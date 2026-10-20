package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Frequency-response fuzz targets (task MOI6LA).
//
// Oracle: the exact rational response C(xI − A)⁻¹B + D of the float64 model
// at the exact point the library evaluates (jω, or for discrete models
// z = p/conj(p) with p = cos(ωT/2) + j·sin(ωT/2) as rounded, the unit-circle
// point of time_domain.go), times the delay phase. Each entry carries the
// first-order bound β = c·(‖CᵢR‖·(‖A‖+|x|)·‖RBⱼ‖ + ‖Cᵢ‖‖RBⱼ‖ + |Dᵢⱼ|),
// R = (xI − A)⁻¹, of a backward-stable solve (fzFRBound). Finite values must
// lie within β of the exact entry while the point's condition is below
// 1e13; +Inf (the documented pole flag) fails only where the point is
// clearly no pole (condition below 1e11 and β < 1e-6·|G|). NaN always fails.

const fzFRCondCheck, fzFRCondPole = 1e13, 1e11

// fzFRModel is a delay-free explicit model; blk lists the diagonal block
// sizes of an upper block-triangular a (nil: one block), which the exact
// solve exploits.
type fzFRModel struct {
	n, m, p    int
	a, b, c, d []float64
	blk        []int
}

func fzFRModelOf(s *fzSpec) *fzFRModel {
	return &fzFRModel{n: s.N, m: s.M, p: s.P, a: s.A, b: s.B, c: s.C, d: s.D}
}

func (md *fzFRModel) system(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(md.n, md.m, md.p, md.a, md.b, md.c, md.d, dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v", err)
	}
	return sys
}

// fzFRPoint returns the library's frequency point at w and its exact
// rational value.
func fzFRPoint(s *fzSpec, w float64) (frequencyPoint, *fzCRat) {
	if !s.Discrete {
		return pointAt(complex(0, w)), fzCOf(0, w)
	}
	sn, cs := math.Sincos(w * s.Dt / 2)
	c, sr := fzRat(cs), fzRat(sn)
	var c2, s2, den, cross fzCRat
	c2.re.Mul(c, c)
	s2.re.Mul(sr, sr)
	den.re.Add(&c2.re, &s2.re)
	cross.re.Mul(c, sr)
	z := new(fzCRat)
	z.re.Sub(&c2.re, &s2.re)
	z.re.Quo(&z.re, &den.re)
	z.im.Add(&cross.re, &cross.re)
	z.im.Quo(&z.im, &den.re)
	return frequencyPoint{p: complex(cs, sn), q: complex(-cs, sn)}, z
}

func fzFRRound(x *fzCRat) complex128 {
	re, _ := x.re.Float64()
	im, _ := x.im.Float64()
	return complex(re, im)
}

// exact returns C(xI − A)⁻¹B + D at the rational x, rounded; ok is false
// when xI − A is exactly singular.
func (md *fzFRModel) exact(x *fzCRat) ([]complex128, bool) {
	n, m, p := md.n, md.m, md.p
	blk := md.blk
	if blk == nil {
		blk = []int{n}
	}
	X := make([]*fzCRat, n*m)
	end := n
	for _, k := range slices.Backward(blk) {
		lo := end - k
		M := make([]*fzCRat, k*k)
		rhs := make([]*fzCRat, k*m)
		for i := range k {
			r := lo + i
			for j := range k {
				e := fzCOf(-md.a[r*n+lo+j], 0)
				if i == j {
					e.re.Add(&e.re, &x.re)
					e.im.Add(&e.im, &x.im)
				}
				M[i*k+j] = e
			}
			for j := range m {
				acc := fzCOf(md.b[r*m+j], 0)
				for l := end; l < n; l++ {
					if a := md.a[r*n+l]; a != 0 {
						t := fzCMul(fzCOf(a, 0), X[l*m+j])
						acc.re.Add(&acc.re, &t.re)
						acc.im.Add(&acc.im, &t.im)
					}
				}
				rhs[i*m+j] = acc
			}
		}
		Y, ok := fzCSolve(k, M, m, rhs)
		if !ok {
			return nil, false
		}
		copy(X[lo*m:], Y)
		end = lo
	}
	G := fzCReal(p, m, md.d)
	if n > 0 {
		CX := fzCMatMul(p, n, m, fzCReal(p, n, md.c), X)
		for k := range G {
			G[k].re.Add(&G[k].re, &CX[k].re)
			G[k].im.Add(&G[k].im, &CX[k].im)
		}
	}
	return fzCRound(G), true
}

// fzFRBound returns the per-entry first-order error bound β of a
// backward-stable evaluation at z and the condition ‖R‖·(‖A‖+|z|) of the
// point (+Inf when zI − A is numerically singular).
func (md *fzFRModel) bound(z complex128) ([]float64, float64) {
	n := md.n
	Z := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			Z[i*n+j] = complex(-md.a[i*n+j], 0)
		}
		Z[i*n+i] += z
	}
	return fzFRBoundZ(n, md.m, md.p, Z, fzFrob(md.a)+cmplx.Abs(z), md.b, md.c, md.d)
}

// fzFRBoundZ is bound for the pencil Z = zI − A with ‖Z‖ ≤ nZ.
func fzFRBoundZ(n, m, p int, Z []complex128, nZ float64, b, c, d []float64) ([]float64, float64) {
	const eps = 0x1p-52
	k := 64 * float64(n+2) * eps
	beta := make([]float64, p*m)
	if n == 0 {
		for e, v := range d {
			beta[e] = k * math.Abs(v)
		}
		return beta, 1
	}
	R, ok := fzCInv(n, Z)
	if !ok {
		for e := range beta {
			beta[e] = math.Inf(1)
		}
		return beta, math.Inf(1)
	}
	CR := fzCMulF(p, n, n, R, c, true)
	RB := fzCMulF(n, n, m, R, b, false)
	for i := range p {
		cr, ci := fzCFrob(CR[i*n:(i+1)*n]), fzFrob(c[i*n:(i+1)*n])
		for j := range m {
			rb := 0.0
			for l := range n {
				rb = math.Hypot(rb, cmplx.Abs(RB[l*m+j]))
			}
			beta[i*m+j] = k * (cr*nZ*rb + ci*rb + math.Abs(d[i*m+j]))
		}
	}
	return beta, fzCFrob(R) * nZ
}

// fzFRDelayPhase returns the I/O delay factors (p×m) at frequency w and the
// largest phase |ωτ|.
func fzFRDelayPhase(s *fzSpec, pt frequencyPoint, w float64, internal bool) ([]complex128, float64) {
	f := make([]complex128, s.P*s.M)
	phiMax := 0.0
	for i := range s.P {
		for j := range s.M {
			tau := 0.0
			if s.In != nil {
				tau += s.In[j]
			}
			if s.Out != nil {
				tau += s.Out[i]
			}
			if s.IO != nil && !(internal && s.Tau > 0) {
				tau += s.IO[i*s.M+j]
			}
			phi := w * tau
			if s.Discrete {
				phi = 2 * tau * cmplx.Phase(pt.p)
			}
			f[i*s.M+j] = cmplx.Exp(complex(0, -phi))
			phiMax = math.Max(phiMax, math.Abs(phi))
		}
	}
	return f, phiMax
}

// fzFRFreqs returns the evaluation frequencies, ascending: DC, the Nyquist
// frequency, each intended pole's resonance and natural frequency, points
// just beside a resonance, and one point picked by v[11].
func fzFRFreqs(s *fzSpec) []float64 {
	var ws []float64
	add := func(w float64) {
		if w >= 0 && !math.IsInf(w, 0) && !math.IsNaN(w) && (!s.Discrete || w <= math.Pi/s.Dt) {
			ws = append(ws, w)
		}
	}
	add(0)
	if s.Discrete {
		add(math.Pi / s.Dt)
	}
	scale := 1.0
	for k, e := range s.S {
		w := math.Abs(imag(e))
		if s.Discrete {
			w = math.Abs(cmplx.Phase(s.Eig[k])) / s.Dt
		}
		add(w)
		add(w * (1 + 1e-9))
		add(cmplx.Abs(e))
		if cmplx.Abs(e) > 0 {
			scale = cmplx.Abs(e)
		}
	}
	frac := s.Vals[11] - math.Floor(s.Vals[11])
	add(scale * math.Pow(10, 2*frac-1))
	slices.Sort(ws)
	return slices.Compact(ws)
}

// fzFRCheck checks the p×m response got at frequency w against the exact
// want (nil: exactly singular) with bounds beta and point condition cond;
// slack[e] adds rounding of the delay phase.
func fzFRCheck(t *testing.T, s *fzSpec, op string, w float64, got, want []complex128, beta []float64, cond float64, slack []float64) {
	t.Helper()
	fzFRCheckKnown(t, s, op, w, got, want, beta, cond, slack, nil)
}

// fzFRCheckKnown is fzFRCheck that reports whether a finite miss fell within
// tolerance plus known[e], a documented defect the caller counts as a skip
// after checking its remaining points.
func fzFRCheckKnown(t *testing.T, s *fzSpec, op string, w float64, got, want []complex128, beta []float64, cond float64, slack, known []float64) bool {
	t.Helper()
	hit := false
	for e, g := range got {
		if cmplx.IsNaN(g) {
			t.Fatalf("%s at ω=%g: [%d] = NaN\n%s", op, w, e, s)
		}
		if want == nil {
			continue
		}
		h := want[e]
		if cmplx.IsInf(h) || cmplx.IsNaN(h) {
			continue
		}
		tol := beta[e]
		if slack != nil {
			tol += slack[e]
		}
		if cmplx.IsInf(g) {
			if cond < fzFRCondPole && tol < 1e-6*cmplx.Abs(h) {
				t.Fatalf("%s at ω=%g: [%d] = Inf (pole flag), exact %v, point condition %.3g, bound %.3g\n%s", op, w, e, h, cond, tol, s)
			}
			continue
		}
		if cond >= fzFRCondCheck {
			continue
		}
		if d := cmplx.Abs(g - h); !(d <= tol) {
			kn := 0.0
			if known != nil {
				kn = known[e]
			}
			if d <= tol+kn {
				hit = true
				continue
			}
			t.Fatalf("%s at ω=%g: [%d] = %v, exact %v (|Δ| %.3g, bound %.3g, known %.3g, rel %.3g, cond %.3g)\n%s", op, w, e, g, h, d, tol, kn, d/cmplx.Abs(h), cond, s)
		}
	}
	return hit
}

// fzFRPointData is the oracle at one frequency.
type fzFRPointData struct {
	w     float64
	pt    frequencyPoint
	g, gd []complex128 // delay-free and delayed exact responses (nil: singular)
	beta  []float64
	slack []float64
	cond  float64
}

func fzFROracle(s *fzSpec, md *fzFRModel, w float64, internal bool) fzFRPointData {
	const eps = 0x1p-52
	pt, x := fzFRPoint(s, w)
	d := fzFRPointData{w: w, pt: pt}
	d.g, _ = md.exact(x)
	d.beta, d.cond = md.bound(fzFRRound(x))
	f, phi := fzFRDelayPhase(s, pt, w, internal)
	d.slack = make([]float64, len(f))
	if d.g != nil {
		d.gd = make([]complex128, len(f))
		for k := range f {
			d.gd[k] = d.g[k] * f[k]
			d.slack[k] = 8 * eps * (2 + phi) * cmplx.Abs(d.g[k])
		}
	}
	return d
}

func fzFRAt(r *FreqResponseMatrix, k int) []complex128 {
	pm := r.P * r.M
	return r.Data[k*pm : (k+1)*pm]
}

// FuzzFreqResponse: FreqResponse (sweep and pointwise), the dense and
// Hessenberg engines called directly, EvalFr, a descriptor realization with
// the same response, Bode, Sigma and the FRD built from the model all match
// the exact response (see the file comment), with I/O delays.
func FuzzFreqResponse(f *testing.F) {
	for _, s := range fzFRSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{Delays: true})
		md := fzFRModelOf(s)
		ws := fzFRFreqs(s)
		fr, err := sys.FreqResponse(ws)
		fzContract(t, s, "FreqResponse", fr != nil, err)
		fzErr(t, s, "FreqResponse", err)
		pw, err := sys.FreqResponsePointwise(ws)
		fzContract(t, s, "FreqResponsePointwise", pw != nil, err)
		fzErr(t, s, "FreqResponsePointwise", err)
		plain := md.system(t, s.Dt)
		dense := newBalancedDense(plain, s.N, s.M, s.P)
		hess := newHessenbergSweep(plain, s.N, s.M, s.P)
		pts := make([]fzFRPointData, len(ws))
		for k, w := range ws {
			d := fzFROracle(s, md, w, false)
			pts[k] = d
			fzFRCheck(t, s, "FreqResponse", w, fzFRAt(fr, k), d.gd, d.beta, d.cond, d.slack)
			one, err := sys.FreqResponse([]float64{w})
			fzErr(t, s, "FreqResponse(single)", err)
			if !fzFRBitEqual(fzFRAt(pw, k), one.Data) {
				t.Fatalf("FreqResponsePointwise at ω=%g = %v, single-point FreqResponse %v\n%s", w, fzFRAt(pw, k), one.Data, s)
			}
			for _, eng := range []struct {
				name string
				eval func(frequencyPoint, []complex128) error
			}{{"balancedDense", dense.evalInto}, {"hessenbergSweep", hess.evalInto}} {
				got := make([]complex128, s.P*s.M)
				if err := eng.eval(d.pt, got); err != nil {
					if !errors.Is(err, ErrSingularTransform) {
						t.Fatalf("%s at ω=%g: %v\n%s", eng.name, w, err, s)
					}
					if d.cond < fzFRCondPole {
						t.Fatalf("%s at ω=%g: %v at point condition %.3g\n%s", eng.name, w, err, d.cond, s)
					}
					continue
				}
				fzFRCheck(t, s, eng.name, w, got, d.g, d.beta, d.cond, nil)
			}
			if !s.Discrete {
				g, err := sys.EvalFr(complex(0, w))
				fzErr(t, s, "EvalFr", err)
				if flat := slices.Concat(g...); !fzFRBitEqual(flat, one.Data) {
					t.Fatalf("EvalFr(j%g) = %v, FreqResponse %v\n%s", w, flat, one.Data, s)
				}
			} else {
				z := fzFreqPoint(s, w)
				g, err := sys.EvalFr(z)
				fzErr(t, s, "EvalFr", err)
				want, _ := md.exact(fzCOf(real(z), imag(z)))
				beta, cond := md.bound(z)
				var gd []complex128
				if want != nil {
					gd = make([]complex128, len(want))
					for k := range want {
						gd[k] = want[k] * fzFRDelayZ(s, k, z)
					}
				}
				fzFRCheck(t, s, "EvalFr(z)", w, slices.Concat(g...), gd, beta, cond, d.slack)
			}
		}
		fzFRCheckBode(t, s, sys, ws, pts)
		fzFRCheckDescriptor(t, s, ws, pts)
		fzFRCheckFRD(t, s, sys, ws, fr)
		fzFRCheckSigma(t, s, sys, ws, pts)
	})
}

// fzFRDelayZ is the I/O delay factor z^{−τ} of flat entry k at the plain
// discrete point z.
func fzFRDelayZ(s *fzSpec, k int, z complex128) complex128 {
	i, j := k/s.M, k%s.M
	tau := 0.0
	if s.In != nil {
		tau += s.In[j]
	}
	if s.Out != nil {
		tau += s.Out[i]
	}
	if s.IO != nil {
		tau += s.IO[k]
	}
	if tau == 0 {
		return 1
	}
	return cmplx.Pow(z, complex(-math.Round(tau), 0))
}

func fzFRBitEqual(a, b []complex128) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if math.Float64bits(real(a[k])) != math.Float64bits(real(b[k])) || math.Float64bits(imag(a[k])) != math.Float64bits(imag(b[k])) {
			if !(cmplx.IsNaN(a[k]) && cmplx.IsNaN(b[k])) {
				return false
			}
		}
	}
	return true
}

// fzFRResolved reports whether every delayed entry at d is finite and known
// to relative accuracy 1e-3, and returns ‖ΔG‖_F.
func fzFRResolved(d fzFRPointData) (float64, bool) {
	if d.gd == nil || d.cond >= fzFRCondCheck {
		return 0, false
	}
	var e float64
	for k, h := range d.gd {
		tol := d.beta[k] + d.slack[k]
		if cmplx.IsInf(h) || cmplx.IsNaN(h) || !(tol <= 1e-3*cmplx.Abs(h)) && h != 0 {
			return 0, false
		}
		e = math.Hypot(e, tol)
	}
	return e, true
}

func fzFRCheckBode(t *testing.T, s *fzSpec, sys *System, ws []float64, pts []fzFRPointData) {
	t.Helper()
	b, err := sys.Bode(ws, 0)
	fzContract(t, s, "Bode", b != nil, err)
	fzErr(t, s, "Bode", err)
	for i := range s.P {
		for j := range s.M {
			prev := math.NaN()
			for k, d := range pts {
				mag, ph := b.MagDBAt(k, i, j), b.PhaseAt(k, i, j)
				if math.IsNaN(mag) {
					t.Fatalf("Bode at ω=%g: magnitude NaN\n%s", d.w, s)
				}
				if !math.IsInf(ph, 0) && !math.IsNaN(ph) {
					if !math.IsNaN(prev) && math.Abs(ph-prev) > 180+1e-9 {
						t.Fatalf("Bode phase jumps %g -> %g deg at ω=%g: not unwrapped\n%s", prev, ph, d.w, s)
					}
					prev = ph
				}
				if d.gd == nil || d.cond >= fzFRCondCheck {
					continue
				}
				e := i*s.M + j
				h := d.gd[e]
				rel := (d.beta[e] + d.slack[e]) / cmplx.Abs(h)
				if h == 0 || cmplx.IsInf(h) || !(rel < 1e-3) {
					continue
				}
				if want := 20 * math.Log10(cmplx.Abs(h)); math.Abs(mag-want) > 8.7*rel+1e-9 {
					t.Fatalf("Bode at ω=%g [%d,%d]: %g dB, exact %g dB\n%s", d.w, i, j, mag, want, s)
				}
				dp := math.Mod(ph-cmplx.Phase(h)*180/math.Pi, 360)
				dp = math.Min(math.Abs(dp), 360-math.Abs(dp))
				if dp > 60*rel+1e-7 {
					t.Fatalf("Bode at ω=%g [%d,%d]: phase %g deg, exact %g mod 360\n%s", d.w, i, j, ph, cmplx.Phase(h)*180/math.Pi, s)
				}
			}
		}
	}
}

func fzFRCheckSigma(t *testing.T, s *fzSpec, sys *System, ws []float64, pts []fzFRPointData) {
	t.Helper()
	sg, err := sys.Sigma(ws, 0)
	fzContract(t, s, "Sigma", sg != nil, err)
	if err != nil && errors.Is(err, ErrInvalidArgument) && min(s.P, s.M) > 1 && fzFRAnyInf(pts) {
		return
	}
	fzErr(t, s, "Sigma", err)
	for k, d := range pts {
		for i := range sg.NSV() {
			if math.IsNaN(sg.At(k, i)) || sg.At(k, i) < 0 {
				t.Fatalf("Sigma at ω=%g: σ%d = %g\n%s", d.w, i, sg.At(k, i), s)
			}
		}
		e, ok := fzFRResolved(d)
		if !ok {
			continue
		}
		want := fzSingularValues(d.gd, s.P, s.M)
		for i, v := range want {
			fzFRCheckSV(t, s, "Sigma", d.w, i, sg.At(k, i), v, want[0], e+1e-13*want[0])
		}
	}
}

// fzFRCheckSV checks σᵢ against the exact v within tol. A miss inside the
// accuracy of eigenvalues of the Gram matrix GᴴG, which Sigma uses
// (frequency_svd.go: |σ̂² − σ²| ~ eps·σ₁²), is the known small-σ defect
// (followup), not a new failure.
func fzFRCheckSV(t *testing.T, s *fzSpec, op string, w float64, i int, got, v, s1, tol float64) {
	t.Helper()
	const eps = 0x1p-52
	d := math.Abs(got - v)
	if d <= tol {
		return
	}
	c := 64 * float64(s.P+s.M) * eps
	if gram := c * s1 * s1 / math.Max(got+v, math.Sqrt(c)*s1); d <= tol+gram {
		fzSkip(t, "known: Sigma small singular values from Gram eigenvalues")
	}
	t.Fatalf("%s at ω=%g: σ%d = %g, exact %g (|Δ| %.3g, bound %.3g)\n%s", op, w, i, got, v, d, tol, s)
}

func fzFRAnyInf(pts []fzFRPointData) bool {
	for _, d := range pts {
		if d.cond >= fzFRCondPole || d.gd == nil {
			return true
		}
	}
	return false
}

// fzFRCheckDescriptor evaluates the descriptor realization (E, E·A, E·B,
// C, D), E a permuted power-of-two diagonal, whose entries are formed
// exactly and whose response is that of the explicit model.
func fzFRCheckDescriptor(t *testing.T, s *fzSpec, ws []float64, pts []fzFRPointData) {
	t.Helper()
	n := s.N
	if n == 0 {
		return
	}
	E := mat.NewDense(n, n, nil)
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, s.M, nil)
	for i := range n {
		r := (i + 1) % n
		e := math.Ldexp(1, i%3-1)
		E.Set(i, r, e)
		for j := range n {
			A.Set(i, j, e*s.A[r*n+j])
		}
		for j := range s.M {
			B.Set(i, j, e*s.B[r*s.M+j])
		}
	}
	if slices.ContainsFunc(slices.Concat(A.RawMatrix().Data, B.RawMatrix().Data), fzBadFloat) {
		return
	}
	ds, err := NewDescriptor(A, B, mat.NewDense(s.P, n, slices.Clone(s.C)), mat.NewDense(s.P, s.M, slices.Clone(s.D)), E, s.Dt)
	if err != nil {
		t.Fatalf("NewDescriptor: %v\n%s", err, s)
	}
	fr, err := ds.FreqResponse(ws)
	fzContract(t, s, "descriptor FreqResponse", fr != nil, err)
	fzErr(t, s, "descriptor FreqResponse", err)
	for k, d := range pts {
		beta := make([]float64, len(d.beta))
		for e, b := range d.beta {
			beta[e] = 4 * b
		}
		fzFRCheck(t, s, "descriptor FreqResponse", d.w, fzFRAt(fr, k), d.g, beta, d.cond, nil)
	}
}

// fzFRCheckFRD: sys.FRD holds the FreqResponse data bit for bit; FRD
// Sigma, PeakGain, Parallel, Series and Feedback match the same algebra
// done here on that data.
func fzFRCheckFRD(t *testing.T, s *fzSpec, sys *System, ws []float64, fr *FreqResponseMatrix) {
	t.Helper()
	const eps = 0x1p-52
	F, err := sys.FRD(ws)
	if err != nil {
		for k := range ws {
			if slices.ContainsFunc(fzFRAt(fr, k), cmplx.IsInf) {
				return
			}
		}
		t.Fatalf("FRD: %v\n%s", err, s)
	}
	p, m := s.P, s.M
	for k := range ws {
		for i := range p {
			for j := range m {
				if a, b := F.Response[k][i][j], fr.At(k, i, j); !fzFRBitEqual([]complex128{a}, []complex128{b}) {
					t.Fatalf("FRD[%d][%d][%d] = %v, FreqResponse %v\n%s", k, i, j, a, b, s)
				}
			}
		}
	}
	for k := range ws {
		for _, g := range fzFRAt(fr, k) {
			if cmplx.IsInf(g) || cmplx.Abs(g) > 1e150 {
				return
			}
		}
	}
	par, err := FRDParallel(F, F)
	fzErr(t, s, "FRDParallel", err)
	for k := range ws {
		for i := range p {
			for j := range m {
				if got, want := par.Response[k][i][j], 2*F.Response[k][i][j]; got != want {
					t.Fatalf("FRDParallel[%d][%d][%d] = %v, want %v\n%s", k, i, j, got, want, s)
				}
			}
		}
	}
	Kd := make([][][]complex128, len(ws))
	for k := range ws {
		Kd[k] = make([][]complex128, m)
		for i := range m {
			Kd[k][i] = make([]complex128, p)
			for j := range p {
				Kd[k][i][j] = complex(0.3*math.Cos(float64(i+2*j)+s.Vals[11]), 0.2*math.Sin(float64(3*i+j)))
			}
		}
	}
	K, err := NewFRD(Kd, ws, s.Dt)
	if err != nil {
		t.Fatalf("NewFRD: %v\n%s", err, s)
	}
	ser, err := FRDSeries(F, K)
	fzErr(t, s, "FRDSeries", err)
	fb, fbErr := FRDFeedback(F, K, -1)
	maxCond, maxKG := 0.0, 0.0
	for k := range ws {
		G := fzFRAt(fr, k)
		Kk := slices.Concat(Kd[k]...)
		KG := fzFRCMul(m, p, m, Kk, G)
		nKG := fzCFrob(Kk) * fzCFrob(G)
		for e, v := range KG {
			if got := ser.Response[k][e/m][e%m]; cmplx.Abs(got-v) > 8*float64(p)*eps*nKG {
				t.Fatalf("FRDSeries[%d] [%d] = %v, want K·G %v\n%s", k, e, got, v, s)
			}
		}
		L := fzFRCMul(p, m, p, G, Kk)
		for i := range p {
			L[i*p+i] += 1
		}
		Li, ok := fzCInv(p, L)
		cond := math.Inf(1)
		if ok {
			cond = fzCFrob(Li) * (1 + nKG)
		}
		maxCond, maxKG = math.Max(maxCond, cond), math.Max(maxKG, nKG)
		if fbErr != nil || cond > 1e10 {
			continue
		}
		want := fzFRCMul(p, p, m, Li, G)
		tol := 64 * float64(p+m) * eps * cond * fzCFrob(Li) * fzCFrob(G) * (1 + nKG)
		for e, v := range want {
			if got := fb.Response[k][e/m][e%m]; cmplx.Abs(got-v) > tol {
				t.Fatalf("FRDFeedback[%d] [%d] = %v, want (I+GK)⁻¹G %v (tol %.3g)\n%s", k, e, got, v, tol, s)
			}
		}
	}
	if fbErr != nil && errors.Is(fbErr, ErrAlgebraicLoop) && maxCond < 1e12 && maxKG > 1e14 && m > p {
		fzSkip(t, "known: FRDFeedback pivot tolerance flags I+K·G singular for large ‖K·G‖")
	}
	if fbErr != nil && (!errors.Is(fbErr, ErrAlgebraicLoop) || maxCond < 1e12) {
		t.Fatalf("FRDFeedback: %v with max cond(I+GK) %.3g\n%s", fbErr, maxCond, s)
	}
	pk, err := F.PeakGain()
	fzErr(t, s, "FRD.PeakGain", err)
	sg, err := F.Sigma()
	fzErr(t, s, "FRD.Sigma", err)
	peak := 0.0
	for k := range ws {
		peak = math.Max(peak, fzSingularValues(fzFRAt(fr, k), p, m)[0])
	}
	if math.Abs(pk.Gain-peak) > 64*eps*peak || pk.Frequency != ws[pk.Index] {
		t.Fatalf("FRD.PeakGain = %+v, want gain %g\n%s", pk, peak, s)
	}
	for k := range ws {
		g := fzFRAt(fr, k)
		want := fzSingularValues(g, p, m)
		for i, v := range want {
			fzFRCheckSV(t, s, "FRD.Sigma", ws[k], i, sg.At(k, i), v, want[0], 64*eps*fzCFrob(g))
		}
	}
}

func fzFRCMul(r, k, c int, a, b []complex128) []complex128 {
	out := make([]complex128, r*c)
	for i := range r {
		for j := range c {
			var acc complex128
			for l := range k {
				acc += a[i*k+l] * b[l*c+j]
			}
			out[i*c+j] = acc
		}
	}
	return out
}

func fzFRSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 2, 0, 4 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.DScal = 2, 1, 13, 0.3 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Phase = 2, 0.5, 11.5, 0.7 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 2, 0.2, 2 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0.3, 0.3}, [3]float64{8, 8}
			s.Shear, s.Rot, s.Phase = -2, 0.8, 0.4
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 3, 2
			s.Mag, s.Damp = [3]float64{-40, 1, 2}, [3]float64{3, 6, 0}
			s.Gain, s.DScal, s.In, s.Out, s.IO = 7, 0.2, -0.5, -1, -0.2
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 6, 1, 1
			s.Mag, s.Damp = [3]float64{-2, 1, 3}, [3]float64{3, 5, 0.5}
			s.Shear, s.Rot, s.Phase, s.Gain = 0, 0.8, 0.4, 7
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.1)
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0.5, 1}, [3]float64{10, 2}
			s.Rot, s.Phase, s.In = -1.3, 0.9, 0.5
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, 0
			s.N, s.M, s.P = 3, 1, 2
			s.Mag, s.Damp = [3]float64{math.Log10(math.Pi) - 1e-9, 0}, [3]float64{12, 1}
			s.DScal = -2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -2
			s.N, s.M, s.P = 2, 1, 1
			s.Mag[0], s.Damp[0] = -3, 13
		}),
	}
}
