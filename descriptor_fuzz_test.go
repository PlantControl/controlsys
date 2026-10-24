package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzDesc is a singular-E descriptor model built in Weierstrass form from a
// decoded spec: the slow part (s.A, s.B, s.C) joins an algebraic part
// (I, B2, C2) with nilpotent E-block N, and both are mixed by Q, Z:
// E = Q·diag(I, N)·Z, A = Q·diag(A1, I)·Z, B = Q·[B1; B2], C = [C1 C2]·Z.
// G(s) = D + C1(sI−A1)⁻¹B1 − Σ_k s^k C2 N^k B2, proper iff C2 N^k B2 = 0
// for k ≥ 1, the reduced static term is D − C2 B2.
//
//	v[12] algebraic order ni = 1 + ⌊3|v|⌋ mod 3 and N's chain breaks
//	v[13] ≥ 0: B2 only on chain heads (N·B2 = 0, proper); < 0: generic
//	v[14] Givens angles of Q, Z;  v[15] row/column scaling 10^clamp(|v|,0,2)
type fzDesc struct {
	n, nf, ni int
	a, e      []float64
	b, c      []float64
	dRed      []float64
	proper    int // 1 proper, 0 improper, -1 undecided
	chain     int
	cond      float64
}

func fzBuildDesc(s *fzSpec) *fzDesc {
	nf, m, p := s.N, s.M, s.P
	v := s.Vals
	code := int(math.Floor(math.Mod(math.Abs(v[12])*3, 24)))
	ni := 1 + code%3
	n := nf + ni
	N := make([]float64, ni*ni)
	for i := 0; i+1 < ni; i++ {
		if (code/3)>>i&1 == 0 {
			N[i*ni+i+1] = 1
		}
	}
	head := func(i int) bool { return i == 0 || N[(i-1)*ni+i] == 0 }
	ph := s.Vals[11]
	B2 := make([]float64, ni*m)
	C2 := make([]float64, p*ni)
	for i := range ni {
		for j := range m {
			if v[13] < 0 || head(i) {
				B2[i*m+j] = math.Sin(1.3+ph*float64(i+1)+0.7*float64(j)) + 0.25
			}
		}
	}
	for i := range p {
		for j := range ni {
			C2[i*ni+j] = math.Cos(0.4+ph*float64(j+2)-0.9*float64(i)) + 0.2
		}
	}
	d := &fzDesc{n: n, nf: nf, ni: ni, proper: 1}
	for i, run := 0, 1; i < ni; i++ {
		if i > 0 && !head(i) {
			run++
		} else {
			run = 1
		}
		d.chain = max(d.chain, run)
	}
	if v[13] < 0 {
		mx := 0.0
		pk := slicesCopy(B2)
		for range ni {
			pk = fzMatMul(N, pk, ni, ni, m)
			mx = math.Max(mx, fzMaxAbs(fzMatMul(C2, pk, p, ni, m)))
		}
		switch {
		case mx == 0:
			d.proper = 1
		case mx > 1e-3*fzMaxAbs(C2)*fzMaxAbs(B2):
			d.proper = 0
		default:
			d.proper = -1
		}
	}
	E0 := make([]float64, n*n)
	A0 := make([]float64, n*n)
	B0 := make([]float64, n*m)
	C0 := make([]float64, p*n)
	for i := range nf {
		E0[i*n+i] = 1
		copy(A0[i*n:i*n+nf], s.A[i*nf:(i+1)*nf])
		copy(B0[i*m:(i+1)*m], s.B[i*m:(i+1)*m])
	}
	for i := range ni {
		copy(E0[(nf+i)*n+nf:(nf+i)*n+n], N[i*ni:(i+1)*ni])
		A0[(nf+i)*n+nf+i] = 1
		copy(B0[(nf+i)*m:(nf+i+1)*m], B2[i*m:(i+1)*m])
	}
	for i := range p {
		copy(C0[i*n:i*n+nf], s.C[i*nf:(i+1)*nf])
		copy(C0[i*n+nf:(i+1)*n], C2[i*ni:(i+1)*ni])
	}
	d.dRed = slicesCopy(s.D)
	c2b2 := fzMatMul(C2, B2, p, ni, m)
	for k := range d.dRed {
		d.dRed[k] -= c2b2[k]
	}
	sc := math.Pow(10, fzClamp(math.Abs(v[15]), 0, 2))
	Q := fzMixer(n, v[14], sc, 0)
	Z := fzMixer(n, v[14]+0.77, sc, 1)
	d.e = fzMatMul(fzMatMul(Q, E0, n, n, n), Z, n, n, n)
	d.a = fzMatMul(fzMatMul(Q, A0, n, n, n), Z, n, n, n)
	d.b = fzMatMul(Q, B0, n, n, m)
	d.c = fzMatMul(C0, Z, p, n, n)
	d.cond = sc * sc
	return d
}

// fzMixer returns Givens sweep · diag(10^(k·log10(sc)/(n−1))), rows or
// columns scaled, so its 2-norm condition is at most sc.
func fzMixer(n int, th, sc float64, side int) []float64 {
	M := make([]float64, n*n)
	for i := range n {
		M[i*n+i] = 1
	}
	for k := 0; k+1 < n; k++ {
		c, s := math.Cos(th+0.61*float64(k)), math.Sin(th+0.61*float64(k))
		for j := range n {
			x, y := M[k*n+j], M[(k+1)*n+j]
			M[k*n+j], M[(k+1)*n+j] = c*x-s*y, s*x+c*y
		}
	}
	for k := range n {
		f := 1.0
		if n > 1 {
			f = math.Pow(sc, float64(k)/float64(n-1))
		}
		for j := range n {
			if side == 0 {
				M[j*n+k] *= f
			} else {
				M[k*n+j] *= f
			}
		}
	}
	return M
}

func fzMatMul(a, b []float64, r, k, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for l := range k {
			x := a[i*k+l]
			for j := range c {
				out[i*c+j] += x * b[l*c+j]
			}
		}
	}
	return out
}

func fzMaxAbs(a []float64) float64 {
	m := 0.0
	for _, x := range a {
		m = math.Max(m, math.Abs(x))
	}
	return m
}

func slicesCopy(a []float64) []float64 { return append([]float64(nil), a...) }

// fzExactDescResponse returns C(sE−A)⁻¹B + D exactly over rationals.
func fzExactDescResponse(n, m, p int, a, e, b, c, d []float64, s complex128) ([]complex128, bool) {
	cols := n + m
	M := make([]*fzCRat, n*cols)
	for i := range n {
		for j := range cols {
			z := new(fzCRat)
			if j < n {
				ev := fzRat(e[i*n+j])
				z.re.Mul(ev, fzRat(real(s)))
				z.re.Sub(&z.re, fzRat(a[i*n+j]))
				z.im.Mul(ev, fzRat(imag(s)))
			} else {
				z.re.Set(fzRat(b[i*m+j-n]))
			}
			M[i*cols+j] = z
		}
	}
	for k := range n {
		piv := -1
		for i := k; i < n; i++ {
			if !M[i*cols+k].isZero() {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range cols {
			M[k*cols+j], M[piv*cols+j] = M[piv*cols+j], M[k*cols+j]
		}
		for i := range n {
			if i == k || M[i*cols+k].isZero() {
				continue
			}
			f := fzCQuo(M[i*cols+k], M[k*cols+k])
			for j := k; j < cols; j++ {
				t := fzCMul(f, M[k*cols+j])
				M[i*cols+j].re.Sub(&M[i*cols+j].re, &t.re)
				M[i*cols+j].im.Sub(&M[i*cols+j].im, &t.im)
			}
		}
	}
	out := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			var acc fzCRat
			acc.re.Set(fzRat(d[i*m+j]))
			for k := range n {
				x := fzCQuo(M[k*cols+n+j], M[k*cols+k])
				ck := fzRat(c[i*n+k])
				var t1, t2 fzCRat
				t1.re.Mul(ck, &x.re)
				t2.im.Mul(ck, &x.im)
				acc.re.Add(&acc.re, &t1.re)
				acc.im.Add(&acc.im, &t2.im)
			}
			re, _ := acc.re.Float64()
			im, _ := acc.im.Float64()
			out[i*m+j] = complex(re, im)
		}
	}
	return out, true
}

// fzDescSens returns ‖C R‖·‖R B‖·(|s|‖E‖+‖A‖) + ‖C‖‖R B‖ + ‖D‖ with
// R = (sE−A)⁻¹ in float64 (Frobenius norms): the first-order effect of
// normwise O(eps) perturbations of the pencil on C(sE−A)⁻¹B + D.
func fzDescSens(n, m, p int, a, e, b, c, d []float64, s complex128) float64 {
	inv, ok := fzCInverse(n, func(i, j int) complex128 {
		return s*complex(e[i*n+j], 0) - complex(a[i*n+j], 0)
	})
	if !ok {
		return math.Inf(1)
	}
	at := func(i, j int) complex128 { return inv[i*n+j] }
	var cr, rb float64
	for i := range p {
		for j := range n {
			var x complex128
			for k := range n {
				x += complex(c[i*n+k], 0) * at(k, j)
			}
			cr += real(x)*real(x) + imag(x)*imag(x)
		}
	}
	for i := range n {
		for j := range m {
			var x complex128
			for k := range n {
				x += at(i, k) * complex(b[k*m+j], 0)
			}
			rb += real(x)*real(x) + imag(x)*imag(x)
		}
	}
	return math.Sqrt(cr)*math.Sqrt(rb)*(cmplx.Abs(s)*fzFrob(e)+fzFrob(a)) + fzFrob(c)*math.Sqrt(rb) + fzFrob(d)
}

// fzCInverse inverts the n×n matrix el by Gauss–Jordan with partial pivoting.
func fzCInverse(n int, el func(i, j int) complex128) ([]complex128, bool) {
	w := 2 * n
	M := make([]complex128, n*w)
	for i := range n {
		for j := range n {
			M[i*w+j] = el(i, j)
		}
		M[i*w+n+i] = 1
	}
	for k := range n {
		piv, best := -1, 0.0
		for i := k; i < n; i++ {
			if a := cmplx.Abs(M[i*w+k]); a > best {
				piv, best = i, a
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range w {
			M[k*w+j], M[piv*w+j] = M[piv*w+j], M[k*w+j]
		}
		inv := 1 / M[k*w+k]
		for j := range w {
			M[k*w+j] *= inv
		}
		for i := range n {
			if i == k || M[i*w+k] == 0 {
				continue
			}
			f := M[i*w+k]
			for j := range w {
				M[i*w+j] -= f * M[k*w+j]
			}
		}
	}
	out := make([]complex128, n*n)
	for i := range n {
		copy(out[i*n:(i+1)*n], M[i*w+n:(i+1)*w])
	}
	return out, true
}

// FuzzDescriptor: singular-E descriptor models with known Weierstrass
// structure (fzBuildDesc). Oracles: IsProper equals the constructed
// properness; TransferFunction and Step succeed exactly when proper and
// IsProper agrees with them; EvalFr, the TF and Stable+Unstable of Stabsep
// match the exact rational C(sE−A)⁻¹B + D of the float64 model within the
// first-order pencil sensitivity (and, for the TF, the polynomial-form
// conditioning of FuzzSSToTF); Poles are the intended slow eigenvalues
// within Bauer–Fike × cond(Q)·cond(Z); the Step response matches that of the
// explicit slow model (A1, B1, C1, D − C2B2).
func FuzzDescriptor(f *testing.F) {
	for _, s := range fzDescSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, fzCheckDescriptor)
}

func fzCheckDescriptor(t *testing.T, s *fzSpec) {
	const eps = 0x1p-52
	if s.NormJ*s.CondT > 1e6 {
		fzSkip(t, "slow part out of scale with the algebraic part")
	}
	d := fzBuildDesc(s)
	if s.NormJ*s.CondT*d.cond*d.cond > 0.1*math.Pow(eps, -1/float64(d.chain)) {
		fzSkip(t, "slow eigenvalues within the ε^(−1/k) split of the infinite chain")
	}
	n, m, p := d.n, s.M, s.P
	for _, blk := range [][]float64{d.a, d.e, d.b, d.c} {
		for _, x := range blk {
			if fzBadFloat(x) {
				fzSkip(t, "descriptor entries out of range")
			}
		}
	}
	sys, err := NewDescriptor(mat.NewDense(n, n, d.a), mat.NewDense(n, m, d.b), mat.NewDense(p, n, d.c), mat.NewDense(p, m, slicesCopy(s.D)), mat.NewDense(n, n, d.e), s.Dt)
	fzContract(t, s, "NewDescriptor", sys != nil, err)
	fzErr(t, s, "NewDescriptor", err)

	pr, err := sys.IsProper()
	if err != nil {
		t.Fatalf("IsProper: %v (constructed proper=%d)\n%s", err, d.proper, s)
	}
	if d.proper >= 0 && pr != (d.proper == 1) {
		t.Fatalf("IsProper = %v, constructed proper=%d ni=%d\n%s", pr, d.proper, d.ni, s)
	}

	pts := fzConvPoints(s)
	pts = append(pts, complex(0.31, 1.7), complex(-2.3, 0.4))
	for _, x := range pts {
		g, ok := fzExactDescResponse(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
		if !ok {
			continue
		}
		sens := fzDescSens(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
		tol := 1e3 * float64(n*n) * eps * sens
		if !(tol <= 1e-4*(1+fzMaxAbsC(g))) {
			continue
		}
		got, err := sys.EvalFr(x)
		fzContract(t, s, "EvalFr", got != nil, err)
		fzErr(t, s, "EvalFr", err)
		for i := range p {
			for j := range m {
				if dd := cmplx.Abs(got[i][j] - g[i*m+j]); !(dd <= tol) {
					t.Fatalf("EvalFr(%v)[%d,%d] = %v, exact %v (|Δ| %g, tol %g)\n%s", x, i, j, got[i][j], g[i*m+j], dd, tol, s)
				}
			}
		}
	}

	poles, err := sys.Poles()
	fzErr(t, s, "Poles", err)
	var finite []complex128
	for _, z := range poles {
		if !cmplx.IsInf(z) && !cmplx.IsNaN(z) {
			finite = append(finite, z)
		}
	}
	if s.Decidable() && d.proper == 1 {
		r := 64 * s.EigErr() * d.cond * d.cond
		if r < 1e-3*(1+fzMaxAbsC(s.Eig)) {
			if _, _, msg := fzMatchEig(finite, s.Eig, r); msg != "" {
				t.Fatalf("Poles of descriptor: %s\ngot %v want %v\n%s", msg, poles, s.Eig, s)
			}
		}
	}

	sep, err := Stabsep(sys)
	fzContract(t, s, "Stabsep", sep != nil, err)
	if err == nil {
		for _, x := range pts {
			g, ok := fzExactDescResponse(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
			if !ok {
				continue
			}
			sens := fzDescSens(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
			tol := 1e5 * float64(n*n) * eps * sens * d.cond
			if !(tol <= 1e-4*(1+fzMaxAbsC(g))) {
				continue
			}
			gs, err1 := sep.Stable.EvalFr(x)
			gu, err2 := sep.Unstable.EvalFr(x)
			if err1 != nil || err2 != nil {
				continue
			}
			for i := range p {
				for j := range m {
					h := gs[i][j] + gu[i][j]
					if dd := cmplx.Abs(h - g[i*m+j]); !(dd <= tol) {
						t.Fatalf("Stabsep sum(%v)[%d,%d] = %v, exact %v (|Δ| %g, tol %g)\n%s", x, i, j, h, g[i*m+j], dd, tol, s)
					}
				}
			}
		}
	} else if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("Stabsep: %v\n%s", err, s)
	}

	res, err := sys.TransferFunction(nil)
	fzContract(t, s, "TransferFunction", res != nil, err)
	if err != nil {
		if errors.Is(err, ErrImproperModel) && !pr {
			fzSkip(t, "improper descriptor")
		}
		t.Fatalf("TransferFunction: %v (IsProper %v)\n%s", err, pr, s)
	}
	tf := res.TF
	if improper := fzTFImproper(tf); improper == pr {
		t.Fatalf("TransferFunction improper=%v, IsProper %v\nnum=%v den=%v\n%s", improper, pr, tf.Num, tf.Den, s)
	}
	nfc := d.nf
	var cp []float64
	if nfc > 0 {
		cp = fzRatFloats(fzCharPolyExact(s.A, nfc))
	}
	nA := math.Max(fzFrob(s.A), s.CondT*s.NormJ) * d.cond
	nB := math.Max(fzFrob(s.B), fzMaxAbs(d.b)) * d.cond
	kk := 64 * float64((n+1)*(n+1))
	for _, x := range pts {
		g, ok := fzExactDescResponse(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
		if !ok {
			continue
		}
		cx := 1.0
		if nfc > 0 {
			cx = cmplx.Abs(fzBigPolyEval(cp, x))
		}
		if cx == 0 || math.IsInf(cx, 0) {
			continue
		}
		sens := fzDescSens(n, m, p, d.a, d.e, d.b, d.c, s.D, x)
		for i := range p {
			nC := math.Max(fzFrob(s.C[i*nfc:(i+1)*nfc]), fzFrob(d.c[i*n:(i+1)*n])) * d.cond
			nu := math.Max(nA, math.Max(nB, nC)) + cmplx.Abs(x)
			kap := math.Pow(nu, float64(nfc)) / cx
			if !(kk*eps*kap <= 1e-3) || !(1e3*float64(n*n)*eps*sens <= 1e-4*(1+fzMaxAbsC(g))) {
				continue
			}
			for j := range m {
				h := g[i*m+j]
				hp := fzBigPolyEval(tf.Num[i][j], x) / fzBigPolyEval(tf.Den[i], x)
				tol := kk*eps*kap*(cmplx.Abs(h)+math.Abs(d.dRed[i*m+j])+nC*nB/nu) + 1e3*float64(n*n)*eps*sens
				if dd := cmplx.Abs(hp - h); !(dd <= tol) {
					t.Fatalf("TF(%v)[%d,%d] = %v, exact %v (|Δ| %g, tol %g, κ %g)\nnum=%v den=%v\n%s", x, i, j, hp, h, dd, tol, kap, tf.Num[i][j], tf.Den[i], s)
				}
			}
		}
	}

	if d.nf == 0 || d.proper != 1 {
		return
	}
	ex, err := New(mat.NewDense(d.nf, d.nf, slicesCopy(s.A)), mat.NewDense(d.nf, m, slicesCopy(s.B)), mat.NewDense(p, d.nf, slicesCopy(s.C)), mat.NewDense(p, m, slicesCopy(d.dRed)), s.Dt)
	if err != nil {
		t.Fatalf("New(slow part): %v\n%s", err, s)
	}
	ref, err := Step(ex, 0)
	if err != nil {
		fzSkip(t, "Step(slow part): "+err.Error())
	}
	T := ref.T[len(ref.T)-1]
	ref, err = Step(ex, T)
	if err != nil {
		fzSkip(t, "Step(slow part, T): "+err.Error())
	}
	got, err := Step(sys, T)
	fzContract(t, s, "Step", got != nil, err)
	fzErr(t, s, "Step", err)
	if len(got.T) != len(ref.T) {
		fzSkip(t, "Step grids differ")
	}
	gy, ry := fzTimeRespData(got), fzTimeRespData(ref)
	scale := 1 + fzMaxAbs(ry)
	if math.IsInf(scale, 0) || math.IsNaN(scale) {
		fzSkip(t, "Step overflows")
	}
	ktol := 1e-6 * d.cond * d.cond * scale * math.Max(1, s.CondT*s.CondT)
	for k := range ry {
		if dd := math.Abs(gy[k] - ry[k]); !(dd <= ktol) {
			t.Fatalf("Step sample %d = %v, slow-model %v (|Δ| %g, tol %g)\n%s", k, gy[k], ry[k], dd, ktol, s)
		}
	}
}

func fzMaxAbsC(a []complex128) float64 {
	m := 0.0
	for _, x := range a {
		m = math.Max(m, cmplx.Abs(x))
	}
	return m
}

func fzTimeRespData(r *TimeResponse) []float64 {
	return mat.DenseCopyOf(r.Y).RawMatrix().Data
}

func fzDescSeeds() []fzSeed {
	off := fzSeed{In: 0, Out: 0.5, IO: 0.3, Tau: 0, Shear: -10, Rot: 0.37, DScal: 1, Gain: 6, Phase: 0.8}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, 2.5 }),
		with(func(s *fzSeed) { s.N, s.Damp[0], s.In, s.Out = 2, 1.5, 0.7, -1 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out = 3, 2, 2, [3]float64{0, 0.5}, [3]float64{1.5, 3}, 1.0, 0.5
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.In, s.Out, s.IO = 0, 1, 2, 0.4, 1, 0 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out, s.Tau = 3, 1, 2, [3]float64{0.3, -0.2}, [3]float64{1.2, 1}, 1.4, 1, 1.5
		}),
		with(func(s *fzSeed) {
			s.N, s.Mag, s.Damp, s.In, s.Out, s.IO = 4, [3]float64{-1, 1}, [3]float64{2, -1}, 0.7, -0.3, 0.6
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out, s.Tau = 5, 2, 1, [3]float64{0, 0.5, 1}, [3]float64{2, 3, 1}, 2.0, 0.2, 2
		}),
	}
}

// fzTFImproper reports whether a channel numerator of tf outranks its
// denominator (leading coefficients first, exact zeros trimmed).
func fzTFImproper(tf *TransferFunc) bool {
	deg := func(c []float64) int {
		for i, x := range c {
			if x != 0 {
				return len(c) - 1 - i
			}
		}
		return -1
	}
	for i, row := range tf.Num {
		for _, num := range row {
			if deg(num) > deg(tf.Den[i]) {
				return true
			}
		}
	}
	return false
}

// TestDescriptorChainPerturbedA: an improper index-3 chain beside a slow pair
// at |λ| 1e4 (kind 0x40). Rounding of A (‖A‖ 1.4e4) turns the staircase Z
// and leaks ~1e-12 into the deflated E blocks, above the E-only rank tol;
// the staircase must still deflate the chain, so IsProper is false, the TF
// is improper and matches the exact response, Poles is the slow pair, not
// three spurious ~2e5 ones, and IsStable is true (base errored).
func TestDescriptorChainPerturbedA(t *testing.T) {
	s := fzDecode(t, 0x40, [fzNVals]float64{83, 3, 0, 0, 0, 0, 6, 1, 0, -10, 0.37, -83.2, 64.7, -1, 0.3, 0})
	d := fzBuildDesc(s)
	if d.proper != 0 || d.nf != 2 || d.chain != 3 {
		t.Fatalf("spec: proper %d nf %d chain %d", d.proper, d.nf, d.chain)
	}
	n, m, p := d.n, s.M, s.P
	rng := rand.New(rand.NewPCG(1, 2))
	pert := func(x []float64) []float64 {
		y := slicesCopy(x)
		for i := range y {
			for range rng.IntN(9) {
				y[i] = math.Nextafter(y[i], math.Inf(2*rng.IntN(2)-1))
			}
		}
		return y
	}
	for k := range 200 {
		a, e, b, c := pert(d.a), pert(d.e), pert(d.b), pert(d.c)
		sys, err := NewDescriptor(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, slicesCopy(s.D)), mat.NewDense(n, n, e), s.Dt)
		if err != nil {
			t.Fatal(err)
		}
		if pr, err := sys.IsProper(); err != nil || pr {
			t.Fatalf("run %d: IsProper = %v, %v; want false", k, pr, err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("run %d: TransferFunction: %v", k, err)
		}
		if !fzTFImproper(res.TF) {
			t.Fatalf("run %d: TF proper, num=%v den=%v", k, res.TF.Num, res.TF.Den)
		}
		var pts []complex128
		if k%25 == 0 {
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Fatalf("run %d: Step err = %v, want ErrImproperModel", k, err)
			}
			pts = []complex128{complex(-2.3, 0.4), 300i}
		}
		for _, x := range pts {
			w, ok := fzExactDescResponse(n, m, p, a, e, b, c, s.D, x)
			if !ok {
				t.Fatalf("run %d: exact response at %v failed", k, x)
			}
			g, err := res.TF.Eval(x)
			if err != nil {
				t.Fatal(err)
			}
			for i := range p {
				for j := range m {
					if dd := cmplx.Abs(g[i][j] - w[i*m+j]); !(dd <= 1e-6*(1+cmplx.Abs(w[i*m+j]))) {
						t.Fatalf("run %d: G(%v)[%d,%d] = %v, exact %v", k, x, i, j, g[i][j], w[i*m+j])
					}
				}
			}
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatalf("run %d: Poles: %v", k, err)
		}
		if _, _, msg := fzMatchEig(poles, s.Eig, 1e-6*fzMaxAbsC(s.Eig)); msg != "" {
			t.Fatalf("run %d: Poles %v, want %v: %s", k, poles, s.Eig, msg)
		}
		if st, err := sys.IsStable(); err != nil || !st {
			t.Fatalf("run %d: IsStable = %v, %v; want true", k, st, err)
		}
	}
}

// TestDescriptorBlurredChainProper: index-3..6 chains whose staircase
// leaks rounding of A into the deflated blocks and Qᵀ·B, so c·N^k·b of a
// proper model sits above the ε-level static cut and IsProper reported
// false: rank decided within the leak (deflatedRank; [6] seed 205, amd64
// also [6] seed 84), or every rank clear but A row blocks ill-conditioned
// (JDKLOX; [5] 328, 422, [6] 452, 875, 916, 961, [3] 988: b carries ~1e-9
// of ‖b‖ on chain states B does not drive). Oracle: the structured model,
// B on chain heads.
func TestDescriptorBlurredChainProper(t *testing.T) {
	for _, tc := range []struct {
		chains []int
		seed   uint64
	}{{[]int{6}, 205}, {[]int{6}, 84}, {[]int{5}, 66},
		{[]int{5}, 328}, {[]int{5}, 422}, {[]int{6}, 452}, {[]int{6}, 875},
		{[]int{6}, 916}, {[]int{6}, 961}, {[]int{3}, 988}} {
		rng := rand.New(rand.NewPCG(tc.seed, 77))
		nf := 1 + rng.IntN(3)
		sys, oracle, _ := chainDescriptor(t, rng, nf, tc.chains, 0, true)
		if pr, err := sys.IsProper(); err != nil || !pr {
			t.Fatalf("%v seed %d: IsProper = %v, %v; want true", tc.chains, tc.seed, pr, err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil || fzTFImproper(res.TF) {
			t.Fatalf("%v seed %d: TransferFunction err %v, want proper TF", tc.chains, tc.seed, err)
		}
		if _, err := Step(sys, 1); err != nil {
			t.Fatalf("%v seed %d: Step: %v", tc.chains, tc.seed, err)
		}
		for _, s := range []complex128{1i, complex(0.5, 3), 20i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, oracle, s)[0][0]
			if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-6*math.Max(1, cmplx.Abs(w))) {
				t.Fatalf("%v seed %d: G(%v) = %v, want %v", tc.chains, tc.seed, s, g[0][0], w)
			}
		}
	}
}

// TestDescriptorNonNormalChainImproper: an improper index-5 chain whose
// staircase N is strongly non-normal (‖N‖ ≈ 106 from entries off the chain
// path, while the s⁴ coefficient c₀·N₀₁·N₁₂·N₂₃·N₃₄·b₄ is 0.118). The
// cut 1e3·n·ε·‖c‖‖N‖⁴‖b‖ ≈ 0.13 dropped that term, so G(1i) was off by
// 0.118 (JDKLOX [5] seed 837). Oracle: the structured model's pencil
// response at |s| ≤ 3, where the float data still matches it.
func TestDescriptorNonNormalChainImproper(t *testing.T) {
	rng := rand.New(rand.NewPCG(837, 77))
	nf := 1 + rng.IntN(3)
	sys, oracle, _ := chainDescriptor(t, rng, nf, []int{5}, 0, false)
	if pr, err := sys.IsProper(); err != nil || pr {
		t.Fatalf("IsProper = %v, %v; want false", pr, err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil || !fzTFImproper(res.TF) {
		t.Fatalf("TransferFunction err %v, want improper TF", err)
	}
	for _, s := range []complex128{0, 1i, complex(0.5, 3), complex(-2, 1)} {
		g, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		w := pencilResponse(t, oracle, s)[0][0]
		if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-6*math.Max(1, cmplx.Abs(w))) {
			t.Fatalf("G(%v) = %v, want %v", s, g[0][0], w)
		}
	}
	if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("Step err = %v, want ErrImproperModel", err)
	}
}

// TestDescriptorChainBesideSmallEPole: proper index-2/3 chains beside a
// finite pole whose E singular value is 1e-2..1e-4 relative (far above the
// rank floor). That pole turns the computed infinite subspace by ε·egap,
// σ₁(E)/σ_r, leaving c·N·b up to 5× the A-side cut, so IsProper said false
// and Step/TF ErrImproperModel or an improper ratio whose s-term diverged
// from the response (XSDPXI, on base: [3] 1e-3 111/200; [3] 1e-2, [2]
// 1e-4 and [2 1] 1e-4 together 23/600, ≤ 6 now on arm64 and amd64).
// Oracle: the structured model's pencil response. The cut is widened by
// egap only when the reduction matches the direct (sE − A) solve at the
// data scale; reductions too inaccurate for that stay improper as before.
func TestDescriptorChainBesideSmallEPole(t *testing.T) {
	skipSlowUnderRace(t)
	falseMild := 0
	for _, tc := range []struct {
		chains []int
		sr     float64
		mild   bool
	}{{[]int{3}, 1e-2, true}, {[]int{2}, 1e-4, true}, {[]int{2, 1}, 1e-4, true}, {[]int{3}, 1e-3, false}} {
		false_ := 0
		for seed := range uint64(200) {
			rng := rand.New(rand.NewPCG(seed, 77))
			nf := 1 + rng.IntN(3)
			sys, oracle, _ := chainDescriptor(t, rng, nf, tc.chains, tc.sr, true)
			pr, err := sys.IsProper()
			if err != nil {
				t.Fatalf("%v %g seed %d: IsProper: %v", tc.chains, tc.sr, seed, err)
			}
			res, terr := sys.TransferFunction(nil)
			if terr != nil {
				t.Fatalf("%v %g seed %d: TransferFunction: %v", tc.chains, tc.sr, seed, terr)
			}
			if fzTFImproper(res.TF) == pr {
				t.Fatalf("%v %g seed %d: IsProper %v, improper TF %v", tc.chains, tc.sr, seed, pr, !pr)
			}
			if !pr {
				false_++
				continue
			}
			if _, err := Step(sys, 1); err != nil {
				t.Fatalf("%v %g seed %d: Step: %v", tc.chains, tc.sr, seed, err)
			}
			for _, s := range []complex128{0.37i, complex(0.5, 3), 1e3i} {
				g, err := res.TF.Eval(s)
				if err != nil {
					t.Fatal(err)
				}
				w := pencilResponse(t, oracle, s)[0][0]
				if d := cmplx.Abs(g[0][0] - w); !(d <= 2e-6*math.Max(1, cmplx.Abs(w))) {
					t.Fatalf("%v %g seed %d: G(%v) = %v, want %v", tc.chains, tc.sr, seed, s, g[0][0], w)
				}
			}
		}
		if tc.mild {
			falseMild += false_
		} else if false_ > 90 {
			t.Errorf("%v %g: %d/200 proper chains IsProper false, want ≤ 90", tc.chains, tc.sr, false_)
		}
		for seed := range uint64(50) {
			rng := rand.New(rand.NewPCG(seed, 78))
			nf := 1 + rng.IntN(3)
			sys, _, _ := chainDescriptor(t, rng, nf, tc.chains, tc.sr, false)
			if pr, err := sys.IsProper(); err != nil || pr {
				t.Fatalf("%v %g improper seed %d: IsProper = %v, %v; want false", tc.chains, tc.sr, seed, pr, err)
			}
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Fatalf("%v %g improper seed %d: Step err = %v, want ErrImproperModel", tc.chains, tc.sr, seed, err)
			}
		}
	}
	if falseMild > 10 {
		t.Errorf("%d/600 mild proper chains IsProper false, want ≤ 10", falseMild)
	}
}

func fzSISODescriptor(t *testing.T, n int, a, e, b, c []float64) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(n, n, a), mat.NewDense(n, 1, b), mat.NewDense(1, n, c), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(n, n, e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}
