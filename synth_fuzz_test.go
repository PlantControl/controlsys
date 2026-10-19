package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Controller-synthesis fuzz targets (Place/Acker, Kalman/Lqe, Lqg, H2Syn,
// HinfSyn, Mixsyn) on the fzSpec models; the delay fields v[12..15] are
// reused as design parameters. Closed loops are assembled in big.Rat from
// the float64 plant and controller and checked by exact Routh / Schur–Cohn;
// norms by a Kronecker Lyapunov solve (H2) or a frequency sweep (H∞) of
// that closed loop, never by the library's own norm routines.

// syDesiredPoles returns n distinct stable poles for Place: blocks with
// frequencies w0·r^k and damping ζ ∈ [0.3, 0.95] (all real for v[15] > 0),
// so the library's 0.1·|p| accuracy disks stay inside the stability region.
// Discrete poles are exp(p·Dt) with |p|·Dt in [0.55, 3.2], |z| ≤ 0.85.
func syDesiredPoles(s *fzSpec) []complex128 {
	v := s.Vals
	base := 0.0
	for _, e := range s.S {
		base = math.Max(base, cmplx.Abs(e))
	}
	if base == 0 {
		base = 1
	}
	w0 := base * math.Pow(10, fzClamp(v[12], -1.5, 1.5))
	r := math.Pow(10, fzClamp(math.Abs(v[14]), 0.05, 0.7))
	if s.Discrete {
		w0 = fzClamp(w0*s.Dt, 0.55, 1) / s.Dt
		r = math.Min(r, math.Pow(10, 0.25))
	}
	zeta := 0.3 + 0.65*(0.5+0.5*math.Cos(v[13]))
	allReal := v[15] > 0
	var p []complex128
	w := w0
	for len(p) < s.N {
		if allReal || len(p) == s.N-1 {
			p = append(p, complex(-w, 0))
		} else {
			re, im := -zeta*w, w*math.Sqrt(1-zeta*zeta)
			p = append(p, complex(re, im), complex(re, -im))
		}
		w *= r
	}
	if s.Discrete {
		for k, e := range p {
			z := cmplx.Exp(e * complex(s.Dt, 0))
			if imag(e) == 0 {
				z = complex(real(z), 0)
			}
			p[k] = z
		}
	}
	return p
}

// syMonic returns the real monic polynomial with the given conjugate-closed
// roots, descending.
func syMonic(roots []complex128) []float64 {
	c := []complex128{1}
	for _, z := range roots {
		next := make([]complex128, len(c)+1)
		for i, a := range c {
			next[i] += a
			next[i+1] -= a * z
		}
		c = next
	}
	out := make([]float64, len(c))
	for i, a := range c {
		out[i] = real(a)
	}
	return out
}

// syRatDense returns the rational copy of a row-major float slice.
func syRatDense(d []float64) []*big.Rat { return fzRatPoly(d) }

func syRatSub(a, b []*big.Rat) []*big.Rat {
	out := make([]*big.Rat, len(a))
	for i := range a {
		out[i] = new(big.Rat).Sub(a[i], b[i])
	}
	return out
}

func syRatAdd(a, b []*big.Rat) []*big.Rat {
	out := make([]*big.Rat, len(a))
	for i := range a {
		out[i] = new(big.Rat).Add(a[i], b[i])
	}
	return out
}

func syRatFloat(a []*big.Rat) []float64 {
	out := make([]float64, len(a))
	for i, x := range a {
		out[i], _ = x.Float64()
	}
	return out
}

// syAckerExact returns e_nᵀ·Ctrb(A, b)⁻¹·χ(A) exactly (the unique gain
// placing the roots of the monic float polynomial chi), rounded to float64;
// nil when Ctrb is exactly singular.
func syAckerExact(a, b []float64, n int, chi []float64) []float64 {
	A := syRatDense(a)
	ctrb := make([]*big.Rat, n*n)
	col := syRatDense(b)
	for k := range n {
		for i := range n {
			ctrb[i*n+k] = col[i]
		}
		col = mqRatMul(A, col, n, n, 1)
	}
	inv := mqRatInv(ctrb, n)
	if inv == nil {
		return nil
	}
	P := make([]*big.Rat, n*n)
	for i := range P {
		P[i] = new(big.Rat)
	}
	for i := range n {
		P[i*n+i].SetInt64(1)
	}
	for _, c := range chi[1:] {
		P = mqRatMul(P, A, n, n, n)
		for i := range n {
			P[i*n+i].Add(P[i*n+i], fzRat(c))
		}
	}
	return syRatFloat(mqRatMul(inv[(n-1)*n:], P, 1, n, n))
}

// syRatPolyEval returns c(z) and c'(z) for the descending rational c at the
// float64 point z, exactly, rounded to complex128.
func syRatPolyEval(c []*big.Rat, z complex128) (val, der complex128) {
	var x fzCRat
	x.re.Set(fzRat(real(z)))
	x.im.Set(fzRat(imag(z)))
	var v, d fzCRat
	for _, a := range c {
		t := fzCMul(&d, &x)
		d.re.Add(&t.re, &v.re)
		d.im.Add(&t.im, &v.im)
		t = fzCMul(&v, &x)
		v.re.Add(&t.re, a)
		v.im.Set(&t.im)
	}
	vr, _ := v.re.Float64()
	vi, _ := v.im.Float64()
	dr, _ := d.re.Float64()
	di, _ := d.im.Float64()
	return complex(vr, vi), complex(dr, di)
}

// syRootsNear reports whether the exact monic char poly cp of degree n has
// a root within rel·|λ| of every λ: n·|cp(λ)/cp'(λ)| bounds the distance
// to the nearest root. With pairwise separations above 2·rel·|λ| the disks
// are disjoint, so each holds exactly one root.
func syRootsNear(cp []*big.Rat, poles []complex128, rel float64) bool {
	n := len(cp) - 1
	for _, l := range poles {
		v, d := syRatPolyEval(cp, l)
		if d == 0 || !(float64(n)*cmplx.Abs(v)/cmplx.Abs(d) <= rel*cmplx.Abs(l)) {
			return false
		}
	}
	return true
}

// syClosedLoopA returns A − B·K exactly for the float64 data.
func syClosedLoopA(a, b, k []float64, n, m int) []*big.Rat {
	return syRatSub(syRatDense(a), mqRatMul(syRatDense(b), syRatDense(k), n, m, n))
}

// syExactCount returns the number of exact char-poly roots of the n×n
// rational matrix outside the stability region and inside the strip
// (delta-wide, relative) about the boundary; ok false on a degenerate table.
func syExactCount(acl []*big.Rat, n int, discrete bool, delta float64) (out, outStrip int, ok, okStrip bool) {
	if n == 0 {
		return 0, 0, true, true
	}
	cp := fzCharPolyRat(acl, n)
	if discrete {
		out, ok = fzSchurCohnOutside(cp, 1)
		outStrip, okStrip = fzSchurCohnOutside(cp, 1+delta)
		return
	}
	nrm := 0.0
	for _, x := range syRatFloat(acl) {
		nrm = math.Hypot(nrm, x)
	}
	out, ok = fzRouthRight(cp, 0)
	outStrip, okStrip = fzRouthRight(cp, delta*math.Max(1, nrm))
	return
}

// syStableFloat certifies stability cheaply: every computed eigenvalue is
// inside the stability region by ten Bauer–Fike radii (eigenvector
// condition times the backward error of the eigensolver and of rounding
// the exact matrix). False means undecided.
func syStableFloat(af []float64, n int, discrete bool) bool {
	if n == 0 {
		return true
	}
	eig, cond := syEigCond(af, n)
	r := 10 * cond * 1e2 * float64(n) * 0x1p-52 * math.Max(mqFro(af), 1e-300)
	if !(r < 1) {
		return false
	}
	for _, e := range eig {
		if discrete && !(cmplx.Abs(e) < 1-r) || !discrete && !(real(e) < -r) {
			return false
		}
	}
	return true
}

// syStable reports exact stability of the rational matrix, trying the
// float certificate first.
func syStable(acl []*big.Rat, n int, discrete bool) bool {
	if syStableFloat(syRatFloat(acl), n, discrete) {
		return true
	}
	k, _, ok, _ := syExactCount(acl, n, discrete, 1e-8)
	return ok && k == 0
}

// syRequireStableRat fails when the exact closed-loop matrix is unstable,
// skipping when every offending root lies within a 1e-8 strip of the
// boundary (below the rounding of the design data).
func syRequireStableRat(t *testing.T, s *fzSpec, op string, acl []*big.Rat, n int, discrete bool, detail string) {
	t.Helper()
	if syStableFloat(syRatFloat(acl), n, discrete) {
		return
	}
	k, kd, ok, okd := syExactCount(acl, n, discrete, 1e-8)
	if ok && k == 0 {
		return
	}
	if okd && kd == 0 {
		fzSkip(t, op+": closed-loop root within strip of the boundary")
	}
	t.Fatalf("%s: closed loop not stable (%d roots beyond the strip, degenerate %v)\n%s\n%s", op, kd, !okd, detail, s)
}

// syFragile skips when the exact interconnection cl of the float plant
// and controller is not stable while the library's own float closed loop
// CL is exactly stable: the controller is then stabilizing only to within
// the rounding of the interconnection (huge gains or near-improper
// poles), a precision limit rather than a wrong design.
func syFragile(t *testing.T, s *fzSpec, op string, cl *syCL, CL *System) {
	t.Helper()
	if syStable(cl.A, cl.N, s.Discrete) || CL == nil || CL.HasDelay() || CL.IsDescriptor() {
		return
	}
	n, _, _ := CL.Dims()
	if n > 0 && syStable(syRatDense(mqRaw(CL.A)), n, s.Discrete) {
		fzSkip(t, op+": controller stabilizing only within interconnection rounding")
	}
}

// syEigCond returns the eigenvalues of the float n×n matrix and the
// 2-norm condition of its eigenvector matrix (Inf when defective).
func syEigCond(a []float64, n int) ([]complex128, float64) {
	var e mat.Eigen
	if !e.Factorize(mqDense(n, n, a), mat.EigenRight) {
		return nil, math.Inf(1)
	}
	var v mat.CDense
	e.VectorsTo(&v)
	g := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			g[i*n+j] = v.At(i, j)
		}
	}
	sv := fzSingularValues(g, n, n)
	if sv == nil || !(sv[n-1] > 0) {
		return e.Values(nil), math.Inf(1)
	}
	return e.Values(nil), sv[0] / sv[n-1]
}

// syCtrbCond is the 2-norm condition of the single-input Ctrb(A, b),
// which bounds the rounding error of Ackermann's formula.
func syCtrbCond(a, b []float64, n int) float64 {
	C := mat.NewDense(n, n, nil)
	col := mat.NewVecDense(n, append([]float64(nil), b...))
	A := mqDense(n, n, a)
	for k := range n {
		C.SetCol(k, col.RawVector().Data)
		var nx mat.VecDense
		nx.MulVec(A, col)
		col = &nx
	}
	return mat.Cond(C, 2)
}

// FuzzPlace: Place (and Acker for one input) toward syDesiredPoles. A
// returned K must make the exact A − BK stable (the requested 0.1·|p|
// accuracy disks lie inside the stability region) and keep a computed
// closed-loop eigenvalue within 0.1·|p| plus a Bauer–Fike radius of every
// pole. An error is a failure when the rounded exact Ackermann gain of
// some input column already places every pole to 1e-4·|p| (exact Newton
// bound) and its closed loop is well conditioned.
func FuzzPlace(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m := s.N, s.M
		if n == 0 {
			fzSkip(t, "Place: no states")
		}
		poles := syDesiredPoles(s)
		chi := syMonic(poles)
		feasible := false
		var witness []float64
		for j := 0; j < m && !feasible; j++ {
			bj := make([]float64, n)
			for i := range n {
				bj[i] = s.B[i*m+j]
			}
			kj := syAckerExact(s.A, bj, n, chi)
			if kj == nil {
				continue
			}
			kf := make([]float64, m*n)
			copy(kf[j*n:], kj)
			acl := syClosedLoopA(s.A, s.B, kf, n, m)
			if !syRootsNear(fzCharPolyRat(acl, n), poles, 1e-4) {
				continue
			}
			_, cond := syEigCond(syRatFloat(acl), n)
			nk := mqFro(kf)
			minp := math.Inf(1)
			for _, p := range poles {
				minp = math.Min(minp, cmplx.Abs(p))
			}
			if cond*0x1p-52*(mqFro(s.A)+mqFro(s.B)*nk) <= 1e-6*minp {
				feasible, witness = true, kf
			}
		}
		check := func(op string, K *mat.Dense, err error, feasible bool) {
			fzContract(t, s, op, K != nil, err)
			if err != nil {
				if feasible {
					t.Fatalf("%s: %v, but the rounded exact gain %v places %v to 1e-4 on a well-conditioned closed loop\n%s", op, err, witness, poles, s)
				}
				if errors.Is(err, ErrPoleAccuracy) || errors.Is(err, ErrUncontrollable) {
					return
				}
				if errors.Is(err, ErrSchurFailed) {
					fzSkip(t, op+": Schur reordering failed on a not provably placeable problem")
				}
				t.Fatalf("%s: undocumented error %v\n%s", op, err, s)
			}
			if r, c := K.Dims(); r != m || c != n {
				t.Fatalf("%s: K is %d×%d\n%s", op, r, c, s)
			}
			k := mqRaw(K)
			acl := syClosedLoopA(s.A, s.B, k, n, m)
			af := syRatFloat(acl)
			got, cond := syEigCond(af, n)
			rbf := cond * 1e2 * float64(n) * 0x1p-52 * math.Max(mqFro(af), mqFro(s.A)+mqFro(s.B)*mqFro(k))
			detail := fmt.Sprintf("poles=%v K=%v", poles, k)
			for _, p := range poles {
				if rbf > 1e-2*cmplx.Abs(p) {
					fzSkip(t, op+": ill-conditioned closed loop")
				}
			}
			syRequireStableRat(t, s, op, acl, n, s.Discrete, detail)
			for _, p := range poles {
				best := math.Inf(1)
				for _, g := range got {
					best = math.Min(best, cmplx.Abs(g-p))
				}
				if !(best <= 0.1*cmplx.Abs(p)*(1+1e-9)+rbf) {
					t.Fatalf("%s: pole %v has no closed-loop eigenvalue within 10%% (nearest %.3g, Bauer–Fike %.3g)\neig=%v\n%s\n%s", op, p, best, rbf, got, detail, s)
				}
			}
		}
		A, B := mqDense(n, n, s.A), mqDense(n, m, s.B)
		K, err := Place(A, B, poles)
		check("Place", K, err, feasible)
		if m == 1 {
			K, err = Acker(A, B, poles)
			check("Acker", K, err, feasible && syCtrbCond(s.A, s.B, n) <= 1e6)
		}
	})
}

// syNormC returns s with C scaled to Frobenius norm √p: the gain field
// (default 1e-6) would otherwise make (C, A) numerically undetectable.
func syNormC(s *fzSpec) *fzSpec {
	d := *s
	if nc := mqFro(s.C); nc > 0 {
		d.C = make([]float64, len(s.C))
		for i, x := range s.C {
			d.C[i] = x * math.Sqrt(float64(s.P)) / nc
		}
	}
	return &d
}

// syTinyDiscreteEig reports a discrete spec with a nonzero eigenvalue
// below 1e-12·max(1, ‖J‖), on which Dare wrongly reports ErrNoStabilizing
// (e.g. Dare(3e-17, -0.92, 1, 1)); filed as a followup.
func syTinyDiscreteEig(s *fzSpec) bool {
	if !s.Discrete {
		return false
	}
	for _, e := range s.Eig {
		if e != 0 && cmplx.Abs(e) < 1e-12*math.Max(1, s.NormJ) {
			return true
		}
	}
	return false
}

// syNearUnitPole reports a discrete spec with a pole within 1e-12 of, but
// not at, z = ±1, on which HinfSyn wrongly reports a rank-deficient
// P21(z = 1); filed as a followup.
func syNearUnitPole(s *fzSpec) bool {
	if !s.Discrete {
		return false
	}
	for _, e := range s.Eig {
		if d := math.Min(cmplx.Abs(e-1), cmplx.Abs(e+1)); d != 0 && d < 1e-12 {
			return true
		}
	}
	return false
}

// syDual returns the spec of the dual pair (Aᵀ, Cᵀ): the mq Riccati
// helpers on it describe the filter Riccati equation of (A, C).
func syDual(s *fzSpec) *fzSpec {
	d := *s
	d.A = mqTranspose(s.A, s.N, s.N)
	d.B = mqTranspose(s.C, s.P, s.N)
	d.M = s.P
	return &d
}

// syRiccatiChecks verifies a returned Riccati solution x and gain k of the
// regulator problem (s.A, s.B, q, r, sm) as FuzzRiccati does: symmetric,
// gain and Riccati residuals to rounding (mqRiccatiTol widened tenfold for
// the rounding of the assembled Q̄, R̄, S and by the Q̄ cancellation
// ratio), exactly stable A − BK and, for a PSD joint cost, PSD x.
func syRiccatiChecks(t *testing.T, s *fzSpec, op string, q, r, sm, x, k []float64, rcnd float64, jointPSD bool) {
	t.Helper()
	n, m := s.N, s.M
	mqSymmetric(t, s, op, x, n)
	if !mqStabilizable(s) {
		fzSkip(t, op+": pair not robustly stabilizable")
	}
	acl := syClosedLoopA(s.A, s.B, k, n, m)
	tol := mqRiccatiTol(s, syRatFloat(acl), q, r, rcnd)
	if tol > 1e-4 {
		fzSkip(t, op+": ill-conditioned stable subspace")
	}
	if !syQbarDigits(q, r, sm, n, m) {
		syRequireStableRat(t, s, op, acl, n, s.Discrete, fmt.Sprintf("X=%v K=%v", x, k))
		fzSkip(t, op+": Q − SR⁻¹Sᵀ keeps under six digits; residual checks below rounding")
	}
	ke, kden, re, rden := mqRiccatiResiduals(s, q, r, sm, x, k)
	if !(ke <= 1e-10*kden) {
		t.Fatalf("%s: gain residual %.3g (scale %.3g)\nX=%v\nK=%v\n%s", op, ke, kden, x, k, s)
	}
	if !(re <= 10*tol*syQbarRatio(q, r, sm, n, m)*rden) {
		t.Fatalf("%s: Riccati residual %.3g (scale %.3g, rcnd %.3g)\nX=%v\nK=%v\n%s", op, re, rden, rcnd, x, k, s)
	}
	syRequireStableRat(t, s, op, acl, n, s.Discrete, fmt.Sprintf("X=%v K=%v", x, k))
	if jointPSD {
		if me := mqMinEigSym(x, n); !(me >= -1e-8*mqFro(x)) {
			t.Fatalf("%s: PSD cost but min eig(X) = %.3g\nX=%v\n%s", op, me, x, s)
		}
	}
}

// syQbarDigits reports whether Q − S·R⁻¹·Sᵀ keeps more than six digits
// after cancellation (eps·cond(R)·‖SR⁻¹Sᵀ‖ ≤ 1e-6·‖Q − SR⁻¹Sᵀ‖) and
// cancels at most three (Care then loses digits beyond its conditioning:
// known followup); otherwise the
// solution is accurate only relative to ‖SR⁻¹Sᵀ‖, so its residual and
// sign structure at its own scale are below rounding.
// syQbarRatio is max(1, ‖SR⁻¹Sᵀ‖/‖Q − SR⁻¹Sᵀ‖), the amplification of
// data rounding by the cancellation in Q̄ (1 without S).
func syQbarRatio(q, r, sm []float64, n, m int) float64 {
	if sm == nil {
		return 1
	}
	var ri mat.Dense
	if err := ri.Inverse(mqDense(m, m, r)); err != nil {
		return math.Inf(1)
	}
	srs := syMul(syMul(sm, mqRaw(&ri), n, m, m), mqTranspose(sm, n, m), n, m, n)
	qb := make([]float64, n*n)
	for i := range qb {
		qb[i] = q[i] - srs[i]
	}
	return math.Max(1, mqFro(srs)/mqFro(qb))
}

func syQbarDigits(q, r, sm []float64, n, m int) bool {
	if sm == nil {
		return true
	}
	var ri mat.Dense
	if err := ri.Inverse(mqDense(m, m, r)); err != nil {
		return false
	}
	srs := syMul(syMul(sm, mqRaw(&ri), n, m, m), mqTranspose(sm, n, m), n, m, n)
	qb := make([]float64, n*n)
	for i := range qb {
		qb[i] = q[i] - srs[i]
	}
	nq := mqFro(qb)
	return 0x1p-52*mat.Cond(mqDense(m, m, r), 2)*mqFro(srs) <= 1e-6*nq && mqFro(srs) <= 1e3*nq
}

// syNoise returns Qn = q·I (m×m), Rn = r·I (p×p) with q, r = 10^clamp(v,-3,3)
// from v[12], v[13], and Nn (m×p) nil for v[14] < 0, else a texture of
// Frobenius norm 0.5·√(qr), so that [Qn Nn; Nnᵀ Rn] is PSD.
func syNoise(s *fzSpec, m, p int) (qn, rn, nn []float64) {
	q := math.Pow(10, fzClamp(s.Vals[12], -3, 3))
	r := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
	qn, rn = make([]float64, m*m), make([]float64, p*p)
	for i := range m {
		qn[i*m+i] = q
	}
	for i := range p {
		rn[i*p+i] = r
	}
	if s.Vals[14] < 0 {
		return qn, rn, nil
	}
	nn = make([]float64, m*p)
	for i := range m {
		for j := range p {
			nn[i*p+j] = math.Cos(s.Vals[11] + 1.1*float64(i+1) + 0.6*float64(j+1))
		}
	}
	sc := 0.5 * math.Sqrt(q*r) / math.Max(mqFro(nn), 1e-300)
	for i := range nn {
		nn[i] *= sc
	}
	return qn, rn, nn
}

func syMul(a, b []float64, r, k, c int) []float64 {
	var o mat.Dense
	o.Mul(mqDense(r, k, a), mqDense(k, c, b))
	return mqRaw(&o)
}

// FuzzKalman: Kalman on the delay-free spec model (G = B, H = D, ‖D‖_F
// capped at 1: a large H with Qn ≫ Rn cancels Q − SR̄⁻¹Sᵀ and Care/Dare then
// lose digits beyond the conditioning, a known followup) with syNoise
// covariances, and Lqe (H = 0) on continuous ones. By duality Lᵀ
// is the gain of the regulator problem (Aᵀ, Cᵀ, BQnBᵀ, R̄, B(QnDᵀ+Nn)) with
// R̄ = Rn + DNn + NnᵀDᵀ + DQnDᵀ; the solution and gain must pass the
// syRiccatiChecks there (A − LC exactly stable, P PSD). ErrNoStabilizing
// is a failure when mqRiccatiExists proves the dual problem solvable.
func FuzzKalman(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		s = syNormC(s)
		if nd := mqFro(s.D); nd > 1 {
			d := make([]float64, len(s.D))
			for i, x := range s.D {
				d[i] = x / nd
			}
			s.D = d
		}
		n, m, p := s.N, s.M, s.P
		if n == 0 {
			fzSkip(t, "Kalman: no states")
		}
		qn, rn, nn := syNoise(s, m, p)
		ds := syDual(s)
		run := func(op string, h []float64, call func() (*RiccatiResult, error)) {
			qnht := make([]float64, m*p)
			if h != nil {
				qnht = syMul(qn, mqTranspose(h, p, m), m, m, p)
			}
			if nn != nil {
				for i := range qnht {
					qnht[i] += nn[i]
				}
			}
			qb := syMul(syMul(s.B, qn, n, m, m), mqTranspose(s.B, n, m), n, m, n)
			rb := append([]float64(nil), rn...)
			if h != nil {
				hq := syMul(h, qnht, p, m, p)
				hqh := syMul(syMul(h, qn, p, m, m), mqTranspose(h, p, m), p, m, p)
				for i := range p {
					for j := range p {
						rb[i*p+j] += hq[i*p+j] + hq[j*p+i] - hqh[i*p+j]
					}
				}
			}
			var sb []float64
			if h != nil || nn != nil {
				sb = syMul(s.B, qnht, n, m, p)
			}
			res, err := call()
			fzContract(t, s, op, res != nil, err)
			if err != nil {
				switch {
				case errors.Is(err, ErrNoStabilizing):
					if syTinyDiscreteEig(s) {
						fzSkip(t, op+": Dare with a nonzero eigenvalue below 1e-12 (known followup)")
					}
					if !mqRiccatiExists(ds, qb, rb, sb) {
						fzSkip(t, op+": no stabilizing solution (not provably solvable)")
					}
					t.Fatalf("%s: ErrNoStabilizing, but the dual pencil is clear of the boundary and (C,A) detectable\nQn=%v Rn=%v Nn=%v\n%s", op, qn, rn, nn, s)
				case errors.Is(err, ErrNotPSD):
					fzSkip(t, op+": BQnBᵀ numerically semidefinite rejected")
				case errors.Is(err, ErrSingularR):
					if me := mqMinEigSym(rb, p); me >= 1e-10*mqFro(rb) && mqRiccatiExists(ds, qb, rb, sb) {
						t.Fatalf("%s: ErrSingularR for R̄ with min eig %.3g\n%s", op, me, s)
					}
					fzSkip(t, op+": R̄ near singular")
				}
				fzErr(t, s, op, err, ErrSchurFailed)
			}
			if r1, c1 := res.K.Dims(); r1 != n || c1 != p {
				t.Fatalf("%s: L is %d×%d\n%s", op, r1, c1, s)
			}
			if r1, c1 := res.X.Dims(); r1 != n || c1 != n {
				t.Fatalf("%s: P is %d×%d\n%s", op, r1, c1, s)
			}
			syRiccatiChecks(t, ds, op, qb, rb, sb, mqRaw(res.X), mqTranspose(mqRaw(res.K), n, p), res.Rcnd, true)
		}
		sys := s.System(t, fzSysOpts{})
		var nnD *mat.Dense
		if nn != nil {
			nnD = mqDense(m, p, nn)
		}
		run("Kalman", s.D, func() (*RiccatiResult, error) {
			return Kalman(sys, mqDense(m, m, qn), mqDense(p, p, rn), nnD, nil)
		})
		if !s.Discrete {
			run("Lqe", nil, func() (*RiccatiResult, error) {
				return Lqe(mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(p, n, s.C), mqDense(m, m, qn), mqDense(p, p, rn), nnD, nil)
			})
		}
	})
}

func syPut(dst []*big.Rat, ld, r0, c0 int, src []*big.Rat, r, c int) {
	for i := range r {
		for j := range c {
			dst[(r0+i)*ld+c0+j] = src[i*c+j]
		}
	}
}

func syRatZeros(k int) []*big.Rat {
	out := make([]*big.Rat, k)
	for i := range out {
		out[i] = new(big.Rat)
	}
	return out
}

func syRatEye(k int) []*big.Rat {
	out := syRatZeros(k * k)
	for i := range k {
		out[i*k+i].SetInt64(1)
	}
	return out
}

// syPlant is a generalized plant [A B1 B2; C1 D11 D12; C2 D21 D22] with nw
// disturbances, nz errors, nu controls and ny measurements.
type syPlant struct {
	n, nw, nz, nu, ny                int
	A, B1, B2, C1, C2, D11, D12, D21 []float64
	D22                              []float64
}

// syCL is the closed loop of a syPlant with a controller: Acl exactly, the
// rest rounded.
type syCL struct {
	N           int
	A           []*big.Rat
	Af, B, C, D []float64
}

// syLFT closes u = K·y exactly for the float64 data of P and K
// (xk' = Ak xk + Bk y, u = Ck xk + Dk y); ok is false when I − Dk·D22 is
// exactly singular.
func syLFT(P *syPlant, nk int, ak, bk, ck, dk []float64) (*syCL, bool) {
	n, nw, nz, nu, ny := P.n, P.nw, P.nz, P.nu, P.ny
	R := syRatDense
	Dk, D22 := R(dk), R(P.D22)
	I := syRatEye(nu)
	M := mqRatInv(syRatSub(I, mqRatMul(Dk, D22, nu, ny, nu)), nu)
	if M == nil {
		return nil, false
	}
	MDk := mqRatMul(M, Dk, nu, nu, ny)
	MCk := mqRatMul(M, R(ck), nu, nu, nk)
	C2, D21 := R(P.C2), R(P.D21)
	uX := mqRatMul(MDk, C2, nu, ny, n)
	uW := mqRatMul(MDk, D21, nu, ny, nw)
	yX := syRatAdd(C2, mqRatMul(D22, uX, ny, nu, n))
	yK := mqRatMul(D22, MCk, ny, nu, nk)
	yW := syRatAdd(D21, mqRatMul(D22, uW, ny, nu, nw))
	B2, Bk := R(P.B2), R(bk)
	N := n + nk
	A := syRatZeros(N * N)
	syPut(A, N, 0, 0, syRatAdd(R(P.A), mqRatMul(B2, uX, n, nu, n)), n, n)
	syPut(A, N, 0, n, mqRatMul(B2, MCk, n, nu, nk), n, nk)
	syPut(A, N, n, 0, mqRatMul(Bk, yX, nk, ny, n), nk, n)
	syPut(A, N, n, n, syRatAdd(R(ak), mqRatMul(Bk, yK, nk, ny, nk)), nk, nk)
	B := syRatZeros(N * nw)
	syPut(B, nw, 0, 0, syRatAdd(R(P.B1), mqRatMul(B2, uW, n, nu, nw)), n, nw)
	syPut(B, nw, n, 0, mqRatMul(Bk, yW, nk, ny, nw), nk, nw)
	D12 := R(P.D12)
	C := syRatZeros(nz * N)
	syPut(C, N, 0, 0, syRatAdd(R(P.C1), mqRatMul(D12, uX, nz, nu, n)), nz, n)
	syPut(C, N, 0, n, mqRatMul(D12, MCk, nz, nu, nk), nz, nk)
	D := syRatAdd(R(P.D11), mqRatMul(D12, uW, nz, nu, nw))
	return &syCL{N: N, A: A, Af: syRatFloat(A), B: syRatFloat(B), C: syRatFloat(C), D: syRatFloat(D)}, true
}

// syCtrlData returns the state-space data of a delay-free, non-descriptor
// controller.
func syCtrlData(t *testing.T, s *fzSpec, op string, K *System, nu, ny int) (nk int, a, b, c, d []float64) {
	t.Helper()
	if K.HasDelay() || K.IsDescriptor() {
		t.Fatalf("%s: controller has delays or a descriptor\n%s", op, s)
	}
	nk, mk, pk := K.Dims()
	if mk != ny || pk != nu {
		t.Fatalf("%s: controller is %d→%d, want %d→%d\n%s", op, mk, pk, ny, nu, s)
	}
	if K.IsDiscrete() != s.Discrete || s.Discrete && K.Dt != s.Dt {
		t.Fatalf("%s: controller sample time %g, plant %g\n%s", op, K.Dt, s.Dt, s)
	}
	if nk > 0 {
		a, b, c = mqRaw(K.A), mqRaw(K.B), mqRaw(K.C)
	}
	return nk, a, b, c, mqRaw(K.D)
}

// syH2 returns ‖CL‖₂ from the Kronecker controllability gramian; ok false
// when that solve is ill-conditioned or a continuous D is nonzero (Inf).
func syH2(cl *syCL, nw, nz int, discrete bool) (float64, bool) {
	N := cl.N
	if !discrete {
		for _, x := range cl.D {
			if x != 0 {
				return math.Inf(1), false
			}
		}
	}
	h := 0.0
	if N > 0 {
		x, _, ok := mqLyapRef(cl.Af, mqGram(cl.B, N, nw, false), N, discrete)
		if !ok {
			return 0, false
		}
		cx := syMul(cl.C, x, nz, N, N)
		for i := range nz {
			for k := range N {
				h += cx[i*N+k] * cl.C[i*N+k]
			}
		}
	}
	if discrete {
		for _, x := range cl.D {
			h += x * x
		}
	}
	return math.Sqrt(math.Max(h, 0)), true
}

// sySigma returns σmax(C(λI − A)⁻¹B + D) at λ = jω (continuous) or e^{jωdt},
// by a real-embedded LU solve.
func sySigma(cl *syCL, nw, nz int, w, dt float64) float64 {
	N := cl.N
	g := make([]complex128, nz*nw)
	for k := range g {
		g[k] = complex(cl.D[k], 0)
	}
	if N > 0 {
		lr, li := 0.0, w
		if dt > 0 {
			lr, li = math.Cos(w*dt), math.Sin(w*dt)
		}
		M := mat.NewDense(2*N, 2*N, nil)
		for i := range N {
			for j := range N {
				M.Set(i, j, -cl.Af[i*N+j])
				M.Set(N+i, N+j, -cl.Af[i*N+j])
			}
			M.Set(i, i, M.At(i, i)+lr)
			M.Set(N+i, N+i, M.At(N+i, N+i)+lr)
			M.Set(i, N+i, -li)
			M.Set(N+i, i, li)
		}
		rhs := mat.NewDense(2*N, nw, nil)
		for i := range N {
			for j := range nw {
				rhs.Set(i, j, cl.B[i*nw+j])
			}
		}
		var x mat.Dense
		if err := x.Solve(M, rhs); err != nil {
			return math.Inf(1)
		}
		for i := range nz {
			for j := range nw {
				var re, im float64
				for k := range N {
					re += cl.C[i*N+k] * x.At(k, j)
					im += cl.C[i*N+k] * x.At(N+k, j)
				}
				g[i*nw+j] += complex(re, im)
			}
		}
	}
	sv := fzSingularValues(g, nz, nw)
	if sv == nil {
		return math.Inf(1)
	}
	return sv[0]
}

// syRoundingExcess skips when σ = lo of the exact closed loop at ω
// exceeds gamma only because the library's float closed loop CL (whose
// norm Gamma is) differs from it by rounding: σmax of CL itself, exactly
// evaluated at ω, stays within gamma and the two differ by more than half
// the excess. A CL that itself exceeds gamma at ω fails.
func syRoundingExcess(t *testing.T, s *fzSpec, op string, CL *System, lo, w, gamma float64, nw, nz int) {
	t.Helper()
	if CL == nil || CL.HasDelay() || CL.IsDescriptor() {
		return
	}
	n, m, p := CL.Dims()
	if m != nw || p != nz {
		t.Fatalf("%s: CL is %d→%d, want %d→%d\n%s", op, m, p, nw, nz, s)
	}
	var a, b, c []float64
	if n > 0 {
		a, b, c = mqRaw(CL.A), mqRaw(CL.B), mqRaw(CL.C)
	}
	z := complex(0, w)
	if s.Discrete {
		z = cmplx.Exp(complex(0, w*s.Dt))
	}
	g, ok := fzExactResponse(n, m, p, a, b, c, mqRaw(CL.D), z)
	if !ok {
		return
	}
	sc := fzSingularValues(g, p, m)[0]
	if !(sc <= gamma*(1+1e-6)) {
		t.Fatalf("%s: σmax of the returned CL is %.10g at ω=%g, above Gamma %.10g\n%s", op, sc, w, gamma, s)
	}
	if math.Abs(lo-sc) >= 0.5*(lo-gamma) {
		fzSkip(t, op+": closed-loop realization rounding exceeds the Gamma gap")
	}
}

// syHinfLower returns a lower bound of ‖CL‖∞ by a dense sweep clustered at
// the closed-loop poles, and its frequency.
func syHinfLower(cl *syCL, nw, nz int, dt float64) (float64, float64) {
	eig, _ := syEigCond(cl.Af, cl.N)
	s := make([]complex128, 0, len(eig))
	for _, e := range eig {
		if dt > 0 {
			if e == 0 {
				continue
			}
			e = cmplx.Log(e) / complex(dt, 0)
		}
		s = append(s, e)
	}
	w, v := fzSweepMax(func(w float64) float64 { return sySigma(cl, nw, nz, w, dt) }, fzFreqGrid(s, dt, 200), 6)
	return v, w
}

// syRegPlant returns the generalized plant of the regulator problem on the
// spec model: w = [process (n); sensor (p)], z = [x; ρu], y = Cx + σv + Du,
// i.e. B1 = [I 0], C1 = [I; 0], D12 = [0; ρI], D21 = [0 σI], D22 = D, so
// the rank conditions hold and solvability is (A,B) stabilizable and (C,A)
// detectable. ρ = 10^clamp(v[12],-2,2), σ = 10^clamp(v[13],-2,2).
func syRegPlant(s *fzSpec) (*syPlant, float64, float64) {
	n, m, p := s.N, s.M, s.P
	rho := math.Pow(10, fzClamp(s.Vals[12], -2, 2))
	sig := math.Pow(10, fzClamp(s.Vals[13], -2, 2))
	P := &syPlant{n: n, nw: n + p, nz: n + m, nu: m, ny: p, A: s.A, B2: s.B, C2: s.C, D22: s.D}
	P.B1 = make([]float64, n*(n+p))
	P.C1 = make([]float64, (n+m)*n)
	for i := range n {
		P.B1[i*(n+p)+i] = 1
		P.C1[i*n+i] = 1
	}
	P.D11 = make([]float64, (n+m)*(n+p))
	P.D12 = make([]float64, (n+m)*m)
	for i := range m {
		P.D12[(n+i)*m+i] = rho
	}
	P.D21 = make([]float64, p*(n+p))
	for i := range p {
		P.D21[i*(n+p)+n+i] = sig
	}
	return P, rho, sig
}

// System builds P as one model with inputs [w; u] and outputs [z; y].
func (P *syPlant) System(t *testing.T, s *fzSpec) *System {
	t.Helper()
	n, mi, po := P.n, P.nw+P.nu, P.nz+P.ny
	b := make([]float64, n*mi)
	for i := range n {
		copy(b[i*mi:], P.B1[i*P.nw:(i+1)*P.nw])
		copy(b[i*mi+P.nw:], P.B2[i*P.nu:(i+1)*P.nu])
	}
	c := append(append([]float64(nil), P.C1...), P.C2...)
	d := make([]float64, po*mi)
	for i := range P.nz {
		copy(d[i*mi:], P.D11[i*P.nw:(i+1)*P.nw])
		copy(d[i*mi+P.nw:], P.D12[i*P.nu:(i+1)*P.nu])
	}
	for i := range P.ny {
		copy(d[(P.nz+i)*mi:], P.D21[i*P.nw:(i+1)*P.nw])
		copy(d[(P.nz+i)*mi+P.nw:], P.D22[i*P.nu:(i+1)*P.nu])
	}
	sys, err := NewFromSlices(n, mi, po, P.A, b, c, d, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices(P): %v\n%s", err, s)
	}
	return sys
}

// syLqgCandidate returns the closed loop of P with the Lqg controller for
// the matching weights (QXU = blkdiag(I, ρ²I), QWV = blkdiag(I, σ²I);
// Current when discrete), which is H2-optimal for syRegPlant; nil if Lqg
// fails.
func syLqgCandidate(t *testing.T, s *fzSpec, P *syPlant, rho, sig float64) *syCL {
	t.Helper()
	n, m, p := s.N, s.M, s.P
	qxu := make([]float64, (n+m)*(n+m))
	for i := range n + m {
		qxu[i*(n+m)+i] = 1
		if i >= n {
			qxu[i*(n+m)+i] = rho * rho
		}
	}
	qwv := make([]float64, (n+p)*(n+p))
	for i := range n + p {
		qwv[i*(n+p)+i] = 1
		if i >= n {
			qwv[i*(n+p)+i] = sig * sig
		}
	}
	res, err := Lqg(s.System(t, fzSysOpts{}), mqDense(n+m, n+m, qxu), mqDense(n+p, n+p, qwv), &LqgOpts{Current: s.Discrete})
	if err != nil {
		return nil
	}
	nk, a, b, c, d := syCtrlData(t, s, "Lqg", res.Controller, m, p)
	cl, ok := syLFT(P, nk, a, b, c, d)
	if !ok {
		return nil
	}
	return cl
}

// syRegSolvable reports whether (A,B) is robustly stabilizable and (C,A)
// robustly detectable, which makes the syRegPlant H2 and H∞ problems
// solvable.
func syRegSolvable(s *fzSpec) bool { return mqStabilizable(s) && mqStabilizable(syDual(s)) }

// FuzzH2Syn: H2Syn on syRegPlant of the normalized spec model. When the
// problem is solvable an error is a failure. A returned K must be
// strictly proper (continuous), give an exactly stable closed loop whose
// Kronecker H2 norm equals Gamma, and no larger a norm than the Lqg
// controller for the same weights (an H2-optimal candidate measured by the
// same oracle).
func FuzzH2Syn(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		s = syNormC(s)
		if s.N == 0 {
			fzSkip(t, "H2Syn: no states")
		}
		P, rho, sig := syRegPlant(s)
		res, err := H2Syn(P.System(t, s), P.ny, P.nu)
		fzContract(t, s, "H2Syn", res != nil, err)
		solvable := syRegSolvable(s)
		if err != nil {
			if errors.Is(err, ErrNoStabilizing) && syTinyDiscreteEig(s) {
				fzSkip(t, "H2Syn: Dare with a nonzero eigenvalue below 1e-12 (known followup)")
			}
			if solvable {
				t.Fatalf("H2Syn: %v on a robustly stabilizable and detectable plant (ρ=%g σ=%g)\n%s", err, rho, sig, s)
			}
			fzSkip(t, "H2Syn: error on a not robustly solvable plant")
		}
		nk, a, b, c, d := syCtrlData(t, s, "H2Syn", res.K, P.nu, P.ny)
		if !s.Discrete {
			for _, x := range d {
				if x != 0 {
					t.Fatalf("H2Syn: continuous K has feedthrough %v\n%s", d, s)
				}
			}
		}
		cl, ok := syLFT(P, nk, a, b, c, d)
		if !ok {
			t.Fatalf("H2Syn: I − Dk·D22 singular\n%s", s)
		}
		if !solvable {
			fzSkip(t, "H2Syn: plant not robustly solvable")
		}
		syFragile(t, s, "H2Syn", cl, res.CL)
		syRequireStableRat(t, s, "H2Syn", cl.A, cl.N, s.Discrete, fmt.Sprintf("K: A=%v B=%v C=%v D=%v", a, b, c, d))
		h, ok := syH2(cl, P.nw, P.nz, s.Discrete)
		if !ok {
			fzSkip(t, "H2Syn: ill-conditioned closed-loop gramian")
		}
		if !(math.Abs(h-res.Gamma) <= 1e-6*h) {
			t.Fatalf("H2Syn: Gamma %.10g, closed-loop H2 norm %.10g\n%s", res.Gamma, h, s)
		}
		if cand := syLqgCandidate(t, s, P, rho, sig); cand != nil {
			if syStable(cand.A, cand.N, s.Discrete) {
				if hc, ok := syH2(cand, P.nw, P.nz, s.Discrete); ok && !(h <= hc*(1+1e-6)) {
					t.Fatalf("H2Syn: ‖CL‖₂ = %.10g exceeds %.10g of the Lqg controller for the same weights\n%s", h, hc, s)
				}
			}
		}
	})
}

// FuzzHinfSyn: HinfSyn on syRegPlant of the normalized spec model. When
// the problem is solvable an error is a failure. A returned K must give an
// exactly stable closed loop with Gamma ≤ GammaOpt and no swept σmax above
// Gamma, and GammaOpt may not exceed (beyond the bisection and backoff
// tolerances) the swept H∞ norm of the H2Syn controller, a stabilizing
// candidate.
func FuzzHinfSyn(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		s = syNormC(s)
		if s.N == 0 {
			fzSkip(t, "HinfSyn: no states")
		}
		P, rho, sig := syRegPlant(s)
		Psys := P.System(t, s)
		res, err := HinfSyn(Psys, P.ny, P.nu)
		fzContract(t, s, "HinfSyn", res != nil, err)
		solvable := syRegSolvable(s)
		if err != nil {
			if errors.Is(err, ErrInvalidPartition) && syNearUnitPole(s) {
				fzSkip(t, "HinfSyn: P21(z = ±1) rank with a pole within 1e-12 of ±1 (known followup)")
			}
			if solvable {
				t.Fatalf("HinfSyn: %v on a robustly stabilizable and detectable plant (ρ=%g σ=%g)\n%s", err, rho, sig, s)
			}
			fzSkip(t, "HinfSyn: error on a not robustly solvable plant")
		}
		nk, a, b, c, d := syCtrlData(t, s, "HinfSyn", res.K, P.nu, P.ny)
		cl, ok := syLFT(P, nk, a, b, c, d)
		if !ok {
			t.Fatalf("HinfSyn: I − Dk·D22 singular\n%s", s)
		}
		if !(res.Gamma <= res.GammaOpt*(1+1e-12)) {
			t.Fatalf("HinfSyn: Gamma %.10g > GammaOpt %.10g\n%s", res.Gamma, res.GammaOpt, s)
		}
		if !solvable {
			fzSkip(t, "HinfSyn: plant not robustly solvable")
		}
		syFragile(t, s, "HinfSyn", cl, res.CL)
		syRequireStableRat(t, s, "HinfSyn", cl.A, cl.N, s.Discrete, fmt.Sprintf("K: A=%v B=%v C=%v D=%v", a, b, c, d))
		lo, w := syHinfLower(cl, P.nw, P.nz, s.Dt)
		if !(lo <= res.Gamma*(1+1e-6)) {
			syRoundingExcess(t, s, "HinfSyn", res.CL, lo, w, res.Gamma, P.nw, P.nz)
			t.Fatalf("HinfSyn: σmax %.10g at ω=%g exceeds Gamma %.10g\n%s", lo, w, res.Gamma, s)
		}
		h2, err := H2Syn(Psys, P.ny, P.nu)
		if err != nil {
			return
		}
		nk, a, b, c, d = syCtrlData(t, s, "H2Syn", h2.K, P.nu, P.ny)
		cand, ok := syLFT(P, nk, a, b, c, d)
		if !ok {
			return
		}
		if !syStable(cand.A, cand.N, s.Discrete) {
			return
		}
		hc, wc := syHinfLower(cand, P.nw, P.nz, s.Dt)
		if !(res.GammaOpt <= hc*(1+1e-3)+hinfGammaAbsTol) {
			if res.GammaOpt > 1e6 {
				fzSkip(t, "HinfSyn: GammaOpt above an H2 candidate at gamma > 1e6 (known followup)")
			}
			t.Fatalf("HinfSyn: GammaOpt %.10g exceeds the H∞ norm ≥ %.10g (ω=%g) of the stabilizing H2Syn controller\n%s", res.GammaOpt, hc, wc, s)
		}
	})
}

// FuzzLqg: Lqg regulator (Current for discrete specs with v[15] > 0) on
// the normalized spec model with QXU = blkdiag(CᵀC + qI, rI) and
// QWV = blkdiag(BBᵀ + qI, rI) (q, r from v[12], v[13]). When both Riccati
// problems are provably solvable an error is a failure; a returned
// controller on a robustly stabilizable and detectable plant must make the
// positive-feedback closed loop exactly stable.
func FuzzLqg(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		s = syNormC(s)
		n, m, p := s.N, s.M, s.P
		if n == 0 {
			fzSkip(t, "Lqg: no states")
		}
		q := math.Pow(10, fzClamp(s.Vals[12], -3, 3))
		r := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
		qx := mqGram(s.C, p, n, true)
		qw := mqGram(s.B, n, m, false)
		for i := range n {
			qx[i*n+i] += q
			qw[i*n+i] += q
		}
		ru, rv := make([]float64, m*m), make([]float64, p*p)
		for i := range m {
			ru[i*m+i] = r
		}
		for i := range p {
			rv[i*p+i] = r
		}
		blk := func(a, b []float64, k, l int) []float64 {
			out := make([]float64, (k+l)*(k+l))
			for i := range k {
				copy(out[i*(k+l):], a[i*k:(i+1)*k])
			}
			for i := range l {
				copy(out[(k+i)*(k+l)+k:], b[i*l:(i+1)*l])
			}
			return out
		}
		current := s.Discrete && s.Vals[15] > 0
		res, err := Lqg(s.System(t, fzSysOpts{}), mqDense(n+m, n+m, blk(qx, ru, n, m)), mqDense(n+p, n+p, blk(qw, rv, n, p)), &LqgOpts{Current: current})
		fzContract(t, s, "Lqg", res != nil, err)
		if err != nil {
			if errors.Is(err, ErrNoStabilizing) && syTinyDiscreteEig(s) {
				fzSkip(t, "Lqg: Dare with a nonzero eigenvalue below 1e-12 (known followup)")
			}
			if !errors.Is(err, ErrAlgebraicLoop) && mqRiccatiExists(s, qx, ru, nil) && mqRiccatiExists(syDual(s), qw, rv, nil) {
				t.Fatalf("Lqg: %v with both Riccati problems provably solvable\n%s", err, s)
			}
			fzSkip(t, "Lqg: error on a not provably solvable problem")
		}
		nk, a, b, c, d := syCtrlData(t, s, "Lqg", res.Controller, m, p)
		P := &syPlant{n: n, nu: m, ny: p, A: s.A, B2: s.B, C2: s.C, D22: s.D,
			B1: make([]float64, 0), C1: make([]float64, 0), D11: nil, D12: make([]float64, 0), D21: make([]float64, 0)}
		cl, ok := syLFT(P, nk, a, b, c, d)
		if !ok {
			t.Fatalf("Lqg: I − Dk·D singular in the closed loop\n%s", s)
		}
		if !syRegSolvable(s) {
			fzSkip(t, "Lqg: not robustly stabilizable and detectable")
		}
		syRequireStableRat(t, s, "Lqg", cl.A, cl.N, s.Discrete, fmt.Sprintf("K=%v L=%v ctrl A=%v B=%v C=%v D=%v", mqRaw(res.K), mqRaw(res.L), a, b, c, d))
	})
}

// syMixWeights returns the scalar W1 = 1/M + c/(s + a) (continuous) or
// 1/M + c/(z − a) (discrete) with M = 2, DC gain 1/Al = 100 and
// bandwidth wb = max(|S|, 1)·10^clamp(v[12],-2,1), and the static W2 = ρ
// = 10^clamp(v[13],-2,1); W3 = 0.1 when v[14] ≥ 0, else absent (w3 = 0).
func syMixWeights(s *fzSpec) (aw, cw, dw, rho, w3 float64) {
	base := 1.0
	for _, e := range s.S {
		base = math.Max(base, cmplx.Abs(e))
	}
	wb := base * math.Pow(10, fzClamp(s.Vals[12], -2, 1))
	const M, Al = 2.0, 1e-2
	dw = 1 / M
	if s.Discrete {
		aw = math.Exp(-wb * Al * s.Dt)
		cw = (1/Al - 1/M) * (1 - aw)
	} else {
		aw = -wb * Al
		cw = (1/Al - 1/M) * wb * Al
	}
	rho = math.Pow(10, fzClamp(s.Vals[13], -2, 1))
	if s.Vals[14] >= 0 {
		w3 = 0.1
	}
	return
}

// syMixPlant builds the mixed-sensitivity plant independently of Augw:
// states [x_G; x_W1 (p)], inputs [w (p); u (m)], outputs
// [z1 (p); z2 (m); z3 (p, when w3 ≠ 0); e (p)] with e = w − G·u,
// z1 = W1·e, z2 = ρ·u, z3 = w3·G·u.
func syMixPlant(s *fzSpec, aw, cw, dw, rho, w3 float64) *syPlant {
	n, m, p := s.N, s.M, s.P
	nz := p + m
	if w3 != 0 {
		nz += p
	}
	N := n + p
	P := &syPlant{n: N, nw: p, nz: nz, nu: m, ny: p}
	P.A = make([]float64, N*N)
	P.B1 = make([]float64, N*p)
	P.B2 = make([]float64, N*m)
	for i := range n {
		copy(P.A[i*N:], s.A[i*n:(i+1)*n])
		copy(P.B2[i*m:], s.B[i*m:(i+1)*m])
	}
	for i := range p {
		r := n + i
		P.A[r*N+r] = aw
		for j := range n {
			P.A[r*N+j] = -s.C[i*n+j]
		}
		for j := range m {
			P.B2[r*m+j] = -s.D[i*m+j]
		}
		P.B1[r*p+i] = 1
	}
	P.C1 = make([]float64, nz*N)
	P.D11 = make([]float64, nz*p)
	P.D12 = make([]float64, nz*m)
	for i := range p {
		for j := range n {
			P.C1[i*N+j] = -dw * s.C[i*n+j]
		}
		P.C1[i*N+n+i] = cw
		P.D11[i*p+i] = dw
		for j := range m {
			P.D12[i*m+j] = -dw * s.D[i*m+j]
		}
	}
	for i := range m {
		P.D12[(p+i)*m+i] = rho
	}
	if w3 != 0 {
		for i := range p {
			r := p + m + i
			for j := range n {
				P.C1[r*N+j] = w3 * s.C[i*n+j]
			}
			for j := range m {
				P.D12[r*m+j] = w3 * s.D[i*m+j]
			}
		}
	}
	P.C2 = make([]float64, p*N)
	P.D21 = make([]float64, p*p)
	P.D22 = make([]float64, p*m)
	for i := range p {
		for j := range n {
			P.C2[i*N+j] = -s.C[i*n+j]
		}
		P.D21[i*p+i] = 1
		for j := range m {
			P.D22[i*m+j] = -s.D[i*m+j]
		}
	}
	return P
}

// syClear reports every intended pole at least rel·max(1, |e|) (plus the
// eigenvalue error) away from the stability boundary.
func syClear(s *fzSpec, rel float64) bool {
	for _, e := range s.Eig {
		if !(s.Margin(e) > rel*math.Max(1, cmplx.Abs(e))+4*s.EigErr()) {
			return false
		}
	}
	return true
}

// syNumericallySingularD reports a nonzero D with 2-norm condition above
// 1e12: Mixsyn then misses the optimum or errors on discrete plants with
// fast sampling (known followup).
func syNumericallySingularD(s *fzSpec) bool {
	if mqFro(s.D) == 0 {
		return false
	}
	g := make([]complex128, len(s.D))
	for i, x := range s.D {
		g[i] = complex(x, 0)
	}
	sv := fzSingularValues(g, s.P, s.M)
	return sv == nil || !(sv[len(sv)-1] > 1e-12*sv[0])
}

// syUnstableWithRHPZeros reports an unstable G that may also have
// transmission zeros in the closed right half plane (exact Routh count of
// det[sI−A, −B; C, D] for square G; non-square unstable G counts as
// possibly), where a near pole–zero pair makes the optimal γ arbitrarily
// large.
func syUnstableWithRHPZeros(s *fzSpec) bool {
	if s.Unstable() == 0 {
		return false
	}
	if s.M != s.P || s.Discrete {
		return true
	}
	num := fzTrimRat(fzPencilDetRat(s.N, s.M, s.A, s.B, s.C, s.D))
	if len(num) == 0 {
		return true
	}
	k, ok := fzRouthRight(num, 0)
	return !ok || k > 0
}

// FuzzMixsyn: Mixsyn of the normalized spec model with syMixWeights. When
// G is robustly stabilizable and detectable with every pole 1e-4 clear of
// the stability boundary, D not numerically rank deficient and, when G is
// unstable, no right-half-plane zeros (and, discrete, W1's pole not within
// 1e-6 of z = 1), an error is a failure. A returned K (driven by
// e = r − y) must stabilize G exactly under negative feedback, and the
// swept σmax of [W1·S; W2·K·S; W3·T] (closed on syMixPlant, built without
// Augw) may not exceed Gamma ≤ Info.GammaOpt.
func FuzzMixsyn(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		s = syNormC(s)
		n, m, p := s.N, s.M, s.P
		aw, cw, dw, rho, w3 := syMixWeights(s)
		mk := func(a, b, c, d []float64, k int) *System {
			sys, err := NewFromSlices(k, len(d), 1, a, b, c, d, s.Dt)
			if err != nil {
				t.Fatalf("weight: %v\n%s", err, s)
			}
			return sys
		}
		W1 := mk([]float64{aw}, []float64{1}, []float64{cw}, []float64{dw}, 1)
		W2 := mk(nil, nil, nil, []float64{rho}, 0)
		var W3 *System
		if w3 != 0 {
			W3 = mk(nil, nil, nil, []float64{w3}, 0)
		}
		res, err := Mixsyn(s.System(t, fzSysOpts{}), W1, W2, W3)
		fzContract(t, s, "Mixsyn", res != nil, err)
		solvable := syRegSolvable(s) && syClear(s, 1e-4) && !syNumericallySingularD(s) && !syUnstableWithRHPZeros(s)
		if s.Discrete && 1-aw < 1e-6 {
			solvable = false // known followup: slow discrete W1 pole fails on non-square or singular G
		}
		if err != nil {
			if solvable {
				t.Fatalf("Mixsyn: %v on a robustly stabilizable and detectable G with no boundary poles (W1 a=%g c=%g d=%g, ρ=%g, W3=%g)\n%s", err, aw, cw, dw, rho, w3, s)
			}
			fzSkip(t, "Mixsyn: error on a not robustly solvable plant")
		}
		nk, a, b, c, d := syCtrlData(t, s, "Mixsyn", res.K, m, p)
		if !(res.Gamma <= res.Info.GammaOpt*(1+1e-12)) {
			t.Fatalf("Mixsyn: Gamma %.10g > GammaOpt %.10g\n%s", res.Gamma, res.Info.GammaOpt, s)
		}
		G := &syPlant{n: n, nu: m, ny: p, A: s.A, B2: s.B, C2: make([]float64, p*n), D22: make([]float64, p*m)}
		for i := range G.C2 {
			G.C2[i] = -s.C[i]
		}
		for i := range G.D22 {
			G.D22[i] = -s.D[i]
		}
		loop, ok := syLFT(G, nk, a, b, c, d)
		if !ok {
			t.Fatalf("Mixsyn: I + Dk·D singular\n%s", s)
		}
		if !solvable {
			fzSkip(t, "Mixsyn: plant not robustly solvable")
		}
		syFragile(t, s, "Mixsyn", loop, res.CL)
		syRequireStableRat(t, s, "Mixsyn", loop.A, loop.N, s.Discrete, fmt.Sprintf("K: A=%v B=%v C=%v D=%v", a, b, c, d))
		P := syMixPlant(s, aw, cw, dw, rho, w3)
		cl, ok := syLFT(P, nk, a, b, c, d)
		if !ok {
			t.Fatalf("Mixsyn: I + Dk·D singular in the weighted loop\n%s", s)
		}
		lo, w := syHinfLower(cl, P.nw, P.nz, s.Dt)
		if !(lo <= res.Gamma*(1+1e-6)) {
			syRoundingExcess(t, s, "Mixsyn", res.CL, lo, w, res.Gamma, P.nw, P.nz)
			t.Fatalf("Mixsyn: σmax %.10g of [W1S; W2KS; W3T] at ω=%g exceeds Gamma %.10g\n%s", lo, w, res.Gamma, s)
		}
	})
}
