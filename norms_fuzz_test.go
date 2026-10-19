package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzH2Norm: on decidable delay-free models H2Norm returns ErrUnstable (and
// Norm(·,2) +Inf) exactly when an intended pole is unstable, +Inf for a
// continuous model with D ≠ 0, and otherwise the energy √tr(BᵀXB) (+‖D‖²_F
// when discrete) with X the exact rational observability gramian of the
// float64 model.
func FuzzH2Norm(f *testing.F) {
	for _, s := range fzH2NormSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		h2, err := H2Norm(sys)
		nrm, nerr := Norm(sys, NormH2)
		fzContract(t, s, "Norm(2)", nerr == nil, nerr)
		if math.IsNaN(nrm) || nrm < 0 || err == nil && (math.IsNaN(h2) || h2 < 0) {
			t.Fatalf("H2Norm = %v, Norm(2) = %v\n%s", h2, nrm, s)
		}
		fzDescriptorNorm(t, s, "H2Norm", func(d *System) (float64, error) { return H2Norm(d) }, h2, err)
		dZero := true
		for _, x := range s.D {
			dZero = dZero && x == 0
		}
		if s.N == 0 {
			fzErr(t, s, "H2Norm", err)
			want := math.Sqrt(fzFrob2(s.D))
			if !s.Discrete && !dZero {
				want = math.Inf(1)
			}
			if h2 != want && math.Abs(h2-want) > 1e-14*want || nrm != h2 {
				t.Fatalf("H2Norm = %v, Norm(2) = %v, want %v\n%s", h2, nrm, want, s)
			}
			return
		}
		var unstable bool
		if s.Decidable() {
			unstable = s.Unstable() > 0
		} else if unstable = fzSurelyNotStable(s); !unstable {
			fzSkip(t, "verdict unresolvable: pole within error band of stability boundary")
		}
		if unstable {
			if !errors.Is(err, ErrUnstable) {
				t.Fatalf("H2Norm = %v, %v for an unstable model, want ErrUnstable\n%s", h2, err, s)
			}
			if !math.IsInf(nrm, 1) || nerr != nil {
				t.Fatalf("Norm(2) = %v, %v for an unstable model, want +Inf\n%s", nrm, nerr, s)
			}
			return
		}
		fzErr(t, s, "Norm(2)", nerr)
		fzErr(t, s, "H2Norm", err)
		if nrm != h2 {
			t.Fatalf("Norm(2) = %v, H2Norm = %v\n%s", nrm, h2, s)
		}
		if !s.Discrete && !dZero {
			if !math.IsInf(h2, 1) {
				t.Fatalf("H2Norm = %v for a continuous model with D ≠ 0, want +Inf\n%s", h2, s)
			}
			return
		}
		tol := fzNormTol(s)
		if tol > 1e-3 {
			fzSkip(t, "H2: gramian ill-conditioned")
		}
		want, ok := fzH2Exact(s)
		if !ok {
			fzSkip(t, "H2: exact Lyapunov system singular")
		}
		if math.IsInf(h2, 0) || math.Abs(h2-want) > tol*want+1e-300 {
			t.Fatalf("H2Norm = %v, exact gramian gives %v (rel %.3g, tol %.3g)\n%s", h2, want, math.Abs(h2-want)/want, tol, s)
		}
	})
}

// FuzzLinfNorm: on decidable delay-free models Norm(·,Inf) is finite even
// for unstable models, attained at the frequency linfNorm returns, not
// exceeded on a pole-clustered sweep, certified from above by the exact
// γ-crossing test (fzPeakCertified), and equal to HinfNorm when stable.
func FuzzLinfNorm(f *testing.F) {
	for _, s := range fzLinfNormSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		norm, omega, err := linfNorm(sys)
		fzErr(t, s, "linfNorm", err)
		nrm, nerr := Norm(sys, math.Inf(1))
		fzErr(t, s, "Norm(Inf)", nerr)
		if nrm != norm {
			t.Fatalf("Norm(Inf) = %v, linfNorm = %v\n%s", nrm, norm, s)
		}
		if math.IsNaN(norm) || math.IsNaN(omega) || norm < 0 {
			t.Fatalf("linfNorm = %v at %v\n%s", norm, omega, s)
		}
		fzDescriptorNorm(t, s, "Norm(Inf)", func(d *System) (float64, error) { return Norm(d, math.Inf(1)) }, nrm, nerr)
		if !s.Decidable() {
			fzSkip(t, "verdict unresolvable: pole within error band of stability boundary")
		}
		if math.IsInf(norm, 0) {
			t.Fatalf("linfNorm = %v with no pole near the stability boundary\n%s", norm, s)
		}
		fzCheckPeak(t, s, "linfNorm", norm, omega)
		if s.Unstable() == 0 {
			h, _, err := HinfNorm(sys)
			fzErr(t, s, "HinfNorm", err)
			if math.Abs(h-norm) > 1e-9*norm {
				t.Fatalf("HinfNorm = %v, Norm(Inf) = %v for a stable model\n%s", h, norm, s)
			}
		}
	})
}

// fzCheckPeak checks a finite peak gain of a decidable model: attained at
// omega (σmax(D) for +Inf), at least the exact σmax at the best sweep
// frequency and at σmax(D), and certified by fzPeakCertified, all to the
// relative sensitivity fzNormTol (at least 1e-6).
func fzCheckPeak(t *testing.T, s *fzSpec, op string, norm, omega float64) {
	t.Helper()
	tol := 1e-6
	if s.N > 0 {
		ct := math.Max(tol, fzNormTol(s))
		if ct > 1e-2 {
			fzSkip(t, "peak: realization too ill-conditioned")
		}
		tol = ct
	}
	if s.N == 0 {
		if want := fzSigmaMax(fzComplexD(s), s.P, s.M); math.Abs(norm-want) > tol*want {
			t.Fatalf("%s = %v, want σmax(D) = %v\n%s", op, norm, want, s)
		}
		return
	}
	at := func(w float64) float64 {
		g, ok := fzExactResponse(s.N, s.M, s.P, s.A, s.B, s.C, s.D, fzFreqPoint(s, w))
		if !ok {
			fzSkip(t, "frequency hits an exact pole")
		}
		return fzSigmaMax(g, s.P, s.M)
	}
	var got float64
	if math.IsInf(omega, 1) {
		if s.Discrete {
			t.Fatalf("%s omega = +Inf for a discrete model\n%s", op, s)
		}
		got = fzSigmaMax(fzComplexD(s), s.P, s.M)
	} else {
		if omega < 0 || s.Discrete && omega > math.Pi/s.Dt*(1+1e-12) {
			t.Fatalf("%s omega = %v outside [0, Nyquist]\n%s", op, omega, s)
		}
		got = at(omega)
	}
	if got < norm*(1-tol) {
		t.Fatalf("%s = %v at ω=%v, but exact σmax there is %v\n%s", op, norm, omega, got, s)
	}
	wbest, _ := fzSweepMax(func(w float64) float64 {
		return fzSigmaMax(fzResponseFloat(s, fzFreqPoint(s, w)), s.P, s.M)
	}, fzFreqGrid(s.S, s.Dt, 60), 2)
	if peak := at(wbest); norm < peak*(1-tol) {
		t.Fatalf("%s = %v at ω=%v, but exact σmax at ω=%v is %v\n%s", op, norm, omega, wbest, peak, s)
	}
	if !s.Discrete {
		if d := fzSigmaMax(fzComplexD(s), s.P, s.M); norm < d*(1-tol) {
			t.Fatalf("%s = %v below σmax(D) = %v\n%s", op, norm, d, s)
		}
	}
	gamma := norm * (1 + tol)
	switch ok, why := fzPeakCertified(s, gamma); {
	case why == "skip":
		fzSkip(t, "peak certificate: z = -1 is an exact pole")
	case !ok:
		t.Fatalf("%s = %v under-reported: exact γ-crossing test at γ = %v (rel %.3g) %s\n%s", op, norm, gamma, tol, why, s)
	}
}

// fzNormTol is the relative sensitivity of a peak or H2 norm of the float64
// model to the backward error of the library: the eigenvalue error bound
// over the smallest distance of an intended pole to the stability boundary.
func fzNormTol(s *fzSpec) float64 {
	m := math.Inf(1)
	for _, e := range s.Eig {
		m = math.Min(m, s.Margin(e))
	}
	return 1e-9 + s.EigErr()/m
}

// fzDescriptorNorm checks the descriptor contract of a norm routine: E = I
// gives the explicit result, a nonsingular diagonal E with A, B scaled to
// the same transfer function returns ErrDescriptorUnsupported.
func fzDescriptorNorm(t *testing.T, s *fzSpec, op string, norm func(*System) (float64, error), want float64, wantErr error) {
	t.Helper()
	if s.N == 0 {
		return
	}
	n := s.N
	A := mat.NewDense(n, n, append([]float64(nil), s.A...))
	B := mat.NewDense(n, s.M, append([]float64(nil), s.B...))
	C := mat.NewDense(s.P, n, s.C)
	D := mat.NewDense(s.P, s.M, s.D)
	id, err := NewDescriptor(A, B, C, D, eyeDense(n), s.Dt)
	if err != nil {
		t.Fatalf("NewDescriptor(E=I): %v\n%s", err, s)
	}
	got, err := norm(id)
	if (err == nil) != (wantErr == nil) || err == nil && got != want && !(math.IsNaN(got) && math.IsNaN(want)) {
		t.Fatalf("%s with E = I: %v, %v; explicit %v, %v\n%s", op, got, err, want, wantErr, s)
	}
	E := mat.NewDense(n, n, nil)
	for i := range n {
		e := math.Ldexp(1, i%3-1)
		E.Set(i, i, e)
		for j := range n {
			A.Set(i, j, e*A.At(i, j))
		}
		for j := range s.M {
			B.Set(i, j, e*B.At(i, j))
		}
	}
	ds, err := NewDescriptor(A, B, C, D, E, s.Dt)
	if err != nil {
		t.Fatalf("NewDescriptor: %v\n%s", err, s)
	}
	if v, err := norm(ds); !errors.Is(err, ErrDescriptorUnsupported) {
		t.Fatalf("%s of a descriptor model = %v, %v, want ErrDescriptorUnsupported\n%s", op, v, err, s)
	}
}

func fzFrob2(d []float64) float64 {
	v := 0.0
	for _, x := range d {
		v += x * x
	}
	return v
}

// fzH2Exact returns √tr(BᵀXB) (+‖D‖²_F when discrete), X solving
// AᵀX + XA + CᵀC = 0 (AᵀXA − X + CᵀC = 0) exactly over big.Rat.
func fzH2Exact(s *fzSpec) (float64, bool) {
	n, m, p := s.N, s.M, s.P
	A, B, C := fzRatPoly(s.A), fzRatPoly(s.B), fzRatPoly(s.C)
	idx := make([]int, n*n)
	k := 0
	for i := range n {
		for j := i; j < n; j++ {
			idx[i*n+j], idx[j*n+i] = k, k
			k++
		}
	}
	nu := k
	M := make([]*big.Rat, nu*(nu+1))
	for i := range M {
		M[i] = new(big.Rat)
	}
	var t big.Rat
	row := 0
	for i := range n {
		for j := i; j < n; j++ {
			r := M[row*(nu+1) : (row+1)*(nu+1)]
			if s.Discrete {
				for a := range n {
					for b := range n {
						r[idx[a*n+b]].Add(r[idx[a*n+b]], t.Mul(A[a*n+i], A[b*n+j]))
					}
				}
				r[idx[i*n+j]].Sub(r[idx[i*n+j]], big.NewRat(1, 1))
			} else {
				for a := range n {
					r[idx[a*n+j]].Add(r[idx[a*n+j]], A[a*n+i])
					r[idx[i*n+a]].Add(r[idx[i*n+a]], A[a*n+j])
				}
			}
			for l := range p {
				r[nu].Sub(r[nu], t.Mul(C[l*n+i], C[l*n+j]))
			}
			row++
		}
	}
	x, ok := fzRatSolve(M, nu)
	if !ok {
		return 0, false
	}
	tr := new(big.Rat)
	for c := range m {
		for i := range n {
			for j := range n {
				var u big.Rat
				u.Mul(B[i*m+c], B[j*m+c])
				tr.Add(tr, u.Mul(&u, x[idx[i*n+j]]))
			}
		}
	}
	if s.Discrete {
		for _, d := range s.D {
			tr.Add(tr, t.Mul(fzRat(d), fzRat(d)))
		}
	}
	v, _ := tr.Float64()
	return math.Sqrt(math.Max(v, 0)), true
}

// fzRatSolve solves the r×r system with augmented last column in place
// (Gauss–Jordan); ok is false when it is singular.
func fzRatSolve(M []*big.Rat, r int) ([]*big.Rat, bool) {
	c := r + 1
	var t big.Rat
	for col := range r {
		piv := -1
		for i := col; i < r; i++ {
			if M[i*c+col].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range c {
			M[piv*c+j], M[col*c+j] = M[col*c+j], M[piv*c+j]
		}
		inv := new(big.Rat).Inv(M[col*c+col])
		for j := col; j < c; j++ {
			M[col*c+j].Mul(M[col*c+j], inv)
		}
		for i := range r {
			if i == col || M[i*c+col].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(M[i*c+col])
			for j := col; j < c; j++ {
				M[i*c+j].Sub(M[i*c+j], t.Mul(f, M[col*c+j]))
			}
		}
	}
	x := make([]*big.Rat, r)
	for i := range r {
		x[i] = M[i*c+r]
	}
	return x, true
}

func fzRatMul(a, b []*big.Rat, r, k, c int) []*big.Rat {
	out := make([]*big.Rat, r*c)
	var t big.Rat
	for i := range r {
		for j := range c {
			v := new(big.Rat)
			for l := range k {
				v.Add(v, t.Mul(a[i*k+l], b[l*c+j]))
			}
			out[i*c+j] = v
		}
	}
	return out
}

func fzRatT(a []*big.Rat, r, c int) []*big.Rat {
	out := make([]*big.Rat, r*c)
	for i := range r {
		for j := range c {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}

// fzRatInv returns the inverse of the n×n rational matrix, or nil.
func fzRatInv(a []*big.Rat, n int) []*big.Rat {
	c := 2 * n
	M := make([]*big.Rat, n*c)
	for i := range n {
		for j := range n {
			M[i*c+j] = new(big.Rat).Set(a[i*n+j])
			M[i*c+n+j] = new(big.Rat)
		}
		M[i*c+n+i].SetInt64(1)
	}
	var t big.Rat
	for col := range n {
		piv := -1
		for i := col; i < n; i++ {
			if M[i*c+col].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil
		}
		for j := range c {
			M[piv*c+j], M[col*c+j] = M[col*c+j], M[piv*c+j]
		}
		inv := new(big.Rat).Inv(M[col*c+col])
		for j := range c {
			M[col*c+j].Mul(M[col*c+j], inv)
		}
		for i := range n {
			if i == col || M[i*c+col].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(M[i*c+col])
			for j := range c {
				M[i*c+j].Sub(M[i*c+j], t.Mul(f, M[col*c+j]))
			}
		}
	}
	out := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			out[i*n+j] = M[i*c+n+j]
		}
	}
	return out
}

// fzRatPosDef reports whether the symmetric n×n rational matrix is
// positive definite (exact Cholesky pivots).
func fzRatPosDef(a []*big.Rat, n int) bool {
	M := make([]*big.Rat, len(a))
	for i, x := range a {
		M[i] = new(big.Rat).Set(x)
	}
	var t big.Rat
	for k := range n {
		if M[k*n+k].Sign() <= 0 {
			return false
		}
		for i := k + 1; i < n; i++ {
			f := new(big.Rat).Quo(M[i*n+k], M[k*n+k])
			for j := k; j < n; j++ {
				M[i*n+j].Sub(M[i*n+j], t.Mul(f, M[k*n+j]))
			}
		}
	}
	return true
}

// fzRatCayley maps the discrete model to the continuous one with
// G_c(s) = G_d((1+s)/(1−s)), exactly: Ac = (I+A)⁻¹(A−I), Bc = (I+A)⁻¹B,
// Cc = 2C(I+A)⁻¹, Dc = D − C(I+A)⁻¹B. The unit circle maps to the
// imaginary axis and z = −1 to s = ∞. ok is false when −1 is an exact
// eigenvalue of A.
func fzRatCayley(n, m, p int, A, B, C, D []*big.Rat) (Ac, Bc, Cc, Dc []*big.Rat, ok bool) {
	IA := make([]*big.Rat, n*n)
	AmI := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			IA[i*n+j] = new(big.Rat).Set(A[i*n+j])
			AmI[i*n+j] = new(big.Rat).Set(A[i*n+j])
		}
		IA[i*n+i].Add(IA[i*n+i], big.NewRat(1, 1))
		AmI[i*n+i].Sub(AmI[i*n+i], big.NewRat(1, 1))
	}
	inv := fzRatInv(IA, n)
	if inv == nil {
		return nil, nil, nil, nil, false
	}
	Ac = fzRatMul(inv, AmI, n, n, n)
	Bc = fzRatMul(inv, B, n, n, m)
	CI := fzRatMul(C, inv, p, n, n)
	Cc = make([]*big.Rat, p*n)
	for i, x := range CI {
		Cc[i] = new(big.Rat).Add(x, x)
	}
	CB := fzRatMul(CI, B, p, n, m)
	Dc = make([]*big.Rat, p*m)
	for i := range Dc {
		Dc[i] = new(big.Rat).Sub(D[i], CB[i])
	}
	return Ac, Bc, Cc, Dc, true
}

// fzPeakCertified reports whether σmax(G) < γ on the whole stability
// boundary of the float64 model, exactly: R = γ²I − DᵀD is positive definite
// and the Hamiltonian
//
//	H = [F, BR⁻¹Bᵀ; −Cᵀ(I + DR⁻¹Dᵀ)C, −Fᵀ],  F = A + BR⁻¹DᵀC,
//
// whose imaginary eigenvalues jω are the frequencies where γ is a singular
// value of G(jω) (A has none on the axis), has none: its characteristic
// polynomial is q(s²) and q has no root in (−∞, 0] (Sturm). Discrete models
// go through fzRatCayley. why describes a failure, or is "skip" when −1 is
// an exact pole.
func fzPeakCertified(s *fzSpec, gamma float64) (ok bool, why string) {
	n, m, p := s.N, s.M, s.P
	A, B, C, D := fzRatPoly(s.A), fzRatPoly(s.B), fzRatPoly(s.C), fzRatPoly(s.D)
	if s.Discrete {
		var cok bool
		if A, B, C, D, cok = fzRatCayley(n, m, p, A, B, C, D); !cok {
			return false, "skip"
		}
	}
	return fzRatPeakCertified(n, m, p, A, B, C, D, fzRat(gamma))
}

func fzRatPeakCertified(n, m, p int, A, B, C, D []*big.Rat, gamma *big.Rat) (bool, string) {
	g2 := new(big.Rat).Mul(gamma, gamma)
	Dt := fzRatT(D, p, m)
	R := fzRatMul(Dt, D, m, p, m)
	for i := range R {
		R[i].Neg(R[i])
	}
	for i := range m {
		R[i*m+i].Add(R[i*m+i], g2)
	}
	if !fzRatPosDef(R, m) {
		return false, "finds γ ≤ σmax(D) (at ω = ∞ or Nyquist)"
	}
	Ri := fzRatInv(R, m)
	BRi := fzRatMul(B, Ri, n, m, m)
	F := fzRatMul(BRi, fzRatMul(Dt, C, m, p, n), n, m, n)
	for i := range F {
		F[i].Add(F[i], A[i])
	}
	G := fzRatMul(BRi, fzRatT(B, n, m), n, m, n)
	W := fzRatMul(fzRatMul(D, Ri, p, m, m), Dt, p, m, p)
	for i := range p {
		W[i*p+i].Add(W[i*p+i], big.NewRat(1, 1))
	}
	Ct := fzRatT(C, p, n)
	Q := fzRatMul(fzRatMul(Ct, W, n, p, p), C, n, p, n)
	h := 2 * n
	H := make([]*big.Rat, h*h)
	for i := range n {
		for j := range n {
			H[i*h+j] = F[i*n+j]
			H[i*h+n+j] = G[i*n+j]
			H[(n+i)*h+j] = new(big.Rat).Neg(Q[i*n+j])
			H[(n+i)*h+n+j] = new(big.Rat).Neg(F[j*n+i])
		}
	}
	cp := fzCharPolyRat(H, h)
	q := make([]*big.Rat, n+1)
	for k := range cp {
		if k%2 == 1 {
			if cp[k].Sign() != 0 {
				return false, "oracle error: Hamiltonian characteristic polynomial not even"
			}
			continue
		}
		q[k/2] = cp[k]
	}
	if q[n].Sign() == 0 {
		return false, "finds a crossing at ω = 0"
	}
	if k := fzSturmCount(q); k > 0 {
		return false, "finds a crossing on the stability boundary"
	}
	return true, ""
}

// fzSturmCount returns the number of distinct real roots of the descending
// rational polynomial q in (−∞, 0); q(0) must be nonzero.
func fzSturmCount(q []*big.Rat) int {
	q = fzTrimRat(q)
	deg := len(q) - 1
	if deg <= 0 {
		return 0
	}
	dq := make([]*big.Rat, deg)
	for i := range deg {
		dq[i] = new(big.Rat).Mul(q[i], big.NewRat(int64(deg-i), 1))
	}
	seq := [][]*big.Rat{q, dq}
	for {
		a, b := seq[len(seq)-2], seq[len(seq)-1]
		r := fzRatPolyRem(a, b)
		if len(r) == 0 {
			break
		}
		for _, x := range r {
			x.Neg(x)
		}
		seq = append(seq, r)
	}
	changes := func(signs []int) int {
		k, last := 0, 0
		for _, s := range signs {
			if s == 0 {
				continue
			}
			if last != 0 && s != last {
				k++
			}
			last = s
		}
		return k
	}
	var atNegInf, atZero []int
	for _, p := range seq {
		sg := p[0].Sign()
		if (len(p)-1)%2 == 1 {
			sg = -sg
		}
		atNegInf = append(atNegInf, sg)
		atZero = append(atZero, p[len(p)-1].Sign())
	}
	return changes(atNegInf) - changes(atZero)
}

// fzRatPolyRem returns a mod b with exactly zero leading terms trimmed.
func fzRatPolyRem(a, b []*big.Rat) []*big.Rat {
	r := make([]*big.Rat, len(a))
	for i, x := range a {
		r[i] = new(big.Rat).Set(x)
	}
	var t big.Rat
	for len(r) >= len(b) {
		f := new(big.Rat).Quo(r[0], b[0])
		for i := range b {
			r[i].Sub(r[i], t.Mul(f, b[i]))
		}
		r = r[1:]
	}
	return fzTrimRat(r)
}

func TestFzNormOracles(t *testing.T) {
	// G = 1/(s²+0.2s+1): ‖G‖₂² = 1/(2·0.2), peak 1/(2ζ√(1−ζ²)) at ω = √(1−2ζ²).
	s := &fzSpec{N: 2, M: 1, P: 1, A: []float64{0, 1, -1, -0.2}, B: []float64{0, 1}, C: []float64{1, 0}, D: []float64{0}}
	if h, ok := fzH2Exact(s); !ok || math.Abs(h-math.Sqrt(2.5)) > 1e-15 {
		t.Errorf("H2 = %v,%v, want √2.5", h, ok)
	}
	peak := 1 / (2 * 0.1 * math.Sqrt(1-0.01))
	if ok, why := fzPeakCertified(s, peak*(1+1e-9)); !ok {
		t.Errorf("γ just above the peak not certified: %s", why)
	}
	if ok, _ := fzPeakCertified(s, peak*(1-1e-9)); ok {
		t.Error("γ just below the peak certified")
	}
	// D ≠ 0, MIMO: G = [1/(s+1); 2] has peak √5 at ω = 0, σ → 2 at ∞.
	s = &fzSpec{N: 1, M: 1, P: 2, A: []float64{-1}, B: []float64{1}, C: []float64{1, 0}, D: []float64{0, 2}}
	if ok, why := fzPeakCertified(s, math.Sqrt(5)*(1+1e-12)); !ok {
		t.Errorf("√5(1+ε) not certified: %s", why)
	}
	if ok, _ := fzPeakCertified(s, math.Sqrt(5)*(1-1e-12)); ok {
		t.Error("√5(1−ε) certified")
	}
	if ok, _ := fzPeakCertified(s, 1.999); ok {
		t.Error("γ < σmax(D) certified")
	}
	// G = [s/(s²+s+4), 1/(s+3)] (MIMO, peak away from 0 and ∞): compare a sweep.
	s = &fzSpec{N: 3, M: 2, P: 1, A: []float64{0, 1, 0, -4, -1, 0, 0, 0, -3}, B: []float64{0, 0, 1, 0, 0, 1}, C: []float64{0, 1, 1}, D: []float64{0, 0.5}}
	_, sw := fzSweepMax(func(w float64) float64 {
		return fzSigmaMax(fzResponseFloat(s, complex(0, w)), 1, 2)
	}, fzFreqGrid([]complex128{complex(-0.5, math.Sqrt(3.75)), -3}, 0, 400), 3)
	if ok, why := fzPeakCertified(s, sw*(1+1e-8)); !ok {
		t.Errorf("sweep peak %v(1+1e-8) not certified: %s", sw, why)
	}
	if ok, _ := fzPeakCertified(s, sw*(1-1e-8)); ok {
		t.Errorf("sweep peak %v(1−1e-8) certified", sw)
	}
	// Discrete G = 1/(z−0.5) (dt 1): ‖G‖₂² = 1/(1−0.25), peak 2 at z = 1;
	// G = 1/(z+0.5) peaks at Nyquist; G = 0.5 + 1/(z-0.9) mixes D.
	for _, c := range []struct {
		a, d, h2, peak float64
	}{{0.5, 0, math.Sqrt(4.0 / 3), 2}, {-0.5, 0, math.Sqrt(4.0 / 3), 2}, {0.9, 0.5, math.Sqrt(0.25 + 1/0.19), 10.5}} {
		s = &fzSpec{Discrete: true, Dt: 1, N: 1, M: 1, P: 1, A: []float64{c.a}, B: []float64{1}, C: []float64{1}, D: []float64{c.d}}
		if h, ok := fzH2Exact(s); !ok || math.Abs(h-c.h2) > 1e-14 {
			t.Errorf("a=%v: H2 = %v, want %v", c.a, h, c.h2)
		}
		if ok, why := fzPeakCertified(s, c.peak*(1+1e-12)); !ok {
			t.Errorf("a=%v: peak(1+ε) not certified: %s", c.a, why)
		}
		if ok, _ := fzPeakCertified(s, c.peak*(1-1e-12)); ok {
			t.Errorf("a=%v: peak(1−ε) certified", c.a)
		}
	}
	// Discrete 2×2 lightly damped pair: Cayley image matches the response.
	z := cmplx.Rect(0.95, 0.7)
	s = &fzSpec{Discrete: true, Dt: 0.1, N: 2, M: 1, P: 1, A: []float64{real(z), imag(z), -imag(z), real(z)}, B: []float64{1, 0.3}, C: []float64{0.2, 1}, D: []float64{0.1}}
	_, sw = fzSweepMax(func(w float64) float64 {
		return fzSigmaMax(fzResponseFloat(s, fzFreqPoint(s, w)), 1, 1)
	}, fzFreqGrid([]complex128{cmplx.Log(z) * 10}, 0.1, 400), 3)
	if ok, why := fzPeakCertified(s, sw*(1+1e-8)); !ok {
		t.Errorf("discrete sweep peak %v(1+1e-8) not certified: %s", sw, why)
	}
	if ok, _ := fzPeakCertified(s, sw*(1-1e-8)); ok {
		t.Errorf("discrete sweep peak %v(1−1e-8) certified", sw)
	}
	if k := fzSturmCount(fzRatPoly([]float64{1, 3, -4})); k != 1 {
		t.Errorf("Sturm count of x²+3x−4 on (−∞,0) = %d, want 1", k)
	}
	if k := fzSturmCount(fzRatPoly([]float64{1, 2, 1, 0, 4})); k != 0 {
		t.Errorf("Sturm count of x⁴+2x³+x²+4 = %d, want 0", k)
	}
	if k := fzSturmCount(fzRatPoly([]float64{1, 4, 4})); k != 1 {
		t.Errorf("Sturm count of (x+2)² = %d, want 1", k)
	}
}

func fzH2NormSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzHinfNormSeeds(),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 6, 3, 3
			s.Mag, s.Damp = [3]float64{-1, 0.3, 2}, [3]float64{3, 5, 0.5}
			s.Shear, s.Rot, s.Phase, s.Gain = -1, 0.8, 0.4, 7
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.2)
			s.N, s.M, s.P = 5, 2, 3
			s.Mag, s.Damp = [3]float64{0.3, -0.4, 0.2}, [3]float64{5, 2, 1}
			s.Shear, s.Rot, s.DScal = -0.5, 1.7, 0.8
		}),
		with(func(s *fzSeed) { s.N, s.M, s.Mag[0], s.Damp[0], s.Rot = 2, 2, 1, 7, 2.2 }),
		with(func(s *fzSeed) { s.N, s.P, s.Mag[0], s.Damp[0], s.Damp[1] = 3, 2, 0.5, -3, 1 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -2
			s.N, s.Mag[0], s.Damp[0] = 2, 1.5, 6.5
		}),
	)
}

func fzLinfNormSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzH2NormSeeds(),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 2, 2
			s.Mag, s.Damp = [3]float64{0.2, -0.5, 1}, [3]float64{-4, 3, -1}
			s.Shear, s.Rot, s.Phase, s.DScal = -0.7, 0.9, 1.2, 0.3
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.05)
			s.N, s.M, s.P = 4, 2, 1
			s.Mag, s.Damp = [3]float64{1, 0.5}, [3]float64{-5, 2}
			s.Rot, s.DScal = 0.6, -0.4
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Gain = 2, 0, -6, 3 }),
	)
}
