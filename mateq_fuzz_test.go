package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Matrix-equation fuzz targets (Lyap/DLyap, Gram, Covar, Care/Dare) on the
// fzSpec models. The delay fields v[12..14] are reused as equation options.
// Value oracle: the n²×n² Kronecker form of the Lyapunov equation solved by
// dense LU, trusted only when its 2-norm condition is at most mqMaxCond.
// Riccati oracle: exact closed-loop stability (Routh / Schur–Cohn on the
// float64 A−BK), residuals, and, when the library reports no stabilizing
// solution, an exact count of Hamiltonian / symplectic-pencil roots in a
// strip about the stability boundary plus a PBH stabilizability margin.

const mqMaxCond = 1e8

func mqDense(r, c int, d []float64) *mat.Dense {
	return mat.NewDense(r, c, append([]float64(nil), d...))
}

func mqFro(d []float64) float64 {
	s := 0.0
	for _, x := range d {
		s = math.Hypot(s, x)
	}
	return s
}

func mqRaw(m mat.Matrix) []float64 {
	return mat.DenseCopyOf(m).RawMatrix().Data
}

// mqKron returns the row-major Kronecker matrix of X ↦ AX + XAᵀ
// (continuous) or X ↦ AXAᵀ − X (discrete) on row-major vec(X).
func mqKron(a []float64, n int, discrete bool) *mat.Dense {
	nn := n * n
	L := mat.NewDense(nn, nn, nil)
	for i := range n {
		for j := range n {
			row := i*n + j
			for k := range n {
				for l := range n {
					col := k*n + l
					v := 0.0
					if discrete {
						v = a[i*n+k] * a[j*n+l]
						if row == col {
							v--
						}
					} else {
						if l == j {
							v += a[i*n+k]
						}
						if k == i {
							v += a[j*n+l]
						}
					}
					L.Set(row, col, v)
				}
			}
		}
	}
	return L
}

// mqLyapRef solves AX + XAᵀ + Q = 0 (AXAᵀ − X + Q = 0) through its
// Kronecker form L. cond is the structured condition (2‖A‖ or ‖A‖²+1)/σmin(L),
// which bounds the relative effect of eps·‖A‖ data errors; ok is false when
// it exceeds mqMaxCond.
func mqLyapRef(a, q []float64, n int, discrete bool) (x []float64, cond float64, ok bool) {
	L := mqKron(a, n, discrete)
	var svd mat.SVD
	if !svd.Factorize(L, mat.SVDNone) {
		return nil, math.Inf(1), false
	}
	sv := svd.Values(nil)
	na := mat.Norm(mqDense(n, n, a), 2)
	scale := 2 * na
	if discrete {
		scale = na*na + 1
	}
	cond = math.Max(scale, sv[0]) / sv[len(sv)-1]
	if !(cond <= mqMaxCond) {
		return nil, cond, false
	}
	rhs := mat.NewVecDense(n*n, nil)
	for k := range n * n {
		rhs.SetVec(k, -q[k])
	}
	var v mat.VecDense
	if err := v.SolveVec(L, rhs); err != nil {
		return nil, cond, false
	}
	x = make([]float64, n*n)
	for k := range x {
		x[k] = v.AtVec(k)
	}
	return x, cond, true
}

// mqLyapResidual is ‖AX+XAᵀ+Q‖/(2‖A‖‖X‖+‖Q‖) or the discrete analogue.
func mqLyapResidual(a, x, q []float64, n int, discrete bool) float64 {
	A, X, Q := mqDense(n, n, a), mqDense(n, n, x), mqDense(n, n, q)
	var r, t mat.Dense
	na, nx, nq := mqFro(a), mqFro(x), mqFro(q)
	den := 2*na*nx + nq
	if discrete {
		t.Mul(A, X)
		r.Mul(&t, A.T())
		r.Sub(&r, X)
		den = na*na*nx + nx + nq
	} else {
		r.Mul(A, X)
		t.Mul(X, A.T())
		r.Add(&r, &t)
	}
	r.Add(&r, Q)
	if den == 0 {
		return mqFro(mqRaw(&r))
	}
	return mqFro(mqRaw(&r)) / den
}

func mqMinEigSym(x []float64, n int) float64 {
	S := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			S.SetSym(i, j, (x[i*n+j]+x[j*n+i])/2)
		}
	}
	var e mat.EigenSym
	if !e.Factorize(S, false) {
		return math.NaN()
	}
	return e.Values(nil)[0]
}

func mqSymmetric(t *testing.T, s *fzSpec, op string, x []float64, n int) {
	t.Helper()
	for i := range n {
		for j := i + 1; j < n; j++ {
			if x[i*n+j] != x[j*n+i] {
				t.Fatalf("%s: X not symmetric at (%d,%d): %v vs %v\n%s", op, i, j, x[i*n+j], x[j*n+i], s)
			}
		}
	}
}

// mqGram returns M·Mᵀ (trans false) or Mᵀ·M for the r×c row-major M,
// exactly symmetric.
func mqGram(mm []float64, r, c int, trans bool) []float64 {
	k, inner := r, c
	at := func(i, l int) float64 { return mm[i*c+l] }
	if trans {
		k, inner = c, r
		at = func(i, l int) float64 { return mm[l*c+i] }
	}
	g := make([]float64, k*k)
	for i := range k {
		for j := i; j < k; j++ {
			v := 0.0
			for l := range inner {
				v += at(i, l) * at(j, l)
			}
			g[i*k+j], g[j*k+i] = v, v
		}
	}
	return g
}

func mqTranspose(a []float64, r, c int) []float64 {
	t := make([]float64, r*c)
	for i := range r {
		for j := range c {
			t[j*r+i] = a[i*c+j]
		}
	}
	return t
}

// FuzzLyap: Lyap (continuous spec) or DLyap (discrete) with Q = BBᵀ
// (v[12] < 0, PSD) or BBᵀ − CᵀC. When the Kronecker form is well
// conditioned the solve must succeed, match the Kronecker solution, be
// exactly symmetric, have a small residual, and be PSD for PSD Q and a
// decidably stable A. Ill-conditioned inputs may error with
// ErrSingularEquation; returned values must still have a small residual.
func FuzzLyap(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n := s.N
		if n == 0 {
			fzSkip(t, "Lyap: no states")
		}
		psd := s.Vals[12] < 0
		q := mqGram(s.B, n, s.M, false)
		if !psd {
			cc := mqGram(s.C, s.P, n, true)
			for k := range q {
				q[k] -= cc[k]
			}
		}
		op := "Lyap"
		solve := Lyap
		if s.Discrete {
			op, solve = "DLyap", DLyap
		}
		X, err := solve(mqDense(n, n, s.A), mqDense(n, n, q), nil)
		fzContract(t, s, op, X != nil, err)
		xr, cond, ok := mqLyapRef(s.A, q, n, s.Discrete)
		if !ok {
			if err != nil {
				fzErr(t, s, op, err, ErrSingularEquation, ErrSchurFailed)
			}
			x := mqRaw(X)
			if r := mqLyapResidual(s.A, x, q, n, s.Discrete); !(r <= 1e-8) {
				fzSkip(t, op+": ill-conditioned value with large residual")
			}
			fzSkip(t, op+": Kronecker form ill-conditioned")
		}
		fzErr(t, s, op, err, ErrSchurFailed)
		x := mqRaw(X)
		mqSymmetric(t, s, op, x, n)
		if r := mqLyapResidual(s.A, x, q, n, s.Discrete); !(r <= 1e-10) {
			t.Fatalf("%s: relative residual %.3g (cond %.3g)\nX=%v\n%s", op, r, cond, x, s)
		}
		diff := make([]float64, n*n)
		for k := range diff {
			diff[k] = x[k] - xr[k]
		}
		tol := 1e3 * float64(n*n) * 0x1p-52 * cond
		if e, w := mqFro(diff), mqFro(xr)+mqFro(q)/math.Max(mqFro(s.A), 1e-300); !(e <= tol*w) {
			t.Fatalf("%s: ‖X−Xref‖ = %.3g > %.3g (cond %.3g)\nX=%v\nXref=%v\n%s", op, e, tol*w, cond, x, xr, s)
		}
		if psd && s.Unstable() == 0 && s.Decidable() {
			if me := mqMinEigSym(x, n); !(me >= -tol*mqFro(xr)) {
				t.Fatalf("%s: stable A, PSD Q, min eig(X) = %.3g\n%s", op, me, s)
			}
		}
	})
}

// FuzzGram: Gram of the delay-free spec model, type by v[12] (0..3).
// Decidably unstable models must return ErrUnstableGramian; decidably
// stable ones the Kronecker-reference gramian (PSD, symmetric), and for the
// factor types an upper triangular R with RᵀR equal to it.
func FuzzGram(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n := s.N
		typ := GramType(int(math.Floor(fzClamp(s.Vals[12], 0, 3.5))))
		sys := s.System(t, fzSysOpts{})
		W, err := Gram(sys, typ)
		fzContract(t, s, "Gram", W != nil, err)
		if n == 0 {
			fzErr(t, s, "Gram", err, ErrDimensionMismatch)
			t.Fatalf("Gram: value for a model without states\n%s", s)
		}
		if !s.Decidable() {
			fzSkip(t, "Gram: pole within error band of stability boundary")
		}
		if s.Unstable() > 0 {
			if !errors.Is(err, ErrUnstableGramian) {
				t.Fatalf("Gram: unstable model gave err=%v, want ErrUnstableGramian\n%s", err, s)
			}
			return
		}
		fzErr(t, s, "Gram", err, ErrSchurFailed)
		a, q := s.A, mqGram(s.B, n, s.M, false)
		if typ == GramObservability || typ == GramObservabilityFactor {
			a, q = mqTranspose(s.A, n, n), mqGram(s.C, s.P, n, true)
		}
		xr, cond, ok := mqLyapRef(a, q, n, s.Discrete)
		if !ok {
			fzSkip(t, "Gram: Kronecker form ill-conditioned")
		}
		w := mqRaw(W)
		if r, c := W.Dims(); r != n || c != n {
			t.Fatalf("Gram: %d×%d, want %d×%d\n%s", r, c, n, n, s)
		}
		if typ == GramControllabilityFactor || typ == GramObservabilityFactor {
			for i := range n {
				for j := range i {
					if w[i*n+j] != 0 {
						t.Fatalf("Gram factor not upper triangular at (%d,%d)=%v\n%s", i, j, w[i*n+j], s)
					}
				}
			}
			w = mqGram(w, n, n, true)
		} else {
			mqSymmetric(t, s, "Gram", w, n)
		}
		tol := 1e3 * float64(n*n) * 0x1p-52 * cond
		diff := make([]float64, n*n)
		for k := range diff {
			diff[k] = w[k] - xr[k]
		}
		scale := mqFro(xr) + mqFro(q)/math.Max(mqFro(a), 1e-300)
		if e := mqFro(diff); !(e <= tol*scale) {
			t.Fatalf("Gram(type %d): ‖W−Wref‖ = %.3g > %.3g (cond %.3g)\nW=%v\nWref=%v\n%s", typ, e, tol*scale, cond, w, xr, s)
		}
		if me := mqMinEigSym(w, n); !(me >= -tol*scale) {
			t.Fatalf("Gram(type %d): min eig %.3g\n%s", typ, me, s)
		}
	})
}

// FuzzCovar: Covar of the delay-free spec model with PSD W = LLᵀ.
// Decidably unstable models must return ErrUnstable; stable ones
// C·X·Cᵀ (+ D·W·Dᵀ discrete) from the Kronecker X, with ±Inf exactly
// where D·W·Dᵀ is nonzero for continuous models.
func FuzzCovar(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m, p := s.N, s.M, s.P
		l := make([]float64, m*m)
		for i := range m {
			for j := 0; j <= i; j++ {
				l[i*m+j] = math.Cos(s.Vals[11] + 1.1*float64(i+1) + 0.6*float64(j+1))
			}
		}
		wd := mqGram(l, m, m, false)
		W := mqDense(m, m, wd)
		sys := s.System(t, fzSysOpts{})
		P, err := Covar(sys, W)
		fzContract(t, s, "Covar", P != nil, err)
		if !s.Decidable() {
			fzSkip(t, "Covar: pole within error band of stability boundary")
		}
		if s.Unstable() > 0 {
			if !errors.Is(err, ErrUnstable) {
				t.Fatalf("Covar: unstable model gave err=%v, want ErrUnstable\n%s", err, s)
			}
			return
		}
		fzErr(t, s, "Covar", err, ErrSchurFailed)
		var dw, dwd mat.Dense
		D := mqDense(p, m, s.D)
		dw.Mul(D, W)
		dwd.Mul(&dw, D.T())
		ref := mat.NewDense(p, p, nil)
		tol, scale := 0.0, 0.0
		if n > 0 {
			var bw, bwb mat.Dense
			B := mqDense(n, m, s.B)
			bw.Mul(B, W)
			bwb.Mul(&bw, B.T())
			q := mqRaw(&bwb)
			for i := range n {
				for j := i + 1; j < n; j++ {
					v := (q[i*n+j] + q[j*n+i]) / 2
					q[i*n+j], q[j*n+i] = v, v
				}
			}
			xr, cond, ok := mqLyapRef(s.A, q, n, s.Discrete)
			if !ok {
				fzSkip(t, "Covar: Kronecker form ill-conditioned")
			}
			var cx mat.Dense
			C := mqDense(p, n, s.C)
			cx.Mul(C, mqDense(n, n, xr))
			ref.Mul(&cx, C.T())
			tol = 1e3 * float64(n*n) * 0x1p-52 * cond
			nc := mqFro(s.C)
			scale = nc * nc * (mqFro(xr) + mqFro(q)/math.Max(mqFro(s.A), 1e-300))
		}
		if s.Discrete {
			ref.Add(ref, &dwd)
			scale += mqFro(mqRaw(&dwd))
			tol = math.Max(tol, 1e3*0x1p-52)
		}
		for i := range p {
			for j := range p {
				g, w := P.At(i, j), ref.At(i, j)
				if v := dwd.At(i, j); !s.Discrete && v != 0 {
					if !math.IsInf(g, int(math.Copysign(1, v))) {
						t.Fatalf("Covar[%d,%d] = %v, want %v·Inf (DWDᵀ=%v)\n%s", i, j, g, math.Copysign(1, v), v, s)
					}
					continue
				}
				if !(math.Abs(g-w) <= tol*scale+1e-300) {
					t.Fatalf("Covar[%d,%d] = %v, want %v (tol %.3g)\nP=%v\nref=%v\n%s", i, j, g, w, tol*scale, mqRaw(P), mqRaw(ref), s)
				}
			}
		}
	})
}

// mqRatInv inverts the m×m rational matrix by Gauss–Jordan; nil if
// singular.
func mqRatInv(a []*big.Rat, m int) []*big.Rat {
	w := 2 * m
	M := make([]*big.Rat, m*w)
	for i := range m {
		for j := range m {
			M[i*w+j] = new(big.Rat).Set(a[i*m+j])
			M[i*w+m+j] = new(big.Rat)
			if i == j {
				M[i*w+m+j].SetInt64(1)
			}
		}
	}
	var t big.Rat
	for c := range m {
		piv := -1
		for i := c; i < m; i++ {
			if M[i*w+c].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil
		}
		for j := range w {
			M[piv*w+j], M[c*w+j] = M[c*w+j], M[piv*w+j]
		}
		inv := new(big.Rat).Inv(M[c*w+c])
		for j := range w {
			M[c*w+j].Mul(M[c*w+j], inv)
		}
		for i := range m {
			if i == c || M[i*w+c].Sign() == 0 {
				continue
			}
			fct := new(big.Rat).Set(M[i*w+c])
			for j := range w {
				M[i*w+j].Sub(M[i*w+j], t.Mul(fct, M[c*w+j]))
			}
		}
	}
	out := make([]*big.Rat, m*m)
	for i := range m {
		for j := range m {
			out[i*m+j] = M[i*w+m+j]
		}
	}
	return out
}

func mqRatMul(a, b []*big.Rat, r, k, c int) []*big.Rat {
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

func mqRatT(a []*big.Rat, r, c int) []*big.Rat {
	t := make([]*big.Rat, r*c)
	for i := range r {
		for j := range c {
			t[j*r+i] = a[i*c+j]
		}
	}
	return t
}

// mqDetPoly returns det(z·L − M) exactly, descending of length d+1, by
// Newton interpolation at z = 0..d.
func mqDetPoly(L, M []*big.Rat, d int) []*big.Rat {
	y := make([]*big.Rat, d+1)
	var t big.Rat
	for k := range y {
		z := big.NewRat(int64(k), 1)
		P := make([]*big.Rat, d*d)
		for i := range P {
			P[i] = new(big.Rat).Sub(t.Mul(z, L[i]), M[i])
		}
		_, y[k] = fzRatElim(P, d, d)
	}
	for j := 1; j <= d; j++ {
		for k := d; k >= j; k-- {
			y[k] = new(big.Rat).Quo(new(big.Rat).Sub(y[k], y[k-1]), big.NewRat(int64(j), 1))
		}
	}
	poly := []*big.Rat{new(big.Rat).Set(y[d])}
	for k := d - 1; k >= 0; k-- {
		poly = fzRatPolyMul(poly, []*big.Rat{big.NewRat(1, 1), big.NewRat(-int64(k), 1)})
		poly[len(poly)-1].Add(poly[len(poly)-1], y[k])
	}
	return poly
}

// mqStabilizable reports whether the spec's (A,B) has a PBH
// stabilizability margin σmin([λI−A, B]) well above the eigenvalue error at
// every intended mode near or outside the stability boundary.
func mqStabilizable(s *fzSpec) bool {
	n, m := s.N, s.M
	ab := math.Hypot(mqFro(s.A), mqFro(s.B))
	band := 1e-3*math.Max(1, ab) + 4*s.EigErr()
	for _, e := range s.Eig {
		near := real(e) > -band
		if s.Discrete {
			near = cmplx.Abs(e) > 1-band
		}
		if !near {
			continue
		}
		g := make([]complex128, n*(n+m))
		for i := range n {
			for j := range n {
				g[i*(n+m)+j] = complex(-s.A[i*n+j], 0)
			}
			g[i*(n+m)+i] += e
			for j := range m {
				g[i*(n+m)+n+j] = complex(s.B[i*m+j], 0)
			}
		}
		sv := fzSingularValues(g, n, n+m)
		if sv == nil || !(sv[n-1] > band) {
			return false
		}
	}
	return true
}

// mqRiccatiExists reports whether a stabilizing solution provably exists
// for the float64 data: mqStabilizable, and the exact Hamiltonian
// (continuous) or symplectic pencil (discrete) of Ā = A − BR⁻¹Sᵀ,
// G = BR⁻¹Bᵀ, Q̄ = Q − SR⁻¹Sᵀ has no root within a relative strip of the
// boundary (discrete: widened by the eps·‖[A Q]‖·‖[A G]‖ sensitivity of
// the symplectic eigenvalues, and only when ‖B‖²‖Q‖/λmin(R) ≤ 1e8 so that
// R + BᵀXB is numerically definite). With S, Q̄ must also be computable
// from the data: eps·cond(R)·‖SR⁻¹Sᵀ‖ well below ‖Q̄‖.
func mqRiccatiExists(s *fzSpec, q, r, sm []float64) bool {
	n, m := s.N, s.M
	if !mqStabilizable(s) {
		return false
	}
	A, B, Q := fzRatPoly(s.A), fzRatPoly(s.B), fzRatPoly(q)
	Ri := mqRatInv(fzRatPoly(r), m)
	if Ri == nil {
		return false
	}
	Bt := mqRatT(B, n, m)
	G := mqRatMul(mqRatMul(B, Ri, n, m, m), Bt, n, m, n)
	if sm != nil {
		S := fzRatPoly(sm)
		F := mqRatMul(Ri, mqRatT(S, n, m), m, m, n)
		BF := mqRatMul(B, F, n, m, n)
		SF := mqRatMul(S, F, n, m, n)
		nsf, nqb := 0.0, 0.0
		for k := range A {
			A[k] = new(big.Rat).Sub(A[k], BF[k])
			Q[k] = new(big.Rat).Sub(Q[k], SF[k])
			f1, _ := SF[k].Float64()
			f2, _ := Q[k].Float64()
			nsf, nqb = math.Hypot(nsf, f1), math.Hypot(nqb, f2)
		}
		if !(1e3*0x1p-52*mat.Cond(mqDense(m, m, r), 2)*nsf <= 1e-3*nqb) {
			return false
		}
	}
	d := 2 * n
	L := make([]*big.Rat, d*d)
	M := make([]*big.Rat, d*d)
	for i := range L {
		L[i], M[i] = new(big.Rat), new(big.Rat)
	}
	hn, na, ng, nq := 0.0, 0.0, 0.0, 0.0
	for i := range n {
		for j := range n {
			a, _ := A[i*n+j].Float64()
			gq, _ := G[i*n+j].Float64()
			qq, _ := Q[i*n+j].Float64()
			na, ng, nq = math.Max(na, math.Abs(a)), math.Max(ng, math.Abs(gq)), math.Max(nq, math.Abs(qq))
			hn = math.Max(na, math.Max(ng, nq))
			if s.Discrete {
				M[i*d+j].Set(A[i*n+j])
				M[(n+i)*d+j].Neg(Q[i*n+j])
				L[i*d+n+j].Set(G[i*n+j])
				L[(n+i)*d+n+j].Set(A[j*n+i])
			} else {
				M[i*d+j].Set(A[i*n+j])
				M[i*d+n+j].Neg(G[i*n+j])
				M[(n+i)*d+j].Neg(Q[i*n+j])
				M[(n+i)*d+n+j].Neg(A[j*n+i])
			}
		}
		L[i*d+i].SetInt64(1)
		if s.Discrete {
			M[(n+i)*d+n+i].SetInt64(1)
		} else {
			L[(n+i)*d+n+i].SetInt64(1)
		}
	}
	p := fzTrimRat(mqDetPoly(L, M, d))
	if len(p) == 0 {
		return false
	}
	if s.Discrete {
		delta := math.Max(1e-6, 1e3*0x1p-52*(1+na+nq)*(1+na+ng))
		nb := mqFro(s.B)
		if delta > 0.1 || !(nb*nb*mqFro(q) <= 1e8*mqMinEigSym(r, m)) {
			return false
		}
		inf := d - (len(p) - 1)
		k1, ok1 := fzSchurCohnOutside(p, 1+delta)
		k2, ok2 := fzSchurCohnOutside(p, 1-delta)
		return ok1 && ok2 && k1+inf == n && k2+inf == n
	}
	delta := 1e-6 * math.Max(1, hn)
	k1, ok1 := fzRouthRight(p, delta)
	k2, ok2 := fzRouthRight(p, -delta)
	return ok1 && ok2 && k1 == n && k2 == n
}

// mqRiccatiResiduals returns the gain residual ‖MK − N‖ with
// M = R (+BᵀXB) and N = BᵀX(A) + Sᵀ, and the Riccati residual, each with
// its scale.
func mqRiccatiResiduals(s *fzSpec, q, r, sm, x, k []float64) (ke, kden, re, rden float64) {
	n, m := s.N, s.M
	A, B, X, K := mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(n, n, x), mqDense(m, n, k)
	Q, R := mqDense(n, n, q), mqDense(m, m, r)
	var xb, lhs, rhs, tmp mat.Dense
	xb.Mul(X, B)
	if s.Discrete {
		var xa mat.Dense
		xa.Mul(X, A)
		rhs.Mul(B.T(), &xa)
		tmp.Mul(B.T(), &xb)
		tmp.Add(&tmp, R)
	} else {
		rhs.Mul(B.T(), X)
		tmp.CloneFrom(R)
	}
	if sm != nil {
		rhs.Add(&rhs, mqDense(n, m, sm).T())
	}
	lhs.Mul(&tmp, K)
	var kr mat.Dense
	kr.Sub(&lhs, &rhs)
	kden = mqFro(mqRaw(&tmp))*mqFro(k) + mqFro(mqRaw(&rhs))
	ke = mqFro(mqRaw(&kr))
	var res1, t1, t2 mat.Dense
	var den float64
	var xbs mat.Dense
	if s.Discrete {
		var xa mat.Dense
		xa.Mul(X, A)
		res1.Mul(A.T(), &xa)
		res1.Sub(&res1, X)
		xbs.Mul(A.T(), &xb)
		na := mqFro(s.A)
		den = na*na*mqFro(x) + mqFro(x)
	} else {
		t1.Mul(A.T(), X)
		t2.Mul(X, A)
		res1.Add(&t1, &t2)
		xbs.CloneFrom(&xb)
		den = 2 * mqFro(s.A) * mqFro(x)
	}
	if sm != nil {
		xbs.Add(&xbs, mqDense(n, m, sm))
	}
	var corr mat.Dense
	corr.Mul(&xbs, K)
	res1.Sub(&res1, &corr)
	res1.Add(&res1, Q)
	den += mqFro(mqRaw(&xbs))*mqFro(k) + mqFro(q)
	if sm != nil {
		var ri, f, srs, xbf mat.Dense
		if err := ri.Inverse(R); err == nil {
			Sm := mqDense(n, m, sm)
			f.Mul(&ri, Sm.T())
			srs.Mul(Sm, &f)
			xbf.Mul(&xb, &f)
			den += mqFro(mqRaw(&srs)) + 2*mqFro(mqRaw(&xbf))
		}
	}
	return ke, kden, mqFro(mqRaw(&res1)), den
}

// mqRiccatiTol bounds the relative Riccati residual of a Schur-method
// solution: eps over the relative separation of the stable and antistable
// eigenvalues (closed-loop margin from mat.Eigen of A−BK) and over Rcnd.
func mqRiccatiTol(s *fzSpec, acl, q, r []float64, rcnd float64) float64 {
	sep, _ := mqRiccatiSep(s, acl, q, r)
	if !(sep > 0) || !(rcnd > 0) {
		return math.Inf(1)
	}
	return math.Max(1e-9, 1e2*0x1p-52/(sep*rcnd))
}

// mqRiccatiSep returns the relative stable/antistable separation 2·margin/h
// and the Hamiltonian scale h = max(‖A‖, ‖BR⁻¹Bᵀ‖, ‖Q‖) (at least 1 for
// discrete); sep is 0 when it cannot be computed.
func mqRiccatiSep(s *fzSpec, acl, q, r []float64) (sep, h float64) {
	n, m := s.N, s.M
	var e mat.Eigen
	if !e.Factorize(mqDense(n, n, acl), mat.EigenNone) {
		return 0, 0
	}
	margin := math.Inf(1)
	for _, v := range e.Values(nil) {
		if s.Discrete {
			margin = math.Min(margin, 1-cmplx.Abs(v))
		} else {
			margin = math.Min(margin, -real(v))
		}
	}
	var ri, g, bri mat.Dense
	if err := ri.Inverse(mqDense(m, m, r)); err != nil {
		return 0, 0
	}
	B := mqDense(n, m, s.B)
	bri.Mul(B, &ri)
	g.Mul(&bri, B.T())
	h = math.Max(mqFro(s.A), math.Max(mqFro(mqRaw(&g)), mqFro(q)))
	if s.Discrete {
		h = math.Max(h, 1)
	}
	return math.Max(0, 2*margin/h), h
}

// mqRiccatiBelowSchurAccuracy reports ‖X‖ below eps·h/sep, the absolute
// accuracy of the Schur method: Care/Dare then return X with no correct
// digits although X is well determined by the data (known followup).
func mqRiccatiBelowSchurAccuracy(s *fzSpec, acl, q, r, x []float64) bool {
	return mqRiccatiSchurRelErr(s, acl, q, r, x) > 1
}

// mqRiccatiSchurRelErr returns eps·h/(sep·‖X‖), the relative accuracy of
// the unscaled Schur method: for ‖X‖ ≪ h (e.g. Q ≈ 1e-12 against ‖A‖ ≈ 1)
// X keeps only that many digits (known followup: no cost scaling).
func mqRiccatiSchurRelErr(s *fzSpec, acl, q, r, x []float64) float64 {
	sep, h := mqRiccatiSep(s, acl, q, r)
	if !(sep > 0) {
		return 0
	}
	return 0x1p-52 * h / (sep * mqFro(x))
}

// FuzzRiccati: Care (continuous spec) or Dare (discrete) with A, B from
// the spec, Q = CᵀC, R = DᵀD + rI (r = 10^clamp(v[13],-3,3)) and cross
// term by v[14]: none (< -3), S = CᵀD (≤ 0, joint PSD cost) or a generic
// texture. A returned solution must be exactly symmetric, satisfy the
// Riccati and gain equations to rounding, give an exactly stable A − BK
// (unless a closed-loop root is within a tiny strip of the boundary) and,
// for a PSD joint cost, be PSD. ErrNoStabilizing is a failure when
// mqRiccatiExists proves a stabilizing solution exists.
func FuzzRiccati(f *testing.F) {
	for _, s := range fzMatEqSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m, p := s.N, s.M, s.P
		if n == 0 {
			fzSkip(t, "Riccati: no states")
		}
		q := mqGram(s.C, p, n, true)
		r := mqGram(s.D, p, m, true)
		rr := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
		for i := range m {
			r[i*m+i] += rr
		}
		var sm []float64
		jointPSD := true
		switch v := s.Vals[14]; {
		case v < -3:
		case v <= 0:
			var cd mat.Dense
			cd.Mul(mqDense(p, n, s.C).T(), mqDense(p, m, s.D))
			sm = mqRaw(&cd)
		default:
			jointPSD = false
			sm = make([]float64, n*m)
			for i := range n {
				for j := range m {
					sm[i*m+j] = math.Sqrt(rr) * math.Cos(s.Vals[11]+0.8*float64(i+1)+1.7*float64(j+1))
				}
			}
		}
		var opts *RiccatiOpts
		if sm != nil {
			opts = &RiccatiOpts{S: mqDense(n, m, sm)}
		}
		op, solve := "Care", Care
		if s.Discrete {
			op, solve = "Dare", Dare
		}
		res, err := solve(mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(n, n, q), mqDense(m, m, r), opts)
		fzContract(t, s, op, res != nil, err)
		if err != nil {
			switch {
			case errors.Is(err, ErrNoStabilizing):
				if (s.Discrete && !jointPSD) || !mqRiccatiExists(s, q, r, sm) {
					fzSkip(t, op+": no stabilizing solution (not provably solvable)")
				}
				t.Fatalf("%s: ErrNoStabilizing, but the Hamiltonian/symplectic pencil is clear of the boundary and (A,B) stabilizable\nQ=%v R=%v S=%v\n%s", op, q, r, sm, s)
			case errors.Is(err, ErrNotPSD):
				if me := mqMinEigSym(q, n); me >= 1e-10*mqFro(q) {
					t.Fatalf("%s: ErrNotPSD for Q with min eig %.3g\n%s", op, me, s)
				}
				fzSkip(t, op+": Q = CᵀC numerically semidefinite rejected")
			case errors.Is(err, ErrSingularR):
				if me := mqMinEigSym(r, m); me >= 1e-10*mqFro(r) {
					t.Fatalf("%s: ErrSingularR for R with min eig %.3g\n%s", op, me, s)
				}
				fzSkip(t, op+": R near singular")
			}
			fzErr(t, s, op, err, ErrSchurFailed)
		}
		x, k := mqRaw(res.X), mqRaw(res.K)
		if r1, c1 := res.X.Dims(); r1 != n || c1 != n {
			t.Fatalf("%s: X is %d×%d\n%s", op, r1, c1, s)
		}
		if r1, c1 := res.K.Dims(); r1 != m || c1 != n {
			t.Fatalf("%s: K is %d×%d\n%s", op, r1, c1, s)
		}
		mqSymmetric(t, s, op, x, n)
		if !mqStabilizable(s) {
			fzSkip(t, op+": (A,B) not robustly stabilizable")
		}
		A, B, K := mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(m, n, k)
		var bk, acl mat.Dense
		bk.Mul(B, K)
		acl.Sub(A, &bk)
		ad := mqRaw(&acl)
		tol := mqRiccatiTol(s, ad, q, r, res.Rcnd)
		if tol > 1e-4 {
			fzSkip(t, op+": ill-conditioned stable subspace")
		}
		ke, kden, re, rden := mqRiccatiResiduals(s, q, r, sm, x, k)
		if !(ke <= 1e-10*kden) {
			t.Fatalf("%s: gain residual %.3g (scale %.3g)\nX=%v\nK=%v\n%s", op, ke, kden, x, k, s)
		}
		tinyX := mqRiccatiBelowSchurAccuracy(s, ad, q, r, x)
		resFail := !(re <= math.Max(tol, 1e2*mqRiccatiSchurRelErr(s, ad, q, r, x))*rden)
		if resFail && !tinyX {
			t.Fatalf("%s: Riccati residual %.3g (scale %.3g, rcnd %.3g)\nX=%v\nK=%v\n%s", op, re, rden, res.Rcnd, x, k, s)
		}
		cp := fzCharPolyExact(ad, n)
		delta := 1e-8 * math.Max(1, mqFro(ad))
		var kk, kd int
		var ok, okd bool
		if s.Discrete {
			kk, ok = fzSchurCohnOutside(cp, 1)
			kd, okd = fzSchurCohnOutside(cp, 1+1e-8)
		} else {
			kk, ok = fzRouthRight(cp, 0)
			kd, okd = fzRouthRight(cp, delta)
		}
		if !ok || kk > 0 {
			if okd && kd == 0 {
				fzSkip(t, op+": closed-loop root within strip of the boundary")
			}
			t.Fatalf("%s: A−BK not stable (%d roots beyond the strip, degenerate %v)\nX=%v\nK=%v\n%s", op, kd, !okd, x, k, s)
		}
		if len(res.Eig) != n {
			t.Fatalf("%s: %d closed-loop eigenvalues, want %d\n%s", op, len(res.Eig), n, s)
		}
		var tr complex128
		for _, e := range res.Eig {
			tr += e
		}
		trA := 0.0
		for i := range n {
			trA += ad[i*n+i]
		}
		hs := 1e6
		if !s.Discrete {
			nb := mqFro(s.B)
			hs = mqFro(q) + nb*nb/rr
		}
		if d := cmplx.Abs(tr - complex(trA, 0)); !(d <= 1e-6*float64(n)*(mqFro(s.A)+mqFro(mqRaw(&bk))+1e-6*hs)) {
			t.Fatalf("%s: Σ Eig = %v, trace(A−BK) = %v\nEig=%v\n%s", op, tr, trA, res.Eig, s)
		}
		const tinyMsg = ": ‖X‖ below Schur-method accuracy eps·h/sep, X inaccurate (known followup)"
		if resFail {
			fzSkip(t, op+tinyMsg)
		}
		if jointPSD {
			if me := mqMinEigSym(x, n); !(me >= -1e-8*mqFro(x)) {
				if tinyX {
					fzSkip(t, op+tinyMsg)
				}
				t.Fatalf("%s: PSD cost but min eig(X) = %.3g\nX=%v\n%s", op, me, x, s)
			}
		}
	})
}

func fzMatEqSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzIsStableSeeds(),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0.3, -0.4}, [3]float64{1.5, -1}
			s.Shear, s.Rot, s.In, s.Out, s.IO = -0.5, 0.9, -1, 0.5, -2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P = 5, 2, 3
			s.Mag, s.Damp = [3]float64{0.2, 0.5, -0.3}, [3]float64{-1.5, 2, 1}
			s.Shear, s.Rot, s.In, s.Out, s.IO = -1, 1.3, 2, 0, 0.5
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 3, 1, 1
			s.Mag, s.Damp = [3]float64{0, 0.5}, [3]float64{13, -2}
			s.In = 1
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -0.5
			s.N, s.M, s.P = 6, 3, 1
			s.Mag, s.Damp = [3]float64{0, -0.2, 0.4}, [3]float64{-3, 2.5, -1.2}
			s.Shear, s.Rot, s.Gain, s.DScal, s.In, s.Out, s.IO = -0.7, -1.4, 7, 0.3, 3, -2, 1
		}),
	)
}
