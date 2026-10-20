package controlsys

import (
	"errors"
	"math"
	"slices"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzGramian solves A·X + X·Aᵀ + W = 0 (continuous) or A·X·Aᵀ − X + W = 0
// (discrete) by a dense Kronecker solve, returning X and the condition of
// the solve including the rounding of forming K (‖A‖-scale·‖K⁻¹‖, so that
// a⊗a − 1 cancellation near |a| = 1 counts); ok is false when it fails.
func fzGramian(a, w []float64, n int, discrete bool) (x []float64, cond float64, ok bool) {
	N := n * n
	K := mat.NewDense(N, N, nil)
	for i := range n {
		for j := range n {
			r := i*n + j
			if discrete {
				for k := range n {
					for l := range n {
						K.Set(r, k*n+l, K.At(r, k*n+l)+a[i*n+k]*a[j*n+l])
					}
				}
				K.Set(r, r, K.At(r, r)-1)
				continue
			}
			for k := range n {
				K.Set(r, k*n+j, K.At(r, k*n+j)+a[i*n+k])
				K.Set(r, i*n+k, K.At(r, i*n+k)+a[j*n+k])
			}
		}
	}
	rhs := mat.NewVecDense(N, nil)
	for i := range N {
		rhs.SetVec(i, -w[i])
	}
	var sol mat.VecDense
	if err := sol.SolveVec(K, rhs); err != nil {
		if _, ok := errors.AsType[mat.Condition](err); !ok {
			return nil, 0, false
		}
	}
	x = make([]float64, N)
	for i := range n {
		for j := range n {
			x[i*n+j] = (sol.AtVec(i*n+j) + sol.AtVec(j*n+i)) / 2
		}
	}
	var svd mat.SVD
	if !svd.Factorize(K, mat.SVDNone) {
		return nil, 0, false
	}
	sv := svd.Values(nil)
	scale := 2*fzFrob(a) + 1
	if discrete {
		scale = fzFrob(a)*fzFrob(a) + 1
	}
	return x, scale / sv[N-1], true
}

// fzHSVOracle returns the Hankel singular values from Kronecker-solved
// gramians as eig(Q^½·P·Q^½) (Weyl-stable in P, Q), descending, and an
// absolute error bound on their squares.
func fzHSVOracle(s *fzSpec) (hsv []float64, tol2 float64, ok bool) {
	const eps = 0x1p-52
	n, m, p := s.N, s.M, s.P
	bbT := make([]float64, n*n)
	cTc := make([]float64, n*n)
	for i := range n {
		for j := range n {
			for k := range m {
				bbT[i*n+j] += s.B[i*m+k] * s.B[j*m+k]
			}
			for k := range p {
				cTc[i*n+j] += s.C[k*n+i] * s.C[k*n+j]
			}
		}
	}
	P, kp, ok1 := fzGramian(s.A, bbT, n, s.Discrete)
	Q, kq, ok2 := fzGramian(fzTranspose(s.A, n, n), cTc, n, s.Discrete)
	if !ok1 || !ok2 || math.IsInf(kp, 0) || math.IsInf(kq, 0) {
		return nil, 0, false
	}
	var eq mat.EigenSym
	if !eq.Factorize(mat.NewSymDense(n, Q), true) {
		return nil, 0, false
	}
	var V mat.Dense
	eq.VectorsTo(&V)
	lam := eq.Values(nil)
	half := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			var v float64
			for k := range n {
				v += V.At(i, k) * math.Sqrt(math.Max(lam[k], 0)) * V.At(j, k)
			}
			half.Set(i, j, v)
		}
	}
	var t1, M mat.Dense
	t1.Mul(half, mat.NewDense(n, n, P))
	M.Mul(&t1, half)
	sym := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, (M.At(i, j)+M.At(j, i))/2)
		}
	}
	var em mat.EigenSym
	if !em.Factorize(sym, false) {
		return nil, 0, false
	}
	sq := em.Values(nil)
	sort.Sort(sort.Reverse(sort.Float64Slice(sq)))
	hsv = make([]float64, n)
	for i, v := range sq {
		hsv[i] = math.Sqrt(math.Max(v, 0))
	}
	tol2 = 1e3 * float64(n*n) * eps * (kp + kq + float64(n)) * fzFrob(P) * fzFrob(Q)
	return hsv, tol2, true
}

func fzMatsSpec(s *fzSpec, g [4][]float64, n int) *fzSpec {
	return &fzSpec{N: n, M: s.M, P: s.P, A: g[0], B: g[1], C: g[2], D: g[3], Dt: s.Dt, Discrete: s.Discrete}
}

// FuzzBalred: on decidable delay-free models Balreal/Balred reject exactly
// the unstable ones; otherwise HSV match the Kronecker-gramian oracle,
// TL·TR = I, the balanced model has the original response, and the order-r
// reduction (v[14] → r = ⌊clamp((v+3)/6,0,1)·(n+1)⌋; v[13] ≥ 0 MatchDC
// else Truncate) is stable when σ_r−1 > σ_r, keeps the DC gain under
// MatchDC, and obeys sup_ω σmax(G−Gr) ≤ 2·Σ_{i≥r} σ_i on a pole-clustered
// sweep confirmed by exact rational responses.
func FuzzBalred(f *testing.F) {
	for _, s := range fzBalredSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		const eps = 0x1p-52
		sys := s.System(t, fzSysOpts{})
		br, err := Balreal(sys)
		fzContract(t, s, "Balreal", br != nil, err)
		if s.N == 0 {
			fzErr(t, s, "Balreal", err, ErrDimensionMismatch)
		}
		dec := s.Decidable()
		if dec && s.Unstable() > 0 {
			if !errors.Is(err, ErrUnstable) {
				t.Fatalf("Balreal on %d unstable poles: err %v, want ErrUnstable\n%s", s.Unstable(), err, s)
			}
			return
		}
		if !dec {
			fzSkip(t, "pole within error band of stability boundary")
		}
		if errors.Is(err, ErrUnstable) {
			t.Fatalf("Balreal: ErrUnstable on a decidably stable model\n%s", s)
		}
		fzErr(t, s, "Balreal", err, ErrNotMinimal)
		n := s.N
		hsv, tol2, ok := fzHSVOracle(s)
		if !ok || tol2 > 1e-6*hsv[0]*hsv[0] {
			fzSkip(t, "gramian oracle ill-conditioned")
		}
		if len(br.HSV) != n {
			t.Fatalf("Balreal: %d HSV for n=%d\n%s", len(br.HSV), n, s)
		}
		for i, v := range br.HSV {
			if !(v > 0) || i > 0 && v > br.HSV[i-1] {
				t.Fatalf("Balreal HSV %v not positive and descending\n%s", br.HSV, s)
			}
			if d := math.Abs(v*v - hsv[i]*hsv[i]); d > 10*tol2 {
				t.Fatalf("Balreal HSV[%d] = %v, gramian oracle %v (|Δσ²| %.3g > %.3g)\n%s", i, v, hsv[i], d, 10*tol2, s)
			}
		}
		if hsv[n-1]*hsv[n-1] <= 10*tol2 {
			fzSkip(t, "numerically non-minimal: smallest HSV within oracle error")
		}
		condT := mat.Norm(br.TL, 2) * mat.Norm(br.TR, 2)
		if condT > 1e8 {
			fzSkip(t, "balancing transformation ill-conditioned")
		}
		var I mat.Dense
		I.Mul(br.TL, br.TR)
		for i := range n {
			for j := range n {
				want := 0.0
				if i == j {
					want = 1
				}
				if d := math.Abs(I.At(i, j) - want); d > 1e3*float64(n)*eps*(condT+br.HSV[0]/br.HSV[n-1]) {
					t.Fatalf("Balreal: (TL·TR)[%d,%d] = %v\n%s", i, j, I.At(i, j), s)
				}
			}
		}
		orig := [4][]float64{s.A, s.B, s.C, s.D}
		fzRespCheck(t, s, "Balreal", orig, fzSysMats(br.Sys), n, n, 1e4*float64(n)*condT)

		order := min(int(fzClamp((s.Vals[14]+3)/6, 0, 1)*float64(n+1)), n)
		method := Truncate
		if s.Vals[13] >= 0 {
			method = MatchDC
		}
		gap := order == 0 || order == n || br.HSV[order-1] > br.HSV[order]*(1+1e-6)
		red, hsvR, err := Balred(sys, order, BalredOptions{StateProjection: method})
		fzContract(t, s, "Balred", red != nil, err)
		if !gap {
			fzErr(t, s, "Balred", err, ErrSingularA22, ErrNotMinimal)
		}
		fzErr(t, s, "Balred", err)
		if !slices.Equal(hsvR, br.HSV) {
			t.Fatalf("Balred HSV %v differ from Balreal %v\n%s", hsvR, br.HSV, s)
		}
		nr, _, _ := red.Dims()
		if nr != order {
			t.Fatalf("Balred order %d, want %d\n%s", nr, order, s)
		}
		if !gap {
			fzSkip(t, "repeated HSV at the truncation order")
		}
		fzRequireStable(t, s, "Balred", red)
		var disc float64
		for _, v := range hsv[order:] {
			disc += v
		}
		slack := float64(n)*math.Sqrt(10*tol2) + 1e4*float64(n)*eps*condT*condT*(hsv[0]+fzFrob(s.D))
		bound := 2*disc*(1+1e-6) + slack
		gr := fzSysMats(red)
		rs := fzMatsSpec(s, gr, nr)
		diff := func(x complex128, exact bool) (float64, bool) {
			var g0, g1 []complex128
			if exact {
				var ok0, ok1 bool
				g0, ok0 = fzExactResponse(n, s.M, s.P, s.A, s.B, s.C, s.D, x)
				g1, ok1 = fzExactResponse(nr, s.M, s.P, gr[0], gr[1], gr[2], gr[3], x)
				if !ok0 || !ok1 {
					return 0, false
				}
			} else {
				g0, g1 = fzResponseFloat(s, x), fzResponseFloat(rs, x)
			}
			d := make([]complex128, len(g0))
			for k := range d {
				d[k] = g0[k] - g1[k]
			}
			return fzSigmaMax(d, s.P, s.M), true
		}
		wbest, _ := fzSweepMax(func(w float64) float64 {
			v, _ := diff(fzFreqPoint(s, w), false)
			return v
		}, fzFreqGrid(s.S, s.Dt, 60), 2)
		if e, ok := diff(fzFreqPoint(s, wbest), true); ok && e > bound {
			t.Fatalf("Balred order %d (method %d): σmax(G−Gr) = %.6g at ω=%v exceeds 2·Σ discarded HSV = %.6g (+slack %.3g)\n%s", order, method, e, wbest, 2*disc, slack, s)
		}
		if method == MatchDC {
			x0 := complex(0, 0)
			if s.Discrete {
				x0 = 1
			}
			e, ok := diff(x0, true)
			M := make([]complex128, n*n)
			for i := range n {
				for j := range n {
					M[i*n+j] = complex(-s.A[i*n+j], 0)
				}
				M[i*n+i] += x0
			}
			sv := fzSingularValues(M, n, n)
			amp := math.Max(1, (fzFrob(s.A)+1)/sv[n-1])
			if ok && e > slack*amp*10 {
				t.Fatalf("Balred MatchDC order %d: DC gain differs by %.3g (> %.3g)\n%s", order, e, slack*amp*10, s)
			}
		}
	})
}

// fzBalredSeeds: stable continuous and discrete MIMO models, an unstable
// one, every order fraction and both projections.
func fzBalredSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0], s.IO = 2, 2.5, 0 }),
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, -2.5 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0, 1}, [3]float64{1.5, 0.7}
			s.Shear, s.Rot, s.IO, s.Out = -1, 0.8, -1, 1
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 1, 2
			s.Mag, s.Damp = [3]float64{-0.5, 0.5, 1}, [3]float64{1, 2, 1}
			s.Shear, s.Rot, s.IO, s.Out = -0.7, 1.3, 1.5, -1
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P = 6, 2, 3
			s.Mag, s.Damp = [3]float64{0.3, -0.2, 0.5}, [3]float64{1.2, 1, 2}
			s.Rot, s.IO, s.Out = 1.1, 0.5, 2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.2)
			s.N, s.M, s.P = 3, 2, 1
			s.Mag, s.Damp = [3]float64{0.1, 0.4}, [3]float64{1.5, 1}
			s.IO, s.Out = 3, -1
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 0, 2, 2, 3 }),
	}
}
