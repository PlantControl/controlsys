package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzEraMarkov returns Y0 = D, Yk = C·A^(k-1)·B for k < count by plain
// float64 recursion on the given row-major matrices.
func fzEraMarkov(n, m, p int, a, b, c, d []float64, count int) [][]float64 {
	out := make([][]float64, count)
	out[0] = append([]float64(nil), d...)
	x := append([]float64(nil), b...)
	next := make([]float64, n*m)
	for k := 1; k < count; k++ {
		y := make([]float64, p*m)
		for i := range p {
			for j := range m {
				v := 0.0
				for l := range n {
					v += c[i*n+l] * x[l*m+j]
				}
				y[i*m+j] = v
			}
		}
		out[k] = y
		for i := range n {
			for j := range m {
				v := 0.0
				for l := range n {
					v += a[i*n+l] * x[l*m+j]
				}
				next[i*m+j] = v
			}
		}
		x, next = next, x
	}
	return out
}

func fzEraHankelSV(y [][]float64, r, p, m int) []float64 {
	h := mat.NewDense(r*p, r*m, nil)
	for i := range r {
		for j := range r {
			for bi := range p {
				for bj := range m {
					h.Set(i*p+bi, j*m+bj, y[i+j+1][bi*m+bj])
				}
			}
		}
	}
	var svd mat.SVD
	if !svd.Factorize(h, mat.SVDNone) {
		return nil
	}
	return svd.Values(nil)
}

func fzEraRaw(s *System) (a, b, c, d []float64) {
	n, m, p := s.Dims()
	get := func(x *mat.Dense, r, k int) []float64 {
		v := make([]float64, r*k)
		for i := range r {
			for j := range k {
				v[i*k+j] = x.At(i, j)
			}
		}
		return v
	}
	if n > 0 {
		a, b, c = get(s.A, n, n), get(s.B, n, m), get(s.C, p, n)
	}
	return a, b, c, get(s.D, p, m)
}

// FuzzIdentERA: ERA on the exact Markov parameters of a decoded discrete
// model recovers a realization of order n whose Markov parameters
// reproduce the data within K·eps·σ1·(σ1/σn) (σ: singular values of the
// Hankel matrix, computed independently), whose poles match the intended
// eigenvalues, with D = Y0, HSV = σ, order n−1 valid and order n+1 either
// valid or ErrInvalidOrder. Non-minimal or rank-ambiguous inputs skip.
func FuzzIdentERA(f *testing.F) {
	for _, s := range fzEraSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if !s.Discrete || s.N == 0 {
			fzSkip(t, "ERA needs a discrete model with states")
		}
		n, m, p := s.N, s.M, s.P
		r := n + int(s.Kind>>1&1) + 1
		count := 2*r + 1
		rho := 0.0
		for _, e := range s.Eig {
			rho = math.Max(rho, cmplx.Abs(e))
		}
		if math.Pow(math.Max(rho, 1), float64(count)) > 1e6 {
			fzSkip(t, "Markov growth beyond relative precision")
		}
		y := fzEraMarkov(n, m, p, s.A, s.B, s.C, s.D, count)
		ymax := 0.0
		for _, yk := range y[1:] {
			for _, v := range yk {
				if fzBadFloat(v) {
					fzSkip(t, "denormal Markov parameter")
				}
				ymax = math.Max(ymax, math.Abs(v))
			}
		}
		sv := fzEraHankelSV(y, r, p, m)
		if sv == nil || sv[0] == 0 || len(sv) < n {
			fzSkip(t, "zero Hankel matrix")
		}
		const eps = 0x1p-52
		dims := float64(r * max(p, m))
		libTol := dims * eps * sv[0]
		if sv[n-1] < 1e4*libTol || sv[n-1] < 1e-9*sv[0] {
			fzSkip(t, "numerically non-minimal (σn/σ1 near rank tolerance)")
		}
		kappa := sv[0] / sv[n-1]
		markov := make([]*mat.Dense, count)
		for k := range count {
			markov[k] = mat.NewDense(p, m, append([]float64(nil), y[k]...))
		}

		res, err := ERA(markov, n, s.Dt)
		fzContract(t, s, "ERA", res != nil, err)
		fzErr(t, s, "ERA", err)
		if res.Sys == nil {
			t.Fatalf("ERA: nil system\n%s", s)
		}
		rn, rm, rp := res.Sys.Dims()
		if rn != n || rm != m || rp != p || res.Sys.Dt != s.Dt {
			t.Fatalf("ERA: dims %d×%d×%d dt=%g, want %d×%d×%d dt=%g\n%s", rn, rm, rp, res.Sys.Dt, n, m, p, s.Dt, s)
		}
		if len(res.HSV) != len(sv) {
			t.Fatalf("ERA: %d HSV, want %d\n%s", len(res.HSV), len(sv), s)
		}
		for i := range sv {
			if math.Abs(res.HSV[i]-sv[i]) > 100*dims*eps*sv[0] {
				t.Fatalf("ERA: HSV[%d] = %g, want %g\n%s", i, res.HSV[i], sv[i], s)
			}
		}
		ra, rb, rc, rd := fzEraRaw(res.Sys)
		for k, v := range rd {
			if v != y[0][k] {
				t.Fatalf("ERA: D = %v, want Y0 = %v\n%s", rd, y[0], s)
			}
		}
		got := fzEraMarkov(n, m, p, ra, rb, rc, rd, count)
		tol := 1e3 * dims * eps * math.Max(sv[0], ymax) * kappa
		for k := 1; k < count; k++ {
			for i, v := range got[k] {
				if !(math.Abs(v-y[k][i]) <= tol) {
					t.Fatalf("ERA: Markov Y%d[%d] = %g, want %g (tol %g, σ1/σn=%.3g)\n%s", k, i, v, y[k][i], tol, kappa, s)
				}
			}
		}

		eigTol := 1e3 * s.EigErr() * kappa
		if eigTol < 1e-3*math.Max(1, s.NormJ) {
			var eig mat.Eigen
			if !eig.Factorize(mat.NewDense(n, n, ra), mat.EigenNone) {
				t.Fatalf("ERA: eigen-decomposition of identified A failed\n%s", s)
			}
			fzEraMatchEig(t, s, eig.Values(nil), eigTol)
			if s.Decidable() && fzEraClear(s, eigTol) {
				stable, err := res.Sys.IsStable()
				if err != nil {
					t.Fatalf("ERA: IsStable: %v\n%s", err, s)
				}
				if stable != (s.Unstable() == 0) {
					t.Fatalf("ERA: identified stable=%v, intended unstable poles %d\n%s", stable, s.Unstable(), s)
				}
			}
		}

		if n > 1 {
			lo, err := ERA(markov, n-1, s.Dt)
			fzContract(t, s, "ERA(n-1)", lo != nil, err)
			fzErr(t, s, "ERA(n-1)", err)
			if ln, _, _ := lo.Sys.Dims(); ln != n-1 {
				t.Fatalf("ERA(n-1): order %d\n%s", ln, s)
			}
			for _, v := range fzEraFlat(lo.Sys) {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("ERA(n-1): non-finite realization\n%s", s)
				}
			}
		}
		if r >= n+1 && n+1 <= len(sv) {
			hi, err := ERA(markov, n+1, s.Dt)
			fzContract(t, s, "ERA(n+1)", hi != nil, err)
			if err != nil && !errors.Is(err, ErrInvalidOrder) {
				t.Fatalf("ERA(n+1): error %v, want ErrInvalidOrder or a value\n%s", err, s)
			}
			if err == nil && sv[n] < libTol/1e3 {
				t.Fatalf("ERA(n+1): accepted order above Hankel rank (σ%d/σ1 = %g)\n%s", n+1, sv[n]/sv[0], s)
			}
		}
	})
}

func fzEraFlat(s *System) []float64 {
	a, b, c, d := fzEraRaw(s)
	return append(append(append(a, b...), c...), d...)
}

func fzEraMatchEig(t *testing.T, s *fzSpec, got []complex128, tol float64) {
	t.Helper()
	used := make([]bool, len(got))
	for _, want := range s.Eig {
		best, bi := math.Inf(1), -1
		for i, g := range got {
			if !used[i] && cmplx.Abs(g-want) < best {
				best, bi = cmplx.Abs(g-want), i
			}
		}
		if bi < 0 || best > tol*math.Max(1, cmplx.Abs(want)) {
			t.Fatalf("ERA: poles %v, want %v (tol %g)\n%s", got, s.Eig, tol, s)
		}
		used[bi] = true
	}
}

func fzEraClear(s *fzSpec, tol float64) bool {
	for _, e := range s.Eig {
		if s.Margin(e) <= 4*tol*math.Max(1, cmplx.Abs(e)) {
			return false
		}
	}
	return true
}

func fzEraSeeds() []fzSeed {
	return []fzSeed{
		{Discrete: true, N: 1, M: 1, P: 1, Mag: [3]float64{0}, Damp: [3]float64{2}, Gain: 6, LogDt: -1, Shear: -10},
		{Discrete: true, N: 2, M: 1, P: 1, Mag: [3]float64{0.3}, Damp: [3]float64{1.5}, Gain: 6, LogDt: -1, Shear: -1, Rot: 0.4},
		{Discrete: true, N: 3, M: 2, P: 1, Mag: [3]float64{0.5, -0.3}, Damp: [3]float64{1.2, 3}, Gain: 6, LogDt: -1, Shear: -0.5, Rot: 0.7, Phase: 0.3},
		{Discrete: true, N: 4, M: 2, P: 2, Mag: [3]float64{0.2, 0.8}, Damp: [3]float64{1.7, 2.2}, Gain: 7, DScal: 0.5, LogDt: -1.2, Shear: -1, Rot: 1.1, Phase: 1.3},
		{Discrete: true, N: 5, M: 3, P: 2, Mag: [3]float64{0.1, 0.6, -0.2}, Damp: [3]float64{1.4, 2.5, 3}, Gain: 5, LogDt: -0.8, Shear: -2, Rot: -0.6, Phase: 2.1},
		{Discrete: true, N: 6, M: 3, P: 3, Mag: [3]float64{0.1, 0.5, 0.9}, Damp: [3]float64{1.4, 2.0, 1.1}, Gain: 6, DScal: -1, LogDt: -1.1, Shear: -1.5, Rot: 0.9, Phase: -0.7},
		{Discrete: true, N: 2, M: 1, P: 2, Mag: [3]float64{0.2}, Damp: [3]float64{-1.8}, Gain: -6, LogDt: -1, Shear: -1, Rot: 0.3},
		{Discrete: true, N: 3, M: 1, P: 1, Mag: [3]float64{1, -0.5}, Damp: [3]float64{13, 2}, Gain: 6, LogDt: -1.5, Shear: -1, Rot: 0.5},
	}
}
