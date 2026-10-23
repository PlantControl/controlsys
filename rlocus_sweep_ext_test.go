package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// Second frozen RootLocus regression sweep (track rlocus, wave 20), built
// from the wave-19 review families: D ≠ 0 at 20% (either sign, often with a
// strictly proper part of relative degree ≥ 2), non-orthogonal dyadic
// similarities T·A·T⁻¹ (T unit upper triangular), Givens rotations,
// observable companions, near-multiple poles, a double pole beside a simple
// pole 2⁻¹² relative apart, an origin double pole and slow loops down to
// 2⁻²⁰. Verdicts and oracles are those of rlocus_sweep_test.go; the golden
// testdata/rlocus_sweep_ext_base.json holds base verdicts (arm64 AND amd64).
// Regenerate it only with base code: RLOCUS_SWEEPX_WRITE=<path>.

const (
	rlsxCases = 2000
	rlsxSeed1 = 0x5eed20
	rlsxSeed2 = 0x0dd5eed
)

var rlsxCycle = []float64{-0.75, 1.5, -2, 0.25, 3, -0.5, 1, -1, 0.5, 2}

func rlsxGenRoots(r *rand.Rand, cnt int, sc float64) []complex128 {
	if cnt >= 3 {
		switch r.IntN(8) {
		case 0:
			p := -float64(8+r.IntN(40)) / 32 * sc
			out := []complex128{complex(p, 0), complex(p, 0), complex(p*(1+math.Ldexp(1, -12)), 0)}
			return append(out, rlswGenRoots(r, cnt-3, sc, false)...)
		case 1:
			return append([]complex128{0, 0}, rlswGenRoots(r, cnt-2, sc, false)...)
		}
	}
	return rlswGenRoots(r, cnt, sc, true)
}

// rlsxSimilar applies A ← T·A·T⁻¹, b ← T·b, c ← c·T⁻¹ with a unit upper
// triangular dyadic T (bidiagonal or full), computed exactly and rounded once.
func rlsxSimilar(r *rand.Rand, n int, a, b, c []float64, full bool) {
	rlsxSimilarAt(r.IntN(len(rlsxCycle)), n, a, b, c, full)
}

func rlsxSimilarAt(off, n int, a, b, c []float64, full bool) {
	T := make([]*big.Rat, n*n)
	k := 0
	for i := range n {
		for j := range n {
			switch {
			case i == j:
				T[i*n+j] = big.NewRat(1, 1)
			case j == i+1 || (full && j > i):
				T[i*n+j] = fzRat(rlsxCycle[(off+k)%len(rlsxCycle)])
				k++
			default:
				T[i*n+j] = new(big.Rat)
			}
		}
	}
	Ti := make([]*big.Rat, n*n)
	for j := range n {
		for i := n - 1; i >= 0; i-- {
			x := new(big.Rat)
			if i == j {
				x.SetInt64(1)
			}
			for m := i + 1; m < n; m++ {
				x.Sub(x, new(big.Rat).Mul(T[i*n+m], Ti[m*n+j]))
			}
			Ti[i*n+j] = x
		}
	}
	mul := func(x, y []*big.Rat, rows, inner, cols int) []*big.Rat {
		out := make([]*big.Rat, rows*cols)
		for i := range rows {
			for j := range cols {
				s := new(big.Rat)
				for m := range inner {
					s.Add(s, new(big.Rat).Mul(x[i*inner+m], y[m*cols+j]))
				}
				out[i*cols+j] = s
			}
		}
		return out
	}
	rat := func(v []float64) []*big.Rat {
		out := make([]*big.Rat, len(v))
		for i, x := range v {
			out[i] = fzRat(x)
		}
		return out
	}
	ta := mul(mul(T, rat(a), n, n, n), Ti, n, n, n)
	tb := mul(T, rat(b), n, n, 1)
	tc := mul(rat(c), Ti, 1, n, n)
	for i := range a {
		a[i] = rlswF(ta[i])
	}
	for i := range n {
		b[i], c[i] = rlswF(tb[i]), rlswF(tc[i])
	}
}

func rlsxGenerate() []rlswCase {
	r := rand.New(rand.NewPCG(rlsxSeed1, rlsxSeed2))
	var cases []rlswCase
	for len(cases) < rlsxCases {
		n := 2 + r.IntN(5)
		e := -20 + r.IntN(28)
		sc := math.Ldexp(1, e)
		poles := rlsxGenRoots(r, n, sc)
		nz := r.IntN(n)
		if r.IntN(3) == 0 {
			nz = r.IntN(n - 1)
		}
		zeros := rlswGenRoots(r, nz, sc, r.IntN(4) == 0)
		if len(zeros) > nz {
			zeros = zeros[:nz]
			if nz > 0 && imag(zeros[nz-1]) > 0 {
				zeros[nz-1] = complex(real(zeros[nz-1]), 0)
			}
		}
		sign := 1.0
		if r.IntN(2) == 0 {
			sign = -1
		}
		k := sign * float64(1+r.IntN(15)) / 4 * math.Ldexp(1, e*(n-nz))
		d := 0.0
		if r.IntN(5) == 0 {
			d = float64(r.IntN(2)*2-1) * float64(1+r.IntN(8)) / 4
		}
		den := rlswPolyFromRoots(poles)
		zp := rlswPolyFromRoots(zeros)
		num := make([]*big.Rat, n+1)
		for i := range num {
			num[i] = new(big.Rat).Mul(fzRat(d), den[i])
			if j := i - (n - nz); j >= 0 {
				num[i].Add(num[i], new(big.Rat).Mul(fzRat(k), zp[j]))
			}
		}
		if len(fzTrimRat(num)) == 0 {
			continue
		}
		a := make([]float64, n*n)
		b := make([]float64, n)
		c := make([]float64, n)
		for i := 0; i < n-1; i++ {
			a[i*n+i+1] = 1
		}
		for j := range n {
			a[(n-1)*n+j] = -rlswF(den[n-j])
			c[j] = rlswF(new(big.Rat).Sub(num[n-j], new(big.Rat).Mul(fzRat(d), den[n-j])))
		}
		b[n-1] = 1
		desc := "companion"
		kind := r.IntN(10)
		if kind == 1 || kind == 8 {
			at := make([]float64, n*n)
			for i := range n {
				for j := range n {
					at[i*n+j] = a[j*n+i]
				}
			}
			a, b, c = at, c, b
			desc = "observable"
		}
		switch kind {
		case 2, 3:
			rlswRotate(r, n, a, b, c)
			desc = "rotated"
		case 4, 5:
			rlsxSimilar(r, n, a, b, c, false)
			desc = "similar-bidiag"
		case 6:
			rlsxSimilar(r, n, a, b, c, true)
			desc = "similar-full"
		case 7:
			rlswRotate(r, n, a, b, c)
			rlsxSimilar(r, n, a, b, c, false)
			desc = "rotated+similar"
		case 8:
			rlsxSimilar(r, n, a, b, c, false)
			desc = "observable+similar"
		case 9:
			for i := range n {
				ks := r.IntN(9) - 4
				for j := range n {
					a[i*n+j] = math.Ldexp(a[i*n+j], ks)
					a[j*n+i] = math.Ldexp(a[j*n+i], -ks)
				}
				b[i] = math.Ldexp(b[i], ks)
				c[i] = math.Ldexp(c[i], -ks)
			}
			desc = "scaled"
		}
		cs := rlswCase{n: n, a: a, b: b, c: c, d: d, num: num, den: den, poles: poles, zeros: zeros, desc: desc}
		if r.IntN(6) == 0 {
			cs.dt, cs.L = 1, 1+r.IntN(3)
		}
		cases = append(cases, cs)
	}
	return cases
}

// rlsxPrecisionLimits: see rlswPrecisionLimits. All are rotated+similar
// realizations whose eigenvalues eps-level entry changes move by O(1)
// relative.
var rlsxPrecisionLimits = map[int]string{
	215:  "amd64: rotated poles 0, 0, -1.3e-6, -3.8e-6 in a unit-norm A: float charpoly 2.8e-6, -1.8e-6±2.9e-6i, -4.4e-6 (eig ~5e-5 off); 1-ulp perturbations give [] or one root 6.5e-7..9.1e-7",
	394:  "D=1.75, rotated triple pole 4.395e-3: float charpoly splits it into 4.37e-3±4.2e-5i, 4.44e-3, and the float TF has no breakaway there (oFlt [-7.3e-3]); amd64 eig resolves a near pair",
	477:  "triple pole -15.25: float charpoly splits it by 2e-3 (complex pair); 1-ulp perturbations give [], unposed or a pair -15.25002/-15.24998",
	685:  "rotated near triple pole -2.9564e-5 (double + 7.2e-9 apart) in a unit-norm A: float charpoly -2.9525e-5, -2.9587e-5±3.5e-8i; 1-ulp perturbations give [], a pair -2.9567e-5/-2.9565e-5 or a far root 1.7e16",
	885:  "D=-1.5, double pole -1.648e-3: float charpoly poles -1.53e-3±2.5e-4i, -1.94e-3; 1-ulp perturbations give [] or far pairs ±1.7e8..4.6e8",
	958:  "rotated triple pole -1.5259e-4 in a unit-norm A: float charpoly -1.5263e-4, -1.5257e-4±3.7e-8i, oFlt unposed; 1-ulp perturbations give [-3.553e-4 1.048e-4] (no triple point, as returned) or unposed",
	1042: "poles ~1e-3: 1-ulp perturbations move the real pole 7.9e-4..9.3e-4 and give [], a far root -1.6e8..-2.6e8 or three near ones",
	1567: "rotated near triple pole -3.967e-4 (double + 9.7e-8 apart) in a unit-norm A: float charpoly -4.008e-4, -3.948e-4±3.5e-6i (1% off); oFlt and 1-ulp perturbations give only far roots ±1.4e16..5.3e17",
	1647: "D=-1.25, poles ~1e-5: 1-ulp perturbations move the poles to ~1.5e-4 (10x) and leave the oracle unposed",
	1752: "amd64: rotated double pole at 0: float charpoly ±6.4e-9i (oFlt only a far root 4e8) and amd64 eig splits it ±7.2e-8i, past the √u·σ origin window; 1-ulp perturbations give ~1e-15..1e-19 or that plus a far root 1.5e8..2.6e8",
	1766: "relative degree 6, poles ~5e-6: float charpoly poles ~2.3e-3; 1-ulp perturbations give [] or one root in -7.3e-4..-5.9e-4",
}

// TestRootLocusSweepExtNoRegression fails when a component that base got
// right on the second frozen sweep is now wrong (or errors).
func TestRootLocusSweepExtNoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2000-case sweep")
	}
	skipSlowUnderRace(t)
	rlswCheck(t, rlsxGenerate(), "testdata/rlocus_sweep_ext_base.json", "RLOCUS_SWEEPX_WRITE", rlsxPrecisionLimits)
}
