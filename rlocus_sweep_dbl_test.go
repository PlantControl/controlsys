package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// Third frozen RootLocus regression sweep (track rlocus, wave 20 review 3).
// Families: a double real pole beside a simple pole 2⁻¹¹…2⁻¹⁴ relative
// apart in exact companion/observable forms (some dyadic-similar, some
// D ≠ 0); Givens-rotated k(s+z)/(s²(s+p)); a triple zero of a biproper
// loop; zeros at the origin of multiplicity 2–3 beside a multiple pole.
// Verdicts and oracles are those of rlocus_sweep_test.go; the golden
// testdata/rlocus_sweep_dbl_base.json holds base verdicts (arm64 AND
// amd64). Regenerate it only with base code: RLOCUS_SWEEPD_WRITE=<path>.

const (
	rlsdCases = 1500
	rlsdSeed1 = 0x5eed2003
	rlsdSeed2 = 0xd0b1e
)

func rlsdCase(r *rand.Rand, poles, zeros []complex128, num []*big.Rat, d float64, form int) rlswCase {
	n := len(poles)
	den := rlswPolyFromRoots(poles)
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := 0; i < n-1; i++ {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -rlswF(den[n-j])
		c[j] = rlswF(new(big.Rat).Sub(num[n-j], new(big.Rat).Mul(fzRat(d), den[n-j])))
	}
	b[n-1] = 1
	desc := "companion"
	if form == 1 || form == 3 {
		at := make([]float64, n*n)
		for i := range n {
			for j := range n {
				at[i*n+j] = a[j*n+i]
			}
		}
		a, b, c = at, c, b
		desc = "observable"
	}
	switch form {
	case 2:
		rlsxSimilar(r, n, a, b, c, false)
		desc = "similar-bidiag"
	case 3:
		rlsxSimilar(r, n, a, b, c, false)
		desc = "observable+similar"
	case 4:
		rlswRotate(r, n, a, b, c)
		desc = "rotated"
	}
	return rlswCase{n: n, a: a, b: b, c: c, d: d, num: num, den: den, poles: poles, zeros: zeros, desc: desc}
}

func rlsdNum(n int, den []*big.Rat, zeros []complex128, k, d float64) []*big.Rat {
	zp := rlswPolyFromRoots(zeros)
	num := make([]*big.Rat, n+1)
	for i := range num {
		num[i] = new(big.Rat).Mul(fzRat(d), den[i])
		if j := i - (n - len(zeros)); j >= 0 {
			num[i].Add(num[i], new(big.Rat).Mul(fzRat(k), zp[j]))
		}
	}
	return num
}

func rlsdGenerate() []rlswCase {
	r := rand.New(rand.NewPCG(rlsdSeed1, rlsdSeed2))
	var cases []rlswCase
	sign := func() float64 { return float64(r.IntN(2)*2 - 1) }
	for len(cases) < rlsdCases {
		e := -20 + r.IntN(28)
		sc := math.Ldexp(1, e)
		var poles, zeros []complex128
		var num []*big.Rat
		d, form := 0.0, r.IntN(4)
		switch fam := r.IntN(10); {
		case fam < 6:
			n := 3 + r.IntN(4)
			p := -float64(8+r.IntN(40)) / 32 * sc
			if r.IntN(4) == 0 {
				p = -p
			}
			poles = []complex128{complex(p, 0), complex(p, 0), complex(p*(1+math.Ldexp(sign(), -(11+r.IntN(4)))), 0)}
			poles = append(poles, rlswGenRoots(r, n-3, sc, false)...)
			nz := r.IntN(n)
			zeros = rlswGenRoots(r, nz, sc, false)[:nz]
			if nz > 0 && imag(zeros[nz-1]) > 0 {
				zeros[nz-1] = complex(real(zeros[nz-1]), 0)
			}
			if r.IntN(7) == 0 {
				d = sign() * float64(1+r.IntN(8)) / 4
			}
			k := sign() * float64(1+r.IntN(15)) / 4 * math.Ldexp(1, e*(n-nz))
			num = rlsdNum(n, rlswPolyFromRoots(poles), zeros, k, d)
		case fam < 8:
			p := sign() * float64(1+r.IntN(64)) / 32 * sc
			z := sign() * float64(1+r.IntN(64)) / 64 * sc
			k := sign() * float64(1+r.IntN(16)) / 8 * math.Ldexp(1, 2*e)
			poles, zeros = []complex128{0, 0, complex(-p, 0)}, []complex128{complex(-z, 0)}
			num = rlsdNum(3, rlswPolyFromRoots(poles), zeros, k, 0)
			form = 4
		case fam < 9:
			n := 4 + r.IntN(3)
			poles = rlswGenRoots(r, n, sc, true)
			z := -float64(r.IntN(64)) / 64 * sc
			zeros = []complex128{complex(z, 0), complex(z, 0), complex(z, 0)}
			zeros = append(zeros, rlswGenRoots(r, n-3, sc, false)...)[:n]
			if imag(zeros[n-1]) > 0 {
				zeros[n-1] = complex(real(zeros[n-1]), 0)
			}
			d = sign() * float64(1+r.IntN(8)) / 4
			num = rlsdNum(n, rlswPolyFromRoots(poles), zeros, d, 0)
			form = r.IntN(2)
		default:
			n := 4 + r.IntN(3)
			m := 2 + r.IntN(2)
			p := -float64(8+r.IntN(40)) / 32 * sc
			poles = []complex128{complex(p, 0), complex(p, 0)}
			if r.IntN(2) == 0 {
				poles = append(poles, complex(p, 0))
			}
			poles = append(poles, rlswGenRoots(r, n-len(poles), sc, true)...)[:n]
			if imag(poles[n-1]) > 0 {
				poles[n-1] = complex(real(poles[n-1]), 0)
			}
			zeros = make([]complex128, m)
			if n-m-1 > 0 && r.IntN(2) == 0 {
				zeros = append(zeros, complex(float64(r.IntN(97)-48)/32*sc, 0))
			}
			k := sign() * float64(1+r.IntN(15)) / 4 * math.Ldexp(1, e*(n-len(zeros)))
			num = rlsdNum(n, rlswPolyFromRoots(poles), zeros, k, 0)
			form = r.IntN(2)
		}
		if len(fzTrimRat(num)) == 0 {
			continue
		}
		cases = append(cases, rlsdCase(r, poles, zeros, num, d, form))
	}
	return cases
}

// rlsdPrecisionLimits: see rlswPrecisionLimits. Every exact double pole is
// 1-ulp sensitive; these are listed because eig cannot order the three
// cluster poles (each u-merged with the others, the pair mean no better
// conditioned than its gap) and the converted D confirms no double, so base
// got them only by its absolute 1e-4 merge radius, which is scale-blind:
// the pinned base 512 near triple has the same coefficient-level signature
// (p(c) below u·Σ|pᵢ||c|ⁱ at the p' root) and no break point.
var rlsdPrecisionLimits = map[int]string{
	128:  "similar-bidiag double -1.25 + simple 2^-12 apart: eig -1.25030, -1.250018, -1.249983 all u-merged (u·κ to 7.5e-3)",
	135:  "companion double -6.40869e-4 + simple 2^-14 apart: eig pair ±9.5e-9i and the simple all u-merged (u·κ 2.3e-7, gaps 4.4e-8)",
	515:  "observable+similar double -0.0292969 + simple: eig -0.028964±5.1e-4i, -0.03005, -0.03118; oInt unposed, float-TF oracle [] while the converted TF keeps the cluster",
	706:  "amd64: observable+similar double -0.34375 + simple 2^-12 apart: eig three reals -0.343678, -0.343713, -0.343774 all u-merged",
	728:  "observable+similar double -9.155e-5 + simple 2^-11 apart: eig resolves none of the cluster; stationary point -5.125e-5 of the float TF has unresolved sign of G",
	1073: "similar-bidiag double -0.052734 + simple 2^-13 apart: eig pair ±4.6e-7i u-merged with the simple (u·κ 1.3e-5, gap 6.5e-6)",
	1099: "similar-bidiag double 7.629e-5 + simple 2^-13 apart: eig pair ±1.3e-10i (8x √u·σ_b), converted D confirms no double",
	1433: "observable double 3.8147e-4 + simple 2^-14 apart: eig pair ±6.1e-9i and the simple all u-merged (u·κ 4.7e-8, gaps 2.6e-8)",
	1436: "observable double 1.0986e-3 + simple 2^-13 apart: eig pair ±8.2e-9i, converted D confirms no double",
	1452: "arm64: observable+similar D=0.5 double -5.371e-3 + simple 2^-11 apart: eig -5.310e-3±1.05e-4i, -5.490e-3 (u·κ 2.4e-3); float-TF oracle drops the double",
}

// TestRootLocusSweepDblNoRegression fails when a component that base got
// right on the third frozen sweep is now wrong (or errors).
func TestRootLocusSweepDblNoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 1500-case sweep")
	}
	skipSlowUnderRace(t)
	rlswCheck(t, rlsdGenerate(), "testdata/rlocus_sweep_dbl_base.json", "RLOCUS_SWEEPD_WRITE", rlsdPrecisionLimits)
}
