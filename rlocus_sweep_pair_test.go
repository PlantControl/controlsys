package controlsys

import (
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"
)

// Fresh seeded RootLocus sweep (track rlocus, wave 20 review 4): a
// resolved complex pole pair p(1+2^−k1) ± |p|·2^−k2·i beside a simple pole
// p, 0–3 further poles, D ≠ 0, in the rlsdCase forms. Negative controls
// (every fifth case) carry an exact double or triple pole p instead, whose
// K = 0 break point must stay; another fifth has a double or triple real
// zero of G − D (one break point, a multiple root of N'D − ND'). Judged by the exact oracles of
// rlocus_sweep_test.go; base 03edce8 is compared offline
// (RLOCUS_SWEEPP_DUMP=<path> writes the verdicts), so the test fails only
// on a duplicate point and logs the counts.

const (
	rlspCases = 1000
	rlspSeed1 = 0x5eed2004
	rlspSeed2 = 0xba1d
)

func rlspGenerate() []rlswCase {
	r := rand.New(rand.NewPCG(rlspSeed1, rlspSeed2))
	sign := func() float64 { return float64(r.IntN(2)*2 - 1) }
	var cases []rlswCase
	for len(cases) < rlspCases {
		e := -20 + r.IntN(28)
		sc := math.Ldexp(1, e)
		p := -float64(8+r.IntN(40)) / 32 * sc
		if r.IntN(4) == 0 {
			p = -p
		}
		var poles []complex128
		mz := 0
		switch {
		case len(cases)%5 == 3:
			poles = rlswGenRoots(r, 3+r.IntN(3), sc, true)
			mz = 2 + r.IntN(2)
		case len(cases)%5 == 4:
			poles = []complex128{complex(p, 0), complex(p, 0)}
			if r.IntN(2) == 0 {
				poles = append(poles, complex(p, 0))
			}
		default:
			re := p * (1 + sign()*math.Ldexp(1, -(8+r.IntN(9))))
			im := math.Abs(p) * math.Ldexp(1, -(8+r.IntN(9)))
			poles = []complex128{complex(p, 0), complex(re, im), complex(re, -im)}
		}
		if mz == 0 {
			poles = append(poles, rlswGenRoots(r, r.IntN(4), sc, false)...)
		}
		n := len(poles)
		nz := r.IntN(n)
		if mz > 0 {
			mz = min(mz, n-1)
			nz = mz + r.IntN(n-mz)
		}
		zeros := rlswGenRoots(r, nz-mz, sc, false)[:nz-mz]
		if nz > mz && imag(zeros[nz-mz-1]) > 0 {
			zeros[nz-mz-1] = complex(real(zeros[nz-mz-1]), 0)
		}
		if mz > 0 {
			z := complex(float64(r.IntN(97)-48)/32*sc, 0)
			zeros = append(zeros, slices.Repeat([]complex128{z}, mz)...)
		}
		d := sign() * float64(1+r.IntN(8)) / 4
		k := sign() * float64(1+r.IntN(15)) / 4 * math.Ldexp(1, e*(n-nz))
		cases = append(cases, rlsdCase(r, poles, zeros, rlsdNum(n, rlswPolyFromRoots(poles), zeros, k, d), d, r.IntN(5)))
	}
	return cases
}

func TestRootLocusSweepPair(t *testing.T) {
	if testing.Short() {
		t.Skip("seeded 1000-case sweep")
	}
	skipSlowUnderRace(t)
	cases := rlspGenerate()
	ok, posed, dups := rlswRun(t, cases)
	if path := os.Getenv("RLOCUS_SWEEPP_DUMP"); path != "" {
		if err := os.WriteFile(path, []byte(rlswHex(ok)+"\n"+rlswHex(posed)+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var np, nok, ctlPosed, ctlOK int
	for i := range cases {
		if posed[i]&rlswBreak == 0 {
			continue
		}
		np++
		good := ok[i]&rlswBreak != 0
		if good {
			nok++
		}
		if i%5 >= 3 {
			ctlPosed++
			if good {
				ctlOK++
			}
		}
	}
	t.Logf("Breakaway posed %d, ok %d (multiple-pole/zero cases posed %d, ok %d)", np, nok, ctlPosed, ctlOK)
	for k, i := range dups {
		if k < 5 {
			sys, _ := cases[i].system()
			res, _ := RootLocus(sys, []float64{0, 1})
			t.Errorf("case %d: Breakaway %v lists a point twice\n%s", i, res.Breakaway, &cases[i])
		}
	}
}
