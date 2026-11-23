package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// balanceSimilarityK is the C4GXH2 R2 constant of the within-terms
// similarity bound.
const balanceSimilarityK = 8

type balanceSimCase struct {
	name    string
	n, m, p int
	dt      float64
	// xin is the input's exact abscissa: max Re λ, or max |λ|² − 1 for
	// discrete models.
	xin        float64
	a, b, c, d []float64
}

func (tc balanceSimCase) sys(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// balanceSimilarityRatios returns, in 320-bit arithmetic, the C4GXH2 R2
// ratios ‖A_b − TR⁻¹·A·TR‖_F / ((n+1)·ε·(‖|TL|·|A|·|TR|‖_F +
// ‖|TL|·|TR|‖_F·‖A_b‖_F)) and the same form for B_b against TR⁻¹·B.
func balanceSimilarityRatios(t *testing.T, sys *System, br *BalrealResult) (ra, rb float64) {
	t.Helper()
	n, _, _ := sys.Dims()
	TR := hbFrom(br.TR)
	xa := hbSolve(TR, hbMul(hbFrom(sys.A), TR))
	xb := hbSolve(TR, hbFrom(sys.B))
	var p, q, s mat.Dense
	atl, atr := absDense(br.TL), absDense(br.TR)
	p.Mul(atl, absDense(sys.A))
	q.Mul(&p, atr)
	s.Mul(atl, atr)
	pb := mat.Dense{}
	pb.Mul(atl, absDense(sys.B))
	u := float64(n+1) * eps()
	ra = hbFro(hbAdd(hbFrom(br.Sys.A), xa, -1)) / (u * (mat.Norm(&q, 2) + mat.Norm(&s, 2)*mat.Norm(br.Sys.A, 2)))
	rb = hbFro(hbAdd(hbFrom(br.Sys.B), xb, -1)) / (u * (mat.Norm(&pb, 2) + mat.Norm(&s, 2)*mat.Norm(br.Sys.B, 2)))
	return ra, rb
}

func requireBalanceSimilarity(t *testing.T, what string, sys *System, br *BalrealResult) {
	t.Helper()
	ra, rb := balanceSimilarityRatios(t, sys, br)
	if !(ra <= balanceSimilarityK) || !(rb <= balanceSimilarityK) {
		t.Fatalf("%s: similarity ratios A %.3g, B %.3g > K = %d", what, ra, rb, balanceSimilarityK)
	}
}

// exactRootsBeyond counts the roots of det(λI − A) with Re λ > x, or for
// discrete models with |λ|² − 1 > x, exactly.
func exactRootsBeyond(t *testing.T, A *mat.Dense, discrete bool, x float64) int {
	t.Helper()
	n, _ := A.Dims()
	cp := fzCharPolyExact(mat.DenseCopyOf(A).RawMatrix().Data, n)
	var k int
	var ok bool
	if discrete {
		k, ok = fzSchurCohnOutside(cp, math.Sqrt(1+x))
	} else {
		k, ok = fzRouthRight(cp, x)
	}
	if !ok {
		t.Fatalf("degenerate exact root count at %g", x)
	}
	return k
}

// requireAbscissaNear asserts that the exact abscissa of A is negative and
// within rel of x relative (x < 0).
func requireAbscissaNear(t *testing.T, what string, A *mat.Dense, discrete bool, x, rel float64) {
	t.Helper()
	if k := exactRootsBeyond(t, A, discrete, x*(1-rel)); k != 0 {
		t.Fatalf("%s: %d exact roots beyond %.6g, abscissa %.6g − %g relative", what, k, x*(1-rel), x, rel)
	}
	if k := exactRootsBeyond(t, A, discrete, x*(1+rel)); k == 0 {
		t.Fatalf("%s: no exact root beyond %.6g, abscissa %.6g + %g relative", what, x*(1+rel), x, rel)
	}
}

var balanceSimClusters = []balanceSimCase{balSimRepro, balSimNsc587, balSimNscd144, balSimHarvest}

// C4GXH2: near-scalar pole clusters whose float64 balancing pairs TL and TR
// to only about 1e-8 (base returned an exactly unstable realization) give a
// nil realization that is exactly stable at the input's abscissa.
func TestBalreal_NearScalarClusterRealizationStable(t *testing.T) {
	for _, tc := range balanceSimClusters {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			requireAbscissaNear(t, "input", sys.A, tc.dt > 0, tc.xin, 1e-3)
			br, err := Balreal(sys)
			if err != nil {
				t.Fatalf("Balreal: %v", err)
			}
			requireStable(t, "Balreal", br.Sys)
			requireAbscissaNear(t, "Balreal", br.Sys.A, tc.dt > 0, tc.xin, 1e-3)
		})
	}
}

// C4GXH2 R2: Balreal's realization is the similarity of the stored model by
// TR within the rounding of its product terms.
func TestBalreal_RealizationIsSimilarityOfStoredModel(t *testing.T) {
	cases := []struct {
		name string
		sys  func(t *testing.T) *System
	}{
		{"well-conditioned MIMO", balanceMIMOFixture},
		{certRotZ855.name, certRotZ855.sys},
		{certRotZ526.name, certRotZ526.sys},
	}
	for _, tc := range balanceSimClusters {
		cases = append(cases, struct {
			name string
			sys  func(t *testing.T) *System
		}{tc.name, tc.sys})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			br, err := Balreal(sys)
			if err != nil {
				t.Fatalf("Balreal: %v", err)
			}
			requireBalanceSimilarity(t, "Balreal", sys, br)
			if f, tau := exactCertNorm(br.TL, br.TR), certTau(br.HSV); f > tau {
				t.Fatalf("exact ‖TL·TR − I‖ = %.3g > τ = %.3g", f, tau)
			}
		})
	}
}

// C4GXH2 R2: a cluster where base passes IsStable with its margin halved
// (nsc/688) keeps the input's margin.
func TestBalreal_StableButShiftedMarginCorrected(t *testing.T) {
	tc := balSimNsc688
	sys := tc.sys(t)
	u := uncorrectedBalancing(t, sys)
	if k := exactRootsBeyond(t, u.Sys.A, false, tc.xin*0.9); k == 0 {
		t.Fatal("fixture: uncorrected margin not shifted")
	}
	br, err := Balreal(sys)
	if err != nil {
		t.Fatalf("Balreal: %v", err)
	}
	requireAbscissaNear(t, "Balreal", br.Sys.A, false, tc.xin, 1e-3)
}

func balanceMIMOFixture(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(4, 2, 2,
		[]float64{-1, 0.5, 0, 0.25, -0.5, -2, 1, 0, 0, -0.75, -3, 0.5, 0.25, 0, -1, -4},
		[]float64{1, 0, 0.5, 1, -1, 0.5, 0, 2},
		[]float64{1, 0, -0.5, 2, 0, 1, 1, -1},
		[]float64{0, 0.5, 0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// C4GXH2 R3: when ‖fl(TL·TR) − I‖_F is within its rounding term, TL and Sys
// are the uncorrected balancing, bit for bit.
func TestBalreal_PairingErrorWithinRoundingUnchanged(t *testing.T) {
	sys := balanceMIMOFixture(t)
	u := uncorrectedBalancing(t, sys)
	if newBalanceCert(u).pairingInexact() {
		t.Fatal("fixture: pairing error above its rounding term")
	}
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !mat.Equal(br.TL, u.TL) || !mat.Equal(br.TR, u.TR) || !sameModel(br.Sys, u.Sys) {
		t.Fatal("gate-off Balreal differs from the uncorrected balancing")
	}
}

// C4GXH2 R4: a realization IsStable does not confirm is returned with
// ErrAccuracyUnverified.
func TestBalreal_UnconfirmedStabilityReturnsResult(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1, []float64{-1, 0, 0, 0.5}, []float64{1, 1}, []float64{1, 1}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	in := &BalrealResult{Sys: sys, HSV: []float64{1, 0.5}, TL: mat.NewDense(2, 2, []float64{1, 0, 0, 1}), TR: mat.NewDense(2, 2, []float64{1, 0, 0, 1})}
	br, err := confirmBalancedStable("Balreal", in)
	if br != in || !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("confirmBalancedStable = %v, %v; want the result with ErrAccuracyUnverified", br != nil, err)
	}
	stable := &BalrealResult{Sys: balanceMIMOFixture(t)}
	if br, err := confirmBalancedStable("Balreal", stable); br != stable || err != nil {
		t.Fatalf("stable realization: %v, %v", br != nil, err)
	}
}

// C4GXH2: Balred at order n is Balreal, bit for bit and with the same error,
// on gate-on and gate-off models.
func TestBalred_OrderNMatchesBalreal(t *testing.T) {
	cases := append([]balanceSimCase{balSimNsc688}, balanceSimClusters...)
	check := func(t *testing.T, sys *System) {
		n, _, _ := sys.Dims()
		br, err1 := Balreal(sys)
		red, hsv, err2 := Balred(sys, n, BalredOptions{})
		if (err1 == nil) != (err2 == nil) || errors.Is(err1, ErrAccuracyUnverified) != errors.Is(err2, ErrAccuracyUnverified) {
			t.Fatalf("Balreal err %v, Balred(n) err %v", err1, err2)
		}
		if (br == nil) != (red == nil) {
			t.Fatalf("Balreal value %v, Balred(n) value %v", br != nil, red != nil)
		}
		if br != nil && (!sameModel(red, br.Sys) || !mat.Equal(mat.NewDense(1, n, hsv), mat.NewDense(1, n, br.HSV))) {
			t.Fatal("Balred(n) differs from Balreal")
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) { check(t, tc.sys(t)) })
	}
	t.Run("well-conditioned MIMO", func(t *testing.T) { check(t, balanceMIMOFixture(t)) })
}

// C4GXH2 R5: below order n Balred reduces the uncorrected balanced
// realization, including where Balreal corrects TL.
func TestBalred_BelowOrderNReducesUncorrectedRealization(t *testing.T) {
	for _, tc := range []balanceSimCase{balSimNsc688, balSimNsc587} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			if !newBalanceCert(uncorrectedBalancing(t, sys)).pairingInexact() {
				t.Fatal("fixture: gate off")
			}
			for _, proj := range []StateProjection{MatchDC, Truncate} {
				for k := range tc.n {
					red, _, err := Balred(sys, k, BalredOptions{StateProjection: proj})
					if red == nil {
						continue
					}
					want, werr := balredModredOf(t, sys, k, proj)
					if werr != nil || !sameModel(red, want) {
						t.Fatalf("Balred(%d, %v) (err %v) not bit-identical to Modred of the uncorrected realization (%v)", k, proj, err, werr)
					}
				}
			}
		})
	}
}
