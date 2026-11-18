package controlsys

import (
	"errors"
	"math"
	"testing"
)

// stiffRouteRefuted are nearSingularEDraw models (PCG(seed, 7704), seeds
// 830 and 908, SFF7CU) frozen: E full rank with σ_n ≈ 1.5·k·ε, so
// TransferFunction takes the split of the ill-conditioned E
// (stiffExplicitParts, two parts). The parts' summed transfer function is
// outside the stored response's terms near 1e12j (1.26× and 1.18× the refute
// threshold at 320 bits), and was returned with nil; the Weierstrass
// reduction verifies.
var stiffRouteRefuted = map[string]descriptorLiteral{
	"nse 830": {
		n: 3, m: 1, p: 1,
		a: []float64{0.2158228740484498, 0.8305264019101678, 0.9718217935503646, -0.36122021087470585, 1.2187883331096414, -1.0012983966853084, 0.2775528205297164, 1.2747962685739316, -0.7784444956242289},
		b: []float64{0.6586441444878, -0.9927130534348952, 1.1285319500637605},
		c: []float64{-0.35645438250140216, -0.1153686951690519, -0.05630179431958161},
		d: []float64{0.005776131226906678},
		e: []float64{0.13282946479226734, 0.03340901610619286, 0.6132120818890021, -0.9841323935685651, 0.21400927849309087, 0.054230553744352206, -0.034317165409514065, -0.0337637735412456, -0.40877828505900293},
	},
	"nse 908": {
		n: 3, m: 1, p: 1,
		a: []float64{0.7348163746926777, 1.093876315546675, -0.6171857639345739, 1.3767645475027364, 0.29837135995046254, 0.2618058359285137, 0.8523022337138665, 0.2703541179452867, 0.2983336346234952},
		b: []float64{-0.6465064833350833, 1.13332523875493, -0.6228277821181378},
		c: []float64{-0.9997304172825862, 1.2265899691908255, -2.0451531735142767},
		d: []float64{-0.8844932989295631},
		e: []float64{0.3480195361911391, -0.3521251362405099, 0.2585365561579185, -0.4202764197006433, 0.3683774932254563, -0.46511239776992575, 0.0927546504543273, -0.31184436897091405, -0.517320259780385},
	},
}

// TestTransferFunctionDescriptorStiffRouteVerified checks that a transfer
// function from the split of an ill-conditioned E is certified against the
// pencil like the reduction's: nil only when within the stored response's
// terms (withinTermsWorst at 320 bits), else ErrAccuracyUnverified.
func TestTransferFunctionDescriptorStiffRouteVerified(t *testing.T) {
	for name, l := range stiffRouteRefuted {
		t.Run(name, func(t *testing.T) {
			sys := l.system(t)
			if _, _, ok := stiffExplicitParts(sys); !ok {
				t.Fatal("split of the ill-conditioned E does not apply")
			}
			res, err := sys.TransferFunction(nil)
			if res == nil || valueErr(err) != nil {
				t.Fatalf("TransferFunction = %v, %v; want a value", res, err)
			}
			worst, at := withinTermsWorst(t, sys, res.TF, math.Inf(1))
			if err == nil && worst > 1 {
				t.Errorf("nil error but %.3g× the refute threshold at %v", worst, at)
			}
			if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
				t.Errorf("err = %v", err)
			}
		})
	}
}
