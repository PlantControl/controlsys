package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"
)

// E2AY73: Dgebal counts the diagonal in its norms, so the state whose A row
// is diagonal-only (a_33 = ε) was scaled until B or C was inflated ~1e13 and
// the staircase dropped a needed state: order 1, den [1 −ε], ~100% error.
// Oracle: EvalFr and MinimalRealization (order 2).
func TestTransferFunctionDgebalDiagonalOnlyRow(t *testing.T) {
	for _, e := range []float64{1e-12, 1e-20, 1e-40, 1e-69} {
		t.Run(fmt.Sprint(e), func(t *testing.T) {
			sys, err := NewFromSlices(3, 1, 1,
				[]float64{1, 0, -3.1622776601683795, 0, 1, -3.1622776601683795, 0, 0, e},
				[]float64{-0.38211320287195655, 0.7882242424655154, 0.8038113259779769},
				[]float64{-0.9291407314223547, 0.6502534890410231, 0.5087003937751763},
				[]float64{0}, 100)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.MinimalOrder != 2 {
				t.Errorf("MinimalOrder %d, want 2", res.MinimalOrder)
			}
			if err := tfMaxRelErr(t, sys, res.TF, []complex128{complex(1.3, 0.2), cmplx.Exp(0.5i), 2}); err > 1e-10 {
				t.Errorf("rel err %g", err)
			}
		})
	}
}

// AULHZC #73: the Dgebal scale [1 1 1 1.14e-13] inflated C, so TF had
// order 2 and 19% error; MinimalRealization gives order 3.
func TestTransferFunctionDgebalInflatedC(t *testing.T) {
	sys, err := NewFromSlices(4, 1, 1,
		[]float64{
			0.7214294796518126, -0.3045829930861377, -0.00041016443744005306, -4.0917982190015664e-18,
			-0.3045829930861377, 0.12979857966165384, -0.0009701301233358639, 1.7299845394387501e-18,
			-0.00041016443744002493, -0.0009701301233358758, 0.0010844921094764262, 2.026087561733075e-44,
			-8.123803525336596e-45, -1.9214602234118876e-44, -2.0260875617330751e-44, -2.319905867103475e-44,
		},
		[]float64{-0.9667981925794609, -0.01238866346289056, 0.9601702866503661, 0.5260775173811046},
		[]float64{-8.568887533689471e-07, -2.1079579943077972e-07, 9.931849187581927e-07, -4.3137684497061916e-07},
		[]float64{0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 3 {
		t.Errorf("MinimalOrder %d, want 3", res.MinimalOrder)
	}
	pts := []complex128{1e-3, complex(1.1613383804294024, 0.06822811668058017), cmplx.Exp(0.5i), 2}
	if err := tfMaxRelErr(t, sys, res.TF, pts); err > 1e-10 {
		t.Errorf("rel err %g", err)
	}
}
