package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestZPKModelUnderResolvedClusterStaysStable checks ZPKModel of a stable
// discrete model, a matched discretization with eight poles clustered near
// z = 1 whose float64 transfer function resolves only five (ZAR4NK review,
// rv4 gen50 seed 315). Sminreal keeps all eight states; Reduce at its
// default tolerance cuts three genuine modes and moves a kept pole outside
// the unit circle. Every pole of the result must stay inside it, as the
// 320-bit eigenvalues of the stored matrices do.
func TestZPKModelUnderResolvedClusterStaysStable(t *testing.T) {
	a := []float64{
		0.99927883380801308, -0.00066623442551771994, 1, 0, 0, 0, 0, 0,
		0, 0.9988655983997069, 1, 0, 0, 0, 0, 0,
		0, 0, 0.99824112066267334, -9.5609123602513613e-05, 11391.04152484306, 3.9964570699785993, 3042.8857805394318, 3.9995141704843755,
		0, 0, 0, 0.99798623997114788, 11391.04152484306, 3.9964570699785993, 3042.8857805394318, 3.9995141704843755,
		0, 0, 0, 0, 0.99822853498929975, 0.00035053134925941453, 0, 0,
		0, 0, 0, 0, -0.00035053134925941453, 0.99822853498929975, 3042.8857805394318, 3.9995141704843755,
		0, 0, 0, 0, 0, 0, 0.99975708524218787, 0.0013142217490952264,
		0, 0, 0, 0, 0, 0, -0.0013142217490952264, 0.99975708524218787,
	}
	b := []float64{0, 0, 1, 1, 0, 1, 0, 1}
	c := []float64{-3.5198921176889935e-20, -5.6563889658802085e-20, 8.4900880969708536e-17, 0, 0, 0, 0, 0}
	sys, err := New(mat.NewDense(8, 8, a), mat.NewDense(8, 1, b), mat.NewDense(1, 8, c), mat.NewDense(1, 1, nil), 1e-3)
	if err != nil {
		t.Fatal(err)
	}

	starts, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, starts)
	if !ok {
		t.Fatal("oracle poles unresolved")
	}
	for _, p := range modes.poles {
		if !(cmplx.Abs(p) < 1-1e-4) {
			t.Fatalf("oracle eigenvalue %v not inside the unit circle", p)
		}
	}

	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 5 || len(res.RowDegrees) != 1 || res.RowDegrees[0] != 5 {
		t.Fatalf("MinimalOrder %d RowDegrees %v, want 5 [5]", res.MinimalOrder, res.RowDegrees)
	}
	poles := res.ZPK.Poles[0][0]
	if len(poles) != 5 || len(res.ZPK.Zeros[0][0]) != 4 {
		t.Fatalf("got %d poles and %d zeros, want 5 and 4", len(poles), len(res.ZPK.Zeros[0][0]))
	}
	for _, p := range poles {
		if !(cmplx.Abs(p) < 1) {
			t.Errorf("pole %v outside the unit circle", p)
		}
	}
}
