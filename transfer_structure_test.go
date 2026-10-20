package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func tfMaxRelErr(t *testing.T, sys *System, tf *TransferFunc, pts []complex128) float64 {
	t.Helper()
	worst := 0.0
	for _, z := range pts {
		want, err := sys.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tf.Eval(z)
		if err != nil {
			t.Fatal(err)
		}
		mag := 0.0
		for i := range want {
			for j := range want[i] {
				mag = math.Max(mag, cmplx.Abs(want[i][j]))
			}
		}
		for i := range want {
			for j := range want[i] {
				worst = math.Max(worst, cmplx.Abs(got[i][j]-want[i][j])/mag)
			}
		}
	}
	return worst
}

// w13 #21: two decoupled 5-state blocks, both driven by both inputs, each
// seen by one output. Each row's denominator has degree 5, as MATLAB tf(ss)
// (sminreal per channel); the full-model staircase rotation left [10 10].
func TestTransferFunctionBlockDecoupledRows(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 13))
	const n, k = 10, 5
	a := mat.NewDense(n, n, nil)
	b := mat.NewDense(n, 2, nil)
	c := mat.NewDense(2, n, nil)
	for blk := range 2 {
		off := blk * k
		for i := range k {
			for j := range k {
				a.Set(off+i, off+j, rng.NormFloat64())
			}
			a.Set(off+i, off+i, a.At(off+i, off+i)-4)
			b.Set(off+i, 0, rng.NormFloat64())
			b.Set(off+i, 1, rng.NormFloat64())
			c.Set(blk, off+i, rng.NormFloat64())
		}
	}
	for _, dt := range []float64{0, 0.1} {
		ad := a
		if dt != 0 {
			ad = mat.NewDense(n, n, nil)
			ad.Scale(0.1, a)
		}
		sys, err := New(ad, b, c, mat.NewDense(2, 2, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(res.RowDegrees, []int{5, 5}) || res.MinimalOrder != 10 {
			t.Errorf("dt=%g: RowDegrees %v MinimalOrder %d, want [5 5] 10", dt, res.RowDegrees, res.MinimalOrder)
		}
		pts := []complex128{0.3i, 1.7i, 5i}
		if dt != 0 {
			pts = []complex128{cmplx.Exp(0.3i), cmplx.Exp(1.7i), -0.5}
		}
		if e := tfMaxRelErr(t, sys, res.TF, pts); e > 1e-10 {
			t.Errorf("dt=%g: TF rel err %g", dt, e)
		}
	}
}

// Unrotated Kalman-form models hide their unstable blocks behind exact
// zeros; the row stage judged them after the staircase rotations had
// turned those zeros into noise and kept hidden unstable poles.
func TestTransferFunctionStructuralKalmanRows(t *testing.T) {
	bad, cases := 0, 0
	for id := range 1200 {
		rng := rand.New(rand.NewPCG(uint64(id), 0x9e3779b97f4a7c15))
		kc := newKalmanCase(rng)
		if kc.rotated {
			continue
		}
		cases++
		res, err := kc.sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range res.RowDegrees {
			if d != kc.r {
				bad++
				t.Errorf("id %d: RowDegrees %v, want all %d", id, res.RowDegrees, kc.r)
				break
			}
		}
		if e := kalmanRespErr(kc.sys, res.TF.Eval); e > 1e-6 {
			t.Errorf("id %d: response rel err %g", id, e)
		}
	}
	if cases < 200 {
		t.Fatalf("only %d unrotated cases", cases)
	}
}

// Rotated Kalman-form models: the staircase rotations and the row's
// Hessenberg reduction leave ~100-200·eps·‖A‖ in a hidden state's
// subdiagonal, which the row stage kept as an unstable pole at the plain
// ctrbf floor, depending on the output scale. The margin is a few times
// eps·‖A‖, so a handful of the 40 variants may still flip with platform
// rounding (amd64 2, arm64 0; 33 of 40 before).
func TestTransferFunctionRowFloorRotatedKalman(t *testing.T) {
	over := 0
	for _, id := range []int{26, 34, 72, 82, 103, 124, 162, 165} {
		rng := rand.New(rand.NewPCG(uint64(id), 0x9e3779b97f4a7c15))
		kc := newKalmanCase(rng)
		for _, k := range []float64{-8, -4, 0, 4, 8} {
			sys := kc.sys.Copy()
			sys.C.Scale(math.Pow(10, k), sys.C)
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, d := range res.RowDegrees {
				if d < kc.r {
					t.Errorf("id %d, C×1e%g: RowDegrees %v below %d", id, k, res.RowDegrees, kc.r)
				}
				if d > kc.r {
					over++
					t.Logf("id %d, C×1e%g: RowDegrees %v, want all %d", id, k, res.RowDegrees, kc.r)
					break
				}
			}
			if e := kalmanRespErr(sys, res.TF.Eval); e > 1e-6 {
				t.Errorf("id %d, C×1e%g: response rel err %g", id, k, e)
			}
		}
	}
	if over > 5 {
		t.Errorf("%d of 40 variants keep hidden states (≤5)", over)
	}
}
