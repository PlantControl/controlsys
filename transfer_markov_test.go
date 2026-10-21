package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var markovPlantA = []float64{
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
	-0.12647558884588878, -0.9280652544060892, -2.237882725322083, -2.3914032576451225,
}

const markovPlantZero = -0.15444209679158707

func markovPlant(t *testing.T, c []float64) *System {
	t.Helper()
	sys, err := NewFromSlices(4, 1, 1, markovPlantA, []float64{0, 0, 0, 1}, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func requireMatchedDCGain(t *testing.T, name string, sys *System) {
	t.Helper()
	want, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	for _, dt := range []float64{0.1, 1.03} {
		d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("%s dt %v: C2D matched: %v", name, dt, err)
		}
		got, err := d.EvalFr(1)
		if err != nil {
			t.Fatal(err)
		}
		if cmplx.Abs(got[0][0]-want[0][0]) > 1e-9*cmplx.Abs(want[0][0]) {
			t.Errorf("%s dt %v: matched DC gain %v, want %v", name, dt, got[0][0], want[0][0])
		}
	}
}

func TestTransferFunctionRelativeDegreeNoise(t *testing.T) {
	sys := markovPlant(t, []float64{-markovPlantZero, 1, 0, 0})
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	num := res.TF.Num[0][0]
	if len(num) != 2 || math.Abs(num[0]-1) > 1e-12 || math.Abs(num[1]+markovPlantZero) > 1e-12 {
		t.Fatalf("num %v, want [1 %v]", num, -markovPlantZero)
	}

	zpk, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	pencil, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	zs := zpk.ZPK.Zeros[0][0]
	if len(pencil) != 1 || len(zs) != 1 || cmplx.Abs(zs[0]-pencil[0]) > 1e-12 || cmplx.Abs(zs[0]-markovPlantZero) > 1e-12 {
		t.Fatalf("ZPKModel zeros %v, Zeros %v, want [%v]", zs, pencil, markovPlantZero)
	}
	if g := zpk.ZPK.Gain[0][0]; math.Abs(g-1) > 1e-12 {
		t.Fatalf("ZPKModel gain %v, want 1 (C·A²·B)", g)
	}
	requireMatchedDCGain(t, "companion", sys)

	z, err := NewZPK([]complex128{markovPlantZero}, zpk.ZPK.Poles[0][0], 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	obs, err := z.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	requireMatchedDCGain(t, "observable companion", obs.Sys)
}

func TestTransferFunctionRelativeDegreeDense(t *testing.T) {
	tm := mat.NewDense(4, 4, []float64{
		2, -1, 0.5, 0.3,
		0.7, 3, -0.2, 1,
		-0.4, 0.6, 1.5, -0.8,
		1.1, -0.3, 0.9, 2.5,
	})
	var ti mat.Dense
	if err := ti.Inverse(tm); err != nil {
		t.Fatal(err)
	}
	var a, b, c, tmp mat.Dense
	tmp.Mul(tm, mat.NewDense(4, 4, markovPlantA))
	a.Mul(&tmp, &ti)
	b.Mul(tm, mat.NewDense(4, 1, []float64{0, 0, 0, 1}))
	c.Mul(mat.NewDense(1, 4, []float64{-markovPlantZero, 1, 0, 0}), &ti)
	sys, err := New(&a, &b, &c, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if num := res.TF.Num[0][0]; len(num) != 2 {
		t.Fatalf("num %v, want degree 1", num)
	}
	zpk, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if zs := zpk.ZPK.Zeros[0][0]; len(zs) != 1 || cmplx.Abs(zs[0]-markovPlantZero) > 1e-9 {
		t.Fatalf("zeros %v, want [%v]", zs, markovPlantZero)
	}
	if g := zpk.ZPK.Gain[0][0]; math.Abs(g-1) > 1e-9 {
		t.Fatalf("gain %v, want 1", g)
	}
	requireMatchedDCGain(t, "dense", sys)
}

func TestTransferFunctionKeepsResolvedLeadingCoefficient(t *testing.T) {
	sys := markovPlant(t, []float64{-markovPlantZero, 1, 0, 1e-6})
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	num := res.TF.Num[0][0]
	if len(num) != 4 || math.Abs(num[0]-1e-6) > 1e-15 {
		t.Fatalf("num %v, want leading C·B = 1e-6 kept", num)
	}
}

// A normwise noise floor in the Dgebal-scaled Hessenberg basis would drop
// this C·B ≈ 2.37, resolved by the data to full precision.
func TestTransferFunctionKeepsBadlyScaledLeadingCoefficient(t *testing.T) {
	a := []float64{
		-0.994305989795155, 0, 270.2881663379678, -0.6698537112714668, 0,
		-54.52373306646294, -0.0043658812494023445, -0.00023532525529211683, 1812.2681666291357, -1322.641331629807,
		0, 0, -8.449380626297876e-05, 0, 0,
		0.1465471607339703, 0, 0.06643075306620273, 0.16429010970310684, 0,
		0.06745456144880582, 0, 0, 0, -0.06645911668006037,
	}
	b := []float64{0, 0, -1.041564850307128, -0.008342229291739755, 6.538208485331508}
	c := []float64{
		0, 21.960720072330407, 726.6545545417971, 0, -856.9630148436084,
		0, -3043.678203624804, -1.4174721421666965, 0.20335238182148607, 0.13624260971861127,
	}
	sys, err := NewFromSlices(5, 1, 2, a, b, c, []float64{0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	cb := 0.0
	for k := range 5 {
		cb += c[5+k] * b[k]
	}
	num := res.TF.Num[1][0]
	if len(num) != 5 || math.Abs(num[0]-cb) > 0.2*math.Abs(cb) {
		t.Fatalf("row 1 num %v, want degree 4 with leading ≈ C·B = %v", num, cb)
	}
}

func TestZPKRoundTripRelativeDegree(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for trial := range 300 {
		np := 2 + rng.IntN(5)
		nz := rng.IntN(np)
		var poles, zeros []complex128
		for len(poles) < np {
			if np-len(poles) >= 2 && rng.IntN(2) == 0 {
				re, im := -0.05-2*rng.Float64(), 0.05+2*rng.Float64()
				poles = append(poles, complex(re, im), complex(re, -im))
			} else {
				poles = append(poles, complex(-0.05-2*rng.Float64(), 0))
			}
		}
		for range nz {
			zeros = append(zeros, complex(-0.05-2*rng.Float64(), 0))
		}
		k := 0.1 + 3*rng.Float64()
		z, err := NewZPK(zeros, poles, k, 0)
		if err != nil {
			t.Fatal(err)
		}
		ss, err := z.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		got, err := ss.Sys.ZPKModel(nil)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if len(got.ZPK.Zeros[0][0]) != nz || math.Abs(got.ZPK.Gain[0][0]-k) > 1e-9*k {
			t.Fatalf("trial %d: zeros %v gain %v, want %v gain %v", trial, got.ZPK.Zeros[0][0], got.ZPK.Gain[0][0], zeros, k)
		}
	}
}
