package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strconv"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// mildConditionedEDescriptor is U·(sW − F)·V with F upper triangular, W
// diagonal with one entry 1e-5 and U, V random orthogonal: LU cond(E) ≈ 3e5,
// in the window where stiffDescriptorParts applies but the LU fold is
// accurate.
func mildConditionedEDescriptor(t *testing.T) *System {
	t.Helper()
	const n = 6
	rng := rand.New(rand.NewPCG(7, 383801))
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		F.Set(i, i, rng.NormFloat64()-0.3)
		W.Set(i, i, 0.5+rng.Float64())
		for j := i + 1; j < n; j++ {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	W.Set(n-1, n-1, 1e-5)
	b, c := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		b.Set(i, 0, rng.NormFloat64())
		c.Set(0, i, rng.NormFloat64())
	}
	D := mat.NewDense(1, 1, []float64{rng.NormFloat64()})
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	sys, err := NewDescriptor(&A, &B, &C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.StateName = make([]string, n)
	for i := range n {
		sys.StateName[i] = "x" + strconv.Itoa(i)
	}
	return sys
}

func foldGateOracle(t *testing.T, sys *System) stepOracleModes {
	t.Helper()
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, poles)
	if !ok {
		t.Fatal("pencil oracle did not converge")
	}
	return modes
}

func foldGateC2DError(t *testing.T, d *System, modes stepOracleModes, method C2DMethod, dt float64) float64 {
	t.Helper()
	if method == C2DMethodImpulse {
		d = d.Copy()
		d.D.Zero()
	}
	worst := 0.0
	for _, th := range []float64{0.05, 0.5, 2, 3} {
		z := cmplx.Exp(complex(0, th))
		var want complex128
		switch method {
		case C2DMethodFOH:
			want = modes.foh(0, 0, dt, z)
		case C2DMethodTustin:
			want = modes.tustin(0, 0, dt, z)
		case C2DMethodImpulse:
			want = modes.impulseTail(0, 0, dt, z)
		default:
			want = modes.zoh(0, 0, dt, z)
		}
		got := pencilResponse(t, d, z)[0][0]
		worst = math.Max(worst, cmplx.Abs(got-want)/math.Max(1, cmplx.Abs(want)))
	}
	return worst
}

// A full-rank E with LU cond(E) in [1e4, 1e6) (WOIVWF narrowing, wave 38)
// takes the LU fold in C2D and Step: its error is cond(E)·ε-sized, so the
// explicit result already matches the 320-bit pencil oracle and keeps the
// state coordinates and names, at the cost of the explicit C2D.
func TestC2DMildConditionedEKeepsLUFold(t *testing.T) {
	sys := mildConditionedEDescriptor(t)
	var lu mat.LU
	lu.Factorize(sys.E)
	if c := lu.Cond(); c < illConditionedECond || c >= foldAccurateECond {
		t.Fatalf("LU cond(E) = %.3g, want in [%g, %g)", c, illConditionedECond, foldAccurateECond)
	}
	if _, ok := stiffDescriptorParts(sys); !ok {
		t.Fatal("stiffDescriptorParts does not apply; the case no longer tests the gate")
	}
	modes := foldGateOracle(t, sys)
	const dt = 0.1
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse} {
		d, err := sys.C2D(dt, C2DOptions{Method: method})
		if err != nil {
			t.Fatalf("method %v: %v", method, err)
		}
		if len(d.StateName) != len(sys.StateName) || d.StateName[0] != "x0" {
			t.Errorf("method %v: StateName %v, want the LU fold's state names", method, d.StateName)
		}
		if e := foldGateC2DError(t, d, modes, method, dt); e > 1e-9 {
			t.Errorf("method %v: response error %.3g vs pencil oracle, want <= 1e-9", method, e)
		}
	}
	st, err := Step(sys, 5)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for k := 1; k < len(st.T); k++ {
		want := modes.step(0, 0, st.T[k])
		worst = math.Max(worst, math.Abs(st.Y.At(0, k)-want)/math.Max(1, math.Abs(want)))
	}
	if worst > 1e-7 {
		t.Errorf("Step error %.3g vs pencil oracle, want <= 1e-7", worst)
	}
}

// At LU cond(E) ≈ 1.4e8 (equilibrated 1e6; revCase band 653 of the WOIVWF
// sweep) the LU fold misplaces the ZOH response by 2.5e-8, so C2D still
// discretizes the pencil parts, which match the oracle and drop the names.
func TestC2DFoldInaccurateEFollowsPencil(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{-1.6500955130433037, 1.549317110825327, 1.0249012147140975, -0.768864521490068}),
		mat.NewDense(2, 1, []float64{-1.283229147671321, -0.1766429729161167}),
		mat.NewDense(1, 2, []float64{-0.7798368392258331, -0.57684030559386}),
		mat.NewDense(1, 1, []float64{1.699158936847553}),
		mat.NewDense(2, 2, []float64{0.0020762888275396942, 0.002212237899607853, 0.6340618055731662, 0.6755812636359746}),
		0)
	if err != nil {
		t.Fatal(err)
	}
	sys.StateName = []string{"x0", "x1"}
	var lu mat.LU
	lu.Factorize(sys.E)
	if c := lu.Cond(); c < foldAccurateECond {
		t.Fatalf("LU cond(E) = %.3g, want >= %g", c, foldAccurateECond)
	}
	modes := foldGateOracle(t, sys)
	const dt = 0.1
	explicit, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	fold, err := explicit.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e := foldGateC2DError(t, fold, modes, C2DMethodZOH, dt); e <= 1e-9 {
		t.Fatalf("LU fold ZOH error %.3g; the case no longer needs the pencil", e)
	}
	d, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e := foldGateC2DError(t, d, modes, C2DMethodZOH, dt); e > 1e-9 {
		t.Errorf("ZOH error %.3g vs pencil oracle, want <= 1e-9", e)
	}
	if d.StateName != nil {
		t.Errorf("StateName %v, want none for pencil parts", d.StateName)
	}
}
