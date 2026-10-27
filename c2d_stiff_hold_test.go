package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffHoldOracleChecks compares C2D (ZOH, FOH, impulse) at dt and Step(5)
// of sys with the exact methods applied to the 320-bit modes.
func stiffHoldOracleChecks(t *testing.T, name string, sys *System, modes stepOracleModes, dt, c2dTol, stepTol float64) {
	t.Helper()
	_, m, p := sys.Dims()
	for _, mt := range []struct {
		method C2DMethod
		want   func(i, j int, z complex128) complex128
	}{
		{C2DMethodZOH, func(i, j int, z complex128) complex128 { return modes.zoh(i, j, dt, z) }},
		{C2DMethodFOH, func(i, j int, z complex128) complex128 { return modes.foh(i, j, dt, z) }},
		{C2DMethodImpulse, func(i, j int, z complex128) complex128 { return modes.impulseTail(i, j, dt, z) }},
	} {
		d, err := sys.C2D(dt, C2DOptions{Method: mt.method})
		if err != nil {
			t.Errorf("%s %s: C2D: %v", name, mt.method, err)
			continue
		}
		if mt.method == C2DMethodImpulse {
			d = d.Copy()
			d.D.Zero()
		}
		worst := 0.0
		for _, th := range []float64{0.05, 0.5, 2, 3} {
			z := cmplx.Exp(complex(0, th))
			g := pencilResponse(t, d, z)
			for i := range p {
				for j := range m {
					want := mt.want(i, j, z)
					worst = math.Max(worst, cmplx.Abs(g[i][j]-want)/math.Max(1, cmplx.Abs(want)))
				}
			}
		}
		if !(worst <= c2dTol) {
			t.Errorf("%s %s: C2D(%g) off the exact hold by %.3g", name, mt.method, dt, worst)
		}
	}
	st, err := Step(sys, 5)
	if err != nil {
		t.Errorf("%s: Step: %v", name, err)
		return
	}
	worst := 0.0
	_, cols := st.Y.Dims()
	for s := 1; s < cols; s++ {
		for j := range m {
			for i := range p {
				worst = math.Max(worst, stepOracleRel(st.Y.At(j*p+i, s), modes.step(i, j, st.T[s])))
			}
		}
	}
	if !(worst <= stepTol) {
		t.Errorf("%s: Step off the exact response by %.3g", name, worst)
	}
}

// A triangular explicit A = [-0.5 1; 0 f] with a stiff f, and its
// diagonal-E twin (A = [-0.5 1; 0 -1], E = diag(1, -1/f), which
// equilibrates to the same explicit model): one exponential of [A B; 0 0]·T
// rounds the slow mode to ε·|f|·T (f = -1e14: C2D 3e-3, Step 8e-4 off).
// ZOH, FOH and impulse C2D and Step decouple the modes and match the exact
// holds; ZOH keeps the state coordinates, Ad and Bd in closed form.
func TestC2DStiffTriangularExplicitMatchesExactHold(t *testing.T) {
	const dt = 0.1
	for _, f := range []float64{-1e9, -1e12, -6e12, -1e14} {
		C := mat.NewDense(1, 2, []float64{1, 1})
		D := mat.NewDense(1, 1, nil)
		sys, err := New(mat.NewDense(2, 2, []float64{-0.5, 1, 0, f}), mat.NewDense(2, 1, []float64{1, -f}), C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		twin, err := NewDescriptor(mat.NewDense(2, 2, []float64{-0.5, 1, 0, -1}), mat.NewDense(2, 1, []float64{1, 1}), C, D,
			mat.NewDense(2, 2, []float64{1, 0, 0, -1 / f}), 0)
		if err != nil {
			t.Fatal(err)
		}
		for name, s := range map[string]*System{"explicit": sys, "diagonal E": twin} {
			label := fmt.Sprintf("%s f %g", name, f)
			pol, err := s.Poles()
			if err != nil {
				t.Fatal(err)
			}
			modes, ok := newStepOracleModes(s, pol)
			if !ok {
				t.Fatalf("%s: oracle did not resolve the poles %v", label, pol)
			}
			stiffHoldOracleChecks(t, label, s, modes, dt, 1e-12, 1e-9)
		}

		d, err := sys.C2D(dt, C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		e1, ef := math.Exp(-0.5*dt), math.Exp(f*dt)
		slow := -math.Expm1(-0.5*dt) / 0.5
		wantA := []float64{e1, (e1 - ef) / (-0.5 - f), 0, ef}
		wantB := []float64{slow + f/(0.5+f)*(slow-(-math.Expm1(f*dt))/(-f)), -math.Expm1(f * dt)}
		for k, w := range wantA {
			if got := d.A.At(k/2, k%2); math.Abs(got-w) > 1e-15*math.Max(1, math.Abs(w)) {
				t.Errorf("f %g: Ad[%d][%d] = %.17g, want %.17g", f, k/2, k%2, got, w)
			}
		}
		for k, w := range wantB {
			if got := d.B.At(k, 0); math.Abs(got-w) > 1e-14*math.Max(1, math.Abs(w)) {
				t.Errorf("f %g: Bd[%d] = %.17g, want %.17g", f, k, got, w)
			}
		}
	}
}

// twoStiffDescriptor is the reviewer's "vis" draw (seed i, PCG stream
// 383801): A = U·F·V, E = U·W·V with F upper triangular, W diagonal in
// [0.5, 1.5) except two trailing singular values in [4e-16, 1e-14] and
// [1e-4, 1e-3]. Both stiff modes land in one fast part of stiffPencilSplit.
func twoStiffDescriptor(t *testing.T, i uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(i, 383801))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	logu := func(lo, hi float64) float64 {
		return math.Exp(math.Log(lo) + rng.Float64()*(math.Log(hi)-math.Log(lo)))
	}
	n = max(n, 3)
	small := []float64{logu(4e-16, 1e-14), logu(1e-4, 1e-3)}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for r := range n {
		F.Set(r, r, rng.NormFloat64()-0.3)
		W.Set(r, r, 0.5+rng.Float64())
		for c := r + 1; c < n; c++ {
			F.Set(r, c, rng.NormFloat64())
		}
	}
	for q, s := range small {
		W.Set(n-1-q, n-1-q, s)
	}
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for r := range n {
		for j := range m {
			b.Set(r, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, r, rng.NormFloat64())
		}
	}
	for r := range p {
		for j := range m {
			D.Set(r, j, rng.NormFloat64())
		}
	}
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
	return sys
}

// A full-rank E with two stiff singular values (1e-15 and 1e-4 beside
// O(1)) has two stiff poles (vis 8: -7.3e14 and -8192) in one fast part.
// Folded whole, that part kept -8192 only to ε·‖E⁻¹A‖ (-8192.375) and its
// one exponential lost the slow mode beside 1e14: C2D 4e-3 and Step 1e-6
// off. The fast pencil is split again, so ZOH, FOH and impulse C2D and
// Step match the 320-bit partial fractions of the pencil.
func TestC2DTwoStiffSingularValuesFollowPencil(t *testing.T) {
	resolved := 0
	for _, seed := range []uint64{8, 12, 26} {
		sys := twoStiffDescriptor(t, seed)
		name := fmt.Sprintf("seed %d", seed)
		if !matlabRankFull(t, sys.E) {
			t.Fatalf("%s: E reads rank deficient", name)
		}
		pol, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", name, err)
		}
		modes, ok := newStepOracleModes(sys, pol)
		if !ok {
			continue
		}
		resolved++
		stiffHoldOracleChecks(t, name, sys, modes, 0.1, 1e-9, 1e-8)
	}
	if resolved < 2 {
		t.Fatalf("only %d models resolved", resolved)
	}
}
