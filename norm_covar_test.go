package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestNorm_H2_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := Norm(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := 0.707106781186548
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("Norm(sys,2) = %g, want %g", got, want)
	}
}

func TestNorm_Inf_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := Norm(sys, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	want := 1.0
	if math.Abs(got-want) > 1e-6 {
		t.Errorf("Norm(sys,Inf) = %g, want %g", got, want)
	}
}

func TestNorm_H2_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	got, err := Norm(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := 1.154700538379252
	if math.Abs(got-want)/want > 1e-6 {
		t.Errorf("Norm(sys,2) = %g, want %g", got, want)
	}
}

func TestNorm_Inf_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	got, err := Norm(sys, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	want := 2.0
	if math.Abs(got-want) > 1e-6 {
		t.Errorf("Norm(sys,Inf) = %g, want %g", got, want)
	}
}

func TestNorm_InvalidType(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Norm(sys, 3)
	if err == nil {
		t.Fatal("expected error for invalid normType")
	}
}

func TestCovar_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	P, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	want := 0.5
	if math.Abs(P.At(0, 0)-want) > 1e-10 {
		t.Errorf("Covar = %g, want %g", P.At(0, 0), want)
	}
}

func TestCovar_SISO_Continuous_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}), 0)

	P, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(P.At(0, 0), 1) {
		t.Errorf("Covar = %g, want +Inf (white noise through D)", P.At(0, 0))
	}
}

func TestCovar_MIMO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 1, []float64{0, 0}), 0)

	P, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	r, c := P.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("P dims = %d×%d, want 2×2", r, c)
	}
	want := mat.NewDense(2, 2, []float64{1.0 / 12, 0, 0, 1.0 / 6})
	if !matEqual(P, want, 1e-10) {
		t.Errorf("Covar =\n%v\nwant\n%v", mat.Formatted(P), mat.Formatted(want))
	}
}

func TestCovar_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	P, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	want := 4.0 / 3.0
	if math.Abs(P.At(0, 0)-want) > 1e-10 {
		t.Errorf("Covar = %g, want %g", P.At(0, 0), want)
	}
}

func TestCovar_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestCovar_BadW(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Covar(sys, mat.NewDense(2, 2, nil))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestLsim_Step_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	steps := 501
	tVec := make([]float64, steps)
	uMat := mat.NewDense(steps, 1, nil)
	for i := range steps {
		tVec[i] = float64(i) * 0.01
		uMat.Set(i, 0, 1.0)
	}

	resp, err := Lsim(sys, uMat, tVec, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, cols := resp.Y.Dims()
	yEnd := resp.Y.At(0, cols-1)
	want := 1 - math.Exp(-5)
	if math.Abs(yEnd-want) > 1e-3 {
		t.Errorf("y(end) = %g, want %g", yEnd, want)
	}
}

func TestLsim_InitialCondition_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	steps := 201
	tVec := make([]float64, steps)
	uMat := mat.NewDense(steps, 1, nil)
	for i := range steps {
		tVec[i] = float64(i) * 0.01
	}
	x0 := mat.NewVecDense(1, []float64{5})

	resp, err := Lsim(sys, uMat, tVec, x0)
	if err != nil {
		t.Fatal(err)
	}

	_, cols := resp.Y.Dims()
	yEnd := resp.Y.At(0, cols-1)
	want := 5 * math.Exp(-2)
	if math.Abs(yEnd-want) > 1e-3 {
		t.Errorf("y(end) = %g, want %g", yEnd, want)
	}
}

func TestLsim_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	steps := 10
	tVec := make([]float64, steps)
	uMat := mat.NewDense(steps, 1, nil)
	for i := range steps {
		tVec[i] = float64(i) * 0.1
		uMat.Set(i, 0, 1.0)
	}

	resp, err := Lsim(sys, uMat, tVec, nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp.Y.At(0, 0) != 0 {
		t.Errorf("y(0) = %g, want 0 (no feedthrough)", resp.Y.At(0, 0))
	}
	yEnd := resp.Y.At(0, steps-1)
	if yEnd <= 0 {
		t.Errorf("y(end) = %g, want positive", yEnd)
	}
}

func TestLsim_BadDims(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Lsim(sys, mat.NewDense(5, 2, nil), []float64{0, 1, 2, 3, 4}, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestCovar_ContinuousFeedthroughEntries(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	B := mat.NewDense(2, 2, []float64{1, 0.3, 0, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0.4, 1, -0.2, 0.7})
	D := mat.NewDense(3, 2, []float64{0.5, 0, 0, 0, 0, -0.3})
	W := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	P, err := Covar(sys, W)
	if err != nil {
		t.Fatal(err)
	}
	X, err := Lyap(A, mat.NewDense(2, 2, []float64{1.18, 0.6, 0.6, 2}), nil)
	if err != nil {
		t.Fatal(err)
	}
	var CX, want mat.Dense
	CX.Mul(C, X)
	want.Mul(&CX, C.T())
	for i := range 3 {
		for j := range 3 {
			got := P.At(i, j)
			switch {
			case (i == 0 && j == 0) || (i == 2 && j == 2):
				if !math.IsInf(got, 1) {
					t.Errorf("P[%d,%d] = %g, want +Inf", i, j, got)
				}
			case math.Abs(got-want.At(i, j)) > 1e-12:
				t.Errorf("P[%d,%d] = %g, want %g", i, j, got, want.At(i, j))
			}
		}
	}
}

func TestCovar_StaticGain(t *testing.T) {
	D := mat.NewDense(2, 2, []float64{1, 0.5, 0, 0})
	W := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	for _, dt := range []float64{0, 0.1} {
		g, err := NewGain(D, dt)
		if err != nil {
			t.Fatal(err)
		}
		P, err := Covar(g, W)
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		want := mat.NewDense(2, 2, []float64{2.25, 0, 0, 0})
		if dt == 0 {
			want.Set(0, 0, math.Inf(1))
		}
		if !mat.Equal(P, want) {
			t.Errorf("dt=%v: Covar =\n%v\nwant\n%v", dt, mat.Formatted(P), mat.Formatted(want))
		}
	}
}

func TestCovar_DiscreteInternalDelay(t *testing.T) {
	sys := internalDelayFixture(t, 0.1, mat.NewDense(2, 1, []float64{0.2, 0}))
	W := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 0.5})
	want := mat.NewDense(2, 2, nil)
	for _, hk := range discreteImpulseLFT(sys, 4000) {
		var hw, hwh mat.Dense
		hw.Mul(hk, W)
		hwh.Mul(&hw, hk.T())
		want.Add(want, &hwh)
	}
	P, err := Covar(sys, W)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(P, want, 1e-9) {
		t.Errorf("Covar =\n%v\nwant (impulse sum)\n%v", mat.Formatted(P), mat.Formatted(want))
	}
}

func delayedCovarPlant(t *testing.T, dt float64, a []float64) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, a),
		mat.NewDense(2, 2, []float64{1, 0.4, -0.3, 1}),
		mat.NewDense(2, 2, []float64{1, 0.5, -0.2, 0.8}),
		mat.NewDense(2, 2, []float64{0.5, -0.1, 0.3, 0.2}),
		dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestCovar_DiscreteExternalDelays(t *testing.T) {
	W := mat.NewDense(2, 2, []float64{2, 0.7, 0.7, 1})
	cases := []struct {
		name          string
		input, output []float64
		io            *mat.Dense
	}{
		{name: "output", output: []float64{0, 3}},
		{name: "input", input: []float64{2, 0}},
		{name: "input+output", input: []float64{1, 0}, output: []float64{0, 3}},
		{name: "io", io: mat.NewDense(2, 2, []float64{0, 2, 1, 0})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := delayedCovarPlant(t, 0.1, []float64{0.5, 0.3, -0.2, 0.4})
			tau := mat.NewDense(2, 2, nil)
			for i := range 2 {
				for k := range 2 {
					v := 0.0
					if tc.input != nil {
						v += tc.input[k]
					}
					if tc.output != nil {
						v += tc.output[i]
					}
					if tc.io != nil {
						v += tc.io.At(i, k)
					}
					tau.Set(i, k, v)
				}
			}
			if tc.input != nil {
				if err := sys.SetInputDelay(tc.input); err != nil {
					t.Fatal(err)
				}
			}
			if tc.output != nil {
				if err := sys.SetOutputDelay(tc.output); err != nil {
					t.Fatal(err)
				}
			}
			if tc.io != nil {
				if err := sys.SetDelay(tc.io); err != nil {
					t.Fatal(err)
				}
			}

			const steps = 400
			markov := make([]*mat.Dense, steps)
			markov[0] = mat.DenseCopyOf(sys.D)
			Ak := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
			for k := 1; k < steps; k++ {
				var CA, h mat.Dense
				CA.Mul(sys.C, Ak)
				h.Mul(&CA, sys.B)
				markov[k] = &h
				var next mat.Dense
				next.Mul(Ak, sys.A)
				Ak = &next
			}
			want := mat.NewDense(2, 2, nil)
			for k := range steps + 10 {
				g := mat.NewDense(2, 2, nil)
				for i := range 2 {
					for l := range 2 {
						if s := k - int(tau.At(i, l)); s >= 0 && s < steps {
							g.Set(i, l, markov[s].At(i, l))
						}
					}
				}
				var gw, gwg mat.Dense
				gw.Mul(g, W)
				gwg.Mul(&gw, g.T())
				want.Add(want, &gwg)
			}

			P, err := Covar(sys, W)
			if err != nil {
				t.Fatal(err)
			}
			if !matEqual(P, want, 1e-9) {
				t.Errorf("Covar =\n%v\nwant (delayed impulse sum)\n%v", mat.Formatted(P), mat.Formatted(want))
			}
		})
	}
}

// covarModalOracle returns ∫ g_ik(σ) g_jl(σ) dσ for A = [-1 2; 0 -3], where
// g_ik(σ) = D_ik δ(σ-τ_ik) + c_iᵀ e^{A(σ-τ_ik)} b_k for σ > τ_ik, and
// e^{At} = [e^{-t}, e^{-t}-e^{-3t}; 0, e^{-3t}].
func covarModalOracle(B, C, D *mat.Dense, i, k, j, l int, tauIK, tauJL float64) float64 {
	coef := func(r, c int) (float64, float64) {
		c1, c2 := C.At(r, 0), C.At(r, 1)
		b1, b2 := B.At(0, c), B.At(1, c)
		return c1 * (b1 + b2), c2*b2 - c1*b2
	}
	h := func(r, c int, s float64) float64 {
		a1, a3 := coef(r, c)
		return a1*math.Exp(-s) + a3*math.Exp(-3*s)
	}
	ai1, ai3 := coef(i, k)
	aj1, aj3 := coef(j, l)
	T := math.Max(tauIK, tauJL)
	lam := []float64{1, 3}
	ai := []float64{ai1, ai3}
	aj := []float64{aj1, aj3}
	v := 0.0
	for p := range 2 {
		for q := range 2 {
			v += ai[p] * aj[q] * math.Exp(-lam[p]*(T-tauIK)-lam[q]*(T-tauJL)) / (lam[p] + lam[q])
		}
	}
	if math.Abs(tauIK-tauJL) < 1e-12 {
		return v
	}
	if tauIK > tauJL {
		v += D.At(i, k) * h(j, l, tauIK-tauJL)
	}
	if tauJL > tauIK {
		v += D.At(j, l) * h(i, k, tauJL-tauIK)
	}
	return v
}

func TestCovar_ContinuousExternalDelays(t *testing.T) {
	W := mat.NewDense(2, 2, []float64{2, 0.7, 0.7, 1})
	cases := []struct {
		name          string
		input, output []float64
		io            *mat.Dense
		d             []float64
	}{
		{name: "output", output: []float64{0.5, 0}, d: []float64{0, 0, 0, 0}},
		{name: "input", input: []float64{0.3, 0}, d: []float64{0, 0, 0, 0}},
		{name: "input+output+D", input: []float64{0.3, 0}, output: []float64{0.5, 0}, d: []float64{0.5, 0, 0, 0}},
		{name: "io+D", io: mat.NewDense(2, 2, []float64{0.2, 0.3, 0, 0.6}), d: []float64{0, 0.4, -0.3, 0}},
		{name: "rounded equal D paths", io: mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.3}),
			input: []float64{0.2, 0.2}, output: []float64{0.3, 0.1}, d: []float64{0.5, 0, 0, 0.4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := delayedCovarPlant(t, 0, []float64{-1, 2, 0, -3})
			sys.D = mat.NewDense(2, 2, tc.d)
			tau := mat.NewDense(2, 2, nil)
			for i := range 2 {
				for k := range 2 {
					v := 0.0
					if tc.input != nil {
						v += tc.input[k]
					}
					if tc.output != nil {
						v += tc.output[i]
					}
					if tc.io != nil {
						v += tc.io.At(i, k)
					}
					tau.Set(i, k, v)
				}
			}
			if tc.input != nil {
				if err := sys.SetInputDelay(tc.input); err != nil {
					t.Fatal(err)
				}
			}
			if tc.output != nil {
				if err := sys.SetOutputDelay(tc.output); err != nil {
					t.Fatal(err)
				}
			}
			if tc.io != nil {
				if err := sys.SetDelay(tc.io); err != nil {
					t.Fatal(err)
				}
			}

			P, err := Covar(sys, W)
			if err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				for j := range 2 {
					want, inf := 0.0, false
					for k := range 2 {
						for l := range 2 {
							if math.Abs(tau.At(i, k)-tau.At(j, l)) < 1e-12 && sys.D.At(i, k)*W.At(k, l)*sys.D.At(j, l) != 0 {
								inf = true
							}
							want += W.At(k, l) * covarModalOracle(sys.B, sys.C, sys.D, i, k, j, l, tau.At(i, k), tau.At(j, l))
						}
					}
					got := P.At(i, j)
					if inf {
						if !math.IsInf(got, 0) {
							t.Errorf("P[%d,%d] = %g, want Inf", i, j, got)
						}
						continue
					}
					if math.Abs(got-want) > 1e-9 {
						t.Errorf("P[%d,%d] = %.15g, want %.15g", i, j, got, want)
					}
				}
			}
		})
	}
}
