package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorDCBars is the worst entry of |ĥ − g0|/(bar₀ + allowance) for
// the discrete result out of the continuous descriptor sys: ĥ and g0 exact
// for the stored entries (fraction-free solves), bar₀ the descriptor bar
// and the allowance ε·S of out where it is within √ε·max(|g0|, T0) (the
// C2D DC contract). ok is false where an exact gain is singular.
func descriptorDCBars(t *testing.T, sys, out *System) (float64, bool) {
	t.Helper()
	g0, _, ok0 := exactDCGainAtNonsingular(nil, sys.A, sys.B, sys.C, sys.D, 0)
	h, _, ok := exactDCGainAtNonsingular(nil, out.A, out.B, out.C, out.D, 1)
	if !ok0 || !ok {
		return 0, false
	}
	b := newDCBar(ioDelayFree(sys))
	allow := holdAllowance(out)
	p, m := g0.Dims()
	worst := 0.0
	for k := range p * m {
		i, j := k/m, k%m
		a := 0.0
		if allow != nil && allow[k] <= math.Sqrt(0x1p-52)*math.Max(math.Abs(g0.At(i, j)), b.terms[k]) {
			a = allow[k]
		}
		bar := b.fullBar(k)
		if b.lu == nil {
			bar = b.bar[k]
		}
		worst = math.Max(worst, math.Abs(h.At(i, j)-g0.At(i, j))/(bar+a))
	}
	return worst, true
}

// checkDescriptorDCContract asserts the C2D DC contract of a ZOH, FOH or
// Tustin result: nil within the bar, unverified with a model, any other
// error without one. It returns "nil", "unv" or "err".
func checkDescriptorDCContract(t *testing.T, name string, sys, out *System, err error) string {
	t.Helper()
	switch {
	case err == nil:
		if bars, ok := descriptorDCBars(t, sys, out); ok && !(bars <= 1) {
			t.Errorf("%s: nil with DC %.3g bars off DCGain", name, bars)
		}
		return "nil"
	case errors.Is(err, ErrAccuracyUnverified):
		if out == nil {
			t.Errorf("%s: unverified without a model", name)
		}
		return "unv"
	default:
		if out != nil {
			t.Errorf("%s: model with %v", name, err)
		}
		return "err"
	}
}

// Every ZOH, FOH and Tustin C2D of the frozen EJDS3U census rows meets
// the DC contract, and the rows the PR-1 routes fix keep their verdicts:
//   - stfr/45, /91 (rank-deficient E, triangular standard form with the
//     stiff mode leading): nil within the bar once the hold keeps that
//     order (R1); base was 5e10 and 4e6 bars;
//   - cond/13 (LU cond 4.6e5): nil through the pencil parts (R4); the LU
//     fold was 65 bars at dt = 0.01;
//   - cond/644 (LU cond 2.4e3, ‖A·dt‖ large): ZOH at dt = 1 nil in the real
//     Schur coordinates of E⁻¹A, without state names (R2); the fold was
//     2e12 bars;
//   - cond/111 at dt = 0.01: 13.9 bars in fold and 41 in Schur
//     coordinates, unverified (R5);
//   - strad1m/47, /283 (band, split refused): the fold, always unverified
//     (U-2).
func TestC2DDescriptorDCContractFrozenRows(t *testing.T) {
	want := map[string]string{
		"stfr/45@0.01/zoh": "nil", "stfr/45@0.01/foh": "nil", "stfr/45@1/zoh": "nil",
		"stfr/91@0.01/zoh": "nil", "stfr/91@1/zoh": "nil",
		"cond/13@0.01/zoh": "nil", "cond/13@0.01/foh": "nil", "cond/13@1/zoh": "nil",
		"cond/644@1/zoh": "nil", "cond/644@1/tustin": "nil",
		"cond/111@0.01/zoh": "unv", "cond/111@0.01/foh": "unv",
		"strad1m/47@0.0001/zoh": "unv", "strad1m/47@0.01/foh": "unv", "strad1m/47@0.01/tustin": "unv",
		"strad1m/283@1/zoh": "unv",
	}
	for _, r := range descriptorDCGateRows {
		sys, err := r.system()
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				key := fmt.Sprintf("%s@%g/%s", r.name, dt, m)
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				got := checkDescriptorDCContract(t, key, sys, out, err)
				if w, ok := want[key]; ok && got != w {
					t.Errorf("%s: %s (%v), want %s", key, got, err, w)
				}
				if key == "cond/644@1/zoh" && out != nil && out.StateName != nil {
					t.Errorf("%s: state names %v, want none (real Schur coordinates)", key, out.StateName)
				}
			}
		}
	}
}

// C2DMap keeps the LU fold of a full-rank E (T3IVBX is PR-2) and returns
// it with its map and ErrAccuracyUnverified where the DC of the result is
// outside the bar against the model passed: cond/13 at dt = 0.01 is 65
// bars on the fold. Within the bar it stays nil.
func TestC2DMapDescriptorDCUnverified(t *testing.T) {
	for _, tc := range []struct {
		row  string
		dt   float64
		want string
	}{{"cond/13", 0.01, "unv"}, {"cond/644", 0.01, "nil"}} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == tc.row {
				var err error
				if sys, err = r.system(); err != nil {
					t.Fatal(err)
				}
			}
		}
		out, g, err := sys.C2DMap(tc.dt, C2DOptions{})
		got := checkDescriptorDCContract(t, tc.row, sys, out, err)
		if got != tc.want {
			t.Errorf("%s: %s (%v), want %s", tc.row, got, err, tc.want)
		}
		if got == "unv" && g == nil {
			t.Errorf("%s: unverified without G", tc.row)
		}
	}
}

// mimoDescriptor is a 2×2 dense descriptor with E = U·diag(1, …, 10^-x)·Wᵀ.
func mimoDescriptor(t *testing.T, x float64) *System {
	t.Helper()
	A := mat.NewDense(4, 4, []float64{
		-0.9, 0.3, -0.2, 0.4,
		0.5, -1.1, 0.25, -0.3,
		-0.1, 0.6, -0.7, 0.2,
		0.3, -0.2, 0.45, -1.3})
	c, s := math.Cos(0.7), math.Sin(0.7)
	U := mat.NewDense(4, 4, []float64{c, -s, 0, 0, s, c, 0, 0, 0, 0, c, s, 0, 0, -s, c})
	W := mat.NewDense(4, 4, []float64{c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1})
	var T, E mat.Dense
	T.Mul(U, mat.NewDiagDense(4, []float64{1, 0.6, 0.35, math.Pow(10, -x)}))
	E.Mul(&T, W.T())
	B := mat.NewDense(4, 2, []float64{1, -0.4, 0.3, 0.8, -0.7, 0.2, 0.5, 1.1})
	C := mat.NewDense(2, 4, []float64{0.6, -0.9, 0.4, 0.2, -0.3, 0.5, 1.2, -0.8})
	D := mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3})
	sys, err := NewDescriptor(A, B, C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// The DC check of C2D is entrywise for MIMO descriptors: every entry of a
// nil result lies within its own bar (row C_i, column B_j, |D_ij|), from
// a well-conditioned E through the band to cond 1e12.
func TestC2DDescriptorDCContractMIMO(t *testing.T) {
	nils := 0
	for _, x := range []float64{1, 4.5, 5.5, 8, 12} {
		sys := mimoDescriptor(t, x)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				if checkDescriptorDCContract(t, fmt.Sprintf("x=%g dt=%g %s", x, dt, m), sys, out, err) == "nil" {
					nils++
				}
			}
		}
	}
	if nils == 0 {
		t.Error("no nil result; the check is not exercised")
	}
}

// integratorDescriptor has an exact pole at s = 0 (first column of A
// zero) and an ill-conditioned triangular E; with decoupled it has a
// second output reached only by a state no path joins to the integrator.
func integratorDescriptor(t *testing.T, decoupled bool) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{0, 0.4, -0.3, 0, -1.2, 0.5, 0, 0.2, -0.8})
	E := mat.NewDense(3, 3, []float64{1, 0.3, -0.2, 0, 1, 0.4, 0, 0, 1e-7})
	B := mat.NewDense(3, 1, []float64{0.7, -1.1, 0.9})
	C := mat.NewDense(1, 3, []float64{1, 0.5, -0.4})
	D := mat.NewDense(1, 1, []float64{0.2})
	if decoupled {
		A.Set(2, 1, 0)
		A.Set(0, 2, 0)
		A.Set(1, 2, 0)
		E.Set(0, 2, 0)
		E.Set(1, 2, 0)
		C = mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 0, 0.6})
		D = mat.NewDense(2, 1, []float64{0.2, -0.1})
	}
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// An entry of DCGain that is infinite (a pole at DC) is not checked: a
// SISO descriptor with an integrator keeps its nil ZOH, FOH and Tustin.
// Where another entry is finite no bar exists for it, and the result is
// unverified.
func TestC2DDescriptorDCContractIntegrator(t *testing.T) {
	siso := integratorDescriptor(t, false)
	g, err := siso.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) {
		t.Fatalf("setup: DCGain %v, %v; want ±Inf", g, err)
	}
	mixed := integratorDescriptor(t, true)
	g, err = mixed.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) || math.IsInf(g.At(1, 0), 0) {
		t.Fatalf("setup: DCGain %v, %v; want ±Inf and a finite entry", g, err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		if out, err := siso.C2D(0.1, C2DOptions{Method: m}); err != nil || out == nil {
			t.Errorf("SISO %s: %v, want nil", m, err)
		}
		if out, err := mixed.C2D(0.1, C2DOptions{Method: m}); !errors.Is(err, ErrAccuracyUnverified) || out == nil {
			t.Errorf("mixed %s: %v, want a model with ErrAccuracyUnverified", m, err)
		}
	}
}

// The DC check reserves the rounding bounds of both gains: for a result
// whose DC exceeds g0 by δ, nil requires δ + err(ĥ) + err(g0) ≤ bar +
// allowance, so the largest δ it accepts leaves that room. Caller ẋ = −x + u
// (E = I as a descriptor, g0 = 1, T0 = 1); result Φ = 1/2, Γ = (1+δ)/2,
// whose DC 1 + δ is exact.
func TestC2DDescriptorDCGateErrorBounds(t *testing.T) {
	one := func(v float64) *mat.Dense { return mat.NewDense(1, 1, []float64{v}) }
	sys, err := NewDescriptor(one(-1), one(1), one(1), one(0), one(1), 0)
	if err != nil {
		t.Fatal(err)
	}
	gate := newHoldGate(sys)
	bar := gate.bar.fullBar(0)
	accepted := -1.0
	for k := range 200 {
		delta := float64(k) * 0x1p-52
		out := &System{A: one(0.5), B: one((1 + delta) / 2), C: one(1), D: one(0), Dt: 0.1}
		if gate.check(out) != nil {
			break
		}
		accepted = delta
	}
	if accepted < 0 {
		t.Fatal("the exact gain is rejected")
	}
	edge := &System{A: one(0.5), B: one((1 + accepted) / 2), C: one(1), D: one(0), Dt: 0.1}
	_, herr, _ := holdDC(edge)
	limit := bar + holdAllowance(edge)[0] - herr[0] - 0x1p-52*1 - halfUlp(1)
	if !(accepted <= limit+0x1p-53) || !(accepted+0x1p-52 > limit-0x1p-53) {
		t.Errorf("largest δ accepted %.4g, want the bar less both rounding bounds, %.4g", accepted, limit)
	}
}
