package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffSchurDelayCase is a model with one internal delay whose explicit A
// is upper quasi-triangular with one mode of modulus fast beside slow ones
// (complex: a slow complex pair and two inputs with D21 ≠ 0), as an
// explicit model or as a descriptor with diagonal E (E⁻¹A keeps the
// shape).
func stiffSchurDelayCase(t *testing.T, fast float64, complexPair, descriptor bool) *System {
	t.Helper()
	var a, b, c, b2, c2, d12, d21 *mat.Dense
	if complexPair {
		a = mat.NewDense(3, 3, []float64{-0.5, 1.2, 0.3, -0.8, -0.5, 0.7, 0, 0, -fast})
		b = mat.NewDense(3, 2, []float64{1, 0.2, 0.4, -1, fast, 2 * fast})
		c = mat.NewDense(1, 3, []float64{1, -0.6, 1})
		b2 = mat.NewDense(3, 1, []float64{0.3, -0.2, 0.1 * fast})
		c2 = mat.NewDense(1, 3, []float64{0.4, 0.2, -0.3})
		d12 = mat.NewDense(1, 1, []float64{0.5})
		d21 = mat.NewDense(1, 2, []float64{0.2, -0.3})
	} else {
		a = mat.NewDense(2, 2, []float64{-1, 0.5, 0, -fast})
		b = mat.NewDense(2, 1, []float64{1, fast})
		c = mat.NewDense(1, 2, []float64{1, 1})
		b2 = mat.NewDense(2, 1, []float64{0.3, 0.1})
		c2 = mat.NewDense(1, 2, []float64{0.4, 0.2})
		d12 = mat.NewDense(1, 1, []float64{0.5})
		d21 = mat.NewDense(1, 1, nil)
	}
	n, m := b.Dims()
	var sys *System
	var err error
	if descriptor {
		e := mat.NewDense(n, n, nil)
		for i := range n {
			e.Set(i, i, 2/float64(i+1))
		}
		var ea, eb, eb2 mat.Dense
		ea.Mul(e, a)
		eb.Mul(e, b)
		eb2.Mul(e, b2)
		sys, err = NewDescriptor(&ea, &eb, c, mat.NewDense(1, m, nil), e, 0)
		b2 = &eb2
	} else {
		sys, err = New(a, b, c, mat.NewDense(1, m, nil), 0)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.3}, b2, c2, d12, d21, mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

// delayChainRoundingBound bounds the rounding of the exact delay chain
// relative to max(1, |y|): each of the steps samples applies order+1 ≤ K+1
// propagator blocks of size na to the chain state, K = ⌊t_end/τ⌋, and
// forms the output from n states, each product rounding to ε relative.
func delayChainRoundingBound(steps, na, n int, tEnd, tau float64) float64 {
	return 0x1p-52 * float64(steps) * (math.Floor(tEnd/tau) + 1) * float64(na*n)
}

// Step, Lsim (with x0), Impulse and Initial of internal-delay models whose
// explicit A is in real Schur form with a stiff mode follow the 320-bit
// method-of-steps response within the chain's rounding bound. One
// exponential of the chain generator rounded the slow modes to ε·‖A‖·dt per
// sample: 7e-6 off at fast = 1e12 and 6e-4 at 1e15 (WI4F5E).
func TestStepInternalDelayStiffSchurFollowsMethodOfSteps(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		name                    string
		fast                    float64
		complexPair, descriptor bool
	}{
		{"triangular-1e12", 1e12, false, false},
		{"triangular-1e15", 1e15, false, false},
		{"diagonal-E-1e15", 1e15, false, true},
		{"complex-pair-1e13", 1e13, true, false},
	}
	const tFinal = 0.5
	for _, tc := range cases {
		sys := stiffSchurDelayCase(t, tc.fast, tc.complexPair, tc.descriptor)
		oSys := *sys
		n, m, _ := sys.Dims()
		if oSys.E == nil {
			oSys.E = mat.NewDense(n, n, nil)
			for i := range n {
				oSys.E.Set(i, i, 1)
			}
		}
		o := newLFTStepOracle(t, &oSys)
		st, err := Step(sys, tFinal)
		if err != nil {
			t.Fatalf("%s: Step: %v", tc.name, err)
		}
		steps := len(st.T)
		u := mat.NewDense(steps, m, nil)
		for k := range steps {
			for j := range m {
				u.Set(k, j, 1)
			}
		}
		x0 := make([]float64, n)
		for i := range n {
			x0[i] = 0.7 - 0.45*float64(i)
		}
		ls, err := Lsim(sys, u, st.T, mat.NewVecDense(n, x0))
		if err != nil {
			t.Fatalf("%s: Lsim: %v", tc.name, err)
		}
		in, err := Initial(sys, mat.NewVecDense(n, x0), tFinal)
		if err != nil {
			t.Fatalf("%s: Initial: %v", tc.name, err)
		}
		var im *TimeResponse
		if allZeroDense(sys.LFT.D21) {
			if im, err = Impulse(sys, tFinal); err != nil {
				t.Fatalf("%s: Impulse: %v", tc.name, err)
			}
		}
		check := func(api string, got, want float64, na int, T []float64) {
			tol := delayChainRoundingBound(len(T), na, n, T[len(T)-1], o.tau)
			if d := stepOracleRel(got, want); d > tol {
				t.Errorf("%s: %s off the method of steps by %.3g > %.3g", tc.name, api, d, tol)
			}
		}
		for _, k := range []int{steps / 2, steps - 1} {
			tk := st.T[k]
			if nearDelayGrid(tk, o.tau) {
				continue
			}
			free := o.response(-1, x0, tk)[0]
			forced := free
			for j := range m {
				w := o.response(j, nil, tk)[0]
				forced += w
				check("Step", st.Y.At(j, k), w, n+1, st.T)
			}
			check("Lsim", ls.Y.At(0, k), forced, n+m, st.T)
			if k < len(in.T) && in.T[k] == tk {
				check("Initial", in.Y.At(0, k), free, n, in.T)
			}
			if im != nil && k < len(im.T) && im.T[k] == tk {
				for j := range m {
					bj := make([]float64, n)
					for s := range n {
						bj[s] = sys.B.At(s, j) / oSys.E.At(s, s)
					}
					check("Impulse", im.Y.At(j, k), o.response(-1, bj, tk)[0], n, im.T)
				}
			}
		}
	}
}
