package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPadeDelayOrder1(t *testing.T) {
	tau := 0.5
	sys, err := PadeDelay(tau, 1)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 1 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (1,1,1)", n, m, p)
	}
	if !sys.IsContinuous() {
		t.Error("should be continuous")
	}

	// H(s) = (1 - tau*s/2) / (1 + tau*s/2)
	// At s=0: H(0)=1
	tfRes, _ := sys.TransferFunction(nil)
	h0 := mustEval(t, tfRes.TF, 0)
	if math.Abs(real(h0[0][0])-1) > 1e-12 {
		t.Errorf("H(0) = %v, want 1", h0[0][0])
	}
}

func TestPadeDelayOrder2FreqResponse(t *testing.T) {
	tau := 1.0
	sys, err := PadeDelay(tau, 2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 2 {
		t.Errorf("state dim = %d, want 2", n)
	}

	tfRes, _ := sys.TransferFunction(nil)

	// Pade should approximate exp(-s*tau) well for low frequencies
	freqs := []float64{0.1, 0.5, 1.0}
	for _, w := range freqs {
		s := complex(0, w)
		hPade := mustEval(t, tfRes.TF, s)[0][0]
		hExact := cmplx.Exp(-s * complex(tau, 0))

		// Magnitude should be ~1 (allpass property)
		magPade := cmplx.Abs(hPade)
		if math.Abs(magPade-1) > 1e-10 {
			t.Errorf("w=%v: |H_pade| = %v, want 1 (allpass)", w, magPade)
		}

		// Phase should match exp(-j*w*tau) at low frequencies
		phasePade := cmplx.Phase(hPade)
		phaseExact := cmplx.Phase(hExact)
		if math.Abs(phasePade-phaseExact) > 0.05 {
			t.Errorf("w=%v: phase_pade=%v, phase_exact=%v", w, phasePade, phaseExact)
		}
	}
}

func TestPadeDelayHighOrder(t *testing.T) {
	tau := 2.0
	for order := 1; order <= 10; order++ {
		sys, err := PadeDelay(tau, order)
		if err != nil {
			t.Fatalf("order %d: %v", order, err)
		}
		n, _, _ := sys.Dims()
		if n != order {
			t.Errorf("order %d: state dim = %d", order, n)
		}

		tfRes, _ := sys.TransferFunction(nil)
		h0 := mustEval(t, tfRes.TF, 0)[0][0]
		if math.Abs(real(h0)-1) > 1e-8 || math.Abs(imag(h0)) > 1e-8 {
			t.Errorf("order %d: H(0) = %v, want 1", order, h0)
		}
	}
}

func TestPadeDelayAllpass(t *testing.T) {
	tau := 0.3
	sys, _ := PadeDelay(tau, 5)
	tfRes, _ := sys.TransferFunction(nil)

	for _, w := range []float64{0.01, 0.1, 1, 5, 10, 50} {
		s := complex(0, w)
		h := mustEval(t, tfRes.TF, s)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1", w, mag)
		}
	}
}

func TestPadeDelayConvergence(t *testing.T) {
	tau := 1.0
	w := 2.0
	s := complex(0, w)
	exact := cmplx.Exp(-s * complex(tau, 0))

	prevErr := math.Inf(1)
	for order := 1; order <= 6; order++ {
		sys, _ := PadeDelay(tau, order)
		tfRes, _ := sys.TransferFunction(nil)
		h := mustEval(t, tfRes.TF, s)[0][0]
		curErr := cmplx.Abs(h - exact)
		if curErr >= prevErr {
			t.Errorf("order %d error %v >= order %d error %v (should converge)", order, curErr, order-1, prevErr)
		}
		prevErr = curErr
	}
}

func TestPadeDelayZeroTau(t *testing.T) {
	sys, err := PadeDelay(0, 3)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 0 {
		t.Errorf("zero delay should be pure gain, got n=%d", n)
	}
}

func TestPadeDelayNegativeTau(t *testing.T) {
	_, err := PadeDelay(-1, 2)
	if !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("expected ErrNegativeDelay, got %v", err)
	}
}

func TestPadeDelayInvalidOrder(t *testing.T) {
	_, err := PadeDelay(1.0, 0)
	if err == nil {
		t.Error("order 0 should error")
	}
	_, err = PadeDelay(1.0, 11)
	if err == nil {
		t.Error("order 11 should error")
	}
}

func TestPadeDelayStability(t *testing.T) {
	for _, tau := range []float64{0.1, 1.0, 5.0} {
		for order := 1; order <= 10; order++ {
			sys, _ := PadeDelay(tau, order)
			stable, err := sys.IsStable()
			if err != nil {
				t.Fatalf("tau=%v order=%d: %v", tau, order, err)
			}
			if !stable {
				t.Errorf("tau=%v order=%d: Pade should be stable", tau, order)
			}
		}
	}
}

// Julia: freqresp(pade(1, 1), Ω) == freqresp(tf([-1/2, 1], [1/2, 1]), Ω)
func TestPadeDelayOrder1ExactCoeffs(t *testing.T) {
	sys, _ := PadeDelay(1.0, 1)
	tfRes, _ := sys.TransferFunction(nil)

	// Exact TF: (-s/2 + 1)/(s/2 + 1)
	for _, w := range []float64{0, 0.5, 1, 2, 5} {
		s := complex(0, w)
		got := mustEval(t, tfRes.TF, s)[0][0]
		want := (-s/2 + 1) / (s/2 + 1)
		if cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("w=%v: got %v, want %v", w, got, want)
		}
	}
}

// Julia: freqresp(pade(1, 2), Ω) ≈ freqresp(tf([1/12, -1/2, 1], [1/12, 1/2, 1]), Ω)
func TestPadeDelayOrder2ExactCoeffs(t *testing.T) {
	sys, _ := PadeDelay(1.0, 2)
	tfRes, _ := sys.TransferFunction(nil)

	// Exact coefficients in descending power: [s^2/12, -s/2, 1] / [s^2/12, s/2, 1]
	numCoeffs := []float64{1.0 / 12, -0.5, 1.0} // s^2, s, 1
	denCoeffs := []float64{1.0 / 12, 0.5, 1.0}

	for _, freq := range []float64{0, 0.5, 1, 2, 5} {
		s := complex(0, freq)
		got := mustEval(t, tfRes.TF, s)[0][0]

		// Horner: evaluate descending-order polynomials
		sn := complex(0, 0)
		sd := complex(0, 0)
		for k := range numCoeffs {
			sn = sn*s + complex(numCoeffs[k], 0)
			sd = sd*s + complex(denCoeffs[k], 0)
		}
		want := sn / sd
		if cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("freq=%v: got %v, want %v", freq, got, want)
		}
	}
}

// Julia: for (n, tol) in enumerate([0.05; 1e-3; 1e-5; ...])
//
//	evalfr(pade(0.8, n), 1im) ≈ exp(-0.8im) atol=tol
func TestPadeDelayAccuracyByOrder(t *testing.T) {
	tau := 0.8
	s := complex(0, 1)
	exact := cmplx.Exp(complex(-tau, 0) * s)

	// Tolerances tighten with order up to ~6, then SS→TF roundtrip numerical
	// errors dominate. Julia avoids this by evaluating the LFT directly.
	tols := []float64{0.05, 1e-3, 1e-5, 1e-7, 1e-9, 1e-10}
	for n := 1; n <= 6; n++ {
		sys, _ := PadeDelay(tau, n)

		stable, _ := sys.IsStable()
		if !stable {
			t.Errorf("order %d: should be stable", n)
		}

		tfRes, _ := sys.TransferFunction(nil)
		h0 := mustEval(t, tfRes.TF, 0)[0][0]
		if math.Abs(real(h0)-1) > 1e-8 {
			t.Errorf("order %d: H(0) = %v, want 1", n, h0)
		}

		hAt2i := mustEval(t, tfRes.TF, complex(0, 2))[0][0]
		if math.Abs(cmplx.Abs(hAt2i)-1) > 1e-8 {
			t.Errorf("order %d: |H(2i)| = %v, want 1", n, cmplx.Abs(hAt2i))
		}

		h1i := mustEval(t, tfRes.TF, s)[0][0]
		if cmplx.Abs(h1i-exact) > tols[n-1] {
			t.Errorf("order %d: H(i) error = %v, tol = %v", n, cmplx.Abs(h1i-exact), tols[n-1])
		}
	}
}

// Julia: series composition P1*delay → freq response matches P1_fr * exp(-jω*τ)
func TestPadeSeriesFreqResponse(t *testing.T) {
	tau := 1.0
	plant, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	pade, _ := PadeDelay(tau, 5)

	series, err := Series(plant, pade)
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.1, 0.5, 1.0} {
		s := complex(0, w)
		plantFR := complex(1, 0) / (s + 1)
		exact := plantFR * cmplx.Exp(-s*complex(tau, 0))

		tfRes, _ := series.TransferFunction(nil)
		got := mustEval(t, tfRes.TF, s)[0][0]
		if cmplx.Abs(got-exact) > 0.01 {
			t.Errorf("w=%v: got %v, want %v (err=%v)", w, got, exact, cmplx.Abs(got-exact))
		}
	}
}

// Julia: feedback(1, P1*pade(τ,n)) freq response
func TestPadeFeedbackFreqResponse(t *testing.T) {
	tau := 1.0
	plant, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	pade, _ := PadeDelay(tau, 5)

	delayed, _ := Series(plant, pade)
	ctrl, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	cl, err := Feedback(delayed, ctrl, -1)
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.1, 0.5, 1.0} {
		s := complex(0, w)
		P1fr := complex(1, 0) / (s + 1)
		delayFR := cmplx.Exp(-s * complex(tau, 0))
		exact := P1fr * delayFR / (1 + P1fr*delayFR)

		tfRes, _ := cl.TransferFunction(nil)
		got := mustEval(t, tfRes.TF, s)[0][0]
		if cmplx.Abs(got-exact) > 0.02 {
			t.Errorf("w=%v: got %v, want %v (err=%v)", w, got, exact, cmplx.Abs(got-exact))
		}
	}
}

func TestPadeDelayWithFeedback(t *testing.T) {
	tau := 0.5
	pade, _ := PadeDelay(tau, 3)

	// Plant: 1/(s+1) with delay tau
	plant, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)

	// Approximate delayed plant = Series(plant, pade)
	delayed, err := Series(plant, pade)
	if err != nil {
		t.Fatal(err)
	}

	// Controller: simple gain
	ctrl, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	// Feedback should work since no Delay field set
	_, err = Feedback(delayed, ctrl, -1)
	if err != nil {
		t.Errorf("feedback with Pade-approximated delay should work: %v", err)
	}
}

func TestPadeSISO(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{-2}, []float64{1}, []float64{3}, []float64{0}, 0)
	sys.InputDelay = []float64{0.5}

	result, err := sys.Pade(3)
	if err != nil {
		t.Fatal(err)
	}

	if result.HasDelay() {
		t.Error("result should be delay-free")
	}

	n, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims: m=%d p=%d, want 1,1", m, p)
	}
	if n != 1+3 {
		t.Errorf("states = %d, want 4 (1 original + 3 from Pade order 3)", n)
	}

	tfRes, _ := result.TransferFunction(nil)
	for _, w := range []float64{0.1, 0.5, 1.0} {
		s := complex(0, w)
		got := mustEval(t, tfRes.TF, s)[0][0]
		plantFR := complex(3, 0) / (s + 2)
		exact := plantFR * cmplx.Exp(-s*complex(0.5, 0))
		if cmplx.Abs(got-exact) > 0.02 {
			t.Errorf("w=%v: err=%v", w, cmplx.Abs(got-exact))
		}
	}
}

func TestPadeMIMO(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0)
	sys.InputDelay = []float64{0.1, 0.2}

	result, err := sys.Pade(3)
	if err != nil {
		t.Fatal(err)
	}

	if result.HasDelay() {
		t.Error("result should be delay-free")
	}

	n, _, _ := result.Dims()
	if n != 2+6 {
		t.Errorf("states = %d, want 8 (2 original + 2*3 Pade)", n)
	}
}

func TestPadeOutputDelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.OutputDelay = []float64{0.3}

	result, err := sys.Pade(4)
	if err != nil {
		t.Fatal(err)
	}

	if result.HasDelay() {
		t.Error("result should be delay-free")
	}

	n, _, _ := result.Dims()
	if n != 1+4 {
		t.Errorf("states = %d, want 5", n)
	}

	tfRes, _ := result.TransferFunction(nil)
	for _, w := range []float64{0.1, 0.5, 1.0} {
		s := complex(0, w)
		got := mustEval(t, tfRes.TF, s)[0][0]
		exact := cmplx.Exp(-s*complex(0.3, 0)) / (s + 1)
		if cmplx.Abs(got-exact) > 0.01 {
			t.Errorf("w=%v: err=%v", w, cmplx.Abs(got-exact))
		}
	}
}

func TestPadeIODelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 2, 2,
		[]float64{-1},
		[]float64{1, 1},
		[]float64{1, 1},
		[]float64{0, 0, 0, 0},
		0)
	sys.Delay = mat.NewDense(2, 2, []float64{
		0.1, 0.2,
		0.3, 0.4,
	})

	result, err := sys.Pade(3)
	if err != nil {
		t.Fatal(err)
	}

	if result.HasDelay() {
		t.Error("result should be delay-free")
	}

	_, m, p := result.Dims()
	if m != 2 || p != 2 {
		t.Errorf("I/O dims changed: m=%d p=%d", m, p)
	}
}

func TestPadeInternalDelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.LFT = &LFTDelay{
		Tau: []float64{0.5},
		B2:  mat.NewDense(1, 1, []float64{0.5}),
		C2:  mat.NewDense(1, 1, []float64{1}),
		D12: mat.NewDense(1, 1, []float64{0}),
		D21: mat.NewDense(1, 1, []float64{0}),
		D22: mat.NewDense(1, 1, []float64{0}),
	}

	result, err := sys.Pade(3)
	if err != nil {
		t.Fatal(err)
	}

	if result.HasDelay() || result.HasInternalDelay() {
		t.Error("result should be delay-free")
	}

	n, _, _ := result.Dims()
	if n != 1+3 {
		t.Errorf("states = %d, want 4", n)
	}
}

func TestPadeNoDelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)

	result, err := sys.Pade(3)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := result.Dims()
	if n != 1 {
		t.Errorf("no-delay system should return copy, got n=%d", n)
	}
}

func TestPadeDiscrete(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 0.1)

	_, err := sys.Pade(3)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("expected ErrWrongDomain, got %v", err)
	}
}

func TestPadeAllpass(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	sys.InputDelay = []float64{1.0}

	result, err := sys.Pade(5)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := result.TransferFunction(nil)
	for _, w := range []float64{0.01, 0.1, 1, 5, 10} {
		s := complex(0, w)
		h := mustEval(t, tfRes.TF, s)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-6 {
			t.Errorf("w=%v: |H| = %v, want 1", w, mag)
		}
	}
}

func TestPadeDelayRejectsNonFiniteDelay(t *testing.T) {
	for _, tau := range []float64{math.NaN(), math.Inf(1)} {
		if _, err := PadeDelay(tau, 2); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("PadeDelay(%v) err = %v, want ErrInvalidArgument", tau, err)
		}
	}
	if _, err := PadeDelay(-1, 2); !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("PadeDelay(-1) err = %v, want ErrNegativeDelay", err)
	}
}
