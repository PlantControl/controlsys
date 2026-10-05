package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

func TestThiranDelayOrder1(t *testing.T) {
	dt := 0.1
	tau := 0.15 // 1.5 samples
	sys, err := thiranDelay(tau, 1, dt)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.IsDiscrete() {
		t.Error("should be discrete")
	}
	n, m, p := sys.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (_,1,1)", n, m, p)
	}
}

func TestThiranDelayAllpass(t *testing.T) {
	dt := 1.0
	D := 3.4 // 3.4 samples
	sys, err := thiranDelay(D*dt, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	for _, w := range []float64{0.01, 0.1, 0.5, 1.0, 2.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := mustEval(t, tfRes.TF, z)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1 (allpass)", w, mag)
		}
	}
}

func TestThiranDelayGroupDelay(t *testing.T) {
	dt := 1.0
	D := 2.7
	order := 3
	sys, err := thiranDelay(D*dt, order, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	// Group delay at DC should approximate D samples
	dw := 1e-6
	z1 := cmplx.Exp(complex(0, dw))
	z2 := cmplx.Exp(complex(0, 2*dw))
	h1 := mustEval(t, tfRes.TF, z1)[0][0]
	h2 := mustEval(t, tfRes.TF, z2)[0][0]
	phase1 := cmplx.Phase(h1)
	phase2 := cmplx.Phase(h2)
	groupDelay := -(phase2 - phase1) / dw

	if math.Abs(groupDelay-D) > 0.01 {
		t.Errorf("group delay at DC = %v, want %v", groupDelay, D)
	}
}

func TestThiranDelayIntegerFallback(t *testing.T) {
	dt := 0.1
	tau := 0.5 // exactly 5 samples
	sys, err := thiranDelay(tau, 1, dt)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 5 {
		t.Errorf("integer delay: state dim = %d, want 5", n)
	}

	// Should act as pure z^{-5}
	tfRes, _ := sys.TransferFunction(nil)
	z := cmplx.Exp(complex(0, 0.3))
	h := mustEval(t, tfRes.TF, z)[0][0]
	expected := cmplx.Pow(z, -5)
	if cmplx.Abs(h-expected) > 1e-10 {
		t.Errorf("integer delay: H(z)=%v, want z^{-5}=%v", h, expected)
	}
}

func TestThiranDelayNegative(t *testing.T) {
	_, err := thiranDelay(-1, 1, 0.1)
	if !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("expected ErrNegativeDelay, got %v", err)
	}
}

func TestThiranDelayInvalidOrder(t *testing.T) {
	_, err := thiranDelay(0.5, 0, 0.1)
	if !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("order 0: err = %v, want ErrInvalidOrder", err)
	}
	_, err = thiranDelay(0.5, 11, 0.1)
	if !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("order 11: err = %v, want ErrInvalidOrder", err)
	}
}

func TestThiranDelayStability(t *testing.T) {
	dt := 1.0
	for _, D := range []float64{0.6, 1.5, 2.7, 3.7, 5.1} {
		order := min(max(int(math.Floor(D)), 1), 10)
		sys, err := thiranDelay(D*dt, order, dt)
		if err != nil {
			t.Fatalf("D=%v order=%d: %v", D, order, err)
		}
		stable, err := sys.IsStable()
		if err != nil {
			t.Fatalf("D=%v order=%d: stability check failed: %v", D, order, err)
		}
		if !stable {
			t.Errorf("D=%v order=%d: Thiran should be stable", D, order)
		}
	}
}

// Julia: thiran(2*Ts, Ts) == 1/z^2 for integer delays
func TestThiranIntegerDelayExact(t *testing.T) {
	for _, nSamples := range []int{1, 2, 3, 5} {
		for _, dt := range []float64{1.0, 0.5, 1.1} {
			tau := float64(nSamples) * dt
			sys, err := thiranDelay(tau, 1, dt)
			if err != nil {
				t.Fatalf("D=%d dt=%v: %v", nSamples, dt, err)
			}
			tfRes, _ := sys.TransferFunction(nil)

			for _, w := range []float64{0.1, 0.5, 1.0, 2.0} {
				z := cmplx.Exp(complex(0, w*dt))
				got := mustEval(t, tfRes.TF, z)[0][0]
				want := cmplx.Pow(z, complex(float64(-nSamples), 0))
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("D=%d dt=%v w=%v: got %v, want z^{-%d}=%v", nSamples, dt, w, got, nSamples, want)
				}
			}
		}
	}
}

// Julia: thiran(pi, 1) uses order=ceil(pi)=4 internally with different stability bounds.
// We use order=3; stability requires D > N-1.
// Verify allpass + group delay properties match the fractional delay.
func TestThiranPiDelay(t *testing.T) {
	D := math.Pi
	dt := 1.0
	sys, err := thiranDelay(D*dt, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	for _, w := range []float64{0.01, 0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := mustEval(t, tfRes.TF, z)[0][0]

		// Allpass: |H(z)| = 1
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1", w, mag)
		}
	}

	// Group delay at DC should ≈ π samples
	dw := 1e-6
	z1 := cmplx.Exp(complex(0, dw))
	z2 := cmplx.Exp(complex(0, 2*dw))
	h1 := mustEval(t, tfRes.TF, z1)[0][0]
	h2 := mustEval(t, tfRes.TF, z2)[0][0]
	gd := -(cmplx.Phase(h2) - cmplx.Phase(h1)) / dw
	if math.Abs(gd-D) > 0.01 {
		t.Errorf("group delay = %v, want π ≈ %v", gd, D)
	}
}

// Verify Thiran coefficients directly against Julia's known values.
// Julia: thiran(pi, 1) with order=4 has specific coefficients.
// We compute order=4 coefficients directly to cross-validate our coefficient formula.
func TestThiranCoeffsDirectly(t *testing.T) {
	D := math.Pi
	N := 4
	a := thiranCoeffs(D, N)

	wantA := []float64{
		1.0,
		0.8290601401044773,
		-0.03424682772398137,
		0.0042438423976339556,
		-0.00031815668236122736,
	}

	for k := 0; k <= N; k++ {
		if math.Abs(a[k]-wantA[k]) > 1e-12 {
			t.Errorf("a[%d] = %.16g, want %.16g", k, a[k], wantA[k])
		}
	}
}

// Verify integerDelaySS frequency response matches exp(-jω*d*dt)
func TestIntegerDelaySSFreqResponse(t *testing.T) {
	for _, d := range []int{1, 3, 7} {
		dt := 0.1
		sys, _ := integerDelaySS(d, dt)
		tfRes, _ := sys.TransferFunction(nil)

		for _, w := range []float64{0.5, 1.0, 5.0} {
			z := cmplx.Exp(complex(0, w*dt))
			got := mustEval(t, tfRes.TF, z)[0][0]
			want := cmplx.Exp(complex(0, -w*float64(d)*dt))
			if cmplx.Abs(got-want) > 1e-12 {
				t.Errorf("d=%d w=%v: got %v, want %v", d, w, got, want)
			}
		}
	}
}

func TestThiranDelayRejectsBelowStabilityBound(t *testing.T) {
	for _, tc := range []struct {
		samples float64
		order   int
	}{{0.01, 3}, {2 - 1e-6, 3}, {0.9, 2}, {1.5, 3}} {
		_, err := thiranDelay(tc.samples*0.1, tc.order, 0.1)
		if !errors.Is(err, ErrFractionalDelay) {
			t.Errorf("D=%g N=%d: err = %v, want ErrFractionalDelay", tc.samples, tc.order, err)
		}
	}
}

func TestThiranDelayShortStableDelays(t *testing.T) {
	for _, tc := range []struct {
		name    string
		tau     float64
		order   int
		wantDen []float64
		tol     float64
	}{
		{"matlab-thiran-2.4", 0.24, 3, []float64{1, 0.5294, -0.04813, 0.004159}, 5e-4},
		{"first-order-0.2", 0.02, 1, []float64{1, 2.0 / 3}, 1e-14},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dt := 0.1
			sys, err := thiranDelay(tc.tau, tc.order, dt)
			if err != nil {
				t.Fatal(err)
			}
			if n, m, p := sys.Dims(); n != tc.order || m != 1 || p != 1 {
				t.Fatalf("dims = %d,%d,%d, want %d,1,1", n, m, p, tc.order)
			}
			den := thiranCoeffs(tc.tau/dt, tc.order)
			for k, want := range tc.wantDen {
				if math.Abs(den[k]-want) > tc.tol*math.Max(1, math.Abs(want)) {
					t.Errorf("den[%d] = %.6g, want %.6g", k, den[k], want)
				}
			}
			assertThiranStableAllpass(t, sys, tc.tau/dt)
		})
	}
	sys, err := thiranDelay(0.02, 1, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 || cmplx.Abs(poles[0]-complex(-2.0/3, 0)) > 1e-14 {
		t.Fatalf("first-order poles = %v, want -2/3", poles)
	}
}

func TestThiranDelayNearStabilityBound(t *testing.T) {
	for _, order := range []int{1, 2, 3, 5} {
		samples := float64(order-1) + 1e-3
		sys, err := thiranDelay(samples*0.1, order, 0.1)
		if err != nil {
			t.Fatalf("N=%d D=%g: %v", order, samples, err)
		}
		assertThiranStableAllpass(t, sys, samples)
	}
}

func TestThiranDelayIntegerIgnoresOrderBound(t *testing.T) {
	for _, tc := range []struct{ samples, order int }{{0, 1}, {0, 3}, {2, 3}, {1, 4}} {
		sys, err := thiranDelay(float64(tc.samples)*0.1, tc.order, 0.1)
		if err != nil {
			t.Fatalf("D=%d N=%d: %v", tc.samples, tc.order, err)
		}
		if n, _, _ := sys.Dims(); n != tc.samples {
			t.Fatalf("D=%d N=%d: states = %d, want %d", tc.samples, tc.order, n, tc.samples)
		}
		resp, err := sys.FreqResponse([]float64{0.7, 9})
		if err != nil {
			t.Fatal(err)
		}
		for k, w := range []float64{0.7, 9} {
			want := cmplx.Exp(complex(0, -w*0.1*float64(tc.samples)))
			if got := resp.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12 {
				t.Fatalf("D=%d N=%d w=%g: H = %v, want %v", tc.samples, tc.order, w, got, want)
			}
		}
	}
}

func assertThiranStableAllpass(t *testing.T, sys *System, samples float64) {
	t.Helper()
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, pole := range poles {
		if cmplx.Abs(pole) >= 1 {
			t.Fatalf("D=%g: pole %v outside unit circle", samples, pole)
		}
	}
	omega := []float64{0.01, 0.5, 2, 5, 9}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		if mag := cmplx.Abs(resp.At(k, 0, 0)); math.Abs(mag-1) > 1e-9 {
			t.Fatalf("D=%g w=%g: |H| = %.12g, want 1", samples, omega[k], mag)
		}
	}
	gd := -cmplx.Phase(resp.At(0, 0, 0)) / (omega[0] * sys.Dt)
	if math.Abs(gd-samples) > 1e-3 {
		t.Fatalf("D=%g: low-frequency group delay = %g samples", samples, gd)
	}
}

func TestThiranDelayInvalidDt(t *testing.T) {
	_, err := thiranDelay(1.0, 1, 0)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
	_, err = thiranDelay(1.0, 1, -1)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
}

func TestThiranDelayAutomaticOrder(t *testing.T) {
	for _, tc := range []struct {
		tau, dt float64
		states  int
	}{
		{0.24, 0.1, 3},
		{0.05, 0.1, 1},
		{1.37, 0.25, 6},
		{0.3, 0.1, 3},
		{0, 0.1, 0},
	} {
		sys, err := ThiranDelay(tc.tau, tc.dt)
		if err != nil {
			t.Fatalf("tau=%g dt=%g: %v", tc.tau, tc.dt, err)
		}
		n, m, p := sys.Dims()
		if n != tc.states || m != 1 || p != 1 || sys.Dt != tc.dt {
			t.Errorf("tau=%g: dims (%d,%d,%d) Dt=%g, want (%d,1,1) Dt=%g", tc.tau, n, m, p, sys.Dt, tc.states, tc.dt)
		}
		D := tc.tau / tc.dt
		// Allpass with phase delay D at low frequency (maximally flat group delay).
		for _, w := range []float64{1e-4, 0.3, 1.2} {
			h, err := sys.EvalFr(cmplx.Exp(complex(0, w)))
			if err != nil {
				t.Fatal(err)
			}
			if a := cmplx.Abs(h[0][0]); math.Abs(a-1) > 1e-9 {
				t.Errorf("tau=%g w=%g |H| = %g, want 1", tc.tau, w, a)
			}
			if w == 1e-4 {
				if pd := -cmplx.Phase(h[0][0]) / w; math.Abs(pd-D) > 1e-6 {
					t.Errorf("tau=%g phase delay %g, want %g", tc.tau, pd, D)
				}
			}
		}
		if stable, err := sys.IsStable(); err != nil || !stable {
			t.Errorf("tau=%g: stable=%v err=%v", tc.tau, stable, err)
		}
	}
}

func TestThiranDelayRejectsInvalidArgs(t *testing.T) {
	for _, tc := range []struct {
		tau, dt float64
		want    error
	}{
		{-0.1, 0.1, ErrNegativeDelay},
		{math.NaN(), 0.1, ErrNegativeDelay},
		{math.Inf(1), 0.1, ErrNegativeDelay},
		{0.1, 0, ErrInvalidSampleTime},
		{0.1, math.NaN(), ErrInvalidSampleTime},
		{0.1, math.Inf(1), ErrInvalidSampleTime},
	} {
		_, err := ThiranDelay(tc.tau, tc.dt)
		if !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), "ThiranDelay: ") {
			t.Errorf("tau=%g dt=%g: err = %v, want ThiranDelay: ... %v", tc.tau, tc.dt, err, tc.want)
		}
	}
}
