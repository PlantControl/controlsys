package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

func mustEvalFr(t *testing.T, sys *System, z complex128) complex128 {
	t.Helper()
	h, err := sys.EvalFr(z)
	if err != nil {
		t.Fatal(err)
	}
	return h[0][0]
}

func TestConversionThiranChannelOrder1(t *testing.T) {
	sys, err := conversionThiranChannel(1.5, 0.1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.IsDiscrete() {
		t.Error("should be discrete")
	}
	n, m, p := sys.Dims()
	if n != 1 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (1,1,1)", n, m, p)
	}
	if len(sys.InputDelay) != 1 || sys.InputDelay[0] != 1 {
		t.Errorf("InputDelay = %v, want [1]", sys.InputDelay)
	}
}

func TestConversionThiranChannelAllpass(t *testing.T) {
	dt := 1.0
	sys, err := conversionThiranChannel(3.4, dt, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0.01, 0.1, 0.5, 1.0, 2.0} {
		mag := cmplx.Abs(mustEvalFr(t, sys, cmplx.Exp(complex(0, w*dt))))
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1 (allpass)", w, mag)
		}
	}
}

func TestConversionThiranChannelGroupDelay(t *testing.T) {
	D := 2.7
	sys, err := conversionThiranChannel(D, 1.0, 3)
	if err != nil {
		t.Fatal(err)
	}
	dw := 1e-6
	h1 := mustEvalFr(t, sys, cmplx.Exp(complex(0, dw)))
	h2 := mustEvalFr(t, sys, cmplx.Exp(complex(0, 2*dw)))
	groupDelay := -(cmplx.Phase(h2) - cmplx.Phase(h1)) / dw
	if math.Abs(groupDelay-D) > 0.01 {
		t.Errorf("group delay at DC = %v, want %v", groupDelay, D)
	}
}

func TestThiranDelayIntegerFallback(t *testing.T) {
	sys, err := ThiranDelay(0.5, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 5 {
		t.Errorf("integer delay: state dim = %d, want 5", n)
	}
	tfRes, _ := sys.TransferFunction(nil)
	z := cmplx.Exp(complex(0, 0.3))
	h := mustEval(t, tfRes.TF, z)[0][0]
	expected := cmplx.Pow(z, -5)
	if cmplx.Abs(h-expected) > 1e-10 {
		t.Errorf("integer delay: H(z)=%v, want z^{-5}=%v", h, expected)
	}
}

func TestConversionThiranChannelIntegerIsInputDelay(t *testing.T) {
	for _, order := range []int{1, 3} {
		sys, err := conversionThiranChannel(5, 0.1, order)
		if err != nil {
			t.Fatal(err)
		}
		if n, _, _ := sys.Dims(); n != 0 || len(sys.InputDelay) != 1 || sys.InputDelay[0] != 5 {
			t.Fatalf("N=%d: states %d InputDelay %v, want 0 and [5]", order, n, sys.InputDelay)
		}
		z := cmplx.Exp(complex(0, 0.3))
		if h := mustEvalFr(t, sys, z); cmplx.Abs(h-cmplx.Pow(z, -5)) > 1e-10 {
			t.Errorf("N=%d: H(z)=%v, want z^{-5}", order, h)
		}
	}
}

func TestConversionThiranChannelRejectsInvalidArgs(t *testing.T) {
	for _, samples := range []float64{-10, math.NaN(), math.Inf(1)} {
		if sys, err := conversionThiranChannel(samples, 0.1, 1); !errors.Is(err, ErrNegativeDelay) || sys != nil {
			t.Errorf("samples=%g: %v, %v; want nil, ErrNegativeDelay", samples, sys, err)
		}
	}
	for _, order := range []int{0, -1} {
		if sys, err := conversionThiranChannel(5, 0.1, order); !errors.Is(err, ErrInvalidConversionOptions) || sys != nil {
			t.Errorf("order %d: %v, %v; want nil, ErrInvalidConversionOptions", order, sys, err)
		}
	}
}

func TestConversionThiranChannelStability(t *testing.T) {
	for _, D := range []float64{0.6, 1.5, 2.7, 3.7, 5.1} {
		order := min(max(int(math.Floor(D)), 1), 10)
		sys, err := conversionThiranChannel(D, 1.0, order)
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
			exact, err := ThiranDelay(tau, dt)
			if err != nil {
				t.Fatalf("D=%d dt=%v: %v", nSamples, dt, err)
			}
			channel, err := conversionThiranChannel(float64(nSamples), dt, 1)
			if err != nil {
				t.Fatalf("D=%d dt=%v: %v", nSamples, dt, err)
			}
			for _, sys := range []*System{exact, channel} {
				for _, w := range []float64{0.1, 0.5, 1.0, 2.0} {
					z := cmplx.Exp(complex(0, w*dt))
					got := mustEvalFr(t, sys, z)
					want := cmplx.Pow(z, complex(float64(-nSamples), 0))
					if cmplx.Abs(got-want) > 1e-10 {
						t.Errorf("D=%d dt=%v w=%v: got %v, want z^{-%d}=%v", nSamples, dt, w, got, nSamples, want)
					}
				}
			}
		}
	}
}

// Julia: thiran(pi, 1) uses order=ceil(pi)=4. With order 3 the live path keeps
// one whole sample as InputDelay; allpass and DC group delay must still be pi.
func TestConversionThiranChannelPiDelay(t *testing.T) {
	D := math.Pi
	dt := 1.0
	sys, err := conversionThiranChannel(D, dt, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0.01, 0.1, 0.5, 1.0} {
		mag := cmplx.Abs(mustEvalFr(t, sys, cmplx.Exp(complex(0, w*dt))))
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1", w, mag)
		}
	}
	dw := 1e-6
	h1 := mustEvalFr(t, sys, cmplx.Exp(complex(0, dw)))
	h2 := mustEvalFr(t, sys, cmplx.Exp(complex(0, 2*dw)))
	gd := -(cmplx.Phase(h2) - cmplx.Phase(h1)) / dw
	if math.Abs(gd-D) > 0.01 {
		t.Errorf("group delay = %v, want π ≈ %v", gd, D)
	}
}

// Julia: thiran(pi, 1) has order 4 with these denominator coefficients.
var juliaThiranPiDen = []float64{
	1.0,
	0.8290601401044773,
	-0.03424682772398137,
	0.0042438423976339556,
	-0.00031815668236122736,
}

func TestThiranCoeffsDirectly(t *testing.T) {
	a := thiranCoeffs(math.Pi, 4)
	for k, want := range juliaThiranPiDen {
		if math.Abs(a[k]-want) > 1e-12 {
			t.Errorf("a[%d] = %.16g, want %.16g", k, a[k], want)
		}
	}
}

func TestConversionThiranChannelMatchesJuliaPi(t *testing.T) {
	sys, err := conversionThiranChannel(math.Pi, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := sys.Dims(); n != 4 || len(sys.InputDelay) != 0 {
		t.Fatalf("states %d InputDelay %v, want 4 and none", n, sys.InputDelay)
	}
	for _, w := range []float64{0.01, 0.4, 1.3, 3} {
		z := cmplx.Exp(complex(0, w))
		var num, den complex128
		for k, a := range juliaThiranPiDen {
			den += complex(a, 0) * cmplx.Pow(z, complex(float64(-k), 0))
			num += complex(juliaThiranPiDen[4-k], 0) * cmplx.Pow(z, complex(float64(-k), 0))
		}
		if got := mustEvalFr(t, sys, z); cmplx.Abs(got-num/den) > 1e-12 {
			t.Errorf("w=%g: H = %v, want %v", w, got, num/den)
		}
	}
}

// Verify integerDelaySS frequency response matches exp(-jω*d*dt)
func TestIntegerDelaySSFreqResponse(t *testing.T) {
	for _, d := range []float64{1, 3, 7} {
		dt := 0.1
		sys, _ := integerDelaySS(d, dt)
		tfRes, _ := sys.TransferFunction(nil)

		for _, w := range []float64{0.5, 1.0, 5.0} {
			z := cmplx.Exp(complex(0, w*dt))
			got := mustEval(t, tfRes.TF, z)[0][0]
			want := cmplx.Exp(complex(0, -w*float64(d)*dt))
			if cmplx.Abs(got-want) > 1e-12 {
				t.Errorf("d=%g w=%v: got %v, want %v", d, w, got, want)
			}
		}
	}
}

// MATLAB c2d caps the Thiran order at ceil(D), so delays a fixed order-N
// filter could not stabilize (D <= N-1) get a lower-order stable filter.
func TestConversionThiranChannelShortDelayLowersOrder(t *testing.T) {
	for _, tc := range []struct {
		samples float64
		order   int
	}{{0.01, 3}, {2 - 1e-6, 3}, {0.9, 2}, {1.5, 3}} {
		sys, err := conversionThiranChannel(tc.samples, 0.1, tc.order)
		if err != nil {
			t.Fatalf("D=%g N=%d: %v", tc.samples, tc.order, err)
		}
		if n, _, _ := sys.Dims(); n != int(math.Ceil(tc.samples)) || len(sys.InputDelay) != 0 {
			t.Fatalf("D=%g N=%d: states %d InputDelay %v, want %d and none", tc.samples, tc.order, n, sys.InputDelay, int(math.Ceil(tc.samples)))
		}
		assertThiranStableAllpass(t, sys, tc.samples)
	}
}

func TestConversionThiranChannelShortStableDelays(t *testing.T) {
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
			sys, err := conversionThiranChannel(tc.tau/dt, dt, tc.order)
			if err != nil {
				t.Fatal(err)
			}
			if n, m, p := sys.Dims(); n != tc.order || m != 1 || p != 1 || len(sys.InputDelay) != 0 {
				t.Fatalf("dims = %d,%d,%d InputDelay %v, want %d,1,1 and none", n, m, p, sys.InputDelay, tc.order)
			}
			tfRes, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			den, num := tfRes.TF.Den[0], tfRes.TF.Num[0][0]
			if len(den) != len(tc.wantDen) || len(num) != len(tc.wantDen) {
				t.Fatalf("num %v den %v, want degree %d", num, den, tc.order)
			}
			for k, want := range tc.wantDen {
				if got := den[k] / den[0]; math.Abs(got-want) > tc.tol*math.Max(1, math.Abs(want)) {
					t.Errorf("den[%d] = %.6g, want %.6g", k, got, want)
				}
				if got := num[tc.order-k] / den[0]; math.Abs(got-want) > tc.tol*math.Max(1, math.Abs(want)) {
					t.Errorf("num[%d] = %.6g, want %.6g", tc.order-k, got, want)
				}
			}
			assertThiranStableAllpass(t, sys, tc.tau/dt)
		})
	}
	sys, err := conversionThiranChannel(0.2, 0.1, 1)
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

func TestConversionThiranChannelNearStabilityBound(t *testing.T) {
	for _, order := range []int{1, 2, 3, 5} {
		samples := float64(order-1) + 1e-3
		sys, err := conversionThiranChannel(samples, 0.1, order)
		if err != nil {
			t.Fatalf("N=%d D=%g: %v", order, samples, err)
		}
		if n, _, _ := sys.Dims(); n != order || len(sys.InputDelay) != 0 {
			t.Fatalf("N=%d: states %d InputDelay %v, want %d and none", order, n, sys.InputDelay, order)
		}
		assertThiranStableAllpass(t, sys, samples)
	}
}

func TestThiranDelayIntegerExactShift(t *testing.T) {
	for _, samples := range []int{0, 1, 2} {
		sys, err := ThiranDelay(float64(samples)*0.1, 0.1)
		if err != nil {
			t.Fatalf("D=%d: %v", samples, err)
		}
		if n, _, _ := sys.Dims(); n != samples {
			t.Fatalf("D=%d: states = %d, want %d", samples, n, samples)
		}
		resp, err := sys.FreqResponse([]float64{0.7, 9})
		if err != nil {
			t.Fatal(err)
		}
		for k, w := range []float64{0.7, 9} {
			want := cmplx.Exp(complex(0, -w*0.1*float64(samples)))
			if got := resp.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12 {
				t.Fatalf("D=%d w=%g: H = %v, want %v", samples, w, got, want)
			}
		}
	}
}

func TestConversionThiranChannelIntegerIgnoresOrder(t *testing.T) {
	for _, tc := range []struct{ samples, order int }{{0, 1}, {0, 3}, {2, 3}, {1, 4}} {
		sys, err := conversionThiranChannel(float64(tc.samples), 0.1, tc.order)
		if err != nil {
			t.Fatalf("D=%d N=%d: %v", tc.samples, tc.order, err)
		}
		if n, _, _ := sys.Dims(); n != 0 {
			t.Fatalf("D=%d N=%d: states = %d, want 0", tc.samples, tc.order, n)
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
		{0.1, -1, ErrInvalidSampleTime},
		{0.1, math.NaN(), ErrInvalidSampleTime},
		{0.1, math.Inf(1), ErrInvalidSampleTime},
	} {
		_, err := ThiranDelay(tc.tau, tc.dt)
		if !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), "ThiranDelay: ") {
			t.Errorf("tau=%g dt=%g: err = %v, want ThiranDelay: ... %v", tc.tau, tc.dt, err, tc.want)
		}
	}
}
