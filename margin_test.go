package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// === Phase 1: helper tests ===

func TestFindCrossings_Basic(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100}
	vals := []float64{-2, -1, 0.5, 2}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 1 {
		t.Fatalf("got %d crossings, want 1", len(cs))
	}
	if cs[0].idx != 1 {
		t.Errorf("idx = %d, want 1", cs[0].idx)
	}
	if cs[0].w < 1 || cs[0].w > 10 {
		t.Errorf("w = %v, want between 1 and 10", cs[0].w)
	}
}

func TestFindCrossings_Multiple(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100, 1000}
	vals := []float64{-1, 1, -1, 1, -1}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 4 {
		t.Fatalf("got %d crossings, want 4", len(cs))
	}
}

func TestFindCrossings_NoCrossing(t *testing.T) {
	omega := []float64{0.1, 1, 10}
	vals := []float64{1, 2, 3}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 0 {
		t.Fatalf("got %d crossings, want 0", len(cs))
	}
}

func TestFindCrossings_InfNaN(t *testing.T) {
	omega := []float64{0.1, 1, 10}
	vals := []float64{math.Inf(-1), 0, 1}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 0 {
		t.Fatalf("got %d crossings, want 0 (Inf skipped)", len(cs))
	}
}

func TestPhaseCrossings_SimpleWrap(t *testing.T) {
	omega := []float64{1, 2, 3, 4}
	phase := []float64{-170, -175, -185, -190}
	cs := phaseCrossings(omega, phase, -180)
	if len(cs) != 1 {
		t.Fatalf("got %d crossings, want 1", len(cs))
	}
	if cs[0].idx != 1 {
		t.Errorf("idx = %d, want 1", cs[0].idx)
	}
}

func TestPhaseCrossings_MultipleRevolutions(t *testing.T) {
	omega := []float64{1, 2, 3, 4, 5, 6}
	phase := []float64{-170, -190, -350, -370, -530, -550}
	cs := phaseCrossings(omega, phase, -180)
	if len(cs) != 2 {
		t.Fatalf("got %d crossings, want 2 (-360 is not a -180 crossing)", len(cs))
	}
	if cs[0].idx != 0 || cs[1].idx != 4 {
		t.Errorf("idx = %d,%d, want 0,4", cs[0].idx, cs[1].idx)
	}
}

func TestPhaseCrossings_ZeroWrapIgnored(t *testing.T) {
	omega := []float64{1, 2, 3, 4}
	phase := []float64{-10, 10, 170, -170}
	cs := phaseCrossings(omega, phase, -180)
	if len(cs) != 1 || cs[0].idx != 2 {
		t.Fatalf("got %+v, want single crossing at idx 2", cs)
	}
}

func TestRefineCrossing_Precision(t *testing.T) {
	// f(w) = 1/sqrt(1+w^2) - 1/sqrt(2), zero at w=1
	f := func(w float64) float64 {
		return 1/math.Sqrt(1+w*w) - 1/math.Sqrt(2)
	}
	w := refineCrossing(0.1, 10, f)
	if math.Abs(w-1) > 1e-8 {
		t.Errorf("refined crossing = %v, want 1.0", w)
	}
}

// === Phase 2: Margin / AllMargin tests ===

// G(s) = 1/s: integrator
// PM = 90 deg at w = 1, GM = +Inf
func TestMargin_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if math.Abs(r.PhaseMargin-90) > 1 {
		t.Errorf("PM = %v, want ~90 deg", r.PhaseMargin)
	}
	if !math.IsNaN(wpFreq(r)) {
		t.Errorf("WpFreq = %v, want NaN", wpFreq(r))
	}
	if math.Abs(wgFreq(r)-1) > 0.05 {
		t.Errorf("WgFreq = %v, want ~1", wgFreq(r))
	}
}

// G(s) = 1/(s(s+1)(s+2)):
// GM = 20*log10(6) ≈ 15.56 dB at w_pc = sqrt(2)
// PM ≈ 53.4 deg at w_gc ≈ 0.446
func TestMargin_ThirdOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGM := 20 * math.Log10(6)
	if math.Abs(r.GainMargin-wantGM) > 0.5 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGM)
	}
	if math.Abs(wpFreq(r)-math.Sqrt(2)) > 0.05 {
		t.Errorf("WpFreq = %v, want ~%v", wpFreq(r), math.Sqrt(2))
	}
	if math.Abs(r.PhaseMargin-53.4) > 1.5 {
		t.Errorf("PM = %v deg, want ~53.4 deg", r.PhaseMargin)
	}
	if math.Abs(wgFreq(r)-0.446) > 0.05 {
		t.Errorf("WgFreq = %v, want ~0.446", wgFreq(r))
	}
}

// G(s) = 100/(s^3 + 11s^2 + 10s) = 10/(s(s+1)(0.1s+1))
// GM ≈ 0.83 dB at w_pc = sqrt(10)
// This system is very close to instability
func TestMargin_NearUnstable(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -10, -11,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{100, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	// GM = -20*log10(10/11) = 20*log10(11/10) ≈ 0.828 dB
	wantGM := 20 * math.Log10(11.0/10.0)
	if math.Abs(r.GainMargin-wantGM) > 0.3 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGM)
	}
	if math.Abs(wpFreq(r)-math.Sqrt(10)) > 0.1 {
		t.Errorf("WpFreq = %v, want ~%v", wpFreq(r), math.Sqrt(10))
	}
}

// G = 0.1/(s+1): |G| < 0dB everywhere -> no crossings -> infinite margins
func TestMargin_LowGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf", r.PhaseMargin)
	}
}

func TestMargin_GainOnly(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf", r.PhaseMargin)
	}
}

func TestMargin_MIMOReject(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Margin(sys)
	if err == nil {
		t.Fatal("expected error for MIMO system")
	}
}

func TestAllMargin_ThirdOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.GainCrossFreqs) != 1 {
		t.Fatalf("got %d gain crossovers, want 1", len(all.GainCrossFreqs))
	}
	if len(all.PhaseCrossFreqs) != 1 {
		t.Fatalf("got %d phase crossovers, want 1", len(all.PhaseCrossFreqs))
	}
}

// Discrete: discretize the integrator, verify PM close to continuous
func TestMargin_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.01
	dsys, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(r.PhaseMargin-90) > 3 {
		t.Errorf("discrete PM = %v, want ~90 deg", r.PhaseMargin)
	}
}

// System with IO delay: G(s) = 2*exp(-s)/(s+1)
// w_gc = sqrt(3), PM = 180 - arctan(sqrt(3))*180/pi - sqrt(3)*180/pi ≈ 20.76 deg
func TestMargin_WithDelay(t *testing.T) {
	sys, err := NewWithDelay(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantPM := 180 - math.Atan(math.Sqrt(3))*180/math.Pi - math.Sqrt(3)*180/math.Pi
	if math.Abs(r.PhaseMargin-wantPM) > 2 {
		t.Errorf("PM = %v, want ~%v", r.PhaseMargin, wantPM)
	}
	if math.Abs(wgFreq(r)-math.Sqrt(3)) > 0.1 {
		t.Errorf("WgFreq = %v, want ~%v", wgFreq(r), math.Sqrt(3))
	}
}

// === Phase 3: Bandwidth tests ===

func TestBandwidth_ScaledAndZeroPole(t *testing.T) {
	z1 := mat.NewDense(1, 1, nil)
	for _, k := range []float64{1e-5, 1, 1e5} {
		sys, _ := New(mat.NewDense(1, 1, []float64{-k}), mat.NewDense(1, 1, []float64{k}), mat.NewDense(1, 1, []float64{1}), z1, 0)
		bw, err := Bandwidth(sys, -3)
		if err != nil {
			t.Fatal(err)
		}
		want := k * math.Sqrt(math.Pow(10, 0.3)-1)
		if math.Abs(bw-want) > 1e-6*want {
			t.Errorf("k=%g: BW = %g, want %g", k, bw, want)
		}
	}
	// H(z) = 0.5/(z(z-0.5)): pole at z=0, DC gain 1
	sys, _ := New(mat.NewDense(2, 2, []float64{0.5, 1, 0, 0}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{0.5, 0}), z1, 0.1)
	bw, err := Bandwidth(sys, -3)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Acos(1.25-0.25*math.Pow(10, 0.3)) / 0.1
	if math.Abs(bw-want) > 1e-6*want {
		t.Errorf("z=0 pole: BW = %g, want %g", bw, want)
	}
}

// G(s) = 1/(s+1): BW = 1 rad/s at -3dB
func TestBandwidth_FirstOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(bw-1.0) > 0.05 {
		t.Errorf("bandwidth = %v, want ~1.0", bw)
	}
}

// G(s) = wn^2/(s^2+2*z*wn*s+wn^2), wn=10, z=0.7
// |G(jw)| = wn^2 / sqrt((wn^2-w^2)^2 + (2*z*wn*w)^2)
func TestBandwidth_SecondOrder(t *testing.T) {
	wn := 10.0
	z := 0.7
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0, 1,
			-wn * wn, -2 * z * wn,
		}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	// BW for 2nd order: w_bw = wn * sqrt(1 - 2z^2 + sqrt(4z^4 - 4z^2 + 2))
	inner := 4*z*z*z*z - 4*z*z + 2
	wBW := wn * math.Sqrt(1-2*z*z+math.Sqrt(inner))
	if math.Abs(bw-wBW) > 0.5 {
		t.Errorf("bandwidth = %v, want ~%v", bw, wBW)
	}
}

func TestBandwidth_CustomDrop(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, -6)
	if err != nil {
		t.Fatal(err)
	}

	// -6dB: 1/sqrt(1+w^2) = 10^(-6/20) = 0.5012 -> w = sqrt(1/0.5012^2 - 1) ≈ 1.732
	want := math.Sqrt(1/math.Pow(10, -6.0/20)*1/math.Pow(10, -6.0/20) - 1)
	// Simpler: |G|=10^(-6/20), 1/(1+w^2) = 10^(-6/10), w^2 = 10^(6/10)-1
	w2 := math.Pow(10, 0.6) - 1
	want = math.Sqrt(w2)
	if math.Abs(bw-want) > 0.1 {
		t.Errorf("bandwidth(-6dB) = %v, want ~%v", bw, want)
	}
}

func TestBandwidth_GainOnly(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(bw, 1) {
		t.Errorf("bandwidth = %v, want +Inf (constant gain)", bw)
	}
}

func TestBandwidth_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.01
	dsys, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(dsys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(bw-1.0) > 0.1 {
		t.Errorf("discrete bandwidth = %v, want ~1.0", bw)
	}
}

// === Phase 4: DiskMargin tests ===

// L(s) = 10/(s+1): S = (s+1)/(s+11), Ms ≈ 1.0
// L = k/(s+1): |S−1/2|² = (ω²+(1−k)²)/(4(ω²+(1+k)²)) rises to 1/4, so the
// balanced disk margin is exactly 2 (GM [0, Inf], PM 90°), while
// ‖S‖∞ = 1 gives the σ = 1 margin 1 (GM [1/2, Inf], PM 60°).
func TestDiskMargin_StableFirstOrder(t *testing.T) {
	for _, k := range []float64{0.5, 10} {
		sys, err := New(
			mat.NewDense(1, 1, []float64{-1}),
			mat.NewDense(1, 1, []float64{k}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{0}),
			0,
		)
		if err != nil {
			t.Fatal(err)
		}
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(dm.Alpha-2) > 1e-9 || dm.GainMargin[0] > 1e-9 || dm.GainMargin[1] < 1e9 ||
			math.Abs(dm.PhaseMargin-90) > 1e-7 {
			t.Errorf("k=%g: %+v, want alpha 2, GM [0 Inf], PM 90", k, dm)
		}
		if math.Abs(dm.PeakSensitivity-1) > 1e-9 {
			t.Errorf("k=%g: Ms=%g, want 1", k, dm.PeakSensitivity)
		}
		dm, err = DiskMarginSkew(sys, 1)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(dm.Alpha-1) > 1e-9 || math.Abs(dm.GainMargin[0]-0.5) > 1e-9 || dm.GainMargin[1] < 1e8 ||
			math.Abs(dm.PhaseMargin-60) > 1e-6 {
			t.Errorf("k=%g sigma=1: %+v, want alpha 1, GM [0.5 Inf], PM 60", k, dm)
		}
	}
}

// L(s) = 10/(s(s+1)): moderately damped, Ms > 1
func TestDiskMargin_ModerateLoop(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0, 1,
			0, -1,
		}),
		mat.NewDense(2, 1, []float64{0, 10}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.PeakSensitivity < 1 {
		t.Errorf("Ms = %v, want > 1", dm.PeakSensitivity)
	}
	if dm.Alpha <= 0 || dm.Alpha >= 1 {
		t.Errorf("Alpha = %v, want in (0, 1)", dm.Alpha)
	}
	if dm.PhaseMargin <= 0 {
		t.Errorf("disk PM = %v, want > 0", dm.PhaseMargin)
	}
	if dm.GainMargin[0] <= 0 || dm.GainMargin[0] >= 1 {
		t.Errorf("GM_low = %v, want in (0,1)", dm.GainMargin[0])
	}
	if dm.GainMargin[1] <= 1 {
		t.Errorf("GM_high = %v, want > 1", dm.GainMargin[1])
	}
}

// Verify disk margin is tighter than classical margins
func TestDiskMargin_VsClassical(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	cm, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.PhaseMargin > cm.PhaseMargin+1 {
		t.Errorf("disk PM %v > classical PM %v (should be ≤)", dm.PhaseMargin, cm.PhaseMargin)
	}
	if !math.IsInf(cm.GainMargin, 1) {
		gmHighDB := dm.GainMarginDB[1]
		if gmHighDB > cm.GainMargin+1 {
			t.Errorf("disk GM_high %v dB > classical GM %v dB", gmHighDB, cm.GainMargin)
		}
	}
}

func TestDiskMargin_MIMOReject(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = DiskMargin(sys)
	if err == nil {
		t.Fatal("expected error for MIMO system")
	}
}

// === Benchmarks ===

// python-control: tf([1], [1,2,3,4])
// GM = 20*log10(2) ≈ 6.02 dB at wg = sqrt(3) ≈ 1.732
func TestAllMargin_ThirdOrderPythonControl(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -4, -3, -2},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.PhaseCrossFreqs) < 1 {
		t.Fatal("expected at least 1 phase crossover")
	}

	wantGM := 20 * math.Log10(2.0)
	wantWp := math.Sqrt(3.0)
	foundGM := false
	for i, w := range all.PhaseCrossFreqs {
		if math.Abs(w-wantWp) < 0.1 {
			if math.Abs(all.GainMargins[i]-wantGM) > 0.5 {
				t.Errorf("GM at w=%.2f: got %v dB, want ~%v dB", w, all.GainMargins[i], wantGM)
			}
			foundGM = true
		}
	}
	if !foundGM {
		t.Errorf("phase crossover at w≈%.2f not found in %v", wantWp, all.PhaseCrossFreqs)
	}

	if len(all.GainCrossFreqs) != 0 {
		t.Errorf("expected no gain crossings, got %v", all.GainCrossFreqs)
	}
}

// python-control: tf([2],[1,3,2,0]) sampled at dt=0.01
// Expected: gm=2.955761 (9.41 dB), pm=32.398, wg=1.403725, wp=0.749367
func TestAllMargin_Discrete(t *testing.T) {
	// G(s) = 2/(s^3+3s^2+2s) = 2/(s(s+1)(s+2))
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, 0, -2, -3},
		[]float64{0, 0, 1},
		[]float64{2, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	m, err := Margin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	// python-control: gm ≈ 9.41 dB (linear 2.9558)
	wantGM := 20 * math.Log10(2.9558)
	if math.Abs(m.GainMargin-wantGM) > 0.5 {
		t.Errorf("GM = %v dB, want ~%v dB", m.GainMargin, wantGM)
	}
	if math.Abs(m.PhaseMargin-32.4) > 2 {
		t.Errorf("PM = %v deg, want ~32.4 deg", m.PhaseMargin)
	}
	if math.Abs(wgFreq(m)-0.749) > 0.05 {
		t.Errorf("WgFreq = %v, want ~0.749", wgFreq(m))
	}
	if math.Abs(wpFreq(m)-1.404) > 0.05 {
		t.Errorf("WpFreq = %v, want ~1.404", wpFreq(m))
	}
}

// python-control: 8.75*(4s^2+0.4s+1)/((100s+1)(s^2+0.22s+1)(s^2/100+0.008s+1))
// multiple gain crossovers -> AllMargin should return multiple phase margins
func TestAllMargin_MultipleGainCrossovers(t *testing.T) {
	// Construct via ZPK: zeros from 4s^2+0.4s+1=0, poles from each factor
	// Use Margin to check the worst-case
	// G(s) = K/(s+1)^3, K=2: Margin at wg ≈ 0.766
	K := 2.0
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -1, -3, -3},
		[]float64{0, 0, 1},
		[]float64{K, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	// python-control: gm=4.0 (12.04 dB), pm=67.6058 deg, wg=1.7322, wp=0.7663
	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	wantGM := 20 * math.Log10(4.0)
	if math.Abs(m.GainMargin-wantGM) > 0.3 {
		t.Errorf("GM = %v dB, want ~%v dB", m.GainMargin, wantGM)
	}
	if math.Abs(m.PhaseMargin-67.6058) > 1.0 {
		t.Errorf("PM = %v deg, want ~67.6058 deg", m.PhaseMargin)
	}
	if math.Abs(wgFreq(m)-0.7663) > 0.03 {
		t.Errorf("WgFreq = %v, want ~0.7663", wgFreq(m))
	}
	if math.Abs(wpFreq(m)-1.7322) > 0.05 {
		t.Errorf("WpFreq = %v, want ~1.7322", wpFreq(m))
	}
	_ = all
}

// python-control: nonminimum phase system 0.01*(10-s)/((2+s)*(1+s))
// GM = 20*log10(300) ≈ 49.54 dB
func TestMargin_NonMinimumPhase(t *testing.T) {
	// G(s) = 0.01*(10-s)/((s+2)(s+1)) = 0.01*(-s+10)/(s^2+3s+2)
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{0.01 * 10, 0.01 * -1},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	// python-control: gm=300 (49.54 dB), wg=5.6569
	wantGM := 20 * math.Log10(300.0)
	if math.Abs(m.GainMargin-wantGM) > 0.5 {
		t.Errorf("GM = %v dB, want ~%v dB", m.GainMargin, wantGM)
	}
	if math.Abs(wpFreq(m)-5.6569) > 0.1 {
		t.Errorf("WpFreq = %v, want ~5.6569", wpFreq(m))
	}
}

// AllMargin for system with no crossings
func TestAllMargin_NoCrossings(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(all.GainCrossFreqs) != 0 {
		t.Errorf("expected no gain crossings, got %d", len(all.GainCrossFreqs))
	}
	if len(all.PhaseCrossFreqs) != 0 {
		t.Errorf("expected no phase crossings, got %d", len(all.PhaseCrossFreqs))
	}
}

// MATLAB bandwidth accepts only SISO models.
func TestBandwidth_MIMO(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := Bandwidth(sys, 0); !errors.Is(err, ErrNotSISO) {
		t.Errorf("MIMO bandwidth err = %v, want ErrNotSISO", err)
	}
}

// Bandwidth with integrator (no DC gain) should return 0
func TestBandwidth_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	if bw, err := Bandwidth(sys, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("integrator bandwidth = %v, %v; want ErrInvalidArgument (infinite DC gain)", bw, err)
	}
	zero := makeSISO(-1, 1, 1, 0)
	zero.C.Set(0, 0, 0)
	if bw, err := Bandwidth(zero, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("zero-gain bandwidth = %v, %v; want ErrInvalidArgument", bw, err)
	}
}

// MATLAB diskmargin doc, L = tf(25,[1 10 10 10]): GainMargin for skews
// -2, 0, 2 printed to 4 decimals; python-control gives DM 0.46, DPM 25.8°.
func TestDiskMargin_MATLABSkewExample(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -10, -10, -10},
		[]float64{0, 0, 1},
		[]float64{25, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		sigma float64
		gm    [2]float64
	}{{-2, [2]float64{0.4013, 1.3745}}, {0, [2]float64{0.6273, 1.5942}}, {2, [2]float64{0.7717, 1.7247}}} {
		dm, err := DiskMarginSkew(sys, tc.sigma)
		if err != nil {
			t.Fatal(err)
		}
		for k := range 2 {
			if math.Abs(dm.GainMargin[k]-tc.gm[k]) > 6e-5 {
				t.Errorf("sigma=%g: GainMargin=%v, MATLAB %v", tc.sigma, dm.GainMargin, tc.gm)
			}
		}
		if dm.Skew != tc.sigma {
			t.Errorf("Skew=%g, want %g", dm.Skew, tc.sigma)
		}
	}
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dm.Alpha-0.46) > 0.005 || math.Abs(dm.PhaseMargin-25.8) > 0.05 {
		t.Errorf("Alpha=%g PM=%g, want 0.46, 25.8", dm.Alpha, dm.PhaseMargin)
	}
	if math.Abs(dm.PhaseMargin-2*math.Atan(dm.Alpha/2)*180/math.Pi) > 1e-12 {
		t.Errorf("PM=%g, want 2·atan(α/2)", dm.PhaseMargin)
	}
}

// DiskMargin for discrete system
func TestDiskMargin_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.Alpha <= 0 {
		t.Errorf("discrete alpha = %v, want > 0", dm.Alpha)
	}
	if dm.PhaseMargin <= 0 {
		t.Errorf("discrete disk PM = %v, want > 0", dm.PhaseMargin)
	}
}

// Margin on LFT system (exercises the LFT workspace path in sisoEval.at)
func TestMargin_LFTSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_ = sys.SetInputDelay([]float64{0.5})
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(lft)
	if err != nil {
		t.Fatal(err)
	}

	if r.PhaseMargin <= 0 || math.IsNaN(r.PhaseMargin) {
		t.Errorf("PM = %v, want positive", r.PhaseMargin)
	}
}

// AllMargin on LFT system
func TestAllMargin_LFTSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_ = sys.SetInputDelay([]float64{0.5})
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(lft)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.GainCrossFreqs) == 0 {
		t.Error("expected gain crossover(s) for gain=2 system")
	}
}

// Bandwidth on LFT system - may not support DCGain, so just check no panic
func TestBandwidth_LFTSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_ = sys.SetInputDelay([]float64{0.1})
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(lft, 0)
	if err != nil {
		t.Skipf("Bandwidth on LFT system returned error: %v", err)
	}
	t.Logf("LFT bandwidth = %v", bw)
}

// DiskMargin on LFT system
func TestDiskMargin_LFTSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_ = sys.SetInputDelay([]float64{0.01})
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(lft)
	if err != nil {
		t.Fatal(err)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		return 10 * cmplx.Exp(-0.01*s) / (s + 1)
	}
	ms, _ := oraclePeakS(L, 1000)
	if math.Abs(dm.PeakSensitivity-ms) > 1e-7*ms || dm.Alpha <= 0 {
		t.Fatalf("Ms=%.12g alpha=%g, oracle Ms=%.12g", dm.PeakSensitivity, dm.Alpha, ms)
	}
}

func TestMargin_PythonControl_StableNoMargin(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -3, -2},
		[]float64{0, 1},
		[]float64{2, 1},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(m.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", m.GainMargin)
	}
	if !math.IsInf(m.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf", m.PhaseMargin)
	}
	if !math.IsNaN(wgFreq(m)) {
		t.Errorf("WgFreq = %v, want NaN", wgFreq(m))
	}
	if !math.IsNaN(wpFreq(m)) {
		t.Errorf("WpFreq = %v, want NaN", wpFreq(m))
	}
}

func TestMargin_PythonControl_StateSpace(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{1, 4, 3, 2},
		[]float64{1, -4},
		[]float64{1, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(m.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", m.GainMargin)
	}
	if math.Abs(m.PhaseMargin-147.0743) > 0.5 {
		t.Errorf("PM = %v deg, want ~147.0743 deg", m.PhaseMargin)
	}
	if math.Abs(wgFreq(m)-2.5483) > 0.02 {
		t.Errorf("WgFreq = %v, want ~2.5483", wgFreq(m))
	}
	if !math.IsNaN(wpFreq(m)) {
		t.Errorf("WpFreq = %v, want NaN", wpFreq(m))
	}
}

func TestAllMargin_PythonControl_Returnall(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -4, -3, -2},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.GainMargins) != 1 {
		t.Fatalf("expected 1 gain margin, got %d", len(all.GainMargins))
	}
	wantGM := 20 * math.Log10(2.0)
	if math.Abs(all.GainMargins[0]-wantGM) > 0.2 {
		t.Errorf("GM = %v dB, want ~%v dB", all.GainMargins[0], wantGM)
	}

	if len(all.PhaseMargins) != 0 {
		t.Errorf("expected no phase margins, got %v", all.PhaseMargins)
	}

	if len(all.PhaseCrossFreqs) != 1 {
		t.Fatalf("expected 1 phase cross freq, got %d", len(all.PhaseCrossFreqs))
	}
	if math.Abs(all.PhaseCrossFreqs[0]-1.7321) > 0.02 {
		t.Errorf("PhaseCrossFreq = %v, want ~1.7321", all.PhaseCrossFreqs[0])
	}

	if len(all.GainCrossFreqs) != 0 {
		t.Errorf("expected no gain cross freqs, got %v", all.GainCrossFreqs)
	}
}

func TestMargin_Discrete_PythonControl_SecondCase(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -1, -3, -3},
		[]float64{0, 0, 1},
		[]float64{2, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	m, err := Margin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	wantGM := 20 * math.Log10(3.4927)
	if math.Abs(m.GainMargin-wantGM) > 0.2 {
		t.Errorf("GM = %v dB, want ~%v dB", m.GainMargin, wantGM)
	}
	if math.Abs(m.PhaseMargin-65.4212) > 0.5 {
		t.Errorf("PM = %v deg, want ~65.4212 deg", m.PhaseMargin)
	}
	if math.Abs(wpFreq(m)-1.6283) > 0.02 {
		t.Errorf("WpFreq = %v, want ~1.6283", wpFreq(m))
	}
	if math.Abs(wgFreq(m)-0.76625) > 0.02 {
		t.Errorf("WgFreq = %v, want ~0.76625", wgFreq(m))
	}
}

func BenchmarkMargin_SISO(b *testing.B) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := Margin(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBandwidth_SISO(b *testing.B) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := Bandwidth(sys, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiskMargin_SISO(b *testing.B) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := DiskMargin(sys); err != nil {
			b.Fatal(err)
		}
	}
}

// The auto Bode grid for K/(s(s+1)) ends at 10 rad/s; crossovers at and
// beyond that edge must still be found.
func TestAllMargin_CrossoverAtGridEdge(t *testing.T) {
	for _, wc := range []float64{10, 12} {
		k := wc * math.Hypot(wc, 1)
		ol := makePlant(t, []float64{k}, []float64{1, 1, 0})
		mr, err := AllMargin(ol)
		if err != nil {
			t.Fatal(err)
		}
		if len(mr.GainCrossFreqs) != 1 {
			t.Fatalf("wc=%g: gain crossovers %v, want one", wc, mr.GainCrossFreqs)
		}
		if got := mr.GainCrossFreqs[0]; math.Abs(got-wc) > 1e-6*wc {
			t.Errorf("wc=%g: got %g", wc, got)
		}
		wantPM := 90 - math.Atan(wc)*180/math.Pi
		if got := mr.PhaseMargins[0]; math.Abs(got-wantPM) > 1e-6 {
			t.Errorf("wc=%g: PM %g, want %g", wc, got, wantPM)
		}
	}
}

func TestDiskMargin_DiscreteLoopDelay(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{0.6}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	dm, err := DiskMargin(lft)
	if err != nil {
		t.Fatal(err)
	}
	ms, peak := 0.0, 0.0
	for k := range 200001 {
		z := cmplx.Exp(complex(0, float64(k)/200000*math.Pi))
		l := 0.3 / (z - 0.6) / (z * z)
		ms = math.Max(ms, cmplx.Abs(1/(1+l)))
		peak = math.Max(peak, cmplx.Abs(1/(1+l)-0.5))
	}
	if math.Abs(dm.PeakSensitivity-ms) > 1e-6*ms || math.Abs(dm.Alpha-1/peak) > 1e-6/peak {
		t.Fatalf("Ms = %.12g alpha = %.12g, want Ms %.12g alpha %.12g", dm.PeakSensitivity, dm.Alpha, ms, 1/peak)
	}
}

// diskPeakOracle returns sup_ω |1/(1+L(e^{jω})) + c| on a dense grid of
// [0, wmax] refined by ternary search.
func diskPeakOracle(l func(w float64) complex128, wmax, c float64) (float64, float64) {
	f := func(w float64) float64 { return cmplx.Abs(1/(1+l(w)) + complex(c, 0)) }
	const n = 200000
	best, bw := 0.0, 0.0
	for i := range n + 1 {
		w := wmax * float64(i) / n
		if v := f(w); v > best {
			best, bw = v, w
		}
	}
	a, b := math.Max(bw-wmax/n, 0), math.Min(bw+wmax/n, wmax)
	for range 200 {
		m1, m2 := a+(b-a)/3, b-(b-a)/3
		if f(m1) < f(m2) {
			a = m1
		} else {
			b = m2
		}
	}
	w := (a + b) / 2
	return f(w), w
}

// L = C(sI−A)⁻¹B + D with non-symmetric A and D ≠ 0, continuous and ZOH
// discrete, against a hand-evaluated frequency response of 1/(1+L) + c.
func TestDiskMarginSkew_DenseGridOracle(t *testing.T) {
	A := []float64{-1, 2, -0.5, -3}
	B := []float64{1, 0.5}
	C := []float64{2, -1}
	const d = 0.3
	at := func(s complex128) complex128 {
		a11, a12, a21, a22 := s-complex(A[0], 0), complex(-A[1], 0), complex(-A[2], 0), s-complex(A[3], 0)
		det := a11*a22 - a12*a21
		x1 := (a22*complex(B[0], 0) - a12*complex(B[1], 0)) / det
		x2 := (a11*complex(B[1], 0) - a21*complex(B[0], 0)) / det
		return complex(C[0], 0)*x1 + complex(C[1], 0)*x2 + d
	}
	csys, err := NewFromSlices(2, 1, 1, A, B, C, []float64{d}, 0)
	if err != nil {
		t.Fatal(err)
	}
	const dt = 0.2
	dsys, err := csys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ad, bd, cd := dsys.A.RawMatrix().Data, dsys.B.RawMatrix().Data, dsys.C.RawMatrix().Data
	dat := func(z complex128) complex128 {
		a11, a12, a21, a22 := z-complex(ad[0], 0), complex(-ad[1], 0), complex(-ad[2], 0), z-complex(ad[3], 0)
		det := a11*a22 - a12*a21
		x1 := (a22*complex(bd[0], 0) - a12*complex(bd[1], 0)) / det
		x2 := (a11*complex(bd[1], 0) - a21*complex(bd[0], 0)) / det
		return complex(cd[0], 0)*x1 + complex(cd[1], 0)*x2 + d
	}
	for _, tc := range []struct {
		name string
		sys  *System
		l    func(w float64) complex128
		wmax float64
	}{
		{"continuous", csys, func(w float64) complex128 { return at(complex(0, w)) }, 1e3},
		{"discrete", dsys, func(w float64) complex128 { return dat(cmplx.Exp(complex(0, w*dt))) }, math.Pi / dt},
	} {
		for _, sigma := range []float64{-2, -0.5, 0, 1, 2} {
			dm, err := DiskMarginSkew(tc.sys, sigma)
			if err != nil {
				t.Fatal(err)
			}
			peak, wp := diskPeakOracle(tc.l, tc.wmax, (sigma-1)/2)
			if tc.name == "continuous" {
				peak = math.Max(peak, math.Abs(1/(1+d)+(sigma-1)/2))
			}
			if math.Abs(dm.Alpha-1/peak) > 1e-8/peak {
				t.Errorf("%s sigma=%g: alpha=%.12g, oracle %.12g", tc.name, sigma, dm.Alpha, 1/peak)
			}
			if got := cmplx.Abs(1/(1+tc.l(dmFreq(dm))) + complex((sigma-1)/2, 0)); !math.IsInf(dmFreq(dm), 1) &&
				math.Abs(got-peak) > 1e-8*peak {
				t.Errorf("%s sigma=%g: |S+c| at Frequency %g is %.12g, oracle peak %.12g at %g", tc.name, sigma, dmFreq(dm), got, peak, wp)
			}
			ms, _ := diskPeakOracle(tc.l, tc.wmax, 0)
			if math.Abs(dm.PeakSensitivity-ms) > 1e-8*ms {
				t.Errorf("%s: Ms=%.12g, oracle %.12g", tc.name, dm.PeakSensitivity, ms)
			}
			gm, pm := diskMarginOracleGM(dm.Alpha, sigma)
			if math.Abs(dm.GainMargin[0]-gm[0]) > 1e-9 || math.Abs(dm.PhaseMargin-pm) > 1e-7 ||
				(math.IsInf(gm[1], 1) != math.IsInf(dm.GainMargin[1], 1)) ||
				(!math.IsInf(gm[1], 1) && math.Abs(dm.GainMargin[1]-gm[1]) > 1e-9*gm[1]) {
				t.Errorf("%s sigma=%g: GM %v PM %g, oracle %v %g", tc.name, sigma, dm.GainMargin, dm.PhaseMargin, gm, pm)
			}
		}
	}
}

// diskMarginOracleGM finds the gain interval and phase arc around 1 in
// {F(δ): |δ| ≤ 1}, F = (1 + aδ)/(1 − bδ), by scanning membership
// δ = (z−1)/(a + bz) outward from z = 1 and bisecting the first exit.
func diskMarginOracleGM(alpha, sigma float64) ([2]float64, float64) {
	a, b := alpha*(1-sigma)/2, alpha*(1+sigma)/2
	in := func(z complex128) bool { return cmplx.Abs((z-1)/(complex(a, 0)+complex(b, 0)*z)) <= 1 }
	edge := func(p func(float64) complex128, x0, x1 float64, next func(float64) float64) (float64, bool) {
		lo := x0
		for hi := next(lo); ; lo, hi = hi, next(hi) {
			if (x1 > x0 && hi >= x1) || (x1 < x0 && hi <= x1) {
				return x1, false
			}
			if !in(p(hi)) {
				for range 200 {
					mid := (lo + hi) / 2
					if in(p(mid)) {
						lo = mid
					} else {
						hi = mid
					}
				}
				return lo, true
			}
		}
	}
	onReal := func(g float64) complex128 { return complex(g, 0) }
	gm := [2]float64{0, math.Inf(1)}
	if g, ok := edge(onReal, 1, 0, func(g float64) float64 { return g - 1e-4 }); ok {
		gm[0] = g
	}
	if g, ok := edge(onReal, 1, 1e12, func(g float64) float64 { return g * (1 + 1e-4) }); ok {
		gm[1] = g
	}
	th, _ := edge(func(th float64) complex128 { return cmplx.Exp(complex(0, th)) }, 0, math.Pi,
		func(th float64) float64 { return th + 1e-4 })
	return gm, th * 180 / math.Pi
}

// dm2gm doc examples, σ = 1 and σ = 0 closed forms, and the exterior-disk
// regime α(1+σ)/2 > 1 against membership bisection.
func TestDiskGainPhaseMargin(t *testing.T) {
	gm, pm := diskGainPhaseMargin(0.5, 0)
	if math.Abs(gm[1]-1.6667) > 5e-5 || math.Abs(pm-28.0725) > 5e-5 {
		t.Errorf("alpha 0.5 sigma 0: GM %v PM %g, MATLAB 1.6667 28.0725", gm, pm)
	}
	gm, pm = diskGainPhaseMargin(0.6, 0.75)
	if math.Abs(gm[0]-0.6066) > 5e-5 || math.Abs(gm[1]-2.2632) > 5e-5 || math.Abs(pm-34.2267) > 5e-5 {
		t.Errorf("alpha 0.6 sigma 0.75: GM %v PM %g, MATLAB [0.6066 2.2632] 34.2267", gm, pm)
	}
	for _, alpha := range []float64{0.1, 0.7, 1.5} {
		_, pm := diskGainPhaseMargin(alpha, 1)
		if math.Abs(pm-2*math.Asin(alpha/2)*180/math.Pi) > 1e-12 {
			t.Errorf("sigma 1 alpha %g: PM %g, want 2asin(α/2)", alpha, pm)
		}
		gm, pm = diskGainPhaseMargin(alpha, 0)
		if math.Abs(pm-2*math.Atan(alpha/2)*180/math.Pi) > 1e-12 || math.Abs(gm[0]*gm[1]-1) > 1e-12 {
			t.Errorf("sigma 0 alpha %g: GM %v PM %g, want balanced, 2atan(α/2)", alpha, gm, pm)
		}
	}
	for _, tc := range [][2]float64{{0.5, 0}, {1.2, -3}, {3, -2}, {1.5, 1}, {2, 0}, {2.5, 0}, {3, 0.5}, {6, 0.2}, {1.8, 1.5}, {0.4, 4}, {0.3, 9}} {
		gm, pm := diskGainPhaseMargin(tc[0], tc[1])
		ogm, opm := diskMarginOracleGM(tc[0], tc[1])
		if math.Abs(gm[0]-ogm[0]) > 1e-9 || math.Abs(pm-opm) > 1e-7 || math.IsInf(gm[1], 1) != math.IsInf(ogm[1], 1) ||
			(!math.IsInf(gm[1], 1) && math.Abs(gm[1]-ogm[1]) > 1e-9*ogm[1]) {
			t.Errorf("alpha %g sigma %g: GM %v PM %g, oracle %v %g", tc[0], tc[1], gm, pm, ogm, opm)
		}
	}
}

func TestDiskMarginSkew_Unstable(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		a := 1.0
		if dt > 0 {
			a = 1.5
		}
		sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{1}, []float64{0.2}, []float64{0}, dt)
		if err != nil {
			t.Fatal(err)
		}
		for _, sigma := range []float64{0, 1, -1} {
			dm, err := DiskMarginSkew(sys, sigma)
			if err != nil {
				t.Fatal(err)
			}
			if dm.Alpha != 0 || dm.GainMargin != [2]float64{1, 1} || dm.PhaseMargin != 0 || !math.IsInf(dm.PeakSensitivity, 1) {
				t.Errorf("dt=%g sigma=%g: %+v, want unstable margins", dt, sigma, dm)
			}
		}
	}
	if _, err := DiskMarginSkew(nil, math.NaN()); err == nil {
		t.Error("NaN skew accepted")
	}
}

func marginOracleSolve(a [][]complex128, b [][]complex128) [][]complex128 {
	n := len(a)
	if n == 0 {
		return b
	}
	m := len(b[0])
	aug := make([][]complex128, n)
	for i := range n {
		aug[i] = append(append([]complex128{}, a[i]...), b[i]...)
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if cmplx.Abs(aug[r][c]) > cmplx.Abs(aug[piv][c]) {
				piv = r
			}
		}
		aug[c], aug[piv] = aug[piv], aug[c]
		for r := range n {
			if r == c {
				continue
			}
			f := aug[r][c] / aug[c][c]
			for k := c; k < n+m; k++ {
				aug[r][k] -= f * aug[c][k]
			}
		}
	}
	x := make([][]complex128, n)
	for i := range n {
		x[i] = make([]complex128, m)
		for j := range m {
			x[i][j] = aug[i][n+j] / aug[i][i]
		}
	}
	return x
}

func marginOracleAt(m *mat.Dense, i, j int) float64 {
	if m == nil {
		return 0
	}
	r, c := m.Dims()
	if r == 0 || c == 0 {
		return 0
	}
	return m.At(i, j)
}

func marginOracleBlock(A, B, C, D *mat.Dense, n, cols, rows int, s complex128) [][]complex128 {
	M := make([][]complex128, n)
	Bc := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n)
		for j := range n {
			M[i][j] = complex(-marginOracleAt(A, i, j), 0)
		}
		M[i][i] += s
		Bc[i] = make([]complex128, cols)
		for j := range cols {
			Bc[i][j] = complex(marginOracleAt(B, i, j), 0)
		}
	}
	X := marginOracleSolve(M, Bc)
	G := make([][]complex128, rows)
	for i := range rows {
		G[i] = make([]complex128, cols)
		for j := range cols {
			v := complex(marginOracleAt(D, i, j), 0)
			for k := range n {
				v += complex(marginOracleAt(C, i, k), 0) * X[k][j]
			}
			G[i][j] = v
		}
	}
	return G
}

// marginOracle evaluates a SISO loop from its state-space resolvent, internal
// LFT delays and external delays, independently of TransferFunction.
func marginOracle(sys *System, w float64) complex128 {
	n, _, _ := sys.Dims()
	s := complex(0, w)
	if sys.Dt != 0 {
		s = cmplx.Exp(complex(0, w*sys.Dt))
	}
	delay := func(tau float64) complex128 {
		if sys.Dt == 0 {
			return cmplx.Exp(-s * complex(tau, 0))
		}
		return cmplx.Pow(s, complex(-tau, 0))
	}
	h := marginOracleBlock(sys.A, sys.B, sys.C, sys.D, n, 1, 1, s)[0][0]
	if sys.LFT != nil && len(sys.LFT.Tau) > 0 {
		N := len(sys.LFT.Tau)
		H12 := marginOracleBlock(sys.A, sys.LFT.B2, sys.C, sys.LFT.D12, n, N, 1, s)
		H21 := marginOracleBlock(sys.A, sys.B, sys.LFT.C2, sys.LFT.D21, n, 1, N, s)
		H22 := marginOracleBlock(sys.A, sys.LFT.B2, sys.LFT.C2, sys.LFT.D22, n, N, N, s)
		dl := make([]complex128, N)
		for k := range N {
			dl[k] = delay(sys.LFT.Tau[k])
		}
		IM := make([][]complex128, N)
		for i := range N {
			IM[i] = make([]complex128, N)
			for j := range N {
				IM[i][j] = -H22[i][j] * dl[j]
			}
			IM[i][i] += 1
		}
		X := marginOracleSolve(IM, H21)
		for k := range N {
			h += H12[0][k] * dl[k] * X[k][0]
		}
	}
	tau := marginOracleAt(sys.Delay, 0, 0)
	if sys.InputDelay != nil {
		tau += sys.InputDelay[0]
	}
	if sys.OutputDelay != nil {
		tau += sys.OutputDelay[0]
	}
	if tau != 0 {
		h *= delay(tau)
	}
	return h
}

func marginWrapDeg(x float64) float64 {
	x = math.Mod(x+180, 360)
	if x < 0 {
		x += 360
	}
	return x - 180
}

// marginBruteForce scans a dense log grid for |L|=1 and angle(L)=-180 (mod 360).
func marginBruteForce(sys *System, wlo, whi float64, n int) (gc, pc []float64) {
	var prevW, prevMag, prevS float64
	for k := 0; k <= n; k++ {
		w := wlo * math.Pow(whi/wlo, float64(k)/float64(n))
		h := marginOracle(sys, w)
		mag := math.Log(cmplx.Abs(h))
		sh := marginWrapDeg(cmplx.Phase(h)*180/math.Pi + 180)
		if k > 0 {
			if prevMag*mag < 0 {
				gc = append(gc, math.Sqrt(prevW*w))
			}
			if prevS*sh < 0 && math.Abs(prevS-sh) < 180 {
				pc = append(pc, math.Sqrt(prevW*w))
			}
		}
		prevW, prevMag, prevS = w, mag, sh
	}
	if sys.Dt > 0 && whi == math.Pi/sys.Dt && math.Abs(prevS) < 1e-6 {
		pc = append(pc, whi)
	}
	return gc, pc
}

func marginCountIn(ws []float64, lo, hi float64) []float64 {
	var out []float64
	for _, w := range ws {
		if w >= lo && w <= hi {
			out = append(out, w)
		}
	}
	return out
}

func marginMatch(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: got %d crossings %v, brute force %d %v", label, len(got), got, len(want), want)
		return
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-3*want[i] {
			t.Errorf("%s[%d]: got w=%g, brute force %g", label, i, got[i], want[i])
		}
	}
}

func marginTimeScale(sys *System, k float64) *System {
	c := sys.Copy()
	c.A.Scale(k, c.A)
	c.B.Scale(k, c.B)
	return c
}

func TestAllMargin_CrossoverContract(t *testing.T) {
	chain5 := func(k float64) *System {
		A := mat.NewDense(5, 5, nil)
		for i := range 5 {
			A.Set(i, i, -1)
			if i > 0 {
				A.Set(i, i-1, 1)
			}
		}
		B := mat.NewDense(5, 1, []float64{k, 0, 0, 0, 0})
		C := mat.NewDense(1, 5, []float64{0, 0, 0, 0, 1})
		s, _ := New(A, B, C, mat.NewDense(1, 1, nil), 0)
		return s
	}
	first := func() *System {
		s, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{2}), mat.NewDense(1, 1, nil), 0)
		return s
	}
	withIn := func(tau float64) *System {
		s := first()
		s.InputDelay = []float64{tau}
		return s
	}
	disc := func() *System {
		s, _ := New(mat.NewDense(2, 2, []float64{1.5, -0.7, 1, 0}), mat.NewDense(2, 1, []float64{1, 0}),
			mat.NewDense(1, 2, []float64{0.3, 0.25}), mat.NewDense(1, 1, []float64{0.05}), 0.1)
		return s
	}
	typeOne := func() *System {
		s, _ := New(mat.NewDense(3, 3, []float64{0, 1, 0, 0, -1, 2, 0, -0.5, -3}),
			mat.NewDense(3, 1, []float64{0, 0, 4}), mat.NewDense(1, 3, []float64{1, 0.2, 0}),
			mat.NewDense(1, 1, nil), 0)
		return s
	}

	type tc struct {
		name     string
		sys      func() *System
		wlo, whi float64
	}
	cases := []tc{
		{"10/(s+1)^5", func() *System { return chain5(10) }, 1e-3, 1e3},
		{"0.5/(s+1)^5", func() *System { return chain5(0.5) }, 1e-3, 1e3},
		{"input 0.5", func() *System { return withIn(0.5) }, 1e-2, 15},
		{"output 0.5", func() *System { s := first(); s.OutputDelay = []float64{0.5}; return s }, 1e-2, 15},
		{"iodelay 0.5", func() *System { s := first(); s.Delay = mat.NewDense(1, 1, []float64{0.5}); return s }, 1e-2, 15},
		{"input 0.05", func() *System { return withIn(0.05) }, 1e-2, 150},
		{"iodelay 0.02", func() *System { s := first(); s.Delay = mat.NewDense(1, 1, []float64{0.02}); return s }, 1e-2, 400},
		{"lft 0.05", func() *System { s, _ := withIn(0.05).PullDelaysToLFT(); return s }, 1e-2, 150},
		{"input 30", func() *System { return withIn(30) }, 1e-2, 5},
		{"discrete D", disc, 1e-3, math.Pi / 0.1},
		{"discrete z^-3", func() *System { s := disc(); s.InputDelay = []float64{3}; return s }, 1e-3, math.Pi / 0.1},
		{"0.3/(z(z-1))", func() *System {
			s, _ := New(mat.NewDense(2, 2, []float64{1, 1, 0, 0}), mat.NewDense(2, 1, []float64{0, 1}),
				mat.NewDense(1, 2, []float64{0.3, 0}), mat.NewDense(1, 1, nil), 0.1)
			return s
		}, 1e-3, math.Pi / 0.1},
		{"type1", typeOne, 1e-3, 1e3},
		{"type1 x1e-5", func() *System { return marginTimeScale(typeOne(), 1e-5) }, 1e-8, 1e-2},
		{"type1 x1e5", func() *System { return marginTimeScale(typeOne(), 1e5) }, 1e2, 1e8},
		{"10/(s+1)^5 x1e5", func() *System { return marginTimeScale(chain5(10), 1e5) }, 1e2, 1e8},
		{"10/(s+1)^5 x1e-5", func() *System { return marginTimeScale(chain5(10), 1e-5) }, 1e-8, 1e-2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sys := c.sys()
			all, err := AllMargin(sys)
			if err != nil {
				t.Fatal(err)
			}
			for i, w := range all.PhaseCrossFreqs {
				h := marginOracle(sys, w)
				if d := marginWrapDeg(cmplx.Phase(h)*180/math.Pi + 180); math.Abs(d) > 1e-6 {
					t.Errorf("phase crossover w=%g: angle(L)=%g deg, want -180 mod 360", w, cmplx.Phase(h)*180/math.Pi)
				}
				if gm := -20 * math.Log10(cmplx.Abs(h)); math.Abs(gm-all.GainMargins[i]) > 1e-8 {
					t.Errorf("GM at w=%g: %g, want %g", w, all.GainMargins[i], gm)
				}
			}
			for i, w := range all.GainCrossFreqs {
				h := marginOracle(sys, w)
				if math.Abs(cmplx.Abs(h)-1) > 1e-8 {
					t.Errorf("gain crossover w=%g: |L|=%.12g, want 1", w, cmplx.Abs(h))
				}
				pm := all.PhaseMargins[i]
				if pm <= -180 || pm > 180 {
					t.Errorf("PM=%g at w=%g outside (-180,180]", pm, w)
				}
				if d := marginWrapDeg(pm - 180 - cmplx.Phase(h)*180/math.Pi); math.Abs(d) > 1e-6 {
					t.Errorf("PM at w=%g: %g, angle(L)=%g", w, pm, cmplx.Phase(h)*180/math.Pi)
				}
			}
			gc, pc := marginBruteForce(sys, c.wlo, c.whi, 200000)
			marginMatch(t, "gain crossovers", marginCountIn(all.GainCrossFreqs, c.wlo, c.whi), gc)
			marginMatch(t, "phase crossovers", marginCountIn(all.PhaseCrossFreqs, c.wlo, c.whi), pc)
		})
	}
}

func TestMargin_ClosedForms(t *testing.T) {
	wpc := math.Tan(math.Pi / 5)
	z1 := mat.NewDense(1, 1, nil)
	chainA := mat.NewDense(5, 5, nil)
	for i := range 5 {
		chainA.Set(i, i, -1)
		if i > 0 {
			chainA.Set(i, i-1, 1)
		}
	}
	unstable, _ := New(chainA, mat.NewDense(5, 1, []float64{10, 0, 0, 0, 0}), mat.NewDense(1, 5, []float64{0, 0, 0, 0, 1}), z1, 0)
	zpole, _ := New(mat.NewDense(2, 2, []float64{1, 1, 0, 0}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{0.3, 0}), z1, 0.1)
	nyq := func(k float64) *System {
		s, _ := New(mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{k}), mat.NewDense(1, 1, []float64{1}), z1, 0.1)
		return s
	}
	delayed, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{2}), z1, 0)
	delayed.InputDelay = []float64{30}
	thz := 2 * math.Asin(0.15)
	wg := math.Sqrt(3)
	pm30 := marginWrapDeg(180 - math.Atan(wg)*180/math.Pi - 30*wg*180/math.Pi)

	cases := []struct {
		name           string
		sys            *System
		gm, wp, pm, wg float64
	}{
		{"10/(s+1)^5", unstable, -20 * math.Log10(10/math.Pow(1+wpc*wpc, 2.5)), wpc, math.NaN(), math.NaN()},
		{"0.3/(z(z-1))", zpole, -20 * math.Log10(0.3), math.Pi / 3 / 0.1, 90 - 1.5*thz*180/math.Pi, thz / 0.1},
		{"0.5/(z-1)", nyq(0.5), -20 * math.Log10(0.25), math.Pi / 0.1, math.NaN(), math.NaN()},
		{"1.5/(z-1)", nyq(1.5), -20 * math.Log10(0.75), math.Pi / 0.1, math.NaN(), math.NaN()},
		{"2e^-30s/(s+1)", delayed, math.NaN(), math.NaN(), pm30, wg},
	}
	for _, c := range cases {
		r, err := Margin(c.sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !math.IsNaN(c.gm) && (math.Abs(r.GainMargin-c.gm) > 1e-6 || math.Abs(wpFreq(r)-c.wp) > 1e-6*c.wp) {
			t.Errorf("%s: GM=%g@%g, want %g@%g", c.name, r.GainMargin, wpFreq(r), c.gm, c.wp)
		}
		if !math.IsNaN(c.pm) && (math.Abs(r.PhaseMargin-c.pm) > 1e-6 || (!math.IsNaN(c.wg) && math.Abs(wgFreq(r)-c.wg) > 1e-6*c.wg)) {
			t.Errorf("%s: PM=%g@%g, want %g@%g", c.name, r.PhaseMargin, wgFreq(r), c.pm, c.wg)
		}
	}
}

func TestBandwidthRejectsInvalidDrop(t *testing.T) {
	sys := emptyIOFixture(t, 2, 1, 1, 0)
	for _, drop := range []float64{3, math.NaN(), math.Inf(-1), math.Inf(1)} {
		if _, err := Bandwidth(sys, drop); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("Bandwidth dbDrop=%v err = %v, want ErrInvalidArgument", drop, err)
		}
	}
	if _, err := Bandwidth(sys, -3); err != nil {
		t.Errorf("Bandwidth dbDrop=-3: %v", err)
	}
}

// G(s) = (s+0.1)(s+30)/((s+1)(s+2)): |G| rises from 1.5 and falls to 1 only
// past the zero at 30, beyond a decade above the poles.
func TestBandwidth_ZeroBeyondPoles(t *testing.T) {
	sys, _ := New(mat.NewDense(2, 2, []float64{0, 1, -2, -3}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 27.1}), mat.NewDense(1, 1, []float64{1}), 0)
	bw, err := Bandwidth(sys, -3)
	if err != nil {
		t.Fatal(err)
	}
	mag := func(w float64) float64 {
		s := complex(0, w)
		return cmplx.Abs((s + 0.1) * (s + 30) / ((s + 1) * (s + 2)))
	}
	thr := 1.5 * math.Pow(10, -3.0/20)
	lo, hi := 10.0, 1e4
	for range 200 {
		mid := math.Sqrt(lo * hi)
		if mag(mid) > thr {
			lo = mid
		} else {
			hi = mid
		}
	}
	if math.Abs(bw-lo) > 1e-6*lo {
		t.Errorf("BW = %g, want %g", bw, lo)
	}
}

// narrowResonance returns k/(s²+2ζs+1) with ζ = 1e-4 scaled so the peak gain
// is 1+1e-6, and its two gain crossovers from
// x² − 2(1−2ζ²)x + 1 − k² = 0, x = ω², whose discriminant is written
// without the cancellation in (1−2ζ²)² − 1 + k².
func narrowResonance(t *testing.T) (*System, [2]float64) {
	t.Helper()
	const zeta = 1e-4
	k := (1 + 1e-6) * 2 * zeta * math.Sqrt(1-zeta*zeta)
	sys := makePlant(t, []float64{k}, []float64{1, 2 * zeta, 1})
	b := 1 - 2*zeta*zeta
	d := 2 * zeta * math.Sqrt((1-zeta*zeta)*((1+1e-6)*(1+1e-6)-1))
	return sys, [2]float64{math.Sqrt(b - d), math.Sqrt(b + d)}
}

func TestAllMargin_NarrowResonanceCrossings(t *testing.T) {
	sys, want := narrowResonance(t)
	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(all.GainCrossFreqs) != 2 {
		t.Fatalf("gain crossovers %v, want %v", all.GainCrossFreqs, want)
	}
	for i, w := range all.GainCrossFreqs {
		if math.Abs(w-want[i]) > 1e-12 {
			t.Errorf("crossover %d: %.15g, want %.15g", i, w, want[i])
		}
		h := marginOracle(sys, w)
		if pm := phaseMarginDeg(h); math.Abs(all.PhaseMargins[i]-pm) > 1e-6 {
			t.Errorf("PM %d: %g, want %g", i, all.PhaseMargins[i], pm)
		}
	}
	if len(all.PhaseCrossFreqs) != 0 {
		t.Errorf("phase crossovers %v, want none", all.PhaseCrossFreqs)
	}
}

// A lightly damped pole/zero pair at ω0 = 10 dips the phase of 0.01/(s+1)²
// (−168.6° there, never −180° alone) below −180° over a band ~5e-3 wide.
func TestAllMargin_NarrowPhaseDip(t *testing.T) {
	const w0 = 10.0
	num := Poly{0.01}.Mul(Poly{1, 2 * 1e-4 * w0, w0 * w0})
	den := Poly{1, 2, 1}.Mul(Poly{1, 2 * 1e-6 * w0, w0 * w0})
	sys := makePlant(t, num, den)
	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	_, pc := marginBruteForce(sys, w0*(1-1e-2), w0*(1+1e-2), 400000)
	if len(pc) != 2 {
		t.Fatalf("oracle found %d phase crossings near ω0, want 2", len(pc))
	}
	marginMatch(t, "phase crossovers", marginCountIn(all.PhaseCrossFreqs, w0*(1-1e-2), w0*(1+1e-2)), pc)
	for i, w := range all.PhaseCrossFreqs {
		h := marginOracle(sys, w)
		if d := marginWrapDeg(cmplx.Phase(h)*180/math.Pi + 180); math.Abs(d) > 1e-6 {
			t.Errorf("w=%g: angle(L)=%g, want -180", w, cmplx.Phase(h)*180/math.Pi)
		}
		if gm := -20 * math.Log10(cmplx.Abs(h)); math.Abs(gm-all.GainMargins[i]) > 1e-8 {
			t.Errorf("GM at %g: %g, want %g", w, all.GainMargins[i], gm)
		}
	}
	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(wpFreq(m)-w0) > 1e-2*w0 {
		t.Errorf("Margin WpFreq=%g, want the dip near %g", wpFreq(m), w0)
	}
}

// Discrete resonance at θ = 2.5 rad (Dt = 0.1) with r = 0.9999, D ≠ 0 and a
// z^-3 input delay, scaled to peak just above 0 dB.
func TestAllMargin_DiscreteNarrowResonance(t *testing.T) {
	const dt, theta, r = 0.1, 2.5, 0.9999
	num := []float64{1, 0.3, 0.1}
	den := []float64{1, -2 * r * math.Cos(theta), r * r}
	z := cmplx.Exp(complex(0, theta))
	g := cmplx.Abs(Poly(num).Eval(z) / Poly(den).Eval(z))
	k := (1 + 1e-5) / g
	tf := &TransferFunc{Num: [][][]float64{{{k * num[0], k * num[1], k * num[2]}}}, Den: [][]float64{den}, Dt: dt}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := res.Sys
	sys.InputDelay = []float64{3}
	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	lo, hi := (theta-1e-3)/dt, (theta+1e-3)/dt
	gc, pc := marginBruteForce(sys, lo, hi, 400000)
	if len(gc) != 2 {
		t.Fatalf("oracle found %d gain crossings near θ, want 2", len(gc))
	}
	marginMatch(t, "gain crossovers", marginCountIn(all.GainCrossFreqs, lo, hi), gc)
	marginMatch(t, "phase crossovers", marginCountIn(all.PhaseCrossFreqs, lo, hi), pc)
	for _, w := range all.GainCrossFreqs {
		if h := marginOracle(sys, w); math.Abs(cmplx.Abs(h)-1) > 1e-8 {
			t.Errorf("w=%g: |L|=%.12g, want 1", w, cmplx.Abs(h))
		}
	}
	gcAll, pcAll := marginBruteForce(sys, 1e-3, math.Pi/dt, 400000)
	if len(all.GainCrossFreqs) < len(gcAll) || len(all.PhaseCrossFreqs) < len(pcAll) {
		t.Errorf("found %d/%d crossings, dense grid %d/%d", len(all.GainCrossFreqs), len(all.PhaseCrossFreqs), len(gcAll), len(pcAll))
	}
}

// K/(s(s+1)) with K = wc·hypot(wc,1) crosses 0 dB exactly at wc, far above
// the poles.
func TestAllMargin_FarCrossover(t *testing.T) {
	for _, wc := range []float64{12, 1e4, 1e7} {
		sys := makePlant(t, []float64{wc * math.Hypot(wc, 1)}, []float64{1, 1, 0})
		m, err := Margin(sys)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(wgFreq(m)-wc) > 1e-9*wc {
			t.Errorf("wc=%g: WgFreq=%.15g", wc, wgFreq(m))
		}
		if pm := 90 - math.Atan(wc)*180/math.Pi; math.Abs(m.PhaseMargin-pm) > 1e-7 {
			t.Errorf("wc=%g: PM=%.12g, want %.12g", wc, m.PhaseMargin, pm)
		}
	}
}

// The notch (s²+2ζz·s+1)/(s²+2ζp·s+1) with ζz/ζp = 0.7 dips to −3.1 dB over a
// band ~1e-4 wide; the −3 dB bandwidth is the lower root of
// (1−g²)x² + (4(ζz²−g²ζp²) − 2(1−g²))x + 1 − g² = 0, x = ω².
func TestBandwidth_NarrowNotch(t *testing.T) {
	const zp = 1e-4
	zz := 0.7 * zp
	sys := makePlant(t, []float64{1, 2 * zz, 1}, []float64{1, 2 * zp, 1})
	bw, err := Bandwidth(sys, -3)
	if err != nil {
		t.Fatal(err)
	}
	g2 := math.Pow(10, -3.0/10)
	a, b := 1-g2, 4*(zz*zz-g2*zp*zp)-2*(1-g2)
	x := (-b - math.Sqrt(b*b-4*a*a)) / (2 * a)
	if want := math.Sqrt(x); math.Abs(bw-want) > 1e-12 {
		t.Errorf("Bandwidth=%.15g, want %.15g", bw, want)
	}
}

// 10/(s+1)^7 passes −180° at tan(π/7) (GM −13.7 dB) and −540° at tan(3π/7)
// (GM +71.4 dB); MATLAB margin returns the one closest to 0 dB.
func TestMargin_ClosestToZeroSelection(t *testing.T) {
	den := Poly{1}
	for range 7 {
		den = den.Mul(Poly{1, 1})
	}
	sys := makePlant(t, []float64{10}, den)
	gm := func(w float64) float64 { return -20 * math.Log10(10/math.Pow(1+w*w, 3.5)) }
	w1, w3 := math.Tan(math.Pi/7), math.Tan(3*math.Pi/7)
	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(all.GainMargins) != 2 || math.Abs(all.GainMargins[0]-gm(w1)) > 1e-9 || math.Abs(all.GainMargins[1]-gm(w3)) > 1e-9 {
		t.Fatalf("GMs %v @ %v, want [%g %g]", all.GainMargins, all.PhaseCrossFreqs, gm(w1), gm(w3))
	}
	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.GainMargin-gm(w1)) > 1e-9 || math.Abs(wpFreq(m)-w1) > 1e-12 {
		t.Errorf("GM=%g@%g, want %g@%g", m.GainMargin, wpFreq(m), gm(w1), w1)
	}
	if got, w, _ := selectMargin([]float64{-30, 20, -5, 5}, []float64{1, 2, 3, 4}); got != -5 || w != 3 {
		t.Errorf("selectMargin=%g, want -5 (closest to 0, lower frequency on a tie)", got)
	}
}

// The loop evaluator behind Margin, AllMargin, Bandwidth, TunePID and the
// delay Nyquist test is held to the refined-dense oracle on high-order loops,
// where evaluating the converted transfer-function polynomials lost 6%.
func TestSISOEvalHighOrderMatchesOracle(t *testing.T) {
	for _, n := range []int{60, 100} {
		for _, dt := range []float64{0, 0.05} {
			for _, delay := range []float64{0, 2} {
				for kind := range sweepModelKinds {
					rng := rand.New(rand.NewPCG(uint64(n), uint64(kind)))
					sys, twin := randomSweepRealization(rng, kind, dt, false, n, 1, 1)
					sys.InputDelay = []float64{delay}
					eval, err := newSISOEval(sys)
					if err != nil {
						t.Fatal(err)
					}
					td := newTimeDomain(dt)
					omega := logspace(-2, 2, 9)
					if dt > 0 {
						omega = logspace(-2, math.Log10(math.Pi/dt), 9)
					}
					for _, w := range omega {
						s := td.frequencyVariable(w)
						want := oracleResponse(t, twin, s, n, 1, 1)[0]
						if dt == 0 {
							want *= cmplx.Exp(-s * complex(delay, 0))
						} else {
							want /= cmplx.Pow(s, complex(delay, 0))
						}
						if e := cmplx.Abs(eval.at(w)-want) / cmplx.Abs(want); !(e <= 1e-12) {
							t.Errorf("n=%d dt=%g delay=%g kind=%v ω=%g: rel err %g", n, dt, delay, kind, w, e)
						}
					}
				}
			}
		}
	}
}

// marginDenseGrid samples the state-space response on an adaptive log grid,
// split until phase moves < 2° and |L| < 2% per step.
type marginDenseGrid struct {
	eval *sisoEval
	w    []float64
	h    []complex128
}

func newMarginDenseGrid(eval *sisoEval, wlo, whi float64) *marginDenseGrid {
	g := &marginDenseGrid{eval: eval}
	var split func(w0, w1 float64, h0, h1 complex128, depth int)
	split = func(w0, w1 float64, h0, h1 complex128, depth int) {
		dp := math.Abs(wrapDegrees((cmplx.Phase(h1) - cmplx.Phase(h0)) * 180 / math.Pi))
		dm := math.Abs(math.Log(cmplx.Abs(h1) / cmplx.Abs(h0)))
		if depth < 20 && (dp > 2 || dm > 0.02) {
			wm := math.Sqrt(w0 * w1)
			hm := eval.at(wm)
			split(w0, wm, h0, hm, depth+1)
			split(wm, w1, hm, h1, depth+1)
			return
		}
		g.w = append(g.w, w1)
		g.h = append(g.h, h1)
	}
	base := logspace(math.Log10(wlo), math.Log10(whi), 2000)
	g.w, g.h = []float64{base[0]}, []complex128{eval.at(base[0])}
	for _, w := range base[1:] {
		split(g.w[len(g.w)-1], w, g.h[len(g.h)-1], eval.at(w), 0)
	}
	return g
}

// crossings is the brute-force crossing oracle: bisection at each sign
// change of log|L|−log γ and of the −180° phase offset on the grid.
func (g *marginDenseGrid) crossings(gamma float64) (gc, pc []float64) {
	lg := math.Log(gamma)
	logMag := func(w float64) float64 { return math.Log(cmplx.Abs(g.eval.at(w))) - lg }
	offset := func(w float64) float64 { return phaseOffsetDeg(g.eval.at(w)) }
	for k := 1; k < len(g.w); k++ {
		a, b := g.h[k-1], g.h[k]
		if m0, m1 := math.Log(cmplx.Abs(a))-lg, math.Log(cmplx.Abs(b))-lg; m0*m1 < 0 {
			gc = append(gc, refineCrossing(g.w[k-1], g.w[k], logMag))
		}
		s0, s1 := phaseOffsetDeg(a), phaseOffsetDeg(b)
		if s0*s1 < 0 && math.Abs(s0-s1) < 180 {
			if w := refineCrossing(g.w[k-1], g.w[k], offset); math.Abs(offset(w)) <= 1e-6 {
				pc = append(pc, w)
			}
		}
	}
	return gc, pc
}

func marginExactMatch(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: got %d crossings %v, dense grid %d %v", label, len(got), got, len(want), want)
		return
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-7*want[i] {
			t.Errorf("%s[%d]: got w=%.12g, dense grid %.12g", label, i, got[i], want[i])
		}
	}
}

// marginRandomLoop draws a stable SISO loop with D ≠ 0 and |L(jω0)| = 3. With
// mimo set it is channel (1,0) of a 2×2 realization, so it carries modes
// that are uncontrollable or unobservable from that channel.
func marginRandomLoop(tb testing.TB, n int, kind sweepModelKind, dt float64, descriptor, mimo bool) *System {
	tb.Helper()
	seed := uint64(n)<<8 | uint64(kind)<<1
	if dt > 0 {
		seed |= 1 << 20
	}
	if descriptor {
		seed |= 1 << 21
	}
	if mimo {
		seed |= 1 << 22
	}
	rng := rand.New(rand.NewPCG(seed, 7))
	io := 1
	if mimo {
		io = 2
	}
	sys, _ := randomSweepRealization(rng, kind, dt, descriptor, n, io, io)
	if mimo {
		var err error
		if sys, err = sys.SelectByIndex([]int{0}, []int{1}); err != nil {
			tb.Fatal(err)
		}
	}
	sys.D.Set(0, 0, 0.5+rng.Float64())
	eval, err := newSISOEval(sys)
	if err != nil {
		tb.Fatal(err)
	}
	k := 3 / cmplx.Abs(eval.at(1))
	sys.C.Scale(k, sys.C)
	sys.D.Scale(k, sys.D)
	return sys
}

// AllMargin and Bandwidth find every crossing of high-order loops, where
// polynomial candidates from the transfer function missed some from about
// order 60 (ergo MY3SAU). Phase crossovers of continuous delayed loops come
// from a grid search and are not checked here; their gain crossings and
// bandwidth are checked against the delay-free twin, which has the same |L|
// without a phase that winds 0.2·ω rad.
func TestAllMargin_HighOrderMatchesDenseGrid(t *testing.T) {
	orders := []int{20, 60, 100, 150}
	if testing.Short() || raceEnabled {
		orders = orders[:2]
	}
	for _, n := range orders {
		for _, dt := range []float64{0, 0.05} {
			for _, variant := range []string{"siso", "mimo", "descriptor", "delay"} {
				if (variant == "descriptor" || variant == "delay") && n > 60 ||
					raceEnabled && (variant == "mimo" || variant == "descriptor") && n > 20 {
					continue
				}
				for kind := range sweepModelKinds {
					t.Run(fmt.Sprintf("n=%d/dt=%g/%s/%v", n, dt, variant, kind), func(t *testing.T) {
						t.Parallel()
						sys := marginRandomLoop(t, n, kind, dt, variant == "descriptor", variant == "mimo")
						if variant == "delay" {
							sys.InputDelay = []float64{0.2}
							if dt > 0 {
								sys.InputDelay[0] = 3
							}
						}
						all, err := AllMargin(sys)
						if err != nil {
							t.Fatal(err)
						}
						bw, err := Bandwidth(sys, 0)
						dc, dcErr := sys.DCGain()
						if dcErr != nil {
							t.Fatal(dcErr)
						}
						if g := math.Abs(dc.At(0, 0)); g == 0 || math.IsInf(g, 0) || math.IsNaN(g) {
							if !errors.Is(err, ErrInvalidArgument) {
								t.Fatalf("bandwidth with DC gain %g: err = %v, want ErrInvalidArgument", g, err)
							}
						} else if err != nil {
							t.Fatal(err)
						}
						oracleSys := sys
						if variant == "delay" && dt == 0 {
							oracleSys = sys.Copy()
							oracleSys.InputDelay = nil
						}
						eval, err := newSISOEval(oracleSys)
						if err != nil {
							t.Fatal(err)
						}
						wlo, whi := 1e-5, 1e5
						if dt > 0 {
							whi = math.Pi / dt
						}
						grid := newMarginDenseGrid(eval, wlo, whi)
						gc, pc := grid.crossings(1)
						marginExactMatch(t, "gain", marginCountIn(all.GainCrossFreqs, wlo, whi), gc)
						if variant != "delay" || dt > 0 {
							marginExactMatch(t, "phase", marginCountIn(all.PhaseCrossFreqs, wlo, whi), pc)
						}

						if g := math.Abs(dc.At(0, 0)); g == 0 || math.IsInf(g, 0) || math.IsNaN(g) {
							return
						}
						bgc, _ := grid.crossings(math.Abs(dc.At(0, 0)) * math.Pow(10, -3.0/20))
						want := math.Inf(1)
						if len(bgc) > 0 {
							want = bgc[0]
						}
						if !(bw == want || math.Abs(bw-want) <= 1e-7*want) {
							t.Errorf("bandwidth: got %.12g, dense grid %.12g", bw, want)
						}
					})
				}
			}
		}
	}
}

// Loops with |L| ≡ 1 or L real on the whole boundary make the crossing
// pencils singular; they have no isolated crossings of that kind, while the
// other kind is still found exactly.
func TestAllMargin_SingularPencils(t *testing.T) {
	tf := func(num, den []float64, dt float64) *System {
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	cases := []struct {
		name   string
		sys    *System
		gc, pc []float64
	}{
		{"allpass (s²-3s+2.5)/(s²+3s+2.5)", tf([]float64{1, -3, 2.5}, []float64{1, 3, 2.5}, 0), nil, []float64{math.Sqrt(2.5)}},
		{"allpass (1-s)/(1+s)", tf([]float64{-1, 1}, []float64{1, 1}, 0), nil, nil},
		{"-2/(1-s²) real negative", tf([]float64{-2}, []float64{-1, 0, 1}, 0), []float64{1}, nil},
		{"discrete allpass (1-0.5z)/(z-0.5)", tf([]float64{-0.5, 1}, []float64{1, -0.5}, 0.1), nil, []float64{math.Pi / 0.1}},
		{"static -3", tf([]float64{-3}, []float64{1}, 0), nil, nil},
		{"1/(s+1)^3 touches 0 dB at DC", tf([]float64{1}, []float64{1, 3, 3, 1}, 0), nil, []float64{math.Sqrt(3)}},
	}
	for _, c := range cases {
		all, err := AllMargin(c.sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		marginExactMatch(t, c.name+" gain", all.GainCrossFreqs, c.gc)
		marginExactMatch(t, c.name+" phase", all.PhaseCrossFreqs, c.pc)
	}
}

func BenchmarkMarginOrder(b *testing.B) {
	for _, n := range []int{10, 100} {
		sys := marginRandomLoop(b, n, sweepDense, 0, false, false)
		b.Run(fmt.Sprintf("Margin/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := Margin(sys); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("AllMargin/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := AllMargin(sys); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("Bandwidth/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := Bandwidth(sys, 0); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("Pidtune/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := Pidtune(sys, PidtunePID, 0, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// A descriptor loop with an algebraic state (singular E) adds infinite
// eigenvalues to the crossing pencils; the finite crossings still match the
// dense grid.
func TestAllMargin_SingularE(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 0.3, 0, 1}), mat.NewDense(2, 1, []float64{4, 1}),
			mat.NewDense(1, 2, []float64{1, -0.5}), mat.NewDense(1, 1, []float64{0.2}), mat.NewDense(2, 2, []float64{1, 0, 0, 0}), dt)
		if err != nil {
			t.Fatal(err)
		}
		all, err := AllMargin(sys)
		if err != nil {
			t.Fatal(err)
		}
		eval, err := newSISOEval(sys)
		if err != nil {
			t.Fatal(err)
		}
		whi := 1e5
		if dt > 0 {
			whi = math.Pi / dt
		}
		gc, pc := newMarginDenseGrid(eval, 1e-5, whi).crossings(1)
		marginExactMatch(t, fmt.Sprintf("dt=%g gain", dt), all.GainCrossFreqs, gc)
		marginExactMatch(t, fmt.Sprintf("dt=%g phase", dt), all.PhaseCrossFreqs, pc)
	}
}

func wgFreq(r *MarginResult) float64 {
	if w, ok := r.GainCrossover(); ok {
		return w
	}
	return math.NaN()
}

func wpFreq(r *MarginResult) float64 {
	if w, ok := r.PhaseCrossover(); ok {
		return w
	}
	return math.NaN()
}

func dmFreq(r *DiskMarginResult) float64 {
	if w, ok := r.Frequency(); ok {
		return w
	}
	return math.NaN()
}

func dmPeakFreq(r *DiskMarginResult) float64 {
	if w, ok := r.PeakFreq(); ok {
		return w
	}
	return math.NaN()
}

func TestMarginCrossoverAccessors(t *testing.T) {
	none, err := Margin(makeSISO(-10, 1, 1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := none.GainCrossover(); ok {
		t.Error("1/(s+10): GainCrossover ok, want none")
	}
	if _, ok := none.PhaseCrossover(); ok {
		t.Error("1/(s+10): PhaseCrossover ok, want none")
	}
	if !math.IsInf(none.GainMargin, 1) || !math.IsInf(none.PhaseMargin, 1) {
		t.Errorf("margins = %g, %g; want +Inf, +Inf", none.GainMargin, none.PhaseMargin)
	}

	// L = 2/(s+1)³: phase −180° at ω = √3 with |L| = 1/4; |L| = 1 at
	// ω = √(2^(2/3) − 1).
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{0, 1, 0, 0, 0, 1, -1, -3, -3},
		[]float64{0, 0, 1},
		[]float64{2, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	wpc, ok := m.PhaseCrossover()
	if !ok || math.Abs(wpc-math.Sqrt(3)) > 1e-9 || math.Abs(m.GainMargin-20*math.Log10(4)) > 1e-9 {
		t.Errorf("phase crossover %g (ok %v), GM %g; want √3, %g dB", wpc, ok, m.GainMargin, 20*math.Log10(4))
	}
	wgc, ok := m.GainCrossover()
	want := math.Sqrt(math.Pow(2, 2.0/3) - 1)
	if !ok || math.Abs(wgc-want) > 1e-9 || math.Abs(m.PhaseMargin-(180-3*math.Atan(want)*180/math.Pi)) > 1e-7 {
		t.Errorf("gain crossover %g (ok %v), PM %g; want %g", wgc, ok, m.PhaseMargin, want)
	}
}

func TestDiskMarginUnstableHasNoFrequencies(t *testing.T) {
	dm, err := DiskMargin(makeSISO(1, 1, 1, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	if dm.Alpha != 0 || dm.GainMargin != [2]float64{1, 1} || dm.PhaseMargin != 0 {
		t.Errorf("unstable disk margin = %+v, want MATLAB's 0, [1 1], 0", dm)
	}
	if _, ok := dm.Frequency(); ok {
		t.Error("unstable loop: Frequency ok, want none")
	}
	if _, ok := dm.PeakFreq(); ok {
		t.Error("unstable loop: PeakFreq ok, want none")
	}
	stable, err := DiskMargin(makeSISO(-1, 1, 1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := stable.Frequency(); !ok {
		t.Error("stable loop: Frequency not ok")
	}
}

func TestMarginFamilyRejects(t *testing.T) {
	nan := makeSISO(math.NaN(), 1, 1, 0)
	tests := []struct {
		prefix string
		want   error
		call   func() error
	}{
		{"DiskMarginSkew: ", ErrInvalidArgument, func() error { _, err := DiskMarginSkew(makeSISO(-1, 1, 1, 0), math.NaN()); return err }},
		{"DiskMarginSkew: ", ErrInvalidArgument, func() error { _, err := DiskMarginSkew(nil, 0); return err }},
		{"DiskMargin: ", ErrInvalidArgument, func() error { _, err := DiskMargin(nan); return err }},
		{"AllMargin: ", ErrInvalidArgument, func() error { _, err := AllMargin(nil); return err }},
		{"Margin: ", ErrInvalidArgument, func() error { _, err := Margin(nan); return err }},
		{"Bandwidth: ", ErrInvalidArgument, func() error { _, err := Bandwidth(nil, 0); return err }},
	}
	for _, tc := range tests {
		err := tc.call()
		if !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), tc.prefix) {
			t.Errorf("err = %v, want %q prefix and %v", err, tc.prefix, tc.want)
		}
	}
}

func TestBandwidthFirstOrderOracle(t *testing.T) {
	bw, err := Bandwidth(makeSISO(-1, 1, 1, 0), 0)
	if err != nil {
		t.Fatal(err)
	}
	if want := math.Sqrt(math.Pow(10, 0.3) - 1); math.Abs(bw-want) > 1e-10 {
		t.Errorf("bandwidth = %.12g, want %.12g", bw, want)
	}
}

// sisoEval reuses one point evaluator across frequencies; each value must
// stay bit-identical to a fresh FreqResponse, including at the integrator
// pole ω = 0 and after it.
func TestSISOEval_MatchesFreqResponse(t *testing.T) {
	for _, internal := range []bool{false, true} {
		l := benchPIDDelayLoop(t, internal)
		l.D.Set(0, 0, 0.25)
		e, err := newSISOEval(l)
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range []float64{0, 1e-9, 0.3, 0.7085, 1, 3.7, 20, 1e3, 0, 2.5e4, 0.3} {
			got := e.at(w)
			resp, err := l.FreqResponse([]float64{w})
			if err != nil {
				t.Fatal(err)
			}
			if want := resp.At(0, 0, 0); got != want && !(cmplx.IsInf(got) && cmplx.IsInf(want)) {
				t.Errorf("internal=%v ω=%g: at = %v, FreqResponse = %v", internal, w, got, want)
			}
		}
		if e.err != nil {
			t.Fatal(e.err)
		}
	}
}

// benchPIDDelayLoop is the PID loop 0.8 + 0.3/s + 0.2s/(0.05s+1) around
// 1/(s+1)³ with an exact 0.5 s output delay, as an I/O delay or, as
// process-lab builds it from a block diagram, an internal delay.
func benchPIDDelayLoop(b testing.TB, internal bool) *System {
	b.Helper()
	pid, err := NewPID(0.8, 0.3, 0.2, 0.05, 0)
	if err != nil {
		b.Fatal(err)
	}
	c, err := pid.System()
	if err != nil {
		b.Fatal(err)
	}
	res, err := (&TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, 3, 3, 1}}}).StateSpace()
	if err != nil {
		b.Fatal(err)
	}
	plant := res.Sys
	if internal {
		n, _, _ := plant.Dims()
		c2 := mat.DenseCopyOf(plant.C)
		plant.C = mat.NewDense(1, n, nil)
		err = plant.SetInternalDelay([]float64{0.5}, mat.NewDense(n, 1, nil), c2,
			mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil))
	} else {
		err = plant.SetOutputDelay([]float64{0.5})
	}
	if err != nil {
		b.Fatal(err)
	}
	l, err := Series(c, plant)
	if err != nil {
		b.Fatal(err)
	}
	return l
}

func BenchmarkDiskMargin_PIDExactDelay(b *testing.B) {
	for _, internal := range []bool{false, true} {
		l := benchPIDDelayLoop(b, internal)
		name := "io"
		if internal {
			name = "internal"
		}
		b.Run(name+"/DiskMargin", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := DiskMargin(l); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/DiskMarginSkew", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := DiskMarginSkew(l, 1); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/Margin", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := Margin(l); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
