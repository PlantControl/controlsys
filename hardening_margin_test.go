package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestMargin_ThirdOrderContinuous(t *testing.T) {
	// G(s) = 1/(s+1)^3: DC gain = 1, so no gain crossover.
	// GM = 18.06 dB at w_pc = sqrt(3), PM = +Inf (no gain crossover).
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1, -3, -3,
		},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGmDB := 20 * math.Log10(8)
	wantWpc := math.Sqrt(3)

	if math.Abs(r.GainMargin-wantGmDB)/wantGmDB > 0.05 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGmDB)
	}
	if math.Abs(wpFreq(r)-wantWpc)/wantWpc > 0.05 {
		t.Errorf("WpFreq = %v, want ~%v", wpFreq(r), wantWpc)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf (no gain crossover)", r.PhaseMargin)
	}
}

func TestMargin_ThirdOrderWithGain(t *testing.T) {
	// G(s) = 2/(s+1)^3: DC gain = 2 > 1, so gain crossover exists.
	// GM = 12.04 dB at w_pc = sqrt(3), PM ≈ 67.6° at wgc ≈ 0.766
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1, -3, -3,
		},
		[]float64{0, 0, 1},
		[]float64{2, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGmDB := 20 * math.Log10(4)
	wantWpc := math.Sqrt(3)
	wantPm := 67.6
	wantWgc := 0.766

	if math.Abs(r.GainMargin-wantGmDB)/wantGmDB > 0.05 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGmDB)
	}
	if math.Abs(wpFreq(r)-wantWpc)/wantWpc > 0.05 {
		t.Errorf("WpFreq = %v, want ~%v", wpFreq(r), wantWpc)
	}
	if math.Abs(r.PhaseMargin-wantPm) > 2 {
		t.Errorf("PM = %v deg, want ~%v deg", r.PhaseMargin, wantPm)
	}
	if math.Abs(wgFreq(r)-wantWgc)/wantWgc > 0.05 {
		t.Errorf("WgFreq = %v, want ~%v", wgFreq(r), wantWgc)
	}
}

func TestMargin_IntegratorSystem(t *testing.T) {
	// G(s) = 1/(s(s+1)): phase goes from -90 to -180, never crosses -180.
	// GM = +Inf, PM ≈ 52° at wgc ≈ 0.786
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{
			0, 1,
			0, -1,
		},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
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
		t.Errorf("GM = %v, want +Inf (phase never crosses -180)", r.GainMargin)
	}

	wantPm := 52.0
	if math.Abs(r.PhaseMargin-wantPm) > 3 {
		t.Errorf("PM = %v deg, want ~%v deg", r.PhaseMargin, wantPm)
	}
	if wgFreq(r) < 0.5 || wgFreq(r) > 1.0 {
		t.Errorf("WgFreq = %v, want ~0.786", wgFreq(r))
	}
}

func TestMargin_DiscreteTime(t *testing.T) {
	// G(s) = 10/((s+1)(s+10)) discretized at dt=0.1
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{
			0, 1,
			-10, -11,
		},
		[]float64{0, 10},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	if r.GainMargin <= 0 {
		t.Errorf("GM = %v dB, want positive (stable system)", r.GainMargin)
	}
	if r.PhaseMargin <= 0 {
		t.Errorf("PM = %v deg, want positive (stable system)", r.PhaseMargin)
	}
}

func TestBode_PhaseUnwrapping_ThirdOrder(t *testing.T) {
	// G(s) = 1/(s+1)^3: phase at high freq approaches -270°.
	// With multiple Bode points, unwrapping should produce continuous phase.
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1, -3, -3,
		},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := make([]float64, 200)
	for i := range omega {
		omega[i] = math.Pow(10, -2+4*float64(i)/float64(len(omega)-1))
	}

	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	for k := 1; k < len(omega); k++ {
		diff := math.Abs(bode.PhaseAt(k, 0, 0) - bode.PhaseAt(k-1, 0, 0))
		if diff > 180 {
			t.Errorf("phase jump at w=%v: %v -> %v (diff=%v)",
				omega[k], bode.PhaseAt(k-1, 0, 0), bode.PhaseAt(k, 0, 0), diff)
			break
		}
	}

	lastPhase := bode.PhaseAt(len(omega)-1, 0, 0)
	if lastPhase > -250 {
		t.Errorf("final phase = %v deg, want approaching -270 for 3rd order", lastPhase)
	}
}

func TestBode_PhaseUnwrapping_FourthOrder(t *testing.T) {
	// G(s) = 24/((s+1)(s+2)(s+3)(s+4)): phase goes from 0 to -360°.
	sys, err := NewFromSlices(4, 1, 1,
		[]float64{
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			-24, -50, -35, -10,
		},
		[]float64{0, 0, 0, 24},
		[]float64{1, 0, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := make([]float64, 200)
	for i := range omega {
		omega[i] = math.Pow(10, -1+3*float64(i)/float64(len(omega)-1))
	}

	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	for k := 1; k < len(omega); k++ {
		diff := math.Abs(bode.PhaseAt(k, 0, 0) - bode.PhaseAt(k-1, 0, 0))
		if diff > 180 {
			t.Errorf("phase jump at w=%v: %v -> %v (diff=%v)",
				omega[k], bode.PhaseAt(k-1, 0, 0), bode.PhaseAt(k, 0, 0), diff)
			break
		}
	}

	lastPhase := bode.PhaseAt(len(omega)-1, 0, 0)
	if lastPhase > -300 {
		t.Errorf("final phase = %v deg, want approaching -360 for 4th order", lastPhase)
	}
}

func TestMargin_StableHighGain(t *testing.T) {
	// G(s) = 100/(s^3 + 8s^2 + 17s + 10) = 100/((s+1)(s+2)(s+5))
	// DC gain = 100/10 = 10. Stable, with both gain and phase crossovers.
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-10, -17, -8,
		},
		[]float64{0, 0, 1},
		[]float64{100, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if r.GainMargin <= 0 {
		t.Errorf("GM = %v dB, want positive (stable system)", r.GainMargin)
	}
	if r.PhaseMargin <= 0 || r.PhaseMargin > 90 {
		t.Errorf("PM = %v deg, want small positive", r.PhaseMargin)
	}
	if math.IsNaN(wgFreq(r)) || math.IsNaN(wpFreq(r)) {
		t.Errorf("crossover freqs should be finite: WgFreq=%v, WpFreq=%v", wgFreq(r), wpFreq(r))
	}
}

func TestBode_MIMOSystem(t *testing.T) {
	// 2x2 diagonal: G11=1/(s+1), G22=1/(s+2), off-diagonal=0
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	mag00 := bode.MagDBAt(0, 0, 0)
	wantMag00 := 20 * math.Log10(1.0/math.Sqrt(2))
	if math.Abs(mag00-wantMag00) > 0.1 {
		t.Errorf("(0,0) mag = %v dB, want ~%v dB", mag00, wantMag00)
	}

	mag11 := bode.MagDBAt(0, 1, 1)
	wantMag11 := 20 * math.Log10(1.0/math.Sqrt(5))
	if math.Abs(mag11-wantMag11) > 0.1 {
		t.Errorf("(1,1) mag = %v dB, want ~%v dB", mag11, wantMag11)
	}

	mag01 := bode.MagDBAt(0, 0, 1)
	if mag01 > -60 {
		t.Errorf("(0,1) off-diagonal mag = %v dB, want << 0 (decoupled)", mag01)
	}
}

func TestMargin_PureGain(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf for static gain < 1", r.GainMargin)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf for static gain < 1", r.PhaseMargin)
	}
}

func TestMargin_PureGainAboveUnity(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{2.0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf (no phase crossover for pure gain)", r.GainMargin)
	}
}

func TestAllMargin_ThirdOrderContinuous(t *testing.T) {
	// G(s) = 2/(s+1)^3: has both gain and phase crossovers.
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1, -3, -3,
		},
		[]float64{0, 0, 1},
		[]float64{2, 0, 0},
		[]float64{0},
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

	wantGm := 20 * math.Log10(4)
	if math.Abs(all.GainMargins[0]-wantGm)/wantGm > 0.05 {
		t.Errorf("GainMargin[0] = %v dB, want ~%v dB", all.GainMargins[0], wantGm)
	}
	if math.Abs(all.PhaseMargins[0]-67.6) > 2 {
		t.Errorf("PhaseMargin[0] = %v deg, want ~67.6 deg", all.PhaseMargins[0])
	}
}

func TestAllMargin_IntegratorSystem(t *testing.T) {
	// G(s) = 1/(s(s+1)): no phase crossover, one gain crossover.
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{
			0, 1,
			0, -1,
		},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.PhaseCrossFreqs) != 0 {
		t.Errorf("got %d phase crossovers, want 0 (phase never reaches -180)", len(all.PhaseCrossFreqs))
	}
	if len(all.GainCrossFreqs) != 1 {
		t.Fatalf("got %d gain crossovers, want 1", len(all.GainCrossFreqs))
	}
	if all.PhaseMargins[0] < 40 || all.PhaseMargins[0] > 60 {
		t.Errorf("PhaseMargin[0] = %v deg, want ~52", all.PhaseMargins[0])
	}
}
