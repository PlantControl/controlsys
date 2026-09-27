package controlsys

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestTunableRealSetSampleAndFixedBehavior(t *testing.T) {
	k, err := NewTunableReal("K", 2, TunableBounds{Lower: 1, Upper: 4})
	if err != nil {
		t.Fatalf("NewTunableReal: %v", err)
	}
	if k.Name() != "K" || k.Value() != 2 || k.Bounds().Lower != 1 || k.Bounds().Upper != 4 {
		t.Fatalf("unexpected parameter state: %#v", k)
	}
	if err := k.SetValue(5); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("SetValue outside bounds err = %v, want ErrDimensionMismatch", err)
	}
	if err := k.SetValue(3); err != nil {
		t.Fatalf("SetValue: %v", err)
	}
	sample := map[string]float64{"K": 1.5}
	sampled, err := k.Sample(sample)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	if sampled.Value() != 1.5 || k.Value() != 3 {
		t.Fatalf("sampled/current values = %g/%g, want 1.5/3", sampled.Value(), k.Value())
	}
	k.SetFixed(true)
	randomized, err := k.RandomSample(rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("RandomSample fixed: %v", err)
	}
	if randomized.Value() != k.Value() {
		t.Fatalf("fixed random value = %g, want %g", randomized.Value(), k.Value())
	}
}

func TestTunableRealRejectsInvalidBounds(t *testing.T) {
	if _, err := NewTunableReal("bad", 0, TunableBounds{Lower: 2, Upper: 1}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("invalid bounds err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := NewTunableReal("", 0, TunableBounds{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("empty name err = %v, want ErrDimensionMismatch", err)
	}
}

func TestTunableGainPIDTFAndSSCurrentValues(t *testing.T) {
	k, _ := NewTunableReal("K", 2, TunableBounds{Lower: 0, Upper: 10})
	gain := NewTunableGain("gain", [][]*TunableReal{{k}}, 0)
	gainSys, err := gain.CurrentSystem()
	if err != nil {
		t.Fatalf("gain CurrentSystem: %v", err)
	}
	if gainSys.D.At(0, 0) != 2 {
		t.Fatalf("gain D = %g, want 2", gainSys.D.At(0, 0))
	}

	kp, _ := NewTunableReal("Kp", 1.2, TunableBounds{Lower: 0, Upper: 5})
	ki, _ := NewTunableReal("Ki", 0.3, TunableBounds{Lower: 0, Upper: 5})
	kd, _ := NewTunableReal("Kd", 0.4, TunableBounds{Lower: 0, Upper: 5})
	pid := NewTunablePID("pid", kp, ki, kd, 0.1, 0)
	pidSys, err := pid.CurrentSystem()
	if err != nil {
		t.Fatalf("pid CurrentSystem: %v", err)
	}
	if _, m, p := pidSys.Dims(); m != 1 || p != 1 {
		t.Fatalf("PID dims = (_, %d, %d), want (_, 1, 1)", m, p)
	}

	numGain, _ := NewTunableReal("num", 3, TunableBounds{Lower: 1, Upper: 5})
	tf := NewTunableTF("tf", [][][]*TunableReal{{{numGain}}}, [][]float64{{1, 2}}, 0)
	tfSys, err := tf.CurrentSystem()
	if err != nil {
		t.Fatalf("tf CurrentSystem: %v", err)
	}
	resp, err := tfSys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := real(resp[0][0]), 1.5; math.Abs(got-want) > 1e-12 {
		t.Fatalf("tf dc = %g, want %g", got, want)
	}

	a11, _ := NewTunableReal("a11", -1, TunableBounds{Lower: -5, Upper: -0.1})
	ss := NewTunableSS(
		"ss",
		[][]*TunableReal{{a11, fixedReal(t, "a12", 0.5)}, {fixedReal(t, "a21", -2), fixedReal(t, "a22", -3)}},
		[][]*TunableReal{{fixedReal(t, "b11", 1)}, {fixedReal(t, "b21", -1)}},
		[][]*TunableReal{{fixedReal(t, "c11", 2), fixedReal(t, "c12", -0.5)}},
		[][]*TunableReal{{fixedReal(t, "d11", 0.25)}},
		0,
	)
	ssSys, err := ss.CurrentSystem()
	if err != nil {
		t.Fatalf("ss CurrentSystem: %v", err)
	}
	if ssSys.A.At(0, 1) != 0.5 || ssSys.A.At(1, 0) != -2 {
		t.Fatalf("non-symmetric A not preserved: %v", mat.Formatted(ssSys.A))
	}
}

func TestTunableBlockDeterministicAndRandomSampling(t *testing.T) {
	k, _ := NewTunableReal("K", 2, TunableBounds{Lower: 1, Upper: 4})
	block := NewTunableGain("gain", [][]*TunableReal{{k}}, 0)

	sampled, err := block.Sample(map[string]float64{"K": 3.5})
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	sys, err := sampled.CurrentSystem()
	if err != nil {
		t.Fatal(err)
	}
	if sys.D.At(0, 0) != 3.5 || k.Value() != 2 {
		t.Fatalf("sampled/current gain = %g/%g, want 3.5/2", sys.D.At(0, 0), k.Value())
	}

	randomized, err := block.RandomSample(rand.New(rand.NewSource(4)))
	if err != nil {
		t.Fatalf("RandomSample: %v", err)
	}
	randSys, err := randomized.CurrentSystem()
	if err != nil {
		t.Fatal(err)
	}
	got := randSys.D.At(0, 0)
	if got < 1 || got > 4 {
		t.Fatalf("random gain = %g outside [1,4]", got)
	}
}

func fixedReal(t *testing.T, name string, value float64) *TunableReal {
	t.Helper()
	p, err := NewTunableReal(name, value, TunableBounds{})
	if err != nil {
		t.Fatal(err)
	}
	p.SetFixed(true)
	return p
}
