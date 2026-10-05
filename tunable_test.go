package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestTunableRealSetSampleAndFixedBehavior(t *testing.T) {
	k, err := newBoundedReal("K", 2, 1, 4)
	if err != nil {
		t.Fatalf("NewTunableReal: %v", err)
	}
	if k.Name() != "K" || k.Value() != 2 || k.Bounds().Lower != 1 || k.Bounds().Upper != 4 {
		t.Fatalf("unexpected parameter state: %#v", k)
	}
	if err := k.SetValue(5); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("SetValue outside bounds err = %v, want ErrInvalidArgument", err)
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
	randomized, err := k.RandomSample(rand.New(rand.NewPCG(1, 0)))
	if err != nil {
		t.Fatalf("RandomSample fixed: %v", err)
	}
	if randomized.Value() != k.Value() {
		t.Fatalf("fixed random value = %g, want %g", randomized.Value(), k.Value())
	}
}

func TestTunableRealRejectsInvalidBounds(t *testing.T) {
	if _, err := newBoundedReal("bad", 0, 2, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid bounds err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewTunableReal("", 0); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty name err = %v, want ErrInvalidArgument", err)
	}
}

func TestTunableGainPIDTFAndSSCurrentValues(t *testing.T) {
	k, _ := newBoundedReal("K", 2, 0, 10)
	gain := mustOK(NewTunableGain("gain", [][]*TunableReal{{k}}, 0))
	gainSys, err := gain.CurrentSystem()
	if err != nil {
		t.Fatalf("gain CurrentSystem: %v", err)
	}
	if gainSys.D.At(0, 0) != 2 {
		t.Fatalf("gain D = %g, want 2", gainSys.D.At(0, 0))
	}

	kp, _ := newBoundedReal("Kp", 1.2, 0, 5)
	ki, _ := newBoundedReal("Ki", 0.3, 0, 5)
	kd, _ := newBoundedReal("Kd", 0.4, 0, 5)
	pid := mustOK(NewTunablePID("pid", kp, ki, kd, 0.1, 0))
	pidSys, err := pid.CurrentSystem()
	if err != nil {
		t.Fatalf("pid CurrentSystem: %v", err)
	}
	if _, m, p := pidSys.Dims(); m != 1 || p != 1 {
		t.Fatalf("PID dims = (_, %d, %d), want (_, 1, 1)", m, p)
	}

	numGain, _ := newBoundedReal("num", 3, 1, 5)
	tf := mustOK(NewTunableTF("tf", [][][]*TunableReal{{{numGain}}}, [][]float64{{1, 2}}, 0))
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

	a11, _ := newBoundedReal("a11", -1, -5, -0.1)
	ss := mustOK(NewTunableSS(
		"ss",
		[][]*TunableReal{{a11, fixedReal(t, "a12", 0.5)}, {fixedReal(t, "a21", -2), fixedReal(t, "a22", -3)}},
		[][]*TunableReal{{fixedReal(t, "b11", 1)}, {fixedReal(t, "b21", -1)}},
		[][]*TunableReal{{fixedReal(t, "c11", 2), fixedReal(t, "c12", -0.5)}},
		[][]*TunableReal{{fixedReal(t, "d11", 0.25)}},
		0,
	))
	ssSys, err := ss.CurrentSystem()
	if err != nil {
		t.Fatalf("ss CurrentSystem: %v", err)
	}
	if ssSys.A.At(0, 1) != 0.5 || ssSys.A.At(1, 0) != -2 {
		t.Fatalf("non-symmetric A not preserved: %v", mat.Formatted(ssSys.A))
	}
}

func TestTunableBlockDeterministicAndRandomSampling(t *testing.T) {
	k, _ := newBoundedReal("K", 2, 1, 4)
	block := mustOK(NewTunableGain("gain", [][]*TunableReal{{k}}, 0))

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

	randomized, err := block.RandomSample(rand.New(rand.NewPCG(4, 0)))
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
	p, err := NewTunableReal(name, value)
	if err != nil {
		t.Fatal(err)
	}
	p.SetFixed(true)
	return p
}

func newBoundedReal(name string, value, lower, upper float64) (*TunableReal, error) {
	p, err := NewTunableReal(name, value)
	if err != nil {
		return nil, err
	}
	if err := p.SetBounds(lower, upper); err != nil {
		return nil, err
	}
	return p, nil
}

func mustOK[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func TestTunableRealBoundsAndValidation(t *testing.T) {
	p, err := NewTunableReal("k", 5)
	if err != nil {
		t.Fatal(err)
	}
	if b := p.Bounds(); !math.IsInf(b.Lower, -1) || !math.IsInf(b.Upper, 1) {
		t.Errorf("default bounds = %v, want (-Inf, Inf)", b)
	}
	if err := p.SetBounds(0, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("SetBounds(0,0) with value 5: err = %v, want ErrInvalidArgument", err)
	}
	z, err := NewTunableReal("z", 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := z.SetBounds(0, 0); err != nil {
		t.Errorf("SetBounds(0,0) with value 0: %v", err)
	}
	if err := z.SetValue(1); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("value outside [0,0]: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewTunableReal("k", math.NaN()); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NaN value: err = %v, want ErrInvalidArgument", err)
	}
	if err := p.SetBounds(math.NaN(), 10); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NaN bound: err = %v, want ErrInvalidArgument", err)
	}
	if err := p.SetValue(math.Inf(1)); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Inf SetValue: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := p.RandomSample(rand.New(rand.NewPCG(1, 0))); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("RandomSample unbounded: err = %v, want ErrInvalidArgument", err)
	}
}

func TestTunableRandomSampleUsesOneStream(t *testing.T) {
	a := mustOK(newBoundedReal("a", 0, -1, 1))
	b := mustOK(newBoundedReal("b", 0, -1, 1))
	gain := mustOK(NewTunableGain("g", [][]*TunableReal{{a, b}}, 0))
	if _, err := gain.RandomSample(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil rng: err = %v, want ErrInvalidArgument", err)
	}
	s, err := gain.RandomSample(rand.New(rand.NewPCG(7, 0)))
	if err != nil {
		t.Fatal(err)
	}
	if s.D[0][0].Value() == s.D[0][1].Value() {
		t.Errorf("both parameters drew %g", s.D[0][0].Value())
	}
}

func TestTunableBlockConstructorsValidate(t *testing.T) {
	k := mustOK(NewTunableReal("k", 1))
	cases := []struct {
		name string
		err  error
		want error
	}{
		{"gain nil entry", second(NewTunableGain("g", [][]*TunableReal{{nil}}, 0)), ErrInvalidArgument},
		{"gain ragged", second(NewTunableGain("g", [][]*TunableReal{{k, k}, {k}}, 0)), ErrDimensionMismatch},
		{"gain empty", second(NewTunableGain("g", nil, 0)), ErrInvalidArgument},
		{"gain empty name", second(NewTunableGain("", [][]*TunableReal{{k}}, 0)), ErrInvalidArgument},
		{"gain bad dt", second(NewTunableGain("g", [][]*TunableReal{{k}}, -1)), ErrInvalidSampleTime},
		{"pid nil kp", second(NewTunablePID("c", nil, k, k, 0, 0)), ErrInvalidArgument},
		{"pid NaN tf", second(NewTunablePID("c", k, k, k, math.NaN(), 0)), ErrInvalidArgument},
		{"tf empty", second(NewTunableTF("tf", nil, nil, 0)), ErrInvalidArgument},
		{"tf improper", second(NewTunableTF("tf", [][][]*TunableReal{{{k, k, k}}}, [][]float64{{1, 1}}, 0)), ErrImproperTF},
		{"ss nil", second(NewTunableSS("ss", [][]*TunableReal{{nil}}, nil, nil, nil, 0)), ErrInvalidArgument},
	}
	for _, tc := range cases {
		if !errors.Is(tc.err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, tc.err, tc.want)
		}
	}
}

func second[T any](_ T, err error) error { return err }
