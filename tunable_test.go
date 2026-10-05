package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
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

func TestTunableBlockDeterministicAndRandomSampling(t *testing.T) {
	k, _ := newBoundedReal("K", 2, 1, 4)
	block := mustOK(tunableGainWith("gain", [][]*TunableReal{{k}}))

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
	gain := mustOK(tunableGainWith("g", [][]*TunableReal{{a, b}}))
	if _, err := gain.RandomSample(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil rng: err = %v, want ErrInvalidArgument", err)
	}
	s, err := gain.RandomSample(rand.New(rand.NewPCG(7, 0)))
	if err != nil {
		t.Fatal(err)
	}
	if s.Gain[0][0].Value() == s.Gain[0][1].Value() {
		t.Errorf("both parameters drew %g", s.Gain[0][0].Value())
	}
}

func second[T any](_ T, err error) error { return err }

// tunableGainWith builds a gain block whose entries are the given parameters,
// as MATLAB tunableGain followed by assigning realp entries.
func tunableGainWith(name string, params [][]*TunableReal) (*TunableGain, error) {
	g, err := NewTunableGain(name, len(params), len(params[0]))
	if err != nil {
		return nil, err
	}
	g.Gain = params
	return g, nil
}

func ssResponseOracle(sys *System, s complex128) [][]complex128 {
	n, m, p := sys.Dims()
	out := make([][]complex128, p)
	for i := range out {
		out[i] = make([]complex128, m)
	}
	M := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			M[i*n+j] = -complex(sys.A.At(i, j), 0)
		}
		M[i*n+i] += s
	}
	for j := range m {
		x := make([]complex128, n)
		if n > 0 {
			b := make([]complex128, n)
			for i := range n {
				b[i] = complex(sys.B.At(i, j), 0)
			}
			x = complexSolve(M, b, n)
		}
		for i := range p {
			v := complex(sys.D.At(i, j), 0)
			for k := range n {
				v += complex(sys.C.At(i, k), 0) * x[k]
			}
			out[i][j] = v
		}
	}
	return out
}

func assertResponseAt(t *testing.T, label string, sys *System, s complex128, want [][]complex128) {
	t.Helper()
	got, err := sys.EvalFr(s)
	if err != nil {
		t.Fatalf("%s: EvalFr: %v", label, err)
	}
	if len(got) != len(want) {
		t.Fatalf("%s: %d outputs, want %d", label, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s: %d inputs, want %d", label, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if cmplx.Abs(got[i][j]-want[i][j]) > 1e-9*(1+cmplx.Abs(want[i][j])) {
				t.Errorf("%s: H(%d,%d)(%v) = %v, want %v", label, i, j, s, got[i][j], want[i][j])
			}
		}
	}
}

func freeNames(params []*TunableReal) []string {
	var out []string
	for _, p := range params {
		out = append(out, p.Name())
	}
	return out
}

func TestTunableGainMATLABShapes(t *testing.T) {
	g := mustOK(NewTunableGain("K", 2, 3))
	if got, want := freeNames(g.FreeParameters()), []string{"K.Gain(1,1)", "K.Gain(1,2)", "K.Gain(1,3)", "K.Gain(2,1)", "K.Gain(2,2)", "K.Gain(2,3)"}; !slices.Equal(got, want) {
		t.Fatalf("free = %v, want %v", got, want)
	}
	for _, row := range g.Gain {
		for _, p := range row {
			if b := p.Bounds(); p.Value() != 0 || !math.IsInf(b.Lower, -1) || !math.IsInf(b.Upper, 1) {
				t.Fatalf("%s = %g in %v, want 0 unbounded", p.Name(), p.Value(), b)
			}
		}
	}
	G := mat.NewDense(2, 2, []float64{1, -2, 3, 0.5})
	h := mustOK(NewTunableGainFrom("H", G))
	h.Dt = 0.1
	h.InputName, h.OutputName = []string{"e1", "e2"}, []string{"u1", "u2"}
	G.Set(0, 0, 99)
	if err := h.Gain[1][0].SetValue(4); err != nil {
		t.Fatal(err)
	}
	sys := mustOK(h.CurrentSystem())
	if !mat.Equal(sys.D, mat.NewDense(2, 2, []float64{1, -2, 4, 0.5})) || sys.Dt != 0.1 || !slices.Equal(sys.InputName, []string{"e1", "e2"}) || !slices.Equal(sys.OutputName, []string{"u1", "u2"}) {
		t.Fatalf("gain = %v Dt %g names %v %v", mat.Formatted(sys.D), sys.Dt, sys.InputName, sys.OutputName)
	}
}

func TestTunablePIDFamiliesMatchMATLAB(t *testing.T) {
	cases := []struct {
		typ  PidtuneType
		free []string
	}{
		{"P", []string{"C.Kp"}},
		{"pi", []string{"C.Kp", "C.Ki"}},
		{"PD", []string{"C.Kp", "C.Kd", "C.Tf"}},
		{"PID", []string{"C.Kp", "C.Ki", "C.Kd", "C.Tf"}},
	}
	for _, tc := range cases {
		b := mustOK(NewTunablePID("C", tc.typ, 0))
		if got := freeNames(b.FreeParameters()); !slices.Equal(got, tc.free) {
			t.Errorf("%s free = %v, want %v", tc.typ, got, tc.free)
		}
		if b.Kp.Value() != 0 || b.Ki.Value() != 0 || b.Kd.Value() != 0 || b.Tf.Value() != 1 || b.Tf.Bounds().Lower != 0 {
			t.Errorf("%s initial = %g %g %g %g", tc.typ, b.Kp.Value(), b.Ki.Value(), b.Kd.Value(), b.Tf.Value())
		}
	}
	for _, typ := range []PidtuneType{"I", "PDF", "PIDF", ""} {
		if _, err := NewTunablePID("C", typ, 0); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("type %q: err = %v, want ErrInvalidArgument", typ, err)
		}
	}

	kp, ki, kd, tf := 1.5, 0.7, 0.3, 0.2
	b := mustOK(NewTunablePID("C", PidtunePID, 0))
	for p, v := range map[*TunableReal]float64{b.Kp: kp, b.Ki: ki, b.Kd: kd, b.Tf: tf} {
		if err := p.SetValue(v); err != nil {
			t.Fatal(err)
		}
	}
	s := complex(0.3, 1.7)
	assertResponseAt(t, "continuous PID", mustOK(b.CurrentSystem()), s, [][]complex128{{complex(kp, 0) + complex(ki, 0)/s + complex(kd, 0)*s/(complex(tf, 0)*s+1)}})

	ts := 0.1
	pid := mustOK(NewPID(kp, ki, kd, tf, ts, WithPIDFormulas(BackwardEuler, Trapezoidal)))
	d := mustOK(NewTunablePIDFrom("D", pid))
	if got := freeNames(d.FreeParameters()); !slices.Equal(got, []string{"D.Kp", "D.Ki", "D.Kd", "D.Tf"}) {
		t.Fatalf("PIDFrom free = %v", got)
	}
	z := complex(0.4, 0.8)
	cts := complex(ts, 0)
	iF := cts * z / (z - 1)
	dF := cts / 2 * (z + 1) / (z - 1)
	assertResponseAt(t, "discrete PIDFrom", mustOK(d.CurrentSystem()), z, [][]complex128{{complex(kp, 0) + complex(ki, 0)*iF + complex(kd, 0)/(complex(tf, 0)+dF)}})
}

func TestTunablePIDFromFixesAbsentTerms(t *testing.T) {
	pid := mustOK(NewPID(5, 2.2, 0, 0, 0.1, WithPIDFormulas(BackwardEuler, ForwardEuler)))
	b := mustOK(NewTunablePIDFrom("piblock", pid))
	if got := freeNames(b.FreeParameters()); !slices.Equal(got, []string{"piblock.Kp", "piblock.Ki"}) {
		t.Fatalf("free = %v, want Kp, Ki", got)
	}
	if b.Dt != 0.1 || b.IFormula != BackwardEuler || b.Tf.Value() != 1 {
		t.Fatalf("Dt %g IFormula %v Tf %g", b.Dt, b.IFormula, b.Tf.Value())
	}
	z := complex(-0.2, 0.9)
	assertResponseAt(t, "PI", mustOK(b.CurrentSystem()), z, [][]complex128{{5 + 2.2*0.1*z/(z-1)}})
	if _, err := NewTunablePIDFrom("c", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil sys: err = %v", err)
	}
	if _, err := NewTunablePIDFrom("c", &PID{Kp: math.NaN()}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NaN sys: err = %v", err)
	}
}

func TestTunablePID2MatchesTwoDOFFormula(t *testing.T) {
	b := mustOK(NewTunablePID2("C2", PidtunePI, 0))
	if got := freeNames(b.FreeParameters()); !slices.Equal(got, []string{"C2.Kp", "C2.Ki", "C2.b"}) {
		t.Fatalf("PI free = %v", got)
	}
	src := mustOK(NewPID2(2, 0.5, 0.4, 0.25, 0.6, 1.3, 0))
	c := mustOK(NewTunablePID2From("C2", src))
	if got := freeNames(c.FreeParameters()); !slices.Equal(got, []string{"C2.Kp", "C2.Ki", "C2.Kd", "C2.Tf", "C2.b", "C2.c"}) {
		t.Fatalf("PID free = %v", got)
	}
	s := complex(0.2, 2.1)
	I := 0.5 / s
	D := 0.4 * s / (0.25*s + 1)
	assertResponseAt(t, "PID2", mustOK(c.CurrentSystem()), s, [][]complex128{{2*0.6 + I + D*1.3, -(2 + I + D)}})
	if _, err := NewTunablePID2From("C2", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil sys: err = %v", err)
	}
}

func TestTunableTFMATLABShapes(t *testing.T) {
	b := mustOK(NewTunableTF("F", 1, 2, 0))
	if len(b.Numerator) != 2 || len(b.Denominator) != 3 || !b.Denominator[0].Fixed() || b.Denominator[0].Value() != 1 {
		t.Fatalf("Numerator %d Denominator %d leading fixed %v", len(b.Numerator), len(b.Denominator), b.Denominator[0].Fixed())
	}
	if got := freeNames(b.FreeParameters()); !slices.Equal(got, []string{"F.Numerator(1)", "F.Numerator(2)", "F.Denominator(2)", "F.Denominator(3)"}) {
		t.Fatalf("free = %v", got)
	}
	s := complex(0.5, 1.5)
	assertResponseAt(t, "default continuous", mustOK(b.CurrentSystem()), s, [][]complex128{{1 / ((s + 1) * (s + 1))}})
	for k, v := range []float64{2, -1} {
		_ = b.Numerator[k].SetValue(v)
	}
	for k, v := range []float64{3, 5} {
		_ = b.Denominator[k+1].SetValue(v)
	}
	assertResponseAt(t, "tuned", mustOK(b.CurrentSystem()), s, [][]complex128{{(2*s - 1) / (s*s + 3*s + 5)}})

	d := mustOK(NewTunableTF("G", 0, 2, 0.2))
	z := complex(0.9, 0.3)
	assertResponseAt(t, "default discrete", mustOK(d.CurrentSystem()), z, [][]complex128{{1 / ((z - 0.5) * (z - 0.5))}})

	tf := &TransferFunc{Num: [][][]float64{{{0, 2, 4}}}, Den: [][]float64{{2, 6, 8}}, Dt: 0, InputName: []string{"e"}, OutputName: []string{"u"}}
	f := mustOK(NewTunableTFFrom("H", tf))
	if len(f.Numerator) != 2 || f.Numerator[0].Value() != 1 || f.Denominator[2].Value() != 4 || f.InputName[0] != "e" {
		t.Fatalf("normalized num %v den %v", f.Numerator, f.Denominator)
	}
	assertResponseAt(t, "from tf", mustOK(f.CurrentSystem()), s, [][]complex128{{(2*s + 4) / (2*s*s + 6*s + 8)}})

	mimo := &TransferFunc{Num: [][][]float64{{{1}, {1}}}, Den: [][]float64{{1, 1}}}
	delayed := &TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, 1}}, Delay: [][]float64{{0.5}}}
	for _, tc := range []struct {
		name string
		err  error
		want error
	}{
		{"nz > np", second(NewTunableTF("F", 3, 2, 0)), ErrImproperTF},
		{"negative", second(NewTunableTF("F", -1, 2, 0)), ErrInvalidArgument},
		{"bad ts", second(NewTunableTF("F", 0, 1, -1)), ErrInvalidSampleTime},
		{"empty name", second(NewTunableTF("", 0, 1, 0)), ErrInvalidArgument},
		{"nil sys", second(NewTunableTFFrom("F", nil)), ErrInvalidArgument},
		{"MIMO", second(NewTunableTFFrom("F", mimo)), ErrNotSISO},
		{"delay", second(NewTunableTFFrom("F", delayed)), ErrDelayUnsupported},
		{"improper", second(NewTunableTFFrom("F", &TransferFunc{Num: [][][]float64{{{1, 0, 0}}}, Den: [][]float64{{1, 1}}})), ErrImproperTF},
	} {
		if !errors.Is(tc.err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, tc.err, tc.want)
		}
	}
}

func TestTunableSSStructures(t *testing.T) {
	free := func(b *TunableSS) [][]bool {
		out := make([][]bool, len(b.A))
		for i, row := range b.A {
			for _, p := range row {
				out[i] = append(out[i], !p.Fixed())
			}
		}
		return out
	}
	T, F := true, false
	for _, tc := range []struct {
		astruct TunableSSStructure
		free    [][]bool
		lastCol []float64
	}{
		{"", [][]bool{{T, T, F}, {T, T, T}, {F, T, T}}, []float64{0, 0, -1}},
		{TunableSSFull, [][]bool{{T, T, T}, {T, T, T}, {T, T, T}}, []float64{0, 0, -1}},
		{TunableSSCompanion, [][]bool{{F, F, T}, {T, F, T}, {F, T, T}}, []float64{-1, -3, -3}},
	} {
		b := mustOK(NewTunableSS("S", 3, 2, 2, 0, tc.astruct))
		if got := free(b); !slices.EqualFunc(got, tc.free, slices.Equal) {
			t.Errorf("%q free = %v, want %v", tc.astruct, got, tc.free)
		}
		for i, want := range tc.lastCol {
			if got := b.A[i][2].Value(); got != want {
				t.Errorf("%q A(%d,3) = %g, want %g", tc.astruct, i+1, got, want)
			}
		}
		if b.A[0][0].Name() != "S.A(1,1)" || b.D[1][1].Name() != "S.D(2,2)" || len(b.B) != 3 || len(b.C) != 2 {
			t.Errorf("%q names/dims wrong", tc.astruct)
		}
		poles := mustOK(mustOK(b.CurrentSystem()).Poles())
		for _, p := range poles {
			if cmplx.Abs(p+1) > 1e-4 {
				t.Errorf("%q pole %v, want -1", tc.astruct, p)
			}
		}
	}
	d := mustOK(NewTunableSS("S", 2, 1, 1, 0.1, TunableSSCompanion))
	if d.A[0][1].Value() != 0 || d.A[1][1].Value() != 0 || d.A[1][0].Value() != 1 {
		t.Fatalf("discrete companion A not z^2")
	}

	A := mat.NewDense(3, 3, []float64{-1, 2, 0.3, -0.5, -2, 1, 0.2, -0.4, -3})
	B := mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2})
	C := mat.NewDense(2, 3, []float64{1, 0.3, 0, -0.2, 1, 0.7})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0.4, -0.3})
	for _, dt := range []float64{0, 0.05} {
		Ad := mat.DenseCopyOf(A)
		if dt > 0 {
			Ad.Scale(0.2, Ad)
		}
		sys := mustOK(New(Ad, B, C, D, dt))
		sys.InputName = []string{"u1", "u2"}
		s := complex(0.3, 1.1)
		want := ssResponseOracle(sys, s)
		for _, astruct := range []TunableSSStructure{TunableSSTridiag, TunableSSFull, TunableSSCompanion} {
			b, err := NewTunableSSFrom("S", sys, astruct)
			if err != nil {
				t.Fatalf("dt %g %q: %v", dt, astruct, err)
			}
			cur := mustOK(b.CurrentSystem())
			if cur.Dt != sys.Dt || !slices.Equal(cur.InputName, sys.InputName) {
				t.Errorf("dt %g %q: Dt %g names %v", dt, astruct, cur.Dt, cur.InputName)
			}
			assertResponseAt(t, string(astruct), cur, s, want)
			fixedA := free(b)
			for i := range 3 {
				for j := range 3 {
					if !fixedA[i][j] && math.Abs(b.A[i][j].Value()) > 1e-9 {
						t.Errorf("dt %g %q: fixed A(%d,%d) = %g, want 0", dt, astruct, i+1, j+1, b.A[i][j].Value())
					}
				}
			}
		}
	}
	for _, tc := range []struct {
		name string
		err  error
		want error
	}{
		{"unknown struct", second(NewTunableSS("S", 2, 1, 1, 0, "diag")), ErrInvalidArgument},
		{"negative nx", second(NewTunableSS("S", -1, 1, 1, 0, "")), ErrInvalidArgument},
		{"no outputs", second(NewTunableSS("S", 2, 0, 1, 0, "")), ErrDimensionMismatch},
		{"nil sys", second(NewTunableSSFrom("S", nil, "")), ErrInvalidArgument},
		{"descriptor", second(NewTunableSSFrom("S", mustOK(NewDescriptor(A, B, C, D, mat.NewDense(3, 3, []float64{1, 0, 0, 0, 2, 0, 0, 0, 1}), 0)), "")), ErrDescriptorUnsupported},
	} {
		if !errors.Is(tc.err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, tc.err, tc.want)
		}
	}
}

func TestTunableBlockConstructorsValidate(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want error
	}{
		{"gain empty name", second(NewTunableGain("", 1, 1)), ErrInvalidArgument},
		{"gain negative", second(NewTunableGain("g", -1, 1)), ErrInvalidArgument},
		{"gain zero", second(NewTunableGain("g", 0, 1)), ErrDimensionMismatch},
		{"gain nil G", second(NewTunableGainFrom("g", nil)), ErrInvalidArgument},
		{"gain NaN G", second(NewTunableGainFrom("g", mat.NewDense(1, 1, []float64{math.NaN()}))), ErrInvalidArgument},
		{"pid bad ts", second(NewTunablePID("c", PidtunePI, -1)), ErrInvalidSampleTime},
		{"pid empty name", second(NewTunablePID("", PidtunePI, 0)), ErrInvalidArgument},
		{"pid2 bad type", second(NewTunablePID2("c", "PIDF", 0)), ErrInvalidArgument},
	} {
		if !errors.Is(tc.err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, tc.err, tc.want)
		}
	}
	g := mustOK(NewTunableGain("g", 1, 2))
	g.Gain[0][1] = nil
	if _, err := g.CurrentSystem(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil entry: err = %v", err)
	}
	g.Gain = [][]*TunableReal{{mustOK(NewTunableReal("a", 1)), mustOK(NewTunableReal("b", 1))}, {mustOK(NewTunableReal("c", 1))}}
	if _, err := g.CurrentSystem(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("ragged: err = %v", err)
	}
	g.Gain = [][]*TunableReal{{mustOK(NewTunableReal("a", 1))}}
	g.Dt = -1
	if _, err := g.CurrentSystem(); !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("bad Dt: err = %v", err)
	}
	pid := mustOK(NewTunablePID("c", PidtunePID, 0))
	pid.Kd = nil
	if _, err := pid.CurrentSystem(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Kd: err = %v", err)
	}
}
