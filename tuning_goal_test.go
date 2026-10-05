package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func goalFocusGrid(lo, hi float64) []float64 {
	n := tuningGoalGridPoints
	out := make([]float64, n)
	a, b := math.Log10(lo), math.Log10(hi)
	for k := range out {
		out[k] = math.Pow(10, a+float64(k)*(b-a)/float64(n-1))
	}
	out[0], out[n-1] = lo, hi
	return out
}

func sigmaMax2(m [2][2]complex128) float64 {
	fro := 0.0
	for i := range 2 {
		for j := range 2 {
			fro += real(m[i][j] * cmplx.Conj(m[i][j]))
		}
	}
	det := cmplx.Abs(m[0][0]*m[1][1] - m[0][1]*m[1][0])
	return math.Sqrt((fro + math.Sqrt(math.Max(0, fro*fro-4*det*det))) / 2)
}

func to2x2(h [][]complex128) [2][2]complex128 {
	return [2][2]complex128{{h[0][0], h[0][1]}, {h[1][0], h[1][1]}}
}

func evalPoint(sys *System, w float64) complex128 {
	if sys.IsDiscrete() {
		return cmplx.Exp(complex(0, w*sys.Dt))
	}
	return complex(0, w)
}

func goalMIMOPlant(t *testing.T, dt float64) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1, 2, 0.3, -0.5, -2, 1, 0.2, -0.4, -3})
	if dt > 0 {
		A.Scale(0.2, A)
	}
	sys := mustOK(New(A, mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2}),
		mat.NewDense(2, 3, []float64{1, 0.3, 0, -0.2, 1, 0.7}), mat.NewDense(2, 2, []float64{0.1, 0, 0.4, -0.3}), dt))
	sys.InputName, sys.OutputName = expandName("r", 2), expandName("y", 2)
	return sys
}

func assertGoalValue(t *testing.T, label string, res TuningGoalResult, want float64) {
	t.Helper()
	if math.Abs(res.Value-want) > 1e-9*(1+math.Abs(want)) {
		t.Errorf("%s: f = %.15g, want %.15g", label, res.Value, want)
	}
	if res.Limit != 1 || res.Pass != (want <= 1) || math.Abs(res.Violation-math.Max(0, want-1)) > 1e-9*(1+want) {
		t.Errorf("%s: result %+v inconsistent with f = %g", label, res, want)
	}
}

func TestTrackingGoalMatchesMaxErrorProfile(t *testing.T) {
	responseTime, dcErr, peakErr := 0.5, 0.05, 1.3
	wc := 2 / responseTime
	for _, dt := range []float64{0, 0.05} {
		sys := goalMIMOPlant(t, dt)
		goal := mustOK(mustOK(NewTrackingGoal("r", "y", responseTime, dcErr, peakErr)).WithFocus(0.05, 20))
		res, err := goal.Evaluate(sys)
		if err != nil {
			t.Fatal(err)
		}
		omega := goalFocusGrid(0.05, 20)
		want := 0.0
		for _, w := range omega {
			h := to2x2(ssResponseOracle(sys, evalPoint(sys, w)))
			h[0][0]--
			h[1][1]--
			s := complex(0, w)
			maxErr := cmplx.Abs((complex(peakErr, 0)*s + complex(wc*dcErr, 0)) / (s + complex(wc, 0)))
			want = math.Max(want, sigmaMax2(h)/maxErr)
		}
		assertGoalValue(t, "tracking", res, want)
	}

	siso := makeSISO(-4, 4, 1, 0)
	siso.InputName, siso.OutputName = []string{"r"}, []string{"y"}
	res := mustOK(mustOK(mustOK(NewTrackingGoal("r", "y", 1, 0, 0)).WithFocus(0.01, 100)).Evaluate(siso))
	want := 0.0
	for _, w := range goalFocusGrid(0.01, 100) {
		s := complex(0, w)
		want = math.Max(want, cmplx.Abs(-s/(s+4))/cmplx.Abs((s+2*0.001)/(s+2)))
	}
	assertGoalValue(t, "default errors", res, want)
	nonSquare := goalMIMOPlant(t, 0)
	nonSquare.OutputName = []string{"y", "z"}
	if _, err := mustOK(NewTrackingGoal("r", "y", 1, 0, 0)).Evaluate(nonSquare); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("non-square tracking: err = %v, want ErrDimensionMismatch", err)
	}
}

func mul2(a, b [2][2]complex128) [2][2]complex128 {
	var out [2][2]complex128
	for i := range 2 {
		for j := range 2 {
			for k := range 2 {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func TestGainGoalsMatchSingularValueOracle(t *testing.T) {
	WL := mustOK(NewGain(mat.NewDense(2, 2, []float64{2, 0, 0.5, 1}), 0))
	WR := mustOK(New(mat.NewDense(1, 1, []float64{-3}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(2, 1, []float64{3, 0}), mat.NewDense(2, 2, []float64{0, 0, 0, 1}), 0))
	sys := goalMIMOPlant(t, 0)
	omega := goalFocusGrid(0.1, 10)
	gain := mustOK(mustOK(NewGainGoal("r", "y", 1.7)).WithFocus(0.1, 10))
	weighted := mustOK(mustOK(NewWeightedGainGoal("r", "y", WL, WR)).WithFocus(0.1, 10))
	peak, wpeak, elem := 0.0, 0.0, 0.0
	for _, w := range omega {
		s := complex(0, w)
		h := to2x2(ssResponseOracle(sys, s))
		peak = math.Max(peak, sigmaMax2(h))
		elem = math.Max(elem, cmplx.Abs(h[0][1]))
		wpeak = math.Max(wpeak, sigmaMax2(mul2(mul2(to2x2(ssResponseOracle(WL, s)), h), to2x2(ssResponseOracle(WR, s)))))
	}
	assertGoalValue(t, "gain", mustOK(gain.Evaluate(sys)), peak/1.7)
	assertGoalValue(t, "weighted gain", mustOK(weighted.Evaluate(sys)), wpeak)
	single := mustOK(mustOK(NewGainGoal("r(2)", "y(1)", 1)).WithFocus(0.1, 10))
	assertGoalValue(t, "channel", mustOK(single.Evaluate(sys)), elem)

	bad := mustOK(NewWeightedGainGoal("r", "y", mustOK(NewGain(mat.NewDense(3, 3, nil), 0)), nil))
	if _, err := bad.Evaluate(sys); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("weight size: err = %v, want ErrDimensionMismatch", err)
	}
	discreteWeight := mustOK(NewWeightedGainGoal("r", "y", mustOK(NewGain(mat.NewDense(2, 2, nil), 0.1)), nil))
	if _, err := discreteWeight.Evaluate(sys); !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("weight domain: err = %v, want ErrDomainMismatch", err)
	}
}

func goalLoop(t *testing.T, dt float64) (*GeneralizedClosedLoop, *System, *mat.Dense) {
	t.Helper()
	plant := goalMIMOPlant(t, dt)
	plant.InputName = expandName("u", 2)
	K := mat.NewDense(2, 2, []float64{0.8, -0.2, 0.3, 0.5})
	ctrl := mustOK(NewGain(K, dt))
	ctrl.InputName, ctrl.OutputName = expandName("e", 2), expandName("u", 2)
	loop := mustOK(NewGeneralizedClosedLoop("cl", plant, fixedBlockT(t, ctrl), "y"))
	if err := loop.InsertAnalysisPoint("u", AnalysisPointPlantInput); err != nil {
		t.Fatal(err)
	}
	return loop, plant, K
}

func inv2(m [2][2]complex128) [2][2]complex128 {
	d := m[0][0]*m[1][1] - m[0][1]*m[1][0]
	return [2][2]complex128{{m[1][1] / d, -m[0][1] / d}, {-m[1][0] / d, m[0][0] / d}}
}

func outputSensitivityOracle(plant *System, K *mat.Dense, w float64) [2][2]complex128 {
	P := to2x2(ssResponseOracle(plant, evalPoint(plant, w)))
	Kc := [2][2]complex128{{complex(K.At(0, 0), 0), complex(K.At(0, 1), 0)}, {complex(K.At(1, 0), 0), complex(K.At(1, 1), 0)}}
	M := mul2(P, Kc)
	M[0][0]++
	M[1][1]++
	return inv2(M)
}

func TestSensitivityAndRejectionGoalsOnLoop(t *testing.T) {
	attfact := mustOK(New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{5}), mat.NewDense(1, 1, []float64{0}), 0))
	for _, dt := range []float64{0, 0.05} {
		loop, plant, K := goalLoop(t, dt)
		sens := mustOK(mustOK(NewSensitivityGoal("y", mustOK(NewGain(mat.NewDense(1, 1, []float64{1.5}), 0)))).WithFocus(0.1, 10))
		rej := mustOK(mustOK(NewRejectionGoal("y", attfact)).WithFocus(0.1, 10))
		wantS, wantR := 0.0, 0.0
		for _, w := range goalFocusGrid(0.1, 10) {
			sigma := sigmaMax2(outputSensitivityOracle(plant, K, w))
			wantS = math.Max(wantS, sigma/1.5)
			wantR = math.Max(wantR, sigma*cmplx.Abs(5/(complex(0, w)+1)))
		}
		assertGoalValue(t, "sensitivity", mustOK(sens.Evaluate(loop)), wantS)
		assertGoalValue(t, "rejection", mustOK(rej.Evaluate(loop)), wantR)
	}
	loop, _, _ := goalLoop(t, 0)
	if _, err := mustOK(NewSensitivityGoal("nowhere", mustOK(NewGain(mat.NewDense(1, 1, []float64{1}), 0)))).Evaluate(loop); !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("unknown location: err = %v, want ErrSignalNotFound", err)
	}
	if _, err := NewSensitivityGoal("y", goalMIMOPlant(t, 0)); !errors.Is(err, ErrNotSISO) {
		t.Errorf("MIMO profile: err = %v, want ErrNotSISO", err)
	}
}

func TestTuningGoalSignalNamesOnClosedLoop(t *testing.T) {
	loop, plant, K := goalLoop(t, 0)
	full, elem, row := 0.0, 0.0, 0.0
	for _, w := range goalFocusGrid(0.1, 10) {
		S := outputSensitivityOracle(plant, K, w)
		T := [2][2]complex128{{1 - S[0][0], -S[0][1]}, {-S[1][0], 1 - S[1][1]}}
		full = math.Max(full, sigmaMax2(T))
		elem = math.Max(elem, cmplx.Abs(T[1][0]))
		row = math.Max(row, math.Hypot(cmplx.Abs(T[0][0]), cmplx.Abs(T[0][1])))
	}
	eval := func(in, out string) (TuningGoalResult, error) {
		return mustOK(mustOK(NewGainGoal(in, out, 1)).WithFocus(0.1, 10)).Evaluate(loop)
	}
	for _, tc := range []struct {
		in, out string
		want    float64
	}{
		{"e", "y", full},
		{"y", "y", full},
		{"e(1)", "y(2)", elem},
		{"y", "y(1)", row},
	} {
		res, err := eval(tc.in, tc.out)
		if err != nil {
			t.Fatalf("%s→%s: %v", tc.in, tc.out, err)
		}
		assertGoalValue(t, tc.in+"→"+tc.out, res, tc.want)
	}
	if _, err := eval("u", "y"); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("two analysis points: err = %v, want ErrOptionUnsupported", err)
	}
	if _, err := eval("e", "q"); !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("unknown output: err = %v, want ErrSignalNotFound", err)
	}
	inner, err := eval("u", "u")
	if err != nil {
		t.Fatalf("plant-input point: %v", err)
	}
	if math.Abs(inner.Value-full) < 1e-6 {
		t.Errorf("plant-input loop should differ from plant-output loop")
	}
	if _, err := mustOK(NewGainGoal("r", "y", 1)).Evaluate(plant); !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("System unknown input: err = %v, want ErrSignalNotFound", err)
	}
	if _, err := mustOK(NewGainGoal("u", "y", 1)).Evaluate(plant); err != nil {
		t.Errorf("System channel names: %v", err)
	}
}

func integratorLoop(t *testing.T, gain float64) *System {
	t.Helper()
	return mustOK(New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{gain}), mat.NewDense(1, 1, []float64{0}), 0))
}

func TestLoopShapeGoalCrossoverTolerance(t *testing.T) {
	omega := goalFocusGrid(0.01, 400)
	oracle := func(k, wc, crossTol float64) float64 {
		f := 0.0
		for _, w := range omega {
			L := complex(k*wc, 0) / complex(0, w)
			G := wc / w
			S, T := 1/(1+L), L/(1+L)
			f = math.Max(f, math.Max(cmplx.Abs(S)*G, cmplx.Abs(T)/G)/math.Pow(10, crossTol))
		}
		return f
	}
	wc, crossTol := 2.0, 0.2
	goal := mustOK(mustOK(NewLoopShapeGoal("L", integratorLoop(t, wc), crossTol)).WithFocus(0.01, 400))
	for _, k := range []float64{1, 1.5, 0.7, 2, 0.4} {
		want := oracle(k, wc, crossTol)
		assertGoalValue(t, "loop shape", mustOK(goal.Evaluate(integratorLoop(t, k*wc))), want)
		if inBand := k >= math.Pow(10, -crossTol) && k <= math.Pow(10, crossTol); inBand != (want <= 1) {
			t.Errorf("k = %g: f = %g, crossover-in-band %v", k, want, inBand)
		}
	}
	rangeGoal := mustOK(mustOK(NewLoopShapeGoalWc("L", []float64{1, 4})).WithFocus(0.01, 400))
	assertGoalValue(t, "wcrange", mustOK(rangeGoal.Evaluate(integratorLoop(t, 3))), oracle(1.5, 2, math.Log10(4)/2))
	single := mustOK(mustOK(NewLoopShapeGoalWc("L", []float64{2})).WithFocus(0.01, 400))
	assertGoalValue(t, "wc", mustOK(single.Evaluate(integratorLoop(t, 3))), oracle(1.5, 2, 0.1))
}

func TestMarginsGoalUsesDiskMargin(t *testing.T) {
	// L = 2/(s+1)^3.
	loop := mustOK(New(mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}), mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{2, 0, 0}), mat.NewDense(1, 1, []float64{0}), 0))
	worst := 0.0
	for k := range 200001 {
		s := complex(0, math.Pow(10, -3+6*float64(k)/200000))
		L := 2 / ((s + 1) * (s + 1) * (s + 1))
		worst = math.Max(worst, cmplx.Abs(1/(1+L)-0.5))
	}
	alpha := 1 / worst
	dm := mustOK(DiskMargin(loop))
	if math.Abs(dm.Alpha-alpha) > 1e-6 {
		t.Fatalf("DiskMargin α = %g, oracle %g", dm.Alpha, alpha)
	}
	if gm := 20 * math.Log10((2+dm.Alpha)/(2-dm.Alpha)); math.Abs(gm-dm.GainMarginDB[1]) > 1e-9 {
		t.Fatalf("disk gain margin %g, want %g from α", dm.GainMarginDB[1], gm)
	}
	if pm := 2 * math.Atan(dm.Alpha/2) * 180 / math.Pi; math.Abs(pm-dm.PhaseMargin) > 1e-9 {
		t.Fatalf("disk phase margin %g, want %g from α", dm.PhaseMargin, pm)
	}
	for _, tc := range []struct{ gm, pm float64 }{{3, 10}, {6, 40}, {0, 0}} {
		g := math.Pow(10, tc.gm/20)
		required := math.Max(2*(g-1)/(g+1), 2*math.Tan(tc.pm*math.Pi/360))
		res := mustOK(mustOK(NewMarginsGoal("L", tc.gm, tc.pm)).Evaluate(loop))
		if math.Abs(res.Value-required/alpha) > 1e-6 || res.Pass != (required/alpha <= 1) {
			t.Errorf("gm %g pm %g: f = %g, want %g", tc.gm, tc.pm, res.Value, required/alpha)
		}
	}
	unstable := mustOK(New(mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}), mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{10, 0, 0}), mat.NewDense(1, 1, []float64{0}), 0))
	if res := mustOK(mustOK(NewMarginsGoal("L", 1, 1)).Evaluate(unstable)); !math.IsInf(res.Value, 1) || res.Pass {
		t.Errorf("unstable loop: %+v, want +Inf failing", res)
	}
}

func TestPolesGoalRegion(t *testing.T) {
	// Poles −1±2j and −3: decay 1, damping 1/√5, natural frequency up to 3.
	A := mat.NewDense(3, 3, []float64{-1, 2, 0, -2, -1, 0, 0.5, 0, -3})
	sys := mustOK(New(A, mat.NewDense(3, 1, []float64{1, 0, 1}), mat.NewDense(1, 3, []float64{1, 1, 1}), mat.NewDense(1, 1, []float64{0}), 0))
	for _, tc := range []struct {
		decay, damping, freq, want float64
	}{
		{1.5, 0.5, 2, 1.5},
		{0.5, 0.6, math.Inf(1), 0.6 * math.Sqrt(5)},
		{0, 0, 4, 0.75},
		{0, 0, math.Inf(1), 0},
	} {
		assertGoalValue(t, "poles", mustOK(mustOK(NewPolesGoal("", tc.decay, tc.damping, tc.freq)).Evaluate(sys)), tc.want)
	}
	ts := 0.1
	z := cmplx.Rect(0.8, 0.5)
	disc := mustOK(New(mat.NewDense(2, 2, []float64{real(z), imag(z), -imag(z), real(z)}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), ts))
	s := cmplx.Log(z) / complex(ts, 0)
	decay, damping := -real(s), -real(s)/cmplx.Abs(s)
	assertGoalValue(t, "discrete", mustOK(mustOK(NewPolesGoal("", 1, 0.5, 3)).Evaluate(disc)), math.Max(math.Max(1/decay, 0.5/damping), cmplx.Abs(s)/3))
	deadbeat := mustOK(New(mat.NewDense(2, 2, nil), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), ts))
	assertGoalValue(t, "z = 0", mustOK(mustOK(NewPolesGoal("", 1, 0.5, math.Inf(1))).Evaluate(deadbeat)), 0.5)
	if res := mustOK(mustOK(NewPolesGoal("", 1, 0.5, 100)).Evaluate(deadbeat)); !math.IsInf(res.Value, 1) {
		t.Errorf("z = 0 with maxfreq: f = %g, want +Inf", res.Value)
	}
	if res := mustOK(mustOK(NewPolesGoal("", 0, 0, math.Inf(1))).Evaluate(makeSISO(0.5, 1, 1, 0))); !math.IsInf(res.Value, 1) || res.Pass {
		t.Errorf("unstable pole: %+v, want +Inf failing", res)
	}

	loop, _, _ := goalLoop(t, 0)
	poles := mustOK(mustOK(loop.Sensitivity("u")).Poles())
	minDecay := math.Inf(1)
	for _, p := range poles {
		minDecay = math.Min(minDecay, -real(p))
	}
	assertGoalValue(t, "located", mustOK(mustOK(NewPolesGoal("u", 0.5, 0, math.Inf(1))).Evaluate(loop)), 0.5/minDecay)
}

func TestOvershootGoalSecondOrder(t *testing.T) {
	zeta, wn := 0.3, 2.0
	sys := mustOK(New(mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{wn * wn, 0}), mat.NewDense(1, 1, []float64{0}), 0))
	sys.InputName, sys.OutputName = []string{"r"}, []string{"y"}
	overshoot := 100 * math.Exp(-zeta*math.Pi/math.Sqrt(1-zeta*zeta))
	res := mustOK(mustOK(NewOvershootGoal("r", "y", 20)).Evaluate(sys))
	if math.Abs(res.Value-overshoot/20) > 1e-3 || res.Pass {
		t.Errorf("overshoot f = %g, want %g", res.Value, overshoot/20)
	}
}

func TestTuningGoalConstructorsValidate(t *testing.T) {
	one := mustOK(NewGain(mat.NewDense(1, 1, []float64{1}), 0))
	nan := math.NaN()
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"tracking empty input", second(NewTrackingGoal("", "y", 1, 0, 0))},
		{"tracking responsetime", second(NewTrackingGoal("r", "y", 0, 0, 0))},
		{"tracking NaN dc", second(NewTrackingGoal("r", "y", 1, nan, 0))},
		{"tracking negative peak", second(NewTrackingGoal("r", "y", 1, 0, -1))},
		{"gain zero", second(NewGainGoal("r", "y", 0))},
		{"gain Inf", second(NewGainGoal("r", "y", math.Inf(1)))},
		{"weighted empty", second(NewWeightedGainGoal("r", "", nil, nil))},
		{"rejection nil", second(NewRejectionGoal("y", nil))},
		{"sensitivity empty location", second(NewSensitivityGoal("", one))},
		{"loopshape crosstol", second(NewLoopShapeGoal("y", one, -1))},
		{"loopshape wc order", second(NewLoopShapeGoalWc("y", []float64{2, 1}))},
		{"loopshape wc count", second(NewLoopShapeGoalWc("y", nil))},
		{"margins pm", second(NewMarginsGoal("y", 6, 90))},
		{"margins gm", second(NewMarginsGoal("y", -1, 30))},
		{"margins location", second(NewMarginsGoal("", 6, 30))},
		{"poles damping", second(NewPolesGoal("", 0, 1.5, 1))},
		{"poles decay", second(NewPolesGoal("", -1, 0, 1))},
		{"poles freq", second(NewPolesGoal("", 0, 0, 0))},
		{"overshoot", second(NewOvershootGoal("r", "y", 0))},
		{"focus", second(mustOK(NewGainGoal("r", "y", 1)).WithFocus(0, 1))},
		{"zero goal", second(TuningGoal{}.Evaluate(one))},
		{"nil model", second(mustOK(NewGainGoal("r", "y", 1)).Evaluate(nil))},
	} {
		if !errors.Is(tc.err, ErrInvalidArgument) {
			t.Errorf("%s: err = %v, want ErrInvalidArgument", tc.name, tc.err)
		}
	}
	if _, err := mustOK(NewPolesGoal("", 0, 0, 1)).WithFocus(1, 2); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("poles focus: err = %v, want ErrOptionUnsupported", err)
	}
	g := mustOK(NewMarginsGoal("y", 6, 30))
	if g.Name() != "Margins" || g.WithName("m").Name() != "m" || g.Type() != TuningGoalMargins || TuningGoalPoles.String() != "Poles" {
		t.Errorf("names: %q %q %v", g.Name(), g.WithName("m").Name(), g.Type())
	}
}

func TestTuningGoalOnGeneralizedModel(t *testing.T) {
	gm := mustOK(NewGeneralizedModel("gain", mustOK(NewTunableGainFrom("K", mat.NewDense(1, 1, []float64{0.25})))))
	if err := gm.SetInputName("r"); err != nil {
		t.Fatal(err)
	}
	if err := gm.SetOutputName("y"); err != nil {
		t.Fatal(err)
	}
	assertGoalValue(t, "generalized", mustOK(mustOK(NewGainGoal("r", "y", 0.5)).Evaluate(gm)), 0.5)
}

func TestTuningGoalDefaultGridOnDelayedLoop(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	plant.InputDelay = []float64{0.3}
	loop := mustOK(NewGeneralizedClosedLoop("cl", plant, fixedBlockT(t, mustOK(NewGain(mat.NewDense(1, 1, []float64{0.5}), 0))), "y"))
	res, err := mustOK(NewGainGoal("y", "y", 2)).Evaluate(loop)
	if err != nil {
		t.Fatalf("delayed loop without focus: %v", err)
	}
	// T = 0.5e^{-0.3s}/(s+1+0.5e^{-0.3s}) peaks at 1/3 near DC.
	if res.Value < 0.16 || res.Value > 1.0/6+1e-9 || !res.Pass {
		t.Errorf("f = %g, want just under 1/6", res.Value)
	}
}
