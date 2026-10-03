package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func conversionSISO(t testing.TB, num, den []float64, dt float64) *System {
	t.Helper()
	model := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	result, err := model.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	return result.Sys
}

func TestMatchedForwardIndependentCoefficients(t *testing.T) {
	a := math.Exp(-.1)
	tests := []struct {
		name                       string
		num, den, wantNum, wantDen []float64
	}{
		{"integrator", []float64{1}, []float64{1, 0}, []float64{.1}, []float64{1, -1}},
		{"first order", []float64{1}, []float64{1, 1}, []float64{1 - a}, []float64{1, -a}},
		{"singular DC second order", []float64{11}, []float64{1, 1, 0}, []float64{11 * .1 * (1 - a) / 2, 11 * .1 * (1 - a) / 2}, []float64{1, -(1 + a), a}},
		{"double integrator", []float64{1}, []float64{1, 0, 0}, []float64{.005, .005}, []float64{1, -2, 1}},
		{"zero DC", []float64{1, 0}, []float64{1, 1}, []float64{(1 - a) / .1, -(1 - a) / .1}, []float64{1, -a}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			source := conversionSISO(t, tc.num, tc.den, 0)
			disc, err := source.DiscretizeMatched(.1)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := disc.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			compareConversionCoefficients(t, actual.TF.Num[0][0], tc.wantNum, 1e-9)
			compareConversionCoefficients(t, actual.TF.Den[0], tc.wantDen, 1e-9)
		})
	}
}

func compareConversionCoefficients(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	for len(got) > len(want) && math.Abs(got[0]) < tol {
		got = got[1:]
	}
	if len(got) != len(want) {
		t.Fatalf("coefficients %v, want %v", got, want)
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol*math.Max(1, math.Abs(want[i])) {
			t.Fatalf("coefficients %v, want %v", got, want)
		}
	}
}

func TestMatchedReverseIndependentResponse(t *testing.T) {
	dt := .1
	a, b := math.Exp(-dt), math.Exp(-2*dt)
	tests := []struct {
		name                                   string
		num, den, continuousNum, continuousDen []float64
	}{
		{"gain", []float64{3}, []float64{1}, []float64{3}, []float64{1}},
		{"first order", []float64{1 - a}, []float64{1, -a}, []float64{1}, []float64{1, 1}},
		{"integrator", []float64{dt}, []float64{1, -1}, []float64{1}, []float64{1, 0}},
		{"double integrator", []float64{dt * dt / 2, dt * dt / 2}, []float64{1, -2, 1}, []float64{1}, []float64{1, 0, 0}},
		{"repeated pole", []float64{(1 - a) * (1 - a) / 2, (1 - a) * (1 - a) / 2}, []float64{1, -2 * a, a * a}, []float64{1}, []float64{1, 2, 1}},
		{"nonminimum phase", []float64{-(1 - a) * (1 - b) / (2 * (1 - math.Exp(dt))), math.Exp(dt) * (1 - a) * (1 - b) / (2 * (1 - math.Exp(dt)))}, []float64{1, -(a + b), a * b}, []float64{1, -1}, []float64{1, 3, 2}},
		{"zero DC", []float64{(1 - a) / dt, -(1 - a) / dt}, []float64{1, -a}, []float64{1, 0}, []float64{1, 1}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			source := conversionSISO(t, tc.num, tc.den, dt)
			copy := source.Copy()
			cont, err := source.D2C(C2DMethodMatched)
			if err != nil {
				t.Fatal(err)
			}
			tf, err := cont.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, w := range []float64{.03, .4, 1.7, 4.3, 13} {
				s := complex(0, w)
				want := Poly(tc.continuousNum).Eval(s) / Poly(tc.continuousDen).Eval(s)
				got := tf.TF.Eval(s)[0][0]
				if cmplx.Abs(got-want) > 2e-6*math.Max(1, cmplx.Abs(want)) {
					t.Fatalf("w=%g got=%v want=%v", w, got, want)
				}
			}
			if !reflect.DeepEqual(source, copy) {
				t.Fatal("source mutated")
			}
		})
	}
}

func TestMatchedReverseComplexPolesAndMetadata(t *testing.T) {
	dt := .2
	pole := cmplx.Exp(complex(-1, 2) * complex(dt, 0))
	gain := (1 - pole) * (1 - cmplx.Conj(pole)) / 10
	disc := conversionSISO(t, []float64{real(gain), real(gain)}, []float64{1, -2 * real(pole), cmplx.Abs(pole) * cmplx.Abs(pole)}, dt)
	disc.InputDelay = []float64{2}
	disc.OutputDelay = []float64{3}
	disc.InputName = []string{"u"}
	disc.OutputName = []string{"y"}
	disc.StateName = []string{"old1", "old2"}
	cont, err := disc.D2C(C2DMethodMatched)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := cont.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	compareConversionCoefficients(t, tf.TF.Den[0], []float64{1, 2, 5}, 1e-8)
	if cont.InputDelay[0] != .4 || math.Abs(cont.OutputDelay[0]-.6) > 1e-12 || cont.InputName[0] != "u" || cont.OutputName[0] != "y" || len(cont.StateName) != 0 {
		t.Fatalf("metadata %v %v %v", cont.InputDelay, cont.OutputDelay, cont.StateName)
	}
	cont.InputName[0] = "changed"
	if disc.InputName[0] != "u" {
		t.Fatal("names alias source")
	}
}

func TestMatchedRejectUnsupportedRoots(t *testing.T) {
	for _, den := range [][]float64{{1, 0}, {1, .5}, {1, 1}} {
		sys := conversionSISO(t, []float64{1}, den, .1)
		if _, err := sys.D2C(C2DMethodMatched); !errors.Is(err, ErrSingularTransform) {
			t.Fatalf("den=%v err=%v", den, err)
		}
	}
	sys, _ := New(mat.NewDense(1, 1, []float64{.5}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 2, nil), .1)
	if _, err := sys.D2C(C2DMethodMatched); !errors.Is(err, ErrNotSISO) {
		t.Fatalf("MIMO err=%v", err)
	}
}

func TestLeastSquaresIndependentFrequencyResponse(t *testing.T) {
	tests := []struct {
		name     string
		num, den []float64
		dt       float64
		order    int
	}{
		{"first order", []float64{1}, []float64{1, 1}, .2, 0},
		{"fast dynamics", []float64{400}, []float64{1, 40, 400}, .1, 0},
		{"reduced order", []float64{2}, []float64{1, 3, 2}, .2, 1},
		{"biproper", []float64{1, 2}, []float64{1, 1}, .2, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			source := conversionSISO(t, tc.num, tc.den, 0)
			before := source.Copy()
			result, err := source.DiscretizeLeastSquares(tc.dt, tc.order)
			if err != nil {
				t.Fatal(err)
			}
			tf, err := result.Sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			residual, energy := 0.0, 0.0
			for k := range 827 {
				w := math.Pi * (float64(k) + .371) / (827 * tc.dt)
				want := Poly(tc.num).Eval(complex(0, w)) / Poly(tc.den).Eval(complex(0, w))
				got := tf.TF.Eval(cmplx.Exp(complex(0, w*tc.dt)))[0][0]
				residual = math.Hypot(residual, cmplx.Abs(got-want))
				energy = math.Hypot(energy, cmplx.Abs(want))
			}
			observed := residual / energy
			t.Logf("fit order=%d RMS relative=%g max relative=%g independent RMS=%g stable=%v", result.FitOrder, result.RMSRelativeError, result.MaxRelativeError, observed, result.Stable)
			if observed > .3 {
				t.Fatalf("poor fit %g", observed)
			}
			if math.Abs(observed-result.RMSRelativeError) > .005 {
				t.Fatalf("reported %g, independent %g", result.RMSRelativeError, observed)
			}
			if !reflect.DeepEqual(source, before) {
				t.Fatal("source mutated")
			}
			expected := tc.order
			if expected == 0 {
				expected, _, _ = source.Dims()
			}
			if result.FitOrder != expected {
				t.Fatal("fit order ignored")
			}
			poles, err := result.Sys.Poles()
			if err != nil {
				t.Fatal(err)
			}
			stable := true
			for _, p := range poles {
				stable = stable && cmplx.Abs(p) < 1
			}
			if stable != result.Stable {
				t.Fatal("stability report disagrees with poles")
			}
		})
	}
}

func TestLeastSquaresGainAndInvalidInputs(t *testing.T) {
	gain := conversionSISO(t, []float64{3}, []float64{1}, 0)
	gain.InputDelay = []float64{.4}
	gain.InputName = []string{"u"}
	gain.OutputName = []string{"y"}
	result, err := gain.DiscretizeLeastSquares(.2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.FitOrder != 0 || result.RMSRelativeError != 0 || !result.Stable || result.Sys.D.At(0, 0) != 3 || result.Sys.InputDelay[0] != 2 || result.Sys.InputName[0] != "u" {
		t.Fatalf("gain result %+v", result)
	}
	for _, dt := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := gain.DiscretizeLeastSquares(dt, 1); !errors.Is(err, ErrInvalidSampleTime) {
			t.Fatalf("dt=%g err=%v", dt, err)
		}
	}
	if _, err := gain.DiscretizeLeastSquares(.2, -1); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("negative order err=%v", err)
	}
	gain.InputDelay = []float64{.3}
	if _, err := gain.DiscretizeLeastSquares(.2, 1); !errors.Is(err, ErrFractionalDelay) {
		t.Fatalf("fractional delay err=%v", err)
	}
	integrator := conversionSISO(t, []float64{1}, []float64{1, 0, 0}, 0)
	if _, err := integrator.DiscretizeLeastSquares(.2, 1); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("integrator order err=%v", err)
	}
}

func BenchmarkMatchedConversion(b *testing.B) {
	sys := conversionSISO(b, []float64{1, 4}, []float64{1, 3, 2}, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := sys.DiscretizeMatched(.1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLeastSquaresConversion(b *testing.B) {
	sys := conversionSISO(b, []float64{400}, []float64{1, 40, 400}, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := sys.DiscretizeLeastSquares(.1, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func TestLeastSquaresIntegratorConstraints(t *testing.T) {
	for _, den := range [][]float64{{1, 0}, {1, 0, 0}, {1, 2, 0}} {
		source := conversionSISO(t, []float64{1}, den, 0)
		result, err := source.DiscretizeLeastSquares(.2, 0)
		if err != nil {
			t.Fatal(err)
		}
		model, err := result.Sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(den) == 2 {
			compareConversionCoefficients(t, model.TF.Num[0][0], []float64{.1, .1}, 1e-8)
			compareConversionCoefficients(t, model.TF.Den[0], []float64{1, -1}, 1e-8)
		}
		for _, w := range []float64{.0001, .001, .01} {
			got := model.TF.Eval(cmplx.Exp(complex(0, w*.2)))[0][0]
			want := 1 / Poly(den).Eval(complex(0, w))
			if cmplx.Abs(got-want)/cmplx.Abs(want) > .003 {
				t.Fatalf("den=%v w=%g fitted=%v want=%v", den, w, got, want)
			}
		}
		if math.IsNaN(result.RMSRelativeError) || math.IsInf(result.RMSRelativeError, 0) {
			t.Fatal("nonfinite integrator quality")
		}
	}
}

func TestLeastSquaresOfflineVariableProjectionReference(t *testing.T) {
	fixtures := []struct {
		den, num, wantDen, wantNum []float64
		rms                        float64
	}{
		{[]float64{1, 1}, []float64{1}, []float64{1, -.8103716398716132}, []float64{.09595485569826978, .09377492110570333}, .10381669572326799},
		{[]float64{1, 3, 2}, []float64{2}, []float64{1, -.9015732018101918}, []float64{.0079768232110633, .09977155207930476}, .2805154965480451},
	}
	for _, f := range fixtures {
		source := conversionSISO(t, f.num, f.den, 0)
		result, err := source.DiscretizeLeastSquares(.2, 1)
		if err != nil {
			t.Fatal(err)
		}
		tf, err := result.Sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		compareConversionCoefficients(t, tf.TF.Num[0][0], f.wantNum, .001)
		compareConversionCoefficients(t, tf.TF.Den[0], f.wantDen, .001)
		if math.Abs(result.RMSRelativeError-f.rms) > .001 {
			t.Fatalf("fit RMS=%g reference=%g", result.RMSRelativeError, f.rms)
		}
	}
}

func TestLeastSquaresExactSampledRationalFit(t *testing.T) {
	den := []float64{1, -.4, .1}
	num := []float64{.2, .1, -.03}
	target := make([]complex128, 513)
	for k := range target {
		z := cmplx.Exp(complex(0, math.Pi*float64(k)/512))
		target[k] = Poly(num).Eval(z) / Poly(den).Eval(z)
	}
	gotNum, gotDen, err := fitDiscreteResponse(target, 2)
	if err != nil {
		t.Fatal(err)
	}
	compareConversionCoefficients(t, gotNum, num, 1e-10)
	compareConversionCoefficients(t, gotDen, den, 1e-10)
}

func TestMatchedDeflateRepeatedArtificialZeros(t *testing.T) {
	numerator := Poly{1}
	for range 5 {
		numerator = numerator.Mul(Poly{1, 1})
	}
	reduced, count := matchedDeflateArtificialZeros(numerator)
	if count != 5 || len(reduced) != 1 || reduced[0] != 1 {
		t.Fatalf("deflated=%v count=%d", reduced, count)
	}
}

func BenchmarkLeastSquaresOrders(b *testing.B) {
	for _, tc := range []struct {
		name  string
		den   []float64
		order int
	}{
		{"source2_fit1", []float64{1, 40, 400}, 1},
		{"source2_fit4", []float64{1, 40, 400}, 4},
		{"source8_fit8", []float64{1, 36, 546, 4536, 22449, 67284, 118124, 109584, 40320}, 8},
	} {
		b.Run(tc.name, func(b *testing.B) {
			sys := conversionSISO(b, []float64{tc.den[len(tc.den)-1]}, tc.den, 0)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := sys.DiscretizeLeastSquares(.1, tc.order); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestLeastSquaresOptionsDispatch(t *testing.T) {
	source := conversionSISO(t, []float64{2}, []float64{1, 3, 2}, 0)
	direct, err := source.DiscretizeLeastSquares(.2, 1)
	if err != nil {
		t.Fatal(err)
	}
	dispatch, err := source.DiscretizeWithOpts(.2, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 1})
	if err != nil {
		t.Fatal(err)
	}
	got, err := dispatch.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := direct.Sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	compareConversionCoefficients(t, got.TF.Num[0][0], want.TF.Num[0][0], 1e-12)
	compareConversionCoefficients(t, got.TF.Den[0], want.TF.Den[0], 1e-12)
}

func TestLeastSquaresRejectUnsupportedModelClasses(t *testing.T) {
	source := conversionSISO(t, []float64{1}, []float64{1, 1}, 0)
	source.E = mat.NewDense(1, 1, []float64{2})
	if _, err := source.DiscretizeLeastSquares(.2, 1); !errors.Is(err, ErrDescriptorUnsupported) {
		t.Fatalf("descriptor err=%v", err)
	}
	source.E = nil
	source.LFT = &LFTDelay{Tau: []float64{.3}, B2: mat.NewDense(1, 1, []float64{1}), C2: mat.NewDense(1, 1, []float64{1}), D12: mat.NewDense(1, 1, nil), D21: mat.NewDense(1, 1, nil), D22: mat.NewDense(1, 1, nil)}
	if _, err := source.DiscretizeLeastSquares(.2, 1); !errors.Is(err, ErrFeedbackDelay) {
		t.Fatalf("internal delay err=%v", err)
	}
	mimo, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mimo.DiscretizeLeastSquares(.2, 1); !errors.Is(err, ErrNotSISO) {
		t.Fatalf("MIMO err=%v", err)
	}
}
