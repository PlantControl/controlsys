package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func evalSISO(t *testing.T, sys *System, s complex128) complex128 {
	t.Helper()
	h, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	return h[0][0]
}

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol*math.Max(1, math.Abs(want)) {
		t.Errorf("%s = %.15g, want %.15g", label, got, want)
	}
}

func assertMinimumPhaseStable(t *testing.T, W *System) {
	t.Helper()
	poles, err := W.Poles()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := W.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	inside := func(z complex128) bool {
		if W.IsContinuous() {
			return real(z) < 0
		}
		return cmplx.Abs(z) < 1
	}
	for _, p := range poles {
		if !inside(p) {
			t.Errorf("pole %v outside stability region", p)
		}
	}
	for _, z := range zeros {
		if !inside(z) {
			t.Errorf("zero %v outside stability region", z)
		}
	}
}

func TestMakeweightConstraints(t *testing.T) {
	cases := []struct {
		name       string
		dc, hf, Ts float64
		fm         []float64
		N          int
		minPhase   bool
	}{
		{name: "lowpass", dc: 100, fm: []float64{10}, hf: 0.1, N: 1, minPhase: true},
		{name: "highpass", dc: 0.01, fm: []float64{1, 0.1}, hf: 10, N: 1, minPhase: true},
		{name: "negative", dc: -10, fm: []float64{1, 0.1}, hf: -0.01, N: 1, minPhase: true},
		{name: "rolloff", dc: 10, fm: []float64{3, 1}, hf: 0, N: 1},
		{name: "order2", dc: 0.01, fm: []float64{1, 0.1}, hf: 10, N: 2, minPhase: true},
		{name: "order3", dc: 1000, fm: []float64{2, 5}, hf: 0.5, N: 3, minPhase: true},
		{name: "order4neg", dc: -0.5, fm: []float64{7, 2}, hf: -20, N: 4, minPhase: true},
		{name: "order3rolloff", dc: 10, fm: []float64{3, 1}, hf: 0, N: 3},
		{name: "order2zeroDC", dc: 0, fm: []float64{3, 1}, hf: 5, N: 2},
		{name: "discrete", dc: 100, fm: []float64{10}, hf: 0.1, Ts: 0.05, N: 1, minPhase: true},
		{name: "discreteOrder3", dc: 0.01, fm: []float64{20, 0.3}, hf: 4, Ts: 0.1, N: 3, minPhase: true},
		{name: "discreteOrder2neg", dc: -50, fm: []float64{1, 2}, hf: -0.2, Ts: 0.5, N: 2, minPhase: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			W, err := Makeweight(tc.dc, tc.fm, tc.hf, tc.Ts, tc.N)
			if err != nil {
				t.Fatal(err)
			}
			if W.Dt != tc.Ts {
				t.Fatalf("Dt = %g, want %g", W.Dt, tc.Ts)
			}
			if n, _, _ := W.Dims(); n != tc.N {
				t.Fatalf("order %d, want %d", n, tc.N)
			}
			freq, mag := tc.fm[0], 1.0
			if len(tc.fm) == 2 {
				mag = tc.fm[1]
			}
			var w0, wInf, wFreq complex128
			if tc.Ts == 0 {
				w0 = evalSISO(t, W, 0)
				wInf = complex(W.D.At(0, 0), 0)
				wFreq = evalSISO(t, W, complex(0, freq))
			} else {
				w0 = evalSISO(t, W, 1)
				wInf = evalSISO(t, W, -1)
				wFreq = evalSISO(t, W, cmplx.Exp(complex(0, freq*tc.Ts)))
			}
			const tol = 1e-9
			assertClose(t, "W(0)", real(w0), tc.dc, tol)
			assertClose(t, "imag W(0)", imag(w0), 0, tol)
			assertClose(t, "W(inf)", real(wInf), tc.hf, tol)
			assertClose(t, "imag W(inf)", imag(wInf), 0, tol)
			assertClose(t, "|W(j freq)|", cmplx.Abs(wFreq), mag, tol)
			stable, err := W.IsStable()
			if err != nil || !stable {
				t.Fatalf("IsStable = %v, %v", stable, err)
			}
			if tc.minPhase {
				assertMinimumPhaseStable(t, W)
			}
		})
	}
}

// MATLAB Answers 1947288 reports makeweight(100,10,0.1) as
// (0.1 s + 9.95)/(s + 0.0995); the exact values follow from MATLAB's
// first-order formula p = wc·√((hf²−1)/(1−dc²)), W = (hf s + dc p)/(s + p).
func TestMakeweightMATLABFirstOrder(t *testing.T) {
	W, err := Makeweight(100, []float64{10}, 0.1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	p := 10 * math.Sqrt((0.1*0.1-1)/(1-100*100))
	assertClose(t, "pole", p, 0.0995037190209989, 1e-12)
	poles, err := W.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "pole", real(poles[0]), -p, 1e-12)
	for _, w := range []float64{1e-3, 0.1, 1, 10, 1e3} {
		s := complex(0, w)
		want := (0.1*s + complex(100*p, 0)) / (s + complex(p, 0))
		if got := evalSISO(t, W, s); cmplx.Abs(got-want) > 1e-12*cmplx.Abs(want) {
			t.Errorf("W(j%g) = %v, want %v", w, got, want)
		}
	}
}

// MATLAB's discrete first-order makeweight (rctutil/makeweight.m) is
// (aa z + bb)/(cc z + dd) with bt = wc sin(wc Ts)/(cos(wc Ts) − 1),
// aa = p dc − hf bt, bb = p dc + hf bt, cc = p − bt, dd = p + bt.
func TestMakeweightMATLABDiscreteFirstOrder(t *testing.T) {
	const dc, wc, hf, Ts = 0.2, 4.0, 30.0, 0.3
	W, err := Makeweight(dc, []float64{wc}, hf, Ts, 1)
	if err != nil {
		t.Fatal(err)
	}
	p := wc * math.Sqrt((hf*hf-1)/(1-dc*dc))
	bt := wc * math.Sin(wc*Ts) / (math.Cos(wc*Ts) - 1)
	aa, bb, cc, dd := p*dc-hf*bt, p*dc+hf*bt, p-bt, p+bt
	for _, th := range []float64{0, 0.3, 1, 2, 3} {
		z := cmplx.Exp(complex(0, th))
		want := (complex(aa, 0)*z + complex(bb, 0)) / (complex(cc, 0)*z + complex(dd, 0))
		if got := evalSISO(t, W, z); cmplx.Abs(got-want) > 1e-12*cmplx.Abs(want) {
			t.Errorf("W(e^j%g) = %v, want %v", th, got, want)
		}
	}
}

// For N = 2 the Butterworth pattern gives
// W = (|hf| s² + √2 √(dc hf) w0 s + dc w0²)/(s² + √2 w0 s + w0²),
// w0 = freq·((mag²−hf²)/(dc²−mag²))^(1/4).
func TestMakeweightSecondOrderClosedForm(t *testing.T) {
	const dc, freq, mag, hf = 0.01, 1.0, 0.1, 10.0
	W, err := Makeweight(dc, []float64{freq, mag}, hf, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	w0 := freq * math.Pow((mag*mag-hf*hf)/(dc*dc-mag*mag), 0.25)
	for _, w := range []float64{1e-2, 0.3, 1, 5, 100} {
		s := complex(0, w)
		num := complex(hf, 0)*s*s + complex(math.Sqrt2*math.Sqrt(dc*hf)*w0, 0)*s + complex(dc*w0*w0, 0)
		den := s*s + complex(math.Sqrt2*w0, 0)*s + complex(w0*w0, 0)
		if got, want := evalSISO(t, W, s), num/den; cmplx.Abs(got-want) > 1e-10*cmplx.Abs(want) {
			t.Errorf("W(j%g) = %v, want %v", w, got, want)
		}
	}
}

func TestMakeweightButterworthPattern(t *testing.T) {
	const N = 5
	const dc, freq, mag, hf = 100.0, 2.0, 3.0, 0.5
	W, err := Makeweight(dc, []float64{freq, mag}, hf, 0, N)
	if err != nil {
		t.Fatal(err)
	}
	w0 := freq * math.Pow((mag*mag-hf*hf)/(dc*dc-mag*mag), 1.0/(2*N))
	wz := w0 * math.Pow(dc/hf, 1.0/N)
	check := func(label string, roots []complex128, radius float64) {
		if len(roots) != N {
			t.Fatalf("%s: %d roots, want %d", label, len(roots), N)
		}
		for _, r := range roots {
			assertClose(t, label+" radius", cmplx.Abs(r), radius, 1e-8)
			k := (math.Atan2(imag(r), real(r)) - math.Pi/2) * float64(2*N) / math.Pi
			if math.Abs(k-math.Round(k)) > 1e-6 || int(math.Round(k))%2 == 0 {
				t.Errorf("%s %v not on the Butterworth angles", label, r)
			}
		}
	}
	poles, err := W.Poles()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := W.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	check("pole", poles, w0)
	check("zero", zeros, wz)
}

func TestMakeweightErrors(t *testing.T) {
	cases := []struct {
		name       string
		dc, hf, Ts float64
		fm         []float64
		N          int
		want       error
	}{
		{"gains same side", 2, 3, 0, []float64{1}, 1, ErrInvalidArgument},
		{"mag outside", 0.1, 10, 0, []float64{1, 20}, 1, ErrInvalidArgument},
		{"equal gains", 1, 1, 0, []float64{1}, 1, ErrInvalidArgument},
		{"mag equals dc", 0.1, 10, 0, []float64{1, 0.1}, 1, ErrInvalidArgument},
		{"freq zero", 0.1, 10, 0, []float64{0, 1}, 1, ErrInvalidArgument},
		{"freq negative", 0.1, 10, 0, []float64{-1}, 1, ErrInvalidArgument},
		{"mag zero", 0, 10, 0, []float64{1, 0}, 1, ErrInvalidArgument},
		{"freqMag empty", 0.1, 10, 0, nil, 1, ErrInvalidArgument},
		{"freqMag long", 0.1, 10, 0, []float64{1, 1, 1}, 1, ErrInvalidArgument},
		{"NaN dc", math.NaN(), 10, 0, []float64{1}, 1, ErrInvalidArgument},
		{"Inf hf", 0.1, math.Inf(1), 0, []float64{1}, 1, ErrInvalidArgument},
		{"NaN freq", 0.1, 10, 0, []float64{math.NaN()}, 1, ErrInvalidArgument},
		{"Inf Ts", 0.1, 10, math.Inf(1), []float64{1}, 1, ErrInvalidArgument},
		{"order zero", 0.1, 10, 0, []float64{1}, 0, ErrInvalidArgument},
		{"above Nyquist", 0.1, 10, 1, []float64{4}, 1, ErrInvalidArgument},
		{"at Nyquist", 0.1, 10, 1, []float64{math.Pi}, 1, ErrInvalidArgument},
		{"unspecified Ts", 0.1, 10, -1, []float64{1}, 1, ErrInvalidSampleTime},
		{"opposite signs order 2", -0.1, 10, 0, []float64{1}, 2, ErrOptionUnsupported},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			W, err := Makeweight(tc.dc, tc.fm, tc.hf, tc.Ts, tc.N)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if W != nil {
				t.Fatal("non-nil model with error")
			}
		})
	}
}

// Opposite-sign gains with N = 1 follow MATLAB's formula and give a
// right-half-plane zero.
func TestMakeweightOppositeSignsFirstOrder(t *testing.T) {
	W, err := Makeweight(-0.1, []float64{1}, 10, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "W(0)", real(evalSISO(t, W, 0)), -0.1, 1e-12)
	assertClose(t, "W(inf)", W.D.At(0, 0), 10, 1e-12)
	assertClose(t, "|W(j)|", cmplx.Abs(evalSISO(t, W, 1i)), 1, 1e-12)
	z, err := W.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(z) != 1 || real(z[0]) <= 0 {
		t.Fatalf("zeros %v, want one right-half-plane zero", z)
	}
}
