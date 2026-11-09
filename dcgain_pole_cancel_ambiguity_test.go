package controlsys

import (
	"errors"
	"math"
	"testing"
)

// checkDCGainExactLimitEntries checks that every entry of g is the float
// model's exact limit: the same infinity, or within 1e-9 when finite.
func checkDCGainExactLimitEntries(t *testing.T, name string, n, m, p int, a, b, c, d, g []float64, dt float64) {
	t.Helper()
	x0 := 0.0
	if dt > 0 {
		x0 = 1
	}
	want := dcGainRationalLimit(n, m, p, a, b, c, d, x0)
	for k, v := range g {
		w := want[k]
		if math.IsInf(w, 0) || w == 0 {
			if v != w {
				t.Errorf("%s: entry %d = %v, exact limit %v", name, k, v, w)
			}
		} else if !(math.Abs(v-w) <= 1e-9*math.Abs(w)) {
			t.Errorf("%s: entry %d = %.17g, exact limit %.17g", name, k, v, w)
		}
	}
}

// INGGF2: where the exact limit of the float model and the deflated block
// expansion disagree on a pole at DC (Inf against finite, or the Inf sign),
// DCGain returns the exact limit with ErrAccuracyUnverified rather than
// either reading with a nil error. Over-cancelled discrete Series
// k(z−1)²/((z−a)(z−b))·1/(z−1) (intended 0), Parallel residue cancellation
// k/(z−1) − k(1−a)/((z−1)(z−a)) (intended k/(1−a)) and its zero-gain variant
// have an exact pole at z = 1 the given coefficients leave uncancelled; the
// fuzz rows 0xcc, 0xf7 and the w18#41 row have exact poles the deflation
// misreads or fails on.
func TestDCGainPoleCancelAmbiguityUnverified(t *testing.T) {
	type row struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
		dt         float64
	}
	rows := []row{
		{"over-cancelled Series a=0.1 b=0.7", 3, 1, 1, []float64{1, 0, 0, 0.93000000000000005, 0, -0.069999999999999993, -1.2000000000000002, 1, 0.79999999999999993}, []float64{1, 0, 0}, []float64{1, 0, 1}, []float64{0}, 0.1},
		{"over-cancelled Series PCG row", 3, 1, 1, []float64{1, 0, 0, 43.146116550661567, 0, -0.47070270576561263, -51.179027658665589, 1, 1.3721585387766906}, []float64{1, 0, 0}, []float64{81.515845670571821, 0, 1}, []float64{0}, 0.1},
		{"over-cancelled Series, other order", 3, 1, 1, []float64{0, -0.45000000000000001, 0, 1, 1.3999999999999999, 0, 0, 1, 1}, []float64{0.055, -0.060000000000000026, 0.10000000000000001}, []float64{0, 0, 1}, []float64{0}, 0.1},
		{"Parallel residue cancellation", 3, 1, 1, []float64{1, 0, 0, 0, 0, -0.90000000000000002, 0, 1, 1.8999999999999999}, []float64{1, -0.099999999999999978, 0}, []float64{1, 0, 1}, []float64{0}, 0.1},
		{"Parallel zero-gain cancellation", 3, 1, 1, []float64{0, -0.90000000000000002, 0, 1, 1.8999999999999999, 0, 0, 0, 1}, []float64{-0.90000000000000002, 1, -1}, []float64{0, 1, 1}, []float64{0}, 0.1},
		{"integrators beside a tiny block", 6, 1, 3, []float64{
			-1e-13, 2e-13, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			-1e-13, 1e-13, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			0, 0, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			0, 0, -9999.499987499375, 10099.499987499375, -100, 0,
			0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0,
		}, []float64{-0.9999435261164459, -0.2572434700675131, 0.8623188722876839, 0.7185820465431654, -0.47787916084576904, -0.974246278043856}, []float64{
			704291.0931971872, 444099.1644246522, -991436.3462197612, 196942.88945958653, 864097.183359284, -755650.0977319883,
			996008.9570228793, -237538.8670471864, -842421.282141011, 782230.889887116, 336647.11098329315, -999899.8872783968,
			819288.2452914593, -807458.6576995418, -297202.3258757174, 999623.4801059181, -349133.35794434027, -773881.1359710619,
		}, []float64{331.1729782101074, -314.4549687182428, 279.47226068756777}, 0},
	}
	for _, tc := range []struct {
		name string
		kind uint8
		v    [fzNVals]float64
	}{
		{"fuzz 0xcc", 0xcc, [fzNVals]float64{93.5, -57.75, -5, -19.5, -100, -72.75, -71.75, -48.75, 23.75, 85.75, 84.6, -58.75, -65.5, 12.75, -75.5, 41.75}},
		{"fuzz 0xf7", 0xf7, [fzNVals]float64{-80, -78, 0.5, -0.33333333333333337, 0, 0, 0, 0.5, 0, 12, 0, 0, -10, 57, -10, -10}},
	} {
		s := fzDecode(t, tc.kind, tc.v)
		dt := 0.0
		if s.Discrete {
			dt = s.Dt
		}
		rows = append(rows, row{tc.name, s.N, s.M, s.P, s.A, s.B, s.C, s.D, dt})
	}
	for _, tc := range rows {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		g, err := sys.DCGain()
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: DCGain error %v, want ErrAccuracyUnverified", tc.name, err)
		}
		if g == nil {
			t.Errorf("%s: no gain with %v", tc.name, err)
			continue
		}
		checkDCGainExactLimitEntries(t, tc.name, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, g.RawMatrix().Data, tc.dt)
	}
}

// INGGF2: true integrators, where the exact limit and the deflated
// expansion agree or the exact route is not taken, keep their exact infinite
// gain with a nil error; so do the Series cancellations both routes read as
// finite, and the discrete over-cancelled Series both read as 0.
func TestDCGainPoleCancelRoutesAgreeVerified(t *testing.T) {
	checkDCGainAgainstRationalLimit(t, "tiny integrator", 2, 1, 1, []float64{0, 0, 1, -1.3258457598967608e-08}, []float64{7.29653485561532e-14, 1.0337733234010072e-13}, []float64{0, 1}, []float64{0}, 0)
	checkDCGainAgainstRationalLimit(t, "integrator beside stiff pole", 2, 1, 1, []float64{-1000, 10, 0, 0}, []float64{1, 1}, []float64{1, 1}, []float64{0}, 0)
	checkDCGainAgainstRationalLimit(t, "dense singular block", 3, 2, 2, []float64{0.38629023905802873, 3.851534645926291, -3.23782488498432, 0, 4, -3, 4, 1.6251019182011088, -4.625101918201109}, []float64{-0.5, 0, -1.75, 0, 0, 0}, []float64{-1, -2.176206339653932, 1, 0, -0.06848213510526557, 0}, []float64{5.737696031494993, 0, 0.75, 0}, 0.1)
	s := fzDecode(t, 0xe2, [fzNVals]float64{2.5, -2.75, -5, 89.75, -35, 2, -55.25, -25.5, 100, 19.5, -74.5, -10, -100, -8.75, -86.5, 61.5})
	dt := 0.0
	if s.Discrete {
		dt = s.Dt
	}
	checkDCGainAgainstRationalLimit(t, "fuzz 0xe2", s.N, s.M, s.P, s.A, s.B, s.C, s.D, dt)

	tfd := func(num, den []float64) *System {
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 0.1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	mul := func(a, b []float64) []float64 {
		o := make([]float64, len(a)+len(b)-1)
		for i, x := range a {
			for j, y := range b {
				o[i+j] += x * y
			}
		}
		return o
	}
	dz := []float64{1, -1}
	for _, q := range []struct{ a, b, k float64 }{{0.5, 0.5, 1}, {0.25, 0.75, 2}, {0.1, 0.1, 1}} {
		g2 := tfd(mul([]float64{q.k, -q.k}, dz), mul([]float64{1, -q.a}, []float64{1, -q.b}))
		sys, err := Series(tfd([]float64{1}, dz), g2)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("over-cancelled Series %v: %v", q, err)
		}
		if !(math.Abs(g.At(0, 0)) <= 1e-9) {
			t.Errorf("over-cancelled Series %v: DCGain = %v, want 0", q, g.At(0, 0))
		}
	}
}

// INGGF2, YRJCSJ: Bandwidth on a DC gain returned with ErrAccuracyUnverified
// is computed on that value and keeps the error. Series(7s/(s+1/3), 1/s,
// (s+1)/(s+1e-14)) has its integrator cancelled below rounding beside a pole
// 1e-14 from DC that deflation snaps onto it: the exact limit with the
// sub-rounding residue read as cancelled is the intended k/(a·d), the deflated
// expansion infinite. The bandwidth is that of the model with the
// integrator cancelled by hand. Where the exact DC gain is infinite the
// bandwidth stays undefined (ErrInvalidArgument, no value).
func TestBandwidthOnUnverifiedDCGain(t *testing.T) {
	tf := func(num, den []float64) *System {
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	series := func(ss ...*System) *System {
		s := ss[0]
		for _, x := range ss[1:] {
			var err error
			if s, err = Series(s, x); err != nil {
				t.Fatal(err)
			}
		}
		return s
	}
	k, a, d := 7.0, 1.0/3, 1e-14
	sys := series(tf([]float64{k, 0}, []float64{1, a}), tf([]float64{1}, []float64{1, 0}), tf([]float64{1, 1}, []float64{1, d}))
	g, err := sys.DCGain()
	if !errors.Is(err, ErrAccuracyUnverified) || g == nil {
		t.Fatalf("DCGain = %v, %v; want a value with ErrAccuracyUnverified", g, err)
	}
	if w := k / a / d; !(math.Abs(g.At(0, 0)-w) <= 1e-12*w) {
		t.Fatalf("DCGain = %v, want %v", g.At(0, 0), w)
	}
	want, err := Bandwidth(series(tf([]float64{k}, []float64{1, a}), tf([]float64{1, 1}, []float64{1, d})), -3)
	if err != nil {
		t.Fatal(err)
	}
	bw, err := Bandwidth(sys, -3)
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Errorf("Bandwidth error %v, want ErrAccuracyUnverified", err)
	}
	if !(math.Abs(bw-want) <= 1e-4*want) {
		t.Errorf("Bandwidth = %v, want %v", bw, want)
	}

	over, err := NewFromSlices(3, 1, 1, []float64{1, 0, 0, 0.93000000000000005, 0, -0.069999999999999993, -1.2000000000000002, 1, 0.79999999999999993}, []float64{1, 0, 0}, []float64{1, 0, 1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if bw, err := Bandwidth(over, -3); !errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrAccuracyUnverified) {
		t.Errorf("Bandwidth on an infinite exact DC gain = %v, %v; want ErrInvalidArgument alone", bw, err)
	}
}
