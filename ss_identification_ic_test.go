package controlsys

import (
	"context"
	"math"
	"testing"
)

func ioICInput(n int) []float64 {
	u := make([]float64, n)
	x := uint64(0x9e3779b97f4a7c15)
	level := 1.0
	for k := range u {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		if x%3 == 0 {
			if x&8 == 0 {
				level = -level
			} else {
				level = math.Copysign(1.5-math.Abs(level), -level)
			}
		}
		u[k] = level
	}
	return u
}

type ioICCase struct {
	name    string
	a, b, c []float64
	d       float64
	x0      []float64
	delay   int
}

func (tc ioICCase) fit(t testing.TB, u []float64) IOStateSpaceCandidate {
	t.Helper()
	y := fzIOSimulate(2, tc.a, tc.b, tc.c, tc.d, tc.x0, u, tc.delay)
	res, err := IdentifyIOStateSpace(context.Background(), u[:200], y[:200], u[200:], y[200:], 1, IOStateSpaceOptions{Order: 2, InputDelay: tc.delay, DirectFeedthrough: tc.d != 0, InitialCondition: IOInitialEstimate})
	if err != nil {
		t.Fatalf("%s: %v", tc.name, err)
	}
	return res.Candidates[0]
}

// Noiseless records from a nonzero initial state: the fit must reproduce the
// state-recursion oracle and recover det(zI-A) whatever |a_n| or the
// output/input unit ratio (ssest is exact and unit invariant here).
func TestIdentifyIOStateSpaceEstimateInitialStateSmallLastCoefficientAndUnits(t *testing.T) {
	u := ioICInput(300)
	w48a := []float64{0.7459657511413003, 0.11530070127246744, 0.11530070127246742, 0.04260820664581423}
	w48b := []float64{0.9151008458042321, 0.6333192030863108}
	cases := []ioICCase{
		{"deadbeat pole", []float64{.5, 0, 0, 0}, []float64{1, 1}, []float64{1, 1}, 0, []float64{0, 1}, 0},
		{"small pole", []float64{.5, 0, 0, .01}, []float64{1, 1}, []float64{1, 1}, 0, []float64{0, 1}, 0},
	}
	for _, g := range []float64{1, 1e3, 1e6, 1e-6} {
		cases = append(cases, ioICCase{"delay units", []float64{.75, .1, .1, .04}, []float64{.9, .6}, []float64{.35 * g, .77 * g}, 0, []float64{-1.7, -.8}, 2})
		cases = append(cases, ioICCase{"delay feedthrough units", w48a, w48b, []float64{g * 348300.95243665157, g * 774443.3278635736}, g * -98.77607841952707, []float64{2, -1.5}, 2})
	}
	for _, tc := range cases {
		c := tc.fit(t, u)
		if c.TrainingNRMSE > 1e-10 || c.ValidationNRMSE > 1e-10 {
			t.Errorf("%s c0=%g: training NRMSE %g validation %g (%s)", tc.name, tc.c[0], c.TrainingNRMSE, c.ValidationNRMSE, c.Convergence)
		}
		want := []float64{1, -(tc.a[0] + tc.a[3]), tc.a[0]*tc.a[3] - tc.a[1]*tc.a[2]}
		if len(c.Denominator) != 3+tc.delay {
			t.Fatalf("%s: denominator %v", tc.name, c.Denominator)
		}
		for i, v := range want {
			if math.Abs(c.Denominator[i]-v) > 1e-8 {
				t.Errorf("%s c0=%g: denominator %v, want %v", tc.name, tc.c[0], c.Denominator, want)
				break
			}
		}
	}
}

func BenchmarkIdentifyIOStateSpaceEstimate(b *testing.B) {
	u := ioICInput(300)
	tc := ioICCase{"delay units", []float64{.75, .1, .1, .04}, []float64{.9, .6}, []float64{.35, .77}, 0, []float64{-1.7, -.8}, 2}
	y := fzIOSimulate(2, tc.a, tc.b, tc.c, tc.d, tc.x0, u, tc.delay)
	for i := range y {
		y[i] += 1e-3 * math.Sin(1.3*float64(i)*float64(i))
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := IdentifyIOStateSpace(context.Background(), u[:200], y[:200], u[200:], y[200:], 1, IOStateSpaceOptions{MinOrder: 1, MaxOrder: 4, InputDelay: 2, InitialCondition: IOInitialEstimate}); err != nil {
			b.Fatal(err)
		}
	}
}
