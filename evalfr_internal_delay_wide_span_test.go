package controlsys

import (
	"math/cmplx"
	"testing"
)

// wideSpanPair decodes a 3-state discrete model with poles e^{±i·dt} and
// e^{10^v2·dt}, InputDelay [20 40 60] and an IO delay pattern, and a static
// 3×3 partner with OutputDelay [60 40 20]; interconnected, the delays become
// internal.
func wideSpanPair(t *testing.T, v2, v8 float64) (*System, *System) {
	t.Helper()
	s := fzDecode(t, 0x77, [fzNVals]float64{0, 15, v2, -40, 95, -100, 0.17142857142857143, 0.5, v8, -10, 2.37, 0, 19, -10, 67, -112.75})
	q := fzICPartner(t, s, 0, 3, 3)
	return s.System(t, fzSysOpts{Delays: true}), q.System(t, fzSysOpts{Delays: true})
}

func evalFrFlat(t *testing.T, sys *System, z complex128) []complex128 {
	t.Helper()
	g, err := sys.EvalFr(z)
	if err != nil {
		t.Fatal(err)
	}
	var out []complex128
	for _, row := range g {
		out = append(out, row...)
	}
	return out
}

func checkWideSpan(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	for k := range want {
		if d := cmplx.Abs(got[k] - want[k]); !(d <= 1e-9*cmplx.Abs(want[k])) {
			t.Errorf("%s entry %d = %v, want %v", name, k, got[k], want[k])
		}
	}
}

// HFKNLV: a pole ≥ 1e17 times the others made the double solve of the
// delay-free plant singular, and the pole limit then stepped by the
// spectral scale, where z^{-τ} underflows: EvalFr returned exactly 0 or
// +Inf. The oracle composes the components' own delayed responses.
func TestEvalFr_InternalDelayWidePoleSpan(t *testing.T) {
	z := complex(-0.4933, 0.9495)
	for _, tc := range []struct{ v2, v8 float64 }{{1.3, 0}, {1.6, 0}, {1.8, 0}, {2, 0}, {0, 2}} {
		a, b := wideSpanPair(t, tc.v2, tc.v8)
		ga, gb := evalFrFlat(t, a, z), evalFrFlat(t, b, z)

		par, err := Parallel(a, b)
		if err != nil {
			t.Fatal(err)
		}
		if !par.HasInternalDelay() {
			t.Fatalf("v2=%g: Parallel has no internal delay", tc.v2)
		}
		want := make([]complex128, 9)
		for k := range want {
			want[k] = ga[k] + gb[k]
		}
		checkWideSpan(t, "Parallel", evalFrFlat(t, par, z), want)

		w := cmplx.Phase(z)
		if a.Dt != 1 {
			w /= a.Dt
		}
		zw := cmplx.Rect(1, w*a.Dt)
		gwa, gwb := evalFrFlat(t, a, zw), evalFrFlat(t, b, zw)
		fr, err := par.FreqResponse([]float64{w})
		if err != nil {
			t.Fatal(err)
		}
		for i := range 3 {
			for j := range 3 {
				k := i*3 + j
				checkWideSpan(t, "Parallel FreqResponse", []complex128{fr.At(0, i, j)}, []complex128{gwa[k] + gwb[k]})
			}
		}

		fb, err := Feedback(a, b, -1)
		if err != nil {
			t.Fatal(err)
		}
		l := make([]complex128, 9)
		for i := range 3 {
			l[i*3+i] = 1
			for j := range 3 {
				for k := range 3 {
					l[i*3+j] += ga[i*3+k] * gb[k*3+j]
				}
			}
		}
		li, ok := fzCInv(3, l)
		if !ok {
			t.Fatal("singular loop")
		}
		for i := range 3 {
			for j := range 3 {
				var v complex128
				for k := range 3 {
					v += li[i*3+k] * ga[k*3+j]
				}
				want[i*3+j] = v
			}
		}
		checkWideSpan(t, "Feedback", evalFrFlat(t, fb, z), want)

		blk, err := Append(a, b)
		if err != nil {
			t.Fatal(err)
		}
		var js []Junction
		for i := range 3 {
			js = append(js, Junction{Input: i, Minus: []int{3 + i}}, Junction{Input: 3 + i, Plus: []int{i}})
		}
		cn, err := Connect(blk, js, []int{0, 1, 2}, []int{0, 1, 2})
		if err != nil {
			t.Fatal(err)
		}
		checkWideSpan(t, "Connect", evalFrFlat(t, cn, z), want)
	}
}
