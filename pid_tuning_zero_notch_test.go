package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// routhStable reports whether every root of p (descending) has a negative
// real part, by the Routh array's first column.
func routhStable(p []float64) bool {
	n := len(p)
	rows := [][]float64{{}, {}}
	for i, c := range p {
		rows[i%2] = append(rows[i%2], c)
	}
	for len(rows) < n {
		a, b := rows[len(rows)-2], rows[len(rows)-1]
		if b[0] == 0 {
			return false
		}
		var r []float64
		for j := 0; j+1 < len(a); j++ {
			bj := 0.
			if j+1 < len(b) {
				bj = b[j+1]
			}
			r = append(r, (b[0]*a[j+1]-a[0]*bj)/b[0])
		}
		if len(r) == 0 {
			r = []float64{0}
		}
		rows = append(rows, r)
	}
	for _, r := range rows {
		if r[0]*p[0] <= 0 {
			return false
		}
	}
	return true
}

// Tiny Kp and Ki against Kd put a lightly damped controller zero pair near
// 0.009 rad/s, below the wc/100 analysis band, where |L| notched through
// 0 dB with a 1.2° (|S| 49) and 7.6° margin while the evidence reported 57°.
// Oracle: a dense sweep of L(jω) = G(jω)·C(jω) from closed forms over
// [1e-4·wc, 1e4·wc] for every 0 dB crossing and the |S| peak, and Routh on
// den·s(Tf·s+1) + num·((Kd+Kp·Tf)s² + (Kp+Ki·Tf)s + Ki).
func TestTunePIDZeroNotchBelowBand(t *testing.T) {
	for _, c := range []struct{ num, den []float64 }{
		{[]float64{-3.8562437633148288}, []float64{1, 2.38218255046348, 0.43053693845881924, 0}},
		{[]float64{6.021719327000207}, []float64{1, 3.3266544288844027, 0.13253243642255236, 0}},
	} {
		r, err := TunePID(context.Background(), makePlant(t, c.num, c.den), PidtunePIDF, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%v: %v", c.num, err)
		}
		k := r.Controller
		wc := r.Evidence.AchievedCrossover
		eval := func(p []float64, s complex128) complex128 {
			var v complex128
			for _, x := range p {
				v = v*s + complex(x, 0)
			}
			return v
		}
		loop := func(w float64) complex128 {
			s := complex(0, w)
			return eval(c.num, s) / eval(c.den, s) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(complex(k.Tf, 0)*s+1))
		}
		const n = 400000
		peak, crossings := 0., 0
		prev := cmplx.Abs(loop(1e-4*wc)) - 1
		for i := 1; i <= n; i++ {
			w := 1e-4 * wc * math.Pow(1e8, float64(i)/n)
			l := loop(w)
			peak = math.Max(peak, 1/cmplx.Abs(1+l))
			d := cmplx.Abs(l) - 1
			if math.Signbit(d) != math.Signbit(prev) {
				crossings++
				m := math.Abs(180 + cmplx.Phase(l)*180/math.Pi)
				if m < 57-.1 {
					t.Errorf("%v: 0 dB crossing at %.4g rad/s with margin %.3g°", c.num, w, m)
				}
			}
			prev = d
		}
		if crossings != 1 || len(r.Evidence.GainCrossovers) != 1 {
			t.Errorf("%v: %d crossings, evidence %v", c.num, crossings, r.Evidence.GainCrossovers)
		}
		if peak > 2 {
			t.Errorf("%v: |S| peak %.3g", c.num, peak)
		}
		ch := polyMul(c.den, []float64{k.Tf, 1, 0})
		nc := polyMul(c.num, []float64{k.Kd + k.Kp*k.Tf, k.Kp + k.Ki*k.Tf, k.Ki})
		for i := range nc {
			ch[len(ch)-len(nc)+i] += nc[i]
		}
		if !routhStable(ch) {
			t.Errorf("%v: closed loop %v unstable", c.num, ch)
		}
		a, b := k.Kd+k.Kp*k.Tf, k.Kp+k.Ki*k.Tf
		if slowest := math.Sqrt(k.Ki / a); b*b < 4*a*k.Ki && r.Evidence.FrequencyBand[0] > slowest/10*1.0001 {
			t.Errorf("%v: band %v misses zero scale %.4g", c.num, r.Evidence.FrequencyBand, slowest)
		}
	}
}

// The zero grid reaches a decade below the slowest controller zero and
// brackets a lightly damped pair σ ± jω₀ within 4|σ|.
func TestPIDTuningZeroGrid(t *testing.T) {
	p := pidTuningPlant{at: func(w float64) complex128 { return 1 / complex(1, w) }}
	grid := []float64{1, 10, 100}
	// Kd s² + Kp s + Ki = s² + 0.002 s + 1e-4: zeros −0.001 ± j·0.00995.
	c := PID2{Kp: .002, Ki: 1e-4, Kd: 1, B: 1, C: 1}
	cache := map[float64]complex128{}
	got := pidTuningZeroGrid(p, c, grid, math.Log(10)/60, cache)
	if got[0] > .001*1.0001 {
		t.Fatalf("grid starts at %g, above a tenth of the zero scale 0.01", got[0])
	}
	w0, s := math.Sqrt(1e-4-1e-6), .001
	below, above := 0, 0
	for _, w := range got {
		if w >= w0-4*s && w < w0 {
			below++
		}
		if w > w0 && w <= w0+4*s {
			above++
		}
	}
	if below < 4 || above < 4 {
		t.Fatalf("pair at %g bracketed by %d below, %d above", w0, below, above)
	}
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("grid not increasing at %d", i)
		}
	}
	if len(cache) == 0 {
		t.Fatal("extension samples not cached")
	}
	if g := pidTuningZeroGrid(p, PID2{Kp: 1, Ki: 20, B: 1, C: 1}, grid, math.Log(10)/60, cache); len(g) != len(grid) {
		t.Fatalf("zero at 20, a tenth of it above a tenth of the band start changed the grid: %v", g)
	}
}
