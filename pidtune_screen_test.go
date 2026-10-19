package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A plant with an internal delay feedback cycle is outside both stability
// certificates; Pidtune must return the closed-form design as before the
// screen, not ErrContinuousInternalDelay.
func TestPidtuneInternalDelayCycleUnscreened(t *testing.T) {
	g := makePlant(t, []float64{1}, []float64{1, 1})
	if err := g.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	k, err := NewGain(mat.NewDense(1, 1, []float64{0.2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Feedback(g, k, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !p.HasInternalDelay() {
		t.Fatal("plant has no internal delay")
	}
	pid, err := Pidtune(p, PidtunePI, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(pid.Kp-1.40856) > 1e-4 || math.Abs(pid.Ki-1.28885) > 1e-4 {
		t.Fatalf("Kp=%g Ki=%g, want 1.40856, 1.28885", pid.Kp, pid.Ki)
	}
}

// pidMarkovTail must bound |P(s)| and |s·P(s)| on |s| = w, Re s ≥ 0, and
// decay with the relative degree r (checked for r ≤ 3, above which
// roundoff Markov terms dominate). Oracle: P = num/den evaluated directly on
// the arc, realization scrambled by a non-orthogonal similarity.
func TestPIDMarkovTailBoundsArc(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for trial := range 60 {
		n := 2 + trial%5
		r := 1 + trial%n
		den := Poly(make([]float64, n+1))
		den[0] = 1
		for i := 1; i <= n; i++ {
			den[i] = 4 * (rng.Float64() - 0.3)
		}
		num := Poly(make([]float64, n-r+1))
		for i := range num {
			num[i] = 2*rng.Float64() - 1
		}
		num[0] = math.Copysign(0.2+math.Abs(num[0]), num[0])
		plant := scrambledPlant(t, rng, num, den)
		tail := newPIDMarkovTail(plant)
		normA := min(tail.reals[0].normA, tail.reals[1].normA)
		for _, f := range []float64{1.01, 1.3, 2, 5, 30, 1e3} {
			w := f * normA
			b0, b1 := tail.at(w, 0), tail.at(w, 1)
			for k := 0; k <= 64; k++ {
				s := cmplx.Rect(w, math.Pi/2*(1-float64(k)/32))
				g := num.Eval(s) / den.Eval(s)
				if a := cmplx.Abs(g); a > b0*(1+1e-9) {
					t.Fatalf("trial %d: |P(%v)| = %g > bound %g", trial, s, a, b0)
				}
				if a := cmplx.Abs(s * g); a > b1*(1+1e-9) {
					t.Fatalf("trial %d: |sP(%v)| = %g > bound %g", trial, s, a, b1)
				}
			}
		}
		if r > 3 {
			continue
		}
		w := 1e3 * normA
		want := math.Abs(num[0]) / math.Pow(w, float64(r-1))
		if got := tail.at(w, 1); got > 1.1*want {
			t.Errorf("trial %d (n=%d, r=%d): |sP| bound %g at w=%g, want ≈ %g", trial, n, r, got, w, want)
		}
	}
}

func scrambledPlant(t *testing.T, rng *rand.Rand, num, den Poly) *System {
	t.Helper()
	p := makePlant(t, num, den)
	n, _, _ := p.Dims()
	T := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			T.Set(i, j, 0.4*(rng.Float64()-0.5))
		}
		T.Set(i, i, T.At(i, i)+1)
	}
	var Ti, a, ta, b, c mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	ta.Mul(T, p.A)
	a.Mul(&ta, &Ti)
	b.Mul(T, p.B)
	c.Mul(p.C, &Ti)
	s, err := New(&a, &b, &c, p.D, 0)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// The Pidtune PD design on (0.3s−1)/(s+1)⁴·e^{−5s} at its automatic
// crossover, given explicitly since the automatic design's |S| 7.6 makes it
// error (IBWZGV), has a filter pole at
// 1/Tf ≈ 42 rad/s. The Nyquist grid needs delay spacing only where |L| can
// reach 1 (ω ≲ 2), so the screen certifies it within 2000 points; spacing to
// the filter pole needs > 5000. Oracle: dense winding of 1+L on [0, 10³].
func TestPidtuneDelayedScreenGridBudget(t *testing.T) {
	num, den := Poly{0.3, -1}, Poly{1, 4, 6, 4, 1}
	q := makePlant(t, num, den)
	if err := q.SetInputDelay([]float64{5}); err != nil {
		t.Fatal(err)
	}
	wc, err := findCrossoverFreq(q, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := Pidtune(q, PidtunePD, wc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Tf <= 0 || 1/pid.Tf < 20 {
		t.Fatalf("design %+v lacks a fast filter pole", pid)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		c := complex(pid.Kp, 0) + complex(pid.Kd, 0)*s/(1+complex(pid.Tf, 0)*s)
		return c * num.Eval(s) / den.Eval(s) * cmplx.Exp(-5*s)
	}
	phase, prev := 0.0, 1+L(0)
	for w := 1e-6; w < 1e3; w = math.Min(w*1.001, w+1e-2) {
		f := 1 + L(w)
		step := cmplx.Phase(f / prev)
		if math.Abs(step) > math.Pi/8 {
			t.Fatalf("oracle grid too coarse at ω=%g", w)
		}
		phase += step
		prev = f
	}
	if turns := phase / math.Pi; math.Abs(turns) > 0.1 {
		t.Fatalf("oracle: 1+L winds %g half-turns, design not stable", turns)
	}

	eval, err := newSISOEval(q)
	if err != nil {
		t.Fatal(err)
	}
	stable, ok, err := pidTuningDelayStabilityBudget(q, eval, 0.2, false, 1, 2000)
	if err != nil || !ok {
		t.Fatalf("out of scope: %v", err)
	}
	got, err := stable(&PID2{Kp: pid.Kp, Kd: pid.Kd, Tf: pid.Tf})
	if err != nil || !got {
		t.Fatalf("screen within 2000 points: stable=%v err=%v, want stable", got, err)
	}
}
