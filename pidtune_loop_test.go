package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidtuneLoopPlant returns a random SISO plant of order n with sample time
// dt, scrambled by a non-orthogonal similarity so A is not Hessenberg, and
// with feedthrough d.
func pidtuneLoopPlant(t *testing.T, rng *rand.Rand, n int, dt, d float64) *System {
	t.Helper()
	if n == 0 {
		s, err := NewGain(mat.NewDense(1, 1, []float64{d}), dt)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		for j := range n {
			a.Set(i, j, 2*(rng.Float64()-0.5))
		}
		a.Set(i, i, a.At(i, i)-0.5)
		b.Set(i, 0, 2*rng.Float64()-1)
		c.Set(0, i, 2*rng.Float64()-1)
	}
	if dt > 0 {
		a.Scale(0.5, a)
	}
	s, err := New(a, b, c, mat.NewDense(1, 1, []float64{d}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// pidtuneLoopController returns a random controller of kind k (P, I, PI,
// PDF, PIDF, ideal PD) sampled at dt.
func pidtuneLoopController(rng *rand.Rand, k int, dt float64) PID2 {
	g := func() float64 { return 4 * (rng.Float64() - 0.5) }
	f := []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal}
	c := PID2{Kp: g(), Dt: dt, IFormula: f[rng.IntN(3)], DFormula: f[rng.IntN(3)]}
	switch k {
	case 1:
		c.Kp, c.Ki = 0, g()
	case 2:
		c.Ki = g()
	case 3:
		c.Kd, c.Tf = g(), 0.01+rng.Float64()
	case 4:
		c.Ki, c.Kd, c.Tf = g(), g(), 0.01+rng.Float64()
	case 5:
		c.Kd = g()
	}
	return c
}

// pidtuneLoop must give closedLoopPoles's verdict and, when it forms the
// loop itself, the eigenvalues of Feedback(model, C, −1) bit for bit.
// Oracle: the generic Feedback and IsStable path it replaces, on scrambled
// plants with feedthrough, discrete delays, every controller structure
// (fallbacks included) and ill-posed loops.
func TestPidtuneLoopMatchesFeedback(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 29))
	fast, unstable := 0, 0
	for trial := range 400 {
		n := trial % 6
		dt := 0.0
		if trial%3 == 1 {
			dt = 0.1
		}
		d := 0.0
		if trial%4 == 2 {
			d = 2*rng.Float64() - 1
		}
		plant := pidtuneLoopPlant(t, rng, n, dt, d)
		if dt > 0 && trial%5 == 1 {
			if err := plant.SetInputDelay([]float64{2}); err != nil {
				t.Fatal(err)
			}
		}
		eval, err := newSISOEval(plant)
		if err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(plant, eval, 1)
		if err != nil {
			t.Fatal(err)
		}
		loop := newPidtuneLoop(screen.model)
		if loop == nil {
			t.Fatalf("trial %d: no loop for a delay-free standard model", trial)
		}
		for k := range 12 {
			c := pidtuneLoopController(rng, k%6, dt)
			if k == 11 && d != 0 {
				c = PID2{Kp: -1 / d, Dt: dt}
			}
			want, wantErr := screen.closedLoopPoles(&c)
			loop.poles = nil
			got, gotErr := loop.stable(screen, &c)
			if got != want || (gotErr == nil) != (wantErr == nil) || gotErr != nil && gotErr.Error() != wantErr.Error() {
				t.Fatalf("trial %d, %+v: stable = %v, %v; want %v, %v", trial, c, got, gotErr, want, wantErr)
			}
			if !want {
				unstable++
			}
			if len(loop.poles) == 0 {
				continue
			}
			fast++
			cs, err := (&PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt, IFormula: c.IFormula, DFormula: c.DFormula}).System()
			if err != nil {
				t.Fatal(err)
			}
			cl, err := Feedback(screen.model, cs, -1)
			if err != nil {
				t.Fatal(err)
			}
			ref, err := cl.Poles()
			if err != nil {
				t.Fatal(err)
			}
			if len(ref) != len(loop.poles) {
				t.Fatalf("trial %d, %+v: %d poles, want %d", trial, c, len(loop.poles), len(ref))
			}
			for i, p := range ref {
				q := loop.poles[i]
				if math.Float64bits(real(p)) != math.Float64bits(real(q)) || math.Float64bits(imag(p)) != math.Float64bits(imag(q)) {
					t.Fatalf("trial %d, %+v: pole %d = %v, want %v", trial, c, i, q, p)
				}
			}
		}
	}
	if fast < 2000 || unstable < 500 {
		t.Fatalf("corpus too thin: %d fast-path loops, %d unstable", fast, unstable)
	}
}

// Models outside pidtuneLoop's scope get no loop: Pidtune then screens
// with closedLoopPoles or the Nyquist test.
func TestPidtuneLoopScope(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 1))
	delayed := pidtuneLoopPlant(t, rng, 2, 0, 0)
	if err := delayed.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	descriptor := pidtuneLoopPlant(t, rng, 2, 0, 0)
	descriptor.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	for name, s := range map[string]*System{"delayed": delayed, "descriptor": descriptor} {
		if newPidtuneLoop(s) != nil {
			t.Errorf("%s: got a loop", name)
		}
	}
}

// pidtunePlantAt must return evalPlantAt bit for bit, at a pole (the pole
// limit), repeated frequencies, with I/O delays and on descriptor and
// internally delayed models (which fall back).
func TestPidtunePlantAtMatchesEvalPlantAt(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 13))
	integ := makePlant(t, []float64{1, 2}, []float64{1, 0, 4})
	plants := []*System{integ}
	for trial := range 60 {
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		s := pidtuneLoopPlant(t, rng, 1+trial%7, dt, float64(trial%3)/2)
		switch trial % 5 {
		case 1:
			delay := 0.3
			if dt > 0 {
				delay = 3
			}
			if err := s.SetInputDelay([]float64{delay}); err != nil {
				t.Fatal(err)
			}
		case 2:
			if dt == 0 {
				n, _, _ := s.Dims()
				e := eyeDense(n)
				e.Set(0, 0, 1.5)
				s.E = e
			}
		case 3:
			k, err := NewGain(mat.NewDense(1, 1, []float64{0.2}), dt)
			if err != nil {
				t.Fatal(err)
			}
			delay := 0.4
			if dt > 0 {
				delay = 2
			}
			if err := s.SetInputDelay([]float64{delay}); err != nil {
				t.Fatal(err)
			}
			if s, err = Feedback(s, k, -1); err != nil {
				t.Fatal(err)
			}
		}
		plants = append(plants, s)
	}
	for i, s := range plants {
		at := pidtunePlantAt(s)
		ws := []float64{2, 0.5, 0.5, 1.3, 2, 1e-3, 7}
		if s.Dt > 0 {
			ws = append(ws, 31)
		}
		for _, w := range ws {
			want, wantErr := evalPlantAt(s, w)
			got, gotErr := at(w)
			if (gotErr == nil) != (wantErr == nil) || gotErr != nil && gotErr.Error() != wantErr.Error() {
				t.Fatalf("plant %d, w=%g: err %v, want %v", i, w, gotErr, wantErr)
			}
			if math.Float64bits(real(got)) != math.Float64bits(real(want)) || math.Float64bits(imag(got)) != math.Float64bits(imag(want)) {
				t.Fatalf("plant %d, w=%g: %v, want %v", i, w, got, want)
			}
		}
	}
}

// Every back-off step of an automatic I design on 1/((s−2)(s+1)) is
// closed-loop unstable. Reusing the loop buffers keeps the ~61 failing
// designs within a few allocations each (about 10000 before).
func TestPidtuneScreenBoundBackoffAllocs(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, -1, -2})
	if _, err := Pidtune(plant, PidtuneI, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("err = %v, want ErrPIDTuningTargetUnattainable", err)
	}
	if a := testing.AllocsPerRun(5, func() { _, _ = Pidtune(plant, PidtuneI, 0, nil) }); a > 1500 {
		t.Fatalf("%g allocations, want <= 1500", a)
	}
}
