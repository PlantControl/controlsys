package controlsys

import (
	"math/big"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 4FAOGG: G = (0.1z+0.05)/((z−1)(z−0.999)²), Dt = 0.1. A PI at a low explicit
// crossover leaves a closed-loop pair ~2e-7 inside z = 1, while float
// eigenvalues of the clustered loop err by ~3e-6 and put one outside.
// Oracle: Schur–Cohn in exact rationals on den·denC + num·numC.
func TestPidtuneClusteredPIExplicitStable(t *testing.T) {
	const dt = .1
	num, den := Poly{0.1, 0.05}, Poly{1, -1}.Mul(Poly{1, -0.999}).Mul(Poly{1, -0.999})
	pos, neg := negatedPlant(t, num, den, dt)
	negNum := Poly{-num[0], -num[1]}
	for _, wc := range []float64{5.32e-5, 2.99e-5, 1.68e-5, 5.32e-6} {
		for _, c := range []struct {
			plant *System
			num   Poly
		}{{pos, num}, {neg, negNum}} {
			pid, err := Pidtune(c.plant, PidtunePI, wc, nil)
			if err != nil {
				t.Fatalf("wc=%g: %v", wc, err)
			}
			if !pidtuneExactSchurStable(t, pid, c.num, den, dt) {
				t.Errorf("wc=%g: %+v unstable", wc, pid)
			}
		}
	}
}

// Both pole screens must agree with the exact closed-loop roots across PI
// gains scaled around the low-crossover designs, on either side of the
// Kp sign change where the pair near z = 1 leaves the circle. With negative
// Ki (Y6PD4H) float eigenvalues also put a real root ~4e-7 outside z = 1
// back inside.
func TestPIDStabilityScreenClusteredMatchesExact(t *testing.T) {
	const dt = .1
	num, den := Poly{0.1, 0.05}, Poly{1, -1}.Mul(Poly{1, -0.999}).Mul(Poly{1, -0.999})
	pos, _ := negatedPlant(t, num, den, dt)
	screen, err := newPIDStabilityScreen(pos, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	loop := newPidtuneLoop(pos)
	if loop == nil {
		t.Fatal("no pidtuneLoop")
	}
	var stable, unstable int
	for _, wc := range []float64{3e-4, 5.32e-5, 5.32e-6} {
		pid0, err := computePIDGains(pos, PidtunePI, wc, 60, 1, true)
		if err != nil {
			t.Fatal(err)
		}
		for _, fi := range []float64{-1, -0.01, 0.01, 1, 100, 1e4} {
			for k := -20; k <= 20; k++ {
				pid := *pid0
				pid.Kp *= float64(k)/10 + 0.05
				pid.Ki *= fi
				want := pidtuneExactSchurStable(t, &pid, num, den, dt)
				c := PID2{Kp: pid.Kp, Ki: pid.Ki, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula}
				got, err := screen.closedLoopPoles(&c)
				if err != nil {
					t.Fatal(err)
				}
				fast, err := loop.stable(screen, &c)
				if err != nil {
					t.Fatal(err)
				}
				if got != want || fast != want {
					t.Errorf("wc=%g Kp=%g Ki=%g: screen %v, pidtuneLoop %v, exact %v", wc, pid.Kp, pid.Ki, got, fast, want)
				}
				if want {
					stable++
				} else {
					unstable++
				}
			}
		}
	}
	if stable == 0 || unstable == 0 {
		t.Fatalf("sweep one-sided: %d stable, %d unstable", stable, unstable)
	}
}

// Y6PD4H: random discrete plants with 2–4 poles clustered near z = 1 and PI
// gains around their low-crossover designs. Both screens must match the
// exact roots except for a root within poleStabilityTolerance of the
// circle, which they reject as IsStable does.
func TestPIDStabilityScreenRandomClustersMatchExact(t *testing.T) {
	const dt = .1
	rng := rand.New(rand.NewPCG(1, 2))
	var checked, unstable int
	for range 80 {
		den := Poly{1}
		base := 1 - rng.Float64()*1e-2
		for range 2 + rng.IntN(3) {
			den = den.Mul(Poly{1, -(base - rng.Float64()*1e-3*float64(rng.IntN(2)))})
		}
		num := Poly{rng.Float64()*0.2 - 0.05, 0.05}
		plant, _ := negatedPlant(t, num, den, dt)
		screen, err := newPIDStabilityScreen(plant, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		loop := newPidtuneLoop(plant)
		for _, wc := range []float64{1e-2, 1e-3, 1e-4, 1e-5} {
			pid0, err := computePIDGains(plant, PidtunePI, wc, 60, 1, true)
			if err != nil {
				continue
			}
			for range 10 {
				pid := *pid0
				pid.Kp *= rng.Float64()*4 - 2
				pid.Ki *= (rng.Float64()*4 - 2) * []float64{1, 0.01, 100}[rng.IntN(3)]
				want := pidtuneExactSchurStable(t, &pid, num, den, dt)
				c := PID2{Kp: pid.Kp, Ki: pid.Ki, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula}
				got, err := screen.closedLoopPoles(&c)
				if err != nil {
					t.Fatal(err)
				}
				fast, err := loop.stable(screen, &c)
				if err != nil {
					t.Fatal(err)
				}
				checked++
				if !want {
					unstable++
				}
				if got == want && fast == want {
					continue
				}
				if want && !got && !fast && !pidtuneExactSchurStableRadius(t, &pid, num, den, dt, 1-1e-10) {
					continue
				}
				t.Errorf("den=%v Kp=%g Ki=%g: screen %v, pidtuneLoop %v, exact %v", den, pid.Kp, pid.Ki, got, fast, want)
			}
		}
	}
	if unstable == 0 || unstable == checked {
		t.Fatalf("sweep one-sided: %d of %d unstable", unstable, checked)
	}
}

func bigInts(v ...int64) []*big.Int {
	out := make([]*big.Int, len(v))
	for i, x := range v {
		out[i] = big.NewInt(x)
	}
	return out
}

func bigEqual(a, b []*big.Int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Cmp(b[i]) != 0 {
			return false
		}
	}
	return true
}

func TestIntCharPoly(t *testing.T) {
	for _, c := range []struct {
		m    [][]int64
		want []int64
	}{
		{[][]int64{{5}}, []int64{1, -5}},
		// tr 2, principal 2×2 minors −7, 2, −22, det −34.
		{[][]int64{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}}, []int64{1, -2, -27, 34}},
		// Companion of λ⁴+2λ³−3λ²+5λ−7.
		{[][]int64{{-2, 3, -5, 7}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}, []int64{1, 2, -3, 5, -7}},
	} {
		m := make([][]*big.Int, len(c.m))
		for i, row := range c.m {
			m[i] = bigInts(row...)
		}
		if got := intCharPoly(m); !bigEqual(got, bigInts(c.want...)) {
			t.Errorf("%v: got %v, want %v", c.m, got, c.want)
		}
	}
}

func TestIntShift(t *testing.T) {
	// (x−3)(x+1) at x+2 is (x−1)(x+3).
	if got := intShift(bigInts(1, -2, -3), big.NewInt(2)); !bigEqual(got, bigInts(1, 2, -3)) {
		t.Errorf("got %v", got)
	}
}

func TestIntSchurHurwitz(t *testing.T) {
	for _, c := range []struct {
		p    []int64
		want bool
	}{
		{[]int64{4, 0, -1}, true},   // ±1/2
		{[]int64{2, -1, -1}, false}, // 1, −1/2
		{[]int64{1, 1, 1}, false},   // |z| = 1
		{[]int64{2, -5, 2}, false},  // 2, 1/2
		{[]int64{-8, 4, 2, -1}, true},
	} {
		if got := intSchur(bigInts(c.p...)); got != c.want {
			t.Errorf("Schur %v: got %v", c.p, got)
		}
	}
	for _, c := range []struct {
		p    []int64
		want bool
	}{
		{[]int64{1, 3, 3, 9}, false}, // −3, ±j√3
		{[]int64{10, 30, 30, 89}, true},
		{[]int64{10, 30, 30, 91}, false},
		{[]int64{1, 0, 1}, false},
		{[]int64{-1, -3, -2}, true},
		{[]int64{1, 2, -3}, false},
	} {
		if got := intHurwitz(bigInts(c.p...)); got != c.want {
			t.Errorf("Hurwitz %v: got %v", c.p, got)
		}
	}
}

// Continuous loops with known closed-loop poles: K/(s+1)³ is stable for
// K < 8 with poles ±j√3 at K = 8, and K·(s+2)/(s+1) has its pole at
// −(1+2K)/(1+K), with 1+D1D2 = 1+K negative for K < −1.
func TestPIDTuningExactStableContinuous(t *testing.T) {
	cube := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	lead := &System{A: mat.NewDense(1, 1, []float64{-1}), B: mat.NewDense(1, 1, []float64{1}), C: mat.NewDense(1, 1, []float64{1}), D: mat.NewDense(1, 1, []float64{1})}
	for _, c := range []struct {
		plant  *System
		k      float64
		want   bool
		wantOK bool
	}{
		{cube, 7.9, true, true},
		{cube, 8 - 1e-6, true, true},
		{cube, 8, false, true},
		{cube, 8.1, false, true},
		{lead, -0.4, true, true},
		{lead, -0.75, false, true},
		{lead, -1.5, true, true},
		{lead, -1, false, false},
	} {
		cs, err := (&PID{Kp: c.k}).System()
		if err != nil {
			t.Fatal(err)
		}
		got, ok := pidTuningExactStable(c.plant, cs)
		if got != c.want || ok != c.wantOK {
			t.Errorf("K=%g: got %v,%v want %v,%v", c.k, got, ok, c.want, c.wantOK)
		}
	}
}

// Discrete: K/(z(z−0.5)) has det(z²−0.5z+K), stable for −0.5 < K < 1 with
// |z| = 1 at K = 1.
func TestPIDTuningExactStableDiscrete(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, -0.5, 0})
	plant.Dt = 1
	for _, c := range []struct {
		k    float64
		want bool
	}{{0.99, true}, {1, false}, {1.01, false}, {-0.49, true}, {-0.5, false}} {
		cs, err := (&PID{Kp: c.k, Dt: 1}).System()
		if err != nil {
			t.Fatal(err)
		}
		if got, ok := pidTuningExactStable(plant, cs); got != c.want || !ok {
			t.Errorf("K=%g: got %v,%v want %v", c.k, got, ok, c.want)
		}
	}
}

// Fast-sampled plants put every open-loop pole near z = 1; the 5-fold pole
// is a true cluster the exact test must decide, the 2-pole plant is not.
func BenchmarkPidtuneFastSampled(b *testing.B) {
	for _, c := range []struct {
		name string
		den  []float64
	}{{"O2", []float64{1, 3, 2}}, {"O5", []float64{1, 5, 10, 10, 5, 1}}} {
		r, err := (&TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{c.den}}).StateSpace()
		if err != nil {
			b.Fatal(err)
		}
		d, err := r.Sys.C2D(1e-3, C2DOptions{})
		if err != nil {
			b.Fatal(err)
		}
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				if _, err := Pidtune(d, PidtunePI, 0, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
