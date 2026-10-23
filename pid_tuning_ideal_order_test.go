package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidtuneHiddenModesPlant returns the discrete plant num/den with the
// uncontrollable modes appended to its realization, so the loop
// characteristic is that of num/den times ∏(z−mode); a mode with positive
// imaginary part adds its conjugate pair.
func pidtuneHiddenModesPlant(t *testing.T, num, den Poly, dt float64, modes []complex128) *System {
	t.Helper()
	g := pidtuneBackoffPlant(t, num, den, dt, 0)
	n0, _, _ := g.Dims()
	n := n0
	for _, m := range modes {
		n++
		if imag(m) > 0 {
			n++
		}
	}
	a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n0 {
		for j := range n0 {
			a.Set(i, j, g.A.At(i, j))
		}
		b.Set(i, 0, g.B.At(i, 0))
		c.Set(0, i, g.C.At(0, i))
	}
	i := n0
	for _, m := range modes {
		a.Set(i, i, real(m))
		c.Set(0, i, 0.3)
		if imag(m) > 0 {
			a.Set(i, i+1, imag(m))
			a.Set(i+1, i, -imag(m))
			a.Set(i+1, i+1, real(m))
			c.Set(0, i+1, 0.3)
			i++
		}
		i++
	}
	p, err := New(a, b, c, mat.NewDense(1, 1, []float64{g.D.At(0, 0)}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// 5UUEZD: above pidTuningExactOrder the screen took the float verdict on
// the transfer's characteristic whenever the transfer kept full degree, so
// a corrupted full-degree transfer decided the loop. The closed-loop
// realization now decides there and the transfer is never consulted.
// Oracle: exact Schur–Cohn with z^15.
func TestPIDTuningIdealStableHighOrderFullDegreeTransfer(t *testing.T) {
	const dt, lag = 0.1, 15
	numG, denG := Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}
	numH, denH := Poly{1}, Poly{1}
	for i := range 16 {
		b := 0.1 + 0.03*float64(i)
		numH, denH = numH.Mul(Poly{1, -(b + 0.005)}), denH.Mul(Poly{1, -b})
	}
	num, den := numG.Mul(numH), denG.Mul(denH)
	plant := pidtuneBackoffPlant(t, num, den, dt, 0)
	n, _, _ := plant.Dims()
	if n <= pidTuningExactOrder {
		t.Fatalf("order %d within pidTuningExactOrder", n)
	}
	wrong := func(k float64) func() (*TransferFuncResult, error) {
		return func() (*TransferFuncResult, error) {
			return &TransferFuncResult{TF: &TransferFunc{Num: [][][]float64{{num.Mul(Poly{k})}}, Den: [][]float64{den}, Dt: dt}}, nil
		}
	}
	for _, tc := range []struct {
		k    PID2
		want bool
	}{
		{PID2{Kp: 0.5706220540940695, Ki: 1.0390660705947083, Kd: 0.3000376123041945}, false},
		{PID2{Kp: 0.02, Ki: 0.01, Kd: 0.005}, true},
	} {
		k := tc.k
		k.B, k.C, k.Dt = 1, 1, dt
		if pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag) != tc.want {
			t.Fatalf("%+v: oracle not %v", k, tc.want)
		}
		for _, scale := range []float64{0.01, 100} {
			if got, err := pidTuningIdealStable(plant, wrong(scale), &k, lag); err != nil || got != tc.want {
				t.Errorf("gain ×%g %+v: stable %v %v, oracle %v", scale, k, got, err, tc.want)
			}
		}
	}
}

// 5UUEZD: past pidTuningExactLagOrder states the screen needed the
// transfer function and returned its error; the closed-loop realization
// now decides. The 70 hidden modes, spread on |z| = 0.9 or along
// [0.1, 0.79] where only pidTuningEigenDiscVerdict decides, leave the
// verdict to the second-order core. Oracle: exact Schur–Cohn on the core with z^lag.
func TestPIDTuningIdealStableBeyondExactOrder(t *testing.T) {
	const dt = 0.1
	num, den := Poly{0.3, 0.1}, Poly{1, -1.2, 0.5}
	modes := make([]complex128, 35)
	for i := range modes {
		modes[i] = cmplx.Rect(0.9, (float64(i)+0.5)*math.Pi/35)
	}
	real70 := make([]complex128, 70)
	for i := range real70 {
		real70[i] = complex(0.1+0.01*float64(i), 0)
	}
	failed := func() (*TransferFuncResult, error) { return nil, ErrOverflow }
	for _, plant := range []*System{pidtuneHiddenModesPlant(t, num, den, dt, modes), pidtuneHiddenModesPlant(t, num, den, dt, real70)} {
		for _, lag := range []int{0, 5} {
			for _, k := range []PID2{
				{Kp: 0.2, Ki: 0.5, Kd: 0.02},
				{Kp: 1.5, Ki: 2, Kd: 0.3},
				{Kp: 0.1, Kd: 0.01},
				{Kp: 4, Ki: 1, Kd: 0.05},
			} {
				k.B, k.C, k.Dt = 1, 1, dt
				want := pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag)
				if got, err := pidTuningIdealStable(plant, failed, &k, lag); err != nil || got != want {
					t.Errorf("lag %d %+v: stable %v %v, oracle %v", lag, k, got, err, want)
				}
			}
		}
	}
}

// 5UUEZD: a hidden mode at 1−3e-10 puts the float verdict in doubt. Within
// pidTuningExactLagOrder exact Schur–Cohn certifies it (|z| < 1−1e-10);
// beyond it the loop fails closed instead of keeping the float verdict.
func TestPIDTuningIdealStableHighOrderFailClosed(t *testing.T) {
	const dt = 0.1
	num, den := Poly{0.3, 0.1}, Poly{1, -1.2, 0.5}
	modes := []complex128{1 - 3e-10}
	for i := range 16 {
		modes = append(modes, complex(0.2+0.03*float64(i), 0))
	}
	plant := pidtuneHiddenModesPlant(t, num, den, dt, modes)
	k := PID2{Kp: 0.05, Ki: 0.02, Kd: 0.005, B: 1, C: 1, Dt: dt}
	failed := func() (*TransferFuncResult, error) { return nil, ErrOverflow }
	for _, tc := range []struct {
		lag  int
		want bool
	}{{10, true}, {50, false}} {
		if !pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, tc.lag) {
			t.Fatalf("lag %d: core oracle unstable", tc.lag)
		}
		if _, decided := pidTuningIdealRealizationFloat(plant, &k, tc.lag); decided {
			t.Fatalf("lag %d: float verdict decided", tc.lag)
		}
		if got, err := pidTuningIdealStable(plant, failed, &k, tc.lag); err != nil || got != tc.want {
			t.Errorf("lag %d: stable %v %v, want %v", tc.lag, got, err, tc.want)
		}
	}
}

// 5UUEZD: every decided verdict of pidTuningIdealRealizationFloat matches
// exact Schur–Cohn or Routh–Hurwitz on the dyadic characteristic, over
// random non-symmetric plants, continuous and forward-Euler discrete
// derivatives, every integral formula, D ≠ 0 and lags.
func TestPIDTuningIdealRealizationMatchesExact(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(5, 0x5ee2d))
	decided, total := 0, 0
	for trial := range 120 {
		n := 17 + trial%3
		dt := 0.0
		lag := 0
		if trial%2 == 1 {
			dt = 0.1
			lag = []int{0, 1, 3}[trial/2%3]
		}
		a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		for i := range n {
			for j := range n {
				a.Set(i, j, rng.NormFloat64()/math.Sqrt(float64(n)))
			}
			if dt == 0 {
				a.Set(i, i, a.At(i, i)-1.2)
			} else {
				a.Set(i, i, a.At(i, i)*0.3)
			}
			b.Set(i, 0, rng.NormFloat64())
			c.Set(0, i, rng.NormFloat64())
		}
		d := 0.0
		if trial%4 >= 2 {
			d = rng.NormFloat64()
		}
		plant, err := New(a, b, c, mat.NewDense(1, 1, []float64{d}), dt)
		if err != nil {
			t.Fatal(err)
		}
		for g := range 6 {
			k := PID2{
				Kp: math.Pow(10, -2+2*rng.Float64()), Kd: math.Pow(10, -3+2*rng.Float64()),
				B: 1, C: 1, Dt: dt, IFormula: PIDFormula(g % 3),
			}
			if g != 5 {
				k.Ki = math.Pow(10, -2+2*rng.Float64())
			}
			if rng.IntN(2) == 0 {
				k.Kd = -k.Kd
			}
			stable, ok := pidTuningIdealRealizationFloat(plant, &k, lag)
			total++
			if !ok {
				continue
			}
			decided++
			exact, inScope := pidTuningIdealExact(plant, &k, lag)
			if !inScope {
				t.Fatalf("trial %d: exact out of scope", trial)
			}
			if stable != exact {
				t.Errorf("trial %d n %d dt %g lag %d d %g %+v: float %v, exact %v", trial, n, dt, lag, d, k, stable, exact)
			}
		}
	}
	if decided < total*3/4 {
		t.Errorf("decided %d of %d", decided, total)
	}
}

// 5UUEZD: the polynomial error model of pidTuningFloatVerdict leaves loops
// of many well-separated modes in doubt; every verdict
// pidTuningEigenDiscVerdict decides there matches exact Schur–Cohn on the
// dyadic characteristic, over random non-symmetric discrete plants.
func TestPIDTuningEigenDiscVerdictMatchesExact(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(41, 0x5ee2d))
	disc, total, stables := 0, 0, 0
	for trial := range 12 {
		n, lag := 22+trial%3*2, []int{0, 4, 12}[trial/3%3]
		a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		for i := range n {
			for j := range n {
				a.Set(i, j, 0.3*rng.NormFloat64()/math.Sqrt(float64(n)))
			}
			a.Set(i, i, a.At(i, i)+0.6*rng.Float64())
			b.Set(i, 0, rng.NormFloat64())
			c.Set(0, i, rng.NormFloat64())
		}
		plant, err := New(a, b, c, mat.NewDense(1, 1, nil), 0.1)
		if err != nil {
			t.Fatal(err)
		}
		for range 3 {
			k := PID2{Kp: math.Pow(10, -2+1.5*rng.Float64()), Ki: math.Pow(10, -2+1.5*rng.Float64()), Kd: math.Pow(10, -3+1.5*rng.Float64()), B: 1, C: 1, Dt: 0.1}
			am, abs, ok := pidTuningIdealRealization(plant, &k, lag)
			if !ok {
				t.Fatalf("trial %d: realization failed", trial)
			}
			order, _ := am.Dims()
			stable, decided := pidTuningEigenDiscVerdict(am, false, pidTuningPoleErrorScale*float64(order)*0x1p-52*mat.Norm(abs, 2))
			total++
			if !decided {
				continue
			}
			disc++
			if stable {
				stables++
			}
			exact, inScope := pidTuningIdealExact(plant, &k, lag)
			if !inScope {
				t.Fatalf("trial %d: exact out of scope", trial)
			}
			if stable != exact {
				t.Errorf("trial %d n %d lag %d %+v: disc %v, exact %v", trial, n, lag, k, stable, exact)
			}
		}
	}
	if disc < total/2 || stables == 0 || stables == disc {
		t.Errorf("disc verdict decided %d of %d, %d stable", disc, total, stables)
	}
}

// HVXYLI: within pidTuningExactOrder, past pidTuningExactLagOrder, an
// undecided float verdict on the exact characteristic stood, unlike every
// other loop out of exact scope. A hidden mode at 1−9.99e-11, a root of the
// characteristic outside |z| < 1−1e-10, passed as stable at lag 70; one at
// 1−3e-10, which exact Schur–Cohn certifies at lag 10, passed unresolved.
// Both now take the closed-loop realization's float and Bauer–Fike
// verdicts, then fail closed. A robust mode at 0.9 keeps its verdict at
// lag 70, whose oracle is the core's characteristic, as the modes are
// uncontrollable.
func TestPIDTuningIdealStableLowOrderLongLagFailClosed(t *testing.T) {
	const dt = 0.1
	num, den := Poly{0.3, 0.1}, Poly{1, -1.2, 0.5}
	k := PID2{Kp: 0.05, Ki: 0.02, Kd: 0.005, B: 1, C: 1, Dt: dt}
	failed := func() (*TransferFuncResult, error) { return nil, ErrOverflow }
	core := map[int]bool{}
	for _, tc := range []struct {
		mode float64
		lag  int
		want bool
	}{{1 - 9.99e-11, 70, false}, {1 - 3e-10, 10, true}, {1 - 3e-10, 70, false}, {0.9, 70, true}} {
		plant := pidtuneHiddenModesPlant(t, num, den, dt, []complex128{complex(tc.mode, 0), 0.4, 0.6})
		if n, _, _ := plant.Dims(); n > pidTuningExactOrder {
			t.Fatalf("order %d above pidTuningExactOrder", n)
		}
		stable, ok := core[tc.lag]
		if !ok {
			stable = pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, tc.lag)
			core[tc.lag] = stable
		}
		if !stable {
			t.Fatalf("lag %d: core oracle unstable", tc.lag)
		}
		if got, err := pidTuningIdealStable(plant, failed, &k, tc.lag); err != nil || got != tc.want {
			t.Errorf("mode %v lag %d: stable %v %v, want %v", tc.mode, tc.lag, got, err, tc.want)
		}
	}
}

// HVXYLI: pidTuningEigenDiscVerdict, now the second link of every chain
// out of exact scope, never disagrees with exact Schur–Cohn or
// Routh–Hurwitz on the dyadic characteristic where both decide, over random
// non-symmetric plants of every order up to pidTuningExactOrder and beyond,
// continuous and discrete, D ≠ 0 and lags.
func TestPIDTuningEigenDiscVerdictMatchesExactLowOrder(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(97, 0x4f71))
	disc, total, stables := 0, 0, 0
	for trial := range 60 {
		n := 2 + trial%19
		dt, lag := 0.0, 0
		if trial%3 != 0 {
			dt, lag = 0.1, []int{0, 2, 9, 20}[trial%4]
		}
		a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		for i := range n {
			for j := range n {
				a.Set(i, j, 0.3*rng.NormFloat64()/math.Sqrt(float64(n)))
			}
			if dt == 0 {
				a.Set(i, i, a.At(i, i)-0.2-2*rng.Float64())
			} else {
				a.Set(i, i, a.At(i, i)+0.6*rng.Float64())
			}
			b.Set(i, 0, rng.NormFloat64())
			c.Set(0, i, rng.NormFloat64())
		}
		d := 0.0
		if trial%5 == 4 {
			d = 0.3 * rng.NormFloat64()
		}
		plant, err := New(a, b, c, mat.NewDense(1, 1, []float64{d}), dt)
		if err != nil {
			t.Fatal(err)
		}
		for g := range 4 {
			k := PID2{Kp: math.Pow(10, -2+1.5*rng.Float64()), Kd: math.Pow(10, -3+1.5*rng.Float64()), B: 1, C: 1, Dt: dt, IFormula: PIDFormula(g % 3)}
			if g != 3 {
				k.Ki = math.Pow(10, -2+1.5*rng.Float64())
			}
			am, abs, ok := pidTuningIdealRealization(plant, &k, lag)
			if !ok {
				t.Fatalf("trial %d: realization failed", trial)
			}
			order, _ := am.Dims()
			stable, decided := pidTuningEigenDiscVerdict(am, dt == 0, pidTuningPoleErrorScale*float64(order)*0x1p-52*mat.Norm(abs, 2))
			total++
			if !decided {
				continue
			}
			disc++
			if stable {
				stables++
			}
			exact, inScope := pidTuningIdealExact(plant, &k, lag)
			if !inScope {
				t.Fatalf("trial %d: exact out of scope", trial)
			}
			if stable != exact {
				t.Errorf("trial %d n %d dt %g lag %d d %g %+v: disc %v, exact %v", trial, n, dt, lag, d, k, stable, exact)
			}
		}
	}
	if disc < total/2 || stables == 0 || stables == disc {
		t.Errorf("disc verdict decided %d of %d, %d stable", disc, total, stables)
	}
}
