package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDelayNyquistAxisCluster: open-loop poles crowding the axis made the
// delay Nyquist test report a closed-loop root on the axis or fail. A
// defective pair ±j split by rounding into one RHP and one LHP eigenvalue
// left no indentation (YJINK6/A), as did a near-coincident pair (YJINK6/B);
// tiny real poles next to integrators fell inside the integrator indentation
// and broke its growth check (5IT7DX, RMVV47). Each loop is stable per the
// argument-principle oracle on χ = den + num·e^{-τs}.
func TestDelayNyquistAxisCluster(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den []float64
		tau      float64
	}{
		{"YJINK6/A", []float64{4, 8, 8, 4, 0}, []float64{1, 1, 2, 2, 1, 1}, 0.01},
		{"YJINK6/B", []float64{1.7733920073898735, 0.4376675554475584, 0.0633521995049906, -0.3280049239511615, -0.17086077595238047}, []float64{1, 0.41818100387501744, 1.296612937471446, 0.5422200140535021, 0.42030127740454004, 0.17576273247407195}, 0.5412512384391044},
		{"5IT7DX/A", []float64{2.068398001856945, 0.8594606031323779, 0.06137818585641332}, []float64{1, -0.0002458302753966347, -9.793006777603344e-11, 0}, 0.030620080660797796},
		{"5IT7DX/B", []float64{0.6384213036340597, 0.14090783238879181, 0.005884832216235019}, []float64{1, -0.0001811526579898724, 9.178212441738005e-11, 0}, 0.0010160268481930203},
		{"5IT7DX/C", []float64{0.7803703188031718, 0.21618757769671687, 0.02539130523001864, 0.0010756344031647744}, []float64{1, 7.897042906351579e-07, 1.3910090146822504e-13, 0, 0}, 0.010047134447323788},
		{"RMVV47", []float64{1.0220019540306915, 2.4805485191105365, 1.1305200275357294, 0.01821221818505486}, []float64{1, -0.017718929845052005, -1.4494897693008775e-10, 0, 0}, 0.0044056653976761095},
	} {
		t.Run(c.name, func(t *testing.T) {
			q := quasiLoop{num: c.num, den: c.den, tau: c.tau}
			if n := oracleRHPRoots(t, q.chi, 200, q.den, q.num); n != 0 {
				t.Fatalf("oracle %d RHP roots", n)
			}
			p := makePlant(t, c.num, c.den)
			if err := p.SetInputDelay([]float64{c.tau}); err != nil {
				t.Fatal(err)
			}
			l, err := delayLoopFromSystem(p, "delay loop")
			if err != nil {
				t.Fatal(err)
			}
			res, err := l.nyquist(0.5)
			if err != nil || !res.stable || res.roots != 0 {
				t.Fatalf("nyquist roots=%d stable=%v axisRoot=%v err=%v", res.roots, res.stable, res.axisRoot, err)
			}
			checkStableFeedback(t, p, nil)
		})
	}
}

// TestDelayNyquistOriginClusterParallelDelays (3NHTS4): s³ with poles at
// 1.3e-3 and -1.4e-6 under a two-delay controller. The integrator
// indentation's growth check failed and IsStable errored; the oracle on
// χ = den + num·K(s)·e^{-(τin+τout)s} counts no RHP root.
func TestDelayNyquistOriginClusterParallelDelays(t *testing.T) {
	num := []float64{4.142627854570593, 1.2271211903141908, 0.11276344373801922, 0.003433033463902915, 3.306252694392361e-05}
	den := []float64{1, -0.0013350213545146655, -1.8267006382290283e-09, 0, 0, 0}
	const tin, tout = 0.030755962059159137, 0.10588402844719585
	k1, t1, k2, t2 := 1.2938430639742877, 0.13168802070035582, 0.07787111691968329, 0.07929680193337156
	chi := func(s complex128) complex128 {
		e := func(tau float64) complex128 { return cmplx.Exp(-s * complex(tau, 0)) }
		k := complex(k1, 0)*e(t1) + complex(k2, 0)*e(t2)
		return polyEval(den, s) + polyEval(num, s)*k*e(tin+tout)
	}
	if n := oracleRHPRoots(t, chi, 200, den, num); n != 0 {
		t.Fatalf("oracle %d RHP roots", n)
	}
	p := makePlant(t, num, den)
	if err := p.SetInputDelay([]float64{tin}); err != nil {
		t.Fatal(err)
	}
	if err := p.SetOutputDelay([]float64{tout}); err != nil {
		t.Fatal(err)
	}
	g := func(k, tau float64) *System {
		s, err := NewGain(mat.NewDense(1, 1, []float64{k}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
		return s
	}
	k, err := Parallel(g(k1, t1), g(k2, t2))
	if err != nil {
		t.Fatal(err)
	}
	checkStableFeedback(t, p, k)
}

func checkStableFeedback(t *testing.T, p, k *System) {
	t.Helper()
	cl, err := Feedback(p, k, -1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil || !stable {
		t.Fatalf("IsStable = %v, %v; oracle stable", stable, err)
	}
	peak, _, err := HinfNorm(cl)
	if err != nil || math.IsInf(peak, 0) || math.IsNaN(peak) {
		t.Fatalf("HinfNorm = %g, %v; want finite", peak, err)
	}
}
