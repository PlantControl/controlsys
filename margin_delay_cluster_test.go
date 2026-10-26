package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func marginTFSystem(t *testing.T, num, den []float64, dt, delay float64) *System {
	t.Helper()
	r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if delay > 0 {
		if err := r.Sys.SetInputDelay([]float64{delay}); err != nil {
			t.Fatal(err)
		}
	}
	return r.Sys
}

// checkPhaseCrossOracle requires every phase crossover the exact loop of the
// realization has robustly under rounding, and soundness of each reported one.
func checkPhaseCrossOracle(t *testing.T, sys *System) (required, missed int) {
	t.Helper()
	l := fzMgFromSystem(t, sys)
	res, err := AllMargin(sys)
	if err != nil {
		t.Fatalf("AllMargin: %v", err)
	}
	pc, zero, ok := l.phaseCross()
	if zero || !ok {
		return 0, 0
	}
	for _, c := range pc {
		if c.required {
			required++
			if !fzMgFound(c, res.PhaseCrossFreqs) {
				missed++
				t.Errorf("missed phase crossover ω=%v (|L|=%g), got %v", c.w, math.Exp(l.lnMag(c.w)), res.PhaseCrossFreqs)
			}
		}
	}
	for i, w := range res.PhaseCrossFreqs {
		if !fzMgSound(pc, w, l.phaseOff(w), l.budget(w)+1e-6) {
			t.Errorf("spurious phase crossover ω=%v: exact offset %v°", w, l.phaseOff(w)*180/math.Pi)
		}
		l.checkGM(t, w, res.GainMargins[i], "")
	}
	return required, missed
}

// Discrete loops with a ring of poles near z = 1 and an input delay of d
// samples cross −180° about every 2π/((n/2+d)·Δt) up to Nyquist, where the
// pencil of L − L~ on the absorbed loop loses roots. The scan beside the
// cluster must reach Nyquist (RR6SXJ). Oracle: exact rational loop of the
// realization, crossings robust under rounding of its entries.
func TestAllMarginPhaseCrossDelayedUnitCluster(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		delay    float64
	}{
		{"ring7 delay3", []float64{1.1974308853542987e-16}, []float64{1, -6.184746550403649, 16.242438259051475, -23.437999081676736, 20.022569140290482, -10.09289718715969, 2.766371287397997, -0.31573586749987936}, 3},
		{"ring6 delay3", []float64{1.391607534896511e-15}, []float64{1, -5.996928959245808, 14.984645931608547, -19.969294133971424, 14.969296404721078, -5.984649337733024, 0.9969300946206339}, 3},
		{"ring9 delay3", []float64{8.41798900940185e-14}, []float64{1, -8.99977808507088, 35.99822556730129, -83.99379258894939, 125.98759138433863, -125.9844969880358, 83.9876037961689, -35.993805000779744, 8.998230886657213, -0.9997789716302142}, 3},
		{"ring8 delay2", []float64{4.415434721389033e-14}, []float64{1, -8.036803991456532, 28.258220548727593, -56.77644492436822, 71.29705612326164, -57.30004652305295, 28.781827662944814, -8.261210980106032, 1.0374020840497447}, 2},
		{"ring6 zeros delay3", []float64{5.9841497733304036e-15, -2.804052128450389e-16, -2.0141604770736565e-15}, []float64{1, -5.984748544250353, 14.923837918788877, -19.847865977750004, 14.848055863244074, -5.924122747141322, 0.9848434871087284}, 3},
		{"real ring delay3", []float64{1.1434425150364283e-15}, []float64{1, -3.995537998630547, 5.9862758565987395, -3.985939289452155, 0.9952014452502751}, 3},
		{"ring6 no roots delay3", []float64{-2.5450729808155154e-17}, []float64{1, -6.050512878430315, 15.253110713377083, -20.507304934969966, 15.508379232755557, -6.254722124127703, 1.0510499913967717}, 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if n, _ := checkPhaseCrossOracle(t, marginTFSystem(t, c.num, c.den, 0.01, c.delay)); n == 0 {
				t.Fatal("oracle has no required crossing")
			}
		})
	}
}

// Seeded sweep of the RR6SXJ family: rings of 3 to 8 poles within 0.03 of
// z = 1, tiny gains and delays of 0 to 3 samples; base 721855b missed 468 of
// 3588 required crossings at this size.
func TestAllMarginPhaseCrossDelayedUnitClusterSweep(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(7, 31))
	total, missed := 0, 0
	for range 300 {
		np := 3 + rng.IntN(6)
		var roots []complex128
		for len(roots) < np {
			rho := 0.03 * rng.Float64()
			if len(roots)+1 < np && rng.IntN(2) == 0 {
				p := 1 + cmplx.Rect(rho, rng.Float64()*math.Pi)
				roots = append(roots, p, cmplx.Conj(p))
			} else {
				roots = append(roots, complex(1+rho*(2*rng.Float64()-1), 0))
			}
		}
		den := []complex128{1}
		for _, r := range roots {
			nd := make([]complex128, len(den)+1)
			for i, c := range den {
				nd[i] += c
				nd[i+1] -= c * r
			}
			den = nd
		}
		dr := make([]float64, len(den))
		for i, c := range den {
			dr[i] = real(c)
		}
		num := make([]float64, 1+rng.IntN(3))
		for i := range num {
			num[i] = (2*rng.Float64() - 1) * math.Pow(10, -16+3*rng.Float64())
		}
		n, m := checkPhaseCrossOracle(t, marginTFSystem(t, num, dr, 0.01, float64(rng.IntN(4))))
		total, missed = total+n, missed+m
		if t.Failed() {
			t.Fatalf("num %v den %v", num, dr)
		}
	}
	if total < 500 {
		t.Fatalf("sweep has only %d required crossings", total)
	}
}

func shearRealization(t *testing.T, s *System, k float64) *System {
	t.Helper()
	n, _, _ := s.Dims()
	T := mat.NewDense(n, n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			T.Set(i, j, k)
		}
		T.Set(i, i, 1+k)
	}
	var Ti, A, B, C, tmp mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	tmp.Mul(&Ti, s.A)
	A.Mul(&tmp, T)
	B.Mul(&Ti, s.B)
	C.Mul(s.C, T)
	sys, err := New(&A, &B, &C, s.D, s.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Phase crossovers beside lightly damped pairs that earlier revisions missed
// or found only on some realizations (YIWBXS); they pass on 721855b and are
// pinned against the exact oracle.
func TestAllMarginPhaseCrossLightlyDampedRealizations(t *testing.T) {
	lowPair := marginTFSystem(t, []float64{-4.969549555901367e-05, -136.49013836668794, -6930.639686355759, 0.019408431409671963, -2.972934713546136e-06}, []float64{1, 2e6, 1.000000000001103e8, 0.00859245157249671, 0.08427408999794361, 0, 0}, 0, 0)
	ff := marginTFSystem(t, []float64{-447.67661499638587, -0.27042549718475767, -4.083782716088854e9, -18.734339369971906, 8.388141179041895e-12}, []float64{1, 6.040586492872535e-4, 9.12217118175434e6, -1.009712242922581e-5, -2.672975009877894e-18}, 0, 0)
	a := mat.NewDense(6, 6, []float64{-3.621052778337902, -3.1274712544492207, -0.7036583014885595, 7163.284872846627, -1848.386251864272, 13.572483826745826, -3.068627991847458, -2.650346852858203, -0.5963087789166936, 6186.55146074059, -1596.360702520805, 11.719419435425758, -0.7430135397672757, -0.6417342219340859, -0.1443855357489841, 1392.2100564300247, -359.23583426417133, 2.6394788364652118, -7177.720215704098, -6199.333459425002, -1394.8049710175735, -15.450904840151562, 1483.7338126427182, 28.6681321534807, 1789.8429003980025, 1545.8714809299904, 347.8098476663391, -1606.2058839073006, -241.8606829237323, 112.49988798696154, 13.590458356132176, 11.737958667120829, 2.6409553874873333, 28.54542501926317, 112.52102142587427, -53.314511114599604})
	b := mat.NewDense(6, 1, []float64{-0.9118608758306777, -0.6394617563830491, 0.569750334265312, 0.9442768504319174, -0.06456443148960145, -0.9788186700204795})
	c := mat.NewDense(1, 6, []float64{5.656293014467941e-07, 1.2935760141565169e-06, -1.4020285600901407e-06, -3.8705360231291875e-07, 1.6522893429795368e-06, -6.812822097371732e-07})
	dDominated, err := New(a, b, c, mat.NewDense(1, 1, []float64{-74.50384797884405}), 0)
	if err != nil {
		t.Fatal(err)
	}
	systems := map[string]*System{
		"low pair double integrator": lowPair,
		"feedthrough":                ff,
		"D-dominated":                dDominated,
	}
	for _, k := range []float64{0.3, 1, 3, 10} {
		systems[fmt.Sprintf("feedthrough shear %v", k)] = shearRealization(t, ff, k)
	}
	for name, sys := range systems {
		t.Run(name, func(t *testing.T) {
			if n, _ := checkPhaseCrossOracle(t, sys); n == 0 {
				t.Fatal("oracle has no required crossing")
			}
		})
	}
}
