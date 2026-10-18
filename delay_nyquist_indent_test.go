package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func indentTestPlant(t *testing.T, num, den []float64) *System {
	t.Helper()
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}
	r, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	return r.Sys
}

func indentTestDelayedGains(t *testing.T, k1, t1, k2, t2 float64) *System {
	t.Helper()
	gain := func(k, tau float64) *System {
		g, err := NewGain(mat.NewDense(1, 1, []float64{k}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := g.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
		return g
	}
	k, err := Parallel(gain(k1, t1), gain(k2, t2))
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// Loops with a double or triple integrator, where the state-space solve
// fails at the default indentation ω=1e-7 next to the multiple pole.
// Oracles: argument principle for χ = den + num·K·e^{-τs} on the box
// [0,60]×[-400,400] and Newton roots (rightmost -0.4389, -0.1131±0.2889j,
// -0.1113±3.1034j, 0.9811±1.2163j), and a dense sweep (ratio 1+1e-5) of
// |P/(1+PK)| and |1/(1+L)| from the polynomials. p4, p5: L is finite only
// beyond 1e-4 of the gap to the next pole; argument principle on χ gives 2
// and 3 RHP roots.
func TestDelayNyquistIndentNearMultipleIntegrator(t *testing.T) {
	p0 := func() *System {
		p := indentTestPlant(t, []float64{5.910980332296372, 27.806410726825444, 37.431038620035174, 11.565340194450041}, []float64{1, 0.20389057161026952, -0.07104652183880648, 0, 0})
		if err := p.SetInputDelay([]float64{0.022715415198538882}); err != nil {
			t.Fatal(err)
		}
		return p
	}
	p1 := indentTestPlant(t, []float64{0.46603599034106363, 1.1507662521895636, 0.8492300943505945, 0.24647378460270886, 0.024431190753480315}, []float64{1, 1.0954836376994372, 0, 0, 0})
	if err := p1.SetOutputDelay([]float64{0.02290693554361851}); err != nil {
		t.Fatal(err)
	}
	k1 := indentTestDelayedGains(t, 0.9575000297606748, 0.019511878680882565, -0.022781002697111294, 0.26994534441905227)
	p2 := indentTestPlant(t, []float64{10.747742314693493, 11.151438054546672, 2.1696272808395105}, []float64{1, 4.665255397111082, 0, 0, 0})
	if err := p2.SetOutputDelay([]float64{0.04028063817878855}); err != nil {
		t.Fatal(err)
	}
	k2 := indentTestDelayedGains(t, 1.0443033033837892, 0.1306289578483865, 0.4884653272897968, 0.20653192846415452)
	p3 := func() *System {
		p := indentTestPlant(t, []float64{1.3878930086477943, 5.22648972355323}, []float64{1, 0, 0, 0})
		if err := p.SetInputDelay([]float64{0.1033441077900939}); err != nil {
			t.Fatal(err)
		}
		if err := p.SetOutputDelay([]float64{0.4856267246070171}); err != nil {
			t.Fatal(err)
		}
		return p
	}
	p4 := func() *System {
		p := indentTestPlant(t, []float64{1.7564813852082757, 1.388622390508125}, []float64{1, -0.21409111912503734, 0, 0, 0})
		if err := p.SetInputDelay([]float64{0.1}); err != nil {
			t.Fatal(err)
		}
		return p
	}
	p5 := indentTestPlant(t, []float64{1.1718148861896658, 24.751460741266797}, []float64{1, 0, 0.7958029956388403, 0, 0, 0})
	if err := p5.SetInputDelay([]float64{0.2336578227553776}); err != nil {
		t.Fatal(err)
	}
	k5 := indentTestDelayedGains(t, -0.007883409556943888, 0.06107819230285669, 0, 0)

	cases := []struct {
		name  string
		plant *System
		k     *System
		roots int
		peak  float64
	}{
		{"double integrator nd=1", p0(), nil, 0, 1.7772338088102355},
		{"triple integrator nd=3", p1, k1, 0, 1.698009432484453},
		{"triple integrator out+2 delays", p2, k2, 0, 11.589371381816779},
		{"triple integrator unstable", p3(), nil, 2, math.Inf(1)},
		{"triple integrator, L finite beyond 1e-4 of pole gap", p4(), nil, 2, math.Inf(1)},
		{"triple integrator + axis pair, L finite beyond 1e-4 of pole gap", p5, k5, 3, math.Inf(1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cl, err := Feedback(tc.plant, tc.k, -1)
			if err != nil {
				t.Fatal(err)
			}
			stable, err := cl.IsStable()
			if err != nil {
				t.Fatalf("IsStable: %v", err)
			}
			if stable != (tc.roots == 0) {
				t.Fatalf("IsStable = %v, want %v", stable, tc.roots == 0)
			}
			e, err := newDelayLFT(cl)
			if err != nil {
				t.Fatal(err)
			}
			res, err := e.loop(false).nyquist(0.5)
			if err != nil || e.err != nil {
				t.Fatalf("nyquist: %v, %v", err, e.err)
			}
			if res.axisRoot || res.roots != tc.roots {
				t.Fatalf("roots = %d (axisRoot %v), want %d", res.roots, res.axisRoot, tc.roots)
			}
			peak, _, err := HinfNorm(cl)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.IsInf(tc.peak, 1) {
				if !math.IsInf(peak, 1) {
					t.Fatalf("HinfNorm = %g, want +Inf", peak)
				}
				return
			}
			if math.Abs(peak-tc.peak) > 1e-6*tc.peak {
				t.Fatalf("HinfNorm = %.10g, want %.10g", peak, tc.peak)
			}
		})
	}

	t.Run("DiskMargin", func(t *testing.T) {
		loop, err := Series(k2, p2)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name string
			loop *System
			ms   float64
		}{
			{"double integrator", p0(), 1.352067449972263},
			{"triple integrator", loop, 18.13206091451099},
			{"triple integrator unstable", p3(), math.Inf(1)},
			{"triple integrator, L finite beyond 1e-4 of pole gap", p4(), math.Inf(1)},
		} {
			r, err := DiskMargin(tc.loop)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if math.IsInf(tc.ms, 1) {
				if !math.IsInf(r.PeakSensitivity, 1) {
					t.Errorf("%s: Ms = %g, want +Inf", tc.name, r.PeakSensitivity)
				}
				continue
			}
			if math.Abs(r.PeakSensitivity-tc.ms) > 1e-6*tc.ms {
				t.Errorf("%s: Ms = %.10g, want %.10g", tc.name, r.PeakSensitivity, tc.ms)
			}
		}
	})
}

// Loops with a multiple integrator and another open-loop pole near it, where
// the double solve cannot evaluate L inside 1e-2 of the gap (all errored
// before indentCluster; the last is ergo GFXRCX): the indentation grows
// past indentLimit or encloses the nearby pole (enclosed: poles enclosed
// besides the integrator; inner: those in the RHP). Where the double-double
// fallback evaluates L at axisGap, the plain indentation encloses none. Oracles: argument
// principle for χ = den + num·K·e^{-τs} on a box bounding its RHP roots, and
// a dense sweep (ratio 1+1e-5, refined) of |P/(1+PK)| and |1/(1+KP)| from
// the polynomials.
func TestDelayNyquistIndentPoleCluster(t *testing.T) {
	cases := []struct {
		name            string
		num, den        []float64
		tin             float64
		k               []float64
		enclosed, inner int
		roots           int
		peak, ms        float64
	}{
		{"triple integrator, pole at -0.0028", []float64{1.45142418583699, 1.7771868624571499, 0.7337024564555565, 0.1491670039392469, 0.011972359539458527}, []float64{1, 0.6735426670516077, 0.00188436338869233, 0, 0, 0}, 0.0021047148122163885, nil, 0, 0, 0, 1.38344291476588, 1.00301885979056},
		{"triple integrator, enclosed RHP pole 2.7e-4", []float64{1.4364650281495652, 0.7289142902933975, 0.1579532558455607, 0.012465726823420125}, []float64{1, -0.00026662308744651494, 0, 0, 0}, 0.03743190011595099, []float64{0.9159105466881329, 0.0028822272430659075, -0.08609255836401393, 0.09917128660413921}, 1, 1, 0, 1.73597844290612, 1.04818875003705},
		{"triple integrator, enclosed pole -7.5e-5", []float64{0.5948608803738119, 0.18021188904600305, 0.02373347440976468, 0.0015423370227633894, 3.955326526685069e-05}, []float64{1, 0.08062050596496431, 6.018829832700829e-06, 0, 0, 0}, 0.24120324143499425, nil, 0, 0, 0, 1.4115919346666, 1.15249495294211},
		{"triple integrator, enclosed RHP pole 2.1e-5, unstable", []float64{3.9318328957351754, 9.01550405735846, 7.781485125822608, 2.941510858787714, 0.3164957596584042}, []float64{1, 0.9269292765925116, -2.0846422608156813e-05, 0, 0, 0}, 0.8792379781321229, nil, 0, 0, 2, math.Inf(1), math.Inf(1)},
		{"triple integrator, RHP pole 6.7e-3, delayed gain, unstable", []float64{3.310364316932462, -3.4690921208593335}, []float64{1, -0.0066905036607369794, 0, 0, 0}, 0.3333286130472003, []float64{-0.9519745006566804, 0.2094256006355884, 0, 0}, 0, 0, 2, math.Inf(1), math.Inf(1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := indentTestPlant(t, tc.num, tc.den)
			if err := p.SetInputDelay([]float64{tc.tin}); err != nil {
				t.Fatal(err)
			}
			var k *System
			loop := p
			if tc.k != nil {
				k = indentTestDelayedGains(t, tc.k[0], tc.k[1], tc.k[2], tc.k[3])
				var err error
				if loop, err = Series(k, p); err != nil {
					t.Fatal(err)
				}
			}
			cl, err := Feedback(p, k, -1)
			if err != nil {
				t.Fatal(err)
			}
			stable, err := cl.IsStable()
			if err != nil {
				t.Fatalf("IsStable: %v", err)
			}
			if stable != (tc.roots == 0) {
				t.Fatalf("IsStable = %v, want %v", stable, tc.roots == 0)
			}
			e, err := newDelayLFT(cl)
			if err != nil {
				t.Fatal(err)
			}
			l := e.loop(false)
			res, err := l.nyquist(0.5)
			if err != nil || e.err != nil {
				t.Fatalf("nyquist: %v, %v", err, e.err)
			}
			if res.axisRoot || res.roots != tc.roots {
				t.Fatalf("roots = %d (axisRoot %v), want %d", res.roots, res.axisRoot, tc.roots)
			}
			if len(l.axis) != 1 || l.axis[0].poles()-l.axis[0].mult != tc.enclosed || l.axis[0].inner != tc.inner {
				t.Fatalf("axis indentation %+v, want %d enclosed, %d inner", l.axis, tc.enclosed, tc.inner)
			}
			peak, _, err := HinfNorm(cl)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			r, err := DiskMargin(loop)
			if err != nil {
				t.Fatalf("DiskMargin: %v", err)
			}
			for _, c := range []struct {
				what      string
				got, want float64
			}{{"HinfNorm", peak, tc.peak}, {"Ms", r.PeakSensitivity, tc.ms}} {
				if math.IsInf(c.want, 1) {
					if !math.IsInf(c.got, 1) {
						t.Errorf("%s = %g, want +Inf", c.what, c.got)
					}
				} else if math.Abs(c.got-c.want) > 1e-6*c.want {
					t.Errorf("%s = %.10g, want %.10g", c.what, c.got, c.want)
				}
			}
		})
	}
}
