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
// |P/(1+PK)| and |1/(1+L)| from the polynomials.
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
