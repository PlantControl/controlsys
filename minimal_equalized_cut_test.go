package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// gradedKalmanLiterals are Kalman-structure draws graded by a diagonal
// similarity (newKalmanCaseScaled, id/r.u1.u2, continuous): d4 1838/2.1.1,
// d8 45/2.1.0 and d4 1134/2.0.2.
func gradedKalmanLiterals(t *testing.T) map[string]*System {
	t.Helper()
	mk := func(a, b, c *mat.Dense, dt float64) *System {
		p, _ := c.Dims()
		_, m := b.Dims()
		sys, err := New(a, b, c, mat.NewDense(p, m, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	return map[string]*System{
		"d4-1838": mk(mat.NewDense(4, 4, []float64{
			1.5951934194078166, -2.4627010310628744e-07, 5.661959740160857e-05, -0.2255345729412059,
			184343.00148758528, -0.7560278963065902, -57.852259621632996, -299446.11998223246,
			3213.0465596734, 0.018323517025276233, -0.23738721889152362, 4523.3893411487325,
			0.06303897632182298, 1.4828295418951795e-07, -5.5777138918285015e-05, 1.4906225274725795,
		}), mat.NewDense(4, 1, []float64{
			2.3560723630207483,
			-7.719026526312119e+07,
			184582.12090325906,
			-7.50011730382037,
		}), mat.NewDense(1, 4, []float64{
			1488.6728169714029, -0.014863182800056694, 0.11235149078753619, 7259.7511039091205,
		}), 0),
		"d8-45": mk(mat.NewDense(3, 3, []float64{
			-3.278568536623127, -3.572759808249402e-11, -4.2727450740111847e-07,
			-1.3532474728283737e+10, -1.6017755553401152, -4413.837825447322,
			-3.2647488410020606e+06, -5.243582477313004e-05, 1.447990277032188,
		}), mat.NewDense(3, 2, []float64{
			1.8881522721975423e-06, -4.1106248148058295e-06,
			-55656.139536289695, -54836.899419510526,
			-2.4937926299934894, -5.3327216714158245,
		}), mat.NewDense(1, 3, []float64{
			7.252191797029054e+12, 1178.035022460001, 1.743897009069254e+06,
		}), 0),
		"d4-1134": mk(mat.NewDense(4, 4, []float64{
			0.41342335386815887, 51.98402116777392, -306.3888356911102, 0.0026874570044089266,
			0.07594714105073289, -1.0649425375694965, -6.818423091376761, 5.969372491284363e-05,
			-0.00863753842461081, -0.02216342586765398, 2.515047844001973, -2.7314498138100993e-06,
			1202.9212779481848, 3909.8452656616987, -45150.20628221605, 0.12660232133698765,
		}), mat.NewDense(4, 1, []float64{
			0.0006187022076285131,
			2.3334867341791864e-05,
			1.1358222380117837e-06,
			-0.6600808416973589,
		}), mat.NewDense(2, 4, []float64{
			-0.20164700140260816, 0.8038560453356552, -2.9784232818934573, 0.00011960413232009506,
			0.0010572842975083898, -2.0232150250564302, 7.709588392154627, 9.113109163460326e-05,
		}), 0),
	}
}

// KPGQ4X: on these graded draws the orthogonal staircase of the stored
// realization cuts states, but its rotations mix the graded entries and
// the cut misses the response beyond √ε, so the band rejects it and base
// 938d548 returned the model unreduced. The cut of the
// power-of-two equalized model passes the band and keeps the response.
func TestReduceGradedKalmanTakesEqualizedCut(t *testing.T) {
	lits := gradedKalmanLiterals(t)
	for _, tc := range []struct {
		name  string
		order int
	}{{"d4-1838", 2}, {"d8-45", 2}} {
		sys := lits[tc.name]
		policy := newRealizationTransformPolicy(sys)
		plain, removed, err := reduceByStaircase(sys, policy, &ReduceOpts{}, true, -1)
		if err != nil {
			t.Fatal(err)
		}
		if n, _, _ := sys.Dims(); plain.Order == n || frequencyAxisCutInBand(sys, plain.Sys, removed, math.Sqrt(eps())) {
			t.Fatalf("%s: precondition: plain cut order %d of %d, want a cut refused by the band", tc.name, plain.Order, n)
		}
		res, err := sys.Reduce(nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != tc.order {
			t.Errorf("%s: order %d, want %d", tc.name, res.Order, tc.order)
			continue
		}
		if e := determinedOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
			t.Errorf("%s: response misfit %.3g", tc.name, e)
		}
	}
}

// KPGQ4X guard: d4 1134's plain cut (order 2) passes the band, while
// Equalize keeps order 4. The equalized cut is only a fallback for a cut
// the band refuses, so the plain cut stands.
func TestReduceGradedKalmanKeepsCertifiedPlainCut(t *testing.T) {
	sys := gradedKalmanLiterals(t)["d4-1134"]
	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 2 {
		t.Fatalf("order %d, want 2", res.Order)
	}
	if e := determinedOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
		t.Errorf("response misfit %.3g", e)
	}
	eq, err := sys.Reduce(&ReduceOpts{Equalize: true})
	if err != nil {
		t.Fatal(err)
	}
	if eq.Order != 4 {
		t.Fatalf("precondition: Equalize order %d, want 4", eq.Order)
	}
}
