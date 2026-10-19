package controlsys

import (
	"math"
	"math/big"
	"testing"
)

func exactRationalAtOne(num, den []float64) float64 {
	sum := func(c []float64) *big.Rat {
		s := new(big.Rat)
		for _, v := range c {
			s.Add(s, new(big.Rat).SetFloat64(v))
		}
		return s
	}
	v, _ := new(big.Rat).Quo(sum(num), sum(den)).Float64()
	return v
}

// 4M7RB3: I−A singular to working precision (cond 2e18) with the nearest
// pole off z = 1 (3.4e-5 outside; 1.3e-7 inside): the transfer limit's
// coefficient tolerance took D(1) for 0 and returned +Inf while FreqResponse
// was finite. Oracle: exact rational N(1)/D(1) of the float coefficients.
func TestDCGain_IllConditionedOffPoleFinite(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		dt       float64
	}{
		{[]float64{1.8748015424080903e-16}, []float64{1, -6.656527066165689, 18.974082402962605, -30.020939359334676, 28.473509438245436, -16.187809823253666, 5.1075991528389695, -0.6899147452929796}, 0.1},
		{[]float64{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}, []float64{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}, 0.01},
	} {
		tf := &TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		want := exactRationalAtOne(c.num, c.den)
		g, err := res.Sys.DCGain()
		if err != nil {
			t.Fatalf("%v: %v", c.den, err)
		}
		if got := g.At(0, 0); math.Abs(got-want) > 1e-9*math.Abs(want) {
			t.Errorf("%v: DCGain %g, want %g", c.den, got, want)
		}
		fr, err := res.Sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatal(err)
		}
		if got := real(fr.At(0, 0, 0)); math.Abs(got-want) > 1e-9*math.Abs(want) {
			t.Errorf("%v: FreqResponse(0) %g, want %g", c.den, got, want)
		}
	}
}

func TestDCGain_CompanionIntegratorStaysInfinite(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		den := []float64{1, -1.5, 0.5, 0}
		if dt > 0 {
			den = []float64{1, -2.5, 2, -0.5}
		}
		tf := &TransferFunc{Num: [][][]float64{{{1e-16, 2e-16}}}, Den: [][]float64{den}, Dt: dt}
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		g, err := res.Sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		if !math.IsInf(g.At(0, 0), 1) {
			t.Errorf("dt=%g: DCGain %g, want +Inf", dt, g.At(0, 0))
		}
	}
}
