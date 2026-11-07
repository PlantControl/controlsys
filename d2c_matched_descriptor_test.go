package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// VYRY6S: MATLAB d2c(dss(...),'matched') accepts SISO discrete descriptors;
// D2C reduces a proper descriptor to explicit form before d2cMatched.
func TestD2CMatched_DescriptorMatchesExplicit(t *testing.T) {
	const n, dt = 8, 0.01
	rng := rand.New(rand.NewPCG(6, 6))
	tm := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			tm.Set(i, j, rng.NormFloat64())
		}
		tm.Set(i, i, tm.At(i, i)+3)
	}
	diag := mat.NewDense(n, n, nil)
	for k := range n {
		diag.Set(k, k, math.Exp(-0.2*float64(k+1)*dt))
	}
	var a, b mat.Dense
	a.Mul(tm, diag)
	ones := mat.NewDense(n, 1, nil)
	for i := range n {
		ones.Set(i, 0, 1)
	}
	b.Mul(tm, ones)
	c := mat.NewDense(1, n, nil)
	for i := range n {
		c.Set(0, i, float64(i%3)-0.7)
	}
	desc, err := NewDescriptor(&a, &b, c, mat.NewDense(1, 1, nil), tm, dt)
	if err != nil {
		t.Fatal(err)
	}
	expl, err := New(diag, ones, c, mat.NewDense(1, 1, nil), dt)
	if err != nil {
		t.Fatal(err)
	}
	got, err := desc.D2C(D2COptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatalf("descriptor D2C matched: %v", err)
	}
	want, err := expl.D2C(D2COptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	checkD2CMatchedAgainstOracle(t, "descriptor", expl, got)
	poles, _ := got.Poles()
	for k := 1; k <= n; k++ {
		p := -0.2 * float64(k)
		best := math.Inf(1)
		for _, q := range poles {
			best = math.Min(best, cmplx.Abs(q-complex(p, 0)))
		}
		if !(best <= 1e-8*math.Abs(p)) {
			t.Errorf("no pole near %g: nearest %.3g away", p, best)
		}
	}
	for _, w := range []float64{0.01, 0.3, 2, 30} {
		s := complex(0, w)
		g, err := got.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		h, err := want.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(g[0][0]-h[0][0]) / cmplx.Abs(h[0][0]); !(d <= 1e-8) {
			t.Errorf("w=%g: descriptor %v vs explicit %v (rel %.3g)", w, g[0][0], h[0][0], d)
		}
	}
}

func TestD2CMatched_ImproperDescriptorFails(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{1, 0, 0, 0.5})
	e := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	desc, err := NewDescriptor(a, mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), e, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if proper, _ := desc.IsProper(); proper {
		t.Fatal("model expected improper")
	}
	if _, err := desc.D2C(D2COptions{Method: C2DMethodMatched}); err == nil {
		t.Fatal("improper descriptor: nil error")
	} else if !errors.Is(err, ErrDescriptorUnsupported) {
		t.Fatalf("err = %v, want ErrDescriptorUnsupported", err)
	}
}
