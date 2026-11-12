package controlsys

import (
	"errors"
	"math"
	"math/big"
	"testing"
)

// dcGainExactFloat is the exact DC gain of the float SISO model rounded once.
func dcGainExactFloat(t *testing.T, sys *System) float64 {
	t.Helper()
	_, g, ok := dcGainExactRat(sys)
	if !ok {
		t.Fatal("x0·I − A exactly singular")
	}
	f, _ := g[0].Float64()
	return f
}

// UP7H7O: D cancelling C(x0·I − A)⁻¹B to below an ulp of D left a sum the
// regular-path certificate bounded only absolutely, ε of its terms; DCGain
// returned 0 with nil although the float model's gain is nonzero. The
// literals are the 0.05sᵏ/(s + 0.7)ᵏ realizations of the CNGTCF dipulp rows
// (D moved by k ulps), 0.05s²/(s² + 1.4s + 0.49), and (z − 1)/(z − p/4).
func TestDCGainCancelledDCResolvesFloatModelGain(t *testing.T) {
	tests := []struct {
		name       string
		n          int
		a, b, c    []float64
		d, dt, val float64
	}{
		{"dipulp o1 ulp-1", 1, []float64{-0.7}, []float64{-0.034999999999999996}, []float64{1}, 0.049999999999999996, 0, -2.1807952269422717e-18},
		{"dipulp o2 ulp0", 2, []float64{0, -0.48999999999999994, 1, -1.4}, []float64{-0.024499999999999997, -0.06999999999999999}, []float64{0, 1}, 0.05, 0, 1.3594567648471302e-18},
		{"dipulp o3 ulp-1", 3, []float64{0, 0, -0.3429999999999999, 1, 0, -1.4699999999999998, 0, 1, -2.0999999999999996}, []float64{-0.017149999999999995, -0.0735, -0.10499999999999998}, []float64{0, 0, 1}, 0.049999999999999996, 0, -2.1403352041789637e-18},
		{"dipulp o4 ulp0", 4, []float64{0, 0, 0, -0.24009999999999992, 1, 0, 0, -1.3719999999999997, 0, 1, 0, -2.9399999999999995, 0, 0, 1, -2.8}, []float64{-0.012004999999999997, -0.06859999999999998, -0.147, -0.13999999999999999}, []float64{0, 0, 0, 1}, 0.05, 0, 1.3305567485876247e-18},
		{"dipulp o5 ulp0", 5, []float64{0, 0, 0, 0, -0.16806999999999994, 1, 0, 0, 0, -1.2004999999999997, 0, 1, 0, 0, -3.429999999999999, 0, 0, 1, 0, -4.8999999999999995, 0, 0, 0, 1, -3.5}, []float64{-0.008403499999999998, -0.06002499999999999, -0.17149999999999996, -0.245, -0.17500000000000002}, []float64{0, 0, 0, 0, 1}, 0.05, 0, -1.3530161897950136e-18},
		{"dipulp o6 ulp0", 6, []float64{0, 0, 0, 0, 0, -0.11764899999999995, 1, 0, 0, 0, 0, -1.0084199999999996, 0, 1, 0, 0, 0, -3.601499999999999, 0, 0, 1, 0, 0, -6.859999999999998, 0, 0, 0, 1, 0, -7.35, 0, 0, 0, 0, 1, -4.2}, []float64{-0.005882449999999997, -0.05042099999999999, -0.18007499999999996, -0.3429999999999999, -0.3675, -0.21000000000000002}, []float64{0, 0, 0, 0, 0, 1}, 0.05, 0, 2.7755575615628915e-18},
		{"0.05s²/(s²+1.4s+0.49)", 2, []float64{0, -0.49, 1, -1.4}, []float64{-0.0245, -0.06999999999999999}, []float64{0, 1}, 0.05, 0, -5.664403186863049e-20},
		{"(z-1)/(z-0.175)", 1, []float64{0.175}, []float64{-0.825}, []float64{1}, 1, 0.1, 6.728624391667616e-17},
		{"(z-1)/(z-0.075)", 1, []float64{0.075}, []float64{-0.925}, []float64{1}, 1, 0.1, -4.50090415388577e-17},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(tc.n, 1, 1, tc.a, tc.b, tc.c, []float64{tc.d}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			if exact := dcGainExactFloat(t, sys); exact != tc.val {
				t.Fatalf("exact gain %g, literal %g", exact, tc.val)
			}
			g, err := sys.DCGain()
			if err != nil || !(math.Abs(g.At(0, 0)-tc.val) <= 1e-12*math.Abs(tc.val)) {
				t.Fatalf("DCGain = %v, %v; want %v", g.At(0, 0), err, tc.val)
			}
		})
	}
}

// A float model whose exact DC gain is 0 keeps a certified 0: dyadic
// realizations whose solve is exact, s/(s + 1) and z − 1 over z − 0.5 from
// TransferFunc, and an order-20 upper-bidiagonal chain past the exact check.
func TestDCGainExactZeroStaysCertifiedZero(t *testing.T) {
	type model struct {
		name string
		sys  *System
	}
	var ms []model
	add := func(name string, n, m, p int, a, b, c, d []float64, dt float64) {
		sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
		if err != nil {
			t.Fatal(err)
		}
		ms = append(ms, model{name, sys})
	}
	add("dyadic triangular", 2, 1, 1, []float64{-1, 0.5, 0, -2}, []float64{1, 1}, []float64{1, 1}, []float64{-1.75}, 0)
	add("dyadic discrete", 2, 1, 1, []float64{0.5, 0.25, 0, 0.75}, []float64{1, 1}, []float64{1, 1}, []float64{-8}, 0.1)
	add("MIMO one zero channel", 2, 2, 1, []float64{-1, 0.5, 0, -2}, []float64{1, 0, 1, 1}, []float64{1, 1}, []float64{-1.75, 1}, 0)
	for _, tf := range []*TransferFunc{
		{Num: [][][]float64{{{1, 0}}}, Den: [][]float64{{1, 1}}},
		{Num: [][][]float64{{{1, -1}}}, Den: [][]float64{{1, -0.5}}, Dt: 0.1},
		{Num: [][][]float64{{{2, 0, 0}}}, Den: [][]float64{{1, 3, 2}}},
	} {
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		ms = append(ms, model{"TransferFunc", res.Sys})
	}
	const n = 20
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := range n {
		a[i*n+i], b[i], c[i] = -1, 1, 1
		if i+1 < n {
			a[i*n+i+1] = 0.5
		}
	}
	chain, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, g, _ := dcGainExactRat(chain)
	d, exact := new(big.Rat).Neg(g[0]).Float64()
	if !exact {
		t.Fatal("chain gain not a float")
	}
	add("order-20 chain", n, 1, 1, a, b, c, []float64{d}, 0)
	for _, tc := range ms {
		_, want, ok := dcGainExactRat(tc.sys)
		if !ok {
			t.Fatalf("%s: singular", tc.name)
		}
		g, err := tc.sys.DCGain()
		if err != nil {
			t.Errorf("%s: DCGain error %v", tc.name, err)
			continue
		}
		r := g.RawMatrix()
		for k, w := range want {
			if w.Sign() == 0 && r.Data[k] != 0 {
				t.Errorf("%s: entry %d = %g; want exact 0", tc.name, k, r.Data[k])
			}
		}
		if !hasZero(want) {
			t.Errorf("%s: no exact-zero entry; the control needs one", tc.name)
		}
	}
}

func hasZero(v []*big.Rat) bool {
	for _, x := range v {
		if x.Sign() == 0 {
			return true
		}
	}
	return false
}

// An entry of 0 the certificate cannot prove past the exact check keeps
// ErrAccuracyUnverified or resolves from 0; it is never a nil 0 on a
// nonzero float model. The order-18 model, the dipulp o2 row beside sixteen
// decoupled stable states, resolves on the refined path.
func TestDCGainCancelledDCBeyondExactCheck(t *testing.T) {
	const n = 18
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	a[0*n+1], a[1*n+0], a[1*n+1] = -0.48999999999999994, 1, -1.4
	b[0], b[1], c[1] = -0.024499999999999997, -0.06999999999999999, 1
	for i := 2; i < n; i++ {
		a[i*n+i] = -float64(i)
		if i+1 < n {
			a[i*n+i+1] = 0.25
		}
	}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0.05}, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := dcGainExactFloat(t, sys)
	g, err := sys.DCGain()
	switch {
	case err == nil && g.At(0, 0) == 0:
		t.Fatalf("DCGain = 0 with nil; float model gain %g", want)
	case err == nil && !(math.Abs(g.At(0, 0)-want) <= 1e-8*math.Abs(want)):
		t.Fatalf("DCGain = %g with nil; want %g", g.At(0, 0), want)
	case err != nil && !errors.Is(err, ErrAccuracyUnverified):
		t.Fatalf("DCGain error %v", err)
	}
}
