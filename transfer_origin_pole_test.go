package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// YY7Z5C: on slow graded observable companions with an origin pole the
// staircase floor cut a coupling, and the keepObservable gate never saw
// the loss because every band point below the poles failed evalInto's
// normwise pivot test and was skipped. MATLAB ss2tf keeps all n states.
func TestTransferFunctionGradedCompanionKeepsOriginPole(t *testing.T) {
	cases := []struct {
		col, b []float64
		d      float64
		ws     []float64
	}{
		{
			col: []float64{-0, -1.930942169363784e-18, -4.625743911882556e-15, -4.736047505234965e-12, -2.686333377965866e-09, -9.116247383644804e-07, -0.00018507987260818481, -0.02081298828125},
			b:   []float64{2.2421875, 14336},
			ws:  []float64{3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1},
		},
		{
			col: []float64{-0, -7.26422712612985e-24, 3.5634578994751776e-18, 1.0426791897345855e-14, -1.7826413432472095e-11, -3.074821605523414e-08, -5.050575132350965e-05, -0.0040533962164772674, -0.11291301250457764},
			b:   []float64{1.4586476027034223e-07, 0.00017702579498291016, 0.107421875},
			d:   -4,
			ws:  []float64{3e-5, 1e-4, 1e-3, 1e-2, 1},
		},
	}
	for k, tc := range cases {
		n := len(tc.col)
		a := make([]float64, n*n)
		for i := 1; i < n; i++ {
			a[i*n+i-1] = 1
		}
		for i, v := range tc.col {
			a[i*n+n-1] = v
		}
		b, c := make([]float64, n), make([]float64, n)
		copy(b, tc.b)
		c[n-1] = 1
		exDen, exNum := ratSISOTF(a, b, c, n)
		sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{tc.d}, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		den := res.TF.Den[0]
		if res.MinimalOrder != n || len(den) != n+1 {
			t.Fatalf("case %d: order %d, den %v; want order %d with exact den %v", k, res.MinimalOrder, den, n, exDen)
		}
		if den[n] != 0 {
			t.Errorf("case %d: den[%d] = %g, want 0 (origin pole)", k, n, den[n])
		}
		for i := 1; i < n; i++ {
			if r := math.Abs(den[i]-exDen[i]) / math.Abs(exDen[i]); r > 1e-13 {
				t.Errorf("case %d: den[%d] = %.17g, want %.17g (rel %.2g)", k, i, den[i], exDen[i], r)
			}
		}
		for _, w := range tc.ws {
			s := complex(0, w)
			ex := Poly(exNum).Eval(s)/Poly(exDen).Eval(s) + complex(tc.d, 0)
			if e := cmplx.Abs(res.TF.eval(s)[0][0]-ex) / cmplx.Abs(ex); e > 1e-12 {
				t.Errorf("case %d: response at w=%g off exact by %.3g", k, w, e)
			}
		}
	}
}

// Rows of a graded observable companion are taken from the full-order
// conversion one by one: the eₙᵀ row is exact there, while the full
// conversion of the dense rows lost their response far above the poles,
// which the cut rows keep (seeded MIMO sweep trial 1371).
func TestTransferFunctionGradedCompanionFullRowsPerRow(t *testing.T) {
	cs := newObservableMIMO(
		[]float64{1, 0.001705169677734375, 1.3068784028291702e-06, 5.920067769338289e-10, 1.751269915200329e-13, 3.521489792846935e-17, 4.83880643540895e-21, 4.410882240923436e-25, 2.4380426657192067e-29, 6.2618485959831026e-34},
		[]float64{0, 0, -0.000244140625, 0, -0.005615234375, -23552, 0, 0, 0, 0, 96, 1.0728836059570312e-06, 1.1324882507324219e-06, 6.29425048828125e-05, 1312, 15.75, -0.00164794921875, 0},
		[]float64{0, 0, 0, 0, 0, 0, 0, 0, 1, -1.5, 0.75, -2, -0.5, -1.5, 1.5, -1.5, 0.5, -1.25, -0.75, -0.75, 0.25, 1, -1.5, 0.5, -1.25, -1.75, 0},
		2, 3)
	tf := convertMIMOCompanion(t, cs, false)
	if got := tf.Den[0]; len(got) != cs.n+1 {
		t.Fatalf("row 0 den %v, want %v", got, cs.den)
	}
	for k, want := range cs.den {
		if got := tf.Den[0][k]; math.Abs(got-want) > 1e-13*math.Abs(want) {
			t.Errorf("row 0 den[%d] = %.17g, want %.17g", k, got, want)
		}
	}
	bcol := make([]float64, cs.n)
	for i := range cs.p {
		for j := range cs.m {
			for k := range cs.n {
				bcol[k] = cs.b[k*cs.m+j]
			}
			num := ratObservableNum(cs.den, bcol, cs.c[i*cs.n:(i+1)*cs.n])
			for _, w := range []float64{1e-8, 1e-6, 1e-4, 1e-2, 1, 100} {
				if i > 0 && w < 1 {
					continue
				}
				r := ratRatioJ(num, cs.den, w)
				if e := cmplx.Abs(tf.eval(complex(0, w))[i][j]-r) / cmplx.Abs(r); e > 1e-12 {
					t.Errorf("entry %d,%d at w=%g off exact by %.3g", i, j, w, e)
				}
			}
		}
	}
}

// 4XDPLI: graded companions with an origin pole whose observability cut
// drops it with a zero within 1e-7 relative of it. The cut row misses by
// under 1e-6 from its own slowest pole up, so keepObservable never made the
// full conversion and the type-1 system came back type 0. The exact num
// constant is resolved in float64 (no cancellation), and MATLAB ss2tf keeps
// den = poly(A) with the integrator.
func TestTransferFunctionGradedCompanionKeepsOriginPoleBesideSlowZero(t *testing.T) {
	den1472 := []float64{1, 0.09228515625, 0.004141807556152344, 9.087217040359974e-05, 9.155219231615774e-07, 2.865249149053284e-09, 0}
	b1472 := []float64{2.86102294921875e-06, -14336, 336, -2.86102294921875e-05, 1728, 0}
	ws1472 := []float64{1e-11, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2}
	cases := []struct {
		name      string
		den, b, c []float64
		p, shape  int
		ws        []float64
	}{
		{"1472", den1472, b1472, []float64{0, 0, 0, 0, 0, 1}, 1, 0, ws1472},
		{"1472-mimo", den1472, b1472, []float64{0, 0, 0, 0, 0, 1, 0.5, 1.5, 0.25, 0.25, 1.75, 1.5}, 2, 0, ws1472},
		{"dense-transposed", []float64{1, 2664, 3.174912e+06, 2.177994752e+09, 9.27255625728e+11, 2.47497308504064e+14, 3.986381863282278e+16, 3.4771921332810547e+18, 1.2210569259201331e+20, 0},
			[]float64{-2.384185791015625e-07, 1536, -112, 0, 116, -92, 0.02392578125, 0, -1.28125},
			[]float64{0.75, 1.75, -1, 0.5, -1.25, 0, 0, -0.25, 0}, 1, 2, []float64{2e-10, 2e-9, 1e-6, 1e-2, 1, 1e4}},
		// Reversed companions with a unit row: the band reference is itself
		// off by up to 1e4 below the poles, so the exact direct full row
		// never beat the cut one there (cut error 62.8 and 0.995).
		{"unit-reversed-12", []float64{1, 754, 255356, 5.1226392e+07, 6.755561344e+09, 6.140269664e+11, 3.9174454525952e+13, 1.749820945778688e+15, 5.344039379091456e+16, 1.0575501408690831e+18, 1.2115752641409319e+19, 6.01702493877751e+19, 0},
			[]float64{0.05078125, 0, -2.6702880859375e-05, -8.821487426757812e-06, -524288, -0.0072021484375, 0.21875, 1.7642974853515625e-05, 3.375, 2.288818359375e-05, 1.1444091796875e-05, -15.25},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 1, 1, []float64{1.76e-3, 1.76e-2, 0.08, 1.38, 108}},
		{"unit-reversed-10", []float64{1, 1472, 685248, -9.1623424e+07, -1.87501314048e+11, -5.6863329353728e+13, -1.614542569209856e+15, 1.9241177747501875e+18, 1.944393572627251e+19, -1.1803426695136936e+22, 0},
			[]float64{1.4901161193847656e-06, 0.265625, 0.007080078125, -15872, -245760, 0, -0.2265625, 0, 3.4570693969726562e-06, 1.0251998901367188e-05},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 1, 1, []float64{5.6e-7, 5.6e-6, 0.0577, 592}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.den) - 1
			a, b, c := observableCompanion(tc.den), tc.b, tc.c
			if tc.shape != 0 {
				a, b, c = companionShape(tc.den, tc.b, tc.c, tc.shape)
			}
			sys, err := NewFromSlices(n, 1, tc.p, a, b, c, make([]float64, tc.p), 0)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.p {
				den := res.TF.Den[i]
				if len(den) != n+1 {
					t.Fatalf("row %d den %v, want %v", i, den, tc.den)
				}
				if den[n] != 0 {
					t.Errorf("row %d den[%d] = %g, want 0 (origin pole)", i, n, den[n])
				}
				for k := 1; k < n; k++ {
					if r := math.Abs(den[k]-tc.den[k]) / math.Abs(tc.den[k]); r > 1e-12 {
						t.Errorf("row %d den[%d] = %.17g, want %.17g", i, k, den[k], tc.den[k])
					}
				}
				num := ratObservableNum(tc.den, tc.b, tc.c[i*n:(i+1)*n])
				for _, w := range tc.ws {
					ex := ratRatioJ(num, tc.den, w)
					if e := cmplx.Abs(res.TF.eval(complex(0, w))[i][0]-ex) / cmplx.Abs(ex); e > 1e-9 {
						t.Errorf("row %d at w=%g off exact by %.3g", i, w, e)
					}
				}
			}
		})
	}
}
