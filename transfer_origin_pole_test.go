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
