package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// wm67Model is the properExplicitForm output of FuzzDescriptor/0f2ae9b8:
// upper triangular A with poles −3.16, −1.4e-14, −6.2e-13 and a 3e-12
// coupling between the slow states.
func wm67Model() (a, b, c, d []float64) {
	a = []float64{
		-3.1622776601683773, 3.0017672102592, 13.819891230225107,
		0, -1.3826329222003976e-14, -3.2458753376784857e-12,
		0, 0, -6.213738707494789e-13,
	}
	b = []float64{
		0.6219449773361903, 0.04522290976328859,
		-0.5215451170485351, 1.2581120474674456,
		0.9162945237266072, -0.47637786166968576,
	}
	c = []float64{
		0.1006430222266349, -0.674095197361011, 1.0712551603861844,
		0.010079115259649664, 0.18449276203704465, 1.1046839261205939,
	}
	d = []float64{
		-0.33420726351746655, -0.08435246135701613,
		0.026312929966977205, 0.056201341552568174,
	}
	return a, b, c, d
}

// reverseStates returns PAP, PB, CP for the reversal permutation P.
func reverseStates(n, m, p int, a, b, c []float64) (ra, rb, rc []float64) {
	ra, rb, rc = make([]float64, n*n), make([]float64, n*m), make([]float64, p*n)
	for i := range n {
		for j := range n {
			ra[i*n+j] = a[(n-1-i)*n+n-1-j]
		}
		copy(rb[i*m:(i+1)*m], b[(n-1-i)*m:(n-i)*m])
		for r := range p {
			rc[r*n+i] = c[r*n+n-1-i]
		}
	}
	return ra, rb, rc
}

// checkTFExact compares the TF coefficients with the exact rational ones,
// den = det(sI − A) and num_ij = det [sI − A, −b_j; c_i, d_ij], each
// relative to the largest coefficient magnitude of its polynomial, and the
// response with the exact one at pts.
func checkTFExact(t *testing.T, n, m, p int, a, b, c, d []float64, pts []complex128, tol float64) {
	t.Helper()
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	cmp := func(what string, got []float64, want []float64) {
		t.Helper()
		got = trimLeadingFloatZeros(got)
		for len(want) > 0 && want[0] == 0 {
			want = want[1:]
		}
		scale := 0.0
		for _, v := range want {
			scale = max(scale, math.Abs(v))
		}
		if len(got) != len(want) {
			t.Fatalf("%s: got %v, exact %v", what, got, want)
		}
		for k := range want {
			if e := math.Abs(got[k] - want[k]); e > tol*scale {
				t.Fatalf("%s[%d]: got %v, exact %v (err %.3g)", what, k, got, want, e/scale)
			}
		}
	}
	den := make([]float64, n+1)
	for k, v := range fzCharPolyExact(a, n) {
		den[k], _ = v.Float64()
	}
	for i := range p {
		cmp("den", res.TF.Den[i], den)
		for j := range m {
			bj := make([]float64, n)
			for k := range n {
				bj[k] = b[k*m+j]
			}
			ex := fzPencilDetRat(n, 1, a, bj, c[i*n:(i+1)*n], []float64{d[i*m+j]})
			num := make([]float64, len(ex))
			for k, v := range ex {
				num[k], _ = v.Float64()
			}
			cmp("num", res.TF.Num[i][j], num)
		}
	}
	for _, s := range pts {
		want, ok := fzExactResponse(n, m, p, a, b, c, d, s)
		if !ok {
			continue
		}
		got := res.TF.eval(s)
		for i := range p {
			for j := range m {
				w := want[i*m+j]
				if e := cmplx.Abs(got[i][j]-w) / cmplx.Abs(w); e > tol {
					t.Errorf("TF(%v)[%d,%d] = %v, exact %v (rel %.3g)", s, i, j, got[i][j], w, e)
				}
			}
		}
	}
}

var wm67Points = []complex128{complex(-5.898, 7.430), complex(0.31, 1.7), complex(-2.3, 0.4), 2i, 10, 1i, 10i, -1}

// TestTransferFunctionTriangularSlowPoles: the staircase and Hessenberg
// rotations perturb A by eps·‖A‖, which in the Dgebal-graded basis lost the
// 3e-12 coupling the numerators are summed from (s² coefficient of num 0,0
// 0.33937 vs exact 0.33889, 6.7e-4 response error) and in the Osborne basis
// the near-origin poles (den constant 10% off). The triangular A converts
// in its given basis (WM67N5).
func TestTransferFunctionTriangularSlowPoles(t *testing.T) {
	a, b, c, d := wm67Model()
	checkTFExact(t, 3, 2, 2, a, b, c, d, wm67Points, 1e-10)
}

func TestTransferFunctionLowerTriangularSlowPoles(t *testing.T) {
	a, b, c, d := wm67Model()
	ra, rb, rc := reverseStates(3, 2, 2, a, b, c)
	checkTFExact(t, 3, 2, 2, ra, rb, rc, d, wm67Points, 1e-10)
}

// TestTransferFunctionTriangularSeeded checks seeded triangular and
// diagonal MIMO models with distinct moderate poles against the exact TF.
func TestTransferFunctionTriangularSeeded(t *testing.T) {
	r := rand.New(rand.NewPCG(67, 5))
	for trial := range 300 {
		n, m, p := 1+r.IntN(6), 1+r.IntN(3), 1+r.IntN(3)
		diag := trial%3 == 0
		a := make([]float64, n*n)
		for i := range n {
			a[i*n+i] = -float64(i+1) * (0.5 + r.Float64())
			if diag {
				continue
			}
			for j := i + 1; j < n; j++ {
				if r.IntN(3) > 0 {
					a[i*n+j] = r.NormFloat64()
				}
			}
		}
		rnd := func(k int) []float64 {
			v := make([]float64, k)
			for i := range v {
				v[i] = r.NormFloat64()
			}
			return v
		}
		b, c, d := rnd(n*m), rnd(p*n), rnd(p*m)
		if trial%2 == 1 {
			a, b, c = reverseStates(n, m, p, a, b, c)
		}
		t.Run("", func(t *testing.T) {
			checkTFExact(t, n, m, p, a, b, c, d, []complex128{0.3i, 1 + 2i, -0.7 + 0.1i, 5i}, 1e-11)
		})
	}
}

func BenchmarkTransferFunctionTriangular(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{3, 20} {
		a := make([]float64, n*n)
		for i := range n {
			a[i*n+i] = -float64(i + 1)
			for j := i + 1; j < n; j++ {
				a[i*n+j] = r.NormFloat64()
			}
		}
		bm, cm, dm := make([]float64, 2*n), make([]float64, 2*n), make([]float64, 4)
		for i := range bm {
			bm[i], cm[i] = r.NormFloat64(), r.NormFloat64()
		}
		sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, 2, bm), mat.NewDense(2, n, cm), mat.NewDense(2, 2, dm), 0)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("N%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := sys.TransferFunction(nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
