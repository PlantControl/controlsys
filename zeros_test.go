package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

func TestZeros_SISO_Known(t *testing.T) {
	// H(s) = (s+1)(s+2)/((s+3)(s+4)) → zeros at -1, -2
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 3, 2}}},
		Den: [][]float64{{1, 7, 12}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	want := []complex128{-1, -2}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_SISO_Integrator(t *testing.T) {
	// H(s) = 1/s → no zeros
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 0}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros, got %v", zeros)
	}
}

func TestZeros_SISO_RepeatedZeros(t *testing.T) {
	// H(s) = (s+1)²/((s+2)³) → zeros at -1 (multiplicity 2)
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 2, 1}}},
		Den: [][]float64{{1, 6, 12, 8}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	for _, z := range zeros {
		if cmplx.Abs(z-(-1)) > 1e-6 {
			t.Errorf("expected zero at -1, got %v", z)
		}
	}
}

func TestZeros_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros for gain system, got %v", zeros)
	}
}

func TestZeros_NoInputsOrOutputs(t *testing.T) {
	sys, _ := NewGain(&mat.Dense{}, 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros, got %v", zeros)
	}
}

func TestZeros_MIMO_InvertibleD(t *testing.T) {
	// 2×2 system with invertible D
	// zeros = eig(A - B*D⁻¹*C)
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	// A - B*D⁻¹*C = A - I*I⁻¹*I = A - I = [-1 1; -2 -4]
	// eig([-1 1; -2 -4]) = (-5±sqrt(25-4*(-1)(-4+2)))/2... let's compute:
	// char poly: λ²+5λ+6 = (λ+2)(λ+3) → zeros at -2, -3
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{-2, -3}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_Discrete(t *testing.T) {
	// H(z) = (z-0.5)/(z-0.9) → zero at 0.5
	tf := &TransferFunc{
		Num: [][][]float64{{{1, -0.5}}},
		Den: [][]float64{{1, -0.9}},
		Dt:  0.1,
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 1 {
		t.Fatalf("expected 1 zero, got %d", len(zeros))
	}
	if cmplx.Abs(zeros[0]-0.5) > 1e-10 {
		t.Errorf("expected zero at 0.5, got %v", zeros[0])
	}
}

func TestZeros_SeriesRoundtrip(t *testing.T) {
	// Two SISO systems in series: zeros of series = union of zeros
	// G1: (s+1)/(s+3), G2: (s+2)/(s+4)
	tf1 := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 3}},
	}
	tf2 := &TransferFunc{
		Num: [][][]float64{{{1, 2}}},
		Den: [][]float64{{1, 4}},
	}
	r1, _ := tf1.StateSpace()
	r2, _ := tf2.StateSpace()

	z1, err := r1.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	z2, err := r2.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(z1) != 1 || len(z2) != 1 {
		t.Fatalf("expected 1 zero each, got %d and %d", len(z1), len(z2))
	}
	if cmplx.Abs(z1[0]-(-1)) > 1e-10 {
		t.Errorf("G1 zero: expected -1, got %v", z1[0])
	}
	if cmplx.Abs(z2[0]-(-2)) > 1e-10 {
		t.Errorf("G2 zero: expected -2, got %v", z2[0])
	}
}

func TestZeros_SISO_ComplexZeros(t *testing.T) {
	// H(s) = (s²+1)/((s+1)(s+2)) → zeros at ±j
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0, 1}}},
		Den: [][]float64{{1, 3, 2}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	want := []complex128{complex(0, -1), complex(0, 1)}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_SISO_HighOrder(t *testing.T) {
	// H(s) = (s+1)(s+2)(s+3)/((s+4)(s+5)(s+6)(s+7)) → zeros at -1,-2,-3
	num := Poly{1, 1}.Mul(Poly{1, 2}).Mul(Poly{1, 3})
	den := Poly{1, 4}.Mul(Poly{1, 5}).Mul(Poly{1, 6}).Mul(Poly{1, 7})
	tf := &TransferFunc{
		Num: [][][]float64{{[]float64(num)}},
		Den: [][]float64{[]float64(den)},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{-1, -2, -3}
	assertZerosMatch(t, zeros, want, 1e-8)
}

func TestZeros_MIMO_9x2(t *testing.T) {
	// Reference: 9-state, 2-input, 2-output
	A := mat.NewDense(9, 9, []float64{
		-3.93, -0.00315, 0, 0, 0, 4.03e-5, 0, 0, 0,
		368, -3.05, 3.03, 0, 0, -3.77e-3, 0, 0, 0,
		27.4, 0.0787, -5.96e-2, 0, 0, -2.81e-4, 0, 0, 0,
		-0.0647, -5.2e-5, 0, -0.255, -3.35e-6, 3.6e-7, 6.33e-5, 1.94e-4, 0,
		3850, 17.3, -12.8, -12600, -2.91, -0.105, 12.7, 43.1, 0,
		22400, 18, 0, -35.6, -1.04e-4, -0.414, 90, 56.9, 0,
		0, 0, 2.34e-3, 0, 0, 2.22e-4, -0.203, 0, 0,
		0, 0, 0, -1.27, -1.00e-3, 7.86e-5, 0, -7.17e-2, 0,
		-2.2, -0.00177, 0, -8.44, -1.11e-4, 1.38e-5, 1.49e-3, 6.02e-3, -1e-10,
	})
	B := mat.NewDense(9, 2, []float64{
		0, 0,
		0, 0,
		1.56, 0,
		0, -5.13e-6,
		8.28, -1.55,
		0, 1.78,
		2.33, 0,
		0, -2.45e-2,
		0, 2.94e-5,
	})
	C := mat.NewDense(2, 9, []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1,
	})
	D := mat.NewDense(2, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{
		-2.64128629e+01,
		complex(-2.93193619, -0.419522621),
		-9.52183370e-03,
		complex(-2.93193619, 0.419522621),
		1.69789270e-01,
		5.46527700e-01,
	}
	if len(zeros) != len(want) {
		t.Fatalf("expected %d zeros, got %d: %v", len(want), len(zeros), zeros)
	}
	assertZerosMatch(t, zeros, want, 1e-5)
}

func TestZeros_MIMO_4x3_NonSquare(t *testing.T) {
	// Reference: 4-state, 3-input, 1-output (non-square)
	A := mat.NewDense(4, 4, []float64{
		-6.5, 0.5, 6.5, -6.5,
		-0.5, -5.5, -5.5, 5.5,
		-0.5, 0.5, 0.5, -6.5,
		-0.5, 0.5, -5.5, -0.5,
	})
	B := mat.NewDense(4, 3, []float64{
		0, 1, 0,
		2, 1, 2,
		3, 4, 3,
		3, 2, 3,
	})
	C := mat.NewDense(1, 4, []float64{1, 1, 0, 0})
	D := mat.NewDense(1, 3, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{-6, -7}
	assertZerosMatch(t, zeros, want, 1e-8)
}

func TestZeros_Reinschke(t *testing.T) {
	// Reinschke 1988: 6-state, 2-input, 3-output
	A := mat.NewDense(6, 6, []float64{
		0, 0, 1, 0, 0, 0,
		2, 0, 0, 3, 4, 0,
		0, 0, 5, 0, 0, 6,
		0, 7, 0, 0, 0, 0,
		0, 0, 0, 8, 9, 0,
		0, 0, 0, 0, 0, 0,
	})
	B := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		10, 0,
		0, 11,
	})
	C := mat.NewDense(3, 6, []float64{
		0, 12, 0, 0, 13, 0,
		14, 0, 0, 0, 0, 0,
		15, 0, 16, 0, 0, 0,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{-6.78662791, 3.09432022}
	assertZerosMatch(t, zeros, want, 1e-6)
}

func TestZeros_StaircaseExample(t *testing.T) {
	// Staircase zeros doc example: N=6, M=2, P=3
	// Expected: nu=2, rank=2, zeros at 2.0 and -1.0
	A := mat.NewDense(6, 6, []float64{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 3, 0, 0, 0,
		0, 0, 0, -4, 0, 0,
		0, 0, 0, 0, -1, 0,
		0, 0, 0, 0, 0, 3,
	})
	B := mat.NewDense(6, 2, []float64{
		0, -1,
		-1, 0,
		1, -1,
		0, 0,
		0, 1,
		-1, -1,
	})
	C := mat.NewDense(3, 6, []float64{
		1, 0, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 1,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rank != 2 {
		t.Errorf("rank: got %d, want 2", res.Rank)
	}
	want := []complex128{-1, 2}
	assertZerosMatch(t, res.Zeros, want, 1e-10)
}

func TestZeros_StaircaseNonSymmetric(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{
		1, 0.1, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 3, 0.2, 0, 0,
		0, 0, 0, -4, 0, 0,
		0, 0, 0, 0, -1, 0.15,
		0, 0, 0, 0, 0, 3,
	})
	B := mat.NewDense(6, 2, []float64{
		0, -1,
		-1, 0,
		1, -1,
		0, 0,
		0, 1,
		-1, -1,
	})
	C := mat.NewDense(3, 6, []float64{
		1, 0, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 1,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rank != 2 {
		t.Errorf("rank: got %d, want 2", res.Rank)
	}
	assertZerosMatch(t, res.Zeros, []complex128{-1}, 1e-10)
}

func assertZerosMatch(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected %d zeros, got %d: %v", len(want), len(got), got)
	}
	used := make([]bool, len(want))
	for _, g := range got {
		matched := false
		for j, w := range want {
			if !used[j] && cmplx.Abs(g-w) < tol {
				used[j] = true
				matched = true
				break
			}
		}
		if !matched {
			t.Errorf("unexpected zero %v, want %v", g, want)
		}
	}
}

func pencilMinSingularValue(t *testing.T, sys *System, z complex128) float64 {
	t.Helper()
	n, m, p := sys.Dims()
	R := mat.NewDense(2*(n+p), 2*(n+m), nil)
	set := func(i, j int, v complex128) {
		R.Set(i, j, real(v))
		R.Set(i, j+n+m, -imag(v))
		R.Set(i+n+p, j, imag(v))
		R.Set(i+n+p, j+n+m, real(v))
	}
	for i := range n {
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			set(i, j, complex(sys.A.At(i, j), 0)-z*complex(e, 0))
		}
		for j := range m {
			set(i, n+j, complex(sys.B.At(i, j), 0))
		}
	}
	for i := range p {
		for j := range n {
			set(n+i, j, complex(sys.C.At(i, j), 0))
		}
		for j := range m {
			set(n+i, n+j, complex(sys.D.At(i, j), 0))
		}
	}
	var svd mat.SVD
	if !svd.Factorize(R, mat.SVDNone) {
		t.Fatal("svd failed")
	}
	sv := svd.Values(nil)
	return sv[len(sv)-1] / sv[0]
}

func TestZeros_DescriptorPencil(t *testing.T) {
	E := mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, 0.2, 0, 0.4, 1.5})
	A0 := mat.NewDense(3, 3, []float64{-1, 2, 0.5, -0.5, -3, 1, 0.3, 0.7, -2})
	B0 := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1})
	C := mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 1, -1})
	for _, tc := range []struct {
		name string
		D    *mat.Dense
	}{
		{"D=0", mat.NewDense(2, 2, nil)},
		{"D invertible", mat.NewDense(2, 2, []float64{1, 0.2, 0.3, 2})},
	} {
		for _, dt := range []float64{0, 0.1} {
			var EA, EB mat.Dense
			EA.Mul(E, A0)
			EB.Mul(E, B0)
			desc, err := NewDescriptor(&EA, &EB, C, tc.D, E, dt)
			if err != nil {
				t.Fatal(err)
			}
			expl, err := New(A0, B0, C, tc.D, dt)
			if err != nil {
				t.Fatal(err)
			}
			want, err := expl.Zeros()
			if err != nil {
				t.Fatal(err)
			}
			if len(want) == 0 {
				t.Fatalf("%s: fixture has no zeros", tc.name)
			}
			got, err := desc.Zeros()
			if err != nil {
				t.Fatalf("%s dt=%v: %v", tc.name, dt, err)
			}
			if len(got) != len(want) {
				t.Fatalf("%s dt=%v: got %d zeros %v, want %v", tc.name, dt, len(got), got, want)
			}
			for _, z := range got {
				if r := pencilMinSingularValue(t, desc, z); r > 1e-12 {
					t.Errorf("%s dt=%v: zero %v not a rank drop of (A-zE,B;C,D): sigma_min/sigma_max=%.3g", tc.name, dt, z, r)
				}
			}
			pz, err := Pzmap(desc)
			if err != nil {
				t.Fatal(err)
			}
			if len(pz.Zeros) != len(want) {
				t.Errorf("%s dt=%v: Pzmap zeros %v, want %v", tc.name, dt, pz.Zeros, want)
			}
		}
	}
}

func TestZeros_DescriptorSISO(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{2, 1, 0.5, 3})
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	B := mat.NewDense(2, 1, []float64{1, 0.4})
	C := mat.NewDense(1, 2, []float64{1, -0.7})
	desc, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, []float64{0.5}), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := desc.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got zeros %v, want 2", got)
	}
	for _, z := range got {
		if r := pencilMinSingularValue(t, desc, z); r > 1e-12 {
			t.Errorf("zero %v not a rank drop: %.3g", z, r)
		}
	}
}

func pencilFiniteEigOracle(t *testing.T, sys *System) []complex128 {
	t.Helper()
	n, m, p := sys.Dims()
	if m != p {
		t.Fatal("oracle needs a square system")
	}
	k := n + m
	M := mat.NewDense(k, k, nil)
	N := mat.NewDense(k, k, nil)
	M.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.A)
	M.Slice(0, n, n, k).(*mat.Dense).Copy(sys.B)
	M.Slice(n, k, 0, n).(*mat.Dense).Copy(sys.C)
	M.Slice(n, k, n, k).(*mat.Dense).Copy(sys.D)
	N.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.E)
	ev, err := generalizedPoles(M, N, k)
	if err != nil {
		t.Fatal(err)
	}
	var out []complex128
	for _, v := range ev {
		if cmplx.Abs(v) < 1e6 {
			out = append(out, v)
		}
	}
	return out
}

func TestZeros_SingularDescriptor(t *testing.T) {
	E := mat.NewDense(3, 3, []float64{1, 2, 0, 0, 1, 1, 1, 3, 1})
	A := mat.NewDense(3, 3, []float64{-1, 2, 0.5, -0.5, -3, 1, 0.3, 0.7, -2})
	B := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1})
	C := mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 1, -1})
	for _, D := range []*mat.Dense{mat.NewDense(2, 2, nil), mat.NewDense(2, 2, []float64{1, 0.2, 0.3, 2})} {
		for _, dt := range []float64{0, 0.1} {
			sys, err := NewDescriptor(A, B, C, D, E, dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatalf("dt=%v: %v", dt, err)
			}
			want := pencilFiniteEigOracle(t, sys)
			if len(want) == 0 {
				t.Fatal("fixture has no finite zeros")
			}
			assertZerosMatch(t, res.Zeros, want, 1e-9)
			for _, z := range res.Zeros {
				if r := pencilMinSingularValue(t, sys, z); r > 1e-12 {
					t.Errorf("dt=%v: zero %v not a rank drop: %.3g", dt, z, r)
				}
			}
			if res.Rank != 2 {
				t.Errorf("dt=%v: normal rank %d, want 2", dt, res.Rank)
			}
			pz, err := Pzmap(sys)
			if err != nil {
				t.Fatal(err)
			}
			assertZerosMatch(t, pz.Zeros, want, 1e-9)
		}
	}
}

// algebraicDescriptorFixture embeds x2 = K x1 + L u as algebraic states and
// mixes the coordinates, so the explicit equivalent is known in closed form.
func algebraicDescriptorFixture(t *testing.T, dt float64) (desc, expl *System) {
	t.Helper()
	A11 := mat.NewDense(3, 3, []float64{-1, 2, 0, -0.5, -3, 1, 0.3, 0, -2})
	A12 := mat.NewDense(3, 2, []float64{0.4, -1, 1, 0.2, -0.3, 0.5})
	B1 := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1})
	K := mat.NewDense(2, 3, []float64{0.5, -1, 0.2, 1, 0.3, -0.7})
	L := mat.NewDense(2, 2, []float64{0.2, 1, -0.5, 0.1})
	C1 := mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 1, -1})
	C2 := mat.NewDense(2, 2, []float64{0.3, -1, 2, 0.4})
	D := mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3})

	var Ae, Be, Ce, De, t1 mat.Dense
	t1.Mul(A12, K)
	Ae.Add(A11, &t1)
	t1.Reset()
	t1.Mul(A12, L)
	Be.Add(B1, &t1)
	t1.Reset()
	t1.Mul(C2, K)
	Ce.Add(C1, &t1)
	t1.Reset()
	t1.Mul(C2, L)
	De.Add(D, &t1)
	expl, err := New(&Ae, &Be, &Ce, &De, dt)
	if err != nil {
		t.Fatal(err)
	}

	A := mat.NewDense(5, 5, nil)
	A.Slice(0, 3, 0, 3).(*mat.Dense).Copy(A11)
	A.Slice(0, 3, 3, 5).(*mat.Dense).Copy(A12)
	A.Slice(3, 5, 0, 3).(*mat.Dense).Copy(K)
	A.Slice(3, 5, 3, 5).(*mat.Dense).Copy(mat.NewDense(2, 2, []float64{-1, 0, 0, -1}))
	B := mat.NewDense(5, 2, nil)
	B.Slice(0, 3, 0, 2).(*mat.Dense).Copy(B1)
	B.Slice(3, 5, 0, 2).(*mat.Dense).Copy(L)
	C := mat.NewDense(2, 5, nil)
	C.Slice(0, 2, 0, 3).(*mat.Dense).Copy(C1)
	C.Slice(0, 2, 3, 5).(*mat.Dense).Copy(C2)
	E := mat.NewDense(5, 5, nil)
	for i := range 3 {
		E.Set(i, i, 1)
	}
	P := mat.NewDense(5, 5, []float64{1, 0.3, 0, 0.2, -0.1, 0, 1.2, 0.4, 0, 0.3, 0.5, 0, 0.9, 0.1, 0, -0.2, 0.1, 0, 1.1, 0.4, 0, 0.6, -0.3, 0, 1.3})
	Q := mat.NewDense(5, 5, []float64{0.8, 0, 0.1, -0.4, 0.2, 0.3, 1.1, 0, 0.2, 0, 0, 0.5, 1, 0, -0.3, 0.2, 0, -0.1, 0.9, 0.4, 0, 0.3, 0, 0.1, 1.2})
	var PA, PE, PB, CQ, PAQ, PEQ mat.Dense
	PA.Mul(P, A)
	PAQ.Mul(&PA, Q)
	PE.Mul(P, E)
	PEQ.Mul(&PE, Q)
	PB.Mul(P, B)
	CQ.Mul(C, Q)
	desc, err = NewDescriptor(&PAQ, &PB, &CQ, D, &PEQ, dt)
	if err != nil {
		t.Fatal(err)
	}
	return desc, expl
}

func TestZeros_AlgebraicStatesMatchExplicit(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		desc, expl := algebraicDescriptorFixture(t, dt)
		want := pencilFiniteEigOracle(t, &System{A: expl.A, B: expl.B, C: expl.C, D: expl.D, E: eyeDense(3)})
		if len(want) == 0 {
			t.Fatal("fixture has no zeros")
		}
		res, err := desc.ZerosDetail()
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		assertZerosMatch(t, res.Zeros, want, 1e-9)
		if res.Rank != 2 {
			t.Errorf("dt=%v: rank %d, want 2", dt, res.Rank)
		}
	}
}

func TestZeros_ImproperDescriptor(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	for _, tc := range []struct {
		d    float64
		want []complex128
	}{
		{0, []complex128{0}},
		{2, []complex128{2}},
	} {
		for _, dt := range []float64{0, 0.1} {
			sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, []float64{tc.d}), E, dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatalf("D=%v dt=%v: %v", tc.d, dt, err)
			}
			assertZerosMatch(t, res.Zeros, tc.want, 1e-12)
			if res.Rank != 1 {
				t.Errorf("D=%v dt=%v: rank %d, want 1", tc.d, dt, res.Rank)
			}
		}
	}
}

func TestZeros_NormalRank(t *testing.T) {
	mimo, err := New(
		mat.NewDense(3, 3, []float64{-1, 2, 0, -0.5, -3, 1, 0.3, 0, -2}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1}),
		mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 1, -1}),
		mat.NewDense(2, 2, []float64{1, 0.2, 0.3, 2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := mimo.ZerosDetail()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rank != 2 || len(res.Zeros) != 3 {
		t.Errorf("MIMO invertible D: rank %d zeros %d, want 2 and 3", res.Rank, len(res.Zeros))
	}
	siso, err := New(
		mat.NewDense(2, 2, []float64{-1, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 3}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if res, err = siso.ZerosDetail(); err != nil {
		t.Fatal(err)
	}
	if res.Rank != 1 || len(res.Zeros) != 1 {
		t.Errorf("SISO: rank %d zeros %v, want 1 and one zero", res.Rank, res.Zeros)
	}
}

func internalDelayZerosFixture(t *testing.T, dt float64, m int, D22 *mat.Dense) (sys, closed *System) {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1, 2, 0, -0.5, -3, 1, 0.3, 0, -2})
	if dt > 0 {
		A = mat.NewDense(3, 3, []float64{0.5, 0.2, 0, -0.1, 0.3, 0.4, 0.2, 0, -0.6})
	}
	B := mat.NewDense(3, m, []float64{1, 0, 0, 1, 1, -1}[:3*m])
	C := mat.NewDense(m, 3, []float64{1, 0.5, 0, 0, 1, -1}[:3*m])
	D := mat.NewDense(m, m, []float64{0.1, 0, -0.2, 0.3}[:m*m])
	if m == 1 {
		D = mat.NewDense(1, 1, []float64{0.1})
	}
	B2 := mat.NewDense(3, 1, []float64{0.7, -0.4, 1})
	C2 := mat.NewDense(1, 3, []float64{0.3, 1, -0.5})
	D12 := mat.NewDense(m, 1, []float64{0.6, -0.2}[:m])
	D21 := mat.NewDense(1, m, []float64{0.4, 0.9}[:m])
	var err error
	sys, err = New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}
	tau := 0.3
	if dt > 0 {
		tau = 2
	}
	if err := sys.SetInternalDelay([]float64{tau}, B2, C2, D12, D21, D22); err != nil {
		t.Fatal(err)
	}
	g := 1 / (1 - D22.At(0, 0))
	var Ac, Bc, Cc, Dc, t1 mat.Dense
	t1.Mul(B2, C2)
	t1.Scale(g, &t1)
	Ac.Add(A, &t1)
	t1.Reset()
	t1.Mul(B2, D21)
	t1.Scale(g, &t1)
	Bc.Add(B, &t1)
	t1.Reset()
	t1.Mul(D12, C2)
	t1.Scale(g, &t1)
	Cc.Add(C, &t1)
	t1.Reset()
	t1.Mul(D12, D21)
	t1.Scale(g, &t1)
	Dc.Add(D, &t1)
	closed, err = NewDescriptor(&Ac, &Bc, &Cc, &Dc, eyeDense(3), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys, closed
}

func TestZeros_InternalDelaySetToZero(t *testing.T) {
	singularE := mat.NewDense(3, 3, []float64{1, 2, 0, 0, 1, 1, 1, 3, 1})
	for _, E := range []*mat.Dense{nil, singularE} {
		for _, m := range []int{1, 2} {
			for _, dt := range []float64{0, 0.1} {
				sys, closed := internalDelayZerosFixture(t, dt, m, mat.NewDense(1, 1, []float64{0.25}))
				if E != nil {
					sys.E, closed.E = mat.DenseCopyOf(E), mat.DenseCopyOf(E)
				}
				want := pencilFiniteEigOracle(t, closed)
				got, err := sys.Zeros()
				if err != nil {
					t.Fatalf("E=%v m=%d dt=%v: %v", E != nil, m, dt, err)
				}
				assertZerosMatch(t, got, want, 1e-9)
			}
		}
	}
}

func TestZeros_InternalDelayAlgebraicLoop(t *testing.T) {
	for _, m := range []int{1, 2} {
		sys, _ := internalDelayZerosFixture(t, 0, m, mat.NewDense(1, 1, []float64{1}))
		if _, err := sys.Zeros(); !errors.Is(err, ErrAlgebraicLoop) {
			t.Errorf("m=%d: err %v, want ErrAlgebraicLoop", m, err)
		}
	}
}

func TestZeros_ZeroE(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	C := mat.NewDense(2, 2, []float64{1, 0.5, 0, 1})
	for _, dt := range []float64{0, 0.1} {
		sys, err := NewDescriptor(A, B, C, mat.NewDense(2, 2, []float64{0.2, 0, 0, 0.1}), mat.NewDense(2, 2, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.ZerosDetail()
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		if len(res.Zeros) != 0 || res.Rank != 2 {
			t.Errorf("dt=%v: zeros %v rank %d, want none and rank 2", dt, res.Zeros, res.Rank)
		}
	}
}

func TestZeros_SISOKeepsCancelledInvariantZeros(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{2, 1, 0.5, 3})
	cases := []struct {
		name string
		A, B []float64
		C    []float64
		D    float64
		dt   float64
		want []complex128
	}{
		{"continuous", []float64{-1, 1, 0, -2}, []float64{0, 1}, []float64{1, 1}, 0, 0, []complex128{-2}},
		{"discrete D", []float64{0.5, 0.3, 0, -0.4}, []float64{0.2, 1}, []float64{0, 1}, 0.5, 0.1, []complex128{0.5, -2.4}},
	}
	for _, tc := range cases {
		for _, desc := range []bool{false, true} {
			A, B := mat.NewDense(2, 2, tc.A), mat.NewDense(2, 1, tc.B)
			var sys *System
			var err error
			if desc {
				var EA, EB mat.Dense
				EA.Mul(E, A)
				EB.Mul(E, B)
				sys, err = NewDescriptor(&EA, &EB, mat.NewDense(1, 2, tc.C), mat.NewDense(1, 1, []float64{tc.D}), E, tc.dt)
			} else {
				sys, err = New(A, B, mat.NewDense(1, 2, tc.C), mat.NewDense(1, 1, []float64{tc.D}), tc.dt)
			}
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatalf("%s desc=%v: %v", tc.name, desc, err)
			}
			if res.Rank != 1 {
				t.Errorf("%s desc=%v: rank %d, want 1", tc.name, desc, res.Rank)
			}
			assertZerosMatch(t, res.Zeros, tc.want, 1e-9)
			for _, z := range res.Zeros {
				if r := pencilMinSingularValue(t, sys, z); r > 1e-12 {
					t.Errorf("%s desc=%v: zero %v not a rank drop: %.3g", tc.name, desc, z, r)
				}
			}
		}
	}
}

func TestZerosDetail_StaticGainRank(t *testing.T) {
	cases := []struct {
		d    *mat.Dense
		rank int
	}{
		{mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 2},
		{mat.NewDense(2, 3, []float64{1, 2, 3, 2, 4, 6}), 1},
		{mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0},
	}
	for _, tc := range cases {
		for _, dt := range []float64{0, 0.1} {
			sys, err := NewGain(tc.d, dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatal(err)
			}
			if res.Rank != tc.rank || len(res.Zeros) != 0 {
				t.Errorf("dt=%g D=%v: rank %d zeros %v, want rank %d, no zeros", dt, mat.Formatted(tc.d), res.Rank, res.Zeros, tc.rank)
			}
		}
	}
}

func TestZerosDetail_InvalidInput(t *testing.T) {
	var nilSys *System
	if _, err := nilSys.ZerosDetail(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil ZerosDetail error = %v, want ErrInvalidArgument", err)
	}
	if _, err := nilSys.Zeros(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Zeros error = %v, want ErrInvalidArgument", err)
	}
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.B.Set(1, 0, math.Inf(1))
	if _, err := sys.Zeros(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Inf B error = %v, want ErrInvalidArgument", err)
	}
}

func benchPIDLoop(b *testing.B) *System {
	b.Helper()
	pid, err := NewPID(0.8, 0.3, 0.2, 0.05, 0)
	if err != nil {
		b.Fatal(err)
	}
	c, err := pid.System()
	if err != nil {
		b.Fatal(err)
	}
	plant, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -1, -3, -3}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		b.Fatal(err)
	}
	loop, err := Series(c, plant)
	if err != nil {
		b.Fatal(err)
	}
	return loop
}

func benchGridModels(b *testing.B) map[string]*System {
	return map[string]*System{"PIDLoop": benchPIDLoop(b), "SS10x2x2": benchSysNonSym(b, 10, 2, 2)}
}

func BenchmarkZerosGrid(b *testing.B) {
	for _, name := range []string{"PIDLoop", "SS10x2x2"} {
		sys := benchGridModels(b)[name]
		b.Run("Zeros/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := sys.Zeros(); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("DefaultFrequencyGrid/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := sys.DefaultFrequencyGrid(0); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("Bode/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := sys.Bode(nil, 0); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("FreqResponse/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				omega, err := sys.DefaultFrequencyGrid(0)
				if err != nil {
					b.Fatal(err)
				}
				if _, err := sys.FreqResponse(omega); err != nil {
					b.Fatal(err)
				}
			}
		})
		if name == "PIDLoop" {
			b.Run("Margin/"+name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := Margin(sys); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

type gridCorpusModel struct {
	name string
	sys  *System
}

func gridCorpus(t *testing.T) []gridCorpusModel {
	t.Helper()
	must := func(sys *System, err error) *System {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	rng := rand.New(rand.NewPCG(7, 25))
	randDense := func(r, c int, scale float64) *mat.Dense {
		d := mat.NewDense(max(r, 1), max(c, 1), nil)
		if r == 0 || c == 0 {
			return &mat.Dense{}
		}
		for i := range r {
			for j := range c {
				d.Set(i, j, scale*rng.NormFloat64())
			}
		}
		return d
	}
	randSys := func(n, m, p int, dZero bool, dt float64) *System {
		A := randDense(n, n, 1/math.Sqrt(float64(n)))
		for i := range n {
			if dt > 0 {
				A.Set(i, i, A.At(i, i)*0.5)
			} else {
				A.Set(i, i, A.At(i, i)-1-float64(i)/4)
			}
		}
		D := mat.NewDense(p, m, nil)
		if !dZero {
			D = randDense(p, m, 1)
		}
		return must(New(A, randDense(n, m, 1), randDense(p, n, 1), D, dt))
	}

	var out []gridCorpusModel
	add := func(name string, sys *System) { out = append(out, gridCorpusModel{name, sys}) }

	pid, err := NewPID(0.8, 0.3, 0.2, 0.05, 0)
	if err != nil {
		t.Fatal(err)
	}
	c := must(pid.System())
	plant := must(New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -1, -3, -3}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, nil), 0))
	loop := must(Series(c, plant))
	add("pid-loop", loop)
	add("pid-closed", must(Feedback(loop, nil, -1)))
	add("pid", c)
	for _, n := range []int{2, 10, 40} {
		add(fmt.Sprintf("nonsym-%d-2x2", n), benchSysNonSym(t, n, 2, 2))
		add(fmt.Sprintf("nonsym-%d-3x2", n), benchSysNonSym(t, n, 2, 3))
		add(fmt.Sprintf("integrator-%d-2x3", n), benchIntegratorMIMOSystem(t, n, 3, 2))
	}
	add("static-gain", must(NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)))
	add("no-zeros", must(New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)))
	add("integrator", must(New(mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)))

	for k := range 60 {
		n := 1 + rng.IntN(45)
		m := 1 + rng.IntN(4)
		p := 1 + rng.IntN(4)
		dt := 0.0
		if k%3 == 2 {
			dt = 0.1
		}
		add(fmt.Sprintf("rand-%d-n%d-m%d-p%d-dt%g", k, n, m, p, dt), randSys(n, m, p, k%2 == 0, dt))
	}

	desc := randSys(6, 2, 2, false, 0)
	E := mat.NewDense(6, 6, nil)
	for i := range 4 {
		E.Set(i, i, 1+float64(i))
		E.Set(i, i+1, 0.3)
	}
	add("descriptor-singular", must(NewDescriptor(desc.A, desc.B, desc.C, desc.D, E, 0)))
	Einv := mat.NewDense(6, 6, nil)
	for i := range 6 {
		Einv.Set(i, i, 2+float64(i))
		if i > 0 {
			Einv.Set(i, i-1, 0.5)
		}
	}
	add("descriptor-invertible", must(NewDescriptor(desc.A, desc.B, desc.C, desc.D, Einv, 0)))

	delayed := randSys(5, 2, 2, true, 0)
	if err := delayed.SetInputDelay([]float64{0.3, 0}); err != nil {
		t.Fatal(err)
	}
	if err := delayed.SetOutputDelay([]float64{0, 1.5}); err != nil {
		t.Fatal(err)
	}
	add("io-delay", delayed)
	dl := randSys(4, 1, 1, true, 0)
	if err := dl.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	add("internal-delay", must(Feedback(dl, nil, -1)))
	dd := randSys(4, 1, 1, true, 0.1)
	if err := dd.SetInputDelay([]float64{3}); err != nil {
		t.Fatal(err)
	}
	add("discrete-delay", dd)
	add("discrete-internal-delay", must(Feedback(dd, nil, -1)))
	for _, n := range []int{31, 32, 33, 34, 35, 65} {
		add(fmt.Sprintf("strictly-proper-siso-%d", n), randSys(n, 1, 1, true, 0))
	}
	for _, n := range []int{33, 34, 35} {
		add(fmt.Sprintf("strictly-proper-2x2-%d", n), randSys(n, 2, 2, true, 0))
	}
	return out
}

// TestDefaultFrequencyGridBitIdentical pins DefaultFrequencyGrid bit for bit
// to a frozen copy of the v2.0.0 implementation fed the current Zeros, and
// Zeros to the v2.0.0 staircase/eig values to 1e-9 relative on a broad,
// well-conditioned corpus (balancing and refinement change the last bits;
// the corpus models that differ agree with exact det P(s) roots to ≤ 5e-13).
func TestDefaultFrequencyGridBitIdentical(t *testing.T) {
	sameBits := func(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }
	for _, mdl := range gridCorpus(t) {
		z, err := mdl.sys.Zeros()
		if err != nil {
			t.Fatalf("%s: %v", mdl.name, err)
		}
		wantZ, err := legacyZeros(mdl.sys)
		if err != nil {
			t.Fatalf("%s: legacy: %v", mdl.name, err)
		}
		if len(z) != len(wantZ) {
			t.Errorf("%s: %d zeros, want %d", mdl.name, len(z), len(wantZ))
		} else {
			used := make([]bool, len(z))
			for _, w := range wantZ {
				best := -1
				for i, v := range z {
					if !used[i] && (best < 0 || cmplx.Abs(v-w) < cmplx.Abs(z[best]-w)) {
						best = i
					}
				}
				used[best] = true
				if d := cmplx.Abs(z[best] - w); d > 1e-9*math.Max(cmplx.Abs(w), 1) {
					t.Errorf("%s: zero %v, legacy %v", mdl.name, z[best], w)
				}
			}
		}
		for _, nPoints := range []int{0, 1, 37} {
			omega, err := mdl.sys.DefaultFrequencyGrid(nPoints)
			if err != nil {
				t.Fatalf("%s: %v", mdl.name, err)
			}
			want, err := legacyDefaultFrequencyGrid(mdl.sys, z, nPoints)
			if err != nil {
				t.Fatalf("%s: legacy: %v", mdl.name, err)
			}
			if len(omega) != len(want) {
				t.Fatalf("%s: len %d, want %d", mdl.name, len(omega), len(want))
			}
			for i := range omega {
				if !sameBits(omega[i], want[i]) {
					t.Errorf("%s/%d: omega[%d] = %v, want %v", mdl.name, nPoints, i, omega[i], want[i])
				}
			}
		}
	}
}

func legacyDefaultFrequencyGrid(sys *System, zeros []complex128, nPoints int) ([]float64, error) {
	if nPoints <= 0 {
		nPoints = 200
	}
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	roots := append(append([]complex128{}, poles...), zeros...)
	wMin, wMax := autoFreqRange(sys, roots, systemDelays(sys))
	omega := logspace(math.Log10(wMin), math.Log10(wMax), nPoints)
	if nPoints > 1 {
		omega[nPoints-1] = wMax
	}
	return omega, nil
}

func legacyZeros(sys *System) ([]complex128, error) {
	sys, err := zeroInternalDelays(sys)
	if err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	switch {
	case m == 0 || p == 0 || n == 0:
		return nil, nil
	case sys.IsDescriptor():
		return legacyDescriptorZeros(sys)
	}
	return legacyMimoZeros(sys)
}

func legacyDescriptorZeros(sys *System) ([]complex128, error) {
	n, m, p := sys.Dims()
	var svd mat.SVD
	if !svd.Factorize(sys.E, mat.SVDFull) {
		return nil, ErrSingularTransform
	}
	sv := svd.Values(nil)
	tol := float64(n) * eps() * sv[0]
	r := 0
	for r < n && sv[r] > tol {
		r++
	}
	if r == 0 {
		return nil, nil
	}
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)
	var At, tmp, Bt, Ct mat.Dense
	tmp.Mul(U.T(), sys.A)
	At.Mul(&tmp, &V)
	Bt.Mul(U.T(), sys.B)
	Ct.Mul(sys.C, &V)
	for i := range r {
		s := 1 / sv[i]
		for j := range n {
			At.Set(i, j, At.At(i, j)*s)
		}
		for j := range m {
			Bt.Set(i, j, Bt.At(i, j)*s)
		}
	}
	q := n - r
	mh, ph := m+q, p+q
	at, bt, ct := At.RawMatrix(), Bt.RawMatrix(), Ct.RawMatrix()
	dh := make([]float64, ph*mh)
	copyBlock(dh, mh, 0, 0, at.Data, at.Stride, r, r, q, q)
	copyBlock(dh, mh, 0, q, bt.Data, bt.Stride, r, 0, q, m)
	copyBlock(dh, mh, q, 0, ct.Data, ct.Stride, 0, r, p, q)
	dRaw := sys.D.RawMatrix()
	copyBlock(dh, mh, q, q, dRaw.Data, dRaw.Stride, 0, 0, p, m)
	ah := make([]float64, r*r)
	bh := make([]float64, r*mh)
	ch := make([]float64, ph*r)
	copyBlock(ah, r, 0, 0, at.Data, at.Stride, 0, 0, r, r)
	copyBlock(bh, mh, 0, 0, at.Data, at.Stride, 0, r, r, q)
	copyBlock(bh, mh, 0, q, bt.Data, bt.Stride, 0, 0, r, m)
	copyBlock(ch, r, 0, 0, at.Data, at.Stride, r, 0, q, r)
	copyBlock(ch, r, q, 0, ct.Data, ct.Stride, 0, 0, p, r)
	aug := &System{A: mat.NewDense(r, r, ah), B: mat.NewDense(r, mh, bh), C: mat.NewDense(ph, r, ch), D: mat.NewDense(ph, mh, dh), Dt: sys.Dt}
	return legacyMimoZeros(aug)
}

func legacyMimoZeros(sys *System) ([]complex128, error) {
	n, m, p := sys.Dims()
	if m == p {
		var luD mat.LU
		luD.Factorize(sys.D)
		if !nearZero(luD.Det()) {
			var DinvC mat.Dense
			if err := luD.SolveTo(&DinvC, false, sys.C); err == nil {
				var BDinvC, M mat.Dense
				BDinvC.Mul(sys.B, &DinvC)
				M.Sub(sys.A, &BDinvC)
				var eig mat.Eigen
				if !eig.Factorize(&M, mat.EigenNone) {
					return nil, ErrSchurFailed
				}
				zeros := eig.Values(nil)
				sortZeros(zeros)
				return zeros, nil
			}
		}
	}
	afData, bfData, nu, _, _ := zerosStaircase(sys.A, sys.B, sys.C, sys.D, n, m, p)
	if nu == 0 {
		return nil, nil
	}
	alphar := make([]float64, nu)
	alphai := make([]float64, nu)
	beta := make([]float64, nu)
	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nu, afData, nu, bfData, nu,
		alphar, alphai, beta, nil, 1, nil, 1, work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nu, afData, nu, bfData, nu,
		alphar, alphai, beta, nil, 1, nil, 1, work, lwork) {
		return nil, ErrSchurFailed
	}
	betaTol := float64(nu) * eps()
	var zeros []complex128
	for j := range nu {
		if math.Abs(beta[j]) <= betaTol {
			continue
		}
		re := alphar[j] / beta[j]
		im := alphai[j] / beta[j]
		if math.Abs(im) < math.Abs(re)*eps()*100 {
			im = 0
		}
		zeros = append(zeros, complex(re, im))
	}
	sortZeros(zeros)
	return zeros, nil
}

// exactZero is a root of det P(s) (or of the gcd of the maximal minors of
// P(s) for non-square models) of the stored float64 data and its first-order
// relative sensitivity to entrywise ε perturbations of A, B, C, D (-1 when
// not estimated, 0 for non-square models).
type exactZero struct {
	z    complex128
	cond float64
}

// TestZerosMatchExactPencilRoots checks Zeros against the exact roots of the
// Rosenbrock pencil (big.Rat interpolation of det P(s), square-free
// factorization and 400-bit Durand–Kerner) on models from the zeros sweeps
// where the unrefined eig(A − B·D⁻¹·C) or the unscaled staircase was off:
// small zeros under a huge ‖B·D⁻¹·C‖ (G1810, W11a/b, W65), nearly singular
// D with O(1) entries (F121, D194), tiny but well-conditioned D (D114, D256,
// D306), huge zeros on the staircase path (D8), moderate pairs (G1979) and
// badly scaled decoupling zeros of non-square models (Z30, Z37). Every exact
// zero must be matched to max(1e-10, 1e3·cond) relative; base errors ran
// from 1e-8 to O(1), and drops.
func TestZerosMatchExactPencilRoots(t *testing.T) {
	cases := []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
		want       []exactZero
	}{
		{name: "G1810", n: 6, m: 1, p: 1,
			a:    []float64{-4.628428561556428e-09, 9.390187725384587e-09, -548.4640744021577, -217687.2057123227, 249730.66280976016, -23519.368640980494, -4.2736528272644885e-09, 4.72385326335633e-09, -548.4640743974401, -217687.20571232314, 249730.6628097602, -23519.3686409805, 0, 0, -501.2562893380045, -217738.85947645028, 249735.5275064159, -23519.826792945838, 0, 0, 0, -199498.743710662, 229229.11293210942, -21588.554443561596, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 10000},
			b:    []float64{0.9240197354105631, -0.12123724277303606, -0.9888813762654848, -0.407811976826337, 0.7707029240652387, 0.8201362356363325},
			c:    []float64{-0.2451804419447299, -0.020933597353654593, 0.25871566918744526, -0.14634655591116785, -0.1640910398424718, 0.2524443983048766},
			d:    []float64{9.575742709670695e-10},
			want: []exactZero{{complex(3.54750354912185e+08, 0), 7e-16}, {complex(9295.94528231417, 0), 6e-15}, {complex(10000, 0), 1e-15}, {complex(5.088661354797265e-10, 0), 9e-14}, {complex(-305614.05503905524, 0), 6e-16}, {complex(-517.0344035763295, 0), 7e-16}}},
		{name: "W11a", n: 4, m: 2, p: 2,
			a:    []float64{-23.805839809865468, 61.36401194495934, 7.120192067913228, 19.42191689445281, 61.36401194495934, -158.21546714380403, -18.357484731254203, -50.07414679841618, 7.120192067913223, -18.35748473125419, -178.43089731519538, 58.82091435527909, 19.421916894452817, -50.074146798416194, 58.8209143552791, -39.54779573113518},
			b:    []float64{-0.7909677119144166, 0.9274784307440358, 0.37797774271298024, 0.6083513145322546, 0.9931849187581926, -0.6020119026848236, 0.15337386203786346, -0.9304262721047533},
			c:    []float64{989992.4966004454, -186512.36942257575, -869397.4903498252, 748646.6455973986, 848100.0317104083, -775565.8785102495, -346635.31783502584, 999693.0420352065},
			d:    []float64{-8.481000317104082e-07, -5.748239465332691e-07, 9.502325919585293e-07, 7.539022543433046e-07},
			want: []exactZero{{complex(-36.34051311530382, 0), 1e-15}, {complex(6.317824589537833e+11, 0), 9e-16}, {complex(-191.1041566901274, 0), 3e-16}, {complex(-2.3461404190672883e+13, 0), 3e-15}}},
		{name: "W11b", n: 4, m: 2, p: 2,
			a:    []float64{-34.71026808605036, 29.974632950792188, 15.514653898610378, 67.8016799237025, 29.974632950792195, -25.89382546250945, -13.399869336592557, -58.55971120699381, 15.514653898610375, -13.399869336592557, -196.9801266658022, 13.175509403948787, 67.80167992370254, -58.55971120699383, 13.175509403948793, -142.41577978563802},
			b:    []float64{-0.7909677119144166, 0.9274784307440358, 0.37797774271298024, 0.6083513145322546, 0.9931849187581926, -0.6020119026848236, 0.15337386203786346, -0.9304262721047533},
			c:    []float64{98999.24966004454, -18651.236942257572, -86939.74903498252, 74864.66455973986, 84810.00317104082, -77556.58785102495, -34663.531783502585, 99969.30420352065},
			d:    []float64{-8.481000317104082e-07, -5.748239465332691e-07, 9.502325919585293e-07, 7.539022543433046e-07},
			want: []exactZero{{complex(-75.91789456274333, 0), 9e-16}, {complex(6.317824588174007e+10, 0), 9e-16}, {complex(-178.5253406871323, 0), 4e-16}, {complex(-2.3461404191819517e+12, 0), -1e+00}}},
		{name: "G1979", n: 6, m: 3, p: 3,
			a:    []float64{-0.4351166318816339, 9.43722775777014, -30.660199510687317, 65.72888972254256, -106.19990331878945, 20.09017165596497, -1.519966938431621, 10.995197507712309, -32.47296320499042, 70.27105075712662, -113.58250766356409, -55.28701062141697, -1.4080398217004204, 9.259172525911397, -29.08172359881628, 62.7374951474162, -101.249923782297, 224.11486716390027, -1.874382864237419, 12.325812134082375, -40.04479587250213, 86.21971032804784, -139.05363725689088, -448.49357648665307, -1.0345686506367002, 6.803251924065735, -22.10278978819612, 47.589163628814596, -76.75097977370918, 752.0846455658686, -12.256967630772008, 80.60097178272554, -261.8609976408535, 563.8087311218214, -909.2997825138773, 11.25271216864685},
			b:    []float64{-0.5755504782013432, 0.9964679075571247, -0.43057540477662676, 0.6340062895737355, 0.3474682721812599, -0.9848422969392376, 0.9147423578045313, -0.8105731959717306, -0.09631291684576058, -0.1446212711617233, -0.7811230330551143, 0.9333151120639219, -0.9921143990644506, 0.3926742032638758, 0.5956343152752086, -0.38615760806092936, 0.9912028118634736, -0.6146521488144698},
			c:    []float64{-9.876194833474772e-07, 4.6773184024707355e-07, 6.851938352639844e-07, -9.107638766871487e-07, -9.631291684575705e-08, 9.730377990279994e-07, -6.543155223459515e-07, 9.271460038316638e-07, 5.4842262350017956e-08, -9.626058663135665e-07, 5.675586048115438e-07, 5.956343152752114e-07, -1.3276747223059479e-08, 9.505089147582911e-07, -6.013024834811546e-07, -5.617192758811469e-07, 9.644984462781496e-07, -6.190529399442055e-08},
			d:    []float64{2.1860657172338502, 1.0296212821954618, -0.2893777139829187, -2.7270498599002444, -1.7601458530866816, -0.5153535181679144, 3.1096369392465353, 2.3884346936219694, 1.290151126790327},
			want: []exactZero{{complex(-209.97726537175123, -357.271614724225), 3e-07}, {complex(-0.00010136961368205038, 0), 4e-10}, {complex(3.833530693147111, 0), 4e-09}, {complex(-3.683550216487786e+08, 0), 9e-02}, {complex(1.0000015059536167, 0), 6e-15}, {complex(-209.97726537175123, 357.271614724225), 3e-07}}},
		{name: "W65", n: 2, m: 1, p: 1,
			a:    []float64{0.6863883501311974, 0.8550271126517397, 1.6708868232151954, 0.11946342859322057},
			b:    []float64{-0.026829097335336855, -0.1334147041955197},
			c:    []float64{-70618.48253548339, 155535.57430302448},
			d:    []float64{-4.10051011325682e-09},
			want: []exactZero{{complex(-4.598477256509821e+12, 0), -1e+00}, {complex(0.8008011739084017, 0), -1e+00}}},
		{name: "D8", n: 6, m: 3, p: 3,
			a:    []float64{-0.412768609516037, -0.240784472162629, -0.11924526085290527, 1.233127854639846, 0.8679855776553147, 1.9345345454233371, -1.5401229670978438, 1.4220475846147167, -1.5240593735052743, -0.7405482000183068, 0.19093427151090697, 1.9271540491919557, 0.5509162723168061, 0.589460942707459, -0.4455378704801447, -0.6220213017886462, 0.8245082871863683, -0.8226700279350527, -1.4356856009067593, 0.581157250486645, 0.39151189744744436, 1.7322663892190082, 2.125361433967366, -0.8475242734127031, 1.3541857324774893, -1.5312381266261075, -0.9239899870968635, -1.199956608184256, -1.093368340287153, 0.6601673763838691, 0.3136710534386341, 0.6812569188190344, -0.7626432583883, -1.2039410688270946, 0.6954768771417259, -0.8524853061283231},
			b:    []float64{631.9058125876079, -613.9160050043799, 403.34704211771424, -1241.130851417686, -209.9839190387966, -2260.3622175631717, -2690.3826700592217, 226.422136276644, -563.0301465914309, -453.22713822500896, -938.8964621983381, -1248.7831827595717, 657.0937548615594, -1960.8952740040909, 2272.9709098569465, 358.737705145464, -1384.2261920594642, -1030.6545042962375},
			c:    []float64{-0.0018958791170249672, 0.006924299707614492, 0.002784012917262262, -0.0007099248984085829, 0.00589144936987199, 0.006454605491146864, 0.0046488243662851725, -0.0038060891499823164, 0.007119314153008266, 0.0002652516826731899, -0.004223854226323019, 0.005827623674862694, 0.0009830307278821209, -0.00343066080050516, 0.005657337070105757, -9.029201304748292e-05, -0.0009285433910692707, -0.00876203602415996},
			d:    []float64{-2.284706401545822e-06, 1.1739417999459802e-06, 5.482314510538824e-07, -5.090639689317314e-06, -5.005129080843793e-06, 3.554424474525157e-06, -1.0328999442160287e-06, 2.082405481246147e-06, -4.097518138235058e-06},
			want: []exactZero{{complex(1.2164369759315174e+07, 0), 8e-16}, {complex(1.2612676651578944e+06, 0), 2e-15}, {complex(4.838807339767204, 0), 2e-15}, {complex(-4.78195302481522e+06, 0), 6e-16}, {complex(-1.042081178619403, 0), 1e-15}, {complex(1.7028748881632714, 0), 2e-15}}},
		{name: "D194", n: 5, m: 1, p: 1,
			a:    []float64{1.1569321083185757, -2.1636525804714166, 1.0202515547493227, -2.12498229062491, -2.026488057313932, 2.1721782463242527, -0.5512658498981817, 0.35483698352533644, 0.3911648711602486, 1.1550586198219115, -1.511272309456252, 2.4818085446715275, -1.2562202245282013, 1.3998571092997278, -0.9971438227611733, 0.38304921958036314, -0.013975176914699183, 0.3538834956475452, 2.339900272750084, 0.38029647594304816, 0.9608504243014008, -0.47340607225278775, 0.9860005626681394, -1.587466120422297, 1.4180421593145265},
			b:    []float64{42953.626242139675, 9354.898412927336, 30189.357833808834, 31462.50235265519, -37188.54956805072},
			c:    []float64{2220.57969155985, 3014.9377857470813, -2346.449935268758, -1717.464321124552, -7426.541398574466},
			d:    []float64{4.930743719132478e-09},
			want: []exactZero{{complex(1.5480993085057777, -1.2444046074422581), 9e-16}, {complex(1.5480993085057777, 1.2444046074422581), 9e-16}, {complex(-0.8568362577160114, 2.26656664415271), 7e-16}, {complex(-5.575125445956288e+16, 0), 5e-16}, {complex(-0.8568362577160114, -2.26656664415271), 7e-16}}},
		{name: "D256", n: 5, m: 3, p: 3,
			a:    []float64{-0.7298675053953363, -0.4546176888962651, 0.4199947323468507, -1.6429479383981325, 0.1505624472655963, -0.5462509178519209, -0.24280294935326002, -0.6503911422126291, -0.6922820624217165, 0.5225625964589786, -0.1365198473797083, 0.8642288070714044, -0.03600248838654352, 0.8414656088595299, -1.953433147225942, 0.021803611025643654, -0.23704202626449006, -2.9661149112877045, -1.67522167176854, -0.7263130834663969, 0.32254586388105744, 1.123053359224696, 0.16231575503841533, 0.2582628320181192, -0.5658727503934194},
			b:    []float64{1863.777600891911, 2884.5925268566643, 764.0302715332979, 2780.1709458218534, 282.41835787412174, -2789.6992382307817, -2100.2780593914936, 2205.474499147786, 1204.53272621663, -1211.5143935717738, -882.7037157688906, 1400.896281583576, -608.2295034937025, -2081.900582808672, -1227.1907763696533},
			c:    []float64{-12199.804383504905, -86261.14920142603, 56014.042485510414, -57800.39013999763, 44040.12555297552, -60858.97535443216, -14153.1910697796, 23187.386691798707, 1451.6098218561626, 65275.893949025754, -14412.209755305394, -13049.460718189226, -143981.46581309047, -1433.0052504340842, 117570.63967773535},
			d:    []float64{1.1389778029309243e-11, -2.5565586727851207e-10, 9.034574317837915e-11, 2.589212070834289e-10, 1.1660566912991947e-10, -1.2675396085202366e-11, -9.572574335303939e-11, 7.318057348688277e-11, -7.980261912722933e-11},
			want: []exactZero{{complex(2.004532914015177, 0), 2e-15}, {complex(8.872864625864023e+17, 0), 7e-16}, {complex(-3.4987093276399164, 0), 5e-16}, {complex(-9.442287499531911e+18, 0), 1e-15}, {complex(-6.008260269107745e+17, 0), -1e+00}}},
		{name: "D114", n: 6, m: 3, p: 3,
			a:    []float64{0.3636777869087973, 0.5924257345103151, 0.7567293548407281, 0.9197179570016182, 1.4658763436559534, 1.026037011980008, 1.1543907438952767, -1.1093957660137472, 0.3340714761040145, 0.46271634674900575, -0.5747874583896377, 0.05285554674754922, -1.212563196641513, -0.9149034918065064, -0.06012085857171712, 0.9227954529981108, 2.3865943559091174, -0.43371203022069515, 1.0315421826548155, -1.299539178062922, -1.7681261423647063, 0.30939352269241593, -1.099318887471908, -0.3724439621123856, -1.3741524732922406, -1.8896576514461, 0.9012520940151665, 0.8361276024656585, -0.8761748351381371, -0.34104950776998016, 0.08700649230300161, -0.18991192561325854, 0.5851321824125731, -1.0822414013412485, -2.060538485634095, -0.31587872470892453},
			b:    []float64{-45706.865886678126, 24346.36475833484, 16143.242601652253, -25988.10055824961, -19903.505582589314, -12997.745245334248, -3917.5691810819517, 7842.384332748299, 704.7415700089254, 35326.870352650134, -313.5544697634481, -36073.503098627574, 18787.399353895325, 6764.16705694327, 9774.229415462443, -21880.58046946788, 27537.267994548056, 10731.311520139165},
			c:    []float64{0.095392195814638, 0.07210078930850032, 0.0771898533698513, 0.03856543964158747, -0.30428988282557695, 0.006210402998507205, 0.13781145840177636, 0.10126996315372415, 0.12208690454330723, 0.07416168315772927, 0.172430979804973, 0.05615788554607702, 0.02526548874909237, 0.06560075331275693, -0.056976481782874615, -0.09004499277693988, -0.18193910446438447, -0.0043995642228730775},
			d:    []float64{-3.3709848822824566e-10, -6.542923524229764e-10, 1.9506987735635456e-10, 5.233195073268875e-10, 2.949604007396958e-10, -1.924368533639942e-10, -2.868634981876496e-11, -2.8360553140315434e-10, -1.6679429195286365e-10},
			want: []exactZero{{complex(4.185104308992641e+13, 0), 1e-15}, {complex(1.0743683854815482e+13, 0), 8e-16}, {complex(-0.202313165846461, -0.41789700976967115), 4e-15}, {complex(-1.1588877825786842e+13, 0), 7e-16}, {complex(-0.202313165846461, 0.41789700976967115), 4e-15}, {complex(-2.919456439977083, 0), 8e-16}}},
		{name: "F121", n: 5, m: 3, p: 3,
			a:    []float64{-8733.712820574747, 14254.971168201877, 7996.088092834309, 10446.031200280524, 36960.61559349819, -10570.21484435091, 17537.57324374765, 10428.407000621031, 17376.939894769355, 51103.152959142484, 15254.780785227898, -25899.547028335117, -16537.459792931848, -18024.394543234073, -64594.72666064914, 10750.61577930167, -18429.222174981845, -12064.290231669847, 5445.000927106816, -19899.36247692271, -6142.615898695661, 10529.967348530012, 6893.214538111574, 531.2306402213795, 16734.913506352677},
			b:    []float64{0.2824139943038164, 0.685494772297671, -0.9745527253399078, 0.9998796714034104, -0.518176033203694, -0.4766813676836281, 0.2525192874280824, -0.9627177361043177, 0.7195293103748337, -0.8647824442191548, 0.0031242997956527586, 0.861627863056279, -0.7151758691155459, 0.9643892291755364, -0.25856042221911274},
			c:    []float64{-1392.5561018720143, 2785.795668520113, -408.68124822016534, -2521.550901075446, 2039.0634454860824, 524.9308679214796, 2465.5928815720918, -2119.1317774118393, -1095.4064926797712, 2827.3987585276645, 2195.5346482604355, 985.7832364692156, -2832.9215193393047, 845.9247054217901, 2285.9642561094124},
			d:    []float64{1.041633208556786, 3.1113463794462586, 4.689846369323316, 0.31071745060253275, -1.900299653049017, -3.811301225283856, -1.6450204999907685, 0.5788765492714962, 2.7113817197446886},
			want: []exactZero{{complex(3.5125969410203617e+18, 0), 3e-01}, {complex(-316.623261149899, 0), 8e-14}, {complex(-20904.416128190645, 0), 7e-15}, {complex(-969.5801792591714, 0), 1e-13}, {complex(8328.017608593902, 0), 6e-15}}},
		{name: "Z30", n: 4, m: 3, p: 1,
			a:    []float64{-0.03905939546737262, 504.30668104587033, 0, -9.044444812666236e-09, -0.00013645298880559747, 0.08606444041427717, 0, -2.7702122630761502e-11, -63.24964803930255, 69352.80296585718, -0.4804571675649545, 3.6402117760698606e-06, 4.162997477740486e+08, -4.408928120089685e+09, 0, -0.04293247717761972},
			b:    []float64{0, 0, 0, 0, 0, 0, 0.018438400410559427, -0.1663392453869051, -0.42296936929719814, 0, 0, 0},
			c:    []float64{10415.733473427958, -2.1828180290954965e+06, -3.053930810246446, -4.812248685453375e-05},
			d:    []float64{2.1944915934963483, 0.17698655614264766, -0.7810540500242575},
			want: []exactZero{{complex(0.5576046109291253, -2.152841112098554), 0e+00}, {complex(0.5576046109291253, 2.152841112098554), 0e+00}, {complex(-1.1111366540889658, 0), 0e+00}}},
		{name: "Z37", n: 3, m: 3, p: 2,
			a:    []float64{-3.735896371147827, 515861.2027986823, 8.031981117924415, -3.2742150587866983e-09, 0.358418577736386, 2.6861047342329046e-07, -0.059579589000530495, 8.357995848880155e+06, -0.3886758805798943},
			b:    []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			c:    []float64{1.794557390936581e-07, -29.840546123529585, -2.658497323082792e-06, 4.979836612506655e-08, -52.46854186169821, -6.434165285164795e-06},
			d:    []float64{-11.330828363088486, 1.1364868848954401, -6.05944856224958, 9.874368910297743, -30.907786410484025, 9.625991012176556},
			want: []exactZero{{complex(-1.6722379328139818, 0), 0e+00}, {complex(1.4806359791383596, 0), 0e+00}, {complex(-3.574551720315713, 0), 0e+00}}},
		{name: "D306", n: 6, m: 2, p: 2,
			a:    []float64{1.191871107341627, -1.1080300723877006, -1.012238049752284, -1.1589792862754946, -0.3620259870540774, 0.5161148649672304, 0.25300551213917255, -1.8822263349501176, 0.28059197869554353, 0.7428040919494721, -0.6763943778473808, -0.09996958110497278, -0.7422256644530573, 1.3687334607863395, 0.4576561133942358, 0.018552230864765762, 1.3495561248019285, 0.9102061063846314, 0.7195899527762455, 2.035253365920688, 0.3254058665438855, -2.213377646909225, 0.30054135426421735, -0.4617089155135351, -0.028193422709293314, 1.1150893233498889, -0.37708589133670317, 0.3945296102698054, 0.21853911697925765, 0.4607444824670617, -1.9539646074109815, 0.31494402980430114, -2.5306595406736987, 1.2783652684854308, 1.0378347052732328, -0.9610769541128008},
			b:    []float64{3861.1447525715544, 4497.223460883343, 12426.807505521334, 17770.628640797808, -898.1723247015412, 11422.037682166285, 9216.315484433255, 25868.702149039822, 19319.158165538684, -10994.232339182949, 5190.007895171476, 4041.536632740479},
			c:    []float64{-2.488348291727041, 0.4952488351439305, -18.451409231447627, 10.230645345440017, -10.800667698451383, -12.151682463796071, -0.10969320222083732, 7.448750551107804, 20.244212550966633, -23.75431217964804, -24.42690497059527, -18.22160697794404},
			d:    []float64{3.0971709810038334e-10, -5.352845029146683e-11, 1.4314826442968677e-10, -1.7507178131493212e-10},
			want: []exactZero{{complex(1.0342757885917628e+15, 0), 9e-16}, {complex(-1.4361111359206833, 1.168274617702393), 1e-15}, {complex(-4.233081560593275, 0), 4e-16}, {complex(-1.982469299728712e+15, 0), 2e-15}, {complex(-1.4361111359206833, -1.168274617702393), 1e-15}, {complex(6.499252753162701, 0), 1e-15}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := New(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b),
				mat.NewDense(tc.p, tc.n, tc.c), mat.NewDense(tc.p, tc.m, tc.d), 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := sys.Zeros()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("%d zeros %v, want %d", len(got), got, len(tc.want))
			}
			used := make([]bool, len(got))
			for _, w := range tc.want {
				best := -1
				for i, z := range got {
					if !used[i] && (best < 0 || cmplx.Abs(z-w.z) < cmplx.Abs(got[best]-w.z)) {
						best = i
					}
				}
				used[best] = true
				tol := math.Max(1e-10, 1e3*w.cond)
				if e := cmplx.Abs(got[best]-w.z) / cmplx.Abs(w.z); !(e <= tol) {
					t.Errorf("zero %v, exact %v: relative error %.2g > %.2g\nzeros=%v", got[best], w.z, e, tol, got)
				}
			}
		})
	}
}
