package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestZeros_SISO_Known(t *testing.T) {
	// H(s) = (s+1)(s+2)/((s+3)(s+4)) → zeros at -1, -2
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 3, 2}}},
		Den: [][]float64{{1, 7, 12}},
	}
	res, err := tf.StateSpace(nil)
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
	res, err := tf.StateSpace(nil)
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
	res, err := tf.StateSpace(nil)
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
	res, err := tf.StateSpace(nil)
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
	r1, _ := tf1.StateSpace(nil)
	r2, _ := tf2.StateSpace(nil)

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
	res, err := tf.StateSpace(nil)
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
	res, err := tf.StateSpace(nil)
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
