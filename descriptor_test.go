package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestGeneralizedPoles_Standard(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	expected := []complex128{complex(-1, 0), complex(-2, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestGeneralizedPoles_InfiniteFiltered(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3, 1, 0, -1})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 0})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 {
		t.Fatalf("got %d finite poles, want 1", len(poles))
	}
	if math.Abs(real(poles[0])-(-3)) > 1e-10 {
		t.Errorf("expected pole -3, got %v", poles[0])
	}
}

func TestDescriptorSystem_Poles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	E := mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = E

	if !sys.IsDescriptor() {
		t.Error("expected IsDescriptor() = true")
	}

	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	// E^{-1}*A eigenvalues: [2,0;0,1]^{-1}*[-1,2;0,-3] = [-0.5,1;0,-3]
	expected := []complex128{complex(-0.5, 0), complex(-3, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestDescriptorSystem_Copy(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	cp := sys.Copy()
	if cp.E == nil {
		t.Fatal("Copy did not preserve E")
	}
	cp.E.Set(0, 0, 99)
	if sys.E.At(0, 0) == 99 {
		t.Error("Copy shares E backing array")
	}
}

func TestDescriptorSystem_NonDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if sys.IsDescriptor() {
		t.Error("standard system should not be descriptor")
	}
}

func TestDescriptorSystem_UnsupportedOperationsRejectExplicitly(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	ops := []struct {
		name string
		run  func() error
	}{
		{name: "TransferFunction", run: func() error {
			_, err := sys.TransferFunction(nil)
			return err
		}},
		{name: "Inv", run: func() error {
			_, err := Inv(sys)
			return err
		}},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			if err := op.run(); !errors.Is(err, ErrDescriptorUnsupported) {
				t.Fatalf("got %v, want ErrDescriptorUnsupported", err)
			}
		})
	}
}

func TestDescriptorSystem_FrequencyResponseUsesDescriptorPencil(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{4}),
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0, 1, 5}
	response, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for k, frequency := range omega {
		want := 12/complex(1, 2*frequency) + 0.5
		if got := response.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12 {
			t.Fatalf("response at %g = %v, want %v", frequency, got, want)
		}
	}
	evaluated, err := sys.EvalFr(1i)
	if err != nil {
		t.Fatalf("EvalFr: %v", err)
	}
	if want := 12/complex(1, 2) + 0.5; cmplx.Abs(evaluated[0][0]-want) > 1e-12 {
		t.Fatalf("EvalFr = %v, want %v", evaluated[0][0], want)
	}
}

func TestDescriptorSystem_FrequencyResponsePivotsNonSymmetricPencil(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{0, -2, 3, -1})
	e := mat.NewDense(2, 2, []float64{0, 1, 1, 0.5})
	b := mat.NewDense(2, 2, []float64{1, -1, 2, 0.5})
	c := mat.NewDense(2, 2, []float64{1, 0.25, -0.5, 2})
	d := mat.NewDense(2, 2, []float64{0.1, 0.2, -0.3, 0.4})
	sys, err := NewDescriptor(a, b, c, d, e, 0)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 1)
	pencil := [4]complex128{
		s*complex(e.At(0, 0), 0) - complex(a.At(0, 0), 0),
		s*complex(e.At(0, 1), 0) - complex(a.At(0, 1), 0),
		s*complex(e.At(1, 0), 0) - complex(a.At(1, 0), 0),
		s*complex(e.At(1, 1), 0) - complex(a.At(1, 1), 0),
	}
	det := pencil[0]*pencil[3] - pencil[1]*pencil[2]
	inv := [4]complex128{pencil[3] / det, -pencil[1] / det, -pencil[2] / det, pencil[0] / det}
	want := make([][]complex128, 2)
	for i := range 2 {
		want[i] = make([]complex128, 2)
		for j := range 2 {
			for k := range 2 {
				for l := range 2 {
					want[i][j] += complex(c.At(i, k), 0) * inv[k*2+l] * complex(b.At(l, j), 0)
				}
			}
			want[i][j] += complex(d.At(i, j), 0)
		}
	}

	got, err := sys.EvalFr(s)
	if err != nil {
		t.Fatalf("EvalFr: %v", err)
	}
	response, err := sys.FreqResponse([]float64{1})
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for i := range 2 {
		for j := range 2 {
			if diff := cmplx.Abs(got[i][j] - want[i][j]); diff > 1e-12 {
				t.Fatalf("EvalFr(%d,%d) diff=%g: got %v, want %v", i, j, diff, got[i][j], want[i][j])
			}
			if diff := cmplx.Abs(response.At(0, i, j) - want[i][j]); diff > 1e-12 {
				t.Fatalf("FreqResponse(%d,%d) diff=%g: got %v, want %v", i, j, diff, response.At(0, i, j), want[i][j])
			}
		}
	}
}

func TestDescriptorSystem_IdentityDescriptorUsesStandardOperations(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	if sys.IsDescriptor() {
		t.Fatal("identity E should be treated as standard form")
	}

	ops := []struct {
		name string
		run  func() error
	}{
		{name: "TransferFunction", run: func() error {
			_, err := sys.TransferFunction(nil)
			return err
		}},
		{name: "FreqResponse", run: func() error {
			_, err := sys.FreqResponse([]float64{1})
			return err
		}},
		{name: "EvalFr", run: func() error {
			_, err := sys.EvalFr(1i)
			return err
		}},
		{name: "Inv", run: func() error {
			_, err := Inv(sys)
			return err
		}},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			if err := op.run(); err != nil {
				t.Fatalf("got %v, want nil", err)
			}
		})
	}
}

// index1Descriptor returns the descriptor model P·E0·T, P·A0·T, P·B0, C0·T
// with E0 = diag(I₂, 0) and its hand-reduced explicit oracle obtained by
// eliminating the algebraic state ξ₃ = −(A0₃₁ξ₁ + A0₃₂ξ₂ + B0₃u)/A0₃₃.
// With delays, [B2; C2; D12, D21, D22] ride along as extra inputs/outputs.
func index1Descriptor(t *testing.T, dt float64, delays bool) (desc, oracle *System) {
	t.Helper()
	a0 := []float64{-1, 2, 0.5, 0.3, -3, 1, 0.4, -0.6, 2}
	if dt > 0 {
		a0 = []float64{0.5, 0.2, 0.4, -0.1, 0.7, 0.3, 0.2, -0.5, 1.5}
	}
	A0 := mat.NewDense(3, 3, a0)
	B0 := mat.NewDense(3, 2, []float64{1, 0.3, 0, 1, 0.5, -1})
	C0 := mat.NewDense(2, 3, []float64{1, 0, 0.7, 0.4, 1, -0.2})
	D0 := mat.NewDense(2, 2, []float64{0.2, 0.1, 0, -0.3})
	B20 := mat.NewDense(3, 1, []float64{0.3, -0.2, 0.6})
	C20 := mat.NewDense(1, 3, []float64{0.2, -0.4, 0.5})
	D12 := mat.NewDense(2, 1, []float64{0.1, -0.2})
	D21 := mat.NewDense(1, 2, []float64{0, 0})
	D22 := mat.NewDense(1, 1, []float64{0.05})
	P := mat.NewDense(3, 3, []float64{1, 0.2, -0.3, 0.1, 1.5, 0.2, 0.4, -0.1, 1})
	T := mat.NewDense(3, 3, []float64{0.9, 0.1, 0.3, -0.2, 1.1, 0.4, 0.5, 0, 1.2})

	var e, a, b, c, b2, c2 mat.Dense
	e.Mul(P, mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 0}))
	e.Mul(&e, T)
	a.Mul(P, A0)
	a.Mul(&a, T)
	b.Mul(P, B0)
	c.Mul(C0, T)
	b2.Mul(P, B20)
	c2.Mul(C20, T)
	desc, err := NewDescriptor(&a, &b, &c, D0, &e, dt)
	if err != nil {
		t.Fatal(err)
	}

	a33 := A0.At(2, 2)
	reduce := func(top, left, right float64) float64 { return top - left*right/a33 }
	As := mat.NewDense(2, 2, nil)
	Bs := mat.NewDense(2, 2, nil)
	Cs := mat.NewDense(2, 2, nil)
	Ds := mat.NewDense(2, 2, nil)
	B2s := mat.NewDense(2, 1, nil)
	C2s := mat.NewDense(1, 2, nil)
	D12s := mat.NewDense(2, 1, nil)
	D21s := mat.NewDense(1, 2, nil)
	D22s := mat.NewDense(1, 1, []float64{reduce(D22.At(0, 0), C20.At(0, 2), B20.At(2, 0))})
	for i := range 2 {
		for j := range 2 {
			As.Set(i, j, reduce(A0.At(i, j), A0.At(i, 2), A0.At(2, j)))
			Bs.Set(i, j, reduce(B0.At(i, j), A0.At(i, 2), B0.At(2, j)))
			Cs.Set(i, j, reduce(C0.At(i, j), C0.At(i, 2), A0.At(2, j)))
			Ds.Set(i, j, reduce(D0.At(i, j), C0.At(i, 2), B0.At(2, j)))
		}
		B2s.Set(i, 0, reduce(B20.At(i, 0), A0.At(i, 2), B20.At(2, 0)))
		C2s.Set(0, i, reduce(C20.At(0, i), C20.At(0, 2), A0.At(2, i)))
		D12s.Set(i, 0, reduce(D12.At(i, 0), C0.At(i, 2), B20.At(2, 0)))
		D21s.Set(0, i, reduce(D21.At(0, i), C20.At(0, 2), B0.At(2, i)))
	}
	oracle, err = New(As, Bs, Cs, Ds, dt)
	if err != nil {
		t.Fatal(err)
	}
	if delays {
		tau := []float64{0.4}
		if dt > 0 {
			tau = []float64{3}
		}
		if err := desc.SetInternalDelay(tau, &b2, &c2, D12, D21, D22); err != nil {
			t.Fatal(err)
		}
		if err := oracle.SetInternalDelay(tau, B2s, C2s, D12s, D21s, D22s); err != nil {
			t.Fatal(err)
		}
	}
	return desc, oracle
}

// index2Descriptor mixes a slow block (A_s, B_s) with the nilpotent chain
// ξ₄' = ξ₃, 0 = ξ₄ + β·u, so ξ₄ = −β·u and ξ₃ = −β·u'. Output weights c3 on
// ξ₃ make the model improper; with c3 = 0 the oracle feedthrough is D − c4·β.
func index2Descriptor(t *testing.T, dt float64, improper bool) (desc, oracle *System) {
	t.Helper()
	as := []float64{-1, 2, 0.5, -3}
	if dt > 0 {
		as = []float64{0.5, 0.2, -0.1, 0.7}
	}
	beta := []float64{0.6, -0.4}
	c3 := []float64{0, 0}
	if improper {
		c3 = []float64{0.3, 0}
	}
	c4 := []float64{0.8, -0.5}
	E0 := mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0})
	A0 := mat.NewDense(4, 4, []float64{as[0], as[1], 0, 0, as[2], as[3], 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})
	B0 := mat.NewDense(4, 2, []float64{1, 0.3, 0, 1, 0, 0, beta[0], beta[1]})
	C0 := mat.NewDense(2, 4, []float64{1, 0, c3[0], c4[0], 0.4, 1, c3[1], c4[1]})
	D0 := mat.NewDense(2, 2, []float64{0.2, 0.1, 0, -0.3})
	P := mat.NewDense(4, 4, []float64{1, 0.2, -0.3, 0.1, 0.1, 1.5, 0.2, 0, 0.4, -0.1, 1, 0.3, 0, 0.2, -0.1, 1.1})
	T := mat.NewDense(4, 4, []float64{0.9, 0.1, 0.3, 0, -0.2, 1.1, 0.4, 0.1, 0.5, 0, 1.2, -0.3, 0.1, 0.2, 0, 1})
	var e, a, b, c mat.Dense
	e.Mul(P, E0)
	e.Mul(&e, T)
	a.Mul(P, A0)
	a.Mul(&a, T)
	b.Mul(P, B0)
	c.Mul(C0, T)
	desc, err := NewDescriptor(&a, &b, &c, D0, &e, dt)
	if err != nil {
		t.Fatal(err)
	}
	Ds := mat.NewDense(2, 2, nil)
	for i := range 2 {
		for j := range 2 {
			Ds.Set(i, j, D0.At(i, j)-c4[i]*beta[j])
		}
	}
	oracle, err = New(mat.NewDense(2, 2, as), mat.NewDense(2, 2, []float64{1, 0.3, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}), Ds, dt)
	if err != nil {
		t.Fatal(err)
	}
	return desc, oracle
}

func TestProperExplicitFormMatchesHandReduction(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, delays := range []bool{false, true} {
			desc, oracle := index1Descriptor(t, dt, delays)
			got, err := desc.properExplicitForm()
			if err != nil {
				t.Fatalf("dt=%g delays=%v: %v", dt, delays, err)
			}
			if got.IsDescriptor() || got.E != nil {
				t.Fatalf("dt=%g: reduced model keeps E", dt)
			}
			descMatEqual(t, got.D, oracle.D, 1e-12)
			if delays {
				descMatEqual(t, got.LFT.D12, oracle.LFT.D12, 1e-12)
				descMatEqual(t, got.LFT.D21, oracle.LFT.D21, 1e-12)
				descMatEqual(t, got.LFT.D22, oracle.LFT.D22, 1e-12)
			}
			for _, s := range []complex128{0.3 + 0.7i, -0.2 + 2i, 1.7} {
				assertFreqEqual(t, got, oracle, s, 1e-11)
			}
		}
	}
}

func TestProperExplicitFormIndex2(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		desc, oracle := index2Descriptor(t, dt, false)
		got, err := desc.properExplicitForm()
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		if n, _, _ := got.Dims(); n != 2 {
			t.Fatalf("dt=%g: reduced order %d, want 2", dt, n)
		}
		descMatEqual(t, got.D, oracle.D, 1e-11)
		for _, s := range []complex128{0.3 + 0.7i, -0.2 + 2i} {
			assertFreqEqual(t, got, oracle, s, 1e-11)
		}

		improper, _ := index2Descriptor(t, dt, true)
		if _, err := improper.properExplicitForm(); !errors.Is(err, ErrImproperModel) {
			t.Fatalf("dt=%g improper err = %v, want ErrImproperModel", dt, err)
		}
	}
}

func descMatEqual(t *testing.T, got, want *mat.Dense, tol float64) {
	t.Helper()
	if !mat.EqualApprox(got, want, tol) {
		t.Fatalf("got\n%v\nwant\n%v", mat.Formatted(got), mat.Formatted(want))
	}
}

func assertFreqEqual(t *testing.T, got, want *System, s complex128, tol float64) {
	t.Helper()
	g, err := got.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	w, err := want.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	for i := range w {
		for j := range w[i] {
			if cmplx.Abs(g[i][j]-w[i][j]) > tol*math.Max(1, cmplx.Abs(w[i][j])) {
				t.Fatalf("G(%v)[%d][%d] = %v, want %v", s, i, j, g[i][j], w[i][j])
			}
		}
	}
}

func TestProperExplicitFormIndex1TwoAlgebraicStates(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		a0 := []float64{-1, 2, 0.5, 0.2, 0.3, -3, 1, -0.4, 0.4, -0.6, 2, 0.7, 0.1, 0.5, -0.3, 1.5}
		if dt > 0 {
			a0[0], a0[5] = 0.5, 0.7
		}
		A0 := mat.NewDense(4, 4, a0)
		B0 := mat.NewDense(4, 2, []float64{1, 0.3, 0, 1, 0.5, -1, 0.2, 0.8})
		C0 := mat.NewDense(2, 4, []float64{1, 0, 0.7, -0.3, 0.4, 1, -0.2, 0.6})
		D0 := mat.NewDense(2, 2, []float64{0.2, 0.1, 0, -0.3})
		P := mat.NewDense(4, 4, []float64{1, 0.2, -0.3, 0.1, 0.1, 1.5, 0.2, 0, 0.4, -0.1, 1, 0.3, 0, 0.2, -0.1, 1.1})
		T := mat.NewDense(4, 4, []float64{0.9, 0.1, 0.3, 0, -0.2, 1.1, 0.4, 0.1, 0.5, 0, 1.2, -0.3, 0.1, 0.2, 0, 1})
		var e, a, b, c mat.Dense
		e.Mul(P, mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
		e.Mul(&e, T)
		a.Mul(P, A0)
		a.Mul(&a, T)
		b.Mul(P, B0)
		c.Mul(C0, T)
		desc, err := NewDescriptor(&a, &b, &c, D0, &e, dt)
		if err != nil {
			t.Fatal(err)
		}

		a22 := A0.Slice(2, 4, 2, 4)
		var xa, xb mat.Dense
		if err := xa.Solve(a22, A0.Slice(2, 4, 0, 2)); err != nil {
			t.Fatal(err)
		}
		if err := xb.Solve(a22, B0.Slice(2, 4, 0, 2)); err != nil {
			t.Fatal(err)
		}
		var As, Bs, Cs, Ds, tmp mat.Dense
		tmp.Mul(A0.Slice(0, 2, 2, 4), &xa)
		As.Sub(A0.Slice(0, 2, 0, 2), &tmp)
		tmp.Mul(A0.Slice(0, 2, 2, 4), &xb)
		Bs.Sub(B0.Slice(0, 2, 0, 2), &tmp)
		tmp.Mul(C0.Slice(0, 2, 2, 4), &xa)
		Cs.Sub(C0.Slice(0, 2, 0, 2), &tmp)
		tmp.Mul(C0.Slice(0, 2, 2, 4), &xb)
		Ds.Sub(D0, &tmp)
		oracle, err := New(&As, &Bs, &Cs, &Ds, dt)
		if err != nil {
			t.Fatal(err)
		}

		got, err := desc.properExplicitForm()
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		descMatEqual(t, got.D, oracle.D, 1e-11)
		for _, s := range []complex128{0.3 + 0.7i, -0.2 + 2i} {
			assertFreqEqual(t, got, oracle, s, 1e-11)
		}
	}
}

func siso00(t *testing.T, sys *System) *System {
	t.Helper()
	out := sys.Copy()
	n, _, _ := sys.Dims()
	out.B = mat.DenseCopyOf(sys.B.Slice(0, n, 0, 1))
	out.C = mat.DenseCopyOf(sys.C.Slice(0, 1, 0, n))
	out.D = mat.NewDense(1, 1, []float64{sys.D.At(0, 0)})
	return out
}

func assertConvertedTwin(t *testing.T, label string, got, want *System) {
	t.Helper()
	if got.IsDescriptor() || got.StateName != nil {
		t.Fatalf("%s: E=%v StateName=%v, want explicit model without state names", label, got.E, got.StateName)
	}
	if got.Dt != want.Dt {
		t.Fatalf("%s: Dt = %g, want %g", label, got.Dt, want.Dt)
	}
	points := []complex128{0.3 + 0.7i, -0.2 + 2i, 1.7}
	if got.IsDiscrete() {
		points = []complex128{cmplx.Rect(1, 0.3), cmplx.Rect(1, 2.1), cmplx.Rect(1.3, 0.8), 0.4i}
	}
	for _, s := range points {
		assertFreqEqual(t, got, want, s, 1e-12)
	}
	if !got.IsDiscrete() {
		return
	}
	_, m, _ := got.Dims()
	const steps = 30
	u := mat.NewDense(m, steps, nil)
	for i := range m {
		for k := range steps {
			u.Set(i, k, math.Sin(0.3*float64(k*(i+1)))+0.5*float64(i))
		}
	}
	rw, err := want.Simulate(u, nil, nil)
	if err != nil {
		t.Fatalf("%s: oracle Simulate: %v", label, err)
	}
	rg, err := got.Simulate(u, nil, nil)
	if err != nil {
		t.Fatalf("%s: Simulate: %v", label, err)
	}
	if !mat.EqualApprox(rw.Y, rg.Y, 1e-12) {
		t.Fatalf("%s: simulated outputs differ\nwant %v\ngot  %v", label, mat.Formatted(rw.Y), mat.Formatted(rg.Y))
	}
}

func TestConversionsOfSingularDescriptorMatchReducedTwin(t *testing.T) {
	type conv func(*System) (*System, error)
	for _, delays := range []bool{false, true} {
		cases := map[string]conv{
			"ZOH":     func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{}) },
			"FOH":     func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{Method: C2DMethodFOH}) },
			"Impulse": func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{Method: C2DMethodImpulse}) },
			"Tustin":  func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{Method: C2DMethodTustin}) },
			"TustinWarp": func(s *System) (*System, error) {
				return s.C2D(0.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: 3})
			},
		}
		desc, twin := index1Descriptor(t, 0, delays)
		desc.StateName = []string{"x1", "x2", "x3"}
		for name, f := range cases {
			label := name
			if delays {
				label += "/internal"
			}
			want, err := f(twin)
			if err != nil {
				if _, derr := f(desc); derr == nil {
					t.Fatalf("%s: descriptor converted where twin errs: %v", label, err)
				}
				t.Logf("%s: unsupported for both: %v", label, err)
				continue
			}
			got, err := f(desc)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			assertConvertedTwin(t, label, got, want)
		}

		ddesc, dtwin := index1Descriptor(t, 0.1, delays)
		ddesc.StateName = []string{"x1", "x2", "x3"}
		dcases := map[string]conv{
			"D2C/ZOH":    func(s *System) (*System, error) { return s.D2C(D2COptions{Method: C2DMethodZOH}) },
			"D2C/Tustin": func(s *System) (*System, error) { return s.D2C(D2COptions{Method: C2DMethodTustin}) },
			"D2C/FOH":    func(s *System) (*System, error) { return s.D2C(D2COptions{Method: C2DMethodFOH}) },
			"Undiscretize": func(s *System) (*System, error) {
				return s.D2C(D2COptions{Method: C2DMethodTustin})
			},
			"D2D/ZOH":    func(s *System) (*System, error) { return s.D2D(0.05, D2DOptions{}) },
			"D2D/Tustin": func(s *System) (*System, error) { return s.D2D(0.05, D2DOptions{Method: C2DMethodTustin}) },
		}
		for name, f := range dcases {
			label := name
			if delays {
				label += "/internal"
			}
			want, err := f(dtwin)
			if err != nil {
				if _, derr := f(ddesc); derr == nil {
					t.Fatalf("%s: descriptor converted where twin errs: %v", label, err)
				}
				t.Logf("%s: unsupported for both: %v", label, err)
				continue
			}
			got, err := f(ddesc)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			assertConvertedTwin(t, label, got, want)
		}
	}
}

func TestSISOConversionsOfSingularDescriptorMatchReducedTwin(t *testing.T) {
	cdesc, ctwin := index1Descriptor(t, 0, false)
	cdesc, ctwin = siso00(t, cdesc), siso00(t, ctwin)
	ddesc, dtwin := index1Descriptor(t, 0.1, false)
	ddesc, dtwin = siso00(t, ddesc), siso00(t, dtwin)
	type conv func(*System) (*System, error)
	cases := []struct {
		name       string
		desc, twin *System
		f          conv
	}{
		{"Matched", cdesc, ctwin, func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{Method: C2DMethodMatched}) }},
		{"D2C/Matched", ddesc, dtwin, func(s *System) (*System, error) { return s.D2C(D2COptions{Method: C2DMethodMatched}) }},
		{"LeastSquares", cdesc, ctwin, func(s *System) (*System, error) {
			rSys, r, err := s.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 0})
			if err != nil {
				return nil, err
			}
			if r.FitOrder != 2 {
				return nil, errors.New("fit order is not the reduced order")
			}
			return rSys, nil
		}},
	}
	for _, c := range cases {
		want, err := c.f(c.twin)
		if err != nil {
			t.Fatalf("%s: twin: %v", c.name, err)
		}
		got, err := c.f(c.desc)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		assertConvertedTwin(t, c.name, got, want)
	}
}

func TestConversionResultOfSingularDescriptor(t *testing.T) {
	desc, twin := index1Descriptor(t, 0, true)
	want, err := twin.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := desc.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertConvertedTwin(t, "C2D", got, want)
	if _, _, err := desc.C2DMap(0.1, C2DOptions{}); !errors.Is(err, ErrDescriptorInitialState) {
		t.Fatalf("C2DMap err = %v, want ErrDescriptorInitialState", err)
	}

	ddesc, dtwin := index1Descriptor(t, 0.1, false)
	for name, f := range map[string]func(*System) (*System, error){
		"D2C": func(s *System) (*System, error) { return s.D2C(D2COptions{Method: C2DMethodTustin}) },
		"D2D": func(s *System) (*System, error) { return s.D2D(0.05, D2DOptions{}) },
	} {
		want, err := f(dtwin)
		if err != nil {
			t.Fatal(err)
		}
		got, err := f(ddesc)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		assertConvertedTwin(t, name, got, want)
	}
	if _, _, err := ddesc.D2CMap(D2COptions{Method: C2DMethodTustin}); !errors.Is(err, ErrDescriptorInitialState) {
		t.Fatalf("D2CMap err = %v, want ErrDescriptorInitialState", err)
	}
}

func TestConversionsOfImproperDescriptorRejected(t *testing.T) {
	cont, _ := index2Descriptor(t, 0, true)
	disc, _ := index2Descriptor(t, 0.1, true)
	calls := map[string]func() (*System, error){
		"DiscretizeZOH": func() (*System, error) { return cont.C2D(0.1, C2DOptions{}) },
		"DiscretizeFOH": func() (*System, error) { return cont.C2D(0.1, C2DOptions{Method: C2DMethodFOH}) },
		"Discretize":    func() (*System, error) { return cont.C2D(0.1, C2DOptions{Method: C2DMethodTustin}) },
		"D2C":           func() (*System, error) { return disc.D2C(D2COptions{Method: C2DMethodZOH}) },
		"D2D":           func() (*System, error) { return disc.D2D(0.05, D2DOptions{}) },
		"C2DMap": func() (*System, error) {
			sys, _, err := cont.C2DMap(0.1, C2DOptions{})
			return sys, err
		},
	}
	for name, call := range calls {
		if _, err := call(); !errors.Is(err, ErrImproperModel) || !errors.Is(err, ErrDescriptorUnsupported) {
			t.Fatalf("%s: err = %v, want ErrImproperModel and ErrDescriptorUnsupported", name, err)
		}
	}
	proper, twin := index2Descriptor(t, 0, false)
	want, err := twin.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := proper.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertConvertedTwin(t, "index2 ZOH", got, want)
}
