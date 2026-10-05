package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestModalTruncateRetainsDominantModesAndMetadata(t *testing.T) {
	sys := modalTestSystem(t)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	sys.StateName = []string{"x1", "x2", "x3", "x4"}

	result, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 2})
	if err != nil {
		t.Fatalf("ModalTruncate: %v", err)
	}
	if result.Method != "real-schur-modal-truncate" || result.Order != 2 {
		t.Fatalf("metadata = %#v", result)
	}
	if n, _, _ := result.Sys.Dims(); n != 2 {
		t.Fatalf("reduced order = %d, want 2", n)
	}
	if !sameStrings(result.Sys.InputName, []string{"u"}) || !sameStrings(result.Sys.OutputName, []string{"y"}) {
		t.Fatalf("names = %v/%v", result.Sys.InputName, result.Sys.OutputName)
	}
	if !sameComplexApprox(result.KeptPoles, []complex128{-0.1, -1}, 1e-12) {
		t.Fatalf("kept poles = %v, want [-0.1 -1]", result.KeptPoles)
	}
	var identity mat.Dense
	identity.Mul(result.Projection, result.Basis)
	if !mat.EqualApprox(&identity, mat.NewDiagDense(2, []float64{1, 1}), 1e-11) {
		t.Fatalf("projection*basis =\n%v", mat.Formatted(&identity))
	}
}

func TestModalTruncateIsInvariantUnderSimilarityTransform(t *testing.T) {
	base := modalTestSystem(t)
	transform := mat.NewDense(4, 4, []float64{
		1, 0.2, -0.1, 0.3,
		0.1, 1.2, 0.4, -0.2,
		-0.2, 0.1, 0.9, 0.2,
		0.3, -0.1, 0.2, 1.1,
	})
	equivalent, err := SS2SS(base, transform)
	if err != nil {
		t.Fatalf("SS2SS: %v", err)
	}

	left, err := ModalTruncate(base, &ModalTruncateOptions{Order: 2})
	if err != nil {
		t.Fatalf("ModalTruncate base: %v", err)
	}
	right, err := ModalTruncate(equivalent, &ModalTruncateOptions{Order: 2})
	if err != nil {
		t.Fatalf("ModalTruncate equivalent: %v", err)
	}
	if !sameComplexApprox(left.KeptPoles, right.KeptPoles, 1e-10) {
		t.Fatalf("kept poles differ: %v vs %v", left.KeptPoles, right.KeptPoles)
	}
	omega := []float64{0, 0.1, 1, 10}
	leftResponse, err := left.Sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	rightResponse, err := right.Sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		if delta := cmplx.Abs(leftResponse.At(k, 0, 0) - rightResponse.At(k, 0, 0)); delta > 1e-9 {
			t.Fatalf("response difference at omega=%g is %g", omega[k], delta)
		}
	}
}

func TestModalTruncatePreservesComplexPairs(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-0.1, -2, 0.3,
			2, -0.1, -0.2,
			0, 0, -5,
		}),
		mat.NewDense(3, 1, []float64{1, 2, 3}),
		mat.NewDense(1, 3, []float64{2, -1, 0.5}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 1}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("split-pair order err = %v, want ErrInvalidOrder", err)
	}
	result, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 2})
	if err != nil {
		t.Fatalf("pair-preserving reduction: %v", err)
	}
	if len(result.KeptPoles) != 2 || math.Abs(imag(result.KeptPoles[0])) < 1.9 || cmplx.Conj(result.KeptPoles[0]) != result.KeptPoles[1] {
		t.Fatalf("kept poles = %v, want conjugate pair", result.KeptPoles)
	}
}

func TestModalTruncatePreservesUnstableModesAndAutoSelects(t *testing.T) {
	sys, err := New(
		mat.DenseCopyOf(mat.NewDiagDense(4, []float64{0.2, 0.1, -1, -4})),
		mat.NewDense(4, 1, []float64{1, 1, 1, 1}),
		mat.NewDense(1, 4, []float64{1, 1, 1, 1}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 1}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("unstable-discard order err = %v, want ErrInvalidOrder", err)
	}
	result, err := ModalTruncate(sys, &ModalTruncateOptions{MaxRealPart: -0.5})
	if err != nil {
		t.Fatalf("ModalTruncate auto: %v", err)
	}
	if result.Order != 2 || !sameComplexApprox(result.KeptPoles, []complex128{0.2, 0.1}, 1e-12) {
		t.Fatalf("auto-selected result = order %d poles %v", result.Order, result.KeptPoles)
	}
	if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 5}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("invalid order err = %v, want ErrInvalidOrder", err)
	}
	if _, err := ModalTruncate(modalTestSystem(t), &ModalTruncateOptions{MaxRealPart: 5}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("empty threshold selection err = %v, want ErrInvalidOrder", err)
	}
}

func modalTestSystem(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.DenseCopyOf(mat.NewDiagDense(4, []float64{-0.1, -1, -5, -10})),
		mat.NewDense(4, 1, []float64{1, 2, 3, 4}),
		mat.NewDense(1, 4, []float64{4, -1, 2, 0.5}),
		mat.NewDense(1, 1, []float64{0.2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sameComplexApprox(a, b []complex128, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if cmplx.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	return true
}

func TestModalTruncateKeptPolesBadlyScaledPair(t *testing.T) {
	tests := []struct {
		name              string
		re, im, other, dt float64
		tol               float64
	}{
		{"continuous tiny imag", -1e8, 1, -2e8, 0, 1e-6},
		{"continuous partial cancellation", -1e8, 3.3, -2e8, 0, 1e-6},
		{"discrete near repeated", 0.5, 1e-9, 0.1, 0.1, 1e-14},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := badlyScaledPairSystem(t, tc.re, tc.im, tc.other, tc.dt)
			result, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 2})
			if err != nil {
				t.Fatal(err)
			}
			want := []complex128{complex(tc.re, tc.im), complex(tc.re, -tc.im)}
			if !sameComplexApprox(result.KeptPoles, want, tc.tol) {
				t.Fatalf("kept poles = %v, want %v", result.KeptPoles, want)
			}
			assertPolesMatch(t, "reduced", result.Sys, want, tc.tol)
		})
	}
}
