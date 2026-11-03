package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// CLKIDR: a discrete model with no inputs (or no outputs) absorbs its output
// (or input) delays into a shift register and keeps the empty B (or C), as
// MATLAB absorbDelay is expected to (unverified against MATLAB).
func TestAbsorbDelayInputFreeOutputFree(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, -0.3})

	t.Run("input-free output delay", func(t *testing.T) {
		C := mat.NewDense(2, 2, []float64{1, 2, 3, -1})
		for _, scopes := range [][]AbsorbScope{nil, {AbsorbOutput}, {AbsorbInput, AbsorbOutput}} {
			sys, err := New(A, nil, C, nil, 0.1)
			if err != nil {
				t.Fatal(err)
			}
			if err := sys.SetOutputDelay([]float64{0, 1}); err != nil {
				t.Fatal(err)
			}
			got, err := sys.AbsorbDelay(scopes...)
			if err != nil {
				t.Fatalf("%v: %v", scopes, err)
			}
			if n, m, p := got.Dims(); n != 3 || m != 0 || p != 2 {
				t.Fatalf("%v: dims %d,%d,%d, want 3,0,2", scopes, n, m, p)
			}
			if !got.B.IsEmpty() || !got.D.IsEmpty() || got.OutputDelay != nil {
				t.Fatalf("%v: B, D empty and delay cleared expected", scopes)
			}
			wantA := mat.NewDense(3, 3, []float64{0.5, 0.2, 0, -0.1, -0.3, 0, 3, -1, 0})
			wantC := mat.NewDense(2, 3, []float64{1, 2, 0, 0, 0, 1})
			if !mat.Equal(got.A, wantA) || !mat.Equal(got.C, wantC) {
				t.Fatalf("%v: A=%v C=%v", scopes, mat.Formatted(got.A), mat.Formatted(got.C))
			}
		}

		sys, _ := New(A, nil, C, nil, 0.1)
		_ = sys.SetOutputDelay([]float64{0, 1})
		P, err := Covar(sys, &mat.Dense{})
		if err != nil {
			t.Fatal(err)
		}
		if r, c := P.Dims(); r != 2 || c != 2 || mat.Norm(P, 1) != 0 {
			t.Fatalf("Covar = %v, want 2×2 zeros", mat.Formatted(P))
		}
	})

	t.Run("input-free internal and output delay", func(t *testing.T) {
		C := mat.NewDense(2, 2, []float64{1, 2, 3, -1})
		sys, err := New(A, nil, C, nil, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		B2 := mat.NewDense(2, 1, []float64{1, -1})
		C2 := mat.NewDense(1, 2, []float64{0.5, 0.25})
		D12 := mat.NewDense(2, 1, []float64{1, 0})
		if err := sys.SetInternalDelay([]float64{1}, B2, C2, D12, nil, mat.NewDense(1, 1, []float64{0.2})); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{1, 0}); err != nil {
			t.Fatal(err)
		}
		for _, scopes := range [][]AbsorbScope{nil, {AbsorbOutput}} {
			got, err := sys.Copy().AbsorbDelay(scopes...)
			if err != nil {
				t.Fatalf("%v: %v", scopes, err)
			}
			n, m, p := got.Dims()
			if m != 0 || p != 2 || !got.B.IsEmpty() {
				t.Fatalf("%v: dims %d,%d,%d with B %v", scopes, n, m, p, got.B.IsEmpty())
			}
		}
		if _, err := Covar(sys, &mat.Dense{}); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("output-free input delay", func(t *testing.T) {
		B := mat.NewDense(2, 2, []float64{1, 0, 2, -1})
		for _, scopes := range [][]AbsorbScope{nil, {AbsorbInput}, {AbsorbInput, AbsorbOutput}} {
			sys, err := New(A, B, nil, nil, 0.1)
			if err != nil {
				t.Fatal(err)
			}
			if err := sys.SetInputDelay([]float64{0, 2}); err != nil {
				t.Fatal(err)
			}
			got, err := sys.AbsorbDelay(scopes...)
			if err != nil {
				t.Fatalf("%v: %v", scopes, err)
			}
			if n, m, p := got.Dims(); n != 4 || m != 2 || p != 0 {
				t.Fatalf("%v: dims %d,%d,%d, want 4,2,0", scopes, n, m, p)
			}
			if !got.C.IsEmpty() || !got.D.IsEmpty() || got.InputDelay != nil {
				t.Fatalf("%v: C, D empty and delay cleared expected", scopes)
			}
			wantA := mat.NewDense(4, 4, []float64{
				0.5, 0.2, 0, 0,
				-0.1, -0.3, 0, -1,
				0, 0, 0, 0,
				0, 0, 1, 0,
			})
			wantB := mat.NewDense(4, 2, []float64{1, 0, 2, 0, 0, 1, 0, 0})
			if !mat.Equal(got.A, wantA) || !mat.Equal(got.B, wantB) {
				t.Fatalf("%v: A=%v B=%v", scopes, mat.Formatted(got.A), mat.Formatted(got.B))
			}
		}

		sys, _ := New(A, B, nil, nil, 0.1)
		_ = sys.SetInputDelay([]float64{0, 2})
		withInternal := sys.Copy()
		B2 := mat.NewDense(2, 1, []float64{1, -1})
		C2 := mat.NewDense(1, 2, []float64{0.5, 0.25})
		D21 := mat.NewDense(1, 2, []float64{0, 1})
		if err := withInternal.SetInternalDelay([]float64{1}, B2, C2, nil, D21, mat.NewDense(1, 1, []float64{0.2})); err != nil {
			t.Fatal(err)
		}
		for _, scopes := range [][]AbsorbScope{nil, {AbsorbInput}} {
			got, err := withInternal.Copy().AbsorbDelay(scopes...)
			if err != nil {
				t.Fatalf("internal %v: %v", scopes, err)
			}
			if n, m, p := got.Dims(); n < 4 || m != 2 || p != 0 || !got.C.IsEmpty() {
				t.Fatalf("internal %v: dims %d,%d,%d", scopes, n, m, p)
			}
		}
		W := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
		if _, err := Covar(sys, W); !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("Covar err = %v, want ErrDimensionMismatch", err)
		}
	})
}
