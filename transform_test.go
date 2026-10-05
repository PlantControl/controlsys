package controlsys

import (
	"fmt"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSS2SSTransformsInternalDelayChannels(t *testing.T) {
	T := mat.NewDense(3, 3, []float64{2, 0.3, -0.1, 0.4, 1.5, 0.2, -0.3, 0.1, 1.2})
	for _, dt := range []float64{0, 0.1} {
		orig := fieldLFT(t, dt)
		got, err := SS2SS(orig, T)
		if err != nil {
			t.Fatal(err)
		}
		var want mat.Dense
		want.Mul(T, orig.LFT.B2)
		if !matClose(got.LFT.B2, &want, 1e-12) {
			t.Errorf("dt=%g: B2 = %v, want T·B2 = %v", dt, mat.Formatted(got.LFT.B2), mat.Formatted(&want))
		}
		assertFieldResponse(t, fmt.Sprintf("SS2SS/dt=%g", dt), got, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
	}
}

func TestXpermPermutesStatesNamesAndInternalDelay(t *testing.T) {
	perm := []int{2, 0, 1}
	for _, dt := range []float64{0, 0.1} {
		orig := fieldLFT(t, dt)
		orig.StateName = []string{"a", "b", "c"}
		got, err := Xperm(orig, perm)
		if err != nil {
			t.Fatal(err)
		}
		for i, j := range perm {
			for k, l := range perm {
				if got.A.At(i, k) != orig.A.At(j, l) {
					t.Fatalf("dt=%g: A[%d][%d] = %v, want old A[%d][%d] = %v", dt, i, k, got.A.At(i, k), j, l, orig.A.At(j, l))
				}
			}
			if got.LFT.B2.At(i, 0) != orig.LFT.B2.At(j, 0) || got.LFT.C2.At(0, i) != orig.LFT.C2.At(0, j) {
				t.Errorf("dt=%g: LFT channel for new state %d not taken from old state %d", dt, i, j)
			}
		}
		if want := []string{"c", "a", "b"}; !reflect.DeepEqual(got.StateName, want) {
			t.Errorf("dt=%g: StateName = %v, want %v", dt, got.StateName, want)
		}
		if !reflect.DeepEqual(orig.StateName, []string{"a", "b", "c"}) {
			t.Errorf("dt=%g: source StateName mutated to %v", dt, orig.StateName)
		}
		assertFieldResponse(t, fmt.Sprintf("Xperm/dt=%g", dt), got, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
	}
}

func transformDescriptorLFT(t *testing.T, dt float64) *System {
	t.Helper()
	sys := fieldLFT(t, dt)
	sys.E = mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, 0.2, 0, 0.4, 1.5})
	return sys
}

// TestSS2SSDescriptorMatchesMATLAB pins MATLAB ss2ss (R2021b+) descriptor
// semantics: (E·T⁻¹, A·T⁻¹, B, C·T⁻¹, D), internal-delay B2 unchanged.
func TestSS2SSDescriptorMatchesMATLAB(t *testing.T) {
	T := mat.NewDense(3, 3, []float64{2, 0.3, -0.1, 0.4, 1.5, 0.2, -0.3, 0.1, 1.2})
	var Tinv mat.Dense
	if err := Tinv.Inverse(T); err != nil {
		t.Fatal(err)
	}
	for _, dt := range []float64{0, 0.1} {
		for _, mk := range []func(*testing.T, float64) *System{fieldDescriptor, transformDescriptorLFT} {
			orig := mk(t, dt)
			got, err := SS2SS(orig, T)
			if err != nil {
				t.Fatal(err)
			}
			label := fmt.Sprintf("dt=%g/lft=%v", dt, orig.LFT != nil)
			var want mat.Dense
			want.Mul(orig.E, &Tinv)
			if !matClose(got.E, &want, 1e-12) {
				t.Errorf("%s: E = %v, want E·T⁻¹", label, mat.Formatted(got.E))
			}
			want.Mul(orig.A, &Tinv)
			if !matClose(got.A, &want, 1e-12) {
				t.Errorf("%s: A = %v, want A·T⁻¹", label, mat.Formatted(got.A))
			}
			var wantC mat.Dense
			wantC.Mul(orig.C, &Tinv)
			if !matClose(got.C, &wantC, 1e-12) {
				t.Errorf("%s: C = %v, want C·T⁻¹", label, mat.Formatted(got.C))
			}
			if !mat.Equal(got.B, orig.B) {
				t.Errorf("%s: B changed", label)
			}
			if orig.LFT != nil && !mat.Equal(got.LFT.B2, orig.LFT.B2) {
				t.Errorf("%s: B2 changed", label)
			}
			assertFieldResponse(t, "SS2SS/"+label, got, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
		}
	}
}

func TestXpermDescriptorPermutesE(t *testing.T) {
	perm := []int{2, 0, 1}
	for _, dt := range []float64{0, 0.1} {
		for _, mk := range []func(*testing.T, float64) *System{fieldDescriptor, transformDescriptorLFT} {
			orig := mk(t, dt)
			got, err := Xperm(orig, perm)
			if err != nil {
				t.Fatal(err)
			}
			label := fmt.Sprintf("dt=%g/lft=%v", dt, orig.LFT != nil)
			for i, j := range perm {
				for k, l := range perm {
					if got.E.At(i, k) != orig.E.At(j, l) || got.A.At(i, k) != orig.A.At(j, l) {
						t.Fatalf("%s: E/A[%d][%d] not old [%d][%d]", label, i, k, j, l)
					}
				}
			}
			assertFieldResponse(t, "Xperm/"+label, got, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
		}
	}
}
