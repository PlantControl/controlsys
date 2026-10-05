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
