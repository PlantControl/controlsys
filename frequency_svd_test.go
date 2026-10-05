package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestComplexSVDWorkspaceMaximumMatchesGonum(t *testing.T) {
	tests := []struct {
		name string
		p, m int
		data []complex128
	}{
		{
			name: "Rectangular",
			p:    3, m: 2,
			data: []complex128{1 + 2i, -3 + 0.5i, 0.25 - 4i, 2 + 1i, -1.5 + 0.75i, 0.2 - 0.1i},
		},
		{
			name: "RankDeficient",
			p:    3, m: 3,
			data: []complex128{
				1 + 2i, 2 - 1i, -0.5 + 0.25i,
				2 + 4i, 4 - 2i, -1 + 0.5i,
				-1 - 2i, -2 + 1i, 0.5 - 0.25i,
			},
		},
		{
			name: "NonSymmetric2x2",
			p:    2, m: 2,
			data: []complex128{1e-150 + 2e-150i, -3e-150 + 0.5e-150i, 4e-150 - 2e-150i, 0.25e-150 + 1e-150i},
		},
		{
			name: "LargeScale",
			p:    3, m: 2,
			data: []complex128{1e150 + 2e150i, -3e150, 0.5e150 - 0.25e150i, 2e150 + 1e150i, -1e150, 0.75e150i},
		},
		{
			name: "Vector",
			p:    1, m: 4,
			data: []complex128{1e200, 2e200i, -3e200 + 0.5e200i, 0.25e200 - 0.75e200i},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var workspace *complexSVDWorkspace
			if test.p > 1 && test.m > 1 && (test.p != 2 || test.m != 2) {
				workspace = newComplexSVDWorkspace(test.p, test.m)
			}
			got, ok := workspace.maximumFromFlat(test.data, 0, test.p, test.m)
			if !ok {
				t.Fatal("maximum singular-value decomposition failed")
			}
			want := gonumComplexSingularValues(t, test.data, test.p, test.m)[0]
			if !sameRelative(got, want, 2e-12) {
				t.Fatalf("maximum singular value = %.17g, want %.17g", got, want)
			}
		})
	}
}

func TestComplexSVDWorkspaceScaledSingularValues(t *testing.T) {
	p, m := 4, 3
	data := []complex128{
		1e150 + 2e150i, -3e150, 0.5e150 - 0.25e150i,
		2e150 + 1e150i, -1e150, 0.75e150i,
		-0.5e150, 0.2e150 + 0.1e150i, 4e150 - 2e150i,
		1.5e150 - 0.5e150i, -2e150 + 3e150i, 0.25e150,
	}
	want := gonumComplexSingularValues(t, data, p, m)
	got := make([]float64, min(p, m))
	workspace := newComplexSVDWorkspace(p, m)
	workspace.singularValuesFromFlat(got, data, 0, p, m)
	for i := range got {
		if !sameRelative(got[i], want[i], 3e-12) {
			t.Fatalf("singular value %d = %.17g, want %.17g", i, got[i], want[i])
		}
	}
}

func gonumComplexSingularValues(t *testing.T, data []complex128, p, m int) []float64 {
	t.Helper()
	realForm := mat.NewDense(2*p, 2*m, nil)
	for i := range p {
		for j := range m {
			value := data[i*m+j]
			realForm.Set(i, j, real(value))
			realForm.Set(i, j+m, -imag(value))
			realForm.Set(i+p, j, imag(value))
			realForm.Set(i+p, j+m, real(value))
		}
	}
	var decomposition mat.SVD
	if ok := decomposition.Factorize(realForm, mat.SVDNone); !ok {
		t.Fatal("Gonum SVD factorization failed")
	}
	values := decomposition.Values(nil)
	result := make([]float64, min(p, m))
	for i := range result {
		result[i] = values[2*i]
	}
	return result
}

func sameRelative(got, want, tolerance float64) bool {
	if got == want {
		return true
	}
	return math.Abs(got-want) <= tolerance*math.Max(math.Abs(got), math.Abs(want))
}

func TestMaximumComplex2x2SingularValueNearlyEqual(t *testing.T) {
	for _, e := range []float64{1e-8, 3e-9, 1e-12} {
		want := (math.Sqrt(4+e*e) + e) / 2
		for _, data := range [][]complex128{{1, complex(e, 0), 0, 1}, {1i, complex(0, e), 0, -1}} {
			if got := maximumComplex2x2SingularValue(data); !sameRelative(got, want, 2*eps()) {
				t.Fatalf("e=%g data=%v: sigma max = %.17g, want %.17g", e, data, got, want)
			}
		}
	}
}
