package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
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
			got, err := workspace.maximumFromFlat(test.data, 0, test.p, test.m)
			if err != nil {
				t.Fatal(err)
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
	if err := workspace.singularValuesFromFlat(got, data, 0, p, m); err != nil {
		t.Fatal(err)
	}
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

func TestSigmaAtPoleNoNaN(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, 0}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.Sigma([]float64{1}, 0)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("MIMO Sigma at pole: res=%v err=%v, want ErrInvalidArgument", res, err)
	}

	siso := makeSISO(0, 1, 1, 0)
	r, err := siso.Sigma([]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(r.At(0, 0), 1) {
		t.Fatalf("SISO sigma at pole = %g, want +Inf", r.At(0, 0))
	}
}

func TestFRDPeakGainAtPoleIsInf(t *testing.T) {
	inf := cmplx.Inf()
	f, err := NewFRD([][][]complex128{
		{{1, 0}, {0, 1}},
		{{inf, 0}, {0, 1}},
	}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	pk, err := f.PeakGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(pk.Gain, 1) || pk.Index != 1 {
		t.Fatalf("peak = %+v, want +Inf at index 1", pk)
	}
	if _, err := f.Sigma(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("FRD Sigma at pole err = %v, want ErrInvalidArgument", err)
	}
}

func testUnitary(n int, seed float64) [][]complex128 {
	q := make([][]complex128, n)
	for i := range q {
		q[i] = make([]complex128, n)
		for j := range n {
			q[i][j] = complex(math.Sin(seed*float64(3*i+j+1)), math.Cos(seed*float64(i+5*j+2)))
		}
	}
	for range 2 {
		for j := range n {
			for k := range j {
				var dot complex128
				for i := range n {
					dot += cmplx.Conj(q[i][k]) * q[i][j]
				}
				for i := range n {
					q[i][j] -= dot * q[i][k]
				}
			}
			var norm float64
			for i := range n {
				norm = math.Hypot(norm, cmplx.Abs(q[i][j]))
			}
			for i := range n {
				q[i][j] /= complex(norm, 0)
			}
		}
	}
	return q
}

func TestComplexSVDWorkspaceSmallSingularValuesAbsoluteAccuracy(t *testing.T) {
	tests := []struct {
		name  string
		p, m  int
		sigma []float64
	}{
		{"Square2x2", 2, 2, []float64{224.9, 1.2382e-6}},
		{"Tall4x3", 4, 3, []float64{3e3, 2.5e-4, 1e-9}},
		{"Wide2x5", 2, 5, []float64{1e5, 7e-7}},
		{"Square3x3Graded", 3, 3, []float64{1, 1e-6, 1e-11}},
		{"Tiny2x2", 2, 2, []float64{3e-200, 1e-209}},
		{"Wide3x5", 3, 5, []float64{40, 3e-5, 2e-10}},
		{"Square5x5", 5, 5, []float64{1e4, 2e3, 1e-3, 1e-7, 3e-11}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			u := testUnitary(test.p, 0.37)
			v := testUnitary(test.m, 1.13)
			data := make([]complex128, test.p*test.m)
			nested := make([][]complex128, test.p)
			for i := range test.p {
				nested[i] = data[i*test.m : (i+1)*test.m]
				for j := range test.m {
					var g complex128
					for k, s := range test.sigma {
						g += u[i][k] * complex(s, 0) * cmplx.Conj(v[j][k])
					}
					data[i*test.m+j] = g
				}
			}
			gonum := gonumComplexSingularValues(t, data, test.p, test.m)
			tol := 16 * float64(test.p+test.m) * eps() * test.sigma[0]
			ws := newComplexSVDWorkspace(test.p, test.m)
			for _, nestedForm := range []bool{false, true} {
				got := make([]float64, len(test.sigma))
				var err error
				if nestedForm {
					err = ws.singularValuesFromNested(got, nested, test.p, test.m)
				} else {
					err = ws.singularValuesFromFlat(got, data, 0, test.p, test.m)
				}
				if err != nil {
					t.Fatal(err)
				}
				for i, s := range test.sigma {
					if math.Abs(got[i]-s) > tol || math.Abs(got[i]-gonum[i]) > tol {
						t.Fatalf("nested=%v σ%d = %.17g, constructed %.17g, gonum %.17g (tol %.3g)", nestedForm, i, got[i], s, gonum[i], tol)
					}
				}
			}
		})
	}
}

func TestSigmaSubnormalResponse(t *testing.T) {
	const up = 0x1p1000
	for _, d := range [][]float64{{1e-310, 0, 0, 2e-310}, {3e-309, 1e-309, 0, 2e-309}} {
		sys, err := NewFromSlices(0, 2, 2, nil, nil, nil, d, 0)
		if err != nil {
			t.Fatal(err)
		}
		sv, err := sys.Sigma([]float64{0.5}, 0)
		if err != nil {
			t.Fatalf("D=%v: %v", d, err)
		}
		scaled := make([]complex128, len(d))
		for i, v := range d {
			scaled[i] = complex(v*up, 0)
		}
		want := gonumComplexSingularValues(t, scaled, 2, 2)
		for i, w := range want {
			if got := sv.At(0, i); !sameRelative(got, w/up, 1e-10) {
				t.Fatalf("D=%v: σ%d = %g, want %g", d, i, got, w/up)
			}
		}
	}
}

func TestComplexSVDWorkspaceSubnormalSingularValues(t *testing.T) {
	p, m := 3, 2
	data := []complex128{1e-310, 0, 0, 2e-310i, 0, 0}
	ws := newComplexSVDWorkspace(p, m)
	for _, wide := range []bool{false, true} {
		rows, cols := p, m
		in := data
		if wide {
			rows, cols = m, p
			in = []complex128{1e-310, 0, 0, 0, 2e-310i, 0}
		}
		got := make([]float64, 2)
		if err := ws.singularValuesFromFlat(got, in, 0, rows, cols); err != nil {
			t.Fatal(err)
		}
		for i, w := range []float64{2e-310, 1e-310} {
			if !sameRelative(got[i], w, 1e-10) {
				t.Fatalf("wide=%v σ%d = %g, want %g", wide, i, got[i], w)
			}
		}
	}
}
