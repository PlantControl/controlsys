package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// badlyScaledDescriptor returns the descriptor (A, B, C, 0, diag(e)) and its
// explicit oracle (diag(1/e)·A, diag(1/e)·B, C, 0); E⁻¹ is exactly diagonal,
// so the oracle only rounds once per entry.
func badlyScaledDescriptor(t *testing.T, a, b, c, e []float64, dt float64) (desc, oracle *System) {
	t.Helper()
	n, m := len(e), len(b)/len(e)
	p := len(c) / n
	ea := make([]float64, len(a))
	eb := make([]float64, len(b))
	for i := range n {
		for j := range n {
			ea[i*n+j] = a[i*n+j] / e[i]
		}
		for j := range m {
			eb[i*m+j] = b[i*m+j] / e[i]
		}
	}
	E := mat.NewDense(n, n, nil)
	for i, v := range e {
		E.Set(i, i, v)
	}
	desc, err := NewDescriptor(mat.NewDense(n, n, append([]float64(nil), a...)), mat.NewDense(n, m, append([]float64(nil), b...)),
		mat.NewDense(p, n, append([]float64(nil), c...)), mat.NewDense(p, m, nil), E, dt)
	if err != nil {
		t.Fatal(err)
	}
	oracle, err = New(mat.NewDense(n, n, ea), mat.NewDense(n, m, eb), mat.NewDense(p, n, append([]float64(nil), c...)),
		mat.NewDense(p, m, nil), dt)
	if err != nil {
		t.Fatal(err)
	}
	return desc, oracle
}

// 4NAR5D: E = diag(1e-8, 1) or diag(1, 1e-9) has LU cond(E) above 1/√eps
// but is exactly fixed by power-of-two equilibration.
func TestDescriptorNormsBadlyScaledEAccepted(t *testing.T) {
	cases := []struct {
		name    string
		a, b, c []float64
		e       []float64
		dt      float64
	}{
		{"continuous diag(1e-8,1)", []float64{-1, 1, 0.5, -2}, []float64{1, 1}, []float64{1, 0.5}, []float64{1e-8, 1}, 0},
		{"continuous diag(1,1e-9)", []float64{-1, 1, 0.5, -2}, []float64{1, 1}, []float64{1, 0.5}, []float64{1, 1e-9}, 0},
		{"discrete diag(1e-8,1)", []float64{0.5e-8, 0.2e-8, -0.1, 0.3}, []float64{1e-8, 1}, []float64{1, 0.5}, []float64{1e-8, 1}, 0.1},
		{"discrete diag(1,1e-9)", []float64{0.5, 0.2, -0.1e-9, 0.3e-9}, []float64{1, 1e-9}, []float64{1, 0.5}, []float64{1, 1e-9}, 0.1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			desc, oracle := badlyScaledDescriptor(t, tc.a, tc.b, tc.c, tc.e, tc.dt)
			var lu mat.LU
			lu.Factorize(desc.E)
			if c := lu.Cond(); c <= normDescriptorCondLimit {
				t.Fatalf("LU cond(E) = %.3g, want > %.3g", c, normDescriptorCondLimit)
			}
			wantH2, err := H2Norm(oracle)
			if err != nil {
				t.Fatal(err)
			}
			wantHinf, wantW, err := HinfNorm(oracle)
			if err != nil {
				t.Fatal(err)
			}
			if h2, err := H2Norm(desc); err != nil || !relClose(h2, wantH2, 1e-9) {
				t.Errorf("H2Norm = %v, %v; want %v", h2, err, wantH2)
			}
			if hinf, w, err := HinfNorm(desc); err != nil || !relClose(hinf, wantHinf, 1e-9) ||
				math.Abs(w-wantW) > 1e-6*(1+wantW) {
				t.Errorf("HinfNorm = %v at %v, %v; want %v at %v", hinf, w, err, wantHinf, wantW)
			}
			if n2, err := Norm(desc, 2); err != nil || !relClose(n2, wantH2, 1e-9) {
				t.Errorf("Norm(2) = %v, %v; want %v", n2, err, wantH2)
			}
			if ninf, err := Norm(desc, math.Inf(1)); err != nil || !relClose(ninf, wantHinf, 1e-9) {
				t.Errorf("Norm(Inf) = %v, %v; want %v", ninf, err, wantHinf)
			}
			if !mat.Equal(desc.E, mat.NewDiagDense(2, tc.e)) {
				t.Errorf("E modified: %v", mat.Formatted(desc.E))
			}
		})
	}
}

// 4NAR5D: the equilibration scales the internal-delay channels B2 and C2
// with B and C.
func TestDescriptorNormsBadlyScaledEInternalDelay(t *testing.T) {
	e := []float64{1e-8, 1}
	a := []float64{0.5e-8, 0.2e-8, -0.1, 0.3}
	desc, oracle := badlyScaledDescriptor(t, a, []float64{1e-8, 1}, []float64{1, 0.5}, e, 0.1)
	b2 := []float64{0.3e-8, -0.2}
	if err := desc.SetInternalDelay([]float64{2}, mat.NewDense(2, 1, b2), mat.NewDense(1, 2, []float64{0.1, 0.4}),
		mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{0.1})); err != nil {
		t.Fatal(err)
	}
	if err := oracle.SetInternalDelay([]float64{2}, mat.NewDense(2, 1, []float64{b2[0] / e[0], b2[1] / e[1]}),
		mat.NewDense(1, 2, []float64{0.1, 0.4}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil),
		mat.NewDense(1, 1, []float64{0.1})); err != nil {
		t.Fatal(err)
	}
	wantH2, err := H2Norm(oracle)
	if err != nil {
		t.Fatal(err)
	}
	wantHinf, _, err := HinfNorm(oracle)
	if err != nil {
		t.Fatal(err)
	}
	if h2, err := H2Norm(desc); err != nil || !relClose(h2, wantH2, 1e-9) {
		t.Errorf("H2Norm = %v, %v; want %v", h2, err, wantH2)
	}
	if hinf, _, err := HinfNorm(desc); err != nil || !relClose(hinf, wantHinf, 1e-9) {
		t.Errorf("HinfNorm = %v, %v; want %v", hinf, err, wantHinf)
	}
}

// 4NAR5D: E ill-conditioned after equilibration, or singular by MATLAB
// rank, keeps ErrDescriptorUnsupported wrapping ErrDescriptorSingular.
func TestDescriptorNormsIllConditionedERejected(t *testing.T) {
	cases := []struct {
		name string
		e    []float64
	}{
		{"near-rank-one", []float64{1, 1, 1, 1 + 1e-10}},
		{"singular diag(1,0)", []float64{1, 0, 0, 0}},
		{"rank-deficient diag(1,1e-17)", []float64{1, 0, 0, 1e-17}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 1, 0.5, -2}), mat.NewDense(2, 1, []float64{1, 1}),
				mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, tc.e), 0)
			if err != nil {
				t.Fatal(err)
			}
			check := func(name string, err error) {
				if !errors.Is(err, ErrDescriptorUnsupported) || !errors.Is(err, ErrDescriptorSingular) {
					t.Errorf("%s err = %v; want ErrDescriptorUnsupported wrapping ErrDescriptorSingular", name, err)
				}
			}
			_, err = H2Norm(sys)
			check("H2Norm", err)
			_, _, err = HinfNorm(sys)
			check("HinfNorm", err)
			_, err = Norm(sys, math.Inf(1))
			check("Norm(Inf)", err)
		})
	}
}
