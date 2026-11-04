package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Range tests for HSV, Gram and Balreal when B·Bᵀ or Cᵀ·C leaves float64
// range although the requested quantity does not (YP7DWL).

// kronLyap solves M·X·Mᵀ - X + Q = 0 (discrete) or M·X + X·Mᵀ + Q = 0
// (continuous) through the n²×n² Kronecker system, independent of Lyap.
func kronLyap(t *testing.T, M, Q *mat.Dense, discrete bool) *mat.Dense {
	t.Helper()
	n, _ := M.Dims()
	K := mat.NewDense(n*n, n*n, nil)
	for i := range n {
		for j := range n {
			for k := range n {
				for l := range n {
					var v float64
					if discrete {
						v = M.At(i, k) * M.At(j, l)
						if i == k && j == l {
							v--
						}
					} else {
						if j == l {
							v += M.At(i, k)
						}
						if i == k {
							v += M.At(j, l)
						}
					}
					K.Set(i*n+j, k*n+l, v)
				}
			}
		}
	}
	rhs := mat.NewVecDense(n*n, nil)
	for i := range n {
		for j := range n {
			rhs.SetVec(i*n+j, -Q.At(i, j))
		}
	}
	var x mat.VecDense
	if err := x.SolveVec(K, rhs); err != nil {
		t.Fatal(err)
	}
	return mat.NewDense(n, n, x.RawVector().Data)
}

// symEig2 returns the eigenvalues of a symmetric 2×2 matrix, descending.
func symEig2(W *mat.Dense) [2]float64 {
	a, b, d := W.At(0, 0), W.At(0, 1), W.At(1, 1)
	m, r := (a+d)/2, math.Hypot((a-d)/2, b)
	return [2]float64{m + r, m - r}
}

func rotationScaled(rho, theta float64) *mat.Dense {
	c, s := math.Cos(theta), math.Sin(theta)
	return mat.NewDense(2, 2, []float64{rho * c, rho * s, -rho * s, rho * c})
}

func relClose(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol*math.Abs(want)
}

// rangeCase is a 2-state model with A+Aᵀ = -2a·I (continuous) or
// A·Aᵀ = ρ²·I (discrete), so for B = β·I the controllability gramian is
// β²/(2a)·I or β²/(1-ρ²)·I.
type rangeCase struct {
	name string
	A    *mat.Dense
	dt   float64
	wc   float64
}

func rangeCases() []rangeCase {
	return []rangeCase{
		{"continuous", mat.NewDense(2, 2, []float64{-0.1, 5, -5, -0.1}), 0, 1 / 0.2},
		{"discrete", rotationScaled(0.5, 0.7), 0.1, 1 / 0.75},
	}
}

func (rc rangeCase) system(t *testing.T, beta float64, C *mat.Dense) *System {
	t.Helper()
	sys, err := New(rc.A, mat.NewDense(2, 2, []float64{beta, 0, 0, beta}), C, mat.NewDense(1, 2, nil), rc.dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func (rc rangeCase) observabilityGramian(t *testing.T, C *mat.Dense) *mat.Dense {
	var Q mat.Dense
	Q.Mul(C.T(), C)
	return kronLyap(t, mat.DenseCopyOf(rc.A.T()), &Q, rc.dt != 0)
}

func TestHSV_LargeAndTinyInputScale(t *testing.T) {
	C := mat.NewDense(1, 2, []float64{1, 2})
	for _, rc := range rangeCases() {
		lam := symEig2(rc.observabilityGramian(t, C))
		for _, beta := range []float64{1e155, 1e-160} {
			got, err := HSV(rc.system(t, beta, C))
			if err != nil {
				t.Fatalf("%s β=%g: %v", rc.name, beta, err)
			}
			for i := range 2 {
				want := beta * math.Sqrt(rc.wc*lam[i])
				if !relClose(got[i], want, 1e-12) {
					t.Errorf("%s β=%g: σ%d = %g, want %g", rc.name, beta, i, got[i], want)
				}
			}
		}
	}
}

func TestHSV_OverflowReturnsErrOverflow(t *testing.T) {
	C := mat.NewDense(1, 2, []float64{1e200, 2e200})
	for _, rc := range rangeCases() {
		_, err := HSV(rc.system(t, 1e200, C))
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("%s: err = %v, want ErrOverflow", rc.name, err)
		}
	}
}

func TestGram_LargeInputScale(t *testing.T) {
	const beta = 1e155
	C := mat.NewDense(1, 2, []float64{1, 2})
	for _, rc := range rangeCases() {
		sys := rc.system(t, beta, C)

		_, err := Gram(sys, GramControllability)
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("%s: Wc err = %v, want ErrOverflow", rc.name, err)
		}

		R, err := Gram(sys, GramControllabilityFactor)
		if err != nil {
			t.Fatalf("%s: Rc: %v", rc.name, err)
		}
		r := beta * math.Sqrt(rc.wc)
		if e := relErr(R, mat.NewDense(2, 2, []float64{r, 0, 0, r})); e > 1e-12 {
			t.Errorf("%s: Rc rel err %g: %v", rc.name, e, mat.Formatted(R))
		}

		Wo, err := Gram(sys, GramObservability)
		if err != nil {
			t.Fatalf("%s: Wo: %v", rc.name, err)
		}
		if e := relErr(Wo, rc.observabilityGramian(t, C)); e > 1e-12 {
			t.Errorf("%s: Wo rel err %g", rc.name, e)
		}
	}
}

func TestGram_LargeOutputScale(t *testing.T) {
	const gamma = 1e155
	for _, rc := range rangeCases() {
		C := mat.NewDense(2, 2, []float64{gamma, 0, 0, gamma})
		sys, err := New(rc.A, mat.NewDense(2, 1, []float64{1, 2}), C, mat.NewDense(2, 1, nil), rc.dt)
		if err != nil {
			t.Fatal(err)
		}
		_, err = Gram(sys, GramObservability)
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("%s: Wo err = %v, want ErrOverflow", rc.name, err)
		}
		R, err := Gram(sys, GramObservabilityFactor)
		if err != nil {
			t.Fatalf("%s: Ro: %v", rc.name, err)
		}
		r := gamma * math.Sqrt(rc.wc)
		if e := relErr(R, mat.NewDense(2, 2, []float64{r, 0, 0, r})); e > 1e-12 {
			t.Errorf("%s: Ro rel err %g: %v", rc.name, e, mat.Formatted(R))
		}
	}
}

// TestGram_RepresentableDespiteOverflowingBBt: B·Bᵀ = 1e310·I overflows but
// Wc = 1e310/200·I = 5e307·I does not.
func TestGram_RepresentableDespiteOverflowingBBt(t *testing.T) {
	const beta = 1e155
	rc := rangeCase{A: mat.NewDense(2, 2, []float64{-100, 5, -5, -100})}
	sys := rc.system(t, beta, mat.NewDense(1, 2, []float64{1, 2}))
	W, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	w := beta * (beta / 200)
	if e := relErr(W, mat.NewDense(2, 2, []float64{w, 0, 0, w})); e > 1e-12 {
		t.Errorf("Wc rel err %g: %v", e, mat.Formatted(W))
	}
}

func TestBalreal_LargeInputScale(t *testing.T) {
	const beta = 1e155
	C := mat.NewDense(1, 2, []float64{1, 2})
	for _, rc := range rangeCases() {
		sys := rc.system(t, beta, C)
		br, err := Balreal(sys)
		if err != nil {
			t.Fatalf("%s: %v", rc.name, err)
		}
		lam := symEig2(rc.observabilityGramian(t, C))
		for i := range 2 {
			want := beta * math.Sqrt(rc.wc*lam[i])
			if !relClose(br.HSV[i], want, 1e-12) {
				t.Errorf("%s: σ%d = %g, want %g", rc.name, i, br.HSV[i], want)
			}
		}
		var prod mat.Dense
		prod.Mul(br.TL, br.TR)
		if e := relErr(&prod, eye(2)); e > 1e-12 {
			t.Errorf("%s: TL·TR rel err %g", rc.name, e)
		}
		sigma := mat.NewDense(2, 2, []float64{br.HSV[0], 0, 0, br.HSV[1]})
		for _, typ := range []GramType{GramControllability, GramObservability} {
			W, err := Gram(br.Sys, typ)
			if err != nil {
				t.Fatalf("%s: Gram %d: %v", rc.name, typ, err)
			}
			if e := relErr(W, sigma); e > 1e-10 {
				t.Errorf("%s: balanced gramian %d rel err %g", rc.name, typ, e)
			}
		}
	}
}
