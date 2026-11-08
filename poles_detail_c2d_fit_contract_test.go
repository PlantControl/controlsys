package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// resultContractClass is the outcome class of PolesDetail, C2DFit and
// least-squares C2D under the result contract (JBNVAG, ZOJBZX step 4).
type resultContractClass int

const (
	resultCertified resultContractClass = iota
	resultUnverified
	resultFailed
)

func (c resultContractClass) String() string {
	return [...]string{"certified", "unverified", "failed"}[c]
}

// checkResultContract fails unless (has, err) is of class want: certified
// (value, nil); unverified (value, ErrAccuracyUnverified); failed (nil,
// error wrapping sentinel and not ErrAccuracyUnverified).
func checkResultContract(t *testing.T, what string, want resultContractClass, has bool, err, sentinel error) {
	t.Helper()
	if errors.Is(err, errPencilFitUncertified) {
		t.Errorf("%s: unexported sentinel escapes: %v", what, err)
	}
	switch want {
	case resultCertified:
		if err != nil || !has {
			t.Errorf("%s: value %v, err %v; want a value with nil", what, has, err)
		}
	case resultUnverified:
		if !errors.Is(err, ErrAccuracyUnverified) || !has {
			t.Errorf("%s: value %v, err %v; want a value with ErrAccuracyUnverified", what, has, err)
		}
	case resultFailed:
		if !errors.Is(err, sentinel) || errors.Is(err, ErrAccuracyUnverified) || has {
			t.Errorf("%s: value %v, err %v; want nil with %v", what, has, err, sentinel)
		}
	}
}

// TestPolesDetailContractByClass: an explicit model and a robustly
// nonsingular E are certified; an exact DAE (E = diag(1, 0)) and E within
// MATLAB's rank tolerance of singular return the poles of Poles, bit for
// bit, with ErrAccuracyUnverified (no exemption for exactly singular E,
// ZOJBZX Q9); invalid data and an algebraic loop return nil poles.
func TestPolesDetailContractByClass(t *testing.T) {
	a := []float64{-1, 2, 0.5, -3}
	explicit, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	nanA := polesDetailDescriptor(t, 2, a, []float64{1, 0, 0, 0})
	nanA.A.Set(0, 1, math.NaN())
	loop, _ := internalDelayZerosFixture(t, 0, 2, mat.NewDense(1, 1, []float64{1}))
	for _, c := range []struct {
		name     string
		class    resultContractClass
		sys      *System
		sentinel error
	}{
		{"explicit", resultCertified, explicit, nil},
		{"nonsingular E", resultCertified, polesDetailDescriptor(t, 2, a, []float64{2, 1, -1, 3}), nil},
		{"near-singular E", resultCertified, polesDetailDescriptor(t, 2, a, []float64{1, 1, 1, 1 + math.Ldexp(1, -40)}), nil},
		{"exact DAE", resultUnverified, polesDetailDescriptor(t, 2, a, []float64{1, 0, 0, 0}), nil},
		{"E within rank tolerance", resultUnverified, polesDetailDescriptor(t, 2, a, []float64{1, 1, 1, 1 + math.Ldexp(1, -52)}), nil},
		{"zero E", resultUnverified, polesDetailDescriptor(t, 2, a, []float64{0, 0, 0, 0}), nil},
		{"NaN A", resultFailed, nanA, ErrInvalidArgument},
		{"algebraic loop", resultFailed, loop, ErrAlgebraicLoop},
	} {
		poles, err := c.sys.PolesDetail()
		checkResultContract(t, "PolesDetail "+c.name, c.class, poles != nil, err, c.sentinel)
		if c.class == resultFailed {
			continue
		}
		want, perr := c.sys.Poles()
		if perr != nil || !polesBitsEqual(poles, want) {
			t.Errorf("PolesDetail %s: poles %v, Poles %v, %v", c.name, poles, want, perr)
		}
	}
}

// TestLeastSquaresC2DContractByOperation checks C2DFit and C2D (least
// squares) in each class: an explicit model, not sampled through a pencil,
// is certified (ZOJBZX Q11); a stiff descriptor sampled through its pencil
// is unverified from both, with the same model bit for bit (Q10); a pencil
// fit that cannot retain its integrators, a MIMO model and a non-least-
// squares C2DFit return nil. C2DMap of the routed model keeps its
// ErrOptionUnsupported.
func TestLeastSquaresC2DContractByOperation(t *testing.T) {
	const dt = 0.1
	explicit, err := New(mat.NewDense(2, 2, []float64{0.5, 1, 0, -2}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	routed := nearSingularEContinuousK(t, 116)
	chain := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(0, 60608)),
		[]float64{0, 0.7, -0.4, 0.3, 0, 0, 0.9, -0.2, 0, 0, 0.2, 0.5, 0, 0, 0, -0.8},
		[]float64{1, 0.8, 1, 1e-13}, []float64{0.6, -1.1, 0.7, 0.5}, []float64{1.2, 0.4, 0.8, -0.9})
	for _, s := range []*System{routed, chain} {
		if leastSquaresPencilSlow(s) == nil {
			t.Fatal("fixture no longer exercises the pencil route")
		}
	}
	mimo, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	ls := func(order int) C2DOptions { return C2DOptions{Method: C2DMethodLeastSquares, FitOrder: order} }
	for _, c := range []struct {
		name     string
		class    resultContractClass
		sys      *System
		opts     C2DOptions
		sentinel error
	}{
		{"explicit", resultCertified, explicit, ls(0), nil},
		{"pencil", resultUnverified, routed, ls(0), nil},
		{"pencil order 1", resultUnverified, routed, ls(1), nil},
		{"pencil chain", resultUnverified, chain, ls(0), nil},
		{"pencil chain order 1", resultFailed, chain, ls(1), ErrInvalidOrder},
		{"MIMO", resultFailed, mimo, ls(0), ErrNotSISO},
	} {
		df, fit, ferr := c.sys.C2DFit(dt, c.opts)
		checkResultContract(t, "C2DFit "+c.name, c.class, df != nil, ferr, c.sentinel)
		d, err := c.sys.C2D(dt, c.opts)
		checkResultContract(t, "C2D "+c.name, c.class, d != nil, err, c.sentinel)
		if c.class == resultFailed {
			if fit != (LeastSquaresFit{}) {
				t.Errorf("C2DFit %s: fit %+v with %v, want zero", c.name, fit, ferr)
			}
			continue
		}
		if fit.FitOrder == 0 || !(fit.MaxRelativeError >= 0) {
			t.Errorf("C2DFit %s: fit %+v", c.name, fit)
		}
		a1, b1, c1, d1 := fzRaw(df)
		a2, b2, c2, d2 := fzRaw(d)
		if !slices.Equal(a1, a2) || !slices.Equal(b1, b2) || !slices.Equal(c1, c2) || !slices.Equal(d1, d2) || df.Dt != d.Dt {
			t.Errorf("%s: C2DFit and C2D models differ", c.name)
		}
	}
	if _, _, err := explicit.C2DFit(dt, C2DOptions{}); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("C2DFit zoh: %v, want ErrOptionUnsupported", err)
	}
	out, G, err := routed.C2DMap(dt, ls(0))
	checkResultContract(t, "C2DMap pencil", resultFailed, out != nil || G != nil, err, ErrOptionUnsupported)
}
