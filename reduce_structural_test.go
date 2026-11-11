package controlsys

import (
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// assertExactSubsystem fails unless res keeps exactly the states keep of
// sys: the same entries of A, B, C and D, bit for bit, in order.
func assertExactSubsystem(t *testing.T, name string, sys *System, res *ReduceResult, keep []int) {
	t.Helper()
	if res.Order != len(keep) {
		t.Fatalf("%s: order %d, want %d", name, res.Order, len(keep))
	}
	_, m, p := sys.Dims()
	for ii, i := range keep {
		for jj, j := range keep {
			if g, w := res.Sys.A.At(ii, jj), sys.A.At(i, j); g != w {
				t.Fatalf("%s: A[%d,%d] = %v, want %v", name, ii, jj, g, w)
			}
		}
		for j := range m {
			if g, w := res.Sys.B.At(ii, j), sys.B.At(i, j); g != w {
				t.Fatalf("%s: B[%d,%d] = %v, want %v", name, ii, j, g, w)
			}
		}
		for j := range p {
			if g, w := res.Sys.C.At(j, ii), sys.C.At(j, i); g != w {
				t.Fatalf("%s: C[%d,%d] = %v, want %v", name, j, ii, g, w)
			}
		}
	}
	if !mat.Equal(res.Sys.D, sys.D) {
		t.Fatalf("%s: D changed", name)
	}
}

// assertReduceDeletesLastStateExactly fails unless Reduce under min, unc
// and Equalize returns sys without its last state, entry for entry, and
// the high-precision transfer functions of both agree, also at the
// matched-C2D zero z = -1.
func assertReduceDeletesLastStateExactly(t *testing.T, name string, sys *System) {
	t.Helper()
	n, _, _ := sys.Dims()
	keep := make([]int, n-1)
	for i := range keep {
		keep[i] = i
	}
	full := newStepOraclePencil(sys)
	pts := []complex128{1, cmplx.Rect(1, 0.3), cmplx.Rect(1, 2), -1}
	for _, opts := range []*ReduceOpts{{}, {Mode: ReduceUncontrollable}, {Equalize: true}} {
		label := name + " " + reduceModeName(opts)
		res, err := sys.Reduce(opts)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		assertExactSubsystem(t, label, sys, res, keep)
		cut := newStepOraclePencil(res.Sys)
		for _, z := range pts {
			g, r := matchedOracleTransfer(full, z), matchedOracleTransfer(cut, z)
			if d := cmplx.Abs(g - r); d > 1e-14*cmplx.Abs(g) {
				t.Fatalf("%s: oracle |G-R| = %g at %v, |G| = %g", label, d, z, cmplx.Abs(g))
			}
		}
	}
}

func reduceModeName(opts *ReduceOpts) string {
	name := []string{"all", "unc", "unobs"}[opts.Mode]
	if opts.Equalize {
		name += "+eq"
	}
	return name
}

// x3 is driven by nothing (structurally uncontrollable, but observable
// through x0) and x4 drives nothing (structurally unobservable, but
// reachable through x1); the remaining states are minimal. Each mode
// deletes its structurally disconnected states and keeps the rest in the
// original coordinates, also under Equalize.
func TestReduceRemovesStructurallyDisconnectedStatesExactly(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{
		-1.0, 0.4, 0.0, 0.7, 0.0,
		0.3, -2.0, 0.5, 0.0, 0.0,
		0.0, 0.6, -3.0, 0.0, 0.0,
		0.0, 0.0, 0.0, -0.5, 0.0,
		0.0, 0.9, 0.0, 0.0, -4.0,
	})
	B := mat.NewDense(5, 1, []float64{1, 0, 0.2, 0, 0})
	C := mat.NewDense(1, 5, []float64{1, 0, 0.3, 0, 0})
	for _, dt := range []float64{0, 0.1} {
		a := A
		if dt != 0 {
			var e mat.Dense
			e.Scale(0.1, A)
			for i := range 5 {
				e.Set(i, i, 1+e.At(i, i))
			}
			a = &e
		}
		sys, err := New(a, B, C, mat.NewDense(1, 1, []float64{0.25}), dt)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			opts *ReduceOpts
			keep []int
		}{
			{&ReduceOpts{}, []int{0, 1, 2}},
			{&ReduceOpts{Mode: ReduceUncontrollable}, []int{0, 1, 2, 4}},
			{&ReduceOpts{Mode: ReduceUnobservable}, []int{0, 1, 2, 3}},
			{&ReduceOpts{Equalize: true}, []int{0, 1, 2}},
			{&ReduceOpts{Mode: ReduceUncontrollable, Equalize: true}, []int{0, 1, 2, 4}},
			{&ReduceOpts{Mode: ReduceUnobservable, Equalize: true}, []int{0, 1, 2, 3}},
		} {
			res, err := sys.Reduce(tc.opts)
			if err != nil {
				t.Fatalf("dt=%g %s: %v", dt, reduceModeName(tc.opts), err)
			}
			assertExactSubsystem(t, reduceModeName(tc.opts), sys, res, tc.keep)
		}
	}
}

// A model whose every state is structurally connected (a dense rotation of
// a model with an uncontrollable mode): Reduce is the staircase stage
// alone, bit for bit.
func TestReduceWithoutStructuralRemovalIsStaircaseStage(t *testing.T) {
	q := mat.NewDense(3, 3, []float64{7.0 / 9, -4.0 / 9, -4.0 / 9, -4.0 / 9, 1.0 / 9, -8.0 / 9, -4.0 / 9, -8.0 / 9, 1.0 / 9})
	A0 := mat.NewDense(3, 3, []float64{-1, 0.7, 0.2, 0, -2.5, 0.3, 0, 0, -4})
	B0 := mat.NewDense(3, 1, []float64{1, 1, 0})
	C0 := mat.NewDense(1, 3, []float64{1, -0.5, 2})
	var tmp, a, b, c mat.Dense
	tmp.Mul(q, A0)
	a.Mul(&tmp, q.T())
	b.Mul(q, B0)
	c.Mul(C0, q.T())
	sys, err := New(&a, &b, &c, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if keep := structuralStates(sys, true, true); len(keep) != 3 {
		t.Fatalf("structural states %v, want all 3", keep)
	}
	for _, opts := range []*ReduceOpts{{}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}, {Equalize: true}} {
		got, err := sys.Reduce(opts)
		if err != nil {
			t.Fatal(err)
		}
		want, err := reduceStates(sys, newRealizationTransformPolicy(sys), opts, true, sys)
		if err != nil {
			t.Fatal(err)
		}
		if got.Order != want.Order || !mat.Equal(got.Sys.A, want.Sys.A) || !mat.Equal(got.Sys.B, want.Sys.B) || !mat.Equal(got.Sys.C, want.Sys.C) {
			t.Fatalf("%s: Reduce differs from the staircase stage (order %d vs %d)", reduceModeName(opts), got.Order, want.Order)
		}
	}
}

// Y5H2MH/KICQJA matched-hid rows (*-false): a matched C2D model with a
// planted state, last in the realization, that no input reaches. Before
// the structural stage the staircase's rotated cut was wrong at z = -1 or
// unverifiable, so these came back unreduced; now every mode that removes
// uncontrollable states deletes the planted state exactly, and
// ReduceUnobservable leaves the model as it is.
func TestReduceRemovesPlantedStateOfMatchedModelExactly(t *testing.T) {
	lits := reduceVerifyLiterals(t)
	for _, name := range []string{"403-unc-false", "62-unc-false", "188-unc-false", "301-unc-false", "406-unc-false", "148-unc-false", "221-unc-false-amd64", "246-unc-false-amd64"} {
		sys := lits[name]
		assertReduceDeletesLastStateExactly(t, name, sys)
		res, err := sys.Reduce(&ReduceOpts{Mode: ReduceUnobservable})
		if err != nil {
			t.Fatalf("%s unobs: %v", name, err)
		}
		assertReducedUnchanged(t, name+" unobs", sys, res)
	}
}

// The structural stage keeps the external delays and I/O names.
func TestReduceStructuralStageKeepsDelays(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.5, 0, 0.2, -2, 0, 0, 0, -3})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{0, 1, 0})
	sys, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{0.3}
	sys.InputName = []string{"u"}
	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}
	assertExactSubsystem(t, "input delay", sys, res, []int{0, 1})
	if !slices.Equal(res.Sys.InputDelay, sys.InputDelay) || !slices.Equal(res.Sys.InputName, sys.InputName) {
		t.Fatalf("delay %v names %v, want %v %v", res.Sys.InputDelay, res.Sys.InputName, sys.InputDelay, sys.InputName)
	}
	for _, s := range []complex128{0.4i, 3i} {
		g, _ := sys.EvalFr(s)
		r, _ := res.Sys.EvalFr(s)
		if g[0][0] != r[0][0] && cmplx.Abs(g[0][0]-r[0][0]) > 1e-15*cmplx.Abs(g[0][0]) {
			t.Fatalf("response at %v: %v, want %v", s, r[0][0], g[0][0])
		}
	}
}
