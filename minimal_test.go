package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func markovParams(sys *System, k int) []float64 {
	n, m, p := sys.Dims()
	if k == 0 {
		if p == 0 || m == 0 {
			return nil
		}
		data := make([]float64, p*m)
		for i := range p {
			for j := range m {
				data[i*m+j] = sys.D.At(i, j)
			}
		}
		return data
	}
	if n == 0 || p == 0 || m == 0 {
		return make([]float64, p*m)
	}
	// C * A^(k-1) * B
	pow := mat.NewDense(n, n, nil)
	for i := range n {
		pow.Set(i, i, 1)
	}
	for i := 0; i < k-1; i++ {
		var tmp mat.Dense
		tmp.Mul(pow, sys.A)
		pow.Copy(&tmp)
	}
	var cab mat.Dense
	var tmp2 mat.Dense
	tmp2.Mul(sys.C, pow)
	cab.Mul(&tmp2, sys.B)
	r, c := cab.Dims()
	data := make([]float64, r*c)
	for i := range r {
		for j := range c {
			data[i*c+j] = cab.At(i, j)
		}
	}
	return data
}

func checkMarkovEqual(t *testing.T, orig, reduced *System, maxK int, tol float64) {
	t.Helper()
	for k := 0; k <= maxK; k++ {
		mo := markovParams(orig, k)
		mr := markovParams(reduced, k)
		if len(mo) != len(mr) {
			t.Errorf("Markov param k=%d: length mismatch %d vs %d", k, len(mo), len(mr))
			continue
		}
		for i := range mo {
			if math.Abs(mo[i]-mr[i]) > tol {
				t.Errorf("Markov param k=%d, elem %d: orig=%g reduced=%g", k, i, mo[i], mr[i])
			}
		}
	}
}

func TestReduceAlreadyMinimal(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2, got %d", res.Order)
	}

	checkMarkovEqual(t, sys, res.Sys, 6, 1e-10)
}

func TestReduceUncontrollableBZero(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 2, 0, 0, 0, 3})
	B := mat.NewDense(3, 1, nil)
	C := mat.NewDense(1, 3, []float64{1, 1, 1})
	D := mat.NewDense(1, 1, []float64{5})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for B=0, got %d", res.Order)
	}

	if res.Sys.D.At(0, 0) != 5 {
		t.Errorf("D not preserved: got %g", res.Sys.D.At(0, 0))
	}
}

func TestReduceUnobservableCZero(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 2, 0, 0, 0, 3})
	B := mat.NewDense(3, 1, []float64{1, 1, 1})
	C := mat.NewDense(1, 3, nil)
	D := mat.NewDense(1, 1, []float64{7})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for C=0, got %d", res.Order)
	}

	if res.Sys.D.At(0, 0) != 7 {
		t.Errorf("D not preserved: got %g", res.Sys.D.At(0, 0))
	}
}

func TestReducePartiallyControllableObservable(t *testing.T) {
	// 4-state system: states 1,2 controllable+observable; state 3 uncontrollable; state 4 unobservable
	A := mat.NewDense(4, 4, []float64{
		0, 1, 0, 0,
		-2, -3, 0, 0,
		0, 0, -5, 0,
		0, 0, 0, -7,
	})
	B := mat.NewDense(4, 1, []float64{0, 1, 0, 1})
	C := mat.NewDense(1, 4, []float64{1, 0, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order > 3 {
		t.Errorf("expected order <= 3, got %d", res.Order)
	}

	n, _, _ := sys.Dims()
	checkMarkovEqual(t, sys, res.Sys, 2*n, 1e-8)
}

func TestReduceUncontrollableOnly(t *testing.T) {
	// State 3 is uncontrollable but observable
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		-2, -3, 0,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{0, 1, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(&ReduceOpts{Mode: ReduceUncontrollable})
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2 (controllable only), got %d", res.Order)
	}
}

func TestReduceUnobservableOnly(t *testing.T) {
	// State 3 is controllable but unobservable
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		-2, -3, 0,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{0, 1, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(&ReduceOpts{Mode: ReduceUnobservable})
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2 (observable only), got %d", res.Order)
	}
}

func TestReduceZeroDimension(t *testing.T) {
	sys := &System{A: &mat.Dense{}, B: &mat.Dense{}, C: &mat.Dense{}, D: &mat.Dense{}}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0, got %d", res.Order)
	}
}

func TestMinimalRealizationAlias(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2, got %d", res.Order)
	}
}

func TestReduceMarkovPreserved(t *testing.T) {
	// Controllable canonical form for (s+1)(s+2)(s+3)
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	checkMarkovEqual(t, sys, res.Sys, 2*n, 1e-10)
}

func TestReduceMZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := &mat.Dense{}
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := &mat.Dense{}

	sys := &System{A: A, B: B, C: C, D: D}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for m=0, got %d", res.Order)
	}
}

func TestReducePZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := &mat.Dense{}
	D := &mat.Dense{}

	sys := &System{A: A, B: B, C: C, D: D}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for p=0, got %d", res.Order)
	}
}

// Issue #26: MinimalRealization should not over-reduce a closed-loop PI + plant
// system to order 0. The pole-zero cancellation of the PI integrator against the
// plant's unity DC gain leaves a 2-state minimal realization (the closed-loop
// complex-conjugate pole pair).
func TestReduceIssue26PIPlantLoop(t *testing.T) {
	K, tau1, beta := 3.333, 11998.8, 11998.8
	ctrl, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{K / tau1}),
		mat.NewDense(1, 1, []float64{K * beta / tau1}),
		0,
	)
	if err != nil {
		t.Fatalf("ctrl: %v", err)
	}
	Kp, tauP := 0.56, 66240.0
	plant, err := New(
		mat.NewDense(1, 1, []float64{-1.0 / tauP}),
		mat.NewDense(1, 1, []float64{1.0 / tauP}),
		mat.NewDense(1, 1, []float64{Kp}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatalf("plant: %v", err)
	}

	cl, err := Feedback(ctrl, plant, -1)
	if err != nil {
		t.Fatalf("Feedback: %v", err)
	}
	clToPv, err := Series(cl, plant)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}

	mr, err := clToPv.MinimalRealization()
	if err != nil {
		t.Fatalf("MinimalRealization: %v", err)
	}
	if mr.Order != 2 {
		t.Fatalf("Order=%d, want 2", mr.Order)
	}
	stable, err := mr.Sys.IsStable()
	if err != nil {
		t.Fatalf("IsStable: %v", err)
	}
	if !stable {
		t.Errorf("minimal system should be stable")
	}
	dc, err := mr.Sys.DCGain()
	if err != nil {
		t.Fatalf("DCGain: %v", err)
	}
	if v := dc.At(0, 0); math.Abs(v-1.0) > 1e-6 {
		t.Errorf("DCGain=%g, want ≈1", v)
	}
}

// Issue #26: when the controllability staircase returns ncont=0 (e.g., B=0),
// Reduce must return a cleanly empty 0-state system, not a bogus 1×1 zero
// matrix that IsStable then flags as marginally unstable.
func TestReduceZeroControllableReturnsEmpty(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{0, 0})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{7})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mr, err := sys.MinimalRealization()
	if err != nil {
		t.Fatalf("MinimalRealization: %v", err)
	}
	if mr.Order != 0 {
		t.Fatalf("Order=%d, want 0", mr.Order)
	}
	ar, ac := mr.Sys.A.Dims()
	br, bc := mr.Sys.B.Dims()
	cr, cc := mr.Sys.C.Dims()
	if ar != 0 || ac != 0 {
		t.Errorf("A dims=%dx%d, want 0x0", ar, ac)
	}
	if br != 0 || bc != 0 {
		t.Errorf("B dims=%dx%d, want 0x0", br, bc)
	}
	if cr != 0 || cc != 0 {
		t.Errorf("C dims=%dx%d, want 0x0", cr, cc)
	}
	if v := mr.Sys.D.At(0, 0); v != 7 {
		t.Errorf("D preserved? got %g, want 7", v)
	}
}

// Issue #27: equalize() must match LAPACK DGEBAL + SLICOT TB01ID semantics.
// A deliberately imbalanced 2×2 system should have its row/col 1-norms
// brought into balance with finite entries and identical A-eigenvalues
// (diagonal similarity is a similarity transform).
func TestEqualizeImbalanced2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 1e10, 1e-10, 2})
	B := mat.NewDense(2, 1, []float64{1e10, 1e-10})
	C := mat.NewDense(1, 2, []float64{1e-10, 1e10})

	before := eigSorted(A)

	Aw := mat.DenseCopyOf(A)
	Bw := mat.DenseCopyOf(B)
	Cw := mat.DenseCopyOf(C)
	equalize(Aw, Bw, Cw, 2, 1, 1)

	assertAllFinite(t, "A", Aw)
	assertAllFinite(t, "B", Bw)
	assertAllFinite(t, "C", Cw)

	after := eigSorted(Aw)
	for i := range before {
		diff := cmplx.Abs(before[i] - after[i])
		scale := math.Max(1, cmplx.Abs(before[i]))
		if diff/scale > 1e-9 {
			t.Errorf("eig[%d] changed: before=%v after=%v (rel diff %.2e)",
				i, before[i], after[i], diff/scale)
		}
	}

	beforeRatio := normRatio(A, B, C)
	afterRatio := normRatio(Aw, Bw, Cw)
	if afterRatio >= beforeRatio {
		t.Errorf("balance did not improve: before=%.2e after=%.2e",
			beforeRatio, afterRatio)
	}
}

// Issue #27: equalize with the issue #26 closed-loop should not diverge.
// Before the fix, one pass made A[0,1]=-Inf and B blew up to 1e+186.
func TestEqualizeIssue26SystemFinite(t *testing.T) {
	K, tau1, beta := 3.333, 11998.8, 11998.8
	ctrl, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{K / tau1}),
		mat.NewDense(1, 1, []float64{K * beta / tau1}),
		0,
	)
	Kp, tauP := 0.56, 66240.0
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1.0 / tauP}),
		mat.NewDense(1, 1, []float64{1.0 / tauP}),
		mat.NewDense(1, 1, []float64{Kp}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	cl, _ := Feedback(ctrl, plant, -1)
	clToPv, _ := Series(cl, plant)

	mr, err := clToPv.Reduce(&ReduceOpts{Mode: ReduceAll, Equalize: true})
	if err != nil {
		t.Fatalf("Reduce(Equalize): %v", err)
	}

	n, _, _ := mr.Sys.Dims()
	if n > 0 {
		assertAllFinite(t, "A", mr.Sys.A)
		assertAllFinite(t, "B", mr.Sys.B)
		assertAllFinite(t, "C", mr.Sys.C)
		stable, _ := mr.Sys.IsStable()
		if !stable {
			t.Errorf("balanced minimal system should be stable")
		}
	}
}

func assertAllFinite(t *testing.T, name string, m *mat.Dense) {
	t.Helper()
	r, c := m.Dims()
	for i := range r {
		for j := range c {
			v := m.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("%s[%d,%d] = %v (non-finite)", name, i, j, v)
			}
		}
	}
}

func eigSorted(A *mat.Dense) []complex128 {
	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenNone)
	if !ok {
		return nil
	}
	vals := eig.Values(nil)
	sort.Slice(vals, func(i, j int) bool {
		if real(vals[i]) != real(vals[j]) {
			return real(vals[i]) < real(vals[j])
		}
		return imag(vals[i]) < imag(vals[j])
	})
	return vals
}

func normRatio(A, B, C *mat.Dense) float64 {
	n, _ := A.Dims()
	_, m := B.Dims()
	p, _ := C.Dims()
	maxNorm := 0.0
	minNorm := math.Inf(1)
	for i := range n {
		r, c := 0.0, 0.0
		for j := range n {
			if j == i {
				continue
			}
			r += math.Abs(A.At(i, j))
			c += math.Abs(A.At(j, i))
		}
		for j := range m {
			r += math.Abs(B.At(i, j))
		}
		for j := range p {
			c += math.Abs(C.At(j, i))
		}
		if r > maxNorm {
			maxNorm = r
		}
		if c > maxNorm {
			maxNorm = c
		}
		if r > 0 && r < minNorm {
			minNorm = r
		}
		if c > 0 && c < minNorm {
			minNorm = c
		}
	}
	if minNorm == 0 || math.IsInf(minNorm, 1) {
		return math.Inf(1)
	}
	return maxNorm / minNorm
}

// Issue #26: same check for the observability branch — when the dual staircase
// returns nobs=0 (e.g., C=0), Reduce must return a cleanly empty system.
func TestReduceZeroObservableReturnsEmpty(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{0, 0})
	D := mat.NewDense(1, 1, []float64{9})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mr, err := sys.MinimalRealization()
	if err != nil {
		t.Fatalf("MinimalRealization: %v", err)
	}
	if mr.Order != 0 {
		t.Fatalf("Order=%d, want 0", mr.Order)
	}
	ar, ac := mr.Sys.A.Dims()
	if ar != 0 || ac != 0 {
		t.Errorf("A dims=%dx%d, want 0x0", ar, ac)
	}
	if v := mr.Sys.D.At(0, 0); v != 9 {
		t.Errorf("D preserved? got %g, want 9", v)
	}
}

func TestReduceKeepsExternalDelays(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		orig := fieldIODelay(t, dt)
		res, err := orig.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 2 {
			t.Errorf("dt=%g: order %d, want 2", dt, res.Order)
		}
		assertFieldResponse(t, fmt.Sprintf("dt=%g", dt), res.Sys, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
	}
}

func TestReduceInternalDelayUsesAugmentedChannels(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		orig := fieldLFT(t, dt)
		res, err := orig.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 2 {
			t.Errorf("dt=%g: order %d, want 2", dt, res.Order)
		}
		if res.Sys.internalDelayCount() != 1 {
			t.Fatalf("dt=%g: internal delay dropped", dt)
		}
		assertFieldResponse(t, fmt.Sprintf("dt=%g", dt), res.Sys, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })

		// State 3 is reachable only through the internal-delay channel B2.
		viaB2 := fieldLFT(t, dt)
		viaB2.LFT.B2.Set(2, 0, 0.8)
		res, err = viaB2.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 3 {
			t.Errorf("dt=%g: B2-reachable state removed, order %d, want 3", dt, res.Order)
		}
		assertFieldResponse(t, fmt.Sprintf("viaB2/dt=%g", dt), res.Sys, func(s complex128) [][]complex128 { return fieldOracle(viaB2, s) })
	}
}

func TestReduceInternalDelayToZeroOrder(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		orig := fieldLFT(t, dt)
		orig.C.Zero()
		orig.LFT.C2.Zero()
		res, err := orig.MinimalRealization()
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		if res.Order != 0 || res.Sys.internalDelayCount() != 1 {
			t.Errorf("dt=%g: order %d internal delays %d, want 0 and 1", dt, res.Order, res.Sys.internalDelayCount())
		}
		if err := res.Sys.Validate(); err != nil {
			t.Fatalf("dt=%g: Validate: %v", dt, err)
		}
		assertFieldResponse(t, fmt.Sprintf("dt=%g", dt), res.Sys, func(s complex128) [][]complex128 { return fieldOracle(orig, s) })
	}
}
