package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// nearSingularEFoldOf is one fold of nearSingularEFold, for seeds whose
// other member has no explicit form.
func nearSingularEFoldOf(t *testing.T, seed uint64, rotated bool) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7704))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, twin := nearSingularEDraw(t, rng, n, m, p, k)
	if !rotated {
		sys = twin
	}
	ex, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	return ex
}

// TestReduceResolvedGapPerSideCount checks the per-side count of the
// resolved-gap check (N3P7NO): on these stiff continuous models (‖A‖ up to
// ~1e15 against O(1) slow poles) the staircase cut drops a slow state that
// carries O(1) of the response, and the global count n + nr alone accepts
// it, met by fast-scale points that cannot see the slow poles (exact
// deviation 0.03..3.5 of the response beyond the model's own bound). Each
// must keep its order: the folds unreduced, planted 182 at the structural
// remainder's 5.
func TestReduceResolvedGapPerSideCount(t *testing.T) {
	unobs := &ReduceOpts{Mode: ReduceUnobservable}
	for _, tc := range []struct {
		name string
		sys  func() *System
		want int
	}{
		{"twin 33", func() *System { return nearSingularEFoldOf(t, 33, false) }, 3},
		{"twin 1518", func() *System { return nearSingularEFoldOf(t, 1518, false) }, 3},
		{"twin 1903", func() *System { return nearSingularEFoldOf(t, 1903, false) }, 3},
		{"rotated 1518", func() *System { return nearSingularEFoldOf(t, 1518, true) }, 3},
		{"rotated 1076", func() *System { return nearSingularEFoldOf(t, 1076, true) }, 4},
		{"planted 26", func() *System { return stiffPlantedCase(rand.New(rand.NewPCG(26, 0x5713ff))) }, 4},
		{"planted 182", func() *System { return stiffPlantedCase(rand.New(rand.NewPCG(182, 0x5713ff))) }, 5},
	} {
		sys := tc.sys()
		res, err := sys.Reduce(unobs)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if res.Order != tc.want {
			t.Errorf("%s: order %d, want %d", tc.name, res.Order, tc.want)
			continue
		}
		if e := reducedResponseErr(t, sys, res.Sys, slowProbePoints); e > 1e-8 {
			t.Errorf("%s: response error %g", tc.name, e)
		}
	}
}

// TestReduceResolvedGapDiscreteGlobalCount checks that a discrete model
// takes the global count alone (N3P7NO): fuzz case 4265, whose gap path
// early-accepted the staircase's order-5 cut on amd64 by the float probe
// alone, keeps that cut, now verified by the band (deviation 5.6e-14 of
// the response beyond the model's bound).
func TestReduceResolvedGapDiscreteGlobalCount(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	var kind uint8
	var v [fzNVals]float64
	for range 4266 {
		kind = uint8(rng.IntN(256))
		for i := range v {
			v[i] = 3 * rng.NormFloat64()
		}
	}
	sys := fzDecode(t, kind, v).System(t, fzSysOpts{})
	res, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 5 {
		t.Errorf("order %d, want 5", res.Order)
	}
}

// TestReduceResolvedGapSplitAccepted covers the accept of the split across
// a resolved gap, which no census row reaches (N3P7NO spec item v). A is
// upper triangular, fast state −2⁵⁰ and slow block [−1 0.5; 0 −2] with its
// first state unobservable, the slow block then rotated by the exact
// similarity [1 1; 1 −1] (inverse half of it), so no row or column is
// structurally zero. Against the floor ε·‖A‖ ≈ 0.25 the full staircase
// keeps only the fast state, a cut that fails the check; the split reduces
// each block on its own floor, removes just the unobservable state and is
// verified with the per-side counts, so Reduce returns it at order 2. The
// spec's −2⁴⁰ leaves both slow states to the staircase (its cut still
// fails, but drops no slow state), hence −2⁵⁰; and no dense 2×2 orthogonal
// matrix is exact in binary, so the rotation is the Hadamard similarity,
// orthogonal up to the scale 1/√2 that a similarity ignores.
func TestReduceResolvedGapSplitAccepted(t *testing.T) {
	fast := math.Ldexp(1, 50)
	A := mat.NewDense(3, 3, []float64{
		-1, 0.5, 0.75,
		0, -2, -1.25,
		0, 0, -fast,
	})
	B := mat.NewDense(3, 1, []float64{1, 1.5, fast})
	C := mat.NewDense(1, 3, []float64{0, 1, 1})
	T := mat.NewDense(3, 3, []float64{1, 1, 0, 1, -1, 0, 0, 0, 1})
	Ti := mat.NewDense(3, 3, []float64{0.5, 0.5, 0, 0.5, -0.5, 0, 0, 0, 1})
	var tmp, Ar, Br, Cr mat.Dense
	tmp.Mul(T, A)
	Ar.Mul(&tmp, Ti)
	Br.Mul(T, B)
	Cr.Mul(C, Ti)
	sys, err := New(&Ar, &Br, &Cr, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := exactMinimalOrder(sys); got != 2 {
		t.Fatalf("exact minimal order %d, want 2", got)
	}
	if keep := structuralStates(sys, true, true); len(keep) != 3 {
		t.Fatalf("structural pass keeps %d states, want 3", len(keep))
	}
	policy := newRealizationTransformPolicy(sys)
	opts := &ReduceOpts{}
	tol := math.Sqrt(eps())
	_, cutoff, ok := resolvedGap(sys.A)
	if !ok {
		t.Fatal("no resolved gap")
	}
	base, removed, err := reduceByStaircase(sys, policy, opts, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	if base.Order != 1 || frequencyAxisCutAcrossGap(sys, base.Sys, removed, tol, cutoff) {
		t.Fatalf("staircase cut: order %d, want 1 and rejected", base.Order)
	}
	split, ok := reduceAcrossSpectralGap(sys, policy, opts, cutoff)
	if !ok || split.Order != 2 || !frequencyAxisCutAcrossGap(sys, split.Sys, nil, tol, cutoff) {
		t.Fatalf("split: ok %v, order %d, want 2 and verified", ok, split.Order)
	}
	res, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 2 {
		t.Fatalf("order %d, want 2", res.Order)
	}
	for _, pair := range [][2]*mat.Dense{{res.Sys.A, split.Sys.A}, {res.Sys.B, split.Sys.B}, {res.Sys.C, split.Sys.C}} {
		if !mat.Equal(pair[0], pair[1]) {
			t.Fatalf("result is not the split:\n%v\n%v", mat.Formatted(pair[0]), mat.Formatted(pair[1]))
		}
	}
	if res.Sys.A.At(0, 1) != 0 || res.Sys.A.At(1, 0) != 0 {
		t.Errorf("A not block diagonal:\n%v", mat.Formatted(res.Sys.A))
	}
	if e := reducedResponseErr(t, sys, res.Sys, append(slowProbePoints, 1e14i, 1e16i)); e > 1e-8 {
		t.Errorf("response error %g", e)
	}
}

// TestReduceWindowCapAtOtherPole checks the window cap of
// sharedPoleWindows (N3P7NO): a window may not reach another model pole.
// The model is upper triangular, poles −1, −2 and −2⁴⁰, with the −2 mode
// carrying a residue of 5e-5; the cut keeps −1 and −2⁴⁰ in a Hadamard-rotated
// (exact) realization and drops −2, so it is wrong by 5e-5/|s+2| exactly,
// 2500·tol at s = 2j. The rotation smears the fast scale into the cut's −1:
// β_R ≈ ε·2⁴⁰ ≈ 2.4e-4 ≤ ρ₀, an uncapped window of radius β_R/tol ≈ 1.6e4
// covers −2 and every slow probe, makes them uninformative, and the
// mid-band points, where the fast mode dominates and the error is below
// tol, meet the count: the wrong cut is accepted. Capped at the distance
// to −2 the window leaves −2's probes to the model-determined rule, which
// rejects.
func TestReduceWindowCapAtOtherPole(t *testing.T) {
	const d = 1e-4
	f := -math.Ldexp(1, 40)
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0.5, 0, 0, -2, 0, 0, 0, f}),
		mat.NewDense(3, 1, []float64{1, d, -f}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	r1 := 1 + 0.5*d
	red, err := New(
		mat.NewDense(2, 2, []float64{(-1 + f) / 2, (-1 - f) / 2, (-1 - f) / 2, (-1 + f) / 2}),
		mat.NewDense(2, 1, []float64{r1 - f, r1 + f}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	tol := math.Sqrt(eps())
	if e := reducedResponseErr(t, sys, red, []complex128{2i}); e < 1e3*tol {
		t.Fatalf("cut error %g at s = 2j, want ≥ %g", e, 1e3*tol)
	}
	_, poles, ok := axisProbes(sys, nil)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		t.Fatal("eigenvalues of the cut did not converge")
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	for i, lr := range evR {
		if math.Abs(real(lr)+1) > 1e-3 {
			continue
		}
		if !(betaR[i] <= poles.rho0 && betaR[i]/tol > 1) {
			t.Fatalf("β_R %g, ρ₀ %g: uncapped window would not reach −2", betaR[i], poles.rho0)
		}
		for k, c := range win.centre {
			if c == lr && win.radius[k] > cmplx.Abs(lr+2) {
				t.Errorf("window about %v, radius %g, reaches the pole −2", lr, win.radius[k])
			}
		}
		if win.contain(2i) {
			t.Errorf("probe 2j inside a window: radii %v", win.radius)
		}
	}
	if frequencyAxisCutInBand(sys, red, nil, tol) {
		t.Error("cut that drops the −2 mode accepted")
	}
}
