package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
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

// TestReduceResolvedGapSlowCountAtSlowScale checks the slow count of
// frequencyAxisCutAcrossGap on rotfast 13 (both census corpora), a stiff
// rotated model whose slow poles near 1 sit 1e11 below its fast one (its
// ReduceUnobservable cut is 0.1% off the response at s = 0, where the
// probes are below the model's own bound). The staircase cuts' verified
// probes all lie at
// |s| ≥ 1e6, where the slow dynamics show only at relative |λ_s|/|s| =
// 1e-6, and meet the global count there; none lies at |s| ≤ ρ_s, the
// slow scale, so the slow count refuses them and reduceStates keeps the
// order in every mode.
func TestReduceResolvedGapSlowCountAtSlowScale(t *testing.T) {
	tol := math.Sqrt(eps())
	for _, name := range []string{"rotfast/13", "rotfast/13@arm64"} {
		sys := bandLiterals[name].system(t)
		n, _, _ := sys.Dims()
		_, cutoff, ok := resolvedGap(sys.A)
		if !ok {
			t.Fatalf("%s: no resolved gap", name)
		}
		for _, opts := range []*ReduceOpts{{}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}, {Equalize: true}} {
			policy := newRealizationTransformPolicy(sys)
			if res, err := reduceStates(sys, policy, opts, true, sys); err != nil || res.Order != n {
				t.Errorf("%s %+v: order %v err %v, want %d", name, opts, res.Order, err, n)
			}
			cut, removed, err := reduceByStaircase(sys, policy, opts, true, -1)
			if err != nil {
				t.Fatal(err)
			}
			evs, _, _ := eigenRoundingBounds(sys.A)
			evR, _, _ := eigenRoundingBounds(cut.Sys.A)
			rho := 0.0
			for _, ev := range append(evs, evR...) {
				if a := cmplx.Abs(ev); a < cutoff {
					rho = math.Max(rho, a)
				}
			}
			pts, poles, _ := axisProbes(sys, removed)
			win, _ := sharedPoleWindows(poles, cut.Sys, tol)
			full, rc := newBoundedResponse(sys, pts), newBoundedResponse(cut.Sys, pts)
			slow, other := 0, 0
			for k, s := range pts {
				if classifyProbe(&full, &rc, k, tol, win.contain(s)) != probeVerified {
					continue
				}
				if cmplx.Abs(s) <= rho {
					slow += probeWeight(s)
				} else {
					other += probeWeight(s)
				}
			}
			if slow != 0 || other < n+cut.Order {
				t.Errorf("%s %+v: verified weight %d at |s| ≤ ρ_s = %.4g and %d beyond, want 0 and ≥ %d", name, opts, slow, rho, other, n+cut.Order)
			}
			if frequencyAxisCutAcrossGap(sys, cut.Sys, removed, tol, cutoff) {
				t.Errorf("%s %+v: band accepts the staircase cut", name, opts)
			}
		}
	}
}

// TestReduceResolvedGapDiscreteGlobalCount checks that a discrete model
// takes the global count alone (N3P7NO): fuzz case 4265, whose gap path
// early-accepted the staircase's order-5 cut on amd64 by the float probe
// alone. The order-5 cut, where it is accepted, must be within Reduce's
// contract of the stored model at and beside its poles; on arm64 it is
// 1.07·tol off inside a shared-pole window, which the double-double
// judgement of the window refutes (HO6UP6), so the order is kept there,
// and arm64 takes the band, not the gap path, even on main: there
// TestReduceResolvedGapDiscreteGlobalCountLiteral covers the gap path.
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
	switch res.Order {
	case 5:
		if c, at := contractSweepAtPoles(sys, res.Sys, math.Sqrt(eps())); c > 1 {
			t.Errorf("order 5 cut off by %.3g·tol·max(scale, terms) at %v", c, at)
		}
	case 6:
	default:
		t.Errorf("order %d, want 5 or 6", res.Order)
	}
}

// discreteGapLiteral is fuzz case 12607 frozen on arm64, where fuzz case
// 4265 no longer reaches the gap path: a discrete model with a resolved
// gap whose staircase order-2 cut the gap path accepts by the global count
// alone, and refuses with the per-side count.
var discreteGapLiteral = bandLiteral{
	n: 3, m: 1, p: 1, dt: 0.2248521902777981,
	a: []uint64{0x3f5b91fba28bf12a, 0x3f04ef100fcf92e4, 0xbfa2266fc329581d, 0xbf4196e9d6f5f869, 0xbeeab60b05487ed7, 0x3f8728bf86a3e43e, 0xbfa83a01bad650f1, 0xbf526520b3efcec4, 0x3fefe5dac266846b},
	b: []uint64{0x3fefda4d10f2295c, 0x3fd6f355be841e3a, 0xbfe9b6a7e4375c27},
	c: []uint64{0xbef1ee76fc88b965, 0xbeeed1f8cd4ffde1, 0x3efbe535dbb02f41},
	d: []uint64{0x4000447752ad6536},
}

// TestReduceResolvedGapDiscreteGlobalCountLiteral checks the gap path of
// a discrete model on discreteGapLiteral: on arm64, where it was frozen,
// the gap is resolved, the staircase cut passes frequencyAxisCutAcrossGap
// and Reduce keeps it; on every arch a cut Reduce accepts is within its
// contract.
func TestReduceResolvedGapDiscreteGlobalCountLiteral(t *testing.T) {
	sys := discreteGapLiteral.system(t)
	tol := math.Sqrt(eps())
	res, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if res.Order < 3 {
		if c, at := contractSweepAtPoles(sys, res.Sys, tol); c > 1 {
			t.Errorf("order %d off by %.3g·tol·max(scale, terms) at %v", res.Order, c, at)
		}
	}
	if runtime.GOARCH != "arm64" {
		return
	}
	_, cutoff, ok := resolvedGap(sys.A)
	if !ok || !(cutoff > 0) {
		t.Fatalf("no resolved gap (cutoff %g)", cutoff)
	}
	cut, removed, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), &ReduceOpts{}, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cut.Order != 2 || !frequencyAxisCutAcrossGap(sys, cut.Sys, removed, tol, cutoff) {
		t.Errorf("staircase order %d not accepted across the gap", cut.Order)
	}
	if res.Order != 2 {
		t.Errorf("order %d, want 2", res.Order)
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

// TestReduceWindowCapAtOtherPole checks that a wrong cut whose window
// would reach another model pole is rejected (N3P7NO). The model is upper
// triangular, poles −1, −2 and −2⁴⁰, with the −2 mode carrying a residue
// of 5e-5; the cut keeps −1 and −2⁴⁰ in a Hadamard-rotated (exact)
// realization and drops −2, so it is wrong by 5e-5/|s+2| exactly,
// 2500·tol at s = 2j. The rotation smears the fast scale into the cut's −1:
// β_R ≈ ε·2⁴⁰ ≈ 2.4e-4 ≤ ρ₀, so under the normwise key an uncapped window
// of radius β_R/tol ≈ 1.6e4 covered −2 and every slow probe, and only the
// cap at the distance to −2 left −2's probes to the model-determined rule.
// Under the per-pole key (7NNE5Y) the model's −1 is resolved to
// 2n·β_G ≈ 1e-15 ≪ β_R, so no window forms about the cut's −1 at all and
// the cap is not reached; TestReduceWindowCapAtExactIntegrator is the cap's
// live regression.
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
		for k, c := range win.centre {
			if c == lr {
				t.Errorf("window about %v, radius %g (β_R %g)", lr, win.radius[k], betaR[i])
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
