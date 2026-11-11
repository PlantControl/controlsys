package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hadamardSimilarity returns T·A·T⁻¹, T·B, C·T⁻¹ for T the identity with
// states i and j replaced by the exact Hadamard block [1 1; 1 −1], whose
// inverse is half of it: a dense rotation that rounds nothing in binary.
func hadamardSimilarity(A, B, C *mat.Dense, i, j int) (*mat.Dense, *mat.Dense, *mat.Dense) {
	n, _ := A.Dims()
	T := mat.NewDense(n, n, nil)
	Ti := mat.NewDense(n, n, nil)
	for k := range n {
		T.Set(k, k, 1)
		Ti.Set(k, k, 1)
	}
	T.Set(i, j, 1)
	T.Set(j, i, 1)
	T.Set(j, j, -1)
	Ti.Set(i, i, 0.5)
	Ti.Set(i, j, 0.5)
	Ti.Set(j, i, 0.5)
	Ti.Set(j, j, -0.5)
	var tmp, Ar, Br, Cr mat.Dense
	tmp.Mul(T, A)
	Ar.Mul(&tmp, Ti)
	Br.Mul(T, B)
	Cr.Mul(C, Ti)
	return &Ar, &Br, &Cr
}

func newTestSystem(t *testing.T, A, B, C *mat.Dense) *System {
	t.Helper()
	_, m := B.Dims()
	p, _ := C.Dims()
	sys, err := New(A, B, C, mat.NewDense(p, m, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestReduceSharedPoleWindowKeyedPerPole checks that sharedPoleWindows
// keys a window on the model pole's own resolution ρ_j = min(ρ₀, 2n·β_G)
// (7NNE5Y), not on the normwise ρ₀ = n·ε·‖A_bal‖_F: a cut that smears a
// graded slow pole by its dense norm gets no window, and the band's
// model-determined rule sees it at the pole's own probes.
//
// The twins are the triangular nse folds 33 and 1518 (nearSingularEFoldOf
// with rotated false, frozen), a 1e15 fast state beside two slow poles.
// The staircase cut (order 2 of 3) drops a slow state carrying O(1) of the
// response. On main its slow pole, smeared to β_R ≈ 0.2 (twin 33: −0.128
// against the model's −0.59, β_G = 1.3e-16, ρ₀ = 0.55) took a window of
// radius ~1e7 that made every slow probe uninformative, and the band
// accepted the cut; Reduce kept order 3 only through the per-side count.
// On twin 33 the window goes through the per-pole shares slack, not the
// key.
func TestReduceSharedPoleWindowKeyedPerPole(t *testing.T) {
	tol := math.Sqrt(eps())
	twins := []struct {
		name    string
		a, b, c []float64
		d       []float64
		m       int
	}{
		{
			name: "twin 33",
			a: []float64{
				-0.5900132298553701, 1.0949153124838162, 1.3380268983666948,
				0, -0.5556053658333376, -1.1298696741814158,
				0, 0, -8.236048053224104e+14,
			},
			b: []float64{
				-0.0251550546549375, -0.31110737241205877,
				-1.903266169707275, -0.8638239394267936,
				2.234184957097801e+14, 1.7991592601062482e+15,
			},
			c: []float64{0.6773181193147748, 1.4761793755789903, -1.4767896173593071},
			d: []float64{0.6021407714065655, 0.7576645574259762},
			m: 2,
		},
		{
			name: "twin 1518",
			a: []float64{
				-0.46151774430690434, -3.193852291631417, -1.7932399062339872,
				0, -0.9901461065026029, 1.7003882643434352,
				0, 0, -8.937431033222542e+14,
			},
			b: []float64{1.2257981226499426, 1.0929397008316404, 1.0946488373653068e+15},
			c: []float64{-0.11527199200333871, 0.4463352748349899, -0.45645336629698585},
			d: []float64{-0.2945495813080551},
			m: 1,
		},
	}
	for _, tc := range twins {
		sys, err := New(mat.NewDense(3, 3, tc.a), mat.NewDense(3, tc.m, tc.b),
			mat.NewDense(1, 3, tc.c), mat.NewDense(1, tc.m, tc.d), 0)
		if err != nil {
			t.Fatal(err)
		}
		opts := &ReduceOpts{Mode: ReduceUnobservable}
		base, removed, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), opts, true, -1)
		if err != nil {
			t.Fatal(err)
		}
		if base.Order != 2 {
			t.Fatalf("%s: staircase order %d, want 2", tc.name, base.Order)
		}
		_, poles, ok := axisProbes(sys, removed)
		if !ok {
			t.Fatal("eigenvalues did not converge")
		}
		win, ok := sharedPoleWindows(poles, base.Sys, tol)
		if !ok {
			t.Fatal("eigenvalues of the cut did not converge")
		}
		for k, c := range win.centre {
			if cmplx.Abs(c) < 10 {
				t.Errorf("%s: window about slow cut pole %v, radius %g", tc.name, c, win.radius[k])
			}
		}
		if frequencyAxisCutInBand(sys, base.Sys, removed, tol) {
			t.Errorf("%s: band accepts the cut that drops a slow state", tc.name)
		}
	}

	// Synthetic twin: upper triangular, slow block [−0.2 0.7; 0 −0.3] 0.1
	// apart (inside each other's ρ₀ = 0.75 at 2⁵⁰, so the cap cannot act),
	// fast state −2⁵⁰. The cut keeps states 0 and 2, Hadamard-rotated: its
	// slow pole −0.375 has β_R = 0.25 ≤ ρ₀ but ≫ 2n·β_G ≈ 1e-15, a window
	// of radius 1.7e7 on main, and the probe 2.5j was uninformative there.
	fast := math.Ldexp(1, 50)
	sys := newTestSystem(t,
		mat.NewDense(3, 3, []float64{-0.2, 0.7, 0.5, 0, -0.3, -1.25, 0, 0, -fast}),
		mat.NewDense(3, 1, []float64{1, 1.2505, fast}),
		mat.NewDense(1, 3, []float64{1, 1, 1}))
	Ar, Br, Cr := hadamardSimilarity(
		mat.NewDense(2, 2, []float64{-0.2, 0.5, 0, -fast}),
		mat.NewDense(2, 1, []float64{1, fast}),
		mat.NewDense(1, 2, []float64{1, 1}), 0, 1)
	red := newTestSystem(t, Ar, Br, Cr)
	pts, poles, ok := axisProbes(sys, nil)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		t.Fatal("eigenvalues of the cut did not converge")
	}
	found := false
	for i, lr := range evR {
		if cmplx.Abs(lr) > 1 {
			continue
		}
		found = true
		if !(betaR[i] <= poles.rho0) {
			t.Fatalf("cut pole %v: β_R %g > ρ₀ %g, the normwise key would refuse it too", lr, betaR[i], poles.rho0)
		}
		for j, lg := range poles.evals {
			if cmplx.Abs(lg) < 1 && !(betaR[i] > 6*poles.beta[j]) {
				t.Fatalf("cut pole %v: β_R %g not above 2n·β_G %g of %v", lr, betaR[i], 6*poles.beta[j], lg)
			}
		}
	}
	if !found {
		t.Fatalf("no slow cut pole: %v", evR)
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	for k, c := range win.centre {
		if math.Abs(real(c)) < 1 {
			t.Errorf("window about %v, radius %g", c, win.radius[k])
		}
	}
	k := -1
	for i, s := range pts {
		if cmplx.Abs(s-2.5i) < 1e-9 {
			k = i
		}
	}
	if k < 0 {
		t.Fatalf("no probe at 2.5j: %v", pts)
	}
	if win.contain(pts[k]) {
		t.Errorf("probe %v inside a window", pts[k])
	}
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	if v := classifyProbe(full, cut, k, tol, win.contain(pts[k])); v != probeReject {
		t.Errorf("probe %v: verdict %d, want reject", pts[k], v)
	}
	if frequencyAxisCutInBand(sys, red, nil, tol) {
		t.Error("band accepts the rotated cut that drops −0.3")
	}
}

// TestReduceWindowCapAtExactIntegrator checks the window cap of
// sharedPoleWindows where the per-pole key still leaves a window: an
// unresolved pole keeps the normwise floor ρ₀ (7NNE5Y). The model is an
// exact integrator (zero column: β_G = 0) beside −1, with a 1e-4 residue,
// and −2⁵⁰; the cut keeps the integrator, with −1's DC gain folded into
// B, and the fast state, Hadamard-rotated. Its integrator comes out at
// ≈ −0.125 with β_R = 0.25 ≤ ρ₀ = 0.75, an uncapped radius of 1.7e7 that
// would make −1's probes uninformative; capped at the distance to −1 it
// leaves the corner 1j, where the cut is wrong by 3700·tol, to reject. The
// result is the same on main.
func TestReduceWindowCapAtExactIntegrator(t *testing.T) {
	const d = 1e-4
	fast := math.Ldexp(1, 50)
	sys := newTestSystem(t,
		mat.NewDense(3, 3, []float64{0, 0.5, 0, 0, -1, 0, 0, 0, -fast}),
		mat.NewDense(3, 1, []float64{1, d, fast}),
		mat.NewDense(1, 3, []float64{1, 1, 1}))
	Ar, Br, Cr := hadamardSimilarity(
		mat.NewDense(2, 2, []float64{0, 0, 0, -fast}),
		mat.NewDense(2, 1, []float64{1 + 0.5*d, fast}),
		mat.NewDense(1, 2, []float64{1, 1}), 0, 1)
	red := newTestSystem(t, Ar, Br, Cr)
	tol := math.Sqrt(eps())
	_, poles, ok := axisProbes(sys, nil)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		t.Fatal("eigenvalues of the cut did not converge")
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	found := false
	for i, lr := range evR {
		if cmplx.Abs(lr) > 0.5 {
			continue
		}
		if !(betaR[i] <= poles.rho0 && betaR[i]/tol > 1) {
			t.Fatalf("cut integrator %v: β_R %g, ρ₀ %g: uncapped window would not reach −1", lr, betaR[i], poles.rho0)
		}
		for k, c := range win.centre {
			if c != lr {
				continue
			}
			found = true
			if win.radius[k] > cmplx.Abs(lr+1) {
				t.Errorf("window about %v, radius %g, reaches the pole −1", lr, win.radius[k])
			}
		}
	}
	if !found {
		t.Fatalf("no window about the cut's integrator: cut poles %v, windows %v", evR, win.centre)
	}
	if win.contain(1i) {
		t.Errorf("probe 1j inside a window: radii %v", win.radius)
	}
	if frequencyAxisCutInBand(sys, red, nil, tol) {
		t.Error("cut that drops the −1 mode accepted")
	}
	for _, tc := range []struct {
		opts *ReduceOpts
		want int
	}{{nil, 3}, {&ReduceOpts{Tol: 1e-4}, 2}} {
		res, err := sys.Reduce(tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != tc.want {
			t.Errorf("Reduce(%+v): order %d, want %d", tc.opts, res.Order, tc.want)
		}
	}
}

// TestReduceTolKeepsSlowProbesOfGradedPoles checks P5's per-pole drop
// radius n·ε·|s| + ρ_j (7NNE5Y flag 2): the model of
// TestReduceResolvedGapSplitAccepted with its slow block scaled by 0.1
// (poles −0.1, −0.2, −2⁵⁰, exact minimal order 2). The normwise radius
// ρ₀ = 0.75 dropped the probes 0, 0.1j, 0.2j and 0.25j beside the graded
// slow poles, which the data resolve to 1e-16; at Tol = 1e-2 the slow
// limit ε‖A_bal‖_F/tol = 25 then left the slow side only 2.5j against a
// count of 4, so on main Reduce returned order 3 at Tol 1e-2 and 3e-2 (2
// at the default). Kept, they verify the split.
func TestReduceTolKeepsSlowProbesOfGradedPoles(t *testing.T) {
	fast := math.Ldexp(1, 50)
	Ar, Br, Cr := hadamardSimilarity(
		mat.NewDense(3, 3, []float64{-0.1, 0.05, 0.75, 0, -0.2, -1.25, 0, 0, -fast}),
		mat.NewDense(3, 1, []float64{1, 1.5, fast}),
		mat.NewDense(1, 3, []float64{0, 1, 1}), 0, 1)
	sys := newTestSystem(t, Ar, Br, Cr)
	if got := exactMinimalOrder(sys); got != 2 {
		t.Fatalf("exact minimal order %d, want 2", got)
	}
	pts, _, ok := axisProbes(sys, nil)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	for _, want := range []complex128{0, 0.1i, 0.2i} {
		found := false
		for _, s := range pts {
			if cmplx.Abs(s-want) < 1e-12 {
				found = true
			}
		}
		if !found {
			t.Errorf("probe %v dropped: %v", want, pts)
		}
	}
	for _, tl := range []float64{0, 1e-2, 3e-2} {
		res, err := sys.Reduce(&ReduceOpts{Tol: tl})
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 2 {
			t.Errorf("Tol %g: order %d, want 2", tl, res.Order)
		}
	}
}

// TestReduceSharedPoleWindowNeedsEverySharedPole checks the all-shared
// match of sharedPoleWindows (7NNE5Y): a window forms only when every
// model pole the cut pole shares admits it. The model is a structural
// integrator beside a graded slow pair −0.2, −0.3 and −2⁵⁰ (ρ₀ = 1), in
// both eigenvalue orders; the cut drops −0.3 and Hadamard-rotates
// (−0.2, −2⁵⁰). Its slow pole (−0.375 or −0.125, β_R = 0.25) shares both
// −0.2, whose key fails by sixteen orders, and the integrator, whose
// floor ρ₀ admits it. The first-match rule let it borrow the integrator's
// floor for a window of radius 1.7e7, uncapped since −0.2 lies in 0's
// ρ₀-cluster. Reduce keeps order 4 either way, through the per-side count.
func TestReduceSharedPoleWindowNeedsEverySharedPole(t *testing.T) {
	fast := math.Ldexp(1, 50)
	tol := math.Sqrt(eps())
	for _, tc := range []struct {
		name       string
		a, b       []float64
		ac, bc, cc []float64
		i, j       int
	}{
		{
			name: "integrator first",
			a:    []float64{0, 0.4, 0.3, 0, 0, -0.2, 0.7, 0.5, 0, 0, -0.3, -1.25, 0, 0, 0, -fast},
			b:    []float64{1, 1, 1.2505, fast},
			ac:   []float64{0, 0.4, 0, 0, -0.2, 0.5, 0, 0, -fast},
			bc:   []float64{1, 1, fast},
			i:    1, j: 2,
		},
		{
			name: "integrator last",
			a:    []float64{-0.2, 0.7, 0.5, 0, 0, -0.3, -1.25, 0, 0, 0, -fast, 0, 0.4, 0.3, 0, 0},
			b:    []float64{1, 1.2505, fast, 1},
			ac:   []float64{-0.2, 0.5, 0, 0, -fast, 0, 0.4, 0, 0},
			bc:   []float64{1, fast, 1},
			i:    0, j: 1,
		},
	} {
		sys := newTestSystem(t, mat.NewDense(4, 4, tc.a), mat.NewDense(4, 1, tc.b),
			mat.NewDense(1, 4, []float64{1, 1, 1, 1}))
		Ar, Br, Cr := hadamardSimilarity(mat.NewDense(3, 3, tc.ac), mat.NewDense(3, 1, tc.bc),
			mat.NewDense(1, 3, []float64{1, 1, 1}), tc.i, tc.j)
		red := newTestSystem(t, Ar, Br, Cr)
		_, poles, ok := axisProbes(sys, nil)
		if !ok {
			t.Fatal("eigenvalues did not converge")
		}
		win, ok := sharedPoleWindows(poles, red, tol)
		if !ok {
			t.Fatal("eigenvalues of the cut did not converge")
		}
		for k, c := range win.centre {
			if math.Abs(real(c)) < 1 && win.radius[k] != 0 {
				t.Errorf("%s: window about %v, radius %g", tc.name, c, win.radius[k])
			}
		}
		if frequencyAxisCutInBand(sys, red, nil, tol) {
			t.Errorf("%s: band accepts the cut that drops −0.3", tc.name)
		}
		res, err := sys.Reduce(nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 4 {
			t.Errorf("%s: order %d, want 4", tc.name, res.Order)
		}
	}
}
