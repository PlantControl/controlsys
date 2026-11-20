package controlsys

import (
	"math"
	"math/cmplx"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// keptAxisPoleLiterals are axis-kept census models (a controllable and
// observable mode on the axis beside stable dynamics and one hidden stable
// mode, rotated by a random orthogonal similarity; rcAxisKept) whose Reduce
// cut main accepted while it was off by more than tol just outside the
// window of its kept axis pole, radius max(β_G, β_R)/tol (SSOTWL), frozen
// per arch. 16-unc (arm64, a pair at e^{±j2.83}) needs 1.47·(β_G + β_R)/tol,
// within the window's 2(β_G + β_R)/tol, its cut pole 3.5e-15 from the
// model's: accepted. The others are refused: 417-unc, an integrator pair
// at ±0.0206j, needs 3.1·(β_G + β_R)/tol (arm64) and 5.4· (amd64), its cut
// pole further from the model's than rounding explains; 494-unobs (arm64,
// z = 1) and 64-unc (amd64, z = −1) lie 1.04 and 1.13 times beyond the
// window's first-order bound.
var keptAxisPoleLiterals = map[string]bandLiteral{
	"axis-kept/16-unc@arm64": {
		n: 5, m: 1, p: 1, dt: 0.1,
		a: []uint64{0x3ff53c412058edff, 0xbfc7202dc55f85b0, 0x3fdfdb6c2d6dfcc4, 0x3feb01c86875afbf, 0xbfd69a05cdfe385c, 0x3fd5ab003c6ba644, 0x3fd9e5f98575d3e1, 0x3fbc4f5017f9465a, 0x3fc3c7c5f33de6fa, 0xbfb9ca96bfeac178, 0xbfe875c2bcd7a7ee, 0x3fc78fb777a6e01a, 0xbff0c332fe40e04c, 0xbfb67efb69ddd0c7, 0xbfc0b5fe4d3c6e30, 0xbfe89130049b5484, 0x3fe02a685d5dcb5b, 0xbfd89e4b5f37e9d5, 0xbfeef57ae6c96416, 0x3fd9dd34fd7e1316, 0xbf9ab8d915d0de86, 0x3fe09a49f6d76b2d, 0x3f870adde5729f9d, 0x3ff05ebad0395569, 0x3fbec9191435c87d},
		b: []uint64{0xbfdd34141ebd02c7, 0xbfe65ee098592e01, 0xbfb0c9df3051c35b, 0xbfd46eb583d02e60, 0xbfe9be0853538550},
		c: []uint64{0xbff3760b6212fdec, 0xbfc4963660f18d45, 0xbff6f05362106a93, 0xbff852f50ac2c8fa, 0xbfe60dff2c5d4402},
		d: []uint64{0x0},
	},
	"axis-kept/417-unc@arm64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xbff740ed36b79895, 0x3fd49f2dc1176e86, 0xbfc24876e9eb6e69, 0x3fbd9b236c2c4a8c, 0xbfe518b6467b720d, 0xbfc61f6abe4e4f19, 0xbfd59e24c190410c, 0x3fdc9819df1c7a98, 0xbfcb289d9e5c9c38, 0xbfcfbe41a958b163, 0xbfd184f026b6d6b7, 0x3fdace1c71ba818c, 0x3fd77806dbc15006, 0x3fb6fe9d2010df16, 0x3fc3e86cd9eadb4c, 0xbfcdd24ae02c1768},
		b: []uint64{0x3fd24bdcee73b1b8, 0x3ff08a9cb05e6a96, 0x3fd18c49b246cdcc, 0x3ff10962c7e8f6cc},
		c: []uint64{0x3fbd42325b1454c3, 0x3fe358a75d7b6e18, 0x3ff16e0b2516680e, 0x3ff3c5eb2c168fa8},
		d: []uint64{0x0},
	},
	"axis-kept/494-unobs@arm64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0x3fede3f5e665a1fc, 0x3fa948e5eeedb60e, 0x3fb8029c364b4733, 0xbfa1fb0dab83e54b, 0xbfed61e3858914ab, 0xbfd6a9f0d559dd82, 0x3fd690a52a9d9083, 0x3ff206c237d261fd, 0x3fe93255945cd2f8},
		b: []uint64{0xbfe9bae9df96ce38, 0xbfef5ddc80dae855, 0xbffe09a2f3e91721},
		c: []uint64{0xbfe0321462211838, 0xbfd01bd463426b1e, 0xbfd4ebdc596f908b},
		d: []uint64{0x0},
	},
	"axis-kept/64-unc@amd64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0xbfd089452bba5caf, 0xbfc6c67ef2609c96, 0x3fd09f663a714b3b, 0xc000ef15c9a364e0, 0xbfe0ad5b34bd915c, 0xbff175ff328fe1cf, 0x3fe1821072bc0794, 0xbfabef59d27d30ee, 0x3fbde6901d04d954},
		b: []uint64{0x3ffdb0cb2cb4d0f4, 0xbfb91e98a693d3e0, 0xc000bdcae52dcd28},
		c: []uint64{0x3fa5b02fe40f1fa0, 0xbfe425a43be8d669, 0x40015d2fd226e901},
		d: []uint64{0x0},
	},
	"axis-kept/417-unc@amd64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xbff740ed36b7988d, 0x3fd49f2dc1176e74, 0xbfc24876e9eb6e4b, 0x3fbd9b236c2c4ae3, 0xbfe518b6467b720e, 0xbfc61f6abe4e4f21, 0xbfd59e24c190410c, 0x3fdc9819df1c7aa0, 0xbfcb289d9e5c9c27, 0xbfcfbe41a958b16a, 0xbfd184f026b6d6ba, 0x3fdace1c71ba818c, 0x3fd77806dbc15012, 0x3fb6fe9d2010df0a, 0x3fc3e86cd9eadb49, 0xbfcdd24ae02c176c},
		b: []uint64{0x3fd24bdcee73b1b8, 0x3ff08a9cb05e6a96, 0x3fd18c49b246cdc7, 0x3ff10962c7e8f6cb},
		c: []uint64{0x3fbd42325b1454ba, 0x3fe358a75d7b6e1a, 0x3ff16e0b2516680a, 0x3ff3c5eb2c168fa6},
		d: []uint64{0x0},
	},
}

// contractOutsideAxisWindows is the largest exact |G−R|/(tol·max(scale,
// terms)) of sys against red, with the point, over contractPolesGrid and
// the axis points at 1+2⁻²⁰, 1.1, 2 and 10 times each axis window's radius
// from its centre (sharedPoleWindows), leaving out the points inside an
// axis window, where Reduce certifies nothing.
func contractOutsideAxisWindows(t *testing.T, sys, red *System, tol float64) (float64, complex128) {
	t.Helper()
	pts, ok := contractPolesGrid(sys, red)
	_, mp, okG := axisProbes(sys, nil)
	if !ok || !okG {
		t.Fatal("eigenvalues did not converge")
	}
	win, ok := sharedPoleWindows(mp, red, tol)
	if !ok {
		t.Fatal("cut eigenvalues did not converge")
	}
	for i, c := range win.centre {
		if !win.axisPole[i] {
			continue
		}
		for _, f := range []float64{1 + 0x1p-20, 1.1, 2, 10} {
			r := f * win.radius[i]
			if sys.Dt != 0 {
				pts = append(pts, cmplx.Rect(1, cmplx.Phase(c)+r), cmplx.Rect(1, cmplx.Phase(c)-r))
			} else {
				pts = append(pts, complex(0, imag(c)+r), complex(0, imag(c)-r))
			}
		}
	}
	pts = slices.DeleteFunc(pts, func(s complex128) bool {
		for i, c := range win.centre {
			if win.axisPole[i] && cmplx.Abs(s-c) <= win.radius[i] {
				return true
			}
		}
		return false
	})
	return contractWorst(sys, red, pts, tol)
}

// axisWindowRadiusExcess returns the largest radius·tol/(2(β_G + β_R)) of
// the axis windows of red against sys, β_G the largest rounding bound of
// the model poles the window shares (sharedPoleWindows' share test): at
// most 1, since rounding explains a pole distance of at most 2(β_G + β_R)
// and a computed distance beyond β_G + β_R widens no window.
func axisWindowRadiusExcess(t *testing.T, sys, red *System, tol float64) float64 {
	t.Helper()
	_, mp, ok := axisProbes(sys, nil)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	win, ok := sharedPoleWindows(mp, red, tol)
	evR, betaR, okR := eigenRoundingBounds(red.A)
	if !ok || !okR {
		t.Fatal("cut eigenvalues did not converge")
	}
	nr, _, _ := red.Dims()
	rho0R := 0.0
	for _, a := range newBalancedRealization(red, nr, red.B.RawMatrix().Cols, red.C.RawMatrix().Rows).a {
		rho0R = math.Hypot(rho0R, a)
	}
	rho0R *= float64(nr) * eps()
	worst := 0.0
	for i, c := range win.centre {
		if !win.axisPole[i] {
			continue
		}
		k := slices.Index(evR, c)
		rR := math.Min(rho0R, 2*float64(nr)*betaR[k])
		bG := 0.0
		for j, lg := range mp.evals {
			if cmplx.Abs(c-lg) <= betaR[k]+mp.beta[j]+mp.rho[j]+rR {
				bG = math.Max(bG, mp.beta[j])
			}
		}
		worst = math.Max(worst, win.radius[i]*tol/(2*(bG+betaR[k])))
	}
	return worst
}

// TestReduceCutWithinContractOutsideKeptAxisPoleWindow checks that every
// cut Reduce accepts of keptAxisPoleLiterals, in every mode, is within its
// contract outside the windows of its kept axis poles, at their edges and
// beyond: each window covers every point where rounding of the pole, β_G +
// β_R + min(|λ_R − λ_G|, β_G + β_R) with radius at most 2(β_G + β_R)/tol,
// explains the cut's deviation, and the cut is judged at its edge. On
// arm64 the 16-unc cut, which needs a window wider than (β_G + β_R)/tol,
// is accepted.
func TestReduceCutWithinContractOutsideKeptAxisPoleWindow(t *testing.T) {
	tol := math.Sqrt(eps())
	names := make([]string, 0, len(keptAxisPoleLiterals))
	for k := range keptAxisPoleLiterals {
		names = append(names, k)
	}
	slices.Sort(names)
	for _, name := range names {
		sys := keptAxisPoleLiterals[name].system(t)
		n, _, _ := sys.Dims()
		for _, opts := range []*ReduceOpts{{}, {Equalize: true}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}} {
			res, err := sys.Reduce(opts)
			if err != nil {
				t.Fatalf("%s %+v: %v", name, *opts, err)
			}
			if name == "axis-kept/16-unc@arm64" && runtime.GOARCH == "arm64" && opts.Mode != ReduceUnobservable && res.Order == n {
				t.Errorf("%s %+v: order %d, want a cut", name, *opts, res.Order)
			}
			if res.Order == n {
				continue
			}
			if c, at := contractOutsideAxisWindows(t, sys, res.Sys, tol); c > 1 {
				t.Errorf("%s %+v: order %d off by %.3g·tol·max(scale, terms) at %v outside its axis windows", name, *opts, res.Order, c, at)
			}
			if x := axisWindowRadiusExcess(t, sys, res.Sys, tol); x > 1 {
				t.Errorf("%s %+v: axis window radius %.3g·2(β_G + β_R)/tol", name, *opts, x)
			}
		}
	}
}

// TestReduceRefusesCutJustOutsideAxisPoleWindow checks the refused
// literals of keptAxisPoleLiterals on the arch each was frozen on: where
// the staircase cut main accepted is off by more than tol just outside the
// window of its kept axis pole, the band refuses it, judging the window's
// edge (axisWindowEdges), and Reduce keeps the full order. 417-unc, off by
// 5.2·tol (arm64) and 8.7·tol (amd64) there, must be outside; the edge
// literals, 1.04 and 1.13 times beyond the window's bound under Rosetta and
// arm64, may land within it on real x86 with FMA, and then nothing is
// asserted.
func TestReduceRefusesCutJustOutsideAxisPoleWindow(t *testing.T) {
	tol := math.Sqrt(eps())
	for _, tc := range []struct {
		name string
		opts *ReduceOpts
		must bool
	}{
		{"axis-kept/417-unc@arm64", &ReduceOpts{}, true},
		{"axis-kept/417-unc@arm64", &ReduceOpts{Equalize: true}, true},
		{"axis-kept/417-unc@amd64", &ReduceOpts{}, true},
		{"axis-kept/417-unc@amd64", &ReduceOpts{Equalize: true}, true},
		{"axis-kept/494-unobs@arm64", &ReduceOpts{}, false},
		{"axis-kept/64-unc@amd64", &ReduceOpts{Equalize: true}, false},
	} {
		if _, arch, _ := strings.Cut(tc.name, "@"); arch != runtime.GOARCH {
			continue
		}
		sys := keptAxisPoleLiterals[tc.name].system(t)
		n, _, _ := sys.Dims()
		cut, _, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), tc.opts, true, -1)
		if err != nil || cut.Order == n {
			if tc.must {
				t.Errorf("%s %+v: no staircase cut (%v)", tc.name, *tc.opts, err)
			}
			continue
		}
		if c, at := contractOutsideAxisWindows(t, sys, cut.Sys, tol); !(c > 1) {
			if tc.must {
				t.Errorf("%s %+v: staircase cut within contract outside its axis windows (%.3g at %v)", tc.name, *tc.opts, c, at)
			}
			continue
		}
		if frequencyAxisCutInBand(sys, cut.Sys, nil, tol) {
			t.Errorf("%s %+v: band accepts the staircase cut", tc.name, *tc.opts)
		}
		res, err := sys.Reduce(tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != n {
			t.Errorf("%s %+v: order %d, want %d", tc.name, *tc.opts, res.Order, n)
		}
	}
}
