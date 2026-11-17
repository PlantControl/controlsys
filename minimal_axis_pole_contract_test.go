package controlsys

import (
	"math"
	"math/cmplx"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// contractSweepAtPoles is the largest exact |G−R|/(tol·max(scale, terms))
// of the stored model sys against red (Reduce's contract), with the point,
// over contractSweep's grid, inside the shared-pole windows too, plus every
// eigenvalue's nearest axis point and points beside it down to 1e-14 of
// its scale: the points beside a pole within rounding of the axis, which
// axisProbes drops.
func contractSweepAtPoles(sys, red *System, tol float64) (float64, complex128) {
	_, poles, ok := axisProbes(sys, nil)
	evR, _, okR := eigenRoundingBounds(red.A)
	if !ok || !okR {
		return math.Inf(1), 0
	}
	evs := append(slices.Clone(poles.evals), evR...)
	var pts []complex128
	if sys.Dt != 0 {
		for e := -16.0; e < math.Log10(math.Pi); e += 0.125 {
			pts = append(pts, cmplx.Rect(1, math.Pow(10, e)))
		}
		pts = append(pts, 1, -1)
		for _, ev := range evs {
			if r := cmplx.Abs(ev); r > 0 {
				pts = append(pts, ev/complex(r, 0))
			}
			th := cmplx.Phase(ev)
			for e := -14.0; e <= 0; e += 0.5 {
				d := math.Pow(10, e)
				pts = append(pts, cmplx.Rect(1, th+d), cmplx.Rect(1, th-d))
			}
		}
	} else {
		hi := 1.0
		for _, ev := range poles.evals {
			hi = math.Max(hi, cmplx.Abs(ev))
		}
		for e := -20.0; e <= 3; e += 0.125 {
			pts = append(pts, complex(0, hi*math.Pow(10, e)))
		}
		pts = append(pts, 0)
		for _, ev := range evs {
			pts = append(pts, complex(0, imag(ev)))
			a := math.Max(cmplx.Abs(ev), 1e-300)
			for e := -14.0; e <= 0; e += 0.5 {
				d := a * math.Pow(10, e)
				pts = append(pts, complex(0, imag(ev)+d), complex(0, imag(ev)-d))
			}
		}
	}
	worst, at := 0.0, complex128(0)
	for _, s := range pts {
		g, r, tg := oracleStrictlyProper(sys, s), oracleStrictlyProper(red, s), oracleTerms(sys, s)
		if g == nil || r == nil || tg == nil {
			continue
		}
		scale := 0.0
		for i := range g {
			scale = math.Max(scale, math.Max(oracleDist(g[i], oracleOf(0)), oracleDist(r[i], oracleOf(0))))
		}
		for i := range g {
			if d := oracleDist(g[i], r[i]) / (tol * math.Max(scale, tg[i])); d > worst {
				worst, at = d, s
			}
		}
	}
	return worst, at
}

// axisPoleLiterals are stored models whose Reduce cut main accepted while
// it was off at a pole within rounding of the axis or inside a shared-pole
// window (HO6UP6), frozen per arch where the generator's rounding differs:
// the axis-hid census family (a mode on the axis left uncontrollable by a
// random orthogonal similarity: ±19.6j, z = −1, s = 0, z = 1), the
// Householder-rotated discrete removed integrator of
// TestReduceAcceptsCutBesideExactAxisPole, and fz 6359 (amd64; a pole
// 3e-4 inside z = 1 with β 1.1e-8 that the cut moves by 4.3e-9, 919·tol at
// z = 1 inside its window of radius 0.74) and fz 4265 (arm64; 1.07·tol
// inside a window).
var axisPoleLiterals = map[string]bandLiteral{
	"axis-hid/1-unc@arm64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xbff2050ff686e042, 0x4003067ca853000b, 0x4029fd50896202c1, 0x400421260df197f9, 0xbffd4e90964f7857, 0x3fca3823b77ef445, 0xc029ed2ec0bfbe95, 0xc0121e62c8a06ddb, 0xc028166753d4b068, 0x402a95689af1358b, 0x3fd5e42e8f9048a3, 0xc01b1cde415311d4, 0xbff1de00a996f5bb, 0x4006adbe268476b6, 0x4015e8d4d401dc13, 0xbff687584ce91369},
		b: []uint64{0xbfcb7dfbda19ac41, 0xbfe948580b02155d, 0x3fd5b0a5a51cd344, 0xbff175d8b54840ca},
		c: []uint64{0x3fe19c14e8d5ed9d, 0xbfc229744638cfd9, 0xbff3d5a94b51e65e, 0x3fe4c7b700733db7},
		d: []uint64{0x0},
	},
	"axis-hid/28-unc@arm64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0xbff372227543b993, 0x3fca4a1ebf2c037a, 0x3fd707eed90ca2a4, 0xbfe3ee146f6cf79c, 0xbfe39887d7df5d47, 0x3fc0d560e09bb73d, 0x3fbdbf86f97bbfc1, 0x3fbfba0ae8e15705, 0xbfc472ad478e56dd},
		b: []uint64{0xbfe01baf595bed25, 0xbff60ac610b33230, 0x3fc764a061ddfac2},
		c: []uint64{0x3ff30ab88c95907d, 0x3fb149be78c76cb1, 0xbfdeaff06d26db81},
		d: []uint64{0x0},
	},
	"axis-hid/101-unc@arm64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xc00121716017bb9b, 0x3fdafbb28734b057, 0x3fe8134072999cdc, 0xbfd771ac5ae61a4d, 0x3fe8a374bcb0094d, 0xbff1fa6f1794a27b, 0x3fee099b25deba5f, 0xbfd00336a4fcabda, 0x3ff3f4eaff46d968, 0xbfd6eecf636a698e, 0xbfe59c202d187985, 0xbfcd1b1a81008ada, 0xbfddcaf45092a694, 0x3fccc1640243cd61, 0xbfe25f185fe9e663, 0xbfe41dc2db506671},
		b: []uint64{0xbfa0b58d5140ba87, 0x3fd3d178b333a1f5, 0x3fd244949831f973, 0x3fd3712c27ebf3f1},
		c: []uint64{0xbfe19d8410ebd6cc, 0x3ff09a934a57d4a6, 0x3fd994e084f8ea25, 0x3ffaaa33043effb6},
		d: []uint64{0x0},
	},
	"axis-hid/245-unc@arm64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0xbfc6e68e57f8e9e8, 0xbfcd41a716f7810f, 0xbfa5e2f717bcacb0, 0xbfcabcd4b1f6b2e3, 0xbfc285e71c6dbd87, 0xbff83a0adb345273, 0x3fc50d5f7450ce1c, 0xbfc21e99721a2b66, 0x3fe89e34d54d9682},
		b: []uint64{0x3ffa5f7540007e1c, 0xbfdaeede5941b4a8, 0xbfd5e240070acac6},
		c: []uint64{0xbfdbe6765373b9f0, 0xbfe086e29dc6453a, 0x40023fe1bf6722e8},
		d: []uint64{0x0},
	},
	"axis-hid/1-unc@amd64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xbff2050ff686e044, 0x4003067ca8530009, 0x4029fd50896202c1, 0x400421260df197fc, 0xbffd4e90964f7850, 0x3fca3823b77ef440, 0xc029ed2ec0bfbe93, 0xc0121e62c8a06ddb, 0xc028166753d4b068, 0x402a95689af1358b, 0x3fd5e42e8f9048a0, 0xc01b1cde415311d5, 0xbff1de00a996f5c0, 0x4006adbe268476b7, 0x4015e8d4d401dc12, 0xbff687584ce9136a},
		b: []uint64{0xbfcb7dfbda19ac40, 0xbfe948580b02155d, 0x3fd5b0a5a51cd344, 0xbff175d8b54840ca},
		c: []uint64{0x3fe19c14e8d5ed9e, 0xbfc229744638cfd5, 0xbff3d5a94b51e65f, 0x3fe4c7b700733db8},
		d: []uint64{0x0},
	},
	"axis-hid/28-unc@amd64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0xbff372227543b994, 0x3fca4a1ebf2c037a, 0x3fd707eed90ca2a3, 0xbfe3ee146f6cf79d, 0xbfe39887d7df5d46, 0x3fc0d560e09bb73d, 0x3fbdbf86f97bbfbc, 0x3fbfba0ae8e15704, 0xbfc472ad478e56dc},
		b: []uint64{0xbfe01baf595bed26, 0xbff60ac610b33230, 0x3fc764a061ddfac1},
		c: []uint64{0x3ff30ab88c95907d, 0x3fb149be78c76cac, 0xbfdeaff06d26db7f},
		d: []uint64{0x0},
	},
	"axis-hid/101-unc@amd64": {
		n: 4, m: 1, p: 1, dt: 0,
		a: []uint64{0xc00121716017bb96, 0x3fdafbb28734b049, 0x3fe8134072999cd2, 0xbfd771ac5ae61a42, 0x3fe8a374bcb0094b, 0xbff1fa6f1794a27a, 0x3fee099b25deba64, 0xbfd00336a4fcabe4, 0x3ff3f4eaff46d961, 0xbfd6eecf636a6984, 0xbfe59c202d187980, 0xbfcd1b1a81008aea, 0xbfddcaf45092a68a, 0x3fccc1640243cd58, 0xbfe25f185fe9e664, 0xbfe41dc2db50666e},
		b: []uint64{0xbfa0b58d5140ba78, 0x3fd3d178b333a1f5, 0x3fd244949831f972, 0x3fd3712c27ebf3f1},
		c: []uint64{0xbfe19d8410ebd6c8, 0x3ff09a934a57d4a5, 0x3fd994e084f8ea23, 0x3ffaaa33043effb6},
		d: []uint64{0x0},
	},
	"axis-hid/245-unc@amd64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0xbfc6e68e57f8e9e9, 0xbfcd41a716f7810f, 0xbfa5e2f717bcacb5, 0xbfcabcd4b1f6b2e3, 0xbfc285e71c6dbd87, 0xbff83a0adb345273, 0x3fc50d5f7450ce1c, 0xbfc21e99721a2b66, 0x3fe89e34d54d9682},
		b: []uint64{0x3ffa5f7540007e1c, 0xbfdaeede5941b4a8, 0xbfd5e240070acac6},
		c: []uint64{0xbfdbe6765373b9f0, 0xbfe086e29dc6453a, 0x40023fe1bf6722e8},
		d: []uint64{0x0},
	},
	"fz/6359@amd64": {
		n: 6, m: 1, p: 1, dt: 100,
		a: []uint64{0xc13b69e96c6c27fe, 0x412db1087d1d09d5, 0xc13754aae8e7ea1b, 0x41617a8b243a10d2, 0xc151686af4499e3e, 0x416140766bd9b04f, 0xc140999e607f05a3, 0x4132177c0c2a83c1, 0xc13c2ee9330d681f, 0x41652df0253eb3bf, 0xc15517f9782bd165, 0x4164e78f402ade02, 0x41268733b1f6c7f3, 0xc117e01718e510ff, 0x41235750db75f954, 0xc14cac9702adf6cb, 0x413c8edab5931c3d, 0xc14c4d4e91bb1db6, 0x3fd0de23e2c0b9b5, 0xbfc595554145d9da, 0x3fc27cd7053d356a, 0xbff81836b44ed211, 0x3fe7dc2152022c06, 0xbff7a56de4a1305d, 0xbee0dfa4b7189ad7, 0x3ed59741ab27a69f, 0xbed27e7ccec3393f, 0x3f140d47b764d615, 0xbf07b3f3bfa39a22, 0x3f149f91b0f13aa9, 0x3c096266f7b26f6b, 0xbc003d8ed85baa0b, 0x3bfbd27cae8cfccb, 0xbc3e2a6c9b2762be, 0x3c39f3e634eb4ef7, 0xbc43d1d1705479d8},
		b: []uint64{0xbfefed6ea808cfb2, 0xbfd53b13c2a7400b, 0x3fea3f8e3347d7a5, 0x3fe8a87c2ad68cfe, 0xbfda1cc812c5f0d9, 0xbfefa4a8009b8972},
		c: []uint64{0xbf7d69973b202ed1, 0xbf774e1590de839e, 0x3f863d9173277bca, 0xbf55d266e97814b3, 0xbf847a0fd1a47e01, 0x3f7fef6a81f7b31e},
		d: []uint64{0xbfe6dbb4b5944313},
	},
	"fz/4265@arm64": {
		n: 6, m: 3, p: 1, dt: 1.534899915944551,
		a: []uint64{0x401a10cbd8786b41, 0xc003fd03d521dafc, 0xc025198a6a813043, 0xc0353a13a1613f4e, 0x40380b4b017151fe, 0xc048b5af8e42b4d0, 0x4008bb05ce01de5b, 0xc0030c742be38308, 0xc0018d5aa339e158, 0xc0175c6f6cc085c7, 0x401cbcb67b607854, 0xc02c8957e58213bf, 0x4011484c8a9c57f6, 0xbff5cd8d3402edf2, 0xc01e0672c9177e64, 0xc02ea384046bec9c, 0x403173c348402a03, 0xc041b3a0bf7b3cde, 0xbfdc6a649f89e7d2, 0x3fc0af424d35f401, 0x3fe70b83ff5f8a28, 0x3fffcb591cc152e2, 0xc0037bdc379b0b80, 0x4011c7633af01b36, 0x3fd3952ad2ea3708, 0xbfb6ff1953ddc97a, 0xbfdfc3556c4ad5e1, 0xbfe5c7bdb82736fd, 0x3fe6b83c318742bb, 0xbffb70adbc94b755, 0xbac0de7b5544f157, 0x3aa3cf4f915bc720, 0x3acb5c8e9b95f567, 0x3ad2c319376a4a17, 0xbac6affc6c7fce5e, 0x3af3de4fc6877411},
		b: []uint64{0xbfb4f4b2b93cd1a3, 0x3fecda38ef3ea3e5, 0xbfea83395484ec9f, 0x3fee0798aba21b85, 0xbfc6782da98c06a7, 0xbfe85b9d2baefa4f, 0x3fe2af6eedb502b4, 0xbfefdb92a8cfe6a0, 0x3fdaf65ec46bcc22, 0xbfe408788333bb5a, 0xbfd6da55e97399e0, 0x3fef91fd693ae842, 0xbfed672734fb9f3c, 0x3fe9be9d6ead947c, 0x3fbb4548d54e1de9, 0x3fc135d8864bd5fc, 0x3fe93321a12bf0e7, 0xbfedbf1e57fd24e5},
		c: []uint64{0xbeb325419012777b, 0x3eb51b7ddacef359, 0x3e95fde27b66244a, 0xbeb8a9887a0cd8ca, 0x3ea4e574878d3cfe, 0x3eb1e81c2ce317f5},
		d: []uint64{0x3fe29776f658f122, 0xbfe639ff0a226a0d, 0xbffdc490e202aca0},
	},
	"house@arm64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0x3fd7777777777778, 0x3fcddddddddddddc, 0xbfdddddddddddddc, 0x3fd6c16c16c16c16, 0x3fed27d27d27d27d, 0x3fcb05b05b05b059, 0x3fd3e93e93e93e93, 0x3fcc71c71c71c71b, 0x3fea4fa4fa4fa4f9},
		b: []uint64{0x3fd5555555555556, 0xbfd5555555555555, 0xbff5555555555555},
		c: []uint64{0x3fbc71c71c71c720, 0xc00238e38e38e38e, 0x3fcc71c71c71c71c},
		d: []uint64{0x0},
	},
	"house@amd64": {
		n: 3, m: 1, p: 1, dt: 0.1,
		a: []uint64{0x3fd7777777777777, 0x3fcddddddddddddb, 0xbfdddddddddddddd, 0x3fd6c16c16c16c16, 0x3fed27d27d27d27d, 0x3fcb05b05b05b059, 0x3fd3e93e93e93e92, 0x3fcc71c71c71c71d, 0x3fea4fa4fa4fa4f9},
		b: []uint64{0x3fd5555555555556, 0xbfd5555555555555, 0xbff5555555555555},
		c: []uint64{0x3fbc71c71c71c720, 0xc00238e38e38e38e, 0x3fcc71c71c71c71c},
		d: []uint64{0x0},
	},
}

// TestReduceAcceptedCutMatchesStoredModelAtAxisPoles checks that every cut
// Reduce accepts of axisPoleLiterals is within Reduce's contract of the
// stored model on the whole axis, at and beside its poles and inside the
// shared-pole windows (HO6UP6), and that each literal has, on the arch it
// was frozen on (its @ suffix), a mode whose staircase cut is not: the cut
// main accepted there.
func TestReduceAcceptedCutMatchesStoredModelAtAxisPoles(t *testing.T) {
	tol := math.Sqrt(eps())
	modes := []*ReduceOpts{{}, {Equalize: true}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}}
	names := make([]string, 0, len(axisPoleLiterals))
	for k := range axisPoleLiterals {
		names = append(names, k)
	}
	slices.Sort(names)
	for _, name := range names {
		sys := axisPoleLiterals[name].system(t)
		n, _, _ := sys.Dims()
		wrong := false
		for _, opts := range modes {
			res, err := sys.Reduce(opts)
			if err != nil {
				t.Fatalf("%s %+v: %v", name, *opts, err)
			}
			if res.Order < n {
				if c, at := contractSweepAtPoles(sys, res.Sys, tol); c > 1 {
					t.Errorf("%s %+v: order %d off by %.3g·tol·max(scale, terms) at %v", name, *opts, res.Order, c, at)
				}
			}
			if cut, _, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), opts, true, -1); err == nil && cut.Order < n {
				if c, _ := contractSweepAtPoles(sys, cut.Sys, tol); c > 1 {
					wrong = true
				}
			}
		}
		if _, arch, ok := strings.Cut(name, "@"); ok && arch == runtime.GOARCH && !wrong {
			t.Errorf("%s: no staircase cut outside the contract", name)
		}
	}
}
