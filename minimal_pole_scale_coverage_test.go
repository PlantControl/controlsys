package controlsys

import (
	"math"
	"testing"
)

// poleScaleLiterals are census models whose cuts main accepted with every
// verified point far above a pole group (SCBUFK): matched-hid 287 under
// Equalize (drawn on arm64) and 312 under Equalize (amd64), whose poles
// cluster within 0.02 of z = 1 while the band verified only θ ≥ 0.27 and
// θ ≥ 0.045, 1.2e4·tol and 6.4·tol off at z = 1; and nse-rot/171 as
// the current corpus draws it on amd64 (bandLiterals holds an earlier
// draw), whose ReduceUnobservable cut main accepted.
var poleScaleLiterals = map[string]bandLiteral{
	"matched-hid/287-unobs-true@arm64": {
		n: 6, m: 1, p: 1, dt: 0.01,
		a: []uint64{
			0xc01cc68085a9013d, 0x402141b805ee7ca9, 0xc02ca07135202437, 0xc00a5b79e9a8d5d9, 0x400c32246113ad5f, 0x4010211fe82eaa48,
			0x40293f4ef34f943b, 0xc027a5e23cda590b, 0x403ba2071d592d2e, 0x4012871eec660414, 0xc01acea9acb228eb, 0xc01c92d1a452b236,
			0x40381adbfb84a097, 0xc038c161064b4f08, 0x40484071ef5662fa, 0x40219384a8539717, 0xc0257af17acee7b6, 0xc02adc3debdcd890,
			0xc04580ad39ae46a2, 0x40464ab68fc4b179, 0xc05549a90d3bbaa9, 0xc0308edb6b7f6477, 0x4033e0e10255164f, 0x403810df3324abb6,
			0xc0507dc93e0250fa, 0x405117bb45883a50, 0xc06099329ae0d3a0, 0xc03b876c59e812f9, 0x404017f3fd1d002b, 0x4042abd1239a52cf,
			0x4051821646327ada, 0xc05202799fbef786, 0x406147d89e1ec84c, 0x403a7a00e3d58f32, 0xc03e70da1e8677d0, 0xc043957b01c466fc,
		},
		b: []uint64{0xbfd8f49ac5611948, 0xbfc0dd0311fc4c9f, 0x3fb696628c255f91, 0x3ff6fad261c57672, 0x3fee20d4e9b63462, 0x3fc66b0033cc8c24},
		c: []uint64{0xbd9145647f50031b, 0xbdc17fea32d7273b, 0xbdaf292656ada70c, 0x3d800dbef3d5c473, 0x3d97cf808ec3c914, 0xbd7f56878bd445fc},
		d: []uint64{0},
	},
	"matched-hid/312-unobs-true": {
		n: 6, m: 1, p: 1, dt: 0.001,
		a: []uint64{
			0xbfbb723bf1c223a4, 0x3fc578021bc97bdc, 0xbf83437758b03950, 0x3fbf6db5d7065581, 0xbfe6ea040c32c65a, 0xbfcf66638c627614,
			0xbfeda879b8308bb6, 0x3fe908f393f99651, 0xbff11e7d4e70b9e4, 0x3fe7e6a4941c35e6, 0xbfd41e7a76f7cb5c, 0x3fe3effe402f2b57,
			0x3fe818749155b42e, 0x3fe057bcdb016168, 0x3ff74556d82a8550, 0x3fd9c4e18a98ac82, 0x3fd78740b1aa45b1, 0x3fdefb59a15936e3,
			0xbfc0606901340d2e, 0x3fd10a74e133e7fa, 0x3fd7c0b2c6d15518, 0x3fef69f786531d7e, 0xbfc79a0efd73651a, 0xbfc2acc389e9f63e,
			0x3fe7b5c92fe1af3a, 0xbfff966d4c812ac2, 0xbff9e51f6ee0537c, 0xbff2077432b90d75, 0x4002ba8321964950, 0xbfe7d8ca6d020755,
			0x3fc84eba4927deae, 0xbfe7c83b00d853ba, 0xbfe15a935ea11108, 0xbfe0084587233e8a, 0x3fdfe65fbeb42df9, 0x3fe15aef59d947e8,
		},
		b: []uint64{0x3fd90a639a3bdf1d, 0xbfdc414c96c7bb42, 0x3fd7c9eb6eb0c563, 0xbff35e431de0cb0e, 0xbfb93705ddf70c94, 0x3fd53293763fddcc},
		c: []uint64{0x3df95656279478f7, 0xbe0c4e10aca9e09d, 0xbe0c4d1ef2adb2a6, 0xbdf43dd7f3a6ab18, 0x3e0561257d3e6f60, 0xbdd7ad1789f6016e},
		d: []uint64{0},
	},
	"nse-rot/171-redraw": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{
			0x42dff3e69fb928a8, 0x42e804ea3cdba20d, 0x42f54f0b5189f193,
			0x42e804ea3cdba1c2, 0x42f20e33c15ecaa8, 0x4300049cc7b18b7f,
			0x42f54f0b5189f196, 0x4300049cc7b18b88, 0x430c6bc328b94006,
		},
		b: []uint64{0x42ed73313a247cde, 0x42f62349adea5ff1, 0x4303a3c416e37e07},
		c: []uint64{0xbfdb9db7d6ee84bf, 0xbfe29223aca38a1c, 0x3fed7a01844d932a},
		d: []uint64{0xbfb0dff51e9a5250},
	},
}

// TestReduceRefusesCutWithUncoveredClusterGap checks Reduce's contract on
// models whose cut main accepted on the strength of verified points far
// above a pole group whose own-scale points were all uninformative: the
// matched C2D clusters near z = 1 (matched-hid 287 eq on arm64, 312 eq on
// amd64) and nse-rot/171, whose unresolved slow poles leave no resolved
// gap (ReduceUnobservable on amd64, 2.6e7·tol off at s = 0.35i). Whatever
// Reduce accepts must pass the 320-bit sweep of contract (C).
func TestReduceRefusesCutWithUncoveredClusterGap(t *testing.T) {
	tol := math.Sqrt(eps())
	cases := []struct {
		name string
		lit  bandLiteral
		opts *ReduceOpts
	}{
		{"matched-hid/287-unobs-true@arm64", poleScaleLiterals["matched-hid/287-unobs-true@arm64"], &ReduceOpts{Equalize: true}},
		{"matched-hid/312-unobs-true", poleScaleLiterals["matched-hid/312-unobs-true"], &ReduceOpts{Equalize: true}},
		{"nse-rot/171-redraw", poleScaleLiterals["nse-rot/171-redraw"], &ReduceOpts{Mode: ReduceUnobservable}},
	}
	for _, c := range cases {
		sys := c.lit.system(t)
		res, err := sys.Reduce(c.opts)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if res.Order == c.lit.n {
			continue
		}
		if w, at := contractSweep(sys, res.Sys, tol); !(w <= 1) {
			t.Errorf("%s: order %d cut %.3g·tol·max(scale, terms) off at %v", c.name, res.Order, w, at)
		}
	}
}

// TestPoleScalesCoveredCountsFarPointsByScale checks the count of
// poleScalesCovered: points beyond a group's scale σ fix at most
// 1 + ⌊ln(1/tol)/ln(d/σ)⌋ of its coefficients however many there are,
// points at or below σ count fully, a shared pole counts once, and at
// tol ≥ 1 the count per scale does not apply.
func TestPoleScalesCoveredCountsFarPointsByScale(t *testing.T) {
	tol := math.Sqrt(eps())
	all := func(n int) []bool {
		v := make([]bool, n)
		for i := range v {
			v[i] = true
		}
		return v
	}
	group := []complex128{-1, -1.5, -2, -2.5, -3}
	win := poleWindows{cutPoles: []complex128{-1, -2}}
	far := []complex128{1000i, 3000i, 1e4i, 3e4i}
	if poleScalesCovered(false, group, win, far, all(len(far)), tol) {
		t.Error("7 poles of scale 3 covered by 8 conditions at 333× and beyond (at most 4)")
	}
	near := append([]complex128{1i, 2i, 3i}, far...)
	if !poleScalesCovered(false, group, win, near, all(len(near)), tol) {
		t.Error("not covered with 6 conditions inside the group and 4 beyond")
	}
	pts := []complex128{1000i}
	shared := poleWindows{centre: []complex128{-1, -2}, radius: []float64{1e-9, 1e-9}, cutPoles: []complex128{-1, -2}}
	if !poleScalesCovered(false, []complex128{-1, -2}, shared, pts, all(1), tol) {
		t.Error("two shared poles counted twice")
	}
	if poleScalesCovered(false, []complex128{-1, -2}, poleWindows{cutPoles: []complex128{-1, -2}}, pts, all(1), tol) {
		t.Error("four unshared poles covered by one point")
	}
	if !poleScalesCovered(false, group, win, far, all(len(far)), 2) {
		t.Error("count per scale applied at tol 2")
	}
}

// TestReduceTolAboveOneSkipsScaleCount checks that at Tol ≥ 1, where
// |G−R| ≤ terms ≤ Tol·terms holds at every point, the count per pole scale
// does not refuse a cut: matched-hid/221-unc-true at Tol 2 reduces to 0.
func TestReduceTolAboveOneSkipsScaleCount(t *testing.T) {
	sys := bandLiterals["matched-hid/221-unc-true"].system(t)
	res, err := sys.Reduce(&ReduceOpts{Tol: 2})
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 0 {
		t.Errorf("order %d at Tol 2, want 0", res.Order)
	}
}
