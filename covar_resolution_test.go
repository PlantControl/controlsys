package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// covarExactReference returns the exact P of the delay-free stored model
// driven by W: C·X·Cᵀ (+ D·W·Dᵀ when discrete) with B·W·Bᵀ, X and the
// products in hbPrec bits, and +Inf where continuous noise through D reaches.
func covarExactReference(sys *System, W *mat.Dense) (*mat.Dense, bool) {
	_, _, p := sys.Dims()
	disc := sys.Dt > 0
	b, c, w, d := hbFrom(sys.B), hbFrom(sys.C), hbFrom(W), hbFrom(sys.D)
	x, ok := hsvExactLyap(hbFrom(sys.A), hbMul(hbMul(b, w), b.t()), disc)
	if !ok {
		return nil, false
	}
	P := hbMul(hbMul(c, x), c.t())
	dwd := hbMul(hbMul(d, w), d.t())
	out := mat.NewDense(p, p, nil)
	for i := range p {
		for j := range p {
			v := P.at(i, j)
			if disc {
				v.Add(v, dwd.at(i, j))
			} else if s := dwd.at(i, j).Sign(); s != 0 {
				out.Set(i, j, math.Inf(s))
				continue
			}
			f, _ := v.Float64()
			out.Set(i, j, f)
		}
	}
	return out, true
}

// covarRelError returns ‖P − ref‖_F/‖ref‖_F over the finite entries, +Inf
// when the infinite entries differ.
func covarRelError(P, ref *mat.Dense) float64 {
	r, c := ref.Dims()
	d, n := 0.0, 0.0
	for i := range r {
		for j := range c {
			a, b := P.At(i, j), ref.At(i, j)
			if math.IsInf(a, 0) || math.IsInf(b, 0) {
				if a != b {
					return math.Inf(1)
				}
				continue
			}
			d += (a - b) * (a - b)
			n += b * b
		}
	}
	if n == 0 {
		if d == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return math.Sqrt(d / n)
}

func covarIdentity(m int) *mat.Dense {
	W := mat.NewDense(m, m, nil)
	for i := range m {
		W.Set(i, i, 1)
	}
	return W
}

func TestCovar_WeakCouplingMatchesExact(t *testing.T) {
	for _, c := range h2WeakCoupling {
		sys := c.sys()
		W := covarIdentity(c.m)
		want, ok := covarExactReference(sys, W)
		if !ok {
			t.Fatalf("%s: exact reference failed", c.name)
		}
		P, err := Covar(sys, W)
		if err != nil {
			t.Fatalf("%s: Covar err %v, want nil", c.name, err)
		}
		if e := covarRelError(P, want); e > hsvTau/8 {
			t.Fatalf("%s: Covar relative error %.3g > τ/8\nP=%v\nexact=%v", c.name, e, mat.Formatted(P), mat.Formatted(want))
		}
	}
}

func TestCovar_BelowResolutionUnverified(t *testing.T) {
	for _, c := range hsvBelowResolution {
		P, err := Covar(c.sys(), covarIdentity(c.m))
		if !errors.Is(err, ErrAccuracyUnverified) || P == nil {
			t.Fatalf("%s: Covar = %v, err %v, want P with ErrAccuracyUnverified", c.name, P != nil, err)
		}
		if mat.Norm(P, 2) == 0 {
			t.Fatalf("%s: Covar = 0 for a nonzero transfer function", c.name)
		}
	}
}

func TestCovar_ExactlyZeroTransferFunction(t *testing.T) {
	cancelA, cancelB, cancelC := zeroCancelSystem(false)
	cancel, _ := New(cancelA, cancelB, cancelC, mat.NewDense(1, 1, nil), 0)
	structural := hsvLiteral(3, 2, 1,
		[]float64{-1, 0.5, 0.25, 0, -2, 0, 0, 0.75, -3},
		[]float64{1, -2, 0, 0, 0, 0},
		[]float64{0, 1, 4}, 0)
	discD, err := NewFromSlices(2, 1, 1, []float64{-0.75, 0.25, -0.25, -0.25}, []float64{1, 1}, []float64{1, -1}, []float64{0.75}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		sys  *System
		want float64
	}{{"structural", structural, 0}, {"cancellation", cancel, 0}, {"cancellation discrete with D", discD, 0.5625}} {
		_, m, _ := c.sys.Dims()
		P, err := Covar(c.sys, covarIdentity(m))
		if err != nil {
			t.Fatalf("%s: Covar err %v, want nil", c.name, err)
		}
		if g := P.At(0, 0); math.Abs(g-c.want) > hsvTau*c.want/8 || c.want == 0 && g != 0 {
			t.Fatalf("%s: Covar = %v, want %v", c.name, g, c.want)
		}
	}
	bwZero, err := NewFromSlices(2, 2, 1, []float64{-1, 0.5, -0.25, -2}, []float64{1, 1, 0.5, 0.5}, []float64{1, 2}, []float64{0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	P, err := Covar(bwZero, mat.NewDense(2, 2, []float64{1, -1, -1, 1}))
	if err != nil || P.At(0, 0) != 0 {
		t.Fatalf("B·W = 0 with C·Aᵏ·B ≠ 0: Covar = %v, %v; want 0, nil", P, err)
	}
	ulpA, ulpB, ulpC := zeroCancelSystem(true)
	ulp, _ := New(ulpA, ulpB, ulpC, mat.NewDense(1, 1, nil), 0)
	P, err = Covar(ulp, covarIdentity(1))
	if err == nil && P.At(0, 0) == 0 {
		t.Fatal("one ulp from a zero transfer function: Covar = 0 with nil error")
	}
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("one-ulp model: Covar err %v, want nil or ErrAccuracyUnverified", err)
	}
}

// covarCase is a frozen Covar literal: the model (row-major A, B, C, D)
// and its noise intensity W.
type covarCase struct {
	name       string
	n, m, p    int
	dt         float64
	a, b, c, d []float64
	w          []float64
}

func (c covarCase) sys() *System {
	s, err := NewFromSlices(c.n, c.m, c.p, c.a, c.b, c.c, c.d, c.dt)
	if err != nil {
		panic(err)
	}
	return s
}

func (c covarCase) W() *mat.Dense { return mat.NewDense(c.m, c.m, append([]float64(nil), c.w...)) }

// requireCovarExact checks a nil Covar within τ/8 of the 320-bit reference
// and returns the route.
func requireCovarExact(t *testing.T, c covarCase) hsvRoute {
	t.Helper()
	want, ok := covarExactReference(c.sys(), c.W())
	if !ok {
		t.Fatalf("%s: exact reference failed", c.name)
	}
	P, route, err := covarRouted(c.sys(), c.W())
	if err != nil {
		t.Fatalf("%s: Covar route %d, err %v, want nil", c.name, route, err)
	}
	if e := covarRelError(P, want); e > hsvTau/8 {
		t.Fatalf("%s: Covar relative error %.3g > τ/8\nP=%v\nexact=%v", c.name, e, mat.Formatted(P), mat.Formatted(want))
	}
	return route
}

// canc0/k8|sing (D5BGNA PR3 census, arm64 data).
var covarCanc8 = covarCase{name: "canc0/k8|sing", n: 3, m: 2, p: 1, dt: 0,
	a: []float64{-1.7394648054132271, -0.009272931567071253, -0.08481234788029207, 0.02963016562868459, -2.2462603695996783, -0.061849474553334054, 0.004935835102640507, -0.1054552449335286, -3.425330435280071},
	b: []float64{0.1015309000788589, 0.30459270323148696, -0.27472388992647645, -0.8241716718313963, 0.23664286357002684, 0.7099285883820914},
	c: []float64{-0.23083452364821833, 0.34888531272467704, -0.3650743106223099},
	d: []float64{0, 0},
	w: []float64{9, -3, -3, 1},
}

// canc0/k12|sing (D5BGNA PR3 census, arm64 data).
var covarCanc12 = covarCase{name: "canc0/k12|sing", n: 3, m: 2, p: 1, dt: 0,
	a: []float64{-1.7394648054132271, -0.009272931567071253, -0.08481234788029207, 0.02963016562868459, -2.2462603695996783, -0.061849474553334054, 0.004935835102640507, -0.1054552449335286, -3.425330435280071},
	b: []float64{-0.33421682081313764, -1.002650462439779, 0.4283953126543525, 1.2851859379634456, 0.36832458983923844, 1.104973769517582},
	c: []float64{-0.23083452364821833, 0.34888531272467704, -0.3650743106223099},
	d: []float64{0, 0},
	w: []float64{9, -3, -3, 1},
}

// dcanc0/s8k8|sing (D5BGNA PR3 census, arm64 data).
var covarDCanc = covarCase{name: "dcanc0/s8k8|sing", n: 3, m: 2, p: 1, dt: 0.5,
	a: []float64{-0.04305383934479374, 0.003963072577763349, 0.012276054632485989, -0.014388221006119238, -0.25431112884081836, 0.08693129299267871, 0.036590082663388736, 0.0055695399051666535, -0.06919534343324005},
	b: []float64{4.9905914619202974e-09, -8.181731158483974e-10, 1.618275346576692e-09, 4.8036906750761525e-09, 7.209926945720424e-11, 1.6676666848006282e-10},
	c: []float64{0.27634762714209393, 0.23743335495149076, 0.3994265122692038},
	d: []float64{-0.19465507042701458, -0.5839652091320884},
	w: []float64{9, -3, -3, 1},
}

// ccanc0/k10|sing (D5BGNA PR3 census, arm64 data).
var covarCCanc = covarCase{name: "ccanc0/k10|sing", n: 3, m: 2, p: 1, dt: 0,
	a: []float64{-1.1024056770157737, -0.07778366041651769, 0.07316791329302892, 0.1622142082201562, -1.5676963592781252, -0.0018714182442004736, -0.1163561147659646, -0.17337237755509904, -1.513313625976168},
	b: []float64{0.0044183279063140235, -0.018056714991513734, 0.09917872068812406, 0.19230872462090642, -0.1710097659424631, 0.424225199867703},
	c: []float64{-0.4877260479978658, -0.2290889327337936, -0.19805512021793437},
	d: []float64{0.34491873935746453, 1.0347562181203058},
	w: []float64{9, -3, -3, 1},
}

// wide0/k30|indef (D5BGNA PR3 census, arm64 data).
var covarWideIndef = covarCase{name: "wide0/k30|indef", n: 2, m: 6, p: 1, dt: 0,
	a: []float64{-2.357415256461679, 0.5902559806242645, 0.5902559790683677, -1.58915915618955},
	b: []float64{0.22725498340588807, -0.11870677454966069, 0.10029007255617944, 0.1424222333157417, -0.01737768476194793, -0.12896322080527836, 0.41903424243776993, -0.21888278355949758, 0.1849243257409632, 0.2626115905109382, -0.03204261952988512, -0.23779458968324},
	c: []float64{0.012467146385352033, -0.00676131174492908},
	d: []float64{0, 0, 0, 0, 0, 0},
	w: []float64{-1.2687698084839576, 0.3351162500617363, 0.020249731512023257, -0.10999443213907523, -0.11925561786016597, 0.2713700657243473, 0.3351162500617363, -1.093910884756626, -0.17801224349618863, -0.044464204141030084, 0.03416986523205939, -0.002534166933577745, 0.020249731512023257, -0.17801224349618863, -1.2860074658462022, 0.0016550434724000175, 0.3745347677343966, 0.23680238747669644, -0.10999443213907523, -0.044464204141030084, 0.0016550434724000175, -1.2946052324689914, 0.16027186100266996, 0.07575800151328732, -0.11925561786016597, 0.03416986523205939, 0.3745347677343966, 0.16027186100266996, -1.0710974518588867, -0.0016155570948913722, 0.2713700657243473, -0.002534166933577745, 0.23680238747669644, 0.07575800151328732, -0.0016155570948913722, -1.4713942984719315},
}

// sw0k39r1/112|diag (D5BGNA PR3 census, arm64 data).
var covarSwDiag = covarCase{name: "sw0k39r1/112|diag", n: 5, m: 1, p: 1, dt: 0,
	a: []float64{-1.412773839946212, -0.5285873768515353, 1.2304413579704137, 0.32528491946931365, -1.0392414653445412, -1.5026515776135845, -1.8605366604947728, 0.7700735332678388, 0.05744549749052285, -0.3630798779243396, 0.4832353571926975, 1.2597578967334069, -1.6571093671940569, 0.35706771964192496, 0.5136529364473561, -0.5880141088781332, 0.1622031600425496, -0.02170885275795177, -0.8409934772716063, 0.16487443319661332, -0.3768088190373946, -1.0456565185513431, 0.3230674976499477, -0.4038783108190934, -1.2285866550933535},
	b: []float64{-0.4928489895259175, -0.788858744655321, 0.2262239549491722, 0.29689931442782413, -0.495454613262967},
	c: []float64{-1.0262120974934883, 0.020387728516947173, -0.23988342949260738, 0.2604968622600228, 1.0349253096197248},
	d: []float64{0},
	w: []float64{0.5},
}

// sw0k39r1/112|dense (D5BGNA PR3 census, arm64 data).
var covarSwDense = covarCase{name: "sw0k39r1/112|dense", n: 5, m: 1, p: 1, dt: 0,
	a: []float64{-1.412773839946212, -0.5285873768515353, 1.2304413579704137, 0.32528491946931365, -1.0392414653445412, -1.5026515776135845, -1.8605366604947728, 0.7700735332678388, 0.05744549749052285, -0.3630798779243396, 0.4832353571926975, 1.2597578967334069, -1.6571093671940569, 0.35706771964192496, 0.5136529364473561, -0.5880141088781332, 0.1622031600425496, -0.02170885275795177, -0.8409934772716063, 0.16487443319661332, -0.3768088190373946, -1.0456565185513431, 0.3230674976499477, -0.4038783108190934, -1.2285866550933535},
	b: []float64{-0.4928489895259175, -0.788858744655321, 0.2262239549491722, 0.29689931442782413, -0.495454613262967},
	c: []float64{-1.0262120974934883, 0.020387728516947173, -0.23988342949260738, 0.2604968622600228, 1.0349253096197248},
	d: []float64{0},
	w: []float64{1.25},
}

// sw1k37r1/115|sing (D5BGNA PR3 census, arm64 data).
var covarSwSing = covarCase{name: "sw1k37r1/115|sing", n: 7, m: 2, p: 2, dt: 0.5,
	a: []float64{-0.05458868430989859, 0.026848360883206978, -0.09476291727912371, -0.11785069928786854, -0.26837548813855705, -0.19789382981197295, 0.11752976746138613, -0.07155177920388833, -0.05334942930165884, 0.031848739176287034, 0.04191982035675913, 0.18349804924182467, 0.17065770700346627, -0.04789864765103527, 0.04194474780045602, -0.08344147186141525, -0.10845690983274511, 0.027373604347425973, 0.14762414537541313, -0.021369063348516362, -0.04058517255900489, 0.08806814487936272, -0.060918936006189514, -0.07298079579208115, -0.07536428191560729, -0.029865373511025896, -0.024555125352848078, -0.03658905848954581, 0.2782679494344731, -0.17258994086434087, -0.014829572658121983, 0.050501336003924514, -0.11042040333959426, -0.012183842490032832, -0.16957099187662972, 0.19684625005140527, -0.17364060158107175, -0.043745863867213804, 0.024958008751409553, 0.09467228186055342, -0.05346559990206752, -0.033954144166803056, -0.1333181686358567, 0.04060001769521285, 0.12614939810721582, -0.007635794913814354, 0.03935481346274216, 0.10980789788371859, -0.10685469139842835},
	b: []float64{1.4824588150382052, -0.7724663998735106, -0.9255679565600684, 0.5271820952338867, -0.6974070554210621, -0.1754263331979923, -0.019456745686308285, -0.24967812412286639, -0.4048892524881361, -0.07360760453248527, -0.7600404193258176, -0.28462757358871216, -0.1725917373792737, 0.6677051747548848},
	c: []float64{-3.607558593881649, -2.5241081874334608, -2.728362643903214, 1.2807686114219936, -1.179056524659361, -0.2599718043843529, -2.659381181736648, 0.49501813575721587, 0.8969845073001943, 1.54676458927297, 1.1933397139107305, 0.7354757630635437, -1.9599789926104434, -0.037325888889617076},
	d: []float64{0, 0, 0, 0},
	w: []float64{0.25, -0.5, -0.5, 1},
}

// The census diag (entries 0.5, 2: inexact square roots) and dense W on
// sw0k39r1/112, where the factor path was off by 1.5e-5 and 5.9e-5, a
// singular W and an indefinite W with m = 40 > n.
func TestCovar_GeneralNoiseIntensityRefined(t *testing.T) {
	for _, c := range []covarCase{covarSwDiag, covarSwDense, covarSwSing, covarWideIndef} {
		if route := requireCovarExact(t, c); route != hsvRouteCore {
			t.Fatalf("%s: route %d, want the refined path", c.name, route)
		}
	}
}

// W = v·vᵀ with B·v nearly zero: ρ ≈ 10¹⁷. The plain path is refused; at
// κ = 10⁻⁸ the double-double B·W·Bᵀ resolves P, at κ = 10⁻¹² (ρ ≈ 5·10²⁵)
// it does not, and the tier-1 value is never returned with nil.
func TestCovar_CancellingNoiseIntensity(t *testing.T) {
	if route := requireCovarExact(t, covarCanc8); route != hsvRouteCore {
		t.Fatalf("%s: route %d, want the refined path", covarCanc8.name, route)
	}
	P, route, err := covarRouted(covarCanc12.sys(), covarCanc12.W())
	if !errors.Is(err, ErrAccuracyUnverified) || P == nil || route != hsvRouteFloor {
		t.Fatalf("%s: Covar route %d, err %v, want P with ErrAccuracyUnverified", covarCanc12.name, route, err)
	}
}

// D·W·Dᵀ cancelling: the discrete dcanc literal (tier-1 value 90% off) is
// nil and exact; the continuous ccanc literal's variance is +Inf where the
// float64 D·W·Dᵀ has the wrong sign, and the pattern is symmetric; a
// discrete static gain with D·W·Dᵀ exactly zero returns exactly zero.
func TestCovar_CancellingFeedthrough(t *testing.T) {
	requireCovarExact(t, covarDCanc)
	requireCovarExact(t, covarCCanc)
	P, err := Covar(covarCCanc.sys(), covarCCanc.W())
	if err != nil {
		t.Fatal(err)
	}
	p := covarCCanc.p
	for i := range p {
		if !math.IsInf(P.At(i, i), 1) {
			t.Fatalf("%s: P[%d,%d] = %v, want +Inf", covarCCanc.name, i, i, P.At(i, i))
		}
		for j := range p {
			if a, b := P.At(i, j), P.At(j, i); math.IsInf(a, 0) != math.IsInf(b, 0) || math.IsInf(a, 0) && a != b {
				t.Fatalf("%s: pattern not symmetric at (%d,%d): %v, %v", covarCCanc.name, i, j, a, b)
			}
		}
	}
	gain, err := NewFromSlices(0, 2, 1, nil, nil, nil, []float64{1, 1}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	P, err = Covar(gain, mat.NewDense(2, 2, []float64{1, -1, -1, 1}))
	if err != nil || P.At(0, 0) != 0 {
		t.Fatalf("static gain with D·W·Dᵀ = 0: Covar = %v, %v; want exactly 0, nil", P, err)
	}
}

// Continuous external delays have no refined path. rlrep/23 with a common
// input delay and sw0k39r1/112 with two inputs, distinct delays and a
// correlated W are below the delayed gate and flagged; canc0/k12 with two
// distinct delays (the delay difference undoes the cancellation) passes it,
// and a resolved model with a common delay stays nil and equals the
// delay-free P.
func TestCovar_DelayedBelowResolutionUnverified(t *testing.T) {
	c := hsvBelowResolution[0]
	sys := c.sys()
	if err := sys.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	if P, route, err := covarRouted(sys, covarIdentity(1)); !errors.Is(err, ErrAccuracyUnverified) || P == nil || route != hsvRouteDelayed {
		t.Fatalf("%s delayed: Covar route %d, err %v, want P with ErrAccuracyUnverified", c.name, route, err)
	}
	w := covarSwDiag
	b2 := make([]float64, 2*w.n)
	for i := range w.n {
		b2[2*i], b2[2*i+1] = w.b[i], -0.5*w.b[i]
	}
	weak, err := NewFromSlices(w.n, 2, w.p, w.a, b2, w.c, make([]float64, 2*w.p), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := weak.SetInputDelay([]float64{0.2, 0.7}); err != nil {
		t.Fatal(err)
	}
	correlated := mat.NewDense(2, 2, []float64{2, 0.5, 0.5, 1})
	if P, route, err := covarRouted(weak, correlated); !errors.Is(err, ErrAccuracyUnverified) || P == nil || route != hsvRouteDelayed {
		t.Fatalf("%s, two input delays, correlated W: Covar route %d, err %v, want P with ErrAccuracyUnverified", w.name, route, err)
	}
	two := covarCanc12.sys()
	if err := two.SetInputDelay([]float64{0.2, 0.7}); err != nil {
		t.Fatal(err)
	}
	if P, route, err := covarRouted(two, covarCanc12.W()); err != nil || P == nil || route != hsvRoutePlain {
		t.Fatalf("%s, two input delays: Covar route %d, err %v, want the delayed gate's nil", covarCanc12.name, route, err)
	}
	ok := benchSysNonSym(t, 4, 1, 1)
	if err := ok.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	want, _ := covarExactReference(benchSysNonSym(t, 4, 1, 1), covarIdentity(1))
	P, err := Covar(ok, covarIdentity(1))
	if err != nil || covarRelError(P, want) > hsvTau/8 {
		t.Fatalf("resolved delayed model: Covar err %v, relative error %.3g", err, covarRelError(P, want))
	}
}

// Covar is P for the symmetric part of W: a discrete model with D ≠ 0 and a
// non-symmetric W gives the P of ½(W + Wᵀ), symmetric, feedthrough included.
func TestCovar_NonSymmetricNoiseUsesSymmetricPart(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2, []float64{0.5, 0.25, -0.125, 0.25}, []float64{1, 0.5, -0.5, 1}, []float64{1, 0, 0.5, 1}, []float64{1, 0.25, -0.5, 1}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	W := mat.NewDense(2, 2, []float64{2, 1, -0.5, 1})
	Ws := mat.NewDense(2, 2, []float64{2, 0.25, 0.25, 1})
	P, err := Covar(sys, W)
	if err != nil {
		t.Fatal(err)
	}
	want, ok := covarExactReference(sys, Ws)
	if !ok {
		t.Fatal("exact reference failed")
	}
	if e := covarRelError(P, want); e > hsvTau/8 {
		t.Fatalf("Covar with non-symmetric W: relative error %.3g against P of the symmetric part", e)
	}
	if P.At(0, 1) != P.At(1, 0) {
		t.Fatalf("Covar with non-symmetric W: P not symmetric: %v", mat.Formatted(P))
	}
}
