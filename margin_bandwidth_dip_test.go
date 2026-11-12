package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const bandwidthDipPrec = 320

type bandwidthDipC struct{ re, im *big.Float }

func bandwidthDipF(v float64) *big.Float {
	return new(big.Float).SetPrec(bandwidthDipPrec).SetFloat64(v)
}

func bandwidthDipMul(a, b bandwidthDipC) bandwidthDipC {
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthDipPrec) }
	return bandwidthDipC{
		nf().Sub(nf().Mul(a.re, b.re), nf().Mul(a.im, b.im)),
		nf().Add(nf().Mul(a.re, b.im), nf().Mul(a.im, b.re)),
	}
}

func bandwidthDipDiv(a, b bandwidthDipC) bandwidthDipC {
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthDipPrec) }
	d := nf().Add(nf().Mul(b.re, b.re), nf().Mul(b.im, b.im))
	n := bandwidthDipMul(a, bandwidthDipC{b.re, nf().Neg(b.im)})
	return bandwidthDipC{n.re.Quo(n.re, d), n.im.Quo(n.im, d)}
}

func bandwidthDipSub(a, b bandwidthDipC) bandwidthDipC {
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthDipPrec) }
	return bandwidthDipC{nf().Sub(a.re, b.re), nf().Sub(a.im, b.im)}
}

// bandwidthDipGainSq is |D + C(λI − A)⁻¹B|² of the float64 SISO model in
// big.Float at λ = jν, or z = (1+jν)/(1−jν) on the unit circle when
// discrete, so λ carries no rounding next to DC.
func bandwidthDipGainSq(sys *System, nu float64) *big.Float {
	n, _, _ := sys.Dims()
	one, zero := bandwidthDipF(1), bandwidthDipF(0)
	lam := bandwidthDipC{zero, bandwidthDipF(nu)}
	if sys.IsDiscrete() {
		lam = bandwidthDipDiv(bandwidthDipC{one, bandwidthDipF(nu)}, bandwidthDipC{one, bandwidthDipF(-nu)})
	}
	m := make([]bandwidthDipC, n*n)
	x := make([]bandwidthDipC, n)
	for i := range n {
		for j := range n {
			m[i*n+j] = bandwidthDipC{bandwidthDipF(-sys.A.At(i, j)), bandwidthDipF(0)}
		}
		m[i*n+i] = bandwidthDipC{new(big.Float).SetPrec(bandwidthDipPrec).Add(m[i*n+i].re, lam.re), lam.im}
		x[i] = bandwidthDipC{bandwidthDipF(sys.B.At(i, 0)), bandwidthDipF(0)}
	}
	abs2 := func(a bandwidthDipC) *big.Float {
		v := new(big.Float).SetPrec(bandwidthDipPrec).Mul(a.re, a.re)
		return v.Add(v, new(big.Float).SetPrec(bandwidthDipPrec).Mul(a.im, a.im))
	}
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if abs2(m[i*n+k]).Cmp(abs2(m[p*n+k])) > 0 {
				p = i
			}
		}
		for j := range n {
			m[k*n+j], m[p*n+j] = m[p*n+j], m[k*n+j]
		}
		x[k], x[p] = x[p], x[k]
		for i := k + 1; i < n; i++ {
			f := bandwidthDipDiv(m[i*n+k], m[k*n+k])
			for j := k; j < n; j++ {
				m[i*n+j] = bandwidthDipSub(m[i*n+j], bandwidthDipMul(f, m[k*n+j]))
			}
			x[i] = bandwidthDipSub(x[i], bandwidthDipMul(f, x[k]))
		}
	}
	h := bandwidthDipC{bandwidthDipF(sys.D.At(0, 0)), bandwidthDipF(0)}
	for k := n - 1; k >= 0; k-- {
		for j := k + 1; j < n; j++ {
			x[k] = bandwidthDipSub(x[k], bandwidthDipMul(m[k*n+j], x[j]))
		}
		x[k] = bandwidthDipDiv(x[k], m[k*n+k])
		t := bandwidthDipMul(bandwidthDipC{bandwidthDipF(sys.C.At(0, k)), bandwidthDipF(0)}, x[k])
		h.re.Add(h.re, t.re)
		h.im.Add(h.im, t.im)
	}
	return abs2(h)
}

// bandwidthDipOracle is the first ω where |H| falls 3 dB below |H(DC)|: the
// first of 30 log-spaced ν per decade from 1e-14 past which |H|² is below
// the level, bisected in ν, or +Inf when no sample on the axis, nor
// Nyquist, is.
func bandwidthDipOracle(sys *System) float64 {
	target := bandwidthDipGainSq(sys, 0)
	target.Mul(target, bandwidthDipF(math.Pow(10, -3.0/10)))
	below := func(nu float64) bool { return bandwidthDipGainSq(sys, nu).Cmp(target) < 0 }
	omega := func(nu float64) float64 {
		if sys.IsDiscrete() {
			return 2 * math.Atan(nu) / sys.Dt
		}
		return nu
	}
	top := 1e10
	if sys.IsDiscrete() {
		top = 1e17
	}
	prev := 1e-14
	for k := 1; ; k++ {
		nu := 1e-14 * math.Pow(10, float64(k)/30)
		if nu > top {
			return math.Inf(1)
		}
		if below(nu) {
			lo, hi := prev, nu
			for range 120 {
				mid := math.Sqrt(lo * hi)
				if mid <= lo || mid >= hi {
					break
				}
				if below(mid) {
					hi = mid
				} else {
					lo = mid
				}
			}
			return omega(hi)
		}
		prev = nu
	}
}

// bandwidthDipModel is 0.05·sᵏ/(s+0.7)ᵏ, or 0.05·(1−z)ᵏ/(1−0.9z)ᵏ with Ts 0.1,
// realized with D = 0.05 (0.05/0.9ᵏ) cancelling C(λI − A)⁻¹B at DC, then
// moved by a non-orthogonal similarity so A is non-symmetric, and D offset by
// delta.
func bandwidthDipModel(t *testing.T, k int, discrete bool, delta float64) *System {
	t.Helper()
	num, den := []float64{0.05}, []float64{1}
	dt := 0.0
	for range k {
		if discrete {
			num, den, dt = bandwidthDipPolyMul(num, []float64{-1, 1}), bandwidthDipPolyMul(den, []float64{-0.9, 1}), 0.1
		} else {
			num, den = bandwidthDipPolyMul(num, []float64{1, 0}), bandwidthDipPolyMul(den, []float64{1, 0.7})
		}
	}
	res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	s := res.Sys
	n, _, _ := s.Dims()
	rng := rand.New(rand.NewPCG(uint64(k), 3))
	T := make([]float64, n*n)
	for i := range T {
		T[i] = 0.3 * rng.NormFloat64()
	}
	for i := range n {
		T[i*n+i]++
	}
	Ti := bandwidthDipInv(T, n)
	a, b, c := s.A.RawMatrix().Data, s.B.RawMatrix().Data, s.C.RawMatrix().Data
	ar := bandwidthDipMatMul(bandwidthDipMatMul(Ti, a, n, n, n), T, n, n, n)
	br := bandwidthDipMatMul(Ti, b, n, n, 1)
	cr := bandwidthDipMatMul(c, T, 1, n, n)
	sys, err := NewFromSlices(n, 1, 1, ar, br, cr, []float64{s.D.At(0, 0) + delta}, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func bandwidthDipPolyMul(a, b []float64) []float64 {
	o := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			o[i+j] += x * y
		}
	}
	return o
}

func bandwidthDipInv(a []float64, n int) []float64 {
	m := append([]float64(nil), a...)
	inv := make([]float64, n*n)
	for i := range n {
		inv[i*n+i] = 1
	}
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if math.Abs(m[i*n+k]) > math.Abs(m[p*n+k]) {
				p = i
			}
		}
		for j := range n {
			m[k*n+j], m[p*n+j] = m[p*n+j], m[k*n+j]
			inv[k*n+j], inv[p*n+j] = inv[p*n+j], inv[k*n+j]
		}
		piv := m[k*n+k]
		for j := range n {
			m[k*n+j] /= piv
			inv[k*n+j] /= piv
		}
		for i := range n {
			if i != k {
				f := m[i*n+k]
				for j := range n {
					m[i*n+j] -= f * m[k*n+j]
					inv[i*n+j] -= f * inv[k*n+j]
				}
			}
		}
	}
	return inv
}

func bandwidthDipMatMul(a, b []float64, r, k, c int) []float64 {
	o := make([]float64, r*c)
	for i := range r {
		for j := range c {
			for l := range k {
				o[i*c+j] += a[i*k+l] * b[l*c+j]
			}
		}
	}
	return o
}

// A k-fold DC zero cancelled by D, as in TestBandwidth_ZeroDCGain, leaves a
// DC gain δ far below D. Where the zero's term turns against δ the gain dips
// below the −3 dB level past DC and rises again: two crossings with the same
// sign of |L| − γ at either end. The gain pencil's roots fell off those
// crossings past polish's reach (for γ ≪ D) or were rejected beside the 2k
// clustered roots, and Bandwidth returned +Inf with a nil error (CNGTCF).
func TestBandwidthDipPastDCCrossingCertified(t *testing.T) {
	tests := []struct {
		name     string
		k        int
		discrete bool
		delta    float64
	}{
		{"s^2 delta 1e-17", 2, false, 1e-17},
		{"s^2 delta 1e-12", 2, false, 1e-12},
		{"s^2 delta 3e-7", 2, false, 3.1622776601683795e-07},
		{"s^2 delta 1e-5 pencil resolves", 2, false, 1e-5},
		{"(1-z)^2 delta 1e-15", 2, true, 1e-15},
		{"(1-z)^2 delta 3e-6", 2, true, 3.1622776601683795e-06},
		{"s^4 delta -1e-9", 4, false, -1e-9},
		{"(1-z)^4 delta -1e-7", 4, true, -1e-7},
		{"(1-z)^5 delta -1e-5 clustered roots", 5, true, -1e-5},
		{"(1-z)^6 delta -3e-4 pencil resolves", 6, true, -3.1622776601683794e-04},
		{"s^6 delta 3e-7 shallow dip", 6, false, 3.1622776601683795e-07},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := bandwidthDipModel(t, tc.k, tc.discrete, tc.delta)
			want := bandwidthDipOracle(sys)
			if math.IsInf(want, 1) {
				t.Fatalf("oracle +Inf; the model has no dip")
			}
			got, err := Bandwidth(sys, -3)
			if err != nil || !(math.Abs(got-want) <= 1e-6*want) {
				t.Errorf("Bandwidth = %v, %v; want %v (big.Float oracle)", got, err, want)
			}
		})
	}
}

// A sixth-order zero at z = 1 cancelled by D up to δ = −3.2e-4, moved by a
// non-orthogonal similarity: the gain pencil has no root near the dip at
// ω ≈ 0.436, polish rejects none, and only the cancelled DC gain flags it.
// Bandwidth returned +Inf with a nil error (CNGTCF).
func TestBandwidthDipPastDCWithoutPencilRoots(t *testing.T) {
	sys, err := NewFromSlices(6, 1, 1,
		[]float64{
			1.1119135065068497, 0.2451700708672957, -0.48373002896821476, 0.79869713501956163, 0.86164128238442861, 1.8085370209253553,
			3.335538006146439, 1.2689179767704339, -3.1226850377672082, 0.058554384148514625, 10.939919773114115, 9.0534296400785887,
			-4.5634102517940383, -2.2655155991254805, 6.8102351358093012, -0.52115668811229021, -23.527425625489453, -18.211970273619578,
			1.0960594239195247, 0.62412882611359521, 0.098052993129019148, 0.83526415599869741, 0.34015120929441378, 1.5175334354526195,
			-3.5540230992631932, -3.1997868122897075, 4.5547133809407416, 1.7652713574237269, -19.991338776360578, -14.570327205585711,
			3.722989426071154, 3.6737116677643655, -4.9486242744257281, -2.519815529153024, 24.159852658161331, 16.631674667941965,
		},
		[]float64{0.19550941913789432, 0.26488615721895842, -0.33530992978273572, 0.17151301684386877, -0.022238298843745612, -0.15758872947791533},
		[]float64{0.43150242848364506, 0.29595303599989226, -0.56141805936460032, 0.044112105720684816, 2.0118360478071677, 1.5934460802244248},
		[]float64{0.09376759339192918}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	want := bandwidthDipOracle(sys)
	if got, err := Bandwidth(sys, -3); err != nil || !(math.Abs(got-want) <= 1e-6*want) {
		t.Errorf("Bandwidth = %v, %v; want %v (big.Float oracle)", got, err, want)
	}
}

// 0.05s²/(s + 0.7)² with 0.7·0.7 rounded realizes D = 0.05 cancelling
// C(−A)⁻¹B to 1.4e-18, below an ulp of D: DCGain certified 0 to its own
// rounding (UP7H7O), and Bandwidth took that zero for MATLAB's high-pass
// +Inf, while the model's gain dips 3 dB below its DC value at ω ≈ 2e-9
// (CNGTCF).
func TestBandwidthCancelledDCGainResolved(t *testing.T) {
	pole := 0.7
	res, err := (&TransferFunc{Num: [][][]float64{{{0.05, 0, 0}}}, Den: [][]float64{{1, 2 * pole, pole * pole}}}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := res.Sys
	if g, err := sys.DCGain(); err != nil || g.At(0, 0) != 1.3594567648471302e-18 {
		t.Errorf("DCGain = %v, %v; want 1.3594567648471302e-18 (exact)", g.At(0, 0), err)
	}
	want := bandwidthDipOracle(sys)
	if got, err := Bandwidth(sys, -3); err != nil || !(math.Abs(got-want) <= 1e-6*want) {
		t.Errorf("Bandwidth = %v, %v; want %v (big.Float oracle)", got, err, want)
	}
}

// KQVXXL removed Bandwidth's reading of a certified zero DC gain from the
// double-double DC response (CNGTCF): DCGain certifies a zero only where it
// is exact (UP7H7O, OECG2I, KQVXXL). On the CNGTCF literal, and on its
// descriptor twin with E = 2I, A and B doubled, the DC gain stays the exact
// 1.36e-18 and Bandwidth the oracle's.
func TestBandwidthCancelledDCGainDescriptorTwin(t *testing.T) {
	pole := 0.7
	res, err := (&TransferFunc{Num: [][][]float64{{{0.05, 0, 0}}}, Den: [][]float64{{1, 2 * pole, pole * pole}}}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := res.Sys
	want := bandwidthDipOracle(sys)
	n, _, _ := sys.Dims()
	e := mat.NewDense(n, n, nil)
	for i := range n {
		e.Set(i, i, 2)
	}
	a, b := mat.DenseCopyOf(sys.A), mat.DenseCopyOf(sys.B)
	a.Scale(2, a)
	b.Scale(2, b)
	desc, err := NewDescriptor(a, b, sys.C, sys.D, e, 0)
	if err != nil {
		t.Fatal(err)
	}
	for name, s := range map[string]*System{"explicit": sys, "descriptor": desc} {
		if g, err := s.DCGain(); err != nil || g.At(0, 0) != 1.3594567648471302e-18 {
			t.Errorf("%s: DCGain = %v, %v; want 1.3594567648471302e-18 (exact)", name, g.At(0, 0), err)
		}
		if got, err := Bandwidth(s, -3); err != nil || !(math.Abs(got-want) <= 1e-6*want) {
			t.Errorf("%s: Bandwidth = %v, %v; want %v (big.Float oracle)", name, got, err, want)
		}
	}
}

// Beside the same cancelled DC zeros, a δ whose term grows with the zero's
// keeps |L| above the level: the scan must not report a crossing there.
func TestBandwidthDipPastDCNoCrossingStaysInf(t *testing.T) {
	tests := []struct {
		name     string
		k        int
		discrete bool
		delta    float64
	}{
		{"s delta 1e-17", 1, false, 1e-17},
		{"s delta -1e-12", 1, false, -1e-12},
		{"(1-z) delta 1e-16", 1, true, 1e-16},
		{"(1-z) delta -1e-10", 1, true, -1e-10},
		{"s^2 delta -1e-9", 2, false, -1e-9},
		{"(1-z)^2 delta -1e-12", 2, true, -1e-12},
		{"s^3 delta 1e-15", 3, false, 1e-15},
		{"(1-z)^3 delta -1e-9", 3, true, -1e-9},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := bandwidthDipModel(t, tc.k, tc.discrete, tc.delta)
			if want := bandwidthDipOracle(sys); !math.IsInf(want, 1) {
				t.Fatalf("oracle %g; the model dips", want)
			}
			if got, err := Bandwidth(sys, -3); err != nil || !math.IsInf(got, 1) {
				t.Errorf("Bandwidth = %v, %v; want +Inf", got, err)
			}
		})
	}
}

// dcPinned's ω bounds |L(jω) − L(0)| by tol below it, which is what lets the
// scan start there: check | |L| − |L(0)| | ≤ tol at and below it in big.Float
// on the dip models, whose low-order DC moments cancel to rounding.
func TestBandwidthDCPinnedBoundHolds(t *testing.T) {
	for _, tc := range []struct {
		k        int
		discrete bool
		delta    float64
	}{{1, false, 1e-12}, {2, false, 1e-17}, {3, false, -1e-17}, {4, true, -1e-7}, {5, true, 1e-16}, {6, false, 3.1622776601683795e-07}} {
		sys := bandwidthDipModel(t, tc.k, tc.discrete, tc.delta)
		loop, err := newRationalLoop(sys)
		if err != nil {
			t.Fatal(err)
		}
		res, ok := loop.dcResolvent()
		h0, _, ok0 := loop.ddDC()
		if !ok || !ok0 {
			t.Fatalf("k=%d: dcResolvent %v, ddDC %v", tc.k, ok, ok0)
		}
		tol := h0 * (1 - math.Pow(10, -3.0/20)) / 4
		w := loop.dcPinned(res, tol)
		dc := new(big.Float).Sqrt(bandwidthDipGainSq(sys, 0))
		for _, f := range []float64{1, 0.5, 0.1} {
			nu := w * f
			if tc.discrete {
				nu = math.Tan(w * f * sys.Dt / 2)
			}
			g := new(big.Float).Sqrt(bandwidthDipGainSq(sys, nu))
			d, _ := g.Sub(g, dc).Float64()
			if math.Abs(d) > tol {
				t.Errorf("k=%d discrete=%v: ||L(%g)| − |L(0)|| = %g > tol %g", tc.k, tc.discrete, w*f, math.Abs(d), tol)
			}
		}
	}
}
