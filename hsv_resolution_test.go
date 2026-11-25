package controlsys

import (
	"errors"
	"math"
	"math/big"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hsvExactLyap solves A·X + X·Aᵀ + Q = 0 (A·X·Aᵀ − X + Q = 0 when
// discrete) by a Kronecker solve in hbPrec bits.
func hsvExactLyap(a, q hbMat, discrete bool) (hbMat, bool) {
	n := a.r
	M := newHbMat(n*n, n*n)
	rhs := newHbMat(n*n, 1)
	p := new(big.Float).SetPrec(hbPrec)
	for i := range n {
		for j := range n {
			row := i*n + j
			rhs.d[row].Neg(q.at(i, j))
			if discrete {
				for k := range n {
					for l := range n {
						c := M.at(row, k*n+l)
						c.Add(c, p.Mul(a.at(i, k), a.at(j, l)))
					}
				}
				c := M.at(row, row)
				c.Sub(c, big.NewFloat(1))
				continue
			}
			for k := range n {
				c := M.at(row, k*n+j)
				c.Add(c, a.at(i, k))
				c = M.at(row, i*n+k)
				c.Add(c, a.at(j, k))
			}
		}
	}
	var x hbMat
	ok := func() (ok bool) {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		x = hbSolve(M, rhs)
		return true
	}()
	if !ok {
		return hbMat{}, false
	}
	for _, v := range x.d {
		if v.IsInf() {
			return hbMat{}, false
		}
	}
	x.r, x.c = n, n
	return x, true
}

// hsvPivChol returns F with W = F·Fᵀ by diagonally pivoted Cholesky in
// hbPrec bits, stopping at pivots below 2⁻²⁸⁰ of the largest diagonal.
func hsvPivChol(w hbMat) hbMat {
	n := w.r
	a := newHbMat(n, n)
	for i := range a.d {
		a.d[i].Set(w.d[i])
	}
	f := newHbMat(n, n)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	toF := func(x *big.Float) float64 { v, _ := x.Float64(); return v }
	dmax := 0.0
	for i := range n {
		dmax = math.Max(dmax, toF(a.at(i, i)))
	}
	floor := math.Ldexp(dmax, -280)
	t := new(big.Float).SetPrec(hbPrec)
	k := 0
	for ; k < n; k++ {
		best, bv := -1, floor
		for i := k; i < n; i++ {
			if v := toF(a.at(perm[i], perm[i])); v > bv {
				best, bv = i, v
			}
		}
		if best < 0 {
			break
		}
		perm[k], perm[best] = perm[best], perm[k]
		piv := perm[k]
		s := new(big.Float).SetPrec(hbPrec).Sqrt(a.at(piv, piv))
		for i := k; i < n; i++ {
			f.at(perm[i], k).Quo(a.at(perm[i], piv), s)
		}
		for i := k + 1; i < n; i++ {
			for j := k + 1; j < n; j++ {
				c := a.at(perm[i], perm[j])
				c.Sub(c, t.Mul(f.at(perm[i], k), f.at(perm[j], k)))
			}
		}
	}
	o := newHbMat(n, max(k, 1))
	for i := range n {
		for j := range k {
			o.at(i, j).Set(f.at(i, j))
		}
	}
	return o
}

// hsvExactReference returns the Hankel singular values of the stored
// explicit model from gramians whose right-hand sides, solves, factors and
// factor product are all in hbPrec bits, and T = √(‖Wc‖_F·‖Wo‖_F).
func hsvExactReference(sys *System) (hsv []float64, T float64, ok bool) {
	n, _, _ := sys.Dims()
	disc := sys.Dt > 0
	b, c := hbFrom(sys.B), hbFrom(sys.C)
	wc, ok1 := hsvExactLyap(hbFrom(sys.A), hbMul(b, b.t()), disc)
	wo, ok2 := hsvExactLyap(hbFrom(mat.DenseCopyOf(sys.A.T())), hbMul(c.t(), c), disc)
	if !ok1 || !ok2 {
		return nil, 0, false
	}
	fro := func(m hbMat) float64 {
		s := new(big.Float).SetPrec(hbPrec)
		p := new(big.Float).SetPrec(hbPrec)
		for _, v := range m.d {
			s.Add(s, p.Mul(v, v))
		}
		f, _ := s.Float64()
		return math.Sqrt(f)
	}
	T = math.Sqrt(fro(wc) * fro(wo))
	g := hbDense(hbMul(hsvPivChol(wo).t(), hsvPivChol(wc)))
	if mat.Norm(g, math.Inf(1)) == 0 {
		return make([]float64, n), T, true
	}
	var svd mat.SVD
	if !svd.Factorize(g, mat.SVDNone) {
		return nil, 0, false
	}
	hsv = svd.Values(nil)
	for len(hsv) < n {
		hsv = append(hsv, 0)
	}
	slices.Sort(hsv)
	slices.Reverse(hsv)
	return hsv[:n], T, true
}

// hsvContractViolation returns the largest ratio of |h[i] − ref[i]| to its
// contract bar: τ·σ₁ for σ₁, max(τ·σ₁, 4·√n·2⁻⁵⁰·T) for the others. A
// value of at most 1 is within the contract.
func hsvContractViolation(h, ref []float64, T float64) float64 {
	if len(h) != len(ref) {
		return math.Inf(1)
	}
	n := float64(len(ref))
	worst := 0.0
	for i := range ref {
		bar := hsvTau * ref[0]
		if i > 0 {
			bar = math.Max(bar, 4*math.Sqrt(n)*0x1p-50*T)
		}
		d := math.Abs(h[i] - ref[i])
		if d == 0 {
			continue
		}
		worst = math.Max(worst, d/bar)
	}
	return worst
}

func hsvLiteral(n, m, p int, a, b, c []float64, dt float64) *System {
	sys, err := NewFromSlices(n, m, p, a, b, c, make([]float64, p*m), dt)
	if err != nil {
		panic(err)
	}
	return sys
}

func TestHSV_ExactlyZeroTransferFunction(t *testing.T) {
	cancelA, cancelB, cancelC := zeroCancelSystem(false)
	ulpA, ulpB, ulpC := zeroCancelSystem(true)
	structural := hsvLiteral(3, 2, 1,
		[]float64{-1, 0.5, 0.25, 0, -2, 0, 0, 0.75, -3},
		[]float64{1, -2, 0, 0, 0, 0},
		[]float64{0, 1, 4}, 0)
	permuted := hsvLiteral(4, 1, 1,
		[]float64{-2, 0, 0.5, 0, 0, -3, 0, 1, 0.25, 0, -1, 0, 0, -0.5, 0, -4},
		[]float64{1, 0, -1, 0},
		[]float64{0, 2, 0, 1}, 0)
	cancel, _ := New(cancelA, cancelB, cancelC, mat.NewDense(1, 1, nil), 0)
	cancelDisc := hsvLiteral(2, 1, 1, []float64{-0.75, 0.25, -0.25, -0.25}, []float64{1, 1}, []float64{1, -1}, 0.1)
	for _, c := range []struct {
		name string
		sys  *System
	}{{"structural", structural}, {"permuted blocks", permuted}, {"cancellation", cancel}, {"cancellation discrete", cancelDisc}} {
		h, err := HSV(c.sys)
		if err != nil {
			t.Fatalf("%s: HSV err %v, want nil", c.name, err)
		}
		n, _, _ := c.sys.Dims()
		if len(h) != n || slices.ContainsFunc(h, func(v float64) bool { return v != 0 }) {
			t.Fatalf("%s: HSV = %v, want %d exact zeros", c.name, h, n)
		}
	}
	ulp, _ := New(ulpA, ulpB, ulpC, mat.NewDense(1, 1, nil), 0)
	h, err := HSV(ulp)
	if err == nil && !slices.ContainsFunc(h, func(v float64) bool { return v != 0 }) {
		t.Fatalf("one ulp from a zero transfer function: HSV = %v with nil error, want a nonzero value or ErrAccuracyUnverified", h)
	}
	ref, T, ok := hsvExactReference(ulp)
	if !ok || !(ref[0] > 0) {
		t.Fatalf("one-ulp model: exact reference %v, ok %v", ref, ok)
	}
	if err == nil && hsvContractViolation(h, ref, T) > 1.0/8 {
		t.Fatalf("one-ulp model: nil HSV %v off the exact %v beyond the contract", h, ref)
	}
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("one-ulp model: HSV err %v, want nil or ErrAccuracyUnverified", err)
	}
}

// hsvCase is a frozen model with its 320-bit reference Hankel singular
// values (right-hand sides, gramians, factors and their product in 320
// bits) and T = √(‖Wc‖_F·‖Wo‖_F).
type hsvCase struct {
	name    string
	n, m, p int
	dt      float64
	a, b, c []float64
	ref     []float64
	T       float64
}

func (c hsvCase) sys() *System { return hsvLiteral(c.n, c.m, c.p, c.a, c.b, c.c, c.dt) }

// Rotated decoupled halves joined by a coupling 2⁻ᵏ, σ₁/T from 1.3·10⁻⁴ down to
// 7.7·10⁻¹¹ (D5BGNA sweep rows sw1k12r1/161, sw0k25r1/666, sw0k31r1/94,
// sw0k20r1/370, sw0k29r1/486).
var hsvWeakCoupling = []hsvCase{
	{name: "sw1k12r1/161 (arm64)", n: 5, m: 2, p: 2, dt: 0.5,
		a:   []float64{-0.20718744159779862, 0.01316994320328305, 0.14985978060656074, 0.04671614595137693, 0.008237280253766702, -0.05595201870259408, -0.11334664574049454, -0.08805131095029504, 0.0003120996293248345, -0.06959421977693793, -0.07576530723463347, 0.09659890252448208, -0.14353010278892522, 0.17910932092437293, 0.08991786342915886, -0.11244487375129604, 0.03556642783098272, -0.13079370464878812, -0.18580760747321395, 0.06319956475955732, -0.057673885929122475, -0.0704328337390896, -0.08893075805571024, 0.027519690928456984, -0.16262820239956763},
		b:   []float64{-1.5821382021369639, 0.19399106622451537, -3.8490710524950615, 0.28403370152515717, 0.5506020205884404, -0.07186769928014483, 0.2632561323427288, -0.00549975735526256, 1.519547421402581, -0.03869249400342974},
		c:   []float64{0.5387800533084519, -1.8472126738648598, -4.485625689860518, 1.1052741969479092, -2.6842228878568357, -0.0025862684479340814, -0.729939838020257, -2.3362922353507996, 0.859936109991954, -1.1540925534193291},
		ref: []float64{0.0038944322200497655, 0.0032099041761435997, 5.8678765801458789e-05, 2.8127144586005824e-05, 1.0136651350955699e-06}, T: 29.080938778671985},
	{name: "sw0k25r1/666 (arm64)", n: 3, m: 2, p: 2, dt: 0,
		a:   []float64{-0.698919397531697, -0.11626789701259527, -0.2156894928764762, -0.11626790153331501, -0.3566787447940287, -0.2173909733550246, -0.21568945848788024, -0.21739094941041343, -0.694401857674274},
		b:   []float64{2.5242414232086237, -0.6310603558021559, 1.3931293525638309, -0.3482823381409577, 2.772615343747268, -0.693153835936817},
		c:   []float64{3.2137913079424463, -4.615559211770078, -0.6067607367306769, 1.4375285664278268, 3.111805617499079, -2.8723122078919507},
		ref: []float64{1.5184819074287478e-06, 2.5443095034471725e-07, 9.5316381092884823e-08}, T: 24.170003859915191},
	{name: "sw0k31r1/94 (arm64)", n: 3, m: 2, p: 1, dt: 0,
		a:   []float64{-0.49999999990020727, -3.7316118728875964, -0.9137243751370522, 3.731611872737284, -0.49999999994385064, -1.1136339600225336, 0.913724375424903, 1.113633959831405, -0.500000000155942},
		b:   []float64{1.1136339597932192, 1.1136339597932192, -0.9137243754714407, -0.9137243754714407, 3.7316118728057175, 3.7316118728057175},
		c:   []float64{-2.935354620884492, 2.315961434198887, 1.4430924727665568},
		ref: []float64{1.4718705547069476e-09, 1.1637781387169172e-09, 1.4598913732199408e-10}, T: 19.137345841101752},
	{name: "sw0k20r1/370 (arm64)", n: 3, m: 2, p: 2, dt: 0,
		a:   []float64{-0.8417128321959634, 0.08756468262139286, 0.2154576481635903, 0.08756414625280498, -0.9515593918192237, 0.11919074222389893, 0.21545408172147584, 0.1191894993584154, -0.7067277759848132},
		b:   []float64{0.37989157109530947, 1.5195662843812379, 0.0852061495228472, 0.3408245980913888, -0.31372329574678987, -1.2548931829871595},
		c:   []float64{-0.2221155043143563, 0.15591581890014927, -0.22661632809635546, -2.0877042871102143, -1.7124932050929795, -2.9931351175802274},
		ref: []float64{3.7417885990937426e-05, 5.8997072997878778e-06, 4.9809047715780996e-18}, T: 5.8612652663520004},
	{name: "sw0k29r1/486 (arm64)", n: 2, m: 1, p: 2, dt: 0,
		a:   []float64{-0.6889019094220478, -0.24241910770248526, -0.24241910583984005, -0.8110980905779525},
		b:   []float64{0.15366436986130116, 0.19719853304507426},
		c:   []float64{0.788794132180297, -0.6146574794452047, 1.577588264360594, -1.2293149588904093},
		ref: []float64{1.236156020721345e-09, 1.9490568496138203e-10}, T: 0.39528470752104738},
}

// σ₁/T below 10⁻¹⁶: census rows rlrep/23, rlrepd/591, rlrepd/746, rlrepd/23
// and the non-minimal nm0k1/14 (D5BGNA).
var hsvBelowResolution = []hsvCase{
	{name: "rlrep/23 (arm64)", n: 3, m: 1, p: 1, dt: 0,
		a:   []float64{-1.0000000000000004, -1.0447196082361515e-16, 3.853544548533148e-17, -1.0447196082361515e-16, -0.9999999999999999, -6.186480626824452e-17, 3.853544548533148e-17, -6.186480626824452e-17, -1},
		b:   []float64{0, 0, -1},
		c:   []float64{1, 0.5, 0},
		ref: []float64{4.5888419898965716e-18, 7.8732081429196909e-19, 1.6770041232182757e-63}, T: 0.55901699437494734},
	{name: "rlrepd/591 (arm64)", n: 3, m: 1, p: 1, dt: 0.5,
		a:   []float64{0.2500000000000001, -1.3242652989672499e-17, 4.0449049961017614e-17, -1.3242652989672499e-17, 0.25000000000000006, 1.1429896905378628e-18, 4.0449049961017614e-17, 1.1429896905378628e-18, 0.25},
		b:   []float64{0.5, 1, -1},
		c:   []float64{0, 2, 2},
		ref: []float64{2.014322425433386e-16, 1.2279547130985336e-16, 1.5698248019915217e-68}, T: 4.5254833995939041},
	{name: "rlrepd/746 (arm64)", n: 2, m: 1, p: 1, dt: 0.5,
		a:   []float64{0.24999999999999994, -5.5927857926893454e-18, -5.5927857926893454e-18, 0.24999999999999994},
		b:   []float64{-1, 0.5},
		c:   []float64{0.5, 1},
		ref: []float64{6.1125189029122795e-18, 3.726263631364825e-18}, T: 1.3333333333333333},
	{name: "rlrepd/23 (amd64)", n: 3, m: 1, p: 1, dt: 0.5,
		a:   []float64{0.5000000000000002, 5.551115123125783e-17, -1.3877787807814457e-17, 5.551115123125783e-17, 0.49999999999999994, 2.7755575615628914e-17, -1.3877787807814457e-17, 2.7755575615628914e-17, 0.5},
		b:   []float64{0, 0, -1},
		c:   []float64{1, 0.5, 0},
		ref: []float64{1.6432366544640058e-32, 7.3958605019482631e-33, 2.1481537824903775e-33}, T: 1.49071198499986},
	{name: "nm0k1/14 (arm64)", n: 7, m: 1, p: 1, dt: 0,
		a:   []float64{-1.9814068900256796, -0.5610082257947525, 0.5848336446102131, 0.7248951384372907, 0.02371920927334836, 0.15364168666248892, -0.05381946683007374, 0.26442403841761014, -1.3429437275394445, -0.20986291440108726, 0.5829054068114948, -0.31707010262868845, -0.868706731067122, 1.2333473053859818, -0.9371831373739978, 0.47417892418721663, -1.8032472054564201, 0.5834739710667346, -1.0967257258049488, 0.9139177901426102, -0.1462847117982208, 0.0639629817638529, 1.2418883788394812, -0.08390713180431678, -1.4666612145150064, 0.5526597117147836, -0.12056617104293198, -1.040839522541043, -0.30276830252588866, -0.4951528135385202, -0.49779591185465855, 0.9559033399015017, -1.4277928739649546, 0.7240960576822717, 0.519747996155457, -0.40623161671553765, 0.5204712308787796, -1.1836200485038746, -0.7859210848068077, 0.05383753271223692, -1.559336964680076, 0.11720069540977046, -0.016797239216044507, 0.48181158198237883, 1.2087934443755055, -0.46863374699775645, 0.8249941779970795, 0.5646442245495465, -1.6686111238184163},
		b:   []float64{0.6010379451059595, -0.3169033873893937, -1.8577021800882436, 1.795021011070716, 3.0679810849717097, 0.6555123581125627, -0.292341276817864},
		c:   []float64{-0.5719202171905547, -0.8337613822181006, 1.1436611986783043, -0.10924499602178711, 0.3762754489404086, 1.5607892469922802, -0.7617135479760997},
		ref: []float64{5.2787289199694992e-16, 1.7372943933823954e-16, 6.0148640669600457e-17, 1.2486804041683301e-17, 2.1828005227426801e-19, 4.0566779970340303e-21, 6.5489661358828603e-50}, T: 6.4186695381347461},
}

// Balreal literals (D5BGNA): rlrep/23 as stored on amd64 (σ₁/T ≈ 10⁻³²,
// certificate passes), the structure-exact sw0k43r0/164 (σ₁/T ≈ 10⁻¹³)
// and sw0k34r1/260 just above the floor (σ₁/T ≈ 2.7·10⁻¹¹).
var balrealResolution = []hsvCase{
	{name: "rlrep/23 (amd64)", n: 3, m: 1, p: 1, dt: 0,
		a:   []float64{-1.0000000000000004, -1.1102230246251565e-16, 2.7755575615628914e-17, -1.1102230246251565e-16, -0.9999999999999999, -5.551115123125783e-17, 2.7755575615628914e-17, -5.551115123125783e-17, -1},
		b:   []float64{0, 0, -1},
		c:   []float64{1, 0.5, 0},
		ref: []float64{7.1290116842757277e-33, 1.9158123218266109e-33, 1.7940448183514191e-34}, T: 0.55901699437494734},
	{name: "sw0k43r0/164 (arm64)", n: 2, m: 2, p: 2, dt: 0,
		a:   []float64{-0.5, 0, 5.684341886080802e-14, -1},
		b:   []float64{0.5, -0.25, 0, 0},
		c:   []float64{0, 2, 0, 0.5},
		ref: []float64{7.7771021017951008e-14, 1.2262217606483436e-14}, T: 0.81490030065033114},
	{name: "sw0k34r1/260 (arm64)", n: 4, m: 1, p: 1, dt: 0,
		a:   []float64{-1.1842193867284352, 0.4699912892895372, 0.13060147863004137, 0.5611564356194776, 0.46999128933468803, -1.518681360736717, -0.32666045182421083, -1.4830400198356182, 0.13060147866409638, -0.3266604518657825, -0.605615502647272, -0.4808621148178207, 0.5611564355973969, -1.4830400196495797, -0.48086211469783147, -2.691483749887575},
		b:   []float64{0.4208264979972426, -0.31253529667274776, -0.24608040245583945, 0.40579697195053877},
		c:   []float64{-0.82338288194468, -1.0791252023578715, 3.6334408371514004, 2.226126031984059},
		ref: []float64{6.5356865581503952e-11, 5.4283138508113641e-12, 9.8184483085583821e-14, 5.5442589326583403e-24}, T: 2.4340457946366301},
}

// gr0g24/4 (D5BGNA): a minimal rotated model under a power-of-two state
// scaling up to 2^±24, then its unscaled original.
var hsvGraded = []hsvCase{
	{name: "gr0g24/4 (arm64)", n: 3, m: 2, p: 2, dt: 0,
		a:   []float64{-2.338807409364855, 3.8523725064968006e+06, -81.99325389160083, 9.276545396779059e-07, -1.7405934386283748, -6.50036854487696e-06, 4.0053224149589234e-05, 10638.872735992587, -0.17059915200677073},
		b:   []float64{-197.35277256260065, 328.56547977595403, -3.21517091264195e-05, 0.00011935158145737104, -1.6109950884567266, -10.855280907014818},
		c:   []float64{-0.00687853242102684, -17887.40112312936, -1.5160766504570127, 0.00024071914294797922, 513.4196228350397, 1.1241412865460647},
		ref: []float64{60.083599479807361, 46.131279414195134, 0.02146334020878319}, T: 64850695.162574053},
	{name: "gr0g24/4-orig (arm64)", n: 3, m: 2, p: 2, dt: 0,
		a:   []float64{-2.338807409364855, 1.836954358337784, -0.6405722960281315, 1.9454325731945996, -1.7405934386283748, -0.10650203823926412, 0.005126812691147422, 0.6493452597651725, -0.17059915200677073},
		b:   []float64{-0.7709092678226588, 1.2834589053748204, -0.26338680116362856, 0.9777281552987835, -0.8054975442283633, -5.427640453507409},
		c:   []float64{-1.760904299782871, -2.183520644913252, -3.0321533009140254, 0.06162410059468268, 0.06267329380310543, 2.2482825730921294},
		ref: []float64{60.083599479807361, 46.131279414195134, 0.021463340208783193}, T: 78.070025716353229},
}

// nm1k4/189 (D5BGNA): Lyap's observability gramian is not Cholesky-factorable
// although ε·T²/σ̃₁² is within γₙ.
var hsvCholeskyFailed = []hsvCase{
	{name: "nm1k4/189 (arm64)", n: 2, m: 1, p: 1, dt: 0.5,
		a:   []float64{-0.3915985425186051, 0.12389305221140544, 0.12389305221140544, -0.35840145748139474},
		b:   []float64{0.09410410357022603, -1.411079167761768},
		c:   []float64{8.761319747031631e-11, -7.665803796795324e-11},
		ref: []float64{1.5522042910257967e-10, 1.3346636678786587e-27}, T: 2.0611879777089731e-10},
}

func TestHSV_WeakCouplingMatchesExact(t *testing.T) {
	for _, c := range hsvWeakCoupling {
		h, err := HSV(c.sys())
		if err != nil {
			t.Fatalf("%s: HSV err %v, want nil", c.name, err)
		}
		if v := hsvContractViolation(h, c.ref, c.T); v > 1.0/8 {
			t.Fatalf("%s: HSV %v off the 320-bit %v by %.3g of the contract bar, want ≤ 1/8", c.name, h, c.ref, v)
		}
	}
}

func TestHSV_BelowResolutionUnverified(t *testing.T) {
	for _, c := range hsvBelowResolution {
		h, err := HSV(c.sys())
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("%s: HSV = %v, err %v, want ErrAccuracyUnverified", c.name, h, err)
		}
		if len(h) != c.n {
			t.Fatalf("%s: %d values, want %d", c.name, len(h), c.n)
		}
		if lim := 64 * math.Sqrt(float64(c.n)) * 0x1p-50 * c.T; !(h[0] <= lim) {
			t.Fatalf("%s: σ̃₁ = %.3g above 64·√n·2⁻⁵⁰·T = %.3g", c.name, h[0], lim)
		}
	}
}

func TestHSV_CertifiedPlainPathUnchanged(t *testing.T) {
	sys := benchSys(t, 10, 2, 3)
	golden := []float64{21.565906565489989, 2.0726713669476391, 0.37499852643977816, 0.0094290682231784918, 6.7210076388549229e-05,
		2.7119254717246375e-07, 7.2829794625773567e-10, 1.3868118926738466e-12, 1.9080809839049438e-15, 1.3575076904510764e-18}
	h, route, err := hsvRouted(sys)
	if err != nil || route != hsvRoutePlain {
		t.Fatalf("HSV route %d, err %v, want the plain path with nil", route, err)
	}
	policy := newEnergyAnalysisPolicy(sys)
	Wc, eB, _ := policy.scaledGramian(GramControllability)
	Wo, eC, _ := policy.scaledGramian(GramObservability)
	plain, chol, err := hsvFromGramians(Wc, Wo, 10, eB+eC)
	if err != nil || !chol || !slices.Equal(h, plain) {
		t.Fatalf("HSV %v not bit-equal to the plain-path %v (chol %v, err %v)", h, plain, chol, err)
	}
	for i, g := range golden {
		if math.Abs(h[i]-g) > hsvTau*golden[0]/8 {
			t.Fatalf("HSV[%d] = %.17g, golden %.17g", i, h[i], g)
		}
	}
}

// hsvUncontrollableState has a decoupled state that B does not reach, so
// the controllability gramian is exactly singular while σ₁/T is O(1).
var hsvUncontrollableState = hsvCase{name: "exactly uncontrollable state", n: 3, m: 1, p: 1,
	a: []float64{-1, 0.5, 0, -0.25, -2, 0, 0, 0, -3},
	b: []float64{1, 0.5, 0},
	c: []float64{1, 2, 1}}

func TestHSV_PlainPathRequiresBothCholeskyFactors(t *testing.T) {
	structural := hsvUncontrollableState
	ref, T, ok := hsvExactReference(structural.sys())
	if !ok {
		t.Fatal("exact reference of the uncontrollable-state model failed")
	}
	structural.ref, structural.T = ref, T
	for _, c := range append([]hsvCase{structural}, hsvCholeskyFailed...) {
		sys := c.sys()
		policy := newEnergyAnalysisPolicy(sys)
		Wc, eB, _ := policy.scaledGramian(GramControllability)
		Wo, eC, _ := policy.scaledGramian(GramObservability)
		plain, chol, err := hsvFromGramians(Wc, Wo, c.n, eB+eC)
		if err != nil || chol {
			t.Fatalf("%s: plain path chol %v, err %v, want a failed Cholesky factorization", c.name, chol, err)
		}
		t2 := math.Sqrt(mat.Norm(Wc, 2)) * math.Sqrt(mat.Norm(Wo, 2))
		if !hsvPlainResolved(c.n, t2, math.Ldexp(plain[0], -(eB+eC))) {
			t.Fatalf("%s: ε·T²/σ̃₁² above γₙ; the literal no longer isolates the Cholesky requirement", c.name)
		}
		h, route, err := hsvRouted(sys)
		if err != nil || route != hsvRouteCore {
			t.Fatalf("%s: HSV route %d, err %v, want the refined path with nil", c.name, route, err)
		}
		if v := hsvContractViolation(h, c.ref, c.T); v > 1.0/8 {
			t.Fatalf("%s: HSV %v off the 320-bit %v by %.3g of the contract bar", c.name, h, c.ref, v)
		}
	}
}

func TestHSV_StateScalingKeepsValues(t *testing.T) {
	scaled, orig := hsvGraded[0], hsvGraded[1]
	hs, errS := HSV(scaled.sys())
	ho, errO := HSV(orig.sys())
	for _, r := range []struct {
		c   hsvCase
		h   []float64
		err error
	}{{scaled, hs, errS}, {orig, ho, errO}} {
		if r.err != nil {
			t.Fatalf("%s: HSV err %v, want nil", r.c.name, r.err)
		}
		if v := hsvContractViolation(r.h, r.c.ref, r.c.T); v > 1.0/8 {
			t.Fatalf("%s: HSV %v off the 320-bit %v by %.3g of the contract bar", r.c.name, r.h, r.c.ref, v)
		}
	}
	for i := range hs {
		if d := math.Abs(hs[i] - ho[i]); d > hsvTau*ho[0]/8 {
			t.Fatalf("HSV[%d]: scaled %.17g, unscaled %.17g", i, hs[i], ho[i])
		}
	}
}
