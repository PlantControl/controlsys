package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descTolUnitOK is the normwise rounding accuracy of a descriptor's
// response that Stabsep and Modsep's SepTol is relative to on the
// generalized Schur route: u·(‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sE − A)⁻¹, u = 4(n+5)ε, Frobenius norms of the stored data. ok is
// false where sE − A is singular or the contract does not apply,
// η = u(‖A‖ + |s|‖E‖)‖M(s)‖ > 0.1.
func descTolUnitOK(sys *System, s complex128) (float64, bool) {
	n, _, _ := sys.Dims()
	K := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			e := sys.E.At(i, j)
			re := real(s)*e - sys.A.At(i, j)
			im := imag(s) * e
			K.Set(i, j, re)
			K.Set(n+i, n+j, re)
			K.Set(i, n+j, -im)
			K.Set(n+i, j, im)
		}
	}
	var inv mat.Dense
	if err := inv.Inverse(K); err != nil {
		return 0, false
	}
	mf := mat.Norm(&inv, 2) / math.Sqrt2
	fro := func(m *mat.Dense) float64 {
		if r, c := m.Dims(); r == 0 || c == 0 {
			return 0
		}
		return mat.Norm(m, 2)
	}
	a, e, b, c, d := fro(sys.A), fro(sys.E), fro(sys.B), fro(sys.C), fro(sys.D)
	u := 4 * float64(n+5) * 0x1p-52
	pen := a + cmplx.Abs(s)*e
	if u*pen*mf > 0.1 || math.IsInf(mf, 0) || math.IsNaN(mf) {
		return 0, false
	}
	return u * (c*mf*pen*mf*b + c*mf*b + d), true
}

// descTolRatio is the largest ‖G − G1 − G2‖_F / descTolUnit over pts, the
// responses evaluated in 320 bits, and how many points were scored.
func descTolRatio(t *testing.T, sys, g1, g2 *System, pts []complex128) (float64, int) {
	t.Helper()
	worst, scored := 0.0, 0
	for _, s := range pts {
		unit, ok := descTolUnitOK(sys, s)
		if !ok {
			continue
		}
		g, a, b := sepEval320(t, sys, s), sepEval320(t, g1, s), sepEval320(t, g2, s)
		var d l2
		for k := range g {
			d.addC(g[k] - a[k] - b[k])
		}
		worst = math.Max(worst, d.norm()/unit)
		scored++
	}
	return worst, scored
}

// descRequireNilWithinBar fails unless err is nil and G1 + G2 is within the
// SepTol bar at pts.
func descRequireNilWithinBar(t *testing.T, name string, sys, g1, g2 *System, err error, pts []complex128) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	r, scored := descTolRatio(t, sys, g1, g2, pts)
	if scored == 0 {
		t.Fatalf("%s: no point scored", name)
	}
	if !(r <= decouplingSepTol) {
		t.Errorf("%s: ‖G − G1 − G2‖ = %.3g SepTol units", name, r)
	}
}

// descRequireSchurFailedOrWithinBar accepts ErrSchurFailed, or nil with
// G1 + G2 within the SepTol bar: the split is never silently wrong.
func descRequireSchurFailedOrWithinBar(t *testing.T, name string, sys, g1, g2 *System, err error, pts []complex128) {
	t.Helper()
	if errors.Is(err, ErrSchurFailed) {
		return
	}
	descRequireNilWithinBar(t, name, sys, g1, g2, err, pts)
}

// descAugment is the explicit sys with one more state: E = diag(I, 0),
// A = diag(A, a), and a row and column of ones in B and C (aug0); with
// rotate it is U·(A, E)·V, U·B, C·V for seeded random orthogonal U, V (augQ).
func descAugment(t *testing.T, sys *System, a, e float64, rotate bool, seed uint64) *System {
	t.Helper()
	n, m, p := sys.Dims()
	N := n + 1
	A := mat.NewDense(N, N, nil)
	E := mat.NewDense(N, N, nil)
	B := mat.NewDense(N, m, nil)
	C := mat.NewDense(p, N, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.A)
	B.Slice(0, n, 0, m).(*mat.Dense).Copy(sys.B)
	C.Slice(0, p, 0, n).(*mat.Dense).Copy(sys.C)
	for i := range n {
		E.Set(i, i, 1)
	}
	A.Set(n, n, a)
	E.Set(n, n, e)
	for j := range m {
		B.Set(n, j, 1)
	}
	for i := range p {
		C.Set(i, n, 1)
	}
	if rotate {
		rng := rand.New(rand.NewPCG(seed, 0xe0a7))
		u := mat.NewDense(N, N, sepRandOrthogonal(rng, N))
		v := mat.NewDense(N, N, sepRandOrthogonal(rng, N))
		var t1, a2, e2, b2, c2 mat.Dense
		t1.Mul(u, A)
		a2.Mul(&t1, v)
		t1.Reset()
		t1.Mul(u, E)
		e2.Mul(&t1, v)
		b2.Mul(u, B)
		c2.Mul(C, v)
		A, E, B, C = &a2, &e2, &b2, &c2
	}
	d, err := NewDescriptor(A, B, C, mat.DenseCopyOf(sys.D), E, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// descTied717 is corpus row tied/717 (PZET7B): a 7-state discrete model
// with near-defective clusters; Modsep at cutoff 1.3486991523489157e-12
// had ‖X‖_F ≈ 1.5e36.
func descTied717(t *testing.T) *System {
	return sepLit(t, 7, 1, 1, []float64{0.99999999999909051, 1, 0, 0, 0, -0.5, 0, 0, 0.99999999999909051, 0, 0, 0, 0.5, 1, 0, 0, -1, 0, 0, 0, 0.5, -0.5, 0, 0, 1, 0, 0, 1, 0, 1, -1, 0, 0.99999999999909051, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
		[]float64{-1, 0.5, -1, 2, 2, -1, 0.5}, []float64{2, 2, 2, -1, 2, 1, 1}, []float64{-1}, 0.5)
}

// descRotZ880 is corpus row rotZ/880 (AST4BK): a near-defective pair at
// ±0.5i split across Modsep's cutoff 0.50000000000000033.
func descRotZ880(t *testing.T) *System {
	return sepLit(t, 6, 1, 1, []float64{-3.637978807091713e-12, 0.5, 0, 1, -1, -0.5, 0, -1.4901161193847656e-08, 0, -0.5, -0.5, 0, 0, 1, -2, 0, -0.5, 0, 0, 0.5, 0, -1.4901161193847656e-08, 0.5, 0, 0, 0, 0, 0, -3.814697265625e-06, 0, 0.5, 0, 0, 1, 0, -3.637978807091713e-12},
		[]float64{0.5, -1, 1, 2, 1, -1}, []float64{1, 0, 0.5, 1, 1, 2}, []float64{0}, 0)
}

// descTied703 is corpus row tied/703 (AST4BK, XVYTBG S2): Stabsep split a
// near-defective cluster at |z| = 1 with ‖X‖_F ≈ 7e10.
func descTied703(t *testing.T) *System {
	return sepLit(t, 7, 2, 2, []float64{1.443190524382989, -0.050485035806474447, -0.26634193994854477, 0.85828945017646341, 0.64795477158400283, 0.36033784085867349, -0.15052878825209876, 1.1382397478984099, -0.60247796032727485, 1.0152146153729023, -0.29575290259434656, -0.72607595010588533, 0.3604816147716407, 1.165848964048005, -0.048998573615816991, 0.92762804129425758, 0.27556973726771189, -1.1314809441056688, 0.021955870293289607, -0.50252492920474778, -0.26156874993868223, 0.22260189466842425, -0.13566173328675235, -0.19213295264304947, 0.11927520848274449, -0.39221504165141591, -0.39308105473485122, 0.36197393681440709, 0.27131412148984024, -0.21002539679009077, 0.77167478395293376, -0.81549087903040496, -0.01029169434313223, -0.3175450711766804, -0.29425673968802485, 0.27632811597450202, -0.70522663918508066, 0.35905746739864702, 0.13201226335371194, -0.53919357479420549, 0.78777643320963497, 0.015474066324836657, 0.63985060030405638, -0.22795638728506321, -0.23751839695644258, -0.3014547078037671, -0.1796510261929623, -0.2833632135826985, 0.98695775132460106},
		[]float64{2, 0.5, 1, 0.5, 0.5, -1, 2, -1, 1, -1, -1, 0, 0.5, 1},
		[]float64{1, 0.5, 1, 0.5, -1, 1, 0, 0, -1, 1, 0, 2, 1, 0.5},
		[]float64{0, 0, 0, 0}, 0.5)
}

func descPoints(sys *System) []complex128 {
	if sys.IsDiscrete() {
		return sepPointsD
	}
	return sepPointsC
}

// TestModsepDescriptorIllConditionedDecouplingErrors: the PZET7B/AST4BK
// corpus rows with one algebraic state appended (E = diag(I, 0), EUNKOP),
// split by the generalized Schur form, returned nil with ‖G − G1 − G2‖ up
// to 6e30 SepTol units of the descriptor bar (tied/717; rotZ/880 1.5e4).
// tied/717 has so large a decoupling that it errors on every arch through
// both tiers; for the others either an error or a split within the bar is
// accepted (their literals are ulp-fragile).
func TestModsepDescriptorIllConditionedDecouplingErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		sys    *System
		cut    float64
		strict bool
	}{
		{"tied/717 aug0", descAugment(t, descTied717(t), 1, 0, false, 0), 1.3486991523489157e-12, true},
		{"tied/717 augQ", descAugment(t, descTied717(t), 1, 0, true, 717), 1.3486991523489157e-12, false},
		{"rotZ/880 aug0", descAugment(t, descRotZ880(t), 1, 0, false, 0), 0.50000000000000033, false},
		{"rotZ/880 augQ", descAugment(t, descRotZ880(t), 1, 0, true, 880), 0.50000000000000033, false},
	} {
		res, err := Modsep(tc.sys, tc.cut)
		if tc.strict {
			if !errors.Is(err, ErrSchurFailed) {
				t.Errorf("%s: Modsep err = %v, want ErrSchurFailed", tc.name, err)
			}
			continue
		}
		var g1, g2 *System
		if err == nil {
			g1, g2 = res.Slow, res.Fast
		}
		descRequireSchurFailedOrWithinBar(t, tc.name, tc.sys, g1, g2, err, descPoints(tc.sys))
	}
}

// TestStabsepDescriptorIllConditionedDecouplingErrors: tied/703 with an
// appended algebraic state (EUNKOP) returned nil from Stabsep with
// ‖G − G1 − G2‖ ≈ 4e6 descriptor SepTol units; tied/717 rotated (augQ)
// 3.7e5. Either an error or a split within the bar is accepted.
func TestStabsepDescriptorIllConditionedDecouplingErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		sys  *System
	}{
		{"tied/703 aug0", descAugment(t, descTied703(t), 1, 0, false, 0)},
		{"tied/703 augQ", descAugment(t, descTied703(t), 1, 0, true, 703)},
		{"tied/717 augQ", descAugment(t, descTied717(t), 1, 0, true, 717)},
	} {
		res, err := Stabsep(tc.sys)
		var g1, g2 *System
		if err == nil {
			g1, g2 = res.Stable, res.Unstable
		}
		descRequireSchurFailedOrWithinBar(t, tc.name, tc.sys, g1, g2, err, descPoints(tc.sys))
	}
}

// TestStabsepDescriptorIllConditionedFoldStaysNil: rotZ/880 with a stable
// fast mode −1e8 in E = diag(I, 1e-8) (EUNKOP fastE): Stabsep folds T11
// with that mode into the explicit Stable part. The fold residual's
// rounding is bounded componentwise; a normwise ε‖T11‖‖Â1‖ rejected this
// accurate split at ζ ≈ 7e5.
func TestStabsepDescriptorIllConditionedFoldStaysNil(t *testing.T) {
	sys := descAugment(t, descRotZ880(t), -1, 1e-8, false, 0)
	res, err := Stabsep(sys)
	var g1, g2 *System
	if err == nil {
		g1, g2 = res.Stable, res.Unstable
	}
	descRequireNilWithinBar(t, "rotZ/880 fastE", sys, g1, g2, err, sepPointsC)
}

// TestStabsepDescriptorBadlyScaledPencilStaysNil: the XACVXD pencil
// (TestStabsep_DescriptorRejectedQZReorder) has a genuine coupling
// ‖R‖ ≈ 1.8e4 and blocks of mixed scale; bounding the diagonal blocks of
// (sT − S)⁻¹ by ‖M‖ rejected every seed at ζ ≈ 35 though the split is
// within 1e-5 units. The θ rule and the ridge bounds certify it (EUNKOP
// ruling 2), and the sum is within the bar on a sweep of s.
func TestStabsepDescriptorBadlyScaledPencilStaysNil(t *testing.T) {
	var pts []complex128
	for k := -8; k <= 16; k++ {
		r := math.Pow(10, float64(k)/2)
		for _, th := range []float64{0.3, 1.1, 2.4} {
			pts = append(pts, cmplx.Rect(r, th))
		}
	}
	for seed := range uint64(5) {
		sys := xacvxdDescriptor(t, seed)
		res, err := Stabsep(sys)
		var g1, g2 *System
		if err == nil {
			g1, g2 = res.Stable, res.Unstable
		}
		descRequireNilWithinBar(t, "XACVXD", sys, g1, g2, err, pts)
	}
}

// xacvxdDescriptor is the TestStabsep_DescriptorRejectedQZReorder model for
// seed: A = blkdiag(H, 1), E = blkdiag(J, 0), the ZHGFWA Dare pencil
// perturbed by 1e-12 relative, discrete Ts = 1, B and C ones.
func xacvxdDescriptor(t *testing.T, seed uint64) *System {
	t.Helper()
	h := []float64{
		11.475236504468711, -4.802980538727953, 0, 0,
		-4.804997870973105e-16, 1.883135785961934e-16, 1, 0,
		1.883135785961934e-16, -7.38023300653887e-17, 0, 1,
		0.0004680437192265164, 0.00021592881520019866, 0, 0,
	}
	j := []float64{
		-0.9260172649172378, 0.37748115799546883, 0.00065672900631252, 0.00161105365515163,
		0, 0, -13.418168006075515, -2.5172613588536494,
		0, 0, 0.597370003150422, -11.258325117032756,
		-3.097492448978087e-05, -2.0823017944789118e-05, -31.53860826496756, -77.36888371806144,
	}
	rng := rand.New(rand.NewPCG(seed, 9))
	A, E := mat.NewDense(5, 5, nil), mat.NewDense(5, 5, nil)
	for i := range 4 {
		for k := range 4 {
			A.Set(i, k, h[i*4+k]*(1+1e-12*rng.NormFloat64()))
		}
	}
	for i := range 4 {
		for k := range 4 {
			E.Set(i, k, j[i*4+k]*(1+1e-12*rng.NormFloat64()))
		}
	}
	A.Set(4, 4, 1)
	B, C := mat.NewDense(5, 1, nil), mat.NewDense(1, 5, nil)
	for i := range 5 {
		B.Set(i, 0, 1)
		C.Set(0, i, 1)
	}
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// descTied354AugQ is the arm64 stored data of corpus row tied/354 with an
// appended algebraic state under random orthogonal U, V (EUNKOP augQ):
// Dgges left a near-defective real pair at λ ≈ −1 (Ts = 0.5) unsplit as a
// 2×2 block, reported both halves real (Dtgsen's Dlag2, as reference LAPACK
// 3.12, ET622B) and set n1 inside it; the split ignored S21 and returned
// Stable + Unstable off by 4 percent.
func descTied354AugQ(t *testing.T) *System {
	t.Helper()
	A := mat.NewDense(9, 9, []float64{-0.614578897865547, 0.2163323975066743, 0.00039735636704525153, -0.9427632238588094, 0.39872174712035796, -0.43833770797804245, -0.2713336869153518, 0.15352389971245392, 0.20732896871466527, 0.32234238237157914, 0.26449267250425934, 0.18928187479713954, 0.39967089154368585, 0.03062351858793012, 0.060050042174932, 0.26470817167278027, -0.41938515605393595, -0.7057423245171166, -0.537874644966477, -0.03426013178737156, -0.416178682818744, 0.6611857584283989, -0.6076105375710097, -1.075186422596192, 0.7189953158126704, 1.1600371128930895, 0.7787148908297377, -0.3994587607281084, -0.33941757654985, 0.020225301028101113, 0.14647349910834448, 0.00371468881664437, -0.5079730762886713, 0.22211382831383517, -0.3777809203701853, 0.3394035795240312, -0.3010920481435536, -0.1520603373233379, 0.11564062501072475, 0.5423419703709488, -0.18731641101763072, -0.03968342395262189, 0.559171059249871, -0.012194575355904277, -0.2776678775783286, -0.43184812577828274, -0.2188168933336735, 0.15253753798911251, -0.14181529093878392, -0.28682210207588377, -0.4884463523796544, 0.5549933805927476, -0.12719125103180282, -0.05104263293728, 0.3392451497911849, 0.5383982653349851, 0.3686354911154195, -0.149353203864944, -0.048580681360121766, -0.10283537433774882, -0.06538717279879622, -0.447345434395953, -0.3554328430275573, 0.3776586076399971, -0.4281077851041599, 0.10294458761565112, 0.6027524253875246, -0.13537360946187413, 0.2683930273738266, -0.10523664449278512, -0.1100442016665232, 0.5654008127281036, -0.6896328188074713, 0.9076105002166338, 0.10189713262314705, -0.3035707835917166, 0.7277784684866053, 0.06922187537741811, -0.39942358883760193, -0.2531689299134612, 0.39969937787742954})
	E := mat.NewDense(9, 9, []float64{-0.1801467446506509, -0.24177300600123133, 0.16413084820982687, 0.044105691487594495, 0.4431011094918468, -0.5431085534016706, 0.1177877326342065, 0.11114592808737361, -0.6017398703785559, -0.2002978150774171, -0.2663148206135841, -0.6580281059141337, 0.49160258054091927, 0.05375282356458766, -0.1497990562381794, -0.1818354224250245, 0.2994762075615745, 0.21217252058916208, -0.13966999123727397, -0.09522373436486427, -0.14556931161378267, -0.01148114578884617, 0.3749263208868808, 0.7388376906189068, 0.12306021234003564, -0.029664359177165124, -0.346966554716156, 0.3306888477189057, -0.39698844380268017, -0.22475807054116567, 0.1150676706741258, -0.14921802300234838, 0.23394185401269305, 0.14641305687413095, -0.05074357028417761, -0.26575020375938946, 0.14470929296401944, 0.21222829691429498, -0.22407574474604378, -0.0987724854852763, 0.29327108616656866, -0.03812662445809364, -0.7636976655101645, -0.39642317321964887, -0.16369875043849355, -0.3596705221487661, -0.025312448615379693, 0.5195998484374093, 0.2985400568014642, 0.04499786431061756, 0.2829591714834515, -0.41612002907128937, 0.38323430219306204, 0.06155782036614787, -0.30116351137499725, -0.37253394655937916, -0.07642124817366161, -0.4555705587413693, -0.5787979552781611, -0.002117414773330535, -0.31295962635815, 0.07985632675402303, -0.292424283315759, -0.6583782633796057, 0.36205628764516246, -0.32349319192678244, -0.2754585289994425, 0.07409957301206217, 0.004125748723610437, 0.23801920486231215, -0.0663438342164137, 0.04573118728765336, 0.18172368364212138, -0.30826456482199, -0.04293851411188292, -0.5962352025782516, 0.4387705618572822, 0.012935106285886486, -0.06334452254231944, 0.3942734268542972, 0.3899805608733722})
	B := mat.NewDense(9, 1, []float64{0.43023216240900103, -0.5615011784736639, -0.6322143171752663, 1.8866631533737053, -0.4256018236400132, -2.0325152628258683, 0.3290019204836995, 0.4045911178996285, -0.4541282567635625})
	C := mat.NewDense(2, 9, []float64{-1.30394374546999, 0.45788073389989825, -2.060113713108877, 0.4546236696588086, 0.5853429568037058, 2.401042195792644, -0.39072174292108786, -1.9559942397510155, -0.23047299371680297, -1.2234102819052866, 0.7460091190727552, -0.17081073163082058, -0.45172842260214413, -0.6199198465061604, 1.2673075181931657, 0.7630980410119333, -1.1728299846507522, 1.1248488999454842})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(2, 1, []float64{2, 1}), E, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}
