package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// valueErr is err with ErrAccuracyUnverified read as a value: the result
// is returned alongside it.
func valueErr(err error) error {
	if errors.Is(err, ErrAccuracyUnverified) {
		return nil
	}
	return err
}

type descriptorLiteral struct {
	n, m, p       int
	a, b, c, d, e []float64
}

func (l descriptorLiteral) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(l.n, l.n, l.a), mat.NewDense(l.n, l.m, l.b), mat.NewDense(l.p, l.n, l.c),
		mat.NewDense(l.p, l.m, l.d), mat.NewDense(l.n, l.n, l.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// smallSTermConverse1437 is smallSTermChainDescriptor(1437, 1e-4) frozen:
// main returns its TF with nil, 2× outside the response's terms and its
// data resolution at 10i.
var smallSTermConverse1437 = descriptorLiteral{
	n: 7, m: 1, p: 1,
	a: []float64{46374.66080079181, 70407.90412243232, -27491.499084996056, 38271.59370294404, -94576.46941955699, 13665.215518519743, -19111.554490094964, -1864.134854861105, -2827.120191394049, 1110.604920768632, -1540.1783085546049, 3790.6432574008513, -545.4676498430496, 769.9274116635823, 82231.63571190204, 124844.8351785187, -48750.91467851064, 67864.04084108435, -167701.14775194356, 24233.949191300962, -33890.04588937459, -160149.8527507583, -243145.14142070763, 94941.46463203855, -132171.12692041133, 326604.11508545646, -47186.71741607811, 66006.56287798451, -28545.63725610445, -43335.2096890594, 16923.45404275081, -23557.98599897478, 58209.3442524883, -8410.196201108567, 11763.648605011947, 27210.30133645959, 41310.69335426423, -16130.228996748348, 22455.403998329657, -55492.834374764025, 8020.0133988625375, -11213.023345243097, 166455.27134960666, 252718.35933762082, -98680.08850837787, 137377.10930291202, -339464.6390917946, 49044.31770408266, -68606.75590375527},
	b: []float64{0.7250000123958005, 0.31473325786160455, 1.57516533625141, -0.28074504344244167, 0.19636412473968023, 1.344965121802612, 2.954018787582521},
	c: []float64{-0.9597157864355866, 1.2786916111712576, 6.571429960409456, -1.2885827177976883, -1.0137746019178846, -4.027511366923849, -0.4989271105608501},
	d: []float64{-0.26940206902944813},
	e: []float64{-0.2597102018071852, -7.0770091847003895, -3.2680782665887818, 4.95288476488326, 4.385292182027589, 0.709715468648181, 0.27180624616108817, -2.119418510770867, -2.071533225446128, -2.2788977537237076, 4.715867742524158, 2.5033673457244117, -4.248893846211508, -4.40037350874882, -0.8338389482361017, -3.9858520604843672, -2.371287495036436, 3.3072686787827585, -0.049055097692518215, -0.3768445615352531, -1.161791960082826, 0.9211584260782746, -0.8880432667296564, -3.737530389366621, 6.6855578163555585, -3.8814633386274022, -0.08339620488549615, -3.0521393801334304, -0.11512649410967334, -2.679006160043617, -0.24565668082235131, 3.261517120080884, 1.2083013058090941, -1.1846748125416402, -0.532874509114702, -1.594143068394627, -4.527913138856705, -1.702460051015961, 3.4702984653925597, 0.6202893977428537, 0.48798981839329464, -1.18753900009236, -2.4564538224491983, -7.888551265445174, 5.593653490799505, -1.2422648025716019, 3.479592901739338, 1.2791320918344924, 4.222617786479007},
}

// largeAChainLowBand7 is largeAChainDescriptor(7) (4491e78's generator)
// frozen: a pole at 1.4e5 beside an index-5 chain, its TF within terms
// only below about 300i.
var largeAChainLowBand7 = descriptorLiteral{
	n: 6, m: 1, p: 1,
	a: []float64{-164961.86302670828, -191933.65815863424, 43512.61069803172, 148103.38993148215, -102814.50829041326, 279831.4320670752, -86021.87624569361, -100089.47600107218, 22692.548423746743, 77232.72408428135, -53612.34275114844, 145922.1345725032, 53863.641367754426, 62664.048633898645, -14206.065126562316, -48358.65526713096, 33569.04920147337, -91372.60377366749, 8198.461009600986, 9545.85571113554, -2164.4992021297903, -7360.656762828236, 5110.385286703791, -13908.778701254965, 209266.07085912296, 243481.9302055681, -55199.10041564666, -187880.06856777158, 130429.80702455963, -354987.4425892351, 168977.13619910204, 196603.87411712596, -44570.434806742065, -151708.71801313382, 105315.7413664686, -286640.15830978635},
	b: []float64{-2.6432764732627447, -1.774349449643725, 1.5631747077864349, -0.34222105427205785, 3.931308717916877, 2.916835236484694},
	c: []float64{0.019402448752582218, -0.1392807668957776, 1.7194935321299398, 0.2538086660117549, 2.1330303434924835, 4.205650781418397},
	d: []float64{-1.4903180211313933},
	e: []float64{-1.5707724715834126, -0.3352218183692016, -0.7856181862914846, 1.2440664280027207, -1.8182560348111692, 0.5525500624515605, -2.6432805548018985, -3.443670386518802, -0.2520558499487239, 3.2824930178140264, 1.7180763924395914, 0.8581486630717798, -0.6130950474872817, -1.8764826098964802, 2.133253113311929, 1.660014292490483, 4.133025109299453, 2.975102756311221, 0.24549042320829723, 1.8884939413105426, -0.8708979207252804, -0.6229547335218208, -1.5438388018899738, -1.286499655222288, 2.8310318048148515, 2.5037204487709834, 1.6935553562595191, -2.632793013288763, 1.7453161118570204, 0.32161069219354643, 1.837324416836137, -1.0590831448559452, 1.01121977507894, -1.4362644539608658, 2.268742750810643, -0.5144150746120015},
}

// TestTransferFunctionDescriptorVerifiedBelowRoots: OXJGBU. Beside a
// large-A pole and a chain the reduction's TF is within the response's
// terms only well below the poles (largeA 7: 1.3e-8 relative at 100i,
// 1.3e-6 at 1000i where the data fix G to 1.9e-5); the certificate counts
// the half-decade ladder under the roots and returns it with nil. Oracle:
// 320-bit solve of the stored pencil.
func TestTransferFunctionDescriptorVerifiedBelowRoots(t *testing.T) {
	sys := largeAChainLowBand7.system(t)
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatalf("TransferFunction: %v", err)
	}
	or := newStepOraclePencil(sys)
	for _, s := range []complex128{0.05i, 0.37i, complex(0.5, 3), 10i} {
		re, im := or.transfer(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))})
		a, _ := re[0][0].Float64()
		b, _ := im[0][0].Float64()
		got, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(got[0][0]-complex(a, b)) / cmplx.Abs(complex(a, b)); !(d <= 1e-9) {
			t.Errorf("G(%v) off the oracle by %.3g relative", s, d)
		}
	}
	if pr, err := sys.IsProper(); err != nil || pr == fzTFImproper(res.TF) {
		t.Errorf("IsProper = %v, %v, TF improper %v", pr, err, fzTFImproper(res.TF))
	}
}

// TestTransferFunctionDescriptorRefutedReturnsUnverified: OXJGBU. Main
// returned smallSTermChainDescriptor(1437, 1e-4)'s TF with nil though it
// misses the stored response at 10i by 2× the larger of √ε times its terms
// and the data resolution; it now comes with ErrAccuracyUnverified, the
// value unchanged, and IsProper and Step share the verdict. Where the
// reduction fails closed instead (amd64: an infinite chain left among the
// finite eigenvalues) there is no value to judge.
func TestTransferFunctionDescriptorRefutedReturnsUnverified(t *testing.T) {
	sys := smallSTermConverse1437.system(t)
	res, err := sys.TransferFunction(nil)
	if errors.Is(err, ErrDescriptorSingular) {
		return
	}
	if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
		t.Fatalf("TransferFunction = %v, %v; want a value with ErrAccuracyUnverified", res, err)
	}
	pr, perr := sys.IsProper()
	if !errors.Is(perr, ErrAccuracyUnverified) {
		t.Errorf("IsProper err %v, want ErrAccuracyUnverified", perr)
	}
	if pr == fzTFImproper(res.TF) {
		t.Errorf("IsProper %v but TF improper %v", pr, fzTFImproper(res.TF))
	}
	if _, serr := Step(sys, 1); pr && !errors.Is(serr, ErrAccuracyUnverified) || !pr && !errors.Is(serr, ErrImproperModel) {
		t.Errorf("Step err %v with IsProper %v", serr, pr)
	}
	s := complex(0, 10.0)
	full := newPencilBoundedResponse(sys, []complex128{s})
	_, poles, _, _, _ := tfShape(res.TF)
	jp := judgeDescriptorTF(sys, res.TF, []complex128{s}, full, tfPoleWindows(res.TF, poles, math.Sqrt(eps())))
	if !jp[0].refuted {
		t.Errorf("10i does not refute the TF (worst %.3g)", jp[0].worst)
	}
}

// TestPencilBoundedResponseBoundsExact: the computational bound comp of
// pencilBoundedResp holds the GEPP response Ĉ(sE−A)⁻¹B̂ against the exact
// one, and dTer the terms T̂ against the exact T, at every determined
// point (η < 1) of frozen chain descriptors beside a large pole, at s
// across the band and past the chain's rounding cluster.
func TestPencilBoundedResponseBoundsExact(t *testing.T) {
	pts := []complex128{0, 0.37i, complex(0.5, 3), 31.5i, 1e3i, 3e4i, 1e6i}
	for name, sys := range map[string]*System{"largeA 7": largeAChainLowBand7.system(t), "small 1437": smallSTermConverse1437.system(t)} {
		n, m, p := sys.Dims()
		full := newPencilBoundedResponse(sys, pts)
		determined := 0
		for k, s := range pts {
			if !full.ok[k] || !(full.eta[k] < 1) {
				continue
			}
			determined++
			ur, ui := exactPencilAt(sys, s)
			if ur == nil {
				t.Fatalf("%s: sE − A singular at %v", name, s)
			}
			for i := range p {
				for j := range m {
					gr, gi, T := new(big.Rat), new(big.Rat), 0.0
					var tmp big.Rat
					for l := range n {
						c := new(big.Rat).SetFloat64(sys.C.At(i, l))
						gr.Add(gr, tmp.Mul(c, ur[l][j]))
						gi.Add(gi, tmp.Mul(c, ui[l][j]))
						a, _ := ur[l][j].Float64()
						b, _ := ui[l][j].Float64()
						T += math.Abs(sys.C.At(i, l)) * cmplx.Abs(complex(a, b))
					}
					T += math.Abs(sys.D.At(i, j))
					a, _ := gr.Float64()
					b, _ := gi.Float64()
					e := k*p*m + i*m + j
					if d := cmplx.Abs(full.g[e] - complex(a, b)); !(d <= full.comp[e]+4*eps()*cmplx.Abs(complex(a, b))) {
						t.Errorf("%s at %v [%d%d]: |Ĝ − G| = %.3g > comp %.3g", name, s, i, j, d, full.comp[e])
					}
					if d := math.Abs(full.terms[e] - T); !(d <= full.dTer[e]+4*eps()*T) {
						t.Errorf("%s at %v [%d%d]: |T̂ − T| = %.3g > δ_T %.3g", name, s, i, j, d, full.dTer[e])
					}
				}
			}
		}
		if determined < 4 {
			t.Errorf("%s: %d determined points", name, determined)
		}
	}
}

// TestTransferFunctionDescriptorRoundingDoubleZeroWindow: OXJGBU. The
// realization of 9s²/(7s + 9) carries the double zero at 0 exactly; its
// reduction splits it by rounding to ±1.9e-8j (amd64: coefficients
// −6.7e-16, 4.4e-16 beside 1.29), where the TF is relatively off. The
// first-order window of either root does not reach the other; the
// cluster window (tfPoleWindows) about their centroid, second order,
// covers both, and the TF returns with nil, within 1e-12 of 9s²/(7s+9)
// away from the zero.
func TestTransferFunctionDescriptorRoundingDoubleZeroWindow(t *testing.T) {
	tf := &TransferFunc{Num: [][][]float64{{{9, 0, 0}}}, Den: [][]float64{{7, 9}}}
	ss, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	res, err := ss.Sys.TransferFunction(nil)
	if err != nil {
		t.Fatalf("TransferFunction: %v", err)
	}
	for _, s := range []complex128{0.37i, complex(0.5, 3), 1e3i} {
		got, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want := 9 * s * s / (7*s + 9)
		if d := cmplx.Abs(got[0][0]-want) / cmplx.Abs(want); !(d <= 1e-12) {
			t.Errorf("G(%v) = %v, want %v", s, got[0][0], want)
		}
	}
	w := tfPoleWindows(&TransferFunc{Num: [][][]float64{{{1.2857142857142856, -6.661338147750939e-16, 4.440892098500626e-16}}}, Den: [][]float64{{1, 1.2857142857142856}}}, []complex128{-1.2857142857142856}, math.Sqrt(eps()))
	for _, s := range []complex128{1.859e-8i, 1e-11i, 0} {
		if !w.contain(s) {
			t.Errorf("%v outside the window of the split double zero", s)
		}
	}
}

// TestTransferFunctionDescriptorWindowUndeterminedDegree: OXJGBU. The
// reduced TF of nearSingularEContinuousK(117306) on amd64 has the row
// denominator s² − 2.69e15·s + 1.26e14, whose leading coefficient lies
// within its own (d+1)ε‖c‖∞ bound: the pole near 2.7e15 is placed nowhere,
// and a cluster about it took in the pole at 0.047 and opened a window
// over the whole axis, so the TF, 2.8× past its contract at 1e12j, was
// certified. Such a polynomial now gets no window.
func TestTransferFunctionDescriptorWindowUndeterminedDegree(t *testing.T) {
	den := []float64{1, -2.6899581798769065e+15, 1.2644841442409705e+14}
	tf := &TransferFunc{Num: [][][]float64{{{-1.577419773320246, 4.0910786504363625e+15, 4.738000349482541e+15}}}, Den: [][]float64{den}}
	_, poles, _, _, ok := tfShape(tf)
	if !ok {
		t.Fatal("roots")
	}
	w := tfPoleWindows(tf, poles, math.Sqrt(eps()))
	for _, s := range []complex128{1e12i, 1e11i, 1i} {
		if w.contain(s) {
			t.Errorf("%v inside a pole window (%v, %v)", s, w.centre, w.radius)
		}
	}
}

func infinityProbeModel(t *testing.T, d float64) (*System, *TransferFunc) {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 0.5, 0.3, -2}), mat.NewDense(2, 1, []float64{1, -0.7}),
		mat.NewDense(1, 2, []float64{0.4, 1.3}), mat.NewDense(1, 1, []float64{d}), mat.NewDense(2, 2, []float64{2, 1, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	return sys, res.TF
}

// TestDescriptorTFInfinityProbe: OXJGBU, the probe at ∞ that closes the
// ladder above s_dom (ruling r5). With E robustly invertible every model
// within terms of the data is proper with G(∞) = D: an s-term below what
// any ladder point to s_dom can see, a wrong tf(∞), and, at D = 0, a wrong
// first Markov parameter CE⁻¹B each refute; the exact TF does not.
func TestDescriptorTFInfinityProbe(t *testing.T) {
	sys, tf := infinityProbeModel(t, 0.8)
	if descriptorTFInfinityRefutes(sys, tf) || !descriptorTFVerified(sys, tf) {
		t.Fatalf("exact TF refuted at ∞ or unverified: %v / %v", tf.Num, tf.Den)
	}
	sdom := dominatedFrequency(sys)
	hidden := tf.Copy()
	c := 0.5 * math.Sqrt(eps()) * 0.8 / sdom
	hidden.Num[0][0] = Poly(hidden.Num[0][0]).Add(Poly(hidden.Den[0]).Mul(Poly{c, 0}))
	if !descriptorTFInfinityRefutes(sys, hidden) {
		t.Errorf("s-term %.3g (below √ε·|D|/s_dom) not refuted at ∞", c)
	}
	if descriptorTFVerified(sys, hidden) {
		t.Errorf("TF with a hidden s-term verified")
	}
	off := tf.Copy()
	off.Num[0][0] = Poly(off.Num[0][0]).Add(Poly(off.Den[0]).Mul(Poly{1e-6}))
	if !descriptorTFInfinityRefutes(sys, off) {
		t.Errorf("tf(∞) = D + 1e-6 not refuted")
	}
	sys0, tf0 := infinityProbeModel(t, 0)
	if descriptorTFInfinityRefutes(sys0, tf0) {
		t.Fatalf("exact strictly proper TF refuted at ∞: %v / %v", tf0.Num, tf0.Den)
	}
	g1 := tf0.Copy()
	num := trimLeadingZeros(g1.Num[0][0])
	num[0] *= 1 + 1e-6
	g1.Num[0][0] = num
	if !descriptorTFInfinityRefutes(sys0, g1) {
		t.Errorf("first Markov parameter off by 1e-6 not refuted")
	}
	sys2, err := NewDescriptor(mat.NewDense(3, 3, []float64{-1, 0, 0.5, 0, -2, 0, 0.3, 0, -3}), mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}), mat.NewDense(1, 1, nil), mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 1e4, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := sys2.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if descriptorTFInfinityRefutes(sys2, res2.TF) {
		t.Fatalf("exact TF with g₁ = T₁ = 0 refuted at ∞: %v / %v", res2.TF.Num, res2.TF.Den)
	}
	g2 := res2.TF.Copy()
	num2 := trimLeadingZeros(g2.Num[0][0])
	num2[0] *= 1 + 1e-6
	g2.Num[0][0] = num2
	if !descriptorTFInfinityRefutes(sys2, g2) {
		t.Errorf("second Markov parameter off by 1e-6 not refuted (g₁ = T₁ = 0)")
	}
}

// TestDescriptorTFPrunedExactMatchesUnpruned: skipping the exact
// evaluation of open entries whose bounds exclude both outcomes
// (tfPruneExact) gives the verdicts of evaluating every one, over the
// stiff-leak chain family (kdSeedDescriptor, SISO and MIMO).
func TestDescriptorTFPrunedExactMatchesUnpruned(t *testing.T) {
	defer func() { tfPruneExact, tfVerdictMemo = true, true }()
	tfVerdictMemo = false
	for seed := range uint64(12) {
		for _, mimo := range []bool{false, true} {
			sys, _ := kdSeedDescriptor(t, seed, mimo)
			res, err := sys.TransferFunction(nil)
			if res == nil {
				continue
			}
			_ = err
			tfPruneExact = true
			pruned := descriptorTFVerified(sys, res.TF)
			tfPruneExact = false
			full := descriptorTFVerified(sys, res.TF)
			if pruned != full {
				t.Errorf("seed %d mimo %v: pruned verdict %v, unpruned %v", seed, mimo, pruned, full)
			}
		}
	}
}

// exactMarkovParam returns g_k = C(E⁻¹A)^(k−1)E⁻¹B, entry (0, 0), for rational
// a, e (n×n), b (n), c (n), by elimination on E.
func exactMarkovParam(a, e [][]*big.Rat, b, c []*big.Rat, k int) *big.Rat {
	n := len(b)
	solve := func(rhs []*big.Rat) []*big.Rat {
		M := make([][]*big.Rat, n)
		for i := range n {
			M[i] = make([]*big.Rat, n+1)
			for j := range n {
				M[i][j] = new(big.Rat).Set(e[i][j])
			}
			M[i][n] = new(big.Rat).Set(rhs[i])
		}
		var t big.Rat
		for col := range n {
			piv := col
			for M[piv][col].Sign() == 0 {
				piv++
			}
			M[col], M[piv] = M[piv], M[col]
			inv := new(big.Rat).Inv(M[col][col])
			for j := col; j <= n; j++ {
				M[col][j].Mul(M[col][j], inv)
			}
			for r := range n {
				if r != col && M[r][col].Sign() != 0 {
					f := new(big.Rat).Set(M[r][col])
					for j := col; j <= n; j++ {
						M[r][j].Sub(M[r][j], t.Mul(f, M[col][j]))
					}
				}
			}
		}
		out := make([]*big.Rat, n)
		for i := range n {
			out[i] = M[i][n]
		}
		return out
	}
	u := solve(b)
	var t big.Rat
	for range k - 1 {
		w := make([]*big.Rat, n)
		for i := range n {
			w[i] = new(big.Rat)
			for j := range n {
				w[i].Add(w[i], t.Mul(a[i][j], u[j]))
			}
		}
		u = solve(w)
	}
	g := new(big.Rat)
	for i := range n {
		g.Add(g, t.Mul(c[i], u[i]))
	}
	return g
}

// TestMarkovBoundCoversExactPerturbations: the Markov bound of the probe at
// ∞ (markovBounds.bound, k ≥ 2) holds the change of g_k under exact
// relative perturbations of A, E, B and C by ±ε (random signs), with an
// ill-conditioned E whose Skeel factor |E⁻¹||E| reaches 1e8, for k = 2, 3.
func TestMarkovBoundCoversExactPerturbations(t *testing.T) {
	n := 3
	af := []float64{-1, 0.7, 0.5, 0.2, -2, 0.4, 0.3, -0.6, -3}
	ef := []float64{1, 1e4, 0, 0, 1, 1e4, 0, 0, 1}
	bf := []float64{0.3, -1.1, 0.8}
	cf := []float64{1.2, 0.4, -0.9}
	sys, err := NewDescriptor(mat.NewDense(n, n, af), mat.NewDense(n, 1, bf), mat.NewDense(1, n, cf), mat.NewDense(1, 1, nil), mat.NewDense(n, n, ef), 0)
	if err != nil {
		t.Fatal(err)
	}
	mb, ok := newMarkovBounds(sys)
	if !ok {
		t.Fatal("E singular")
	}
	rng := rand.New(rand.NewPCG(5, 77))
	eps := new(big.Rat).SetFloat64(eps())
	pert := func(x float64) *big.Rat {
		r := new(big.Rat).SetFloat64(x)
		f := new(big.Rat).Add(big.NewRat(1, 1), eps)
		if rng.IntN(2) == 0 {
			f.Sub(big.NewRat(1, 1), eps)
		}
		return r.Mul(r, f)
	}
	mats := func(perturb bool) (a, e [][]*big.Rat, b, c []*big.Rat) {
		get := func(x float64) *big.Rat {
			if perturb {
				return pert(x)
			}
			return new(big.Rat).SetFloat64(x)
		}
		a, e = make([][]*big.Rat, n), make([][]*big.Rat, n)
		b, c = make([]*big.Rat, n), make([]*big.Rat, n)
		for i := range n {
			a[i], e[i] = make([]*big.Rat, n), make([]*big.Rat, n)
			for j := range n {
				a[i][j], e[i][j] = get(af[i*n+j]), get(ef[i*n+j])
			}
			b[i], c[i] = get(bf[i]), get(cf[i])
		}
		return
	}
	a0, e0, b0, c0 := mats(false)
	for _, k := range []int{2, 3} {
		g0 := exactMarkovParam(a0, e0, b0, c0, k)
		bound := mb.bound(k, 0, 0)
		worst := 0.0
		for range 40 {
			a, e, b, c := mats(true)
			d, _ := new(big.Rat).Sub(exactMarkovParam(a, e, b, c, k), g0).Float64()
			worst = math.Max(worst, math.Abs(d))
		}
		if !(worst <= bound) || worst == 0 {
			t.Errorf("k=%d: |Δg_k| up to %.3g, bound %.3g", k, worst, bound)
		}
		t.Logf("k=%d: |Δg_k| ≤ %.3g, bound %.3g", k, worst, bound)
	}
}
