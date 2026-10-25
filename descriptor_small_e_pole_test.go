package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzPertULP returns x with each entry moved by up to 8 ulps.
func fzPertULP(rng *rand.Rand, x []float64) []float64 {
	y := slicesCopy(x)
	for i := range y {
		for range rng.IntN(9) {
			y[i] = math.Nextafter(y[i], math.Inf(2*rng.IntN(2)-1))
		}
	}
	return y
}

// TestDescriptorIndexOneIllConditionedKeepsFinitePole: index-1 pencils
// A = U·blkdiag(F, I)·V, E = U·blkdiag(W, 0)·V with cond(U), cond(V) ≈ 1e5.6,
// so the finite part of E sits 1e3·tol above the deflated-block rank tol.
// The staircase took that real singular value as zero within the A-side
// leak (capped at qzInfiniteGap), deflating a finite pole as infinite:
// Poles lost it, IsStable flipped (seeds 145, 418: unstable pole 3.96,
// 1.96 dropped, reported stable), and IsProper/TF/Step said improper where
// base failed closed. An infinite
// chain leaves Ulᵀ·A·Vr at rounding level; these keep it ≥ 1e-12·‖A‖.
// Oracle: eig(W⁻¹F) of the structured model; TF against the exact
// rational response of the float data.
func TestDescriptorIndexOneIllConditionedKeepsFinitePole(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		a, e, b, c []float64
		want       []complex128
	}{
		{"145", 4,
			[]float64{0.04093758609686032, 0.019861776165723044, -0.1394837174199481, -0.14823281804231078, -0.05086389971626707, -0.023433349175990634, 0.16701201602559332, 0.1787917628590964, -0.06909861473817942, -0.03431306004605764, 0.2394225800190722, 0.2536141136690435, -0.04887342266586861, -0.023052975559943647, 0.16319081309008618, 0.1741174217488029},
			[]float64{0.021635002483381703, 0.015326252756955702, -0.09776295421171668, -0.0988889910001219, -0.030374781236385772, -0.02150452632776599, 0.13719822789638317, 0.13878798304202164, -0.0342488565201105, -0.024272180071114646, 0.15480791385970233, 0.15658341807068196, -0.02764602356299864, -0.019578781762076922, 0.12490050615398211, 0.12634322768343606},
			[]float64{-1.1647725845168635, 0.007057187998262715, -0.694329943033263, -0.9539407768604006},
			[]float64{0.8571170486355306, 0.38969675179553315, -0.6833980364600376, 0.30938136824962337},
			[]complex128{complex(-5.0087401181615325, 1.6999973915445912), complex(-5.0087401181615325, -1.6999973915445912), 3.9636052146206753}},
		{"418", 3,
			[]float64{0.08708686639627698, -0.0755642714279615, 0.35059562369357206, 0.16285555300823265, -0.14130834686508972, 0.6556257026636121, 0.010293495586265038, -0.008931596237808884, 0.04143964174050677},
			[]float64{0.06886015084081726, -0.05964634349724191, 0.27736408973798793, 0.12974626813565987, -0.11238360759648687, 0.5226121630047685, 0.00824049436721794, -0.007137669235776534, 0.03319245820925515},
			[]float64{-0.4786210326227082, -1.0810649281165197, -0.43900001561492774},
			[]float64{1.4906928520198188, -0.7996362873282745, 0.04078316091804868},
			[]complex128{-0.05459087127364137, 1.9636176118307498}},
	} {
		wantStable := true
		for _, w := range tc.want {
			wantStable = wantStable && real(w) < 0
		}
		rng := rand.New(rand.NewPCG(5, 23))
		for run := range 20 {
			a, e := tc.a, tc.e
			if run > 0 {
				a, e = fzPertULP(rng, a), fzPertULP(rng, e)
			}
			sys := fzSISODescriptor(t, tc.n, a, e, tc.b, tc.c)
			if st, err := sys.IsStable(); err == nil && st != wantStable {
				t.Fatalf("%s run %d: IsStable = %v, want %v", tc.name, run, st, wantStable)
			}
			if p, err := sys.Poles(); err == nil {
				for _, w := range tc.want {
					found := false
					for _, q := range p {
						found = found || cmplx.Abs(q-w) <= 1e-3*cmplx.Abs(w)
					}
					if !found {
						t.Fatalf("%s run %d: Poles = %v, want %v among them", tc.name, run, p, w)
					}
				}
			}
			if pr, err := sys.IsProper(); err == nil && !pr {
				t.Fatalf("%s run %d: IsProper = false, want true (or an error)", tc.name, run)
			}
			if _, err := Step(sys, 1); errors.Is(err, ErrImproperModel) {
				t.Fatalf("%s run %d: Step err = %v", tc.name, run, err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				continue
			}
			if fzTFImproper(res.TF) {
				t.Fatalf("%s run %d: TF improper, num=%v den=%v", tc.name, run, res.TF.Num, res.TF.Den)
			}
			if run%5 != 0 {
				continue
			}
			for _, x := range []complex128{0.37i, complex(-0.5, 3)} {
				w, ok := fzExactDescResponse(tc.n, 1, 1, a, e, tc.b, tc.c, []float64{0.3}, x)
				if !ok {
					t.Fatalf("%s run %d: exact response at %v failed", tc.name, run, x)
				}
				g, err := res.TF.Eval(x)
				if err != nil {
					t.Fatal(err)
				}
				if d := cmplx.Abs(g[0][0] - w[0]); !(d <= 1e-6*(1+cmplx.Abs(w[0]))) {
					t.Fatalf("%s run %d: G(%v) = %v, exact %v", tc.name, run, x, g[0][0], w[0])
				}
			}
		}
	}
}

// TestDescriptorSmallImpulsiveTermKept: improper index-3 chains whose b
// tail is 1e-11..1e-10 (rvBuild class 4, seeds 1688 and 233). Their
// c·N^k·b lie 5 to 25 times above the ε-level cut, but inside it widened
// by the staircase leak and egap, which dropped them: IsProper true, a
// proper TF off by 1e-10 at |s| ≤ 3 and Step returning a response (base
// was right). Spurious terms that widening is for change at first order
// under ε-perturbation of the coefficients; these keep 3 digits. Oracle:
// the exact rational response of the float data.
func TestDescriptorSmallImpulsiveTermKept(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		a, e, b, c []float64
	}{
		{"1688", 5,
			[]float64{3.5058468481047447, -0.08926974664599595, -0.6163428015598069, 2.132745107031651, 1.1288182468669152, -0.14893926562813065, 0.8747351582223445, -0.41712772209611, 0.6400398287303763, 0.9068352489291812, 3.7376586604638895, -1.8070415813237257, 0.6285425316427545, -0.21080492058205835, 0.24382402985813192, 2.5301045487423353, -1.5009095464300568, -1.464671059682392, 0.5439920693994393, 0.6133631302386705, -1.9490023395929859, 0.9246790125928835, 0.399366846611659, 0.7956175467971218, -1.5176988225868986},
			[]float64{5.210830885391026, 1.3881116346607831, 1.9878021971082493, 2.1478000443055447, 2.5586469050506713, 0.1246740649528188, -0.671732952959139, 0.04897938838354325, -0.48278514433692354, -0.416328134847613, 7.024039512531821, -0.6846398482049412, 2.637121219034744, 4.082507899646409, -1.5821570770607178, 2.2312605235307097, 0.31524271226146244, 0.7783000398471125, -0.24173475628314975, 2.021748094920837, -1.7605253414914368, -0.20934855003130415, -0.7544296521758919, 0.6917458551664293, -1.896779231165194},
			[]float64{-1.0398373724730243, 0.08712898489843006, -1.566207058587595, 0.00015040329462150868, 0.25214375330715066},
			[]float64{2.0531669705373052, -2.3527717178628964, -0.37118258921261227, -0.06958020869693621, -1.2793269909090896}},
		{"233", 6,
			[]float64{-0.6584578170931925, 6.783288038290037, 3.562418257707947, 1.8708203451686867, -1.5152696452478966, -1.4641755854874225, 4.381703986575038, -2.206399724453914, -4.007631086554027, -10.603071868621399, -3.8723945543055374, 4.564617318261405, 0.24853929926586726, 5.270222674582547, 5.877517908012543, 0.33902688888220806, -1.6615530544688533, -2.5548150824502978, -0.2225622805132796, -7.665991633231391, -5.944914686585921, -5.7075357582766015, -0.29856738661266324, 7.01620824074571, -2.400495868226552, -2.065701214076379, -8.393828218048128, 1.7763525167706506, 5.377255541775278, -1.567025421570522, -3.052160486056655, 1.6949051915891804, 0.7662849970674175, -1.157679539823121, 0.9786309340785224, 0.7251621088759906},
			[]float64{0.9044682109141948, -6.29031375196609, 2.50653313439507, -6.059983208398597, -2.1194881059889448, 5.033345244140337, 1.8468334258867003, 1.7633702991347042, 3.2698228352785903, -2.8000578062006287, -2.980367836415298, 0.6883932774276926, 0.5642544948311571, -2.791324881685077, -0.6237146648809788, -1.3425679377621036, -0.7015685180450267, 2.2105180899560097, 0.32206352102288016, 6.021254999615473, -4.243915380215302, 1.6254622556637295, 2.4311953883274207, -6.033050693741315, -0.8414023086965454, -0.17702082586758267, 0.029395256749570463, -3.8632760264467843, -0.6052112546109375, 2.618540911142667, 0.46909484480537994, 0.14843380678103774, -7.203586558876869, -3.396221121203666, 1.8202405967102142, -0.5958875262388669},
			[]float64{1.2496518298811845, 1.998222319518005, 1.97981451881702, 3.6472057084609073, -3.3731275159647, 5.653070498419018},
			[]float64{1.8227889138748148, 0.8217885184476913, 4.068365877327117, 0.15770146857882383, -2.66386221449982, 0.4536177351283461}},
	} {
		rng := rand.New(rand.NewPCG(7, 23))
		for run := range 30 {
			a, e, b, c := tc.a, tc.e, tc.b, tc.c
			if run > 0 {
				a, e, b, c = fzPertULP(rng, a), fzPertULP(rng, e), fzPertULP(rng, b), fzPertULP(rng, c)
			}
			sys := fzSISODescriptor(t, tc.n, a, e, b, c)
			if pr, err := sys.IsProper(); err != nil || pr {
				t.Fatalf("%s run %d: IsProper = %v, %v; want false", tc.name, run, pr, err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil || !fzTFImproper(res.TF) {
				t.Fatalf("%s run %d: TransferFunction err %v, want improper TF", tc.name, run, err)
			}
			if run%10 != 0 {
				continue
			}
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Fatalf("%s run %d: Step err = %v, want ErrImproperModel", tc.name, run, err)
			}
			for _, x := range []complex128{3i, complex(-2, 2)} {
				w, ok := fzExactDescResponse(tc.n, 1, 1, a, e, b, c, []float64{0.3}, x)
				if !ok {
					t.Fatalf("%s run %d: exact response at %v failed", tc.name, run, x)
				}
				g, err := res.TF.Eval(x)
				if err != nil {
					t.Fatal(err)
				}
				if d := cmplx.Abs(g[0][0] - w[0]); !(d <= 2e-12*(1+cmplx.Abs(w[0]))) {
					t.Fatalf("%s run %d: G(%v) = %v, exact %v (diff %.2g)", tc.name, run, x, g[0][0], w[0], d)
				}
			}
		}
	}
}

// TestDescriptorChainKeepsSmallEPole: an infinite chain beside a finite
// pole small in E (σ_r/σ₁ ≈ 5e-8, 1e7 above the MATLAB rank tol) under
// cond(U), cond(V) ≈ 1e4. In a deflated E block that pole's singular value
// fell within the A-side leak of the rank tol and deflatedRank took it as
// zero: Poles dropped the unstable stiff pole (seeds 386: 7409.37, 2172:
// 2277.04; 1854: −2668.3), IsStable flipped to true, and IsProper/TF/Step
// gave an improper TF off by 1e-4..3e-3 where base failed closed. Oracle:
// the structured Weierstrass form (poles f/w); TF against the exact
// rational response of the float data.
func TestDescriptorChainKeepsSmallEPole(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		a, e, b, c []float64
		want       []complex128
		proper     bool
	}{
		{"386", 4,
			[]float64{0.892905025076295, -0.11288359472818811, 0.9121083376923637, -0.06474985778103985, 0.46847707441570297, -0.025600491611808118, 0.4888439986831914, -0.031011494609847604, 0.3040948175501781, -0.03675096186313089, 0.3110230756314638, -0.021847274755270155, 0.25919757166822116, -0.022057008396182534, 0.2681687454551831, -0.017902817068011438},
			[]float64{0.039090653700109564, -0.015922243461823127, 0.03649628155897175, -0.003979387166376351, 0.03947811058918048, -0.0068678529290251734, 0.0403821076458006, -0.003357539145365296, 0.014607023283755748, -0.005410720424973301, 0.013843752738052106, -0.0014482895996516835, 0.017090311446907713, -0.004100339614095816, 0.017050454944971, -0.001534416346945183},
			[]float64{-0.30084654660217397, -0.10967877856694218, -0.10559431595866899, -0.06587309615955014},
			[]float64{0.04359101531472977, 0.09478608888419238, 0.0617121928111728, 0.012238155734080158},
			[]complex128{7409.37, -4.2696}, false},
		{"2172", 6,
			[]float64{-0.0037670257606886824, -0.007534333558418587, -0.0037340504776319406, -0.0016358267147548494, 0.0020932562077547808, -0.007051182977603471, -0.008762732227666416, -0.021784631287861164, 0.03685219790162874, 0.005354194240266594, 0.016892935336291225, -0.004958406734307635, -0.022944618465151372, -0.051878167427000084, 0.024483125256966828, -0.00044560920206025093, 0.02538146215696462, -0.03169271285663985, 0.003417091623774009, 0.007009375520647093, 0.007373602353952956, 0.002279136151688046, -0.0009023226683092553, 0.007614342599610211, 0.033461428203977905, 0.06919721493521581, 0.02416425910692988, 0.012702161208775709, -0.021117183938092986, 0.06093832940793687, 0.0068888509149271976, 0.01839224067772071, -0.039580915997083764, -0.00634702561399234, -0.01611373333267787, 0.001332870310336167},
			[]float64{-0.0007816041745176931, -0.0036098013584827, -0.006786076178720048, -0.0015288167656925314, -0.0010358584763391348, -0.0040944059875368975, 0.0008010236227489541, -0.008688766945771146, -0.0018378185959809284, -0.00010420236396322053, -0.0006932745135124995, -0.0037054114787483275, 0.0023870409320582184, -0.014907199080758701, 0.00862451920074572, 0.0024321444380847783, 0.0011049204152274777, -0.0022419048534452974, -0.0005841607963448808, 0.0006700681844692078, -0.004575916070760294, -0.0010672277976571385, -0.0007823245471057813, -0.0014508727553353576, 0.00248199921596936, 0.025353825362253404, 0.02706574820092621, 0.005859954588711335, 0.004144872588998478, 0.02062641461179422, -0.002094473979436187, 0.005520413620345255, -0.008864789956889479, -0.002340114156119926, -0.00099841765736524, -0.0016973256741709159},
			[]float64{-0.023145546634428398, -0.11921245242090785, -0.2299581473296095, 0.023229190436582104, 0.23990230117121622, 0.11292032969690191},
			[]float64{0.07615822592058047, 0.11217225582365138, 1.003112789362183, 0.21908929747737776, 0.19985098544576227, 0.39458741066664194},
			[]complex128{2277.04, -0.95907}, true},
		{"1854", 5,
			[]float64{0.01017989210285958, -0.020584080467212926, 0.05164182315690744, 0.024001568618924454, 0.09360134432454659, 0.05122278456511631, -0.08489137745099413, 0.2099592844844192, 0.10046455326623703, 0.4395722925197165, -0.0004240204609980562, -0.004424325983754342, 0.011532410778990646, 0.004747450828118146, 0.005199045731007235, 0.006362601012326807, -0.015284635563423847, 0.03840586108300097, 0.017606814859495428, 0.06242478331079874, 0.020018517793076434, -0.0284340996693922, 0.06932881360865033, 0.03410251858559945, 0.16379420264829347},
			[]float64{0.005285537407305326, -0.021112482999938058, 0.0488903396198416, 0.023996430101286263, 0.06989497802888067, 0.015893861198156382, -0.0775267594707307, 0.17987725759868822, 0.08732704501531556, 0.23263403563186663, 0.002824505434612172, -0.006932558424837709, 0.015988681701215784, 0.008117475352608046, 0.030233742013942043, 0.005704823045735848, -0.01856417941143227, 0.04306079178568076, 0.021309738054861624, 0.06802291448530058, 0.0037517108552665647, -0.023485494209275745, 0.05463671817622567, 0.026209062198649886, 0.06305408908276564},
			[]float64{-0.04534447858238751, -0.3879786812689528, 0.04694865027834787, -0.008157914224348478, -0.1926344633457886},
			[]float64{-0.12495085488484634, 0.09864359667198877, -0.12044547296762002, -0.13119227422806057, -0.9490914512399493},
			[]complex128{-2668.3, 1.7444, -0.9397}, true},
	} {
		wantStable := true
		for _, w := range tc.want {
			wantStable = wantStable && real(w) < 0
		}
		rng := rand.New(rand.NewPCG(11, 23))
		for run := range 20 {
			a, e := tc.a, tc.e
			if run > 0 {
				a, e = fzPertULP(rng, a), fzPertULP(rng, e)
			}
			sys := fzSISODescriptor(t, tc.n, a, e, tc.b, tc.c)
			if st, err := sys.IsStable(); err == nil && st != wantStable {
				t.Fatalf("%s run %d: IsStable = %v, want %v", tc.name, run, st, wantStable)
			}
			if p, err := sys.Poles(); err == nil {
				requirePolesContain(t, fmt.Sprintf("%s run %d", tc.name, run), p, tc.want, 1e-4)
			}
			pr, prErr := sys.IsProper()
			if prErr == nil && pr != tc.proper {
				t.Fatalf("%s run %d: IsProper = %v, want %v (or an error)", tc.name, run, pr, tc.proper)
			}
			_, stepErr := Step(sys, 1)
			res, err := sys.TransferFunction(nil)
			if (prErr == nil) != (err == nil) || prErr == nil && pr == errors.Is(stepErr, ErrImproperModel) {
				t.Fatalf("%s run %d: IsProper %v, %v; TransferFunction err %v; Step err %v disagree", tc.name, run, pr, prErr, err, stepErr)
			}
			if err != nil {
				continue
			}
			for _, x := range []complex128{0.37i, complex(-0.5, 3)} {
				w, ok := fzExactDescResponse(tc.n, 1, 1, a, e, tc.b, tc.c, []float64{0.3}, x)
				if !ok {
					t.Fatalf("%s run %d: exact response at %v failed", tc.name, run, x)
				}
				g, err := res.TF.Eval(x)
				if err != nil {
					t.Fatal(err)
				}
				if d := cmplx.Abs(g[0][0] - w[0]); !(d <= 1e-6*(1+cmplx.Abs(w[0]))) {
					t.Fatalf("%s run %d: G(%v) = %v, exact %v", tc.name, run, x, g[0][0], w[0])
				}
			}
		}
	}
}

// condChainDescriptor is chainDescriptor with U = Q₁·diag(10^(−logCond·i/(n−1)))·Q₂
// (V likewise) and Ff, Wf diagonal with Wf[0] = stiffRel.
func condChainDescriptor(t *testing.T, rng *rand.Rand, nf int, chains []int, stiffRel float64, proper bool, logCond float64) (sys, oracle *System, poles []complex128) {
	t.Helper()
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range nf {
		c.Set(0, i, rng.NormFloat64())
		b.Set(i, 0, rng.NormFloat64())
		f, w := rng.NormFloat64(), 0.5+rng.Float64()
		if i == 0 {
			w = stiffRel
		}
		F.Set(i, i, f)
		W.Set(i, i, w)
		poles = append(poles, complex(f/w, 0))
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			c.Set(0, o+i, rng.NormFloat64())
			if !proper || i == 0 {
				b.Set(o+i, 0, rng.NormFloat64())
			}
		}
		o += k
	}
	orth := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		var qr mat.QR
		qr.Factorize(g)
		var q mat.Dense
		qr.QTo(&q)
		return &q
	}
	conditioned := func() *mat.Dense {
		d := mat.NewDense(n, n, nil)
		for i := range n {
			d.Set(i, i, math.Pow(10, -logCond*float64(i)/float64(n-1)))
		}
		var t, out mat.Dense
		t.Mul(orth(), d)
		out.Mul(&t, orth())
		return &out
	}
	U, V := conditioned(), conditioned()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	D := mat.NewDense(1, 1, []float64{0.3})
	sys, err := NewDescriptor(&A, &B, &C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle, poles
}

// TestDescriptorChainBesideSmallEPoleSweep: chains [2], [3], [2 1], [3 1]
// beside a finite pole small in E (stiffRel 1e-1..1e-4) under cond(U),
// cond(V) up to 1e4. Poles must keep every finite pole, IsStable must
// not report an unstable model stable, and a returned TF must match the
// structured oracle to 1e-5 (base reaches 1.6e-6 here). Before
// leakedZeros, 5 of these lost a pole (2 stability flips) and returned a
// TF off by 2.4e-5..7.5e-4.
func TestDescriptorChainBesideSmallEPoleSweep(t *testing.T) {
	skipSlowUnderRace(t)
	for seed := range uint64(3000) {
		rng := rand.New(rand.NewPCG(seed, 707))
		chains := [][]int{{2}, {3}, {2, 1}, {3, 1}}[rng.IntN(4)]
		nf := 1 + rng.IntN(3)
		sr := math.Pow(10, -1-3*rng.Float64())
		proper := rng.IntN(2) == 0
		sys, oracle, poles := condChainDescriptor(t, rng, nf, chains, sr, proper, 4*rng.Float64())
		wantStable := true
		for _, p := range poles {
			wantStable = wantStable && real(p) < 0
		}
		if st, err := sys.IsStable(); err == nil && st && !wantStable {
			t.Fatalf("seed %d: IsStable = true, poles %v", seed, poles)
		}
		if p, err := sys.Poles(); err == nil {
			requirePolesContain(t, fmt.Sprintf("seed %d", seed), p, poles, 1e-3)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			continue
		}
		for _, s := range []complex128{0.37i, complex(-0.5, 3)} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatalf("seed %d: Eval(%v): %v", seed, s, err)
			}
			w := pencilResponse(t, oracle, s)[0][0]
			if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-5*math.Max(1, cmplx.Abs(w))) {
				t.Fatalf("seed %d: G(%v) = %v, want %v", seed, s, g[0][0], w)
			}
		}
	}
}
