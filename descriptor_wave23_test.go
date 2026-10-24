package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
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
