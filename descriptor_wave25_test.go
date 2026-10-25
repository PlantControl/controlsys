package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// largePoleChainA..C: index-5 proper chain beside a finite pole 2.473e5
// (F = diag(2.473e5, f2, f3, I5), W = diag(1, w2, w3, N5), A = U·F·V,
// E = U·W·V, Gaussian U, V; D = 0.3). E is well conditioned; QZ smears the
// chain to |λ| ~ 6e2, and the Weierstrass split moved a residue 5.3e9 onto
// the pole 2.473e5 with D = 2.1e4 and an s-term −0.086 dropped as rounding.
var largePoleChainA = []float64{-187189.11483614656, -184119.93119067902, 247149.16999821848, -159856.5118745356, -35284.54782313305, 216981.0540625069, 462672.158623724, 45336.46195640846, 106333.6763908921, 104591.7817467216, -140396.279826432, 90808.54940551118, 20043.548912723607, -123256.52425932903, -262828.43968283356, -25752.884956238056, -157897.78576569995, -155315.86111496142, 208474.96417667452, -134843.29132827124, -29754.850435690732, 183028.6689711936, 390271.53256964777, 38244.56310939547, 26183.707221923694, 25756.41680480757, -34570.99120665872, 22359.11156419534, 4933.213166093224, -30350.02521846589, -64715.50223435076, -6344.720426434853, -241287.8264548597, -237341.91352833618, 318585.02698025736, -206057.0128348201, -45469.33729077961, 279687.2163900194, 596389.7733272803, 58437.16346737191, -21696.821360826525, -21341.22293758545, 28648.05137290495, -18526.789550894984, -4086.217928410938, 25147.145892097302, 53627.325793551885, 5253.08605487544, 28023.25579529715, 27561.632460348683, -37003.097501851036, 23932.839514579427, 5283.98604800324, -32483.503535155483, -69265.08215298681, -6786.08287478378, 159302.10306347304, 156694.519609872, -210337.82697089785, 136038.38637531857, 30020.060701884857, -184655.5449803762, -393747.7686750795, -38580.652428524234}
var largePoleChainE = []float64{1.3952040838916944, -1.597377879947066, 3.945321321255862, 1.580489463883563, 2.0781926731831426, 6.489814460781639, -1.6912830349144798, 1.0236094987704776, -2.8668786221424982, 3.222112974073026, -2.048038281285443, -0.5250666446389666, -2.385261920118196, -1.7820227421859927, 2.2704874193947235, -1.6122667655565939, -2.6790475795050805, -4.426469155575041, 2.4639059179997016, -0.9980494795870568, 5.254745360565756, -1.6474415073105235, 1.4799222253989226, 1.0560475957135105, -1.0245373497659804, 0.3324359385396623, -1.8205727981271695, 1.5021871757000436, -0.6577726901197012, 2.005640096052203, -1.7866449039523935, 2.942720240835751, -2.5272132578407893, 2.1194360301488433, 3.770054015218346, -0.7096415636289631, -2.490612341080245, 1.0707879108454725, 4.996845194993217, -0.31515290025928494, -0.4373049148923571, 1.7949653683882263, -1.4532976685581511, 0.019653023377930786, -2.843536170078665, -2.068780893135851, 1.971537227621646, 0.6352176935057168, -1.6172325405933254, 2.2284429866647995, 4.17684870633686, -1.894443922872639, -1.65016242468186, 1.2382486077410018, 1.610747287410093, -2.1990726761693464, 1.0255434297585357, -3.479182118812013, 0.6861629085670752, 1.6758299678261728, 6.386622713339333, 0.8950548340924217, -4.34788492032227, 0.30298345350629197}
var largePoleChainB = []float64{-0.8591258295953126, -0.3401200749319086, 4.125685710650072, 0.8500337810153541, -1.3255580727754868, -0.8577367267584565, -0.8489782950254071, 2.4644968432566063}
var largePoleChainC = []float64{-3.5110472217753177, 1.8977194510182793, 0.34828575188916144, -2.8943708024306, -2.9050191314794764, 0.832156942217585, 1.3830388417038522, 0.3713314577465813}

// TestTransferFunction_ChainBesideLargePole: BAQHZG #6. Base returned a
// proper TF 0.03 off at 0.37i (−0.409917 − 0.235195i, amd64 −0.157592i)
// and 0.86 off at 10i; its reductionMatches never ran, and its points at
// |s| ~ ‖A‖/‖E‖ ~ 1e5 cannot be solved directly. Oracle: an exact rational
// solve of (sE − A) from the float data, −0.409917 − 0.203396i at 0.37i,
// at |s| ≤ 1 where the data determine the response to 1e-8 relative
// (above |s| ~ 1e2 the smeared chain eigenvalues of the float data take
// over). Poles keep {2.47307e5, −0.9166, −1.6097}.
func TestTransferFunction_ChainBesideLargePole(t *testing.T) {
	sys, err := NewDescriptor(mat.NewDense(8, 8, largePoleChainA), mat.NewDense(8, 1, largePoleChainB),
		mat.NewDense(1, 8, largePoleChainC), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(8, 8, largePoleChainE), 0)
	if err != nil {
		t.Fatal(err)
	}
	if pr, err := sys.IsProper(); err != nil || !pr {
		t.Fatalf("IsProper = %v, %v, want true", pr, err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	requirePolesContain(t, "poles", poles, []complex128{2.47307089e5, -0.91664174, -1.60971647}, 1e-4)
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []complex128{0.37i, 1i, complex(0.3, 0.4), 0.05i} {
		g, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want, ok := fzExactDescResponse(8, 1, 1, largePoleChainA, largePoleChainE, largePoleChainB, largePoleChainC, []float64{0.3}, s)
		if !ok {
			t.Fatal("exact solve failed")
		}
		if d := cmplx.Abs(g[0][0] - want[0]); !(d <= 1e-8*cmplx.Abs(want[0])) {
			t.Errorf("TF(%v) = %v, want %v (|diff| %.3g)", s, g[0][0], want[0], d)
		}
	}
}

// TestTransferFunction_StiffProperChainSweep: BAQHZG #3 and #5. Beside a
// finite pole small in E (Wf[0] = stiffRel) the split leaks an O(1)
// residue into the nilpotent block; the first-order absorbStiffLeak left a
// second-order remainder, so base read proper chains improper ([5] 1e-4:
// 43/200, [6] 1e-3: 13/200; seed 125/77 [3] 1e-6: 18× off natively, 2.3e6
// relative on amd64 with IsProper true). Oracle: the structured model of
// chainDescriptor; a returned TF must match it to 1e-5 up to 1e6i. An
// ErrDescriptorSingular from the split (amd64, [3] 1e-5 seed 125, as on
// base) counts as a false read.
func TestTransferFunction_StiffProperChainSweep(t *testing.T) {
	check := func(label string, sys, orc *System) bool {
		t.Helper()
		pr, err := sys.IsProper()
		if err != nil || !pr {
			return false
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", label, err)
		}
		for _, s := range []complex128{0.37i, complex(0.5, 3), 30i, 1e3i, 1e4i, 1e6i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, orc, s)[0][0]
			if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-5*math.Max(1, cmplx.Abs(w))) {
				t.Fatalf("%s: G(%v) = %v, want %v", label, s, g[0][0], w)
			}
		}
		return true
	}
	rng := rand.New(rand.NewPCG(125, 77))
	nf := 1 + rng.IntN(3)
	sys, orc, _ := chainDescriptor(t, rng, nf, []int{3}, 1e-6, true)
	if !check("seed 125 [3] 1e-6", sys, orc) {
		t.Errorf("seed 125 [3] 1e-6: proper chain IsProper false")
	}
	for _, tc := range []struct {
		chain    int
		stiffRel float64
	}{{5, 1e-4}, {6, 1e-3}, {3, 1e-5}} {
		improper := 0
		for seed := range uint64(200) {
			rng := rand.New(rand.NewPCG(seed, 77))
			nf := 1 + rng.IntN(3)
			sys, orc, _ := chainDescriptor(t, rng, nf, []int{tc.chain}, tc.stiffRel, true)
			if !check(fmt.Sprintf("[%d] %g seed %d", tc.chain, tc.stiffRel, seed), sys, orc) {
				improper++
			}
		}
		if improper > 3 {
			t.Errorf("[%d] %g: %d/200 proper chains read improper or failed, want ≤ 3", tc.chain, tc.stiffRel, improper)
		}
	}
}
