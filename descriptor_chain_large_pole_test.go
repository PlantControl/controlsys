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

// TestDescriptorStiffChainConditioned: EVSFPV. condChainDescriptor chains
// [4], [5] whose E singular value of the finite pole (σ/σ₁ 6e-9 to 1.8e-8)
// is above MATLAB's rank floor but within qzInfiniteGap of the staircase
// tol, so descriptorRank sets stiff, beside index ≥ 2. The staircase
// stopped after the null space of E, and IsProper, Step and
// TransferFunction returned ErrDescriptorSingular ("infinite Jordan chain
// left among the finite eigenvalues"), improper and proper alike. MATLAB
// rank keeps the pole finite (D62TAA): deflateStiffChain deflates the whole
// chain, and the split stands when it matches the pencil at the data scale
// and 4ε copies of the data reduce alike. [5] seed 92 improper failed
// closed: its split misses the direct solve at the data scale by 6.6e-10
// (> 1e-10) and every 4ε copy failed alike; MATLAB isproper gives false.
// It now stands, as improper, when its refit at the same degree matches
// the (sE − A) samples below the data scale with dominant s-terms
// (refitPolyKeepDegree). Oracle: the structured model's pencil response
// and finite pole count.
func TestDescriptorStiffChainConditioned(t *testing.T) {
	for _, tc := range []struct {
		chain    int
		seed     uint64
		proper   bool
		singular bool
	}{{4, 92, false, false}, {5, 92, false, false}, {5, 193, false, false}, {4, 92, true, false}, {5, 92, true, false}} {
		label := fmt.Sprintf("[%d] seed %d proper %v", tc.chain, tc.seed, tc.proper)
		rng := rand.New(rand.NewPCG(tc.seed, 911))
		nf := 1 + rng.IntN(3)
		sr := math.Pow(10, -2-2*rng.Float64())
		sys, oracle, _ := condChainDescriptor(t, rng, nf, []int{tc.chain}, sr, tc.proper, 1+2*rng.Float64())
		n, _, _ := sys.Dims()
		var svd mat.SVD
		if !svd.Factorize(sys.E, mat.SVDNone) {
			t.Fatal("SVD failed")
		}
		sv := svd.Values(nil)
		if _, stiff, _ := descriptorRank(sv, sv[0], n, true); !stiff {
			t.Fatalf("%s: E not stiff, σ_{n-1}/σ₁ = %g", label, sv[n-2]/sv[0])
		}
		if _, _, _, _, k, _, _, _, ok := deflateStiffChain(sys.A, sys.E); !ok || k != nf {
			t.Errorf("%s: deflateStiffChain nf %d ok %v, want %d finite", label, k, ok, nf)
		}
		pr, err := sys.IsProper()
		_, serr := Step(sys, 1)
		res, terr := sys.TransferFunction(nil)
		if tc.singular {
			for _, e := range []error{err, serr, terr} {
				if !errors.Is(e, ErrDescriptorSingular) {
					t.Errorf("%s: err %v, want ErrDescriptorSingular", label, e)
				}
			}
			continue
		}
		if err != nil || pr != tc.proper {
			t.Fatalf("%s: IsProper = %v, %v; want %v", label, pr, err, tc.proper)
		}
		if tc.proper && serr != nil || !tc.proper && !errors.Is(serr, ErrImproperModel) {
			t.Fatalf("%s: Step err = %v", label, serr)
		}
		if terr != nil {
			t.Fatalf("%s: TransferFunction: %v", label, terr)
		}
		if fzTFImproper(res.TF) == tc.proper {
			t.Fatalf("%s: TF improper %v, IsProper %v", label, !tc.proper, tc.proper)
		}
		for _, s := range []complex128{0.37i, complex(0.5, 3), 1e2i, 1e4i, 1e7i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, oracle, s)[0][0]
			if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-5*math.Max(1, cmplx.Abs(w))) {
				t.Errorf("%s: G(%v) = %v, want %v", label, s, g[0][0], w)
			}
		}
	}
}

// TestDescriptorStiffChainSweep: EVSFPV. Proper [4] chains beside a pole
// of E singular value 3e-8..1e-9 relative (chainDescriptor), stiff for
// descriptorRank: base read 3/300 proper and 290 failed closed;
// deflateStiffChain with its checks reads 46 (53 amd64), and the 7 (6)
// whose reduction misses the oracle are base's (another path). Over the chain
// sweeps (index 2..5, 10800 draws) the stiff chain split only turned
// ErrDescriptorSingular into results that match the oracle.
func TestDescriptorStiffChainSweep(t *testing.T) {
	good, bad := 0, 0
	for seed := range uint64(300) {
		rng := rand.New(rand.NewPCG(seed, 5))
		nf := 1 + rng.IntN(3)
		sr := math.Pow(10, -7.5-1.5*rng.Float64())
		sys, oracle, _ := chainDescriptor(t, rng, nf, []int{4}, sr, true)
		pr, err := sys.IsProper()
		if err != nil {
			continue
		}
		res, terr := sys.TransferFunction(nil)
		if !pr || terr != nil {
			bad++
			continue
		}
		ok := true
		for _, s := range []complex128{0.37i, complex(0.5, 3), 1e2i, 1e4i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, oracle, s)[0][0]
			ok = ok && cmplx.Abs(g[0][0]-w) <= 1e-5*math.Max(1, cmplx.Abs(w))
		}
		if ok {
			good++
		} else {
			bad++
		}
	}
	t.Logf("%d match, %d do not", good, bad)
	if good < 35 || bad > 10 {
		t.Errorf("%d proper [4] chains match the oracle (want ≥ 35), %d do not (want ≤ 10)", good, bad)
	}
}

// refutedChainDescriptor: MIMO 3×3 index-6 proper chain beside the finite
// pole 1e5/w (F = diag(1e5, f2, f3, I6), W = diag(w1, w2, w3, N6),
// w ∈ [0.5, 1.5); b nonzero at the finite states and the chain's first,
// c and D Gaussian; A = U·F·V, E = U·W·V, Gaussian U, V). Oracle: the
// structured model (F, W, b, c, D).
func refutedChainDescriptor(t *testing.T, seed uint64) (sys, oracle *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 4242))
	nf, k, p, m := 3, 6, 3, 3
	n := nf + k
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range nf {
		f := rng.NormFloat64()
		if i == 0 {
			f = 1e5
		}
		F.Set(i, i, f)
		W.Set(i, i, 0.5+rng.Float64())
	}
	for i := range k {
		F.Set(nf+i, nf+i, 1)
		if i+1 < k {
			W.Set(nf+i, nf+i+1, 1)
		}
	}
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			if i <= nf {
				b.Set(i, j, rng.NormFloat64())
			}
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64())
		}
	}
	U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			U.Set(i, j, rng.NormFloat64())
			V.Set(i, j, rng.NormFloat64())
		}
	}
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, 0); err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

// TestDescriptorRefutedReductionFailsClosed: BAQHZG. Beside an index ≥ 2
// chain the split misallocates residue to a large finite pole; the direct
// (sE − A) samples of refitFromPencil show it (mismatch 1e10..1e15 times
// their uncertainty; correct reductions ≤ 6), but there were too few to
// refit (B/55: 8 at |s| 0.01..10, 10 needed) or none fit, and base
// returned the reduction as proper: B/55 TF(0.05i) −0.545 − 15.44i
// against −0.544 + 0.285i, 3e5 relative at 1e3i; MIMO chains [6] beside
// 1.67e5 (seed 0: 1e5 relative). Now IsProper, Step and TransferFunction
// fail closed alike; proper chains whose reduction is right still return
// TFs that match the structured oracle. KLVET5: 35 more of the 200 MIMO
// chains (59 before, 94 now) read proper with a TF that matches, where
// their split had passed residue for s-terms the pencil solves refute.
func TestDescriptorRefutedReductionFailsClosed(t *testing.T) {
	failsClosed := func(label string, sys *System) bool {
		t.Helper()
		_, perr := sys.IsProper()
		_, serr := Step(sys, 1)
		_, terr := sys.TransferFunction(nil)
		n := 0
		for _, e := range []error{perr, serr, terr} {
			if errors.Is(e, ErrDescriptorSingular) {
				n++
			}
		}
		if n != 0 && n != 3 {
			t.Errorf("%s: IsProper %v, Step %v, TF %v disagree", label, perr, serr, terr)
		}
		return n == 3
	}
	rng := rand.New(rand.NewPCG(55, 911))
	chains := [][]int{{2}, {3}, {4}, {5}, {6}, {2, 3}, {3, 2}}[rng.IntN(7)]
	proper := rng.IntN(3) != 0
	nf := 1 + rng.IntN(3)
	sr := math.Pow(10, -1-5*rng.Float64())
	sys, orc, _ := condChainDescriptor(t, rng, nf, chains, sr, proper, 1+2*rng.Float64())
	// Borderline: fails closed natively; FMA contraction (-race) resolves it.
	if !failsClosed("B/55", sys) {
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("B/55: TF %v without failing closed", err)
		}
		for _, s := range []complex128{0.37i, complex(0.5, 3), 20i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			want := pencilResponse(t, orc, s)[0][0]
			if d := cmplx.Abs(g[0][0] - want); !(d <= 1e-8*math.Max(1, cmplx.Abs(want))) {
				t.Errorf("B/55: G(%v) = %v, want %v (or ErrDescriptorSingular)", s, g[0][0], want)
			}
		}
	}

	good, closed := 0, 0
	for seed := range uint64(200) {
		label := fmt.Sprintf("C/%d", seed)
		sys, oracle := refutedChainDescriptor(t, seed)
		if failsClosed(label, sys) {
			closed++
			continue
		}
		pr, err := sys.IsProper()
		if err != nil || !pr {
			continue
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", label, err)
		}
		for _, s := range []complex128{0.05i, 0.37i, complex(0.5, 3), 30i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, oracle, s)
			for i := range w {
				for j := range w[i] {
					if d := cmplx.Abs(g[i][j] - w[i][j]); !(d <= 1e-5*math.Max(1, cmplx.Abs(w[i][j]))) {
						t.Errorf("%s: IsProper true, G(%v)[%d][%d] = %v, want %v", label, s, i, j, g[i][j], w[i][j])
					}
				}
			}
		}
		good++
	}
	t.Logf("%d proper TFs match, %d fail closed", good, closed)
	if good < 85 || closed == 0 {
		t.Errorf("%d proper TFs match the oracle (want ≥ 85), %d fail closed (want > 0)", good, closed)
	}
}
