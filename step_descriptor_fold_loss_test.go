package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stepDescriptorCase is a descriptor model (A, B, C, D, E), row-major.
type stepDescriptorCase struct {
	name          string
	n, m, p       int
	a, b, c, d, e []float64
}

func (c stepDescriptorCase) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b), mat.NewDense(c.p, c.n, c.c),
		mat.NewDense(c.p, c.m, c.d), mat.NewDense(c.n, c.n, c.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// stepFoldLossCases are nearSingularEContinuousK draws (seeds 355, 46, 2210,
// 4312; HK6D3N, WOIVWF) with E of full MATLAB rank and cond(E) ~ 1e16,
// whose LU fold E⁻¹A (ToExplicit) misplaces a slow pole by O(0.1): 355 to
// exactly 0, 2210 to +0.099 (a stable model simulated as diverging), 46
// and 4312 into the left half-plane.
var stepFoldLossCases = []stepDescriptorCase{
	{name: "seed 355", n: 2, m: 1, p: 1,
		a: []float64{0.22399326286783883, 0.8402255269963316, -0.3783616459247681, -0.9974968703206462},
		b: []float64{0.9097247634578738, -0.7662664682401428},
		c: []float64{-1.5350587018630855, -1.7388160724801482},
		d: []float64{-0.28270713794423386},
		e: []float64{0.5245065011343613, -0.19913998015010564, -0.001982441398173575, 0.000752675781572609}},
	{name: "seed 46", n: 4, m: 1, p: 1,
		a: []float64{0.5414084488417618, 1.5679446698902575, -1.1476050053714222, 0.48714585101009084, 0.29842073369612226, 0.4784530477013085, 0.3534586715827879, 0.03826668165634744, -0.3468957827943742, -0.4597765039600787, -0.600492027779978, -0.38066805901593326, 0.08850027223164926, 0.7730903828348706, 0.3437969696434461, -0.16787335839168466},
		b: []float64{-1.0866238501149552, 1.8868658258980304, -0.6186020487654416, -0.5625192378754207},
		c: []float64{2.4110016145960063, 0.6405766884785891, -0.175372503188046, 1.3249776484032758},
		d: []float64{0.4486562990926004},
		e: []float64{0.5569856441747267, -0.16455967036395622, 0.1667268459574106, -0.15144396102195812, -0.004920043352418276, 0.22960000410890716, -0.32854613002690847, -0.598615526025465, 0.37687524678697715, 0.06877283632692356, 0.30197676082100755, -0.3218519098316367, 0.28244638106340114, -0.26209416073907515, -0.5040335106125737, -0.08710489309758754}},
	{name: "seed 2210", n: 4, m: 2, p: 1,
		a: []float64{-1.0136393817160623, 0.41020766831170286, -0.09408133555158105, -0.644179004526586, 1.07418114205301, -0.9218150379285601, -1.231723246386637, -0.4687636393685979, -0.4784512557501716, 0.10403339381831038, -0.20208321866594542, -0.33632021307486626, -0.6580025519940232, 0.32980882689020774, 0.3231255322725699, 0.21958078122713193},
		b: []float64{-0.3626171625118464, 0.1944985799807775, -0.7708366418667391, 1.251931747212658, -0.40474469480612923, -1.0336305172761757, -0.9947060093932467, 0.0016913529662841962},
		c: []float64{1.7945515896385373, 0.9289844921303549, -0.7535614308719014, 0.3635245322319877},
		d: []float64{-1.1233747942280599, -1.527784706145031},
		e: []float64{0.06431405765055008, -0.09793366984003522, 0.5845997476492618, 0.07814003323238274, 0.243435295920452, 0.37896528924668094, 0.044798667658080306, -0.2597269810728716, 0.05389243080007255, 0.20790324116625455, 0.16412401124056547, 0.40771103662944136, -0.04850114202678998, 0.25795304454007617, -0.5177231761406489, 0.2995265591040745}},
	{name: "seed 4312", n: 3, m: 1, p: 1,
		a: []float64{-1.508799949444929, -0.9908733309271828, -0.23306747705955372, -2.6807188376475897, -0.574824929849027, -0.6439455090966051, -0.4911676708095334, 0.4326453223586859, -0.3770803441025254},
		b: []float64{-1.8026187197654808, -0.5174592290459684, -0.11128888859008108},
		c: []float64{-0.6301857100380259, 1.9627097714501562, 0.1977142507505732},
		d: []float64{0.8647726842582911},
		e: []float64{0.20418495938098255, 0.12201940186391004, -0.6947173956684332, 0.048638532495039424, 0.2703977125745278, -0.18385760644761337, 0.11292817202362494, -0.7530086338224561, -0.3217698127466444}},
}

// stiffDescriptorModels returns the fold-loss cases and further full-rank
// ill-conditioned-E draws (nearSingularEContinuousK seeds whose base Step
// was off the pencil by 1e-3 to 0.4, and the nearSingularEModel k = 0.7,
// n = 2, seed 2 model of WOIVWF), each with its 320-bit pencil modes; models
// whose E reads rank deficient or whose oracle does not resolve n distinct
// poles on this platform are skipped.
func stiffDescriptorModels(t *testing.T) (names []string, systems []*System, modes []stepOracleModes) {
	t.Helper()
	add := func(name string, sys *System) {
		if !matlabRankFull(t, sys.E) {
			return
		}
		pol, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", name, err)
		}
		m, ok := newStepOracleModes(sys, pol)
		if !ok {
			return
		}
		names, systems, modes = append(names, name), append(systems, sys), append(modes, m)
	}
	for _, c := range stepFoldLossCases {
		add(c.name, c.system(t))
	}
	for _, seed := range []uint64{8, 33, 54, 109, 190, 1439} {
		add(fmt.Sprintf("continuous-k seed %d", seed), nearSingularEContinuousK(t, seed))
	}
	sys, _ := nearSingularEModel(t, 2, 2, 0.7)
	add("k 0.7 n 2 seed 2", sys)
	if len(systems) < 4 {
		t.Fatalf("only %d models resolved", len(systems))
	}
	return names, systems, modes
}

func stepOracleRel(got, want float64) float64 {
	d := math.Abs(got-want) / math.Max(1, math.Abs(want))
	if math.IsNaN(d) {
		return math.Inf(1)
	}
	return d
}

// On a full-rank E whose fold E⁻¹A loses the slow poles to cond(E)·ε, Step
// and Lsim follow the pencil: every sample after t = 0 matches the exact
// partial-fraction step response of the 320-bit generalized eigenvalues,
// and a stable model is not simulated as diverging.
func TestStepFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		_, m, p := sys.Dims()
		st, err := Step(sys, 5)
		if err != nil {
			t.Errorf("%s: Step: %v", names[k], err)
			continue
		}
		worst := 0.0
		_, cols := st.Y.Dims()
		for s := 1; s < cols; s++ {
			for j := range m {
				for i := range p {
					worst = math.Max(worst, stepOracleRel(st.Y.At(j*p+i, s), modes[k].step(i, j, st.T[s])))
				}
			}
		}
		if worst > 1e-7 {
			t.Errorf("%s: Step off the pencil by %.3g (oracle poles %v)", names[k], worst, modes[k].poles)
		}
		u := mat.NewDense(cols, m, nil)
		for s := range cols {
			u.Set(s, 0, 1)
		}
		ls, err := Lsim(sys, u, st.T, nil)
		if err != nil {
			t.Errorf("%s: Lsim: %v", names[k], err)
			continue
		}
		worst = 0
		for s := 1; s < cols; s++ {
			for i := range p {
				worst = math.Max(worst, stepOracleRel(ls.Y.At(i, s), modes[k].step(i, 0, st.T[s])))
			}
		}
		if worst > 1e-7 {
			t.Errorf("%s: Lsim of a unit step off the pencil by %.3g", names[k], worst)
		}
	}
}

// Impulse of the same models matches Σ Rᵢ·e^{λᵢt} of the pencil after
// t = 0; the step form of the derived model would cancel C·B ~ 1e14 of the
// stiff mode against its state.
func TestImpulseFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		_, m, p := sys.Dims()
		im, err := Impulse(sys, 5)
		if err != nil {
			t.Errorf("%s: Impulse: %v", names[k], err)
			continue
		}
		worst := 0.0
		_, cols := im.Y.Dims()
		for s := 1; s < cols; s++ {
			for j := range m {
				for i := range p {
					want := 0.0
					for q, l := range modes[k].poles {
						if real(l)*im.T[s] > -700 {
							want += real(modes[k].residues[q][i][j] * cmplx.Exp(l*complex(im.T[s], 0)))
						}
					}
					worst = math.Max(worst, stepOracleRel(im.Y.At(j*p+i, s), want))
				}
			}
		}
		if worst > 1e-7 {
			t.Errorf("%s: Impulse off the pencil by %.3g", names[k], worst)
		}
	}
}

// C2D (ZOH) of the same models is the exact ZOH of the pencil's modes:
// its frequency response on the unit circle matches the 320-bit partial
// fractions and its stability is the pencil's.
func TestC2DZOHFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		_, m, p := sys.Dims()
		d, err := sys.C2D(0.1, C2DOptions{})
		if err != nil {
			t.Errorf("%s: C2D: %v", names[k], err)
			continue
		}
		worst := 0.0
		for _, th := range []float64{0.05, 0.5, 2, 3} {
			z := cmplx.Exp(complex(0, th))
			g := pencilResponse(t, d, z)
			for i := range p {
				for j := range m {
					want := modes[k].zoh(i, j, 0.1, z)
					worst = math.Max(worst, cmplx.Abs(g[i][j]-want)/math.Max(1, cmplx.Abs(want)))
				}
			}
		}
		if !(worst <= 1e-9) {
			t.Errorf("%s: C2D frequency response off the pencil by %.3g", names[k], worst)
		}
		got, err := d.IsStable()
		if err != nil {
			t.Fatal(err)
		}
		if got != modes[k].stable() {
			t.Errorf("%s: IsStable(C2D) = %v, pencil poles %v", names[k], got, modes[k].poles)
		}
	}
}
