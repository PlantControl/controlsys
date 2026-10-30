package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Step and Initial of a stiff-E descriptor with internal delays take the
// route Lsim takes on their automatic grid (PJIDAF): at cond(E) about 1e8,
// where cond(E)·ε is below dt·ρ on the 1e-5 grid of tFinal 1, Step is the
// exact delay chain of the LU fold, at its cond(E)·ε accuracy; at about
// 1e13, where the fold would lose the slow modes, it stays the ZOH step
// response of the generalized Schur parts, within lftStepTol. Both against
// the 320-bit method of steps.
func TestStepInternalDelayStiffDescriptorRouteFollowsAutoGrid(t *testing.T) {
	for _, tc := range []struct {
		k    float64
		fold bool
	}{{8, true}, {13, false}} {
		sys := stiffGridCase(t, tc.k)
		if !matlabRankFull(t, sys.E) {
			t.Fatalf("k %g: E not of full rank", tc.k)
		}
		var lu mat.LU
		lu.Factorize(sys.E)
		cond := lu.Cond()
		st, err := Step(sys, 1)
		if err != nil {
			t.Fatal(err)
		}
		var want *TimeResponse
		if tc.fold {
			zs, err := sys.zeroStateResponseForm()
			if err != nil {
				t.Fatal(err)
			}
			var ok bool
			if want, ok, err = delayChainAuto(zs, 1, stepResponse); err != nil || !ok {
				t.Fatalf("k %g: no delay chain of the fold: %v", tc.k, err)
			}
		} else {
			ex, ok := stiffLFTExplicit(sys)
			if !ok {
				t.Fatalf("k %g: stiffLFTExplicit: no split", tc.k)
			}
			plan, err := prepareAutoTimeResponse(ex, 1, 0)
			if err != nil {
				t.Fatal(err)
			}
			if want, err = plan.allInputResponse(stepResponse); err != nil {
				t.Fatal(err)
			}
		}
		if !mat.Equal(st.Y, want.Y) || len(st.T) != len(want.T) {
			t.Errorf("k %g (cond(E) %.3g): Step is not the response of the fold route %v", tc.k, cond, tc.fold)
		}
		o := newLFTStepOracle(t, sys)
		x0 := []float64{0.7, -0.4, 1.1}
		in, err := Initial(sys, mat.NewVecDense(3, x0), 1)
		if err != nil {
			t.Fatal(err)
		}
		worstStep, worstInitial := 0.0, 0.0
		for _, s := range lftOracleSamples(len(st.T)) {
			if !nearDelayGrid(st.T[s], o.tau) {
				worstStep = math.Max(worstStep, stepOracleRel(st.Y.At(0, s), o.response(0, nil, st.T[s])[0]))
			}
		}
		for _, s := range lftOracleSamples(len(in.T)) {
			if !nearDelayGrid(in.T[s], o.tau) {
				worstInitial = math.Max(worstInitial, stepOracleRel(in.Y.At(0, s), o.response(-1, x0, in.T[s])[0]))
			}
		}
		tol := lftStepTol
		if tc.fold {
			tol = 10 * cond * 0x1p-52
		}
		t.Logf("k %g cond(E) %.3g: Step %.3g, Initial %.3g", tc.k, cond, worstStep, worstInitial)
		if worstStep > tol || worstInitial > tol {
			t.Errorf("k %g (cond(E) %.3g): Step off the pencil by %.3g, Initial by %.3g, want <= %.3g", tc.k, cond, worstStep, worstInitial, tol)
		}
	}
}

// stiffQZFoldCase is a descriptor with one internal delay whose pencil
// was built in generalized Schur form with one stiff E diagonal (stiffQZ
// generator, PJIDAF seeds 50, 54, 331): cond(E) 3.3e6, 1.2e6, 2.2e9.
type stiffQZFoldCase struct {
	seed                    int
	n, m, p                 int
	tau                     float64
	a, b, c, d, e           []float64
	b2, c2, d12, d21        []float64
	inputDelay, outputDelay []float64
}

func (tc stiffQZFoldCase) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b), mat.NewDense(tc.p, tc.n, tc.c),
		mat.NewDense(tc.p, tc.m, tc.d), mat.NewDense(tc.n, tc.n, tc.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{tc.tau}, mat.NewDense(tc.n, 1, tc.b2), mat.NewDense(1, tc.n, tc.c2),
		mat.NewDense(tc.p, 1, tc.d12), mat.NewDense(1, tc.m, tc.d21), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	sys.InputDelay, sys.OutputDelay = tc.inputDelay, tc.outputDelay
	return sys
}

var stiffQZFoldCases = []stiffQZFoldCase{
	{seed: 50, n: 3, m: 1, p: 1, tau: 0.3,
		a:   []float64{-0.81981769965881091, 0.96165132347033133, -0.57855779247322914, -1.28429785623257, -1.7607754896686059, 0.46692845387714732, 0.021870811790010918, 1.4860702434097668, -1.1072786517617943},
		b:   []float64{0.55113137291096037, 0.2926872660852699, 0.94688314424278275},
		c:   []float64{0.81541976950329054, 0.85431372447894571, 0.32828240560482891},
		d:   []float64{-0.13347992733768965},
		e:   []float64{0.52870843799765799, -0.70210419040702021, 0.46260768213877373, 0.72000519735268287, 0.43187810843150187, -0.41053862963263327, -0.0051011652156024616, -0.75384611114504319, 0.56573656321188837},
		b2:  []float64{0.29508722063708664, 0.018340412339922452, -0.28979534338328822},
		c2:  []float64{0.39395929906480115, 0.7080681658396677, 0.18662331358078621},
		d12: []float64{-0.49064935254211656},
		d21: []float64{-0.067429927748446958}},
	{seed: 54, n: 2, m: 2, p: 2, tau: 0.3,
		a:          []float64{-1.7060383325322708, 1.0541554650037057, 1.170861870743682, -1.296335328232278},
		b:          []float64{1.3883236165756938, -0.69304701103587973, -0.30647945398343779, -0.019884446256282828},
		c:          []float64{1.0583234928508212, 0.46184596993107063, -1.1591880704469837, 0.86311905670508016},
		d:          []float64{-0.5326837775037645, 0.39961805998055017, 0.12267972131153124, -0.39530903617998359},
		e:          []float64{0.58354837809164262, -0.83881151138091847, -0.50669100410583912, 0.72833740627953769},
		b2:         []float64{-0.085308525996063089, 0.45719241224515561},
		c2:         []float64{-0.41445298788738233, -0.10801155086719304},
		d12:        []float64{0.032773411100341376, 0.62294767685354913},
		d21:        []float64{-0.43120197972156693, 0.064334609891980463},
		inputDelay: []float64{0.2, 0}, outputDelay: []float64{0.1, 0}},
	{seed: 331, n: 2, m: 2, p: 1, tau: 0.25,
		a:   []float64{-1.2156500190759345, -0.85968799276925612, 0.67134702081495856, -1.7399279546193609},
		b:   []float64{0.90502348418131517, -2.6462400462937747, 1.8513300264873553, 1.7288272961633382},
		c:   []float64{-0.97058193525002245, -1.2448466728581082},
		d:   []float64{0.076818877709705635, -0.40613300439165945},
		e:   []float64{1.0507836868146694, -0.51293719196247223, -0.92681055980577365, 0.45242004933215396},
		b2:  []float64{0.80790624134344569, 0.1820290608387497},
		c2:  []float64{-0.33317615234674164, -0.88107425985498444},
		d12: []float64{0.50029471237676337},
		d21: []float64{-0.36630779987948325, 0.57052027190265153}},
}

// Step and Initial over tFinal 1.2 of stiffQZFoldCases (cond(E) 1.2e6 to
// 2.2e9) stay within 10·cond(E)·ε of the 320-bit method of steps, I/O
// delays included (PJIDAF): the 1.2e-5 auto grid takes the exact chain of
// the fold, where the ZOH discretization of the parts was off by up to
// 2.4e-5 (seed 50, cond(E) 3.3e6, bound 7.3e-9).
func TestStepInternalDelayStiffDescriptorFoldAccurateOnFineAutoGrid(t *testing.T) {
	for _, tc := range stiffQZFoldCases {
		sys := tc.system(t)
		var lu mat.LU
		lu.Factorize(sys.E)
		tol := 10 * lu.Cond() * 0x1p-52
		x0 := make([]float64, tc.n)
		for i := range x0 {
			x0[i] = math.Cos(float64(3*i + 1))
		}
		st, err := Step(sys, 1.2)
		if err != nil {
			t.Fatalf("seed %d: Step: %v", tc.seed, err)
		}
		in, err := Initial(sys, mat.NewVecDense(tc.n, x0), 1.2)
		if err != nil {
			t.Fatalf("seed %d: Initial: %v", tc.seed, err)
		}
		o := newLFTStepOracle(t, sys)
		worstStep, worstInitial := 0.0, 0.0
		for _, s := range lftOracleSamples(len(st.T)) {
			for j := range tc.m {
				want, ok := o.delayedResponse(sys, j, nil, st.T[s])
				if !ok {
					continue
				}
				for i := range tc.p {
					worstStep = math.Max(worstStep, stepOracleRel(st.Y.At(j*tc.p+i, s), want[i]))
				}
			}
		}
		for _, s := range lftOracleSamples(len(in.T)) {
			want, ok := o.delayedResponse(sys, -1, x0, in.T[s])
			if !ok {
				continue
			}
			for i := range tc.p {
				worstInitial = math.Max(worstInitial, stepOracleRel(in.Y.At(i, s), want[i]))
			}
		}
		t.Logf("seed %d cond(E) %.3g: Step %.3g, Initial %.3g, bound %.3g", tc.seed, lu.Cond(), worstStep, worstInitial, tol)
		if worstStep > tol || worstInitial > tol {
			t.Errorf("seed %d (cond(E) %.3g): Step off the pencil by %.3g, Initial by %.3g, want <= %.3g", tc.seed, lu.Cond(), worstStep, worstInitial, tol)
		}
	}
}
