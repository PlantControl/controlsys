package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Step and Initial over tFinal 120 of stiffQZFoldCases seed 50 (cond(E)
// 3.3e6) keep the ZOH discretization of the generalized Schur parts
// (DIUI37): cond(E)·ε is below dt·ρ on the 1.2e-3 grid, but the exact
// chain of the LU fold exceeds delayChainMaxWork there, so the fold would
// take the ZOH discretization too, after building the rejected chain.
func TestStepInternalDelayStiffDescriptorKeepsPartsWhenFoldChainOverBudget(t *testing.T) {
	const tf = 120
	tc := stiffQZFoldCases[0]
	if tc.seed != 50 {
		t.Fatalf("stiffQZFoldCases[0] is seed %d, want 50", tc.seed)
	}
	sys := tc.system(t)
	x0 := mat.NewVecDense(tc.n, []float64{0.7, -0.4, 1.1})

	zs, err := sys.zeroStateResponseForm()
	if err != nil {
		t.Fatal(err)
	}
	explicit, tau, ok := delayChainModel(zs)
	if !ok {
		t.Fatal("fold outside the delay chain class")
	}
	tg, dt, err := newTimeResponsePlanner(zs).grid(tf, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := newDelayChain(explicit, tau, []int{0}, dt, len(tg), delayShiftCount(zs)); ok {
		t.Fatal("Step: fold chain within budget")
	}
	sim, _, _, err := sys.timeResponseForm(x0)
	if err != nil {
		t.Fatal(err)
	}
	free := sim.Copy()
	free.InputDelay, free.Delay = nil, nil
	explicit, tau, _ = delayChainModel(free)
	tg, dt, err = newTimeResponsePlanner(free).grid(tf, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := newDelayChain(explicit, tau, nil, dt, len(tg), delayShiftCount(free)); ok {
		t.Fatal("Initial: fold chain within budget")
	}

	ex, ok := stiffLFTExplicit(sys)
	if !ok {
		t.Fatal("stiffLFTExplicit: no split")
	}
	plan, err := prepareAutoTimeResponse(ex, tf, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := plan.allInputResponse(stepResponse)
	if err != nil {
		t.Fatal(err)
	}
	st, err := Step(sys, tf)
	if err != nil {
		t.Fatal(err)
	}
	if len(st.T) != len(want.T) || !mat.Equal(st.Y, want.Y) {
		t.Error("Step is not the ZOH step response of the parts")
	}

	exi, xt, ok := stiffLFTExplicitState(sys, x0)
	if !ok {
		t.Fatal("stiffLFTExplicitState: no split")
	}
	simi, simX0, _, err := exi.timeResponseForm(xt)
	if err != nil {
		t.Fatal(err)
	}
	freei := simi.Copy()
	freei.InputDelay, freei.Delay = nil, nil
	plani, err := prepareAutoTimeResponse(freei, tf, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, m, _ := plani.sim.Dims()
	resp, err := plani.sim.Simulate(mat.NewDense(m, plani.steps, nil), simX0, nil)
	if err != nil {
		t.Fatal(err)
	}
	wanti := plani.response(resp.Y)
	in, err := Initial(sys, x0, tf)
	if err != nil {
		t.Fatal(err)
	}
	if len(in.T) != len(wanti.T) || !mat.Equal(in.Y, wanti.Y) {
		t.Error("Initial is not the ZOH free response of the parts")
	}
}
