package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// delayChainWork grows strictly with the truncation order, so newDelayChain
// rejecting at the first over-budget order of its doubling search matches
// rejecting the final order (DIUI37).
func TestDelayChainWorkGrowsWithOrder(t *testing.T) {
	for _, blocks := range []float64{1, 9, 401} {
		for _, steps := range []float64{2, 1001, 100001} {
			for _, na := range []float64{1, 3, 6} {
				for _, shifts := range []float64{1, 3} {
					prev := delayChainWork(blocks, steps, 0, na, shifts)
					for order := 1.0; order <= 64; order++ {
						w := delayChainWork(blocks, steps, order, na, shifts)
						if !(w > prev) {
							t.Fatalf("blocks %g steps %g na %g shifts %g: work %g at order %g, %g at order %g", blocks, steps, na, shifts, w, order, prev, order-1)
						}
						prev = w
					}
				}
			}
		}
	}
}

// The fold chains of stiffQZFoldCases over tFinal 1.2, 12, 120 and auto:
// delayChainWithinBudget is necessary for newDelayChain, and an accepted
// chain fits delayChainMaxWork at its final order.
func TestDelayChainWithinBudgetNecessaryForFoldChain(t *testing.T) {
	accepted, rejected := 0, 0
	for _, tc := range stiffQZFoldCases {
		sys := tc.system(t)
		x0 := make([]float64, tc.n)
		for i := range x0 {
			x0[i] = float64(i+1) / float64(tc.n)
		}
		zs, err := sys.zeroStateResponseForm()
		if err != nil {
			t.Fatal(err)
		}
		sim, _, _, err := sys.timeResponseForm(mat.NewVecDense(tc.n, x0))
		if err != nil {
			t.Fatal(err)
		}
		free := sim.Copy()
		free.InputDelay, free.Delay = nil, nil
		for _, tf := range []float64{1.2, 12, 120, 0} {
			for k, form := range []*System{zs, free} {
				explicit, tau, ok := delayChainModel(form)
				if !ok {
					t.Fatalf("seed %d: fold outside the delay chain class", tc.seed)
				}
				tg, dt, err := newTimeResponsePlanner(form).grid(tf, 0)
				if err != nil {
					t.Fatal(err)
				}
				var inputs []int
				if k == 0 {
					inputs = []int{0}
				}
				shifts := delayShiftCount(form)
				fits := delayChainWithinBudget(form, dt, len(tg), len(inputs), shifts)
				c, ok := newDelayChain(explicit, tau, inputs, dt, len(tg), shifts)
				if !ok {
					rejected++
					continue
				}
				accepted++
				if !fits {
					t.Errorf("seed %d tf %g form %d: chain accepted, delayChainWithinBudget false", tc.seed, tf, k)
				}
				if w := delayChainWork(float64(c.blocks), float64(c.steps), float64(c.order), float64(c.na), float64(shifts)); w > delayChainMaxWork {
					t.Errorf("seed %d tf %g form %d: accepted at order %d with work %g", tc.seed, tf, k, c.order, w)
				}
			}
		}
	}
	if accepted == 0 || rejected == 0 {
		t.Fatalf("accepted %d, rejected %d chains: want both", accepted, rejected)
	}
}

func BenchmarkNewDelayChainOverBudget(b *testing.B) {
	tc := stiffQZFoldCases[0]
	sys, err := NewDescriptor(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b), mat.NewDense(tc.p, tc.n, tc.c),
		mat.NewDense(tc.p, tc.m, tc.d), mat.NewDense(tc.n, tc.n, tc.e), 0)
	if err != nil {
		b.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{tc.tau}, mat.NewDense(tc.n, 1, tc.b2), mat.NewDense(1, tc.n, tc.c2),
		mat.NewDense(tc.p, 1, tc.d12), mat.NewDense(1, tc.m, tc.d21), mat.NewDense(1, 1, nil)); err != nil {
		b.Fatal(err)
	}
	zs, err := sys.zeroStateResponseForm()
	if err != nil {
		b.Fatal(err)
	}
	explicit, tau, _ := delayChainModel(zs)
	tg, dt, err := newTimeResponsePlanner(zs).grid(120, 0)
	if err != nil {
		b.Fatal(err)
	}
	shifts := delayShiftCount(zs)
	for b.Loop() {
		if _, ok := newDelayChain(explicit, tau, []int{0}, dt, len(tg), shifts); ok {
			b.Fatal("chain within budget")
		}
	}
}
