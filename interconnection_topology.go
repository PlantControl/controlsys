package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type interconnectionTopology struct {
	plan interconnectionPlan
}

type interconnectionDelayPlan struct {
	inputDelay  []float64
	outputDelay []float64
	ioDelay     *mat.Dense
	requiresLFT bool
}

func newInterconnectionTopology(sys1, sys2 *System) interconnectionTopology {
	return interconnectionTopology{plan: newInterconnectionPlan(sys1, sys2)}
}

func (t interconnectionTopology) seriesRequiresLFT() bool {
	return t.seriesDelayPlan().requiresLFT
}

func (t interconnectionTopology) seriesDelayPlan() interconnectionDelayPlan {
	plan := t.plan
	if plan.sys1.HasInternalDelay() || plan.sys2.HasInternalDelay() {
		return interconnectionDelayPlan{requiresLFT: true}
	}
	if !plan.sys1.HasDelay() && !plan.sys2.HasDelay() {
		return interconnectionDelayPlan{}
	}
	ioCheck := seriesDelay(plan.sys1, plan.sys2)
	if ioCheck == nil && (plan.sys1.Delay != nil || plan.sys2.Delay != nil) {
		return interconnectionDelayPlan{requiresLFT: true}
	}
	if plan.sys1.OutputDelay == nil && plan.sys2.InputDelay == nil {
		inDel, outDel, ioDelay := seriesInputOutputDelay(plan.sys1, plan.sys2, plan.p2, plan.m1, ioCheck)
		return interconnectionDelayPlan{inputDelay: inDel, outputDelay: outDel, ioDelay: ioDelay}
	}
	first := t.seriesIntermediateDelay(0)
	for k := 1; k < plan.p1; k++ {
		d := t.seriesIntermediateDelay(k)
		if !delaysEqual(d, first, max(math.Abs(d), math.Abs(first))) {
			return interconnectionDelayPlan{requiresLFT: true}
		}
	}
	inDel, outDel, ioDelay := seriesInputOutputDelay(plan.sys1, plan.sys2, plan.p2, plan.m1, ioCheck)
	return interconnectionDelayPlan{inputDelay: inDel, outputDelay: outDel, ioDelay: ioDelay}
}

func (t interconnectionTopology) seriesIntermediateDelay(k int) float64 {
	plan := t.plan
	var delay float64
	if plan.sys1.OutputDelay != nil {
		delay += plan.sys1.OutputDelay[k]
	}
	if plan.sys2.InputDelay != nil {
		delay += plan.sys2.InputDelay[k]
	}
	return delay
}

func (t interconnectionTopology) parallelRequiresLFT() bool {
	return t.parallelDelayPlan().requiresLFT
}

func (t interconnectionTopology) parallelDelayPlan() interconnectionDelayPlan {
	plan := t.plan
	if plan.sys1.HasInternalDelay() || plan.sys2.HasInternalDelay() {
		return interconnectionDelayPlan{requiresLFT: true}
	}
	if !plan.sys1.HasDelay() && !plan.sys2.HasDelay() {
		return interconnectionDelayPlan{}
	}
	td1 := t.totalDelayOrZero(plan.sys1)
	td2 := t.totalDelayOrZero(plan.sys2)
	if td1 == nil && td2 == nil {
		inDel, outDel, ioDelay := parallelInputOutputDelay(plan.sys1, plan.sys2, plan.m1, plan.p1, nil)
		return interconnectionDelayPlan{inputDelay: inDel, outputDelay: outDel, ioDelay: ioDelay}
	}
	if td1 == nil {
		td1 = mat.NewDense(plan.p1, plan.m1, nil)
	}
	if td2 == nil {
		td2 = mat.NewDense(plan.p1, plan.m1, nil)
	}
	if !mat.Equal(td1, td2) {
		return interconnectionDelayPlan{requiresLFT: true}
	}
	ioDelay := parallelDelay(plan.sys1, plan.sys2)
	inDel, outDel, ioDelay := parallelInputOutputDelay(plan.sys1, plan.sys2, plan.m1, plan.p1, ioDelay)
	return interconnectionDelayPlan{inputDelay: inDel, outputDelay: outDel, ioDelay: ioDelay}
}

func (t interconnectionTopology) totalDelayOrZero(sys *System) *mat.Dense {
	return newDelayTopology(sys).totalExternal(true)
}
