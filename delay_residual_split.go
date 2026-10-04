package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// residualDelaySplit realizes a system whose total IO delay has no
// input+output decomposition as a column sum or row stack of copies whose
// delay sub-blocks decompose exactly.
type residualDelaySplit struct {
	byRow  bool
	groups []residualDelayGroup
}

type residualDelayGroup struct {
	members     []int
	inputDelay  []float64
	outputDelay []float64
}

// newResidualDelaySplit groups inputs (or outputs, when that needs fewer
// added states) whose total-delay columns (rows) differ by a constant.
func newResidualDelaySplit(sys *System) residualDelaySplit {
	n, _, _ := sys.Dims()
	total := newDelayTopology(sys).totalExternal(true)
	cols, colCost := residualDelayGroups(total, false, n)
	rows, rowCost := residualDelayGroups(total, true, n)
	if rowCost < colCost {
		return residualDelaySplit{byRow: true, groups: rows}
	}
	return residualDelaySplit{groups: cols}
}

func residualDelayGroups(total *mat.Dense, byRow bool, n int) ([]residualDelayGroup, float64) {
	raw := total.RawMatrix()
	p, m := raw.Rows, raw.Cols
	lines, width := m, p
	at := func(line, k int) float64 { return raw.Data[k*raw.Stride+line] }
	if byRow {
		lines, width = p, m
		at = func(line, k int) float64 { return raw.Data[line*raw.Stride+k] }
	}

	keys := make([]float64, lines*width)
	for l := range lines {
		mn := math.Inf(1)
		for k := range width {
			mn = min(mn, at(l, k))
		}
		for k := range width {
			keys[l*width+k] = at(l, k) - mn
		}
	}

	var groups []residualDelayGroup
	for l := range lines {
		key := keys[l*width : l*width+width]
		placed := false
		for g := range groups {
			ref := groups[g].members[0]
			if delayKeysEqual(key, keys[ref*width:ref*width+width]) {
				groups[g].members = append(groups[g].members, l)
				placed = true
				break
			}
		}
		if !placed {
			groups = append(groups, residualDelayGroup{members: []int{l}})
		}
	}

	cost := 0.0
	for g := range groups {
		in, out, _ := DecomposeIODelay(splitDelayBlock(total, byRow, groups[g].members))
		groups[g].inputDelay, groups[g].outputDelay = in, out
		cost += float64(n)
		for _, v := range in {
			cost += v
		}
		for _, v := range out {
			cost += v
		}
	}
	return groups, cost
}

func delayKeysEqual(a, b []float64) bool {
	for k := range a {
		if math.Abs(a[k]-b[k]) > delayTopologyTol {
			return false
		}
	}
	return true
}

func splitDelayBlock(total *mat.Dense, byRow bool, members []int) *mat.Dense {
	p, m := total.Dims()
	if byRow {
		out := mat.NewDense(len(members), m, nil)
		for k, i := range members {
			out.SetRow(k, total.RawRowView(i))
		}
		return out
	}
	out := mat.NewDense(p, len(members), nil)
	for k, j := range members {
		for i := range p {
			out.Set(i, k, total.At(i, j))
		}
	}
	return out
}

// apply realizes each group, carrying only input and output delays, with
// realize, which must leave no external delay, and reassembles the pieces.
func (s residualDelaySplit) apply(sys *System, realize func(*System) (*System, error)) (*System, error) {
	_, m, p := sys.Dims()
	parts := make([]pathPart, len(s.groups))
	for g, group := range s.groups {
		var piece *System
		if s.byRow {
			piece = pathRows(sys, group.members)
		} else {
			piece = pathCols(sys, group.members)
		}
		piece.InputDelay, piece.OutputDelay = group.inputDelay, group.outputDelay
		realized, err := realize(piece)
		if err != nil {
			return nil, err
		}
		if delayMatrixHasNonzero(realized.Delay) || delaySliceHasNonzero(realized.InputDelay) || delaySliceHasNonzero(realized.OutputDelay) {
			return nil, fmt.Errorf("residual delay split: piece kept external delay: %w", ErrFeedbackDelay)
		}
		parts[g] = pathPart{sys: realized, lines: group.members}
	}
	out := stackPathParts(parts, s.byRow, m, p, sys.Dt)
	propagateIONames(out, sys)
	return out, nil
}

func absorbDecomposedDelay(sys *System) (*System, error) {
	cur, err := absorbInputDelay(sys)
	if err != nil {
		return nil, err
	}
	return absorbOutputDelay(cur)
}
