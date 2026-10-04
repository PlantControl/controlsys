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
	pieces := make([]*System, len(s.groups))
	for g, group := range s.groups {
		piece, err := realize(s.piece(sys, group))
		if err != nil {
			return nil, err
		}
		pieces[g] = piece
	}
	res, err := s.assemble(pieces, m, p, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(res, sys)
	return res, nil
}

func (s residualDelaySplit) piece(sys *System, group residualDelayGroup) *System {
	n, m, p := sys.Dims()
	members := group.members
	piece := &System{
		A:           denseCopy(sys.A),
		InputDelay:  group.inputDelay,
		OutputDelay: group.outputDelay,
		Dt:          sys.Dt,
	}
	N := sys.internalDelayCount()
	var lft *LFTDelay
	if N > 0 {
		lft = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D22: copyDelayOrNil(sys.LFT.D22),
		}
	}
	if s.byRow {
		piece.B = denseCopy(sys.B)
		piece.C = selectDenseRows(sys.C, members, n)
		piece.D = selectDenseRows(sys.D, members, m)
		if lft != nil {
			lft.D12 = selectDenseRows(sys.LFT.D12, members, N)
			lft.D21 = copyDelayOrNil(sys.LFT.D21)
		}
	} else {
		piece.B = selectDenseCols(sys.B, members, n)
		piece.C = denseCopy(sys.C)
		piece.D = selectDenseCols(sys.D, members, p)
		if lft != nil {
			lft.D12 = copyDelayOrNil(sys.LFT.D12)
			lft.D21 = selectDenseCols(sys.LFT.D21, members, N)
		}
	}
	piece.LFT = lft
	return piece
}

func selectDenseRows(src *mat.Dense, rows []int, cols int) *mat.Dense {
	if len(rows) == 0 || cols == 0 {
		return &mat.Dense{}
	}
	out := mat.NewDense(len(rows), cols, nil)
	for k, i := range rows {
		out.SetRow(k, src.RawRowView(i)[:cols])
	}
	return out
}

func selectDenseCols(src *mat.Dense, cols []int, rows int) *mat.Dense {
	if len(cols) == 0 || rows == 0 {
		return &mat.Dense{}
	}
	out := mat.NewDense(rows, len(cols), nil)
	sRaw := src.RawMatrix()
	oRaw := out.RawMatrix()
	for i := range rows {
		sRow := sRaw.Data[i*sRaw.Stride:]
		oRow := oRaw.Data[i*oRaw.Stride:]
		for k, j := range cols {
			oRow[k] = sRow[j]
		}
	}
	return out
}

func (s residualDelaySplit) assemble(pieces []*System, m, p int, dt float64) (*System, error) {
	nTot, NTot := 0, 0
	for _, pc := range pieces {
		if delayMatrixHasNonzero(pc.Delay) || delaySliceHasNonzero(pc.InputDelay) || delaySliceHasNonzero(pc.OutputDelay) {
			return nil, fmt.Errorf("residual delay split: piece kept external delay: %w", ErrFeedbackDelay)
		}
		n, _, _ := pc.Dims()
		nTot += n
		NTot += pc.internalDelayCount()
	}

	a := mat.NewDense(max(nTot, 1), max(nTot, 1), nil)
	b := mat.NewDense(max(nTot, 1), max(m, 1), nil)
	c := mat.NewDense(max(p, 1), max(nTot, 1), nil)
	d := mat.NewDense(max(p, 1), max(m, 1), nil)
	var lft *LFTDelay
	if NTot > 0 {
		lft = &LFTDelay{
			Tau: make([]float64, 0, NTot),
			B2:  mat.NewDense(max(nTot, 1), NTot, nil),
			C2:  mat.NewDense(NTot, max(nTot, 1), nil),
			D12: mat.NewDense(max(p, 1), NTot, nil),
			D21: mat.NewDense(NTot, max(m, 1), nil),
			D22: mat.NewDense(NTot, NTot, nil),
		}
	}

	all := func(k int) []int {
		idx := make([]int, k)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	allRows, allCols := all(p), all(m)

	x0, w0 := 0, 0
	for g, pc := range pieces {
		rows, cols := allRows, s.groups[g].members
		if s.byRow {
			rows, cols = s.groups[g].members, allCols
		}
		n, _, _ := pc.Dims()
		N := pc.internalDelayCount()
		if n > 0 {
			setBlock(a, x0, x0, pc.A)
			scatterBlock(b, pc.B, nil, cols, x0, 0)
			scatterBlock(c, pc.C, rows, nil, 0, x0)
		}
		scatterBlock(d, pc.D, rows, cols, 0, 0)
		if N > 0 {
			lft.Tau = append(lft.Tau, pc.LFT.Tau...)
			if n > 0 {
				setBlock(lft.B2, x0, w0, pc.LFT.B2)
				setBlock(lft.C2, w0, x0, pc.LFT.C2)
			}
			scatterBlock(lft.D12, pc.LFT.D12, rows, nil, 0, w0)
			scatterBlock(lft.D21, pc.LFT.D21, nil, cols, w0, 0)
			setBlock(lft.D22, w0, w0, pc.LFT.D22)
		}
		x0 += n
		w0 += N
	}

	var res *System
	var err error
	if nTot == 0 {
		res, err = NewGain(resizeDense(d, p, m), dt)
	} else {
		res, err = newNoCopy(a, resizeDense(b, nTot, m), resizeDense(c, p, nTot), resizeDense(d, p, m), dt)
	}
	if err != nil {
		return nil, err
	}
	if lft != nil {
		if nTot == 0 {
			lft.B2, lft.C2 = &mat.Dense{}, &mat.Dense{}
		} else {
			lft.B2 = resizeDense(lft.B2, nTot, NTot)
			lft.C2 = resizeDense(lft.C2, NTot, nTot)
		}
		lft.D12 = resizeDense(lft.D12, p, NTot)
		lft.D21 = resizeDense(lft.D21, NTot, m)
		res.LFT = lft
	}
	return res, nil
}

// scatterBlock writes src into dst with src row k at rows[k]+r0 (or r0+k
// when rows is nil) and src column k at cols[k]+c0 (or c0+k).
func scatterBlock(dst, src *mat.Dense, rows, cols []int, r0, c0 int) {
	if src == nil || src.IsEmpty() {
		return
	}
	sr, sc := src.Dims()
	dRaw := dst.RawMatrix()
	sRaw := src.RawMatrix()
	for i := range sr {
		di := r0 + i
		if rows != nil {
			di = r0 + rows[i]
		}
		dRow := dRaw.Data[di*dRaw.Stride:]
		sRow := sRaw.Data[i*sRaw.Stride : i*sRaw.Stride+sc]
		for j, v := range sRow {
			dj := c0 + j
			if cols != nil {
				dj = c0 + cols[j]
			}
			dRow[dj] = v
		}
	}
}

func absorbDecomposedDelay(sys *System) (*System, error) {
	cur, err := absorbInputDelay(sys)
	if err != nil {
		return nil, err
	}
	return absorbOutputDelay(cur)
}
