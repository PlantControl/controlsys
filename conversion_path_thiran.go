package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// conversionPathThiran realizes a nondecomposable fractional path-delay matrix
// on a discretized model. Each row (or column, whichever needs fewer copies)
// carrying a fractional delay gets its own copy of the model with a Thiran
// bank on the opposite side; rows without one share a single copy. Integer
// remainders become the discrete path-delay matrix.
func conversionPathThiran(disc *System, delay *mat.Dense, dt float64, opts C2DOptions) (*System, error) {
	_, m, p := disc.Dims()
	fractionalRow, fractionalCol := make([]bool, p), make([]bool, m)
	rows, cols := 0, 0
	for i := range p {
		for j := range m {
			if !isIntegerSampleDelay(delay.At(i, j) / dt) {
				if !fractionalRow[i] {
					fractionalRow[i] = true
					rows++
				}
				if !fractionalCol[j] {
					fractionalCol[j] = true
					cols++
				}
			}
		}
	}
	byRow := rows <= cols
	lines, fractional := p, fractionalRow
	if !byRow {
		lines, fractional = m, fractionalCol
	}
	whole := mat.NewDense(p, m, nil)
	var plain []int
	var parts []pathPart
	for k := range lines {
		if !fractional[k] {
			plain = append(plain, k)
			for o := range otherCount(byRow, m, p) {
				i, j := pathIndex(byRow, k, o)
				whole.Set(i, j, math.Round(delay.At(i, j)/dt))
			}
			continue
		}
		delays := make([]float64, otherCount(byRow, m, p))
		for o := range delays {
			i, j := pathIndex(byRow, k, o)
			delays[o] = delay.At(i, j)
		}
		bank, err := conversionThiranBank(delays, dt, opts.ThiranOrder, opts.DelayModeling)
		if err != nil {
			return nil, err
		}
		for o, samples := range bank.InputDelay {
			i, j := pathIndex(byRow, k, o)
			whole.Set(i, j, samples)
		}
		bank.InputDelay = nil
		line := []int{k}
		part := pathPart{lines: line}
		if byRow {
			part.sys = conversionSeries(bank, pathRows(disc, line))
		} else {
			part.sys = conversionSeries(pathCols(disc, line), bank)
		}
		parts = append(parts, part)
	}
	if len(plain) > 0 {
		part := pathPart{lines: plain}
		if byRow {
			part.sys = pathRows(disc, plain)
		} else {
			part.sys = pathCols(disc, plain)
		}
		parts = append(parts, part)
	}
	out := stackPathParts(parts, byRow, m, p, disc.Dt)
	if delayMatrixHasNonzero(whole) {
		out.Delay = whole
	}
	out.InputDelay, out.OutputDelay = disc.InputDelay, disc.OutputDelay
	metadataFromSystem(disc).applyIO(out)
	out.Notes = disc.Notes
	return out, nil
}

type pathPart struct {
	sys   *System
	lines []int
}

func otherCount(byRow bool, m, p int) int {
	if byRow {
		return m
	}
	return p
}

func pathIndex(byRow bool, line, other int) (int, int) {
	if byRow {
		return line, other
	}
	return other, line
}

func pathRows(sys *System, rows []int) *System {
	n, m, _ := sys.Dims()
	out := &System{A: sys.A, B: sys.B, C: newDense(len(rows), n), D: newDense(len(rows), m), Dt: sys.Dt}
	for r, i := range rows {
		for k := range n {
			out.C.Set(r, k, sys.C.At(i, k))
		}
		for j := range m {
			out.D.Set(r, j, sys.D.At(i, j))
		}
	}
	if q := sys.internalDelayCount(); q > 0 {
		out.LFT = &LFTDelay{Tau: sys.LFT.Tau, B2: sys.LFT.B2, C2: sys.LFT.C2, D12: newDense(len(rows), q), D21: sys.LFT.D21, D22: sys.LFT.D22}
		for r, i := range rows {
			for k := range q {
				out.LFT.D12.Set(r, k, sys.LFT.D12.At(i, k))
			}
		}
	}
	return out
}

func pathCols(sys *System, cols []int) *System {
	n, _, p := sys.Dims()
	out := &System{A: sys.A, B: newDense(n, len(cols)), C: sys.C, D: newDense(p, len(cols)), Dt: sys.Dt}
	for c, j := range cols {
		for k := range n {
			out.B.Set(k, c, sys.B.At(k, j))
		}
		for i := range p {
			out.D.Set(i, c, sys.D.At(i, j))
		}
	}
	if q := sys.internalDelayCount(); q > 0 {
		out.LFT = &LFTDelay{Tau: sys.LFT.Tau, B2: sys.LFT.B2, C2: sys.LFT.C2, D12: sys.LFT.D12, D21: newDense(q, len(cols)), D22: sys.LFT.D22}
		for c, j := range cols {
			for k := range q {
				out.LFT.D21.Set(k, c, sys.LFT.D21.At(k, j))
			}
		}
	}
	return out
}

// stackPathParts combines parts that share inputs (byRow) or outputs into one
// system with block-diagonal dynamics, routing each part to its lines.
func stackPathParts(parts []pathPart, byRow bool, m, p int, dt float64) *System {
	n, q := 0, 0
	for _, part := range parts {
		pn, _, _ := part.sys.Dims()
		n += pn
		q += part.sys.internalDelayCount()
	}
	out := &System{A: newDense(n, n), B: newDense(n, m), C: newDense(p, n), D: newDense(p, m), Dt: dt}
	var lft *LFTDelay
	if q > 0 {
		lft = &LFTDelay{Tau: make([]float64, 0, q), B2: newDense(n, q), C2: newDense(q, n), D12: newDense(p, q), D21: newDense(q, m), D22: newDense(q, q)}
	}
	ns, qs := 0, 0
	for _, part := range parts {
		sys := part.sys
		pn, _, _ := sys.Dims()
		pq := sys.internalDelayCount()
		setBlock(out.A, ns, ns, sys.A)
		if byRow {
			setBlock(out.B, ns, 0, sys.B)
			placeRows(out.C, part.lines, ns, sys.C)
			placeRows(out.D, part.lines, 0, sys.D)
		} else {
			placeCols(out.B, ns, part.lines, sys.B)
			setBlock(out.C, 0, ns, sys.C)
			placeCols(out.D, 0, part.lines, sys.D)
		}
		if pq > 0 {
			lft.Tau = append(lft.Tau, sys.LFT.Tau...)
			setBlock(lft.B2, ns, qs, sys.LFT.B2)
			setBlock(lft.C2, qs, ns, sys.LFT.C2)
			setBlock(lft.D22, qs, qs, sys.LFT.D22)
			if byRow {
				placeRows(lft.D12, part.lines, qs, sys.LFT.D12)
				setBlock(lft.D21, qs, 0, sys.LFT.D21)
			} else {
				setBlock(lft.D12, 0, qs, sys.LFT.D12)
				placeCols(lft.D21, qs, part.lines, sys.LFT.D21)
			}
		}
		ns += pn
		qs += pq
	}
	out.LFT = lft
	return out
}

func placeRows(dst *mat.Dense, rows []int, c0 int, src *mat.Dense) {
	r, c := src.Dims()
	if r == 0 || c == 0 {
		return
	}
	for k, i := range rows {
		for j := range c {
			dst.Set(i, c0+j, src.At(k, j))
		}
	}
}

func placeCols(dst *mat.Dense, r0 int, cols []int, src *mat.Dense) {
	r, c := src.Dims()
	if r == 0 || c == 0 {
		return
	}
	for k, j := range cols {
		for i := range r {
			dst.Set(r0+i, j, src.At(i, k))
		}
	}
}
