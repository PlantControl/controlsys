package controlsys

import "plantcontrol.org/v1/gonum/mat"

// stiffLFTExplicit returns, for a continuous descriptor sys with internal
// delays, the explicit model with the same delay feedback whose descriptor
// part G = (A, E, [B B2], [C; C2], [D D12; D21 D22]) is replaced by the
// parallel connection of the foldInaccurateParts of G, each in real Schur
// form: MATLAB keeps H = lft(G, delay) and G its pencil, so the poles are
// those of (A, E) rather than of the LU fold E⁻¹A. The I/O delays and names
// of sys carry over; state names are dropped. ok is false when sys has no
// internal delays or G does not need the split.
func stiffLFTExplicit(sys *System) (*System, bool) {
	out, _, ok := stiffLFTExplicitState(sys, nil)
	return out, ok
}

// stiffLFTExplicitState is stiffLFTExplicit that also maps a non-nil x0 to
// the coordinates of the returned model: each part takes its share of x0
// from the input columns E·x0 and A·x0 appended to G (freePartShare), so
// the free response of the returned model from xt is that of sys from x0.
func stiffLFTExplicitState(sys *System, x0 *mat.VecDense) (out *System, xt *mat.VecDense, ok bool) {
	if !sys.IsContinuous() || !sys.HasInternalDelay() || !sys.IsDescriptor() {
		return nil, nil, false
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if lu.Cond() < foldAccurateECond {
		return nil, nil, false
	}
	n, m, p := sys.Dims()
	g := conversionAugmentedRational(sys)
	g.E = sys.E
	cols := m + len(sys.LFT.Tau)
	if x0 != nil {
		rows, _ := g.D.Dims()
		b, d := newDense(n, cols+2), newDense(rows, cols+2)
		setBlock(b, 0, 0, g.B)
		setBlock(d, 0, 0, g.D)
		var ex, ax mat.VecDense
		ex.MulVec(sys.E, x0)
		ax.MulVec(sys.A, x0)
		b.SetCol(cols, ex.RawVector().Data)
		b.SetCol(cols+1, ax.RawVector().Data)
		g.B, g.D = b, d
	}
	parts, ok := stiffDescriptorParts(g)
	if !ok {
		return nil, nil, false
	}
	for i, part := range parts {
		if parts[i], ok = realSchurPart(part); !ok {
			return nil, nil, false
		}
	}
	if x0 != nil {
		xt, parts = splitFreeShareColumns(parts, cols, n)
	}
	explicit, err := parallelExplicit(parts)
	if err != nil {
		return nil, nil, false
	}
	out = splitConversionRational(explicit, m, p, sys.LFT.Tau)
	delays := sys.Copy()
	out.Delay, out.InputDelay, out.OutputDelay = delays.Delay, delays.InputDelay, delays.OutputDelay
	propagateNames(out, sys)
	out.StateName = nil
	return out, xt, true
}

// splitFreeShareColumns returns the shares of x0 (freePartShare) of parts
// whose input columns cols and cols+1 are E·x0 and A·x0, and the parts
// without those two columns.
func splitFreeShareColumns(parts []*System, cols, n int) (*mat.VecDense, []*System) {
	views := make([]*System, len(parts))
	for i, part := range parts {
		v := *part
		if pn, _, _ := part.Dims(); pn > 0 {
			v.B = part.B.Slice(0, pn, cols, cols+2).(*mat.Dense)
		}
		views[i] = &v
	}
	shares := make([]float64, 0, n)
	for i := range views {
		shares = append(shares, freePartShare(views, i)...)
	}
	trimmed := make([]*System, len(parts))
	for i, part := range parts {
		c := *part
		if pn, _, _ := part.Dims(); pn > 0 {
			c.B = part.B.Slice(0, pn, 0, cols).(*mat.Dense)
		}
		rows, _ := part.D.Dims()
		c.D = part.D.Slice(0, rows, 0, cols).(*mat.Dense)
		trimmed[i] = &c
	}
	return mat.NewVecDense(len(shares), shares), trimmed
}
