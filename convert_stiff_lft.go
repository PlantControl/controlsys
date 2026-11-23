package controlsys

import (
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffLFTExplicit returns, for a continuous descriptor sys with internal
// delays, the explicit model with the same delay feedback whose descriptor
// part G = (A, E, [B B2], [C; C2], [D D12; D21 D22]) is replaced by the
// parallel connection of the foldInaccurateParts of G, each in real Schur
// form: MATLAB keeps H = lft(G, delay) and G its pencil, so the poles are
// those of (A, E) rather than of the LU fold E⁻¹A. The I/O delays and names
// of sys carry over; state names are dropped. ok is false when sys has no
// internal delays or G does not need the split.
func stiffLFTExplicit(sys *System) (*System, bool) {
	out, ok, err := stiffLFTExplicitWithin(sys, 1)
	return out, ok && err == nil
}

// stiffLFTExplicitWithin is stiffLFTExplicit returning the error of a
// split of G not certified within limit (stiffSplit.within), for C2D.
func stiffLFTExplicitWithin(sys *System, limit float64) (*System, bool, error) {
	sp, ok, err := newStiffLFTSplitWithin(sys, nil, limit)
	if !ok || err != nil {
		return nil, ok, err
	}
	out, _, ok := sp.explicit()
	return out, ok, nil
}

// stiffLFTSampled is stiffLFTExplicitState for a response sampled at the
// spacing dt (Lsim). ok is false, so the caller samples the LU fold E⁻¹A by
// the exact delay chain, when the fold's error cond(E)·ε is at most the
// error dt·ρ of the ZOH discretization of the parts' delay channels, ρ the
// spectral radius of the slow part: over one sample the held delay signal
// moves by up to dt·ρ relative. Internal delays of different lengths have
// no exact chain, so their fold takes the ZOH discretization too and the
// parts are kept. The gate decides before the parts are assembled.
func stiffLFTSampled(sys *System, x0 *mat.VecDense, dt float64) (*System, *mat.VecDense, bool) {
	out, xt, ok, err := stiffLFTSampledWithin(sys, x0, dt)
	return out, xt, ok && err == nil
}

// stiffLFTSampledWithin is stiffLFTSampled returning the error of a split
// of G that is not certified (Lsim).
func stiffLFTSampledWithin(sys *System, x0 *mat.VecDense, dt float64) (*System, *mat.VecDense, bool, error) {
	sp, ok, err := newStiffLFTSplitWithin(sys, x0, 1)
	if !ok || err != nil {
		return nil, nil, ok, err
	}
	if sp.foldAccurate(dt) {
		return nil, nil, false, nil
	}
	out, xt, ok := sp.explicit()
	return out, xt, ok, nil
}

// stiffLFTSampledAuto is stiffLFTSampled for a response sampled at the
// spacing dtOf(out) of the automatic grid of the returned model out (Step,
// Initial), so the parts are assembled before the gate decides. The fold
// also needs chainFits: its exact delay chain, on the fold's own automatic
// grid, within the chain budget (delayChainWithinBudget). Over budget the
// fold takes the ZOH discretization too, after the rejected chain's cost,
// so the parts are kept. A split of G that is not certified returns its
// error.
func stiffLFTSampledAuto(sys *System, x0 *mat.VecDense, dtOf func(out *System) float64, chainFits func() bool) (*System, *mat.VecDense, bool, error) {
	sp, ok, err := newStiffLFTSplitWithin(sys, x0, 1)
	if !ok || err != nil {
		return nil, nil, false, err
	}
	out, xt, ok := sp.explicit()
	if !ok || sp.foldAccurate(dtOf(out)) && chainFits() {
		return nil, nil, false, nil
	}
	return out, xt, true, nil
}

// delayChainWithinBudget reports whether newDelayChain of the internal-delay
// model sys over steps samples of dt, with k input columns and shifts
// source shifts, can pass its delayChainMaxWork check: it passes the check
// at the first truncation order tried, min(K, 4). A model outside
// delayChainModel reports true, leaving the decision to its caller's route.
func delayChainWithinBudget(sys *System, dt float64, steps, k, shifts int) bool {
	explicit, tau, ok := delayChainModel(sys)
	if !ok {
		return true
	}
	n, _, _ := explicit.Dims()
	blocks := math.Floor(float64(steps-1)*dt/tau+gridTol) + 1
	return delayChainWork(blocks, float64(steps), math.Min(blocks-1, 4), float64(n+k), float64(shifts)) <= delayChainMaxWork
}

// foldAccurate reports whether the exact delay chain of the LU fold E⁻¹A,
// with error cond(E)·ε, is at least as accurate on the spacing dt as the
// ZOH discretization of the parts, with error dt·ρ (see stiffLFTSampled).
func (sp *stiffLFTSplit) foldAccurate(dt float64) bool {
	_, _, chain := delayChainModel(sp.sys)
	return chain && sp.cond*0x1p-52 <= dt*schurSpectralRadius(sp.parts[0].A)
}

// stiffLFTExplicitState is stiffLFTExplicit that also maps a non-nil x0 to
// the coordinates of the returned model: each part takes its share of x0
// from the input columns E·x0 and A·x0 appended to G (freePartShare), so
// the free response of the returned model from xt is that of sys from x0.
func stiffLFTExplicitState(sys *System, x0 *mat.VecDense) (*System, *mat.VecDense, bool) {
	sp, ok, err := newStiffLFTSplitWithin(sys, x0, 1)
	if !ok || err != nil {
		return nil, nil, false
	}
	return sp.explicit()
}

// stiffLFTSplit is the generalized Schur split of the descriptor part G of
// stiffLFTExplicitState, with the LU condition estimate of E; the slow
// (first) part is in real Schur form.
type stiffLFTSplit struct {
	sys   *System
	x0    *mat.VecDense
	parts []*System
	cols  int
	cond  float64
}

// newStiffLFTSplitWithin splits G; ok is false when the split does not apply,
// and err reports a split not certified within limit (stiffSplit.within),
// certified with G's own B and C.
func newStiffLFTSplitWithin(sys *System, x0 *mat.VecDense, limit float64) (*stiffLFTSplit, bool, error) {
	if !sys.IsContinuous() || !sys.HasInternalDelay() || !sys.IsDescriptor() {
		return nil, false, nil
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	cond := lu.Cond()
	if cond < foldAccurateECond {
		return nil, false, nil
	}
	n, m, _ := sys.Dims()
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
	split, ok := stiffDescriptorParts(g)
	if !ok {
		return nil, false, nil
	}
	if err := split.within(limit); err != nil {
		return nil, true, err
	}
	parts := split.parts
	if parts[0], ok = realSchurPart(parts[0]); !ok {
		return nil, false, nil
	}
	return &stiffLFTSplit{sys: sys, x0: x0, parts: parts, cols: cols, cond: cond}, true, nil
}

// explicit assembles the parts, each in real Schur form, into the model of
// stiffLFTExplicitState.
func (sp *stiffLFTSplit) explicit() (*System, *mat.VecDense, bool) {
	sys, parts := sp.sys, sp.parts
	for i := 1; i < len(parts); i++ {
		var ok bool
		if parts[i], ok = realSchurPart(parts[i]); !ok {
			return nil, nil, false
		}
	}
	n, m, p := sys.Dims()
	var xt *mat.VecDense
	if sp.x0 != nil {
		xt, parts = splitFreeShareColumns(parts, sp.cols, n)
	}
	explicit, err := parallelExplicit(parts)
	if err != nil {
		return nil, nil, false
	}
	out := splitConversionRational(explicit, m, p, sys.LFT.Tau)
	delays := sys.Copy()
	out.Delay, out.InputDelay, out.OutputDelay = delays.Delay, delays.InputDelay, delays.OutputDelay
	propagateNames(out, sys)
	out.StateName = nil
	return out, xt, true
}

// schurSpectralRadius returns the largest eigenvalue modulus of the upper
// quasi-triangular a.
func schurSpectralRadius(a *mat.Dense) float64 {
	if a == nil || a.IsEmpty() {
		return 0
	}
	n, _ := a.Dims()
	raw := a.RawMatrix()
	if raw.Stride != n {
		raw = mat.DenseCopyOf(a).RawMatrix()
	}
	rho := 0.0
	for _, l := range schurEigenvaluesRaw(raw.Data, n) {
		rho = math.Max(rho, cmplx.Abs(l))
	}
	return rho
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
