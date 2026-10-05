package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// LFTFeedback gives the feedback channel counts of MATLAB
// lft(sys1,sys2,nu,ny).
type LFTFeedback struct {
	// Nu counts the first outputs of sys2 that drive the last inputs of sys1.
	Nu int
	// Ny counts the last outputs of sys1 that drive the first inputs of sys2.
	Ny int
}

// LFT forms the Redheffer star product of sys1 and sys2, as MATLAB
// lft(sys1,sys2,nu,ny) with fb = LFTFeedback{Nu: nu, Ny: ny}: the first Nu
// outputs of sys2 drive the last Nu inputs of sys1, and the last Ny outputs
// of sys1 drive the first Ny inputs of sys2. The result maps [w1; w2] to
// [z1; z2], where w1, z1 are the remaining inputs and outputs of sys1 and
// w2, z2 those of sys2; its states and StateName are sys1's followed by
// sys2's.
//
// Without fb, LFT follows MATLAB lft(sys1,sys2): when sys2 has fewer inputs
// and fewer outputs than sys1 it is the lower LFT (Nu = sys2 outputs, Ny =
// sys2 inputs); when sys1 has fewer inputs and fewer outputs than sys2 it is
// the upper LFT (Nu = sys1 inputs, Ny = sys1 outputs); otherwise
// ErrDimensionMismatch.
//
// Delays are kept through an internal-delay realization. A nil model, a
// negative count or more than one fb returns ErrInvalidArgument, counts that
// do not fit either model ErrDimensionMismatch, mixed domains
// ErrDomainMismatch and an ill-posed loop ErrAlgebraicLoop.
// See https://www.mathworks.com/help/control/ref/inputoutputmodel.lft.html.
func LFT(sys1, sys2 *System, fb ...LFTFeedback) (*System, error) {
	if err := requireSystems("LFT", sys1, sys2); err != nil {
		return nil, err
	}
	n1, m1, p1 := sys1.Dims()
	n2, m2, p2 := sys2.Dims()
	var nu, ny int
	switch {
	case len(fb) > 1:
		return nil, fmt.Errorf("LFT: %d feedback specs, want at most 1: %w", len(fb), ErrInvalidArgument)
	case len(fb) == 1:
		nu, ny = fb[0].Nu, fb[0].Ny
	case m2 < m1 && p2 < p1:
		nu, ny = p2, m2
	case m1 < m2 && p1 < p2:
		nu, ny = m1, p1
	default:
		return nil, fmt.Errorf("LFT: sys1 is %dx%d and sys2 is %dx%d, want one strictly smaller in both inputs and outputs: %w", p1, m1, p2, m2, ErrDimensionMismatch)
	}
	if nu < 0 || ny < 0 {
		return nil, fmt.Errorf("LFT: nu=%d, ny=%d, want non-negative: %w", nu, ny, ErrInvalidArgument)
	}
	if nu > m1 || nu > p2 {
		return nil, fmt.Errorf("LFT: nu=%d exceeds sys1 inputs %d or sys2 outputs %d: %w", nu, m1, p2, ErrDimensionMismatch)
	}
	if ny > p1 || ny > m2 {
		return nil, fmt.Errorf("LFT: ny=%d exceeds sys1 outputs %d or sys2 inputs %d: %w", ny, p1, m2, ErrDimensionMismatch)
	}
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, fmt.Errorf("LFT: %w", err)
	}

	var res *System
	var err error
	if m2 == ny && p2 == nu {
		res, err = lftCloseExternal(sys1, sys2, m1-nu, p1-ny)
	} else {
		res, err = lftStar(sys1, sys2, nu, ny)
	}
	if err != nil {
		return nil, fmt.Errorf("LFT: %w", err)
	}
	if n, _, _ := res.Dims(); n == n1+n2 {
		res.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	}
	return res, nil
}

// lftStar closes the star product as a lower LFT of append(sys1, sys2),
// reordered to [w1; w2 | u; v] inputs and [z1; z2 | y; u'] outputs, with the
// static swap u = u', v = y.
func lftStar(sys1, sys2 *System, nu, ny int) (*System, error) {
	_, m1, p1 := sys1.Dims()
	_, m2, p2 := sys2.Dims()
	app, err := Append(sys1, sys2)
	if err != nil {
		return nil, err
	}
	inputs := make([]int, 0, m1+m2)
	inputs = append(inputs, rangeInts(m1-nu)...)
	inputs = append(inputs, offsetInts(m1+ny, m2-ny)...)
	inputs = append(inputs, offsetInts(m1-nu, nu)...)
	inputs = append(inputs, offsetInts(m1, ny)...)
	outputs := make([]int, 0, p1+p2)
	outputs = append(outputs, rangeInts(p1-ny)...)
	outputs = append(outputs, offsetInts(p1+nu, p2-nu)...)
	outputs = append(outputs, offsetInts(p1-ny, ny)...)
	outputs = append(outputs, offsetInts(p1, nu)...)
	k := nu + ny
	if k == 0 {
		return app, nil
	}
	M, err := app.SelectByIndex(inputs, outputs)
	if err != nil {
		return nil, err
	}
	swap := mat.NewDense(k, k, nil)
	for i := range nu {
		swap.Set(i, ny+i, 1)
	}
	for i := range ny {
		swap.Set(nu+i, i, 1)
	}
	Delta, err := NewGain(swap, sys1.Dt)
	if err != nil {
		return nil, err
	}
	return lftCloseExternal(M, Delta, m1+m2-k, p1+p2-k)
}

func offsetInts(start, n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = start + i
	}
	return s
}

// lftCloseExternal closes the lower loop of M with Delta, keeping M's first
// nw inputs and first nz outputs external. nw and nz count the EXTERNAL
// channels (the complement of MATLAB lft's nu, ny); a nil Delta returns the
// external block. Callers must have checked the partition sizes and domains.
func lftCloseExternal(M, Delta *System, nw, nz int) (*System, error) {
	_, mM, pM := M.Dims()
	if Delta == nil {
		return lftExtract(M, nw, nz)
	}

	needsLFT := M.HasDelay() || M.HasInternalDelay() || Delta.HasDelay() || Delta.HasInternalDelay()
	if needsLFT {
		return lftWithDelay(M, Delta, nw, nz)
	}

	if mM == nw && pM == nz {
		return lftEmptyLoop(M, Delta, nw, nz)
	}
	return lftSimple(M, Delta, nw, nz)
}

// lftEmptyLoop closes an LFT whose loop carries no signals: the result is M's
// upper channels with Delta's states appended, unreachable and unobservable.
func lftEmptyLoop(M, Delta *System, nw, nz int) (*System, error) {
	upper, err := lftExtract(M, nw, nz)
	if err != nil {
		return nil, err
	}
	if nD, _, _ := Delta.Dims(); nD == 0 {
		return upper, nil
	}
	return Append(upper, Delta)
}

func lftExtract(M *System, nw, nz int) (*System, error) {
	nM, _, _ := M.Dims()
	D11 := extractBlock(M.D, 0, 0, nz, nw)

	if nM == 0 {
		result, err := lftGain(D11, nz, nw, M.Dt)
		if err != nil {
			return nil, err
		}
		lftVisibleMetadata(M, nw, nz).applyIOOwned(result)
		return result, nil
	}

	B1 := extractBlock(M.B, 0, 0, nM, nw)
	C1 := extractBlock(M.C, 0, 0, nz, nM)
	result, err := newNoCopy(denseCopy(M.A), B1, C1, D11, M.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(M.E)
	lftVisibleMetadata(M, nw, nz).applyIOOwned(result)
	return result, nil
}

func lftSimple(M, Delta *System, nw, nz int) (*System, error) {
	nM, mM, pM := M.Dims()
	nD, _, _ := Delta.Dims()

	w := pM - nz
	z := mM - nw

	D11 := extractBlock(M.D, 0, 0, nz, nw)
	D12 := extractBlock(M.D, 0, nw, nz, z)
	D21 := extractBlock(M.D, nz, 0, w, nw)
	D22 := extractBlock(M.D, nz, nw, w, z)

	B1 := extractBlock(M.B, 0, 0, nM, nw)
	B2 := extractBlock(M.B, 0, nw, nM, z)
	C1 := extractBlock(M.C, 0, 0, nz, nM)
	C2 := extractBlock(M.C, nz, 0, w, nM)

	F, err := solveLFTLoop(D22, Delta.D, w)
	if err != nil {
		return nil, err
	}

	Phi := mulDense(Delta.D, F)
	G := lftLoopGain(Phi, D22, z)

	PhiC2 := mulDense(Phi, C2)
	PhiD21 := mulDense(Phi, D21)

	n := nM + nD

	Dcl := addMulDims(nz, nw, D11, D12, PhiD21)
	if n == 0 {
		result, err := lftGain(Dcl, nz, nw, M.Dt)
		if err != nil {
			return nil, err
		}
		lftVisibleMetadata(M, nw, nz).applyIOOwned(result)
		return result, nil
	}

	Acl := mat.NewDense(n, n, nil)
	Bcl := newDense(n, nw)
	Ccl := newDense(nz, n)

	var FC2, FD21, FD22Cd, GCd *mat.Dense
	if nD > 0 {
		FC2 = mulDense(F, C2)
		FD21 = mulDense(F, D21)
		FD22Cd = mulDims(w, nD, F, mulDense(D22, Delta.C))
		GCd = mulDense(G, Delta.C)
	}

	if nM > 0 {
		setBlock(Acl, 0, 0, M.A)
		addBlock(Acl, 0, 0, mulDense(B2, PhiC2))
		setBlock(Bcl, 0, 0, B1)
		addBlock(Bcl, 0, 0, mulDense(B2, PhiD21))
		setBlock(Ccl, 0, 0, C1)
		addBlock(Ccl, 0, 0, mulDense(D12, PhiC2))
	}
	if nD > 0 {
		setBlock(Acl, nM, nM, Delta.A)
		addBlock(Acl, nM, nM, mulDense(Delta.B, FD22Cd))
		setBlock(Bcl, nM, 0, mulDense(Delta.B, FD21))
	}
	if nM > 0 && nD > 0 {
		setBlock(Acl, 0, nM, mulDense(B2, GCd))
		setBlock(Acl, nM, 0, mulDense(Delta.B, FC2))
	}
	if nD > 0 {
		setBlock(Ccl, 0, nM, mulDense(D12, GCd))
	}

	result, err := newNoCopy(Acl, Bcl, Ccl, Dcl, M.Dt)
	if err != nil {
		return nil, err
	}
	result.E = blkDiagDescriptorE(M, Delta)
	lftVisibleMetadata(M, nw, nz).applyIOOwned(result)
	return result, nil
}

// lftGain builds a static LFT result. A gain with no inputs or no outputs but
// not both carries no dimensions, so it is rejected as in NewFromSlices.
func lftGain(D *mat.Dense, nz, nw int, dt float64) (*System, error) {
	if err := lftGainDims(nz, nw); err != nil {
		return nil, err
	}
	return buildSystem(nil, nil, nil, D, dt, nil)
}

func lftGainDims(nz, nw int) error {
	if (nz == 0) != (nw == 0) {
		return fmt.Errorf("%dx%d static gain cannot be stored: %w", nz, nw, ErrDimensionMismatch)
	}
	return nil
}

// lftLoopGain returns I + Phi·D22, the z×z gain from Delta's outputs through
// the closed loop.
func lftLoopGain(Phi, D22 *mat.Dense, z int) *mat.Dense {
	G := mulDims(z, z, Phi, D22)
	for i := range z {
		G.Set(i, i, G.At(i, i)+1)
	}
	return G
}

func solveLFTLoop(D22M, DDelta *mat.Dense, w int) (*mat.Dense, error) {
	return solveIdentityMinusProduct(D22M, DDelta, w, "lft", ErrAlgebraicLoop)
}

func lftWithDelay(M, Delta *System, nw, nz int) (*System, error) {
	_, mM, pM := M.Dims()

	savedInputDelay := selectLeadingVisibleDelays(M.InputDelay, nw)
	savedOutputDelay := selectLeadingVisibleDelays(M.OutputDelay, nz)

	mCopy := M.Copy()
	mCopy.InputDelay = clearLeadingDelays(mCopy.InputDelay, nw)
	mCopy.OutputDelay = clearLeadingDelays(mCopy.OutputDelay, nz)

	mLFT, err := mCopy.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	dLFT, err := Delta.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	NM := mLFT.internalDelayCount()
	ND := dLFT.internalDelayCount()
	N := NM + ND

	mH, _, err := mLFT.GetDelayModel()
	if err != nil {
		return nil, err
	}
	dH, _, err := dLFT.GetDelayModel()
	if err != nil {
		return nil, err
	}

	nM, _, _ := mH.Dims()
	nD, _, _ := dH.Dims()

	w := pM - nz
	z := mM - nw
	n := nM + nD
	mTotal := nw + N
	pTotal := nz + N

	D11 := extractBlock(mH.D, 0, 0, nz, nw)
	D12p := extractBlock(mH.D, 0, nw, nz, z)
	D21p := extractBlock(mH.D, nz, 0, w, nw)
	D22p := extractBlock(mH.D, nz, nw, w, z)

	DDe := extractBlock(dH.D, 0, 0, z, w)

	F, err := solveLFTLoop(D22p, DDe, w)
	if err != nil {
		return nil, err
	}

	Phi := mulDense(DDe, F)
	G := lftLoopGain(Phi, D22p, z)

	B1 := extractBlock(mH.B, 0, 0, nM, nw)
	B2p := extractBlock(mH.B, 0, nw, nM, z)
	C1 := extractBlock(mH.C, 0, 0, nz, nM)
	C2p := extractBlock(mH.C, nz, 0, w, nM)

	BDe := extractBlock(dH.B, 0, 0, nD, w)
	CDe := extractBlock(dH.C, 0, 0, z, nD)

	PhiC2p := mulDense(Phi, C2p)
	PhiD21p := mulDense(Phi, D21p)
	var FC2p, FD21p, FD22pCDe, GCDe *mat.Dense
	if nD > 0 || N > 0 {
		FC2p = mulDense(F, C2p)
		FD21p = mulDense(F, D21p)
		FD22pCDe = mulDense(F, mulDense(D22p, CDe))
		GCDe = mulDense(G, CDe)
	}

	Acl := mat.NewDense(max(n, 1), max(n, 1), nil)
	Bcl := mat.NewDense(max(n, 1), max(mTotal, 1), nil)
	Ccl := mat.NewDense(max(pTotal, 1), max(n, 1), nil)
	Dcl := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)

	if nM > 0 {
		setBlock(Acl, 0, 0, mH.A)
		addBlock(Acl, 0, 0, mulDense(B2p, PhiC2p))
	}
	if nD > 0 {
		setBlock(Acl, nM, nM, dH.A)
		addBlock(Acl, nM, nM, mulDense(BDe, FD22pCDe))
	}
	if nM > 0 && nD > 0 {
		setBlock(Acl, 0, nM, mulDense(B2p, GCDe))
		setBlock(Acl, nM, 0, mulDense(BDe, FC2p))
	}

	if nM > 0 {
		setBlock(Bcl, 0, 0, B1)
		addBlock(Bcl, 0, 0, mulDense(B2p, PhiD21p))
	}
	if nD > 0 {
		setBlock(Bcl, nM, 0, mulDense(BDe, FD21p))
	}

	if nM > 0 {
		setBlock(Ccl, 0, 0, C1)
		addBlock(Ccl, 0, 0, mulDense(D12p, PhiC2p))
	}
	if nD > 0 {
		setBlock(Ccl, 0, nM, mulDense(D12p, GCDe))
	}

	setBlock(Dcl, 0, 0, D11)
	addBlock(Dcl, 0, 0, mulDense(D12p, PhiD21p))

	if N == 0 {
		if n == 0 {
			if err := lftGainDims(nz, nw); err != nil {
				return nil, err
			}
			Acl = &mat.Dense{}
		} else {
			Acl = resizeDense(Acl, n, n)
		}
		Bcl = resizeDense(Bcl, n, nw)
		Ccl = resizeDense(Ccl, nz, n)
		Dcl = resizeDense(Dcl, nz, nw)

		sys, err := newNoCopy(Acl, Bcl, Ccl, Dcl, M.Dt)
		if err != nil {
			return nil, err
		}
		sys.E = blkDiagDescriptorE(mH, dH)
		if savedInputDelay.hasNonzero {
			sys.InputDelay = savedInputDelay.values
		}
		if savedOutputDelay.hasNonzero {
			sys.OutputDelay = savedOutputDelay.values
		}
		lftVisibleMetadata(M, nw, nz).applyIOOwned(sys)
		return sys, nil
	}

	var D12iMu, D12iMw *mat.Dense
	var D21iMext, D21iMz *mat.Dense
	var PhiD12iMw, FD12iMw *mat.Dense
	if NM > 0 {
		D12iMu = extractBlock(mH.D, 0, nw+z, nz, NM)
		D12iMw = extractBlock(mH.D, nz, nw+z, w, NM)
		D21iMext = extractBlock(mH.D, nz+w, 0, NM, nw)
		D21iMz = extractBlock(mH.D, nz+w, nw, NM, z)
		PhiD12iMw = mulDense(Phi, D12iMw)
		FD12iMw = mulDense(F, D12iMw)
	}

	var D12iD, D21iD *mat.Dense
	var GD12iD, FD22pD12iD *mat.Dense
	if ND > 0 {
		D12iD = extractBlock(dH.D, 0, w, z, ND)
		D21iD = extractBlock(dH.D, z, 0, ND, w)
		GD12iD = mulDense(G, D12iD)
		FD22pD12iD = mulDense(F, mulDense(D22p, D12iD))
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(nz, 1), N, nil)
	d21 := mat.NewDense(N, max(nw, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if NM > 0 {
		B2iM := extractBlock(mH.B, 0, nw+z, nM, NM)
		C2iM := extractBlock(mH.C, nz+w, 0, NM, nM)
		D22iM := extractBlock(mH.D, nz+w, nw+z, NM, NM)

		if nM > 0 {
			t := addMulDims(nM, NM, B2iM, B2p, PhiD12iMw)
			setBlock(b2, 0, 0, t)
		}
		if nD > 0 {
			setBlock(b2, nM, 0, mulDense(BDe, FD12iMw))
		}

		if nM > 0 {
			t := addMulDims(NM, nM, C2iM, D21iMz, PhiC2p)
			setBlock(c2, 0, 0, t)
		}
		if nD > 0 {
			setBlock(c2, 0, nM, mulDense(D21iMz, GCDe))
		}

		{
			t := addMulDims(nz, NM, D12iMu, D12p, PhiD12iMw)
			setBlock(d12, 0, 0, t)
		}

		{
			t := addMulDims(NM, nw, D21iMext, D21iMz, PhiD21p)
			setBlock(d21, 0, 0, t)
		}

		{
			t := addMulDims(NM, NM, D22iM, D21iMz, PhiD12iMw)
			setBlock(d22, 0, 0, t)
		}
	}

	if ND > 0 {
		B2iD := extractBlock(dH.B, 0, w, nD, ND)
		C2iD := extractBlock(dH.C, z, 0, ND, nD)
		D22iD := extractBlock(dH.D, z, w, ND, ND)

		if nM > 0 {
			setBlock(b2, 0, NM, mulDense(B2p, GD12iD))
		}
		if nD > 0 {
			t := addMulDims(nD, ND, B2iD, BDe, FD22pD12iD)
			setBlock(b2, nM, NM, t)
		}

		if nM > 0 {
			setBlock(c2, NM, 0, mulDense(D21iD, FC2p))
		}
		if nD > 0 {
			t := addMulDims(ND, nD, C2iD, D21iD, FD22pCDe)
			setBlock(c2, NM, nM, t)
		}

		setBlock(d12, 0, NM, mulDense(D12p, GD12iD))

		setBlock(d21, NM, 0, mulDense(D21iD, FD21p))

		{
			t := addMulDims(ND, ND, D22iD, D21iD, FD22pD12iD)
			setBlock(d22, NM, NM, t)
		}
	}

	if NM > 0 && ND > 0 {
		setBlock(d22, 0, NM, mulDense(D21iMz, GD12iD))
		setBlock(d22, NM, 0, mulDense(D21iD, FD12iMw))
	}

	if n == 0 {
		Acl = &mat.Dense{}
	} else {
		Acl = resizeDense(Acl, n, n)
	}
	Bcl = resizeDense(Bcl, n, mTotal)
	Ccl = resizeDense(Ccl, pTotal, n)
	Dcl = resizeDense(Dcl, pTotal, mTotal)

	b2 = resizeDense(b2, n, N)
	c2 = resizeDense(c2, N, n)
	d12 = resizeDense(d12, nz, N)
	d21 = resizeDense(d21, N, nw)

	setBlock(Bcl, 0, nw, b2)
	setBlock(Ccl, nz, 0, c2)
	setBlock(Dcl, 0, nw, d12)
	setBlock(Dcl, nz, 0, d21)
	setBlock(Dcl, nz, nw, d22)

	H := &System{A: Acl, B: Bcl, C: Ccl, D: Dcl, E: blkDiagDescriptorE(mH, dH), Dt: M.Dt}

	tau := make([]float64, N)
	if mLFT.LFT != nil {
		copy(tau, mLFT.LFT.Tau)
	}
	if dLFT.LFT != nil {
		copy(tau[NM:], dLFT.LFT.Tau)
	}

	result, err := SetDelayModel(H, tau)
	if err != nil {
		return nil, err
	}
	if savedInputDelay.hasNonzero {
		result.InputDelay = savedInputDelay.values
	}
	if savedOutputDelay.hasNonzero {
		result.OutputDelay = savedOutputDelay.values
	}
	lftVisibleMetadata(M, nw, nz).applyIOOwned(result)
	return result, nil
}
