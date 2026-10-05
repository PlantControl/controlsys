package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

func LFT(M, Delta *System, nu, ny int) (*System, error) {
	if M == nil {
		return nil, fmt.Errorf("lft: M cannot be nil: %w", ErrDimensionMismatch)
	}
	_, mM, pM := M.Dims()
	if nu < 0 || ny < 0 {
		return nil, fmt.Errorf("lft: nu and ny must be non-negative: %w", ErrDimensionMismatch)
	}
	if nu > mM {
		return nil, fmt.Errorf("lft: nu=%d > M inputs=%d: %w", nu, mM, ErrDimensionMismatch)
	}
	if ny > pM {
		return nil, fmt.Errorf("lft: ny=%d > M outputs=%d: %w", ny, pM, ErrDimensionMismatch)
	}

	if Delta != nil {
		_, mD, pD := Delta.Dims()
		if pM-ny != mD {
			return nil, fmt.Errorf("lft: M lower outputs %d != Delta inputs %d: %w", pM-ny, mD, ErrDimensionMismatch)
		}
		if mM-nu != pD {
			return nil, fmt.Errorf("lft: M lower inputs %d != Delta outputs %d: %w", mM-nu, pD, ErrDimensionMismatch)
		}
		if err := domainMatch(M, Delta); err != nil {
			return nil, fmt.Errorf("lft: %w", err)
		}
	}

	if Delta == nil {
		return lftExtract(M, nu, ny)
	}

	needsLFT := M.HasDelay() || M.HasInternalDelay() || Delta.HasDelay() || Delta.HasInternalDelay()
	if needsLFT {
		return lftWithDelay(M, Delta, nu, ny)
	}

	if mM == nu && pM == ny {
		return lftEmptyLoop(M, Delta, nu, ny)
	}
	return lftSimple(M, Delta, nu, ny)
}

// lftEmptyLoop closes an LFT whose loop carries no signals: the result is M's
// upper channels with Delta's states appended, unreachable and unobservable.
func lftEmptyLoop(M, Delta *System, nu, ny int) (*System, error) {
	upper, err := lftExtract(M, nu, ny)
	if err != nil {
		return nil, err
	}
	if nD, _, _ := Delta.Dims(); nD == 0 {
		return upper, nil
	}
	return Append(upper, Delta)
}

func lftExtract(M *System, nu, ny int) (*System, error) {
	nM, _, _ := M.Dims()
	D11 := extractBlock(M.D, 0, 0, ny, nu)

	if nM == 0 {
		result, err := lftGain(D11, ny, nu, M.Dt)
		if err != nil {
			return nil, err
		}
		lftVisibleMetadata(M, nu, ny).applyIOOwned(result)
		return result, nil
	}

	B1 := extractBlock(M.B, 0, 0, nM, nu)
	C1 := extractBlock(M.C, 0, 0, ny, nM)
	result, err := newNoCopy(denseCopy(M.A), B1, C1, D11, M.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(M.E)
	lftVisibleMetadata(M, nu, ny).applyIOOwned(result)
	return result, nil
}

func lftSimple(M, Delta *System, nu, ny int) (*System, error) {
	nM, mM, pM := M.Dims()
	nD, _, _ := Delta.Dims()

	w := pM - ny
	z := mM - nu

	D11 := extractBlock(M.D, 0, 0, ny, nu)
	D12 := extractBlock(M.D, 0, nu, ny, z)
	D21 := extractBlock(M.D, ny, 0, w, nu)
	D22 := extractBlock(M.D, ny, nu, w, z)

	B1 := extractBlock(M.B, 0, 0, nM, nu)
	B2 := extractBlock(M.B, 0, nu, nM, z)
	C1 := extractBlock(M.C, 0, 0, ny, nM)
	C2 := extractBlock(M.C, ny, 0, w, nM)

	F, err := solveLFTLoop(D22, Delta.D, w)
	if err != nil {
		return nil, err
	}

	Phi := mulDense(Delta.D, F)
	G := lftLoopGain(Phi, D22, z)

	PhiC2 := mulDense(Phi, C2)
	PhiD21 := mulDense(Phi, D21)

	n := nM + nD

	Dcl := addMulDims(ny, nu, D11, D12, PhiD21)
	if n == 0 {
		result, err := lftGain(Dcl, ny, nu, M.Dt)
		if err != nil {
			return nil, err
		}
		lftVisibleMetadata(M, nu, ny).applyIOOwned(result)
		return result, nil
	}

	Acl := mat.NewDense(n, n, nil)
	Bcl := newDense(n, nu)
	Ccl := newDense(ny, n)

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
	lftVisibleMetadata(M, nu, ny).applyIOOwned(result)
	return result, nil
}

// lftGain builds a static LFT result. A gain with no inputs or no outputs but
// not both carries no dimensions, so it is rejected as in NewFromSlices.
func lftGain(D *mat.Dense, ny, nu int, dt float64) (*System, error) {
	if err := lftGainDims(ny, nu); err != nil {
		return nil, err
	}
	return buildSystem(nil, nil, nil, D, dt, nil)
}

func lftGainDims(ny, nu int) error {
	if (ny == 0) != (nu == 0) {
		return fmt.Errorf("lft: %dx%d static gain cannot be stored: %w", ny, nu, ErrDimensionMismatch)
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

func lftWithDelay(M, Delta *System, nu, ny int) (*System, error) {
	_, mM, pM := M.Dims()

	savedInputDelay := selectLeadingVisibleDelays(M.InputDelay, nu)
	savedOutputDelay := selectLeadingVisibleDelays(M.OutputDelay, ny)

	mCopy := M.Copy()
	mCopy.InputDelay = clearLeadingDelays(mCopy.InputDelay, nu)
	mCopy.OutputDelay = clearLeadingDelays(mCopy.OutputDelay, ny)

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

	mH, _ := mLFT.GetDelayModel()
	dH, _ := dLFT.GetDelayModel()

	nM, _, _ := mH.Dims()
	nD, _, _ := dH.Dims()

	w := pM - ny
	z := mM - nu
	n := nM + nD
	mTotal := nu + N
	pTotal := ny + N

	D11 := extractBlock(mH.D, 0, 0, ny, nu)
	D12p := extractBlock(mH.D, 0, nu, ny, z)
	D21p := extractBlock(mH.D, ny, 0, w, nu)
	D22p := extractBlock(mH.D, ny, nu, w, z)

	DDe := extractBlock(dH.D, 0, 0, z, w)

	F, err := solveLFTLoop(D22p, DDe, w)
	if err != nil {
		return nil, err
	}

	Phi := mulDense(DDe, F)
	G := lftLoopGain(Phi, D22p, z)

	B1 := extractBlock(mH.B, 0, 0, nM, nu)
	B2p := extractBlock(mH.B, 0, nu, nM, z)
	C1 := extractBlock(mH.C, 0, 0, ny, nM)
	C2p := extractBlock(mH.C, ny, 0, w, nM)

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
			if err := lftGainDims(ny, nu); err != nil {
				return nil, err
			}
			Acl = &mat.Dense{}
		} else {
			Acl = resizeDense(Acl, n, n)
		}
		Bcl = resizeDense(Bcl, n, nu)
		Ccl = resizeDense(Ccl, ny, n)
		Dcl = resizeDense(Dcl, ny, nu)

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
		lftVisibleMetadata(M, nu, ny).applyIOOwned(sys)
		return sys, nil
	}

	var D12iMu, D12iMw *mat.Dense
	var D21iMext, D21iMz *mat.Dense
	var PhiD12iMw, FD12iMw *mat.Dense
	if NM > 0 {
		D12iMu = extractBlock(mH.D, 0, nu+z, ny, NM)
		D12iMw = extractBlock(mH.D, ny, nu+z, w, NM)
		D21iMext = extractBlock(mH.D, ny+w, 0, NM, nu)
		D21iMz = extractBlock(mH.D, ny+w, nu, NM, z)
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
	d12 := mat.NewDense(max(ny, 1), N, nil)
	d21 := mat.NewDense(N, max(nu, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if NM > 0 {
		B2iM := extractBlock(mH.B, 0, nu+z, nM, NM)
		C2iM := extractBlock(mH.C, ny+w, 0, NM, nM)
		D22iM := extractBlock(mH.D, ny+w, nu+z, NM, NM)

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
			t := addMulDims(ny, NM, D12iMu, D12p, PhiD12iMw)
			setBlock(d12, 0, 0, t)
		}

		{
			t := addMulDims(NM, nu, D21iMext, D21iMz, PhiD21p)
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
	d12 = resizeDense(d12, ny, N)
	d21 = resizeDense(d21, N, nu)

	setBlock(Bcl, 0, nu, b2)
	setBlock(Ccl, ny, 0, c2)
	setBlock(Dcl, 0, nu, d12)
	setBlock(Dcl, ny, 0, d21)
	setBlock(Dcl, ny, nu, d22)

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
	lftVisibleMetadata(M, nu, ny).applyIOOwned(result)
	return result, nil
}
