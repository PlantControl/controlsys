package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// Augw returns the mixed-sensitivity generalized plant of G with weights W1,
// W2 and W3 on the error, control and output signals, as MATLAB
// augw(G,W1,W2,W3); see
// https://www.mathworks.com/help/robust/ref/dynamicsystem.augw.html.
//
// P has inputs [w; u] and outputs [z1; z2; z3; e]:
//
//	z1 = W1·e,  z2 = W2·u,  z3 = W3·G·u,  e = w − G·u,
//
// so P = [W1 −W1·G; 0 W2; 0 W3·G; I −G], and lower LFT(P, K) with u = K·e
// is [W1·S; W2·K·S; W3·T] with S = (I + G·K)⁻¹ and T = G·K·S. A nil weight
// drops its rows. For G with NY outputs and NU inputs, W1 and W3 are SISO
// (applied as W·I) or NY×NY and W2 is SISO or NU×NU. MATLAB marks the
// partition with InputGroup/OutputGroup; this library has no groups, so P's
// channels are named w, u, z1, z2, z3 and e (vector channels expand to
// w(1), w(2), …), with the last NY outputs measurements and the last NU
// inputs controls.
//
// Delays are kept as in the underlying interconnection. W3 may be improper
// (a descriptor model) provided W3·G is proper; a descriptor P is returned
// as an equivalent explicit model, and an improper P returns
// ErrImproperModel. A nil G returns ErrInvalidArgument, wrongly sized
// weights ErrDimensionMismatch and different sample times ErrDomainMismatch.
func Augw(G, W1, W2, W3 *System) (*System, error) {
	return augw("Augw", G, W1, W2, W3)
}

func augw(op string, G, W1, W2, W3 *System) (*System, error) {
	if err := requireSystem(op, G); err != nil {
		return nil, err
	}
	_, nu, ny := G.Dims()
	w1, err := expandWeight(op, "W1", W1, G, ny)
	if err != nil {
		return nil, err
	}
	w2, err := expandWeight(op, "W2", W2, G, nu)
	if err != nil {
		return nil, err
	}
	w3, err := expandWeight(op, "W3", W3, G, ny)
	if err != nil {
		return nil, err
	}
	P, err := augwInterconnect(G, w1, w2, w3, ny, nu)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if P.IsDescriptor() {
		proper, err := P.isProper()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		if !proper {
			return nil, fmt.Errorf("%s: generalized plant is improper (W3·G or a weight improper): %w", op, ErrImproperModel)
		}
		if P, err = P.properExplicitForm(); err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
	}

	var outNames []string
	for _, z := range []struct {
		name string
		w    *System
		size int
	}{{"z1", w1, ny}, {"z2", w2, nu}, {"z3", w3, ny}} {
		if z.w != nil {
			outNames = append(outNames, expandName(z.name, z.size)...)
		}
	}
	P.OutputName = append(outNames, expandName("e", ny)...)
	P.InputName = append(expandName("w", ny), expandName("u", nu)...)
	return P, nil
}

// expandWeight validates the optional weight W of a G channel group of the
// given size, returning nil for nil and W·I (as copies on the diagonal) for a
// SISO W.
func expandWeight(op, name string, W, G *System, size int) (*System, error) {
	if W == nil {
		return nil, nil
	}
	if err := W.validate(); err != nil {
		return nil, fmt.Errorf("%s: %s: %w", op, name, err)
	}
	if err := domainMatch(G, W); err != nil {
		return nil, fmt.Errorf("%s: %s: %w", op, name, err)
	}
	_, m, p := W.Dims()
	switch {
	case m == size && p == size:
		return W, nil
	case m == 1 && p == 1:
		copies := make([]*System, size)
		for i := range copies {
			copies[i] = W
		}
		Wd, err := BlkDiag(copies...)
		if err != nil {
			return nil, fmt.Errorf("%s: %s: %w", op, name, err)
		}
		return Wd, nil
	default:
		return nil, fmt.Errorf("%s: %s is %d×%d, want 1×1 or %d×%d: %w", op, name, p, m, size, size, ErrDimensionMismatch)
	}
}

// augwInterconnect builds P from [w; u] → [w; u; u] → [w; u; G·u] → the
// pre-weight signals [e; u; G·u; e] of the present weights → the weighted
// outputs.
func augwInterconnect(G, W1, W2, W3 *System, ny, nu int) (*System, error) {
	dt := G.Dt
	split := newDense(ny+2*nu, ny+nu)
	for i := range ny + nu {
		split.Set(i, i, 1)
	}
	for i := range nu {
		split.Set(ny+nu+i, ny+i, 1)
	}
	splitSys, err := newGain(split, dt)
	if err != nil {
		return nil, err
	}
	passSys, err := newGain(eyeDense(ny+nu), dt)
	if err != nil {
		return nil, err
	}
	plant, err := BlkDiag(passSys, G)
	if err != nil {
		return nil, err
	}
	H, err := Series(splitSys, plant)
	if err != nil {
		return nil, err
	}

	type rows struct {
		w, u, gu float64
		size     int
		weight   *System
	}
	blocks := []rows{
		{w: 1, gu: -1, size: ny, weight: W1},
		{u: 1, size: nu, weight: W2},
		{gu: 1, size: ny, weight: W3},
		{w: 1, gu: -1, size: ny},
	}
	nout := 0
	var weights []*System
	for i, b := range blocks {
		if b.weight == nil && i < 3 {
			continue
		}
		nout += b.size
	}
	mix := newDense(nout, ny+nu+ny)
	r := 0
	for i, b := range blocks {
		if b.weight == nil && i < 3 {
			continue
		}
		for k := range b.size {
			if b.w != 0 {
				mix.Set(r+k, k, b.w)
			}
			if b.u != 0 {
				mix.Set(r+k, ny+k, b.u)
			}
			if b.gu != 0 {
				mix.Set(r+k, ny+nu+k, b.gu)
			}
		}
		r += b.size
		if b.weight != nil {
			weights = append(weights, b.weight)
		}
	}
	mixSys, err := newGain(mix, dt)
	if err != nil {
		return nil, err
	}
	if H, err = Series(H, mixSys); err != nil {
		return nil, err
	}
	measure, err := newGain(eyeDense(ny), dt)
	if err != nil {
		return nil, err
	}
	Wblk, err := BlkDiag(append(weights, measure)...)
	if err != nil {
		return nil, err
	}
	return Series(H, Wblk)
}

// MixsynResult is the mixed-sensitivity H∞ design of MATLAB
// [K,CL,gamma,info] = mixsyn(G,W1,W2,W3).
type MixsynResult struct {
	// K is the controller, driven by e = r − y and producing u.
	K *System
	// CL is [W1·S; W2·K·S; W3·T] from w to the present z channels.
	CL *System
	// Gamma is the achieved H∞ norm of CL.
	Gamma float64
	// Info is the underlying HinfSyn design on Augw(G, W1, W2, W3).
	Info *HinfSynResult
}

// Mixsyn designs an H∞ controller K for the plant G minimizing the H∞ norm
// of the weighted closed loop [W1·S; W2·K·S; W3·T], as MATLAB
// mixsyn(G,W1,W2,W3); see
// https://www.mathworks.com/help/robust/ref/dynamicsystem.mixsyn.html.
// It is HinfSyn on Augw(G, W1, W2, W3) with NY measurements and NU controls;
// CL is the lower LFT of that plant with K and Gamma its H∞ norm. Weights
// should be stable and G stabilizable and detectable, as MATLAB requires.
// MATLAB's gamTry, gamRange and opts arguments are not supported.
//
// Errors are Augw's, plus: D12 = [−W1(∞)·G(∞); W2(∞); W3·G(∞)] without full
// column rank (for example W2 nil with a strictly proper G) returns
// ErrInvalidPartition, since the Riccati synthesis needs it; a discrete G
// returns HinfSyn's ErrWrongDomain (MATLAB also handles discrete plants,
// this library's HinfSyn does not); delays return ErrDelayUnsupported, as
// MATLAB hinfsyn rejects them; and synthesis failures are HinfSyn's.
func Mixsyn(G, W1, W2, W3 *System) (*MixsynResult, error) {
	const op = "Mixsyn"
	P, err := augw(op, G, W1, W2, W3)
	if err != nil {
		return nil, err
	}
	_, nu, ny := G.Dims()
	_, m, p := P.Dims()
	if p > ny && nu > 0 {
		if !fullColumnRank(mat.DenseCopyOf(P.D.Slice(0, p-ny, m-nu, m))) {
			return nil, fmt.Errorf("%s: D12 = [−W1·D_G; W2(∞); W3·D_G] has rank below %d; W2 needs feedthrough or G must be biproper: %w", op, nu, ErrInvalidPartition)
		}
	}
	info, err := HinfSyn(P, ny, nu)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	CL, err := LFT(P, info.K, LFTFeedback{Nu: nu, Ny: ny})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	gamma, _, err := HinfNorm(CL)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return &MixsynResult{K: info.K, CL: CL, Gamma: gamma, Info: info}, nil
}

// fullColumnRank reports whether the tall matrix M has full column rank,
// relative to its largest singular value.
func fullColumnRank(M *mat.Dense) bool {
	r, c := M.Dims()
	if r < c {
		return false
	}
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDNone) {
		return false
	}
	s := svd.Values(nil)
	return s[c-1] > float64(max(r, c))*eps()*s[0]
}
