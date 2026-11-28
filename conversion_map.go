package controlsys

import (
	"errors"
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// C2DMap converts sys like C2D and also returns the initial-condition map G
// of MATLAB [sysd,G] = c2d(sysc,Ts,opts): x[0] = G·[x0; u0; w0], where x0
// is the continuous state, u0 the input at t = 0 and w0 the internal-delay
// outputs (zero history). G is nd×(n+m+N). Zero-order hold and impulse maps
// are exact; Tustin and FOH maps account for the initial input.
//
// When no exact map exists for the conversion (matched and least-squares
// methods, Thiran-approximated fractional delays, delays that change the
// state dimension, a result with no states), C2DMap returns
// ErrOptionUnsupported; C2D still converts. A singular-E descriptor model,
// whose algebraic states are eliminated, returns ErrDescriptorInitialState.
// For a full-rank continuous descriptor without internal delays, by ZOH,
// FOH, Tustin or impulse, the model and its error are those of C2D,
// bitwise, in whatever state coordinates C2D took (the E⁻¹A fold, its real
// Schur form, or the generalized Schur parts of the pencil), and G maps the
// descriptor state into them: on the parts, G = [W 0] for ZOH and impulse,
// W the map of x0 into the parts' states, and the parts' own Tustin or FOH
// map composed with diag(W, I) otherwise. So does a full-rank continuous
// descriptor with internal delays, by ZOH, FOH or Tustin, whose E has an
// LU condition estimate of 1e6 or more (C2D's stiff LFT split): there G
// maps the descriptor state into the real Schur forms of the parts, and
// its w0 columns are those of the split model. A model returned with an error
// wrapping ErrAccuracyUnverified comes with its G. The error is C2D's
// verdict on the DC gain of the model; G itself is not checked. Other descriptors take
// the model and G of E⁻¹A; with internal delays, by ZOH, FOH or Tustin,
// the error is then C2D's verdict on the DC gain of that model too. An Inf or NaN entry in the result or in G
// returns ErrOverflow.
func (sys *System) C2DMap(dt float64, opts C2DOptions) (*System, *mat.Dense, error) {
	if err := requireFiniteSystem("C2DMap", sys); err != nil {
		return nil, nil, err
	}
	opts, err := normalizeC2DOptions(dt, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	if c2dMapRouted(sys, opts.Method) {
		return sys.c2dMapRouted(dt, opts)
	}
	std, reduced, k, err := conversionStandardFormScaled(sys, "C2DMap")
	if err != nil {
		return nil, nil, err
	}
	if k != 0 {
		return nil, nil, fmt.Errorf("C2DMap: E⁻¹B overflows the descriptor state coordinates: %w", ErrOverflow)
	}
	if reduced {
		return nil, nil, fmt.Errorf("C2DMap: algebraic descriptor states are eliminated: %w", ErrDescriptorInitialState)
	}
	out, err := std.c2d(dt, opts)
	if err != nil && !errors.Is(err, errMatchedUnverified) {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	var gerr error
	if sys.LFT != nil && holdGated(sys, opts.Method) {
		gerr = newHoldGate(sys).check(out)
	}
	if !conversionCanMapState(std, out, opts.Method, dt) || (opts.ThiranOrder > 0 && conversionInitialMapHasFraction(std, dt)) {
		return nil, nil, fmt.Errorf("C2DMap: no exact initial-condition map for this delay structure: %w", ErrOptionUnsupported)
	}
	G, err := conversionStateMap(std, out, opts.Method, opts.PrewarpFrequency, false)
	if err != nil {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	if err := finiteDense("C2DMap", "G", G, ErrOverflow); err != nil {
		return nil, nil, err
	}
	if gerr != nil {
		return out, G, fmt.Errorf("C2DMap: %v: %w", gerr, ErrAccuracyUnverified)
	}
	return out, G, nil
}

// c2dMapRouted reports a C2DMap that follows C2D's route (c2dCoords): a
// continuous descriptor whose E is not rank deficient by the MATLAB rule,
// without internal delays by ZOH, FOH, Tustin or impulse, or with them by
// ZOH, FOH or Tustin where C2D tries the stiff LFT split (LU condition
// estimate of E foldAccurateECond or more).
func c2dMapRouted(sys *System, method C2DMethod) bool {
	switch method {
	case C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse:
	default:
		return false
	}
	if !sys.IsContinuous() || !sys.IsDescriptor() || sys.LFT != nil && method == C2DMethodImpulse {
		return false
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if eRankDeficient(sys.E, &lu) {
		return false
	}
	return sys.LFT == nil || lu.Cond() >= foldAccurateECond
}

// c2dMapRouted is C2DMap of a c2dMapRouted sys: C2D's result and the map
// into its coordinates.
func (sys *System) c2dMapRouted(dt float64, opts C2DOptions) (*System, *mat.Dense, error) {
	rec := &c2dCoords{}
	out, err := sys.c2dRecorded(dt, opts, rec)
	if rec.reduced {
		return nil, nil, fmt.Errorf("C2DMap: algebraic descriptor states are eliminated: %w", ErrDescriptorInitialState)
	}
	if err != nil && !carriesModel(err) {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	if rec.parts == nil && rec.cont != nil && rec.t == nil && rec.foldScale != 0 {
		return nil, nil, fmt.Errorf("C2DMap: E⁻¹B overflows the descriptor state coordinates: %w", ErrOverflow)
	}
	if (rec.parts == nil && rec.cont == nil) || !conversionCanMapState(sys, out, opts.Method, dt) || (opts.ThiranOrder > 0 && conversionInitialMapHasFraction(sys, dt)) {
		return nil, nil, fmt.Errorf("C2DMap: no exact initial-condition map for this delay structure: %w", ErrOptionUnsupported)
	}
	G, gerr := rec.stateMap(sys, out, opts.Method, opts.PrewarpFrequency)
	if gerr != nil {
		return nil, nil, fmt.Errorf("C2DMap: %w", gerr)
	}
	if gerr := finiteDense("C2DMap", "G", G, ErrOverflow); gerr != nil {
		return nil, nil, gerr
	}
	if err != nil {
		return out, G, fmt.Errorf("C2DMap: %v: %w", err, ErrAccuracyUnverified)
	}
	return out, G, nil
}

// c2dCoords records, for C2DMap, the state coordinates C2D discretized a
// continuous descriptor in. On the parts route (discretizeStiffDescriptor)
// parts are the explicit parts, each discretized alone, and shares the maps
// W_k of the descriptor state into their states (partShare). On the plan
// route cont is the explicit model discretized and t the map into its
// state, nil for the LU fold, whose state is the descriptor's; z, when set,
// holds the Schur vectors of the real Schur retry, whose state is zᵀ times
// that of t. A nil *c2dCoords records nothing.
type c2dCoords struct {
	parts     []*System
	shares    []*mat.Dense
	cont      *System
	t, z      *mat.Dense
	foldScale int
	reduced   bool
	gamma1    *mat.Dense
}

// riders returns [E A] of the descriptor sys, the riders whose transforms
// give the shares, or nil for a nil c.
func (c *c2dCoords) riders(sys *System) *mat.Dense {
	if c == nil || !sys.IsDescriptor() {
		return nil
	}
	n, _, _ := sys.Dims()
	r := newDense(n, 2*n)
	if n > 0 {
		r.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.E)
		r.Slice(0, n, n, 2*n).(*mat.Dense).Copy(sys.A)
	}
	return r
}

func (c *c2dCoords) setParts(sp stiffSplit) {
	if c == nil {
		return
	}
	c.parts = sp.parts
	c.shares = make([]*mat.Dense, len(sp.parts))
	for k, part := range sp.parts {
		c.shares[k] = partShare(part, sp.riders[k], k > 0)
	}
}

// setSchurParts records the parts of stiffDescriptorExplicit, each in the
// coordinates of its real Schur form (zs[k]; nil when unchanged), as the map
// t of the parallel connection.
func (c *c2dCoords) setSchurParts(sp stiffSplit, zs []*mat.Dense) {
	if c == nil {
		return
	}
	nt := 0
	for _, part := range sp.parts {
		pk, _, _ := part.Dims()
		nt += pk
	}
	_, nr := sp.riders[0].Dims()
	t := newDense(nt, nr/2)
	off := 0
	for k, part := range sp.parts {
		pk, _, _ := part.Dims()
		w := partShare(part, sp.riders[k], k > 0)
		dst := t.Slice(off, off+pk, 0, nr/2).(*mat.Dense)
		if zs[k] != nil {
			dst.Mul(zs[k].T(), w)
		} else {
			dst.Copy(w)
		}
		off += pk
	}
	c.t = t
}

// setCont records the plan route: src and z as in c2dCoords, with the fold
// scaling 2^foldScale and reduction of the plan's standard form.
func (c *c2dCoords) setCont(src *System, z *mat.Dense, foldScale int, reduced bool) {
	if c == nil {
		return
	}
	c.cont, c.z, c.foldScale, c.reduced = src, z, foldScale, reduced
}

// partShare is the map of the descriptor state into the state of a part of
// stiffDescriptorPartsRiders with riders [E A] taken through it, as
// freePartShare: a fast part takes A_k⁻¹ times the A block, which carries
// no cancellation in the nearly null directions of E, where A_k is
// solvable (1-norm condition below 2⁵²); a slow part, or a fast one with
// near-singular A_k, keeps the E block.
func partShare(part *System, riders *mat.Dense, fast bool) *mat.Dense {
	pk, _, _ := part.Dims()
	_, nr := riders.Dims()
	n := nr / 2
	if fast {
		var lu mat.LU
		lu.Factorize(part.A)
		if lu.Cond() < 0x1p52 {
			var s mat.Dense
			if err := lu.SolveTo(&s, false, riders.Slice(0, pk, n, nr)); err == nil {
				return &s
			}
		}
	}
	return mat.DenseCopyOf(riders.Slice(0, pk, 0, n))
}

// stateMap returns G of the recorded route for out: on the parts, per part
// k its own map (conversionStateMap; for FOH [I −Γ1] with the Γ1 of the
// part's hold, fohInputIntegrals) composed with diag(W_k, I); on the
// plan route, the map of cont composed with diag(T, I), T = zᵀ·t (t = I for
// the fold).
func (c *c2dCoords) stateMap(sys, out *System, method C2DMethod, prewarp float64) (*mat.Dense, error) {
	n, m, _ := sys.Dims()
	nt, _, _ := out.Dims()
	if c.parts != nil {
		G := newDense(nt, n+m)
		off := 0
		for k, part := range c.parts {
			pk, _, _ := part.Dims()
			if pk == 0 {
				continue
			}
			rows := G.Slice(off, off+pk, 0, n+m).(*mat.Dense)
			off += pk
			if method == C2DMethodFOH {
				rows.Slice(0, pk, 0, n).(*mat.Dense).Copy(c.shares[k])
				if m > 0 {
					_, _, gamma1 := fohInputIntegrals(part.A, part.B, out.Dt)
					rows.Slice(0, pk, n, n+m).(*mat.Dense).Scale(-1, gamma1)
				}
				continue
			}
			stub := &System{A: part.A, B: part.B, C: part.C, D: part.D, Dt: out.Dt}
			gp, err := conversionStateMap(part, stub, method, prewarp, false)
			if err != nil {
				return nil, err
			}
			rows.Slice(0, pk, 0, n).(*mat.Dense).Mul(gp.Slice(0, pk, 0, pk), c.shares[k])
			if m > 0 {
				rows.Slice(0, pk, n, n+m).(*mat.Dense).Copy(gp.Slice(0, pk, pk, pk+m))
			}
		}
		return G, nil
	}
	if method == C2DMethodFOH && c.t != nil && c.cont.HasInternalDelay() {
		return c.lftFOHMap(nt)
	}
	gc, err := conversionStateMap(c.cont, out, method, prewarp, false)
	if err != nil {
		return nil, err
	}
	nc, _, _ := c.cont.Dims()
	T := c.t
	if T == nil {
		T = newDense(nc, n)
		for i := range min(nc, n) {
			T.Set(i, i, 1)
		}
	}
	if c.z != nil {
		var zt mat.Dense
		zt.Mul(c.z.T(), T)
		T = &zt
	}
	_, cols := gc.Dims()
	G := newDense(nt, n+cols-nc)
	if nt > 0 && n > 0 {
		G.Slice(0, nt, 0, n).(*mat.Dense).Mul(gc.Slice(0, nt, 0, nc), T)
	}
	if cols > nc {
		G.Slice(0, nt, n, n+cols-nc).(*mat.Dense).Copy(gc.Slice(0, nt, nc, cols))
	}
	return G, nil
}

// lftFOHMap is the FOH G of the stiff LFT route: [T −Γ1], Γ1 the hold of
// [B B2] that C2D's discretization of the split model took (gamma1).
func (c *c2dCoords) lftFOHMap(nt int) (*mat.Dense, error) {
	nc, m, _ := c.cont.Dims()
	_, n := c.t.Dims()
	nw := c.cont.internalDelayCount()
	if nt != nc {
		return nil, fmt.Errorf("foh changes the state dimension: %w", ErrOptionUnsupported)
	}
	G := newDense(nt, n+m+nw)
	G.Slice(0, nt, 0, n).(*mat.Dense).Copy(c.t)
	if nt > 0 && m+nw > 0 {
		if c.gamma1 == nil {
			return nil, fmt.Errorf("foh hold of the split model not recorded: %w", ErrOptionUnsupported)
		}
		G.Slice(0, nt, n, n+m+nw).(*mat.Dense).Scale(-1, c.gamma1)
	}
	return G, nil
}

// D2CMap converts sys like D2C and also returns the map G of MATLAB
// [sysc,G] = d2c(sysd,opts): xc(k·Ts) = G·[xd[k]; u[k]; w[k]]. Its
// availability follows C2DMap. An Inf or NaN entry in the result or in G returns
// ErrOverflow.
func (sys *System) D2CMap(opts D2COptions) (*System, *mat.Dense, error) {
	if err := requireFiniteSystem("D2CMap", sys); err != nil {
		return nil, nil, err
	}
	if opts.Method == "" {
		opts.Method = C2DMethodZOH
	}
	std, reduced, k, err := conversionStandardFormScaled(sys, "D2CMap")
	if err != nil {
		return nil, nil, err
	}
	if k != 0 {
		return nil, nil, fmt.Errorf("D2CMap: E⁻¹B overflows the descriptor state coordinates: %w", ErrOverflow)
	}
	if reduced {
		return nil, nil, fmt.Errorf("D2CMap: algebraic descriptor states are eliminated: %w", ErrDescriptorInitialState)
	}
	out, err := std.d2c(opts)
	if err != nil && !errors.Is(err, errMatchedUnverified) {
		return nil, nil, fmt.Errorf("D2CMap: %w", err)
	}
	if !conversionCanMapState(std, out, opts.Method, std.Dt) {
		return nil, nil, fmt.Errorf("D2CMap: no exact initial-condition map for this delay structure: %w", ErrOptionUnsupported)
	}
	G, err := conversionStateMap(std, out, opts.Method, opts.PrewarpFrequency, true)
	if err != nil {
		return nil, nil, fmt.Errorf("D2CMap: %w", err)
	}
	if err := finiteDense("D2CMap", "G", G, ErrOverflow); err != nil {
		return nil, nil, err
	}
	return out, G, nil
}

func initialStateIdentity(rows, sourceStates, columns int) *mat.Dense {
	mapping := newDense(rows, columns)
	raw := mapping.RawMatrix()
	for i := range min(rows, sourceStates) {
		raw.Data[i*raw.Stride+i] = 1
	}
	return mapping
}

func conversionStateMap(source, out *System, method C2DMethod, prewarp float64, inverse bool) (*mat.Dense, error) {
	n, m, _ := source.Dims()
	nt, _, _ := out.Dims()
	if nt == 0 {
		return nil, fmt.Errorf("converted model has no states: %w", ErrDimensionMismatch)
	}
	if method == C2DMethodMatched || method == C2DMethodLeastSquares {
		return nil, fmt.Errorf("method %q has no exact initial-condition map: %w", method, ErrOptionUnsupported)
	}
	mapping := initialStateIdentity(nt, n, n+m+source.internalDelayCount())
	if method == C2DMethodZOH || method == C2DMethodImpulse || n == 0 {
		return mapping, nil
	}
	raw := mapping.RawMatrix()
	switch method {
	case C2DMethodTustin:
		dt := out.Dt
		if inverse {
			dt = source.Dt
		}
		beta, err := tustinBeta(dt, prewarp)
		if err != nil {
			return nil, err
		}
		scale := beta
		if inverse {
			scale = 2
		}
		a, b := source.A.RawMatrix(), source.B.RawMatrix()
		for i := range n {
			for j := range n {
				value := -a.Data[i*a.Stride+j]
				if inverse {
					value = a.Data[i*a.Stride+j]
				}
				if i == j {
					if inverse {
						value += 1
					} else {
						value += beta
					}
				}
				raw.Data[i*raw.Stride+j] = value / scale
			}
			for j := range m {
				value := b.Data[i*b.Stride+j] / scale
				if !inverse {
					value = -value
				}
				raw.Data[i*raw.Stride+n+j] = value
			}
		}
		if source.HasInternalDelay() {
			b2 := source.LFT.B2.RawMatrix()
			for i := range n {
				for j := range source.internalDelayCount() {
					value := b2.Data[i*b2.Stride+j] / scale
					if !inverse {
						value = -value
					}
					raw.Data[i*raw.Stride+n+m+j] = value
				}
			}
		}

	case C2DMethodFOH:
		continuous := source
		dt := out.Dt
		if inverse {
			continuous = out
			dt = source.Dt
		}
		if nt != n {
			return nil, fmt.Errorf("foh changes the state dimension: %w", ErrOptionUnsupported)
		}
		_, _, k1 := fohKernel(continuous.A, dt)
		var gamma mat.Dense
		if m > 0 {
			gamma.Mul(k1, continuous.B)
		}
		gr := gamma.RawMatrix()
		for i := range n {
			for j := range m {
				value := gr.Data[i*gr.Stride+j]
				if !inverse {
					value = -value
				}
				raw.Data[i*raw.Stride+n+j] = value
			}
		}
		if continuous.HasInternalDelay() {
			var gamma2 mat.Dense
			if n > 0 {
				gamma2.Mul(k1, continuous.LFT.B2)
			}
			g2 := gamma2.RawMatrix()
			for i := range n {
				for j := range source.internalDelayCount() {
					value := g2.Data[i*g2.Stride+j]
					if !inverse {
						value = -value
					}
					raw.Data[i*raw.Stride+n+m+j] = value
				}
			}
		}
	default:
		return nil, fmt.Errorf("method %q has no initial-condition map: %w", method, ErrOptionUnsupported)
	}
	return mapping, nil
}

func conversionInitialMapHasFraction(sys *System, dt float64) bool {
	if conversionHasFractionalExternalDelay(sys, dt) || conversionHasFractionalPathDelay(sys.Delay, dt) {
		return true
	}
	if sys.HasInternalDelay() {
		for _, tau := range sys.LFT.Tau {
			if !isIntegerSampleDelay(tau / dt) {
				return true
			}
		}
	}
	return false
}

func conversionCanMapState(source, out *System, method C2DMethod, dt float64) bool {
	if !source.HasDelay() && !out.HasDelay() {
		return true
	}
	n, _, _ := source.Dims()
	nt, _, _ := out.Dims()
	if method == C2DMethodTustin {
		return nt == n && source.internalDelayCount() == out.internalDelayCount()
	}
	if method == C2DMethodFOH && nt == n && source.HasInternalDelay() && !hasExternalDelay(source, true) {
		for _, tau := range source.LFT.Tau {
			if source.IsContinuous() && !isIntegerSampleDelay(tau/dt) {
				return false
			}
		}
		return true
	}
	if method != C2DMethodZOH || nt < n {
		return false
	}
	for _, part := range [][]float64{source.InputDelay, source.OutputDelay} {
		for _, tau := range part {
			if !isIntegerSampleDelay(tau/dt) && source.IsContinuous() {
				return false
			}
		}
	}
	if source.Delay != nil {
		r, c := source.Delay.Dims()
		for i := range r {
			for j := range c {
				if source.IsContinuous() && !isIntegerSampleDelay(source.Delay.At(i, j)/dt) {
					return false
				}
			}
		}
	}
	if source.HasInternalDelay() {
		for _, tau := range source.LFT.Tau {
			if source.IsContinuous() && !isIntegerSampleDelay(tau/dt) {
				return false
			}
		}
	}
	return true
}
