package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// C2D converts a continuous-time model to discrete time with sample time
// dt, as MATLAB sysd = c2d(sysc,Ts,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.c2d.html. The zero
// C2DOptions selects zero-order hold, the MATLAB default; MATLAB
// c2d(sysc,Ts,method) is C2D(dt, C2DOptions{Method: method}). Input, output
// and I/O delays become sample delays (fractional ones per opts), internal
// delays stay internal, and a singular-E descriptor model is first reduced as
// MATLAB dss2ss does. A model without internal delays whose full-rank E
// is too ill-conditioned for E⁻¹A (as in TransferFunction; for impulse and
// for a model with I/O delays, also LU condition estimate 1e6 or more) is
// discretized from its generalized Schur parts, which keep the poles of the
// pencil (A, E), by every method but least squares (matched from the
// transfer function of the pencil), its I/O delays as for an explicit
// model; the result then has new state coordinates and no state names. The
// split is certified against the descriptor bar: where the stiff and slow
// modes cannot be separated reliably (a split that fails or is not
// certified, for ZOH and FOH within half the decoupling budget), C2D
// returns an error wrapping ErrSchurFailed and a nil model, every method
// alike, except that ZOH, FOH and Tustin of a delay-free model whose LU
// condition estimate is below 1e6 return the model of E⁻¹A with an error
// wrapping ErrAccuracyUnverified. With internal delays, all
// methods that accept them (ZOH, FOH, Tustin) treat the descriptor part G of H = lft(G, delay) the
// same way and keep the delay feedback. ZOH, FOH and impulse of a stiff A (ρ(A)·dt ≥ 1000) in real Schur
// shape, such as those parts, exponentiate its modes of different scale
// separately; with all modes stiff, any A takes Γ = A⁻¹(e^{A·dt} − I)·B. An already discrete model returns ErrWrongDomain;
// invalid options return ErrInvalidConversionOptions; a result with an Inf or
// NaN entry in any matrix returns ErrOverflow.
//
// ZOH, FOH and Tustin of a continuous descriptor model are certified at
// DC: a nil error means that the DC gain of the
// result, exact for its stored entries, lies entry by entry within the
// descriptor bar u·(‖C_i‖‖M‖‖A‖‖M‖‖B_j‖ + ‖C_i‖‖M‖‖B_j‖ + |D_ij|),
// M = A⁻¹, u = 4(n+5)ε (Frobenius norms of the delay-free sys, row i of C,
// column j of B) of g0 = DCGain of the delay-free sys, plus ε·S_ij, the
// rounding of the result's own entries (Ad, Bd, Cd, Dd):
// S = |Dd| + Σ|Cd||X| + |Cd(I − Ad)⁻¹|·(|Ad||X| + |Bd|), X = (I − Ad)⁻¹Bd,
// counted only where ε·S_ij ≤ √ε·max(|g0_ij|, Σ|C_i||A⁻¹B_j| + |D_ij|).
// With I/O delays the check is on the model returned, its internal delays
// (unity at DC) closed. With internal delays, g0 is the DC gain of sys
// with its delays closed (unity at DC), the pencil [A B2; C2 D22 − I] with
// input [B; D21] and output [C D12], certified by its own solve or else
// its DCGain as a delay-free descriptor, and the bar keeps each delay
// channel k as its own block: u·(ĉ·m̂·â·m̂·b̂ + ĉ·m̂·b̂ + |D_ij|),
// u = 4(n+N+5)ε, with ĉ = [‖C_i‖, |D12_ik|], b̂ = [‖B_j‖; |D21_kj|],
// â = [‖A‖, ‖B2_·k‖; ‖C2_k·‖, |D22_kl|] and m̂ the same block norms of the
// pencil's inverse, so that rescaling a delay channel or the state
// equation, which leaves the model the same, leaves the bar the same; T0
// is the pencil's Σ|C_i||X_j| + |D_ij|. Where C2, D21 and D22 are zero
// the delayed signals vanish, and g0 and the bar are those of the
// delay-free core. An entry where g0 is infinite (a pole at DC) is not
// checked; where other entries are, there is no bar and the model is
// unverified. Where the model
// of a dense E⁻¹A without internal delays fails that check, ZOH and FOH
// discretize its real Schur form instead, whose result has new state coordinates and no state names.
// The model is returned with an error wrapping ErrAccuracyUnverified where
// its DC gain is not within the bar, DCGain fails or is unverified, or the
// split was refused as above. Neither the response away from DC nor the
// state coordinates are certified.
//
// A least-squares fit of a
// descriptor sampled through its pencil is not certified (see C2DFit) and is
// returned with an error wrapping ErrAccuracyUnverified.
//
// Matched (SISO) maps poles and zeros by z = e^{s·dt}, adds zeros at z = −1
// up to relative degree one, and matches the DC gain to DCGain of sys, as
// MATLAB matches the DC gains, through the factors 1 − e^{s·dt} formed
// from the continuous roots (else, where that fails the bar below, those of
// the stored discrete roots, if their DC product rescaled to DCGain is
// within √ε·max(|DCGain|, |product|, Σ|C||X| + |D|) of it). A nil error
// certifies that the DC gain of the result, exact for its stored entries,
// lies within the descriptor bar u·(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + |D|),
// M = A⁻¹, u = 4(n+5)ε (Frobenius norms of the delay-free sys) of that
// DCGain; where A is exactly singular (decided exactly up to order 16) and
// the exact limit is finite, within u·(2|DCGain − D| + |D|), the floor of
// that bar with the group inverse of A in place of A⁻¹. A pole or zero at DC is the
// stored data's exact limit, as DCGain returns it: where DCGain is 0, ±Inf
// or within ε·(|D| + Σ|C||X|) of 0, the exact expansion of the transfer
// function of sys at s = 0 (order up to 16) gives the order k of its lowest
// term; the roots nearest DC are put at z = 1 exactly, |k| of them net (with
// as many more cancelling pairs as the exact multiplicities allow, where
// that brings the roots' leading coefficient within terms), and the gain
// matches the exact coefficient of that term times dt^−k. A nil error then
// certifies that the result's DC gain is exactly that of sys: at a pole,
// its own exact expansion at z = 1 has order k and a coefficient of that
// sign; at a zero, its DC gain is 0 or within the bar of 0. A root near DC
// but not at it is mapped like any other. Neither the response away from
// DC (the leading coefficient included) nor the root locations are
// certified. The model is returned with an error wrapping
// ErrAccuracyUnverified when DCGain fails or is unverified, or A is
// singular beyond order 16; when the expansion is unavailable or the roots
// cannot hold the order; when the DC product (leading coefficient) of the
// roots mapped is not within √ε·max(|DCGain|, |product|, Σ|C||X| + |D|)
// (√ε·max(|coefficient|, |product|)) of it; or when the DC gain of the
// result is not within the bar (not that of sys). That model is the best
// available but can be poor.
//
// Every other error comes with a nil model.
func (sys *System) C2D(dt float64, opts C2DOptions) (*System, error) {
	if err := requireFiniteSystem("C2D", sys); err != nil {
		return nil, err
	}
	return c2dExit(sys.c2d(dt, opts))
}

// c2dExit maps the result of c2d to that of C2D: errPencilFitUncertified,
// which comes with the fit, becomes ErrAccuracyUnverified.
func c2dExit(out *System, err error) (*System, error) {
	if carriesModel(err) {
		return out, fmt.Errorf("C2D: %v: %w", err, ErrAccuracyUnverified)
	}
	if err != nil {
		return nil, fmt.Errorf("C2D: %w", err)
	}
	return out, nil
}

// c2d is the exit shared by C2D, C2DMap, D2D and MakeWeight. A least-squares
// fit through a descriptor pencil returns errPencilFitUncertified with the
// model; only C2D (least squares) reaches it.
func (sys *System) c2d(dt float64, opts C2DOptions) (*System, error) {
	return sys.c2dRecorded(dt, opts, nil)
}

// c2dRecorded is c2d recording, in a non-nil rec, the coordinates the
// result was discretized in (c2dCoords), for C2DMap.
func (sys *System) c2dRecorded(dt float64, opts C2DOptions, rec *c2dCoords) (*System, error) {
	out, err := sys.c2dUnchecked(dt, opts, true, rec)
	if err != nil && !carriesModel(err) {
		return nil, err
	}
	if err := finiteSystemMatrices("result", out, ErrOverflow); err != nil {
		return nil, err
	}
	return out, err
}

// c2dUnguarded is C2D without the check of its result, for Step and Lsim:
// a diverging plant's non-finite samples are reported by their exit check
// (boundedTimeResponse) as ErrOverflow of the response, not of a model.
func (sys *System) c2dUnguarded(dt float64, opts C2DOptions) (*System, error) {
	if err := requireFiniteSystem("C2D", sys); err != nil {
		return nil, err
	}
	return c2dExit(sys.c2dUnchecked(dt, opts, false, nil))
}

// stiffSplitLimit is the ζ″ limit (stiffSplit.within) at which C2D with
// method takes the stiff split's parts: stiffHoldZeta for ZOH and FOH from
// C2D, C2DMap and D2D (public), else 1. Step and Lsim discretize through
// c2dUnguarded and keep 1.
func stiffSplitLimit(method C2DMethod, public bool) float64 {
	if public && (method == "" || method == C2DMethodZOH || method == C2DMethodFOH) {
		return stiffHoldZeta
	}
	return 1
}

// c2dUnchecked is C2D without the check of its result; public reports a
// call from C2D, C2DMap or D2D (stiffSplitLimit). A non-nil rec records
// the coordinates of the result (c2dCoords); the result is the same.
func (sys *System) c2dUnchecked(dt float64, opts C2DOptions, public bool, rec *c2dCoords) (*System, error) {
	if public && holdGated(sys, opts.Method) {
		return sys.c2dDescriptorGated(dt, opts, rec)
	}
	if opts.Method != C2DMethodLeastSquares && sys.IsContinuous() {
		limit := stiffSplitLimit(opts.Method, public)
		route := foldInaccuratePartsRiders
		if opts.Method == C2DMethodMatched {
			route = stiffDescriptorPartsRiders
		}
		if sp, ok := route(sys, rec.riders(sys)); ok {
			if err := sp.within(limit); err != nil {
				return nil, err
			}
			normalized, err := normalizeC2DOptions(dt, opts)
			if err != nil {
				return nil, err
			}
			rec.setParts(sp)
			return discretizeStiffDescriptor(sys, sp.parts, dt, normalized)
		}
		px, ok, err := stiffDescriptorExplicit(sys, opts.Method, limit, rec)
		if err != nil {
			return nil, err
		}
		if ok {
			sys = px
		}
	}
	if !sys.HasDelay() && !sys.IsDescriptor() && opts.Method != C2DMethodLeastSquares {
		normalized, err := normalizeC2DOptions(dt, opts)
		if err != nil {
			return nil, err
		}
		switch normalized.Method {
		case C2DMethodZOH:
			return sys.discretizeZOH(dt)
		case C2DMethodTustin:
			return sys.discretizeTustin(dt, normalized.PrewarpFrequency)
		case C2DMethodFOH:
			return sys.discretizeModifiedFOH(dt)
		case C2DMethodImpulse:
			if sys.IsDiscrete() {
				return nil, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
			}
			return sys.discretizeImpulseParity(dt)
		case C2DMethodMatched:
			return sys.discretizeMatched(dt, sys)
		}
	}
	plan, err := newC2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	plan.rec = rec
	return plan.run()
}

// c2dDescriptorGated is c2dUnchecked of a public C2D that holdGate judges.
// A delay-free sys takes its stiffDescriptorParts at the stiffPencilSplit
// gate (cond(E) 1e4): the LU fold there is off by cond(E)·ε, beyond the
// descriptor bar, and no hold of the fold recovers it. A split refused at
// an LU condition estimate below foldAccurateECond takes the fold, always
// unverified (its DC can pass while the response off DC is O(1) wrong);
// above it the error stays. Every result is gated (holdGate); a dense
// standard form failing the gate is discretized again in real Schur
// coordinates (c2dPlan.discretizeMethod).
func (sys *System) c2dDescriptorGated(dt float64, opts C2DOptions, rec *c2dCoords) (*System, error) {
	gate := newHoldGate(sys)
	limit := stiffSplitLimit(opts.Method, true)
	if !sys.HasDelay() {
		if sp, ok := stiffDescriptorPartsRiders(sys, rec.riders(sys)); ok {
			err := sp.within(limit)
			switch {
			case err == nil:
				normalized, err := normalizeC2DOptions(dt, opts)
				if err != nil {
					return nil, err
				}
				out, err := discretizeStiffDescriptor(sys, sp.parts, dt, normalized)
				if err != nil {
					return nil, err
				}
				rec.setParts(sp)
				return out, gate.check(out)
			case sp.eCond >= foldAccurateECond:
				return nil, err
			}
			gate.refused = err
		}
	} else {
		px, ok, err := stiffDescriptorExplicit(sys, opts.Method, limit, rec)
		if err != nil {
			return nil, err
		}
		if ok {
			sys = px
		}
	}
	plan, err := newC2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	plan.gate = gate
	plan.rec = rec
	return plan.run()
}

func (sys *System) discretizeTustin(dt, prewarp float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
	}
	beta, err := tustinBeta(dt, prewarp)
	if err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: prewarp})
	}
	out, err := bilinear(sys, -beta, -1, 1, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = dt
	policy := newDelayConversionPolicy(dt)
	if out, err = policy.applyDiscreteDelayFields(out, sys); err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	return out, nil
}

func (sys *System) undiscretizeTustin(prewarp float64) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("system already continuous: %w", ErrWrongDomain)
	}
	beta, err := tustinBeta(sys.Dt, prewarp)
	if err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return undiscretizeInternalTustin(sys, beta)
	}
	out, err := bilinear(sys, 1, beta, 1, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = 0
	policy := newDelayConversionPolicy(sys.Dt)
	policy.applyContinuousDelayFields(out, sys)
	propagateNames(out, sys)
	return out, nil
}

// bilinear maps (A, B, C, D) to (pbeta·I − 2αβ·W, scaleB·W·B, scaleC·C·W,
// D − C·W·B), W = (A + palpha·I)⁻¹, in the integratorOrder of A.
func bilinear(sys *System, palpha, pbeta, alpha, beta float64) (*System, error) {
	pole := 0.0
	if sys.IsDiscrete() {
		pole = 1
	}
	scaleB, scaleC := beta, 2.0
	if palpha < 0 {
		scaleB, scaleC = -2, -beta
	}
	image := pbeta - 2*alpha*beta/(pole+palpha)
	return integratorBilinear(sys, pole, palpha, image, scaleB, scaleC, func(sys *System) (*System, error) {
		return bilinearInverse(sys, palpha, pbeta, alpha, beta)
	})
}

func bilinearInverse(sys *System, palpha, pbeta, alpha, beta float64) (*System, error) {
	n, m, p := sys.Dims()

	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A: newDense(0, 0),
			B: denseCopy(sys.B),
			C: denseCopy(sys.C),
			D: D,
		}
		propagateNames(out, sys)
		return out, nil
	}

	A := mat.NewDense(n, n, nil)
	A.Copy(sys.A)

	B := denseCopy(sys.B)
	C := denseCopy(sys.C)

	aRaw := A.RawMatrix()
	for i := range n {
		aRaw.Data[i*aRaw.Stride+i] += palpha
	}

	var lu mat.LU
	lu.Factorize(A)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("bilinear: (palpha·I + A) is singular: %w", ErrSingularTransform)
	}

	if m > 0 {
		err := lu.SolveTo(B, false, B)
		if err != nil {
			return nil, fmt.Errorf("bilinear: LU solve for B failed: %w", ErrSingularTransform)
		}
	}

	if p > 0 && m > 0 {
		D.Mul(C, B)
		D.Scale(-1, D)
		D.Add(D, sys.D)
	}

	twoAB := 2.0 * alpha * beta
	if math.IsInf(twoAB, 0) {
		return nil, fmt.Errorf("bilinear: 2*alpha*beta overflows: %w", ErrOverflow)
	}

	scaleB, scaleC := beta, 2.0
	if palpha < 0 {
		scaleB, scaleC = -2, -beta
	}
	if m > 0 {
		B.Scale(scaleB, B)
	}

	Ainv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	eyeRaw := eye.RawMatrix()
	for i := range n {
		eyeRaw.Data[i*eyeRaw.Stride+i] = 1
	}
	err := lu.SolveTo(Ainv, false, eye)
	if err != nil {
		return nil, fmt.Errorf("bilinear: LU inverse failed: %w", ErrSingularTransform)
	}

	if p > 0 {
		C.Mul(C, Ainv)
		C.Scale(scaleC, C)
	}
	Ainv.Scale(-twoAB, Ainv)
	ainvRaw := Ainv.RawMatrix()
	for i := range n {
		ainvRaw.Data[i*ainvRaw.Stride+i] += pbeta
	}

	if math.IsInf(denseNorm(Ainv), 0) || math.IsNaN(denseNorm(Ainv)) {
		return nil, fmt.Errorf("bilinear: result contains Inf/NaN: %w", ErrOverflow)
	}

	out := &System{A: Ainv, B: B, C: C, D: D}
	propagateNames(out, sys)
	return out, nil
}

// C2DOptions mirrors MATLAB c2dOptions; see
// https://www.mathworks.com/help/control/ref/c2doptions.html. Zero values
// select the MATLAB defaults: Method "" is zero-order hold, ThiranOrder 0
// rounds fractional delays to the nearest sample (tustin and matched only),
// DelayModeling "" models extra delays as internal delays, PrewarpFrequency 0
// (rad/s, tustin only, below Nyquist) disables prewarping, and FitOrder 0
// ("auto") fits least-squares models of the source order n. Above the
// minimal order the fit keeps a redundant near-cancelling pole-zero pair
// whose pole is not imposed stable and can land on or beyond z = −1, as
// MATLAB's can; the MATLAB remedy, reducing FitOrder (or the model, with
// minreal), applies to such unstable poles and z = −1 cancellations.
type C2DOptions struct {
	Method           C2DMethod
	ThiranOrder      int
	DelayModeling    C2DDelayModeling
	PrewarpFrequency float64
	FitOrder         int
}

// C2DMethod names a continuous/discrete conversion method, as the MATLAB
// c2d, d2c and d2d method strings. The empty method selects C2DMethodZOH.
type C2DMethod string

const (
	// C2DMethodZOH assumes piecewise-constant inputs (zero-order hold).
	C2DMethodZOH C2DMethod = "zoh"
	// C2DMethodTustin is the bilinear approximation, optionally prewarped.
	C2DMethodTustin C2DMethod = "tustin"
	// C2DMethodFOH assumes piecewise-linear inputs (triangle approximation).
	C2DMethodFOH C2DMethod = "foh"
	// C2DMethodImpulse is impulse-invariant discretization (c2d only).
	C2DMethodImpulse C2DMethod = "impulse"
	// C2DMethodMatched is zero-pole matching (SISO only).
	C2DMethodMatched C2DMethod = "matched"
	// C2DMethodLeastSquares fits the frequency response up to Nyquist (c2d,
	// SISO only). C2D returns the fit as MATLAB does; C2DFit also reports
	// its quality. A fit through a descriptor pencil returns
	// ErrAccuracyUnverified with the model from both.
	C2DMethodLeastSquares C2DMethod = "least-squares"
)

// C2DDelayModeling selects how C2D models delays that become extra
// discrete delays, as the MATLAB c2dOptions DelayModeling option. The empty
// value selects C2DDelayModelingInternal, the MATLAB default.
type C2DDelayModeling string

const (
	// C2DDelayModelingState models extra delays as additional states.
	C2DDelayModelingState C2DDelayModeling = "state"
	// C2DDelayModelingInternal models extra delays as internal delays.
	C2DDelayModelingInternal C2DDelayModeling = "delay"
)

func mergeDelays(existing, decomposed []float64) []float64 {
	if decomposed == nil {
		return existing
	}
	allZero := true
	for _, v := range decomposed {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return existing
	}
	n := len(decomposed)
	out := make([]float64, n)
	copy(out, decomposed)
	for i := range existing {
		if i < n {
			out[i] += existing[i]
		}
	}
	return out
}

func convertSliceDelayToDiscrete(delay []float64, dt float64, thiranOrder int) ([]float64, error) {
	if delay == nil {
		return nil, nil
	}
	out := make([]float64, len(delay))
	for i, tau := range delay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		rounded := math.Round(samples)
		if math.Abs(samples-rounded) < 1e-9 {
			out[i] = rounded
		} else if thiranOrder > 0 {
			out[i] = 0
		} else {
			return nil, fmt.Errorf("delay[%d]=%g not integer multiple of dt=%g: %w",
				i, tau, dt, ErrFractionalDelay)
		}
	}
	return out, nil
}

func convertSliceDelayToContinuous(delay []float64, dt float64) []float64 {
	if delay == nil {
		return nil
	}
	out := make([]float64, len(delay))
	for i, d := range delay {
		out[i] = d * dt
	}
	return out
}

func (sys *System) discretizeZOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
	}
	if err := validateConversionSampleTime(dt); err != nil {
		return nil, err
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodZOH})
	}

	n, m, _ := sys.Dims()

	D := denseCopy(sys.D)
	C := denseCopy(sys.C)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: dt,
		}
		propagateNames(out, sys)
		return newDelayConversionPolicy(dt).applyDiscreteDelayFields(out, sys)
	}

	if m == 0 {
		Ad, ok := stiffExp(sys.A, dt)
		if !ok {
			Ad = expmScaled(sys.A, dt)
		}
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		propagateNames(out, sys)
		return newDelayConversionPolicy(dt).applyDiscreteDelayFields(out, sys)
	}

	Ad, gammas, ok := stiffModalHold(sys.A, sys.B, dt, zohHoldLeaf)
	var Bd *mat.Dense
	if ok {
		Bd = gammas[0]
	} else {
		Ad, Bd = zohAugmented(sys.A, sys.B, dt)
	}
	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	policy := newDelayConversionPolicy(dt)
	out, err := policy.applyDiscreteDelayFields(out, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	return out, nil
}

// zohAugmented returns e^{a·dt} and ∫₀^dt e^{a·s} ds·b from one exponential
// of [a b; 0 0]·dt, in the integratorOrder of a.
func zohAugmented(a, b *mat.Dense, dt float64) (ad, bd *mat.Dense) {
	ad, gammas := integratorHold(a, b, dt, func(a, b *mat.Dense) (*mat.Dense, []*mat.Dense) {
		ad, bd := zohAugmentedExp(a, b, dt)
		return ad, []*mat.Dense{bd}
	})
	return ad, gammas[0]
}

func zohAugmentedExp(a, b *mat.Dense, dt float64) (ad, bd *mat.Dense) {
	n, _ := a.Dims()
	_, m := b.Dims()
	nm := n + m
	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := a.RawMatrix()
	bRaw := b.RawMatrix()
	for i := range n {
		mRow := mRaw.Data[i*mRaw.Stride:]
		aRow := aRaw.Data[i*aRaw.Stride : i*aRaw.Stride+n]
		for j, v := range aRow {
			mRow[j] = v * dt
		}
		bRow := bRaw.Data[i*bRaw.Stride : i*bRaw.Stride+m]
		for j, v := range bRow {
			mRow[n+j] = v * dt
		}
	}

	var eM mat.Dense
	expGuarded(&eM, M)

	emRaw := eM.RawMatrix()
	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	for i := range n {
		copy(adData[i*n:], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		copy(bdData[i*m:], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
	}
	return mat.NewDense(n, n, adData), mat.NewDense(n, m, bdData)
}

// stiffDescriptorParts returns, for a delay-free descriptor sys with
// full-rank ill-conditioned E, the explicit parts of stiffExplicitParts,
// whose sum has the transfer function and the poles of the pencil (A, E).
// The LU fold E⁻¹A (ToExplicit) of such an E moves the slow poles by
// cond(E)·ε, up to a stability flip, and one matrix exponential of the
// stiff pole beside them loses the slow dynamics to the squarings. ok is
// false when the fold is accurate, E is rank deficient, or the generalized
// Schur form zeroed the stiff part's E (deflated). A split that is not
// certified is ok with sp.err set (stiffSplit): the caller returns that
// error, never the parts.
func stiffDescriptorParts(sys *System) (stiffSplit, bool) {
	return stiffDescriptorPartsRiders(sys, nil)
}

// stiffDescriptorPartsRiders is stiffDescriptorParts with the riders of
// stiffExplicitSplitRiders.
func stiffDescriptorPartsRiders(sys *System, riders *mat.Dense) (stiffSplit, bool) {
	if !sys.IsDescriptor() || sys.HasDelay() {
		return stiffSplit{}, false
	}
	sp, ok := stiffExplicitSplitRiders(sys, riders)
	if !ok || sp.deflated {
		return stiffSplit{}, false
	}
	return sp, true
}

// foldAccurateECond is the 1-norm condition estimate of E below which C2D
// impulse, C2D with I/O delays, Step and Lsim with I/O delays, and the
// free response of Initial and Lsim take the LU fold E⁻¹A (ToExplicit) of
// a full-rank E even when stiffDescriptorParts applies: its cond(E)·ε
// error then stays under 1e-9 relative in the discrete response and 1e-7
// in the step response, where the pencil parts cost a generalized Schur
// split and several matrix exponentials. Delay-free ZOH, FOH and Tustin
// C2D (c2dDescriptorGated), the delay-free zero-state Step and Lsim
// (zeroStateRoute) and Impulse take the parts from the stiffPencilSplit
// gate (cond(E) 1e4) instead; a split those refuse below foldAccurateECond
// returns the fold unverified, above it the error.
const foldAccurateECond = 1e6

// foldInaccurateParts is stiffDescriptorParts for an E whose LU condition
// estimate is foldAccurateECond or more.
func foldInaccurateParts(sys *System) (stiffSplit, bool) {
	return foldInaccuratePartsRiders(sys, nil)
}

// foldInaccuratePartsRiders is foldInaccurateParts with the riders of
// stiffExplicitSplitRiders.
func foldInaccuratePartsRiders(sys *System, riders *mat.Dense) (stiffSplit, bool) {
	if !sys.IsDescriptor() {
		return stiffSplit{}, false
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if lu.Cond() < foldAccurateECond {
		return stiffSplit{}, false
	}
	return stiffDescriptorPartsRiders(sys, riders)
}

// splitFastPencil returns explicit parts of the fast descriptor part of
// stiffPencilSplit, split again by the ordered generalized Schur form at the
// stiffCut of the moduli of its fold, nominally |λ| < qzLargeRatio·ρ, into
// the modes below the cut and the rest, recursively. Folded whole by
// svdExplicit, a fast part with two stiff singular values of E keeps its
// slower stiff pole only to ε·‖E⁻¹A‖ (1e-6 relative at |λ| ~ 1e4 beside
// 1e15), while the generalized Schur form resolves it to ε relative and the
// triangular fold of the leading block keeps it. zeta is the largest ζ″ of
// the certified splits. A split with modes on both sides of its cut that
// fails or is not certified returns err (errStiffInseparable) with the
// fold, which only TransferFunction, judging the sum, takes. ok is false
// when the fold is exactly singular (deflated).
func splitFastPencil(fast *System) (parts []*System, zeta float64, err error, ok bool) {
	parts, zeta, err, ok, _ = splitFastPencilRiders(fast, nil)
	return
}

// splitFastPencilRiders is splitFastPencil also taking riders (n×r B-like
// columns of fast, nil for none) through each part's transformations, one
// per part.
func splitFastPencilRiders(fast *System, riders *mat.Dense) (parts []*System, zeta float64, err error, ok bool, pr []*mat.Dense) {
	folded, fr, ok := svdExplicitRiders(fast, riders)
	if !ok {
		return nil, 0, nil, false, nil
	}
	whole := func(err error) ([]*System, float64, error, bool, []*mat.Dense) {
		if riders == nil {
			return []*System{folded}, 0, err, true, nil
		}
		return []*System{folded}, 0, err, true, []*mat.Dense{fr}
	}
	n, _, _ := fast.Dims()
	if n == 1 {
		return whole(nil)
	}
	var eig mat.Eigen
	if !eig.Factorize(folded.A, mat.EigenNone) {
		return whole(nil)
	}
	rho := 0.0
	mods := make([]float64, 0, n)
	for _, l := range eig.Values(nil) {
		rho = math.Max(rho, cmplx.Abs(l))
		mods = append(mods, cmplx.Abs(l))
	}
	cut := stiffCut(mods, qzLargeRatio*rho)
	slow, rest, zeta, rs, rr, err := decomposeGeneralizedQZZetaRiders(fast, func(l complex128) bool {
		return cmplx.Abs(l) < cut
	}, nil, riders)
	if err != nil {
		if !splitsBothSides(mods, cut) {
			return whole(nil)
		}
		return whole(errStiffInseparable())
	}
	n1, _, _ := slow.Dims()
	n2, _, _ := rest.Dims()
	if n1 == 0 || n2 == 0 {
		return whole(nil)
	}
	restParts, rz, rerr, ok, restRiders := splitFastPencilRiders(rest, rr)
	if !ok {
		return whole(nil)
	}
	if riders != nil {
		pr = append([]*mat.Dense{rs}, restRiders...)
	}
	return append([]*System{slow}, restParts...), math.Max(zeta, rz), rerr, true, pr
}

// discretizeStiffDescriptor discretizes the descriptor sys from its
// stiffDescriptorParts. ZOH, FOH, impulse and Tustin are linear in the
// model, so each part is discretized on its own and the results are
// connected in parallel, with the metadata of sys; the state
// coordinates are those of the parts, so state names are dropped. Matched
// maps the poles and zeros of the transfer function of the pencil.
func discretizeStiffDescriptor(sys *System, parts []*System, dt float64, opts C2DOptions) (*System, error) {
	if opts.Method == C2DMethodMatched {
		return sys.discretizeMatched(dt, sys)
	}
	disc := make([]*System, len(parts))
	for i, part := range parts {
		var d *System
		var err error
		switch opts.Method {
		case C2DMethodTustin:
			d, err = part.discretizeTustin(dt, opts.PrewarpFrequency)
		case C2DMethodFOH:
			d, err = part.discretizeModifiedFOH(dt)
		case C2DMethodImpulse:
			d, err = part.discretizeImpulseParity(dt)
		default:
			d, err = part.discretizeZOH(dt)
		}
		if err != nil {
			return nil, err
		}
		disc[i] = d
	}
	out, err := parallelExplicit(disc)
	if err != nil {
		return nil, err
	}
	out.Dt = dt
	propagateNames(out, sys)
	out.StateName = nil
	return out, nil
}

// stiffDescriptorExplicit returns, for a continuous descriptor with I/O
// delays and no internal ones (with internal ones, stiffLFTExplicit), the
// parallel connection of the
// foldInaccurateParts of its delay-free core, each in real Schur form, with
// the delays and names of sys and no state names: an explicit model with
// the transfer function and poles of the pencil for the delay paths of
// C2D (and so of Step and Lsim, which discretize through it), which would
// otherwise fold E⁻¹A. Upper quasi-triangular, it lets
// the hold discretize its stiff and slow modes separately
// (stiffModalHold). Matched is left to newC2DPlan, which keeps the
// transfer function of the pencil. A split not certified within limit
// (stiffSplit.within) returns its error. A non-nil rec records the map
// of the descriptor state into the state of the result.
func stiffDescriptorExplicit(sys *System, method C2DMethod, limit float64, rec *c2dCoords) (*System, bool, error) {
	if method == C2DMethodMatched || !sys.IsDescriptor() || !sys.HasDelay() {
		return nil, false, nil
	}
	if sys.LFT != nil {
		if method == C2DMethodImpulse {
			return nil, false, nil
		}
		return stiffLFTExplicitWithin(sys, limit, rec)
	}
	sp, ok := foldInaccuratePartsRiders(ioDelayFreeCore(sys), rec.riders(sys))
	if !ok {
		return nil, false, nil
	}
	if err := sp.within(limit); err != nil {
		return nil, false, err
	}
	parts := slices.Clone(sp.parts)
	zs := make([]*mat.Dense, len(parts))
	for i, part := range parts {
		if parts[i], zs[i], ok = realSchurPartZ(part); !ok {
			return nil, false, nil
		}
	}
	out, err := parallelExplicit(parts)
	if err != nil {
		return nil, false, nil
	}
	propagateNames(out, sys)
	out.StateName = nil
	delays := sys.Copy()
	out.Delay, out.InputDelay, out.OutputDelay = delays.Delay, delays.InputDelay, delays.OutputDelay
	rec.setSchurParts(sp, zs)
	return out, true, nil
}

// ioDelayFreeCore returns sys without its I/O delays (Delay, InputDelay,
// OutputDelay) when it has no LFT; otherwise sys itself.
func ioDelayFreeCore(sys *System) *System {
	if sys.LFT != nil {
		return sys
	}
	core := *sys
	core.Delay, core.InputDelay, core.OutputDelay = nil, nil, nil
	return &core
}

// realSchurPart returns the explicit part, if its A is not already upper
// quasi-triangular, in the coordinates of the real Schur form of A.
func realSchurPart(part *System) (*System, bool) {
	out, _, ok := realSchurPartZ(part)
	return out, ok
}

// realSchurPartZ is realSchurPart also returning the Schur vectors Z (the
// part's state is Z times the result's); nil when part is returned as is.
func realSchurPartZ(part *System) (*System, *mat.Dense, bool) {
	if upperQuasiTriangular(part.A) {
		return part, nil, true
	}
	n, m, p := part.Dims()
	t, z, ok := realSchur(part.A)
	if !ok {
		return nil, nil, false
	}
	Z := mat.NewDense(n, n, z)
	out := part.Copy()
	out.A = mat.NewDense(n, n, t)
	if m > 0 {
		out.B.Mul(Z.T(), part.B)
	}
	if p > 0 {
		out.C.Mul(part.C, Z)
	}
	out.StateName = nil
	return out, Z, true
}

// parallelExplicit returns the explicit parallel connection of parts with
// equal input and output counts: block-diagonal A, stacked B, adjacent C
// columns and summed D.
func parallelExplicit(parts []*System) (*System, error) {
	_, m, p := parts[0].Dims()
	n := 0
	for _, part := range parts {
		k, _, _ := part.Dims()
		n += k
	}
	A, B, C, D := newDense(n, n), newDense(n, m), newDense(p, n), newDense(p, m)
	off := 0
	for _, part := range parts {
		k, _, _ := part.Dims()
		if k > 0 {
			A.Slice(off, off+k, off, off+k).(*mat.Dense).Copy(part.A)
			if m > 0 {
				B.Slice(off, off+k, 0, m).(*mat.Dense).Copy(part.B)
			}
			if p > 0 {
				C.Slice(0, p, off, off+k).(*mat.Dense).Copy(part.C)
			}
		}
		if p > 0 && m > 0 {
			D.Add(D, part.D)
		}
		off += k
	}
	return newNoCopy(A, B, C, D, parts[0].Dt)
}

func discretizeWithInternalDelay(sys *System, dt float64, opts C2DOptions) (*System, error) {
	return discretizeInternalModel(sys, dt, opts)
}

func isStrictlyUpperTriangular(m *mat.Dense) bool {
	if m == nil {
		return true
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		bound := min(raw.Cols-1, i)
		row := raw.Data[i*raw.Stride:]
		for j := 0; j <= bound; j++ {
			if row[j] != 0 {
				return false
			}
		}
	}
	return true
}

// D2COptions mirrors MATLAB d2cOptions. The zero Method selects zero-order
// hold, the MATLAB default; PrewarpFrequency is in rad/s, applies only to
// Tustin, and must be below Nyquist.
type D2COptions struct {
	Method           C2DMethod
	PrewarpFrequency float64
}

// D2C converts a discrete-time model to continuous time, as MATLAB
// sysc = d2c(sysd,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.d2c.html.
// Methods are zoh (the zero-value default), tustin, foh and matched.
// Delays of k samples become delays of k·Ts seconds; internal delays stay
// internal, inverting C2D, except that matched rejects them as MATLAB does.
// Matched maps poles and zeros by s = log(z)/Ts and matches the DC gain to
// DCGain of sys at z = 1 under the contract of C2D's matched method, with
// M = (I − A)⁻¹ and the pencil term ‖A‖ + ‖E‖ (‖E‖ = √n for E = I) in the
// bar; the model is returned with an error wrapping ErrAccuracyUnverified
// in the cases C2D lists. An already continuous model returns
// ErrWrongDomain; a result with an Inf or NaN entry in any matrix returns
// ErrOverflow. Every other error comes with a nil model.
func (sys *System) D2C(opts D2COptions) (*System, error) {
	if err := requireFiniteSystem("D2C", sys); err != nil {
		return nil, err
	}
	out, err := sys.d2c(opts)
	if carriesModel(err) {
		return out, fmt.Errorf("D2C: %v: %w", err, ErrAccuracyUnverified)
	}
	if err != nil {
		return nil, fmt.Errorf("D2C: %w", err)
	}
	return out, nil
}

// d2c is the exit shared by D2C, D2CMap and D2D.
func (sys *System) d2c(opts D2COptions) (*System, error) {
	plan, err := newD2CPlan(sys, opts)
	if err != nil {
		return nil, err
	}
	out, err := plan.run()
	if err != nil && !carriesModel(err) {
		return nil, err
	}
	if err := finiteSystemMatrices("result", out, ErrOverflow); err != nil {
		return nil, err
	}
	return out, err
}

func (sys *System) d2cZOH() (*System, error) {
	return sys.d2cZOHRobust()
}

func d2cPropagateDelays(out, sys *System, dt float64) {
	policy := newDelayConversionPolicy(dt)
	policy.applyContinuousDelayFields(out, sys)
}

// D2DOptions mirrors MATLAB d2dOptions. The zero Method selects zero-order
// hold, the MATLAB default; PrewarpFrequency is in rad/s and applies only to
// Tustin.
type D2DOptions struct {
	Method           C2DMethod
	PrewarpFrequency float64
}

// D2D resamples a discrete-time model to sample time dt, as MATLAB
// sys1 = d2d(sys,Ts,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.d2d.html. It
// converts with D2C and back with C2D using opts.Method, which must be zoh
// (the zero-value default) or tustin; a same-rate request returns a copy.
func (sys *System) D2D(dt float64, opts D2DOptions) (*System, error) {
	if err := requireFiniteSystem("D2D", sys); err != nil {
		return nil, err
	}
	plan, err := newD2DPlan(sys, dt, C2DOptions{Method: opts.Method, PrewarpFrequency: opts.PrewarpFrequency})
	if err == nil {
		var out *System
		if out, err = plan.run(); err == nil {
			return out, nil
		}
	}
	return nil, fmt.Errorf("D2D: %w", err)
}
