package controlsys

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// HinfSynResult is an H∞ controller, as MATLAB [K,CL,gamma,info] =
// hinfsyn(P,nmeas,ncont) returns it (see
// https://www.mathworks.com/help/robust/ref/dynamicsystem.hinfsyn.html):
// X and Y are the state-feedback and filter Riccati solutions at the gamma
// K is built for (at most GammaOpt) and CLPoles the closed-loop poles. For a
// discrete plant X and Y solve the Riccati equations of its
// Tustin-equivalent continuous plant (in the same state coordinates), not
// discrete Riccati equations; when HinfSyn first shifts modes away from
// z = ±1 (see HinfSyn), that of the shifted plant.
type HinfSynResult struct {
	K *System
	// CL is the closed loop of P with K from w to z, LFT(P, K), as MATLAB CL.
	CL *System
	// Gamma is ‖CL‖∞ ≤ GammaOpt, the achieved norm (MATLAB gamma).
	Gamma float64
	// PeakFrequency is where ‖CL‖∞ is attained, in rad/s, as HinfNorm
	// returns it: σ_max(CL) there is within HinfNorm's 1e-10 relative
	// accuracy below Gamma. It is +Inf when a continuous CL only approaches
	// its peak σ_max(D) as ω → ∞, the Nyquist frequency π/Ts playing that
	// role when discrete.
	PeakFrequency float64
	// GammaOpt is the gamma the returned K is verified to meet, Gamma ≤
	// GammaOpt: the larger of the gamma K is built for and Gamma. It is
	// within hinfControllerBackoff (relative) of the smallest achievable
	// gamma unless the central controller is ill-conditioned there, as at an
	// unattained infimum of 0 or a coupling-limited optimum (ρ(XY) = γ²);
	// then it is the smallest such bound among the backed-off controllers.
	GammaOpt float64
	X        *mat.Dense
	Y        *mat.Dense
	CLPoles  []complex128
}

// HinfSyn computes a suboptimal H-infinity output-feedback controller for the
// generalized plant P whose last nmeas outputs are measurements and last
// ncont inputs are controls, bisecting to the smallest achievable gamma, as
// MATLAB hinfsyn(P,nmeas,ncont); see
// https://www.mathworks.com/help/robust/ref/dynamicsystem.hinfsyn.html.
// A nonzero D11 uses the Glover-Doyle general formulas, whose central
// controller may have feedthrough. A nonzero D22 is handled by a loop shift:
// K is designed for D22 = 0 and returned as K0 (I + D22 K0)^-1, giving the
// same closed loop and gamma, as MATLAB's loop-shifting Riccati method does.
// When that shift is ill-posed for the central controller (I + D22·Dk
// singular, e.g. the square one-block problem Mixsyn(G, W1, nil, nil) with a
// biproper G), K0 is the non-central controller of the Youla parameter
// Q = const with ‖Q‖ = γ/2, which also meets gamma.
//
// D12 must have full column rank and D21 full row rank; otherwise HinfSyn
// returns ErrInvalidPartition (MATLAB regularizes the plant instead).
//
// A discrete P (Ts > 0) is mapped to continuous time by the Tustin
// transform, which preserves the H∞ norm and closed-loop stability exactly;
// K is the continuous design mapped back, with P's sample time, so GammaOpt
// equals the continuous design on P.D2C with Tustin. The rank conditions then
// apply to P12 and P21 at z = -1, the unit-circle point Tustin sends to
// s = ∞, in line with MATLAB's requirement that they have no zeros on the
// unit circle. A plant mode at z = -1 is handled by designing for P(-z) and
// reflecting K back. Modes at (or near) both z = 1 and z = -1 are first moved
// by a static output feedback u = D0·y + v, which leaves the closed loops and
// gamma unchanged; K is the design for the shifted plant plus D0, and the
// rank conditions apply to the shifted plant.
//
// Bisection stops when the bracket is within relative 1e-6 or absolute
// hinfGammaAbsTol, like the RelTol and AbsTol of MATLAB hinfsynOptions (see
// https://www.mathworks.com/help/robust/ref/hinfsynoptions.html) but tighter. As MATLAB
// returns the controller of a passing gamma, HinfSyn builds K just above the
// bisection edge and measures ||T_zw||inf on the returned K. If the build
// fails or K misses that gamma (the central controller is numerically
// unreliable near an unattained infimum such as 0, where K grows without
// bound, or a coupling-limited optimum, where one of its modes does), it
// backs gamma off upward, since every gamma above the infimum is
// achievable, until a controller meets it (see hinfDesign.verified).
//
// The Riccati tests are invariant under scaling z or w: each Hamiltonian is
// balanced by a power-of-two symplectic scaling, the semidefiniteness test
// applies to the balanced solution, the constant terms are formed as Gram
// matrices, and the bisection's give-up bound scales with the plant.
func HinfSyn(P *System, nmeas, ncont int) (*HinfSynResult, error) {
	gp, err := partitionGeneralizedPlant("HinfSyn", P, nmeas, ncont)
	if err != nil {
		return nil, err
	}
	var d hinfDesign
	if P.IsDiscrete() {
		d, err = hinfSynDiscrete(gp, P, nmeas, ncont)
	} else {
		d, err = hinfSynScaled(gp.op, P, nmeas, ncont, "D12", "D21")
	}
	if err != nil {
		return nil, err
	}
	return d.verified(gp.op, P, nmeas, ncont)
}

// hinfDesign is a bisected H∞ problem: gammaEdge is the smallest gamma the
// Riccati test accepts and build returns the controller for P at a gamma.
type hinfDesign struct {
	gammaEdge float64
	build     func(gamma float64) (*HinfSynResult, error)
}

// hinfGammaAbsTol is the absolute bisection tolerance. It ends bisection
// toward a zero infimum, and is far below MATLAB's default AbsTol (1e-6) so
// that the relative 1e-6 governs every gamma above 1e-3.
const hinfGammaAbsTol = 1e-9

// hinfControllerBackoff is the first relative gamma margin above the
// bisection edge, and the resolution to which verified narrows it. At the
// edge X or Y grows without bound (or I − XY/γ² becomes singular), so the
// central controller is ill-conditioned and overshoots gamma; 1e-4
// restores a genuine margin.
const hinfControllerBackoff = 1e-4

// hinfBackoffTries bounds the doublings of the gamma margin in verified.
const hinfBackoffTries = 48

// verified builds the controller at gammaEdge plus a margin that starts at
// hinfControllerBackoff·gammaEdge and doubles (at least to hinfGammaAbsTol)
// until the closed loop of P with the returned K meets the gamma it is built
// for. Doubling alone can land up to twice the margin that is needed, so the
// margin is then bisected between the last miss and the pass down to a
// bracket of hinfControllerBackoff·gammaEdge. Near a coupling-limited optimum
// (ρ(XY) → γ²) the central controller has a mode that grows without bound
// and its rounding leaves ‖T_zw‖∞ a little above the gamma it is built for,
// at random, so meeting it is not monotone in the margin; yet every stable
// closed loop is a verified bound. Every try with a finite norm is therefore
// a candidate, its GammaOpt the larger of its build gamma and its norm, and
// the smallest candidate is returned.
func (d hinfDesign) verified(op string, P *System, nmeas, ncont int) (*HinfSynResult, error) {
	var best *HinfSynResult
	var err error
	try := func(margin float64) bool {
		var res *HinfSynResult
		if res, err = d.build(d.gammaEdge + margin); err == nil {
			err = res.measure(op, P, nmeas, ncont)
		}
		if err != nil {
			return false
		}
		met := res.Gamma <= res.GammaOpt
		res.GammaOpt = math.Max(res.GammaOpt, res.Gamma)
		if best == nil || res.GammaOpt < best.GammaOpt {
			best = res
		}
		return met
	}
	margin := hinfControllerBackoff * d.gammaEdge
	for i := range hinfBackoffTries {
		if try(margin) {
			if i > 0 {
				d.narrow(try, margin/2, margin)
			}
			break
		}
		margin = math.Max(2*margin, hinfGammaAbsTol)
	}
	if best == nil {
		return nil, err
	}
	return best, nil
}

// narrow bisects the margin in (lo, hi], try(hi) having met its gamma.
func (d hinfDesign) narrow(try func(float64) bool, lo, hi float64) {
	tol := math.Max(hinfControllerBackoff*d.gammaEdge, hinfGammaAbsTol)
	for range hinfBackoffTries {
		if !(hi-lo > tol) {
			return
		}
		mid := (lo + hi) / 2
		if try(mid) {
			hi = mid
		} else {
			lo = mid
		}
	}
}

// measure records the closed loop of P with K, its norm and peak frequency;
// an unstable loop (infinite norm) is an error.
func (res *HinfSynResult) measure(op string, P *System, nmeas, ncont int) error {
	cl, err := LFT(P, res.K, LFTFeedback{Nu: ncont, Ny: nmeas})
	if err != nil {
		return fmt.Errorf("%s: closing the loop: %w", op, err)
	}
	norm, omega, err := HinfNorm(cl)
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	if math.IsInf(norm, 1) || math.IsNaN(norm) {
		return fmt.Errorf("%s: controller for γ = %g gives ‖T_zw‖∞ = %g: %w", op, res.GammaOpt, norm, ErrGammaNotAchievable)
	}
	res.CL, res.Gamma, res.PeakFrequency = cl, norm, omega
	return nil
}

func hinfSynPartition(gp *generalizedPlantPartition, d12, d21 string) (hinfDesign, error) {
	if err := gp.requireRegularFeedthrough(d12, d21); err != nil {
		return hinfDesign{}, err
	}
	if err := gp.validateControllerChannels(); err != nil {
		return hinfDesign{}, err
	}
	if !allZeroDense(gp.D11) {
		return hinfSynGeneral(gp)
	}
	ric := newHinfD11ZeroRiccatis(gp)
	gamma, err := hinfBisect(0, gp.hinfGammaScale(), ric.feasible)
	if err != nil {
		return hinfDesign{}, fmt.Errorf("%s: %w", gp.op, err)
	}
	return hinfDesign{gammaEdge: gamma, build: func(g float64) (*HinfSynResult, error) { return hinfSynD11Zero(gp, ric, g) }}, nil
}

// hinfSynScaled designs for the continuous P after balancing its states by
// Ssbal, as MATLAB hinfsyn's default AutoScale does: the Riccati solutions of
// a badly scaled realization (B huge, C tiny, as Tustin makes of a slow
// discrete pole at fast sampling) lose all accuracy. K's transfer function
// does not depend on the plant's state coordinates; X and Y are mapped back
// to those of P.
func hinfSynScaled(op string, P *System, nmeas, ncont int, d12, d21 string) (hinfDesign, error) {
	bal, err := Ssbal(P)
	if err != nil {
		return hinfDesign{}, fmt.Errorf("%s: %w", op, err)
	}
	gp, err := partitionGeneralizedPlant(op, bal.Sys, nmeas, ncont)
	if err != nil {
		return hinfDesign{}, err
	}
	d, err := hinfSynPartition(gp, d12, d21)
	if err != nil {
		return hinfDesign{}, err
	}
	t := make([]float64, gp.n)
	for i := range t {
		t[i] = bal.T.At(i, i)
	}
	build := d.build
	d.build = func(g float64) (*HinfSynResult, error) {
		res, err := build(g)
		if err != nil {
			return nil, err
		}
		unscaleGram(res.X, t, false)
		unscaleGram(res.Y, t, true)
		return res, nil
	}
	return d, nil
}

// unscaleGram maps M of the coordinates x̂ = diag(t)·x back to x:
// M_ij·t_i·t_j for a quadratic-form matrix (X), M_ij/(t_i·t_j) when inverse
// (Y).
func unscaleGram(M *mat.Dense, t []float64, inverse bool) {
	if M == nil {
		return
	}
	raw := M.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j := range row {
			if inverse {
				row[j] /= t[i] * t[j]
			} else {
				row[j] *= t[i] * t[j]
			}
		}
	}
}

// hinfSynDiscrete designs for the discrete plant P through the Tustin map
// z = (β+s)/(β−s), β = 2/Ts, or through z = −(β+s)/(β−s) applied to P(−z)
// when ‖(I + A)⁻¹‖ exceeds ‖(I − A)⁻¹‖ (a mode nearer z = −1, which the
// plain map sends to s = ∞; conditioning cannot tell, a scalar's being 1). Both maps take the unit circle onto
// the imaginary axis and the open unit disk onto the open left half-plane.
// When both I + A and I − A are ill-conditioned (modes at or near both
// z = 1 and z = −1, or a strongly non-normal A) it first closes a static output feedback u = D0·y + v,
// which moves those modes, designs K' from y to v, and returns K = K' + D0:
// the closed loops, and so the achievable γ, are identical.
func hinfSynDiscrete(gp *generalizedPlantPartition, P *System, nmeas, ncont int) (hinfDesign, error) {
	op := gp.op
	if cond := tustinCond(P.A); cond > hinfTustinCond {
		if D0, Ps, ok := gp.unitCircleModeShift(P, cond); ok {
			d, err := hinfSynTustin(op, Ps, nmeas, ncont)
			if err != nil {
				return hinfDesign{}, err
			}
			build := d.build
			d.build = func(g float64) (*HinfSynResult, error) {
				res, err := build(g)
				if err != nil {
					return nil, err
				}
				res.K.D.Add(res.K.D, D0)
				return res, nil
			}
			return d, nil
		}
	}
	return hinfSynTustin(op, P, nmeas, ncont)
}

// hinfTustinCond is the tustinCond above which hinfSynDiscrete shifts the
// plant's modes away from z = ±1 before the Tustin map. Beyond it the
// Tustin-equivalent plant is stiff enough to cost the continuous Riccati
// solutions accuracy.
const hinfTustinCond = 1e2

// tustinCond is the condition number of I + A or I − A, whichever is better
// conditioned: that of the better of the two Tustin maps.
func tustinCond(A *mat.Dense) float64 {
	n, _ := A.Dims()
	var plus, minus mat.LU
	IpA, ImA := eyeDense(n), eyeDense(n)
	IpA.Add(IpA, A)
	ImA.Sub(ImA, A)
	plus.Factorize(IpA)
	minus.Factorize(ImA)
	return math.Min(plus.Cond(), minus.Cond())
}

func hinfSynTustin(op string, P *System, nmeas, ncont int) (hinfDesign, error) {
	n, _, _ := P.Dims()
	var plus, minus mat.LU
	IpA, ImA := eyeDense(n), eyeDense(n)
	IpA.Add(IpA, P.A)
	ImA.Sub(ImA, P.A)
	plus.Factorize(IpA)
	minus.Factorize(ImA)
	reflect := plus.Cond()/mat.Norm(IpA, 1) > minus.Cond()/mat.Norm(ImA, 1)
	Pd, point := P, "-1"
	if reflect {
		Pd = P.Copy()
		Pd.A.Scale(-1, Pd.A)
		Pd.B.Scale(-1, Pd.B)
		point = "1"
	}
	Pc, err := Pd.undiscretizeTustin(0)
	if errors.Is(err, ErrSingularTransform) {
		return hinfDesign{}, fmt.Errorf("%s: plant has modes at both z = 1 and z = -1 that no static output feedback moves: %w", op, ErrOptionUnsupported)
	}
	if err != nil {
		return hinfDesign{}, fmt.Errorf("%s: %w", op, err)
	}
	d, err := hinfSynScaled(op, Pc, nmeas, ncont, "P12(z = "+point+")", "P21(z = "+point+")")
	if err != nil {
		return hinfDesign{}, err
	}
	build := d.build
	d.build = func(g float64) (*HinfSynResult, error) {
		res, err := build(g)
		if err != nil {
			return nil, err
		}
		Kc := res.K
		if nk, _, _ := Kc.Dims(); nk > 0 {
			if bal, err := Ssbal(Kc); err == nil {
				Kc = bal.Sys
			}
		}
		K, err := Kc.discretizeTustin(P.Dt, 0)
		if err != nil {
			return nil, fmt.Errorf("%s: mapping the controller back to discrete time: %w", op, err)
		}
		beta := 2 / P.Dt
		sign := complex(1, 0)
		if reflect {
			K.A.Scale(-1, K.A)
			K.B.Scale(-1, K.B)
			sign = -1
		}
		for i, s := range res.CLPoles {
			res.CLPoles[i] = sign * (complex(beta, 0) + s) / (complex(beta, 0) - s)
		}
		res.K = K
		return res, nil
	}
	return d, nil
}

// unitCircleModeShift returns a static output feedback D0 and the plant P
// with u = D0·y + v closed, from [w; v] to [z; y]. With Δ = (I − D22 D0)⁻¹:
//
//	A' = A + B2 D0 Δ C2,  B' = [B1 + B2 D0 Δ D21, B2 (I + D0 Δ D22)],
//	C' = [C1 + D12 D0 Δ C2; Δ C2],
//	D' = [D11 + D12 D0 Δ D21, D12 (I + D0 Δ D22); Δ D21, Δ D22].
//
// Static output feedback can move every mode that is both controllable from
// u and observable from y (Davison & Wang, 1975), which unit-circle modes of
// a stabilizable and detectable plant are. D0 is the best of a fixed set of
// candidates by tustinCond, and ok is false when none improves on cond, that
// of P; any D0 gives the same closed loops, so the choice affects
// conditioning only.
func (gp *generalizedPlantPartition) unitCircleModeShift(P *System, cond float64) (*mat.Dense, *System, bool) {
	m2, p2 := gp.m2, gp.p2
	base := (1 + mat.Norm(gp.A, 2)) / math.Max(mat.Norm(gp.B2, 2)*mat.Norm(gp.C2, 2), math.SmallestNonzeroFloat64)
	B2tC2t := mulDense(mat.DenseCopyOf(gp.B2.T()), mat.DenseCopyOf(gp.C2.T()))
	dirs := []*mat.Dense{B2tC2t}
	for k := 1; k <= 4; k++ {
		D := mat.NewDense(m2, p2, nil)
		for i := range m2 {
			for j := range p2 {
				D.Set(i, j, math.Cos(float64(k*(i*p2+j)+k*k)))
			}
		}
		dirs = append(dirs, D)
	}
	var best *mat.Dense
	bestCond := cond
	for _, dir := range dirs {
		dn := mat.Norm(dir, 2)
		if dn == 0 {
			continue
		}
		for _, t := range []float64{0.1, -0.1, 0.5, -0.5, 1, -1} {
			D0 := mat.NewDense(m2, p2, nil)
			D0.Scale(t*base/dn, dir)
			As, ok := shiftedStateMatrix(gp, D0)
			if !ok {
				continue
			}
			if c := tustinCond(As); c < bestCond {
				best, bestCond = D0, c
			}
		}
	}
	if best == nil {
		return nil, nil, false
	}
	Ps, ok := gp.closeStaticLoop(P, best)
	if !ok {
		return nil, nil, false
	}
	return best, Ps, true
}

// staticLoopGain returns D0·Δ = D0 (I − D22 D0)⁻¹, or false when the loop is
// ill-posed.
func staticLoopGain(gp *generalizedPlantPartition, D0 *mat.Dense) (*mat.Dense, bool) {
	IDD := mulDense(gp.D22, D0)
	IDD.Scale(-1, IDD)
	for i := range gp.p2 {
		IDD.Set(i, i, IDD.At(i, i)+1)
	}
	var lu mat.LU
	lu.Factorize(IDD)
	if !(lu.Cond() < hinfTustinCond) {
		return nil, false
	}
	Delta, err := invertSmall(IDD, gp.p2)
	if err != nil {
		return nil, false
	}
	return mulDense(D0, Delta), true
}

func shiftedStateMatrix(gp *generalizedPlantPartition, D0 *mat.Dense) (*mat.Dense, bool) {
	G, ok := staticLoopGain(gp, D0)
	if !ok {
		return nil, false
	}
	As := mulDense(mulDense(gp.B2, G), gp.C2)
	As.Add(gp.A, As)
	return As, true
}

func (gp *generalizedPlantPartition) closeStaticLoop(P *System, D0 *mat.Dense) (*System, bool) {
	G, ok := staticLoopGain(gp, D0)
	if !ok {
		return nil, false
	}
	n, m1, m2, p1, p2 := gp.n, gp.m1, gp.m2, gp.p1, gp.p2
	Delta := mulDense(gp.D22, G)
	for i := range p2 {
		Delta.Set(i, i, Delta.At(i, i)+1)
	}
	U := mulDense(G, gp.D22)
	for i := range m2 {
		U.Set(i, i, U.At(i, i)+1)
	}
	A := mulDense(mulDense(gp.B2, G), gp.C2)
	A.Add(gp.A, A)
	B := mat.NewDense(n, m1+m2, nil)
	B1 := mulDense(mulDense(gp.B2, G), gp.D21)
	B1.Add(gp.B1, B1)
	setBlock(B, 0, 0, B1)
	setBlock(B, 0, m1, mulDense(gp.B2, U))
	C := mat.NewDense(p1+p2, n, nil)
	C1 := mulDense(mulDense(gp.D12, G), gp.C2)
	C1.Add(gp.C1, C1)
	setBlock(C, 0, 0, C1)
	setBlock(C, p1, 0, mulDense(Delta, gp.C2))
	D := mat.NewDense(p1+p2, m1+m2, nil)
	D11 := mulDense(mulDense(gp.D12, G), gp.D21)
	D11.Add(gp.D11, D11)
	setBlock(D, 0, 0, D11)
	setBlock(D, 0, m1, mulDense(gp.D12, U))
	setBlock(D, p1, 0, mulDense(Delta, gp.D21))
	setBlock(D, p1, m1, mulDense(Delta, gp.D22))
	Ps, err := newDerived(A, B, C, D, P.Dt)
	if err != nil {
		return nil, false
	}
	propagateNames(Ps, P)
	return Ps, true
}

// hinfBisect returns the smallest gamma above gammaLB, to relative 1e-6 or
// absolute hinfGammaAbsTol, that feasible accepts, giving up once the
// doubled upper bound passes hinfGammaCap·max(1, scale).
func hinfBisect(gammaLB, scale float64, feasible func(float64) bool) (float64, error) {
	gammaUB := gammaLB*2 + 1
	limit := hinfGammaCap * math.Max(1, scale)
	for !feasible(gammaUB) {
		gammaUB *= 2
		if gammaUB > limit {
			return 0, fmt.Errorf("no feasible gamma below %g: %w", gammaUB, ErrGammaNotAchievable)
		}
	}
	for gammaUB-gammaLB > math.Max(1e-6*gammaUB, hinfGammaAbsTol) {
		mid := (gammaLB + gammaUB) / 2
		if feasible(mid) {
			gammaUB = mid
		} else {
			gammaLB = mid
		}
	}
	return gammaUB, nil
}

// hinfGammaCap is the gamma, relative to hinfGammaScale, beyond which
// hinfBisect deems a plant infeasible.
const hinfGammaCap = 1e12

// hinfGammaScale is ‖[C1 D11 D12]‖_F·‖[B1; D11; D21]‖_F, the size of the
// performance channels: scaling z or w by k scales it, and the optimal
// gamma, by k, so hinfBisect's give-up bound follows the plant's scaling.
func (gp *generalizedPlantPartition) hinfGammaScale() float64 {
	sq := func(ms ...*mat.Dense) float64 {
		var s float64
		for _, m := range ms {
			if m != nil && !m.IsEmpty() {
				f := mat.Norm(m, 2)
				s += f * f
			}
		}
		return math.Sqrt(s)
	}
	return sq(gp.C1, gp.D11, gp.D12) * sq(gp.B1, gp.D11, gp.D21)
}

func hinfSynD11Zero(gp *generalizedPlantPartition, ric *hinfD11ZeroRiccatis, gamma float64) (*HinfSynResult, error) {
	n := gp.n
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	X, Y, err := ric.solve(gamma)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}

	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	R1inv, err := invertSmall(R1, gp.m2)
	if err != nil {
		return nil, fmt.Errorf("%s: D12ᵀD12: %w", gp.op, err)
	}
	S1 := mulDense(mat.DenseCopyOf(D12.T()), C1)

	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	R2inv, err := invertSmall(R2, gp.p2)
	if err != nil {
		return nil, fmt.Errorf("%s: D21D21ᵀ: %w", gp.op, err)
	}
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	// F = -R1inv * (B2'*X + S1)
	B2tX := mulDense(mat.DenseCopyOf(B2.T()), X)
	B2tX.Add(B2tX, S1)
	F := mulDense(R1inv, B2tX)
	F.Scale(-1, F)

	// L = -(Y*C2' + S2) * R2inv
	YC2t := mulDense(Y, mat.DenseCopyOf(C2.T()))
	YC2t.Add(YC2t, S2)
	L := mulDense(YC2t, R2inv)
	L.Scale(-1, L)

	ginv2 := 1.0 / (gamma * gamma)

	YX := mulDense(Y, X)
	nn := n
	eye := mat.NewDense(nn, nn, nil)
	for i := range nn {
		eye.Set(i, i, 1)
	}
	ZpArg := mat.NewDense(nn, nn, nil)
	ZpArg.Scale(ginv2, YX)
	ZpArg.Sub(eye, ZpArg)

	var lu mat.LU
	lu.Factorize(ZpArg)
	Zp := mat.NewDense(nn, nn, nil)
	if err := lu.SolveTo(Zp, false, eye); err != nil {
		return nil, fmt.Errorf("%s: I − XY/γ² singular at γ = %g: %w", gp.op, gamma, ErrGammaNotAchievable)
	}

	// Ak = A + ginv2*B1*B1'*X + B2*F + Zp*L*(C2 + ginv2*D21*B1'*X); the D21 term feeds the worst-case disturbance into the observer.
	Ak := denseCopy(A)

	tmp1 := mulDense(B1, mat.DenseCopyOf(B1.T()))
	tmp1.Mul(tmp1, X)
	tmp1.Scale(ginv2, tmp1)
	Ak.Add(Ak, tmp1)

	Ak.Add(Ak, mulDense(B2, F))

	ZpL := mulDense(Zp, L)
	C2w := C2
	if !allZeroDense(S2) {
		C2w = mulDense(mat.DenseCopyOf(S2.T()), X)
		C2w.Scale(ginv2, C2w)
		C2w.Add(C2w, C2)
	}
	Ak.Add(Ak, mulDense(ZpL, C2w))

	Bk := mulDense(Zp, L)
	Bk.Scale(-1, Bk)
	Ck := denseCopy(F)

	K, err := gp.newController(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	clPoles, err := gp.closedLoopPoles(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	return &HinfSynResult{K: K, GammaOpt: gamma, X: X, Y: Y, CLPoles: clPoles}, nil
}

// hinfD11ZeroRiccatis holds the γ-independent terms of the D11 = 0 Riccati
// pair, so each γ of a bisection only forms Gx and Gy and solves.
type hinfD11ZeroRiccatis struct {
	errR1, errR2        error
	B1B1t, C1tC1        *mat.Dense
	Ahat, Qhat, B2R1B2t *mat.Dense
	AtildeT, Qy, C2R2C2 *mat.Dense
	Gx, Gy, XY          *mat.Dense
	ham                 *hamiltonianRiccatiWork
}

func newHinfD11ZeroRiccatis(gp *generalizedPlantPartition) *hinfD11ZeroRiccatis {
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	n := gp.n
	r := &hinfD11ZeroRiccatis{}

	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	R1inv, err := invertSmall(R1, gp.m2)
	if err != nil {
		r.errR1 = fmt.Errorf("D12 or D21 not full rank: %w", ErrInvalidPartition)
		return r
	}
	S1 := mulDense(mat.DenseCopyOf(D12.T()), C1)

	r.B1B1t = mulDense(B1, mat.DenseCopyOf(B1.T()))
	r.C1tC1 = mulDense(mat.DenseCopyOf(C1.T()), C1)

	// X-Riccati: Ahat = A - B2*R1inv*S1
	B2R1inv := mulDense(B2, R1inv)
	r.Ahat = mat.NewDense(n, n, nil)
	r.Ahat.Sub(A, mulDense(B2R1inv, S1))

	var ok bool
	if r.Qhat, ok = complementGram(C1, D12); !ok {
		r.errR1 = fmt.Errorf("SVD of D12 failed: %w", ErrSchurFailed)
		return r
	}

	// Gx = ginv2*B1*B1' - B2*R1inv*B2'
	r.B2R1B2t = mulDense(B2R1inv, mat.DenseCopyOf(B2.T()))
	r.Gx = mat.NewDense(n, n, nil)
	r.ham = newHamiltonianRiccatiWork(n)

	// Y-Riccati
	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	R2inv, err := invertSmall(R2, gp.p2)
	if err != nil {
		r.errR2 = fmt.Errorf("D12 or D21 not full rank: %w", ErrInvalidPartition)
		return r
	}
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	// Atilde = A - S2*R2inv*C2
	S2R2inv := mulDense(S2, R2inv)
	Atilde := mat.NewDense(n, n, nil)
	Atilde.Sub(A, mulDense(S2R2inv, C2))
	r.AtildeT = mat.DenseCopyOf(Atilde.T())

	if r.Qy, ok = complementGram(mat.DenseCopyOf(B1.T()), mat.DenseCopyOf(D21.T())); !ok {
		r.errR2 = fmt.Errorf("SVD of D21 failed: %w", ErrSchurFailed)
		return r
	}

	// Gy = ginv2*C1'C1 - C2'*R2inv*C2
	r.C2R2C2 = mulDense(mat.DenseCopyOf(C2.T()), mulDense(R2inv, C2))
	r.Gy = mat.NewDense(n, n, nil)
	return r
}

// complementGram returns MᵀU₂U₂ᵀM for U₂ an orthonormal basis of the
// orthogonal complement of range(D), that is Mᵀ(I − D(DᵀD)⁻¹Dᵀ)M for D of
// full column rank: Qhat = C1ᵀ(I − D12 R1⁻¹ D12ᵀ)C1 with (M, D) = (C1, D12)
// and, transposed, Qy = B1(I − D21ᵀR2⁻¹D21)B1ᵀ. Forming it as a Gram matrix
// keeps it exactly semidefinite, where the subtraction C1ᵀC1 − S1ᵀR1⁻¹S1
// cancels to rounding noise of size eps·‖C1‖² (all of it when D12 is
// square), noise that decides γ-feasibility once C1 is large. ok is false
// when the SVD of D fails.
func complementGram(M, D *mat.Dense) (*mat.Dense, bool) {
	p, k := D.Dims()
	_, n := M.Dims()
	G := mat.NewDense(n, n, nil)
	if p == k {
		return G, true
	}
	var svd mat.SVD
	if !svd.Factorize(D, mat.SVDFull) {
		return nil, false
	}
	var U mat.Dense
	svd.UTo(&U)
	U2 := U.Slice(0, p, k, p)
	var Z mat.Dense
	Z.Mul(U2.T(), M)
	G.Mul(Z.T(), &Z)
	return G, true
}

func (r *hinfD11ZeroRiccatis) feasible(gamma float64) bool {
	_, _, err := r.solve(gamma)
	return err == nil
}

func (r *hinfD11ZeroRiccatis) solve(gamma float64) (*mat.Dense, *mat.Dense, error) {
	if r.errR1 != nil {
		return nil, nil, r.errR1
	}
	ginv2 := 1.0 / (gamma * gamma)

	r.Gx.Scale(ginv2, r.B1B1t)
	r.Gx.Sub(r.Gx, r.B2R1B2t)
	r.ham.setHamiltonian(r.Ahat, r.Gx, r.Qhat)
	X, err := r.ham.solve()
	if err != nil {
		return nil, nil, err
	}
	if r.errR2 != nil {
		return nil, nil, r.errR2
	}

	r.Gy.Scale(ginv2, r.C1tC1)
	r.Gy.Sub(r.Gy, r.C2R2C2)
	r.ham.setHamiltonian(r.AtildeT, r.Gy, r.Qy)
	Y, err := r.ham.solve()
	if err != nil {
		return nil, nil, err
	}

	r.XY = mulInto(r.XY, X, Y)
	var eig mat.Eigen
	ok := eig.Factorize(r.XY, mat.EigenNone)
	if !ok {
		return nil, nil, fmt.Errorf("Riccati solutions not admissible at γ = %g: %w", gamma, ErrGammaNotAchievable)
	}
	vals := eig.Values(nil)
	g2 := gamma * gamma
	for _, v := range vals {
		if math.Abs(real(v)) >= g2 {
			return nil, nil, fmt.Errorf("Riccati solutions not admissible at γ = %g: %w", gamma, ErrGammaNotAchievable)
		}
	}

	return X, Y, nil
}

// solveHamiltonianRiccati returns the stabilizing solution of the Riccati
// equation with Hamiltonian H. H is rejected when an eigenvalue is on the
// imaginary axis to within rounding, where the stable/unstable split is
// arbitrary and yields spurious solutions.
func solveHamiltonianRiccati(H *mat.Dense, n int) (*mat.Dense, error) {
	w := newHamiltonianRiccatiWork(n)
	w.c = 1
	setBlock(w.H, 0, 0, H)
	return w.solve()
}

// riccatiScale returns the power of two nearest √(q/g), the norms of Q and
// G, or 1 when either is zero or not finite.
func riccatiScale(g, q float64) float64 {
	if !(g > 0 && q > 0) || math.IsInf(g, 0) || math.IsInf(q, 0) {
		return 1
	}
	return math.Ldexp(1, int(math.Round(math.Log2(q/g)/2)))
}

// hamiltonianRiccatiWork holds the Hamiltonian and LAPACK buffers reused by
// repeated solves of one order, as along a γ bisection.
type hamiltonianRiccatiWork struct {
	n        int
	H        *mat.Dense
	hData    []float64
	wr, wi   []float64
	vs       []float64
	bwork    []bool
	work     []float64
	u11, u21 []float64
	ipiv     []int
	c        float64
}

func newHamiltonianRiccatiWork(n int) *hamiltonianRiccatiWork {
	nn := 2 * n
	return &hamiltonianRiccatiWork{
		n:     n,
		H:     mat.NewDense(nn, nn, nil),
		hData: make([]float64, nn*nn),
		wr:    make([]float64, nn),
		wi:    make([]float64, nn),
		vs:    make([]float64, nn*nn),
		bwork: make([]bool, nn),
		u11:   make([]float64, n*n),
		u21:   make([]float64, n*n),
		ipiv:  make([]int, n),
	}
}

// setHamiltonian sets H to the symplectically scaled [F c·G; -Q/c -Fᵀ] =
// diag(I, 1/c)·[F G; -Q -Fᵀ]·diag(I, c), whose stable invariant subspace is
// spanned by [I; X/c]; solve returns c times that solution. c (see
// riccatiScale) balances the off-diagonal blocks: scaling z or w by k scales
// ‖Q‖/‖G‖ by k² and X by k², so unscaled, U11 of the basis [U11; U21] ≈
// [I; X] shrinks against U21 and X = U21·U11⁻¹ and the semidefiniteness
// test lose accuracy with k. A power-of-two c makes the scaling exact.
func (w *hamiltonianRiccatiWork) setHamiltonian(F, G, Q *mat.Dense) {
	n := w.n
	w.c = riccatiScale(mat.Norm(G, 1), mat.Norm(Q, 1))
	h := w.H.RawMatrix()
	f, g, q := F.RawMatrix(), G.RawMatrix(), Q.RawMatrix()
	for i := range n {
		top := h.Data[i*h.Stride : i*h.Stride+2*n]
		copy(top[:n], f.Data[i*f.Stride:i*f.Stride+n])
		for j := range n {
			top[n+j] = w.c * g.Data[i*g.Stride+j]
		}
		bottom := h.Data[(n+i)*h.Stride : (n+i)*h.Stride+2*n]
		for j := range n {
			bottom[j] = -q.Data[i*q.Stride+j] / w.c
			bottom[n+j] = -f.Data[j*f.Stride+i]
		}
	}
}

// solve returns the stabilizing Riccati solution for the Hamiltonian in H.
func (w *hamiltonianRiccatiWork) solve() (*mat.Dense, error) {
	n := w.n
	nn := 2 * n
	hRaw := w.H.RawMatrix()
	hData := w.hData
	copyStrided(hData, nn, hRaw.Data, hRaw.Stride, nn, nn)

	wr, wi, vs, bwork := w.wr, w.wi, w.vs, w.bwork

	selctg := func(wr, wi float64) bool { return wr < 0 }

	if w.work == nil {
		var workQuery [1]float64
		impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
			nn, hData, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
		w.work = make([]float64, int(workQuery[0]))
	}
	work := w.work
	lwork := len(work)

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, fmt.Errorf("Hamiltonian Schur form did not converge: %w", ErrSchurFailed)
	}
	if sdim != n {
		return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
	}
	if hasImaginaryAxisEigenvalue(hData, nn, wr, wi) {
		return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
	}

	u11, u21 := w.u11, w.u21
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)

	ipiv := w.ipiv
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
	}

	// X = U21 * U11^{-1}  =>  solve U11' * X' = U21'  =>  transpose approach
	xData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)
	symmetrize(xData, n, n)
	X := mat.NewDense(n, n, xData)
	var eig mat.Eigen
	if !eig.Factorize(X, mat.EigenNone) {
		return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
	}
	vals := eig.Values(nil)
	floor := 1.0
	for _, v := range vals {
		floor = math.Max(floor, math.Abs(real(v)))
	}
	floor *= hamiltonianPSDTol
	for _, v := range vals {
		if real(v) < -floor {
			return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
		}
	}
	if w.c != 1 {
		X.Scale(w.c, X)
	}
	return X, nil
}

// hamiltonianPSDTol bounds the negative eigenvalues solve accepts as rounding
// of a semidefinite solution: those of X/c, the solution of the scaled
// Hamiltonian (see setHamiltonian), above −hamiltonianPSDTol·max(1, ‖X/c‖).
// X/c does not change when z or w is scaled, so neither does the test; the
// relative part keeps it from failing on the rounding of a large X/c.
const hamiltonianPSDTol = 1e-8

// An eigenvalue with |Re| <= hamiltonianAxisCandidate*|lambda| is on the
// imaginary axis when |Re| is also within hamiltonianAxisErr times its
// perturbation bound eps*||T||_F/s, s being its reciprocal condition number.
// Rounding splits an axis pair into a near-Jordan block with tiny s, while
// lightly damped normal modes keep s near 1 and slow modes of a stiff H are
// not candidates. A mode both lightly damped and ill-conditioned is treated
// conservatively.
const (
	hamiltonianAxisCandidate = 1e-4
	hamiltonianAxisErr       = 10
)

// hasImaginaryAxisEigenvalue reports whether the real Schur form t (n x n,
// eigenvalues wr + i wi) has an eigenvalue on the imaginary axis to within
// rounding.
func hasImaginaryAxisEigenvalue(t []float64, n int, wr, wi []float64) bool {
	const c2 = hamiltonianAxisCandidate * hamiltonianAxisCandidate
	var tNorm float64
	var tc, work []float64
	var selected []bool
	var iwork [1]int
	for i := range n {
		re := math.Abs(wr[i])
		if wi[i] < 0 || re*re > c2*(wr[i]*wr[i]+wi[i]*wi[i]) {
			continue
		}
		if tc == nil {
			for r := range n {
				for _, v := range t[r*n+r-min(r, 1) : (r+1)*n] {
					tNorm += v * v
				}
			}
			tNorm = math.Sqrt(tNorm)
			tc = make([]float64, n*n)
			selected = make([]bool, n)
			work = make([]float64, max(1, 2*n))
		}
		copy(tc, t)
		wrc, wic := slices.Clone(wr), slices.Clone(wi)
		clear(selected)
		selected[i] = true
		_, s, _, ok := impl.Dtrsen(1, false, selected, n, tc, n, nil, 1, wrc, wic, work, len(work), iwork[:], 1)
		if !ok || re <= hamiltonianAxisErr*eps()*tNorm/s {
			return true
		}
	}
	return false
}
