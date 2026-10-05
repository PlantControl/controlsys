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

// HinfSynResult is an H∞ controller, as MATLAB hinfsyn returns K, gamma and
// info: X and Y are the state-feedback and filter Riccati solutions at
// GammaOpt and CLPoles the closed-loop poles. For a discrete plant X and Y
// solve the Riccati equations of its Tustin-equivalent continuous plant (in
// the same state coordinates), not discrete Riccati equations; when HinfSyn
// first shifts modes away from z = ±1 (see HinfSyn), that of the shifted
// plant.
type HinfSynResult struct {
	K *System
	// GammaOpt is the gamma K is built for, verified on the returned K:
	// ||T_zw||inf <= GammaOpt. It is within hinfControllerBackoff (relative)
	// of the smallest achievable gamma unless the Riccati solutions are
	// ill-conditioned there, as when the infimum is 0 and unattained; then it
	// is the smallest backed-off gamma whose controller meets it.
	GammaOpt float64
	X        *mat.Dense
	Y        *mat.Dense
	CLPoles  []complex128
	// clNorm is ||T_zw||inf of P with K, as verified.
	clNorm float64
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
// bisection edge and checks ||T_zw||inf <= GammaOpt on the returned K. If the
// build fails or K misses that gamma (the Riccati solutions are numerically
// unreliable near an unattained infimum such as 0, where K grows without
// bound), it backs gamma off upward, since every gamma above the infimum is
// achievable, until a controller meets it.
func HinfSyn(P *System, nmeas, ncont int) (*HinfSynResult, error) {
	gp, err := partitionGeneralizedPlant("HinfSyn", P, nmeas, ncont)
	if err != nil {
		return nil, err
	}
	var d hinfDesign
	if P.IsDiscrete() {
		d, err = hinfSynDiscrete(gp, P, nmeas, ncont)
	} else {
		d, err = hinfSynPartition(gp, "D12", "D21")
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

// hinfControllerBackoff is the relative gamma margin above the bisection
// edge. At the edge X or Y grows without bound, so the central controller is
// ill-conditioned and overshoots gamma; 1e-4 restores a genuine margin.
const hinfControllerBackoff = 1e-4

// hinfBackoffTries bounds the doublings of the gamma margin in verified.
const hinfBackoffTries = 48

// verified builds the controller at gammaEdge plus a margin that starts at
// hinfControllerBackoff·gammaEdge and doubles (at least to hinfGammaAbsTol)
// until the build succeeds and the closed loop of P with the returned K
// meets gamma.
func (d hinfDesign) verified(op string, P *System, nmeas, ncont int) (*HinfSynResult, error) {
	margin := hinfControllerBackoff * d.gammaEdge
	var err error
	for range hinfBackoffTries {
		var res *HinfSynResult
		if res, err = d.build(d.gammaEdge + margin); err == nil {
			if err = res.meets(op, P, nmeas, ncont); err == nil {
				return res, nil
			}
		}
		margin = math.Max(2*margin, hinfGammaAbsTol)
	}
	return nil, err
}

// meets checks ||T_zw||inf <= GammaOpt for the closed loop of P with K and
// records that norm.
func (res *HinfSynResult) meets(op string, P *System, nmeas, ncont int) error {
	cl, err := LFT(P, res.K, LFTFeedback{Nu: ncont, Ny: nmeas})
	if err != nil {
		return fmt.Errorf("%s: closing the loop: %w", op, err)
	}
	norm, _, err := HinfNorm(cl)
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	if !(norm <= res.GammaOpt) {
		return fmt.Errorf("%s: controller for γ = %g gives ‖T_zw‖∞ = %g: %w", op, res.GammaOpt, norm, ErrGammaNotAchievable)
	}
	res.clNorm = norm
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
	gamma, err := hinfBisect(0, func(g float64) bool { return hinfFeasible(gp, g) })
	if err != nil {
		return hinfDesign{}, fmt.Errorf("%s: %w", gp.op, err)
	}
	return hinfDesign{gammaEdge: gamma, build: func(g float64) (*HinfSynResult, error) { return hinfSynD11Zero(gp, g) }}, nil
}

// hinfSynDiscrete designs for the discrete plant P through the Tustin map
// z = (β+s)/(β−s), β = 2/Ts, or through z = −(β+s)/(β−s) applied to P(−z)
// when I + A is worse conditioned than I − A (a mode at or near z = −1,
// which the plain map sends to s = ∞). Both maps take the unit circle onto
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
	reflect := plus.Cond() > minus.Cond()
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
	gp, err := partitionGeneralizedPlant(op, Pc, nmeas, ncont)
	if err != nil {
		return hinfDesign{}, err
	}
	d, err := hinfSynPartition(gp, "P12(z = "+point+")", "P21(z = "+point+")")
	if err != nil {
		return hinfDesign{}, err
	}
	build := d.build
	d.build = func(g float64) (*HinfSynResult, error) {
		res, err := build(g)
		if err != nil {
			return nil, err
		}
		K, err := res.K.discretizeTustin(P.Dt, 0)
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
	Ps, err := New(A, B, C, D, P.Dt)
	if err != nil {
		return nil, false
	}
	propagateNames(Ps, P)
	return Ps, true
}

// hinfBisect returns the smallest gamma above gammaLB, to relative 1e-6 or
// absolute hinfGammaAbsTol, that feasible accepts.
func hinfBisect(gammaLB float64, feasible func(float64) bool) (float64, error) {
	gammaUB := gammaLB*2 + 1
	for !feasible(gammaUB) {
		gammaUB *= 2
		if gammaUB > 1e12 {
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

func hinfSynD11Zero(gp *generalizedPlantPartition, gamma float64) (*HinfSynResult, error) {
	n := gp.n
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	X, Y, err := hinfSolveRiccatis(gp, gamma)
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

func hinfFeasible(gp *generalizedPlantPartition, gamma float64) bool {
	_, _, err := hinfSolveRiccatis(gp, gamma)
	return err == nil
}

func hinfSolveRiccatis(gp *generalizedPlantPartition, gamma float64) (*mat.Dense, *mat.Dense, error) {
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	n := gp.n
	ginv2 := 1.0 / (gamma * gamma)

	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	R1inv, err := invertSmall(R1, gp.m2)
	if err != nil {
		return nil, nil, fmt.Errorf("D12 or D21 not full rank: %w", ErrInvalidPartition)
	}
	S1 := mulDense(mat.DenseCopyOf(D12.T()), C1)

	B1B1t := mulDense(B1, mat.DenseCopyOf(B1.T()))
	C1tC1 := mulDense(mat.DenseCopyOf(C1.T()), C1)

	// X-Riccati: Ahat = A - B2*R1inv*S1
	B2R1inv := mulDense(B2, R1inv)
	Ahat := mat.NewDense(n, n, nil)
	Ahat.Sub(A, mulDense(B2R1inv, S1))

	// Qhat = C1'C1 - S1'*R1inv*S1
	Qhat := mat.NewDense(n, n, nil)
	Qhat.Sub(C1tC1, mulDense(mat.DenseCopyOf(S1.T()), mulDense(R1inv, S1)))

	// Gx = ginv2*B1*B1' - B2*R1inv*B2'
	Gx := mat.NewDense(n, n, nil)
	Gx.Scale(ginv2, B1B1t)
	Gx.Sub(Gx, mulDense(B2R1inv, mat.DenseCopyOf(B2.T())))

	Hx := mat.NewDense(2*n, 2*n, nil)
	setBlock(Hx, 0, 0, Ahat)
	setBlock(Hx, 0, n, Gx)
	negQhat := mat.NewDense(n, n, nil)
	negQhat.Scale(-1, Qhat)
	setBlock(Hx, n, 0, negQhat)
	negAhatT := mat.NewDense(n, n, nil)
	negAhatT.Scale(-1, mat.DenseCopyOf(Ahat.T()))
	setBlock(Hx, n, n, negAhatT)

	X, err := solveHamiltonianRiccati(Hx, n)
	if err != nil {
		return nil, nil, err
	}

	// Y-Riccati
	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	R2inv, err := invertSmall(R2, gp.p2)
	if err != nil {
		return nil, nil, fmt.Errorf("D12 or D21 not full rank: %w", ErrInvalidPartition)
	}
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	// Atilde = A - S2*R2inv*C2
	S2R2inv := mulDense(S2, R2inv)
	Atilde := mat.NewDense(n, n, nil)
	Atilde.Sub(A, mulDense(S2R2inv, C2))

	// Qy = B1*B1' - S2*R2inv*S2'
	Qy := mat.NewDense(n, n, nil)
	Qy.Sub(B1B1t, mulDense(S2R2inv, mat.DenseCopyOf(S2.T())))

	// Gy = ginv2*C1'C1 - C2'*R2inv*C2
	Gy := mat.NewDense(n, n, nil)
	Gy.Scale(ginv2, C1tC1)
	Gy.Sub(Gy, mulDense(mat.DenseCopyOf(C2.T()), mulDense(R2inv, C2)))

	Hy := mat.NewDense(2*n, 2*n, nil)
	setBlock(Hy, 0, 0, mat.DenseCopyOf(Atilde.T()))
	setBlock(Hy, 0, n, Gy)
	negQy := mat.NewDense(n, n, nil)
	negQy.Scale(-1, Qy)
	setBlock(Hy, n, 0, negQy)
	negAtilde := mat.NewDense(n, n, nil)
	negAtilde.Scale(-1, Atilde)
	setBlock(Hy, n, n, negAtilde)

	Y, err := solveHamiltonianRiccati(Hy, n)
	if err != nil {
		return nil, nil, err
	}

	XY := mulDense(X, Y)
	var eig mat.Eigen
	ok := eig.Factorize(XY, mat.EigenNone)
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
	nn := 2 * n
	hRaw := H.RawMatrix()
	hData := make([]float64, nn*nn)
	copyStrided(hData, nn, hRaw.Data, hRaw.Stride, nn, nn)

	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)

	selctg := func(wr, wi float64) bool { return wr < 0 }

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

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

	u11 := make([]float64, n*n)
	u21 := make([]float64, n*n)
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)

	ipiv := make([]int, n)
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

	// Check positive semi-definiteness via eigenvalues
	X := mat.NewDense(n, n, xData)
	var eig mat.Eigen
	eigOk := eig.Factorize(X, mat.EigenNone)
	if !eigOk {
		return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if real(v) < -1e-8 {
			return nil, fmt.Errorf("Hamiltonian has no stabilizing solution: %w", ErrNoStabilizing)
		}
	}

	return X, nil
}

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
