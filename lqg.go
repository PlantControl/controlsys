package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// LqgOpts selects the MATLAB lqg variant. A nil *LqgOpts designs the
// regulator, with the delayed estimator x[n|n-1] in discrete time.
type LqgOpts struct {
	// QI is the p×p weight on the integral xi of the tracking error r - y.
	// Non-nil designs the servo controller, as lqg(sys,QXU,QWV,QI).
	QI *mat.Dense
	// OneDOF makes the servo controller take e = r - y instead of [r; y]
	// ('1dof'). It requires QI.
	OneDOF bool
	// Current uses the current estimate x[n|n] ('current'). Continuous
	// plants reject it with ErrWrongDomain.
	Current bool
}

// LqgResult holds the controller and the gains MATLAB lqg returns in info.
type LqgResult struct {
	// Controller is reg; connect it to the plant with positive feedback.
	Controller *System
	// K is the m×n state-feedback gain Kx.
	K *mat.Dense
	// Ki is the m×p integrator gain; nil without QI.
	Ki *mat.Dense
	// Kw is the m×n process-noise gain; nil unless Current.
	Kw *mat.Dense
	// L is the n×p Kalman gain (predictor gain in discrete time).
	L *mat.Dense
	// Mx and Mw are the n×p innovation gains giving x[n|n] and w[n|n]; nil
	// unless Current.
	Mx *mat.Dense
	Mw *mat.Dense
	// Xc is the regulator Riccati solution, (n+p)×(n+p) with QI.
	Xc *mat.Dense
	// Xf is the Kalman Riccati solution P (steady-state error covariance).
	Xf *mat.Dense
}

// Lqg designs a linear-quadratic-Gaussian controller, matching MATLAB
// lqg(sys,QXU,QWV[,QI][,'1dof'][,'current'])
// (https://www.mathworks.com/help/control/ref/ss.lqg.html) for the plant
//
//	x' = Ax + Bu + w,  y = Cx + Du + v,  E([w;v][w;v]') = QWV
//
// and the cost E{[x;u]' QXU [x;u] + xi' QI xi}. QXU = [Q N; N' R] is
// (n+m)×(n+m) and QWV = [Qn Nn; Nn' Rn] is (n+p)×(n+p); both must be
// symmetric. K is Lqr/Dlqr(A, B, Q, R) with cross term N, and L is the
// Kalman gain for G = I, H = 0 with noise cross-covariance Nn. With QI,
// [Kx Ki] is designed as MATLAB lqi on the plant augmented with
// xi' = r - y (xi[n+1] = xi[n] + Ts(r[n] - y[n]) in discrete time), i.e.
// Aa = [A 0; -C 0], Ba = [B; -D], weights blkdiag(Q, QI), R and [N; 0].
//
// The controller implements
//
//	x̂' = Ax̂ + Bu + L(y - Cx̂ - Du),  u = -Kx x̂ - Ki xi
//
// from input y (regulator), [r; y] (servo) or e = r - y (OneDOF, where the
// estimator sees y = -e), so it is connected to the plant with positive
// feedback. In discrete time x̂ = x[n|n-1] by default. Current uses
//
//	u = -Kx x[n|n-1] - Ki xi - (Kx Mx + Kw Mw)(y - C x[n|n-1] - Du)
//
// with Mx = PC'(CPC' + Rn)⁻¹, Mw = Nn(CPC' + Rn)⁻¹ and
// Kw = (R + Ba'X Ba)⁻¹ Ba'X [I; 0]; a singular I - (Kx Mx + Kw Mw)D makes
// the controller non-causal and returns ErrAlgebraicLoop.
//
// Plants without inputs, outputs or states are rejected with
// ErrDimensionMismatch, descriptor plants with ErrDescriptorRiccati and
// plants with delays with ErrDelayUnsupported.
func Lqg(sys *System, QXU, QWV *mat.Dense, opts *LqgOpts) (*LqgResult, error) {
	policy, err := newControllerObserverPolicy(sys, "Lqg")
	if err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if m == 0 || p == 0 {
		return nil, fmt.Errorf("Lqg: model needs inputs and measured outputs: %w", ErrDimensionMismatch)
	}
	var o LqgOpts
	if opts != nil {
		o = *opts
	}
	servo := o.QI != nil
	if o.OneDOF && !servo {
		return nil, fmt.Errorf("Lqg: OneDOF requires QI: %w", ErrInvalidArgument)
	}
	continuous := sys.IsContinuous()
	if o.Current && continuous {
		return nil, fmt.Errorf("Lqg: Current requires a discrete-time plant: %w", ErrWrongDomain)
	}
	if err := validateLqgWeight("QXU", QXU, n+m); err != nil {
		return nil, err
	}
	if err := validateLqgWeight("QWV", QWV, n+p); err != nil {
		return nil, err
	}

	Q := subDense(QXU, 0, 0, n, n)
	N := subDense(QXU, 0, n, n, m)
	R := subDense(QXU, n, n, m, m)
	Aa, Ba := sys.A, sys.B
	if servo {
		if err := validateLqgWeight("QI", o.QI, p); err != nil {
			return nil, err
		}
		Aa, Ba, Q, N = lqgServoProblem(sys, Q, N, o.QI)
	}
	var ropts *RiccatiOpts
	if !allZeroDense(N) {
		ropts = &RiccatiOpts{S: N}
	}
	var kRes *RiccatiResult
	if continuous {
		kRes, err = Care(Aa, Ba, Q, R, ropts)
	} else {
		kRes, err = Dare(Aa, Ba, Q, R, ropts)
	}
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	Qn := subDense(QWV, 0, 0, n, n)
	Nn := subDense(QWV, 0, n, n, p)
	Rn := subDense(QWV, n, n, p, p)
	if allZeroDense(Nn) {
		Nn = nil
	}
	lRes, err := kalmanGain("Lqg", continuous, sys.A, eyeDense(n), sys.C, nil, Qn, Rn, Nn, nil)
	if err != nil {
		return nil, err
	}

	res := &LqgResult{K: subDense(kRes.K, 0, 0, m, n), L: lRes.K, Xc: kRes.X, Xf: lRes.X}
	if servo {
		res.Ki = subDense(kRes.K, 0, n, m, p)
	}
	var F *mat.Dense
	if o.Current {
		if F, err = res.currentGains(sys, Ba, R, Rn, Nn); err != nil {
			return nil, err
		}
	}
	if res.Controller, err = lqgController(sys, res, F, o.OneDOF); err != nil {
		return nil, err
	}
	return res, nil
}

func validateLqgWeight(name string, W *mat.Dense, dim int) error {
	if W == nil {
		return fmt.Errorf("Lqg: nil %s: %w", name, ErrDimensionMismatch)
	}
	r, c := W.Dims()
	if r != dim || c != dim {
		return fmt.Errorf("Lqg: %s is %dx%d, want %dx%d: %w", name, r, c, dim, dim, ErrDimensionMismatch)
	}
	if !isSymmetric(W, eps()*denseNorm(W)) {
		return fmt.Errorf("Lqg: %s: %w", name, ErrNotSymmetric)
	}
	return nil
}

func lqgIntegratorStep(sys *System) float64 {
	if sys.IsDiscrete() {
		return sys.Dt
	}
	return 1
}

func lqgServoProblem(sys *System, Q, N, QI *mat.Dense) (Aa, Ba, Qa, Na *mat.Dense) {
	n, m, p := sys.Dims()
	Aa, Ba = lqiAugmentation(sys)
	Qa = mat.NewDense(n+p, n+p, nil)
	setBlock(Qa, 0, 0, Q)
	setBlock(Qa, n, n, QI)
	Na = mat.NewDense(n+p, m, nil)
	setBlock(Na, 0, 0, N)
	return Aa, Ba, Qa, Na
}

// lqiAugmentation returns the MATLAB lqi plant augmented with the integral
// of r - y: forward Euler with step Ts in discrete time.
func lqiAugmentation(sys *System) (Aa, Ba *mat.Dense) {
	n, m, p := sys.Dims()
	h := lqgIntegratorStep(sys)
	na := n + p
	Aa = mat.NewDense(na, na, nil)
	setBlock(Aa, 0, 0, sys.A)
	var hc mat.Dense
	hc.Scale(-h, sys.C)
	setBlock(Aa, n, 0, &hc)
	if sys.IsDiscrete() {
		for i := n; i < na; i++ {
			Aa.Set(i, i, 1)
		}
	}
	Ba = mat.NewDense(na, m, nil)
	setBlock(Ba, 0, 0, sys.B)
	var hd mat.Dense
	hd.Scale(-h, sys.D)
	setBlock(Ba, n, 0, &hd)
	return Aa, Ba
}

func (res *LqgResult) currentGains(sys *System, Ba, R, Rn, Nn *mat.Dense) (*mat.Dense, error) {
	n, m, p := sys.Dims()
	PCt := mulDims(n, p, res.Xf, sys.C.T())
	S := mulDims(p, p, sys.C, PCt)
	S.Add(S, Rn)
	var Sinv mat.Dense
	if err := Sinv.Inverse(S); err != nil {
		return nil, fmt.Errorf("Lqg: innovation covariance: %w", ErrSingularEquation)
	}
	res.Mx = mulDims(n, p, PCt, &Sinv)
	res.Mw = mat.NewDense(n, p, nil)
	if Nn != nil {
		res.Mw.Mul(Nn, &Sinv)
	}

	na, _ := res.Xc.Dims()
	BtX := mulDims(m, na, Ba.T(), res.Xc)
	G := mulDims(m, m, BtX, Ba)
	G.Add(G, R)
	res.Kw = mat.NewDense(m, n, nil)
	if err := res.Kw.Solve(G, subDense(BtX, 0, 0, m, n)); err != nil {
		return nil, fmt.Errorf("Lqg: %w", ErrSingularR)
	}

	F := mulDims(m, p, res.K, res.Mx)
	F.Add(F, mulDims(m, p, res.Kw, res.Mw))
	return F, nil
}

// lqgController assembles the controller with states [x̂; xi] and
// u = Cu [x̂; xi] + Du y, where Cu = W[-(Kx - FC), -Ki], Du = -WF and
// W = (I - FD)⁻¹; F is nil unless Current.
func lqgController(sys *System, res *LqgResult, F *mat.Dense, oneDOF bool) (*System, error) {
	n, m, p := sys.Dims()
	q := 0
	if res.Ki != nil {
		q = p
	}
	nc := n + q

	Cu := mat.NewDense(m, nc, nil)
	var neg mat.Dense
	neg.Scale(-1, res.K)
	setBlock(Cu, 0, 0, &neg)
	if q > 0 {
		var negKi mat.Dense
		negKi.Scale(-1, res.Ki)
		setBlock(Cu, 0, n, &negKi)
	}
	Du := mat.NewDense(m, p, nil)
	if F != nil {
		cuX := subDense(Cu, 0, 0, m, n)
		cuX.Add(cuX, mulDims(m, n, F, sys.C))
		setBlock(Cu, 0, 0, cuX)
		Du.Scale(-1, F)

		W := eyeDense(m)
		W.Sub(W, mulDims(m, m, F, sys.D))
		var lu mat.LU
		lu.Factorize(W)
		if nearSingularCondition(lu.Cond()) {
			return nil, fmt.Errorf("Lqg: current estimator I - (Kx*Mx + Kw*Mw)*D singular: %w", ErrAlgebraicLoop)
		}
		var cu, du mat.Dense
		if err := lu.SolveTo(&cu, false, Cu); err != nil {
			return nil, fmt.Errorf("Lqg: %w", ErrAlgebraicLoop)
		}
		if err := lu.SolveTo(&du, false, Du); err != nil {
			return nil, fmt.Errorf("Lqg: %w", ErrAlgebraicLoop)
		}
		Cu, Du = &cu, &du
	}

	BmLD := denseCopy(sys.B)
	BmLD.Sub(BmLD, mulDims(n, m, res.L, sys.D))

	Ac := mat.NewDense(nc, nc, nil)
	axx := mulDims(n, n, res.L, sys.C)
	axx.Sub(sys.A, axx)
	axx.Add(axx, mulDims(n, n, BmLD, subDense(Cu, 0, 0, m, n)))
	setBlock(Ac, 0, 0, axx)
	if q > 0 {
		setBlock(Ac, 0, n, mulDims(n, q, BmLD, subDense(Cu, 0, n, m, q)))
		if sys.IsDiscrete() {
			for i := n; i < nc; i++ {
				Ac.Set(i, i, 1)
			}
		}
	}
	By := denseCopy(res.L)
	By.Add(By, mulDims(n, p, BmLD, Du))

	h := lqgIntegratorStep(sys)
	var Bc, Dc *mat.Dense
	var inputNames []string
	switch {
	case q == 0:
		Bc, Dc = By, Du
		inputNames = copyStringSlice(sys.OutputName)
	case oneDOF:
		Bc = mat.NewDense(nc, p, nil)
		var negBy mat.Dense
		negBy.Scale(-1, By)
		setBlock(Bc, 0, 0, &negBy)
		for i := range p {
			Bc.Set(n+i, i, h)
		}
		Dc = mat.NewDense(m, p, nil)
		Dc.Scale(-1, Du)
	default:
		Bc = mat.NewDense(nc, 2*p, nil)
		setBlock(Bc, 0, p, By)
		for i := range p {
			Bc.Set(n+i, i, h)
			Bc.Set(n+i, p+i, -h)
		}
		Dc = mat.NewDense(m, 2*p, nil)
		setBlock(Dc, 0, p, Du)
		inputNames = concatStringSlices([][]string{nil, sys.OutputName}, []int{p, p})
	}

	ctrl, err := New(Ac, Bc, Cu, Dc, sys.Dt)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}
	ctrl.InputName = inputNames
	ctrl.OutputName = copyStringSlice(sys.InputName)
	return ctrl, nil
}
