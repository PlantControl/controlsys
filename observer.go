package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// Lqe computes the Kalman estimator gain via duality with LQR, as MATLAB
// lqe(A,G,C,Qn,Rn,Nn) (https://www.mathworks.com/help/control/ref/lqe.html)
// for dx/dt = Ax + Gw, y = Cx + v. It solves the continuous CARE for the dual
// system (A', C', G*Qn*G', Rn) with cross term G*Nn and returns observer gain
// L (n×p) such that eig(A - L*C) is stable.
//
// A is n×n, G is n×g (noise input), C is p×n, Qn is g×g, Rn is p×p and Nn,
// the g×p noise cross-covariance E{w v'}, may be nil for 0. opts carries only
// the Workspace (sized NewRiccatiWorkspace(n, p)); opts.S and opts.E return
// ErrOptionUnsupported.
func Lqe(A, G, C, Qn, Rn, Nn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	for _, m := range []struct {
		name string
		m    *mat.Dense
	}{{"A", A}, {"G", G}, {"C", C}} {
		if err := requireFiniteDense("Lqe", m.name, m.m); err != nil {
			return nil, err
		}
	}
	if err := rejectKalmanOpts("Lqe", opts); err != nil {
		return nil, err
	}
	na, nac := A.Dims()
	if na != nac {
		return nil, fmt.Errorf("Lqe: A is %d×%d, want square: %w", na, nac, ErrDimensionMismatch)
	}
	n := na
	ng, g := G.Dims()
	if ng != n {
		return nil, fmt.Errorf("Lqe: G has %d rows, want %d: %w", ng, n, ErrDimensionMismatch)
	}
	p, cn := C.Dims()
	if cn != n {
		return nil, fmt.Errorf("Lqe: C has %d columns, want %d: %w", cn, n, ErrDimensionMismatch)
	}
	if err := validateCovarianceRole("Lqe", covarianceProcessNoise, Qn, g); err != nil {
		return nil, err
	}
	if err := validateCovarianceRole("Lqe", covarianceMeasurementNoise, Rn, p); err != nil {
		return nil, err
	}
	if err := validateNoiseCrossCovariance("Lqe", Nn, g, p); err != nil {
		return nil, err
	}

	if n == 0 {
		return nil, fmt.Errorf("Lqe: no states: %w", ErrDimensionMismatch)
	}
	return kalmanGain("Lqe", true, A, nil, G, C, nil, Qn, Rn, Nn, opts)
}

// rejectKalmanOpts keeps opts to the Workspace: the noise cross-covariance is
// the positional Nn argument, as in MATLAB.
func rejectKalmanOpts(op string, opts *RiccatiOpts) error {
	if opts == nil {
		return nil
	}
	if opts.S != nil {
		return fmt.Errorf("%s: opts.S: pass the noise cross-covariance as Nn: %w", op, ErrOptionUnsupported)
	}
	if opts.E != nil {
		return fmt.Errorf("%s: opts.E: %w", op, ErrOptionUnsupported)
	}
	return nil
}

// Kalman computes the Kalman filter gain for a state-space system, matching
// MATLAB kalman for the plant
//
//	x' = Ax + Gw,  y = Cx + Hw + v
//
// where every input of sys is a noise input w, so G = B and H = D. With
// Rbar = Rn + H*N + N'*H' + H*Qn*H' and Nbar = G*(Qn*H' + N), the gain is
// L = (P*C' + Nbar)*Rbar⁻¹ in continuous time and the predictor gain
// L = (A*P*C' + Nbar)*(C*P*C' + Rbar)⁻¹ in discrete time, with X = P.
// Rbar must be positive definite.
//
// The arguments follow MATLAB kalman(sys,Qn,Rn,Nn)
// (https://www.mathworks.com/help/control/ref/ss.kalman.html): Qn is m×m
// (process noise covariance), Rn is p×p (measurement noise covariance) and
// Nn, the m×p cross-covariance E{w v'}, may be nil for 0. opts carries only
// the Workspace (sized NewRiccatiWorkspace(n, p)); opts.S and opts.E return
// ErrOptionUnsupported. Plants with delays are rejected with
// ErrDelayUnsupported, as MATLAB requires a delay-free (Padé/absorbDelay)
// model.
//
// A descriptor plant E x' = Ax + Gw with nonsingular E is solved with the
// dual generalized Riccati equation (Care/Dare with opts.E = E'). X = P is
// the estimation error covariance, equal to that of the explicit model
// (E⁻¹A, E⁻¹G, C, H), and L is the gain of the descriptor estimator
//
//	E x̂' = Ax̂ + Bu + L(y - Cx̂ - Du)
//
// that Estim and Reg build from sys, i.e. E times the explicit-model gain.
// MATLAB's kalman documentation does not specify descriptor models; this is
// the form consistent with Estim. Singular E returns ErrDescriptorSingular.
func Kalman(sys *System, Qn, Rn, Nn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := requireSystem("Kalman", sys); err != nil {
		return nil, err
	}
	if err := rejectKalmanOpts("Kalman", opts); err != nil {
		return nil, err
	}
	policy, err := newControllerObserverPolicy(sys, "Kalman")
	if err != nil {
		return nil, err
	}
	if policy.p == 0 {
		return nil, fmt.Errorf("Kalman: model has no measured outputs: %w", ErrDimensionMismatch)
	}
	if err := policy.validateNoise(Qn, Rn); err != nil {
		return nil, err
	}
	if err := validateNoiseCrossCovariance("Kalman", Nn, policy.m, policy.p); err != nil {
		return nil, err
	}
	return kalmanGain("Kalman", sys.IsContinuous(), sys.A, sys.E, sys.B, sys.C, sys.D, Qn, Rn, Nn, opts)
}

func validateNoiseCrossCovariance(op string, Nn *mat.Dense, g, p int) error {
	if Nn == nil {
		return nil
	}
	if err := requireFiniteDense(op, "Nn", Nn); err != nil {
		return err
	}
	if r, c := Nn.Dims(); r != g || c != p {
		return fmt.Errorf("%s: Nn is %d×%d, want %d×%d: %w", op, r, c, g, p, ErrDimensionMismatch)
	}
	return nil
}

func kalmanGain(context string, continuous bool, A, E, G, C, H, Qn, Rn, Nn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	n, _ := A.Dims()
	g := Qn.RawMatrix().Rows
	p, _ := C.Dims()
	Qbar, At, Ct := dualRiccatiSetup(A, G, C, Qn, n, g, p)

	if allZeroDense(H) {
		H = nil
	}
	Rbar := Rn
	var Nbar *mat.Dense
	if H != nil || Nn != nil {
		QHtN := mat.NewDense(g, p, nil)
		if H != nil {
			QHtN.Mul(Qn, H.T())
		}
		if Nn != nil {
			QHtN.Add(QHtN, Nn)
		}
		Nbar = mulDense(G, QHtN)
		if H != nil {
			HQHtN := mulDense(H, QHtN)
			var HQHt mat.Dense
			HQHt.Mul(mulDense(H, Qn), H.T())
			Rbar = mat.NewDense(p, p, nil)
			Rbar.Add(HQHtN, HQHtN.T())
			Rbar.Sub(Rbar, &HQHt)
			Rbar.Add(Rbar, Rn)
			rbRaw := Rbar.RawMatrix()
			symmetrize(rbRaw.Data, p, rbRaw.Stride)
		}
	}

	ropts := &RiccatiOpts{S: Nbar}
	if !isIdentityDescriptor(E) {
		ropts.E = mat.DenseCopyOf(E.T())
	}
	if opts != nil {
		ropts.Workspace = opts.Workspace
	}
	var res *RiccatiResult
	var err error
	if continuous {
		res, err = Care(At, Ct, Qbar, Rbar, ropts)
	} else {
		res, err = Dare(At, Ct, Qbar, Rbar, ropts)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", context, err)
	}
	return &RiccatiResult{X: res.X, K: transposeGain(res.K), Eig: res.Eig, Rcnd: res.Rcnd}, nil
}

// Kalmd computes the discrete Kalman filter gain from a continuous plant
// using Van Loan's method for noise covariance discretization.
//
// sys must be continuous. Every input is a process noise input (G = B), as in
// MATLAB kalmd's model x' = Ax + Gw, y = Cx + v: noise feedthrough D ≠ 0 is
// rejected with ErrNoiseFeedthrough, and plants with delays with
// ErrDelayUnsupported. Qn is m×m, Rn is p×p, dt > 0; Qd is discretized with
// Van Loan's method and Rd = Rn/dt.
// opts.S and opts.E are rejected with ErrOptionUnsupported; opts.Workspace is
// used for the discrete Riccati solve and must be sized NewRiccatiWorkspace(n, p).
// A descriptor plant with nonsingular E is discretized as its explicit model
// x' = E⁻¹Ax + E⁻¹Bw, so the result is that of the explicit discrete model;
// singular E returns ErrDescriptorSingular.
func Kalmd(sys *System, Qn, Rn *mat.Dense, dt float64, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := requireSystem("Kalmd", sys); err != nil {
		return nil, err
	}
	if opts != nil && opts.S != nil {
		return nil, fmt.Errorf("Kalmd: cross-term S: %w", ErrOptionUnsupported)
	}
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Kalmd: %w", ErrWrongDomain)
	}
	if dt <= 0 || newTimeDomain(dt).validateSampleTime() != nil {
		return nil, fmt.Errorf("Kalmd: dt %g: %w", dt, ErrInvalidSampleTime)
	}
	policy, err := newControllerObserverPolicy(sys, "Kalmd")
	if err != nil {
		return nil, err
	}
	if policy.p == 0 {
		return nil, fmt.Errorf("Kalmd: model has no measured outputs: %w", ErrDimensionMismatch)
	}
	if err := policy.validateNoise(Qn, Rn); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if !allZeroDense(sys.D) {
		return nil, fmt.Errorf("Kalmd: %w", ErrNoiseFeedthrough)
	}
	if err := policy.rejectOptsE(opts); err != nil {
		return nil, err
	}
	A, G := sys.A, sys.B
	if !isIdentityDescriptor(sys.E) {
		var lu mat.LU
		lu.Factorize(sys.E)
		A, G = new(mat.Dense), new(mat.Dense)
		if err := lu.SolveTo(A, false, sys.A); err != nil {
			return nil, fmt.Errorf("Kalmd: %w", ErrDescriptorSingular)
		}
		if err := lu.SolveTo(G, false, sys.B); err != nil {
			return nil, fmt.Errorf("Kalmd: %w", ErrDescriptorSingular)
		}
	}

	GQnGt := inputNoiseIntensity(G, Qn, n, m)

	// Van Loan augmented matrix: F = [[-A, GQnGt], [0, A']] * dt
	nn := 2 * n
	F := mat.NewDense(nn, nn, nil)
	aRaw := A.RawMatrix()
	gRaw := GQnGt.RawMatrix()
	fRaw := F.RawMatrix()
	for i := range n {
		for j := range n {
			fRaw.Data[i*fRaw.Stride+j] = -aRaw.Data[i*aRaw.Stride+j] * dt
			fRaw.Data[i*fRaw.Stride+n+j] = gRaw.Data[i*gRaw.Stride+j] * dt
			fRaw.Data[(n+i)*fRaw.Stride+n+j] = aRaw.Data[j*aRaw.Stride+i] * dt
		}
	}

	var eF mat.Dense
	eF.Exp(F)
	efRaw := eF.RawMatrix()

	// Ad = eF[n:2n, n:2n]'
	adData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			adData[i*n+j] = efRaw.Data[(n+j)*efRaw.Stride+n+i]
		}
	}
	Ad := mat.NewDense(n, n, adData)

	// Qd = Ad * eF[0:n, n:2n]
	f12Data := make([]float64, n*n)
	copyBlock(f12Data, n, 0, 0, efRaw.Data, efRaw.Stride, 0, n, n, n)
	F12 := mat.NewDense(n, n, f12Data)
	Qd := mat.NewDense(n, n, nil)
	Qd.Mul(Ad, F12)

	// Symmetrize Qd
	qdRaw := Qd.RawMatrix()
	symmetrize(qdRaw.Data, n, qdRaw.Stride)

	// Rd = Rn / dt
	Rd := mat.NewDense(p, p, nil)
	rnRaw := Rn.RawMatrix()
	rdRaw := Rd.RawMatrix()
	for i := range p {
		for j := range p {
			rdRaw.Data[i*rdRaw.Stride+j] = rnRaw.Data[i*rnRaw.Stride+j] / dt
		}
	}

	// Solve discrete Kalman: Dare(Ad', C', Qd, Rd), transpose K
	adtData := make([]float64, n*n)
	adR := Ad.RawMatrix()
	for i := range n {
		for j := range n {
			adtData[i*n+j] = adR.Data[j*adR.Stride+i]
		}
	}
	Adt := mat.NewDense(n, n, adtData)

	cRaw := sys.C.RawMatrix()
	ctData := make([]float64, n*p)
	for i := range n {
		for j := range p {
			ctData[i*p+j] = cRaw.Data[j*cRaw.Stride+i]
		}
	}
	Ct := mat.NewDense(n, p, ctData)

	res, err := Dare(Adt, Ct, Qd, Rd, opts)
	if err != nil {
		return nil, fmt.Errorf("Kalmd: %w", err)
	}

	L := transposeGain(res.K)

	return &RiccatiResult{X: res.X, K: L, Eig: res.Eig, Rcnd: res.Rcnd}, nil
}

// Estim constructs the state estimator of a plant with observer gain L, as
// MATLAB estim(sys,L,sensors,known)
// (https://www.mathworks.com/help/control/ref/ss.estim.html). sensors indexes
// the measured outputs y (nil means all) and known the deterministic inputs u
// (nil means none: every input is stochastic, as MATLAB estim(sys,L)).
// Indices are 0-based, distinct and in range, else ErrInvalidArgument. With
// C2, D22 the rows of C, D for sensors and B2 the columns of B for known, the
// estimator is
//
//	x̂' = (A - L·C2)x̂ + (B2 - L·D22)u + L·y,  [ŷ; x̂] = [C2; I]x̂ + [D22; 0]u
//
// (x̂[k+1] for discrete models), with inputs [u; y] and outputs [ŷ; x̂]. L
// is n×len(sensors). A descriptor E is carried over (E x̂' = ...). Input
// delays of known inputs carry over; output, I/O-matrix and internal delays
// are rejected with ErrDelayUnsupported.
func Estim(sys *System, L *mat.Dense, sensors, known []int) (*System, error) {
	if err := requireSystem("Estim", sys); err != nil {
		return nil, err
	}
	if err := requireFiniteDense("Estim", "L", L); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	if n == 0 {
		return nil, fmt.Errorf("Estim: system has no states: %w", ErrDimensionMismatch)
	}
	if delaySliceHasNonzero(sys.OutputDelay) || delayMatrixHasNonzero(sys.Delay) || sys.HasInternalDelay() {
		return nil, fmt.Errorf("Estim: %w", ErrDelayUnsupported)
	}
	if sensors == nil {
		sensors = identityIndices(p)
	}
	if err := validateChannelIndices("Estim", "sensors", sensors, p); err != nil {
		return nil, err
	}
	if len(sensors) == 0 {
		return nil, fmt.Errorf("Estim: sensors is empty: %w", ErrInvalidArgument)
	}
	if err := validateChannelIndices("Estim", "known", known, m); err != nil {
		return nil, err
	}
	ps, mk := len(sensors), len(known)
	if lr, lc := L.Dims(); lr != n || lc != ps {
		return nil, fmt.Errorf("Estim: L is %d×%d, want %d×%d: %w", lr, lc, n, ps, ErrDimensionMismatch)
	}

	C2 := mat.NewDense(ps, n, nil)
	for i, row := range sensors {
		C2.SetRow(i, mat.Row(nil, row, sys.C))
	}
	B2 := newDense(n, mk)
	D22 := newDense(ps, mk)
	for j, col := range known {
		for i := range n {
			B2.Set(i, j, sys.B.At(i, col))
		}
		for i, row := range sensors {
			D22.Set(i, j, sys.D.At(row, col))
		}
	}

	Ae := mulDims(n, n, L, C2)
	Ae.Sub(sys.A, Ae)

	Be := newDense(n, mk+ps)
	if mk > 0 {
		BmLD := mulDims(n, mk, L, D22)
		BmLD.Sub(B2, BmLD)
		setBlock(Be, 0, 0, BmLD)
	}
	setBlock(Be, 0, mk, L)

	Ce := mat.NewDense(ps+n, n, nil)
	setBlock(Ce, 0, 0, C2)
	for i := range n {
		Ce.Set(ps+i, i, 1)
	}

	De := newDense(ps+n, mk+ps)
	if mk > 0 {
		setBlock(De, 0, 0, D22)
	}

	result, err := New(Ae, Be, Ce, De, sys.Dt)
	if err != nil {
		return nil, fmt.Errorf("Estim: %w", err)
	}
	result.E = copyDescriptorE(sys.E)
	if delaySliceHasNonzero(sys.InputDelay) {
		result.InputDelay = make([]float64, mk+ps)
		for j, col := range known {
			result.InputDelay[j] = sys.InputDelay[col]
		}
	}
	inNames, outNames := selectNames(sys.InputName, known), selectNames(sys.OutputName, sensors)
	result.InputName = concatStringSlices([][]string{inNames, outNames}, []int{mk, ps})
	result.OutputName = concatStringSlices([][]string{outNames, sys.StateName}, []int{ps, n})
	return result, nil
}

func identityIndices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

func validateChannelIndices(op, name string, idx []int, count int) error {
	seen := make(map[int]bool, len(idx))
	for _, i := range idx {
		if i < 0 || i >= count || seen[i] {
			return fmt.Errorf("%s: %s index %d out of range [0,%d) or repeated: %w", op, name, i, count, ErrInvalidArgument)
		}
		seen[i] = true
	}
	return nil
}

// selectNames picks names[idx]; nil names stay nil.
func selectNames(names []string, idx []int) []string {
	if names == nil {
		return nil
	}
	out := make([]string, len(idx))
	for k, i := range idx {
		if i < len(names) {
			out[k] = names[i]
		}
	}
	return out
}

// Reg constructs a regulator (observer-based controller) from a plant,
// state-feedback gain K, and observer gain L.
// The controller takes y (p) as input and produces u (m) as output.
//
// K is m×n, L is n×p. Returns system with n states, p inputs, m outputs.
// A descriptor E is carried over. Plants with delays are rejected with
// ErrDelayUnsupported.
func Reg(sys *System, K, L *mat.Dense) (*System, error) {
	n, m, p, err := validateRegulatorGains("Reg", sys, K, L)
	if err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("Reg: %w", ErrDelayUnsupported)
	}

	// Ar = A - B*K - L*C + L*D*K
	BK := mulDense(sys.B, K)
	LC := mulDense(L, sys.C)
	DK := mulDense(sys.D, K)
	LDK := mulDense(L, DK)

	Ar := mat.NewDense(n, n, nil)
	Ar.Sub(sys.A, BK)
	Ar.Sub(Ar, LC)
	Ar.Add(Ar, LDK)

	Br := denseCopy(L)

	Cr := mat.NewDense(m, n, nil)
	Cr.Scale(-1, K)

	Dr := mat.NewDense(m, p, nil)

	result, err := New(Ar, Br, Cr, Dr, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	result.InputName = copyStringSlice(sys.OutputName)
	result.OutputName = copyStringSlice(sys.InputName)
	return result, nil
}

func transposeGain(K *mat.Dense) *mat.Dense {
	kr, kc := K.Dims()
	lData := make([]float64, kc*kr)
	kRaw := K.RawMatrix()
	for i := range kr {
		src := kRaw.Data[i*kRaw.Stride:]
		for j := range kc {
			lData[j*kr+i] = src[j]
		}
	}
	return mat.NewDense(kc, kr, lData)
}

func dualRiccatiSetup(A, B, C, Qn *mat.Dense, n, m, p int) (GQnGt, At, Ct *mat.Dense) {
	GQnGt = inputNoiseIntensity(B, Qn, n, m)

	aRaw := A.RawMatrix()
	atData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			atData[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}
	At = mat.NewDense(n, n, atData)

	cRaw := C.RawMatrix()
	ctData := make([]float64, n*p)
	for i := range n {
		for j := range p {
			ctData[i*p+j] = cRaw.Data[j*cRaw.Stride+i]
		}
	}
	Ct = mat.NewDense(n, p, ctData)
	return GQnGt, At, Ct
}
