package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// Lqe computes the Kalman estimator gain via duality with LQR, matching
// MATLAB lqe(A,G,C,Qn,Rn,N) for dx/dt = Ax + Gw, y = Cx + v.
// It solves the continuous CARE for the dual system (A', C', G*Qn*G', Rn) with
// cross term G*N and returns observer gain L (n×p) such that eig(A - L*C) is
// stable.
//
// A is n×n, G is n×g (noise input), C is p×n, Qn is g×g, Rn is p×p.
// opts.S, when set, is the g×p noise cross-covariance N = E{w v'}.
func Lqe(A, G, C, Qn, Rn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	n := na
	ng, g := G.Dims()
	if ng != n {
		return nil, ErrDimensionMismatch
	}
	p, cn := C.Dims()
	if cn != n {
		return nil, ErrDimensionMismatch
	}
	if err := validateCovarianceRole("Lqe", covarianceProcessNoise, Qn, g); err != nil {
		return nil, err
	}
	if err := validateCovarianceRole("Lqe", covarianceMeasurementNoise, Rn, p); err != nil {
		return nil, err
	}
	Nn, err := noiseCrossCovariance("Lqe", opts, g, p)
	if err != nil {
		return nil, err
	}

	if n == 0 {
		return &RiccatiResult{X: &mat.Dense{}, K: &mat.Dense{}, Eig: nil}, nil
	}
	return kalmanGain("Lqe", true, A, G, C, nil, Qn, Rn, Nn, opts)
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
// Qn is m×m (process noise covariance), Rn is p×p (measurement noise
// covariance). opts.S, when set, is the m×p cross-covariance N = E{w v'}
// (MATLAB's Nn). Plants with delays are rejected with ErrDelayUnsupported,
// as MATLAB requires a delay-free (Padé/absorbDelay) model.
func Kalman(sys *System, Qn, Rn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
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
	Nn, err := noiseCrossCovariance("Kalman", opts, policy.m, policy.p)
	if err != nil {
		return nil, err
	}
	return kalmanGain("Kalman", sys.IsContinuous(), sys.A, sys.B, sys.C, sys.D, Qn, Rn, Nn, opts)
}

func noiseCrossCovariance(context string, opts *RiccatiOpts, g, p int) (*mat.Dense, error) {
	if opts == nil || opts.S == nil {
		return nil, nil
	}
	r, c := opts.S.Dims()
	if r != g || c != p {
		return nil, fmt.Errorf("%s: noise cross-covariance is %dx%d, want %dx%d: %w", context, r, c, g, p, ErrDimensionMismatch)
	}
	return opts.S, nil
}

func kalmanGain(context string, continuous bool, A, G, C, H, Qn, Rn, Nn *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
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
// opts.S is rejected with ErrOptionUnsupported; opts.Workspace is used for the
// discrete Riccati solve and must be sized NewRiccatiWorkspace(n, p).
func Kalmd(sys *System, Qn, Rn *mat.Dense, dt float64, opts *RiccatiOpts) (*RiccatiResult, error) {
	if opts != nil && opts.S != nil {
		return nil, fmt.Errorf("Kalmd: cross-term S: %w", ErrOptionUnsupported)
	}
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Kalmd: %w", ErrWrongDomain)
	}
	if dt <= 0 || newTimeDomain(dt).validateSampleTime() != nil {
		return nil, ErrInvalidSampleTime
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

	GQnGt := inputNoiseIntensity(sys.B, Qn, n, m)

	// Van Loan augmented matrix: F = [[-A, GQnGt], [0, A']] * dt
	nn := 2 * n
	F := mat.NewDense(nn, nn, nil)
	aRaw := sys.A.RawMatrix()
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

// Estim constructs an estimator system from a plant and observer gain L.
// The estimator takes [u; y] as input and produces [y_hat; x_hat] as output.
//
// L is n×p. Returns system with n states, (m+p) inputs, (p+n) outputs.
// A descriptor E is carried over (E x̂' = A x̂ + ...). Plant input delays are
// applied to the u inputs; output, I/O-matrix and internal delays are rejected
// with ErrDelayUnsupported.
func Estim(sys *System, L *mat.Dense) (*System, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		return nil, fmt.Errorf("Estim: system has no states: %w", ErrDimensionMismatch)
	}
	if delaySliceHasNonzero(sys.OutputDelay) || delayMatrixHasNonzero(sys.Delay) || sys.HasInternalDelay() {
		return nil, fmt.Errorf("Estim: %w", ErrDelayUnsupported)
	}
	lr, lc := L.Dims()
	if lr != n || lc != p {
		return nil, ErrDimensionMismatch
	}

	// Ae = A - L*C
	Ae := mulDims(n, n, L, sys.C)
	Ae.Sub(sys.A, Ae)

	// Be = [B - L*D, L]
	BmLD := mulDims(n, m, L, sys.D)
	if m > 0 {
		BmLD.Sub(sys.B, BmLD)
	}

	mp := m + p
	Be := newDense(n, mp)
	setBlock(Be, 0, 0, BmLD)
	setBlock(Be, 0, m, L)

	// Ce = [C; I_n]
	pn := p + n
	Ce := mat.NewDense(pn, n, nil)
	setBlock(Ce, 0, 0, sys.C)
	for i := range n {
		Ce.Set(p+i, i, 1)
	}

	// De = [D, 0; 0, 0]
	De := newDense(pn, mp)
	setBlock(De, 0, 0, sys.D)

	result, err := New(Ae, Be, Ce, De, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	if delaySliceHasNonzero(sys.InputDelay) {
		result.InputDelay = make([]float64, mp)
		copy(result.InputDelay, sys.InputDelay)
	}
	result.InputName = concatStringSlices([][]string{sys.InputName, sys.OutputName}, []int{m, p})
	result.OutputName = concatStringSlices([][]string{sys.OutputName, sys.StateName}, []int{p, n})
	return result, nil
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
	return
}
