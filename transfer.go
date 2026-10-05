package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	gonumLapack "plantcontrol.org/v1/gonum/lapack/gonum"
	"plantcontrol.org/v1/gonum/mat"
)

var impl gonumLapack.Implementation

type TransferFunc struct {
	Num        [][][]float64
	Den        [][]float64
	Delay      [][]float64 // [p][m]; nil = no delay
	Dt         float64
	InputName  []string
	OutputName []string
}

// Copy returns a deep copy of the transfer-function model.
func (tf *TransferFunc) Copy() *TransferFunc {
	if tf == nil {
		return nil
	}
	return &TransferFunc{
		Num:        copyFloatTensor(tf.Num),
		Den:        copyFloatRows(tf.Den),
		Delay:      copyFloatRows(tf.Delay),
		Dt:         tf.Dt,
		InputName:  copyStringSlice(tf.InputName),
		OutputName: copyStringSlice(tf.OutputName),
	}
}

func (tf *TransferFunc) Dims() (p, m int) {
	p = len(tf.Den)
	if p > 0 && len(tf.Num) > 0 && len(tf.Num[0]) > 0 {
		m = len(tf.Num[0])
	}
	return
}

func (tf *TransferFunc) validateShape() (p, m int, err error) {
	return validateTransferChannelShape(tf)
}

func (tf *TransferFunc) Eval(s complex128) [][]complex128 {
	p, m := tf.Dims()
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		dv := Poly(tf.Den[i]).Eval(s)
		for j := range m {
			nv := Poly(tf.Num[i][j]).Eval(s)
			h := nv / dv
			if tf.Delay != nil && tf.Delay[i][j] != 0 {
				tau := tf.Delay[i][j]
				if tf.Dt == 0 {
					h *= cmplx.Exp(-s * complex(tau, 0))
				} else {
					d := int(math.Round(tau))
					for range d {
						h /= s
					}
				}
			}
			result[i][j] = h
		}
	}
	return result
}

func (tf *TransferFunc) evalInto(s complex128, dst []complex128) {
	p, m := tf.Dims()
	for i := range p {
		dv := Poly(tf.Den[i]).Eval(s)
		for j := range m {
			h := Poly(tf.Num[i][j]).Eval(s) / dv
			if tf.Delay != nil && tf.Delay[i][j] != 0 {
				tau := tf.Delay[i][j]
				if tf.Dt == 0 {
					h *= cmplx.Exp(-s * complex(tau, 0))
				} else {
					d := int(math.Round(tau))
					for range d {
						h /= s
					}
				}
			}
			dst[i*m+j] = h
		}
	}
}

func (tf *TransferFunc) EvalMulti(freqs []complex128) [][][]complex128 {
	result := make([][][]complex128, len(freqs))
	for k, s := range freqs {
		result[k] = tf.Eval(s)
	}
	return result
}

// TransferFuncOpts tolerances are absolute and apply to the balanced
// realization (see TransferFunction); zero selects a norm-relative default.
type TransferFuncOpts struct {
	ControllabilityTol float64
	ObservabilityTol   float64
}

type TransferFuncResult struct {
	TF           *TransferFunc
	MinimalOrder int
	RowDegrees   []int
}

func copyFloatTensor(src [][][]float64) [][][]float64 {
	if src == nil {
		return nil
	}
	dst := make([][][]float64, len(src))
	for i := range src {
		dst[i] = copyFloatRows(src[i])
	}
	return dst
}

// TransferFunction returns the rational transfer matrix with every external
// delay (IODelay, InputDelay, OutputDelay) folded into TransferFunc.Delay.
// Internal (LFT) delays have no TransferFunc form and are rejected.
//
// A is first balanced by an exact power-of-two similarity, which removes
// accuracy loss caused by poor state scaling. The polynomial form itself
// stays ill-conditioned for high order or wide dynamic range: coefficients
// of e.g. a 30-state chain spanning 1e-3..1e3 rad/s, or a discrete model
// with poles clustered near z = 1, can evaluate with O(1) relative error.
// As MATLAB advises ("Using the Right Model Representation"), evaluate
// such models with EvalFr/FreqResponse on the state-space form instead.
func (sys *System) TransferFunction(opts *TransferFuncOpts) (*TransferFuncResult, error) {
	res, err := sys.rationalTransferFunction(opts)
	if err != nil {
		return nil, err
	}
	if total := newDelayTopology(sys).totalExternal(true); total != nil {
		res.TF.Delay = denseToSlice2D(total)
	}
	return res, nil
}

// rationalTransferFunction returns the delay-free rational part; callers
// account for external delays themselves.
func (sys *System) rationalTransferFunction(opts *TransferFuncOpts) (*TransferFuncResult, error) {
	if err := newDescriptorPolicy(sys).requireStandard("TransferFunction"); err != nil {
		return nil, err
	}
	if sys.internalDelayCount() > 0 {
		return nil, fmt.Errorf("TransferFunction: internal delays: %w", ErrDelayNotRepresentable)
	}
	if opts == nil {
		opts = &TransferFuncOpts{}
	}
	return newRowRealizationConverter(sys, opts).convert()
}

type rowRealizationConverter struct {
	sys  *System
	opts *TransferFuncOpts
	n    int
	m    int
	p    int
	tf   *TransferFunc
	rows []int
	dRaw blas64.General
}

func newRowRealizationConverter(sys *System, opts *TransferFuncOpts) rowRealizationConverter {
	n, m, p := sys.Dims()

	tf := &TransferFunc{
		Num: make([][][]float64, p),
		Den: make([][]float64, p),
		Dt:  sys.Dt,
	}
	var dRaw blas64.General
	if sys.D != nil {
		dRaw = sys.D.RawMatrix()
	}
	return rowRealizationConverter{
		sys:  sys,
		opts: opts,
		n:    n,
		m:    m,
		p:    p,
		tf:   tf,
		rows: make([]int, p),
		dRaw: dRaw,
	}
}

func (c rowRealizationConverter) convert() (*TransferFuncResult, error) {
	if c.n == 0 || c.p == 0 || c.m == 0 {
		c.fillStaticRows()
		return c.result(0), nil
	}

	a, b, cm := c.balancedABC()
	stair, err := controllabilityStaircase(a, b, cm, c.opts.ControllabilityTol, false)
	if err != nil {
		return nil, err
	}
	ncont := stair.NCont

	if ncont == 0 {
		c.fillStaticRows()
		return c.result(0), nil
	}

	totalOrder := c.convertDynamicRows(stair, ncont)
	return c.result(totalOrder), nil
}

// balancedABC returns D⁻¹AD, D⁻¹B, CD for the exact power-of-two scaling D
// from Dgebal. The transfer function is unchanged, but the orthogonal
// staircase and Hessenberg reductions are only normwise backward stable, so
// without it small entries of badly scaled A (Padé cascades, stiff chains)
// are swamped and coefficients lose most of their digits.
func (c rowRealizationConverter) balancedABC() (a, b, cm *mat.Dense) {
	n, m, p := c.n, c.m, c.p
	aData := make([]float64, n*n)
	raw := c.sys.A.RawMatrix()
	copyStrided(aData, n, raw.Data, raw.Stride, n, n)
	scale := make([]float64, n)
	impl.Dgebal(lapack.Scale, n, aData, n, scale)
	b = mat.NewDense(n, m, nil)
	b.Copy(c.sys.B)
	cm = mat.NewDense(p, n, nil)
	cm.Copy(c.sys.C)
	bRaw, cRaw := b.RawMatrix(), cm.RawMatrix()
	for i, d := range scale {
		if d == 1 {
			continue
		}
		inv := 1 / d
		for j := range m {
			bRaw.Data[i*bRaw.Stride+j] *= inv
		}
		for r := range p {
			cRaw.Data[r*cRaw.Stride+i] *= d
		}
	}
	return mat.NewDense(n, n, aData), b, cm
}

func (c rowRealizationConverter) convertDynamicRows(stair *StaircaseResult, ncont int) int {
	Ac := extractSubmatrix(stair.A, 0, ncont, 0, ncont)
	Bc := extractSubmatrix(stair.B, 0, ncont, 0, c.m)
	Cc := extractSubmatrix(stair.C, 0, c.p, 0, ncont)
	var ws *tfWorkspace
	if ncont > 0 {
		ws = newTFWorkspace(ncont, c.m)
	}

	totalOrder := 0
	for i := 0; i < c.p; i++ {
		nobs, den, numCoeffs := ssToTFRow(Ac, Bc, Cc, i, ncont, c.m, c.opts.ObservabilityTol, ws)
		c.rows[i] = nobs
		totalOrder += nobs

		c.tf.Den[i] = den
		c.tf.Num[i] = make([][]float64, c.m)
		for j := 0; j < c.m; j++ {
			num := numCoeffs[j]
			dij := c.dRaw.Data[i*c.dRaw.Stride+j]
			if dij != 0 {
				numWithD := Poly(num).Add(Poly(den).Scale(dij))
				c.tf.Num[i][j] = []float64(numWithD)
			} else {
				c.tf.Num[i][j] = num
			}
		}
	}
	return totalOrder
}

func (c rowRealizationConverter) fillStaticRows() {
	for i := 0; i < c.p; i++ {
		c.tf.Den[i] = []float64{1}
		c.tf.Num[i] = make([][]float64, c.m)
		for j := 0; j < c.m; j++ {
			if c.p > 0 && c.m > 0 {
				c.tf.Num[i][j] = []float64{c.dRaw.Data[i*c.dRaw.Stride+j]}
			} else {
				c.tf.Num[i][j] = []float64{0}
			}
		}
	}
}

func (c rowRealizationConverter) result(order int) *TransferFuncResult {
	c.tf.InputName = copyStringSlice(c.sys.InputName)
	c.tf.OutputName = copyStringSlice(c.sys.OutputName)
	return &TransferFuncResult{TF: c.tf, MinimalOrder: order, RowDegrees: c.rows}
}

type tfWorkspace struct {
	aData    []float64
	bsimo    []float64
	cData    []float64
	workDlrf []float64
	workC    []float64
	tauHrd   []float64
	workHrd  []float64
	workOrm  []float64
}

func newTFWorkspace(n, m int) *tfWorkspace {
	ws := &tfWorkspace{
		aData:    make([]float64, n*n),
		bsimo:    make([]float64, n),
		cData:    make([]float64, m*n),
		workDlrf: make([]float64, max(n, 1)),
		workC:    make([]float64, max(m, 1)),
	}
	if n > 1 {
		ws.tauHrd = make([]float64, n-1)
		workQuery := make([]float64, 1)
		impl.Dgehrd(n, 0, n-1, ws.aData, n, ws.tauHrd, workQuery, -1)
		lwork := int(workQuery[0])
		impl.Dormhr(blas.Right, blas.NoTrans, m, n, 0, n-1, ws.aData, n, ws.tauHrd, ws.cData, n, workQuery, -1)
		if l2 := int(workQuery[0]); l2 > lwork {
			lwork = l2
		}
		ws.workHrd = make([]float64, lwork)
		ws.workOrm = ws.workHrd
	}
	return ws
}

func formDualSIMO(Ac, Bc, Cc *mat.Dense, row, n, m int, ws *tfWorkspace) {
	aData := ws.aData
	acRaw := Ac.RawMatrix()
	for i := range n {
		for j := range n {
			aData[i*n+j] = acRaw.Data[j*acRaw.Stride+i]
		}
	}
	ccRaw := Cc.RawMatrix()
	copy(ws.bsimo, ccRaw.Data[row*ccRaw.Stride:row*ccRaw.Stride+n])
	cData := ws.cData
	bcRaw := Bc.RawMatrix()
	for i := range m {
		for j := range n {
			cData[i*n+j] = bcRaw.Data[j*bcRaw.Stride+i]
		}
	}
}

func observabilityReduction(n, m int, ws *tfWorkspace, obsTol float64) (nobs int, b1, scaleB float64) {
	aData := ws.aData
	bsimo := ws.bsimo
	cData := ws.cData

	smlnum := math.SmallestNonzeroFloat64 / eps()
	bignum := 1.0 / smlnum

	maxA := 0.0
	for _, v := range aData[:n*n] {
		if a := math.Abs(v); a > maxA {
			maxA = a
		}
	}
	maxB := 0.0
	for _, v := range bsimo[:n] {
		if a := math.Abs(v); a > maxB {
			maxB = a
		}
	}

	if maxB == 0 {
		return 0, 0, 1
	}

	scaleA := 1.0
	if maxA > 0 && maxA < smlnum {
		scaleA = smlnum / maxA
	} else if maxA > bignum {
		scaleA = bignum / maxA
	}
	if scaleA != 1.0 {
		for i := range aData[:n*n] {
			aData[i] *= scaleA
		}
	}

	scaleB = 1.0
	if maxB < smlnum {
		scaleB = smlnum / maxB
	} else if maxB > bignum {
		scaleB = bignum / maxB
	}
	if scaleB != 1.0 {
		for i := range bsimo[:n] {
			bsimo[i] *= scaleB
		}
	}

	frobA := 0.0
	for _, v := range aData[:n*n] {
		frobA += v * v
	}
	frobA = math.Sqrt(frobA)

	norm1B := 0.0
	for _, v := range bsimo[:n] {
		norm1B += math.Abs(v)
	}

	tol := obsTol
	if tol == 0 {
		tol = float64(n) * eps() * math.Max(frobA, norm1B)
	}

	if norm1B <= tol {
		return 0, 0, scaleB
	}

	beta, tau := impl.Dlarfg(n, bsimo[0], bsimo[1:], 1)
	bsimo[0] = 1.0 // v[0]=1 for reflector

	impl.Dlarf(blas.Right, n, n, bsimo, 1, tau, aData, n, ws.workDlrf)
	impl.Dlarf(blas.Left, n, n, bsimo, 1, tau, aData, n, ws.workDlrf)
	impl.Dlarf(blas.Right, m, n, bsimo, 1, tau, cData, n, ws.workC)

	b1 = beta
	for i := range bsimo[:n] {
		bsimo[i] = 0
	}
	bsimo[0] = b1

	tauHrd := ws.tauHrd
	if n > 1 {
		impl.Dgehrd(n, 0, n-1, aData, n, tauHrd, ws.workHrd, len(ws.workHrd))
		impl.Dormhr(blas.Right, blas.NoTrans, m, n, 0, n-1, aData, n, tauHrd, cData, n, ws.workOrm, len(ws.workOrm))
	}

	for j := range n {
		for i := j + 2; i < n; i++ {
			aData[i*n+j] = 0
		}
	}

	tolScan := math.Max(tol, float64(n)*eps()*math.Max(frobA, math.Abs(b1/scaleB)*scaleB))
	nobs = n
	for j := 0; j < n-1; j++ {
		subdiag := aData[(j+1)*n+j]
		if math.Abs(subdiag) <= tolScan {
			nobs = j + 1
			break
		}
	}

	if nobs == 0 {
		return 0, 0, scaleB
	}

	if scaleA != 1.0 {
		invSA := 1.0 / scaleA
		for j := 0; j < nobs; j++ {
			for i := 0; i < nobs; i++ {
				aData[j*n+i] *= invSA
			}
		}
	}
	b1 /= scaleB

	return nobs, b1, scaleB
}

func vPolyRecurrence(aData []float64, n, nobs int) []Poly {
	wPolys := make([]Poly, nobs+1)
	wPolys[0] = Poly{1}
	mulBuf := make(Poly, 0, nobs+2)
	addBuf := make(Poly, 0, nobs+2)
	scaleBuf := make(Poly, 0, nobs+1)
	linPoly := Poly{1, 0}

	poolSize := (nobs + 1) * (nobs + 2) / 2
	pool := make([]float64, poolSize)
	offset := 0

	for k := 1; k <= nobs; k++ {
		r := nobs - k
		linPoly[1] = -aData[r*n+r]
		mulBuf = linPoly.MulTo(mulBuf, wPolys[k-1])
		for i := 1; i < k; i++ {
			hij := aData[r*n+r+i]
			if hij != 0 {
				scaleBuf = wPolys[k-1-i].ScaleTo(scaleBuf, -hij)
				addBuf, mulBuf = mulBuf, mulBuf.AddTo(addBuf, scaleBuf)
			}
		}
		wPolys[k] = pool[offset : offset+len(mulBuf)]
		copy(wPolys[k], mulBuf)
		offset += len(mulBuf)
	}

	return wPolys
}

func computeNumerators(cData []float64, n, nobs, m int, scale []float64, wPolys []Poly) [][]float64 {
	nums := make([][]float64, m)
	scaleBuf := make(Poly, 0, nobs+1)
	addBuf := make(Poly, 0, nobs+2)
	for j := range m {
		var num Poly
		for k := range nobs {
			coeff := cData[j*n+k] * scale[k]
			if coeff != 0 {
				scaleBuf = wPolys[nobs-1-k].ScaleTo(scaleBuf, coeff)
				if len(num) == 0 {
					num = make(Poly, len(scaleBuf))
					copy(num, scaleBuf)
				} else {
					addBuf, num = num, num.AddTo(addBuf, scaleBuf)
				}
			}
		}
		if len(num) == 0 {
			nums[j] = []float64{0}
		} else {
			nums[j] = []float64(num)
		}
	}
	return nums
}

func ssToTFRow(Ac, Bc, Cc *mat.Dense, row, ncont, m int, obsTol float64, ws *tfWorkspace) (int, []float64, [][]float64) {
	zeroReturn := func() (int, []float64, [][]float64) {
		den := []float64{1}
		nums := make([][]float64, m)
		for j := range m {
			nums[j] = []float64{0}
		}
		return 0, den, nums
	}

	if ncont == 0 {
		return zeroReturn()
	}

	n := ncont

	formDualSIMO(Ac, Bc, Cc, row, n, m, ws)

	nobs, b1, _ := observabilityReduction(n, m, ws, obsTol)
	if nobs == 0 {
		return zeroReturn()
	}

	aData := ws.aData

	scale := make([]float64, nobs)
	scale[0] = b1
	for k := 1; k < nobs; k++ {
		scale[k] = scale[k-1] * aData[k*n+(k-1)]
	}

	// Scale upper triangle so V-poly recurrence works on monic form
	for i := range nobs {
		for j := i + 1; j < nobs; j++ {
			aData[i*n+j] *= scale[j] / scale[i]
		}
	}

	wPolys := vPolyRecurrence(aData, n, nobs)
	den := []float64(wPolys[nobs])
	nums := computeNumerators(ws.cData, n, nobs, m, scale, wPolys)

	return nobs, den, nums
}

type StateSpaceOpts struct {
	MinimalTol float64
}

type StateSpaceResult struct {
	Sys          *System
	MinimalOrder int
	BlockSizes   []int
}

func (tf *TransferFunc) StateSpace(opts *StateSpaceOpts) (*StateSpaceResult, error) {
	p, m, err := tf.validateShape()
	if err != nil {
		return nil, err
	}

	if p == 0 || m == 0 {
		sys, _ := NewGain(&mat.Dense{}, tf.Dt)
		sys.InputName = copyStringSlice(tf.InputName)
		sys.OutputName = copyStringSlice(tf.OutputName)
		return &StateSpaceResult{Sys: sys, MinimalOrder: 0}, nil
	}

	if !tf.Isproper() {
		return nil, ErrImproperTF
	}

	degrees := make([]int, p)
	totalN := 0
	for i := range p {
		degrees[i] = len(tf.Den[i]) - 1
		totalN += degrees[i]
	}

	if totalN == 0 {
		D := mat.NewDense(p, m, nil)
		dRaw := D.RawMatrix()
		for i := range p {
			scale := 1.0 / tf.Den[i][0]
			for j := range m {
				dRaw.Data[i*dRaw.Stride+j] = scale * tf.Num[i][j][0]
			}
		}
		sys, _ := NewGain(D, tf.Dt)
		if tf.Delay != nil {
			sys.Delay = slice2DToDense(tf.Delay)
		}
		sys.InputName = copyStringSlice(tf.InputName)
		sys.OutputName = copyStringSlice(tf.OutputName)
		return &StateSpaceResult{Sys: sys, MinimalOrder: 0, BlockSizes: degrees}, nil
	}

	smlnum := math.SmallestNonzeroFloat64 / eps()
	bignum := 1.0 / smlnum

	A := mat.NewDense(totalN, totalN, nil)
	B := mat.NewDense(totalN, m, nil)
	C := mat.NewDense(p, totalN, nil)
	D := mat.NewDense(p, m, nil)

	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()
	dRaw := D.RawMatrix()

	ja := 0
	for i := range p {
		di := degrees[i]
		if di == 0 {
			scale := 1.0 / tf.Den[i][0]
			for j := range m {
				dRaw.Data[i*dRaw.Stride+j] = scale * tf.Num[i][j][0]
			}
			continue
		}

		leading := tf.Den[i][0]

		if math.Abs(leading) < smlnum {
			return nil, fmt.Errorf("row %d: %w", i, ErrSingularDenom)
		}

		umax := 0.0
		for j := range m {
			if v := math.Abs(tf.Num[i][j][0]); v > umax {
				umax = v
			}
		}

		if math.Abs(leading) < 1 && umax > math.Abs(leading)*bignum {
			return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
		}

		if di >= 1 {
			dmx := 0.0
			for k := 1; k <= di; k++ {
				if v := math.Abs(tf.Den[i][k]); v > dmx {
					dmx = v
				}
			}
			if math.Abs(leading) >= 1 {
				if umax > 1 && (dmx/math.Abs(leading)) > (bignum/umax) {
					return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
				}
			} else {
				if umax > 1 && dmx > (bignum*math.Abs(leading))/umax {
					return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
				}
			}
		}

		scale := 1.0 / leading

		for k := 0; k < di-1; k++ {
			aRaw.Data[(ja+k+1)*aRaw.Stride+(ja+k)] = 1
		}

		// Pad numerator to di+1 coefficients (left-pad with zeros)
		padNum := make([][]float64, m)
		for j := range m {
			padNum[j] = make([]float64, di+1)
			nj := len(tf.Num[i][j])
			off := di + 1 - nj
			for idx := range nj {
				if off+idx >= 0 {
					padNum[j][off+idx] = tf.Num[i][j][idx]
				}
			}
		}

		for k := range di {
			row := ja + di - 1 - k
			temp := -scale * tf.Den[i][k+1]
			aRaw.Data[row*aRaw.Stride+(ja+di-1)] = temp
			for j := range m {
				bRaw.Data[row*bRaw.Stride+j] = padNum[j][k+1] + temp*padNum[j][0]
			}
		}

		if ja+di < totalN {
			aRaw.Data[(ja+di)*aRaw.Stride+(ja+di-1)] = 0
		}

		for j := range m {
			dRaw.Data[i*dRaw.Stride+j] = scale * padNum[j][0]
		}

		// C row
		cRaw.Data[i*cRaw.Stride+(ja+di-1)] = scale

		ja += di
	}

	sys, err := newNoCopy(A, B, C, D, tf.Dt)
	if err != nil {
		return nil, err
	}

	if tf.Delay != nil {
		sys.Delay = slice2DToDense(tf.Delay)
	}

	sys.InputName = copyStringSlice(tf.InputName)
	sys.OutputName = copyStringSlice(tf.OutputName)
	return &StateSpaceResult{
		Sys:          sys,
		MinimalOrder: totalN,
		BlockSizes:   degrees,
	}, nil
}

// Isproper reports whether the delay-free part of sys, including internal
// delay channels, is proper. Explicit models always are; a descriptor model
// is proper iff its infinite eigenvalues are all nondynamic.
func (sys *System) Isproper() bool {
	if !sys.IsDescriptor() {
		return true
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if !luNearSingular(&lu) {
		return true
	}
	n, m, p := sys.Dims()
	N := sys.internalDelayCount()
	if m+N == 0 || p+N == 0 {
		return true
	}
	F, ok := regularDescriptorShift(sys.A, sys.E, n)
	if !ok {
		return false
	}
	B := newDense(n, m+N)
	C := newDense(p+N, n)
	setBlock(B, 0, 0, sys.B)
	setBlock(C, 0, 0, sys.C)
	if N > 0 {
		setBlock(B, 0, m, sys.LFT.B2)
		setBlock(C, p, 0, sys.LFT.C2)
	}
	var K, Fb mat.Dense
	if F.SolveTo(&K, false, sys.E) != nil || F.SolveTo(&Fb, false, B) != nil {
		return false
	}
	// With F = A-σE, K = F⁻¹E and ν = 1/(s-σ):
	// G(s) = D - ν·C(νI-K)⁻¹F⁻¹B, so G is proper iff that transfer has at
	// most a simple pole at ν = 0, i.e. K's zero eigenvalue is semisimple
	// in a minimal realization.
	aux, err := New(&K, &Fb, C, newDense(p+N, m+N), 0)
	if err != nil {
		return false
	}
	red, err := aux.Reduce(nil)
	if err != nil {
		return false
	}
	if red.Order == 0 {
		return true
	}
	Kr := red.Sys.A
	var K2 mat.Dense
	K2.Mul(Kr, Kr)
	return numericRank(Kr) == numericRank(&K2)
}

func regularDescriptorShift(A, E *mat.Dense, n int) (*mat.LU, bool) {
	var best *mat.LU
	bestCond := math.Inf(1)
	for _, sigma := range []float64{0, 1, -1, 0.618, -2.414, 3.7} {
		F := mat.NewDense(n, n, nil)
		F.Scale(-sigma, E)
		F.Add(A, F)
		lu := new(mat.LU)
		lu.Factorize(F)
		if c := lu.Cond(); c < bestCond {
			best, bestCond = lu, c
		}
	}
	if best == nil || nearSingularCondition(bestCond) {
		return nil, false
	}
	return best, true
}

func numericRank(M *mat.Dense) int {
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDNone) {
		return 0
	}
	sv := svd.Values(nil)
	if len(sv) == 0 || sv[0] == 0 {
		return 0
	}
	r, c := M.Dims()
	tol := float64(max(r, c)) * 1e3 * eps() * sv[0]
	rank := 0
	for _, v := range sv {
		if v > tol {
			rank++
		}
	}
	return rank
}

func (tf *TransferFunc) Isproper() bool {
	p, m := tf.Dims()
	for i := range p {
		denDeg := len(tf.Den[i]) - 1
		for j := range m {
			numDeg := len(tf.Num[i][j]) - 1
			if numDeg > denDeg {
				return false
			}
		}
	}
	return true
}
