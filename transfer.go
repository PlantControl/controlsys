package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	gonumLapack "plantcontrol.org/v1/gonum/lapack/gonum"
	"plantcontrol.org/v1/gonum/mat"
)

var impl gonumLapack.Implementation

// TransferFunc is a p×m transfer-function model, as MATLAB tf. Each output
// row i shares the denominator Den[i]; Num[i][j] is the numerator of channel
// (i, j). Coefficients are in descending powers of s (or z).
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

// Dims returns the number of outputs p and inputs m.
func (tf *TransferFunc) Dims() (p, m int) {
	p = len(tf.Den)
	if p > 0 && len(tf.Num) > 0 && len(tf.Num[0]) > 0 {
		m = len(tf.Num[0])
	}
	return
}

func (tf *TransferFunc) validateShape() (p, m int, err error) {
	p, m, err = validateTransferChannelShape(tf)
	if err != nil {
		return 0, 0, err
	}
	if err := newTimeDomain(tf.Dt).validateSampleTime(); err != nil {
		return 0, 0, err
	}
	return p, m, nil
}

func (tf *TransferFunc) validateFiniteCoeffs() error {
	for i, den := range tf.Den {
		if !finiteSlice(den) {
			return fmt.Errorf("denominator %d has non-finite coefficients: %w", i, ErrInvalidArgument)
		}
		for j, num := range tf.Num[i] {
			if !finiteSlice(num) {
				return fmt.Errorf("numerator (%d,%d) has non-finite coefficients: %w", i, j, ErrInvalidArgument)
			}
		}
	}
	return nil
}

// Eval evaluates the frequency response H(s) at the complex point s, as
// MATLAB evalfr(sys, s). Delays contribute exp(-s·tau) (continuous) or
// s^-tau (discrete, tau whole samples). A malformed model returns an error.
func (tf *TransferFunc) Eval(s complex128) ([][]complex128, error) {
	if err := tf.validateEval(); err != nil {
		return nil, fmt.Errorf("TransferFunc.Eval: %w", err)
	}
	return tf.eval(s), nil
}

func (tf *TransferFunc) validateEval() error {
	if _, _, err := tf.validateShape(); err != nil {
		return err
	}
	for i, row := range tf.Delay {
		for j, tau := range row {
			if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
				return fmt.Errorf("delay (%d,%d) is %g: %w", i, j, tau, ErrNegativeDelay)
			}
			if tf.Dt > 0 && tau != math.Trunc(tau) {
				return fmt.Errorf("delay (%d,%d) is %g samples: %w", i, j, tau, ErrFractionalDelay)
			}
		}
	}
	return nil
}

func (tf *TransferFunc) eval(s complex128) [][]complex128 {
	p, m := tf.Dims()
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		dv := Poly(tf.Den[i]).Eval(s)
		for j := range m {
			h := Poly(tf.Num[i][j]).Eval(s) / dv
			if tf.Delay != nil && tf.Delay[i][j] != 0 {
				tau := tf.Delay[i][j]
				if tf.Dt == 0 {
					h *= cmplx.Exp(-s * complex(tau, 0))
				} else {
					h *= cmplx.Pow(s, complex(-tau, 0))
				}
			}
			result[i][j] = h
		}
	}
	return result
}

// EvalMulti evaluates Eval at each point of freqs.
func (tf *TransferFunc) EvalMulti(freqs []complex128) ([][][]complex128, error) {
	if err := tf.validateEval(); err != nil {
		return nil, fmt.Errorf("TransferFunc.EvalMulti: %w", err)
	}
	result := make([][][]complex128, len(freqs))
	for k, s := range freqs {
		result[k] = tf.eval(s)
	}
	return result, nil
}

// TransferFuncOpts tolerances are absolute and apply to the balanced,
// I/O-scaled realization (see TransferFunction); zero selects a norm-relative default.
type TransferFuncOpts struct {
	ControllabilityTol float64
	ObservabilityTol   float64
}

// TransferFuncResult is the result of (*System).TransferFunction.
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
// Each output row is converted from the states it structurally sees, as
// MATLAB tf(ss) applies sminreal per channel, so exactly decoupled states
// never enter its denominator. [A B; C ·] is then balanced by an exact
// power-of-two state similarity, and each input and output scaled by a
// power of two toward ‖A‖, which removes accuracy loss caused by poor state
// or I/O scaling (a tiny but nonzero B or C is not dropped as
// uncontrollable) and makes the orders nearly independent of I/O scaling
// (rounding noise near the rank floor aside). When that scaling overflows
// a coefficient the unscaled realization is used, and when coefficients
// overflow float64 regardless the error is ErrOverflow. The polynomial form
// itself stays ill-conditioned for high order or wide dynamic range:
// coefficients of e.g. a 30-state chain spanning 1e-3..1e3 rad/s, or a
// discrete model with poles clustered near z = 1, can evaluate with O(1)
// relative error. As MATLAB advises ("Using the Right Model
// Representation"), evaluate such models with EvalFr/FreqResponse on the
// state-space form instead.
//
// A descriptor model is first reduced to explicit form as MATLAB tf(dss)
// does: nonsingular E is folded in, and singular E is reduced by eliminating
// the algebraic states (dss2ss). An improper model keeps its polynomial
// part, as MATLAB tf(dss): the s^k coefficients −C∞·N^k·A∞⁻¹B∞ of the
// nilpotent block (rounding-level ones dropped, as IsProper decides) are
// added to the transfer function of the explicit part, so the numerator
// degree exceeds the denominator's. MinimalOrder and RowDegrees then
// describe that explicit part.
func (sys *System) TransferFunction(opts *TransferFuncOpts) (*TransferFuncResult, error) {
	std, poly, err := transferStandardForm(sys)
	if err != nil {
		return nil, err
	}
	res, err := std.rationalTransferFunction(opts)
	if err != nil {
		return nil, err
	}
	addPolynomialPart(res.TF, poly)
	if total := newDelayTopology(sys).totalExternal(true); total != nil {
		res.TF.Delay = denseToSlice2D(total)
	}
	return res, nil
}

// transferStandardForm is conversionStandardForm keeping the polynomial part
// of an improper descriptor (see weierstrassReduceParts).
func transferStandardForm(sys *System) (*System, []*mat.Dense, error) {
	if !sys.IsDescriptor() {
		return sys, nil, nil
	}
	if explicit, err := sys.ToExplicit(); err == nil {
		return explicit, nil, nil
	}
	out, poly, err := sys.weierstrassReduceParts()
	if err == nil {
		if n, m, p := out.Dims(); n == 0 {
			err = storableStaticGain("descriptor has no dynamic states", p, m)
		}
	}
	if err != nil {
		return nil, nil, fmt.Errorf("TransferFunction: %w: %w", ErrDescriptorUnsupported, err)
	}
	return out, poly, nil
}

// addPolynomialPart adds Σ poly[k-1]·s^k to every channel of tf.
func addPolynomialPart(tf *TransferFunc, poly []*mat.Dense) {
	if len(poly) == 0 {
		return
	}
	for i, den := range tf.Den {
		for j := range tf.Num[i] {
			q := make(Poly, len(poly)+1)
			for k, c := range poly {
				q[len(poly)-1-k] = c.At(i, j)
			}
			start := 0
			for start < len(poly) && q[start] == 0 {
				start++
			}
			if start == len(poly) {
				continue
			}
			num := Poly(tf.Num[i][j]).Add(Poly(den).Mul(q[start:]))
			for len(num) > 1 && num[0] == 0 {
				num = num[1:]
			}
			tf.Num[i][j] = num
		}
	}
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
	// companion[i] reports that row i came from directRowTF on a companion
	// A, where its recurrence is exact.
	companion []bool
	dRaw      blas64.General
	// trims[i*m+j] holds the Markov trim candidates of entry (i, j), most
	// trimmed first; trimRows applies them after the realization is chosen.
	trims [][][]float64
	// keepObs disables the observability cut (see keepObservable).
	keepObs bool
	// noDirect disables the Hessenberg path of convertDynamicRows (tests).
	noDirect bool
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
		sys:       sys,
		opts:      opts,
		n:         n,
		m:         m,
		p:         p,
		tf:        tf,
		rows:      make([]int, p),
		companion: make([]bool, p),
		dRaw:      dRaw,
		trims:     make([][][]float64, p*m),
	}
}

// fresh returns an empty converter of sys with the options of c.
func (c rowRealizationConverter) fresh(sys *System) rowRealizationConverter {
	f := newRowRealizationConverter(sys, c.opts)
	f.noDirect = c.noDirect
	return f
}

func (c rowRealizationConverter) convert() (*TransferFuncResult, error) {
	if c.n == 0 || c.p == 0 || c.m == 0 {
		c.fillStaticRows()
		return c.result(0), nil
	}

	if groups := c.structuralRowGroups(); groups != nil {
		return c.convertRowGroups(groups)
	}
	pick, order, err := c.choose()
	if err != nil {
		return nil, err
	}
	if slices.Min(pick.rows) < c.n {
		pick, order = pick.keepObservable(order)
	}
	return pick.result(order), nil
}

// keepObservable replaces rows of c, a conversion in which a row lost
// states, by those of the full-order Dgebal conversion when c misses the
// response in the band of its own and of the full one's poles and zeros by
// more than 1e-6 and the full row misses it by 1e3× less. MATLAB tf(ss) takes den = poly(A)
// after only structural reduction, while the observability cut is by
// magnitude: the last Hessenberg subdiagonal of a slow companion is an exact
// product of tiny couplings, which the cut drops with the origin pole and a
// zero (U3H63X). A truly hidden mode leaves c matching everywhere, so it
// stays dropped. The band of c alone starts at its slowest kept pole, above
// a cut origin pole and its near-cancelling zero, where c can miss by less
// than 1e-6 (4XDPLI), so the full conversion is always made and the joint
// band decides. The default check points sit far above such poles, where the
// lost state barely shows; the band points below them are solved in
// double-double where GEPP rejects them, and the full conversion takes a
// row the staircase floor cut directly (see directRowTF) when it can.
func (c rowRealizationConverter) keepObservable(order int) (rowRealizationConverter, int) {
	if c.opts.ObservabilityTol != 0 {
		return c, order
	}
	c.trimRows()
	bd := newBalancedDense(c.sys, c.n, c.m, c.p)
	full := c.fresh(c.sys)
	full.keepObs = true
	fullOrder, _, err := full.convertFrom(false)
	if err != nil || fullOrder <= order {
		return c, order
	}
	full.trimRows()
	ck := sampleBandCheck(c.sys, bd, bandCheckPoints(c.sys, append(full.tf.polys(), c.tf.polys()...)), bd.e == nil)
	eKept, ok := ck.resolvedErr(c.tf)
	if !ok || !(eKept > 1e-6) {
		return c, order
	}
	// Rows are independent, so each takes the full row where that row is
	// the better one, unless it misses by over 10% a resolved point the kept
	// row matches to 1e-6: a direct row exact on a graded companion must not
	// carry along dense rows whose full conversion fixes the response below
	// the poles but loses it far above them. Better is 1e3× less worst miss,
	// or no point missed by more than twice the kept row's miss (or 1e-6) and
	// one missed 1e3× less: at the band edge the reference itself can miss a
	// small row by 1e-2, the same for both rows. A direct full row on a
	// companion A is exact, den = poly(A) as in MATLAB ss2tf, so it needs
	// no margin: below the poles of a reversed unit-row companion the
	// reference itself misses by up to 1e4 (4XDPLI).
	order = 0
	for i := range c.p {
		eK := ck.rowPointErrs(c.tf, i)
		eF := ck.rowPointErrs(full.tf, i)
		worstK, worstF, loses := 0.0, 0.0, false
		dominates, gains := true, false
		for k := range eK {
			if eK[k] < 0 || eF[k] < 0 {
				continue
			}
			worstK, worstF = max(worstK, eK[k]), max(worstF, eF[k])
			loses = loses || eK[k] <= 1e-6 && !(eF[k] <= 0.1)
			dominates = dominates && eF[k] <= max(2*eK[k], 1e-6)
			gains = gains || eK[k] > 1e-6 && eF[k] <= 1e-3*eK[k]
		}
		if !loses && worstK > 1e-6 && (full.companion[i] || worstF <= 1e-3*worstK || dominates && gains) {
			c.tf.Den[i], c.tf.Num[i], c.rows[i] = full.tf.Den[i], full.tf.Num[i], full.rows[i]
		}
		order += c.rows[i]
	}
	return c, order
}

// polys returns the row dens and the numerators.
func (tf *TransferFunc) polys() [][]float64 {
	var out [][]float64
	for i, den := range tf.Den {
		out = append(out, den)
		out = append(out, tf.Num[i]...)
	}
	return out
}

// bandCheckPoints adds jω, ω log-spaced over [r, R] and r/10, to the
// continuous points of newTFCheck, r and R estimating the smallest and
// largest nonzero root moduli over polys; discrete models get six points on
// the unit circle.
func bandCheckPoints(sys *System, polys [][]float64) []complex128 {
	var pts []complex128
	if !sys.IsContinuous() {
		for _, th := range []float64{0.05, 0.3, 0.7, 1.3, 1.9, 2.8} {
			pts = append(pts, cmplx.Rect(1, th))
		}
		return pts
	}
	rho := frobNorm(sys.A)
	if rho == 0 || math.IsInf(rho, 0) {
		rho = 1
	}
	for _, w := range []float64{1e-2, 1, 1e2} {
		pts = append(pts, complex(0, rho*w))
	}
	lo, hi := math.Inf(1), 0.0
	for _, p := range polys {
		p = trimLeadingFloatZeros(p)
		for len(p) > 1 && p[len(p)-1] == 0 {
			p = p[:len(p)-1]
		}
		d := len(p) - 1
		if d < 1 {
			continue
		}
		up, down := 0.0, 0.0
		for k := 1; k <= d; k++ {
			up = max(up, math.Pow(math.Abs(p[k]/p[0]), 1/float64(k)))
			down = max(down, math.Pow(math.Abs(p[d-k]/p[d]), 1/float64(k)))
		}
		if up > 0 && down > 0 && !math.IsInf(up, 0) && !math.IsInf(down, 0) {
			lo, hi = min(lo, 1/down), max(hi, up)
		}
	}
	if hi == 0 {
		return pts
	}
	nb := 8
	if hi <= lo {
		nb = 2
	}
	ratio := math.Pow(hi/lo, 1/float64(nb-1))
	for w := range nb {
		pts = append(pts, complex(0, lo*math.Pow(ratio, float64(w))))
	}
	return append(pts, complex(0, lo/10))
}

// resolvedErr is relErr per entry, relative to the entry, over the points
// where the entry is resolved (sens < 10% of it).
func (ck tfCheck) resolvedErr(tf *TransferFunc) (worst float64, ok bool) {
	for _, e := range ck.rowPointErrs(tf, -1) {
		if e >= 0 {
			worst, ok = max(worst, e), true
		}
	}
	return worst, ok
}

// rowPointErrs is resolvedErr per point over output row only, or all rows
// when row < 0; −1 marks a point resolving no entry.
func (ck tfCheck) rowPointErrs(tf *TransferFunc, only int) []float64 {
	out := make([]float64, len(ck.s))
	for k, s := range ck.s {
		got := tf.eval(s)
		m := len(got[0])
		worst := -1.0
		for i, row := range got {
			if only >= 0 && i != only {
				continue
			}
			if dv := Poly(tf.Den[i]).Eval(s); cmplx.IsInf(dv) || cmplx.IsNaN(dv) {
				continue
			}
			rowMag := ck.rowMag(k, i, m)
			for j, v := range row {
				ref := ck.ref[k][i*m+j]
				scale := max(cmplx.Abs(ref), eps()*rowMag)
				if len(ck.sens) == len(ck.s) && ck.sens[k] != nil && ck.sens[k][i*m+j] >= 0.1*scale {
					continue
				}
				e := cmplx.Abs(v-ref) / scale
				if math.IsNaN(e) {
					e = math.Inf(1)
				}
				worst = max(worst, e)
			}
		}
		out[k] = worst
	}
	return out
}

// rowMag is the largest reference magnitude of output row i at point k,
// or that of all rows when the row is zero there.
func (ck tfCheck) rowMag(k, i, m int) float64 {
	mag := 0.0
	for _, v := range ck.ref[k][i*m : (i+1)*m] {
		mag = max(mag, cmplx.Abs(v))
	}
	if mag == 0 {
		return ck.mag[k]
	}
	return mag
}

// choose returns the Dgebal or Osborne converter, filled, and its order.
func (c rowRealizationConverter) choose() (rowRealizationConverter, int, error) {
	order, direct, err := c.convertFrom(false)
	if err != nil {
		return c, 0, err
	}
	zero, tiny := c.diagonalZeros()
	full := slices.Min(c.rows) == c.n
	if !zero && !tiny && full {
		return c, order, nil
	}
	alt, altOrder, altDirect, useAlt := c.preferOsborne(zero, full)
	// A direct row is exact in either candidate, so with more rows the
	// check points above the poles see only the others, where Osborne's
	// wrong low den coefficients barely show: a direct row that took the
	// Osborne miss out of the check flipped the choice and the other rows
	// to wrong, unstable dens. Then the staircase conversions decide.
	if c.p > 1 && !c.noDirect && direct+altDirect > 0 {
		sc := c.fresh(c.sys)
		sc.noDirect = true
		if _, _, err := sc.convertFrom(false); err != nil {
			useAlt = false
		} else if _, _, _, useAlt = sc.preferOsborne(zero, full); useAlt && alt.tf == nil {
			alt = c.fresh(c.sys)
			if altOrder, _, err = alt.convertFrom(true); err != nil {
				useAlt = false
			}
		}
	}
	if useAlt {
		return alt, altOrder, nil
	}
	return c, order, nil
}

// preferOsborne reports whether the Osborne conversion alt beats the
// filled Dgebal converter c; alt.tf stays nil when the check of c alone
// keeps c.
func (c rowRealizationConverter) preferOsborne(zero, full bool) (alt rowRealizationConverter, altOrder, altDirect int, ok bool) {
	check := newTFCheck(c.sys, c.n, c.m, c.p)
	gate := 1e3 * float64(c.n) * eps()
	// With only rotation noise on the diagonal Dgebal's grading costs
	// digits (P6VF2M: a lost zero), but Osborne's often costs more in the
	// low den coefficients the check points above the poles miss, so then
	// only a gross Dgebal miss switches, and only to an Osborne result
	// within 1e-10 of the response at those points: the right ones of a
	// near-double integrator reach 2e-11, those with a wrong DC den 1e-9.
	coarse := !zero && full
	if coarse {
		gate = 1e-6
	}
	e0, ok := check.relErr(c.tf)
	if !ok || e0 <= gate {
		return alt, 0, 0, false
	}
	alt = c.fresh(c.sys)
	altOrder, altDirect, err := alt.convertFrom(true)
	if err != nil {
		return alt, 0, altDirect, false
	}
	// Far above the poles the untrimmed Markov noise of either result can
	// exceed the error of one that lost states, which barely shows there
	// (WGMRIE; U3H63X: an exact Dgebal slow companion against an Osborne
	// one missing a state), so only the trimmed results are compared.
	// Once trimmed, a Dgebal result within 1e3·n·eps can still carry
	// 1e4·eps den errors on a graded zero-diagonal chain (P6VF2M), so then
	// the trimmed errors are compared outright, by a 4× margin against ties.
	c.trimRows()
	alt.trimRows()
	e0, _ = check.relErr(c.tf)
	e1, ok := check.relErr(alt.tf)
	switch {
	case !ok:
	case coarse:
		ok = e0 > gate && e1 < e0 && e1 <= 1e-10
	case e0 > gate:
		ok = e1 < e0
	default:
		ok = 4*e1 < e0
	}
	return alt, altOrder, altDirect, ok
}

// rowGroup is a set of output rows that structurally see the same states.
type rowGroup struct {
	rows, states []int
}

// structuralRowGroups groups the output rows by the states each sees
// structurally: reachable from B and reaching that row of C through the
// sparsity pattern of A, as MATLAB tf(ss) applies sminreal per channel.
// Exactly decoupled states are then never left to the rank decisions,
// whose rotations turn their zeros into noise. It returns nil when every
// row sees all n states.
func (c rowRealizationConverter) structuralRowGroups() []rowGroup {
	n := c.n
	aRaw, bRaw, cRaw := c.sys.A.RawMatrix(), c.sys.B.RawMatrix(), c.sys.C.RawMatrix()
	reach := make([]bool, n)
	queue := make([]int, 0, n)
	for i := range n {
		for _, v := range bRaw.Data[i*bRaw.Stride : i*bRaw.Stride+c.m] {
			if v != 0 {
				reach[i] = true
				queue = append(queue, i)
				break
			}
		}
	}
	for len(queue) > 0 {
		j := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		for i := range n {
			if !reach[i] && aRaw.Data[i*aRaw.Stride+j] != 0 {
				reach[i] = true
				queue = append(queue, i)
			}
		}
	}
	seen := make([]bool, n*c.p)
	full := true
	for k := range c.p {
		obs := seen[k*n : (k+1)*n]
		for j, v := range cRaw.Data[k*cRaw.Stride : k*cRaw.Stride+n] {
			if v != 0 {
				obs[j] = true
				queue = append(queue, j)
			}
		}
		for len(queue) > 0 {
			i := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			for j, v := range aRaw.Data[i*aRaw.Stride : i*aRaw.Stride+n] {
				if !obs[j] && v != 0 {
					obs[j] = true
					queue = append(queue, j)
				}
			}
		}
		for j := range n {
			obs[j] = obs[j] && reach[j]
			full = full && obs[j]
		}
	}
	if full {
		return nil
	}
	var groups []rowGroup
rows:
	for k := range c.p {
		obs := seen[k*n : (k+1)*n]
		for g := range groups {
			if slices.Equal(obs, seen[groups[g].rows[0]*n:(groups[g].rows[0]+1)*n]) {
				groups[g].rows = append(groups[g].rows, k)
				continue rows
			}
		}
		var states []int
		for j, v := range obs {
			if v {
				states = append(states, j)
			}
		}
		groups = append(groups, rowGroup{rows: []int{k}, states: states})
	}
	return groups
}

// convertRowGroups converts each row group from the subsystem of the
// states it sees.
func (c rowRealizationConverter) convertRowGroups(groups []rowGroup) (*TransferFuncResult, error) {
	aRaw, bRaw, cRaw := c.sys.A.RawMatrix(), c.sys.B.RawMatrix(), c.sys.C.RawMatrix()
	total := 0
	for _, g := range groups {
		r, q := len(g.states), len(g.rows)
		if r == 0 {
			for _, k := range g.rows {
				c.tf.Den[k] = []float64{1}
				c.tf.Num[k] = make([][]float64, c.m)
				for j := range c.m {
					c.tf.Num[k][j] = []float64{c.dRaw.Data[k*c.dRaw.Stride+j]}
				}
				c.rows[k] = 0
			}
			continue
		}
		a, b, cm, d := newDense(r, r), newDense(r, c.m), newDense(q, r), newDense(q, c.m)
		for ii, i := range g.states {
			for jj, j := range g.states {
				a.Set(ii, jj, aRaw.Data[i*aRaw.Stride+j])
			}
			for j := range c.m {
				b.Set(ii, j, bRaw.Data[i*bRaw.Stride+j])
			}
		}
		for kk, k := range g.rows {
			for jj, j := range g.states {
				cm.Set(kk, jj, cRaw.Data[k*cRaw.Stride+j])
			}
			for j := range c.m {
				d.Set(kk, j, c.dRaw.Data[k*c.dRaw.Stride+j])
			}
		}
		sub, err := newNoCopy(a, b, cm, d, c.sys.Dt)
		if err != nil {
			return nil, err
		}
		res, err := c.fresh(sub).convert()
		if err != nil {
			return nil, err
		}
		for kk, k := range g.rows {
			c.tf.Num[k] = res.TF.Num[kk]
			c.tf.Den[k] = res.TF.Den[kk]
			c.rows[k] = res.RowDegrees[kk]
		}
		total += res.MinimalOrder
	}
	return c.result(total), nil
}

// convertFrom fills the rows from the Dgebal or, with osborne, the
// [A B; C ·]-balanced realization (see balancedABC); direct counts the rows
// converted in the given basis (see directRowTF).
func (c rowRealizationConverter) convertFrom(osborne bool) (order, direct int, err error) {
	order, direct, scaled, err := c.convertBalanced(osborne, true)
	if err != nil {
		return 0, 0, err
	}
	if scaled && !c.finiteCoeffs() {
		if order, direct, _, err = c.convertBalanced(osborne, false); err != nil {
			return 0, 0, err
		}
	}
	if !c.finiteCoeffs() {
		return 0, 0, fmt.Errorf("TransferFunction: %w", ErrOverflow)
	}
	return order, direct, nil
}

// diagonalZeros reports whether A has an exactly zero diagonal entry, and
// whether it has one within n·eps·‖A‖_F, the rounding a rotation leaves on
// the diagonal of a zero-diagonal chain, which Dgebal grades like a zero.
func (c rowRealizationConverter) diagonalZeros() (zero, tiny bool) {
	raw := c.sys.A.RawMatrix()
	tol := float64(c.n) * eps() * frobNorm(c.sys.A)
	for i := range c.n {
		v := math.Abs(raw.Data[i*raw.Stride+i])
		zero = zero || v == 0
		tiny = tiny || v <= tol
	}
	return zero, tiny
}

// tfCheck holds reference responses of the state-space model at a few
// points, from the balanced dense solve behind EvalFr.
type tfCheck struct {
	s   []complex128
	ref [][]complex128
	mag []float64
	// sens, when set, bounds per entry the first-order change of ref under
	// relative perturbations of eps in each entry of A, B, C and D.
	sens [][]float64
}

// newTFCheck samples the delay-free response at e^{jθ}, θ ∈ {0.7, 1.9, 2.8},
// or jω, ω ∈ ‖A‖_F·{1e-2, 1, 1e2}, skipping poles and zero responses.
func newTFCheck(sys *System, n, m, p int) tfCheck {
	var pts []complex128
	if sys.IsContinuous() {
		rho := frobNorm(sys.A)
		if rho == 0 || math.IsInf(rho, 0) {
			rho = 1
		}
		for _, w := range []float64{1e-2, 1, 1e2} {
			pts = append(pts, complex(0, rho*w))
		}
	} else {
		for _, th := range []float64{0.7, 1.9, 2.8} {
			pts = append(pts, cmplx.Rect(1, th))
		}
	}
	return sampleTFCheck(newBalancedDense(sys, n, m, p), pts, m, p)
}

// trimCheckPoints is bandCheckPoints of the row's den, so that a trim is
// also checked in and below the band of that row's poles.
func trimCheckPoints(sys *System, den []float64) []complex128 {
	return bandCheckPoints(sys, [][]float64{den})
}

func sampleTFCheck(bd *balancedDense, pts []complex128, m, p int) tfCheck {
	return sampleCheck(nil, bd, pts, m, p, false)
}

// sampleTrimCheck is sampleTFCheck with sens, the trim gate's per-point
// resolution of the response.
func sampleTrimCheck(bd *balancedDense, pts []complex128, m, p int) tfCheck {
	return sampleCheck(nil, bd, pts, m, p, bd.e == nil)
}

// sampleBandCheck is sampleCheck of sys with the double-double fallback,
// as EvalFr: on a slow graded companion evalInto's normwise pivot test
// rejects every band point below the poles, where a cut origin pole shows
// (YY7Z5C).
func sampleBandCheck(sys *System, bd *balancedDense, pts []complex128, sens bool) tfCheck {
	return sampleCheck(sys, bd, pts, bd.m, bd.p, sens)
}

// sampleCheck samples bd at pts; with sys set, a point evalInto rejects as
// singular is solved in double-double instead of skipped.
func sampleCheck(sys *System, bd *balancedDense, pts []complex128, m, p int, sens bool) tfCheck {
	var ck tfCheck
	var sw *sensWork
	for _, s := range pts {
		dst := make([]complex128, p*m)
		fallback := false
		if err := bd.evalInto(pointAt(s), dst); err != nil {
			if sys == nil || !errors.Is(err, errSingularPencil) || !ddResponseInto(sys, pointAt(s), dst) || finiteResponse(dst) != nil {
				continue
			}
			fallback = true
		}
		mag := 0.0
		for _, v := range dst {
			mag = max(mag, cmplx.Abs(v))
		}
		if mag == 0 || math.IsInf(mag, 0) || math.IsNaN(mag) {
			continue
		}
		ck.s, ck.ref, ck.mag = append(ck.s, s), append(ck.ref, dst), append(ck.mag, mag)
		if sens {
			if sw == nil {
				sw = newSensWork(bd.n, m, p)
			}
			var b []float64
			if !fallback || sw.solveX(bd, s) {
				b = sw.bound(bd, s)
			}
			ck.sens = append(ck.sens, b)
		}
	}
	return ck
}

// solveX sets bd.x to (sI−Â)⁻¹B̂ by GEPP without evalInto's pivot test, for
// bound at a point evalInto rejected; it reports false on an exactly
// singular pencil.
func (sw *sensWork) solveX(bd *balancedDense, s complex128) bool {
	n, m := bd.n, bd.m
	for i, v := range bd.a {
		sw.lu[i] = complex(-v, 0)
	}
	for i := range n {
		sw.lu[i*n+i] += s
	}
	if !cLUFactor(sw.lu, sw.piv, n) {
		return false
	}
	for i, v := range bd.b[:n*m] {
		bd.x[i] = complex(v, 0)
	}
	cLUSolve(sw.lu, sw.piv, bd.x, n, m)
	return true
}

type sensWork struct {
	lu  []complex128
	y   []complex128 // row-major n×p
	u   []float64    // row-major n×m
	piv []int
}

func newSensWork(n, m, p int) *sensWork {
	return &sensWork{lu: make([]complex128, n*n), y: make([]complex128, n*p), u: make([]float64, n*m), piv: make([]int, n)}
}

// bound returns eps·(|y_i|ᵀ|Â||x_j| + |y_i|ᵀ|b̂_j| + |ĉ_i||x_j| + |d_ij|) per
// entry, x = (sI−Â)⁻¹B̂ from the solve evalInto just made at s and
// yᵢ = (sI−Â)⁻ᵀĉᵢᵀ: the first-order componentwise condition of G(s), the same
// in every diagonal state scaling. Where it is near |G(s)| the given
// coefficients do not determine the response at s. The result is nil when
// the transposed pencil is singular.
func (sw *sensWork) bound(bd *balancedDense, s complex128) []float64 {
	n, m, p := bd.n, bd.m, bd.p
	a, x := bd.a, bd.x
	for i := range n {
		for j := range n {
			sw.lu[i*n+j] = complex(-a[j*n+i], 0)
		}
		sw.lu[i*n+i] += s
	}
	if !cLUFactor(sw.lu, sw.piv, n) {
		return nil
	}
	for k := range n {
		for i := range p {
			sw.y[k*p+i] = complex(bd.c[i*n+k], 0)
		}
	}
	cLUSolve(sw.lu, sw.piv, sw.y, n, p)
	for k := range n {
		row := a[k*n : (k+1)*n]
		for j := range m {
			v := math.Abs(bd.b[k*m+j])
			for l, al := range row {
				v += math.Abs(al) * cmplx.Abs(x[l*m+j])
			}
			sw.u[k*m+j] = v
		}
	}
	out := make([]float64, p*m)
	for i := range p {
		ci := bd.c[i*n : (i+1)*n]
		for j := range m {
			v := 0.0
			for k := range n {
				v += cmplx.Abs(sw.y[k*p+i])*sw.u[k*m+j] + math.Abs(ci[k])*cmplx.Abs(x[k*m+j])
			}
			if bd.d != nil {
				v += math.Abs(bd.d[i*bd.dStride+j])
			}
			out[i*m+j] = v * eps()
		}
	}
	return out
}

// relErr is the worst max-entry error of tf relative to the max-entry
// reference over the points; ok is false without a usable point.
func (ck tfCheck) relErr(tf *TransferFunc) (worst float64, ok bool) {
	for k, s := range ck.s {
		got := tf.eval(s)
		m := len(got[0])
		for i, row := range got {
			for j, v := range row {
				e := cmplx.Abs(v-ck.ref[k][i*m+j]) / ck.mag[k]
				if math.IsNaN(e) {
					e = math.Inf(1)
				}
				worst = max(worst, e)
			}
		}
	}
	return worst, len(ck.s) > 0
}

// trimOK reports whether numerator trimmed over den matches entry (i, j) of
// an m-input model at every point to within twice the error of numerator
// full at that point, or 1e3·n·eps, relative to the entry, floored at eps
// times the largest entry of row i: rows are independent, and eps of a far
// larger row hid a small exact row's lost degree (TOIGMD). A point where
// full is inaccurate gives no slack at another. With sens, a point where
// the entry is unresolved (sens ≥ |ref|) also allows sens/|ref|, as long as
// some point resolves the entry to 10%: a fast mode in a dense basis leaves
// the response at and above its band undetermined by the given
// coefficients, where only a noise numerator can match it. Resolved points
// keep the 2× rule, so a trim never costs accuracy the data determine.
func (ck tfCheck) trimOK(i, j, m, n int, trimmed, full, den []float64) bool {
	if len(ck.s) == 0 {
		return false
	}
	floor := 1e3 * float64(n) * eps()
	sens := func(int, float64) float64 { return 0 }
	if len(ck.sens) == len(ck.s) {
		resolved := false
		for k := range ck.s {
			if ck.sens[k] != nil {
				ref := ck.ref[k][i*m+j]
				resolved = resolved || ck.sens[k][i*m+j] < 0.1*max(cmplx.Abs(ref), eps()*ck.rowMag(k, i, m))
			}
		}
		if resolved {
			sens = func(k int, scale float64) float64 {
				if ck.sens[k] == nil {
					return 0
				}
				if r := ck.sens[k][i*m+j] / scale; r >= 1 {
					return r
				}
				return 0
			}
		}
	}
	for k, s := range ck.s {
		ref := ck.ref[k][i*m+j]
		scale := max(cmplx.Abs(ref), eps()*ck.rowMag(k, i, m))
		d := Poly(den).Eval(s)
		eT := cmplx.Abs(Poly(trimmed).Eval(s)/d-ref) / scale
		eF := cmplx.Abs(Poly(full).Eval(s)/d-ref) / scale
		if !(eT <= max(2*eF, floor, sens(k, scale))) {
			return false
		}
	}
	return true
}

// convertBalanced fills the rows from the balanced realization, I/O-scaled
// when ioScale; scaled reports whether any I/O scale was not 1. Scaling a
// tiny B or C up toward a huge ‖A‖ can overflow coefficients that are
// finite without it.
func (c rowRealizationConverter) convertBalanced(osborne, ioScale bool) (order, direct int, scaled bool, err error) {
	a, b, cm, inExp, outExp := c.balancedABC(osborne, ioScale)
	stair, err := controllabilityStaircaseFloor(a, b, cm, c.opts.ControllabilityTol, ctrbfFloor(a, c.opts.ControllabilityTol), false)
	if err != nil {
		return 0, 0, false, err
	}
	if stair.NCont < c.n && !osborne {
		a2, b2, cm2, inExp2, outExp2 := c.balancedABC(true, ioScale)
		alt, err := controllabilityStaircaseFloor(a2, b2, cm2, c.opts.ControllabilityTol, ctrbfFloor(a2, c.opts.ControllabilityTol), false)
		if err != nil {
			return 0, 0, false, err
		}
		if alt.NCont == c.n {
			stair, inExp, outExp = alt, inExp2, outExp2
			a, b, cm = a2, b2, cm2
		}
	}
	scaled = slices.ContainsFunc(inExp, func(e int) bool { return e != 0 }) ||
		slices.ContainsFunc(outExp, func(e int) bool { return e != 0 })
	if stair.NCont == 0 {
		clear(c.rows)
		c.fillStaticRows()
		return 0, 0, scaled, nil
	}
	floor := ctrbfFloor(stair.A, c.opts.ControllabilityTol)
	if c.opts.ControllabilityTol == 0 {
		// The staircase rotations plus the row's Hessenberg reduction leave
		// up to ~200·eps·‖A‖ in the subdiagonal of a hidden state, above
		// 10·n·eps·‖A‖₁ for small n.
		floor *= 10
	}
	if c.keepObs {
		floor = 0
	}
	order, direct = c.convertDynamicRows(stair, stair.NCont, floor, inExp, outExp, a, b, cm)
	return order, direct, scaled, nil
}

func (c rowRealizationConverter) finiteCoeffs() bool {
	bad := func(x float64) bool { return math.IsInf(x, 0) || math.IsNaN(x) }
	for i := range c.p {
		if slices.ContainsFunc(c.tf.Den[i], bad) {
			return false
		}
		for _, num := range c.tf.Num[i] {
			if slices.ContainsFunc(num, bad) {
				return false
			}
		}
	}
	return true
}

// balancedABC returns D⁻¹AD, D⁻¹B·diag(2^inExp), diag(2^outExp)·CD for an
// exact power-of-two state scaling D, followed by power-of-two I/O scales
// toward ‖A‖ (reduceIOExponents, up or down), so entry (i, j) of the
// transfer matrix is only scaled by 2^(inExp[j]+outExp[i]). The staircase
// floor and the orthogonal reductions are relative to ‖A‖: without them a
// tiny but nonzero B or C is dropped as uncontrollable, and badly scaled A
// (Padé cascades, stiff chains) loses most of its digits. D is from Dgebal
// on A, or with osborne from the balance of [A B; C ·] (SLICOT TB01ID).
// Dgebal is far more accurate on stiff chains but, counting the diagonal
// in its norms, grades a chain of zero-diagonal states (shift registers,
// integrator and companion chains) toward a small diagonal entry, so the
// coefficients lose up to all digits; it can also shrink a companion
// chain's couplings below the staircase floor. So when states are lost
// convertBalanced takes the osborne realization if it is fully
// controllable (a partial count from either is equally ambiguous), and
// when A has a zero diagonal entry or a row lost states (the cost gate:
// Dgebal can also inflate B or C by ~1e13 past a state whose A row is
// diagonal-only, E2AY73) and the Dgebal result misses the state-space
// response at a few sample points by more than
// 1e3·n·eps, convert keeps whichever result misses it less. MATLAB tf(ss)
// removes only structurally disconnected states, so keeping states that
// are controllable in some exact scaling is on MATLAB's side.
func (c rowRealizationConverter) balancedABC(osborne, ioScale bool) (a, b, cm *mat.Dense, inExp, outExp []int) {
	n, m, p := c.n, c.m, c.p
	br := newRealizationCopy(c.sys, n, m, p)
	if osborne {
		br.balance(nil)
	} else {
		scale := make([]float64, n)
		impl.Dgebal(lapack.Scale, n, br.a, n, scale)
		for i, d := range scale {
			if d == 1 {
				continue
			}
			inv := 1 / d
			for j := range m {
				br.b[i*m+j] *= inv
			}
			for r := range p {
				br.c[r*n+i] *= d
			}
		}
	}
	a, b, cm = mat.NewDense(n, n, br.a), mat.NewDense(n, m, br.b), mat.NewDense(p, n, br.c)
	exps := make([]int, m+p)
	inExp, outExp = exps[:m], exps[m:]
	if !ioScale {
		return a, b, cm, inExp, outExp
	}
	if in, out := reduceIOExponents(a, b, cm); in != nil {
		copy(inExp, in)
		copy(outExp, out)
		ldexpIO(b, cm, inExp, outExp, 1)
	}
	return a, b, cm, inExp, outExp
}

// convertDynamicRows converts each row from the staircase form, a row
// keeping all n states from the balanced realization (a, b, cm) itself
// when that is already Hessenberg (see directRowTF), and returns the total
// order and the number of such direct rows.
func (c rowRealizationConverter) convertDynamicRows(stair *staircaseResult, ncont int, floor float64, inExp, outExp []int, a, b, cm *mat.Dense) (totalOrder, direct int) {
	Ac := extractSubmatrix(stair.A, 0, ncont, 0, ncont)
	Bc := extractSubmatrix(stair.B, 0, ncont, 0, c.m)
	Cc := extractSubmatrix(stair.C, 0, c.p, 0, ncont)
	var ws *tfWorkspace
	if ncont > 0 {
		ws = newTFWorkspace(ncont, c.m)
	}

	markov := newMarkovTrim(c.sys)
	zeroEigs, isCompanion := -1, -1
	for i := 0; i < c.p; i++ {
		nobs, den, numCoeffs := ssToTFRow(Ac, Bc, Cc, i, ncont, c.m, c.opts.ObservabilityTol, floor, inExp, outExp[i], ws)
		var bhData []float64
		if nobs > 0 {
			bhData = ws.cData
		}
		stride := ncont
		c.companion[i] = false
		// The full-order candidate of keepObservable also takes the direct
		// row where the staircase floor cut a graded companion's coupling.
		if (nobs == c.n || c.keepObs) && !c.noDirect {
			if dDen, dNums, dbh, ok := directRowTF(a, b, cm, i, c.n, c.m, inExp, outExp[i]); ok {
				den, numCoeffs, bhData = dDen, dNums, dbh
				nobs, stride = c.n, c.n
				if isCompanion < 0 {
					isCompanion = 0
					if t, _ := companionForm(a, c.n); t >= 0 {
						isCompanion = 1
					}
				}
				c.companion[i] = isCompanion == 1
				direct++
			}
		}
		if nobs == c.n {
			if zeroEigs < 0 {
				zeroEigs = isolatedZeroEigs(c.sys.A)
			}
			clear(den[len(den)-zeroEigs:])
		}
		c.rows[i] = nobs
		totalOrder += nobs

		c.tf.Den[i] = den
		c.tf.Num[i] = make([][]float64, c.m)
		for j := 0; j < c.m; j++ {
			dij := c.dRaw.Data[i*c.dRaw.Stride+j]
			withD := func(num []float64) []float64 {
				if dij == 0 {
					return num
				}
				return Poly(num).Add(Poly(den).Scale(dij))
			}
			var bh []float64
			if j*stride < len(bhData) {
				bh = bhData[j*stride : j*stride+nobs]
			}
			c.trims[i*c.m+j] = nil
			if cands := markov.trim(i, j, nobs, numCoeffs[j], bh); len(cands) > 1 || len(cands[0]) != len(numCoeffs[j]) {
				c.trims[i*c.m+j] = cands
			}
			c.tf.Num[i][j] = withD(numCoeffs[j])
		}
	}
	return totalOrder, direct
}

// isolatedZeroEigs counts the zero eigenvalues of A that Dgebal's
// permutation isolates, exact factors s of det(sI − A): a row keeping all
// states has that den, whose trailing coefficients the reductions leave as
// rounding noise instead of zeros (a k-fold origin pole then splits by
// noise^(1/k)).
func isolatedZeroEigs(A *mat.Dense) int {
	n, _ := A.Dims()
	a := make([]float64, n*n)
	raw := A.RawMatrix()
	for i := range n {
		copy(a[i*n:(i+1)*n], raw.Data[i*raw.Stride:i*raw.Stride+n])
	}
	ilo, ihi := impl.Dgebal(lapack.Permute, n, a, n, make([]float64, n))
	k := 0
	for i := range n {
		if (i < ilo || i > ihi || ilo == ihi) && a[i*n+i] == 0 {
			k++
		}
	}
	return k
}

// trimRows replaces each numerator that has Markov trim candidates by the
// most trimmed one that passes trimOK against a check sampled in the band of
// that row's den, else by the last candidate. It runs on the chosen
// realization, or on both candidates once the untrimmed comparison in convert
// has kept Dgebal.
func (c rowRealizationConverter) trimRows() {
	var bd *balancedDense
	for i := range c.p {
		var check *tfCheck
		den := c.tf.Den[i]
		for j := range c.m {
			cands := c.trims[i*c.m+j]
			if cands == nil {
				continue
			}
			c.trims[i*c.m+j] = nil
			dij := c.dRaw.Data[i*c.dRaw.Stride+j]
			withD := func(num []float64) []float64 {
				if dij == 0 {
					return num
				}
				return Poly(num).Add(Poly(den).Scale(dij))
			}
			num := cands[len(cands)-1]
			for _, cand := range cands[:len(cands)-1] {
				if check == nil {
					if bd == nil {
						bd = newBalancedDense(c.sys, c.n, c.m, c.p)
					}
					ck := sampleTrimCheck(bd, trimCheckPoints(c.sys, den), c.m, c.p)
					check = &ck
				}
				if check.trimOK(i, j, c.m, c.n, withD(cand), withD(num), den) {
					num = cand
					break
				}
			}
			c.tf.Num[i][j] = withD(num)
		}
	}
}

// markovTrim drops leading numerator coefficients that are zero within the
// rounding of the given A, B, C. The orthogonal reductions leave rounding
// noise in such a coefficient instead of an exact zero, which gives TF.ZPK a
// spurious huge zero and a tiny gain (MATLAB tf(ss) builds the numerators
// from the zeros, so its degree is exact). Over a monic denominator the
// leading coefficient is C·B, resolved only beyond 2·n·eps·|C|·|B|, and only
// then are later ones trimmed. With the coefficients of s^(deg-1) … s^(deg-k)
// zero, that of s^(deg-1-k) is the Markov parameter h_k = C·A^k·B, resolved
// beyond (k+2)·n·eps·|C|·|A|^k·|B|; but with a fast mode in a dense basis
// h_k, k ≥ 1, is dominated by the rounding of A while the coefficients, its
// sums over the denominator, are resolved. So that count, and the leading
// entries of the row's observability-Hessenberg b within n·eps·‖b‖ (which
// misses noise the Hessenberg basis amplifies), are only candidates, each
// kept if it matches the state-space response.
type markovTrim struct {
	a, b, c blas64.General
	n, m    int
	// krylov[j] holds A^k·b_j, |A|^k·|b_j| at [2kn, 2kn+2n) for k < count[j].
	krylov [][]float64
	count  []int
}

func newMarkovTrim(sys *System) markovTrim {
	n, m, _ := sys.Dims()
	return markovTrim{a: sys.A.RawMatrix(), b: sys.B.RawMatrix(), c: sys.C.RawMatrix(), n: n, m: m}
}

// vec returns A^k·b_j and |A|^k·|b_j|, k < n, computing them on demand.
func (t *markovTrim) vec(j, k int) (v, u []float64) {
	n := t.n
	if t.krylov == nil {
		t.krylov, t.count = make([][]float64, t.m), make([]int, t.m)
	}
	if t.krylov[j] == nil {
		buf := make([]float64, 2*n*n)
		for i := range n {
			buf[i] = t.b.Data[i*t.b.Stride+j]
			buf[n+i] = math.Abs(buf[i])
		}
		t.krylov[j], t.count[j] = buf, 1
	}
	buf := t.krylov[j]
	for ; t.count[j] <= k; t.count[j]++ {
		prev := buf[2*(t.count[j]-1)*n:]
		next := buf[2*t.count[j]*n:]
		for i := range n {
			row := t.a.Data[i*t.a.Stride : i*t.a.Stride+n]
			sv, su := 0.0, 0.0
			for l, a := range row {
				sv += a * prev[l]
				su += math.Abs(a) * prev[n+l]
			}
			next[i], next[n+i] = sv, su
		}
	}
	return buf[2*k*n : 2*k*n+n], buf[2*k*n+n : 2*k*n+2*n]
}

func (t *markovTrim) negligible(row, j, k int) bool {
	n := t.n
	crow := t.c.Data[row*t.c.Stride : row*t.c.Stride+n]
	h, bound := 0.0, 0.0
	if k == 0 {
		for l, cv := range crow {
			v := cv * t.b.Data[l*t.b.Stride+j]
			h += v
			bound += math.Abs(v)
		}
	} else {
		v, u := t.vec(j, k)
		for l, cv := range crow {
			h += cv * v[l]
			bound += math.Abs(cv) * u[l]
		}
	}
	bound *= float64((k+2)*n) * eps()
	return math.Abs(h) <= bound && !math.IsInf(bound, 0)
}

// trim returns the trimmed candidates for num over a monic denominator of
// degree deg, most trimmed first, every trim from the larger of the two
// counts down; the last, dropping at most a negligible C·B, needs no check.
// bh is input j in the row's observability-Hessenberg basis; nil (a row
// of companionRowTF, which has none) leaves only the Markov counts.
func (t *markovTrim) trim(row, j, deg int, num, bh []float64) [][]float64 {
	if len(num) < 2 || len(num) > deg || bh != nil && len(bh) != deg || !t.negligible(row, j, 0) {
		return [][]float64{num}
	}
	off := deg - len(num)
	safe := num
	if off == 0 {
		safe = num[1:]
	}
	lead := max(0, 1-off)
	for lead < len(num)-1 && t.negligible(row, j, off+lead) {
		lead++
	}
	r := 0
	if bh != nil {
		tol := float64(deg) * eps() * blas64.Implementation().Dnrm2(deg, bh, 1)
		for r < deg-1 && math.Abs(bh[r]) <= tol {
			r++
		}
	}
	var cands [][]float64
	for k := min(max(lead, r-off), len(num)-1); k > len(num)-len(safe); k-- {
		cands = append(cands, num[k:])
	}
	return append(cands, safe)
}

func (c rowRealizationConverter) fillStaticRows() {
	clear(c.trims)
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

// capSISONumerator drops the numerator coefficients of a SISO model with
// D = 0 above degree deg(den) − r, r from sisoRelativeDegree: they are
// combinations of Markov parameters h_1 … h_(r−1) that vanish within the
// rounding of A, B, C. capSISOZeros caps Zeros at n − r by the same test
// (tzero parity), so the TF keeps the zeros Zeros returns instead of far
// noise zeros.
func (c rowRealizationConverter) capSISONumerator() {
	if c.m != 1 || c.p != 1 || len(c.tf.Num[0][0]) < 2 {
		return
	}
	r, ok := sisoRelativeDegree(c.sys)
	num := c.tf.Num[0][0]
	if keep := len(c.tf.Den[0]) - r; ok && keep >= 1 && len(num) > keep {
		c.tf.Num[0][0] = num[len(num)-keep:]
	}
}

func (c rowRealizationConverter) result(order int) *TransferFuncResult {
	c.trimRows()
	c.capSISONumerator()
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

// observabilityReduction reduces the dual single-input pair (Aᵀ, cᵀ) of one
// output row to Hessenberg form and returns its observable order. c is zero
// at n·eps·max(‖A‖_F, ‖c‖₁) (or obsTol). A Hessenberg subdiagonal is zero at
// max(obsTol, floor), floor being 10× the ctrbf floor of the full model
// whose rotations left the rounding noise; it is never judged against ‖c‖,
// so the order does not depend on the output scale.
func observabilityReduction(n, m int, ws *tfWorkspace, obsTol, floor float64) (nobs int, b1, scaleB float64) {
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

	norm1B := 0.0
	for _, v := range bsimo[:n] {
		norm1B += math.Abs(v)
	}

	tol := obsTol
	if tol == 0 {
		frobA := 0.0
		for _, v := range aData[:n*n] {
			frobA += v * v
		}
		frobA = math.Sqrt(frobA)
		if math.IsInf(frobA, 1) {
			frobA = frobNorm(mat.NewDense(n, n, aData[:n*n]))
		}
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

	tolScan := math.Max(obsTol, floor*scaleA)
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

func computeNumerators(cData []float64, n, nobs, m int, scale []float64, inExp []int, wPolys []Poly) [][]float64 {
	nums := make([][]float64, m)
	scaleBuf := make(Poly, 0, nobs+1)
	addBuf := make(Poly, 0, nobs+2)
	for j := range m {
		var num Poly
		for k := range nobs {
			coeff := math.Ldexp(cData[j*n+k], -inExp[j]) * scale[k]
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

// ssToTFRow returns the observable order, denominator and numerators of
// output row; the numerators undo the I/O scales 2^inExp[j]·2^outExp before
// their products are formed, so they do not overflow when the unscaled
// coefficients are finite.
func ssToTFRow(Ac, Bc, Cc *mat.Dense, row, ncont, m int, obsTol, floor float64, inExp []int, outExp int, ws *tfWorkspace) (int, []float64, [][]float64) {
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

	nobs, b1, _ := observabilityReduction(n, m, ws, obsTol, floor)
	if nobs == 0 {
		return zeroReturn()
	}

	den, nums := hessenbergTF(ws.aData, ws.cData, n, nobs, m, b1, inExp, outExp)
	return nobs, den, nums
}

// hessenbergTF returns the den and numerators of the leading nobs states of
// the dual form (upper Hessenberg aData with stride n, input b1·e1, outputs
// the m rows of cData), undoing the I/O scales as ssToTFRow. It overwrites
// the strict upper triangle of aData.
func hessenbergTF(aData, cData []float64, n, nobs, m int, b1 float64, inExp []int, outExp int) ([]float64, [][]float64) {
	scale := make([]float64, nobs)
	scale[0] = math.Ldexp(b1, -outExp)
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
	nums := computeNumerators(cData, n, nobs, m, scale, inExp, wPolys)
	return den, nums
}

// directRowTF converts output row of the balanced (a, b, cm) without the
// staircase and Hessenberg rotations when hessenbergDual or, for a companion
// with dense b and c, companionRowTF applies: on a
// graded companion (a slow or observable companion) the rotations mix the
// scales and cost up to all digits of the low den coefficients, or of the
// numerator far above the poles, while the recurrence in the given basis
// is exact on a companion and otherwise sums the same products of the
// given entries det(sI − A) and the cofactors expand to. bh is the
// Hessenberg-basis input (or output, in the primal form) for markovTrim.
func directRowTF(a, b, cm *mat.Dense, row, n, m int, inExp []int, outExp int) (den []float64, nums [][]float64, bh []float64, ok bool) {
	aData, cData, b1, primal, ok := hessenbergDual(a, b, cm, row, n, m)
	if !ok {
		if den, nums, ok = companionRowTF(a, b, cm, row, n, m, inExp, outExp); !ok {
			den, nums, ok = triangularRowTF(a, b, cm, row, n, m, inExp, outExp)
		}
		return den, nums, nil, ok
	}
	if primal {
		den, nums = hessenbergTF(aData, cData, n, n, 1, b1, []int{outExp}, inExp[0])
	} else {
		den, nums = hessenbergTF(aData, cData, n, n, m, b1, inExp, outExp)
	}
	bad := func(x float64) bool { return math.IsInf(x, 0) || math.IsNaN(x) }
	if slices.ContainsFunc(den, bad) || slices.ContainsFunc(nums, func(p []float64) bool { return slices.ContainsFunc(p, bad) }) {
		return nil, nil, nil, false
	}
	return den, nums, cData, true
}

// companionRowTF converts output row of (a, b, cm) in the given basis when
// a is a companion of any b and c: in the observable form a nonzero
// subdiagonal h and otherwise nonzeros only in the last column g, or its
// transpose or reversal (last row, first column, first row) with b and c
// swapped or reversed to match. Then det(sI − A) has the coefficients
// −H_i·g_i, H_i = h_{i+1}⋯h_{n−1}, exact on a power-of-two scaling, and
// num(s) = Σ_k c_k·P_k(s) with x_k = P_k/det from the state equations
// s·x_k = h_k·x_{k−1} + g_k·x_{n−1} + b_k; P_k's coefficient of s^j is
//
//	Σ_{l ≤ min(j,k), i = j+k+1−l ≤ n−1} H^k_l·H_i·(g_l·b_i − b_l·g_i) + H^k_{l₀}·b_{l₀},
//
// l₀ = j+k+1−n ≥ 0, H^k_l = h_{l+1}⋯h_k: the products its cofactor
// expansion gives, without the pairs that cancel in the recurrence. With
// dense b and c the rotations of the staircase path mix the scales of a
// graded companion and lose its low den coefficients (N2JIRH).
func companionRowTF(a, b, cm *mat.Dense, row, n, m int, inExp []int, outExp int) (den []float64, nums [][]float64, ok bool) {
	ar, br, cr := a.RawMatrix(), b.RawMatrix(), cm.RawMatrix()
	at := func(i, j int) float64 { return ar.Data[i*ar.Stride+j] }
	t, maxA := companionForm(a, n)
	smlnum := math.SmallestNonzeroFloat64 / eps()
	if maxA != 0 && (maxA < smlnum || maxA > 1/smlnum) {
		return nil, nil, false
	}
	if t < 0 {
		return nil, nil, false
	}
	rev := func(k int) int { return n - 1 - k }
	canon := func(i, j int) float64 {
		switch t {
		case 0:
			return at(i, j)
		case 1:
			return at(rev(i), rev(j))
		case 2:
			return at(j, i)
		default:
			return at(rev(j), rev(i))
		}
	}
	h, g := make([]float64, n), make([]float64, n)
	for i := range n {
		g[i] = canon(i, n-1)
		if i > 0 {
			if h[i] = canon(i, i-1); h[i] == 0 {
				return nil, nil, false
			}
		}
	}
	hk := make([]float64, n*n)
	for k := range n {
		hk[k*n+k] = 1
		for l := k - 1; l >= 0; l-- {
			hk[k*n+l] = hk[k*n+l+1] * h[l+1]
		}
	}
	hn := hk[(n-1)*n:]
	den = make([]float64, n+1)
	den[0] = 1
	for i := range n {
		if g[i] != 0 {
			den[n-i] = -hn[i] * g[i]
		}
	}
	crow := cr.Data[row*cr.Stride : row*cr.Stride+n]
	bv, cv := make([]float64, n), make([]float64, n)
	nums = make([][]float64, m)
	for col := range m {
		for k := range n {
			src := k
			if t == 1 || t == 3 {
				src = rev(k)
			}
			bk := math.Ldexp(br.Data[src*br.Stride+col], -inExp[col])
			ck := math.Ldexp(crow[src], -outExp)
			if t >= 2 {
				bk, ck = ck, bk
			}
			bv[k], cv[k] = bk, ck
		}
		num := make([]float64, n)
		for k, ck := range cv {
			if ck == 0 {
				continue
			}
			hrow := hk[k*n:]
			for j := range n {
				p := 0.0
				for l := max(0, j+k+2-n); l <= min(j, k); l++ {
					i := j + k + 1 - l
					p += hrow[l] * hn[i] * (g[l]*bv[i] - bv[l]*g[i])
				}
				if l0 := j + k + 1 - n; l0 >= 0 {
					p += hrow[l0] * bv[l0]
				}
				num[n-1-j] += ck * p
			}
		}
		nums[col] = num
	}
	return den, nums, true
}

// triangularRowTF converts output row of (a, b, cm) in the given basis when
// a is upper or lower triangular (a lower one is read reversed). Then
// det(sI − A) = Π(s − a_ii), and with x = (sI − A)⁻¹b_col solved by back
// substitution, x_k = P_k/Π_{i≥k}(s − a_ii),
//
//	P_k = b_k·Π_{i>k}(s − a_ii) + Σ_{j>k} a_kj·P_j·Π_{k<i<j}(s − a_ii),
//
// and num = Σ_k c_k·P_k·Π_{i<k}(s − a_ii): sums of products of the given
// entries. Any rotation perturbs A by eps·‖A‖, which loses near-origin
// poles in an Osborne basis and, in a Dgebal-graded one, the tiny couplings
// the numerator sums are built from (WM67N5).
func triangularRowTF(a, b, cm *mat.Dense, row, n, m int, inExp []int, outExp int) (den []float64, nums [][]float64, ok bool) {
	ar, br, cr := a.RawMatrix(), b.RawMatrix(), cm.RawMatrix()
	upper, lower := true, true
	maxA := 0.0
	for i := range n {
		for j, v := range ar.Data[i*ar.Stride : i*ar.Stride+n] {
			if v != 0 {
				maxA = max(maxA, math.Abs(v))
				upper = upper && j >= i
				lower = lower && j <= i
			}
		}
	}
	smlnum := math.SmallestNonzeroFloat64 / eps()
	if !upper && !lower || maxA != 0 && (maxA < smlnum || maxA > 1/smlnum) {
		return nil, nil, false
	}
	idx := func(k int) int { return k }
	if !upper {
		idx = func(k int) int { return n - 1 - k }
	}
	u := make([]float64, n*n)
	for i := range n {
		for j := i; j < n; j++ {
			u[i*n+j] = ar.Data[idx(i)*ar.Stride+idx(j)]
		}
	}
	// Polynomials are ascending in s, of length n+1; mulLin multiplies one
	// of degree < deg by (s − r).
	mulLin := func(p []float64, deg int, r float64) {
		for k := deg; k > 0; k-- {
			p[k] = p[k-1] - r*p[k]
		}
		p[0] *= -r
	}
	w := n + 1
	q := make([]float64, (n+1)*w)
	qk := func(k int) []float64 { return q[(k+1)*w : (k+2)*w] }
	qk(n - 1)[0] = 1
	for k := n - 2; k >= -1; k-- {
		copy(qk(k), qk(k+1))
		mulLin(qk(k), n-1-k, u[(k+1)*n+k+1])
	}
	P := make([]float64, n*w)
	// acc sets dst to Σ_{j>k} coef_j·P_j·Π_{k<i<j}(s − a_ii); P_j has degree
	// n−1−j, and so has the sum over j' ≥ j.
	acc := func(dst []float64, k int, coef []float64) {
		clear(dst)
		have := false
		for j := n - 1; j > k; j-- {
			if have {
				mulLin(dst, n-1-j, u[j*n+j])
			}
			if v := coef[j]; v != 0 {
				have = true
				pj := P[j*w : j*w+n-j]
				for l, x := range pj {
					dst[l] += v * x
				}
			}
		}
	}
	den = make([]float64, n+1)
	for k := range n + 1 {
		den[k] = qk(-1)[n-k]
	}
	crow := make([]float64, n)
	for j := range n {
		crow[j] = math.Ldexp(cr.Data[row*cr.Stride+idx(j)], -outExp)
	}
	numBuf := make([]float64, w)
	nums = make([][]float64, m)
	for col := range m {
		for k := n - 1; k >= 0; k-- {
			dst := P[k*w : (k+1)*w]
			acc(dst, k, u[k*n:(k+1)*n])
			if bk := math.Ldexp(br.Data[idx(k)*br.Stride+col], -inExp[col]); bk != 0 {
				for l, x := range qk(k)[:n-k] {
					dst[l] += bk * x
				}
			}
		}
		acc(numBuf, -1, crow)
		num := make([]float64, n)
		for k := range n {
			num[k] = numBuf[n-1-k]
		}
		nums[col] = num
	}
	return den, nums, true
}

// companionForm returns the companion shape of a that companionRowTF
// reads (0 observable, 1 its reversal, 2 its transpose, 3 both), or −1,
// and the largest entry magnitude of a.
func companionForm(a *mat.Dense, n int) (t int, maxA float64) {
	ar := a.RawMatrix()
	var shape [4]bool
	for i := range shape {
		shape[i] = true
	}
	for i := range n {
		for j := range n {
			v := ar.Data[i*ar.Stride+j]
			if v == 0 {
				continue
			}
			maxA = max(maxA, math.Abs(v))
			sub, sup := i == j+1, j == i+1
			shape[0] = shape[0] && (sub || j == n-1)
			shape[1] = shape[1] && (sup || j == 0)
			shape[2] = shape[2] && (sup || i == n-1)
			shape[3] = shape[3] && (sub || i == 0)
		}
	}
	return slices.Index(shape[:], true), maxA
}

// hessenbergDual returns a dual form of output row of (a, b, cm), aData
// and cData, by an exact transpose and reversal, and its input coefficient
// b1, when a is Hessenberg with nonzero subdiagonal and the
// row, or with m = 1 the input, is a multiple of the unit vector at the
// matching end: lower Hessenberg with c ∝ e₁ᵀ or upper with c ∝ eₙᵀ (the
// dual (Aᵀ, cᵀ, Bᵀ)), or upper with b ∝ e₁ or lower with b ∝ eₙ (primal:
// (A, b, c) itself, cData holding the row).
func hessenbergDual(a, b, cm *mat.Dense, row, n, m int) (aData, cData []float64, b1 float64, primal, ok bool) {
	ar, br, cr := a.RawMatrix(), b.RawMatrix(), cm.RawMatrix()
	at := func(i, j int) float64 { return ar.Data[i*ar.Stride+j] }
	upper, lower := true, true
	maxA := 0.0
	for i := range n {
		for j := range n {
			v := at(i, j)
			maxA = max(maxA, math.Abs(v))
			if v != 0 {
				upper = upper && i <= j+1
				lower = lower && j <= i+1
			}
		}
		if !upper && !lower {
			return nil, nil, 0, false, false
		}
	}
	smlnum := math.SmallestNonzeroFloat64 / eps()
	if maxA != 0 && (maxA < smlnum || maxA > 1/smlnum) {
		return nil, nil, 0, false, false
	}
	crow := cr.Data[row*cr.Stride : row*cr.Stride+n]
	unit := func(v []float64, k int) bool {
		for i, x := range v {
			if (x != 0) != (i == k) {
				return false
			}
		}
		return true
	}
	unitB := func(k int) bool {
		if m != 1 {
			return false
		}
		for i := range n {
			if (br.Data[i*br.Stride] != 0) != (i == k) {
				return false
			}
		}
		return true
	}
	var mode int
	switch {
	case lower && unit(crow, 0):
		mode = 1
	case upper && unit(crow, n-1):
		mode = 2
	case upper && unitB(0):
		mode = 3
	case lower && unitB(n-1):
		mode = 4
	default:
		return nil, nil, 0, false, false
	}
	aData, cData = make([]float64, n*n), make([]float64, m*n)
	switch mode {
	case 1:
		b1 = crow[0]
		for i := range n {
			for j := range n {
				aData[i*n+j] = at(j, i)
			}
			for j := range m {
				cData[j*n+i] = br.Data[i*br.Stride+j]
			}
		}
	case 2:
		b1 = crow[n-1]
		for i := range n {
			for j := range n {
				aData[i*n+j] = at(n-1-j, n-1-i)
			}
			for j := range m {
				cData[j*n+i] = br.Data[(n-1-i)*br.Stride+j]
			}
		}
	case 3:
		b1, primal = br.Data[0], true
		for i := range n {
			copy(aData[i*n:(i+1)*n], ar.Data[i*ar.Stride:i*ar.Stride+n])
		}
		copy(cData[:n], crow)
	default:
		b1, primal = br.Data[(n-1)*br.Stride], true
		for i := range n {
			for j := range n {
				aData[i*n+j] = at(n-1-i, n-1-j)
			}
			cData[i] = crow[n-1-i]
		}
	}
	for k := 1; k < n; k++ {
		if aData[k*n+k-1] == 0 {
			return nil, nil, 0, false, false
		}
	}
	return aData, cData, b1, primal, true
}

// StateSpaceResult is the result of (*TransferFunc).StateSpace.
type StateSpaceResult struct {
	// Sys is the realization; delays and I/O names carry over.
	Sys *System
	// MinimalOrder is the realization order, the sum of the row denominator
	// degrees (not necessarily minimal; use MinimalRealization).
	MinimalOrder int
	// BlockSizes[i] is the denominator degree of output row i, the size of
	// its observable-companion block in Sys.A (0 for static rows).
	BlockSizes []int
}

// StateSpace returns an observable-companion realization with one block per
// output row, as MATLAB ss(tf) (SLICOT TD04AD); like MATLAB it is not
// reduced to minimal order. An improper model gives, as MATLAB ss(tf), a
// descriptor realization with singular E: the companion blocks of the
// proper remainders, then per improper row i a nilpotent chain of q_i+1
// states realizing the polynomial quotient of degree q_i (MinimalOrder
// counts them; BlockSizes does not). A malformed or empty model returns
// ErrDimensionMismatch, a NaN or Inf coefficient ErrInvalidArgument and a
// zero leading denominator coefficient ErrSingularDenom.
func (tf *TransferFunc) StateSpace() (*StateSpaceResult, error) {
	res, err := tf.stateSpace()
	if err != nil {
		return nil, fmt.Errorf("TransferFunc.StateSpace: %w", err)
	}
	return res, nil
}

func (tf *TransferFunc) stateSpace() (*StateSpaceResult, error) {
	p, m, err := tf.validateShape()
	if err != nil {
		return nil, err
	}
	if p == 0 || m == 0 {
		return nil, fmt.Errorf("model has %d outputs and %d inputs: %w", p, m, ErrDimensionMismatch)
	}
	if err := tf.validateFiniteCoeffs(); err != nil {
		return nil, err
	}
	if !tf.isProper() {
		return tf.improperStateSpace(p, m)
	}

	smlnum := math.SmallestNonzeroFloat64 / eps()
	bignum := 1.0 / smlnum

	degrees := make([]int, p)
	totalN := 0
	for i := range p {
		if math.Abs(tf.Den[i][0]) < smlnum {
			return nil, fmt.Errorf("row %d: %w", i, ErrSingularDenom)
		}
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
		sys, err := NewGain(D, tf.Dt)
		if err != nil {
			return nil, err
		}
		if tf.Delay != nil {
			sys.Delay = slice2DToDense(tf.Delay)
		}
		sys.InputName = copyStringSlice(tf.InputName)
		sys.OutputName = copyStringSlice(tf.OutputName)
		return &StateSpaceResult{Sys: sys, MinimalOrder: 0, BlockSizes: degrees}, nil
	}

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

// improperStateSpace splits each channel into quotient q_ij and proper
// remainder r_ij of num_ij/den_i, realizes r with stateSpace and appends,
// per row, E z' = z + B u with E the upper shift and B_k = -[q_ij]_k, so
// z_0 = Σ_k s^k Σ_j [q_ij]_k u_j.
func (tf *TransferFunc) improperStateSpace(p, m int) (*StateSpaceResult, error) {
	smlnum := math.SmallestNonzeroFloat64 / eps()
	for i := range p {
		if math.Abs(tf.Den[i][0]) < smlnum {
			return nil, fmt.Errorf("row %d: %w", i, ErrSingularDenom)
		}
	}
	rem := &TransferFunc{Num: make([][][]float64, p), Den: tf.Den, Dt: tf.Dt,
		Delay: tf.Delay, InputName: tf.InputName, OutputName: tf.OutputName}
	quot := make([][][]float64, p)
	chain := make([]int, p)
	nPoly := 0
	for i := range p {
		den := tf.Den[i]
		d := len(den) - 1
		rem.Num[i] = make([][]float64, m)
		quot[i] = make([][]float64, m)
		for j := range m {
			num := trimLeadingFloatZeros(tf.Num[i][j])
			if len(num)-1 <= d {
				rem.Num[i][j] = num
				continue
			}
			r := append([]float64(nil), num...)
			q := make([]float64, len(num)-d)
			for k := range q {
				c := r[k] / den[0]
				if math.IsInf(c, 0) || math.IsNaN(c) {
					return nil, fmt.Errorf("row %d: polynomial quotient: %w", i, ErrOverflow)
				}
				q[k] = c
				for l := 1; l <= d; l++ {
					r[k+l] -= c * den[l]
				}
			}
			for _, v := range r {
				if math.IsInf(v, 0) || math.IsNaN(v) {
					return nil, fmt.Errorf("row %d: polynomial remainder: %w", i, ErrOverflow)
				}
			}
			if d == 0 {
				rem.Num[i][j] = []float64{0}
			} else {
				rem.Num[i][j] = r[len(q):]
			}
			quot[i][j] = q
			chain[i] = max(chain[i], len(q))
		}
		nPoly += chain[i]
	}
	res, err := rem.stateSpace()
	if err != nil || nPoly == 0 {
		return res, err
	}
	base := res.Sys
	n0, _, _ := base.Dims()
	n := n0 + nPoly
	A := mat.NewDense(n, n, nil)
	E := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	D := mat.DenseCopyOf(base.D)
	aRaw, eRaw, bRaw, cRaw, dRaw := A.RawMatrix(), E.RawMatrix(), B.RawMatrix(), C.RawMatrix(), D.RawMatrix()
	if n0 > 0 {
		a0, b0, c0 := base.A.RawMatrix(), base.B.RawMatrix(), base.C.RawMatrix()
		for r := range n0 {
			copy(aRaw.Data[r*aRaw.Stride:r*aRaw.Stride+n0], a0.Data[r*a0.Stride:r*a0.Stride+n0])
			copy(bRaw.Data[r*bRaw.Stride:r*bRaw.Stride+m], b0.Data[r*b0.Stride:r*b0.Stride+m])
			eRaw.Data[r*eRaw.Stride+r] = 1
		}
		for r := range p {
			copy(cRaw.Data[r*cRaw.Stride:r*cRaw.Stride+n0], c0.Data[r*c0.Stride:r*c0.Stride+n0])
		}
	}
	off := n0
	for i := range p {
		for j := range m {
			if q := quot[i][j]; len(q) > 0 {
				dRaw.Data[i*dRaw.Stride+j] += q[len(q)-1]
			}
		}
		qi := chain[i]
		if qi == 0 {
			continue
		}
		for k := range qi {
			aRaw.Data[(off+k)*aRaw.Stride+off+k] = 1
			if k+1 < qi {
				eRaw.Data[(off+k)*eRaw.Stride+off+k+1] = 1
			}
			for j := range m {
				q := quot[i][j]
				if idx := len(q) - 1 - k; k > 0 && idx >= 0 {
					bRaw.Data[(off+k)*bRaw.Stride+j] = -q[idx]
				}
			}
		}
		cRaw.Data[i*cRaw.Stride+off] = 1
		off += qi
	}
	sys, err := newDescriptorOwned(A, B, C, D, E, tf.Dt)
	if err != nil {
		return nil, err
	}
	if tf.Delay != nil {
		sys.Delay = slice2DToDense(tf.Delay)
	}
	sys.InputName = copyStringSlice(tf.InputName)
	sys.OutputName = copyStringSlice(tf.OutputName)
	return &StateSpaceResult{Sys: sys, MinimalOrder: n, BlockSizes: res.BlockSizes}, nil
}

// IsProper reports whether the delay-free part of sys, including internal
// delay channels, is proper, as MATLAB isproper. Explicit models always are;
// a descriptor model is proper iff its infinite eigenvalues are all
// nondynamic. The decision is the Weierstrass reduction Step and
// TransferFunction use, so they agree; s^k coefficients at rounding level
// count as zero. A singular pencil (A, E), for which properness is
// undefined, returns ErrDescriptorSingular, and a numerical failure deciding
// it returns an error rather than false. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.isproper.html.
func (sys *System) IsProper() (bool, error) {
	if err := requireFiniteSystem("IsProper", sys); err != nil {
		return false, err
	}
	ok, err := sys.isProper()
	if err != nil {
		return false, fmt.Errorf("IsProper: %w", err)
	}
	return ok, nil
}

func (sys *System) isProper() (bool, error) {
	if !sys.IsDescriptor() {
		return true, nil
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if !luNearSingular(&lu) {
		return true, nil
	}
	n, m, p := sys.Dims()
	N := sys.internalDelayCount()
	if m+N == 0 || p+N == 0 {
		return true, nil
	}
	if _, ok := regularDescriptorShift(sys.A, sys.E, n); !ok {
		return false, fmt.Errorf("pencil (A, E) is singular: %w", ErrDescriptorSingular)
	}
	if _, err := sys.weierstrassReduce(); err != nil {
		if errors.Is(err, ErrImproperModel) {
			return false, nil
		}
		return false, err
	}
	return true, nil
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

// IsProper reports whether every channel's numerator degree is at most its
// row denominator degree, as MATLAB isproper. A malformed model returns an
// error.
func (tf *TransferFunc) IsProper() (bool, error) {
	if _, _, err := tf.validateShape(); err != nil {
		return false, fmt.Errorf("TransferFunc.IsProper: %w", err)
	}
	return tf.isProper(), nil
}

func (tf *TransferFunc) isProper() bool {
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
