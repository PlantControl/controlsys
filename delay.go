package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

func NewWithDelay(A, B, C, D, delay *mat.Dense, dt float64) (*System, error) {
	sys, err := New(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	if delay != nil {
		if err := sys.SetDelay(delay); err != nil {
			return nil, err
		}
	}
	return sys, nil
}

func (sys *System) SetDelay(delay *mat.Dense) error {
	_, m, p := sys.Dims()
	if err := validateDelay(delay, p, m, sys.Dt); err != nil {
		return err
	}
	sys.Delay = copyDelayOrNil(delay)
	return nil
}

func (sys *System) SetInputDelay(delay []float64) error {
	_, m, _ := sys.Dims()
	if err := validateSliceDelay(delay, m, sys.Dt); err != nil {
		return err
	}
	if delay == nil {
		sys.InputDelay = nil
		return nil
	}
	sys.InputDelay = make([]float64, len(delay))
	copy(sys.InputDelay, delay)
	return nil
}

func (sys *System) SetOutputDelay(delay []float64) error {
	_, _, p := sys.Dims()
	if err := validateSliceDelay(delay, p, sys.Dt); err != nil {
		return err
	}
	if delay == nil {
		sys.OutputDelay = nil
		return nil
	}
	sys.OutputDelay = make([]float64, len(delay))
	copy(sys.OutputDelay, delay)
	return nil
}

// SetInternalDelay sets the internal delays tau of sys and the LFT blocks
// that connect them: the delayed signals enter through B2 and D12 and leave
// through C2, D21 and D22, as in the H partition of GetDelayModel. An empty
// tau removes all internal delays. Each tau must be finite and positive, and
// an integer number of samples for a discrete model.
func (sys *System) SetInternalDelay(tau []float64, B2, C2, D12, D21, D22 *mat.Dense) error {
	if sys == nil {
		return fmt.Errorf("SetInternalDelay: system is nil: %w", ErrInvalidArgument)
	}
	if len(tau) == 0 {
		sys.LFT = nil
		return nil
	}
	n, m, p := sys.Dims()
	N := len(tau)
	if err := validateInternalTau(tau, sys.Dt); err != nil {
		return fmt.Errorf("SetInternalDelay: %w", err)
	}
	if err := validateLFTDims(n, m, p, N, B2, C2, D12, D21, D22); err != nil {
		return fmt.Errorf("SetInternalDelay: %w", err)
	}
	tauCopy := make([]float64, N)
	copy(tauCopy, tau)
	sys.LFT = &LFTDelay{
		Tau: tauCopy,
		B2:  denseCopySafe(B2, n, N),
		C2:  denseCopySafe(C2, N, n),
		D12: denseCopySafe(D12, p, N),
		D21: denseCopySafe(D21, N, m),
		D22: denseCopySafe(D22, N, N),
	}
	return nil
}

func validateLFTDims(n, m, p, N int, B2, C2, D12, D21, D22 *mat.Dense) error {
	check := func(name string, mat *mat.Dense, wantR, wantC int) error {
		empty := wantR == 0 || wantC == 0
		if mat == nil {
			if empty {
				return nil
			}
			return fmt.Errorf("%s required when InternalDelay is set: %w", name, ErrDimensionMismatch)
		}
		r, c := mat.Dims()
		if empty && r == 0 && c == 0 {
			return nil
		}
		if r != wantR || c != wantC {
			return fmt.Errorf("%s %d×%d != %d×%d: %w", name, r, c, wantR, wantC, ErrDimensionMismatch)
		}
		return nil
	}
	if err := check("B2", B2, n, N); err != nil {
		return err
	}
	if err := check("C2", C2, N, n); err != nil {
		return err
	}
	if err := check("D12", D12, p, N); err != nil {
		return err
	}
	if err := check("D21", D21, N, m); err != nil {
		return err
	}
	return check("D22", D22, N, N)
}

func validateSliceDelay(delay []float64, expected int, dt float64) error {
	if delay == nil {
		return nil
	}
	if len(delay) != expected {
		return fmt.Errorf("delay length %d != %d: %w", len(delay), expected, ErrDimensionMismatch)
	}
	for _, v := range delay {
		if err := validateDelayValue(v, dt); err != nil {
			return err
		}
	}
	return nil
}

// validateInternalTau requires each internal delay to be a valid delay value
// and nonzero, since a zero internal delay closes an algebraic loop.
func validateInternalTau(tau []float64, dt float64) error {
	for i, v := range tau {
		if err := validateDelayValue(v, dt); err != nil {
			return fmt.Errorf("tau[%d]: %w", i, err)
		}
		if v == 0 {
			return fmt.Errorf("tau[%d] is 0: %w", i, ErrZeroInternalDelay)
		}
	}
	return nil
}

// validateDelayValue requires a finite non-negative delay, as MATLAB's delay
// properties do, that is an integer sample count for discrete models.
func validateDelayValue(v, dt float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("delay %g must be finite: %w", v, ErrInvalidArgument)
	}
	if v < 0 {
		return fmt.Errorf("delay %g: %w", v, ErrNegativeDelay)
	}
	if dt > 0 && math.Round(v) != v {
		return fmt.Errorf("delay %g: %w", v, ErrFractionalDelay)
	}
	return nil
}

// TotalDelay returns the p×m matrix of total I/O delays, input plus output
// plus I/O delay of each channel, as MATLAB totaldelay(sys)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.totaldelay.html);
// entries are zero for channels without delay. Internal delays are not
// included. Units are time units for continuous models and samples for
// discrete ones. A model with no inputs or no outputs has no delay matrix and
// returns ErrDimensionMismatch.
func (sys *System) TotalDelay() (*mat.Dense, error) {
	if err := requireSystem("TotalDelay", sys); err != nil {
		return nil, err
	}
	_, m, p := sys.Dims()
	if p == 0 || m == 0 {
		return nil, fmt.Errorf("TotalDelay: model is %d×%d, has no delay matrix: %w", p, m, ErrDimensionMismatch)
	}
	return mat.NewDense(p, m, effectiveIODelayData(sys, p, m, true)), nil
}

func effectiveIODelayMatrix(sys *System, p, m int, includeDelayMatrix bool) *mat.Dense {
	if !hasExternalDelay(sys, includeDelayMatrix) || p == 0 || m == 0 {
		return nil
	}
	data := effectiveIODelayData(sys, p, m, includeDelayMatrix)
	return mat.NewDense(p, m, data)
}

func effectiveIODelayData(sys *System, p, m int, includeDelayMatrix bool) []float64 {
	data := make([]float64, p*m)
	if includeDelayMatrix && sys.Delay != nil {
		raw := sys.Delay.RawMatrix()
		for i := range p {
			copy(data[i*m:i*m+m], raw.Data[i*raw.Stride:i*raw.Stride+m])
		}
	}
	if sys.InputDelay != nil {
		for j := range m {
			for i := range p {
				data[i*m+j] += sys.InputDelay[j]
			}
		}
	}
	if sys.OutputDelay != nil {
		for i := range p {
			for j := range m {
				data[i*m+j] += sys.OutputDelay[i]
			}
		}
	}
	return data
}

func hasExternalDelay(sys *System, includeDelayMatrix bool) bool {
	return sys.InputDelay != nil || sys.OutputDelay != nil || (includeDelayMatrix && sys.Delay != nil)
}

func (sys *System) HasDelay() bool {
	if sys.HasInternalDelay() {
		return true
	}
	return delayMatrixHasNonzero(sys.Delay) ||
		delaySliceHasNonzero(sys.InputDelay) ||
		delaySliceHasNonzero(sys.OutputDelay)
}

func (sys *System) HasInternalDelay() bool {
	if sys.LFT == nil {
		return false
	}
	for _, v := range sys.LFT.Tau {
		if v != 0 {
			return true
		}
	}
	return false
}

func (tf *TransferFunc) HasDelay() bool {
	if tf.Delay == nil {
		return false
	}
	for _, row := range tf.Delay {
		for _, v := range row {
			if v != 0 {
				return true
			}
		}
	}
	return false
}

// AbsorbScope selects which delays AbsorbDelay absorbs, as the scope
// argument of MATLAB absorbDelay.
type AbsorbScope string

// AbsorbDelay scopes. AbsorbAll is the default and absorbs every delay.
const (
	AbsorbInput    AbsorbScope = "input"
	AbsorbOutput   AbsorbScope = "output"
	AbsorbIO       AbsorbScope = "io"
	AbsorbInternal AbsorbScope = "internal"
	AbsorbAll      AbsorbScope = "all"
)

// DefaultPadeOrder is the Padé order AbsorbDelay uses for continuous delays.
const DefaultPadeOrder = 5

// AbsorbDelay replaces the delays selected by scopes with model dynamics, as
// MATLAB absorbDelay(sys,scope)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.absorbdelay.html).
// With no scopes every delay is absorbed; several scopes absorb their union,
// like a MATLAB scope array. Unselected delays are kept.
//
// A discrete delay of k samples becomes k states at z = 0, exactly. A
// continuous delay has no finite-dimensional representation, so it is
// replaced by its Padé approximation of order DefaultPadeOrder; use Pade to
// choose the order. An unknown scope returns ErrInvalidArgument.
func (sys *System) AbsorbDelay(scopes ...AbsorbScope) (*System, error) {
	if err := requireSystem("AbsorbDelay", sys); err != nil {
		return nil, err
	}
	set, err := absorbScopeSet(scopes)
	if err != nil {
		return nil, err
	}
	cur := sys
	for _, scope := range set {
		if cur, err = absorbDelayScope(cur, scope); err != nil {
			return nil, fmt.Errorf("AbsorbDelay: %w", err)
		}
	}
	return cur, nil
}

// absorbScopeSet validates scopes and orders their union as AbsorbAll does:
// internal, I/O, input, output.
func absorbScopeSet(scopes []AbsorbScope) ([]AbsorbScope, error) {
	order := []AbsorbScope{AbsorbInternal, AbsorbIO, AbsorbInput, AbsorbOutput}
	want := make(map[AbsorbScope]bool, len(scopes))
	for _, s := range scopes {
		switch s {
		case AbsorbAll:
			return []AbsorbScope{AbsorbAll}, nil
		case AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbInternal:
			want[s] = true
		default:
			return nil, fmt.Errorf("AbsorbDelay: unknown scope %q: %w", s, ErrInvalidArgument)
		}
	}
	if len(want) == 0 {
		return []AbsorbScope{AbsorbAll}, nil
	}
	set := make([]AbsorbScope, 0, len(want))
	for _, s := range order {
		if want[s] {
			set = append(set, s)
		}
	}
	return set, nil
}

func absorbDelayScope(sys *System, scope AbsorbScope) (*System, error) {
	pending := sys.HasDelay()
	if scope == AbsorbInternal {
		pending = sys.HasInternalDelay()
	}
	if !pending {
		return sys.Copy(), nil
	}
	if scope == AbsorbInternal {
		return absorbInternalDelay(sys)
	}

	if sys.IsContinuous() {
		switch scope {
		case AbsorbInput:
			return absorbInputDelayContinuous(sys, DefaultPadeOrder)
		case AbsorbOutput:
			return absorbOutputDelayContinuous(sys, DefaultPadeOrder)
		case AbsorbIO:
			return absorbIODelayContinuous(sys, DefaultPadeOrder)
		default:
			return sys.Pade(DefaultPadeOrder)
		}
	}

	switch scope {
	case AbsorbInput:
		return absorbInputDelay(sys)
	case AbsorbOutput:
		return absorbOutputDelay(sys)
	case AbsorbIO:
		return absorbIODelay(sys)
	default:
		return absorbAllDelay(sys)
	}
}

func absorbAllDelay(sys *System) (*System, error) {
	cur := sys
	var err error

	if cur.HasInternalDelay() {
		cur, err = absorbInternalDelay(cur)
		if err != nil {
			return nil, err
		}
	}

	if cur.Delay != nil {
		cur, err = absorbIODelay(cur)
		if err != nil {
			return nil, err
		}
	}

	hasInput := false
	for _, v := range cur.InputDelay {
		if v != 0 {
			hasInput = true
			break
		}
	}
	if hasInput {
		cur, err = absorbInputDelay(cur)
		if err != nil {
			return nil, err
		}
	}

	hasOutput := false
	for _, v := range cur.OutputDelay {
		if v != 0 {
			hasOutput = true
			break
		}
	}
	if hasOutput {
		cur, err = absorbOutputDelay(cur)
		if err != nil {
			return nil, err
		}
	}

	return cur, nil
}

func absorbInternalDelay(sys *System) (*System, error) {
	N := sys.internalDelayCount()
	if N == 0 {
		return sys.Copy(), nil
	}

	if sys.IsDiscrete() {
		return absorbInternalDiscreteDelay(sys)
	}
	return absorbInternalContinuousDelay(sys)
}

func absorbInternalDiscreteDelay(sys *System) (*System, error) {
	internal := *sys
	internal.Delay = nil
	internal.InputDelay = nil
	internal.OutputDelay = nil
	H, tau, err := internal.GetDelayModel()
	if err != nil {
		return nil, err
	}
	n, mN, pN := H.Dims()
	N := len(tau)
	m := mN - N
	p := pN - N

	delays := make([]int, N)
	totalShift := 0
	for j := range N {
		delays[j] = int(math.Round(tau[j]))
		totalShift += delays[j]
	}

	if totalShift == 0 {
		cp := sys.Copy()
		cp.LFT = nil
		return cp, nil
	}

	nAug := n + totalShift

	// H partitions:
	// B = [B1 | B2], size n×(m+N)
	// C = [C1; C2], size (p+N)×n
	// D = [D11 D12; D21 D22], size (p+N)×(m+N)
	hB := H.B.RawMatrix()
	hC := H.C.RawMatrix()
	hD := H.D.RawMatrix()

	aAug := make([]float64, nAug*nAug)
	bAug := make([]float64, nAug*m)
	cAug := make([]float64, p*nAug)
	dAug := make([]float64, p*m)

	if n > 0 {
		hA := H.A.RawMatrix()
		for i := range n {
			copy(aAug[i*nAug:i*nAug+n], hA.Data[i*hA.Stride:i*hA.Stride+n])
		}
		for i := range n {
			copy(bAug[i*m:i*m+m], hB.Data[i*hB.Stride:i*hB.Stride+m])
		}
		for i := range p {
			copy(cAug[i*nAug:i*nAug+n], hC.Data[i*hC.Stride:i*hC.Stride+n])
		}
	}

	for i := range p {
		copy(dAug[i*m:i*m+m], hD.Data[i*hD.Stride:i*hD.Stride+m])
	}

	// For each internal delay j, build a shift chain of length d_j.
	// The chain connects z_j (output of H's lower block) to w_j (input to H's lower block).
	//
	// Shift chain states: s_1, s_2, ..., s_{d_j}
	//   s_1[k+1] = z_j[k] = C2[j,:]*x[k] + D21[j,:]*u[k] + D22[j,:]*w[k]
	//   s_t[k+1] = s_{t-1}[k]  for t=2..d_j
	//   w_j[k] = s_{d_j}[k]
	//
	// Since w depends on shift states and D22 may couple delays,
	// but each delay has tau_j > 0, so D22 only creates coupling through
	// delayed paths. For the discrete case with d_j >= 1, w_j[k] reads
	// from the last shift state, which was written at least 1 step ago.
	// So the loop w -> D22*w is resolved by the shift registers, no algebraic loop.

	offset := n
	for j := range N {
		dj := delays[j]
		if dj == 0 {
			continue
		}

		// s_1[k+1] = C2[j,:]*x[k] + D21[j,:]*u[k]
		// (D22 contribution is handled after all shift chains are placed)
		if n > 0 {
			for col := range n {
				aAug[offset*nAug+col] = hC.Data[(p+j)*hC.Stride+col]
			}
		}
		for col := range m {
			bAug[offset*m+col] = hD.Data[(p+j)*hD.Stride+col]
		}

		for t := 1; t < dj; t++ {
			aAug[(offset+t)*nAug+(offset+t-1)] = 1
		}

		offset += dj
	}

	// Now handle the coupling: s_1[k+1] += D22[j,:]*w[k]
	// where w_j[k] = s_{last_j}[k] (last state of chain j).
	// Build a map from delay index to its last shift state column.
	lastState := make([]int, N)
	off := n
	for j := range N {
		lastState[j] = off + delays[j] - 1
		off += delays[j]
	}

	off = n
	for j := range N {
		dj := delays[j]
		if dj == 0 {
			continue
		}
		for k := range N {
			dk := delays[k]
			if dk == 0 {
				continue
			}
			d22val := hD.Data[(p+j)*hD.Stride+(m+k)]
			if d22val != 0 {
				aAug[off*nAug+lastState[k]] += d22val
			}
		}
		off += dj
	}

	// B2 columns of H feed into original states: A[i,:] already has B2*w contribution
	// through the shift chain last states.
	// Original A_aug[i, lastState[j]] += B2[i,j] for the original state rows.
	if n > 0 {
		for i := range n {
			for j := range N {
				if delays[j] == 0 {
					continue
				}
				b2val := hB.Data[i*hB.Stride+(m+j)]
				if b2val != 0 {
					aAug[i*nAug+lastState[j]] += b2val
				}
			}
		}
	}

	// D12 columns: C_aug[i, lastState[j]] += D12[i,j]
	for i := range p {
		for j := range N {
			if delays[j] == 0 {
				continue
			}
			d12val := hD.Data[i*hD.Stride+(m+j)]
			if d12val != 0 {
				cAug[i*nAug+lastState[j]] += d12val
			}
		}
	}

	result, err := newNoCopy(
		mat.NewDense(nAug, nAug, aAug),
		denseFromData(nAug, m, bAug),
		denseFromData(p, nAug, cAug),
		denseFromData(p, m, dAug),
		sys.Dt,
	)
	if err != nil {
		return nil, err
	}
	result.E = augmentDescriptorE(sys.E, n, nAug)
	result.Delay = copyDelayOrNil(sys.Delay)
	if sys.InputDelay != nil {
		result.InputDelay = make([]float64, len(sys.InputDelay))
		copy(result.InputDelay, sys.InputDelay)
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, len(sys.OutputDelay))
		copy(result.OutputDelay, sys.OutputDelay)
	}
	propagateIONames(result, sys)
	return result, nil
}

func absorbInternalContinuousDelay(sys *System) (*System, error) {
	internal := *sys
	internal.Delay = nil
	internal.InputDelay = nil
	internal.OutputDelay = nil
	result, err := padeCloseInternalDelay(&internal, DefaultPadeOrder)
	if err != nil {
		return nil, fmt.Errorf("absorbInternalDelay: %w", err)
	}
	result.Delay = copyDelayOrNil(sys.Delay)
	result.InputDelay = copySliceOrNil(sys.InputDelay)
	result.OutputDelay = copySliceOrNil(sys.OutputDelay)
	propagateIONames(result, sys)
	return result, nil
}

func addBlock(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
	if src == nil {
		return
	}
	sr, sc := src.Dims()
	if sr == 0 || sc == 0 {
		return
	}
	dRaw := dst.RawMatrix()
	sRaw := src.RawMatrix()
	for i := range sr {
		dRow := dRaw.Data[(r0+i)*dRaw.Stride+c0:]
		sRow := sRaw.Data[i*sRaw.Stride:]
		for j := range sc {
			dRow[j] += sRow[j]
		}
	}
}

func absorbIODelay(sys *System) (*System, error) {
	if sys.Delay == nil {
		cp := sys.Copy()
		return cp, nil
	}

	_, m, p := sys.Dims()

	hasNonzero := false
	raw := sys.Delay.RawMatrix()
	for i := range p {
		for j := range m {
			if raw.Data[i*raw.Stride+j] != 0 {
				hasNonzero = true
				break
			}
		}
		if hasNonzero {
			break
		}
	}
	if !hasNonzero {
		cp := sys.Copy()
		cp.Delay = nil
		return cp, nil
	}

	inDel, outDel, residual := DecomposeIODelay(sys.Delay)
	if delayMatrixHasNonzero(residual) {
		return newResidualDelaySplit(sys).apply(sys, absorbDecomposedDelay)
	}

	cp := sys.Copy()

	if cp.InputDelay == nil {
		cp.InputDelay = make([]float64, m)
	}
	for j := range m {
		cp.InputDelay[j] += inDel[j]
	}

	if cp.OutputDelay == nil {
		cp.OutputDelay = make([]float64, p)
	}
	for i := range p {
		cp.OutputDelay[i] += outDel[i]
	}

	cp.Delay = residual

	cur := cp
	var err error

	hasInput := false
	for _, v := range cur.InputDelay {
		if v != 0 {
			hasInput = true
			break
		}
	}
	if hasInput {
		cur, err = absorbInputDelay(cur)
		if err != nil {
			return nil, err
		}
	}

	hasOutput := false
	for _, v := range cur.OutputDelay {
		if v != 0 {
			hasOutput = true
			break
		}
	}
	if hasOutput {
		cur, err = absorbOutputDelay(cur)
		if err != nil {
			return nil, err
		}
	} else {
		cur.OutputDelay = nil
	}

	allZeroInput := true
	for _, v := range cur.InputDelay {
		if v != 0 {
			allZeroInput = false
			break
		}
	}
	if allZeroInput {
		cur.InputDelay = nil
	}

	return cur, nil
}

func absorbInputDelay(sys *System) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("absorbInputDelay: %w", ErrWrongDomain)
	}

	n, m, p := sys.Dims()

	totalShift := 0
	delays := make([]int, m)
	if sys.InputDelay != nil {
		for j, d := range sys.InputDelay {
			delays[j] = int(math.Round(d))
			totalShift += delays[j]
		}
	}

	if totalShift == 0 {
		cp := sys.Copy()
		cp.InputDelay = nil
		return cp, nil
	}

	nAug := n + totalShift

	aAug := make([]float64, nAug*nAug)
	bAug := make([]float64, nAug*m)
	cAug := make([]float64, p*nAug)
	dAug := make([]float64, p*m)

	if n > 0 {
		aRaw := sys.A.RawMatrix()
		for i := range n {
			copy(aAug[i*nAug:i*nAug+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
		}
	}

	if n > 0 {
		cRaw := sys.C.RawMatrix()
		for i := range p {
			copy(cAug[i*nAug:i*nAug+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
		}
	}

	bRaw := sys.B.RawMatrix()
	dRaw := sys.D.RawMatrix()

	lastShift := make([]int, m)
	offset := n
	for j := range m {
		dj := delays[j]
		lastShift[j] = -1
		if dj == 0 {
			if n > 0 {
				for i := range n {
					bAug[i*m+j] = bRaw.Data[i*bRaw.Stride+j]
				}
			}
			for i := range p {
				dAug[i*m+j] = dRaw.Data[i*dRaw.Stride+j]
			}
			continue
		}

		bAug[offset*m+j] = 1

		for t := 1; t < dj; t++ {
			aAug[(offset+t)*nAug+(offset+t-1)] = 1
		}

		last := offset + dj - 1
		lastShift[j] = last
		if n > 0 {
			for i := range n {
				aAug[i*nAug+last] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
		for i := range p {
			cAug[i*nAug+last] += dRaw.Data[i*dRaw.Stride+j]
		}

		offset += dj
	}

	aAugMat := mat.NewDense(nAug, nAug, aAug)
	bAugMat := mat.NewDense(nAug, m, bAug)
	cAugMat := mat.NewDense(p, nAug, cAug)
	dAugMat := mat.NewDense(p, m, dAug)

	augSys, err := newNoCopy(aAugMat, bAugMat, cAugMat, dAugMat, sys.Dt)
	if err != nil {
		return nil, err
	}
	augSys.E = augmentDescriptorE(sys.E, n, nAug)
	augSys.LFT = inputShiftLFT(sys.LFT, n, nAug, lastShift)
	augSys.Delay = copyDelayOrNil(sys.Delay)
	if sys.OutputDelay != nil {
		augSys.OutputDelay = make([]float64, len(sys.OutputDelay))
		copy(augSys.OutputDelay, sys.OutputDelay)
	}
	propagateIONames(augSys, sys)
	return augSys, nil
}

func absorbOutputDelay(sys *System) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("absorbOutputDelay: %w", ErrWrongDomain)
	}

	n, m, p := sys.Dims()

	totalShift := 0
	delays := make([]int, p)
	if sys.OutputDelay != nil {
		for i, d := range sys.OutputDelay {
			delays[i] = int(math.Round(d))
			totalShift += delays[i]
		}
	}

	if totalShift == 0 {
		cp := sys.Copy()
		cp.OutputDelay = nil
		return cp, nil
	}

	nAug := n + totalShift

	aAug := make([]float64, nAug*nAug)
	bAug := make([]float64, nAug*m)
	cAug := make([]float64, p*nAug)
	dAug := make([]float64, p*m)

	if n > 0 {
		aRaw := sys.A.RawMatrix()
		for i := range n {
			copy(aAug[i*nAug:i*nAug+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
		}

		bRaw := sys.B.RawMatrix()
		for i := range n {
			copy(bAug[i*m:i*m+m], bRaw.Data[i*bRaw.Stride:i*bRaw.Stride+m])
		}
	}

	cRaw := sys.C.RawMatrix()
	dRaw := sys.D.RawMatrix()

	firstShift := make([]int, p)
	offset := n
	for i := range p {
		di := delays[i]
		firstShift[i] = -1
		if di == 0 {
			if n > 0 {
				copy(cAug[i*nAug:i*nAug+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
			}
			copy(dAug[i*m:i*m+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
			continue
		}

		firstShift[i] = offset
		// w_1[k+1] = C[i,:]*x[k] + D[i,:]*u[k]
		if n > 0 {
			copy(aAug[offset*nAug:offset*nAug+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
		}
		copy(bAug[offset*m:offset*m+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])

		// w_{t+1}[k+1] = w_t[k]
		for t := 1; t < di; t++ {
			aAug[(offset+t)*nAug+(offset+t-1)] = 1
		}

		// y_i[k] = w_{d_i}[k]
		cAug[i*nAug+(offset+di-1)] = 1

		offset += di
	}

	aAugMat := mat.NewDense(nAug, nAug, aAug)
	bAugMat := mat.NewDense(nAug, m, bAug)
	cAugMat := mat.NewDense(p, nAug, cAug)
	dAugMat := mat.NewDense(p, m, dAug)

	augSys, err := newNoCopy(aAugMat, bAugMat, cAugMat, dAugMat, sys.Dt)
	if err != nil {
		return nil, err
	}
	augSys.E = augmentDescriptorE(sys.E, n, nAug)
	augSys.LFT = outputShiftLFT(sys.LFT, n, nAug, firstShift)
	augSys.Delay = copyDelayOrNil(sys.Delay)
	if sys.InputDelay != nil {
		augSys.InputDelay = make([]float64, len(sys.InputDelay))
		copy(augSys.InputDelay, sys.InputDelay)
	}
	propagateIONames(augSys, sys)
	return augSys, nil
}

// augmentDescriptorE extends E with identity rows for the shift-register
// states appended after the first n states.
func augmentDescriptorE(E *mat.Dense, n, nAug int) *mat.Dense {
	if E == nil {
		return nil
	}
	out := mat.NewDense(nAug, nAug, nil)
	setBlock(out, 0, 0, E)
	raw := out.RawMatrix()
	for i := n; i < nAug; i++ {
		raw.Data[i*raw.Stride+i] = 1
	}
	return out
}

func copyLFTWithStates(lft *LFTDelay, n, nAug, m, p int) *LFTDelay {
	q := len(lft.Tau)
	out := &LFTDelay{
		Tau: append([]float64(nil), lft.Tau...),
		B2:  newDense(nAug, q),
		C2:  newDense(q, nAug),
		D12: denseCopySafe(lft.D12, p, q),
		D21: denseCopySafe(lft.D21, q, m),
		D22: denseCopySafe(lft.D22, q, q),
	}
	if n > 0 {
		setBlock(out.B2, 0, 0, lft.B2)
		setBlock(out.C2, 0, 0, lft.C2)
	}
	return out
}

// inputShiftLFT moves each delayed input's D21 column onto the C2 column of
// its register's last state, so internal delays see the delayed input.
func inputShiftLFT(lft *LFTDelay, n, nAug int, lastShift []int) *LFTDelay {
	if lft == nil || len(lft.Tau) == 0 {
		return nil
	}
	m := len(lastShift)
	p, _ := lft.D12.Dims()
	out := copyLFTWithStates(lft, n, nAug, m, p)
	c2 := out.C2.RawMatrix()
	d21 := out.D21.RawMatrix()
	for j, last := range lastShift {
		if last < 0 {
			continue
		}
		for k := range len(lft.Tau) {
			c2.Data[k*c2.Stride+last] = d21.Data[k*d21.Stride+j]
			d21.Data[k*d21.Stride+j] = 0
		}
	}
	return out
}

// outputShiftLFT moves each delayed output's D12 row onto the B2 row of its
// register's first state, so the register captures the full undelayed output.
func outputShiftLFT(lft *LFTDelay, n, nAug int, firstShift []int) *LFTDelay {
	if lft == nil || len(lft.Tau) == 0 {
		return nil
	}
	_, m := lft.D21.Dims()
	out := copyLFTWithStates(lft, n, nAug, m, len(firstShift))
	q := len(lft.Tau)
	b2 := out.B2.RawMatrix()
	d12 := out.D12.RawMatrix()
	for i, first := range firstShift {
		if first < 0 {
			continue
		}
		copy(b2.Data[first*b2.Stride:first*b2.Stride+q], d12.Data[i*d12.Stride:i*d12.Stride+q])
		clear(d12.Data[i*d12.Stride : i*d12.Stride+q])
	}
	return out
}

func buildPadeBank(delays []float64, order int) (*System, error) {
	return buildContinuousPadeDelayBank(delays, order)
}

func absorbInputDelayContinuous(sys *System, order int) (*System, error) {
	hasInput := false
	if sys.InputDelay != nil {
		for _, v := range sys.InputDelay {
			if v != 0 {
				hasInput = true
				break
			}
		}
	}
	if !hasInput {
		cp := sys.Copy()
		cp.InputDelay = nil
		return cp, nil
	}

	bank, err := buildPadeBank(sys.InputDelay, order)
	if err != nil {
		return nil, fmt.Errorf("absorbInputDelayContinuous: %w", err)
	}

	plant := sys.Copy()
	plant.InputDelay = nil

	result, err := Series(bank, plant)
	if err != nil {
		return nil, fmt.Errorf("absorbInputDelayContinuous: %w", err)
	}

	return result, nil
}

func absorbOutputDelayContinuous(sys *System, order int) (*System, error) {
	hasOutput := false
	if sys.OutputDelay != nil {
		for _, v := range sys.OutputDelay {
			if v != 0 {
				hasOutput = true
				break
			}
		}
	}
	if !hasOutput {
		cp := sys.Copy()
		cp.OutputDelay = nil
		return cp, nil
	}

	bank, err := buildPadeBank(sys.OutputDelay, order)
	if err != nil {
		return nil, fmt.Errorf("absorbOutputDelayContinuous: %w", err)
	}

	plant := sys.Copy()
	plant.OutputDelay = nil

	result, err := Series(plant, bank)
	if err != nil {
		return nil, fmt.Errorf("absorbOutputDelayContinuous: %w", err)
	}

	return result, nil
}

func absorbIODelayContinuous(sys *System, order int) (*System, error) {
	if sys.Delay == nil {
		cp := sys.Copy()
		return cp, nil
	}

	_, m, p := sys.Dims()

	hasNonzero := false
	raw := sys.Delay.RawMatrix()
	for i := range p {
		for j := range m {
			if raw.Data[i*raw.Stride+j] != 0 {
				hasNonzero = true
				break
			}
		}
		if hasNonzero {
			break
		}
	}
	if !hasNonzero {
		cp := sys.Copy()
		cp.Delay = nil
		return cp, nil
	}

	inDel, outDel, residual := DecomposeIODelay(sys.Delay)
	if delayMatrixHasNonzero(residual) {
		return newResidualDelaySplit(sys).apply(sys, func(piece *System) (*System, error) {
			cur, err := absorbInputDelayContinuous(piece, order)
			if err != nil {
				return nil, err
			}
			return absorbOutputDelayContinuous(cur, order)
		})
	}

	cp := sys.Copy()

	if cp.InputDelay == nil {
		cp.InputDelay = make([]float64, m)
	}
	for j := range m {
		cp.InputDelay[j] += inDel[j]
	}

	if cp.OutputDelay == nil {
		cp.OutputDelay = make([]float64, p)
	}
	for i := range p {
		cp.OutputDelay[i] += outDel[i]
	}

	cp.Delay = residual

	cur := cp
	var err error

	hasInput := false
	for _, v := range cur.InputDelay {
		if v != 0 {
			hasInput = true
			break
		}
	}
	if hasInput {
		cur, err = absorbInputDelayContinuous(cur, order)
		if err != nil {
			return nil, err
		}
	}

	hasOutput := false
	for _, v := range cur.OutputDelay {
		if v != 0 {
			hasOutput = true
			break
		}
	}
	if hasOutput {
		cur, err = absorbOutputDelayContinuous(cur, order)
		if err != nil {
			return nil, err
		}
	} else {
		cur.OutputDelay = nil
	}

	allZeroInput := true
	for _, v := range cur.InputDelay {
		if v != 0 {
			allZeroInput = false
			break
		}
	}
	if allZeroInput {
		cur.InputDelay = nil
	}

	return cur, nil
}

// DecomposeIODelay splits ioDelay into input delays, output delays and a
// nonnegative residual with ioDelay[i][j] = out[i] + in[j] + residual[i][j].
// Derived values within roundoff of zero relative to the largest delay are
// returned as exact zeros.
func DecomposeIODelay(ioDelay *mat.Dense) (inputDelay, outputDelay []float64, residual *mat.Dense) {
	raw := ioDelay.RawMatrix()
	p, m := raw.Rows, raw.Cols

	scale := 0.0
	for i := range p {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+m] {
			scale = max(scale, math.Abs(v))
		}
	}
	inI, outI, resI := decomposeInputFirst(raw.Data, raw.Stride, p, m)
	inO, outO, resO := decomposeOutputFirst(raw.Data, raw.Stride, p, m)
	snapDelayRoundoff(outI, scale)
	snapDelayRoundoff(resI, scale)
	snapDelayRoundoff(inO, scale)
	snapDelayRoundoff(resO, scale)

	sumI, sumO := 0.0, 0.0
	for _, v := range resI {
		sumI += v
	}
	for _, v := range resO {
		sumO += v
	}

	if sumI <= sumO {
		return inI, outI, mat.NewDense(p, m, resI)
	}
	return inO, outO, mat.NewDense(p, m, resO)
}

func decomposeInputFirst(data []float64, stride, p, m int) (inputDelay, outputDelay, residual []float64) {
	inputDelay = make([]float64, m)
	outputDelay = make([]float64, p)
	residual = make([]float64, p*m)

	for j := range m {
		mn := math.Inf(1)
		for i := range p {
			if v := data[i*stride+j]; v < mn {
				mn = v
			}
		}
		inputDelay[j] = mn
	}

	for i := range p {
		for j := range m {
			residual[i*m+j] = data[i*stride+j] - inputDelay[j]
		}
	}

	for i := range p {
		mn := math.Inf(1)
		for j := range m {
			if v := residual[i*m+j]; v < mn {
				mn = v
			}
		}
		outputDelay[i] = mn
	}

	for i := range p {
		for j := range m {
			residual[i*m+j] -= outputDelay[i]
		}
	}

	return
}

func decomposeOutputFirst(data []float64, stride, p, m int) (inputDelay, outputDelay, residual []float64) {
	inputDelay = make([]float64, m)
	outputDelay = make([]float64, p)
	residual = make([]float64, p*m)

	for i := range p {
		mn := math.Inf(1)
		for j := range m {
			if v := data[i*stride+j]; v < mn {
				mn = v
			}
		}
		outputDelay[i] = mn
	}

	for i := range p {
		for j := range m {
			residual[i*m+j] = data[i*stride+j] - outputDelay[i]
		}
	}

	for j := range m {
		mn := math.Inf(1)
		for i := range p {
			if v := residual[i*m+j]; v < mn {
				mn = v
			}
		}
		inputDelay[j] = mn
	}

	for i := range p {
		for j := range m {
			residual[i*m+j] -= inputDelay[j]
		}
	}

	return
}

// splitIODelayForInitialState moves Delay into InputDelay and OutputDelay,
// split as DecomposeIODelay (and PullDelaysToLFT) does, when x0 is nonzero.
// MATLAB ss has no I/O delay matrix: ss(tf) distributes it over input and
// output delays, and an output delay τ_i delays the whole output, so the free
// response y_i = C_i·x(t−τ_i) is zero before τ_i while delay lines start
// empty (https://www.mathworks.com/help/control/ug/specifying-time-delays.html).
// A Delay with no exact input+output split needs extra states that x0 cannot
// seed, so a nonzero x0 then returns ErrDelayUnsupported.
func (sys *System) splitIODelayForInitialState(x0 *mat.VecDense) (*System, error) {
	if sys.Delay == nil || x0 == nil || allZeroVec(x0) {
		return sys, nil
	}
	in, out, residual := DecomposeIODelay(sys.Delay)
	if delayMatrixHasNonzero(residual) {
		return nil, fmt.Errorf("nonzero x0 with an I/O delay matrix that has no input+output split: %w", ErrDelayUnsupported)
	}
	_, m, p := sys.Dims()
	cp := sys.Copy()
	if cp.InputDelay == nil {
		cp.InputDelay = make([]float64, m)
	}
	for j, d := range in {
		cp.InputDelay[j] += d
	}
	if cp.OutputDelay == nil {
		cp.OutputDelay = make([]float64, p)
	}
	for i, d := range out {
		cp.OutputDelay[i] += d
	}
	cp.Delay = nil
	return cp, nil
}

type delayEntry struct {
	row, col int
	tau      float64
	kind     byte // 'i' input, 'o' output, 'd' io-delay
}

// PullDelaysToLFT returns a copy of the system where all I/O delays (InputDelay,
// OutputDelay, and IODelay) have been pulled into the InternalDelay (LFT) structure.
func (sys *System) PullDelaysToLFT() (*System, error) {
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	hasIODelay := sys.Delay != nil || sys.InputDelay != nil || sys.OutputDelay != nil
	if !hasIODelay {
		return sys.Copy(), nil
	}

	n, m, p := sys.Dims()
	cur := sys.Copy()

	if cur.Delay != nil {
		inDel, outDel, residual := DecomposeIODelay(cur.Delay)
		if cur.InputDelay == nil {
			cur.InputDelay = make([]float64, m)
		}
		for j := range m {
			cur.InputDelay[j] += inDel[j]
		}
		if cur.OutputDelay == nil {
			cur.OutputDelay = make([]float64, p)
		}
		for i := range p {
			cur.OutputDelay[i] += outDel[i]
		}

		hasResidual := delayMatrixHasNonzero(residual)
		if (n > 0 || sys.internalDelayCount() > 0) && hasResidual {
			return newResidualDelaySplit(sys).apply(sys, (*System).PullDelaysToLFT)
		}
		if hasResidual {
			// Only static systems without internal delays reach here: merge
			// InputDelay/OutputDelay into the residual for overlapping channels
			// to avoid parallel double-counting of feedthrough gains.
			resRaw := residual.RawMatrix()
			if n == 0 && cur.InputDelay != nil {
				for j := range m {
					if cur.InputDelay[j] == 0 {
						continue
					}
					colHasRes := false
					for i := range p {
						if resRaw.Data[i*resRaw.Stride+j] > 0 {
							colHasRes = true
							break
						}
					}
					if colHasRes {
						for i := range p {
							resRaw.Data[i*resRaw.Stride+j] += cur.InputDelay[j]
						}
						cur.InputDelay[j] = 0
					}
				}
			}
			if n == 0 && cur.OutputDelay != nil {
				for i := range p {
					if cur.OutputDelay[i] == 0 {
						continue
					}
					rowHasRes := false
					row := resRaw.Data[i*resRaw.Stride : i*resRaw.Stride+m]
					for _, v := range row {
						if v > 0 {
							rowHasRes = true
							break
						}
					}
					if rowHasRes {
						for j := range row {
							row[j] += cur.OutputDelay[i]
						}
						cur.OutputDelay[i] = 0
					}
				}
			}
			cur.Delay = residual
		} else {
			cur.Delay = nil
		}
	}

	var entries []delayEntry
	if cur.InputDelay != nil {
		for j, tau := range cur.InputDelay {
			if tau > 0 {
				entries = append(entries, delayEntry{0, j, tau, 'i'})
			}
		}
	}
	if cur.OutputDelay != nil {
		for i, tau := range cur.OutputDelay {
			if tau > 0 {
				entries = append(entries, delayEntry{i, 0, tau, 'o'})
			}
		}
	}
	if cur.Delay != nil {
		raw := cur.Delay.RawMatrix()
		for i := range p {
			for j := range m {
				v := raw.Data[i*raw.Stride+j]
				if v > 0 {
					entries = append(entries, delayEntry{i, j, v, 'd'})
				}
			}
		}
	}

	N0 := sys.internalDelayCount()
	Nnew := len(entries)
	N := N0 + Nnew
	taus := make([]float64, 0, N)
	if N0 > 0 {
		taus = append(taus, sys.LFT.Tau...)
	}
	for _, e := range entries {
		taus = append(taus, e.tau)
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N0 > 0 {
		if n > 0 {
			setBlock(b2, 0, 0, sys.LFT.B2)
			setBlock(c2, 0, 0, sys.LFT.C2)
		}
		setBlock(d12, 0, 0, sys.LFT.D12)
		setBlock(d21, 0, 0, sys.LFT.D21)
		setBlock(d22, 0, 0, sys.LFT.D22)
	}

	newB := mat.DenseCopyOf(cur.B)
	newC := mat.DenseCopyOf(cur.C)
	newD := mat.DenseCopyOf(cur.D)

	inputDelayIdx := make(map[int]int)
	{
		tmpIdx := N0
		for _, e := range entries {
			if e.kind == 'i' {
				inputDelayIdx[e.col] = tmpIdx
			}
			tmpIdx++
		}
	}
	outputDelayRows := make(map[int]bool)
	for _, e := range entries {
		if e.kind == 'o' {
			outputDelayRows[e.row] = true
		}
	}

	b2Raw := b2.RawMatrix()
	c2Raw := c2.RawMatrix()
	d12Raw := d12.RawMatrix()
	d21Raw := d21.RawMatrix()
	d22Raw := d22.RawMatrix()
	newBRaw := newB.RawMatrix()
	newCRaw := newC.RawMatrix()
	newDRaw := newD.RawMatrix()
	curBRaw := cur.B.RawMatrix()
	curCRaw := cur.C.RawMatrix()
	curDRaw := cur.D.RawMatrix()

	idx := N0
	for _, e := range entries {
		switch e.kind {
		case 'i':
			j := e.col
			for k := range N0 {
				d22Raw.Data[k*d22Raw.Stride+idx] = d21Raw.Data[k*d21Raw.Stride+j]
				d21Raw.Data[k*d21Raw.Stride+j] = 0
			}
			if n > 0 {
				for i := range n {
					b2Raw.Data[i*b2Raw.Stride+idx] = curBRaw.Data[i*curBRaw.Stride+j]
					newBRaw.Data[i*newBRaw.Stride+j] = 0
				}
			}
			for i := range p {
				dVal := curDRaw.Data[i*curDRaw.Stride+j]
				if !outputDelayRows[i] {
					d12Raw.Data[i*d12Raw.Stride+idx] = dVal
				}
				newDRaw.Data[i*newDRaw.Stride+j] = 0
			}
			d21Raw.Data[idx*d21Raw.Stride+j] = 1

		case 'o':
			i := e.row
			for k := range N0 {
				d22Raw.Data[idx*d22Raw.Stride+k] = d12Raw.Data[i*d12Raw.Stride+k]
				d12Raw.Data[i*d12Raw.Stride+k] = 0
			}
			if n > 0 {
				for j := range n {
					c2Raw.Data[idx*c2Raw.Stride+j] = curCRaw.Data[i*curCRaw.Stride+j]
					newCRaw.Data[i*newCRaw.Stride+j] = 0
				}
			}
			for j := range m {
				dVal := curDRaw.Data[i*curDRaw.Stride+j]
				if inIdx, ok := inputDelayIdx[j]; ok {
					d22Raw.Data[idx*d22Raw.Stride+inIdx] = dVal
				} else {
					d21Raw.Data[idx*d21Raw.Stride+j] = dVal
				}
				newDRaw.Data[i*newDRaw.Stride+j] = 0
			}
			d12Raw.Data[i*d12Raw.Stride+idx] = 1

		case 'd':
			i, j := e.row, e.col
			d21Raw.Data[idx*d21Raw.Stride+j] = 1
			d12Raw.Data[i*d12Raw.Stride+idx] = newDRaw.Data[i*newDRaw.Stride+j]
			newDRaw.Data[i*newDRaw.Stride+j] = 0
		}
		idx++
	}

	if n == 0 {
		b2 = &mat.Dense{}
		c2 = &mat.Dense{}
	} else {
		b2 = resizeDense(b2, n, N)
		c2 = resizeDense(c2, N, n)
	}
	d12 = resizeDense(d12, p, N)
	d21 = resizeDense(d21, N, m)
	d22 = resizeDense(d22, N, N)

	res := &System{
		A:  denseCopy(cur.A),
		B:  newB,
		C:  newC,
		D:  newD,
		E:  cur.E,
		Dt: cur.Dt,
		LFT: &LFTDelay{
			Tau: taus,
			B2:  b2,
			C2:  c2,
			D12: d12,
			D21: d21,
			D22: d22,
		},
	}
	propagateIONames(res, sys)
	res.StateName = copyStringSlice(sys.StateName)
	return res, nil
}

// GetDelayModel returns the delay-free model H and internal delays tau of sys,
// as MATLAB [H,tau] = getDelayModel(sys)
// (https://www.mathworks.com/help/control/ref/getdelaymodel.html). Input,
// output and I/O delays are first pulled into internal delays. The last
// len(tau) inputs and outputs of H are the delay channels: sys is the LFT of H
// closed by exp(-s·tau) (z^-tau for discrete models). A model without delays
// returns a copy of sys and an empty tau.
func (sys *System) GetDelayModel() (H *System, tau []float64, err error) {
	if err := requireSystem("GetDelayModel", sys); err != nil {
		return nil, nil, err
	}
	if !sys.HasDelay() {
		return sys.Copy(), []float64{}, nil
	}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		return nil, nil, fmt.Errorf("GetDelayModel: %w", err)
	}

	// Get augmented model H from LFT structure
	n, m, p := lft.Dims()
	N := len(lft.LFT.Tau)
	tau = make([]float64, N)
	copy(tau, lft.LFT.Tau)

	H = &System{
		A:  denseCopy(lft.A),
		B:  newDense(n, m+N),
		C:  newDense(p+N, n),
		D:  newDense(p+N, m+N),
		E:  copyDescriptorE(lft.E),
		Dt: lft.Dt,
	}
	if n > 0 {
		setBlock(H.B, 0, 0, lft.B)
		setBlock(H.B, 0, m, lft.LFT.B2)
		setBlock(H.C, 0, 0, lft.C)
		setBlock(H.C, p, 0, lft.LFT.C2)
	}
	setBlock(H.D, 0, 0, lft.D)
	setBlock(H.D, 0, m, lft.LFT.D12)
	setBlock(H.D, p, 0, lft.LFT.D21)
	setBlock(H.D, p, m, lft.LFT.D22)

	return H, tau, nil
}

// SetDelayModel builds a model with internal delays tau from the delay-free
// model H partitioned as in GetDelayModel, the inverse of GetDelayModel;
// it mirrors MATLAB setDelayModel(H,tau)
// (https://www.mathworks.com/help/control/ref/setdelaymodel.html). The last
// len(tau) inputs and outputs of H are the delay channels. Each tau must be
// finite and positive, and an integer number of samples for a discrete H. An
// empty tau returns a copy of H.
func SetDelayModel(H *System, tau []float64) (*System, error) {
	if err := requireSystem("SetDelayModel", H); err != nil {
		return nil, err
	}
	N := len(tau)
	if N == 0 {
		return H.Copy(), nil
	}
	if err := validateInternalTau(tau, H.Dt); err != nil {
		return nil, fmt.Errorf("SetDelayModel: %w", err)
	}

	n, mN, pN := H.Dims()
	if mN < N || pN < N {
		return nil, fmt.Errorf("SetDelayModel: H is %d×%d, too small for %d internal delays: %w",
			pN, mN, N, ErrDimensionMismatch)
	}
	m := mN - N
	p := pN - N

	var aMat *mat.Dense
	if n > 0 {
		aMat = mat.DenseCopyOf(H.A)
	} else {
		aMat = &mat.Dense{}
	}

	bData := make([]float64, n*m)
	b2Data := make([]float64, n*N)
	cData := make([]float64, p*n)
	c2Data := make([]float64, N*n)
	dData := make([]float64, p*m)
	d12Data := make([]float64, p*N)
	d21Data := make([]float64, N*m)
	d22Data := make([]float64, N*N)

	if n > 0 {
		hbRaw := H.B.RawMatrix()
		for i := range n {
			copy(bData[i*m:i*m+m], hbRaw.Data[i*hbRaw.Stride:i*hbRaw.Stride+m])
			copy(b2Data[i*N:i*N+N], hbRaw.Data[i*hbRaw.Stride+m:i*hbRaw.Stride+mN])
		}

		hcRaw := H.C.RawMatrix()
		for i := range p {
			copy(cData[i*n:i*n+n], hcRaw.Data[i*hcRaw.Stride:i*hcRaw.Stride+n])
		}
		for i := range N {
			copy(c2Data[i*n:i*n+n], hcRaw.Data[(p+i)*hcRaw.Stride:(p+i)*hcRaw.Stride+n])
		}
	}

	hdRaw := H.D.RawMatrix()
	for i := range p {
		copy(dData[i*m:i*m+m], hdRaw.Data[i*hdRaw.Stride:i*hdRaw.Stride+m])
		copy(d12Data[i*N:i*N+N], hdRaw.Data[i*hdRaw.Stride+m:i*hdRaw.Stride+mN])
	}
	for i := range N {
		copy(d21Data[i*m:i*m+m], hdRaw.Data[(p+i)*hdRaw.Stride:(p+i)*hdRaw.Stride+m])
		copy(d22Data[i*N:i*N+N], hdRaw.Data[(p+i)*hdRaw.Stride+m:(p+i)*hdRaw.Stride+mN])
	}

	tauCopy := make([]float64, N)
	copy(tauCopy, tau)

	b2Mat, c2Mat := &mat.Dense{}, &mat.Dense{}
	if n > 0 {
		b2Mat = mat.NewDense(n, N, b2Data)
		c2Mat = mat.NewDense(N, n, c2Data)
	}

	var bMat, cMat, dMat *mat.Dense
	if n > 0 && m > 0 {
		bMat = mat.NewDense(n, m, bData)
	} else if n > 0 {
		bMat = newDense(n, m)
	} else {
		bMat = &mat.Dense{}
	}
	if p > 0 && n > 0 {
		cMat = mat.NewDense(p, n, cData)
	} else if p > 0 {
		cMat = newDense(p, n)
	} else {
		cMat = &mat.Dense{}
	}
	if p > 0 && m > 0 {
		dMat = mat.NewDense(p, m, dData)
	} else {
		dMat = newDense(p, m)
	}

	result := &System{
		A:  aMat,
		B:  bMat,
		C:  cMat,
		D:  dMat,
		E:  copyDescriptorE(H.E),
		Dt: H.Dt,
		LFT: &LFTDelay{
			Tau: tauCopy,
			B2:  b2Mat,
			C2:  c2Mat,
			D12: denseFromData(p, N, d12Data),
			D21: denseFromData(N, m, d21Data),
			D22: mat.NewDense(N, N, d22Data),
		},
	}
	propagateIONames(result, H)
	return result, nil
}

func copyDelayOrNil(m *mat.Dense) *mat.Dense {
	if m == nil {
		return nil
	}
	r, c := m.Dims()
	if r == 0 || c == 0 {
		return nil
	}
	return mat.DenseCopyOf(m)
}

func validateDelay(delay *mat.Dense, p, m int, dt float64) error {
	if delay == nil {
		return nil
	}
	dr, dc := delay.Dims()
	if dr != p || dc != m {
		return fmt.Errorf("delay %d×%d != p×m %d×%d: %w", dr, dc, p, m, ErrDimensionMismatch)
	}
	raw := delay.RawMatrix()
	for i := range dr {
		for j := range dc {
			if err := validateDelayValue(raw.Data[i*raw.Stride+j], dt); err != nil {
				return err
			}
		}
	}
	return nil
}

func convertDelayToDiscrete(delay *mat.Dense, dt float64) (*mat.Dense, error) {
	return newTimeDomain(dt).convertDelayMatrixToDiscrete(delay)
}

func convertDelayToContinuous(delay *mat.Dense, dt float64) *mat.Dense {
	return newTimeDomain(dt).convertDelayMatrixToContinuous(delay)
}

func denseToSlice2D(m *mat.Dense) [][]float64 {
	if m == nil {
		return nil
	}
	raw := m.RawMatrix()
	r, c := raw.Rows, raw.Cols
	out := make([][]float64, r)
	for i := range r {
		out[i] = make([]float64, c)
		copy(out[i], raw.Data[i*raw.Stride:i*raw.Stride+c])
	}
	return out
}

func slice2DToDense(s [][]float64) *mat.Dense {
	if s == nil {
		return nil
	}
	p := len(s)
	if p == 0 {
		return nil
	}
	m := len(s[0])
	data := make([]float64, p*m)
	for i := range p {
		copy(data[i*m:], s[i][:m])
	}
	return mat.NewDense(p, m, data)
}

func (sys *System) MinimalLFT() (*System, error) {
	if !sys.HasInternalDelay() {
		return sys.Copy(), nil
	}

	N := len(sys.LFT.Tau)
	n, m, p := sys.Dims()

	keep := make([]int, 0, N)
	for j := range N {
		if !isZeroGainChannel(sys, j, n, m, p, N) {
			keep = append(keep, j)
		}
	}

	if len(keep) == 0 {
		result := sys.Copy()
		result.LFT = nil
		return result, nil
	}

	cur := sys
	if len(keep) < N {
		cur = lftSelectChannels(sys, keep, n, m, p)
	}

	merged := lftMergeProportional(cur, n, m, p)

	if merged == cur && cur == sys {
		return sys.Copy(), nil
	}
	if merged == cur {
		return cur, nil
	}
	return merged, nil
}

func lftSelectChannels(sys *System, keep []int, n, m, p int) *System {
	Nk := len(keep)
	newTau := make([]float64, Nk)
	newB2 := newDense(n, Nk)
	newC2 := newDense(Nk, n)
	newD12 := newDense(p, Nk)
	newD21 := newDense(Nk, m)
	newD22 := newDense(Nk, Nk)

	b2Raw := rawOrEmpty(sys.LFT.B2)
	c2Raw := rawOrEmpty(sys.LFT.C2)
	d12Raw := rawOrEmpty(sys.LFT.D12)
	d21Raw := rawOrEmpty(sys.LFT.D21)
	d22Raw := rawOrEmpty(sys.LFT.D22)
	nb2 := newB2.RawMatrix()
	nc2 := newC2.RawMatrix()
	nd12 := newD12.RawMatrix()
	nd21 := newD21.RawMatrix()
	nd22 := newD22.RawMatrix()

	for ki, j := range keep {
		newTau[ki] = sys.LFT.Tau[j]
		for i := range n {
			nb2.Data[i*nb2.Stride+ki] = b2Raw.Data[i*b2Raw.Stride+j]
		}
		for i := range n {
			nc2.Data[ki*nc2.Stride+i] = c2Raw.Data[j*c2Raw.Stride+i]
		}
		for i := range p {
			nd12.Data[i*nd12.Stride+ki] = d12Raw.Data[i*d12Raw.Stride+j]
		}
		for i := range m {
			nd21.Data[ki*nd21.Stride+i] = d21Raw.Data[j*d21Raw.Stride+i]
		}
		for kj, jj := range keep {
			nd22.Data[ki*nd22.Stride+kj] = d22Raw.Data[j*d22Raw.Stride+jj]
		}
	}

	result := sys.Copy()
	result.LFT = &LFTDelay{Tau: newTau, B2: newB2, C2: newC2, D12: newD12, D21: newD21, D22: newD22}
	return result
}

func lftMergeProportional(sys *System, n, m, p int) *System {
	const tol = 1e-12
	N := len(sys.LFT.Tau)
	if N < 2 {
		return sys
	}

	c2Raw := rawOrEmpty(sys.LFT.C2)
	d21Raw := rawOrEmpty(sys.LFT.D21)
	d22Raw := rawOrEmpty(sys.LFT.D22)

	merged := make([]int, N)
	for i := range merged {
		merged[i] = i
	}
	alpha := make([]float64, N)
	for i := range alpha {
		alpha[i] = 1
	}

	for i := range N {
		if merged[i] != i {
			continue
		}
		for j := i + 1; j < N; j++ {
			if merged[j] != j {
				continue
			}
			if math.Abs(sys.LFT.Tau[i]-sys.LFT.Tau[j]) > tol {
				continue
			}
			if !d22ZeroCrossCoupling(d22Raw, i, j, N, tol) {
				continue
			}
			a, ok := proportionalRows(c2Raw, d21Raw, i, j, n, m, tol)
			if !ok {
				continue
			}
			merged[j] = i
			alpha[j] = a
		}
	}

	changed := false
	for i := range N {
		if merged[i] != i {
			changed = true
			break
		}
	}
	if !changed {
		return sys
	}

	reps := make([]int, 0, N)
	for i := range N {
		if merged[i] == i {
			reps = append(reps, i)
		}
	}
	Nk := len(reps)

	repIdx := make(map[int]int, Nk)
	for ki, r := range reps {
		repIdx[r] = ki
	}

	b2Raw := rawOrEmpty(sys.LFT.B2)
	d12Raw := rawOrEmpty(sys.LFT.D12)

	newTau := make([]float64, Nk)
	newB2 := newDense(n, Nk)
	newC2 := newDense(Nk, n)
	newD12 := newDense(p, Nk)
	newD21 := newDense(Nk, m)
	newD22 := newDense(Nk, Nk)
	nb2 := newB2.RawMatrix()
	nc2 := newC2.RawMatrix()
	nd12 := newD12.RawMatrix()
	nd21 := newD21.RawMatrix()
	nd22 := newD22.RawMatrix()

	for ki, r := range reps {
		newTau[ki] = sys.LFT.Tau[r]
		for i := range n {
			nc2.Data[ki*nc2.Stride+i] = c2Raw.Data[r*c2Raw.Stride+i]
		}
		for i := range m {
			nd21.Data[ki*nd21.Stride+i] = d21Raw.Data[r*d21Raw.Stride+i]
		}
	}

	for j := range N {
		r := merged[j]
		ki := repIdx[r]
		a := alpha[j]
		for i := range n {
			nb2.Data[i*nb2.Stride+ki] += a * b2Raw.Data[i*b2Raw.Stride+j]
		}
		for i := range p {
			nd12.Data[i*nd12.Stride+ki] += a * d12Raw.Data[i*d12Raw.Stride+j]
		}
	}

	for ki, ri := range reps {
		for kj, rj := range reps {
			nd22.Data[ki*nd22.Stride+kj] = d22Raw.Data[ri*d22Raw.Stride+rj]
		}
	}

	result := sys.Copy()
	result.LFT = &LFTDelay{Tau: newTau, B2: newB2, C2: newC2, D12: newD12, D21: newD21, D22: newD22}
	return result
}

func d22ZeroCrossCoupling(d22Raw blas64.General, i, j, N int, tol float64) bool {
	if math.Abs(d22Raw.Data[i*d22Raw.Stride+j]) > tol {
		return false
	}
	if math.Abs(d22Raw.Data[j*d22Raw.Stride+i]) > tol {
		return false
	}
	return true
}

func proportionalRows(c2Raw, d21Raw blas64.General, i, j, n, m int, tol float64) (float64, bool) {
	var a float64
	found := false

	for k := range n {
		vi := c2Raw.Data[i*c2Raw.Stride+k]
		vj := c2Raw.Data[j*c2Raw.Stride+k]
		if math.Abs(vi) <= tol && math.Abs(vj) <= tol {
			continue
		}
		if math.Abs(vi) <= tol {
			return 0, false
		}
		ratio := vj / vi
		if !found {
			a = ratio
			found = true
		} else if math.Abs(ratio-a) > tol*math.Max(1, math.Abs(a)) {
			return 0, false
		}
	}

	for k := range m {
		vi := d21Raw.Data[i*d21Raw.Stride+k]
		vj := d21Raw.Data[j*d21Raw.Stride+k]
		if math.Abs(vi) <= tol && math.Abs(vj) <= tol {
			continue
		}
		if math.Abs(vi) <= tol {
			return 0, false
		}
		ratio := vj / vi
		if !found {
			a = ratio
			found = true
		} else if math.Abs(ratio-a) > tol*math.Max(1, math.Abs(a)) {
			return 0, false
		}
	}

	if !found {
		return 0, false
	}
	return a, true
}

func isZeroGainChannel(sys *System, j, n, m, p, N int) bool {
	const tol = 1e-15
	b2Raw := rawOrEmpty(sys.LFT.B2)
	for i := range n {
		if math.Abs(b2Raw.Data[i*b2Raw.Stride+j]) > tol {
			return false
		}
	}
	d12Raw := rawOrEmpty(sys.LFT.D12)
	for i := range p {
		if math.Abs(d12Raw.Data[i*d12Raw.Stride+j]) > tol {
			return false
		}
	}
	c2Raw := rawOrEmpty(sys.LFT.C2)
	for i := range n {
		if math.Abs(c2Raw.Data[j*c2Raw.Stride+i]) > tol {
			return false
		}
	}
	d21Raw := rawOrEmpty(sys.LFT.D21)
	for i := range m {
		if math.Abs(d21Raw.Data[j*d21Raw.Stride+i]) > tol {
			return false
		}
	}
	d22Raw := rawOrEmpty(sys.LFT.D22)
	for i := range N {
		if math.Abs(d22Raw.Data[j*d22Raw.Stride+i]) > tol {
			return false
		}
		if math.Abs(d22Raw.Data[i*d22Raw.Stride+j]) > tol {
			return false
		}
	}
	return true
}

// ZeroDelayApprox returns sys with every internal delay set to zero, closing
// the delay loops algebraically. A model without internal delays is returned
// as a copy. An ill-posed loop (I-D22 singular) returns ErrAlgebraicLoop.
func (sys *System) ZeroDelayApprox() (*System, error) {
	if err := requireSystem("ZeroDelayApprox", sys); err != nil {
		return nil, err
	}
	if sys.internalDelayCount() == 0 {
		return sys.Copy(), nil
	}

	N := len(sys.LFT.Tau)
	n, m, p := sys.Dims()

	ImD22 := mat.NewDense(N, N, nil)
	d22Raw := sys.LFT.D22.RawMatrix()
	imRaw := ImD22.RawMatrix()
	for i := range N {
		for j := range N {
			imRaw.Data[i*imRaw.Stride+j] = -d22Raw.Data[i*d22Raw.Stride+j]
		}
		imRaw.Data[i*imRaw.Stride+i] += 1
	}

	var lu mat.LU
	lu.Factorize(ImD22)
	condition := lu.Cond()
	if nearSingularCondition(condition) {
		return nil, fmt.Errorf(
			"zero delay approximation: %w",
			newAlgebraicLoopError(ImD22, sys.LFT.D22, condition),
		)
	}

	eye := mat.NewDense(N, N, nil)
	eyeRaw := eye.RawMatrix()
	for i := range N {
		eyeRaw.Data[i*eyeRaw.Stride+i] = 1
	}
	E := mat.NewDense(N, N, nil)
	if err := lu.SolveTo(E, false, eye); err != nil {
		return nil, fmt.Errorf(
			"zero delay approximation: %w",
			newAlgebraicLoopError(ImD22, sys.LFT.D22, condition),
		)
	}

	ED21 := mulDims(N, m, E, sys.LFT.D21)
	Da := addMulDims(p, m, sys.D, sys.LFT.D12, ED21)

	var result *System
	var err error
	if n == 0 {
		result, err = NewGain(Da, sys.Dt)
	} else {
		EC2 := mulDims(N, n, E, sys.LFT.C2)
		Aa := addMulDims(n, n, sys.A, sys.LFT.B2, EC2)
		Ba := addMulDims(n, m, sys.B, sys.LFT.B2, ED21)
		Ca := addMulDims(p, n, sys.C, sys.LFT.D12, EC2)

		result, err = newNoCopy(Aa, Ba, Ca, Da, sys.Dt)
	}
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	result.Delay = copyDelayOrNil(sys.Delay)
	if sys.InputDelay != nil {
		result.InputDelay = make([]float64, len(sys.InputDelay))
		copy(result.InputDelay, sys.InputDelay)
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, len(sys.OutputDelay))
		copy(result.OutputDelay, sys.OutputDelay)
	}
	propagateNames(result, sys)
	return result, nil
}
