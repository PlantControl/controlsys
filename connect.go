package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// requireSystems rejects nil models. Interconnections skip Validate because
// they accept fractional discrete delays, which the Thiran feedback path
// approximates and Validate rejects.
func requireSystems(op string, systems ...*System) error {
	for i, sys := range systems {
		if sys == nil {
			return fmt.Errorf("%s: system %d is nil: %w", op, i+1, ErrInvalidArgument)
		}
	}
	return nil
}

func domainMatch(sys1, sys2 *System) error {
	return newTimeDomain(sys1.Dt).ensureCompatible(newTimeDomain(sys2.Dt))
}

func setBlock(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
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
		copy(dRaw.Data[(r0+i)*dRaw.Stride+c0:], sRaw.Data[i*sRaw.Stride:i*sRaw.Stride+sc])
	}
}

type interconnectionPlan struct {
	sys1, sys2 *System
	n1, m1, p1 int
	n2, m2, p2 int
}

func newInterconnectionPlan(sys1, sys2 *System) interconnectionPlan {
	n1, m1, p1 := sys1.Dims()
	n2, m2, p2 := sys2.Dims()
	return interconnectionPlan{
		sys1: sys1, sys2: sys2,
		n1: n1, m1: m1, p1: p1,
		n2: n2, m2: m2, p2: p2,
	}
}

// Series connects sys1 and sys2 in cascade, the output of sys1 driving the
// input of sys2 (y = sys2·sys1·u), as MATLAB series(sys1,sys2); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.series.html.
// Delays are kept exactly, as internal delays when they no longer sit at the
// boundary. Mismatched sizes return ErrDimensionMismatch, different sample
// times ErrDomainMismatch and nil models ErrInvalidArgument.
func Series(sys1, sys2 *System) (*System, error) {
	if err := requireSystems("Series", sys1, sys2); err != nil {
		return nil, err
	}
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	topology := newInterconnectionTopology(sys1, sys2)
	plan := topology.plan
	if plan.p1 != plan.m2 {
		return nil, fmt.Errorf("Series: sys1 outputs %d != sys2 inputs %d: %w", plan.p1, plan.m2, ErrDimensionMismatch)
	}

	if topology.seriesRequiresLFT() {
		return seriesLFT(sys1, sys2)
	}

	return seriesSimple(sys1, sys2, topology.seriesDelayPlan())
}

func seriesSimple(sys1, sys2 *System, delayPlan interconnectionDelayPlan) (*System, error) {
	n1, m1, _ := sys1.Dims()
	n2, _, p2 := sys2.Dims()
	n := n1 + n2
	m := m1
	p := p2

	ioDelay := delayPlan.ioDelay
	inDel, outDel := delayPlan.inputDelay, delayPlan.outputDelay

	buildSeries := func(A, B, C, D *mat.Dense) (*System, error) {
		sys, err := buildSystem(A, B, C, D, sys1.Dt, ioDelay)
		if err != nil {
			return nil, err
		}
		sys.E = blkDiagDescriptorE(sys1, sys2)
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		seriesMetadata(sys1, sys2, n1, n2).applyAllOwned(sys)
		return sys, nil
	}

	if n1 == 0 && n2 == 0 {
		return buildSeries(nil, nil, nil, mulDims(p, m, sys2.D, sys1.D))
	}

	if n1 == 0 {
		return buildSeries(denseCopy(sys2.A), mulDims(n2, m, sys2.B, sys1.D), denseCopy(sys2.C), mulDims(p, m, sys2.D, sys1.D))
	}

	if n2 == 0 {
		return buildSeries(denseCopy(sys1.A), denseCopy(sys1.B), mulDims(p, n1, sys2.D, sys1.C), mulDims(p, m, sys2.D, sys1.D))
	}

	A := mat.NewDense(n, n, nil)
	setBlock(A, 0, 0, sys1.A)
	setBlock(A, n1, 0, mulDense(sys2.B, sys1.C))
	setBlock(A, n1, n1, sys2.A)

	B := newDense(n, m)
	setBlock(B, 0, 0, sys1.B)
	setBlock(B, n1, 0, mulDense(sys2.B, sys1.D))

	C := newDense(p, n)
	setBlock(C, 0, 0, mulDense(sys2.D, sys1.C))
	setBlock(C, 0, n1, sys2.C)

	return buildSeries(A, B, C, mulDims(p, m, sys2.D, sys1.D))
}

func seriesDelay(sys1, sys2 *System) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	_, m1, p1 := sys1.Dims()
	_, _, p2 := sys2.Dims()

	d1 := ioDelayOrZero(sys1.Delay, p1, m1)
	d2 := ioDelayOrZero(sys2.Delay, p2, p1)

	result := mat.NewDense(p2, m1, nil)
	for i := range p2 {
		for j := range m1 {
			ref := d1.At(0, j) + d2.At(i, 0)
			for k := 1; k < p1; k++ {
				if math.Abs(d1.At(k, j)+d2.At(i, k)-ref) > 1e-12 {
					return nil
				}
			}
			result.Set(i, j, ref)
		}
	}
	return result
}

func seriesInputOutputDelay(sys1, sys2 *System, p, m int, ioDelay *mat.Dense) (inputDelay, outputDelay []float64, resultIO *mat.Dense) {
	_, _, p1 := sys1.Dims()

	inputDelay = copySliceOrNil(sys1.InputDelay)
	outputDelay = copySliceOrNil(sys2.OutputDelay)

	intermediate := make([]float64, p1)
	hasIntermediate := false
	for k := range p1 {
		var od, id float64
		if sys1.OutputDelay != nil {
			od = sys1.OutputDelay[k]
		}
		if sys2.InputDelay != nil {
			id = sys2.InputDelay[k]
		}
		intermediate[k] = od + id
		if intermediate[k] != 0 {
			hasIntermediate = true
		}
	}

	if !hasIntermediate {
		return inputDelay, outputDelay, ioDelay
	}

	minIntermediate := intermediate[0]
	for k := 1; k < p1; k++ {
		minIntermediate = math.Min(minIntermediate, intermediate[k])
	}

	if ioDelay == nil {
		ioDelay = mat.NewDense(p, m, nil)
	} else {
		ioDelay = mat.DenseCopyOf(ioDelay)
	}

	raw := ioDelay.RawMatrix()
	for i := range p {
		for j := range m {
			raw.Data[i*raw.Stride+j] += minIntermediate
		}
	}

	return inputDelay, outputDelay, ioDelay
}

func copySliceOrNil(s []float64) []float64 {
	if s == nil {
		return nil
	}
	out := make([]float64, len(s))
	copy(out, s)
	return out
}

func ioDelayOrZero(delay *mat.Dense, p, m int) *mat.Dense {
	if delay != nil {
		return delay
	}
	return mat.NewDense(p, m, nil)
}

// Parallel sums sys1 and sys2 driven by the same input (y = (sys1+sys2)·u),
// as MATLAB parallel(sys1,sys2); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.parallel.html.
// Delays common to both are kept at the boundary and the rest become
// internal delays. Errors follow Series.
func Parallel(sys1, sys2 *System) (*System, error) {
	if err := requireSystems("Parallel", sys1, sys2); err != nil {
		return nil, err
	}
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	topology := newInterconnectionTopology(sys1, sys2)
	plan := topology.plan
	if plan.m1 != plan.m2 || plan.p1 != plan.p2 {
		return nil, fmt.Errorf("Parallel: dims (%d,%d) vs (%d,%d): %w", plan.p1, plan.m1, plan.p2, plan.m2, ErrDimensionMismatch)
	}

	if topology.parallelRequiresLFT() {
		return parallelLFT(sys1, sys2)
	}

	return parallelSimple(sys1, sys2, topology.parallelDelayPlan())
}

func parallelSimple(sys1, sys2 *System, delayPlan interconnectionDelayPlan) (*System, error) {
	n1, m1, p1 := sys1.Dims()
	n2, _, _ := sys2.Dims()
	n := n1 + n2
	m := m1
	p := p1

	ioDelay := delayPlan.ioDelay
	inDel, outDel := delayPlan.inputDelay, delayPlan.outputDelay

	buildParallel := func(A, B, C, D *mat.Dense) (*System, error) {
		sys, err := buildSystem(A, B, C, D, sys1.Dt, ioDelay)
		if err != nil {
			return nil, err
		}
		sys.E = blkDiagDescriptorE(sys1, sys2)
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		parallelMetadata(sys1, sys2, n1, n2).applyAllOwned(sys)
		return sys, nil
	}

	if n1 == 0 && n2 == 0 {
		return buildParallel(nil, nil, nil, addDims(p, m, sys1.D, sys2.D))
	}

	if n1 == 0 {
		return buildParallel(denseCopy(sys2.A), denseCopy(sys2.B), denseCopy(sys2.C), addDims(p, m, sys1.D, sys2.D))
	}

	if n2 == 0 {
		return buildParallel(denseCopy(sys1.A), denseCopy(sys1.B), denseCopy(sys1.C), addDims(p, m, sys1.D, sys2.D))
	}

	A := mat.NewDense(n, n, nil)
	setBlock(A, 0, 0, sys1.A)
	setBlock(A, n1, n1, sys2.A)

	B := newDense(n, m)
	setBlock(B, 0, 0, sys1.B)
	setBlock(B, n1, 0, sys2.B)

	C := newDense(p, n)
	setBlock(C, 0, 0, sys1.C)
	setBlock(C, 0, n1, sys2.C)

	return buildParallel(A, B, C, addDims(p, m, sys1.D, sys2.D))
}

func parallelDelay(sys1, sys2 *System) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	_, m, p := sys1.Dims()
	d1 := ioDelayOrZero(sys1.Delay, p, m)
	d2 := ioDelayOrZero(sys2.Delay, p, m)
	if mat.Equal(d1, d2) {
		return mat.DenseCopyOf(d1)
	}
	return nil
}

func parallelInputOutputDelay(sys1, sys2 *System, m, p int, ioDelay *mat.Dense) (inputDelay, outputDelay []float64, resultIO *mat.Dense) {
	resultIO = ioDelay
	inputDelay = mergeParallelDelay(sys1.InputDelay, sys2.InputDelay, m)
	outputDelay = mergeParallelDelay(sys1.OutputDelay, sys2.OutputDelay, p)

	in1 := sliceOrZeros(sys1.InputDelay, m)
	in2 := sliceOrZeros(sys2.InputDelay, m)
	out1 := sliceOrZeros(sys1.OutputDelay, p)
	out2 := sliceOrZeros(sys2.OutputDelay, p)

	commonIn := sliceOrZeros(inputDelay, m)
	commonOut := sliceOrZeros(outputDelay, p)

	hasResidual := false
	residual1 := mat.NewDense(p, m, nil)
	residual2 := mat.NewDense(p, m, nil)
	r1Raw := residual1.RawMatrix()
	r2Raw := residual2.RawMatrix()
	for i := range p {
		for j := range m {
			v1 := (in1[j] - commonIn[j]) + (out1[i] - commonOut[i])
			v2 := (in2[j] - commonIn[j]) + (out2[i] - commonOut[i])
			r1Raw.Data[i*r1Raw.Stride+j] = v1
			r2Raw.Data[i*r2Raw.Stride+j] = v2
			if v1 != 0 || v2 != 0 {
				hasResidual = true
			}
		}
	}

	if !hasResidual {
		return
	}

	if !mat.Equal(residual1, residual2) {
		if resultIO == nil {
			resultIO = mat.NewDense(p, m, nil)
		} else {
			resultIO = mat.DenseCopyOf(resultIO)
		}
	}

	return
}

func mergeParallelDelay(d1, d2 []float64, n int) []float64 {
	if d1 == nil && d2 == nil {
		return nil
	}
	s1 := sliceOrZeros(d1, n)
	s2 := sliceOrZeros(d2, n)
	out := make([]float64, n)
	allZero := true
	for i := range n {
		out[i] = math.Min(s1[i], s2[i])
		if out[i] != 0 {
			allZero = false
		}
	}
	if allZero && d1 == nil && d2 == nil {
		return nil
	}
	return out
}

func sliceOrZeros(s []float64, n int) []float64 {
	if s != nil {
		return s
	}
	return make([]float64, n)
}

// Feedback returns the closed-loop model of plant with controller in the
// feedback path. sign is -1 for negative feedback, +1 for positive; a nil
// controller closes unit feedback. The result is exact by default: every
// plant and controller delay lies inside the loop and becomes an internal
// delay, matching MATLAB feedback.
// Pass WithApproximatedDelays, WithPadeOrder, or WithThiranOrder to receive
// a delay-free rational model instead.
func Feedback(plant, controller *System, sign float64, opts ...FeedbackOption) (*System, error) {
	if plant == nil {
		return nil, fmt.Errorf("Feedback: plant is nil: %w", ErrInvalidArgument)
	}
	if sign != 1 && sign != -1 {
		return nil, fmt.Errorf("Feedback: sign %g must be -1 or +1: %w", sign, ErrInvalidArgument)
	}
	if controller == nil {
		_, p, m := plant.Dims()
		if p != m {
			return nil, fmt.Errorf("Feedback: plant must be square when controller is nil, got %d outputs × %d inputs: %w", p, m, ErrDimensionMismatch)
		}
		eyeData := make([]float64, m*m)
		for i := range m {
			eyeData[i*(m+1)] = 1
		}
		var err error
		controller, err = NewGain(mat.NewDense(m, m, eyeData), plant.Dt)
		if err != nil {
			return nil, err
		}
	}
	if err := domainMatch(plant, controller); err != nil {
		return nil, err
	}
	if cfg := newFeedbackConfig(opts); cfg.approximateDelays {
		return feedbackWithApproximatedDelays(plant, controller, sign, cfg)
	}
	n1, m1, p1 := plant.Dims()
	n2, m2, p2 := controller.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("Feedback: plant outputs %d != controller inputs %d: %w", p1, m2, ErrDimensionMismatch)
	}
	if m1 != p2 {
		return nil, fmt.Errorf("Feedback: plant inputs %d != controller outputs %d: %w", m1, p2, ErrDimensionMismatch)
	}
	if plant.HasDelay() || controller.HasDelay() {
		return feedbackWithLFT(plant, controller, sign)
	}

	feedbackSign := sign
	// User convention: sign=-1 for negative feedback, +1 for positive.
	// AB05ND formula uses opposite convention internally.
	sign = -sign

	n := n1 + n2
	m := m1
	p := p1

	E21, err := solveFeedbackFeedthrough(plant.D, controller.D, feedbackSign, p1, "Feedback", ErrSingularTransform)
	if err != nil {
		return nil, err
	}

	E12 := eyeOrEmptyDense(m1)
	if m1 > 0 {
		tmp2 := mulDims(m1, m1, mulDims(m1, p1, controller.D, E21), plant.D)
		tmp2.Scale(sign, tmp2)
		E12.Sub(E12, tmp2)
	}
	D := mulDims(p, m, E21, plant.D)

	if n1 == 0 && n2 == 0 {
		sys, err := buildSystem(nil, nil, nil, D, plant.Dt, nil)
		if err != nil {
			return nil, err
		}
		sys.InputName = copyStringSlice(plant.InputName)
		sys.OutputName = copyStringSlice(plant.OutputName)
		sys.StateName = concatStringSlices([][]string{plant.StateName, controller.StateName}, []int{n1, n2})
		return sys, nil
	}

	A := mat.NewDense(n, n, nil)
	Bres := newDense(n, m)
	C := newDense(p, n)

	if n1 > 0 {
		B1E12 := mulDims(n1, m1, plant.B, E12)
		block := mulDims(n1, n1, mulDims(n1, p1, B1E12, controller.D), plant.C)
		block.Scale(sign, block)
		setBlock(A, 0, 0, plant.A)
		subBlock(A, 0, 0, block)

		if n2 > 0 {
			block2 := mulDims(n1, n2, B1E12, controller.C)
			block2.Scale(-sign, block2)
			setBlock(A, 0, n1, block2)
		}

		setBlock(Bres, 0, 0, B1E12)
	}

	if n2 > 0 {
		B2E21 := mulDims(n2, p1, controller.B, E21)

		if n1 > 0 {
			setBlock(A, n1, 0, mulDims(n2, n1, B2E21, plant.C))
		}

		setBlock(A, n1, n1, controller.A)
		B2E21D1 := mulDims(n2, m1, B2E21, plant.D)
		block3 := mulDims(n2, n2, B2E21D1, controller.C)
		block3.Scale(sign, block3)
		subBlock(A, n1, n1, block3)

		setBlock(Bres, n1, 0, B2E21D1)
	}

	if n1 > 0 {
		setBlock(C, 0, 0, mulDims(p1, n1, E21, plant.C))
	}
	if n2 > 0 && p1 > 0 {
		E21D1C2 := mulDims(p1, n2, mulDims(p1, m1, E21, plant.D), controller.C)
		E21D1C2.Scale(-sign, E21D1C2)
		setBlock(C, 0, n1, E21D1C2)
	}

	sys, err := buildSystem(A, Bres, C, D, plant.Dt, nil)
	if err != nil {
		return nil, err
	}
	sys.E = blkDiagDescriptorE(plant, controller)
	sys.InputName = copyStringSlice(plant.InputName)
	sys.OutputName = copyStringSlice(plant.OutputName)
	sys.StateName = concatStringSlices([][]string{plant.StateName, controller.StateName}, []int{n1, n2})
	return sys, nil
}

func subBlock(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
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
			dRow[j] -= sRow[j]
		}
	}
}

// Append stacks the inputs and outputs of sys1 and sys2 into a block-diagonal
// model, as MATLAB append(sys1,sys2); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.append.html. It
// equals BlkDiag(sys1, sys2). Errors follow Series.
func Append(sys1, sys2 *System) (*System, error) {
	if err := requireSystems("Append", sys1, sys2); err != nil {
		return nil, err
	}
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	n1, m1, p1 := sys1.Dims()
	n2, m2, p2 := sys2.Dims()

	n := n1 + n2
	m := m1 + m2
	p := p1 + p2

	if sys1.HasInternalDelay() || sys2.HasInternalDelay() {
		return BlkDiag(sys1, sys2)
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	D := mat.NewDense(max(p, 1), max(m, 1), nil)

	if n1 > 0 {
		setBlock(A, 0, 0, sys1.A)
		setBlock(B, 0, 0, sys1.B)
		setBlock(C, 0, 0, sys1.C)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, sys2.A)
		setBlock(B, n1, m1, sys2.B)
		setBlock(C, p1, n1, sys2.C)
	}
	setBlock(D, 0, 0, sys1.D)
	setBlock(D, p1, m1, sys2.D)

	delay := appendDelay(sys1, sys2, p1, m1, p2, m2)
	inDel := concatDelay(sys1.InputDelay, m1, sys2.InputDelay, m2)
	outDel := concatDelay(sys1.OutputDelay, p1, sys2.OutputDelay, p2)

	if n == 0 {
		sys, err := NewGain(resizeDense(D, p, m), sys1.Dt)
		if err != nil {
			return nil, err
		}
		sys.Delay = delay
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		appendInternalDelay(sys, sys1, sys2, n1, n2, m1, m2, p1, p2)
		sys.InputName = concatStringSlices([][]string{sys1.InputName, sys2.InputName}, []int{m1, m2})
		sys.OutputName = concatStringSlices([][]string{sys1.OutputName, sys2.OutputName}, []int{p1, p2})
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)

	sys, err := buildSystem(A, B, C, D, sys1.Dt, delay)
	if err != nil {
		return nil, err
	}
	sys.E = blkDiagDescriptorE(sys1, sys2)
	sys.InputDelay = inDel
	sys.OutputDelay = outDel
	appendInternalDelay(sys, sys1, sys2, n1, n2, m1, m2, p1, p2)
	sys.InputName = concatStringSlices([][]string{sys1.InputName, sys2.InputName}, []int{m1, m2})
	sys.OutputName = concatStringSlices([][]string{sys1.OutputName, sys2.OutputName}, []int{p1, p2})
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func resizeDense(src *mat.Dense, r, c int) *mat.Dense {
	sr, sc := src.Dims()
	if sr == r && sc == c {
		return src
	}
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	dst := mat.NewDense(r, c, nil)
	minR := min(sr, r)
	minC := min(sc, c)
	sRaw := src.RawMatrix()
	dRaw := dst.RawMatrix()
	for i := range minR {
		copy(dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+minC], sRaw.Data[i*sRaw.Stride:i*sRaw.Stride+minC])
	}
	return dst
}

func appendDelay(sys1, sys2 *System, p1, m1, p2, m2 int) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	p := p1 + p2
	m := m1 + m2
	d := mat.NewDense(p, m, nil)
	if sys1.Delay != nil {
		setBlock(d, 0, 0, sys1.Delay)
	}
	if sys2.Delay != nil {
		setBlock(d, p1, m1, sys2.Delay)
	}
	return d
}

func concatDelay(d1 []float64, n1 int, d2 []float64, n2 int) []float64 {
	if d1 == nil && d2 == nil {
		return nil
	}
	out := make([]float64, n1+n2)
	if d1 != nil {
		copy(out, d1)
	}
	if d2 != nil {
		copy(out[n1:], d2)
	}
	return out
}

func mulDense(a, b *mat.Dense) *mat.Dense {
	ar, _ := a.Dims()
	_, bc := b.Dims()
	if ar == 0 || bc == 0 {
		return &mat.Dense{}
	}
	r := mat.NewDense(ar, bc, nil)
	r.Mul(a, b)
	return r
}

func seriesLFT(sys1, sys2 *System) (*System, error) {
	savedInput1 := copySliceOrNil(sys1.InputDelay)
	savedOutput2 := copySliceOrNil(sys2.OutputDelay)

	s1Stripped := sys1.Copy()
	s1Stripped.InputDelay = nil
	s2Stripped := sys2.Copy()
	s2Stripped.OutputDelay = nil

	s1, err := s1Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	s2, err := s2Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	n1, m1, _ := s1.Dims()
	n2, _, p2 := s2.Dims()
	N1 := s1.internalDelayCount()
	N2 := s2.internalDelayCount()
	N := N1 + N2
	n := n1 + n2
	m := m1
	p := p2

	tau := make([]float64, 0, N)
	if s1.LFT != nil {
		tau = append(tau, s1.LFT.Tau...)
	}
	if s2.LFT != nil {
		tau = append(tau, s2.LFT.Tau...)
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	if n1 > 0 {
		setBlock(A, 0, 0, s1.A)
		setBlock(B, 0, 0, s1.B)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, s2.A)
	}
	if n1 > 0 && n2 > 0 {
		setBlock(A, n1, 0, mulDense(s2.B, s1.C))
	}
	if n2 > 0 {
		setBlock(B, n1, 0, mulDense(s2.B, s1.D))
	}
	if n1 > 0 {
		setBlock(C, 0, 0, mulDense(s2.D, s1.C))
	}
	if n2 > 0 {
		setBlock(C, 0, n1, s2.C)
	}
	D := mulDims(p, m, s2.D, s1.D)

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)
	if n == 0 {
		A, B, C = nil, nil, nil
	}

	if N == 0 {
		sys, err := newNoCopy(A, B, C, D, s1.Dt)
		if err != nil {
			return nil, err
		}
		sys.E = blkDiagDescriptorE(s1, s2)
		sys.InputDelay = savedInput1
		sys.OutputDelay = savedOutput2
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys2.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 {
		if n1 > 0 {
			setBlock(b2, 0, 0, s1.LFT.B2)
		}
		if n2 > 0 {
			setBlock(b2, n1, 0, mulDense(s2.B, s1.LFT.D12))
		}
		if n1 > 0 {
			setBlock(c2, 0, 0, s1.LFT.C2)
		}
		setBlock(d12, 0, 0, mulDense(s2.D, s1.LFT.D12))
		setBlock(d21, 0, 0, s1.LFT.D21)
		setBlock(d22, 0, 0, s1.LFT.D22)
		if N2 > 0 {
			setBlock(d22, N1, 0, mulDense(s2.LFT.D21, s1.LFT.D12))
		}
	}

	if N2 > 0 {
		if n2 > 0 {
			setBlock(b2, n1, N1, s2.LFT.B2)
		}
		if n2 > 0 {
			setBlock(c2, N1, n1, s2.LFT.C2)
		}
		if n1 > 0 {
			c2sub := mulDense(s2.LFT.D21, s1.C)
			r, c := c2sub.Dims()
			for i := range r {
				for j := range c {
					c2.Set(N1+i, j, c2.At(N1+i, j)+c2sub.At(i, j))
				}
			}
		}
		setBlock(d12, 0, N1, s2.LFT.D12)
		d21sub := mulDense(s2.LFT.D21, s1.D)
		setBlock(d21, N1, 0, d21sub)
		setBlock(d22, N1, N1, s2.LFT.D22)
		if N1 > 0 {
			setBlock(d22, N1, 0, mulDense(s2.LFT.D21, s1.LFT.D12))
		}
	}

	b2 = resizeDense(b2, n, N)
	c2 = resizeDense(c2, N, n)
	d12 = resizeDense(d12, p, N)
	d21 = resizeDense(d21, N, m)

	sys, err := newNoCopy(A, B, C, D, s1.Dt)
	if err != nil {
		return nil, err
	}
	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  b2,
		C2:  c2,
		D12: d12,
		D21: d21,
		D22: d22,
	}
	sys.E = blkDiagDescriptorE(s1, s2)
	sys.InputDelay = savedInput1
	sys.OutputDelay = savedOutput2
	sys.InputName = copyStringSlice(sys1.InputName)
	sys.OutputName = copyStringSlice(sys2.OutputName)
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func parallelLFT(sys1, sys2 *System) (*System, error) {
	_, m1, p1 := sys1.Dims()
	commonIn := mergeParallelDelay(sys1.InputDelay, sys2.InputDelay, m1)
	commonOut := mergeParallelDelay(sys1.OutputDelay, sys2.OutputDelay, p1)

	s1Stripped := sys1.Copy()
	s2Stripped := sys2.Copy()
	if commonIn != nil {
		if s1Stripped.InputDelay == nil {
			s1Stripped.InputDelay = make([]float64, m1)
		}
		if s2Stripped.InputDelay == nil {
			s2Stripped.InputDelay = make([]float64, m1)
		}
		for j := range m1 {
			s1Stripped.InputDelay[j] -= commonIn[j]
			s2Stripped.InputDelay[j] -= commonIn[j]
		}
	}
	if commonOut != nil {
		if s1Stripped.OutputDelay == nil {
			s1Stripped.OutputDelay = make([]float64, p1)
		}
		if s2Stripped.OutputDelay == nil {
			s2Stripped.OutputDelay = make([]float64, p1)
		}
		for i := range p1 {
			s1Stripped.OutputDelay[i] -= commonOut[i]
			s2Stripped.OutputDelay[i] -= commonOut[i]
		}
	}

	s1, err := s1Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	s2, err := s2Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	n1, _, _ := s1.Dims()
	n2, _, _ := s2.Dims()
	N1 := s1.internalDelayCount()
	N2 := s2.internalDelayCount()
	N := N1 + N2
	n := n1 + n2
	m := m1
	p := p1

	tau := make([]float64, 0, N)
	if s1.LFT != nil {
		tau = append(tau, s1.LFT.Tau...)
	}
	if s2.LFT != nil {
		tau = append(tau, s2.LFT.Tau...)
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	D := mat.NewDense(max(p, 1), max(m, 1), nil)

	if n1 > 0 {
		setBlock(A, 0, 0, s1.A)
		setBlock(B, 0, 0, s1.B)
		setBlock(C, 0, 0, s1.C)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, s2.A)
		setBlock(B, n1, 0, s2.B)
		setBlock(C, 0, n1, s2.C)
	}
	setBlock(D, 0, 0, s1.D)
	addBlock(D, 0, 0, s2.D)

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)

	if N == 0 {
		sys, err := newNoCopy(A, B, C, D, s1.Dt)
		if err != nil {
			return nil, err
		}
		sys.E = blkDiagDescriptorE(s1, s2)
		sys.InputDelay = commonIn
		sys.OutputDelay = commonOut
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys1.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 {
		if n1 > 0 {
			setBlock(b2, 0, 0, s1.LFT.B2)
		}
		if n1 > 0 {
			setBlock(c2, 0, 0, s1.LFT.C2)
		}
		setBlock(d12, 0, 0, s1.LFT.D12)
		setBlock(d21, 0, 0, s1.LFT.D21)
		setBlock(d22, 0, 0, s1.LFT.D22)
	}

	if N2 > 0 {
		if n2 > 0 {
			setBlock(b2, n1, N1, s2.LFT.B2)
		}
		if n2 > 0 {
			setBlock(c2, N1, n1, s2.LFT.C2)
		}
		setBlock(d12, 0, N1, s2.LFT.D12)
		setBlock(d21, N1, 0, s2.LFT.D21)
		setBlock(d22, N1, N1, s2.LFT.D22)
	}

	b2 = resizeDense(b2, n, N)
	c2 = resizeDense(c2, N, n)
	d12 = resizeDense(d12, p, N)
	d21 = resizeDense(d21, N, m)

	sys, err := newNoCopy(A, B, C, D, s1.Dt)
	if err != nil {
		return nil, err
	}
	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  b2,
		C2:  c2,
		D12: d12,
		D21: d21,
		D22: d22,
	}
	sys.E = blkDiagDescriptorE(s1, s2)
	sys.InputDelay = commonIn
	sys.OutputDelay = commonOut
	sys.InputName = copyStringSlice(sys1.InputName)
	sys.OutputName = copyStringSlice(sys1.OutputName)
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func appendInternalDelay(sys, sys1, sys2 *System, n1, n2, m1, m2, p1, p2 int) {
	N1 := sys1.internalDelayCount()
	N2 := sys2.internalDelayCount()
	if N1 == 0 && N2 == 0 {
		return
	}

	n := n1 + n2
	m := m1 + m2
	p := p1 + p2
	N := N1 + N2

	tau := make([]float64, N)
	if N1 > 0 {
		copy(tau, sys1.LFT.Tau)
	}
	if N2 > 0 {
		copy(tau[N1:], sys2.LFT.Tau)
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 && sys1.LFT.B2 != nil {
		setBlock(b2, 0, 0, sys1.LFT.B2)
	}
	if N2 > 0 && sys2.LFT.B2 != nil {
		setBlock(b2, n1, N1, sys2.LFT.B2)
	}
	if N1 > 0 && sys1.LFT.C2 != nil {
		setBlock(c2, 0, 0, sys1.LFT.C2)
	}
	if N2 > 0 && sys2.LFT.C2 != nil {
		setBlock(c2, N1, n1, sys2.LFT.C2)
	}
	if N1 > 0 && sys1.LFT.D12 != nil {
		setBlock(d12, 0, 0, sys1.LFT.D12)
	}
	if N2 > 0 && sys2.LFT.D12 != nil {
		setBlock(d12, p1, N1, sys2.LFT.D12)
	}
	if N1 > 0 && sys1.LFT.D21 != nil {
		setBlock(d21, 0, 0, sys1.LFT.D21)
	}
	if N2 > 0 && sys2.LFT.D21 != nil {
		setBlock(d21, N1, m1, sys2.LFT.D21)
	}
	if N1 > 0 && sys1.LFT.D22 != nil {
		setBlock(d22, 0, 0, sys1.LFT.D22)
	}
	if N2 > 0 && sys2.LFT.D22 != nil {
		setBlock(d22, N1, N1, sys2.LFT.D22)
	}

	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  resizeDense(b2, n, N),
		C2:  resizeDense(c2, N, n),
		D12: resizeDense(d12, p, N),
		D21: resizeDense(d21, N, m),
		D22: resizeDense(d22, N, N),
	}
}

// BlkDiag returns the block-diagonal model of systems, as MATLAB
// blkdiag(sys1,...,sysN); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.blkdiag.html.
// One system returns a copy. No systems or a nil system return
// ErrInvalidArgument; different sample times return ErrDomainMismatch.
func BlkDiag(systems ...*System) (*System, error) {
	if len(systems) == 0 {
		return nil, fmt.Errorf("BlkDiag: no systems provided: %w", ErrInvalidArgument)
	}
	if err := requireSystems("BlkDiag", systems...); err != nil {
		return nil, err
	}
	if len(systems) == 1 {
		return systems[0].Copy(), nil
	}

	dt := systems[0].Dt
	for i := 1; i < len(systems); i++ {
		if systems[i].Dt != dt {
			return nil, fmt.Errorf("BlkDiag: system %d Dt=%v != Dt=%v: %w", i, systems[i].Dt, dt, ErrDomainMismatch)
		}
	}

	anyInternalDelay := false
	for _, s := range systems {
		if s.HasInternalDelay() {
			anyInternalDelay = true
			break
		}
	}

	srcs := systems
	if anyInternalDelay {
		srcs = make([]*System, len(systems))
		for i, s := range systems {
			pulled, err := s.PullDelaysToLFT()
			if err != nil {
				return nil, err
			}
			srcs[i] = pulled
		}
	}

	k := len(srcs)
	ns := make([]int, k)
	ms := make([]int, k)
	ps := make([]int, k)
	var nTotal, mTotal, pTotal int
	for i, s := range srcs {
		ns[i], ms[i], ps[i] = s.Dims()
		nTotal += ns[i]
		mTotal += ms[i]
		pTotal += ps[i]
	}

	A := mat.NewDense(max(nTotal, 1), max(nTotal, 1), nil)
	B := mat.NewDense(max(nTotal, 1), max(mTotal, 1), nil)
	C := mat.NewDense(max(pTotal, 1), max(nTotal, 1), nil)
	D := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)

	nOff, mOff, pOff := 0, 0, 0
	for i, s := range srcs {
		if ns[i] > 0 {
			setBlock(A, nOff, nOff, s.A)
			setBlock(B, nOff, mOff, s.B)
			setBlock(C, pOff, nOff, s.C)
		}
		setBlock(D, pOff, mOff, s.D)
		nOff += ns[i]
		mOff += ms[i]
		pOff += ps[i]
	}

	A = resizeDense(A, nTotal, nTotal)
	B = resizeDense(B, nTotal, mTotal)
	C = resizeDense(C, pTotal, nTotal)
	D = resizeDense(D, pTotal, mTotal)

	var delay *mat.Dense
	hasIODelay := false
	for _, s := range srcs {
		if s.Delay != nil {
			hasIODelay = true
			break
		}
	}
	if hasIODelay {
		delay = mat.NewDense(pTotal, mTotal, nil)
		pOff, mOff = 0, 0
		for i, s := range srcs {
			if s.Delay != nil {
				setBlock(delay, pOff, mOff, s.Delay)
			}
			pOff += ps[i]
			mOff += ms[i]
		}
	}

	var inDel []float64
	hasIn := false
	for _, s := range srcs {
		if s.InputDelay != nil {
			hasIn = true
			break
		}
	}
	if hasIn {
		inDel = make([]float64, mTotal)
		off := 0
		for i, s := range srcs {
			if s.InputDelay != nil {
				copy(inDel[off:], s.InputDelay)
			}
			off += ms[i]
		}
	}

	var outDel []float64
	hasOut := false
	for _, s := range srcs {
		if s.OutputDelay != nil {
			hasOut = true
			break
		}
	}
	if hasOut {
		outDel = make([]float64, pTotal)
		off := 0
		for i, s := range srcs {
			if s.OutputDelay != nil {
				copy(outDel[off:], s.OutputDelay)
			}
			off += ps[i]
		}
	}

	var inNames, outNames, stNames [][]string
	for _, s := range systems {
		inNames = append(inNames, s.InputName)
		outNames = append(outNames, s.OutputName)
		stNames = append(stNames, s.StateName)
	}

	if nTotal == 0 {
		sys, err := NewGain(D, dt)
		if err != nil {
			return nil, err
		}
		sys.Delay = delay
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		blkDiagInternalDelay(sys, srcs, ns, ms, ps, nTotal, mTotal, pTotal)
		sys.InputName = concatStringSlices(inNames, ms)
		sys.OutputName = concatStringSlices(outNames, ps)
		sys.StateName = concatStringSlices(stNames, ns)
		return sys, nil
	}

	sys, err := newNoCopy(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	sys.E = blkDiagDescriptorE(srcs...)
	sys.Delay = delay
	sys.InputDelay = inDel
	sys.OutputDelay = outDel
	blkDiagInternalDelay(sys, srcs, ns, ms, ps, nTotal, mTotal, pTotal)
	sys.InputName = concatStringSlices(inNames, ms)
	sys.OutputName = concatStringSlices(outNames, ps)
	sys.StateName = concatStringSlices(stNames, ns)
	return sys, nil
}

func blkDiagInternalDelay(sys *System, srcs []*System, ns, ms, ps []int, nTotal, mTotal, pTotal int) {
	var NTotal int
	for _, s := range srcs {
		NTotal += s.internalDelayCount()
	}
	if NTotal == 0 {
		return
	}

	tau := make([]float64, 0, NTotal)
	for _, s := range srcs {
		if s.LFT != nil {
			tau = append(tau, s.LFT.Tau...)
		}
	}

	n := nTotal
	m := mTotal
	p := pTotal

	b2 := mat.NewDense(max(n, 1), NTotal, nil)
	c2 := mat.NewDense(NTotal, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), NTotal, nil)
	d21 := mat.NewDense(NTotal, max(m, 1), nil)
	d22 := mat.NewDense(NTotal, NTotal, nil)

	nOff, mOff, pOff, NOff := 0, 0, 0, 0
	for i, s := range srcs {
		Ni := s.internalDelayCount()
		if Ni > 0 {
			if ns[i] > 0 && s.LFT.B2 != nil {
				setBlock(b2, nOff, NOff, s.LFT.B2)
			}
			if ns[i] > 0 && s.LFT.C2 != nil {
				setBlock(c2, NOff, nOff, s.LFT.C2)
			}
			if s.LFT.D12 != nil {
				setBlock(d12, pOff, NOff, s.LFT.D12)
			}
			if s.LFT.D21 != nil {
				setBlock(d21, NOff, mOff, s.LFT.D21)
			}
			if s.LFT.D22 != nil {
				setBlock(d22, NOff, NOff, s.LFT.D22)
			}
		}
		nOff += ns[i]
		mOff += ms[i]
		pOff += ps[i]
		NOff += Ni
	}

	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  resizeDense(b2, n, NTotal),
		C2:  resizeDense(c2, NTotal, n),
		D12: resizeDense(d12, p, NTotal),
		D21: resizeDense(d21, NTotal, m),
		D22: resizeDense(d22, NTotal, NTotal),
	}
}

// Connect closes internal connections of the block-diagonal model sys, as
// the index-based MATLAB form sysc = connect(blksys,connections,inputs,outputs);
// see https://www.mathworks.com/help/control/ref/dynamicsystem.connect.html.
// For name-based connections use ConnectByName.
//
// The MATLAB connections list maps to the m×p gain matrix Q: the row
// [i, j, -k] (u(i) is driven by w(j) - w(k), 1-based) is Q[i-1][j-1] = 1 and
// Q[i-1][k-1] = -1, so u = Q·y + v with v the external input. Q may hold any
// real gains. inputs and outputs are 0-based indices of the inputs of sys
// kept as external inputs and the outputs kept as external outputs; they must
// be non-empty, in range and unique (ErrInvalidArgument). Q of the wrong size
// returns ErrDimensionMismatch and a singular I - Q·D ErrAlgebraicLoop.
func Connect(sys *System, Q *mat.Dense, inputs, outputs []int) (*System, error) {
	if err := requireSystems("Connect", sys); err != nil {
		return nil, err
	}
	if Q == nil {
		return nil, fmt.Errorf("Connect: Q is nil: %w", ErrInvalidArgument)
	}
	n, m, p := sys.Dims()

	qr, qc := Q.Dims()
	if qr != m || qc != p {
		return nil, fmt.Errorf("Connect: Q size %dx%d != %dx%d: %w", qr, qc, m, p, ErrDimensionMismatch)
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("Connect: inputs must be non-empty: %w", ErrInvalidArgument)
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("Connect: outputs must be non-empty: %w", ErrInvalidArgument)
	}

	inSeen := make(map[int]bool, len(inputs))
	for _, idx := range inputs {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("Connect: input index %d outside [0,%d): %w", idx, m, ErrInvalidArgument)
		}
		if inSeen[idx] {
			return nil, fmt.Errorf("Connect: duplicate input index %d: %w", idx, ErrInvalidArgument)
		}
		inSeen[idx] = true
	}
	outSeen := make(map[int]bool, len(outputs))
	for _, idx := range outputs {
		if idx < 0 || idx >= p {
			return nil, fmt.Errorf("Connect: output index %d outside [0,%d): %w", idx, p, ErrInvalidArgument)
		}
		if outSeen[idx] {
			return nil, fmt.Errorf("Connect: duplicate output index %d: %w", idx, ErrInvalidArgument)
		}
		outSeen[idx] = true
	}

	if sys.HasDelay() || sys.HasInternalDelay() {
		return connectWithDelay(sys, Q, inputs, outputs)
	}

	return connectSimple(sys, Q, inputs, outputs, n, m, p)
}

func connectWithDelay(sys *System, Q *mat.Dense, inputs, outputs []int) (*System, error) {
	_, m, p := sys.Dims()

	qRaw := Q.RawMatrix()
	openInput := func(j int) bool {
		for _, v := range qRaw.Data[j*qRaw.Stride : j*qRaw.Stride+p] {
			if v != 0 {
				return false
			}
		}
		return true
	}
	openOutput := func(i int) bool {
		for k := range m {
			if qRaw.Data[k*qRaw.Stride+i] != 0 {
				return false
			}
		}
		return true
	}
	savedInputDelay, hoistInputs := selectHoistableDelays(sys.InputDelay, inputs, openInput)
	savedOutputDelay, hoistOutputs := selectHoistableDelays(sys.OutputDelay, outputs, openOutput)

	sCopy := sys.Copy()
	sCopy.InputDelay = clearDelayIndexes(sCopy.InputDelay, hoistInputs)
	sCopy.OutputDelay = clearDelayIndexes(sCopy.OutputDelay, hoistOutputs)

	sLFT, err := sCopy.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	N := sLFT.internalDelayCount()
	H, tau := sLFT.GetDelayModel()

	nH, _, _ := H.Dims()

	Dcore := extractBlock(H.D, 0, 0, p, m)
	E, err := solveIdentityMinusProduct(Q, Dcore, m, "Connect", ErrAlgebraicLoop)
	if err != nil {
		return nil, err
	}

	EQ := mat.NewDense(m, p, nil)
	EQ.Mul(E, Q)

	DEQ := mat.NewDense(p, p, nil)
	DEQ.Mul(Dcore, EQ)
	Er := eyeDense(p)
	Er.Add(Er, DEQ)

	Bcore := extractBlock(H.B, 0, 0, nH, m)
	Ccore := extractBlock(H.C, 0, 0, p, nH)

	mExt := len(inputs)
	pExt := len(outputs)
	mTotal := mExt + N
	pTotal := pExt + N

	var BEQ *mat.Dense
	if nH > 0 {
		BEQ = mat.NewDense(nH, p, nil)
		BEQ.Mul(Bcore, EQ)
	}

	var Dout, Din, DinEQ *mat.Dense
	if N > 0 {
		Dout = extractBlock(H.D, 0, m, p, N)
		Din = extractBlock(H.D, p, 0, N, m)
		DinEQ = mat.NewDense(N, p, nil)
		DinEQ.Mul(Din, EQ)
	}

	Acl := mat.NewDense(max(nH, 1), max(nH, 1), nil)
	if nH > 0 {
		setBlock(Acl, 0, 0, H.A)
		BEQC := mat.NewDense(nH, nH, nil)
		BEQC.Mul(BEQ, Ccore)
		addBlock(Acl, 0, 0, BEQC)
	}

	Bcl := mat.NewDense(max(nH, 1), max(mTotal, 1), nil)
	if nH > 0 {
		Bfull := mat.NewDense(nH, m, nil)
		Bfull.Mul(Bcore, E)
		bfRaw := Bfull.RawMatrix()
		for k, j := range inputs {
			for i := range nH {
				Bcl.Set(i, k, bfRaw.Data[i*bfRaw.Stride+j])
			}
		}
		if N > 0 {
			Bdelay := extractBlock(H.B, 0, m, nH, N)
			BEQDout := mat.NewDense(nH, N, nil)
			BEQDout.Mul(BEQ, Dout)
			b2mod := mat.NewDense(nH, N, nil)
			b2mod.Add(Bdelay, BEQDout)
			setBlock(Bcl, 0, mExt, b2mod)
		}
	}

	Cfull := mat.NewDense(p, max(nH, 1), nil)
	if nH > 0 {
		Cfull.Mul(Er, Ccore)
	}
	Dfull := mat.NewDense(p, m, nil)
	Dfull.Mul(Er, Dcore)

	Ccl := mat.NewDense(max(pTotal, 1), max(nH, 1), nil)
	if nH > 0 {
		cfRaw := Cfull.RawMatrix()
		ccRaw := Ccl.RawMatrix()
		for k, i := range outputs {
			copy(ccRaw.Data[k*ccRaw.Stride:k*ccRaw.Stride+nH],
				cfRaw.Data[i*cfRaw.Stride:i*cfRaw.Stride+nH])
		}
		if N > 0 {
			Cdelay := extractBlock(H.C, p, 0, N, nH)
			DinEQC := mat.NewDense(N, nH, nil)
			DinEQC.Mul(DinEQ, Ccore)
			c2mod := mat.NewDense(N, nH, nil)
			c2mod.Add(Cdelay, DinEQC)
			setBlock(Ccl, pExt, 0, c2mod)
		}
	}

	Dcl := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)
	for ki, oi := range outputs {
		for kj, ij := range inputs {
			Dcl.Set(ki, kj, Dfull.At(oi, ij))
		}
	}

	if N > 0 {
		ErDout := mat.NewDense(p, N, nil)
		ErDout.Mul(Er, Dout)
		erRaw := ErDout.RawMatrix()
		for ki, oi := range outputs {
			for j := range N {
				Dcl.Set(ki, mExt+j, erRaw.Data[oi*erRaw.Stride+j])
			}
		}

		DinE := mat.NewDense(N, m, nil)
		DinE.Mul(Din, E)
		deRaw := DinE.RawMatrix()
		for i := range N {
			for kj, ij := range inputs {
				Dcl.Set(pExt+i, kj, deRaw.Data[i*deRaw.Stride+ij])
			}
		}

		D22 := extractBlock(H.D, p, m, N, N)
		d22cross := mat.NewDense(N, N, nil)
		d22cross.Mul(DinEQ, Dout)
		d22mod := mat.NewDense(N, N, nil)
		d22mod.Add(D22, d22cross)
		setBlock(Dcl, pExt, mExt, d22mod)
	}

	Acl = resizeDense(Acl, nH, nH)
	Bcl = resizeDense(Bcl, nH, mTotal)
	Ccl = resizeDense(Ccl, pTotal, nH)
	Dcl = resizeDense(Dcl, pTotal, mTotal)

	if nH == 0 {
		Acl = &mat.Dense{}
	}

	Hresult := &System{A: Acl, B: Bcl, C: Ccl, D: Dcl, E: sLFT.E, Dt: sys.Dt}
	result, err := SetDelayModel(Hresult, tau)
	if err != nil {
		return nil, err
	}

	if savedInputDelay.hasNonzero {
		result.InputDelay = savedInputDelay.values
	}
	if savedOutputDelay.hasNonzero {
		result.OutputDelay = savedOutputDelay.values
	}

	result.InputName = selectStringSlice(sys.InputName, inputs)
	result.OutputName = selectStringSlice(sys.OutputName, outputs)

	return result, nil
}

func connectSimple(sys *System, Q *mat.Dense, inputs, outputs []int, n, m, p int) (*System, error) {
	mExt := len(inputs)
	pExt := len(outputs)

	E, err := solveIdentityMinusProduct(Q, sys.D, m, "Connect", ErrAlgebraicLoop)
	if err != nil {
		return nil, err
	}

	EQ := mat.NewDense(m, p, nil)
	EQ.Mul(E, Q)

	if n == 0 {
		DEQ := mat.NewDense(p, p, nil)
		DEQ.Mul(sys.D, EQ)
		Er := eyeDense(p)
		Er.Add(Er, DEQ)

		Dfull := mat.NewDense(p, m, nil)
		Dfull.Mul(Er, sys.D)

		Dcl := mat.NewDense(pExt, mExt, nil)
		for ki, oi := range outputs {
			for kj, ij := range inputs {
				Dcl.Set(ki, kj, Dfull.At(oi, ij))
			}
		}
		result, err := NewGain(Dcl, sys.Dt)
		if err != nil {
			return nil, err
		}
		result.InputName = selectStringSlice(sys.InputName, inputs)
		result.OutputName = selectStringSlice(sys.OutputName, outputs)
		result.StateName = copyStringSlice(sys.StateName)
		return result, nil
	}

	BEQC := mat.NewDense(n, n, nil)
	BEQ := mat.NewDense(n, p, nil)
	BEQ.Mul(sys.B, EQ)
	BEQC.Mul(BEQ, sys.C)
	Acl := mat.NewDense(n, n, nil)
	Acl.Add(sys.A, BEQC)

	Bfull := mat.NewDense(n, m, nil)
	Bfull.Mul(sys.B, E)

	DEQ := mat.NewDense(p, p, nil)
	DEQ.Mul(sys.D, EQ)
	Er := eyeDense(p)
	Er.Add(Er, DEQ)

	Cfull := mat.NewDense(p, n, nil)
	Cfull.Mul(Er, sys.C)

	Dfull := mat.NewDense(p, m, nil)
	Dfull.Mul(Er, sys.D)

	Bcl := mat.NewDense(n, mExt, nil)
	bfRaw := Bfull.RawMatrix()
	bcRaw := Bcl.RawMatrix()
	for k, j := range inputs {
		for i := range n {
			bcRaw.Data[i*bcRaw.Stride+k] = bfRaw.Data[i*bfRaw.Stride+j]
		}
	}

	Ccl := mat.NewDense(pExt, n, nil)
	cfRaw := Cfull.RawMatrix()
	ccRaw := Ccl.RawMatrix()
	for k, i := range outputs {
		copy(ccRaw.Data[k*ccRaw.Stride:k*ccRaw.Stride+n], cfRaw.Data[i*cfRaw.Stride:i*cfRaw.Stride+n])
	}

	Dcl := mat.NewDense(pExt, mExt, nil)
	for ki, oi := range outputs {
		for kj, ij := range inputs {
			Dcl.Set(ki, kj, Dfull.At(oi, ij))
		}
	}

	result, err := newNoCopy(Acl, Bcl, Ccl, Dcl, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	result.InputName = selectStringSlice(sys.InputName, inputs)
	result.OutputName = selectStringSlice(sys.OutputName, outputs)
	result.StateName = copyStringSlice(sys.StateName)
	return result, nil
}

func buildSystem(A, B, C, D *mat.Dense, dt float64, delay *mat.Dense) (*System, error) {
	if A == nil && B == nil && C == nil {
		sys, err := NewGain(D, dt)
		if err != nil {
			return nil, err
		}
		sys.Delay = delay
		return sys, nil
	}
	sys, err := newNoCopy(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	sys.Delay = delay
	return sys, nil
}
