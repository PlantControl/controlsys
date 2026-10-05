package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func conversionHasFractionalExternalDelay(sys *System, dt float64) bool {
	for _, delays := range [][]float64{sys.InputDelay, sys.OutputDelay} {
		for _, tau := range delays {
			if !isIntegerSampleDelay(tau / dt) {
				return true
			}
		}
	}
	return false
}

func conversionDelayParts(delays []float64, size int, dt float64) (whole, residual []float64, err error) {
	whole, residual = make([]float64, size), make([]float64, size)
	if len(delays) == 0 {
		return whole, residual, nil
	}
	if len(delays) != size {
		return nil, nil, ErrDimensionMismatch
	}
	for i, tau := range delays {
		if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
			return nil, nil, ErrNegativeDelay
		}
		samples := tau / dt
		if math.IsInf(samples, 0) {
			return nil, nil, ErrOverflow
		}
		if isIntegerSampleDelay(samples) {
			whole[i] = math.Round(samples)
		} else {
			whole[i] = math.Floor(samples)
			residual[i] = (samples - whole[i]) * dt
		}
	}
	return whole, residual, nil
}

func applyConversionExternalDelays(cont, disc *System, input, output []float64, dt float64, opts C2DOptions) (*System, error) {
	if opts.Method == C2DMethodZOH && !cont.HasInternalDelay() {
		nsys := *cont
		nsys.InputDelay = input
		nsys.OutputDelay = output
		if conversionHasFractionalExternalDelay(&nsys, dt) {
			out, err := discretizeZOHExternal(&nsys, disc, dt)
			if err != nil {
				return nil, err
			}
			if opts.DelayModeling == C2DDelayModelingState {
				return absorbConversionInternal(out)
			}
			return out, nil
		}
	}
	policy := newDelayConversionPolicy(dt, opts.ThiranOrder, 0)
	if opts.Method == C2DMethodTustin || opts.Method == C2DMethodMatched {
		if opts.ThiranOrder == 0 {
			var err error
			disc.InputDelay, err = roundConversionDelays(input, dt)
			if err != nil {
				return nil, err
			}
			disc.OutputDelay, err = roundConversionDelays(output, dt)
			if err != nil {
				return nil, err
			}
			return disc, nil
		}
		return applyConversionThiran(disc, input, output, dt, opts)
	}
	return policy.applyDiscreteExternal(disc, input, output)
}

func roundConversionDelays(delays []float64, dt float64) ([]float64, error) {
	if delays == nil {
		return nil, nil
	}
	out := make([]float64, len(delays))
	for i, tau := range delays {
		if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
			return nil, ErrNegativeDelay
		}
		out[i] = math.Round(tau / dt)
		if math.IsInf(out[i], 0) {
			return nil, ErrOverflow
		}
	}
	return out, nil
}

func roundConversionPathDelays(delays *mat.Dense, dt float64) (*mat.Dense, error) {
	if delays == nil {
		return nil, nil
	}
	out := denseCopy(delays)
	raw := out.RawMatrix()
	for i := range raw.Rows {
		for j := range raw.Cols {
			tau := raw.Data[i*raw.Stride+j]
			if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
				return nil, ErrNegativeDelay
			}
			tau = math.Round(tau/dt) * dt
			if math.IsInf(tau, 0) {
				return nil, ErrOverflow
			}
			raw.Data[i*raw.Stride+j] = tau
		}
	}
	return out, nil
}

func conversionZOHKernel(a *mat.Dense, b *mat.Dense, period float64) (ad, bd *mat.Dense) {
	n, _ := a.Dims()
	_, m := b.Dims()
	ad, bd = newDense(n, n), newDense(n, m)
	if n == 0 {
		return ad, bd
	}
	augmented := newDense(n+m, n+m)
	ar, br, qr := a.RawMatrix(), b.RawMatrix(), augmented.RawMatrix()
	for i := range n {
		for j := range n {
			qr.Data[i*qr.Stride+j] = period * ar.Data[i*ar.Stride+j]
		}
		for j := range m {
			qr.Data[i*qr.Stride+n+j] = period * br.Data[i*br.Stride+j]
		}
	}
	var exp mat.Dense
	exp.Exp(augmented)
	er := exp.RawMatrix()
	adr, bdr := ad.RawMatrix(), bd.RawMatrix()
	for i := range n {
		copy(adr.Data[i*adr.Stride:i*adr.Stride+n], er.Data[i*er.Stride:i*er.Stride+n])
		copy(bdr.Data[i*bdr.Stride:i*bdr.Stride+m], er.Data[i*er.Stride+n:i*er.Stride+n+m])
	}
	return ad, bd
}

func discretizeZOHExternal(cont, disc *System, dt float64) (*System, error) {
	n, m, p := cont.Dims()
	inputWhole, inputResidual, err := conversionDelayParts(cont.InputDelay, m, dt)
	if err != nil {
		return nil, err
	}
	outputWhole, outputResidual, err := conversionDelayParts(cont.OutputDelay, p, dt)
	if err != nil {
		return nil, err
	}
	out := disc.Copy()
	if cont.InputDelay != nil {
		out.InputDelay = inputWhole
	}
	if cont.OutputDelay != nil {
		out.OutputDelay = outputWhole
	}
	out.B = newDense(n, m)
	out.C = newDense(p, n)
	out.D = newDense(p, m)
	oldB := newDense(n, m)
	oldD, newerD := newDense(p, m), newDense(p, m)
	for j := range m {
		if n > 0 {
			column := newDense(n, 1)
			for i := range n {
				column.Set(i, 0, cont.B.At(i, j))
			}
			_, newB := conversionZOHKernel(cont.A, column, dt-inputResidual[j])
			for i := range n {
				out.B.Set(i, j, newB.At(i, 0))
				oldB.Set(i, j, disc.B.At(i, j)-newB.At(i, 0))
			}
		}
	}
	for i := range p {
		rho := outputResidual[i]
		var backA, backB *mat.Dense
		if n > 0 {
			backA, backB = conversionZOHKernel(cont.A, cont.B, -rho)
			for k := range n {
				v := 0.
				for l := range n {
					v += cont.C.At(i, l) * backA.At(l, k)
				}
				out.C.Set(i, k, v)
			}
		}
		for j := range m {
			total := rho + inputResidual[j]
			if total == 0 {
				out.D.Set(i, j, cont.D.At(i, j))
				continue
			}
			oldPart := 0.
			olderPart := 0.
			if n > 0 {
				limited := backB
				if total > dt {
					_, forward := conversionZOHKernel(cont.A, cont.B, dt-inputResidual[j])
					limited = newDense(n, m)
					limited.Mul(backA, forward)
					limited.Scale(-1, limited)
				}
				for k := range n {
					oldPart += cont.C.At(i, k) * limited.At(k, j)
					olderPart += cont.C.At(i, k) * (backB.At(k, j) - limited.At(k, j))
				}
			}
			if total <= dt {
				oldPart += cont.D.At(i, j)
			} else {
				olderPart += cont.D.At(i, j)
			}
			oldD.Set(i, j, oldPart)
			newerD.Set(i, j, olderPart)
		}
	}
	return attachConversionInputHistory(out, oldB, oldD, newerD), nil
}

func attachConversionInputHistory(sys *System, oldB, oldD, olderD *mat.Dense) *System {
	n, m, p := sys.Dims()
	type entry struct{ channel, lag int }
	entries := []entry{}
	for j := range m {
		first, second := false, false
		for i := range n {
			first = first || oldB.At(i, j) != 0
		}
		for i := range p {
			first = first || oldD.At(i, j) != 0
			second = second || olderD.At(i, j) != 0
		}
		if first {
			entries = append(entries, entry{j, 1})
		}
		if second {
			entries = append(entries, entry{j, 2})
		}
	}
	if len(entries) == 0 {
		return sys
	}
	count := len(entries)
	lft := &LFTDelay{Tau: make([]float64, count), B2: newDense(n, count), C2: newDense(count, n), D12: newDense(p, count), D21: newDense(count, m), D22: newDense(count, count)}
	for k, e := range entries {
		lft.Tau[k] = float64(e.lag)
		lft.D21.Set(k, e.channel, 1)
		if e.lag == 1 {
			for i := range n {
				lft.B2.Set(i, k, oldB.At(i, e.channel))
			}
			for i := range p {
				lft.D12.Set(i, k, oldD.At(i, e.channel))
			}
		} else {
			for i := range p {
				lft.D12.Set(i, k, olderD.At(i, e.channel))
			}
		}
	}
	sys.LFT = lft
	return sys
}

func conversionStateAsDelay(sys *System) *System {
	n, m, p := sys.Dims()
	if n == 0 {
		return sys
	}
	out := &System{A: newDense(0, 0), B: newDense(0, m), C: newDense(p, 0), D: denseCopy(sys.D), Dt: sys.Dt}
	out.LFT = &LFTDelay{Tau: make([]float64, n), B2: newDense(0, n), C2: newDense(n, 0), D12: denseCopy(sys.C), D21: denseCopy(sys.B), D22: denseCopy(sys.A)}
	for i := range n {
		out.LFT.Tau[i] = 1
	}
	propagateIONames(out, sys)
	return out
}

func applyConversionThiran(disc *System, input, output []float64, dt float64, opts C2DOptions) (*System, error) {
	metadata := metadataFromSystem(disc)
	states, _, _ := disc.Dims()
	var err error
	disc.InputDelay, err = convertSliceDelayToDiscrete(input, dt, opts.ThiranOrder)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(output, dt, opts.ThiranOrder)
	if err != nil {
		return nil, err
	}
	for _, side := range []struct {
		delays []float64
		input  bool
	}{{input, true}, {output, false}} {
		fractional := false
		for _, tau := range side.delays {
			fractional = fractional || !isIntegerSampleDelay(tau/dt)
		}
		if !fractional {
			continue
		}
		bank, bankErr := conversionThiranBank(side.delays, dt, opts.ThiranOrder, opts.DelayModeling)
		if bankErr != nil {
			return nil, bankErr
		}
		if side.input {
			disc.InputDelay = nil
			disc = conversionSeries(bank, disc)
		} else {
			bank.OutputDelay, bank.InputDelay = bank.InputDelay, nil
			disc.OutputDelay = nil
			disc = conversionSeries(disc, bank)
		}
	}
	metadata.applyIO(disc)
	disc.Notes = metadata.notes
	if n, _, _ := disc.Dims(); n == states {
		disc.StateName = copyStringSlice(metadata.state)
	} else {
		disc.StateName = nil
	}
	return disc, nil
}

func conversionAugmentedRational(sys *System) *System {
	n, m, p := sys.Dims()
	count := len(sys.LFT.Tau)
	out := &System{A: denseCopy(sys.A), B: newDense(n, m+count), C: newDense(p+count, n), D: newDense(p+count, m+count), Dt: sys.Dt}
	if n > 0 {
		setBlock(out.B, 0, 0, sys.B)
		setBlock(out.B, 0, m, sys.LFT.B2)
		setBlock(out.C, 0, 0, sys.C)
		setBlock(out.C, p, 0, sys.LFT.C2)
	}
	setBlock(out.D, 0, 0, sys.D)
	setBlock(out.D, 0, m, sys.LFT.D12)
	setBlock(out.D, p, 0, sys.LFT.D21)
	setBlock(out.D, p, m, sys.LFT.D22)
	return out
}

func splitConversionRational(augmented *System, m, p int, tau []float64) *System {
	n, _, _ := augmented.Dims()
	count := len(tau)
	out := &System{A: denseCopy(augmented.A), B: newDense(n, m), C: newDense(p, n), D: newDense(p, m), Dt: augmented.Dt}
	out.LFT = &LFTDelay{Tau: append([]float64(nil), tau...), B2: newDense(n, count), C2: newDense(count, n), D12: newDense(p, count), D21: newDense(count, m), D22: newDense(count, count)}
	copyConversionBlock(out.B, augmented.B, 0, 0)
	copyConversionBlock(out.LFT.B2, augmented.B, 0, m)
	copyConversionBlock(out.C, augmented.C, 0, 0)
	copyConversionBlock(out.LFT.C2, augmented.C, p, 0)
	copyConversionBlock(out.D, augmented.D, 0, 0)
	copyConversionBlock(out.LFT.D12, augmented.D, 0, m)
	copyConversionBlock(out.LFT.D21, augmented.D, p, 0)
	copyConversionBlock(out.LFT.D22, augmented.D, p, m)
	return out
}

func copyConversionBlock(dst, src *mat.Dense, row, col int) {
	dr, sr := dst.RawMatrix(), src.RawMatrix()
	for i := range dr.Rows {
		copy(dr.Data[i*dr.Stride:i*dr.Stride+dr.Cols], sr.Data[(row+i)*sr.Stride+col:(row+i)*sr.Stride+col+dr.Cols])
	}
}

func discretizeInternalModel(sys *System, dt float64, opts C2DOptions) (*System, error) {
	_, m, p := sys.Dims()
	tau := make([]float64, len(sys.LFT.Tau))
	if opts.Method == C2DMethodImpulse || opts.Method == C2DMethodMatched || opts.Method == C2DMethodLeastSquares {
		return nil, fmt.Errorf("method %s does not support internal feedback delays: %w", opts.Method, ErrFeedbackDelay)
	}
	if opts.Method == C2DMethodTustin && opts.ThiranOrder > 0 {
		return discretizeInternalThiran(sys, dt, opts)
	}
	for i, value := range sys.LFT.Tau {
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, ErrZeroInternalDelay
		}
		samples := value / dt
		if opts.Method != C2DMethodTustin && !isIntegerSampleDelay(samples) {
			return discretizeHoldFeedback(sys, nil, nil, dt, opts)
		}
		tau[i] = math.Round(samples)
	}
	rational := conversionAugmentedRational(sys)
	var disc *System
	var err error
	switch opts.Method {
	case C2DMethodTustin:
		beta, betaErr := tustinBeta(dt, opts.PrewarpFrequency)
		if betaErr != nil {
			return nil, betaErr
		}
		disc, err = bilinear(rational, -beta, -1, 1, beta)
		if disc != nil {
			disc.Dt = dt
		}
	case C2DMethodZOH:
		disc, err = rational.discretizeZOH(dt)
	case C2DMethodFOH:
		disc, err = rational.discretizeModifiedFOH(dt)
	default:
		return nil, fmt.Errorf("method %q with internal delays: %w", opts.Method, ErrOptionUnsupported)
	}
	if err != nil {
		return nil, err
	}
	out := splitConversionRational(disc, m, p, tau)
	if sys.Delay != nil {
		out.Delay, err = convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
	}
	propagateNames(out, sys)
	out, err = eliminateConversionZeroDelays(out)
	if err != nil {
		return nil, err
	}
	return out, err
}

func undiscretizeInternalTustin(sys *System, beta float64) (*System, error) {
	return undiscretizeInternal(sys, func(rational *System) (*System, error) {
		return bilinear(rational, 1, beta, 1, beta)
	})
}

// undiscretizeInternal converts the delay-free augmented rational H(z) and maps
// each internal delay z^-k to exp(-s·k·Ts), the inverse of the c2d mapping.
func undiscretizeInternal(sys *System, convert func(*System) (*System, error)) (*System, error) {
	_, m, p := sys.Dims()
	cont, err := convert(conversionAugmentedRational(sys))
	if err != nil {
		return nil, err
	}
	cont.Dt = 0
	tau := make([]float64, len(sys.LFT.Tau))
	for i, value := range sys.LFT.Tau {
		tau[i] = value * sys.Dt
	}
	out := splitConversionRational(cont, m, p, tau)
	if sys.Delay != nil {
		out.Delay = convertDelayToContinuous(sys.Delay, sys.Dt)
	}
	out.InputDelay = convertSliceDelayToContinuous(sys.InputDelay, sys.Dt)
	out.OutputDelay = convertSliceDelayToContinuous(sys.OutputDelay, sys.Dt)
	propagateNames(out, sys)
	return out, nil
}

func absorbConversionInternal(sys *System) (*System, error) {
	if !sys.HasInternalDelay() {
		return sys, nil
	}
	return absorbInternalDiscreteDelay(sys)
}

func eliminateConversionZeroDelays(sys *System) (*System, error) {
	n, m, p := sys.Dims()
	zeros, remaining := []int{}, []int{}
	for k, tau := range sys.LFT.Tau {
		if tau == 0 {
			zeros = append(zeros, k)
		} else {
			remaining = append(remaining, k)
		}
	}
	if len(zeros) == 0 {
		return sys, nil
	}
	h := conversionAugmentedRational(sys)
	rows, cols := make([]int, p), make([]int, m)
	for i := range p {
		rows[i] = i
	}
	for j := range m {
		cols[j] = j
	}
	for _, k := range remaining {
		rows = append(rows, p+k)
		cols = append(cols, m+k)
	}
	nz := len(zeros)
	bz, cz := newDense(n, nz), newDense(nz, n)
	dz0, d0z, loop := newDense(len(rows), nz), newDense(nz, len(cols)), newDense(nz, nz)
	out := &System{A: denseCopy(h.A), B: newDense(n, len(cols)), C: newDense(len(rows), n), D: newDense(len(rows), len(cols)), Dt: sys.Dt}
	for i := range n {
		for j, k := range cols {
			out.B.Set(i, j, h.B.At(i, k))
		}
		for j, k := range zeros {
			bz.Set(i, j, h.B.At(i, m+k))
		}
	}
	for i, k := range rows {
		for j := range n {
			out.C.Set(i, j, h.C.At(k, j))
		}
		for j, l := range cols {
			out.D.Set(i, j, h.D.At(k, l))
		}
		for j, l := range zeros {
			dz0.Set(i, j, h.D.At(k, m+l))
		}
	}
	for i, k := range zeros {
		for j := range n {
			cz.Set(i, j, h.C.At(p+k, j))
		}
		for j, l := range cols {
			d0z.Set(i, j, h.D.At(p+k, l))
		}
		for j, l := range zeros {
			v := -h.D.At(p+k, m+l)
			if i == j {
				v++
			}
			loop.Set(i, j, v)
		}
	}
	var lu mat.LU
	lu.Factorize(loop)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("rounded internal delays create an algebraic loop: %w", ErrAlgebraicLoop)
	}
	solvedD := newDense(nz, len(cols))
	if len(cols) > 0 {
		if err := lu.SolveTo(solvedD, false, d0z); err != nil {
			return nil, ErrAlgebraicLoop
		}
		var extra mat.Dense
		extra.Mul(dz0, solvedD)
		out.D.Add(out.D, &extra)
	}
	if n > 0 {
		solvedC := newDense(nz, n)
		if err := lu.SolveTo(solvedC, false, cz); err != nil {
			return nil, ErrAlgebraicLoop
		}
		var extra mat.Dense
		extra.Mul(bz, solvedC)
		out.A.Add(out.A, &extra)
		if len(cols) > 0 {
			extra.Reset()
			extra.Mul(bz, solvedD)
			out.B.Add(out.B, &extra)
		}
		if len(rows) > 0 {
			extra.Reset()
			extra.Mul(dz0, solvedC)
			out.C.Add(out.C, &extra)
		}
	}
	tau := make([]float64, len(remaining))
	for i, k := range remaining {
		tau[i] = sys.LFT.Tau[k]
	}
	if len(tau) > 0 {
		out = splitConversionRational(out, m, p, tau)
	}
	out.InputDelay = append([]float64(nil), sys.InputDelay...)
	out.OutputDelay = append([]float64(nil), sys.OutputDelay...)
	out.Delay = copyDelayOrNil(sys.Delay)
	propagateNames(out, sys)
	return out, nil
}

func discretizeFOHFractionalChannel(cont *System, dt, tau float64) (*System, error) {
	n, m, p := cont.Dims()
	if m != 1 || p != 1 {
		return nil, ErrNotSISO
	}
	whole, residual, err := conversionDelayParts([]float64{tau}, 1, dt)
	if err != nil {
		return nil, err
	}
	rho := residual[0]
	if rho == 0 {
		out, err := cont.discretizeModifiedFOH(dt)
		if out != nil {
			out.InputDelay = whole
		}
		return out, err
	}
	out := cont.Copy()
	out.Dt = dt
	out.InputDelay = whole
	oldB, oldD := newDense(n, 1), newDense(1, 1)
	out.D = newDense(1, 1)
	out.D.Set(0, 0, cont.D.At(0, 0)*(1-rho/dt))
	oldD.Set(0, 0, cont.D.At(0, 0)*rho/dt)
	if n > 0 {
		ad, bd := conversionZOHKernel(cont.A, cont.B, dt)
		_, k0r, k1r := fohKernel(cont.A, rho)
		ef, _, k1f := fohKernel(cont.A, dt-rho)
		var weighted, plus, minus, minusKernel, adplus mat.Dense
		weighted.Sub(k0r, k1r)
		weighted.Scale(rho/dt, &weighted)
		minusKernel.Mul(ef, &weighted)
		minus.Mul(&minusKernel, cont.B)
		plus.Mul(k1f, cont.B)
		plus.Scale((dt-rho)/dt, &plus)
		adplus.Mul(ad, &plus)
		out.A = ad
		out.B = newDense(n, 1)
		out.B.Add(bd, &adplus)
		out.B.Sub(out.B, &plus)
		out.B.Sub(out.B, &minus)
		oldB.Copy(&minus)
		var feedthrough mat.Dense
		feedthrough.Mul(cont.C, &plus)
		out.D.Add(out.D, &feedthrough)
	}
	return attachConversionInputHistory(out, oldB, oldD, newDense(1, 1)), nil
}

func discretizeDelayedChannels(sys *System, dt float64, opts C2DOptions) (*System, error) {
	n, m, p := sys.Dims()
	if sys.HasInternalDelay() {
		return nil, ErrFeedbackDelay
	}
	if m == 0 || p == 0 {
		return nil, ErrDimensionMismatch
	}
	var combined *System
	for i := range p {
		for j := range m {
			tau := 0.
			if len(sys.InputDelay) > 0 {
				tau += sys.InputDelay[j]
			}
			if len(sys.OutputDelay) > 0 {
				tau += sys.OutputDelay[i]
			}
			if sys.Delay != nil {
				tau += sys.Delay.At(i, j)
			}
			cont := &System{A: denseCopy(sys.A), B: newDense(n, 1), C: newDense(1, n), D: newDense(1, 1)}
			for k := range n {
				cont.B.Set(k, 0, sys.B.At(k, j))
				cont.C.Set(0, k, sys.C.At(i, k))
			}
			cont.D.Set(0, 0, sys.D.At(i, j))
			var disc *System
			var err error
			switch opts.Method {
			case C2DMethodZOH:
				disc, err = cont.discretizeZOH(dt)
				if err == nil {
					cont.InputDelay = []float64{tau}
					disc, err = discretizeZOHExternal(cont, disc, dt)
				}
			case C2DMethodFOH:
				disc, err = discretizeFOHFractionalChannel(cont, dt, tau)
			case C2DMethodImpulse:
				disc, err = discretizeImpulseDelayedChannel(cont, dt, tau)
			default:
				return nil, fmt.Errorf("method %q with fractional channel delays: %w", opts.Method, ErrOptionUnsupported)
			}
			if err != nil {
				return nil, err
			}
			if disc.HasDelay() {
				disc, err = disc.PullDelaysToLFT()
				if err != nil {
					return nil, err
				}
			}
			expanded := embedConversionChannel(disc, m, p, i, j)
			if combined == nil {
				combined = expanded
			} else {
				combined, err = Parallel(combined, expanded)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	propagateIONames(combined, sys)
	if opts.DelayModeling == C2DDelayModelingState {
		return absorbConversionInternal(combined)
	}
	return combined, nil
}

func embedConversionChannel(sys *System, m, p, output, input int) *System {
	n, _, _ := sys.Dims()
	out := &System{A: denseCopy(sys.A), B: newDense(n, m), C: newDense(p, n), D: newDense(p, m), Dt: sys.Dt}
	for i := range n {
		out.B.Set(i, input, sys.B.At(i, 0))
		out.C.Set(output, i, sys.C.At(0, i))
	}
	out.D.Set(output, input, sys.D.At(0, 0))
	if sys.LFT != nil {
		count := len(sys.LFT.Tau)
		out.LFT = &LFTDelay{Tau: append([]float64(nil), sys.LFT.Tau...), B2: denseCopy(sys.LFT.B2), C2: denseCopy(sys.LFT.C2), D12: newDense(p, count), D21: newDense(count, m), D22: denseCopy(sys.LFT.D22)}
		for k := range count {
			out.LFT.D12.Set(output, k, sys.LFT.D12.At(0, k))
			out.LFT.D21.Set(k, input, sys.LFT.D21.At(k, 0))
		}
	}
	return out
}

func conversionHasFractionalPathDelay(delay *mat.Dense, dt float64) bool {
	if delay == nil {
		return false
	}
	raw := delay.RawMatrix()
	for i := range raw.Rows {
		for j := range raw.Cols {
			if !isIntegerSampleDelay(raw.Data[i*raw.Stride+j] / dt) {
				return true
			}
		}
	}
	return false
}

func discretizeImpulseDelayedChannel(cont *System, dt, tau float64) (*System, error) {
	if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return nil, ErrNegativeDelay
	}
	steps := tau / dt
	if isIntegerSampleDelay(steps) {
		steps = math.Round(steps)
	} else {
		steps = math.Ceil(steps)
	}
	if math.IsInf(steps, 0) {
		return nil, ErrOverflow
	}
	out, err := cont.discretizeImpulseParity(dt)
	if err != nil {
		return nil, err
	}
	n, _, _ := cont.Dims()
	if n > 0 {
		shift, _ := conversionZOHKernel(cont.A, newDense(n, 0), steps*dt-tau)
		out.C.Mul(cont.C, shift)
		out.D.Mul(out.C, cont.B)
		out.D.Scale(dt, out.D)
	}
	out.InputDelay = []float64{steps}
	return out, nil
}
