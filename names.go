package controlsys

import (
	"fmt"
	"slices"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
)

func copyStringSlice(s []string) []string {
	if s == nil {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

func concatStringSlices(slices [][]string, lengths []int) []string {
	allNil := true
	for _, s := range slices {
		if s != nil {
			allNil = false
			break
		}
	}
	if allNil {
		return nil
	}
	total := 0
	for _, l := range lengths {
		total += l
	}
	out := make([]string, 0, total)
	for i, s := range slices {
		if s != nil {
			out = append(out, s...)
		} else {
			out = append(out, make([]string, lengths[i])...)
		}
	}
	return out
}

func selectStringSlice(names []string, indices []int) []string {
	if names == nil {
		return nil
	}
	out := make([]string, len(indices))
	for i, idx := range indices {
		if idx < len(names) {
			out[i] = names[idx]
		}
	}
	return out
}

func lookupSignalIndex(names []string, name string) (int, error) {
	for i, n := range names {
		if n == name {
			return i, nil
		}
	}
	return -1, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
}

func lookupSignalIndices(names []string, targets []string) ([]int, error) {
	indices := make([]int, len(targets))
	for i, t := range targets {
		idx, err := lookupSignalIndex(names, t)
		if err != nil {
			return nil, err
		}
		indices[i] = idx
	}
	return indices, nil
}

func propagateNames(result, src *System) {
	metadataFromSystem(src).applyAll(result)
}

func propagateIONames(result, src *System) {
	metadataFromSystem(src).applyIO(result)
}

func expandName(name string, count int) []string {
	if count == 1 {
		return []string{name}
	}
	out := make([]string, count)
	for i := range out {
		out[i] = fmt.Sprintf("%s(%d)", name, i+1)
	}
	return out
}

func autoLabel(prefix string, count int) []string {
	out := make([]string, count)
	for i := range out {
		out[i] = fmt.Sprintf("%s%d", prefix, i+1)
	}
	return out
}

func (sys *System) SetInputName(names ...string) error {
	_, m, _ := sys.Dims()
	if len(names) == 1 && m > 1 {
		sys.InputName = expandName(names[0], m)
		return nil
	}
	if len(names) != m {
		return fmt.Errorf("SetInputName: got %d names for %d inputs: %w", len(names), m, ErrDimensionMismatch)
	}
	sys.InputName = copyStringSlice(names)
	return nil
}

func (sys *System) SetOutputName(names ...string) error {
	_, _, p := sys.Dims()
	if len(names) == 1 && p > 1 {
		sys.OutputName = expandName(names[0], p)
		return nil
	}
	if len(names) != p {
		return fmt.Errorf("SetOutputName: got %d names for %d outputs: %w", len(names), p, ErrDimensionMismatch)
	}
	sys.OutputName = copyStringSlice(names)
	return nil
}

func (sys *System) SetStateName(names ...string) error {
	n, _, _ := sys.Dims()
	if len(names) == 1 && n > 1 {
		sys.StateName = expandName(names[0], n)
		return nil
	}
	if len(names) != n {
		return fmt.Errorf("SetStateName: got %d names for %d states: %w", len(names), n, ErrDimensionMismatch)
	}
	sys.StateName = copyStringSlice(names)
	return nil
}

func (sys *System) inputLabels() []string {
	_, m, _ := sys.Dims()
	if sys.InputName != nil {
		return sys.InputName
	}
	return autoLabel("u", m)
}

func (sys *System) outputLabels() []string {
	_, _, p := sys.Dims()
	if sys.OutputName != nil {
		return sys.OutputName
	}
	return autoLabel("y", p)
}

func (sys *System) stateLabels() []string {
	n, _, _ := sys.Dims()
	if sys.StateName != nil {
		return sys.StateName
	}
	return autoLabel("x", n)
}

func (sys *System) String() string {
	n, m, p := sys.Dims()
	if n == 0 && m == 0 && p == 0 {
		return "Empty state-space model.\n"
	}

	sl := sys.stateLabels()
	il := sys.inputLabels()
	ol := sys.outputLabels()

	var b strings.Builder

	maxColW := func(labels []string) int {
		w := 0
		for _, l := range labels {
			if len(l) > w {
				w = len(l)
			}
		}
		if w < 8 {
			w = 8
		}
		return w
	}

	writeMatrix := func(name string, mat *mat.Dense, rowLabels, colLabels []string) {
		if mat == nil {
			return
		}
		r, c := mat.Dims()
		if r == 0 || c == 0 {
			return
		}

		colW := maxColW(colLabels)
		rowW := 0
		for _, l := range rowLabels {
			if len(l) > rowW {
				rowW = len(l)
			}
		}
		if rowW < 4 {
			rowW = 4
		}

		fmt.Fprintf(&b, "%s =\n", name)
		fmt.Fprintf(&b, "%*s", rowW+2, "")
		for _, l := range colLabels {
			fmt.Fprintf(&b, "%*s", colW+1, l)
		}
		b.WriteString("\n")
		raw := mat.RawMatrix()
		for i := range r {
			fmt.Fprintf(&b, "  %*s", rowW, rowLabels[i])
			for j := range c {
				fmt.Fprintf(&b, "%*g", colW+1, raw.Data[i*raw.Stride+j])
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	if n > 0 {
		writeMatrix("A", sys.A, sl, sl)
		writeMatrix("B", sys.B, sl, il)
		writeMatrix("C", sys.C, ol, sl)
	}
	writeMatrix("D", sys.D, ol, il)

	if sys.Dt == 0 {
		b.WriteString("Continuous-time state-space model.\n")
	} else {
		fmt.Fprintf(&b, "Discrete-time state-space model (Dt = %g).\n", sys.Dt)
	}
	return b.String()
}

func (sys *System) SelectByIndex(inputs, outputs []int) (*System, error) {
	n, m, p := sys.Dims()
	for _, idx := range inputs {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("select: input index %d out of range [0,%d): %w", idx, m, ErrDimensionMismatch)
		}
	}
	for _, idx := range outputs {
		if idx < 0 || idx >= p {
			return nil, fmt.Errorf("select: output index %d out of range [0,%d): %w", idx, p, ErrDimensionMismatch)
		}
	}

	mSel := len(inputs)
	pSel := len(outputs)
	allStates := rangeInts(n)

	var result *System
	var err error
	Dsel := selectDense(sys.D, outputs, inputs)
	if n == 0 {
		if err := storableStaticGain("select", pSel, mSel); err != nil {
			return nil, err
		}
		result, err = NewGain(Dsel, sys.Dt)
	} else {
		result, err = newNoCopy(denseCopy(sys.A), nonEmptyDense(selectDense(sys.B, allStates, inputs)), nonEmptyDense(selectDense(sys.C, outputs, allStates)), nonEmptyDense(Dsel), sys.Dt)
	}
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	if n == 0 {
		metadataFromSystem(sys).selectIO(inputs, outputs).applyIOOwned(result)
	} else {
		metadataFromSystem(sys).selectIO(inputs, outputs).applyAllOwned(result)
	}

	if sys.Delay != nil {
		result.Delay = copyDelayOrNil(selectDense(sys.Delay, outputs, inputs))
	}
	if sys.InputDelay != nil {
		result.InputDelay = make([]float64, mSel)
		for k, j := range inputs {
			result.InputDelay[k] = sys.InputDelay[j]
		}
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, pSel)
		for k, i := range outputs {
			result.OutputDelay[k] = sys.OutputDelay[i]
		}
	}
	if N := sys.internalDelayCount(); N > 0 {
		allDelays := rangeInts(N)
		result.LFT = &LFTDelay{
			Tau: slices.Clone(sys.LFT.Tau),
			B2:  denseCopy(sys.LFT.B2),
			C2:  denseCopy(sys.LFT.C2),
			D12: selectDense(sys.LFT.D12, outputs, allDelays),
			D21: selectDense(sys.LFT.D21, allDelays, inputs),
			D22: denseCopy(sys.LFT.D22),
		}
	}

	return result, nil
}

func selectDense(src *mat.Dense, rows, cols []int) *mat.Dense {
	dst := newDense(len(rows), len(cols))
	if len(rows) == 0 || len(cols) == 0 {
		return dst
	}
	sRaw := src.RawMatrix()
	dRaw := dst.RawMatrix()
	for k, i := range rows {
		for l, j := range cols {
			dRaw.Data[k*dRaw.Stride+l] = sRaw.Data[i*sRaw.Stride+j]
		}
	}
	return dst
}

func (sys *System) SelectByName(inputs, outputs []string) (*System, error) {
	inIdx, err := lookupSignalIndices(sys.InputName, inputs)
	if err != nil {
		return nil, fmt.Errorf("select inputs: %w", err)
	}
	outIdx, err := lookupSignalIndices(sys.OutputName, outputs)
	if err != nil {
		return nil, fmt.Errorf("select outputs: %w", err)
	}
	return sys.SelectByIndex(inIdx, outIdx)
}

type Connection struct {
	From string
	To   string
	Gain float64
}

func ConnectByName(systems []*System, connections []Connection, inputs, outputs []string) (*System, error) {
	aug, err := BlkDiag(systems...)
	if err != nil {
		return nil, fmt.Errorf("connectbyname: %w", err)
	}

	_, m, p := aug.Dims()

	inIdx, err := lookupSignalIndices(aug.InputName, inputs)
	if err != nil {
		return nil, fmt.Errorf("connectbyname inputs: %w", err)
	}
	outIdx, err := lookupSignalIndices(aug.OutputName, outputs)
	if err != nil {
		return nil, fmt.Errorf("connectbyname outputs: %w", err)
	}

	Q := mat.NewDense(m, p, nil)
	for _, c := range connections {
		fromIdx, err := lookupSignalIndex(aug.OutputName, c.From)
		if err != nil {
			return nil, fmt.Errorf("connectbyname connection from: %w", err)
		}
		toIdx, err := lookupSignalIndex(aug.InputName, c.To)
		if err != nil {
			return nil, fmt.Errorf("connectbyname connection to: %w", err)
		}
		gain := c.Gain
		if gain == 0 {
			gain = 1
		}
		Q.Set(toIdx, fromIdx, gain)
	}

	result, err := Connect(aug, Q, inIdx, outIdx)
	if err != nil {
		return nil, withAlgebraicLoopSignals(err, aug.InputName, aug.OutputName)
	}
	return result, nil
}
