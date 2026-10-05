package controlsys

type visibleDelaySelection struct {
	values     []float64
	hasNonzero bool
}

func selectVisibleDelays(delay []float64, indexes []int) visibleDelaySelection {
	out := visibleDelaySelection{values: make([]float64, len(indexes))}
	if delay == nil {
		return out
	}
	for k, idx := range indexes {
		out.values[k] = delay[idx]
		if delay[idx] != 0 {
			out.hasNonzero = true
		}
	}
	return out
}

// selectHoistableDelays returns the delays of the selected channels that may
// stay external because open reports them outside every feedback path, plus
// the indexes of those channels.
func selectHoistableDelays(delay []float64, indexes []int, open func(int) bool) (visibleDelaySelection, []int) {
	out := visibleDelaySelection{values: make([]float64, len(indexes))}
	var hoist []int
	for k, idx := range indexes {
		if !open(idx) {
			continue
		}
		hoist = append(hoist, idx)
		if delay != nil && delay[idx] != 0 {
			out.values[k] = delay[idx]
			out.hasNonzero = true
		}
	}
	return out, hoist
}

func selectLeadingVisibleDelays(delay []float64, n int) visibleDelaySelection {
	out := visibleDelaySelection{values: make([]float64, n)}
	if delay == nil {
		return out
	}
	copy(out.values, delay[:n])
	out.hasNonzero = delaySliceHasNonzero(out.values)
	return out
}

func clearDelayIndexes(delay []float64, indexes []int) []float64 {
	if delay == nil {
		return nil
	}
	for _, idx := range indexes {
		delay[idx] = 0
	}
	if delaySliceHasNonzero(delay) {
		return delay
	}
	return nil
}

func clearLeadingDelays(delay []float64, n int) []float64 {
	if delay == nil {
		return nil
	}
	for i := range n {
		delay[i] = 0
	}
	if delaySliceHasNonzero(delay) {
		return delay
	}
	return nil
}
