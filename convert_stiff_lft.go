package controlsys

import "plantcontrol.org/v1/gonum/mat"

// stiffLFTExplicit returns, for a continuous descriptor sys with internal
// delays, the explicit model with the same delay feedback whose descriptor
// part G = (A, E, [B B2], [C; C2], [D D12; D21 D22]) is replaced by the
// parallel connection of the foldInaccurateParts of G, each in real Schur
// form: MATLAB keeps H = lft(G, delay) and G its pencil, so the poles are
// those of (A, E) rather than of the LU fold E⁻¹A. The I/O delays and names
// of sys carry over; state names are dropped. ok is false when sys has no
// internal delays or G does not need the split.
func stiffLFTExplicit(sys *System) (*System, bool) {
	if !sys.IsContinuous() || !sys.HasInternalDelay() || !sys.IsDescriptor() {
		return nil, false
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if lu.Cond() < foldAccurateECond {
		return nil, false
	}
	_, m, p := sys.Dims()
	g := conversionAugmentedRational(sys)
	g.E = sys.E
	parts, ok := stiffDescriptorParts(g)
	if !ok {
		return nil, false
	}
	for i, part := range parts {
		if parts[i], ok = realSchurPart(part); !ok {
			return nil, false
		}
	}
	explicit, err := parallelExplicit(parts)
	if err != nil {
		return nil, false
	}
	out := splitConversionRational(explicit, m, p, sys.LFT.Tau)
	delays := sys.Copy()
	out.Delay, out.InputDelay, out.OutputDelay = delays.Delay, delays.InputDelay, delays.OutputDelay
	propagateNames(out, sys)
	out.StateName = nil
	return out, true
}
