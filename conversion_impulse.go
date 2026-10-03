package controlsys

func (sys *System) discretizeImpulseParity(dt float64) (*System, error) {
	n, m, p := sys.Dims()
	out := &System{A: newDense(n, n), B: denseCopy(sys.B), C: denseCopy(sys.C), D: newDense(p, m), Dt: dt}
	if n > 0 {
		scaled := newDense(n, n)
		scaled.Scale(dt, sys.A)
		out.A.Exp(scaled)
		if m > 0 {
			out.B.Mul(out.A, sys.B)
			out.B.Scale(dt, out.B)
			if p > 0 {
				out.D.Mul(sys.C, sys.B)
				out.D.Scale(dt, out.D)
			}
		}
	}
	if sys.Delay != nil {
		var err error
		out.Delay, err = convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
	}
	propagateNames(out, sys)
	return out, nil
}
