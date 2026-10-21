package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// fzProcCase is a decoded process model with its exact modal form:
// G(s) = d + Σ r_k/(s−λ_k) times e^{−Ls}.
type fzProcCase struct {
	st     ProcessStructure
	par    ProcessParameters
	lambda []complex128
	res    []complex128
	d      float64
	dt     float64
	data   ProcessFitData
	fast   float64
	slow   float64
	cond   float64
}

func fzProcModal(st ProcessStructure, p ProcessParameters) (lambda, res []complex128, d float64) {
	lead := 1.0
	for _, tau := range p.TimeConstants {
		lambda = append(lambda, complex(-1/tau, 0))
		lead *= tau
	}
	if st.UnderdampedPair {
		w, z := p.NaturalFrequency, p.Damping
		re, im := -z*w, w*math.Sqrt(1-z*z)
		lambda = append(lambda, complex(re, im), complex(re, -im))
		lead /= w * w
	}
	if st.Integrator {
		lambda = append(lambda, 0)
	}
	for k, l := range lambda {
		den := complex(lead, 0)
		for j, o := range lambda {
			if j != k {
				den *= l - o
			}
		}
		res = append(res, complex(p.Gain, 0)*(complex(p.ZeroTime, 0)*l+1)/den)
	}
	if len(lambda) == 1 && p.ZeroTime != 0 {
		d = p.Gain * p.ZeroTime / lead
	}
	return lambda, res, d
}

func fzProcPhi(l complex128, h float64) complex128 {
	z := l * complex(h, 0)
	if cmplx.Abs(z) < 1e-5 {
		return complex(h, 0) * (1 + z/2 + z*z/6)
	}
	return (cmplx.Exp(z) - 1) / l
}

// fzProcSimulate is the exact sampled response of the modal form to the
// zero-order-held input delayed by L, from rest with zero input before
// the record. scale is max_k Σ|r_j x_j| (cancellation in the modal sum).
func fzProcSimulate(lambda, res []complex128, d, L, dt float64, u []float64) (y []float64, scale float64) {
	whole := int(math.Floor(L / dt))
	frac := L - float64(whole)*dt
	if frac < dt*1e-12 {
		frac = 0
	}
	at := func(i int) float64 {
		if i < 0 || i >= len(u) {
			return 0
		}
		return u[i]
	}
	x := make([]complex128, len(lambda))
	step := func(h, w float64) {
		for k, l := range lambda {
			x[k] = cmplx.Exp(l*complex(h, 0))*x[k] + fzProcPhi(l, h)*complex(w, 0)
		}
	}
	y = make([]float64, len(u))
	for k := range u {
		idx := k - whole
		if frac > 0 {
			idx--
		}
		v := complex(d*at(idx), 0)
		s := math.Abs(d * at(idx))
		for j := range x {
			v += res[j] * x[j]
			s += cmplx.Abs(res[j] * x[j])
		}
		y[k] = real(v)
		scale = math.Max(scale, s)
		if frac > 0 {
			step(frac, at(k-whole-1))
			step(dt-frac, at(k-whole))
		} else {
			step(dt, at(k-whole))
		}
	}
	return y, scale
}

func fzProcBuild(t *testing.T, s *fzSpec) *fzProcCase {
	t.Helper()
	if s.Discrete || s.N == 0 || s.N > 3 {
		fzSkip(t, "process models: continuous, 1 to 3 poles")
	}
	cs := &fzProcCase{}
	pairs := 0
	for k := 0; k < len(s.S); k++ {
		e := s.S[k]
		switch {
		case real(e) > 0:
			fzSkip(t, "unstable pole: not a process model")
		case e == 0:
			if cs.st.Integrator {
				fzSkip(t, "double integrator")
			}
			cs.st.Integrator = true
		case imag(e) != 0:
			pairs++
			cs.st.UnderdampedPair = true
			cs.par.NaturalFrequency = cmplx.Abs(e)
			cs.par.Damping = -real(e) / cmplx.Abs(e)
			cs.st.Poles += 2
			k++
		default:
			cs.par.TimeConstants = append(cs.par.TimeConstants, -1/real(e))
			cs.st.Poles++
		}
	}
	if cs.st.Poles == 0 || pairs > 1 {
		fzSkip(t, "no process pole")
	}
	if cs.st.UnderdampedPair && (cs.par.Damping < 0.02 || cs.par.Damping > 0.99) {
		fzSkip(t, "damping outside the default bounds")
	}
	sortFloat := func(v []float64) {
		for i := 1; i < len(v); i++ {
			for j := i; j > 0 && v[j] < v[j-1]; j-- {
				v[j], v[j-1] = v[j-1], v[j]
			}
		}
	}
	sortFloat(cs.par.TimeConstants)
	cs.par.Gain = math.Pow(10, fzClamp(math.Abs(s.Vals[6]), 0, 12)-6)
	if s.Vals[6] < 0 {
		cs.par.Gain = -cs.par.Gain
	}
	cs.fast, cs.slow = 0, math.Inf(1)
	var poles []complex128
	for _, tau := range cs.par.TimeConstants {
		cs.fast, cs.slow = math.Max(cs.fast, 1/tau), math.Min(cs.slow, 1/tau)
		poles = append(poles, complex(-1/tau, 0))
	}
	if cs.st.UnderdampedPair {
		w, z := cs.par.NaturalFrequency, cs.par.Damping
		cs.fast, cs.slow = math.Max(cs.fast, w), math.Min(cs.slow, z*w)
		poles = append(poles, complex(-z*w, w*math.Sqrt(1-z*z)))
	}
	for i, a := range poles {
		for j := range i {
			if cmplx.Abs(a-poles[j]) < 1e-2*math.Max(cmplx.Abs(a), cmplx.Abs(poles[j])) {
				fzSkip(t, "near-repeated poles (modal oracle ill-conditioned)")
			}
		}
	}
	if cs.fast/cs.slow > 100 {
		fzSkip(t, "time-scale spread beyond the record length")
	}
	cs.dt = math.Pow(10, fzClamp(s.Vals[8], -0.5, 0.3)) / (4 * cs.fast)
	n := int(math.Ceil(8 / (cs.slow * cs.dt)))
	n = min(max(n, 60), 1500)
	train := n * 7 / 10
	duration := float64(train-1) * cs.dt
	if math.Abs(s.Vals[7]) > 0.5 {
		cs.st.Zero = true
		cs.par.ZeroTime = math.Cos(s.Vals[11]+0.3) * 2 / cs.slow
		cs.par.ZeroTime = fzClamp(cs.par.ZeroTime, -0.9*duration, 0.9*duration)
	}
	if s.In != nil {
		cs.st.Delay = true
		cs.par.Delay = math.Min(s.In[0]*cs.dt, duration/5)
	}
	cs.lambda, cs.res, cs.d = fzProcModal(cs.st, cs.par)
	hold := max(1, int(0.5/(cs.slow*cs.dt)))
	u := fzIOInput(s, n, hold)
	y, scale := fzProcSimulate(cs.lambda, cs.res, cs.d, cs.par.Delay, cs.dt, u)
	ymax := 0.0
	for _, v := range y {
		ymax = math.Max(ymax, math.Abs(v))
	}
	if ymax == 0 {
		fzSkip(t, "zero response")
	}
	cs.cond = scale / ymax
	cs.data = ProcessFitData{Input: u, Output: y, SampleTime: cs.dt, TrainingSamples: train}
	return cs
}

func fzProcCheckPredicted(t *testing.T, s *fzSpec, cs *fzProcCase, op string, r *ProcessFitResult, tol float64) {
	t.Helper()
	lambda, res, d := fzProcModal(r.Structure, r.Parameters)
	want, scale := fzProcSimulate(lambda, res, d, r.Parameters.Delay, cs.dt, cs.data.Input)
	ymax := 0.0
	for _, v := range want {
		ymax = math.Max(ymax, math.Abs(v))
	}
	bound := tol * math.Max(scale, ymax)
	if len(r.Predicted) != len(want) || len(r.Residuals) != len(want) {
		t.Fatalf("%s: %d predicted, %d residuals, want %d\n%s", op, len(r.Predicted), len(r.Residuals), len(want), s)
	}
	for k, v := range want {
		if !(math.Abs(r.Predicted[k]-r.Offset-v) <= bound) {
			t.Fatalf("%s: predicted[%d] = %g, exact response of reported parameters %+v is %g (tol %g, cond %g)\n%s",
				op, k, r.Predicted[k], r.Parameters, v, bound, cs.cond, s)
		}
		if r.Residuals[k] != cs.data.Output[k]-r.Predicted[k] {
			t.Fatalf("%s: residual[%d] = %g, want output − predicted = %g\n%s", op, k, r.Residuals[k], cs.data.Output[k]-r.Predicted[k], s)
		}
	}
}

// FuzzIdentProcess: on the exact sampled response (independent modal ZOH
// oracle, fractional delay split at the hold switch) of a decoded process
// model (1–3 real or paired poles, integrator, zero, delay), EvaluateProcess
// at the true parameters reproduces the record (NRMSE and every predicted
// sample within 1e-8·cond), and FitProcess started near the truth returns
// a fit no worse than its start whose Predicted is the exact response of
// its reported Parameters.
func FuzzIdentProcess(f *testing.F) {
	for _, s := range fzProcSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		cs := fzProcBuild(t, s)
		ctx := context.Background()
		g := 1e3 * math.Abs(cs.par.Gain)
		opts := ProcessFitOptions{Structure: cs.st, Bounds: ProcessBounds{MinGain: -g, MaxGain: g}}
		ev, err := EvaluateProcess(ctx, cs.data, opts, cs.par)
		fzContract(t, s, "EvaluateProcess", ev != nil, err)
		fzErr(t, s, "EvaluateProcess", err)
		tol := 1e-8 * math.Max(1, cs.cond)
		if !(ev.TrainingNRMSE <= tol) || !(ev.ValidationNRMSE <= tol) {
			t.Fatalf("EvaluateProcess(truth %+v %+v): NRMSE train %g validation %g > %g (dt %g)\n%s",
				cs.st, cs.par, ev.TrainingNRMSE, ev.ValidationNRMSE, tol, cs.dt, s)
		}
		fzProcCheckPredicted(t, s, cs, "EvaluateProcess", ev, tol)
		if ev.System == nil {
			t.Fatalf("EvaluateProcess: nil System\n%s", s)
		}
		if cs.par.Delay > 0 && (len(ev.System.InputDelay) != 1 || ev.System.InputDelay[0] != cs.par.Delay) {
			t.Fatalf("EvaluateProcess: System input delay %v, want %g\n%s", ev.System.InputDelay, cs.par.Delay, s)
		}

		init := cs.par
		init.TimeConstants = append([]float64(nil), cs.par.TimeConstants...)
		for i := range init.TimeConstants {
			init.TimeConstants[i] *= 1 + 0.15*math.Sin(float64(i+1)+s.Vals[10])
		}
		if init.NaturalFrequency != 0 {
			init.NaturalFrequency *= 1 + 0.1*math.Cos(s.Vals[10])
		}
		if init.Delay > 0 {
			init.Delay *= 0.8
		}
		init.ZeroTime *= 0.9
		start, err := EvaluateProcess(ctx, cs.data, opts, init)
		fzContract(t, s, "EvaluateProcess(start)", start != nil, err)
		fzErr(t, s, "EvaluateProcess(start)", err)
		opts.Initial, opts.Starts, opts.MaxEvaluations = &init, 1, 300
		fit, err := FitProcess(ctx, cs.data, opts)
		fzContract(t, s, "FitProcess", fit != nil, err)
		fzErr(t, s, "FitProcess", err)
		if !(fit.TrainingNRMSE <= start.TrainingNRMSE*(1+1e-6)+tol) {
			t.Fatalf("FitProcess: training NRMSE %g worse than its start %g\n%s", fit.TrainingNRMSE, start.TrainingNRMSE, s)
		}
		if fit.Evaluations < 1 || fit.Evaluations > opts.MaxEvaluations {
			t.Fatalf("FitProcess: %d evaluations, budget %d\n%s", fit.Evaluations, opts.MaxEvaluations, s)
		}
		if fzProcRegular(fit.Structure, fit.Parameters) {
			fzProcCheckPredicted(t, s, cs, "FitProcess", fit, tol)
		}
	})
}

// fzProcRegular reports whether fitted parameters stay in the regime the
// truth was drawn from: spread at most 100, poles 1% apart.
func fzProcRegular(st ProcessStructure, p ProcessParameters) bool {
	lambda, _, _ := fzProcModal(st, p)
	fast, slow := 0.0, math.Inf(1)
	for i, l := range lambda {
		if l == 0 {
			continue
		}
		r := math.Abs(real(l))
		if imag(l) != 0 {
			fast = math.Max(fast, cmplx.Abs(l))
		} else {
			fast = math.Max(fast, r)
		}
		slow = math.Min(slow, r)
		for j := range i {
			if cmplx.Abs(l-lambda[j]) < 1e-2*math.Max(cmplx.Abs(l), cmplx.Abs(lambda[j])) {
				return false
			}
		}
	}
	return fast/slow <= 100
}

func fzProcSeeds() []fzSeed {
	return []fzSeed{
		{N: 1, M: 1, P: 1, Mag: [3]float64{0}, Damp: [3]float64{2}, Gain: 6, Shear: -10, In: fzNoDelay},
		{N: 1, M: 1, P: 1, Mag: [3]float64{-0.5}, Damp: [3]float64{2}, Gain: 7, Shear: -10, In: 0.4},
		{N: 1, M: 1, P: 1, Mag: [3]float64{0.3}, Damp: [3]float64{2}, Gain: -6.5, DScal: 1, Phase: 0.5, Shear: -10, In: -0.5},
		{N: 2, M: 1, P: 1, Mag: [3]float64{0}, Damp: [3]float64{1.7}, Gain: 6, Shear: -10, In: 0.2},
		{N: 2, M: 1, P: 1, Mag: [3]float64{0}, Damp: [3]float64{2.6}, Gain: 6, DScal: -1, Phase: 2, Shear: -10, In: fzNoDelay},
		{N: 2, M: 1, P: 1, Mag: [3]float64{0.2}, Damp: [3]float64{1.0}, Gain: 6, Shear: -10, In: 0.5},
		{N: 3, M: 1, P: 1, Mag: [3]float64{0, 0.4}, Damp: [3]float64{1.7, 2}, Gain: 6, Shear: -10, In: 0.3},
		{N: 3, M: 1, P: 1, Mag: [3]float64{0, 0.5}, Damp: [3]float64{2.5, 2}, Gain: 6, DScal: 1, Phase: 1, Shear: -10, In: fzNoDelay},
		{N: 2, M: 1, P: 1, Mag: [3]float64{0.1, -40}, Damp: [3]float64{2, 2}, Gain: 6, Shear: -10, In: 0.1},
		{N: 3, M: 1, P: 1, Mag: [3]float64{0.1, -40}, Damp: [3]float64{1.5, 2}, Gain: 6, Shear: -10, In: 0.1},
	}
}
