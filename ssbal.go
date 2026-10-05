package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type SsbalResult struct {
	Sys *System
	T   *mat.Dense
}

// SsbalOption configures Ssbal.
type SsbalOption func(*ssbalConfig)

type ssbalConfig struct {
	condT float64
}

// WithCondT bounds the condition number of the Ssbal transformation T, as
// MATLAB ssbal(sys,condT). Values below 1 and NaN are treated as 1 (T a
// multiple of the identity). The default is +Inf (unconstrained).
func WithCondT(condT float64) SsbalOption {
	return func(c *ssbalConfig) {
		c.condT = condT
	}
}

// Ssbal balances a delay-free model by a diagonal similarity transformation T,
// returning (T·E/T, T·A/T, T·B, C/T, D) so that the rows and columns of
// [T·(|A|+|E|)/T, T·B; C/T, 0] have approximately equal 1-norms, as MATLAB
// ssbal. Descriptor models are supported; E is scaled like A and an explicit
// model keeps E nil.
//
// The entries of T are powers of two, so the scaling is exact. As in MATLAB
// ssbal, the states are never permuted. Diagonal entries of A and E are
// excluded from the norms and the Osborne iteration follows SLICOT TB01ID, so
// T may differ from MATLAB's LAPACK-balance-based choice. With WithCondT the
// exponents of T are compressed linearly (as MATLAB abcbalance) until
// cond(T) ≤ condT holds strictly; MATLAB only compresses once cond(T) exceeds
// 10·condT.
func Ssbal(sys *System, opts ...SsbalOption) (*SsbalResult, error) {
	cfg := ssbalConfig{condT: math.Inf(1)}
	for _, o := range opts {
		o(&cfg)
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireDelayFree("Ssbal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		eye := &mat.Dense{}
		return &SsbalResult{Sys: policy.zeroOrderCopy(), T: eye}, nil
	}

	br := newRealizationCopy(sys, n, m, p)
	t := make([]float64, n)
	for i := range t {
		t[i] = 1
	}
	br.balance(t)
	condT := cfg.condT
	if !(condT >= 1) {
		condT = 1
	}
	if boundSsbalCond(t, condT) {
		br = newRealizationCopy(sys, n, m, p)
		scaleRealization(&br, t)
	}

	newSys, err := policy.result(
		mat.NewDense(n, n, br.a), mat.NewDense(n, m, br.b), mat.NewDense(p, n, br.c), denseCopy(sys.D))
	if err != nil {
		return nil, err
	}
	if sys.E != nil {
		newSys.E = mat.NewDense(n, n, br.e)
	}

	T := mat.NewDense(n, n, nil)
	for i, v := range t {
		T.Set(i, i, v)
	}
	return &SsbalResult{Sys: newSys, T: T}, nil
}

// boundSsbalCond maps the power-of-two exponents of t affinely onto
// [kmin, kmin+⌊log2 condT⌋] when max(t)/min(t) > condT, keeping both
// endpoints exact so cond(diag(t)) ≤ condT. It reports whether t changed.
func boundSsbalCond(t []float64, condT float64) bool {
	kmin, kmax := math.MaxInt, math.MinInt
	for _, v := range t {
		_, e := math.Frexp(v)
		kmin, kmax = min(kmin, e), max(kmax, e)
	}
	span := kmax - kmin
	if math.Ldexp(1, span) <= condT {
		return false
	}
	l := math.Floor(math.Log2(condT))
	for i, v := range t {
		_, e := math.Frexp(v)
		t[i] = math.Ldexp(1, kmin-1+int(math.Round(float64(e-kmin)*l/float64(span))))
	}
	return true
}

// scaleRealization applies x̂ = diag(t)·x: Â = T·A/T, Ê = T·E/T, B̂ = T·B,
// Ĉ = C/T.
func scaleRealization(br *balancedRealization, t []float64) {
	n, m, p := br.n, br.m, br.p
	for i := range n {
		for j := range n {
			f := t[i] / t[j]
			br.a[i*n+j] *= f
			if br.e != nil {
				br.e[i*n+j] *= f
			}
		}
		for k := range m {
			br.b[i*m+k] *= t[i]
		}
		for k := range p {
			br.c[k*n+i] /= t[i]
		}
	}
}
