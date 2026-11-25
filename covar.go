package controlsys

import (
	"fmt"
	"math"
	"math/big"

	"plantcontrol.org/v1/gonum/mat"
)

// Covar computes the steady-state output covariance P = E[y yᵀ] of sys driven
// by white noise with intensity W, as MATLAB covar(sys,W). For continuous
// models, entries fed by white noise through D are infinite.
//
// Delays are honoured exactly. Discrete delays (input, output, I/O and
// internal) are absorbed as poles at z=0 before the Lyapunov solve. For
// continuous models, input, output and I/O delays shift each path's impulse
// response, so P_ij = Σ_kl W_kl ∫ g_ik(t) g_jl(t) dt with
// g_ik(t) = D_ik δ(t-τ_ik) + c_i e^{A(t-τ_ik)} b_k for t > τ_ik; a D-path entry
// is infinite only when two feedthrough paths share the same total delay.
// Continuous internal delays have no finite-order form and return
// ErrContinuousInternalDelay. A model with no outputs has no covariance and
// returns ErrDimensionMismatch; with no inputs P is the p×p zero matrix.
// An Inf or NaN entry of P returns ErrOverflow, except the continuous entries
// white noise through D makes infinite, whatever their state part holds.
// A model IsStable rejects, within its rounding of the band, returns
// ErrUnstable. The state covariance carries Lyap's (DLyap's) certificate: an equation
// within its singular band returns ErrSingularEquation, an uncertified one
// P with ErrAccuracyUnverified.
//
// With a nil error ‖P̃ − P‖_F ≤ 10⁻⁵·‖P‖_F over the finite entries, P that
// of the symmetric part of W, and the infinite entries are exactly those
// whose D·W·Dᵀ entry is nonzero, with its sign; one entry far below ‖P‖_F
// can still be off by more relative to itself. The float64 state covariance
// is gated against its rounding, including cancellation in B·W·Bᵀ and
// D·W·Dᵀ. A P it cannot resolve is recomputed from B·W·Bᵀ formed in
// double-double, the state covariance refined and held in double-double,
// and C·X·Cᵀ accumulated in double-double (with D·W·Dᵀ added exactly for
// discrete models). When ‖P‖_F is below the double-double rounding of its
// terms too, or the refinement fails, P is returned with
// ErrAccuracyUnverified. Continuous models with external delays have no
// refined path: below the float64 rounding of their terms P is returned
// with ErrAccuracyUnverified. A model whose C·Aᵏ·B·W vanishes in exact
// arithmetic (up to 32 states) has a zero state part, returned with a nil
// error.
func Covar(sys *System, W *mat.Dense) (*mat.Dense, error) {
	P, _, err := covarRouted(sys, W)
	return P, err
}

// covarInfo is the route of a Covar result and the quantities its gates
// used: N = max(n, ⌈m/16⌉), the tier-1 ρ and a_D, the tier-2 ρ and the last
// double-double correction δ (NaN when not evaluated).
type covarInfo struct {
	route         hsvRoute
	N             int
	rho1, aD      float64
	rho2, delta   float64
	terms, terms2 float64
}

func covarRouted(sys *System, W *mat.Dense) (*mat.Dense, hsvRoute, error) {
	P, info, err := covarTiered(sys, W)
	return P, info.route, err
}

func covarTiered(sys *System, W *mat.Dense) (*mat.Dense, covarInfo, error) {
	info := covarInfo{rho1: math.NaN(), aD: math.NaN(), rho2: math.NaN(), delta: math.NaN(), terms: math.NaN(), terms2: math.NaN()}
	if err := requireFiniteSystem("Covar", sys); err != nil {
		return nil, info, err
	}
	if err := requireStandardCovarianceSystem(sys, "Covar"); err != nil {
		return nil, info, err
	}
	_, m, _ := sys.Dims()
	if err := validateCovarianceRole("Covar", covarianceInputNoise, W, m); err != nil {
		return nil, info, err
	}
	var err error
	if sys.IsDiscrete() && sys.HasDelay() {
		sys, err = sys.absorbDelay()
	} else {
		sys, err = finiteDimensionalModel(sys, "Covar")
	}
	if err != nil {
		return nil, info, err
	}
	n, m, p := sys.Dims()

	stable, err := sys.IsStable()
	if err != nil {
		return nil, info, err
	}
	if !stable {
		return nil, info, fmt.Errorf("Covar: system is unstable: %w", ErrUnstable)
	}
	if p == 0 {
		return nil, info, fmt.Errorf("Covar: model has no outputs: %w", ErrDimensionMismatch)
	}
	if m == 0 {
		return mat.NewDense(p, p, nil), info, nil
	}
	info.N = max(n, (m+15)/16)
	if sys.IsContinuous() && sys.HasDelay() {
		P, dt, err := covarContinuousDelayed(sys, W)
		if err != nil || n == 0 || covarNoState(sys, W) {
			return P, info, err
		}
		info.terms = dt.terms
		if covarGate(hsvTier1K*float64(dt.N)*eps()*dt.terms, finiteFrobenius(P)) {
			info.route = hsvRoutePlain
			return P, info, nil
		}
		info.route = hsvRouteDelayed
		return P, info, fmt.Errorf("Covar: continuous external delays have no refined path and P is below the float64 rounding of its terms: %w", ErrAccuracyUnverified)
	}

	fd := newCovarFeedthrough(sys, W)
	N := float64(info.N)
	P := mat.NewDense(p, p, nil)
	var unv error
	var st covarState
	if n > 0 {
		var err error
		st, err = covarStateTerm(P, sys, W, covarNeedsScaling(sys.B, sys.C, W))
		if err, unv = unverifiedLyapunov(err); err != nil {
			return nil, info, err
		}
	}
	if err := fd.finish(P, info.N, false); err != nil {
		return nil, info, err
	}
	if unv != nil {
		return P, info, covarUnverified(unv)
	}
	if n == 0 || covarNoState(sys, W) {
		Z, err := fd.zeroState(info.N)
		info.route = hsvRouteZero
		return Z, info, err
	}
	info.terms = st.terms
	pn := finiteFrobenius(P)
	aD := 0.0
	if sys.IsDiscrete() {
		aD = fd.aBound
	}
	if rb := st.rhoBound(); covarGate(hsvTier1K*N*eps()*(rb*st.terms+aD), pn) {
		info.rho1, info.aD, info.route = rb, aD, hsvRoutePlain
		return P, info, nil
	}
	info.rho1 = st.rho()
	if sys.IsDiscrete() {
		aD = fd.aD()
	}
	info.aD = aD
	if covarGate(hsvTier1K*N*eps()*(info.rho1*st.terms+aD), pn) {
		info.route = hsvRoutePlain
		return P, info, nil
	}

	r, err := covarRefined(sys, W)
	if err != nil {
		info.route = hsvRouteCoreFailed
		return P, info, fmt.Errorf("Covar: refined state covariance failed (%v), plain-path P not verified: %w", err, ErrAccuracyUnverified)
	}
	info.rho2, info.delta, info.terms2 = r.rho, r.delta, r.terms
	Pr, err := fd.refinedP(r, info.N)
	if err != nil {
		info.route = hsvRouteCoreFailed
		return P, info, fmt.Errorf("Covar: refined state covariance failed (%v), plain-path P not verified: %w", err, ErrAccuracyUnverified)
	}
	if covarGate((hsvDDK*N*0x1p-100*r.rho+r.delta)*r.terms, finiteFrobenius(Pr)) {
		info.route = hsvRouteCore
		return Pr, info, nil
	}
	if covarZeroNoiseTransfer(sys, W) == transferZero {
		Z, err := fd.zeroState(info.N)
		info.route = hsvRouteZero
		return Z, info, err
	}
	info.route = hsvRouteFloor
	return Pr, info, fmt.Errorf("Covar: P below the double-double resolution of its terms, not resolved to %g relative: %w", hsvTau, ErrAccuracyUnverified)
}

// covarGate reports whether the rounding bound is within τ·‖P‖_F. A P with
// no finite entry (pn = +Inf) passes; a zero, NaN or infinite bound against
// a finite pn fails.
func covarGate(bound, pn float64) bool {
	if math.IsInf(pn, 1) {
		return true
	}
	return pn > 0 && bound <= hsvTau*pn
}

// covarNoState reports whether the stored B, C or W is exactly zero, so
// that the state part of P is exactly zero.
func covarNoState(sys *System, W *mat.Dense) bool {
	return zeroDense(sys.B) || zeroDense(sys.C) || zeroDense(W)
}

// finiteFrobenius returns the Frobenius norm of the finite entries of P;
// the entries white noise through D makes infinite carry no rounding. With
// no finite entry it returns +Inf, so that the gates pass.
func finiteFrobenius(P *mat.Dense) float64 {
	raw := P.RawMatrix()
	top, finite := 0.0, false
	for i := range raw.Rows {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			if isFinite(v) {
				finite = true
				top = math.Max(top, math.Abs(v))
			}
		}
	}
	if !finite {
		return math.Inf(1)
	}
	if top == 0 {
		return 0
	}
	s := 0.0
	for i := range raw.Rows {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			if isFinite(v) {
				x := v / top
				s += x * x
			}
		}
	}
	return top * math.Sqrt(s)
}

// covarFeedthrough is the feedthrough part D·W·Dᵀ of P: F̃ as float64 (the
// product base Covar forms, symmetrized when W is not symmetric) and
// a = ½(|D|·|W|·|D|ᵀ + its transpose), the bound on its rounding. The exact
// symmetric part F of D·W·Dᵀ is formed on demand.
type covarFeedthrough struct {
	sys      *System
	W        *mat.Dense
	p        int
	zero     bool
	F, a     *mat.Dense
	aBound   float64
	exact    []*big.Rat
	exactSet bool
}

func newCovarFeedthrough(sys *System, W *mat.Dense) *covarFeedthrough {
	_, m, p := sys.Dims()
	fd := &covarFeedthrough{sys: sys, W: W, p: p, zero: zeroDense(sys.D) || zeroDense(W)}
	if fd.zero {
		return fd
	}
	var DW mat.Dense
	DW.Mul(sys.D, W)
	fd.F = new(mat.Dense)
	fd.F.Mul(&DW, sys.D.T())
	if !symmetricDense(W, m) {
		symmetrizeDense(fd.F, p)
	}
	dn := mat.Norm(sys.D, 2)
	fd.aBound = dn * dn * mat.Norm(W, 2)
	return fd
}

func symmetricDense(W *mat.Dense, m int) bool {
	for i := range m {
		for j := range i {
			if W.At(i, j) != W.At(j, i) {
				return false
			}
		}
	}
	return true
}

func symmetrizeDense(M *mat.Dense, p int) {
	for i := range p {
		for j := range i {
			v := 0.5 * (M.At(i, j) + M.At(j, i))
			M.Set(i, j, v)
			M.Set(j, i, v)
		}
	}
}

// absBound forms a once and returns it.
func (fd *covarFeedthrough) absBound() *mat.Dense {
	if fd.a == nil {
		_, m, p := fd.sys.Dims()
		D := fd.sys.D
		ad := mat.NewDense(p, m, nil)
		for i := range p {
			for l := range m {
				s := 0.0
				for k := range m {
					s += math.Abs(D.At(i, k) * fd.W.At(k, l))
				}
				ad.Set(i, l, s)
			}
		}
		fd.a = mat.NewDense(p, p, nil)
		for i := range p {
			for j := range p {
				s := 0.0
				for l := range m {
					s += ad.At(i, l)*math.Abs(D.At(j, l)) + ad.At(j, l)*math.Abs(D.At(i, l))
				}
				fd.a.Set(i, j, 0.5*s)
			}
		}
	}
	return fd.a
}

// aD returns a_D = ‖a‖_F (0 when D or W is zero).
func (fd *covarFeedthrough) aD() float64 {
	if fd.zero {
		return 0
	}
	return mat.Norm(fd.absBound(), 2)
}

// exactF returns the exact symmetric part of D·W·Dᵀ, row-major p×p.
func (fd *covarFeedthrough) exactF() []*big.Rat {
	if fd.exactSet {
		return fd.exact
	}
	fd.exactSet = true
	_, m, p := fd.sys.Dims()
	fd.exact = make([]*big.Rat, p*p)
	r := func(x float64) *big.Rat { return new(big.Rat).SetFloat64(x) }
	half := big.NewRat(1, 2)
	ws := make([]*big.Rat, m*m)
	for k := range m {
		for l := range m {
			ws[k*m+l] = new(big.Rat).Mul(half, new(big.Rat).Add(r(fd.W.At(k, l)), r(fd.W.At(l, k))))
		}
	}
	t := new(big.Rat)
	for i := range p {
		for j := range p {
			s := new(big.Rat)
			for k := range m {
				dik := fd.sys.D.At(i, k)
				if fd.zero || dik == 0 {
					continue
				}
				for l := range m {
					if djl := fd.sys.D.At(j, l); djl != 0 {
						t.Mul(r(dik), r(djl))
						s.Add(s, t.Mul(t, ws[k*m+l]))
					}
				}
			}
			fd.exact[i*p+j] = s
		}
	}
	return fd.exact
}

// sign returns the sign of the feedthrough entry (i, j): of F̃ when
// |F̃_ij| > 16·N·ε·a_ij, else of the exact F_ij.
func (fd *covarFeedthrough) sign(i, j, N int) int {
	if fd.zero {
		return 0
	}
	v := fd.F.At(i, j)
	if math.Abs(v) > hsvTier1K*float64(N)*eps()*fd.absBound().At(i, j) {
		return int(math.Copysign(1, v))
	}
	return fd.exactF()[i*fd.p+j].Sign()
}

// finish adds the feedthrough part to the state part in P: for discrete
// models the sum P + F̃ (P + F exact when exact is set), for continuous
// models the infinite entries of the exact pattern.
func (fd *covarFeedthrough) finish(P *mat.Dense, N int, exact bool) error {
	p := fd.p
	if fd.sys.IsDiscrete() {
		if fd.zero {
			return finiteDense("Covar", "P", P, ErrOverflow)
		}
		if exact {
			F := fd.exactF()
			for i := range p {
				for j := range p {
					v, _ := new(big.Rat).Add(new(big.Rat).SetFloat64(P.At(i, j)), F[i*p+j]).Float64()
					P.Set(i, j, v)
				}
			}
		} else {
			P.Add(P, fd.F)
		}
		return finiteDense("Covar", "P", P, ErrOverflow)
	}
	for i := range p {
		for j := range p {
			if err := finishCovarEntry(P, i, j, float64(fd.sign(i, j, N))); err != nil {
				return err
			}
		}
	}
	return nil
}

// zeroState returns P with an exactly zero state part: for discrete models
// F̃ when 16·N·ε·a_D ≤ τ·‖F̃‖_F, else the exact F; for continuous models
// the exact infinite pattern.
func (fd *covarFeedthrough) zeroState(N int) (*mat.Dense, error) {
	Z := mat.NewDense(fd.p, fd.p, nil)
	if fd.sys.IsDiscrete() && !fd.zero {
		exact := !covarGate(hsvTier1K*float64(N)*eps()*fd.aD(), mat.Norm(fd.F, 2))
		return Z, fd.finish(Z, N, exact)
	}
	return Z, fd.finish(Z, N, false)
}

// refinedP rounds the double-double state part of r once, with the exact F
// added first for discrete models, and adds the continuous pattern.
func (fd *covarFeedthrough) refinedP(r *covarRefinedResult, N int) (*mat.Dense, error) {
	p := fd.p
	P := mat.NewDense(p, p, nil)
	if fd.sys.IsDiscrete() && !fd.zero {
		F := fd.exactF()
		for i := range p {
			for j := range p {
				x := r.p[i*p+j]
				s := new(big.Rat).Add(ratLdexp(x.hi, r.e), ratLdexp(x.lo, r.e))
				s.Add(s, F[i*p+j])
				v, _ := s.Float64()
				P.Set(i, j, v)
			}
		}
		return P, finiteDense("Covar", "P", P, ErrOverflow)
	}
	for i := range p {
		for j := range p {
			x := r.p[i*p+j]
			P.Set(i, j, math.Ldexp(x.hi+x.lo, r.e))
		}
	}
	return P, fd.finish(P, N, false)
}

func ratLdexp(x float64, e int) *big.Rat {
	r := new(big.Rat).SetFloat64(x)
	if e == 0 || x == 0 {
		return r
	}
	s := new(big.Int).Lsh(big.NewInt(1), uint(absInt(e)))
	if e > 0 {
		return r.Mul(r, new(big.Rat).SetInt(s))
	}
	return r.Quo(r, new(big.Rat).SetInt(s))
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// covarRefinedResult is the double-double state part C·X·Cᵀ of P (p×p,
// times 2ᵉ), the tier-2 ρ, terms = ‖C‖_F²·‖X‖_F (unscaled) and δ.
type covarRefinedResult struct {
	p            []dd
	e            int
	rho, terms   float64
	delta, corr1 float64
}

// covarRefined computes the state part of P from S = ½(B·W·Bᵀ + B·Wᵀ·Bᵀ)
// formed in double-double from the pow2-scaled stored B and W, X refined
// against S with X held in double-double, and C·X·Cᵀ accumulated in
// double-double.
func covarRefined(sys *System, W *mat.Dense) (*covarRefinedResult, error) {
	n, m, p := sys.Dims()
	Bs, eB := pow2Scaled(sys.B)
	Cs, eC := pow2Scaled(sys.C)
	Ws, eW := pow2Scaled(W)
	a := make([]float64, n*n)
	b := make([]float64, n*m)
	c := make([]float64, p*n)
	w := make([]float64, m*m)
	copyStrided(a, n, sys.A.RawMatrix().Data, sys.A.RawMatrix().Stride, n, n)
	copyStrided(b, m, Bs.RawMatrix().Data, Bs.RawMatrix().Stride, n, m)
	copyStrided(c, n, Cs.RawMatrix().Data, Cs.RawMatrix().Stride, p, n)
	copyStrided(w, m, Ws.RawMatrix().Data, Ws.RawMatrix().Stride, m, m)
	q, rho := covarNoiseDD(b, w, n, m)
	discrete := !sys.IsContinuous()
	s, err := newSchurLyapunov(mat.NewDense(n, n, append([]float64(nil), a...)), n, discrete, &LyapunovWorkspace{})
	if err != nil {
		return nil, err
	}
	qf := make([]float64, n*n)
	for i, v := range q {
		qf[i] = v.hi + v.lo
	}
	xf := make([]float64, n*n)
	if err := s.solve(qf, n, xf); err != nil {
		return nil, err
	}
	x := make([]dd, n*n)
	for i := range n {
		for j := range n {
			x[i*n+j] = dd{0.5 * (xf[i*n+j] + xf[j*n+i]), 0}
		}
	}
	r := make([]float64, n*n)
	d := make([]float64, n*n)
	hi := make([]float64, n*n)
	phase := func(steps int, done, accept float64) (float64, error) {
		corr := math.Inf(1)
		for range steps {
			covarResidualDD(a, x, q, n, discrete, r)
			if err := s.solve(r, n, d); err != nil {
				return 0, err
			}
			for i := range x {
				hi[i] = x[i].hi
			}
			dn, xn := scaledNorm2(d), scaledNorm2(hi)
			for i := range n {
				for j := i; j < n; j++ {
					v := x[i*n+j].add(dd{0.5 * (d[i*n+j] + d[j*n+i]), 0})
					x[i*n+j], x[j*n+i] = v, v
				}
			}
			next := 0.0
			if dn != 0 {
				next = dn / xn
			}
			if next <= done {
				return next, nil
			}
			if !(next < corr/2) {
				corr = next
				break
			}
			corr = next
		}
		if !(corr <= accept) {
			return 0, fmt.Errorf("Covar: state covariance refinement stalled at %.3g: %w", corr, ErrSingularEquation)
		}
		return corr, nil
	}
	res := &covarRefinedResult{rho: rho}
	if res.corr1, err = phase(gramianRefineSteps, gramianRefineDone, gramianRefineAccept); err != nil {
		return nil, err
	}
	if res.delta, err = phase(gramianDDSteps, gramianDDDone, gramianDDAccept); err != nil {
		return nil, err
	}
	cx := make([]dd, p*n)
	for i := range p {
		for k := range n {
			var t dd
			for l := range n {
				t = t.add(x[l*n+k].mul(dd{c[i*n+l], 0}))
			}
			cx[i*n+k] = t
		}
	}
	res.e = 2*eB + eW + 2*eC
	res.p = make([]dd, p*p)
	for i := range p {
		for j := i; j < p; j++ {
			var t dd
			for k := range n {
				t = t.add(cx[i*n+k].mul(dd{c[j*n+k], 0}))
				t = t.add(cx[j*n+k].mul(dd{c[i*n+k], 0}))
			}
			res.p[i*p+j] = dd{0.5 * t.hi, 0.5 * t.lo}
			res.p[j*p+i] = res.p[i*p+j]
		}
	}
	for i := range x {
		hi[i] = x[i].hi
	}
	cn := scaledNorm2(c)
	res.terms = math.Ldexp(cn*cn*scaledNorm2(hi), res.e)
	return res, nil
}

// covarNoiseDD returns S = ½(B·W·Bᵀ + B·Wᵀ·Bᵀ) in double-double (B·W from
// exact two-products) and ρ = ‖½(|B|·|W|·|B|ᵀ + its transpose)‖_F/‖S‖_F,
// +Inf when S is zero.
func covarNoiseDD(b, w []float64, n, m int) (q []dd, rho float64) {
	bw := make([]dd, n*m)
	abw := make([]float64, n*m)
	for i := range n {
		for l := range m {
			var s dd
			a := 0.0
			for k := range m {
				s = s.add(ddProd(b[i*m+k], w[k*m+l]))
				a += math.Abs(b[i*m+k] * w[k*m+l])
			}
			bw[i*m+l] = s
			abw[i*m+l] = a
		}
	}
	q = make([]dd, n*n)
	qn := make([]float64, 0, n*n)
	an := make([]float64, 0, n*n)
	for i := range n {
		for j := i; j < n; j++ {
			var s dd
			a := 0.0
			for l := range m {
				s = s.add(bw[i*m+l].mul(dd{b[j*m+l], 0}))
				s = s.add(bw[j*m+l].mul(dd{b[i*m+l], 0}))
				a += abw[i*m+l]*math.Abs(b[j*m+l]) + abw[j*m+l]*math.Abs(b[i*m+l])
			}
			s = dd{0.5 * s.hi, 0.5 * s.lo}
			a *= 0.5
			q[i*n+j], q[j*n+i] = s, s
			qn = append(qn, s.hi)
			an = append(an, a)
			if i != j {
				qn = append(qn, s.hi)
				an = append(an, a)
			}
		}
	}
	nq := scaledNorm2(qn)
	if nq == 0 {
		return q, math.Inf(1)
	}
	return q, scaledNorm2(an) / nq
}

// covarResidualDD writes into r the Lyapunov residual A·X + X·Aᵀ + S
// (continuous) or A·X·Aᵀ − X + S (discrete) of the double-double X and S,
// accumulated in double-double and rounded once.
func covarResidualDD(a []float64, x, q []dd, n int, discrete bool, r []float64) {
	var ax []dd
	if discrete {
		ax = make([]dd, n*n)
		for i := range n {
			for j := range n {
				var s dd
				for l := range n {
					s = s.add(ddProd(a[i*n+l], x[l*n+j].hi))
					s = s.add(ddProd(a[i*n+l], x[l*n+j].lo))
				}
				ax[i*n+j] = s
			}
		}
	}
	for i := range n {
		for j := i; j < n; j++ {
			var s dd
			if discrete {
				for l := range n {
					s = s.add(ax[i*n+l].mul(dd{a[j*n+l], 0}))
				}
				s = s.add(x[i*n+j].neg())
			} else {
				for l := range n {
					s = s.add(ddProd(a[i*n+l], x[l*n+j].hi))
					s = s.add(ddProd(a[i*n+l], x[l*n+j].lo))
					s = s.add(ddProd(x[i*n+l].hi, a[j*n+l]))
					s = s.add(ddProd(x[i*n+l].lo, a[j*n+l]))
				}
			}
			s = s.add(q[i*n+j])
			v := s.hi + s.lo
			r[i*n+j], r[j*n+i] = v, v
		}
	}
}

// covarZeroNoiseTransfer decides C·Aᵏ·B·W = 0, k = 0..n−1, on the stored
// data through the realization ([A B; 0 0], [0; W], [C 0]), whose Markov
// parameters are C·Aᵏ⁻¹·B·W; the cap applies to the model order n.
func covarZeroNoiseTransfer(sys *System, W *mat.Dense) exactZeroVerdict {
	n, m, p := sys.Dims()
	A := mat.NewDense(n+m, n+m, nil)
	B := mat.NewDense(n+m, m, nil)
	C := mat.NewDense(p, n+m, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, sys.A.At(i, j))
		}
		for j := range m {
			A.Set(i, n+j, sys.B.At(i, j))
		}
		for j := range p {
			C.Set(j, i, sys.C.At(j, i))
		}
	}
	for i := range m {
		for j := range m {
			B.Set(n+i, j, W.At(i, j))
		}
	}
	return exactZeroTransferSteps(A, B, C, n, n+1)
}

// covarUnverified wraps an uncertified state covariance for Covar's result.
func covarUnverified(unv error) error {
	if unv == nil {
		return nil
	}
	return fmt.Errorf("Covar: %w", unv)
}

// covarScaleExponentLimit bounds the binary exponents of max|B|, max|C| and
// max|W| within which Covar skips power-of-two scaling: Q = B·W·Bᵀ and
// C·X·Cᵀ then stay far from overflow and underflow, and an X large enough to
// overflow needs an A so ill-conditioned that the Lyapunov solve rejects it.
const covarScaleExponentLimit = 64

func covarNeedsScaling(B, C, W *mat.Dense) bool {
	for _, m := range [3]*mat.Dense{B, C, W} {
		amax := maxAbsDense(m)
		if amax == 0 || math.IsInf(amax, 0) {
			continue
		}
		if _, e := math.Frexp(amax); e > covarScaleExponentLimit || e < -covarScaleExponentLimit {
			return true
		}
	}
	return false
}

// covarStateTerm sets P to C·X·Cᵀ, X the state covariance under B·W·Bᵀ. With
// scale, B, C and W are first scaled by powers of two, which is exact, so
// that the Lyapunov solve neither overflows nor underflows spuriously. An
// uncertified X (ErrAccuracyUnverified) still sets P and is returned, with
// the quantities of the tier-1 gate.
func covarStateTerm(P *mat.Dense, sys *System, W *mat.Dense, scale bool) (st covarState, err error) {
	n, m, p := sys.Dims()
	B, C, Ws := sys.B, sys.C, W
	var eB, eC, eW int
	if scale {
		B, eB = pow2Scaled(sys.B)
		C, eC = pow2Scaled(sys.C)
		Ws, eW = pow2Scaled(W)
	}
	Q := inputNoiseIntensity(B, Ws, n, m)

	// symmetrize Q for Lyapunov solver
	qRaw := Q.RawMatrix()
	for i := range n {
		for j := i + 1; j < n; j++ {
			avg := (qRaw.Data[i*qRaw.Stride+j] + qRaw.Data[j*qRaw.Stride+i]) / 2
			qRaw.Data[i*qRaw.Stride+j] = avg
			qRaw.Data[j*qRaw.Stride+i] = avg
		}
	}

	st.B, st.W, st.n, st.m = B, Ws, n, m
	st.s = mat.Norm(Q, 2)
	var X *mat.Dense
	if sys.IsContinuous() {
		X, err = lyapChecked(sys.A, Q, nil)
	} else {
		X, err = dlyapChecked(sys.A, Q, nil)
	}
	if X == nil {
		return st, err
	}

	var CX mat.Dense
	CX.Mul(C, X)
	P.Mul(&CX, C.T())
	cn := mat.Norm(C, 2)
	st.terms = cn * cn * mat.Norm(X, 2)
	if e := 2*eB + eW + 2*eC; e != 0 {
		st.terms = math.Ldexp(st.terms, e)
		pRaw := P.RawMatrix()
		for i := range p {
			row := pRaw.Data[i*pRaw.Stride : i*pRaw.Stride+p]
			for j, v := range row {
				row[j] = math.Ldexp(v, e)
			}
		}
	}
	return st, err
}

// covarState holds the tier-1 gate quantities of covarStateTerm: terms =
// ‖C‖_F²·‖X‖_F and the (scaled) B, W and ‖S‖_F of its right-hand side S.
type covarState struct {
	terms float64
	B, W  *mat.Dense
	n, m  int
	s     float64
}

// rhoBound returns ‖B‖_F²·‖W‖_F/‖S‖_F ≥ ρ, +Inf when S is zero.
func (st covarState) rhoBound() float64 {
	if !(st.s > 0) {
		return math.Inf(1)
	}
	bn := mat.Norm(st.B, 2)
	return bn * bn * mat.Norm(st.W, 2) / st.s
}

// rho returns ρ = ‖½(|B|·|W|·|B|ᵀ + its transpose)‖_F/‖S‖_F, +Inf when S
// is zero.
func (st covarState) rho() float64 {
	if !(st.s > 0) {
		return math.Inf(1)
	}
	return covarAbsNoise(st.B, st.W, st.n, st.m) / st.s
}

// covarAbsNoise returns ‖½(|B|·|W|·|B|ᵀ + its transpose)‖_F.
func covarAbsNoise(B, W *mat.Dense, n, m int) float64 {
	abw := make([]float64, n*m)
	for i := range n {
		for l := range m {
			a := 0.0
			for k := range m {
				a += math.Abs(B.At(i, k) * W.At(k, l))
			}
			abw[i*m+l] = a
		}
	}
	an := make([]float64, n*n)
	for i := range n {
		for j := range n {
			a := 0.0
			for l := range m {
				a += abw[i*m+l]*math.Abs(B.At(j, l)) + abw[j*m+l]*math.Abs(B.At(i, l))
			}
			an[i*n+j] = 0.5 * a
		}
	}
	return scaledNorm2(an)
}

// covarContinuousDelayed evaluates P_ij = Σ_kl W_kl ∫ g_ik g_jl for a
// continuous model with external delays τ_ik. Inputs sharing a delay column
// share one copy of the state, so the state cross-covariances
// Y_GH = Σ_{k∈G,l∈H} W_kl ∫ e^{At} b_k b_lᵀ e^{Aᵀt} dt come from a single
// Lyapunov solve on blkdiag(A,…,A). An entry is infinite when its grouped
// impulse sum is nonzero, decided exactly where float64 cannot (as
// covarFeedthrough.sign). dt holds the tier-1 gate: N_d = max(n·ng², ⌈m/16⌉)
// and terms_d = μ·‖C‖_F·(ρ_d·‖C‖_F·‖Y‖_F + χ·‖B‖_F·‖D‖_F·‖W‖_F), μ the
// largest √(‖E‖₁·‖E‖_∞) of the e^{A·|δ|}, δ ≠ 0, used (at least 1), ρ_d that
// of the augmented right-hand side and χ = 1 when a D cross term was added.
func covarContinuousDelayed(sys *System, W *mat.Dense) (P *mat.Dense, dt covarDelayedTerms, err error) {
	n, m, p := sys.Dims()
	tau := effectiveIODelayData(sys, p, m, true)

	group := make([]int, m)
	var reps []int
	for k := range m {
		group[k] = -1
		for g, r := range reps {
			same := true
			for i := range p {
				if !sameDelay(tau[i*m+k], tau[i*m+r]) {
					same = false
					break
				}
			}
			if same {
				group[k] = g
				break
			}
		}
		if group[k] < 0 {
			group[k] = len(reps)
			reps = append(reps, k)
		}
	}
	ng := len(reps)

	dt.N = max(n*ng*ng, (m+15)/16)
	mu, chi := 1.0, 0.0
	expCache := map[float64]*mat.Dense{}
	expA := func(t float64) *mat.Dense {
		if e, ok := expCache[t]; ok {
			return e
		}
		var at, e mat.Dense
		at.Scale(t, sys.A)
		expGuarded(&e, &at)
		expCache[t] = &e
		if t != 0 {
			mu = math.Max(mu, math.Sqrt(mat.Norm(&e, 1)*mat.Norm(&e, math.Inf(1))))
		}
		return &e
	}

	P = mat.NewDense(p, p, nil)
	var unv error
	if n > 0 {
		N := n * ng
		aAug := mat.NewDense(N, N, nil)
		bAug := mat.NewDense(N, m, nil)
		for g := range ng {
			aAug.Slice(g*n, (g+1)*n, g*n, (g+1)*n).(*mat.Dense).Copy(sys.A)
		}
		for k := range m {
			for r := range n {
				bAug.Set(group[k]*n+r, k, sys.B.At(r, k))
			}
		}
		Q := inputNoiseIntensity(bAug, W, N, m)
		symmetrize(Q.RawMatrix().Data, N, N)
		rhoD := math.Inf(1)
		if qn := mat.Norm(Q, 2); qn > 0 {
			rhoD = covarAbsNoise(bAug, W, N, m) / qn
		}
		Y, err := lyapChecked(aAug, Q, nil)
		if err, unv = unverifiedLyapunov(err); err != nil {
			return nil, dt, err
		}
		cn := mat.Norm(sys.C, 2)
		defer func() {
			dt.terms = mu * cn * (rhoD*cn*mat.Norm(Y, 2) + chi*mat.Norm(sys.B, 2)*mat.Norm(sys.D, 2)*mat.Norm(W, 2))
		}()

		ci := mat.NewDense(1, n, nil)
		cj := mat.NewDense(1, n, nil)
		var tmp, row, v mat.Dense
		for i := range p {
			ci.Copy(sys.C.Slice(i, i+1, 0, n))
			for j := range p {
				cj.Copy(sys.C.Slice(j, j+1, 0, n))
				sum := 0.0
				for g, kg := range reps {
					for h, lh := range reps {
						ygh := Y.Slice(g*n, (g+1)*n, h*n, (h+1)*n)
						delta := tau[i*m+kg] - tau[j*m+lh]
						if sameDelay(tau[i*m+kg], tau[j*m+lh]) {
							delta = 0
						}
						if delta >= 0 {
							tmp.Mul(ci, ygh)
							row.Mul(&tmp, expA(delta).T())
						} else {
							tmp.Mul(ci, expA(-delta))
							row.Mul(&tmp, ygh)
						}
						v.Mul(&row, cj.T())
						sum += v.At(0, 0)
					}
				}
				for k := range m {
					for l := range m {
						wkl := W.At(k, l)
						if wkl == 0 {
							continue
						}
						delta := tau[i*m+k] - tau[j*m+l]
						switch {
						case sameDelay(tau[i*m+k], tau[j*m+l]):
						case delta > 0 && sys.D.At(i, k) != 0:
							row.Mul(cj, expA(delta))
							sum += sys.D.At(i, k) * wkl * dotCol(&row, sys.B, l)
							chi = 1
						case delta < 0 && sys.D.At(j, l) != 0:
							row.Mul(ci, expA(-delta))
							sum += sys.D.At(j, l) * wkl * dotCol(&row, sys.B, k)
							chi = 1
						}
					}
				}
				P.Set(i, j, sum)
			}
		}
	}

	for i := range p {
		for j := range p {
			impulse, bound := 0.0, 0.0
			exact := new(big.Rat)
			for k := range m {
				for l := range m {
					if sameDelay(tau[i*m+k], tau[j*m+l]) {
						v := sys.D.At(i, k) * W.At(k, l) * sys.D.At(j, l)
						impulse += v
						bound += math.Abs(v)
					}
				}
			}
			if bound > 0 && math.Abs(impulse) <= hsvTier1K*float64(dt.N)*eps()*bound {
				for k := range m {
					for l := range m {
						if sameDelay(tau[i*m+k], tau[j*m+l]) {
							t := new(big.Rat).SetFloat64(sys.D.At(i, k))
							t.Mul(t, new(big.Rat).SetFloat64(W.At(k, l)))
							t.Mul(t, new(big.Rat).SetFloat64(sys.D.At(j, l)))
							exact.Add(exact, t)
						}
					}
				}
				impulse = float64(exact.Sign())
			}
			if err := finishCovarEntry(P, i, j, impulse); err != nil {
				return nil, dt, err
			}
		}
	}
	return P, dt, covarUnverified(unv)
}

// covarDelayedTerms is the tier-1 gate of the delayed path: count N and
// terms.
type covarDelayedTerms struct {
	N     int
	terms float64
}

// finishCovarEntry makes P[i,j] infinite when feedthrough noise reaches it, and
// otherwise rejects a non-finite state part.
func finishCovarEntry(P *mat.Dense, i, j int, feedthrough float64) error {
	if feedthrough != 0 {
		P.Set(i, j, math.Inf(int(math.Copysign(1, feedthrough))))
		return nil
	}
	if v := P.At(i, j); !isFinite(v) {
		return fmt.Errorf("Covar: P[%d,%d] is %g: %w", i, j, v, ErrOverflow)
	}
	return nil
}

func dotCol(row, B *mat.Dense, col int) float64 {
	_, n := row.Dims()
	s := 0.0
	for r := range n {
		s += row.At(0, r) * B.At(r, col)
	}
	return s
}

// sameDelay treats delays assembled from different input/output/IO sums as
// equal when they differ only by rounding.
func sameDelay(a, b float64) bool {
	return math.Abs(a-b) <= 1e-12*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
