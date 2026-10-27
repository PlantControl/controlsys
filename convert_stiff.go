package controlsys

import (
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// holdLeaf discretizes one mode cluster (a, b) over dt: e^{a·dt} and the
// input integrals of the hold. fast reports |λ|·dt ≥ 1/qzLargeRatio for
// every eigenvalue λ of a. b has no columns when the hold needs e^{a·dt}
// only.
type holdLeaf func(a, b *mat.Dense, dt float64, fast bool) (ad *mat.Dense, gammas []*mat.Dense, ok bool)

// stiffModalHold is the hold discretization of (A, B) by its mode
// clusters: the real Schur form of A is split, by the ratio of
// stiffExplicitParts, into modes with |λ| < qzLargeRatio·ρ(A) and the rest,
// recursively, the clusters are decoupled by a Sylvester equation and each
// is discretized by leaf. Ad and the integrals are returned in the
// coordinates of A. One exponential of A·dt with ‖A·dt‖ ≫ 1 rounds every
// mode to ε·‖A·dt‖ in its squarings, which swamps the slow ones beside a
// stiff one (|λ|·dt ~ 1e13: 1e-3). ok is false when ‖A‖₁·dt or ρ(A)·dt is
// below 1/qzLargeRatio, so callers keep their single exponential, or when
// the split fails.
func stiffModalHold(a, b *mat.Dense, dt float64, leaf holdLeaf) (ad *mat.Dense, gammas []*mat.Dense, ok bool) {
	n, _ := a.Dims()
	if n == 0 || !(mat.Norm(a, 1)*dt*qzLargeRatio >= 1) {
		return nil, nil, false
	}
	return splitModalHold(a, b, dt, leaf, true)
}

func splitModalHold(a, b *mat.Dense, dt float64, leaf holdLeaf, root bool) (*mat.Dense, []*mat.Dense, bool) {
	n, _ := a.Dims()
	_, m := b.Dims()
	if n == 1 {
		fast := math.Abs(a.At(0, 0))*dt*qzLargeRatio >= 1
		if root && !fast {
			return nil, nil, false
		}
		return leaf(a, b, dt, fast)
	}
	slowLeaf := func() (*mat.Dense, []*mat.Dense, bool) {
		if root {
			return nil, nil, false
		}
		return leaf(a, b, dt, false)
	}
	t, z, ok := realSchur(a)
	if !ok {
		return slowLeaf()
	}
	rho := 0.0
	for _, l := range schurEigenvaluesRaw(t, n) {
		rho = math.Max(rho, cmplx.Abs(l))
	}
	if !(rho*dt*qzLargeRatio >= 1) {
		return slowLeaf()
	}
	n1, err := orderSchurGroupFirst(t, z, n, func(l complex128) bool { return cmplx.Abs(l) < qzLargeRatio*rho })
	if err != nil {
		return slowLeaf()
	}
	if n1 == 0 {
		return leaf(a, b, dt, true)
	}
	x, err := separateSchurBlocks(t, n, n1, "stiff hold")
	if err != nil {
		return slowLeaf()
	}
	n2 := n - n1
	zm := mat.NewDense(n, n, z)
	xm := mat.NewDense(n1, n2, x)
	z1 := zm.Slice(0, n, 0, n1)
	z2 := zm.Slice(0, n, n1, n)
	v1 := mat.DenseCopyOf(z1)
	v2 := mat.DenseCopyOf(z2)
	v2.Mul(z1, xm)
	v2.Add(v2, z2)
	p1 := mat.DenseCopyOf(z1.T())
	var xz mat.Dense
	xz.Mul(xm, z2.T())
	p1.Sub(p1, &xz)
	p2 := mat.DenseCopyOf(z2.T())
	b1, b2 := newDense(n1, m), newDense(n2, m)
	if m > 0 {
		b1.Mul(p1, b)
		b2.Mul(p2, b)
	}
	ad1, g1, ok := splitModalHold(extractModalBlock(t, n, 0, n1, 0, n1), b1, dt, leaf, false)
	if !ok {
		return slowLeaf()
	}
	ad2, g2, ok := splitModalHold(extractModalBlock(t, n, n1, n, n1, n), b2, dt, leaf, false)
	if !ok {
		return slowLeaf()
	}
	ad := mat.NewDense(n, n, nil)
	var vp, tmp mat.Dense
	vp.Mul(v1, ad1)
	ad.Mul(&vp, p1)
	vp.Reset()
	vp.Mul(v2, ad2)
	tmp.Mul(&vp, p2)
	ad.Add(ad, &tmp)
	gammas := make([]*mat.Dense, len(g1))
	for k := range g1 {
		g := newDense(n, m)
		if m > 0 {
			var h mat.Dense
			g.Mul(v1, g1[k])
			h.Mul(v2, g2[k])
			g.Add(g, &h)
		}
		gammas[k] = g
	}
	return ad, gammas, true
}

// realSchur returns the real Schur form a = z·t·zᵀ, row-major.
func realSchur(a *mat.Dense) (t, z []float64, ok bool) {
	n, _ := a.Dims()
	t = make([]float64, n*n)
	ar := a.RawMatrix()
	copyStrided(t, n, ar.Data, ar.Stride, n, n)
	z = make([]float64, n*n)
	wr, wi := make([]float64, n), make([]float64, n)
	bwork := make([]bool, n)
	query := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, query, -1, bwork)
	work := make([]float64, int(query[0]))
	_, ok = impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, work, len(work), bwork)
	return t, z, ok
}

func expmScaled(a *mat.Dense, dt float64) *mat.Dense {
	n, _ := a.Dims()
	scaled := mat.NewDense(n, n, nil)
	scaled.Scale(dt, a)
	var e mat.Dense
	e.Exp(scaled)
	return mat.DenseCopyOf(&e)
}

// fastInputIntegral returns ∫₀^dt e^{a·s} ds·b = a⁻¹(ad − I)·b for a fast
// cluster (|λ|·dt ≥ 1/qzLargeRatio, |λ| within 1/qzLargeRatio of ρ(a)),
// where a is well conditioned and ad − I does not cancel.
func fastInputIntegral(a, ad, b *mat.Dense) (*mat.Dense, bool) {
	n, _ := a.Dims()
	_, m := b.Dims()
	if m == 0 {
		return newDense(n, 0), true
	}
	var lu mat.LU
	lu.Factorize(a)
	if luNearSingular(&lu) {
		return nil, false
	}
	gap := mat.DenseCopyOf(ad)
	for i := range n {
		gap.Set(i, i, gap.At(i, i)-1)
	}
	rhs := mat.NewDense(n, m, nil)
	rhs.Mul(gap, b)
	out := mat.NewDense(n, m, nil)
	if err := lu.SolveTo(out, false, rhs); err != nil {
		return nil, false
	}
	return out, true
}

// zohHoldLeaf returns (e^{a·dt}, [Γ₀]), Γ₀ = ∫₀^dt e^{a·s} ds·b: closed
// form for one state, a⁻¹(e^{a·dt} − I)·b for a fast cluster, else the
// augmented exponential of discretizeZOH.
func zohHoldLeaf(a, b *mat.Dense, dt float64, fast bool) (*mat.Dense, []*mat.Dense, bool) {
	n, _ := a.Dims()
	_, m := b.Dims()
	if n == 1 && a.At(0, 0) != 0 {
		x := a.At(0, 0)
		g := newDense(1, m)
		if m > 0 {
			g.Scale(math.Expm1(x*dt)/x, b)
		}
		return mat.NewDense(1, 1, []float64{math.Exp(x * dt)}), []*mat.Dense{g}, true
	}
	if fast {
		ad := expmScaled(a, dt)
		if g, ok := fastInputIntegral(a, ad, b); ok {
			return ad, []*mat.Dense{g}, true
		}
	}
	if m == 0 {
		return expmScaled(a, dt), []*mat.Dense{newDense(n, 0)}, true
	}
	ad, g := zohAugmented(a, b, dt)
	return ad, []*mat.Dense{g}, true
}

// fohHoldLeaf returns (e^{a·dt}, [Γ₀, Γ₁]) of fohInputIntegrals; for a
// fast cluster Γ₁ = a⁻¹(Γ₀/dt − b).
func fohHoldLeaf(a, b *mat.Dense, dt float64, fast bool) (*mat.Dense, []*mat.Dense, bool) {
	n, _ := a.Dims()
	_, m := b.Dims()
	if fast && m > 0 {
		ad, gs, _ := zohHoldLeaf(a, b, dt, true)
		g0 := gs[0]
		var lu mat.LU
		lu.Factorize(a)
		if !luNearSingular(&lu) {
			rhs := mat.NewDense(n, m, nil)
			rhs.Scale(1/dt, g0)
			rhs.Sub(rhs, b)
			g1 := mat.NewDense(n, m, nil)
			if err := lu.SolveTo(g1, false, rhs); err == nil {
				return ad, []*mat.Dense{g0, g1}, true
			}
		}
	}
	ad, g0, g1 := fohAugmented(a, b, dt)
	return ad, []*mat.Dense{g0, g1}, true
}

// expHoldLeaf returns e^{a·dt} alone.
func expHoldLeaf(a, _ *mat.Dense, dt float64, _ bool) (*mat.Dense, []*mat.Dense, bool) {
	if n, _ := a.Dims(); n == 1 {
		return mat.NewDense(1, 1, []float64{math.Exp(a.At(0, 0) * dt)}), nil, true
	}
	return expmScaled(a, dt), nil, true
}

// stiffExp is e^{a·dt} by stiffModalHold; ok is false when a·dt is not
// stiff.
func stiffExp(a *mat.Dense, dt float64) (*mat.Dense, bool) {
	n, _ := a.Dims()
	ad, _, ok := stiffModalHold(a, newDense(n, 0), dt, expHoldLeaf)
	return ad, ok
}
