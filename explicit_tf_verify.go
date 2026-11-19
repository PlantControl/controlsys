package controlsys

import (
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// explicitTFVerified reports whether tf, the transfer function of the
// explicit model sys (route tfRouteExplicit) or of the E⁻¹ fold of the
// descriptor sys (tfRouteFold), is within terms of the stored data, by
// descriptorTFVerified's contract, points and verdict, sys taken as the
// descriptor (A, B, C, D, I) when explicit. Each point is first tried by
// the fast tier (explicitFastTier) on the explicit model, which only
// verifies; a point it leaves undecided goes to the pencil judge
// (judgeDescriptorCached), which alone refutes, the points after a refuted
// one in a batch left unjudged. The fold's model is
// (E⁻¹A, E⁻¹B, C, D) as ToExplicit forms it, and the fast tier runs on it
// only when foldWithinRounding charges its whole residual to one
// ε-rounding of A and B; otherwise every point goes to the pencil judge.
// A model whose B or C is zero has the response D everywhere, and tf is
// verified at once when it is D exactly (staticTFExact).
func explicitTFVerified(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) bool {
	if n, m, p := sys.Dims(); n == 0 || m == 0 || p == 0 {
		return true
	}
	if (allZeroDense(sys.B) || allZeroDense(sys.C)) && staticTFExact(sys, tf) {
		return true
	}
	if !tfVerdictMemo {
		v, _, _ := explicitRouteJudgeWhy(sys, route, tf, c)
		return v
	}
	keySys := sys
	if route == tfRouteExplicit {
		keySys = identityPencil(sys)
	}
	key := tfVerdictKey(keySys, tf)
	tfVerdicts.Lock()
	v, ok := tfVerdicts.m[key]
	tfVerdicts.Unlock()
	if ok {
		return v
	}
	v, _, _ = explicitRouteJudgeWhy(sys, route, tf, c)
	tfVerdicts.Lock()
	if tfVerdicts.m == nil || len(tfVerdicts.m) >= tfVerdictMemoSize {
		tfVerdicts.m = make(map[string]bool)
	}
	tfVerdicts.m[key] = v
	tfVerdicts.Unlock()
	return v
}

// explicitRouteJudgeWhy is explicitTFVerified unmemoized, also saying why
// tf is not verified (descriptorTFJudgeWhy), prefixed "pencil: " where every
// point of the fold goes to the pencil judge and "fold residual: " where
// the fold's residual is carried into the fast tier's bound (foldBound).
func explicitRouteJudgeWhy(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) (bool, []int, string) {
	v, ref, why, _ := explicitRouteJudge(sys, route, tf, c)
	return v, ref, why
}

// explicitRouteJudge is explicitRouteJudgeWhy also counting the points the
// pencil judge took, −1 where it took every point.
func explicitRouteJudge(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) (bool, []int, string, int) {
	if route == tfRouteExplicit {
		return explicitTFJudge(identityPencil(sys), sys, nil, tf, nil, true)
	}
	ex, k, err := sys.explicitForm("TransferFunction")
	if err == nil {
		fb, gate := newFoldBound(sys, ex, k)
		switch {
		case gate:
			return explicitTFJudge(sys, ex, nil, tf, c, false)
		case fb != nil:
			v, ref, why, np := explicitTFJudge(sys, ex, fb, tf, c, false)
			if why != "" {
				why = "fold residual: " + why
			}
			return v, ref, why, np
		}
	}
	v, ref, why := descriptorTFJudgeWhy(sys, tf, c)
	return v, ref, "pencil: " + why, -1
}

// explicitTFJudge is descriptorTFJudgeWhy of tf against the descriptor
// pencil with each point tried first by the fast tier on the explicit
// model, whose response is pencil's, that of one ε-rounding of its A and
// B (foldWithinRounding), or, with fb, the fold whose residual fb carries;
// identity marks pencil as (A, B, C, D, I) of model, whose point at ∞
// identityInfinityRefutes judges. It also counts the points the pencil
// judge took.
func explicitTFJudge(pencil, model *System, fb *foldBound, tf *TransferFunc, c *pencilCache, identity bool) (bool, []int, string, int) {
	rows := newTFRows(tf)
	slow := pencilPointJudge(pencil, tf, rows, c)
	var fast *explicitFastTier
	np := 0
	mk := func(poles []complex128) pointJudge {
		judge := slow(poles)
		var buf []descriptorTFPoint
		return func(pts []complex128) []descriptorTFPoint {
			if fast == nil {
				fast = newExplicitFastTier(model, rows)
				fast.fold = fb
			}
			if cap(buf) < len(pts) {
				buf = make([]descriptorTFPoint, len(pts))
			}
			out := buf[:len(pts)]
			clear(out)
			for k, s := range pts {
				if fast.verify(s) {
					out[k] = descriptorTFPoint{determined: true, verified: true}
					continue
				}
				np++
				if out[k] = judge([]complex128{s})[0]; out[k].refuted {
					break
				}
			}
			return out
		}
	}
	v, ref, why := descriptorScheduleWhy(pencil, tf, mk, func() bool {
		if identity {
			return identityInfinityRefutes(pencil, tf, c)
		}
		return descriptorTFInfinityRefutesCached(pencil, tf, c)
	})
	return v, ref, why, np
}

// identityInfinityRefutes is descriptorTFInfinityRefutesCached for the
// descriptor (A, B, C, D, I) of an explicit model, deciding in floating
// point where the margin allows: t₀ is the quotient of the leading
// coefficients, rounded as the exact series is, and where D = 0 = t₀ the
// first Markov parameter g₁ = CB is compensated against t₁, the next
// quotient. An entry within 1% of its threshold, or one whose order 1 is
// all zero terms, takes the exact judgement.
func identityInfinityRefutes(sys *System, tf *TransferFunc, c *pencilCache) bool {
	n, m, p := sys.Dims()
	if n == 0 || n > exactDetMaxOrder {
		return false
	}
	full := c.infinity(sys)
	if !full.ok[0] || !(full.eta[0] < 1) {
		return false
	}
	if tfPolyDegree(tf) >= 1 {
		return true
	}
	eta := full.eta[0]
	tol := math.Sqrt(eps())
	exact := func() bool { return descriptorTFInfinityRefutesCached(sys, tf, c) }
	for i := range p {
		den := trimLeadingZeros(tf.Den[i])
		for j := range m {
			num := trimLeadingZeros(tf.Num[i][j])
			off := len(den) - len(num)
			t0 := 0.0
			if off == 0 && len(num) > 0 {
				if t0 = num[0] / den[0]; t0 == 0 {
					return exact()
				}
			}
			d := sys.D.At(i, j)
			lhs, rhs := math.Abs(t0-d), tol*math.Max(math.Abs(t0), math.Abs(d))+eps()*math.Abs(d)/(1-eta)
			if !isFinite(lhs) || !isFinite(rhs) {
				return exact()
			}
			if lhs > rhs {
				return true
			}
			if d != 0 || t0 != 0 {
				continue
			}
			t1 := 0.0
			if off == 1 && len(num) > 0 {
				if t1 = num[0] / den[0]; t1 == 0 {
					return exact()
				}
			}
			var g compensatedSum
			T := 0.0
			for l := range n {
				g.addProd(sys.C.At(i, l), sys.B.At(l, j))
				T += math.Abs(sys.C.At(i, l) * sys.B.At(l, j))
			}
			if T == 0 && t1 == 0 {
				return exact()
			}
			gk := g.value()
			gErr := dot2Err(n)*T + eps()*math.Abs(gk)
			diff := math.Abs(t1 - gk)
			thr := tol*math.Max(math.Max(math.Abs(t1), math.Abs(gk)), T) + full.data[i*m+j]
			switch {
			case !isFinite(diff) || !isFinite(thr):
				return exact()
			case diff+gErr <= 0.99*thr:
			case diff-gErr > 1.01*thr:
				return true
			default:
				return exact()
			}
		}
	}
	return false
}

// staticTFExact reports whether every entry of tf is exactly D_ij of sys,
// num = D_ij·den coefficient by coefficient without rounding: the response
// of a model with B or C zero is D at every point, the stored data's terms.
func staticTFExact(sys *System, tf *TransferFunc) bool {
	for i, den := range tf.Den {
		for j, num := range tf.Num[i] {
			d := sys.D.At(i, j)
			if len(num) > len(den) {
				return false
			}
			off := len(den) - len(num)
			for k, dk := range den {
				want := d * dk
				if math.FMA(d, dk, -want) != 0 || !isFinite(want) {
					return false
				}
				got := 0.0
				if k >= off {
					got = num[k-off]
				}
				if got != want {
					return false
				}
			}
		}
	}
	return true
}

// identityPencil is the explicit sys as the descriptor (A, B, C, D, I),
// delays dropped.
func identityPencil(sys *System) *System {
	n, _, _ := sys.Dims()
	return &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D, E: eyeDense(n), Dt: sys.Dt}
}

// foldWithinRounding reports whether ex, the explicit form (E⁻¹A,
// 2ᵏ·E⁻¹B, 2⁻ᵏ·C, D) of the descriptor sys as explicitForm returns it with
// its state scaling k, is exactly the fold of A + ΔA and B + ΔB with |ΔA| ≤
// ε|A| and |ΔB| ≤ ε|B| componentwise (foldResidual).
func foldWithinRounding(sys, ex *System, k int) bool {
	_, _, gate, _ := foldResidual(sys, ex, k)
	return gate
}

// foldResidual bounds the residuals R_A = E·Ā − A and R_B = E·B̄ − 2ᵏ·B of
// ex, the explicit form (Ā, B̄, C̄, D̄) = (E⁻¹A, 2ᵏ·E⁻¹B, 2⁻ᵏ·C, D) of the
// descriptor sys as explicitForm returns it with its state scaling k: ra
// (n×n) and rb (n×m) bound |R_A| and |R_B| componentwise, each formed in
// compensated arithmetic with its evaluation bound counted, each row of E
// and column of the solution scaled by a power of two so that no product
// over- or underflows. gate reports |R_A| ≤ ε|A| and |R_B| ≤ ε|2ᵏB| (a zero
// entry allowing no residual): ex is then exactly the fold of one
// ε-rounding of A and B. ok is false, and gate with it, where a scaling is
// inexact, a bound is not finite, or C̄ and D̄ are not 2⁻ᵏ·C and D exactly.
func foldResidual(sys, ex *System, k int) (ra, rb []float64, gate, ok bool) {
	n, m, p := sys.Dims()
	for i := range p {
		for l := range n {
			if v := sys.C.At(i, l); ex.C.At(i, l) != math.Ldexp(v, -k) || !exactLdexp(v, -k) {
				return nil, nil, false, false
			}
		}
		for j := range m {
			if ex.D.At(i, j) != sys.D.At(i, j) {
				return nil, nil, false, false
			}
		}
	}
	e := denseData(sys.E)
	eExp := make([]int, n)
	es := make([]float64, n*n)
	for i := range n {
		eExp[i] = maxExp(e[i*n : (i+1)*n])
		for l := range n {
			v, ok := scaledExact(e[i*n+l], -eExp[i])
			if !ok {
				return nil, nil, false, false
			}
			es[i*n+l] = v
		}
	}
	gate = true
	resid := func(x, ref []float64, cols, shift int) ([]float64, bool) {
		out := make([]float64, n*cols)
		xs := make([]float64, n)
		col := make([]float64, n)
		for j := range cols {
			for l := range n {
				col[l] = x[l*cols+j]
			}
			xe := maxExp(col)
			for l := range n {
				v, ok := scaledExact(col[l], -xe)
				if !ok {
					return nil, false
				}
				xs[l] = v
			}
			for i := range n {
				a, ok := scaledExact(ref[i*cols+j], shift-eExp[i]-xe)
				if !ok {
					return nil, false
				}
				var r compensatedSum
				mag := math.Abs(a)
				for l := range n {
					r.addProd(es[i*n+l], xs[l])
					mag += math.Abs(es[i*n+l] * xs[l])
				}
				r.add(-a)
				bound := math.Abs(r.value())*(1+2*eps()) + dot2Err(n+1)*mag*(1+float64(n+4)*eps()) + float64(4*n+16)*0x1p-1074
				if !(bound <= eps()*math.Abs(a)*(1-4*eps())) {
					gate = false
				}
				v := math.Ldexp(bound*(1+2*eps()), eExp[i]+xe)
				if v < 0x1p-1000 {
					v += 0x1p-1074
				}
				if !isFinite(v) {
					return nil, false
				}
				out[i*cols+j] = v
			}
		}
		return out, true
	}
	ra, okA := resid(denseData(ex.A), denseData(sys.A), n, 0)
	if !okA {
		return nil, nil, false, false
	}
	rb, okB := resid(denseData(ex.B), denseData(sys.B), m, k)
	if !okB {
		return nil, nil, false, false
	}
	return ra, rb, gate, true
}

// foldBound carries the residual of a fold that is not one ε-rounding of
// A and B (foldResidual's gate fails) into the fast tier's bound on the
// stored solution, so that the fold's points still take the fast tier.
// With F = sE − A, M = (sI − Ā)⁻¹ and z = R_B + R_A·x_f for the fold's
// solution x_f, F = E(sI − Ā) + R_A and F·x = E·B̄ − R_B give
// x − x_f = −M·E⁻¹·(I + R_A·M·E⁻¹)⁻¹·z, so with K ≥ |E⁻¹| and
// v = |M|·K·ζ, ζ ≥ |z|, |x − x_f| ≤ Σₖ Pᵏv ≤ v/(1 − q) where P = |M|·K·|R_A|
// passes the check Pv ≤ q·v, q < 1; |M| is bounded by the comparison
// matrices of the sweep (upper) in its balanced coordinates. K applies
// |Ê⁻¹|(I − |R_E|)⁻¹ for the computed inverse Ê⁻¹ with R_E = I − E·Ê⁻¹
// bounded in compensated arithmetic: with S the row scales of E and ρ =
// ‖S⁻¹|R_E|S‖_∞ < 1/2, Σₖ|R_E|ᵏy ≤ y + ρ/(1 − ρ)·‖S⁻¹y‖_∞·S𝟙. A point
// where a check fails is left to the pencil judge; foldBound never refutes.
type foldBound struct {
	n, m     int
	ra, rb   []float64
	einv     []float64
	sRow     []float64
	rhoE     float64
	zeta, ky []float64
	v, w     []float64
}

// newFoldBound is foldResidual's verdict with, where the gate fails, the
// foldBound of ex; nil where the fold's residual cannot be carried (ok
// false, E's computed inverse not finite or ρ ≥ 1/2).
func newFoldBound(sys, ex *System, k int) (*foldBound, bool) {
	ra, rb, gate, ok := foldResidual(sys, ex, k)
	if gate || !ok {
		return nil, gate
	}
	n, m, _ := sys.Dims()
	var lu mat.LU
	lu.Factorize(sys.E)
	var inv mat.Dense
	if err := lu.SolveTo(&inv, false, eyeDense(n)); err != nil {
		return nil, false
	}
	einv := denseData(&inv)
	e := denseData(sys.E)
	sRow := make([]float64, n)
	for i := range n {
		sRow[i] = math.Ldexp(1, maxExp(e[i*n:(i+1)*n]))
	}
	rhoE := 0.0
	for i := range n {
		row := 0.0
		for c := range n {
			var r compensatedSum
			mag := 0.0
			if i == c {
				r.add(1)
				mag = 1
			}
			for l := range n {
				r.addProd(-e[i*n+l], einv[l*n+c])
				mag += math.Abs(e[i*n+l] * einv[l*n+c])
			}
			b := math.Abs(r.value())*(1+2*eps()) + dot2Err(n+1)*mag*(1+float64(n+4)*eps()) + float64(4*n+16)*0x1p-1074
			row += b * sRow[c] / sRow[i]
		}
		rhoE = math.Max(rhoE, row*(1+float64(n+2)*eps()))
	}
	if !(rhoE < 0.5) {
		return nil, false
	}
	for i, v := range einv {
		einv[i] = math.Abs(v)
		if !isFinite(einv[i]) {
			return nil, false
		}
	}
	return &foldBound{n: n, m: m, ra: ra, rb: rb, einv: einv, sRow: sRow, rhoE: rhoE,
		zeta: make([]float64, n), ky: make([]float64, n), v: make([]float64, n), w: make([]float64, n)}, false
}

// applyK sets fb.ky to an upper bound on |E⁻¹|·y for the nonnegative y.
func (fb *foldBound) applyK(y []float64) bool {
	n := fb.n
	nrm := 0.0
	for i, v := range y {
		nrm = math.Max(nrm, v/fb.sRow[i])
	}
	c := fb.rhoE / (1 - fb.rhoE) * nrm * (1 + 4*eps())
	for i := range n {
		var acc float64
		for l, a := range fb.einv[i*n : (i+1)*n] {
			acc += a * (y[l] + c*fb.sRow[l])
		}
		fb.ky[i] = acc * (1 + float64(2*n+4)*eps())
		if !isFinite(fb.ky[i]) {
			return false
		}
	}
	return true
}

// add adds to f.rho, the fast tier's bound on the error of its solution
// against the fold's in the sweep's balanced coordinates, the bound on the
// fold's distance from the stored solution (see foldBound).
func (fb *foldBound) add(f *explicitFastTier) bool {
	n, m := fb.n, fb.m
	t := f.t
	clear(fb.zeta)
	for l := range n {
		xa := 0.0
		for j := range m {
			x := f.xf[j*n+l]
			if f.pair {
				x += f.xl[j*n+l]
			}
			xa += cabs1(x)
		}
		fb.w[l] = (xa*(1+2*termsEps) + float64(m)*f.rho[l]) / t[l]
	}
	for i := range n {
		var z float64
		for j := range m {
			z += fb.rb[i*m+j]
		}
		for l := range n {
			z += fb.ra[i*n+l] * fb.w[l]
		}
		fb.zeta[i] = z * (1 + float64(n+m+2)*eps())
	}
	if !fb.applyK(fb.zeta) {
		return false
	}
	for i := range n {
		fb.v[i] = t[i] * fb.ky[i]
	}
	if !f.upper(fb.v) {
		return false
	}
	for l := range n {
		fb.w[l] = fb.v[l] / t[l]
	}
	for i := range n {
		var z float64
		for l := range n {
			z += fb.ra[i*n+l] * fb.w[l]
		}
		fb.zeta[i] = z * (1 + float64(n+2)*eps())
	}
	if !fb.applyK(fb.zeta) {
		return false
	}
	pv := fb.w
	for i := range n {
		pv[i] = t[i] * fb.ky[i]
	}
	if !f.upper(pv) {
		return false
	}
	q := 0.0
	for i, u := range pv {
		if u == 0 {
			continue
		}
		if !(fb.v[i] > 0) {
			return false
		}
		q = max(q, u/fb.v[i])
	}
	q *= 1 + 8*float64(n)*termsEps
	if !(q < 1) {
		return false
	}
	g := 1 / (1 - q) * (1 + 2*termsEps)
	for i := range n {
		f.rho[i] += fb.v[i] * g
		if !isFinite(f.rho[i]) {
			return false
		}
	}
	return true
}

// maxExp is the binary exponent of the largest |v|, 0 when all are zero or
// one is not finite.
func maxExp(v []float64) int {
	mx := 0.0
	for _, x := range v {
		mx = math.Max(mx, math.Abs(x))
	}
	if mx == 0 || !isFinite(mx) {
		return 0
	}
	return frexpExp(mx)
}

// scaledExact is v·2ᵏ, false where that is not exact.
func scaledExact(v float64, k int) (float64, bool) {
	r := math.Ldexp(v, k)
	return r, isFinite(r) && exactLdexp(v, k)
}

// explicitFastTier verifies a transfer function against an explicit
// model's response at single points in O(n²) each, after one Hessenberg
// reduction of its balanced realization (hessenbergSweep). At s it takes
// the refined solution x̂ = x + Qy of the sweep, forms the residual r = b −
// (sI − Â)x̂ of every column against the exact point in compensated
// arithmetic with its bound (pencilResidualRow), bounds |M|·Σⱼ|rⱼ|, M =
// (sI − Â)⁻¹, by the comparison matrices of the factors (termsCert step 3),
// so that |G − ĝ| ≤ δ = |ĉᵢ|·|M|Σⱼ|rⱼ| + o for ĝ = ĉᵢx̂ⱼ + d formed in
// compensated arithmetic, o its rounding bound. An entry verifies when
// |ĝ − t| + δ + β ≤ √ε·max(scale, T), t the transfer function's value with
// error bound β (tfRows.bounded), scale the least the row's largest |G| or
// |t| can be and T the least the entry's terms Σ|ĉᵢ||x| + |d| can be
// (|x| ≥ |x̂| − |M|Σⱼ|rⱼ|); a point verifies when all its entries do. It never refutes: a
// point the sweep does not solve (a refinement stall, an underflowing
// state, a singular pivot) or whose bound does not verify is left to the
// pencil judge.
type explicitFastTier struct {
	hs      *hessenbergSweep
	n, m, p int
	rows    tfRows
	dst, xf []complex128
	rho, tw []float64
	r, g    []complex128
	beta    []float64
	delta   []float64
	tlo     []float64
	// t is the balancing: x̂ in the sweep's coordinates is t∘x.
	t []float64
	// xl is the refinement correction kept apart from xf where pair is
	// set, the solution then xf + xl; res is its residual's scratch.
	xl, res []complex128
	pair    bool
	fold    *foldBound
	// cb bounds the inverse for upper: the sweep, or the dense solver bd
	// (built on first use) where the sweep's reduction error leaves the
	// comparison check short.
	cb comparisonBounds
	bd *balancedDense
	rm []complex128
	// post is the sweep's bound with its reduction error measured
	// (sweepPost), formed where the a-priori bound first fails.
	post   *sweepPost
	noPost bool
	y0     []float64
}

func newExplicitFastTier(sys *System, rows tfRows) *explicitFastTier {
	n, m, p := sys.Dims()
	br := newRealizationCopy(sys, n, m, p)
	t := make([]float64, n)
	for i := range t {
		t[i] = 1
	}
	br.balance(t)
	hs := newHessenbergSweepOf(sys, br)
	f := &explicitFastTier{hs: hs, cb: hs, y0: make([]float64, n), n: n, m: m, p: p, rows: rows, t: t,
		xl: make([]complex128, n*m), res: make([]complex128, n*m),
		dst: make([]complex128, p*m), xf: make([]complex128, n*m),
		rho: make([]float64, n), tw: make([]float64, n),
		r: make([]complex128, m), beta: make([]float64, m),
		g: make([]complex128, m), delta: make([]float64, m), tlo: make([]float64, m)}
	if n == 0 || hs.nonFinite {
		return f
	}
	hs.certBuf = make([]float64, n)
	hs.absQT = make([]float64, n*n)
	for i, v := range hs.q {
		hs.absQT[i] = math.Abs(v)
	}
	var fr l2
	for _, v := range hs.a {
		fr.add(v)
	}
	hs.normF = fr.norm()
	return f
}

// verify reports whether every entry of the transfer function verifies at
// s (see explicitFastTier): first on the plain Hessenberg solve, then on
// the sweep's refined one (evalInto).
func (f *explicitFastTier) verify(s complex128) bool {
	n, m, hs := f.n, f.m, f.hs
	if n == 0 || hs.nonFinite || cmplx.IsNaN(s) || cmplx.IsInf(s) {
		return false
	}
	pt := pointAt(s)
	if hs.factor(pt) != nil {
		return false
	}
	y := hs.y
	for i := range n {
		for j := range m {
			y[j*n+i] = complex(hs.bt[i*m+j], 0)
		}
	}
	hs.solve(y)
	qMulCols(f.xf, y, hs.q, n, m)
	f.pair = false
	if f.check(pt, s) {
		return true
	}
	if f.refinePair(pt) && f.pairResidual(pt) && f.entries(s) {
		return true
	}
	f.pair = false
	if err := hs.evalInto(pt, f.dst); err == nil && hs.via == viaSweepSolve {
		qMulCols(f.xf, hs.y, hs.q, n, m)
		for i, v := range hs.x {
			f.xf[i] += v
		}
		if f.check(pt, s) {
			return true
		}
	}
	return f.dense(pt, s)
}

// dense is verify on GEPP of sI − Â (balancedDense), whose comparison
// bound carries no reduction error: the plain solve, then the pair with
// its correction.
func (f *explicitFastTier) dense(pt frequencyPoint, s complex128) bool {
	n, m := f.n, f.m
	if f.bd == nil {
		f.bd = newBalancedDense(f.hs.sys, n, m, f.p)
		f.rm = make([]complex128, n*m)
	}
	bd := f.bd
	if bd.nonFinite || !bd.factorAt(pt) {
		return false
	}
	f.cb = bd
	defer func() { f.cb = f.hs }()
	for i, v := range bd.b[:n*m] {
		f.rm[i] = complex(v, 0)
	}
	cLUSolve(bd.pencil, bd.piv, f.rm, n, m)
	for i := range n {
		for j := range m {
			f.xf[j*n+i] = f.rm[i*m+j]
		}
	}
	f.pair = false
	if f.check(pt, s) {
		return true
	}
	for j := range m {
		x := f.xf[j*n : (j+1)*n]
		for i := range n {
			rc, _ := pencilResidualRow(x, bd.a[i*n:(i+1)*n], i, bd.b[i*m+j], pt.p, pt.q, 1)
			f.rm[i*m+j] = rc
		}
	}
	cLUSolve(bd.pencil, bd.piv, f.rm, n, m)
	for i := range n {
		for j := range m {
			v := f.rm[i*m+j]
			if cmplx.IsNaN(v) || cmplx.IsInf(v) {
				return false
			}
			f.xl[j*n+i] = v
		}
	}
	f.pair = true
	return f.pairResidual(pt) && f.entries(s)
}

// refinePair sets f.xl to the correction (sI − Â)⁻¹r̂ of f.xf through the
// sweep's factors, r̂ the residual of f.xf against the exact point in
// compensated arithmetic, and sets f.pair: the solution is then the
// unrounded pair f.xf + f.xl.
func (f *explicitFastTier) refinePair(pt frequencyPoint) bool {
	n, m, hs := f.n, f.m, f.hs
	for j := range m {
		x := f.xf[j*n : (j+1)*n]
		for i := range n {
			rc, _ := pencilResidualRow(x, hs.a[i*n:(i+1)*n], i, hs.b[i*m+j], pt.p, pt.q, 1)
			f.res[j*n+i] = rc
		}
	}
	qTMulCols(hs.y, f.res, hs.q, n, m)
	hs.solve(hs.y)
	qMulCols(f.xl, hs.y, hs.q, n, m)
	for _, v := range f.xl {
		if cmplx.IsNaN(v) || cmplx.IsInf(v) {
			return false
		}
	}
	f.pair = true
	return true
}

// pairResidual is residual for the pair f.xf + f.xl: each row of
// b − (sI − Â)(x_h + x_l) summed in compensated arithmetic over both words,
// within dot2Err(k) of its terms' magnitudes for its k products and sums.
func (f *explicitFastTier) pairResidual(pt frequencyPoint) bool {
	n, m, hs := f.n, f.m, f.hs
	s := pt.p
	sr, si := real(s), imag(s)
	sa := cabs1(s)
	clear(f.rho)
	for j := range m {
		xh, xl := f.xf[j*n:(j+1)*n], f.xl[j*n:(j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			var re, im compensatedSum
			bi := hs.b[i*m+j]
			re.add(bi)
			mag := math.Abs(bi)
			k := 1
			for l, a := range row {
				if a == 0 {
					continue
				}
				re.addProd(a, real(xh[l]))
				re.addProd(a, real(xl[l]))
				im.addProd(a, imag(xh[l]))
				im.addProd(a, imag(xl[l]))
				mag += math.Abs(a) * (cabs1(xh[l]) + cabs1(xl[l]))
				k += 2
			}
			for _, x := range []complex128{xh[i], xl[i]} {
				re.addProd(-sr, real(x))
				re.addProd(si, imag(x))
				im.addProd(-sr, imag(x))
				im.addProd(-si, real(x))
				mag += sa * cabs1(x)
			}
			k += 4
			e := dot2Err(k)*mag*(1+float64(k+16)*termsEps) + float64(8*k+64)*max(sa, 1)*0x1p-1074
			r := complex(re.value(), im.value())
			f.rho[i] += (termsAbs(r)*(1+2*termsEps) + e) * (1 + 4*termsEps)
		}
	}
	if !f.upper(f.rho) {
		return false
	}
	return f.fold == nil || f.fold.add(f)
}

// check reports whether every entry verifies at s with the solution in
// f.xf and the sweep factored at pt.
func (f *explicitFastTier) check(pt frequencyPoint, s complex128) bool {
	if f.residual(pt, false) && f.entries(s) {
		return true
	}
	return f.residual(pt, true) && f.entries(s)
}

// residual sets f.rho to a bound on |M|·Σⱼ|rⱼ| of f.xf at pt, the residual
// in working precision with its rounding bound (nnzᵢ + 5)(ε/2)Sᵢ, as
// termsCert step 1 forms it, or compensated (pencilResidualRow); false
// where the comparison bound fails.
func (f *explicitFastTier) residual(pt frequencyPoint, compensated bool) bool {
	n, m, hs := f.n, f.m, f.hs
	s := pt.p
	sa := cabs1(s)
	clear(f.rho)
	for j := range m {
		x := f.xf[j*n : (j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			bi := hs.b[i*m+j]
			if compensated {
				rc, e := pencilResidualRow(x, row, i, bi, pt.p, pt.q, 1)
				f.rho[i] += termsAbs(rc)*(1+4*termsEps)*(1+2*termsEps) + e*(1+4*termsEps)
				continue
			}
			var ar, ai, sum float64
			nnz := 0
			for k, a := range row {
				if a == 0 {
					continue
				}
				ar += a * real(x[k])
				ai += a * imag(x[k])
				sum += math.Abs(a) * cabs1(x[k])
				nnz++
			}
			r := complex(bi+ar, ai) - s*x[i]
			bound := float64(nnz+5) * termsEps / 2 * (math.Abs(bi) + sa*cabs1(x[i]) + sum)
			f.rho[i] += (termsAbs(r) + bound) * (1 + 2*termsEps)
		}
	}
	if !f.upper(f.rho) {
		return false
	}
	return f.fold == nil || f.fold.add(f)
}

// entries reports whether every entry verifies at s with f.rho bounding
// the solution's error.
func (f *explicitFastTier) entries(s complex128) bool {
	n, m, p, hs := f.n, f.m, f.p, f.hs
	tol := math.Sqrt(eps())
	for i := range p {
		if !f.rows.bounded(i, s, f.r, f.beta) {
			return false
		}
		cr := hs.c[i*n : (i+1)*n]
		w := 0.0
		for k, c := range cr {
			w += math.Abs(c) * f.rho[k]
		}
		w *= 1 + 2*float64(n+2)*termsEps
		scale := 0.0
		for j := range m {
			x := f.xf[j*n : (j+1)*n]
			var xl []complex128
			if f.pair {
				xl = f.xl[j*n : (j+1)*n]
			}
			var re, im compensatedSum
			mag, tlo := 0.0, 0.0
			for k, c := range cr {
				re.addProd(c, real(x[k]))
				im.addProd(c, imag(x[k]))
				xk := x[k]
				ax := cabs1(x[k])
				if xl != nil {
					re.addProd(c, real(xl[k]))
					im.addProd(c, imag(xl[k]))
					xk += xl[k]
					ax += cabs1(xl[k])
				}
				mag += math.Abs(c) * ax
				tlo += math.Abs(c) * math.Max(termsAbs(xk)*(1-4*termsEps)-f.rho[k], 0)
			}
			d := 0.0
			if hs.d != nil {
				d = hs.d[i*hs.dStride+j]
				re.add(d)
			}
			g := complex(re.value(), im.value())
			o := dot2Err(2*n+2)*(mag+math.Abs(d))*(1+float64(2*n+4)*termsEps) + termsEps*cabs1(g) + float64(8*n+16)*0x1p-1074
			f.g[j], f.delta[j] = g, w+o
			f.tlo[j] = (tlo + math.Abs(d)) * (1 - 2*float64(n+2)*termsEps)
			scale = math.Max(scale, math.Max(cmplx.Abs(f.r[j])-f.beta[j], cmplx.Abs(g)-f.delta[j]))
		}
		for j := range m {
			thr := tol * math.Max(scale, f.tlo[j]) * (1 - 8*termsEps)
			lhs := (cmplx.Abs(f.g[j]-f.r[j])*(1+2*termsEps) + f.delta[j] + f.beta[j]) * (1 + 2*termsEps)
			if !(lhs <= thr) {
				return false
			}
		}
	}
	return true
}

// upper overwrites the nonnegative y with an upper bound on |M|y from the
// comparison matrices of the sweep's factors at its last point
// (termsCert.comparisonUpper), false where the q check fails.
func (f *explicitFastTier) upper(y []float64) bool {
	if f.cb != comparisonBounds(f.hs) {
		return f.upperBy(f.cb, y)
	}
	copy(f.y0, y)
	if f.upperBy(f.hs, y) {
		return true
	}
	if f.post == nil && !f.noPost {
		f.post = newSweepPost(f.hs)
		f.noPost = f.post == nil
	}
	if f.post == nil {
		return false
	}
	copy(y, f.y0)
	return f.upperBy(f.post, y)
}

// upperBy is upper with the inverse bounded by cb.
func (f *explicitFastTier) upperBy(cb comparisonBounds, y []float64) bool {
	cb.nsolve(y)
	cb.factorError(f.tw, y)
	cb.nsolve(f.tw)
	q := 0.0
	for i, v := range y {
		if f.tw[i] == 0 {
			continue
		}
		if !(v > 0) {
			return false
		}
		q = max(q, f.tw[i]/v)
	}
	q *= 1 + 8*float64(f.n)*termsEps
	if !(q < 1) {
		return false
	}
	g := 1 / (1 - q) * (1 + 2*termsEps)
	for i := range y {
		y[i] *= g
		if !isFinite(y[i]) {
			return false
		}
	}
	return true
}

// sweepPost bounds the inverse of sI − Â from the sweep's factors with the
// reduction's error measured a posteriori instead of hessenbergSweep's
// a-priori rank-one (n² + 4n)ε‖Â‖_F term, which beside the poles of a large
// model fails the comparison check. With E₄ = ÂQ − QH and E₃ = I − QᵀQ for
// the computed Q and H, sI − Â = G⁻¹ − W for G = Q·F_H⁻¹·Q⁻¹, F_H = P⁻¹L̂Û
// the factored sI − H = F_H − Δ, |Δ| ≤ γP⁻¹|L̂||Û|, and W = (QΔ + E₄)Q⁻¹;
// so |G|y ≤ |Q|N_H|Q⁻¹|y (nsolve) and |W|y ≤ (|Q|γP⁻¹|L̂||Û| + |E₄|)|Q⁻¹|y
// (factorError) feed comparisonUpper's check. |Q⁻¹|y ≤ |Qᵀ|y +
// ρ₃/(1 − ρ₃)·‖|Qᵀ|y‖_∞·𝟙 for ρ₃ ≥ ‖E₃‖_∞ < 1/2. Ē₄ and ρ₃ are formed
// once in compensated arithmetic with their evaluation bounds.
type sweepPost struct {
	hs    *hessenbergSweep
	absE4 []float64
	c3    float64
	z, t  []float64
}

// newSweepPost is the sweepPost of hs, nil where ρ₃ ≥ 1/2 or a bound is
// not finite.
func newSweepPost(hs *hessenbergSweep) *sweepPost {
	n := hs.n
	q, h, a := hs.q, hs.h, hs.a
	bound := func(r *compensatedSum, mag float64, k int) float64 {
		return math.Abs(r.value())*(1+2*eps()) + dot2Err(k)*mag*(1+float64(k+4)*eps()) + float64(4*k+16)*0x1p-1074
	}
	e4 := make([]float64, n*n)
	for i := range n {
		for j := range n {
			var r compensatedSum
			mag := 0.0
			for l := range n {
				r.addProd(a[i*n+l], q[l*n+j])
				r.addProd(-q[i*n+l], h[l*n+j])
				mag += math.Abs(a[i*n+l]*q[l*n+j]) + math.Abs(q[i*n+l]*h[l*n+j])
			}
			e4[i*n+j] = bound(&r, mag, 2*n)
			if !isFinite(e4[i*n+j]) {
				return nil
			}
		}
	}
	rho3 := 0.0
	for i := range n {
		row := 0.0
		for j := range n {
			var r compensatedSum
			mag := 0.0
			if i == j {
				r.add(1)
				mag = 1
			}
			for l := range n {
				r.addProd(-q[l*n+i], q[l*n+j])
				mag += math.Abs(q[l*n+i] * q[l*n+j])
			}
			row += bound(&r, mag, n+1)
		}
		rho3 = math.Max(rho3, row*(1+float64(n+2)*eps()))
	}
	if !(rho3 < 0.5) {
		return nil
	}
	return &sweepPost{hs: hs, absE4: e4, c3: rho3 / (1 - rho3) * (1 + 4*eps()), z: make([]float64, n), t: make([]float64, n)}
}

// qinv sets sp.z to the bound on |Q⁻¹|y.
func (sp *sweepPost) qinv(y []float64) {
	n, q, z := sp.hs.n, sp.hs.absQT, sp.z
	clear(z)
	for k, yk := range y {
		if yk == 0 {
			continue
		}
		for i, a := range q[k*n : (k+1)*n] {
			z[i] += a * yk
		}
	}
	mx := 0.0
	for i := range z {
		z[i] *= 1 + float64(n+2)*eps()
		mx = max(mx, z[i])
	}
	for i := range z {
		z[i] += sp.c3 * mx
	}
}

// qmul sets dst to |Q|·t, rounded up.
func (sp *sweepPost) qmul(dst, t []float64) {
	n, q := sp.hs.n, sp.hs.absQT
	for i := range n {
		var v float64
		for k, a := range q[i*n : (i+1)*n] {
			v += a * t[k]
		}
		dst[i] = v * (1 + float64(n+2)*eps())
	}
}

func (sp *sweepPost) nsolve(y []float64) {
	hs, n, t := sp.hs, sp.hs.n, sp.t
	sp.qinv(y)
	copy(t, sp.z)
	u, mult := hs.absFactors()
	for k := range n - 1 {
		if hs.swapped[k] {
			t[k], t[k+1] = t[k+1], t[k]
		}
		t[k+1] += mult[k] * t[k]
	}
	for i := n - 1; i >= 0; i-- {
		row := u[i*n : (i+1)*n]
		v := t[i]
		for k := i + 1; k < n; k++ {
			v += row[k] * t[k]
		}
		t[i] = v * row[i] * (1 + float64(n+2)*eps())
	}
	sp.qmul(y, t)
}

func (sp *sweepPost) factorError(dst, y []float64) {
	hs, n, t := sp.hs, sp.hs.n, sp.t
	sp.qinv(y)
	copy(t, sp.z)
	u, mult := hs.absFactors()
	for i := range n {
		row := u[i*n : (i+1)*n]
		v := t[i] * sqrt2 / row[i]
		for k := i + 1; k < n; k++ {
			v += row[k] * t[k]
		}
		t[i] = v
	}
	for k := n - 2; k >= 0; k-- {
		t[k+1] += mult[k] * t[k]
		if hs.swapped[k] {
			t[k], t[k+1] = t[k+1], t[k]
		}
	}
	sp.qmul(dst, t)
	g := float64(2*n+4) * eps() * (1 + float64(2*n+4)*eps())
	for i := range n {
		var v float64
		for k, e := range sp.absE4[i*n : (i+1)*n] {
			v += e * sp.z[k]
		}
		dst[i] = g*dst[i] + v*(1+float64(n+2)*eps())
	}
}
