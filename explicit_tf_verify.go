package controlsys

import (
	"math"
	"math/cmplx"
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
func explicitTFVerified(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) bool {
	if n, m, p := sys.Dims(); n == 0 || m == 0 || p == 0 {
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
// tf is not verified (descriptorTFJudgeWhy), prefixed "pencil: " where the
// fold's gate (foldWithinRounding) sends every point to the pencil judge.
func explicitRouteJudgeWhy(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) (bool, []int, string) {
	if route == tfRouteExplicit {
		return explicitTFJudgeWhy(identityPencil(sys), sys, tf, nil, true)
	}
	ex, k, err := sys.explicitForm("TransferFunction")
	if err != nil || !foldWithinRounding(sys, ex, k) {
		v, ref, why := descriptorTFJudgeWhy(sys, tf, c)
		return v, ref, "pencil: " + why
	}
	return explicitTFJudgeWhy(sys, ex, tf, c, false)
}

// explicitTFJudgeWhy is descriptorTFJudgeWhy of tf against the descriptor
// pencil with each point tried first by the fast tier on the explicit
// model, whose response is pencil's or that of one ε-rounding of its A and
// B (foldWithinRounding); identity marks pencil as (A, B, C, D, I) of
// model, whose point at ∞ identityInfinityRefutes judges.
func explicitTFJudgeWhy(pencil, model *System, tf *TransferFunc, c *pencilCache, identity bool) (bool, []int, string) {
	rows := newTFRows(tf)
	slow := pencilPointJudge(pencil, tf, rows, c)
	var fast *explicitFastTier
	mk := func(poles []complex128) pointJudge {
		judge := slow(poles)
		return func(pts []complex128) []descriptorTFPoint {
			if fast == nil {
				fast = newExplicitFastTier(model, rows)
			}
			out := make([]descriptorTFPoint, len(pts))
			for k, s := range pts {
				if fast.verify(s) {
					out[k] = descriptorTFPoint{determined: true, verified: true}
					continue
				}
				if out[k] = judge([]complex128{s})[0]; out[k].refuted {
					break
				}
			}
			return out
		}
	}
	return descriptorScheduleWhy(pencil, tf, mk, func() bool {
		if identity {
			return identityInfinityRefutes(pencil, tf, c)
		}
		return descriptorTFInfinityRefutesCached(pencil, tf, c)
	})
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

// identityPencil is the explicit sys as the descriptor (A, B, C, D, I),
// delays dropped.
func identityPencil(sys *System) *System {
	n, _, _ := sys.Dims()
	return &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D, E: eyeDense(n), Dt: sys.Dt}
}

// foldWithinRounding reports whether ex, the explicit form (E⁻¹A,
// 2ᵏ·E⁻¹B, 2⁻ᵏ·C, D) of the descriptor sys as explicitForm returns it with
// its state scaling k, is exactly the fold of A + ΔA and B + ΔB with |ΔA| ≤
// ε|A| and |ΔB| ≤ ε|B| componentwise: |E·Ā − A| ≤ ε|A| and |E·B̄·2⁻ᵏ − B| ≤
// ε|B|, the residuals formed in compensated arithmetic with their
// evaluation bound counted, each row of E and column of the solution
// scaled by a power of two so that no product over- or underflows; false
// where a scaling is inexact, or where C·2⁻ᵏ and D are not the stored C
// and D exactly. A zero entry of A or B allows no residual.
func foldWithinRounding(sys, ex *System, k int) bool {
	n, m, p := sys.Dims()
	e := denseData(sys.E)
	eExp := make([]int, n)
	es := make([]float64, n*n)
	for i := range n {
		eExp[i] = maxExp(e[i*n : (i+1)*n])
		for l := range n {
			v, ok := scaledExact(e[i*n+l], -eExp[i])
			if !ok {
				return false
			}
			es[i*n+l] = v
		}
	}
	check := func(x, ref []float64, cols, shift int) bool {
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
					return false
				}
				xs[l] = v
			}
			for i := range n {
				a, ok := scaledExact(ref[i*cols+j], shift-eExp[i]-xe)
				if !ok {
					return false
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
					return false
				}
			}
		}
		return true
	}
	for i := range p {
		for l := range n {
			if v := sys.C.At(i, l); ex.C.At(i, l) != math.Ldexp(v, -k) || !exactLdexp(v, -k) {
				return false
			}
		}
		for j := range m {
			if ex.D.At(i, j) != sys.D.At(i, j) {
				return false
			}
		}
	}
	return check(denseData(ex.A), denseData(sys.A), n, 0) && check(denseData(ex.B), denseData(sys.B), m, k)
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
}

func newExplicitFastTier(sys *System, rows tfRows) *explicitFastTier {
	n, m, p := sys.Dims()
	hs := newHessenbergSweep(sys, n, m, p)
	f := &explicitFastTier{hs: hs, n: n, m: m, p: p, rows: rows,
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
	if f.check(pt, s) {
		return true
	}
	if err := hs.evalInto(pt, f.dst); err != nil || hs.via != viaSweepSolve {
		return false
	}
	qMulCols(f.xf, hs.y, hs.q, n, m)
	for i, v := range hs.x {
		f.xf[i] += v
	}
	return f.check(pt, s)
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
	return f.upper(f.rho)
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
			var re, im compensatedSum
			mag, tlo := 0.0, 0.0
			for k, c := range cr {
				re.addProd(c, real(x[k]))
				im.addProd(c, imag(x[k]))
				mag += math.Abs(c) * cabs1(x[k])
				tlo += math.Abs(c) * math.Max(termsAbs(x[k])*(1-2*termsEps)-f.rho[k], 0)
			}
			d := 0.0
			if hs.d != nil {
				d = hs.d[i*hs.dStride+j]
				re.add(d)
			}
			g := complex(re.value(), im.value())
			o := dot2Err(n+2)*(mag+math.Abs(d))*(1+float64(n+4)*termsEps) + termsEps*cabs1(g) + float64(4*n+16)*0x1p-1074
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
	hs := f.hs
	hs.nsolve(y)
	hs.factorError(f.tw, y)
	hs.nsolve(f.tw)
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
