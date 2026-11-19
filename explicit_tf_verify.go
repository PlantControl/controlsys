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
	key := tfVerdictKeyOf(sys, tf, route == tfRouteExplicit)
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
		return foldInfinityRefutes(pencil, model, fb, tf, c)
	})
	return v, ref, why, np
}

// foldInfinityRefutes is descriptorTFInfinityRefutesCached for the E⁻¹
// fold ex of sys, deciding in floating point where the margin allows, as
// identityInfinityRefutes: t₀ against D, and where D = 0 = t₀ the first
// Markov parameter g₁ = C·E⁻¹B = C̄·B̄ − C̄·2⁻ᵏE⁻¹R_B·… estimated by the
// compensated C̄·B̄ of the fold within dot2Err of its terms plus |C̄|·K·R̄_B
// (foldBound, built on demand). An entry within 1% of its threshold, or
// one whose order 1 is all zero terms, takes the exact judgement.
func foldInfinityRefutes(sys, ex *System, fb *foldBound, tf *TransferFunc, c *pencilCache) bool {
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
	built := false
	kb := make([]float64, n*m)
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
			if !built {
				built = true
				if fb == nil {
					_, k, err := sys.explicitForm("TransferFunction")
					if err == nil {
						fb, _ = foldBoundOf(sys, ex, k, true)
					}
				}
				if fb == nil {
					return exact()
				}
				col := make([]float64, n)
				for jj := range m {
					for l := range n {
						col[l] = fb.rbd[l*m+jj]
					}
					if !fb.applyK(col) {
						return exact()
					}
					for l := range n {
						kb[l*m+jj] = fb.ky[l]
					}
				}
			}
			t1 := 0.0
			if off == 1 && len(num) > 0 {
				if t1 = num[0] / den[0]; t1 == 0 {
					return exact()
				}
			}
			var g compensatedSum
			T, kt := 0.0, 0.0
			for l := range n {
				cv := ex.C.At(i, l)
				g.addProd(cv, ex.B.At(l, j))
				T += math.Abs(cv * ex.B.At(l, j))
				kt += math.Abs(cv) * kb[l*m+j]
			}
			if T == 0 && t1 == 0 {
				return exact()
			}
			gk := g.value()
			gErr := (dot2Err(n)*T + eps()*math.Abs(gk) + kt) * (1 + float64(n+4)*eps())
			diff := math.Abs(t1 - gk)
			lo := tol*math.Max(math.Max(math.Abs(t1), math.Abs(gk)-gErr), T-gErr) + full.data[i*m+j]
			hi := tol*math.Max(math.Max(math.Abs(t1), math.Abs(gk)+gErr), T+gErr) + full.data[i*m+j]
			switch {
			case !isFinite(diff) || !isFinite(hi) || !isFinite(gErr):
				return exact()
			case diff+gErr <= 0.99*lo:
			case diff-gErr > 1.01*hi:
				return true
			default:
				return exact()
			}
		}
	}
	return false
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
	ra, rb, _, _, _, _, gate, ok = foldResidualFull(sys, ex, k)
	return ra, rb, gate, ok
}

// foldResidualFull is foldResidual also returning the residuals' computed
// values raV and rbV and the bounds raE and rbE on their errors.
func foldResidualFull(sys, ex *System, k int) (ra, rb, raV, rbV, raE, rbE []float64, gate, ok bool) {
	n, m, p := sys.Dims()
	for i := range p {
		for l := range n {
			if v := sys.C.At(i, l); ex.C.At(i, l) != math.Ldexp(v, -k) || !exactLdexp(v, -k) {
				return nil, nil, nil, nil, nil, nil, false, false
			}
		}
		for j := range m {
			if ex.D.At(i, j) != sys.D.At(i, j) {
				return nil, nil, nil, nil, nil, nil, false, false
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
				return nil, nil, nil, nil, nil, nil, false, false
			}
			es[i*n+l] = v
		}
	}
	gate = true
	resid := func(x, ref []float64, cols, shift int) ([]float64, []float64, []float64, bool) {
		out := make([]float64, n*cols)
		val := make([]float64, n*cols)
		errs := make([]float64, n*cols)
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
					return nil, nil, nil, false
				}
				xs[l] = v
			}
			for i := range n {
				a, ok := scaledExact(ref[i*cols+j], shift-eExp[i]-xe)
				if !ok {
					return nil, nil, nil, false
				}
				var rs, rc float64
				mag := math.Abs(a)
				for l, ev := range es[i*n : (i+1)*n] {
					p := float64(ev * xs[l])
					t := rs + p
					bp := t - rs
					rc += (rs - (t - bp)) + (p - bp) + math.FMA(ev, xs[l], -p)
					rs = t
					mag += math.Abs(p)
				}
				r := compensatedSum{rs, rc}
				r.add(-a)
				bound := math.Abs(r.value())*(1+2*eps()) + dot2Err(n+1)*mag*(1+float64(n+4)*eps()) + float64(4*n+16)*0x1p-1074
				if !(bound <= eps()*math.Abs(a)*(1-4*eps())) {
					gate = false
				}
				v := math.Ldexp(bound*(1+2*eps()), eExp[i]+xe)
				if v < 0x1p-1000 {
					v += 0x1p-1074
				}
				rv := math.Ldexp(r.value(), eExp[i]+xe)
				ev := math.Ldexp((2*eps()*math.Abs(r.value())+dot2Err(n+1)*mag*(1+float64(n+4)*eps())+float64(4*n+16)*0x1p-1074)*(1+2*eps()), eExp[i]+xe)
				ev += 2*eps()*math.Abs(rv) + 0x1p-1074
				if !isFinite(v) || !isFinite(rv) || !isFinite(ev) {
					return nil, nil, nil, false
				}
				out[i*cols+j], val[i*cols+j], errs[i*cols+j] = v, rv, ev
			}
		}
		return out, val, errs, true
	}
	ra, raV, raE, okA := resid(denseData(ex.A), denseData(sys.A), n, 0)
	if !okA {
		return nil, nil, nil, nil, nil, nil, false, false
	}
	rb, rbV, rbE, okB := resid(denseData(ex.B), denseData(sys.B), m, k)
	if !okB {
		return nil, nil, nil, nil, nil, nil, false, false
	}
	return ra, rb, raV, rbV, raE, rbE, gate, true
}

// foldBound verifies points of a fold that is not one ε-rounding of A and
// B (foldResidual's gate fails) against the stored pencil F = sE − A. The
// fast tier's plain solution x̂ of the fold (in the 2ᵏ-scaled state, B_s =
// 2ᵏB) gets the correction d = (sI − Ā)⁻¹Ê⁻¹r̂ from the residual r̂ of the
// stored pencil, formed in compensated arithmetic within e_s
// (storedResidual). With R_A = EĀ − A, R_E = I − EÊ⁻¹, r_d = Ê⁻¹r̂ −
// (sI − Ā)d and δ₁ the rounding of Ê⁻¹r̂, F = E(sI − Ā) + R_A gives the
// residual of x̂ + d as r₂ = (r − r̂) + R_E·r̂ − E·δ₁ + E·r_d − R_A·d, so
// |r₂| ≤ e_s + |R_E||r̂| + γ|E||Ê⁻¹||r̂| + |E||r_d| + |R_A||d|, every term but
// e_s second order. Then |x − x̂ − d| ≤ |F⁻¹||r₂|, F⁻¹ = M·E⁻¹·(I + R_A·M·
// E⁻¹)⁻¹, M = (sI − Ā)⁻¹: with K ≥ |E⁻¹| and v = |M|·K·|r₂|, it is at most
// v/(1 − q) where P = |M|·K·|R_A| passes Pv ≤ q·v, q < 1, |M| bounded as in
// the fast tier (upper): with y = K·|r₂| ≥ 0 and v = |M|y, K·|R_A|·v ≤ q·y
// gives Pv ≤ q·|M|y = q·v, |M| being nonnegative. K applies |Ê⁻¹|(I − |R_E|)⁻¹: with S the row scales
// of E and ρ = ‖S⁻¹|R_E|S‖_∞ < 1/2, Σₖ|R_E|ᵏy ≤ y + ρ/(1 − ρ)·‖S⁻¹y‖_∞·S𝟙.
// A point where a check fails is left to the pencil judge; foldBound never
// refutes.
type foldBound struct {
	n, m          int
	e             []float64
	ra, rbd       []float64
	raV, rbV      []float64
	raE, rbE      []float64
	xr, rr        []complex128
	y1v           []float64
	ax, ar        []float64
	einv, absEinv []float64
	absE, absRE   []float64
	sRow          []float64
	rhoE          float64
	r             []complex128
	rb, r2, ky, v []float64
	w             []float64
	y1            []complex128
}

// newFoldBound is foldResidual's verdict with, where the gate fails, the
// foldBound of ex; nil where the fold's residual cannot be carried (ok
// false, 2ᵏB not exact, E's computed inverse not finite or ρ ≥ 1/2).
func newFoldBound(sys, ex *System, k int) (*foldBound, bool) {
	return foldBoundOf(sys, ex, k, false)
}

// foldBoundOf is newFoldBound, with force also building the foldBound of a
// fold whose gate passes.
func foldBoundOf(sys, ex *System, k int, force bool) (*foldBound, bool) {
	ra, rbd, raV, rbV, raE, rbE, gate, ok := foldResidualFull(sys, ex, k)
	if !ok || gate && !force {
		return nil, gate
	}
	n, m, _ := sys.Dims()
	var lu mat.LU
	lu.Factorize(sys.E)
	var inv mat.Dense
	if err := lu.SolveTo(&inv, false, eyeDense(n)); err != nil {
		return nil, gate
	}
	einv := denseData(&inv)
	e := denseData(sys.E)
	sRow := make([]float64, n)
	for i := range n {
		sRow[i] = math.Ldexp(1, maxExp(e[i*n:(i+1)*n]))
	}
	var prod, absProd mat.Dense
	prod.Mul(sys.E, &inv)
	ae, ai := mat.DenseCopyOf(sys.E), mat.DenseCopyOf(&inv)
	ae.Apply(func(_, _ int, v float64) float64 { return math.Abs(v) }, ae)
	ai.Apply(func(_, _ int, v float64) float64 { return math.Abs(v) }, ai)
	absProd.Mul(ae, ai)
	gRE := gammaN(n+2) * (1 + gammaN(n+2)) * (1 + 4*eps())
	absRE := make([]float64, n*n)
	rhoE := 0.0
	for i := range n {
		row := 0.0
		for c := range n {
			r := -prod.At(i, c)
			if i == c {
				r++
			}
			b := math.Abs(r)*(1+2*eps()) + gRE*(absProd.At(i, c)+1) + float64(4*n+16)*0x1p-1074
			absRE[i*n+c] = b
			row += b * sRow[c] / sRow[i]
		}
		rhoE = math.Max(rhoE, row*(1+float64(n+2)*eps()))
	}
	if !(rhoE < 0.5) {
		return nil, gate
	}
	abs := func(v []float64) ([]float64, bool) {
		out := make([]float64, len(v))
		for i, x := range v {
			if out[i] = math.Abs(x); !isFinite(out[i]) {
				return nil, false
			}
		}
		return out, true
	}
	absEinv, ok1 := abs(einv)
	absE, ok2 := abs(e)
	if !ok1 || !ok2 {
		return nil, gate
	}
	return &foldBound{n: n, m: m, e: e, ra: ra, rbd: rbd, raV: raV, rbV: rbV, raE: raE, rbE: rbE, einv: einv, absEinv: absEinv, absE: absE, absRE: absRE,
		sRow: sRow, rhoE: rhoE, r: make([]complex128, n*m), rb: make([]float64, n*m), r2: make([]float64, n),
		ky: make([]float64, n), v: make([]float64, n), w: make([]float64, n), y1: make([]complex128, n*m),
		xr: make([]complex128, n), rr: make([]complex128, n), y1v: make([]float64, n), ax: make([]float64, n), ar: make([]float64, n)}, gate
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
		for l, a := range fb.absEinv[i*n : (i+1)*n] {
			acc += a * (y[l] + c*fb.sRow[l])
		}
		fb.ky[i] = acc * (1 + float64(2*n+4)*eps())
		if !isFinite(fb.ky[i]) {
			return false
		}
	}
	return true
}

// storedResidual sets fb.r to B_s − (sE − A)x for x = f.xf/t and fb.rb to
// a bound on its error. With r_f = B̄ − (sI − Ā)x the residual of the fold
// in compensated arithmetic within e_f (pencilResidualRow, in the sweep's
// coordinates), E·B̄ = B_s + R_B and sE − A = E(sI − Ā) + R_A give
// B_s − (sE − A)x = E·r_f − R_A·x − R_B; it is formed in working precision
// from the computed R̂_A, R̂_B (foldResidualFull), within |E|e_f +
// |R_A − R̂_A||x| + |R_B − R̂_B| + γ_{2n+3}(|E||r̂_f| + |R̂_A||x| + |R̂_B|).
func (fb *foldBound) storedResidual(f *explicitFastTier, pt frequencyPoint) bool {
	n, m, hs := fb.n, fb.m, f.hs
	g := gammaN(2*n + 3)
	for j := range m {
		xb := f.xf[j*n : (j+1)*n]
		xr, rr := fb.xr, fb.rr
		for i := range n {
			rc, e := denseResidualRow(xb, hs.a[i*n:(i+1)*n], i, hs.b[i*m+j], pt.p)
			it := 1 / f.t[i]
			rr[i] = complex(real(rc)*it, imag(rc)*it)
			fb.w[i] = (e + 2*termsEps*cabs1(rc)) * it
			xr[i] = complex(real(xb[i])*it, imag(xb[i])*it)
			fb.ax[i] = cabs1(xr[i])
			fb.ar[i] = cabs1(rr[i])
		}
		for i := range n {
			var ar, ai float64
			var mag, er float64
			for l, ev := range fb.e[i*n : (i+1)*n] {
				av := fb.raV[i*n+l]
				ar += ev*real(rr[l]) - av*real(xr[l])
				ai += ev*imag(rr[l]) - av*imag(xr[l])
				aev := math.Abs(ev)
				mag += aev*fb.ar[l] + math.Abs(av)*fb.ax[l]
				er += aev*fb.w[l] + fb.raE[i*n+l]*fb.ax[l]
			}
			acc := complex(ar, ai)
			acc -= complex(fb.rbV[i*m+j], 0)
			mag += math.Abs(fb.rbV[i*m+j])
			er += fb.rbE[i*m+j]
			fb.r[i*m+j] = acc
			fb.rb[i*m+j] = (er + g*mag) * (1 + 4*eps())
			if !isFinite(fb.rb[i*m+j]) || cmplx.IsNaN(acc) || cmplx.IsInf(acc) {
				return false
			}
		}
	}
	return true
}

// verify is the fast tier's verify of the fold against the stored pencil
// (see foldBound): the plain solution of the fold, corrected from the
// stored residual, with f.rho bounding its error.
func (fb *foldBound) verify(f *explicitFastTier, pt frequencyPoint, s complex128) bool {
	n, m, hs := fb.n, fb.m, f.hs
	if !fb.storedResidual(f, pt) {
		return false
	}
	for i := range n {
		for j := range m {
			var acc complex128
			for l, a := range fb.einv[i*n : (i+1)*n] {
				acc += complex(a, 0) * fb.r[l*m+j]
			}
			fb.y1[i*m+j] = acc
		}
	}
	for j := range m {
		for i := range n {
			f.res[j*n+i] = fb.y1[i*m+j] * complex(f.t[i], 0)
		}
	}
	qTMulCols(hs.y, f.res, hs.q, n, m)
	hs.solve(hs.y)
	qMulCols(f.xl, hs.y, hs.q, n, m)
	f.pair = true
	sa := cabs1(s)
	clear(fb.r2)
	gn := gammaN(n + 2)
	for j := range m {
		d := f.xl[j*n : (j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			var ar, ai, sum float64
			nnz := 0
			for k, a := range row {
				if a == 0 {
					continue
				}
				ar += a * real(d[k])
				ai += a * imag(d[k])
				sum += math.Abs(a) * cabs1(d[k])
				nnz++
			}
			yb := f.res[j*n+i]
			rd := yb + complex(ar, ai) - s*d[i]
			bound := float64(nnz+6) * termsEps / 2 * (cabs1(yb) + sa*cabs1(d[i]) + sum)
			fb.w[i] = ((termsAbs(rd) + bound) * (1 + 2*termsEps)) / f.t[i]
		}
		for l := range n {
			var ei float64
			for q, a := range fb.absEinv[l*n : (l+1)*n] {
				ei += a * cabs1(fb.r[q*m+j])
			}
			fb.ky[l] = ei
		}
		for i := range n {
			var re, ee, ed, ra float64
			for l := range n {
				re += fb.absRE[i*n+l] * cabs1(fb.r[l*m+j])
				ee += fb.absE[i*n+l] * fb.ky[l]
				ed += fb.absE[i*n+l] * fb.w[l]
				ra += fb.ra[i*n+l] * cabs1(d[l]) / f.t[l]
			}
			fb.r2[i] += (fb.rb[i*m+j] + re + gn*ee + ed + ra) * (1 + float64(2*n+8)*eps())
		}
	}
	if !fb.applyK(fb.r2) {
		return false
	}
	for i := range n {
		fb.v[i] = f.t[i] * fb.ky[i]
	}
	copy(fb.y1v, fb.v)
	if !f.upper(fb.v) {
		return false
	}
	for l := range n {
		fb.w[l] = fb.v[l] / f.t[l]
	}
	for i := range n {
		var z float64
		for l := range n {
			z += fb.ra[i*n+l] * fb.w[l]
		}
		fb.r2[i] = z * (1 + float64(n+2)*eps())
	}
	if !fb.applyK(fb.r2) {
		return false
	}
	q := 0.0
	for i := range n {
		u := f.t[i] * fb.ky[i]
		if u == 0 {
			continue
		}
		if !(fb.y1v[i] > 0) {
			return false
		}
		q = max(q, u/fb.y1v[i])
	}
	q *= 1 + 8*float64(n)*termsEps
	if !(q < 1) {
		return false
	}
	g := 1 / (1 - q) * (1 + 2*termsEps)
	for i := range n {
		f.rho[i] = fb.v[i] * g
		if !isFinite(f.rho[i]) {
			return false
		}
	}
	return f.entries(s)
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
// the plain Hessenberg solution x̂, forms the residual r = b − (sI − Â)x̂ of
// every column against the exact point in working precision with its
// rounding bound, and bounds |M|·Σⱼ|rⱼ|, M = (sI − Â)⁻¹ (upper): by the
// comparison matrices of the sweep's factors (termsCert step 3), else with
// the reduction's error measured (sweepPost), else by the approximate
// inverse of GEPP on sI − Â (inverseBound, O(n³) at that point). Then |G −
// ĝ| ≤ δ = |ĉᵢ|·|M|Σⱼ|rⱼ| + o for ĝ = ĉᵢx̂ⱼ + d, o its rounding bound. Where
// that does not verify, the solution becomes the unrounded pair x̂ + x_l,
// x_l the correction from the compensated residual (refinePair), whose
// residual is bounded at second order (pairResidual). An entry verifies
// when |ĝ − t| + δ + β ≤ √ε·max(scale, T), t the transfer function's value
// with error bound β (tfRows.bounded), scale the least the row's largest
// |G| or |t| can be and T the least the entry's terms Σ|ĉᵢ||x| + |d| can be
// (|x| ≥ |x̂| − |M|Σⱼ|rⱼ|); a point verifies when all its entries do. With
// fold set, the point is verified against the stored pencil of a fold
// instead (foldBound.verify). It never refutes: a point the sweep does
// not solve or whose bound does not verify is left to the pencil judge.
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
	resE    []float64
	pair    bool
	fold    *foldBound
	// Where the sweep's comparison matrices and sweepPost leave upper's
	// check short, the approximate inverse of GEPP on sI − Â at pt
	// (inverseBound, bd's factors, absA = |Â|) bounds |M|, formed once
	// per point.
	bd    *balancedDense
	absA  []float64
	pt    frequencyPoint
	ib    *inverseBound
	ibPt  frequencyPoint
	ibSet bool
	// post is the sweep's bound with its reduction error measured
	// (sweepPost), formed where the a-priori bound first fails.
	post   *sweepPost
	noPost bool
	// preferPair skips the plain solution's check after a point that
	// needed the pair.
	preferPair bool
	// tfR and tfB hold the transfer function's rows and bounds at tfAt.
	tfR         []complex128
	tfB         []float64
	tfAt        complex128
	tfSet, tfOK bool
	y0          []float64
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
	f := &explicitFastTier{hs: hs, y0: make([]float64, n), tfR: make([]complex128, p*m), tfB: make([]float64, p*m), n: n, m: m, p: p, rows: rows, t: t,
		xl: make([]complex128, n*m), res: make([]complex128, n*m), resE: make([]float64, n*m),
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
// s (see explicitFastTier): first on the plain Hessenberg solve with its
// working-precision residual, then on the pair of it and its correction
// (refinePair, pairResidual).
func (f *explicitFastTier) verify(s complex128) bool {
	n, m, hs := f.n, f.m, f.hs
	if n == 0 || hs.nonFinite || cmplx.IsNaN(s) || cmplx.IsInf(s) {
		return false
	}
	pt := pointAt(s)
	f.pt = pt
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
	if f.fold != nil {
		return f.fold.verify(f, pt, s)
	}
	if !f.preferPair && f.residual(pt) && f.entries(s) {
		return true
	}
	if f.refinePair(pt) && f.pairResidual(pt) && f.entries(s) {
		f.preferPair = true
		return true
	}
	f.pair = false
	if f.preferPair && f.residual(pt) && f.entries(s) {
		f.preferPair = false
		return true
	}
	return false
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
			rc, e := denseResidualRow(x, hs.a[i*n:(i+1)*n], i, hs.b[i*m+j], pt.p)
			f.res[j*n+i], f.resE[j*n+i] = rc, e
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

// pairResidual is residual for the pair f.xf + f.xl: the residual of the
// pair is r̂ + e − (sI − Â)x_l, r̂ the compensated residual of f.xf and |e|
// its bound (refinePair), and with x_l at the rounding of f.xf the product
// (sI − Â)x_l is formed in working precision with the rounding bound
// (nnzᵢ + 5)(ε/2)Sᵢ of termsCert step 1, r̂ in the place of b.
func (f *explicitFastTier) pairResidual(pt frequencyPoint) bool {
	n, m, hs := f.n, f.m, f.hs
	s := pt.p
	sa := cabs1(s)
	clear(f.rho)
	for j := range m {
		xl := f.xl[j*n : (j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			var ar, ai, sum float64
			nnz := 0
			for k, a := range row {
				if a == 0 {
					continue
				}
				ar += a * real(xl[k])
				ai += a * imag(xl[k])
				sum += math.Abs(a) * cabs1(xl[k])
				nnz++
			}
			rc := f.res[j*n+i]
			r := rc + complex(ar, ai) - s*xl[i]
			bound := float64(nnz+6) * termsEps / 2 * (cabs1(rc) + sa*cabs1(xl[i]) + sum)
			f.rho[i] += (termsAbs(r)+bound)*(1+2*termsEps) + f.resE[j*n+i]*(1+4*termsEps)
		}
	}
	return f.upper(f.rho)
}

// residual sets f.rho to a bound on |M|·Σⱼ|rⱼ| of f.xf at pt, the residual
// in working precision with its rounding bound (nnzᵢ + 5)(ε/2)Sᵢ, as
// termsCert step 1 forms it; false where a bound fails.
func (f *explicitFastTier) residual(pt frequencyPoint) bool {
	n, m, hs := f.n, f.m, f.hs
	s := pt.p
	sa := cabs1(s)
	clear(f.rho)
	for j := range m {
		x := f.xf[j*n : (j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			bi := hs.b[i*m+j]
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
	m, p := f.m, f.p
	if f.tfAt != s || !f.tfSet {
		f.tfSet, f.tfAt, f.tfOK = true, s, true
		for i := range p {
			if !f.rows.bounded(i, s, f.tfR[i*m:(i+1)*m], f.tfB[i*m:(i+1)*m]) {
				f.tfOK = false
				break
			}
		}
	}
	if !f.tfOK {
		return false
	}
	for i := range p {
		copy(f.r, f.tfR[i*m:(i+1)*m])
		copy(f.beta, f.tfB[i*m:(i+1)*m])
		if !f.row(i, false) && !f.row(i, true) {
			return false
		}
	}
	return true
}

// row reports whether every entry of output row i verifies, its outputs
// ĉᵢx̂ⱼ + d summed in working precision within γ_{2n+2} of their terms,
// or with compensated, in compensated arithmetic within dot2Err.
func (f *explicitFastTier) row(i int, compensated bool) bool {
	n, m, hs := f.n, f.m, f.hs
	tol := math.Sqrt(eps())
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
		var pr, pi float64
		mag, tlo := 0.0, 0.0
		for k, c := range cr {
			xk := x[k]
			ax := cabs1(x[k])
			if compensated {
				re.addProd(c, real(x[k]))
				im.addProd(c, imag(x[k]))
			} else {
				pr += c * real(x[k])
				pi += c * imag(x[k])
			}
			if xl != nil {
				if compensated {
					re.addProd(c, real(xl[k]))
					im.addProd(c, imag(xl[k]))
				} else {
					pr += c * real(xl[k])
					pi += c * imag(xl[k])
				}
				xk += xl[k]
				ax += cabs1(xl[k])
			}
			mag += math.Abs(c) * ax
			tlo += math.Abs(c) * math.Max(termsAbs(xk)*(1-4*termsEps)-f.rho[k], 0)
		}
		d := 0.0
		if hs.d != nil {
			d = hs.d[i*hs.dStride+j]
		}
		var g complex128
		var o float64
		if compensated {
			re.add(d)
			g = complex(re.value(), im.value())
			o = dot2Err(2*n+2)*(mag+math.Abs(d))*(1+float64(2*n+4)*termsEps) + termsEps*cabs1(g) + float64(8*n+16)*0x1p-1074
		} else {
			g = complex(pr+d, pi)
			o = gammaN(2*n+3)*(mag+math.Abs(d))*(1+float64(2*n+4)*termsEps) + float64(8*n+16)*0x1p-1074
		}
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
	return true
}

// upper overwrites the nonnegative y with an upper bound on |M|y from the
// comparison matrices of the sweep's factors at its last point
// (termsCert.comparisonUpper), false where the q check fails.
func (f *explicitFastTier) upper(y []float64) bool {
	copy(f.y0, y)
	if f.upperBy(f.hs, y) {
		return true
	}
	if f.post == nil && !f.noPost {
		f.post = newSweepPost(f.hs)
		f.noPost = f.post == nil
	}
	if f.post != nil {
		copy(y, f.y0)
		if f.upperBy(f.post, y) {
			return true
		}
	}
	if !f.ibSet || f.ibPt != f.pt {
		f.ibSet, f.ibPt, f.ib = true, f.pt, nil
		if f.bd == nil {
			f.bd = newBalancedDense(f.hs.sys, f.n, f.m, f.p)
			f.absA = make([]float64, len(f.hs.a))
			for i, v := range f.hs.a {
				f.absA[i] = math.Abs(v)
			}
		}
		if bd := f.bd; !bd.nonFinite && bd.factorAt(f.pt) {
			f.ib = newInverseBound(&f.hs.balancedRealization, f.absA, f.pt, func(y []complex128) {
				cLUSolve(bd.pencil, bd.piv, y, f.n, f.n)
			})
		}
	}
	if f.ib == nil {
		return false
	}
	copy(y, f.y0)
	return f.ib.upper(y)
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

// denseResidualRow is pencilResidualRow at a plain point s (p = s, q = −1,
// κ = 1) without its skip of zero entries: b − s·xᵢ + Σₖ aₖxₖ in
// compensated arithmetic, e bounding the distance of the exact residual
// from the sums' unrounded value as there.
func denseResidualRow(x []complex128, row []float64, i int, b float64, s complex128) (r complex128, e float64) {
	var sr, cr, si, ci, inner float64
	for k, a := range row {
		xr, xi := real(x[k]), imag(x[k])
		p := float64(a * xr)
		t := sr + p
		bp := t - sr
		cr += (sr - (t - bp)) + (p - bp) + math.FMA(a, xr, -p)
		sr = t
		p = float64(a * xi)
		t = si + p
		bp = t - si
		ci += (si - (t - bp)) + (p - bp) + math.FMA(a, xi, -p)
		si = t
		inner += math.Abs(a) * (math.Abs(xr) + math.Abs(xi))
	}
	axr, axi := compensatedSum{sr, cr}, compensatedSum{si, ci}
	pr, pi := real(s), imag(s)
	var re, im compensatedSum
	re.add(b)
	xr, xi := real(x[i]), imag(x[i])
	re.addProd(-pr, xr)
	re.addProd(pi, xi)
	im.addProd(-pr, xi)
	im.addProd(-pi, xr)
	re.addScaled(1, axr)
	im.addScaled(1, axi)
	outer := math.Abs(b) + cabs1(s)*cabs1(x[i]) + math.Abs(sr) + math.Abs(cr) + math.Abs(si) + math.Abs(ci)
	k := len(row)
	e = (dot2Err(7)*outer+dot2Err(k)*inner)*(1+float64(k+16)*termsEps) + (float64(4*k)+64)*0x1p-1074
	return complex(re.value(), im.value()), e
}
