package controlsys

import (
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// frequencyAxisCutInBand reports whether the staircase reduction red of
// sys is verified to match it to tol (relative) on the frequency axis, the
// unit circle for a discrete model and the imaginary axis for a continuous
// one; removed holds the stage blocks of the removed states. The staircase
// judges couplings against ε·‖A‖, but the response on the axis is that of
// (sI−A)⁻¹, whose norm can be far above 1/(ε·‖A‖): for modes clustered
// near z = 1 in a graded realization (matched C2D), or for a realization
// graded by a diagonal similarity of many decades. A coupling below the
// floor then carries an O(1) share of the response.
//
// The error of the cut need not peak beside the removed modes: when a
// removed pole is far from the axis the change C̃₂(sI−A₂₂)⁻¹R(sI−A₁₁)⁻¹B₁
// varies like the response itself, and the relative error is set by the
// kept poles and the zeros (matched C2D puts its zeros at z = −1). So
// every point of axisProbes is classified (classifyProbe), and the cut is
// accepted only when none rejects and the verified points give at least
// n + nr interpolation conditions, n and nr the orders of sys and red:
// G − R has a numerator of degree at most n + nr − 1, so agreement at
// n + nr points fixes it in exact arithmetic. A point off the real axis
// counts twice, since G − R is real-rational and agreement at s implies
// agreement at s̄; a point on it (s = 0, z = ±1) counts once.
//
// Agreement within tol at those points gives no uniform bound between
// them: the probes are anchored on the data, not a node set with a known
// Lebesgue constant, and G − R can have poles near the axis between
// probes. Every eigenvalue of A has a probe at its nearest axis point, so
// pole-driven peaks are sampled to first order, while the zeros of G,
// where the relative error peaks, are sampled only by the fixed and decade
// grids. Nor is the check exact: the responses and their bounds are
// evaluated in floating point, so a point where the model's response is
// below its own first-order bound can neither verify nor refute a cut.
// False also when the eigenvalues of A or of red do not converge.
func frequencyAxisCutInBand(sys, red *System, removed []*mat.Dense, tol float64) bool {
	n, _, _ := sys.Dims()
	nr, _, _ := red.Dims()
	pts, poles, ok := axisProbes(sys, removed)
	if !ok {
		return false
	}
	win, ok := sharedPoleWindows(poles, red, tol)
	if !ok {
		return false
	}
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	verified := 0
	for k, s := range pts {
		switch classifyProbe(full, cut, k, tol, win.contain(s)) {
		case probeReject:
			return false
		case probeVerified:
			verified++
			if imag(s) != 0 {
				verified++
			}
		}
	}
	return verified >= n+nr
}

type probeVerdict int

const (
	probeVerified probeVerdict = iota
	// probeUninformative is a point where neither response is determined
	// well enough to verify or refute the cut.
	probeUninformative
	probeReject
)

// classifyProbe judges the cut at point k of full and cut, the responses
// Ĝ, R̂ of the model and the cut with the first-order rounding bounds δ_G,
// δ_R of boundedResponse, per entry against tol·scale, scale the largest
// |Ĝ−D|, |R̂−D| of the point. An entry is verified when
// |Ĝ−R̂| + δ_R ≤ tol·scale and refuted, rejecting the point, when
// |Ĝ−R̂| − δ_G − δ_R > tol·scale: the exact deviation is then provably
// above tol. Otherwise the entry is open:
//   - where the model is determined to tol (δ_G ≤ tol·|Ĝ−D|) the cut is
//     not, and the point rejects, unless inShared: within the window of a
//     pole the model shares with the cut (sharedPoleWindows), where no
//     realization with the cut's conditioning is determined to tol;
//   - where it is not, the point rejects when the cut's response is
//     noise-dominated (δ_R ≥ |R̂−D|) or deviates from the model's by more
//     than the model's own bound (|Ĝ−R̂| − δ_G > tol·scale), and is
//     uninformative otherwise.
//
// A point where GEPP meets an exactly singular model is uninformative, one
// with a singular cut rejects.
func classifyProbe(full, cut boundedResponse, k int, tol float64, inShared bool) probeVerdict {
	if !full.ok[k] {
		return probeUninformative
	}
	if !cut.ok[k] {
		return probeReject
	}
	pm := full.p * full.m
	g, r := full.g[k*pm:(k+1)*pm], cut.g[k*pm:(k+1)*pm]
	scale := 0.0
	for i, v := range g {
		d := complex(full.d[i], 0)
		scale = math.Max(scale, math.Max(cmplx.Abs(v-d), cmplx.Abs(r[i]-d)))
	}
	verdict := probeVerified
	for i, v := range g {
		d := complex(full.d[i], 0)
		dG, dR := full.bound[k*pm+i], cut.bound[k*pm+i]
		diff := cmplx.Abs(v - r[i])
		if diff+dR <= tol*scale {
			continue
		}
		if diff-dG-dR > tol*scale {
			return probeReject
		}
		if dG <= tol*cmplx.Abs(v-d) {
			if !inShared {
				return probeReject
			}
		} else if dR >= cmplx.Abs(r[i]-d) || diff-dG > tol*scale {
			return probeReject
		}
		verdict = probeUninformative
	}
	return verdict
}

// modelPoles holds the eigenvalues of the model's A with their
// componentwise rounding bounds β (eigenRoundingBounds) and the
// eigensolver's resolution ρ₀ = n·ε·‖A_bal‖_F, A_bal the power-of-two
// balanced A of boundedResponse.
type modelPoles struct {
	evals []complex128
	beta  []float64
	rho0  float64
}

// axisProbes returns the deduplicated frequency-axis points the cut is
// judged at: the points beside the removed modes (removedModeProbes) and
// beside every eigenvalue of A (eigenAxisProbes), the fixed grid
// (gapProbePoints, with θ = π for a discrete model: matched and Tustin
// C2D map the zeros at infinity to z = −1, and ZOH sampling zeros approach
// it), and the decade grid (decadeAxisProbes). A point s within
// ρ(s) = n·ε·(|s| + ‖A_bal‖_F) of an eigenvalue is dropped: there sI − A
// is singular to working precision (σ_min(sI−A) ≤ |s−λ|), both responses
// are rounding and the first-order bound does not hold. The eigenvalues'
// own bounds β are not the radius: on a cluster they span the points that
// refute a wrong cut. False when the eigenvalues of A do not converge.
func axisProbes(sys *System, removed []*mat.Dense) ([]complex128, modelPoles, bool) {
	n, m, p := sys.Dims()
	discrete := sys.Dt != 0
	evals, beta, ok := eigenRoundingBounds(sys.A)
	if !ok {
		return nil, modelPoles{}, false
	}
	mp := modelPoles{evals: evals, beta: beta}
	resolved := true
	for k, ev := range evals {
		if !(ev != 0 && mp.beta[k] < cmplx.Abs(ev)) {
			resolved = false
		}
	}
	norm := 0.0
	for _, a := range newBalancedRealization(sys, n, m, p).a {
		norm = math.Hypot(norm, a)
	}
	mp.rho0 = float64(n) * eps() * norm
	var pts []complex128
	push := func(s complex128) {
		if !slices.Contains(pts, s) {
			pts = append(pts, s)
		}
	}
	if rp, ok := removedModeProbes(removed, discrete); ok {
		for _, s := range rp {
			push(s)
		}
	}
	for _, s := range eigenAxisProbes(evals, discrete) {
		push(s)
	}
	for _, s := range gapProbePoints(evals, 0, discrete) {
		push(s)
	}
	if discrete {
		push(complex(-1, 0))
	}
	for _, s := range decadeAxisProbes(evals, resolved, discrete) {
		push(s)
	}
	kept := pts[:0]
	for _, s := range pts {
		rho := float64(n) * eps() * (cmplx.Abs(s) + norm)
		if !slices.ContainsFunc(evals, func(ev complex128) bool { return cmplx.Abs(s-ev) <= rho }) {
			kept = append(kept, s)
		}
	}
	return kept, mp, true
}

// eigenRoundingBounds returns the eigenvalues of A with their
// componentwise rounding bounds β = ε·|y|ᵀ|A||x|/|yᴴx|, computed for a
// zero eigenvalue too (eigenvalueRoundingTerms skips it): 0 when the
// terms are, as for a zero column of A, so a computed zero that rounding
// does move keeps its bound. False when the eigenvalues do not converge.
func eigenRoundingBounds(A *mat.Dense) ([]complex128, []float64, bool) {
	pe, ok := newPoleEigen(A)
	if !ok {
		return nil, nil, false
	}
	beta := make([]float64, len(pe.poles))
	for k := range pe.poles {
		yax, yx := pe.coupling(A, k, k)
		if yax != 0 {
			beta[k] = eps() * yax / cmplx.Abs(yx)
		}
	}
	return pe.poles, beta, true
}

// poleWindows are disks of the frequency plane.
type poleWindows struct {
	centre []complex128
	radius []float64
}

func (w poleWindows) contain(s complex128) bool {
	for i, c := range w.centre {
		if cmplx.Abs(s-c) <= w.radius[i] {
			return true
		}
	}
	return false
}

// sharedPoleWindows returns, for every pole λ_R of red that the model
// shares (|λ_R − λ_G| ≤ β_R + β_G + ρ₀) and determines to tol
// (β_G ≤ tol·max(|λ_G|, ρ₀)), and that the cut carries to the eigensolver's
// resolution (β_R ≤ ρ₀), the disk of radius β_R/tol about it: there a
// rounding-level move of the cut's pole changes the cut's response by more
// than tol, so no realization is determined to tol, however exactly the
// model carries the pole (a kept integrator that the rotated cut holds
// only to rounding). A cut pole known worse than that (β_R > ρ₀) is the
// cut's own ill-conditioning, which the window must not excuse: its β_R/tol
// can span the whole cluster where the model is determined. False when the
// eigenvalues of red do not converge.
func sharedPoleWindows(mp modelPoles, red *System, tol float64) (poleWindows, bool) {
	var w poleWindows
	nr, _, _ := red.Dims()
	if nr == 0 {
		return w, true
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		return w, false
	}
	for i, lr := range evR {
		if !(betaR[i] <= mp.rho0) {
			continue
		}
		for j, lg := range mp.evals {
			bG := mp.beta[j]
			if cmplx.Abs(lr-lg) <= betaR[i]+bG+mp.rho0 && bG <= tol*math.Max(cmplx.Abs(lg), mp.rho0) {
				w.centre = append(w.centre, lr)
				w.radius = append(w.radius, betaR[i]/tol)
				break
			}
		}
	}
	return w, true
}

// eigenAxisProbes returns the axis points beside every eigenvalue, one set
// per conjugate pair: on the imaginary axis j·Im λ and j·|λ|; on the unit
// circle λ/|λ| (z = 1 for λ = 0) and the corner e^{jθ}, θ = min(|ln λ|, π)
// (z = −1 for λ = 0): λ = e^{(−a+jb)T} has |ln λ| = |λ_c|·T, the discrete
// form of j·|λ|. A corner nearer than √ε to z = 1 is skipped: so near its
// pole the model is not determined to √ε.
func eigenAxisProbes(evals []complex128, discrete bool) []complex128 {
	var pts []complex128
	for _, ev := range evals {
		if cmplx.IsNaN(ev) || cmplx.IsInf(ev) || imag(ev) < 0 {
			continue
		}
		if !discrete {
			pts = append(pts, complex(0, imag(ev)), complex(0, cmplx.Abs(ev)))
			continue
		}
		r := cmplx.Abs(ev)
		if r == 0 {
			pts = append(pts, 1, -1)
			continue
		}
		pts = append(pts, ev/complex(r, 0))
		switch th := cmplx.Abs(cmplx.Log(ev)); {
		case th >= math.Pi:
			pts = append(pts, -1)
		case th >= math.Sqrt(eps()):
			pts = append(pts, cmplx.Rect(1, th))
		}
	}
	return pts
}

// decadeAxisProbes returns points a decade apart across the band the
// model's data can carry. Continuous: ω from lo to max|λ|, lo = ε·max|λ|
// (the rounding floor of gapProbePoints) unless every eigenvalue is
// resolved from zero, then max(min|λ|, ε·max|λ|): an unresolved eigenvalue
// can sit anywhere below the resolution. Discrete: θ from lo to π, lo =
// max(min|λ−1|, √ε), or √ε when an eigenvalue is unresolved.
func decadeAxisProbes(evals []complex128, resolved, discrete bool) []complex128 {
	lo, hi := math.Inf(1), 0.0
	for _, ev := range evals {
		w := cmplx.Abs(ev)
		if discrete {
			w = cmplx.Abs(ev - 1)
		}
		if w > 0 && !math.IsInf(w, 0) {
			lo = math.Min(lo, w)
			hi = math.Max(hi, w)
		}
	}
	var pts []complex128
	if discrete {
		lo = math.Max(lo, math.Sqrt(eps()))
		if !resolved || math.IsInf(lo, 0) {
			lo = math.Sqrt(eps())
		}
		for th := lo; th <= math.Pi; th *= 10 {
			pts = append(pts, cmplx.Rect(1, th))
		}
		return pts
	}
	if hi == 0 {
		return nil
	}
	if !resolved {
		lo = eps() * hi
	}
	lo = math.Max(lo, eps()*hi)
	for w := lo; w <= hi; w *= 10 {
		pts = append(pts, complex(0, w))
	}
	return pts
}

// removedModeProbes returns the frequency-axis points beside the
// eigenvalues of the blocks, one set per conjugate pair: z = λ/|λ| (z = 1
// for λ = 0) on the unit circle, and on the imaginary axis both the
// nearest point j·Im λ and j·|λ|. For a real λ the nearest point is s = 0,
// while (sI−A₂₂)⁻¹ keeps its size up to |s| ≈ |λ|, beyond which a
// realization inexact in its Markov parameters (CAᵏB) shows its error;
// j·|λ| sits at that corner. False when the eigenvalues do not converge or
// are not finite.
func removedModeProbes(blocks []*mat.Dense, discrete bool) ([]complex128, bool) {
	var pts []complex128
	push := func(s complex128) {
		if !slices.Contains(pts, s) {
			pts = append(pts, s)
		}
	}
	add := func(ev complex128) bool {
		if cmplx.IsNaN(ev) || cmplx.IsInf(ev) {
			return false
		}
		if imag(ev) < 0 {
			return true
		}
		if !discrete {
			push(complex(0, imag(ev)))
			push(complex(0, cmplx.Abs(ev)))
			return true
		}
		z := complex(1, 0)
		if r := cmplx.Abs(ev); r > 0 {
			z = ev / complex(r, 0)
		}
		push(z)
		return true
	}
	for _, blk := range blocks {
		raw := blk.RawMatrix()
		switch raw.Rows {
		case 1:
			if !add(complex(raw.Data[0], 0)) {
				return nil, false
			}
		case 2:
			a, b, c, d := raw.Data[0], raw.Data[1], raw.Data[raw.Stride], raw.Data[raw.Stride+1]
			h := (a + d) / 2
			disc := (a-d)*(a-d)/4 + b*c
			ev := []complex128{complex(h+math.Sqrt(disc), 0), complex(h-math.Sqrt(disc), 0)}
			if disc < 0 {
				ev = []complex128{complex(h, math.Sqrt(-disc))}
			}
			for _, e := range ev {
				if !add(e) {
					return nil, false
				}
			}
		default:
			var eig mat.Eigen
			if !eig.Factorize(blk, mat.EigenNone) {
				return nil, false
			}
			for _, e := range eig.Values(nil) {
				if !add(e) {
					return nil, false
				}
			}
		}
	}
	return pts, true
}

// boundedResponse is the response G(s) = C(sI−A)⁻¹B + D of an explicit
// model at points s, row-major p×m per point, with for each entry the
// first-order bound ε·(|L||A||U| + |C||U| + |L||B|)ᵢⱼ, U = (sI−A)⁻¹B and
// L = C(sI−A)⁻¹, on its change when every entry of A, B and C moves by ε
// relative: the rounding of the stored data and, to first order, the
// componentwise backward error of GEPP on sI−A. Response and bound are
// invariant under the exact power-of-two balancing the solve uses. ok is
// false at points where GEPP meets an exactly zero pivot.
type boundedResponse struct {
	p, m  int
	d     []float64
	g     []complex128
	bound []float64
	ok    []bool
}

func newBoundedResponse(sys *System, pts []complex128) boundedResponse {
	n, m, p := sys.Dims()
	br := newBalancedRealization(sys, n, m, p)
	f := make([]float64, p*m+len(pts)*p*m+2*n*m+n*p)
	out := boundedResponse{
		p: p, m: m,
		d:     f[:p*m],
		g:     make([]complex128, len(pts)*p*m),
		bound: f[p*m : p*m+len(pts)*p*m],
		ok:    make([]bool, len(pts)),
	}
	f = f[p*m+len(pts)*p*m:]
	au, absU, absV := f[:n*m], f[n*m:2*n*m], f[2*n*m:]
	for i := range p {
		copy(out.d[i*m:(i+1)*m], br.d[i*br.dStride:i*br.dStride+m])
	}
	w := make([]complex128, n*n+n*m+n*p)
	x, u, v := w[:n*n], w[n*n:n*n+n*m], w[n*n+n*m:]
	piv := make([]int, n)
	for k, s := range pts {
		for i, a := range br.a {
			x[i] = complex(-a, 0)
		}
		for i := range n {
			x[i*n+i] += s
		}
		if !cgeppFactor(x, n, piv) {
			continue
		}
		for i, b := range br.b {
			u[i] = complex(b, 0)
		}
		for i := range n {
			for j := range p {
				v[i*p+j] = complex(br.c[j*n+i], 0)
			}
		}
		cgeppSolve(x, n, piv, u, m)
		cgeppSolveT(x, n, piv, v, p)
		for i, z := range u {
			absU[i] = cmplx.Abs(z)
		}
		for i, z := range v {
			absV[i] = cmplx.Abs(z)
		}
		clear(au)
		for i := range n {
			ai := au[i*m : (i+1)*m]
			for l, a := range br.a[i*n : (i+1)*n] {
				if a == 0 {
					continue
				}
				a = math.Abs(a)
				for j, x := range absU[l*m : (l+1)*m] {
					ai[j] += a * x
				}
			}
		}
		g, bound := out.g[k*p*m:(k+1)*p*m], out.bound[k*p*m:(k+1)*p*m]
		for i := range p {
			ci := br.c[i*n : (i+1)*n]
			for j := range m {
				var gij complex128
				t := 0.0
				for l, c := range ci {
					gij += complex(c, 0) * u[l*m+j]
					t += absV[l*p+i]*(au[l*m+j]+math.Abs(br.b[l*m+j])) + math.Abs(c)*absU[l*m+j]
				}
				g[i*m+j] = gij + complex(out.d[i*m+j], 0)
				bound[i*m+j] = eps() * t
			}
		}
		out.ok[k] = true
	}
	return out
}

// cgeppFactor overwrites the n×n row-major x with its partial-pivoting LU
// factors, the unit lower factor's multipliers below the diagonal, and
// records the row swaps in piv; false at an exactly zero or NaN pivot.
func cgeppFactor(x []complex128, n int, piv []int) bool {
	for k := range n {
		p, best := k, cabs1(x[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(x[i*n+k]); v > best {
				p, best = i, v
			}
		}
		if !(best > 0) || math.IsInf(best, 0) {
			return false
		}
		piv[k] = p
		if p != k {
			rk, rp := x[k*n:(k+1)*n], x[p*n:(p+1)*n]
			for j := range rk {
				rk[j], rp[j] = rp[j], rk[j]
			}
		}
		ik := crecip(x[k*n+k])
		rk := x[k*n+k+1 : (k+1)*n]
		for i := k + 1; i < n; i++ {
			f := x[i*n+k] * ik
			if f == 0 {
				continue
			}
			x[i*n+k] = f
			ri := x[i*n+k+1 : (i+1)*n]
			for j, v := range rk {
				ri[j] -= f * v
			}
		}
	}
	return true
}

// cgeppSolve overwrites the n×r row-major y with X⁻¹y for the factors of
// cgeppFactor: PX = LU, the multipliers swapped along with their rows.
func cgeppSolve(x []complex128, n int, piv []int, y []complex128, r int) {
	for k := range n {
		if p := piv[k]; p != k {
			yk, yp := y[k*r:(k+1)*r], y[p*r:(p+1)*r]
			for j := range yk {
				yk[j], yp[j] = yp[j], yk[j]
			}
		}
	}
	for k := range n {
		yk := y[k*r : (k+1)*r]
		for i := k + 1; i < n; i++ {
			if f := x[i*n+k]; f != 0 {
				yi := y[i*r : (i+1)*r]
				for j, v := range yk {
					yi[j] -= f * v
				}
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		yi := y[i*r : (i+1)*r]
		for l := i + 1; l < n; l++ {
			if f := x[i*n+l]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
		ii := crecip(x[i*n+i])
		for j := range yi {
			yi[j] *= ii
		}
	}
}

// cgeppSolveT overwrites the n×r row-major y with X⁻ᵀy for the factors of
// cgeppFactor: PX = LU gives Xᵀ = UᵀLᵀP.
func cgeppSolveT(x []complex128, n int, piv []int, y []complex128, r int) {
	for i := range n {
		yi := y[i*r : (i+1)*r]
		for l := range i {
			if f := x[l*n+i]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
		ii := crecip(x[i*n+i])
		for j := range yi {
			yi[j] *= ii
		}
	}
	for i := n - 1; i >= 0; i-- {
		yi := y[i*r : (i+1)*r]
		for l := i + 1; l < n; l++ {
			if f := x[l*n+i]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		if p := piv[k]; p != k {
			yk, yp := y[k*r:(k+1)*r], y[p*r:(p+1)*r]
			for j := range yk {
				yk[j], yp[j] = yp[j], yk[j]
			}
		}
	}
}
