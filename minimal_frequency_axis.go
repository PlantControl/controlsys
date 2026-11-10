package controlsys

import (
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// frequencyAxisCutInBand reports whether the staircase reduction red of
// sys verifiably matches it to √ε (relative) on the frequency axis, the
// unit circle for a discrete model and the imaginary axis for a continuous
// one; removed holds the stage blocks of the removed states. The staircase
// judges couplings against ε·‖A‖, but the response on the axis is that of
// (sI−A)⁻¹, whose norm can be far above 1/(ε·‖A‖): for modes clustered
// near z = 1 in a graded realization (matched C2D), or for a realization
// graded by a diagonal similarity of many decades. A coupling below the
// floor then carries an O(1) share of the response, and the spectral-gap
// test on |λ| cannot see it when the |λ| show no resolved gap.
//
// To first order, zeroing the coupling R below the removed block A₂₂
// changes the response by C̃₂(sI−A₂₂)⁻¹R(sI−A₁₁)⁻¹B₁. Near a kept pole
// the response carries the same (sI−A₁₁)⁻¹ growth, so the relative change
// stays bounded there; near a removed pole only the change grows. The
// probes are therefore the axis points beside the removed eigenvalues
// (removedModeProbes). Where a probe cannot decide (bandUndetermined), as
// on a removed pole within rounding of the axis, a fixed set is checked as
// well, which also stands in when the removed eigenvalues do not converge:
// the gapProbePoints grid on the circle, and on the imaginary axis
// gapProbePoints at the eigenvalues of A, the cut failing when they do not
// converge.
func frequencyAxisCutInBand(sys, red *System, removed []*mat.Dense) bool {
	discrete := sys.Dt != 0
	if pts, ok := removedModeProbes(removed, discrete); ok {
		switch frequencyAxisBand(sys, red, pts) {
		case bandFail:
			return false
		case bandPass:
			return true
		}
	}
	if discrete {
		return frequencyAxisBand(sys, red, gapProbePoints(nil, 0, true)) != bandFail
	}
	var eig mat.Eigen
	if !eig.Factorize(sys.A, mat.EigenNone) {
		return false
	}
	return frequencyAxisBand(sys, red, gapProbePoints(eig.Values(nil), 0, false)) != bandFail
}

type bandVerdict int

const (
	bandPass bandVerdict = iota
	// bandUndetermined is a pass that needed the model's own rounding
	// bound at some point.
	bandUndetermined
	bandFail
)

// frequencyAxisBand compares sys and red at pts. With the first-order
// rounding bounds δ of boundedResponse, the computed responses Ĝ, R̂ put
// the reduction's response within |Ĝ−R̂| + δ_R of Ĝ, while the model's own
// data determine its response only to δ_G, so a point passes when
// |Ĝ−R̂| + δ_R ≤ √ε·scale + δ_G. Beside a rounded pole of the model, where
// δ_G exceeds the response, no comparison can decide and the point
// passes; a reduction whose own response is that uncertain where the
// model's is not fails.
func frequencyAxisBand(sys, red *System, pts []complex128) bandVerdict {
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	tol := math.Sqrt(eps())
	pm := full.p * full.m
	verdict := bandPass
	for k := range pts {
		if !full.ok[k] {
			verdict = bandUndetermined
			continue
		}
		if !cut.ok[k] {
			return bandFail
		}
		g, r := full.g[k*pm:(k+1)*pm], cut.g[k*pm:(k+1)*pm]
		scale := 0.0
		for i, v := range g {
			d := complex(full.d[i], 0)
			scale = math.Max(scale, math.Max(cmplx.Abs(v-d), cmplx.Abs(r[i]-d)))
		}
		for i, v := range g {
			excess := cmplx.Abs(v-r[i]) + cut.bound[k*pm+i]
			if excess <= tol*scale {
				continue
			}
			if !(excess-full.bound[k*pm+i] <= tol*scale) {
				return bandFail
			}
			verdict = bandUndetermined
		}
	}
	return verdict
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
