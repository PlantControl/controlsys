package controlsys

import (
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// torusFarPoints caps the passages torusFarPassages steps through;
// torusRidgePoints those of a ridge, every one of which it samples.
const (
	torusFarPoints   = 1 << 25
	torusRidgePoints = 1 << 17
)

// torusFarPassages decides the case of torusRisesPast where a maximiser θ*
// of S = σ_max(G∞) has a = Re⟨g0, g1⟩ ≠ 0 and the scan of its passages to
// the end of the span, end, found no sample above S* = sup S. With
// t = 1/ω, G(jω) = G∞(θ) + t·g1(θ) + O(t²) rises above S* only where the
// phases θ = ω·h pass within δ of a maximiser θ_m, value S_m, with
// ½·λ_m·δ² <= S_m − level + e_m·t + r(t): λ_m the least curvature of S at
// θ_m, e_m twice the slope of ‖G∞ + t·g1 + …‖ there at t = 1/end and r
// the Cauchy bound on the remainder of the expansion. For several classes such passages are rare and
// their rise falls as 1/ω, so they end at a finite W, and by horizon the
// expansion stays below level past it over the whole torus. Each passage
// between end and W, found from the crossings of the class of least h, is
// sampled on its window with δ doubled; past the span no grid is certified,
// so like torusAtInf this rests on the expansion. level = max(best, S*)·
// (1 + peakCertTol); the peak is the largest window sample above it, its
// window returned for peak to certify, else best above S*, else S* at
// ω = +Inf, as final. Along a flat class (see flatClasses) S is constant
// and λ_m vanishes: δ, λ_m and the crossings then concern the other
// classes alone, e_m is the largest slope over a grid of the flat ones,
// which a passage meets anywhere, and with one other class every passage
// of it is sampled.
func (e *delayLFT) torusFarPassages(x *highFrequencyExpansion, hf *highFrequencyGain, ms []torusMaximum, best, wBest, end float64) (float64, *pastLimit, float64, bool, error) {
	const budget = 1 << 20
	sup := ms[0].value
	lim := sup * (1 + 1e-9)
	level := max(best, sup) * (1 + peakCertTol)
	t0 := 1 / end
	rem := func(t float64) float64 {
		w := x.radius(t)
		if w == 0 {
			return math.Inf(1)
		}
		return x.remainder(e.tailGain(0, w), t*w)
	}
	r0 := rem(t0)
	if math.IsInf(r0, 1) {
		return 0, nil, 0, false, errTorusUnsupported
	}
	type target struct {
		theta         []float64
		s, slope, lam float64
	}
	var ts []target
	for _, m := range ms {
		neg := make([]float64, len(m.theta))
		for c, v := range m.theta {
			neg[c] = -v
		}
		for _, th := range [][]float64{m.theta, neg} {
			dup := false
			for _, o := range ts {
				if torusDistanceOn(o.theta, th, hf.flat) < 1e-6 {
					dup = true
					break
				}
			}
			if dup {
				continue
			}
			sm := hf.at(th)
			slope, ok := ridgeSlope(x, hf.flat, th, sm, t0)
			if !ok {
				return 0, nil, 0, false, errTorusUnsupported
			}
			if !(sm+slope*t0+r0 >= level) {
				continue
			}
			lam := torusCurvature(hf, th, sm, hf.flat)
			if !(lam > 1e-6*sup) {
				return 0, nil, 0, false, errTorusUnsupported
			}
			ts = append(ts, target{slices.Clone(th), sm, slope, lam})
		}
	}
	if hf.over > lim || !hf.below(lim, hf.samples+budget) {
		return 0, nil, 0, false, errTorusUnsupported
	}
	var starts [][]float64
	for _, tg := range ts {
		starts = append(starts, tg.theta)
	}
	cells := 0
	if hf.flat != nil {
		cells = 12
	}
	W, ok := x.horizon(level, end, 1e12, func(t float64) float64 {
		return x.torusModelMax(t, cells, starts)
	})
	if !ok {
		return 0, nil, 0, false, errTorusUnsupported
	}
	h := make([]float64, hf.dim)
	for i, c := range hf.grp {
		k := 1.0
		if hf.k != nil {
			k = hf.k[i]
		}
		h[c] = e.tau[i] / k
	}
	c0, c1 := -1, -1
	for c := range h {
		switch {
		case hf.flat != nil && hf.flat[c]:
		case c0 < 0 || h[c] < h[c0]:
			c0, c1 = c, c0
		case c1 < 0 || h[c] < h[c1]:
			c1 = c
		}
	}
	bFar, wFar := math.Inf(-1), 0.0
	var extFar []float64
	steps := 0
	var scanErr error
	for _, tg := range ts {
		wEnd := W
		if tg.slope > 0 {
			wEnd = min(W, max(end, tg.slope/(level-tg.s)))
		}
		alpha, gamma := 0.0, 0.0
		if c1 >= 0 {
			alpha = h[c1] / h[c0]
			gamma = (tg.theta[c0]*alpha - tg.theta[c1]) / (2 * math.Pi)
		}
		for B := end; B < wEnd; B *= 2 {
			t := 1 / B
			d2 := 8 * (tg.s - level + tg.slope*t + min(r0, rem(t))) / tg.lam
			if !(d2 > 0) {
				break
			}
			delta := math.Sqrt(d2)
			hiB := min(2*B, wEnd)
			n0 := int64(math.Ceil((B*h[c0] - tg.theta[c0]) / (2 * math.Pi)))
			n1 := int64(math.Floor((hiB*h[c0] - tg.theta[c0]) / (2 * math.Pi)))
			eps := 1.1*delta*(1+alpha)/(2*math.Pi) + 1e-12
			visit := func(n int64) bool {
				if steps++; steps > torusFarPoints {
					return false
				}
				wc := (tg.theta[c0] + 2*math.Pi*float64(n)) / h[c0]
				lo, hi := wc-delta/h[c0], wc+delta/h[c0]
				for c, hc := range h {
					if c == c0 || hf.flat != nil && hf.flat[c] {
						continue
					}
					x := wc*hc - tg.theta[c]
					x -= 2 * math.Pi * math.Round(x/(2*math.Pi))
					lo = max(lo, wc+(-delta-x)/hc)
					hi = min(hi, wc+(delta-x)/hc)
					if lo >= hi {
						return true
					}
				}
				const nw = 8
				ws := make([]float64, nw+1)
				for i := range ws {
					ws[i] = lo + (hi-lo)*float64(i)/nw
				}
				b, w, err := e.scan(ws)
				if err != nil {
					scanErr = err
					return false
				}
				if b > bFar {
					bFar, wFar, extFar = b, w, ws
				}
				return true
			}
			ok := true
			if c1 < 0 {
				if steps+int(n1-n0+1) > torusRidgePoints {
					return 0, nil, 0, false, errTorusUnsupported
				}
				for n := n0; n <= n1 && ok; n++ {
					ok = visit(n)
				}
			} else {
				ok = nearCrossings(alpha, gamma, n0, n1, eps, visit)
			}
			if scanErr != nil {
				return 0, nil, 0, false, scanErr
			}
			if !ok {
				return 0, nil, 0, false, errTorusUnsupported
			}
		}
	}
	if bFar > level {
		return lim, &pastLimit{extFar, bFar, wFar}, W, true, nil
	}
	if best > sup*(1+1e-10) {
		return lim, &pastLimit{nil, best, wBest}, W, true, nil
	}
	return lim, &pastLimit{nil, sup, math.Inf(1)}, W, true, nil
}

// torusCurvature returns the least eigenvalue of −∇²S at θ, S(θ) = s, by
// central differences over the classes not flat, NaN when it cannot be
// formed.
func torusCurvature(hf *highFrequencyGain, theta []float64, s float64, flat []bool) float64 {
	var cs []int
	for c := range theta {
		if flat == nil || !flat[c] {
			cs = append(cs, c)
		}
	}
	th := make([]float64, len(theta))
	at := func(i, j int, di, dj float64) float64 {
		copy(th, theta)
		th[i] += di
		th[j] += dj
		return hf.at(th)
	}
	const d = 1e-4
	hess := mat.NewSymDense(len(cs), nil)
	for a, i := range cs {
		hess.SetSym(a, a, -(at(i, i, d, 0)-2*s+at(i, i, -d, 0))/(d*d))
		for b, j := range cs[:a] {
			v := at(i, j, d, d) - at(i, j, d, -d) - at(i, j, -d, d) + at(i, j, -d, -d)
			hess.SetSym(a, b, -v/(4*d*d))
		}
	}
	var es mat.EigenSym
	if !es.Factorize(hess, false) {
		return math.NaN()
	}
	return slices.Min(es.Values(nil))
}

// torusDistanceOn is torusDistance over the classes not flat.
func torusDistanceOn(a, b []float64, flat []bool) float64 {
	d := 0.0
	for c := range a {
		if flat == nil || !flat[c] {
			d = max(d, torusDistance(a[c:c+1], b[c:c+1]))
		}
	}
	return d
}

// ridgeSlope returns twice the largest slope |‖G∞ + t0·g1 + …‖ − s|/t0 at
// θ, with s = S(θ), over a grid of the flat classes, which a passage close
// to θ in the others crosses anywhere; ok is false where I − D22Δ is
// singular.
func ridgeSlope(x *highFrequencyExpansion, flat []bool, theta []float64, s, t0 float64) (float64, bool) {
	var fs []int
	for c, f := range flat {
		if f {
			fs = append(fs, c)
		}
	}
	n := 1
	if len(fs) > 0 {
		n = max(4, int(math.Pow(4096, 1/float64(len(fs)))))
	}
	th := slices.Clone(theta)
	total := int(math.Pow(float64(n), float64(len(fs))))
	slope := 0.0
	for idx := range total {
		r := idx
		for _, c := range fs {
			th[c] = 2 * math.Pi * float64(r%n) / float64(n)
			r /= n
		}
		if !x.atTheta(th) {
			return 0, false
		}
		slope = max(slope, 2*math.Abs(x.sum(t0)-s)/t0)
	}
	return slope, true
}

// nearCrossings calls f, ascending, for every n in [n0, n1] with the
// distance of n·α + γ to the nearest integer at most eps, and possibly a few
// more, until f returns false; it reports whether every call returned true.
// The n are the points k = n − n0 of the lattice {(k, k·α − m)} in a box
// of width n1 − n0 and height 2·eps, found from its Gauss-reduced basis.
func nearCrossings(alpha, gamma float64, n0, n1 int64, eps float64, f func(n int64) bool) bool {
	if n1 < n0 {
		return true
	}
	frac := func(n int64, a, g float64) float64 {
		p := float64(n) * a
		r := math.FMA(float64(n), a, -p)
		v := p - math.Round(p) + r + g
		return v - math.Round(v)
	}
	if n1-n0 <= 64 {
		for n := n0; n <= n1; n++ {
			if math.Abs(frac(n, alpha, gamma)) <= eps && !f(n) {
				return false
			}
		}
		return true
	}
	a := alpha - math.Round(alpha)
	g := frac(n0, alpha, gamma)
	K := float64(n1 - n0)
	vec := func(c [2]float64) [2]float64 {
		return [2]float64{c[0] * 2 / K, math.FMA(c[0], a, -c[1]) / eps}
	}
	dot := func(u, v [2]float64) float64 { return u[0]*v[0] + u[1]*v[1] }
	c1, c2 := [2]float64{1, 0}, [2]float64{0, 1}
	b1, b2 := vec(c1), vec(c2)
	for range 200 {
		if dot(b1, b1) > dot(b2, b2) {
			c1, c2, b1, b2 = c2, c1, b2, b1
		}
		mu := math.Round(dot(b1, b2) / dot(b1, b1))
		if mu == 0 {
			break
		}
		c2 = [2]float64{c2[0] - mu*c1[0], c2[1] - mu*c1[1]}
		b2 = vec(c2)
	}
	q := [2]float64{1, -g / eps}
	n11 := dot(b1, b1)
	m12 := dot(b1, b2) / n11
	bs := [2]float64{b2[0] - m12*b1[0], b2[1] - m12*b1[1]}
	nbs := dot(bs, bs)
	const r = 1.5
	jq := dot(q, bs) / nbs
	dj := r / math.Sqrt(nbs)
	di := r / math.Sqrt(n11)
	if (2*dj+1)*(2*di+1) > torusFarPoints {
		return false
	}
	var ns []int64
	for j := math.Ceil(jq - dj); j <= jq+dj; j++ {
		iq := dot([2]float64{q[0] - j*b2[0], q[1] - j*b2[1]}, b1) / n11
		for i := math.Ceil(iq - di); i <= iq+di; i++ {
			k := int64(i*c1[0] + j*c2[0])
			if k < 0 || k > n1-n0 {
				continue
			}
			if math.Abs(frac(n0+k, alpha, gamma)) <= eps {
				ns = append(ns, n0+k)
			}
		}
	}
	slices.Sort(ns)
	for _, n := range slices.Compact(ns) {
		if !f(n) {
			return false
		}
	}
	return true
}
