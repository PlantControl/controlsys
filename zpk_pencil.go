package controlsys

import (
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkPencilRows replaces, row by row, the polynomial roots of z (the ZPK of
// the transfer function of sys, with row degrees deg) by the roots MATLAB
// zpk(ss) takes: poles eig(A) of the row's structurally minimal part and,
// per channel, the invariant zeros of that part, with a gain fitted to the
// response of sys. Roots of the transfer-function polynomials lose tight
// clusters, e.g. discrete poles near z = 1 leave the unit circle (HP3RZO).
// The per-channel orders stay those of the transfer function: a row is
// replaced only when the structurally minimal part has its row degree,
// every channel keeps its zero count, and each channel fits the response at
// a probe point no worse than its polynomial roots do. Descriptor models
// keep the polynomial roots.
func (sys *System) zpkPencilRows(z *ZPK, deg []int) {
	n, m, p := sys.Dims()
	if n == 0 || sys.IsDescriptor() || sys.HasDelay() || sys.LFT != nil || len(deg) != p {
		return
	}
	var full []complex128
	for i := range p {
		if deg[i] == 0 {
			continue
		}
		zeros, poles, gains, ok := sys.zpkPencilRow(z, i, deg[i], &full)
		if !ok {
			continue
		}
		for j := range m {
			if gains[j] == 0 {
				continue
			}
			z.Zeros[i][j] = zeros[j]
			z.Poles[i][j] = copyComplex(poles)
			z.Gain[i][j] = gains[j]
		}
	}
}

// zpkPencilRow returns the pencil roots and gains of row i of z, or ok
// false; full caches eig(A), the poles of every row of full degree.
func (sys *System) zpkPencilRow(z *ZPK, i, deg int, full *[]complex128) (zeros [][]complex128, poles []complex128, gains []float64, ok bool) {
	n, m, p := sys.Dims()
	row := sys
	if p > 1 {
		c := mat.NewDense(1, n, nil)
		c.Copy(sys.C.Slice(i, i+1, 0, n))
		d := mat.NewDense(1, m, nil)
		d.Copy(sys.D.Slice(i, i+1, 0, m))
		var err error
		if row, err = New(sys.A, sys.B, c, d, sys.Dt); err != nil {
			return nil, nil, nil, false
		}
	}
	if deg < n {
		var err error
		if row, err = Sminreal(row); err != nil {
			return nil, nil, nil, false
		}
		if nr, _, _ := row.Dims(); nr != deg {
			return nil, nil, nil, false
		}
		if poles, err = row.Poles(); err != nil {
			return nil, nil, nil, false
		}
	} else {
		if *full == nil {
			all, err := sys.Poles()
			if err != nil {
				return nil, nil, nil, false
			}
			if all, ok = matchedConjugateClosed(all); !ok {
				return nil, nil, nil, false
			}
			*full = all
		}
		poles = *full
	}
	if len(poles) != deg {
		return nil, nil, nil, false
	}
	if poles, ok = matchedConjugateClosed(poles); !ok {
		return nil, nil, nil, false
	}
	nr := deg
	zeros = make([][]complex128, m)
	gains = make([]float64, m)
	var roots [][]complex128
	roots = append(roots, poles)
	active := false
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		if len(z.Poles[i][j]) != deg {
			return nil, nil, nil, false
		}
		ch := row
		if m > 1 {
			b := mat.NewDense(nr, 1, nil)
			b.Copy(row.B.Slice(0, nr, j, j+1))
			var err error
			if ch, err = New(row.A, b, row.C, mat.NewDense(1, 1, []float64{row.D.At(0, j)}), sys.Dt); err != nil {
				return nil, nil, nil, false
			}
		}
		zj, err := ch.Zeros()
		if err != nil || len(zj) != len(z.Zeros[i][j]) {
			return nil, nil, nil, false
		}
		if zeros[j], ok = matchedConjugateClosed(zj); !ok {
			return nil, nil, nil, false
		}
		roots = append(roots, zeros[j], z.Zeros[i][j])
		active = true
	}
	if !active {
		return nil, nil, nil, false
	}
	for j := range m {
		roots = append(roots, z.Poles[i][j])
	}
	var s0, s1 complex128
	if sys.IsDiscrete() {
		s0, s1 = matchedD2CProbePoints(sys.Dt, roots...)
	} else {
		s0, s1 = matchedProbePoints(roots...)
	}
	sh := zpkBandProbePoint(sys.Dt, roots)
	g1, err := row.EvalFr(s1)
	if err != nil {
		return nil, nil, nil, false
	}
	gh, err := row.EvalFr(sh)
	if err != nil {
		return nil, nil, nil, false
	}
	var g0 [][]complex128
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		v1, vh := g1[0][j], gh[0][j]
		if !zpkFiniteNonzero(v1) || !zpkFiniteNonzero(vh) {
			return nil, nil, nil, false
		}
		k, ok := zpkMarkovGain(row, j, deg-len(zeros[j]))
		if !ok {
			if g0 == nil {
				if g0, err = row.EvalFr(s0); err != nil {
					return nil, nil, nil, false
				}
			}
			v0 := g0[0][j]
			if !zpkFiniteNonzero(v0) {
				return nil, nil, nil, false
			}
			q := v0 / matchedRootRatio(zeros[j], poles, s0)
			if cmplx.IsNaN(q) || cmplx.IsInf(q) || real(q) == 0 || !(math.Abs(imag(q)) <= 1e-7*cmplx.Abs(q)) {
				return nil, nil, nil, false
			}
			k = real(q)
		}
		gains[j] = k
		misfit := func(zs, ps []complex128, k float64, s, v complex128) float64 {
			return cmplx.Abs(complex(k, 0)*matchedRootRatio(zs, ps, s)-v) / cmplx.Abs(v)
		}
		if e := misfit(zeros[j], poles, k, s1, v1); !(e <= math.Max(2*misfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], s1, v1), 1e-12)) {
			return nil, nil, nil, false
		}
		if e := misfit(zeros[j], poles, k, sh, vh); !(e <= math.Max(2*misfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], sh, vh), 1e-9)) {
			return nil, nil, nil, false
		}
	}
	return zeros, poles, gains, true
}

// zpkMarkovGain returns the leading Markov parameter c·A^(r−1)·b of input
// j of the single-output row (D when r = 0), the gain of a channel of
// relative degree r; ok is false when it cancels to below 1e-6 of its
// magnitude bound |c|·|A|^(r−1)·|b|, so that rounding could dominate it.
func zpkMarkovGain(row *System, j, r int) (float64, bool) {
	if r < 0 {
		return 0, false
	}
	if r == 0 {
		d := row.D.At(0, j)
		return d, d != 0
	}
	a := row.A.RawMatrix()
	b := row.B.RawMatrix()
	c := row.C.RawMatrix()
	n := a.Rows
	v := make([]float64, n)
	w := make([]float64, n)
	for k := range n {
		v[k] = b.Data[k*b.Stride+j]
		w[k] = math.Abs(v[k])
	}
	nv, nw := make([]float64, n), make([]float64, n)
	for range r - 1 {
		for k := range n {
			var s, t float64
			for l, x := range a.Data[k*a.Stride : k*a.Stride+n] {
				s += x * v[l]
				t += math.Abs(x) * w[l]
			}
			nv[k], nw[k] = s, t
		}
		v, nv = nv, v
		w, nw = nw, w
	}
	var k, bound float64
	for l := range n {
		k += c.Data[l] * v[l]
		bound += math.Abs(c.Data[l]) * w[l]
	}
	if math.IsNaN(k) || math.IsInf(bound, 0) || !(math.Abs(k) > 1e-6*bound) {
		return 0, false
	}
	return k, true
}

func zpkFiniteNonzero(v complex128) bool {
	return v != 0 && !cmplx.IsNaN(v) && !cmplx.IsInf(v)
}

// zpkBandProbePoint returns a point at the largest nonzero root modulus of
// the continuous roots (log(r)/dt for discrete roots), at the angle from
// 0.25π to 0.75π farthest from every root, mapped back to z = e^{s·dt}. The
// low-frequency probes of matchedProbePoints fall inside a split cluster at
// s = 0 or z = 1, where every candidate misfits by O(1); this point checks
// the response where the roots are well separated.
func zpkBandProbePoint(dt float64, sets [][]complex128) complex128 {
	var roots []complex128
	rho := 0.0
	for _, set := range sets {
		for _, r := range set {
			if dt > 0 {
				if !(cmplx.Abs(r) > 0 && (real(r) > 0 || math.Abs(imag(r)) > 1e-10*cmplx.Abs(r))) {
					continue
				}
				r = cmplx.Log(r) / complex(dt, 0)
			}
			roots = append(roots, r)
			rho = math.Max(rho, cmplx.Abs(r))
		}
	}
	if !(rho > 0) || math.IsInf(rho, 0) {
		rho = 1
	}
	if dt > 0 {
		rho = math.Min(rho, 0.5*math.Pi/dt)
	}
	var best complex128
	score := -1.0
	for _, theta := range []float64{0.25, 0.4, 0.5, 0.6, 0.75} {
		s := cmplx.Rect(rho, theta*math.Pi)
		d := math.Inf(1)
		for _, r := range roots {
			d = math.Min(d, cmplx.Abs(s-r)/rho)
		}
		if d > score {
			best, score = s, d
		}
	}
	if dt > 0 {
		return cmplx.Exp(best * complex(dt, 0))
	}
	return best
}
