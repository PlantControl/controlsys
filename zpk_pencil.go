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
	for i := range p {
		if deg[i] == 0 {
			continue
		}
		zeros, poles, gains, ok := sys.zpkPencilRow(z, i, deg[i], m)
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

func (sys *System) zpkPencilRow(z *ZPK, i, deg, m int) (zeros [][]complex128, poles []complex128, gains []float64, ok bool) {
	n := sys.A.RawMatrix().Rows
	c := mat.NewDense(1, n, nil)
	c.Copy(sys.C.Slice(i, i+1, 0, n))
	d := mat.NewDense(1, m, nil)
	if sys.D != nil {
		d.Copy(sys.D.Slice(i, i+1, 0, m))
	}
	row, err := New(sys.A, sys.B, c, d, sys.Dt)
	if err != nil {
		return nil, nil, nil, false
	}
	if row, err = Sminreal(row); err != nil {
		return nil, nil, nil, false
	}
	nr, _, _ := row.Dims()
	if nr != deg {
		return nil, nil, nil, false
	}
	poles, err = row.Poles()
	if err != nil || len(poles) != deg {
		return nil, nil, nil, false
	}
	if poles, ok = matchedConjugateClosed(poles); !ok {
		return nil, nil, nil, false
	}
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
		b := mat.NewDense(nr, 1, nil)
		b.Copy(row.B.Slice(0, nr, j, j+1))
		ch, err := New(row.A, b, row.C, mat.NewDense(1, 1, []float64{row.D.At(0, j)}), sys.Dt)
		if err != nil {
			return nil, nil, nil, false
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
	g0, err := row.EvalFr(s0)
	if err != nil {
		return nil, nil, nil, false
	}
	g1, err := row.EvalFr(s1)
	if err != nil {
		return nil, nil, nil, false
	}
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		v0, v1 := g0[0][j], g1[0][j]
		if v0 == 0 || v1 == 0 || cmplx.IsNaN(v0) || cmplx.IsNaN(v1) || cmplx.IsInf(v0) || cmplx.IsInf(v1) {
			return nil, nil, nil, false
		}
		q := v0 / matchedRootRatio(zeros[j], poles, s0)
		if cmplx.IsNaN(q) || cmplx.IsInf(q) || real(q) == 0 || !(math.Abs(imag(q)) <= 1e-7*cmplx.Abs(q)) {
			return nil, nil, nil, false
		}
		misfit := func(zs, ps []complex128, k float64) float64 {
			return cmplx.Abs(complex(k, 0)*matchedRootRatio(zs, ps, s1)-v1) / cmplx.Abs(v1)
		}
		gains[j] = real(q)
		if k, ok := zpkMarkovGain(row, j, deg-len(zeros[j])); ok && math.Abs(k-gains[j]) <= 1e-6*math.Abs(gains[j]) {
			gains[j] = k
		}
		e := misfit(zeros[j], poles, gains[j])
		if !(e <= math.Max(2*misfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j]), 1e-12)) {
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
