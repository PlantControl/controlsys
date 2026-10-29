package controlsys

import (
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkPencilRows replaces, row by row, the polynomial roots of z (the ZPK of
// the transfer function of sys, with row degrees deg) by the roots MATLAB
// zpk(ss) takes: poles eig(A) of the row's structurally minimal part and,
// per channel, the invariant zeros of that part, with the leading Markov
// parameter as gain (a fit to the response at a low-frequency probe when
// that parameter cancels). Roots of the transfer-function polynomials lose
// tight clusters, e.g. discrete poles near z = 1 leave the unit circle
// (HP3RZO). The per-channel orders stay those of the transfer function: a
// row is replaced only when the structurally minimal part has its row
// degree, every channel keeps its zero count, and each channel fits the
// response, at the low-frequency probe and at a probe on the scale of its
// largest root, no worse than its polynomial roots do. Descriptor models
// keep the polynomial roots.
func (sys *System) zpkPencilRows(z *ZPK, deg []int) {
	n, m, p := sys.Dims()
	if n == 0 || sys.IsDescriptor() || sys.HasDelay() || sys.LFT != nil || len(deg) != p {
		return
	}
	if requireFiniteSystem("ZPKModel", sys) != nil {
		return
	}
	w := newZPKPencilWork(n, m)
	for i := range p {
		if deg[i] == 0 {
			continue
		}
		zeros, poles, gains, ok := sys.zpkPencilRow(z, i, deg[i], w)
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

// zpkPencilWork holds the buffers zpkPencilRow reuses across rows and
// channels; full caches eig(A), the poles of every row of full degree.
type zpkPencilWork struct {
	full   []complex128
	c, d   []float64
	b      []float64
	dch    []float64
	zeros  [][]complex128
	gains  []float64
	g      []complex128
	roots  [][]complex128
	probe  []complex128
	markov []float64
	zw     zerosWork
}

func newZPKPencilWork(n, m int) *zpkPencilWork {
	f := make([]float64, 2*n+m+1+m+4*n)
	return &zpkPencilWork{
		c:      f[:n:n],
		d:      f[n : n+m : n+m],
		b:      f[n+m : 2*n+m : 2*n+m],
		dch:    f[2*n+m : 2*n+m+1 : 2*n+m+1],
		gains:  f[2*n+m+1 : 2*n+2*m+1 : 2*n+2*m+1],
		markov: f[2*n+2*m+1:],
		zeros:  make([][]complex128, m),
		g:      make([]complex128, 3*m),
		roots:  make([][]complex128, 0, 1+3*m),
	}
}

// zpkPencilRow returns the pencil roots and gains of row i of z, or ok
// false. The returned zeros and gains slices are w's, valid until the next
// call.
func (sys *System) zpkPencilRow(z *ZPK, i, deg int, w *zpkPencilWork) (zeros [][]complex128, poles []complex128, gains []float64, ok bool) {
	n, m, p := sys.Dims()
	row := sys
	if p > 1 {
		cr, dr := sys.C.RawMatrix(), sys.D.RawMatrix()
		copy(w.c, cr.Data[i*cr.Stride:i*cr.Stride+n])
		copy(w.d, dr.Data[i*dr.Stride:i*dr.Stride+m])
		row = &System{A: sys.A, B: sys.B, C: mat.NewDense(1, n, w.c), D: mat.NewDense(1, m, w.d), Dt: sys.Dt}
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
		if poles, ok = matchedConjugateClosed(poles); !ok {
			return nil, nil, nil, false
		}
	} else {
		if w.full == nil {
			all, err := sys.Poles()
			if err != nil {
				return nil, nil, nil, false
			}
			if all, ok = matchedConjugateClosed(all); !ok {
				return nil, nil, nil, false
			}
			w.full = all
		}
		poles = w.full
	}
	if len(poles) != deg {
		return nil, nil, nil, false
	}
	nr := deg
	zeros, gains = w.zeros, w.gains
	clear(zeros)
	clear(gains)
	roots := append(w.roots[:0], poles)
	ch := row
	bRaw := row.B.RawMatrix()
	if m > 1 {
		ch = &System{A: row.A, B: mat.NewDense(nr, 1, w.b[:nr]), C: row.C, D: mat.NewDense(1, 1, w.dch), Dt: sys.Dt}
	}
	dRaw := row.D.RawMatrix()
	active := false
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		if len(z.Poles[i][j]) != deg {
			return nil, nil, nil, false
		}
		if m > 1 {
			for k := range nr {
				w.b[k] = bRaw.Data[k*bRaw.Stride+j]
			}
			w.dch[0] = dRaw.Data[j]
		}
		res, err := mimoZerosWork(ch, &w.zw)
		if err != nil || len(res.Zeros) != len(z.Zeros[i][j]) {
			return nil, nil, nil, false
		}
		if zeros[j], ok = matchedConjugateClosed(res.Zeros); !ok {
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
	w.roots = roots
	s0, s1, sh := zpkProbePoints(sys.Dt, roots, &w.probe)
	eval := newFrequencyEvaluator(row).pointSolver(1).evalInto
	g1, gh, g0 := w.g[:m], w.g[m:2*m], w.g[2*m:]
	if evalWithPoleLimit(eval, row, pointAt(s1), g1) != nil {
		return nil, nil, nil, false
	}
	if evalWithPoleLimit(eval, row, pointAt(sh), gh) != nil {
		return nil, nil, nil, false
	}
	haveG0 := false
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		v1, vh := g1[j], gh[j]
		if !zpkFiniteNonzero(v1) || !zpkFiniteNonzero(vh) {
			return nil, nil, nil, false
		}
		k, ok := zpkMarkovGain(row, j, deg-len(zeros[j]), w.markov)
		if !ok {
			if !haveG0 {
				if evalWithPoleLimit(eval, row, pointAt(s0), g0) != nil {
					return nil, nil, nil, false
				}
				haveG0 = true
			}
			v0 := g0[j]
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
		if e := zpkMisfit(zeros[j], poles, k, s1, v1); !(e <= math.Max(2*zpkMisfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], s1, v1), 1e-12)) {
			return nil, nil, nil, false
		}
		// eig and tzero carry backward error amplified by root sensitivity;
		// 1e-9 tolerates that and still rejects an O(1) wrong gain.
		if e := zpkMisfit(zeros[j], poles, k, sh, vh); !(e <= math.Max(2*zpkMisfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], sh, vh), 1e-9)) {
			return nil, nil, nil, false
		}
	}
	return zeros, poles, gains, true
}

func zpkMisfit(zs, ps []complex128, k float64, s, v complex128) float64 {
	return cmplx.Abs(complex(k, 0)*matchedRootRatio(zs, ps, s)-v) / cmplx.Abs(v)
}

// zpkMarkovGain returns the leading Markov parameter c·A^(r−1)·b of input
// j of the single-output row (D when r = 0), the gain of a channel of
// relative degree r; ok is false when it cancels to below 1e-6 of its
// magnitude bound |c|·|A|^(r−1)·|b|, so that rounding could dominate it.
func zpkMarkovGain(row *System, j, r int, buf []float64) (float64, bool) {
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
	v, w := buf[:n], buf[n:2*n]
	for k := range n {
		v[k] = b.Data[k*b.Stride+j]
		w[k] = math.Abs(v[k])
	}
	nv, nw := buf[2*n:3*n], buf[3*n:4*n]
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

var zpkProbeAngles = func() (u [5]complex128) {
	for k, theta := range []float64{0.25, 0.4, 0.5, 0.6, 0.75} {
		u[k] = cmplx.Rect(1, theta*math.Pi)
	}
	return u
}()

// zpkProbePoints returns s0 and s1 of matchedProbePoints (matchedD2CProbePoints
// when dt > 0) and a band point sh: on the continuous roots (log(r)/dt for
// discrete roots), at the largest nonzero root modulus and the angle from
// 0.25π to 0.75π farthest from every root, mapped back to z = e^{s·dt}. The
// low-frequency probes fall inside a split cluster at s = 0 or z = 1, where
// every candidate misfits by O(1); sh checks the response where the roots
// are well separated.
func zpkProbePoints(dt float64, sets [][]complex128, scratch *[]complex128) (s0, s1, sh complex128) {
	roots := (*scratch)[:0]
	defer func() { *scratch = roots }()
	lo, hi := math.Inf(1), 0.0
	for k, set := range sets {
		if slices.ContainsFunc(sets[:k], func(t []complex128) bool { return slices.Equal(t, set) }) {
			continue
		}
		for _, r := range set {
			if dt > 0 {
				if !(cmplx.Abs(r) > 0 && (real(r) > 0 || math.Abs(imag(r)) > 1e-10*cmplx.Abs(r))) {
					continue
				}
				r = cmplx.Log(r) / complex(dt, 0)
			}
			roots = append(roots, r)
			if a := cmplx.Abs(r); a > 0 {
				lo, hi = math.Min(lo, a), math.Max(hi, a)
			}
		}
	}
	if math.IsInf(lo, 1) {
		lo = 1
	}
	if !(hi > 0) || math.IsInf(hi, 0) {
		hi = 1
	}
	if dt > 0 {
		hi = math.Min(hi, 0.5*math.Pi/dt)
	}
	far := func(rho float64, u complex128) (complex128, float64) {
		s := complex(rho, 0) * u
		d := math.Inf(1)
		for _, r := range roots {
			e := s - r
			if q := real(e)*real(e) + imag(e)*imag(e); q < d {
				d = q
			}
		}
		return s, d / (rho * rho)
	}
	score := [2]float64{-1, -1}
	for _, f := range []float64{0.5, 1, 2} {
		for _, u := range zpkProbeAngles {
			s, d := far(f*lo, u)
			switch {
			case d > score[0]:
				s1, score[1] = s0, score[0]
				s0, score[0] = s, d
			case d > score[1]:
				s1, score[1] = s, d
			}
		}
	}
	best := -1.0
	for _, u := range zpkProbeAngles {
		if s, d := far(hi, u); d > best {
			sh, best = s, d
		}
	}
	if dt > 0 {
		t := complex(dt, 0)
		return cmplx.Exp(s0 * t), cmplx.Exp(s1 * t), cmplx.Exp(sh * t)
	}
	return s0, s1, sh
}
