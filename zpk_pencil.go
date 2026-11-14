package controlsys

import (
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkModelRows sets, row by row, the zero-pole-gain form of z (the ZPK of
// the transfer function of sys, with row degrees deg) for an explicit,
// delay-free, finite model, and returns the first row no candidate verifies
// against the stored model's response (zpkRowCheck), −1 when every row
// verifies or sys is not such a model. A row of degree 0 is the constant
// Z = D: nil when its structurally minimal part has no states, else
// verified as any candidate (its dynamics lie under TransferFunction's
// rank floor). A row of nonzero degree first tries the form ZPKModel
// selected before it verified rows, (a) when it fits ref as well as the
// polynomial roots (zpkPencilFitsAsWell) and (d) otherwise, so a row that
// verifies keeps that result unchanged; then the candidates, in order, the
// first verified kept:
//
//   - (a) the roots MATLAB zpk(ss) takes on the row's structurally minimal
//     part ref (Sminreal): poles eig(A), per channel the invariant zeros of
//     the system pencil and the leading Markov parameter as gain (a fit to
//     the response at a low-frequency probe when that parameter cancels).
//     A part above the row degree, whose extra modes are hidden
//     uncontrollable or unobservable ones, is reduced by Reduce with its
//     default tolerance, as MATLAB zpk(ss) cancels them; the reduction is
//     kept only when it adds no unstable pole to eig(A) of ref or to the
//     polynomial poles (zpkReducedStabilityKept);
//   - (b) for a part above the row degree, eig(A) and the invariant zeros of
//     ref with the closest pole-zero pairs cancelled, each a certified
//     dropped factor (zpkCancelledRow);
//   - (c) for a part of the row degree, eig(A) of ref with the
//     transfer-function zeros and a gain fitted at a low-frequency probe;
//   - (d) the roots of the transfer-function polynomials, which lose tight
//     root clusters, e.g. discrete poles near z = 1 leave the unit circle
//     (HP3RZO).
//
// Each channel keeps the orders of the transfer function. A row no
// candidate verifies keeps the first candidate tried; every candidate has the
// transfer function's orders, so the result keeps RowDegrees.
func (sys *System) zpkModelRows(z *ZPK, deg []int) int {
	n, m, p := sys.Dims()
	if n == 0 || sys.IsDescriptor() || sys.HasDelay() || sys.LFT != nil || len(deg) != p {
		return -1
	}
	if requireFiniteSystem("ZPKModel", sys) != nil {
		return -1
	}
	w := newZPKPencilWork(n, m)
	bad := -1
	for i := range p {
		if !sys.zpkModelRow(z, i, deg[i], w) && bad < 0 {
			bad = i
		}
	}
	return bad
}

// zpkCandidate is a zero-pole-gain form of one row: per channel its zeros,
// poles and gain; win caches its shared-pole windows (zpkRowCheck).
type zpkCandidate struct {
	zeros, poles [][]complex128
	gains        []float64
	win          *poleWindows
}

// zpkRowCandidate returns the candidate of row i of z with the roots and
// gains given for the channels of nonzero transfer-function gain and z's
// own for the others.
func zpkRowCandidate(z *ZPK, i int, zeros, poles [][]complex128, gains []float64) *zpkCandidate {
	m := len(z.Gain[i])
	c := &zpkCandidate{zeros: make([][]complex128, m), poles: make([][]complex128, m), gains: make([]float64, m)}
	for j := range m {
		if z.Gain[i][j] == 0 || zeros == nil {
			c.zeros[j], c.poles[j], c.gains[j] = copyComplex(z.Zeros[i][j]), copyComplex(z.Poles[i][j]), z.Gain[i][j]
			continue
		}
		c.zeros[j], c.poles[j], c.gains[j] = copyComplex(zeros[j]), copyComplex(poles[j]), gains[j]
	}
	return c
}

func (sys *System) zpkModelRow(z *ZPK, i, deg int, w *zpkPencilWork) bool {
	n, m, p := sys.Dims()
	row := sys
	if p > 1 {
		cr, dr := sys.C.RawMatrix(), sys.D.RawMatrix()
		c, d := make([]float64, n), make([]float64, m)
		copy(c, cr.Data[i*cr.Stride:i*cr.Stride+n])
		copy(d, dr.Data[i*dr.Stride:i*dr.Stride+m])
		row = &System{A: sys.A, B: sys.B, C: mat.NewDense(1, n, c), D: mat.NewDense(1, m, d), Dt: sys.Dt}
	}
	ref, err := Sminreal(row)
	if err != nil {
		return false
	}
	nr, _, _ := ref.Dims()
	if deg == 0 {
		dRaw := row.D.RawMatrix()
		c := &zpkCandidate{zeros: make([][]complex128, m), poles: make([][]complex128, m), gains: make([]float64, m)}
		for j := range m {
			c.gains[j] = dRaw.Data[j]
			z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j] = nil, nil, c.gains[j]
		}
		if nr == 0 {
			return true
		}
		check, ok := newZPKRowCheck(ref)
		return ok && check.check(c) == zpkVerified
	}
	check, ok := newZPKRowCheck(ref)
	var first *zpkCandidate
	try := func(c *zpkCandidate) bool {
		if c == nil {
			return false
		}
		if first == nil {
			first = c
		}
		return ok && check.check(c) == zpkVerified
	}
	keep := func(c *zpkCandidate) {
		for j := range m {
			z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j] = c.zeros[j], c.poles[j], c.gains[j]
		}
	}
	pencil := sys.zpkPencilRow(z, i, deg, ref, w)
	tf := zpkRowCandidate(z, i, nil, nil, nil)
	prior := tf
	if pencil != nil && sys.zpkPencilFitsAsWell(z, i, ref, pencil, w) {
		prior = pencil
	}
	cands := []func() *zpkCandidate{
		func() *zpkCandidate { return prior },
		func() *zpkCandidate {
			if pencil == prior {
				return nil
			}
			return pencil
		},
		func() *zpkCandidate {
			if nr <= deg || !ok {
				return nil
			}
			return zpkCancelledRow(z, i, deg, ref, check, w)
		},
		func() *zpkCandidate {
			if nr != deg || !ok {
				return nil
			}
			return zpkEigenTFZerosRow(z, i, ref, check, w)
		},
		func() *zpkCandidate {
			if tf == prior {
				return nil
			}
			return tf
		},
	}
	for k, f := range cands {
		if c := f(); try(c) {
			if zpkRouteHook != nil {
				zpkRouteHook(i, k)
			}
			keep(c)
			return true
		}
	}
	if zpkRouteHook != nil {
		zpkRouteHook(i, -1)
	}
	keep(first)
	return false
}

// zpkRouteHook is a test-only seam, nil outside tests: it sees, per row, the
// index of the verified candidate, −1 when none verifies.
var zpkRouteHook func(row, candidate int)

// zpkPencilWork holds the buffers zpkPencilRow reuses across rows and
// channels; full caches eig(A), the poles of every row whose structurally
// minimal part is the whole model.
type zpkPencilWork struct {
	full   []complex128
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
	f := make([]float64, n+1+m+4*n)
	return &zpkPencilWork{
		b:      f[:n:n],
		dch:    f[n : n+1 : n+1],
		gains:  f[n+1 : n+m+1 : n+m+1],
		markov: f[n+m+1:],
		zeros:  make([][]complex128, m),
		g:      make([]complex128, 3*m),
		roots:  make([][]complex128, 0, 1+3*m),
	}
}

// zpkPencilRow returns candidate (a) of row i of z, ref the row's
// structurally minimal part, or nil when the reduced part does not have the
// row degree, a channel's zero count differs from the transfer function's,
// or a gain is not determined.
func (sys *System) zpkPencilRow(z *ZPK, i, deg int, ref *System, w *zpkPencilWork) *zpkCandidate {
	n, _, _ := sys.Dims()
	nr, _, _ := ref.Dims()
	if nr < deg {
		return nil
	}
	row := ref
	var poles []complex128
	var ok bool
	if nr > deg {
		red, err := ref.Reduce(nil)
		if err != nil || red.Order != deg {
			return nil
		}
		row = red.Sys
		if poles, err = row.Poles(); err != nil {
			return nil
		}
		if poles, ok = matchedConjugateClosed(poles); !ok {
			return nil
		}
		if !zpkReducedStabilityKept(ref, poles, z.Poles[i], sys.Dt) {
			return nil
		}
	} else if nr == n {
		if w.full == nil {
			all, err := sys.Poles()
			if err != nil {
				return nil
			}
			if all, ok = matchedConjugateClosed(all); !ok {
				return nil
			}
			w.full = all
		}
		poles = w.full
	} else {
		var err error
		if poles, err = ref.Poles(); err != nil {
			return nil
		}
		if poles, ok = matchedConjugateClosed(poles); !ok {
			return nil
		}
	}
	if len(poles) != deg {
		return nil
	}
	zeros, ps, gains, ok := sys.zpkPencilChannels(z, i, deg, row, ref, poles, nil, w)
	if !ok {
		return nil
	}
	return zpkRowCandidate(z, i, zeros, ps, gains)
}

// zpkPencilChannels returns, for each channel of row i of nonzero
// transfer-function gain, the invariant zeros of that channel of row (poles
// its eigenvalues), the channel's poles and its gain: the leading Markov
// parameter of ref, of relative degree nr − (zero count of row), or a fit
// to ref's response at a low-frequency probe when it cancels. drop, when
// non-nil, replaces a channel's zeros and poles (zpkCancelledRow) before
// the gain fit. False when a channel's zero count differs from the
// transfer function's or a gain is not determined.
func (sys *System) zpkPencilChannels(z *ZPK, i, deg int, row, ref *System, poles []complex128, drop func(j int, zeros []complex128) ([]complex128, []complex128, bool), w *zpkPencilWork) ([][]complex128, [][]complex128, []float64, bool) {
	_, m, _ := sys.Dims()
	nr, _, _ := row.Dims()
	zeros, chPoles, gains := make([][]complex128, m), make([][]complex128, m), make([]float64, m)
	roots := append(w.roots[:0], poles)
	ch := row
	bRaw := row.B.RawMatrix()
	if m > 1 {
		ch = &System{A: row.A, B: mat.NewDense(nr, 1, w.b[:nr]), C: row.C, D: mat.NewDense(1, 1, w.dch), Dt: sys.Dt}
	}
	dRaw := row.D.RawMatrix()
	active := false
	rel := make([]int, m)
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
		if err != nil {
			return nil, nil, nil, false
		}
		zs, ok := matchedConjugateClosed(res.Zeros)
		if !ok {
			return nil, nil, nil, false
		}
		rel[j] = nr - len(zs)
		ps := poles
		if drop != nil {
			if zs, ps, ok = drop(j, zs); !ok {
				return nil, nil, nil, false
			}
		}
		if len(zs) != len(z.Zeros[i][j]) || len(ps) != deg {
			return nil, nil, nil, false
		}
		zeros[j], chPoles[j] = zs, ps
		roots = append(roots, zs, z.Zeros[i][j])
		if drop != nil {
			roots = append(roots, ps)
		}
		active = true
	}
	if !active {
		return nil, nil, nil, false
	}
	for j := range m {
		roots = append(roots, z.Poles[i][j])
	}
	w.roots = roots
	s0, _, _ := zpkProbePoints(sys.Dt, roots, &w.probe)
	eval := newFrequencyEvaluator(ref).pointSolver(1).evalInto
	g0 := w.g[2*m:]
	haveG0 := false
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		k, ok := zpkMarkovGain(ref, j, rel[j], w.markov)
		if !ok {
			if !haveG0 {
				if evalWithPoleLimit(eval, ref, pointAt(s0), g0) != nil {
					return nil, nil, nil, false
				}
				haveG0 = true
			}
			if k, ok = zpkFittedGain(zeros[j], chPoles[j], s0, g0[j]); !ok {
				return nil, nil, nil, false
			}
		}
		gains[j] = k
	}
	return zeros, chPoles, gains, true
}

// zpkFittedGain is the real gain k with k·Π(s0 − zeros)/Π(s0 − poles) = v,
// false when v is zero or not finite or the quotient is not real to 1e-7.
func zpkFittedGain(zeros, poles []complex128, s0, v complex128) (float64, bool) {
	if !zpkFiniteNonzero(v) {
		return 0, false
	}
	q := v / matchedRootRatio(zeros, poles, s0)
	if cmplx.IsNaN(q) || cmplx.IsInf(q) || real(q) == 0 || !(math.Abs(imag(q)) <= 1e-7*cmplx.Abs(q)) {
		return 0, false
	}
	return real(q), true
}

// zpkPencilFitsAsWell reports whether the pencil candidate c of row i fits
// the response of ref, at the low-frequency probe and at a probe on the
// scale of its largest root (zpkProbePoints), no worse than the
// transfer-function roots of z do, to 1e-12 and 1e-9 relative: the
// selection ZPKModel made before it verified rows, whose result is tried
// first, so a row that verifies keeps it unchanged.
func (sys *System) zpkPencilFitsAsWell(z *ZPK, i int, ref *System, c *zpkCandidate, w *zpkPencilWork) bool {
	_, m, _ := sys.Dims()
	var poles []complex128
	roots := w.roots[:0]
	for j := range m {
		if z.Gain[i][j] != 0 {
			poles = c.poles[j]
			break
		}
	}
	roots = append(roots, poles)
	for j := range m {
		if z.Gain[i][j] != 0 {
			roots = append(roots, c.zeros[j], z.Zeros[i][j])
		}
	}
	for j := range m {
		roots = append(roots, z.Poles[i][j])
	}
	w.roots = roots
	_, s1, sh := zpkProbePoints(sys.Dt, roots, &w.probe)
	eval := newFrequencyEvaluator(ref).pointSolver(1).evalInto
	g1, gh := w.g[:m], w.g[m:2*m]
	if evalWithPoleLimit(eval, ref, pointAt(s1), g1) != nil || evalWithPoleLimit(eval, ref, pointAt(sh), gh) != nil {
		return false
	}
	misfit := func(zs, ps []complex128, k float64, s, v complex128) float64 {
		return cmplx.Abs(complex(k, 0)*matchedRootRatio(zs, ps, s)-v) / cmplx.Abs(v)
	}
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		v1, vh := g1[j], gh[j]
		if !zpkFiniteNonzero(v1) || !zpkFiniteNonzero(vh) {
			return false
		}
		if e := misfit(c.zeros[j], c.poles[j], c.gains[j], s1, v1); !(e <= math.Max(2*misfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], s1, v1), 1e-12)) {
			return false
		}
		if e := misfit(c.zeros[j], c.poles[j], c.gains[j], sh, vh); !(e <= math.Max(2*misfit(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], sh, vh), 1e-9)) {
			return false
		}
	}
	return true
}

// zpkCancelledRow returns candidate (b) of row i of z: the poles eig(A) of
// ref, of order nr above the row degree, and per channel its invariant
// zeros, with nr − deg pole-zero pairs cancelled, closest first by
// |p − z|/max(d(p), ρ₀), d the distance of p to the axis (|Re p|, or
// ||p| − 1| for a discrete model) and ρ₀ the resolution of ref
// (zpkRowCheck), a conjugate pair with its conjugates. Each cancelled pair
// must satisfy |p − z| ≤ tol·d(p): then |(s − z)/(s − p) − 1| ≤ |p − z|/|s − p|
// ≤ tol at every axis point, a certified dropped factor. Nil when a channel
// has too few zeros, a pair is not so certified, or a gain is not
// determined. The Markov gain is that of ref, unchanged by the cancellation.
func zpkCancelledRow(z *ZPK, i, deg int, ref *System, check *zpkRowCheck, w *zpkPencilWork) *zpkCandidate {
	nr, _, _ := ref.Dims()
	all, ok := matchedConjugateClosed(check.evals)
	if !ok || len(all) != nr {
		return nil
	}
	tol, discrete := math.Sqrt(eps()), ref.Dt != 0
	axisDist := func(p complex128) float64 {
		if discrete {
			return math.Abs(cmplx.Abs(p) - 1)
		}
		return math.Abs(real(p))
	}
	drop := func(_ int, zs []complex128) ([]complex128, []complex128, bool) {
		ps, zs := slices.Clone(all), slices.Clone(zs)
		for need := nr - deg; need > 0; {
			bi, bk, best := -1, -1, math.Inf(1)
			for a, p := range ps {
				if imag(p) < 0 {
					continue
				}
				for b, q := range zs {
					if imag(q) < 0 || (imag(p) > 0) != (imag(q) > 0) {
						continue
					}
					if sc := cmplx.Abs(p-q) / math.Max(axisDist(p), check.rho0); sc < best {
						bi, bk, best = a, b, sc
					}
				}
			}
			if bi < 0 {
				return nil, nil, false
			}
			p, q := ps[bi], zs[bk]
			if !(cmplx.Abs(p-q) <= tol*axisDist(p)) {
				return nil, nil, false
			}
			ps, zs = slices.Delete(ps, bi, bi+1), slices.Delete(zs, bk, bk+1)
			need--
			if imag(p) > 0 {
				ci, ck := zpkNearest(ps, cmplx.Conj(p)), zpkNearest(zs, cmplx.Conj(q))
				if need == 0 || ci < 0 || ck < 0 {
					return nil, nil, false
				}
				ps, zs = slices.Delete(ps, ci, ci+1), slices.Delete(zs, ck, ck+1)
				need--
			}
		}
		return zs, ps, true
	}
	zeros, poles, gains, ok := ref.zpkPencilChannels(z, i, deg, ref, ref, all, drop, w)
	if !ok {
		return nil
	}
	return zpkRowCandidate(z, i, zeros, poles, gains)
}

// zpkNearest is the index of the root of rs nearest r, −1 for none.
func zpkNearest(rs []complex128, r complex128) int {
	best, d := -1, math.Inf(1)
	for k, q := range rs {
		if e := cmplx.Abs(q - r); e < d {
			best, d = k, e
		}
	}
	return best
}

// zpkEigenTFZerosRow returns candidate (c) of row i of z: the poles eig(A)
// of ref, of the row degree, with the transfer-function zeros of each
// channel and its gain fitted at a low-frequency probe, nil when the gain
// is not determined.
func zpkEigenTFZerosRow(z *ZPK, i int, ref *System, check *zpkRowCheck, w *zpkPencilWork) *zpkCandidate {
	_, m, _ := ref.Dims()
	poles, ok := matchedConjugateClosed(check.evals)
	if !ok {
		return nil
	}
	roots := append(w.roots[:0], poles)
	for j := range m {
		roots = append(roots, z.Zeros[i][j], z.Poles[i][j])
	}
	w.roots = roots
	s0, _, _ := zpkProbePoints(ref.Dt, roots, &w.probe)
	g0 := w.g[2*m:]
	eval := newFrequencyEvaluator(ref).pointSolver(1).evalInto
	if evalWithPoleLimit(eval, ref, pointAt(s0), g0) != nil {
		return nil
	}
	zeros, ps, gains := make([][]complex128, m), make([][]complex128, m), make([]float64, m)
	for j := range m {
		if z.Gain[i][j] == 0 {
			continue
		}
		k, ok := zpkFittedGain(z.Zeros[i][j], poles, s0, g0[j])
		if !ok {
			return nil
		}
		zeros[j], ps[j], gains[j] = z.Zeros[i][j], poles, k
	}
	return zpkRowCandidate(z, i, zeros, ps, gains)
}

// zpkReducedStabilityKept reports whether the poles of the reduced row have
// no more unstable roots than eig(A) of its structurally minimal part ref and
// than the polynomial poles of every channel of the row. A reduction that
// cuts genuine modes of a cluster under-resolved at float64, rather than
// hidden ones, moves the kept poles; the response gate cannot see it where
// the polynomial roots misfit by O(1) as well.
func zpkReducedStabilityKept(ref *System, poles []complex128, tf [][]complex128, dt float64) bool {
	u := unstableModalCount(poles, dt)
	if u == 0 {
		return true
	}
	for _, ps := range tf {
		if len(ps) > 0 && u > unstableModalCount(ps, dt) {
			return false
		}
	}
	all, err := ref.Poles()
	return err == nil && u <= unstableModalCount(all, dt)
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
