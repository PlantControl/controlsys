package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// errDelayLoopUnsupported reports a delay loop the Nyquist test cannot
// decide; it wraps ErrDelayUnsupported.
var errDelayLoopUnsupported = fmt.Errorf("delay loop outside Nyquist test scope: %w", ErrDelayUnsupported)

// delayLoop is a continuous SISO loop L whose negative-feedback closed-loop
// characteristic is χ(s) = det(sI-A)·(1+L(s)), with χ entire. Its closed-loop
// RHP root count is rhp - W, where W is the winding of 1+L around the origin
// along the jω axis indented to the right of the imaginary-axis eigenvalues.
type delayLoop struct {
	sys       *System // the loop model, for certified sensitivity peaks
	at        func(w float64) complex128
	eval      *sisoEval // records the first failure of at
	rhp       int
	axis      []axisPole
	tail      func(w float64) float64 // bound on |L(s)| for Re s >= 0, |s| >= w
	tailLimit float64                 // bound on limsup |L(jω)|, ω → ∞
	tau       float64                 // total loop delay, sets the grid density
	scales    []float64
	maxPoints int // grid point budget; 0 selects delayNyquistMaxPoints
}

type axisPole struct {
	w    float64
	mult int
}

// delayNyquist is the verdict of the Nyquist test. With axisRoot set, χ has a
// root on the imaginary axis at ±j·axisW, roots is not counted and w, f are
// incomplete; otherwise roots is the closed-loop RHP root count and w, f
// cover the whole grid.
type delayNyquist struct {
	stable   bool
	roots    int
	axisRoot bool
	axisW    float64
	tailFrom float64 // the tail bound certifies |L| <= lin from here on
	w        []float64
	f        []complex128 // 1+L(jω) on the refined grid
}

const delayNyquistMaxPoints = 1 << 21

// delayLoopFromSystem describes the loop sys for the Nyquist test. Internal
// delays must form an acyclic network, so that the open-loop poles are
// exactly eig(A); a delay feedback cycle gives infinitely many open-loop
// poles and is rejected with ErrContinuousInternalDelay.
func delayLoopFromSystem(sys *System, context string) (*delayLoop, error) {
	if err := newDescriptorPolicy(sys).requireStandard(context); err != nil {
		return nil, err
	}
	bound, err := newRationalTailBound(sys)
	if err != nil {
		return nil, err
	}
	eval, err := newSISOEval(sys)
	if err != nil {
		return nil, err
	}
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	l := &delayLoop{sys: sys, at: eval.at, eval: eval, tail: bound.at, tailLimit: bound.at(math.Inf(1)), tau: sisoLoopDelay(sys)}
	l.addPoles(poles)
	return l, nil
}

func sisoLoopDelay(sys *System) float64 {
	tau := 0.0
	if len(sys.InputDelay) > 0 {
		tau += sys.InputDelay[0]
	}
	if len(sys.OutputDelay) > 0 {
		tau += sys.OutputDelay[0]
	}
	if sys.Delay != nil {
		tau += sys.Delay.At(0, 0)
	}
	if sys.LFT != nil {
		for _, t := range sys.LFT.Tau {
			tau += t
		}
	}
	return tau
}

func (l *delayLoop) addPoles(poles []complex128) {
	for _, p := range poles {
		if a := cmplx.Abs(p); a > 0 {
			l.scales = append(l.scales, a, math.Abs(imag(p)))
		}
		switch classifyPoleStability(p, true, poleStabilityTolerance(p)) {
		case poleClassUnstable:
			l.rhp++
		case poleClassBoundary:
			if imag(p) < 0 {
				continue
			}
			l.addAxisPole(imag(p), 1)
		}
	}
}

func (l *delayLoop) addAxisPole(w float64, mult int) {
	if w < 1e-9 {
		w = 0
	}
	for i := range l.axis {
		if math.Abs(l.axis[i].w-w) <= 1e-6*max(1, w) {
			l.axis[i].mult += mult
			return
		}
	}
	l.axis = append(l.axis, axisPole{w: w, mult: mult})
}

// rationalTailBound bounds |L(s)| for Re s >= 0 and |s| >= w from the
// augmented realization H = Daug + Caug(sI-A)^{-1}Baug of the delay LFT,
// using |c(sI-A)^{-1}b| <= ‖c‖‖b‖/(|s|-‖A‖_F), |e^{-sτ}| <= 1 and the finite
// Neumann series of the acyclic delay loop.
type rationalTailBound struct {
	normA float64
	d     []float64 // (1+N)×(1+N) |Daug|
	c, b  []float64 // row norms of Caug, column norms of Baug
	nd    int
}

func newRationalTailBound(sys *System) (*rationalTailBound, error) {
	n, _, _ := sys.Dims()
	nd := sys.internalDelayCount()
	if nd > 0 && !lftDelayLoopAcyclic(sys, n) {
		return nil, fmt.Errorf("internal delay feedback cycle: %w", ErrContinuousInternalDelay)
	}
	k := 1 + nd
	r := &rationalTailBound{d: make([]float64, k*k), c: make([]float64, k), b: make([]float64, k), nd: nd}
	if n > 0 {
		r.normA = mat.Norm(sys.A, 2)
	}
	r.d[0] = math.Abs(sys.D.At(0, 0))
	r.c[0] = rowNorm(sys.C, 0)
	r.b[0] = colNorm(sys.B, 0)
	if nd > 0 {
		lft := sys.LFT
		for i := range nd {
			r.c[1+i] = rowNorm(lft.C2, i)
			r.b[1+i] = colNorm(lft.B2, i)
			if lft.D12 != nil {
				r.d[1+i] = math.Abs(lft.D12.At(0, i))
			}
			if lft.D21 != nil {
				r.d[(1+i)*k] = math.Abs(lft.D21.At(i, 0))
			}
			if lft.D22 != nil {
				for j := range nd {
					r.d[(1+i)*k+1+j] = math.Abs(lft.D22.At(i, j))
				}
			}
		}
	}
	return r, nil
}

func rowNorm(m *mat.Dense, i int) float64 {
	if m == nil {
		return 0
	}
	_, c := m.Dims()
	if c == 0 {
		return 0
	}
	return mat.Norm(m.RowView(i), 2)
}

func colNorm(m *mat.Dense, j int) float64 {
	if m == nil {
		return 0
	}
	r, _ := m.Dims()
	if r == 0 {
		return 0
	}
	return mat.Norm(m.ColView(j), 2)
}

func (r *rationalTailBound) at(w float64) float64 {
	k := 1 + r.nd
	h := make([]float64, k*k)
	for i := range k {
		for j := range k {
			v := r.d[i*k+j]
			if cb := r.c[i] * r.b[j]; cb > 0 && !math.IsInf(w, 1) {
				if w <= r.normA {
					return math.Inf(1)
				}
				v += cb / (w - r.normA)
			}
			h[i*k+j] = v
		}
	}
	bound := h[0]
	if r.nd == 0 {
		return bound
	}
	nd := r.nd
	h22 := make([]float64, nd*nd)
	for i := range nd {
		copy(h22[i*nd:(i+1)*nd], h[(1+i)*k+1:(2+i)*k])
	}
	sum := make([]float64, nd*nd)
	pow := make([]float64, nd*nd)
	for i := range nd {
		sum[i*nd+i] = 1
		pow[i*nd+i] = 1
	}
	next := make([]float64, nd*nd)
	for range nd - 1 {
		clear(next)
		for i := range nd {
			for l := range nd {
				if p := pow[i*nd+l]; p != 0 {
					for j := range nd {
						next[i*nd+j] += p * h22[l*nd+j]
					}
				}
			}
		}
		pow, next = next, pow
		for i := range sum {
			sum[i] += pow[i]
		}
	}
	for i := range nd {
		for j := range nd {
			bound += h[1+i] * sum[i*nd+j] * h[(1+j)*k]
		}
	}
	return bound
}

// lftDelayLoopAcyclic reports whether the delay-to-delay transfer
// D22 + C2(sI-A)^{-1}B2 has a nilpotent sparsity pattern, so that
// det(I - H22(s)Δ(s)) ≡ 1 and the internal delays add no poles.
func lftDelayLoopAcyclic(sys *System, n int) bool {
	lft := sys.LFT
	nd := len(lft.Tau)
	mag := make([]float64, nd*nd)
	if lft.D22 != nil {
		for i := range nd {
			for j := range nd {
				mag[i*nd+j] = math.Abs(lft.D22.At(i, j))
			}
		}
	}
	if n > 0 && lft.B2 != nil && lft.C2 != nil {
		x := mat.DenseCopyOf(lft.B2)
		var markov, next mat.Dense
		for range n {
			markov.Mul(lft.C2, x)
			for i := range nd {
				for j := range nd {
					mag[i*nd+j] += math.Abs(markov.At(i, j))
				}
			}
			next.Mul(sys.A, x)
			x.Copy(&next)
		}
	}
	scale := 0.0
	for _, v := range mag {
		scale = max(scale, v)
	}
	if scale == 0 {
		return true
	}
	adj := make([]bool, nd*nd)
	for i, v := range mag {
		adj[i] = v > 1e-12*scale
	}
	reach := slices.Clone(adj)
	for range nd {
		next := make([]bool, nd*nd)
		for i := range nd {
			for l := range nd {
				if !reach[i*nd+l] {
					continue
				}
				for j := range nd {
					next[i*nd+j] = next[i*nd+j] || adj[l*nd+j]
				}
			}
		}
		reach = next
	}
	return !slices.Contains(reach, true)
}

// nyquist counts closed-loop RHP roots of χ by the argument principle on a
// delay-aware grid: log spacing, linear spacing π/(8τ), points around every
// open-loop pole magnitude, and bisection wherever arg(1+L) moves by more
// than π/4 between neighbours. The grid ends at R where the tail bound
// certifies |L| <= tailLimit + frac(1-tailLimit) < 1 on the rest of the
// closed right half-plane, so no encirclement is missed beyond R. The count
// is exact up to the grid resolving every passage of 1+L near the origin; a
// closed-loop root within the bisection tolerance of the axis, or an axis
// eigenvalue hidden from L, is reported unstable with axisRoot set. A grid that exceeds the
// point budget or a winding number that does not resolve to an integer
// returns errDelayLoopUnsupported rather than a verdict.
func (l *delayLoop) nyquist(frac float64) (delayNyquist, error) {
	return l.nyquistLinearTo(frac, 0)
}

// nyquistLinearTo is nyquist with the linear delay spacing ended at the
// frequency w1 beyond which the tail bound certifies |L| <= lin, when lin
// exceeds the grid-end level. On [w1, R] 1+L stays in the disk of radius
// lin < 1 around 1, so its phase cannot wind and, for lin < sin(π/8), moves by
// less than π/4 between any two points; the count is unchanged and the grid
// is a subset of the full one.
func (l *delayLoop) nyquistLinearTo(frac, lin float64) (delayNyquist, error) {
	res, err := l.nyquistGrid(frac, lin)
	if err := l.evalErr(); err != nil {
		return delayNyquist{}, err
	}
	return res, err
}

func (l *delayLoop) evalErr() error {
	if l.eval == nil {
		return nil
	}
	return l.eval.err
}

func (l *delayLoop) nyquistGrid(frac, lin float64) (delayNyquist, error) {
	if !(l.tailLimit < 1) {
		return delayNyquist{}, fmt.Errorf("neutral delay loop, |L(j∞)| may reach 1: %w", errDelayLoopUnsupported)
	}
	target := l.tailLimit + frac*(1-l.tailLimit)
	hi := 1.0
	for _, s := range l.scales {
		hi = max(hi, 10*s)
	}
	for _, a := range l.axis {
		hi = max(hi, 10*a.w)
	}
	R := hi
	for range 200 {
		if l.tail(R) <= target {
			break
		}
		R *= 2
	}
	if !(l.tail(R) <= target) || math.IsInf(R, 1) {
		return delayNyquist{}, fmt.Errorf("no finite frequency bound for |L|: %w", errDelayLoopUnsupported)
	}
	w1 := R
	if l.tau > 0 && lin > target {
		lo := 0.0
		for range 100 {
			mid := (lo + w1) / 2
			if l.tail(mid) <= lin {
				w1 = mid
			} else {
				lo = mid
			}
		}
	}
	if l.tau > 0 && w1*8*l.tau/math.Pi > float64(l.pointBudget()) {
		return delayNyquist{}, fmt.Errorf("delay grid exceeds %d points: %w", l.pointBudget(), errDelayLoopUnsupported)
	}

	if w, ok := l.hiddenAxisRoot(); ok {
		return delayNyquist{axisRoot: true, axisW: w}, nil
	}
	grid := l.grid(R, w1)
	segments := l.segments(grid, R)
	size := len(grid) + 2*len(segments)
	out := delayNyquist{tailFrom: w1, w: make([]float64, 0, size), f: make([]complex128, 0, size)}
	var phi, phiStart float64
	for si, seg := range segments {
		base := len(out.w)
		var res refineResult
		var wRoot float64
		out.w, out.f, res, wRoot = l.refine(seg, l.pointBudget()-base, out.w, out.f)
		switch res {
		case refineAxisRoot:
			return delayNyquist{axisRoot: true, axisW: wRoot, w: out.w[:base], f: out.f[:base]}, nil
		case refineBudget:
			return delayNyquist{}, fmt.Errorf("refined grid exceeds %d points: %w", l.pointBudget(), errDelayLoopUnsupported)
		}
		fs := out.f[base:]
		if si == 0 {
			if seg.from == 0 {
				phiStart = 0
				if real(fs[0]) < 0 {
					phiStart = math.Pi
				}
				phi = phiStart
			} else {
				m := l.axisMult(0)
				phiStart = math.Pi * math.Round((cmplx.Phase(fs[0])+float64(m)*math.Pi/2)/math.Pi)
				phi = phiStart - float64(m)*math.Pi/2
				phi += wrapPi(cmplx.Phase(fs[0]) - phi)
			}
		} else {
			phi -= float64(seg.jump) * math.Pi
			phi += wrapPi(cmplx.Phase(fs[0]) - phi)
		}
		for k := 1; k < len(fs); k++ {
			phi += wrapPi(cmplx.Phase(fs[k]) - cmplx.Phase(fs[k-1]))
		}
	}
	fR := out.f[len(out.f)-1]
	turns := (2*(phi-cmplx.Phase(fR)) - 2*phiStart) / (2 * math.Pi)
	w := math.Round(turns)
	if math.Abs(turns-w) > 0.25 {
		return delayNyquist{}, fmt.Errorf("winding %.3g turns is not resolved to an integer: %w", turns, errDelayLoopUnsupported)
	}
	out.roots = l.rhp - int(w)
	out.stable = out.roots == 0
	return out, nil
}

func wrapPi(a float64) float64 {
	return math.Remainder(a, 2*math.Pi)
}

func (l *delayLoop) axisGap(w float64) float64 { return 1e-7 * max(1, w) }

func (l *delayLoop) grid(R, w1 float64) []float64 {
	lo := 1.0
	for _, s := range l.scales {
		if s > 0 {
			lo = min(lo, s)
		}
	}
	for _, a := range l.axis {
		if a.w > 0 {
			lo = min(lo, a.w)
		}
	}
	lo *= 1e-3
	decades := math.Log10(R / lo)
	nlog := int(64*decades) + 2
	step, nlin := 0.0, 0
	if l.tau > 0 {
		step = math.Pi / (8 * l.tau)
		nlin = int(w1/step) + 2
	}
	pts := make([]float64, 0, nlog+nlin+9*len(l.scales)+1)
	for i := range nlog {
		pts = append(pts, lo*math.Pow(R/lo, float64(i)/float64(nlog-1)))
	}
	if l.tau > 0 {
		for w, prev := step, 0.0; w < R && prev < w1; prev, w = w, w+step {
			pts = append(pts, w)
		}
	}
	for _, s := range l.scales {
		for _, d := range []float64{-0.3, -0.1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 0.1, 0.3} {
			if w := s * (1 + d); w > 0 && w < R {
				pts = append(pts, w)
			}
		}
	}
	pts = append(pts, R)
	slices.Sort(pts)
	return slices.Compact(pts)
}

type nyquistSegment struct {
	from, to float64
	pts      []float64
	jump     int // indentation multiplicity at from
}

// segments splits the grid at the indentations around imaginary-axis poles
// and checks that each such eigenvalue is a pole of L of full multiplicity;
// a lower order means an uncancelled hidden mode on the axis.
func (l *delayLoop) segments(grid []float64, R float64) []nyquistSegment {
	axis := slices.Clone(l.axis)
	slices.SortFunc(axis, func(a, b axisPole) int { return cmp.Compare(a.w, b.w) })
	var segs []nyquistSegment
	start, jump := 0.0, 0
	if len(axis) > 0 && axis[0].w == 0 {
		start = l.axisGap(0)
		axis = axis[1:]
	}
	for _, a := range axis {
		g := l.axisGap(a.w)
		segs = append(segs, nyquistSegment{from: start, to: a.w - g, jump: jump})
		start, jump = a.w+g, a.mult
	}
	segs = append(segs, nyquistSegment{from: start, to: R, jump: jump})
	for i := range segs {
		s := &segs[i]
		lo, _ := slices.BinarySearch(grid, math.Nextafter(s.from, math.Inf(1)))
		hi, _ := slices.BinarySearch(grid, s.to)
		hi = max(hi, lo)
		s.pts = make([]float64, 0, hi-lo+2)
		s.pts = append(s.pts, s.from)
		s.pts = append(s.pts, grid[lo:hi]...)
		s.pts = append(s.pts, s.to)
	}
	return segs
}

// hiddenAxisRoot returns an imaginary-axis eigenvalue that is not a pole of L
// of its full multiplicity, judged from the growth of |L| towards it; χ
// vanishes there.
func (l *delayLoop) hiddenAxisRoot() (float64, bool) {
	for _, a := range l.axis {
		g := l.axisGap(a.w)
		near, far := cmplx.Abs(l.at(a.w+g)), cmplx.Abs(l.at(a.w+10*g))
		if !(near > 0) || !(far > 0) || math.IsInf(near, 1) {
			return a.w, true
		}
		if int(math.Round(math.Log10(near/far))) != a.mult {
			return a.w, true
		}
	}
	return 0, false
}

type refineResult int

const (
	refineResolved refineResult = iota
	refineAxisRoot
	refineBudget
)

func (l *delayLoop) pointBudget() int {
	if l.maxPoints > 0 {
		return l.maxPoints
	}
	return delayNyquistMaxPoints
}

// refine evaluates 1+L on the segment, bisecting until consecutive phases
// differ by at most π/4. It reports refineAxisRoot and the root frequency when
// 1+L vanishes or its phase turns within the bisection tolerance, i.e. a
// closed-loop root sits on the axis, and refineBudget when the segment would
// exceed budget points. The points are appended to ws and fs.
func (l *delayLoop) refine(seg nyquistSegment, budget int, ws []float64, fs []complex128) ([]float64, []complex128, refineResult, float64) {
	f := func(w float64) complex128 { return 1 + l.at(w) }
	base := len(ws)
	ws = append(ws, seg.pts[0])
	fs = append(fs, f(seg.pts[0]))
	if !resolvedNonzero(fs[base]) {
		return ws, fs, refineAxisRoot, seg.pts[0]
	}
	type span struct {
		a, b   float64
		fa, fb complex128
	}
	var stack []span
	for k := 1; k < len(seg.pts); k++ {
		stack = append(stack[:0], span{seg.pts[k-1], seg.pts[k], fs[len(fs)-1], f(seg.pts[k])})
		for len(stack) > 0 {
			s := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if !resolvedNonzero(s.fb) {
				return ws, fs, refineAxisRoot, s.b
			}
			if math.Abs(wrapPi(cmplx.Phase(s.fb)-cmplx.Phase(s.fa))) <= math.Pi/4 {
				ws = append(ws, s.b)
				fs = append(fs, s.fb)
				continue
			}
			if s.b-s.a <= 1e-13*max(1, s.b) {
				return ws, fs, refineAxisRoot, (s.a + s.b) / 2
			}
			if len(ws)-base+len(stack) > budget {
				return ws, fs, refineBudget, 0
			}
			mid := (s.a + s.b) / 2
			fm := f(mid)
			stack = append(stack, span{mid, s.b, fm, s.fb}, span{s.a, mid, s.fa, fm})
		}
	}
	return ws, fs, refineResolved, 0
}

func resolvedNonzero(v complex128) bool {
	a := cmplx.Abs(v)
	return a > 1e-12 && !math.IsNaN(a)
}

func (l *delayLoop) axisMult(w float64) int {
	for _, a := range l.axis {
		if a.w == w {
			return a.mult
		}
	}
	return 0
}

// stableClosedLoop reports whether 1/(1+L) is stable.
func (l *delayLoop) stableClosedLoop() (bool, error) {
	res, err := l.nyquist(0.5)
	return res.stable, err
}

type sensitivityPeak struct{ peak, w float64 }

// sensitivityTailLevel is the |L| level that ends the linear delay grid on
// the first sensitivity pass; it stays below sin(π/8) so the sparse tail
// needs no bisection.
const sensitivityTailLevel = 0.3

// sensitivityPeaks returns the closed-loop stability of S = 1/(1+L) and,
// when stable, sup_ω |S(jω) + c| with its frequency for each shift c. The
// grid peak is refined by golden section and then certified to peakCertTol
// on [0, tailFrom] by descriptorResponse.certify on the closed-loop
// descriptor, which bisects wherever a narrow peak could hide between
// samples. Where |L| <= t, S lies in the disk
// of centre 1/(1−t²) and radius t/(1−t²), so |S + c| <= diskBound(t, c).
// The linear delay spacing ends where the tail bound certifies |L| <= t and
// diskBound(t, c) does not exceed the grid peak for any c, so the sparse
// tail cannot hold a larger sample; otherwise the grid is rebuilt with t
// lowered until it does, or with the full linear spacing. Beyond the grid
// |L| <= t with t within 0.1% of tailLimit, and diskBound at t = tailLimit is
// reported with frequency +Inf when it exceeds the grid peak.
func (l *delayLoop) sensitivityPeaks(shifts ...float64) (stable bool, peaks []sensitivityPeak, err error) {
	gs := make([]func(complex128) float64, len(shifts))
	for i, c := range shifts {
		gs[i] = func(f complex128) float64 { return cmplx.Abs(1/f + complex(c, 0)) }
	}
	best := make([]int, len(shifts))
	lin := 0.0
	if l.tau > 0 && l.tailLimit < sensitivityTailLevel {
		lin = sensitivityTailLevel
	}
	var res delayNyquist
	for {
		res, err = l.nyquistLinearTo(1e-3, lin)
		if err != nil || !res.stable {
			return false, nil, err
		}
		need := lin
		for i, g := range gs {
			best[i] = 0
			for k, v := range res.f {
				if g(v) > g(res.f[best[i]]) {
					best[i] = k
				}
			}
			need = min(need, diskBoundLevel(shifts[i], g(res.f[best[i]]), lin))
		}
		if need == lin {
			break
		}
		lin = 0
		if need > l.tailLimit {
			lin = need
		}
	}
	peaks = make([]sensitivityPeak, len(shifts))
	for i := range shifts {
		wPeak, peak := refinePeaks(res, gs[i], best[i], func(w float64) float64 { return gs[i](1 + l.at(w)) })
		peaks[i] = sensitivityPeak{peak, wPeak}
	}
	if err := l.evalErr(); err != nil {
		return false, nil, err
	}
	if err := l.certifyPeaks(res, shifts, peaks); err != nil {
		return false, nil, err
	}
	for i, c := range shifts {
		if inf := diskBound(l.tailLimit, c); inf > peaks[i].peak {
			peaks[i] = sensitivityPeak{inf, math.Inf(1)}
		}
	}
	return true, peaks, nil
}

// certifyPeaks raises each sensitivity peak to within peakCertTol of
// sup |S + c| on [0, res.tailFrom], beyond which the disk bound holds.
func (l *delayLoop) certifyPeaks(res delayNyquist, shifts []float64, peaks []sensitivityPeak) error {
	r, err := newDescriptorResponse(l.sys)
	if err != nil {
		return err
	}
	if err := r.sensitivity(); err != nil {
		return err
	}
	ws := []float64{0}
	for _, w := range res.w {
		if w > res.tailFrom {
			break
		}
		if w > ws[len(ws)-1] {
			ws = append(ws, w)
		}
	}
	if ws[len(ws)-1] < res.tailFrom {
		ws = append(ws, res.tailFrom)
	}
	ps, wps := make([]float64, len(shifts)), make([]float64, len(shifts))
	for i, p := range peaks {
		ps[i], wps[i] = p.peak, p.w
	}
	if err := r.certify(ws, shifts, ps, wps, l.pointBudget(), errDelayLoopUnsupported); err != nil {
		return err
	}
	for i := range peaks {
		peaks[i] = sensitivityPeak{ps[i], wps[i]}
	}
	return nil
}

// nearPeakFraction selects the grid local maxima refined besides the grid
// maximum, so that a peak whose top falls between samples is not lost to a
// neighbour that rounding ranks higher.
const nearPeakFraction = 0.9

// refinePeaks refines by golden section every grid local maximum of g within
// nearPeakFraction of the grid maximum at best, and returns the largest. A
// local maximum is bracketed by its nearest neighbours of distinct frequency:
// the grid can hold points that differ only by rounding, which would leave a
// degenerate bracket on one side of the true peak.
func refinePeaks(res delayNyquist, g func(complex128) float64, best int, at func(float64) float64) (wPeak, peak float64) {
	wPeak, peak = res.w[best], g(res.f[best])
	floor := nearPeakFraction * peak
	n := len(res.w)
	for k := range n {
		gk := g(res.f[k])
		if gk < floor {
			continue
		}
		tol := 1e-9 * max(1, res.w[k])
		a := k
		for a > 0 && res.w[a] > res.w[k]-tol {
			a--
		}
		b := k
		for b < n-1 && res.w[b] < res.w[k]+tol {
			b++
		}
		if (a < k && g(res.f[a]) > gk) || (b > k && g(res.f[b]) > gk) {
			continue
		}
		if w, v := goldenMax(res.w[a], res.w[b], res.w[k], gk, at); v > peak {
			wPeak, peak = w, v
		}
	}
	return wPeak, peak
}

// diskBound bounds |S + c| over |L| <= t.
func diskBound(t, c float64) float64 {
	return math.Abs(1/(1-t*t)+c) + t/(1-t*t)
}

// diskBoundLevel returns the largest level in [0, t] at which
// diskBound(level, c) <= peak, or t when diskBound(t, c) <= peak already;
// diskBound grows with the level because the disks are nested.
func diskBoundLevel(c, peak, t float64) float64 {
	if diskBound(t, c) <= peak {
		return t
	}
	lo, hi := 0.0, t
	if !(diskBound(lo, c) <= peak) {
		return 0
	}
	for range 60 {
		mid := (lo + hi) / 2
		if diskBound(mid, c) <= peak {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

// goldenMax refines a maximum of g bracketed by [a, b], starting from the
// sample (w0, g0), by golden-section search.
func goldenMax(a, b, w0, g0 float64, g func(float64) float64) (float64, float64) {
	const phi = 0.6180339887498949
	x1, x2 := b-phi*(b-a), a+phi*(b-a)
	f1, f2 := g(x1), g(x2)
	for range 100 {
		if b-a <= 1e-12*max(1, b) {
			break
		}
		if f1 > f2 {
			b, x2, f2 = x2, x1, f1
			x1 = b - phi*(b-a)
			f1 = g(x1)
		} else {
			a, x1, f1 = x1, x2, f2
			x2 = a + phi*(b-a)
			f2 = g(x2)
		}
	}
	for _, c := range []struct{ w, f float64 }{{x1, f1}, {x2, f2}} {
		if c.f > g0 {
			w0, g0 = c.w, c.f
		}
	}
	return w0, g0
}
