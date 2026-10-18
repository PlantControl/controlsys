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
	sys        *System // the loop model, for certified sensitivity peaks
	at         func(w float64) complex128
	atS        func(s complex128) (complex128, bool) // L(s) off the axis for clusterWinding; nil disables it
	one        func(w float64) complex128            // 1+L formed directly; nil: 1 + at
	eval       *sisoEval                             // records the first failure of at
	fails      *error                                // where at records its first failure when eval is nil
	rhp        int
	axis       []axisPole
	poles      []complex128
	tail       func(w float64) float64 // bound on |L(s)| for Re s >= 0, |s| >= w
	tailLimit  float64                 // bound on limsup |L(jω)|, ω → ∞
	tau        float64                 // total loop delay, sets the grid density
	scales     []float64
	maxPoints  int           // grid point budget; 0 selects delayNyquistMaxPoints
	cert       loopCertifier // certified local models of 1+L for refine
	certErr    error         // why cert is nil
	stack      []refineSpan  // refine work buffers
	models     []nodeModels
	freeModels []int
	clusters   []axisPole // verified cluster indentations, once adopted
}

type axisPole struct {
	w    float64
	mult int
	gap  float64 // indentation radius, set by indent; 0 selects axisGap
	// order is the pole count of L within the indentation, mult plus the
	// open-loop poles it encloses (see indentCluster); 0 selects mult. inner
	// counts the enclosed RHP poles, which leave the contour's interior.
	order, inner int
	// arcSet marks an indentation whose phase change of 1+L, arc, was
	// measured on the arc by verifyCluster: from j(w-gap) to j(w+gap), or at
	// the origin from gap to j·gap, where 1+L has phase arcStart.
	arcSet        bool
	arc, arcStart float64
}

func (a axisPole) poles() int {
	if a.order > 0 {
		return a.order
	}
	return a.mult
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
	cert, err := newDescriptorCertifier(sys)
	if err != nil {
		return nil, err
	}
	l := &delayLoop{sys: sys, at: eval.at, eval: eval, cert: cert, tail: bound.at, tailLimit: bound.at(math.Inf(1)), tau: sisoLoopDelay(sys)}
	eval.extended, eval.ddAllow = true, l.ddAllowed
	if sys.IsContinuous() {
		l.atS = eval.atComplex
	}
	l.addPoles(poles)
	return l, nil
}

// ddAllowed reports whether the double-double fallback may evaluate L(jω):
// only where a boundary pole is nearest, so the walk resolves indentations
// next to crowded axis poles. Near any other pole a double failure must
// stand, so that clusterWinding counts the roots: there the fallback would
// let the uniform walk miss a phase turn between samples.
func (l *delayLoop) ddAllowed(w float64) bool {
	s := complex(0, w)
	best, boundary := math.Inf(1), false
	for _, p := range l.poles {
		d := cmplx.Abs(p - s)
		b := classifyPoleStability(p, true, poleStabilityTolerance(p)) == poleClassBoundary
		if d < best || (d == best && !b) {
			best, boundary = d, b
		}
	}
	return boundary
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
	l.poles = append(l.poles, poles...)
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
// using |c(sI-A)^{-1}b| <= ‖c‖‖b‖/(|s|-‖A‖₂), |e^{-sτ}| <= 1 and the finite
// Neumann series of the acyclic delay loop. Each block of H is invariant
// under state similarity, so its bound is the smaller of those from the
// given and the jointly balanced (A, Baug, Caug): a companion form's ‖A‖₂
// can exceed its spectral radius by orders of magnitude, which would
// stretch the delay grid to ‖A‖₂.
type rationalTailBound struct {
	reals []tailRealization
	d     []float64 // (1+N)×(1+N) |Daug|
	nd    int
}

type tailRealization struct {
	normA float64
	c, b  []float64 // row norms of Caug, column norms of Baug
}

func newRationalTailBound(sys *System) (*rationalTailBound, error) {
	n, _, _ := sys.Dims()
	nd := sys.internalDelayCount()
	if nd > 0 && !lftDelayLoopAcyclic(sys, n) {
		return nil, fmt.Errorf("internal delay feedback cycle: %w", ErrContinuousInternalDelay)
	}
	k := 1 + nd
	r := &rationalTailBound{d: make([]float64, k*k), nd: nd}
	r.d[0] = math.Abs(sys.D.At(0, 0))
	br := augmentedRealization(sys, n, nd)
	given := br.tailRealization()
	r.reals = []tailRealization{given}
	if n > 0 {
		r.reals[0].normA = spectralNorm(sys.A)
		br.balance(nil)
		bal := br.tailRealization()
		bal.normA = spectralNorm(mat.NewDense(n, n, br.a))
		r.reals = append(r.reals, bal)
	}
	if nd > 0 {
		lft := sys.LFT
		for i := range nd {
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

// augmentedRealization copies (A, [B B2], [C; C2]) unscaled.
func augmentedRealization(sys *System, n, nd int) balancedRealization {
	k := 1 + nd
	br := balancedRealization{n: n, m: k, p: k, a: make([]float64, n*n), b: make([]float64, n*k), c: make([]float64, k*n)}
	if n == 0 {
		return br
	}
	a := sys.A.RawMatrix()
	copyStrided(br.a, n, a.Data, a.Stride, n, n)
	for i := range n {
		br.b[i*k] = sys.B.At(i, 0)
		br.c[i] = sys.C.At(0, i)
	}
	if nd == 0 {
		return br
	}
	lft := sys.LFT
	for i := range n {
		for j := range nd {
			if lft.B2 != nil {
				br.b[i*k+1+j] = lft.B2.At(i, j)
			}
			if lft.C2 != nil {
				br.c[(1+j)*n+i] = lft.C2.At(j, i)
			}
		}
	}
	return br
}

func (br *balancedRealization) tailRealization() tailRealization {
	n, k := br.n, br.m
	t := tailRealization{c: make([]float64, k), b: make([]float64, k)}
	if n == 0 {
		return t
	}
	col := make([]float64, n)
	for j := range k {
		for i := range n {
			col[i] = br.b[i*k+j]
		}
		t.b[j] = mat.Norm(mat.NewVecDense(n, col), 2)
		t.c[j] = mat.Norm(mat.NewVecDense(n, br.c[j*n:(j+1)*n]), 2)
	}
	return t
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
			h[i*k+j] = r.d[i*k+j]
			if math.IsInf(w, 1) {
				continue
			}
			best := math.Inf(1)
			for _, t := range r.reals {
				switch cb := t.c[i] * t.b[j]; {
				case cb == 0:
					best = 0
				case w > t.normA:
					best = min(best, cb/(w-t.normA))
				}
			}
			if math.IsInf(best, 1) {
				return best
			}
			h[i*k+j] += best
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
// open-loop pole magnitude, and bisection until refine certifies every span
// free of zeros of 1+L with phase variation at most spanCertPhase < π, so
// that the wrapped phase steps add up to the winding. The grid ends at R
// where the tail bound certifies |L| <= tailLimit + frac(1-tailLimit) < 1 on
// the rest of the closed right half-plane, so no encirclement is missed
// beyond R. The count is exact up to rounding; a closed-loop root within the
// bisection tolerance of the axis, or an axis eigenvalue hidden from L, is
// reported unstable with axisRoot set. A grid that exceeds the point budget
// or a winding number that does not resolve to an integer returns
// errDelayLoopUnsupported rather than a verdict.
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
	if l.cert == nil {
		if l.certErr != nil {
			return delayNyquist{}, l.certErr
		}
		return delayNyquist{}, fmt.Errorf("delay loop without a certified model: %w", errDelayLoopUnsupported)
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
	if l.clusters != nil {
		l.axis = l.clusters
		return l.winding(R, w1)
	}
	pre := l.failed()
	res, err := l.indentedWinding(R, w1)
	if (err == nil && !res.axisRoot && !l.failed()) || l.atS == nil || pre {
		return res, err
	}
	if alt, ok := l.clusterWinding(R, w1, res); ok {
		return alt, nil
	}
	return res, err
}

func (l *delayLoop) indentedWinding(R, w1 float64) (delayNyquist, error) {
	if w, ok, err := l.indent(); err != nil {
		return delayNyquist{}, err
	} else if ok {
		return delayNyquist{axisRoot: true, axisW: w}, nil
	}
	return l.winding(R, w1)
}

// winding counts the closed-loop RHP roots along the indented axis up to R.
func (l *delayLoop) winding(R, w1 float64) (delayNyquist, error) {
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
		case refineUncertified:
			return delayNyquist{}, fmt.Errorf("phase of 1+L not certified near ω=%g: %w", wRoot, errDelayLoopUnsupported)
		}
		fs := out.f[base:]
		switch {
		case si == 0 && seg.from == 0:
			phiStart = 0
			if real(fs[0]) < 0 {
				phiStart = math.Pi
			}
			phi = phiStart
		case seg.arc != nil:
			if si == 0 {
				phiStart, phi = seg.arc.arcStart, seg.arc.arcStart
			}
			phi += seg.arc.arc
			if d := wrapPi(cmplx.Phase(fs[0]) - phi); math.Abs(d) <= arcSnapTol {
				phi += d
			} else {
				return delayNyquist{}, fmt.Errorf("measured indentation at ω=%g misses 1+L by %.3g rad: %w", seg.arc.w, d, errDelayLoopUnsupported)
			}
		case si == 0:
			m := l.axisMult(0)
			phiStart = math.Pi * math.Round((cmplx.Phase(fs[0])+float64(m)*math.Pi/2)/math.Pi)
			phi = phiStart - float64(m)*math.Pi/2
			phi += wrapPi(cmplx.Phase(fs[0]) - phi)
		default:
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
	for _, a := range l.axis {
		out.roots -= a.inner
	}
	if out.roots < 0 {
		return delayNyquist{}, fmt.Errorf("negative closed-loop root count %d: %w", out.roots, errDelayLoopUnsupported)
	}
	out.stable = out.roots == 0
	return out, nil
}

func wrapPi(a float64) float64 {
	return math.Remainder(a, 2*math.Pi)
}

func (l *delayLoop) axisGap(w float64) float64 { return 1e-7 * max(1, w) }

func (a axisPole) radius(l *delayLoop) float64 {
	if a.gap > 0 {
		return a.gap
	}
	return l.axisGap(a.w)
}

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
	if l.tau > 0 {
		for w, prev := step, 0.0; w < R && prev < w1; prev, w = w, w+step {
			pts = append(pts, w)
		}
	}
	var buf [1024]float64
	extra := slices.Grow(buf[:0], nlog+9*len(l.scales)+1)
	for i := range nlog {
		extra = append(extra, lo*math.Pow(R/lo, float64(i)/float64(nlog-1)))
	}
	for _, s := range l.scales {
		for _, d := range []float64{-0.3, -0.1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 0.1, 0.3} {
			if w := s * (1 + d); w > 0 && w < R {
				extra = append(extra, w)
			}
		}
	}
	extra = append(extra, R)
	slices.Sort(extra)
	return slices.Compact(mergeSorted(pts, extra))
}

// mergeSorted merges the sorted b into the sorted a, in place in a's
// capacity when it holds both.
func mergeSorted(a, b []float64) []float64 {
	i, j := len(a)-1, len(b)-1
	a = slices.Grow(a, len(b))[:len(a)+len(b)]
	for k := len(a) - 1; j >= 0; k-- {
		if i >= 0 && a[i] > b[j] {
			a[k] = a[i]
			i--
		} else {
			a[k] = b[j]
			j--
		}
	}
	return a
}

type nyquistSegment struct {
	from, to float64
	pts      []float64
	jump     int       // indentation multiplicity at from
	arc      *axisPole // the measured indentation ending at from, if any
}

// segments splits the grid at the indentations around imaginary-axis poles
// and checks that each such eigenvalue is a pole of L of full multiplicity;
// a lower order means an uncancelled hidden mode on the axis.
func (l *delayLoop) segments(grid []float64, R float64) []nyquistSegment {
	axis := slices.Clone(l.axis)
	slices.SortFunc(axis, func(a, b axisPole) int { return cmp.Compare(a.w, b.w) })
	var segs []nyquistSegment
	start, jump := 0.0, 0
	var arc *axisPole
	measured := func(a *axisPole) *axisPole {
		if a.arcSet {
			return a
		}
		return nil
	}
	if len(axis) > 0 && axis[0].w == 0 {
		start, arc = axis[0].radius(l), measured(&axis[0])
		axis = axis[1:]
	}
	for i := range axis {
		a := &axis[i]
		g := a.radius(l)
		segs = append(segs, nyquistSegment{from: start, to: a.w - g, jump: jump, arc: arc})
		start, jump, arc = a.w+g, a.poles(), measured(a)
	}
	segs = append(segs, nyquistSegment{from: start, to: R, jump: jump, arc: arc})
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

// indent sets the indentation radius of each imaginary-axis pole and
// returns an imaginary-axis eigenvalue that is not a pole of L of its full
// multiplicity, judged from the growth of |L| over the decade outside the
// indentation; χ vanishes there. The radius starts at axisGap; where L
// cannot be evaluated there, as next to a multiple pole whose pencil the
// solver deems singular, it grows up to indentLimit, so that the decade
// stays within a tenth of the other poles and of the delay scale. An
// enlarged radius needs |L| >= 1e3 on that decade, so that χ ≈ s^m + c has
// no root within it. An off-axis pole within the decade, as one below
// indentLimit's cutoff, voids the check and returns errDelayLoopUnsupported,
// which leaves the count to clusterWinding. A pole at which
// no radius gives finite samples is left to indentCluster: a failed
// evaluation does not show a root.
func (l *delayLoop) indent() (float64, bool, error) {
	for i := range l.axis {
		a := &l.axis[i]
		a.order, a.inner = 0, 0
		g0 := l.axisGap(a.w)
		gmax := l.indentLimit(a.w)
		found := false
		for k := 0; k == 0 || g0*math.Pow(10, float64(k)/2) <= gmax; k++ {
			g := g0 * math.Pow(10, float64(k)/2)
			near, ok1 := l.sample(a.w + g)
			far, ok2 := l.sample(a.w + 10*g)
			if !ok1 || !ok2 {
				continue
			}
			if k > 0 && near > 0 && !(far >= 1e3) {
				return 0, false, fmt.Errorf("|L| = %g is not dominant at the indentation of ω=%g: %w", far, a.w, errDelayLoopUnsupported)
			}
			if l.poleNear(a.w, 10*g) {
				return 0, false, fmt.Errorf("open-loop pole within the indentation decade of ω=%g: %w", a.w, errDelayLoopUnsupported)
			}
			a.gap, found = g, true
			if !(near > 0) || !(far > 0) || int(math.Round(math.Log10(near/far))) != a.mult {
				return a.w, true, nil
			}
			break
		}
		if !found {
			if err := l.indentCluster(a); err != nil {
				return 0, false, err
			}
		}
	}
	return 0, false, nil
}

// indentCluster sets the indentation of the pole a at the origin when L
// cannot be evaluated within indentLimit, as for a multiple integrator with
// another pole close by. The indentation encloses the k open-loop poles
// nearest to the origin, at most D away, for growing k, with a radius g in
// [10D, 0.1·min(next pole, 1, 1/τ)] (from beyond indentLimit for k = 0).
// On its arc L ≈ c·s^-order, order = mult + k, so the arc turns the phase
// by order·π, up to 0.1 rad per pole. The growth of |L| over [g, 10g] must
// match that of the known poles within 0.3 decades, so that no zero of L or
// hidden mode sits nearby, and |L| >= 1e3 at 10g, so that 1+L has no zero
// within the indentation. Enclosed RHP poles leave the contour's interior
// and are counted in inner. Anything else, and an axis pole off the
// origin, returns errDelayLoopUnsupported, never a verdict.
func (l *delayLoop) indentCluster(a *axisPole) error {
	unevaluable := fmt.Errorf("L not finite near the axis pole ω=%g: %w", a.w, errDelayLoopUnsupported)
	if a.w != 0 {
		return unevaluable
	}
	type member struct {
		d     float64
		class poleStabilityClass
	}
	var others []member
	for _, p := range l.poles {
		if d := cmplx.Abs(p); d > 1e-6 {
			others = append(others, member{d, classifyPoleStability(p, true, poleStabilityTolerance(p))})
		}
	}
	slices.SortFunc(others, func(x, y member) int { return cmp.Compare(x.d, y.d) })
	lim := 1.0
	if l.tau > 0 {
		lim = min(lim, 1/l.tau)
	}
	growth := func(g float64) float64 {
		v := float64(a.mult)
		for _, p := range l.poles {
			if cmplx.Abs(p) > 1e-6 {
				v += math.Log10(cmplx.Abs(complex(0, 10*g)-p) / cmplx.Abs(complex(0, g)-p))
			}
		}
		return v
	}
	step := math.Sqrt(math.Sqrt(10))
	inner := 0
	for k := 0; k <= len(others); k++ {
		lo := l.indentLimit(0) * step
		if k > 0 {
			m := others[k-1]
			if m.class == poleClassBoundary {
				break
			}
			if m.class == poleClassUnstable {
				inner++
			}
			lo = 10 * m.d
		}
		next := lim
		if k < len(others) {
			next = min(next, others[k].d)
		}
		for g := lo; g <= 0.1*next; g *= step {
			near, ok1 := l.sample(g)
			far, ok2 := l.sample(10 * g)
			if !ok1 || !ok2 {
				continue
			}
			if !(far >= 1e3) || !(near > 0) || math.Abs(math.Log10(near/far)-growth(g)) > 0.3 {
				return fmt.Errorf("|L| near the poles at the origin does not follow them: %w", errDelayLoopUnsupported)
			}
			a.gap, a.order, a.inner = g, a.mult+k, inner
			return nil
		}
	}
	return unevaluable
}

// sample returns |L(jω)| and whether it is finite, without recording a
// failed evaluation.
func (l *delayLoop) sample(w float64) (float64, bool) {
	slot := l.fails
	if l.eval != nil {
		slot = &l.eval.err
	}
	var saved error
	if slot != nil {
		saved, *slot = *slot, nil
	}
	v := cmplx.Abs(l.at(w))
	ok := !math.IsInf(v, 1) && !math.IsNaN(v)
	if slot != nil {
		ok = ok && *slot == nil
		*slot = saved
	}
	return v, ok
}

// poleNear reports an open-loop pole off the imaginary axis within r of jω,
// which the growth of |L| by mult decades over the indentation decade does
// not account for.
func (l *delayLoop) poleNear(w, r float64) bool {
	return slices.ContainsFunc(l.poles, func(p complex128) bool {
		return cmplx.Abs(p-complex(0, w)) < r && classifyPoleStability(p, true, poleStabilityTolerance(p)) != poleClassBoundary
	})
}

// indentLimit bounds the indentation radius at jω by 1e-2 of the distance
// to the other open-loop poles, of max(1, ω) and of 1/τ.
func (l *delayLoop) indentLimit(w float64) float64 {
	lim := max(1, w)
	if l.tau > 0 {
		lim = min(lim, 1/l.tau)
	}
	for _, p := range l.poles {
		if d := cmplx.Abs(p - complex(0, w)); d > 1e-6*max(1, w) {
			lim = min(lim, d)
		}
	}
	return 1e-2 * lim
}

type refineResult int

const (
	refineResolved refineResult = iota
	refineAxisRoot
	refineBudget
	refineUncertified
)

func (l *delayLoop) pointBudget() int {
	if l.maxPoints > 0 {
		return l.maxPoints
	}
	return delayNyquistMaxPoints
}

// refine evaluates 1+L on the segment, bisecting until each span between
// consecutive points is certified: either the tail bound puts 1+L in the
// disk of radius sin(π/8) around 1 beyond the span's start, or the Taylor
// models of l.cert, at one end covering the span or at both ends each
// covering half (see certifiedSpan), prove that 1+L has no zero on it and that its phase varies by at most spanCertPhase
// (see stadiumPhase). The phase step between consecutive points is then
// exactly the wrapped difference of their phases. It reports refineAxisRoot
// and the root frequency when 1+L vanishes, or when a span of the bisection
// tolerance is not certified and its phase turns by more than π/4, i.e. a
// closed-loop root sits on the axis; refineUncertified when such a span
// turns less; and refineBudget when the segment would exceed budget points.
// The points are appended to ws and fs.
func (l *delayLoop) refine(seg nyquistSegment, budget int, ws []float64, fs []complex128) ([]float64, []complex128, refineResult, float64) {
	f := l.one
	if f == nil {
		f = func(w float64) complex128 { return 1 + l.at(w) }
	}
	base := len(ws)
	l.models, l.freeModels = l.models[:0], l.freeModels[:0]
	node := func(w float64) refineNode { return refineNode{w: w, f: f(w), mi: l.modelSlot()} }
	prev := node(seg.pts[0])
	ws = append(ws, prev.w)
	fs = append(fs, prev.f)
	if !l.resolved(&prev) {
		return ws, fs, refineAxisRoot, prev.w
	}
	stack := l.stack[:0]
	defer func() { l.stack = stack[:0] }()
	for k := 1; k < len(seg.pts); k++ {
		stack = append(stack[:0], refineSpan{prev, node(seg.pts[k])})
		for len(stack) > 0 {
			s := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if !l.resolved(&s.b) {
				return ws, fs, refineAxisRoot, s.b.w
			}
			if l.certifiedSpan(&s) {
				l.freeModels = append(l.freeModels, s.a.mi)
				ws = append(ws, s.b.w)
				fs = append(fs, s.b.f)
				prev = s.b
				continue
			}
			mid := (s.a.w + s.b.w) / 2
			if s.b.w-s.a.w <= 1e-13*max(1, s.b.w) {
				if math.Abs(wrapPi(cmplx.Phase(s.b.f)-cmplx.Phase(s.a.f))) > math.Pi/4 {
					return ws, fs, refineAxisRoot, mid
				}
				return ws, fs, refineUncertified, mid
			}
			if len(ws)-base+len(stack) > budget {
				return ws, fs, refineBudget, 0
			}
			m := node(mid)
			stack = append(stack, refineSpan{m, s.b}, refineSpan{s.a, m})
		}
	}
	return ws, fs, refineResolved, 0
}

// refineNode is a sample of 1+L; mi indexes its models in l.models, shared
// by the copies of the node in adjacent spans.
type refineNode struct {
	w  float64
	f  complex128
	mi int
}

type refineSpan struct{ a, b refineNode }

// nodeModels holds the models of each certifier kind at one node,
// evaluated on demand.
type nodeModels struct {
	m         [2]loopModel
	has, good [2]bool
}

// modelSlot returns a cleared slot of l.models, reusing those of nodes left
// of every pending span.
func (l *delayLoop) modelSlot() int {
	if k := len(l.freeModels); k > 0 {
		i := l.freeModels[k-1]
		l.freeModels = l.freeModels[:k-1]
		l.models[i] = nodeModels{}
		return i
	}
	l.models = append(l.models, nodeModels{})
	return len(l.models) - 1
}

// certifiedSpan bounds the phase variation of 1+L over the span with as
// few model evaluations as it can: first by one model covering the whole
// span, from an end where one is evaluated already or else from the right
// end, which the next span reuses; then by the models of the preferred kind
// at both ends, each covering half; then by every kind at both ends. Each
// stage accepts a superset of the spans the previous one does, and the last
// accepts every span that the best model at each end certifies.
func (l *delayLoop) certifiedSpan(s *refineSpan) bool {
	if l.tail(s.a.w) <= sinPi8 {
		return true
	}
	if math.Abs(wrapPi(cmplx.Phase(s.b.f)-cmplx.Phase(s.a.f))) > spanCertPhase {
		return false
	}
	d := s.b.w - s.a.w
	if l.knownPhase(&s.a, d) <= spanCertPhase || l.knownPhase(&s.b, -d) <= spanCertPhase {
		return true
	}
	if !l.hasModel(&s.b) && l.kindPhase(&s.b, l.firstKind(&s.b), -d) <= spanCertPhase {
		return true
	}
	h := d / 2
	if !l.hasModel(&s.a) {
		l.kindPhase(&s.a, l.firstKind(&s.a), h)
	}
	pa, pb := l.knownPhase(&s.a, h), l.knownPhase(&s.b, -h)
	if pa+pb <= spanCertPhase {
		return true
	}
	if l.cert.kinds() == 1 {
		return false
	}
	ends := [2]*refineNode{&s.a, &s.b}
	if pb > pa {
		ends[0], ends[1] = ends[1], ends[0]
	}
	for _, n := range ends {
		hn := h
		if n == &s.b {
			hn = -h
		}
		p := l.knownPhase(n, hn)
		for k := range 2 {
			if !l.models[n.mi].has[k] {
				p = min(p, l.kindPhase(n, k, hn))
			}
		}
		if n == &s.a {
			pa = p
		} else {
			pb = p
		}
		if pa+pb <= spanCertPhase {
			return true
		}
	}
	return false
}

func (l *delayLoop) hasModel(n *refineNode) bool {
	nm := &l.models[n.mi]
	return nm.has[0] || nm.has[1]
}

// firstKind is the kind to evaluate first at n: the model of S where
// |1+L| > 1, near the open-loop poles, and of 1+L elsewhere.
func (l *delayLoop) firstKind(n *refineNode) int {
	if l.cert.kinds() == 2 && cmplx.Abs(n.f) > 1 {
		return 1
	}
	return 0
}

// knownPhase bounds the phase variation of 1+L over [n.w, n.w+h] by the
// models evaluated at n, +Inf without one.
func (l *delayLoop) knownPhase(n *refineNode, h float64) float64 {
	p := math.Inf(1)
	for k, has := range l.models[n.mi].has {
		if has {
			p = min(p, l.kindPhase(n, k, h))
		}
	}
	return p
}

func (l *delayLoop) kindPhase(n *refineNode, k int, h float64) float64 {
	nm := &l.models[n.mi]
	if !nm.has[k] {
		nm.m[k], nm.good[k] = l.cert.model(k, n.w)
		nm.has[k] = true
	}
	if !nm.good[k] {
		return math.Inf(1)
	}
	return l.cert.phase(k, &nm.m[k], h)
}

// resolved reports whether the phase of the sample n of 1+L is meaningful:
// it clears resolvedNonzero, or 1+L is formed directly and the model at n
// bounds its relative rounding by maxSampleRho. A k-fold closed-loop root at
// distance δ from the axis leaves |1+L| ~ δ^k but a relative rounding that
// grows like 1/δ only.
func (l *delayLoop) resolved(n *refineNode) bool {
	if resolvedNonzero(n.f) {
		return true
	}
	if l.one == nil || !finite(n.f) || n.f == 0 {
		return false
	}
	nm := &l.models[n.mi]
	if !nm.has[0] {
		nm.m[0], nm.good[0] = l.cert.model(0, n.w)
		nm.has[0] = true
	}
	m := &nm.m[0]
	a := cmplx.Abs(n.f)
	return nm.good[0] && m.rho*(m.unit+a) <= maxSampleRho*a
}

// maxSampleRho bounds the relative rounding of a sample of 1+L that falls
// below resolvedNonzero's floor; it moves the sampled phase by less than 0.01.
const maxSampleRho = 0.01

// resolvedNonzero reports whether the sample v of 1+L is above the rounding
// uncertainty eps·(1+|L|) of 1 + L, so that its phase is meaningful.
// Whether a span holds a zero is decided by the certified models in refine,
// not by this test.
func resolvedNonzero(v complex128) bool {
	a := cmplx.Abs(v)
	return a > 64*0x1p-52*(1+cmplx.Abs(v-1)) && !math.IsNaN(a)
}

func (l *delayLoop) axisMult(w float64) int {
	for _, a := range l.axis {
		if a.w == w {
			return a.poles()
		}
	}
	return 0
}

// stableClosedLoop reports whether 1/(1+L) is stable. The linear delay
// spacing ends where the tail bound certifies |L| <= sensitivityTailLevel,
// not at the grid end, so a fast controller pole does not stretch it.
func (l *delayLoop) stableClosedLoop() (bool, error) {
	var res delayNyquist
	var err error
	if l.tau > 0 && l.tailLimit < sensitivityTailLevel {
		res, err = l.nyquistLinearTo(1e-3, sensitivityTailLevel)
	} else {
		res, err = l.nyquist(0.5)
	}
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
	if l.sys == nil {
		return fmt.Errorf("sensitivity peaks need the loop model: %w", errDelayLoopUnsupported)
	}
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

// Cluster indentations. Where the indentation of indent fails, or the axis
// walk meets a pole, because open-loop poles crowd the axis (a defective
// pair split by rounding into one RHP and one LHP eigenvalue, integrators
// with tiny real poles nearby), clusterWinding retries with indentations
// that enclose every pole near the axis point jw0 in a disk |s-jw0| < g,
// with the other poles at least clusterMargin·g away. verifyCluster samples
// L on the whole circle: |L| >= clusterDominance there and a winding of -k
// for the k enclosed eigenvalues give, by Rouché, no closed-loop root in
// the disk and all k eigenvalues poles of L. The RHP members leave the
// contour's interior, in the disk and its mirror, and are counted in inner,
// so the count does not depend on which side of the axis rounding put them.
// The phase change of 1+L on the right arc is measured, not assumed.

const (
	clusterMargin    = 4
	clusterDominance = 4
	clusterNearAxis  = 1e-5 // |Re p| <= clusterNearAxis·max(1,|p|) makes p a candidate
	clusterStep      = 0.1  // max change of arg and log|·| between arc samples
	clusterMaxPoints = 1 << 14
	clusterSweep     = 1.3335214321633240 // 10^(1/8)
	arcSnapTol       = 0.5
)

// errSlot is where l.at records its first failure.
func (l *delayLoop) errSlot() *error {
	if l.eval != nil {
		return &l.eval.err
	}
	return l.fails
}

func (l *delayLoop) failed() bool {
	slot := l.errSlot()
	return slot != nil && *slot != nil
}

// clusterWinding counts the roots with cluster indentations around every
// axis pole, the base failure point and every near-axis pole. It reports
// false, with l unchanged, unless every indentation verifies and the walk
// resolves.
func (l *delayLoop) clusterWinding(R, w1 float64, base delayNyquist) (delayNyquist, bool) {
	ws, required := l.clusterCandidates(base)
	if len(ws) == 0 {
		return delayNyquist{}, false
	}
	slot := l.errSlot()
	var saved error
	if slot != nil {
		saved = *slot
	}
	savedAxis := l.axis
	for range 4 {
		axis, ok := l.buildClusters(ws, required)
		if !ok {
			break
		}
		if slot != nil {
			*slot = nil
		}
		l.axis = axis
		res, err := l.winding(R, w1)
		if err == nil && !l.failed() && !res.axisRoot {
			l.clusters = axis
			return res, true
		}
		if err != nil || l.failed() || !res.axisRoot || clusterCovers(axis, res.axisW) {
			break
		}
		ws, required = append(ws, res.axisW), append(required, true)
	}
	l.axis = savedAxis
	if slot != nil {
		*slot = saved
	}
	return delayNyquist{}, false
}

func clusterCovers(axis []axisPole, w float64) bool {
	return slices.ContainsFunc(axis, func(a axisPole) bool { return math.Abs(a.w-w) <= a.gap })
}

// clusterCandidates returns the cluster centres to try; a required one
// must verify.
func (l *delayLoop) clusterCandidates(base delayNyquist) (ws []float64, required []bool) {
	for _, a := range l.axis {
		ws, required = append(ws, a.w), append(required, true)
	}
	if base.axisRoot {
		ws, required = append(ws, base.axisW), append(required, true)
	}
	for _, p := range l.poles {
		if imag(p) >= 0 && math.Abs(real(p)) <= clusterNearAxis*max(1, cmplx.Abs(p)) {
			ws, required = append(ws, imag(p)), append(required, false)
		}
	}
	return ws, required
}

// buildClusters forms disjoint verified clusters at the candidates that
// together enclose every boundary pole; a cluster replaces those inside it.
func (l *delayLoop) buildClusters(ws []float64, required []bool) ([]axisPole, bool) {
	idx := make([]int, len(ws))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(ws[a], ws[b]) })
	var axis []axisPole
	for _, i := range idx {
		if clusterCovers(axis, ws[i]) {
			continue
		}
		a, ok := l.formCluster(ws[i])
		if !ok {
			if required[i] {
				return nil, false
			}
			continue
		}
		axis = slices.DeleteFunc(axis, func(b axisPole) bool { return math.Abs(a.w-b.w)+b.gap <= a.gap })
		axis = append(axis, a)
	}
	slices.SortFunc(axis, func(a, b axisPole) int { return cmp.Compare(a.w, b.w) })
	for i := 1; i < len(axis); i++ {
		if axis[i-1].w+axis[i-1].gap >= axis[i].w-axis[i].gap {
			return nil, false
		}
	}
	for _, p := range l.poles {
		if classifyPoleStability(p, true, poleStabilityTolerance(p)) != poleClassBoundary {
			continue
		}
		q := complex(real(p), math.Abs(imag(p)))
		if !slices.ContainsFunc(axis, func(a axisPole) bool { return cmplx.Abs(q-complex(0, a.w)) < a.gap }) {
			return nil, false
		}
	}
	return axis, len(axis) > 0
}

// formCluster finds the smallest k such that the k poles nearest to jw0
// fit a disk whose circle verifies.
func (l *delayLoop) formCluster(w0 float64) (axisPole, bool) {
	c := complex(0, w0)
	ds := make([]float64, len(l.poles))
	for i, p := range l.poles {
		ds[i] = cmplx.Abs(p - c)
	}
	slices.Sort(ds)
	scale := max(1, w0)
	for k := 1; k <= len(ds); k++ {
		dn := math.Inf(1)
		if k < len(ds) {
			dn = ds[k]
		}
		lo := max(clusterMargin*ds[k-1], 1e-8*scale)
		hi := min(dn/clusterMargin, scale)
		if w0 > 0 {
			hi = min(hi, w0/clusterMargin)
		}
		if lo > scale {
			break
		}
		if !(lo <= hi) {
			continue
		}
		for _, t := range []float64{0.5, 0.25, 0.75} {
			if a, ok := l.verifyCluster(w0, lo*math.Pow(hi/lo, t), k); ok {
				return a, true
			}
		}
		for g := lo; g <= hi; g *= clusterSweep {
			if a, ok := l.verifyCluster(w0, g, k); ok {
				return a, true
			}
		}
	}
	return axisPole{}, false
}

// verifyCluster samples L on |s-jw0| = g, bisecting until arg L, log|L|
// and arg(1+L) each move by at most clusterStep between samples. It
// requires |L| >= clusterDominance, a winding of L of -k, and agreement
// with l.at where the circle meets the axis; it returns the indentation
// with the measured phase change of 1+L on its right arc.
func (l *delayLoop) verifyCluster(w0, g float64, k int) (axisPole, bool) {
	c := complex(0, w0)
	type sample struct {
		t    float64
		v, f complex128
	}
	at := func(t float64) (sample, bool) {
		sn, cs := math.Sincos(t)
		v, ok := l.atS(c + complex(g*cs, g*sn))
		return sample{t, v, 1 + v}, ok && finite(v) && cmplx.Abs(v) >= clusterDominance
	}
	fine := func(a, b sample) bool {
		return math.Abs(wrapPi(cmplx.Phase(b.v)-cmplx.Phase(a.v))) <= clusterStep &&
			math.Abs(math.Log(cmplx.Abs(b.v)/cmplx.Abs(a.v))) <= clusterStep &&
			math.Abs(wrapPi(cmplx.Phase(b.f)-cmplx.Phase(a.f))) <= clusterStep
	}
	const n0 = 64
	prev, ok := at(-math.Pi)
	if !ok {
		return axisPole{}, false
	}
	var windL, turnF, start float64
	from, to := n0/4, 3*n0/4
	if w0 == 0 {
		from = n0 / 2
	}
	count := 1
	var stack [][2]sample
	for i := 1; i <= n0; i++ {
		b, ok := at(-math.Pi + 2*math.Pi*float64(i)/n0)
		if !ok {
			return axisPole{}, false
		}
		inArc := i > from && i <= to
		stack = append(stack[:0], [2]sample{prev, b})
		for len(stack) > 0 {
			s := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if fine(s[0], s[1]) {
				windL += wrapPi(cmplx.Phase(s[1].v) - cmplx.Phase(s[0].v))
				if inArc {
					turnF += wrapPi(cmplx.Phase(s[1].f) - cmplx.Phase(s[0].f))
				}
				count++
				continue
			}
			if count+len(stack) > clusterMaxPoints || s[1].t-s[0].t < 1e-9 {
				return axisPole{}, false
			}
			m, ok := at((s[0].t + s[1].t) / 2)
			if !ok {
				return axisPole{}, false
			}
			stack = append(stack, [2]sample{m, s[1]}, [2]sample{s[0], m})
		}
		if i == from && w0 == 0 {
			start = 0
			if real(b.f) < 0 {
				start = math.Pi
			}
		}
		prev = b
	}
	if math.Abs(windL/(2*math.Pi)+float64(k)) > 0.05 {
		return axisPole{}, false
	}
	for _, w := range []float64{w0 + g, w0 - g} {
		if w < 0 {
			continue
		}
		v, ok := l.sampleValue(w)
		ref, ok2 := l.atS(complex(0, w))
		if !ok || !ok2 || !(cmplx.Abs(v-ref) <= 1e-6*cmplx.Abs(ref)) {
			return axisPole{}, false
		}
	}
	inner := 0
	for _, p := range l.poles {
		if cmplx.Abs(p-c) < g && classifyPoleStability(p, true, poleStabilityTolerance(p)) == poleClassUnstable {
			inner++
		}
	}
	if w0 > 0 {
		inner *= 2
	}
	return axisPole{w: w0, mult: k, gap: g, order: k, inner: inner, arcSet: true, arc: turnF, arcStart: start}, true
}

// sampleValue returns L(jω) and whether it is finite, without recording a
// failed evaluation.
func (l *delayLoop) sampleValue(w float64) (complex128, bool) {
	slot := l.errSlot()
	var saved error
	if slot != nil {
		saved, *slot = *slot, nil
	}
	v := l.at(w)
	ok := finite(v)
	if slot != nil {
		ok = ok && *slot == nil
		*slot = saved
	}
	return v, ok
}

// atComplex evaluates L(s) at a point off the axis without recording a
// failure.
func (e *sisoEval) atComplex(s complex128) (complex128, bool) {
	pt := pointAt(s)
	if err := e.eval(pt, e.dst); err != nil {
		return 0, false
	}
	if e.delay != nil {
		applyIODelayMatrixAtS(e.sys, pt, e.dst, 1, 1, e.delay)
	}
	return e.dst[0], true
}

// minusOneAt is charMinusOne at a point off the axis, without recording a
// failure.
func (e *delayLFT) minusOneAt(s complex128) (complex128, bool) {
	if e.h22 != nil {
		if err := e.h22(pointAt(s), e.hbuf); err != nil {
			return 0, false
		}
	} else {
		copy(e.hbuf, e.d22Complex())
	}
	return e.loopDet(s, e.hbuf) - 1, true
}
