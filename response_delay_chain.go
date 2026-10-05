package controlsys

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
)

const (
	delayChainTruncTol = 1e-20
	delayChainMaxExpm  = 512
	delayChainMaxWork  = 4e8
)

// delayChain samples continuous responses of models whose internal delays
// all equal τ, exactly at t = k·dt (method of steps). With zero history,
// w(t) = Σ_{j≥1} D22^{j−1}·(C2·x + D21·u)(t − jτ), so ξ_i(t) = [x; u](t − iτ)
// obeys ξ_i' = M0·ξ_i + Σ_j Mj·ξ_{i+j}, a block upper-triangular Toeplitz
// ODE whose exponential F_j(δ) propagates the chain between samples. Jumps
// of x or of the ZOH input at time θ enter block i at θ + iτ.
type delayChain struct {
	na, p, blocks, order int
	tau, dt              float64
	steps                int
	gen                  []*mat.Dense
	out                  []*mat.Dense
	prop                 []*mat.Dense
	expCache             map[float64][]*mat.Dense
}

// chainSource is a train of jumps amp[k]·dir at shift + k·dt in the
// augmented chain coordinates [x; u].
type chainSource struct {
	shift float64
	dir   []float64
	amp   []float64
}

// delayChainModel returns the explicit form of sys and its common internal
// delay when sys is a continuous model the chain samples exactly. A model
// ToExplicit cannot reduce is outside the chain's class (ok false); the
// caller's ZOH path then reports or handles it.
func delayChainModel(sys *System) (*System, float64, bool) {
	if !sys.IsContinuous() || !sys.HasInternalDelay() {
		return nil, 0, false
	}
	tau := sys.LFT.Tau[0]
	for _, v := range sys.LFT.Tau {
		if v <= 0 || math.IsInf(v, 0) || math.Abs(v-tau) > 1e-12*tau {
			return nil, 0, false
		}
	}
	explicit, err := sys.ToExplicit()
	if err != nil {
		return nil, 0, false
	}
	return explicit, tau, true
}

// newDelayChain builds the chain for the given input columns; shifts bounds
// the distinct source shifts, each costing one exponential per block. ok is
// false when the chain would exceed delayChainMaxExpm or delayChainMaxWork;
// callers then fall back to the approximate ZOH discretization.
func newDelayChain(sys *System, tau float64, inputs []int, dt float64, steps, shifts int) (*delayChain, bool) {
	n, _, p := sys.Dims()
	q := len(sys.LFT.Tau)
	k := len(inputs)
	na := n + k
	if na == 0 {
		return nil, false
	}
	K := max(int(math.Floor(float64(steps-1)*dt/tau+gridTol)), 0)
	c := &delayChain{na: na, p: p, blocks: K + 1, tau: tau, dt: dt, steps: steps, expCache: make(map[float64][]*mat.Dense)}

	left := mat.NewDense(q, na, nil)
	top := mat.NewDense(p, na, nil)
	gen0 := mat.NewDense(na, na, nil)
	for i := range q {
		for s := range n {
			left.Set(i, s, sys.LFT.C2.At(i, s))
		}
		for j, in := range inputs {
			left.Set(i, n+j, sys.LFT.D21.At(i, in))
		}
	}
	for i := range p {
		for s := range n {
			top.Set(i, s, sys.C.At(i, s))
		}
		for j, in := range inputs {
			top.Set(i, n+j, sys.D.At(i, in))
		}
	}
	for r := range n {
		for s := range n {
			gen0.Set(r, s, sys.A.At(r, s))
		}
		for j, in := range inputs {
			gen0.Set(r, n+j, sys.B.At(r, in))
		}
	}

	c.gen = make([]*mat.Dense, K+1)
	c.out = make([]*mat.Dense, K+1)
	c.gen[0], c.out[0] = gen0, top
	pow := mat.NewDense(q, na, nil)
	pow.Copy(left)
	for j := 1; j <= K; j++ {
		if j > 1 {
			var next mat.Dense
			next.Mul(sys.LFT.D22, pow)
			pow.Copy(&next)
		}
		if allZeroDense(pow) {
			break
		}
		g := mat.NewDense(na, na, nil)
		g.Slice(0, n, 0, na).(*mat.Dense).Mul(sys.LFT.B2, pow)
		if p > 0 {
			o := mat.NewDense(p, na, nil)
			o.Mul(sys.LFT.D12, pow)
			c.out[j] = o
		}
		c.gen[j] = g
	}

	order := min(K, 4)
	for {
		if (order+1)*na > delayChainMaxExpm {
			return nil, false
		}
		c.order = order
		c.prop = c.expBlocks(dt)
		if order == K || maxAbsDense(c.prop[order]) <= delayChainTruncTol*math.Max(1, maxAbsDense(c.prop[0])) {
			break
		}
		order = min(2*order, K)
	}
	size := float64((c.order + 1) * na)
	work := float64(c.blocks) * (float64(steps)*float64(c.order+1)*float64(na*na) + float64(shifts)*(30*size*size*size+float64(steps)*size))
	if work > delayChainMaxWork {
		return nil, false
	}
	return c, true
}

func maxAbsDense(m *mat.Dense) float64 {
	raw := m.RawMatrix()
	v := 0.0
	for i := range raw.Rows {
		for _, x := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			v = math.Max(v, math.Abs(x))
		}
	}
	return v
}

// expBlocks returns F_0..F_order(δ), the first block row of exp(M·δ).
func (c *delayChain) expBlocks(delta float64) []*mat.Dense {
	na, J := c.na, c.order
	size := (J + 1) * na
	big := mat.NewDense(size, size, nil)
	for r := 0; r <= J; r++ {
		for j := 0; r+j <= J; j++ {
			if c.gen[j] == nil {
				continue
			}
			var blk mat.Dense
			blk.Scale(delta, c.gen[j])
			big.Slice(r*na, (r+1)*na, (r+j)*na, (r+j+1)*na).(*mat.Dense).Copy(&blk)
		}
	}
	var e mat.Dense
	e.Exp(big)
	out := make([]*mat.Dense, J+1)
	for j := range out {
		out[j] = mat.DenseCopyOf(e.Slice(0, na, j*na, (j+1)*na))
	}
	return out
}

func (c *delayChain) expAt(delta float64) []*mat.Dense {
	if f, ok := c.expCache[delta]; ok {
		return f
	}
	var f []*mat.Dense
	if delta == 0 {
		id := mat.NewDense(c.na, c.na, nil)
		for i := range c.na {
			id.Set(i, i, 1)
		}
		f = []*mat.Dense{id}
	} else {
		f = c.expBlocks(delta)
	}
	c.expCache[delta] = f
	return f
}

type chainInjection struct {
	start, block int
	vec          []float64
	amp          []float64
}

func (c *delayChain) injections(sources []chainSource) []chainInjection {
	var out []chainInjection
	for _, s := range sources {
		if !slices.ContainsFunc(s.amp, func(a float64) bool { return a != 0 }) {
			continue
		}
		dir := mat.NewVecDense(c.na, s.dir)
		for i := range c.blocks {
			phi := s.shift + float64(i)*c.tau
			m0 := max(int(math.Ceil(phi/c.dt-gridTol)), 0)
			if m0 >= c.steps {
				break
			}
			delta := float64(m0)*c.dt - phi
			if delta < gridTol*c.dt {
				delta = 0
			}
			f := c.expAt(delta)
			top := min(i, len(f)-1)
			vec := make([]float64, (top+1)*c.na)
			for j := 0; j <= top; j++ {
				seg := mat.NewVecDense(c.na, vec[(top-j)*c.na:(top-j+1)*c.na])
				seg.MulVec(f[j], dir)
			}
			out = append(out, chainInjection{start: m0, block: i - top, vec: vec, amp: s.amp})
		}
	}
	return out
}

// simulate returns the p×steps output sampled at k·dt.
func (c *delayChain) simulate(sources []chainSource) *mat.Dense {
	na := c.na
	Y := mat.NewDense(c.p, c.steps, nil)
	injections := c.injections(sources)
	if len(injections) == 0 {
		return Y
	}
	xi := make([]float64, c.blocks*na)
	next := make([]float64, c.blocks*na)
	yRaw := Y.RawMatrix()
	hi := -1
	for m := range c.steps {
		if m > 0 && hi >= 0 {
			for i := 0; i <= hi; i++ {
				dst := next[i*na : (i+1)*na]
				clear(dst)
				for j := 0; j <= c.order && i+j <= hi; j++ {
					addMulVec(dst, c.prop[j], xi[(i+j)*na:(i+j+1)*na])
				}
			}
			xi, next = next, xi
		}
		for _, inj := range injections {
			k := m - inj.start
			if k < 0 || k >= len(inj.amp) || inj.amp[k] == 0 {
				continue
			}
			a := inj.amp[k]
			base := inj.block * na
			for idx, v := range inj.vec {
				xi[base+idx] += a * v
			}
			hi = max(hi, inj.block+len(inj.vec)/na-1)
		}
		for i := 0; i <= hi; i++ {
			o := c.out[i]
			if o == nil {
				continue
			}
			oRaw := o.RawMatrix()
			blk := xi[i*na : (i+1)*na]
			for r := range c.p {
				row := oRaw.Data[r*oRaw.Stride : r*oRaw.Stride+na]
				s := 0.0
				for idx, v := range row {
					s += v * blk[idx]
				}
				yRaw.Data[r*yRaw.Stride+m] += s
			}
		}
	}
	return Y
}

func addMulVec(dst []float64, a *mat.Dense, x []float64) {
	raw := a.RawMatrix()
	for r := range dst {
		row := raw.Data[r*raw.Stride : r*raw.Stride+raw.Cols]
		s := 0.0
		for idx, v := range row {
			s += v * x[idx]
		}
		dst[r] += s
	}
}

// response runs the chain once per group of outputs sharing source shifts
// (external input/output delays) and fills rows rowOffset+r of Y.
func (c *delayChain) response(Y *mat.Dense, rowOffset int, sourcesFor func(r int) []chainSource) {
	groups := make(map[string][]int)
	var keys []string
	sources := make(map[string][]chainSource)
	for r := range c.p {
		src := sourcesFor(r)
		var key strings.Builder
		for _, s := range src {
			fmt.Fprintf(&key, "%v,", s.shift)
		}
		if _, ok := groups[key.String()]; !ok {
			keys = append(keys, key.String())
			sources[key.String()] = src
		}
		groups[key.String()] = append(groups[key.String()], r)
	}
	yRaw := Y.RawMatrix()
	for _, key := range keys {
		out := c.simulate(sources[key]).RawMatrix()
		for _, r := range groups[key] {
			copy(yRaw.Data[(rowOffset+r)*yRaw.Stride:(rowOffset+r)*yRaw.Stride+c.steps], out.Data[r*out.Stride:r*out.Stride+c.steps])
		}
	}
}

func outputDelayAt(sys *System, r int) float64 {
	if sys.OutputDelay == nil {
		return 0
	}
	return sys.OutputDelay[r]
}

func inputShiftAt(sys *System, r, c int) float64 {
	s := outputDelayAt(sys, r)
	if sys.InputDelay != nil {
		s += sys.InputDelay[c]
	}
	if sys.Delay != nil {
		s += sys.Delay.At(r, c)
	}
	return s
}

func delayShiftCount(sys *System) int {
	_, m, p := sys.Dims()
	seen := make(map[float64]bool)
	for r := range p {
		seen[outputDelayAt(sys, r)] = true
		for c := range m {
			seen[inputShiftAt(sys, r, c)] = true
		}
	}
	return max(len(seen), 1)
}

// delayChainStandardResponse samples the step or impulse response of a
// continuous model with one common internal delay exactly. ok is false when
// the model is outside that class or too large for the chain.
func delayChainStandardResponse(sys *System, t []float64, dt float64, kind standardInputResponse) (*mat.Dense, bool) {
	explicit, tau, ok := delayChainModel(sys)
	if !ok {
		return nil, false
	}
	n, m, p := explicit.Dims()
	if m == 0 || p == 0 {
		return nil, false
	}
	steps := len(t)
	Y := mat.NewDense(p*m, steps, nil)
	if kind == impulseResponse {
		c, ok := newDelayChain(explicit, tau, nil, dt, steps, delayShiftCount(sys))
		if !ok {
			return nil, false
		}
		for in := range m {
			dir := make([]float64, n)
			for s := range n {
				dir[s] = explicit.B.At(s, in)
			}
			c.response(Y, in*p, func(r int) []chainSource {
				return []chainSource{{shift: inputShiftAt(sys, r, in), dir: dir, amp: []float64{1}}}
			})
		}
		return Y, true
	}
	for in := range m {
		c, ok := newDelayChain(explicit, tau, []int{in}, dt, steps, delayShiftCount(sys))
		if !ok {
			return nil, false
		}
		dir := make([]float64, n+1)
		dir[n] = 1
		c.response(Y, in*p, func(r int) []chainSource {
			return []chainSource{{shift: inputShiftAt(sys, r, in), dir: dir, amp: []float64{1}}}
		})
	}
	return Y, true
}

// delayChainForcedResponse samples the response to x0 and the ZOH input u
// (m×steps) exactly; input delays do not act on x0.
func delayChainForcedResponse(sys *System, x0 *mat.VecDense, u *mat.Dense, dt float64, steps int) (*mat.Dense, bool, error) {
	explicit, tau, ok := delayChainModel(sys)
	if !ok {
		return nil, false, nil
	}
	n, m, p := explicit.Dims()
	if p == 0 {
		return nil, false, nil
	}
	if x0 != nil && x0.Len() != n {
		return nil, false, fmt.Errorf("x0 length %d != state dimension %d: %w", x0.Len(), n, ErrDimensionMismatch)
	}
	var inputs []int
	if u != nil {
		inputs = make([]int, m)
		for i := range inputs {
			inputs[i] = i
		}
	}
	c, ok := newDelayChain(explicit, tau, inputs, dt, steps, delayShiftCount(sys))
	if !ok {
		return nil, false, nil
	}
	na := n + len(inputs)
	var xDir []float64
	if x0 != nil {
		xDir = make([]float64, na)
		for s := range n {
			xDir[s] = x0.AtVec(s)
		}
	}
	jumps := make([][]float64, len(inputs))
	dirs := make([][]float64, len(inputs))
	for in := range inputs {
		jumps[in] = make([]float64, steps)
		prev := 0.0
		for k := range steps {
			v := u.At(in, k)
			jumps[in][k] = v - prev
			prev = v
		}
		dirs[in] = make([]float64, na)
		dirs[in][n+in] = 1
	}
	Y := mat.NewDense(p, steps, nil)
	c.response(Y, 0, func(r int) []chainSource {
		var src []chainSource
		if xDir != nil {
			src = append(src, chainSource{shift: outputDelayAt(sys, r), dir: xDir, amp: []float64{1}})
		}
		for in := range inputs {
			src = append(src, chainSource{shift: inputShiftAt(sys, r, in), dir: dirs[in], amp: jumps[in]})
		}
		return src
	})
	return Y, true, nil
}
