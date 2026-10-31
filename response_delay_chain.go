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
// false when the chain would exceed delayChainMaxExpm or delayChainMaxWork,
// checked in float64 before sizing so a delay far below dt cannot overflow
// the block count, and before each exponential so a rejected chain costs
// none at an order already over budget; callers then fall back to the
// approximate ZOH discretization.
func newDelayChain(sys *System, tau float64, inputs []int, dt float64, steps, shifts int) (*delayChain, bool) {
	n, _, p := sys.Dims()
	q := len(sys.LFT.Tau)
	k := len(inputs)
	na := n + k
	if na == 0 {
		return nil, false
	}
	blocks := math.Floor(float64(steps-1)*dt/tau+gridTol) + 1
	if blocks*float64(steps) > delayChainMaxWork {
		return nil, false
	}
	K := max(int(blocks)-1, 0)
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
		if n > 0 {
			g := mat.NewDense(na, na, nil)
			g.Slice(0, n, 0, na).(*mat.Dense).Mul(sys.LFT.B2, pow)
			c.gen[j] = g
		}
		if p > 0 {
			o := mat.NewDense(p, na, nil)
			o.Mul(sys.LFT.D12, pow)
			c.out[j] = o
		}
	}

	order := min(K, 4)
	for {
		if (order+1)*na > delayChainMaxExpm || delayChainWork(float64(c.blocks), float64(steps), float64(order), float64(na), float64(shifts)) > delayChainMaxWork {
			return nil, false
		}
		c.order = order
		c.prop = c.expBlocks(dt)
		if order == K || maxAbsDense(c.prop[order]) <= delayChainTruncTol*math.Max(1, maxAbsDense(c.prop[0])) {
			break
		}
		order = min(2*order, K)
	}
	return c, true
}

// delayChainWork is the work newDelayChain bounds by delayChainMaxWork for a
// chain of the given blocks, steps, truncation order, augmented size na and
// source shifts. It grows with order, so newDelayChain checks it before each
// exponential of the order-doubling search.
func delayChainWork(blocks, steps, order, na, shifts float64) float64 {
	size := (order + 1) * na
	return blocks * (steps*(order+1)*(na*na) + shifts*(30*size*size*size+steps*size))
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
			first := math.Ceil(phi/c.dt - gridTol)
			if first >= float64(c.steps) {
				break
			}
			m0 := max(int(first), 0)
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

// simulate sets row rowOffset+r of Y to output r sampled at k·dt for each
// r in rows. Each sample adds the same products in the same order as the
// plain block recurrence next_i = Σ_j F_j·ξ_{i+j}, y = Σ_i out_i·ξ_i, so
// results do not depend on the kernel chosen for na.
func (c *delayChain) simulate(sources []chainSource, Y *mat.Dense, rowOffset int, rows []int) {
	na := c.na
	yRaw := Y.RawMatrix()
	injections := c.injections(sources)
	if len(injections) == 0 {
		for _, r := range rows {
			clear(yRaw.Data[(rowOffset+r)*yRaw.Stride : (rowOffset+r)*yRaw.Stride+c.steps])
		}
		return
	}
	prop := make([]float64, (c.order+1)*na*na)
	for j, f := range c.prop[:c.order+1] {
		raw := f.RawMatrix()
		for r := range na {
			copy(prop[(j*na+r)*na:(j*na+r+1)*na], raw.Data[r*raw.Stride:r*raw.Stride+na])
		}
	}
	type outBlock struct {
		block int
		rows  []float64
	}
	var outs []outBlock
	for i, o := range c.out[:c.blocks] {
		if o == nil {
			continue
		}
		raw := o.RawMatrix()
		rows := make([]float64, c.p*na)
		for r := range c.p {
			copy(rows[r*na:(r+1)*na], raw.Data[r*raw.Stride:r*raw.Stride+na])
		}
		outs = append(outs, outBlock{i, rows})
	}
	byStart := make([]int, len(injections))
	for k := range byStart {
		byStart[k] = k
	}
	slices.SortStableFunc(byStart, func(a, b int) int { return injections[a].start - injections[b].start })
	active := make([]int, 0, len(injections))
	pending := 0

	xi := make([]float64, c.blocks*na)
	next := make([]float64, c.blocks*na)
	propagate := chainKernel(prop, na, c.order)
	hi := -1
	for m := range c.steps {
		if m > 0 && hi >= 0 {
			propagate(next, xi, hi)
			xi, next = next, xi
		}
		for pending < len(byStart) && injections[byStart[pending]].start <= m {
			k := byStart[pending]
			at, _ := slices.BinarySearch(active, k)
			active = slices.Insert(active, at, k)
			pending++
		}
		kept := active[:0]
		for _, id := range active {
			inj := &injections[id]
			k := m - inj.start
			if k >= len(inj.amp) {
				continue
			}
			kept = append(kept, id)
			if inj.amp[k] == 0 {
				continue
			}
			a := inj.amp[k]
			dst := xi[inj.block*na : inj.block*na+len(inj.vec)]
			for idx, v := range inj.vec {
				dst[idx] += a * v
			}
			hi = max(hi, inj.block+len(inj.vec)/na-1)
		}
		active = kept
		for _, r := range rows {
			acc := 0.0
			for _, ob := range outs {
				if ob.block > hi {
					break
				}
				row := ob.rows[r*na : (r+1)*na]
				blk := xi[ob.block*na : ob.block*na+na]
				s := 0.0
				for idx, v := range row {
					s += v * blk[idx]
				}
				acc += s
			}
			yRaw.Data[(rowOffset+r)*yRaw.Stride+m] = acc
		}
	}
}

// chainKernel returns propagateChain for prop, through a kernel that
// advances two blocks at once for na ≤ 4, with the same sums.
func chainKernel(prop []float64, na, order int) func(next, xi []float64, hi int) {
	switch na {
	case 2:
		p := make([][4]float64, order+1)
		for j := range p {
			copy(p[j][:], prop[4*j:])
		}
		return func(next, xi []float64, hi int) { propagateChain2(next, xi, p, hi) }
	case 3:
		p := make([][9]float64, order+1)
		for j := range p {
			copy(p[j][:], prop[9*j:])
		}
		return func(next, xi []float64, hi int) { propagateChain3(next, xi, p, hi) }
	case 4:
		p := make([][16]float64, order+1)
		for j := range p {
			copy(p[j][:], prop[16*j:])
		}
		return func(next, xi []float64, hi int) { propagateChain4(next, xi, p, hi) }
	}
	return func(next, xi []float64, hi int) { propagateChain(next, xi, prop, na, order, hi) }
}

// propagateChain sets next_i = Σ_{j ≤ min(order, hi−i)} F_j·ξ_{i+j} for
// i ≤ hi, F_j row-major at prop[j·na²:]. Every element sums its row
// products from 0 in column order and then the per-j sums from 0 in j order,
// whatever the kernel; the na-specific kernels advance two blocks at once.
func propagateChain(next, xi, prop []float64, na, order, hi int) {
	nn := na * na
	for i := 0; i <= hi; i++ {
		jn := min(order, hi-i) + 1
		x := xi[i*na : (i+jn)*na]
		dst := next[i*na : (i+1)*na]
		for r := range dst {
			acc := 0.0
			for j := range jn {
				row := prop[j*nn+r*na : j*nn+r*na+na]
				v := x[j*na : j*na+na]
				s := 0.0
				for idx, w := range row {
					s += w * v[idx]
				}
				acc += s
			}
			dst[r] = acc
		}
	}
}

func propagateChain2(next, xi []float64, prop [][4]float64, hi int) {
	order := len(prop) - 1
	i := 0
	for ; i+1 <= hi; i += 2 {
		jn := min(order, hi-i-1) + 1
		var a00, a01, a10, a11 float64
		x := xi[2*i:]
		ps := prop[:jn]
		for len(ps) >= 4 && len(x) >= 10 {
			q0 := &ps[0]
			q1 := &ps[1]
			q2 := &ps[2]
			q3 := &ps[3]
			v := (*[10]float64)(x)
			s000 := 0.0
			s000 += q0[0] * v[0]
			s000 += q0[1] * v[1]
			a00 += s000
			s001 := 0.0
			s001 += q0[2] * v[0]
			s001 += q0[3] * v[1]
			a01 += s001
			s010 := 0.0
			s010 += q0[0] * v[2]
			s010 += q0[1] * v[3]
			a10 += s010
			s011 := 0.0
			s011 += q0[2] * v[2]
			s011 += q0[3] * v[3]
			a11 += s011
			s100 := 0.0
			s100 += q1[0] * v[2]
			s100 += q1[1] * v[3]
			a00 += s100
			s101 := 0.0
			s101 += q1[2] * v[2]
			s101 += q1[3] * v[3]
			a01 += s101
			s110 := 0.0
			s110 += q1[0] * v[4]
			s110 += q1[1] * v[5]
			a10 += s110
			s111 := 0.0
			s111 += q1[2] * v[4]
			s111 += q1[3] * v[5]
			a11 += s111
			s200 := 0.0
			s200 += q2[0] * v[4]
			s200 += q2[1] * v[5]
			a00 += s200
			s201 := 0.0
			s201 += q2[2] * v[4]
			s201 += q2[3] * v[5]
			a01 += s201
			s210 := 0.0
			s210 += q2[0] * v[6]
			s210 += q2[1] * v[7]
			a10 += s210
			s211 := 0.0
			s211 += q2[2] * v[6]
			s211 += q2[3] * v[7]
			a11 += s211
			s300 := 0.0
			s300 += q3[0] * v[6]
			s300 += q3[1] * v[7]
			a00 += s300
			s301 := 0.0
			s301 += q3[2] * v[6]
			s301 += q3[3] * v[7]
			a01 += s301
			s310 := 0.0
			s310 += q3[0] * v[8]
			s310 += q3[1] * v[9]
			a10 += s310
			s311 := 0.0
			s311 += q3[2] * v[8]
			s311 += q3[3] * v[9]
			a11 += s311
			ps = ps[4:]
			x = x[8:]
		}
		for len(ps) >= 1 && len(x) >= 4 {
			q := &ps[0]
			v := (*[4]float64)(x)
			s00 := 0.0
			s00 += q[0] * v[0]
			s00 += q[1] * v[1]
			a00 += s00
			s01 := 0.0
			s01 += q[2] * v[0]
			s01 += q[3] * v[1]
			a01 += s01
			s10 := 0.0
			s10 += q[0] * v[2]
			s10 += q[1] * v[3]
			a10 += s10
			s11 := 0.0
			s11 += q[2] * v[2]
			s11 += q[3] * v[3]
			a11 += s11
			ps = ps[1:]
			x = x[2:]
		}
		for j := jn; j <= min(order, hi-i); j++ {
			q := &prop[j]
			v := (*[2]float64)(xi[2*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			a00 += s0
			s1 := 0.0
			s1 += q[2] * v[0]
			s1 += q[3] * v[1]
			a01 += s1
		}
		dst := (*[4]float64)(next[2*i:])
		dst[0], dst[1] = a00, a01
		dst[2], dst[3] = a10, a11
	}
	for ; i <= hi; i++ {
		var a00, a01 float64
		for j := range prop[:min(order, hi-i)+1] {
			q := &prop[j]
			v := (*[2]float64)(xi[2*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			a00 += s0
			s1 := 0.0
			s1 += q[2] * v[0]
			s1 += q[3] * v[1]
			a01 += s1
		}
		dst := (*[2]float64)(next[2*i:])
		dst[0], dst[1] = a00, a01
	}
}

func propagateChain3(next, xi []float64, prop [][9]float64, hi int) {
	order := len(prop) - 1
	i := 0
	for ; i+1 <= hi; i += 2 {
		jn := min(order, hi-i-1) + 1
		var a00, a01, a02, a10, a11, a12 float64
		x := xi[3*i:]
		ps := prop[:jn]
		for len(ps) >= 2 && len(x) >= 9 {
			q0 := &ps[0]
			q1 := &ps[1]
			v := (*[9]float64)(x)
			s000 := 0.0
			s000 += q0[0] * v[0]
			s000 += q0[1] * v[1]
			s000 += q0[2] * v[2]
			a00 += s000
			s001 := 0.0
			s001 += q0[3] * v[0]
			s001 += q0[4] * v[1]
			s001 += q0[5] * v[2]
			a01 += s001
			s002 := 0.0
			s002 += q0[6] * v[0]
			s002 += q0[7] * v[1]
			s002 += q0[8] * v[2]
			a02 += s002
			s010 := 0.0
			s010 += q0[0] * v[3]
			s010 += q0[1] * v[4]
			s010 += q0[2] * v[5]
			a10 += s010
			s011 := 0.0
			s011 += q0[3] * v[3]
			s011 += q0[4] * v[4]
			s011 += q0[5] * v[5]
			a11 += s011
			s012 := 0.0
			s012 += q0[6] * v[3]
			s012 += q0[7] * v[4]
			s012 += q0[8] * v[5]
			a12 += s012
			s100 := 0.0
			s100 += q1[0] * v[3]
			s100 += q1[1] * v[4]
			s100 += q1[2] * v[5]
			a00 += s100
			s101 := 0.0
			s101 += q1[3] * v[3]
			s101 += q1[4] * v[4]
			s101 += q1[5] * v[5]
			a01 += s101
			s102 := 0.0
			s102 += q1[6] * v[3]
			s102 += q1[7] * v[4]
			s102 += q1[8] * v[5]
			a02 += s102
			s110 := 0.0
			s110 += q1[0] * v[6]
			s110 += q1[1] * v[7]
			s110 += q1[2] * v[8]
			a10 += s110
			s111 := 0.0
			s111 += q1[3] * v[6]
			s111 += q1[4] * v[7]
			s111 += q1[5] * v[8]
			a11 += s111
			s112 := 0.0
			s112 += q1[6] * v[6]
			s112 += q1[7] * v[7]
			s112 += q1[8] * v[8]
			a12 += s112
			ps = ps[2:]
			x = x[6:]
		}
		for len(ps) >= 1 && len(x) >= 6 {
			q := &ps[0]
			v := (*[6]float64)(x)
			s00 := 0.0
			s00 += q[0] * v[0]
			s00 += q[1] * v[1]
			s00 += q[2] * v[2]
			a00 += s00
			s01 := 0.0
			s01 += q[3] * v[0]
			s01 += q[4] * v[1]
			s01 += q[5] * v[2]
			a01 += s01
			s02 := 0.0
			s02 += q[6] * v[0]
			s02 += q[7] * v[1]
			s02 += q[8] * v[2]
			a02 += s02
			s10 := 0.0
			s10 += q[0] * v[3]
			s10 += q[1] * v[4]
			s10 += q[2] * v[5]
			a10 += s10
			s11 := 0.0
			s11 += q[3] * v[3]
			s11 += q[4] * v[4]
			s11 += q[5] * v[5]
			a11 += s11
			s12 := 0.0
			s12 += q[6] * v[3]
			s12 += q[7] * v[4]
			s12 += q[8] * v[5]
			a12 += s12
			ps = ps[1:]
			x = x[3:]
		}
		for j := jn; j <= min(order, hi-i); j++ {
			q := &prop[j]
			v := (*[3]float64)(xi[3*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			s0 += q[2] * v[2]
			a00 += s0
			s1 := 0.0
			s1 += q[3] * v[0]
			s1 += q[4] * v[1]
			s1 += q[5] * v[2]
			a01 += s1
			s2 := 0.0
			s2 += q[6] * v[0]
			s2 += q[7] * v[1]
			s2 += q[8] * v[2]
			a02 += s2
		}
		dst := (*[6]float64)(next[3*i:])
		dst[0], dst[1], dst[2] = a00, a01, a02
		dst[3], dst[4], dst[5] = a10, a11, a12
	}
	for ; i <= hi; i++ {
		var a00, a01, a02 float64
		for j := range prop[:min(order, hi-i)+1] {
			q := &prop[j]
			v := (*[3]float64)(xi[3*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			s0 += q[2] * v[2]
			a00 += s0
			s1 := 0.0
			s1 += q[3] * v[0]
			s1 += q[4] * v[1]
			s1 += q[5] * v[2]
			a01 += s1
			s2 := 0.0
			s2 += q[6] * v[0]
			s2 += q[7] * v[1]
			s2 += q[8] * v[2]
			a02 += s2
		}
		dst := (*[3]float64)(next[3*i:])
		dst[0], dst[1], dst[2] = a00, a01, a02
	}
}

func propagateChain4(next, xi []float64, prop [][16]float64, hi int) {
	order := len(prop) - 1
	i := 0
	for ; i+1 <= hi; i += 2 {
		jn := min(order, hi-i-1) + 1
		var a00, a01, a02, a03, a10, a11, a12, a13 float64
		x := xi[4*i:]
		ps := prop[:jn]
		for len(ps) >= 2 && len(x) >= 12 {
			q0 := &ps[0]
			q1 := &ps[1]
			v := (*[12]float64)(x)
			s000 := 0.0
			s000 += q0[0] * v[0]
			s000 += q0[1] * v[1]
			s000 += q0[2] * v[2]
			s000 += q0[3] * v[3]
			a00 += s000
			s001 := 0.0
			s001 += q0[4] * v[0]
			s001 += q0[5] * v[1]
			s001 += q0[6] * v[2]
			s001 += q0[7] * v[3]
			a01 += s001
			s002 := 0.0
			s002 += q0[8] * v[0]
			s002 += q0[9] * v[1]
			s002 += q0[10] * v[2]
			s002 += q0[11] * v[3]
			a02 += s002
			s003 := 0.0
			s003 += q0[12] * v[0]
			s003 += q0[13] * v[1]
			s003 += q0[14] * v[2]
			s003 += q0[15] * v[3]
			a03 += s003
			s010 := 0.0
			s010 += q0[0] * v[4]
			s010 += q0[1] * v[5]
			s010 += q0[2] * v[6]
			s010 += q0[3] * v[7]
			a10 += s010
			s011 := 0.0
			s011 += q0[4] * v[4]
			s011 += q0[5] * v[5]
			s011 += q0[6] * v[6]
			s011 += q0[7] * v[7]
			a11 += s011
			s012 := 0.0
			s012 += q0[8] * v[4]
			s012 += q0[9] * v[5]
			s012 += q0[10] * v[6]
			s012 += q0[11] * v[7]
			a12 += s012
			s013 := 0.0
			s013 += q0[12] * v[4]
			s013 += q0[13] * v[5]
			s013 += q0[14] * v[6]
			s013 += q0[15] * v[7]
			a13 += s013
			s100 := 0.0
			s100 += q1[0] * v[4]
			s100 += q1[1] * v[5]
			s100 += q1[2] * v[6]
			s100 += q1[3] * v[7]
			a00 += s100
			s101 := 0.0
			s101 += q1[4] * v[4]
			s101 += q1[5] * v[5]
			s101 += q1[6] * v[6]
			s101 += q1[7] * v[7]
			a01 += s101
			s102 := 0.0
			s102 += q1[8] * v[4]
			s102 += q1[9] * v[5]
			s102 += q1[10] * v[6]
			s102 += q1[11] * v[7]
			a02 += s102
			s103 := 0.0
			s103 += q1[12] * v[4]
			s103 += q1[13] * v[5]
			s103 += q1[14] * v[6]
			s103 += q1[15] * v[7]
			a03 += s103
			s110 := 0.0
			s110 += q1[0] * v[8]
			s110 += q1[1] * v[9]
			s110 += q1[2] * v[10]
			s110 += q1[3] * v[11]
			a10 += s110
			s111 := 0.0
			s111 += q1[4] * v[8]
			s111 += q1[5] * v[9]
			s111 += q1[6] * v[10]
			s111 += q1[7] * v[11]
			a11 += s111
			s112 := 0.0
			s112 += q1[8] * v[8]
			s112 += q1[9] * v[9]
			s112 += q1[10] * v[10]
			s112 += q1[11] * v[11]
			a12 += s112
			s113 := 0.0
			s113 += q1[12] * v[8]
			s113 += q1[13] * v[9]
			s113 += q1[14] * v[10]
			s113 += q1[15] * v[11]
			a13 += s113
			ps = ps[2:]
			x = x[8:]
		}
		for len(ps) >= 1 && len(x) >= 8 {
			q := &ps[0]
			v := (*[8]float64)(x)
			s00 := 0.0
			s00 += q[0] * v[0]
			s00 += q[1] * v[1]
			s00 += q[2] * v[2]
			s00 += q[3] * v[3]
			a00 += s00
			s01 := 0.0
			s01 += q[4] * v[0]
			s01 += q[5] * v[1]
			s01 += q[6] * v[2]
			s01 += q[7] * v[3]
			a01 += s01
			s02 := 0.0
			s02 += q[8] * v[0]
			s02 += q[9] * v[1]
			s02 += q[10] * v[2]
			s02 += q[11] * v[3]
			a02 += s02
			s03 := 0.0
			s03 += q[12] * v[0]
			s03 += q[13] * v[1]
			s03 += q[14] * v[2]
			s03 += q[15] * v[3]
			a03 += s03
			s10 := 0.0
			s10 += q[0] * v[4]
			s10 += q[1] * v[5]
			s10 += q[2] * v[6]
			s10 += q[3] * v[7]
			a10 += s10
			s11 := 0.0
			s11 += q[4] * v[4]
			s11 += q[5] * v[5]
			s11 += q[6] * v[6]
			s11 += q[7] * v[7]
			a11 += s11
			s12 := 0.0
			s12 += q[8] * v[4]
			s12 += q[9] * v[5]
			s12 += q[10] * v[6]
			s12 += q[11] * v[7]
			a12 += s12
			s13 := 0.0
			s13 += q[12] * v[4]
			s13 += q[13] * v[5]
			s13 += q[14] * v[6]
			s13 += q[15] * v[7]
			a13 += s13
			ps = ps[1:]
			x = x[4:]
		}
		for j := jn; j <= min(order, hi-i); j++ {
			q := &prop[j]
			v := (*[4]float64)(xi[4*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			s0 += q[2] * v[2]
			s0 += q[3] * v[3]
			a00 += s0
			s1 := 0.0
			s1 += q[4] * v[0]
			s1 += q[5] * v[1]
			s1 += q[6] * v[2]
			s1 += q[7] * v[3]
			a01 += s1
			s2 := 0.0
			s2 += q[8] * v[0]
			s2 += q[9] * v[1]
			s2 += q[10] * v[2]
			s2 += q[11] * v[3]
			a02 += s2
			s3 := 0.0
			s3 += q[12] * v[0]
			s3 += q[13] * v[1]
			s3 += q[14] * v[2]
			s3 += q[15] * v[3]
			a03 += s3
		}
		dst := (*[8]float64)(next[4*i:])
		dst[0], dst[1], dst[2], dst[3] = a00, a01, a02, a03
		dst[4], dst[5], dst[6], dst[7] = a10, a11, a12, a13
	}
	for ; i <= hi; i++ {
		var a00, a01, a02, a03 float64
		for j := range prop[:min(order, hi-i)+1] {
			q := &prop[j]
			v := (*[4]float64)(xi[4*(i+j):])
			s0 := 0.0
			s0 += q[0] * v[0]
			s0 += q[1] * v[1]
			s0 += q[2] * v[2]
			s0 += q[3] * v[3]
			a00 += s0
			s1 := 0.0
			s1 += q[4] * v[0]
			s1 += q[5] * v[1]
			s1 += q[6] * v[2]
			s1 += q[7] * v[3]
			a01 += s1
			s2 := 0.0
			s2 += q[8] * v[0]
			s2 += q[9] * v[1]
			s2 += q[10] * v[2]
			s2 += q[11] * v[3]
			a02 += s2
			s3 := 0.0
			s3 += q[12] * v[0]
			s3 += q[13] * v[1]
			s3 += q[14] * v[2]
			s3 += q[15] * v[3]
			a03 += s3
		}
		dst := (*[4]float64)(next[4*i:])
		dst[0], dst[1], dst[2], dst[3] = a00, a01, a02, a03
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
	for _, key := range keys {
		c.simulate(sources[key], Y, rowOffset, groups[key])
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
