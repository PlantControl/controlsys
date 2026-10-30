package controlsys

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// The delay chain kernels must reproduce the plain block recurrence bit for
// bit (IK5K5L): shipped Step/Initial/Lsim/Impulse samples may not change.

func chainPlainAddMulVec(dst []float64, a *mat.Dense, x []float64) {
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

func chainPlainPropagate(next, xi []float64, prop []*mat.Dense, na, order, hi int) {
	for i := 0; i <= hi; i++ {
		dst := next[i*na : (i+1)*na]
		clear(dst)
		for j := 0; j <= order && i+j <= hi; j++ {
			chainPlainAddMulVec(dst, prop[j], xi[(i+j)*na:(i+j+1)*na])
		}
	}
}

// chainPlainSimulate is the delay chain sampling loop before the kernels.
func chainPlainSimulate(c *delayChain, sources []chainSource) *mat.Dense {
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
			chainPlainPropagate(next, xi, c.prop, na, c.order, hi)
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

// chainSpecialValue mixes in ±0, ±Inf, subnormals and overflowing
// magnitudes. NaNs then arise only from 0·Inf and Inf−Inf, all the default
// NaN; which of two distinct NaN payloads survives a commutative add is set
// by register allocation, not by the source.
func chainSpecialValue(rng *rand.Rand) float64 {
	switch rng.IntN(30) {
	case 0:
		return math.Copysign(0, -1)
	case 1:
		return 0
	case 2:
		return math.Inf(1)
	case 3:
		return math.Inf(-1)
	case 4:
		return 5e-324 * float64(1+rng.IntN(1000))
	case 5:
		return math.Copysign(1e300, rng.NormFloat64())
	}
	return rng.NormFloat64() * math.Pow(10, float64(rng.IntN(9)-4))
}

func TestDelayChainKernelsMatchPlainRecurrenceBits(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 4410))
	for trial := range 400 {
		na := 1 + rng.IntN(6)
		order := []int{0, 1, 2, 3, 5, 8, 16}[rng.IntN(7)]
		blocks := 1 + rng.IntN(24)
		specials := trial%3 == 0
		val := func() float64 {
			if specials {
				return chainSpecialValue(rng)
			}
			return rng.NormFloat64()
		}
		prop := make([]*mat.Dense, order+1)
		flat := make([]float64, 0, (order+1)*na*na)
		for j := range prop {
			prop[j] = mat.NewDense(na, na, nil)
			for r := range na {
				for c := range na {
					v := val()
					prop[j].Set(r, c, v)
					flat = append(flat, v)
				}
			}
		}
		xi := make([]float64, blocks*na)
		for k := range xi {
			xi[k] = val()
		}
		kernel := chainKernel(flat, na, order)
		for hi := range blocks {
			want := make([]float64, blocks*na)
			got := make([]float64, blocks*na)
			chainPlainPropagate(want, xi, prop, na, order, hi)
			kernel(got, xi, hi)
			for k := range (hi + 1) * na {
				if math.Float64bits(got[k]) != math.Float64bits(want[k]) {
					t.Fatalf("trial %d na=%d order=%d hi=%d: next[%d] = %v (%#x), plain %v (%#x)",
						trial, na, order, hi, k, got[k], math.Float64bits(got[k]), want[k], math.Float64bits(want[k]))
				}
			}
		}
	}
}

func TestDelayChainSimulateMatchesPlainLoopBits(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 9021))
	g := func(r, c int, s float64) *mat.Dense {
		d := mat.NewDense(r, c, nil)
		for i := range r {
			for j := range c {
				d.Set(i, j, s*rng.NormFloat64())
			}
		}
		return d
	}
	checked := map[int]bool{}
	for trial := range 60 {
		n, m, p := 1+rng.IntN(5), 1+rng.IntN(2), 1+rng.IntN(3)
		A := g(n, n, 1)
		for i := range n {
			A.Set(i, i, A.At(i, i)-2)
		}
		sys, err := New(A, g(n, m, 1), g(p, n, 1), g(p, m, 0.5), 0)
		if err != nil {
			t.Fatal(err)
		}
		q := 1 + rng.IntN(2)
		tau := make([]float64, q)
		for i := range tau {
			tau[i] = 0.13 + 0.01*float64(trial%5)
		}
		d22 := 0.0
		if trial%2 == 1 {
			d22 = 0.3
		}
		if err := sys.SetInternalDelay(tau, g(n, q, 0.5), g(q, n, 0.5), g(p, q, 0.5), g(q, m, 0.5), g(q, q, d22)); err != nil {
			t.Fatal(err)
		}
		dt := 0.0071 + 0.002*rng.Float64()
		steps := 150 + rng.IntN(100)
		var inputs []int
		if trial%3 != 0 {
			for i := range m {
				inputs = append(inputs, i)
			}
		}
		c, ok := newDelayChain(sys, tau[0], inputs, dt, steps, 2)
		if !ok {
			continue
		}
		checked[c.na] = true
		var sources []chainSource
		x0 := make([]float64, c.na)
		for s := range n {
			x0[s] = rng.NormFloat64()
		}
		sources = append(sources, chainSource{shift: 0, dir: x0, amp: []float64{1}})
		for in := range inputs {
			dir := make([]float64, c.na)
			dir[n+in] = 1
			amp := make([]float64, steps)
			for k := range amp {
				if rng.IntN(4) == 0 {
					amp[k] = rng.NormFloat64()
				}
			}
			amp[rng.IntN(steps)] = math.Copysign(0, -1)
			sources = append(sources, chainSource{shift: 0.05 * float64(rng.IntN(3)), dir: dir, amp: amp})
		}
		want := chainPlainSimulate(c, sources)
		var rows []int
		for r := range p {
			if p == 1 || r != trial%p {
				rows = append(rows, r)
			}
		}
		const mark = 7.5
		got := mat.NewDense(p+1, steps, nil)
		for r := range p + 1 {
			for k := range steps {
				got.Set(r, k, mark)
			}
		}
		c.simulate(sources, got, 1, rows)
		for r := range p + 1 {
			listed := r > 0 && slices.Contains(rows, r-1)
			for k := range steps {
				g := got.At(r, k)
				if !listed {
					if g != mark {
						t.Fatalf("trial %d: unlisted row %d changed at %d to %v", trial, r, k, g)
					}
					continue
				}
				if math.Float64bits(g) != math.Float64bits(want.At(r-1, k)) {
					t.Fatalf("trial %d na=%d order=%d blocks=%d: y[%d][%d] = %v, plain %v", trial, c.na, c.order, c.blocks, r-1, k, g, want.At(r-1, k))
				}
			}
		}
	}
	for _, na := range []int{2, 3, 4, 5} {
		if !checked[na] {
			t.Errorf("no chain with na=%d sampled", na)
		}
	}
}
