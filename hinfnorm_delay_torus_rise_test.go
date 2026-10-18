package controlsys

import (
	"cmp"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

// torusRiseCase is a random model H with nd internal delays, n states and
// m inputs, p outputs (m, p <= 2) in the layout of SetDelayModel, the delays
// in classes cls with multiples k of one phase each.
type torusRiseCase struct {
	n, m, p, nd int
	a, b, c, d  []float64
	tau         []float64
	cls, k      []int
	dim         int
}

// torusRiseSeed draws seed i of kind "irr3", τ = (h, r1·h, r2·h) in three
// classes, or "mixed", τ = (h, k·h, r·h) in two, with r, r1, r2 uniform on
// [1.1, 3.4]: A − 2I and the B, C, D blocks uniform, D22 scaled by 0.6/nd.
func torusRiseSeed(kind string, i int) torusRiseCase {
	rng := rand.New(rand.NewPCG(uint64(i), uint64(len(kind))*7919+13))
	u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
	z := torusRiseCase{n: 2 + rng.IntN(2), m: 1 + rng.IntN(2)}
	z.p = z.m
	if rng.Float64() < 0.3 {
		z.p = 3 - z.m
	}
	h := 0.05 + 0.4*rng.Float64()
	irr := func() float64 { return 1.1 + 2.3*rng.Float64() }
	switch kind {
	case "irr3":
		r1, r2 := irr(), irr()
		z.tau, z.cls, z.k, z.dim = []float64{h, r1 * h, r2 * h}, []int{0, 1, 2}, []int{1, 1, 1}, 3
	case "irr2":
		z.tau, z.cls, z.k, z.dim = []float64{h, irr() * h}, []int{0, 1}, []int{1, 1}, 2
	case "mixed":
		k := 2 + rng.IntN(2)
		z.tau, z.cls, z.k, z.dim = []float64{h, float64(k) * h, irr() * h}, []int{0, 0, 1}, []int{1, k, 1}, 2
	}
	z.nd = len(z.tau)
	n, mm, pp := z.n, z.m+z.nd, z.p+z.nd
	fill := func(r, c int, s func(i, j int) float64) []float64 {
		out := make([]float64, r*c)
		for i := range r {
			for j := range c {
				out[i*c+j] = u(s(i, j))
			}
		}
		return out
	}
	one := func(int, int) float64 { return 1 }
	z.a = fill(n, n, one)
	for i := range n {
		z.a[i*n+i] -= 2
	}
	z.b = fill(n, mm, one)
	z.c = fill(pp, n, func(i, _ int) float64 { return map[bool]float64{true: 0.3, false: 1}[i >= z.p] })
	z.d = fill(pp, mm, func(i, j int) float64 {
		switch {
		case i >= z.p && j >= z.m:
			return 0.6 / float64(z.nd)
		case i >= z.p || j >= z.m:
			return 2
		}
		return 1
	})
	return z
}

func (z torusRiseCase) system(t testing.TB) *System {
	t.Helper()
	H, err := NewFromSlices(z.n, z.m+z.nd, z.p+z.nd, z.a, z.b, z.c, z.d, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, z.tau)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// gain is σ_max(G) at s = jw with each delay i at phase ph[i] in place of
// w·τ_i, the states dropped for w = ∞, by Gauss–Jordan elimination of
// (sI − A)·x − B2·Δ·z = B1, −C2·x + (I − D22·Δ)·z = D21, G = C1·x + D11 +
// D12·Δ·z.
func (z torusRiseCase) gain(w float64, ph []float64) float64 {
	n, nd, m, p := z.n, z.nd, z.m, z.p
	if math.IsInf(w, 1) {
		n = 0
	}
	mm, k := m+nd, n+nd
	dl := make([]complex128, nd)
	for i, t := range ph {
		dl[i] = cmplx.Exp(complex(0, -t))
	}
	re := func(v float64) complex128 { return complex(v, 0) }
	d := func(r, c int) complex128 { return re(z.d[r*mm+c]) }
	a := make([]complex128, k*k)
	x := make([]complex128, k*m)
	for r := range n {
		for q := range n {
			a[r*k+q] = re(-z.a[r*z.n+q])
		}
		a[r*k+r] += complex(0, w)
		for q := range nd {
			a[r*k+n+q] = re(-z.b[r*mm+m+q]) * dl[q]
		}
		for j := range m {
			x[r*m+j] = re(z.b[r*mm+j])
		}
	}
	for i := range nd {
		for q := range n {
			a[(n+i)*k+q] = re(-z.c[(p+i)*z.n+q])
		}
		for q := range nd {
			a[(n+i)*k+n+q] = -d(p+i, m+q) * dl[q]
		}
		a[(n+i)*k+n+i] += 1
		for j := range m {
			x[(n+i)*m+j] = d(p+i, j)
		}
	}
	for c := range k {
		piv := c
		for r := c + 1; r < k; r++ {
			if cmplx.Abs(a[r*k+c]) > cmplx.Abs(a[piv*k+c]) {
				piv = r
			}
		}
		for j := range k {
			a[c*k+j], a[piv*k+j] = a[piv*k+j], a[c*k+j]
		}
		for j := range m {
			x[c*m+j], x[piv*m+j] = x[piv*m+j], x[c*m+j]
		}
		for r := range k {
			if r == c {
				continue
			}
			f := a[r*k+c] / a[c*k+c]
			for j := range k {
				a[r*k+j] -= f * a[c*k+j]
			}
			for j := range m {
				x[r*m+j] -= f * x[c*m+j]
			}
		}
	}
	for r := range k {
		for j := range m {
			x[r*m+j] /= a[r*k+r]
		}
	}
	g := make([]complex128, 4)
	for i := range p {
		for j := range m {
			v := d(i, j)
			for q := range n {
				v += re(z.c[i*z.n+q]) * x[q*m+j]
			}
			for q := range nd {
				v += d(i, m+q) * dl[q] * x[(n+q)*m+j]
			}
			g[i*2+j] = v
		}
	}
	return sigma2x2([4]complex128(g))
}

func (z torusRiseCase) sigma(w float64) float64 {
	ph := make([]float64, z.nd)
	for i, t := range z.tau {
		ph[i] = w * t
	}
	return z.gain(w, ph)
}

// torusSup is sup σ_max(G∞) over the torus of the classes by a grid of 360²
// or 64³ cells, the 16 largest refined by a compass search to 1e-13.
func (z torusRiseCase) torusSup() float64 {
	gInf := func(th []float64) float64 {
		ph := make([]float64, z.nd)
		for i := range ph {
			ph[i] = float64(z.k[i]) * th[z.cls[i]]
		}
		return z.gain(math.Inf(1), ph)
	}
	n := map[int]int{2: 360, 3: 64}[z.dim]
	step := 2 * math.Pi / float64(n)
	type cell struct {
		v  float64
		th []float64
	}
	var cells []cell
	th := make([]float64, z.dim)
	total := int(math.Pow(float64(n), float64(z.dim)))
	for idx := range total {
		r := idx
		for c := range z.dim {
			th[c] = float64(r%n) * step
			r /= n
		}
		cells = append(cells, cell{gInf(th), slices.Clone(th)})
	}
	slices.SortFunc(cells, func(a, b cell) int { return cmp.Compare(b.v, a.v) })
	best := 0.0
	for _, c := range cells[:16] {
		x, v := c.th, c.v
		for s := step; s > 1e-13; {
			moved := false
			for k := range z.dim {
				for _, sg := range []float64{1, -1} {
					y := slices.Clone(x)
					y[k] += sg * s
					if f := gInf(y); f > v {
						x, v, moved = y, f, true
					}
				}
			}
			if !moved {
				s /= 2
			}
		}
		best = max(best, v)
	}
	return best
}

// TestHinfNormTorusFixedPointRise checks models whose torus sup S* of
// σ_max(G∞) lies above the finite grid peak at a fixed point θ* ∈ {0, π}^dim
// where passages rise above S*, by κ > 0, only by O(1/ω²) and only where
// they come within O(1/ω) of θ*: the norm is S* at ω = +Inf, against a
// dense torus-grid oracle, with no sample of a dense scan of the
// closed-form response to ω = 1e5 above it by more than peakCertTol. The
// search for a finite sample above S* used to give up unsupported.
func TestHinfNormTorusFixedPointRise(t *testing.T) {
	type tc struct {
		name  string
		sys   *System
		sigma func(float64) float64
		sup   float64
		tmax  float64
	}
	var cases []tc
	for _, s := range []struct {
		kind string
		i    int
	}{{"irr3", 4}, {"irr3", 193}, {"mixed", 25}, {"mixed", 56}} {
		z := torusRiseSeed(s.kind, s.i)
		cases = append(cases, tc{s.kind, z.system(t), z.sigma, z.torusSup(), slices.Max(z.tau)})
	}
	for _, i := range []int{7, 63} {
		rng := rand.New(rand.NewPCG(uint64(i+1000), 7))
		u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
		d22 := []float64{u(.8), u(.8), u(.8), u(.8)}
		d12 := []float64{u(1), u(1)}
		d21 := []float64{u(1), u(1)}
		h := 0.02 + 0.3*rng.Float64()
		tau := []float64{h, math.Sqrt2 * h}
		sys, resp, _ := feedthroughDifferenceResponse(t, d22, d12, d21, tau, [2]float64{1, 1})
		re := func(v float64) complex128 { return complex(v, 0) }
		sup := torusSupOf(func(p, q float64) float64 {
			e1, e2 := cmplx.Exp(complex(0, -p)), cmplx.Exp(complex(0, -q))
			m := [4]complex128{1 - re(d22[0])*e1, -re(d22[1]) * e2, -re(d22[2]) * e1, 1 - re(d22[3])*e2}
			det := m[0]*m[3] - m[1]*m[2]
			v0 := (m[3]*re(d21[0]) - m[1]*re(d21[1])) / det
			v1 := (m[0]*re(d21[1]) - m[2]*re(d21[0])) / det
			return cmplx.Abs(0.1 + re(d12[0])*e1*v0 + re(d12[1])*e2*v1)
		})
		cases = append(cases, tc{"sqrt2", sys, func(w float64) float64 { return cmplx.Abs(resp(w)) }, sup, tau[1]})
	}
	for i, c := range cases {
		got, w, err := HinfNorm(c.sys)
		if err != nil {
			t.Errorf("%d %s: %v", i, c.name, err)
			continue
		}
		if math.Abs(got-c.sup) > 1e-9*c.sup || !math.IsInf(w, 1) {
			t.Errorf("%d %s: norm %.15g at %g, want torus sup %.15g at +Inf", i, c.name, got, w, c.sup)
		}
		if fin, wf := denseScanMax(c.sigma, 1e5, math.Pi/(8*c.tmax)); fin > c.sup*(1+peakCertTol) {
			t.Errorf("%d %s: finite sample %.15g at %g above torus sup %.15g", i, c.name, fin, wf, c.sup)
		}
	}
}

// denseScanMax returns the largest sample of g on [0, wMax] in steps dw,
// the 8 largest local maxima refined by golden section.
func denseScanMax(g func(float64) float64, wMax, dw float64) (float64, float64) {
	type peak struct{ w, v float64 }
	var top []peak
	prev, cur := math.Inf(-1), g(0)
	for w := 0.0; w <= wMax; w += dw {
		next := g(w + dw)
		if cur >= prev && cur >= next {
			top = append(top, peak{w, cur})
			if len(top) > 64 {
				slices.SortFunc(top, func(a, b peak) int { return cmp.Compare(b.v, a.v) })
				top = top[:8]
			}
		}
		prev, cur = cur, next
	}
	slices.SortFunc(top, func(a, b peak) int { return cmp.Compare(b.v, a.v) })
	best, wBest := 0.0, 0.0
	for _, p := range top[:min(8, len(top))] {
		w, v := goldenMax(max(p.w-dw, 0), p.w+dw, p.w, p.v, g)
		if v > best {
			best, wBest = v, w
		}
	}
	return best, wBest
}
