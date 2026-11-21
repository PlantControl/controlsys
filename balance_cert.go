package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

const (
	// balanceCertCap caps the tolerance on TL·TR − I: above it the balancing
	// is no similarity worth the name, however small the Hankel singular
	// values.
	balanceCertCap = 0x1p-10
	// balanceCertMin is the tolerance floor on TL·TR − I.
	balanceCertMin = 1e-8
	// balanceCertK scales the rounding term K·n·ε·σ₁/σ_h of the tolerance.
	balanceCertK = 4
)

// balanceCert certifies the balancing transformation of a BalrealResult by
// E = TL·TR − I, evaluated with error-free products: the leading h×h block
// is accepted when ‖E[0:h,0:h]‖_F ≤ τ_h = min(2⁻¹⁰, max(10⁻⁸, 4·n·ε·σ₁/σ_h)).
type balanceCert struct {
	n      int
	hsv    []float64
	tl, tr *mat.Dense
	// prefix[h] is ‖E[0:h,0:h]‖_F², nil until computed.
	prefix []float64
}

func newBalanceCert(br *BalrealResult) *balanceCert {
	return &balanceCert{n: len(br.HSV), hsv: br.HSV, tl: br.TL, tr: br.TR}
}

// floor is n·ε·σ₁, the level below which a Hankel singular value is only
// known to rounding.
func (c *balanceCert) floor() float64 { return float64(c.n) * eps() * c.hsv[0] }

// band is the largest Hankel singular value whose direction may be dropped
// when the transformation fails its certificate: where τ_h reaches its cap.
func (c *balanceCert) band() float64 { return c.floor() * balanceCertK / balanceCertCap }

func (c *balanceCert) tau(h int) float64 {
	s := c.hsv[h-1]
	if !(s > 0) {
		return balanceCertCap
	}
	return math.Min(balanceCertCap, math.Max(balanceCertMin, balanceCertK*float64(c.n)*eps()*c.hsv[0]/s))
}

// ok reports whether the leading h×h block of TL·TR − I is within τ_h.
func (c *balanceCert) ok(h int) bool {
	if c.prefix == nil {
		c.compute()
	}
	return math.Sqrt(c.prefix[h]) <= c.tau(h)
}

// full reports whether the whole transformation is certified, accepting
// early when the float64 product plus its rounding bound is within τ_n/2.
func (c *balanceCert) full() bool {
	if c.prefix == nil && c.quickBound() <= c.tau(c.n)/2 {
		return true
	}
	return c.ok(c.n)
}

// quickBound bounds ‖TL·TR − I‖_F from the float64 product: ‖fl(E)‖_F +
// (n+1)·ε·‖|TL|·|TR|‖_F.
func (c *balanceCert) quickBound() float64 {
	n := c.n
	tl, tr := c.tl.RawMatrix(), c.tr.RawMatrix()
	buf := make([]float64, 4*n*n)
	p, q, atl, atr := buf[:n*n], buf[n*n:2*n*n], buf[2*n*n:3*n*n], buf[3*n*n:]
	for i := range n {
		for j := range n {
			atl[i*n+j] = math.Abs(tl.Data[i*tl.Stride+j])
			atr[i*n+j] = math.Abs(tr.Data[i*tr.Stride+j])
		}
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tl, tr, 0, blas64.General{Rows: n, Cols: n, Stride: n, Data: p})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: atl},
		blas64.General{Rows: n, Cols: n, Stride: n, Data: atr},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: q})
	for i := range n {
		p[i*n+i]--
	}
	return scaledNorm2(p) + float64(n+1)*eps()*scaledNorm2(q)
}

// compute evaluates E = TL·TR − I with compensated dot products (FMA
// TwoProduct, TwoSum) and its leading-block norms.
func (c *balanceCert) compute() {
	n := c.n
	tl, tr := c.tl.RawMatrix(), c.tr.RawMatrix()
	col := make([]float64, n)
	e := make([]float64, n*n)
	for j := range n {
		for k := range n {
			col[k] = tr.Data[k*tr.Stride+j]
		}
		for i := range n {
			row := tl.Data[i*tl.Stride : i*tl.Stride+n]
			var s, err float64
			if i == j {
				s = -1
			}
			for k, x := range row {
				y := col[k]
				p := float64(x * y)
				pe := math.FMA(x, y, -p)
				t := float64(s + p)
				z := float64(t - s)
				err += float64(float64(s-float64(t-z))+float64(p-z)) + pe
				s = t
			}
			e[i*n+j] = s + err
		}
	}
	c.prefix = make([]float64, n+1)
	for h := 1; h <= n; h++ {
		k := h - 1
		add := 0.0
		for j := range h {
			add += e[k*n+j] * e[k*n+j]
		}
		for i := range k {
			add += e[i*n+k] * e[i*n+k]
		}
		c.prefix[h] = c.prefix[h-1] + add
	}
}

// certifiedOrder returns the largest h such that the leading h balanced
// states are certified, never keeping part of a tail at or below the
// rounding floor and dropping only directions with HSV ≤ band. ok is false
// when a direction above the band fails its certificate.
func (c *balanceCert) certifiedOrder() (h int, ok bool) {
	floor := c.floor()
	band := c.band()
	nmin := 0
	for _, v := range c.hsv {
		if v > floor {
			nmin++
		}
	}
	h = nmin
	if h == c.n {
		if c.hsv[c.n-1] > band {
			return 0, false
		}
		h = c.n - 1
	}
	for h > 0 && !c.ok(h) {
		if c.hsv[h-1] > band {
			return 0, false
		}
		h--
	}
	return h, h > 0
}
