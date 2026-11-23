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
	// balanceHeadCap caps the tolerance on a leading block of TL·TR − I
	// when the whole transformation fails its certificate.
	balanceHeadCap = 0x1p-6
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
	// e is E = TL·TR − I (row-major n×n) and prefix[h] is ‖E[0:h,0:h]‖_F²,
	// both nil until computed.
	e      []float64
	prefix []float64
	// fe is ‖fl(TL·TR) − I‖_F and round its rounding term
	// (n+1)·ε·‖|TL|·|TR|‖_F, valid once quick is set.
	fe, round float64
	quick     bool
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

func (c *balanceCert) tau(h int) float64 { return c.tauCap(h, balanceCertCap) }

func (c *balanceCert) tauCap(h int, limit float64) float64 {
	s := c.hsv[h-1]
	if !(s > 0) {
		return limit
	}
	return math.Min(limit, math.Max(balanceCertMin, balanceCertK*float64(c.n)*eps()*c.hsv[0]/s))
}

// ok reports whether the leading h×h block of TL·TR − I is within its
// tolerance capped at limit.
func (c *balanceCert) ok(h int, limit float64) bool {
	if c.prefix == nil {
		c.compute()
	}
	return math.Sqrt(c.prefix[h]) <= c.tauCap(h, limit)
}

// nmin is the number of Hankel singular values above the floor.
func (c *balanceCert) nmin() int {
	floor := c.floor()
	k := 0
	for _, v := range c.hsv {
		if v > floor {
			k++
		}
	}
	return k
}

// full reports whether the whole transformation is certified, accepting
// early when the float64 product plus its rounding bound is within τ_n/2.
func (c *balanceCert) full() bool {
	if c.prefix == nil && c.quickBound() <= c.tau(c.n)/2 {
		return true
	}
	return c.ok(c.n, balanceCertCap)
}

// quickBound bounds ‖TL·TR − I‖_F from the float64 product: ‖fl(E)‖_F +
// (n+1)·ε·‖|TL|·|TR|‖_F.
func (c *balanceCert) quickBound() float64 {
	c.pairingTerms()
	return c.fe + c.round
}

// pairingInexact reports whether the float64 pairing error ‖fl(TL·TR) − I‖_F
// exceeds its rounding term (n+1)·ε·‖|TL|·|TR|‖_F, so that TL is not TR⁻¹ to
// the rounding of the product.
func (c *balanceCert) pairingInexact() bool {
	c.pairingTerms()
	return c.fe > c.round
}

func (c *balanceCert) pairingTerms() {
	if c.quick {
		return
	}
	c.quick = true
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
	c.fe, c.round = scaledNorm2(p), float64(n+1)*eps()*scaledNorm2(q)
}

// compute evaluates E = TL·TR − I with compensated dot products (exact
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
				pe := twoProductLo(x, y, p)
				t := float64(s + p)
				z := float64(t - s)
				err += float64(float64(s-float64(t-z))+float64(p-z)) + pe
				s = t
			}
			e[i*n+j] = s + err
		}
	}
	c.e = e
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
// states are certified to the head tolerance, for a transformation failing
// its full certificate: it never keeps part of a tail at or below the
// rounding floor and drops only directions with HSV ≤ band. ok is false when
// a direction above the band fails its certificate.
func (c *balanceCert) certifiedOrder() (h int, ok bool) {
	band := c.band()
	h = c.nmin()
	if h == c.n {
		if c.hsv[c.n-1] > band {
			return 0, false
		}
		h = c.n - 1
	}
	for h > 0 && !c.ok(h, balanceHeadCap) {
		if c.hsv[h-1] > band {
			return 0, false
		}
		h--
	}
	return h, h > 0
}
