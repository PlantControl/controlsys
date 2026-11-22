package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
)

// decouplingSepTol is MATLAB's stabsep/freqsep SepTol default: G1 + G2 may
// differ from G by at most SepTol times the rounding accuracy of G(s).
const decouplingSepTol = 10

// decouplingCertificate decides whether the Sylvester decoupling of an
// ordered real Schur form, T11·X − X·T22 = −T12, gives parts with
// G1 + G2 within decouplingSepTol of the normwise rounding accuracy
// U·ε·(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖), M = (sI − A)⁻¹, U = 4(n+5), of the
// model given to Dgees, at every s. The computed parts are exactly the
// decoupled parts of (T − [0 R; 0 0], B̃ + [δb; 0], C̃ + [0 δc]) with
// R = T11·X − X·T22 + T12 and δb, δc the rounding of B̃1 − X·B̃2 and
// C̃2 + C̃1·X, and the diagonal blocks of (sI − T)⁻¹ are bounded by the
// whole inverse, so (Frobenius norms, budget β·U·ε of the SepTol·U·ε bar
// left after the Schur form and the evaluation):
//
//	P1: ‖C̃1‖‖B̃2‖(‖R̂‖ + ½(n+1)ε((‖T11‖+‖T22‖)‖X‖ + ‖T12‖)) ≤ β·U·ε·‖A‖‖C‖‖B‖
//	P2: ½[(n2+1)‖C̃1‖(‖B̃1‖ + ‖X‖‖B̃2‖) + (n1+1)(‖C̃2‖ + ‖C̃1‖‖X‖)‖B̃2‖] ≤ β·U·‖C‖‖B‖
//
// R̂ is the computed residual; B̃1 and C̃2 are taken before the update.
// The coupling X matters only through outputs that see group 1 (C̃1) and
// inputs that drive group 2 (B̃2): a large X with negligible ‖C̃1‖‖B̃2‖
// is accepted.
type decouplingCertificate struct {
	n, n1, n2           int
	a, b, c             float64
	t11, t22, t12, x, r float64
	b1, b2, c1, c2      float64
}

// decouplingBudget is β: the share of the SepTol·U·ε bar the decoupling may
// use.
const decouplingBudget = decouplingSepTol - 1

func newDecouplingCertificate(sys *System, t, x, bt, ct []float64, n1 int) decouplingCertificate {
	n, m, p := sys.Dims()
	n2 := n - n1
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	d := decouplingCertificate{
		n: n, n1: n1, n2: n2,
		a:   froStrided(aRaw.Data, aRaw.Stride, n, n),
		b:   froStrided(bRaw.Data, bRaw.Stride, n, m),
		c:   froStrided(cRaw.Data, cRaw.Stride, p, n),
		t11: froStrided(t, n, n1, n1),
		t22: froStrided(t[n1*n+n1:], n, n2, n2),
		t12: froStrided(t[n1:], n, n1, n2),
		x:   froStrided(x, n2, n1, n2),
		b1:  froStrided(bt, m, n1, m),
		b2:  froStrided(bt[n1*m:], m, n2, m),
	}
	if p > 0 {
		d.c1, d.c2 = froStrided(ct, n, p, n1), froStrided(ct[n1:], n, p, n2)
	}
	res := make([]float64, n1*n2)
	copyStrided(res, n2, t[n1:], n, n1, n2)
	rg := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: res}
	xg := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: x}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n1, Cols: n1, Stride: n, Data: t}, xg, 1, rg)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, xg, blas64.General{Rows: n2, Cols: n2, Stride: n, Data: t[n1*n+n1:]}, 1, rg)
	d.r = froStrided(res, n2, n1, n2)
	return d
}

// accept reports P1 and P2 for the updated blocks b1 = B̃1 − X·B̃2 and
// c2 = C̃2 + C̃1·X, which must be finite like X and R̂. Ratios to ‖B‖ and
// ‖C‖ keep the products in range; a model with B = 0 or C = 0 has G ≡ D
// and is accepted.
func (d decouplingCertificate) accept(b1, c2 []float64) bool {
	if !isFinite(d.x) || !isFinite(d.r) || !allFinite(b1) || !allFinite(c2) {
		return false
	}
	if d.b == 0 || d.c == 0 {
		return true
	}
	e := eps()
	u := 4 * float64(d.n+5)
	c1, c2n, b1n, b2 := d.c1/d.c, d.c2/d.c, d.b1/d.b, d.b2/d.b
	p1 := c1 * b2 * (d.r + 0.5*float64(d.n+1)*e*((d.t11+d.t22)*d.x+d.t12))
	if !(p1 <= decouplingBudget*u*e*d.a) {
		return false
	}
	p2 := 0.5 * (float64(d.n2+1)*c1*(b1n+d.x*b2) + float64(d.n1+1)*(c2n+c1*d.x)*b2)
	return p2 <= decouplingBudget*u
}

// froStrided is the Frobenius norm of a strided block: a plain sum of
// squares, redone with the scaled l2 accumulator when it overflows or may
// have lost entries to underflow.
func froStrided(data []float64, stride, rows, cols int) float64 {
	if rows == 0 || cols == 0 {
		return 0
	}
	var sum float64
	for i := range rows {
		for _, v := range data[i*stride : i*stride+cols] {
			sum += v * v
		}
	}
	if sum > 0x1p-900 && sum < 0x1p900 {
		return math.Sqrt(sum)
	}
	return froStridedScaled(data, stride, rows, cols)
}

func froStridedScaled(data []float64, stride, rows, cols int) float64 {
	var s l2
	for i := range rows {
		for _, v := range data[i*stride : i*stride+cols] {
			s.add(v)
		}
	}
	return s.norm()
}
