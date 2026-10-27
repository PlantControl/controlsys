package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

const stepOraclePrec = 320

func stepOracleFloat(x float64) *big.Float {
	return new(big.Float).SetPrec(stepOraclePrec).SetFloat64(x)
}

func stepOracleNew() *big.Float { return new(big.Float).SetPrec(stepOraclePrec) }

type stepOracleComplex struct{ re, im *big.Float }

func (z stepOracleComplex) complex128() complex128 {
	r, _ := z.re.Float64()
	i, _ := z.im.Float64()
	return complex(r, i)
}

func stepOracleQuo(a, b stepOracleComplex) stepOracleComplex {
	den := stepOracleNew().Add(stepOracleNew().Mul(b.re, b.re), stepOracleNew().Mul(b.im, b.im))
	re := stepOracleNew().Add(stepOracleNew().Mul(a.re, b.re), stepOracleNew().Mul(a.im, b.im))
	im := stepOracleNew().Sub(stepOracleNew().Mul(a.im, b.re), stepOracleNew().Mul(a.re, b.im))
	return stepOracleComplex{re.Quo(re, den), im.Quo(im, den)}
}

// stepOracleSolve returns M⁻¹X by Gauss-Jordan elimination with partial
// pivoting at stepOraclePrec bits, or nil for an exactly singular M.
func stepOracleSolve(M, X [][]*big.Float) [][]*big.Float {
	n, k := len(M), len(X[0])
	W := make([][]*big.Float, n)
	for i := range n {
		W[i] = make([]*big.Float, n+k)
		for j := range n {
			W[i][j] = stepOracleNew().Set(M[i][j])
		}
		for j := range k {
			W[i][n+j] = stepOracleNew().Set(X[i][j])
		}
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if new(big.Float).Abs(W[r][c]).Cmp(new(big.Float).Abs(W[piv][c])) > 0 {
				piv = r
			}
		}
		W[c], W[piv] = W[piv], W[c]
		if W[c][c].Sign() == 0 {
			return nil
		}
		for r := range n {
			if r == c || W[r][c].Sign() == 0 {
				continue
			}
			f := stepOracleNew().Quo(W[r][c], W[c][c])
			for j := c; j < n+k; j++ {
				W[r][j].Sub(W[r][j], stepOracleNew().Mul(f, W[c][j]))
			}
		}
	}
	out := make([][]*big.Float, n)
	for i := range n {
		out[i] = make([]*big.Float, k)
		for j := range k {
			out[i][j] = stepOracleNew().Quo(W[i][n+j], W[i][i])
		}
	}
	return out
}

// stepOraclePencil holds a descriptor model at stepOraclePrec bits.
type stepOraclePencil struct {
	n, m, p    int
	a, e, b, c [][]*big.Float
	d          *mat.Dense
}

func newStepOraclePencil(sys *System) stepOraclePencil {
	n, m, p := sys.Dims()
	conv := func(x *mat.Dense, r, c int) [][]*big.Float {
		out := make([][]*big.Float, r)
		for i := range r {
			out[i] = make([]*big.Float, c)
			for j := range c {
				v := 0.0
				if x != nil {
					v = x.At(i, j)
				} else if i == j {
					v = 1
				}
				out[i][j] = stepOracleFloat(v)
			}
		}
		return out
	}
	return stepOraclePencil{n: n, m: m, p: p, a: conv(sys.A, n, n), e: conv(sys.E, n, n),
		b: conv(sys.B, n, m), c: conv(sys.C, p, n), d: sys.D}
}

// solveShifted returns (sE − A)⁻¹X for real X, by the real 2n embedding
// of the complex system, as (real part, imaginary part).
func (o stepOraclePencil) solveShifted(s stepOracleComplex, X [][]*big.Float) (re, im [][]*big.Float) {
	n := o.n
	K := make([][]*big.Float, 2*n)
	for i := range 2 * n {
		K[i] = make([]*big.Float, 2*n)
	}
	for i := range n {
		for j := range n {
			r := stepOracleNew().Sub(stepOracleNew().Mul(s.re, o.e[i][j]), o.a[i][j])
			q := stepOracleNew().Mul(s.im, o.e[i][j])
			K[i][j], K[n+i][n+j] = r, stepOracleNew().Set(r)
			K[i][n+j], K[n+i][j] = stepOracleNew().Neg(q), q
		}
	}
	k := len(X[0])
	R := make([][]*big.Float, 2*n)
	for i := range 2 * n {
		R[i] = make([]*big.Float, k)
		for j := range k {
			if i < n {
				R[i][j] = stepOracleNew().Set(X[i][j])
			} else {
				R[i][j] = stepOracleFloat(0)
			}
		}
	}
	S := stepOracleSolve(K, R)
	if S == nil {
		return nil, nil
	}
	return S[:n], S[n:]
}

// root refines a generalized eigenvalue of (A, E) from x0 by Newton on
// log det(λE − A), whose derivative is tr((λE − A)⁻¹E).
func (o stepOraclePencil) root(x0 complex128) (stepOracleComplex, bool) {
	lam := stepOracleComplex{stepOracleFloat(real(x0)), stepOracleFloat(imag(x0))}
	for range 200 {
		re, im := o.solveShifted(lam, o.e)
		if re == nil {
			return lam, true
		}
		tr := stepOracleComplex{stepOracleFloat(0), stepOracleFloat(0)}
		for i := range o.n {
			tr.re.Add(tr.re, re[i][i])
			tr.im.Add(tr.im, im[i][i])
		}
		if tr.re.Sign() == 0 && tr.im.Sign() == 0 {
			return lam, false
		}
		step := stepOracleQuo(stepOracleComplex{stepOracleFloat(1), stepOracleFloat(0)}, tr)
		lam.re.Sub(lam.re, step.re)
		lam.im.Sub(lam.im, step.im)
		if cmplx.Abs(step.complex128()) <= 1e-70*math.Max(1, cmplx.Abs(lam.complex128())) {
			return lam, true
		}
	}
	return lam, false
}

// transfer returns G(s) = C(sE − A)⁻¹B + D at stepOraclePrec bits.
func (o stepOraclePencil) transfer(s stepOracleComplex) (re, im [][]*big.Float) {
	xr, xi := o.solveShifted(s, o.b)
	if xr == nil {
		return nil, nil
	}
	re, im = make([][]*big.Float, o.p), make([][]*big.Float, o.p)
	for i := range o.p {
		re[i], im[i] = make([]*big.Float, o.m), make([]*big.Float, o.m)
		for j := range o.m {
			r, q := stepOracleFloat(o.d.At(i, j)), stepOracleFloat(0)
			for k := range o.n {
				r.Add(r, stepOracleNew().Mul(o.c[i][k], xr[k][j]))
				q.Add(q, stepOracleNew().Mul(o.c[i][k], xi[k][j]))
			}
			re[i][j], im[i][j] = r, q
		}
	}
	return re, im
}

// stepOracleModes is the partial-fraction form G(s) = G(∞) + Σ Rᵢ/(s − λᵢ)
// of a descriptor model with n distinct finite poles, at stepOraclePrec
// bits, so its step response y(t) = G(0) + Σ Rᵢ·e^{λᵢt}/λᵢ is exact to
// float64 for the given float64 coefficients.
type stepOracleModes struct {
	poles    []complex128
	residues [][][]complex128
	dc       *mat.Dense
}

// newStepOracleModes refines starts (one per pole) to the n generalized
// eigenvalues of (A, E); ok is false unless they converge to n distinct
// nonzero roots.
func newStepOracleModes(sys *System, starts []complex128) (stepOracleModes, bool) {
	o := newStepOraclePencil(sys)
	if len(starts) != o.n {
		return stepOracleModes{}, false
	}
	var modes stepOracleModes
	var roots []stepOracleComplex
	for _, s := range starts {
		r, ok := o.root(s)
		if !ok {
			return stepOracleModes{}, false
		}
		v := r.complex128()
		if v == 0 || cmplx.IsNaN(v) || cmplx.IsInf(v) {
			return stepOracleModes{}, false
		}
		for _, q := range modes.poles {
			if cmplx.Abs(q-v) <= 1e-8*math.Max(1, cmplx.Abs(v)) {
				return stepOracleModes{}, false
			}
		}
		modes.poles = append(modes.poles, v)
		roots = append(roots, r)
	}
	for _, r := range roots {
		mag := stepOracleFloat(math.Max(1, cmplx.Abs(r.complex128())) * 1e-60)
		s := stepOracleComplex{stepOracleNew().Add(r.re, mag), stepOracleNew().Set(r.im)}
		gr, gi := o.transfer(s)
		if gr == nil {
			return stepOracleModes{}, false
		}
		res := make([][]complex128, o.p)
		for i := range o.p {
			res[i] = make([]complex128, o.m)
			for j := range o.m {
				z := stepOracleComplex{stepOracleNew().Mul(gr[i][j], mag), stepOracleNew().Mul(gi[i][j], mag)}
				res[i][j] = z.complex128()
			}
		}
		modes.residues = append(modes.residues, res)
	}
	gr, _ := o.transfer(stepOracleComplex{stepOracleFloat(0), stepOracleFloat(0)})
	if gr == nil {
		return stepOracleModes{}, false
	}
	modes.dc = mat.NewDense(o.p, o.m, nil)
	for i := range o.p {
		for j := range o.m {
			v, _ := gr[i][j].Float64()
			modes.dc.Set(i, j, v)
		}
	}
	return modes, true
}

// step returns y_ij(t) for t > 0, dropping modes with Re λ·t < −700
// (e^{λt} underflows).
func (m stepOracleModes) step(i, j int, t float64) float64 {
	y := complex(m.dc.At(i, j), 0)
	for k, l := range m.poles {
		if real(l)*t < -700 {
			continue
		}
		y += m.residues[k][i][j] * cmplx.Exp(l*complex(t, 0)) / l
	}
	return real(y)
}

// zoh returns the exact ZOH discretization G_d(z) = G(∞) + Σ Rᵢ·(e^{λᵢT} −
// 1)/λᵢ/(z − e^{λᵢT}) of the modes at z, sample time dt.
func (m stepOracleModes) zoh(i, j int, dt float64, z complex128) complex128 {
	g := complex(m.dc.At(i, j), 0)
	for k, l := range m.poles {
		g += m.residues[k][i][j] / l
	}
	for k, l := range m.poles {
		e := cmplx.Exp(l * complex(dt, 0))
		g += m.residues[k][i][j] * (e - 1) / l / (z - e)
	}
	return g
}

func (m stepOracleModes) stable() bool {
	for _, l := range m.poles {
		if real(l) >= 0 {
			return false
		}
	}
	return true
}
