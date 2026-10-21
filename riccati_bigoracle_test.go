package controlsys

import (
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const hbPrec = 320

type hbMat struct {
	r, c int
	d    []*big.Float
}

func newHbMat(r, c int) hbMat {
	m := hbMat{r, c, make([]*big.Float, r*c)}
	for i := range m.d {
		m.d[i] = new(big.Float).SetPrec(hbPrec)
	}
	return m
}

func hbFrom(a *mat.Dense) hbMat {
	if a == nil {
		return hbMat{}
	}
	r, c := a.Dims()
	m := newHbMat(r, c)
	for i := range r {
		for j := range c {
			m.d[i*c+j].SetFloat64(a.At(i, j))
		}
	}
	return m
}

func (m hbMat) at(i, j int) *big.Float { return m.d[i*m.c+j] }

func (m hbMat) t() hbMat {
	o := newHbMat(m.c, m.r)
	for i := range m.r {
		for j := range m.c {
			o.at(j, i).Set(m.at(i, j))
		}
	}
	return o
}

func hbMul(a, b hbMat) hbMat {
	o := newHbMat(a.r, b.c)
	p := new(big.Float).SetPrec(hbPrec)
	for i := range a.r {
		for j := range b.c {
			for k := range a.c {
				o.at(i, j).Add(o.at(i, j), p.Mul(a.at(i, k), b.at(k, j)))
			}
		}
	}
	return o
}

func hbAdd(a, b hbMat, sign float64) hbMat {
	o := newHbMat(a.r, a.c)
	p := new(big.Float).SetPrec(hbPrec)
	s := new(big.Float).SetPrec(hbPrec).SetFloat64(sign)
	for i := range o.d {
		o.d[i].Add(a.d[i], p.Mul(s, b.d[i]))
	}
	return o
}

// hbSolve solves a·x = b by Gaussian elimination with partial pivoting.
func hbSolve(a, b hbMat) hbMat {
	n := a.r
	a = hbAdd(a, newHbMat(n, n), 1)
	b = hbAdd(b, newHbMat(b.r, b.c), 1)
	p, q := new(big.Float).SetPrec(hbPrec), new(big.Float).SetPrec(hbPrec)
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if p.Abs(a.at(i, k)).Cmp(q.Abs(a.at(piv, k))) > 0 {
				piv = i
			}
		}
		for j := range n {
			a.d[k*n+j], a.d[piv*n+j] = a.d[piv*n+j], a.d[k*n+j]
		}
		for j := range b.c {
			b.d[k*b.c+j], b.d[piv*b.c+j] = b.d[piv*b.c+j], b.d[k*b.c+j]
		}
		for i := k + 1; i < n; i++ {
			f := new(big.Float).SetPrec(hbPrec).Quo(a.at(i, k), a.at(k, k))
			for j := k; j < n; j++ {
				a.at(i, j).Sub(a.at(i, j), p.Mul(f, a.at(k, j)))
			}
			for j := range b.c {
				b.at(i, j).Sub(b.at(i, j), p.Mul(f, b.at(k, j)))
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		for j := range b.c {
			for k := i + 1; k < n; k++ {
				b.at(i, j).Sub(b.at(i, j), p.Mul(a.at(i, k), b.at(k, j)))
			}
			b.at(i, j).Quo(b.at(i, j), a.at(i, i))
		}
	}
	return b
}

// hewerDareBig is Hewer's iteration for the Dare in 320-bit arithmetic
// from the stabilizing gain K0, with Kronecker Stein solves.
func hewerDareBig(A, B, Q, R, S, K0 *mat.Dense, steps int) *mat.Dense {
	a, b, q, r, k := hbFrom(A), hbFrom(B), hbFrom(Q), hbFrom(R), hbFrom(K0)
	s := hbFrom(S)
	n := a.r
	var x hbMat
	for range steps {
		acl := hbAdd(a, hbMul(b, k), -1)
		qk := hbAdd(q, hbMul(hbMul(k.t(), r), k), 1)
		if S != nil {
			sk := hbMul(s, k)
			qk = hbAdd(hbAdd(qk, sk, -1), sk.t(), -1)
		}
		M := newHbMat(n*n, n*n)
		rhs := newHbMat(n*n, 1)
		p := new(big.Float).SetPrec(hbPrec)
		for i := range n {
			for j := range n {
				row := i*n + j
				rhs.d[row].Set(qk.at(i, j))
				for kk := range n {
					for l := range n {
						c := M.at(row, kk*n+l)
						c.Sub(c, p.Mul(acl.at(kk, i), acl.at(l, j)))
						if kk == i && l == j {
							c.Add(c, big.NewFloat(1))
						}
					}
				}
			}
		}
		xv := hbSolve(M, rhs)
		x = newHbMat(n, n)
		for i := range x.d {
			x.d[i].Set(xv.d[i])
		}
		btx := hbMul(b.t(), x)
		rb := hbAdd(r, hbMul(btx, b), 1)
		rh := hbMul(btx, a)
		if S != nil {
			rh = hbAdd(rh, s.t(), 1)
		}
		k = hbSolve(rb, rh)
	}
	o := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v, _ := x.at(i, j).Float64()
			o.Set(i, j, v)
		}
	}
	return o
}

func TestHewerDareBig_MatchesDare(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.4, -0.3, 1.1})
	B := mat.NewDense(2, 1, []float64{0.5, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0.3, 0.3, 1})
	R := mat.NewDense(1, 1, []float64{0.7})
	S := mat.NewDense(2, 1, []float64{0.1, -0.2})
	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(hewerDareBig(A, B, Q, R, S, res.K, 8), res.X); !(e <= 1e-13) {
		t.Errorf("rel err %g", e)
	}
}

// kleinmanCareBig is Newton–Kleinman's iteration for the Care in 320-bit
// arithmetic from the stabilizing gain K0, with Kronecker Lyapunov solves.
func kleinmanCareBig(A, B, Q, R, S, K0 *mat.Dense, steps int) *mat.Dense {
	a, b, q, r, k := hbFrom(A), hbFrom(B), hbFrom(Q), hbFrom(R), hbFrom(K0)
	s := hbFrom(S)
	n := a.r
	var x hbMat
	for range steps {
		acl := hbAdd(a, hbMul(b, k), -1)
		qk := hbAdd(q, hbMul(hbMul(k.t(), r), k), 1)
		if S != nil {
			sk := hbMul(s, k)
			qk = hbAdd(hbAdd(qk, sk, -1), sk.t(), -1)
		}
		M := newHbMat(n*n, n*n)
		rhs := newHbMat(n*n, 1)
		for i := range n {
			for j := range n {
				row := i*n + j
				rhs.d[row].Neg(qk.at(i, j))
				for kk := range n {
					c := M.at(row, kk*n+j)
					c.Add(c, acl.at(kk, i))
					c = M.at(row, i*n+kk)
					c.Add(c, acl.at(kk, j))
				}
			}
		}
		xv := hbSolve(M, rhs)
		x = newHbMat(n, n)
		for i := range x.d {
			x.d[i].Set(xv.d[i])
		}
		rh := hbMul(b.t(), x)
		if S != nil {
			rh = hbAdd(rh, s.t(), 1)
		}
		k = hbSolve(r, rh)
	}
	o := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v, _ := x.at(i, j).Float64()
			o.Set(i, j, v)
		}
	}
	return o
}

func TestKleinmanCareBig_MatchesCare(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.4, -0.3, 1.1})
	B := mat.NewDense(2, 1, []float64{0.5, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0.3, 0.3, 1})
	R := mat.NewDense(1, 1, []float64{0.7})
	S := mat.NewDense(2, 1, []float64{0.1, -0.2})
	res, err := Care(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(kleinmanCareBig(A, B, Q, R, S, res.K, 8), res.X); !(e <= 1e-13) {
		t.Errorf("rel err %g", e)
	}
}
