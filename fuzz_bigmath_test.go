package controlsys

import (
	"math"
	"math/big"
	"sync"
)

// Arch-independent elementary functions for fuzz decoding (QS7HIC). Go's
// math.Exp/Log/Hypot use assembly on amd64 (Exp switches to FMA on AVX+FMA
// CPUs) and the arm64 compiler fuses x*y±z, so float64 decoding differs by
// CPU. These evaluate in math/big (exact integer arithmetic) and round once,
// so results are identical on every GOARCH and CPU.

const (
	fzElemPrec  = 192
	fzConstPrec = fzElemPrec + 64
)

var fzBigConsts = sync.OnceValue(func() (c struct{ ln2, ln10, pi *big.Float }) {
	atanh := func(k int64) *big.Float { return fzSeries(k, false) }
	c.ln2 = new(big.Float).SetPrec(fzConstPrec).Mul(fzBigInt(2), atanh(3))
	c.ln10 = new(big.Float).SetPrec(fzConstPrec).Mul(fzBigInt(3), c.ln2)
	c.ln10.Add(c.ln10, new(big.Float).Mul(fzBigInt(2), atanh(9)))
	pi := new(big.Float).SetPrec(fzConstPrec).Mul(fzBigInt(16), fzSeries(5, true))
	c.pi = pi.Sub(pi, new(big.Float).Mul(fzBigInt(4), fzSeries(239, true)))
	return c
})

// fzSeries returns atan(1/k) (alt) or atanh(1/k) at fzConstPrec.
func fzSeries(k int64, alt bool) *big.Float {
	sum := new(big.Float).SetPrec(fzConstPrec)
	pow := new(big.Float).SetPrec(fzConstPrec).Quo(fzBigInt(1), fzBigInt(k))
	k2 := fzBigInt(k * k)
	tol := new(big.Float).SetMantExp(fzBigInt(1), -fzConstPrec-8)
	var term big.Float
	term.SetPrec(fzConstPrec)
	for j := int64(0); ; j++ {
		term.Quo(pow, fzBigInt(2*j+1))
		if alt && j%2 == 1 {
			sum.Sub(sum, &term)
		} else {
			sum.Add(sum, &term)
		}
		if term.Cmp(tol) < 0 {
			return sum
		}
		pow.Quo(pow, k2)
	}
}

func fzBigInt(k int64) *big.Float { return new(big.Float).SetPrec(fzConstPrec).SetInt64(k) }

func fzBig(x float64) *big.Float { return new(big.Float).SetPrec(fzElemPrec).SetFloat64(x) }

func fzF64(x *big.Float) float64 {
	f, _ := x.Float64()
	return f
}

// fzBigExp returns e^x.
func fzBigExp(x *big.Float) *big.Float {
	c := fzBigConsts()
	q := new(big.Float).SetPrec(fzElemPrec).Quo(x, c.ln2)
	k := math.Round(fzF64(q))
	r := new(big.Float).SetPrec(fzElemPrec).Mul(fzBig(k), c.ln2)
	r.Sub(x, r)
	const halvings = 16
	r.SetMantExp(r, -halvings)
	sum := fzBig(1)
	term := fzBig(1)
	tol := new(big.Float).SetMantExp(fzBig(1), -fzElemPrec-8)
	for j := int64(1); term.Sign() != 0 && new(big.Float).Abs(term).Cmp(tol) >= 0; j++ {
		term.Mul(term, r)
		term.Quo(term, fzBig(float64(j)))
		sum.Add(sum, term)
	}
	for range halvings {
		sum.Mul(sum, sum)
	}
	return sum.SetMantExp(sum, int(k))
}

// fzBigSinCos returns sin x and cos x.
func fzBigSinCos(x *big.Float) (sin, cos *big.Float) {
	c := fzBigConsts()
	halfPi := new(big.Float).SetPrec(fzConstPrec).SetMantExp(c.pi, -1)
	q := math.Round(fzF64(new(big.Float).SetPrec(fzElemPrec).Quo(x, halfPi)))
	r := new(big.Float).SetPrec(fzConstPrec).Mul(new(big.Float).SetPrec(fzConstPrec).SetFloat64(q), halfPi)
	r.Sub(new(big.Float).SetPrec(fzConstPrec).Set(x), r)
	r.SetPrec(fzElemPrec)
	r2 := new(big.Float).SetPrec(fzElemPrec).Mul(r, r)
	s, cs := fzBig(0), fzBig(0)
	tol := new(big.Float).SetMantExp(fzBig(1), -fzElemPrec-8)
	term := new(big.Float).SetPrec(fzElemPrec).Set(r)
	for j := int64(1); ; j += 2 {
		s.Add(s, term)
		if new(big.Float).Abs(term).Cmp(tol) < 0 {
			break
		}
		term.Mul(term, r2)
		term.Quo(term, fzBig(float64(-(j+1)*(j+2))))
	}
	term = fzBig(1)
	for j := int64(0); ; j += 2 {
		cs.Add(cs, term)
		if new(big.Float).Abs(term).Cmp(tol) < 0 {
			break
		}
		term.Mul(term, r2)
		term.Quo(term, fzBig(float64(-(j+1)*(j+2))))
	}
	switch int64(q) & 3 {
	case 1:
		s, cs = cs, s.Neg(s)
	case 2:
		s, cs = s.Neg(s), cs.Neg(cs)
	case 3:
		s, cs = cs.Neg(cs), s
	}
	return s, cs
}

// fzBigPow10 returns 10^x.
func fzBigPow10(x *big.Float) *big.Float {
	return fzBigExp(new(big.Float).SetPrec(fzElemPrec).Mul(x, fzBigConsts().ln10))
}

func fzPow10(x float64) float64 { return fzF64(fzBigPow10(fzBig(x))) }

func fzCos(x float64) float64 {
	_, c := fzBigSinCos(fzBig(x))
	return fzF64(c)
}

func fzSin(x float64) float64 {
	s, _ := fzBigSinCos(fzBig(x))
	return fzF64(s)
}

// fzArg is ph + a·i + b·j with each operation rounded to float64; the
// conversions block FMA fusion, matching the historical amd64 arguments.
func fzArg(ph, a float64, i int, b float64, j int) float64 {
	return float64(float64(ph+float64(a*float64(i))) + float64(b*float64(j)))
}

// fzNorm2 is the Euclidean norm of x rounded once.
func fzNorm2(x []float64) float64 {
	s := fzBig(0)
	for _, v := range x {
		s.Add(s, fzBMul(fzBig(v), fzBig(v)))
	}
	return fzF64(fzBSqrt(s))
}

// fzCAbs is |e| rounded once.
func fzCAbs(e complex128) float64 {
	re, im := fzBig(real(e)), fzBig(imag(e))
	re.Mul(re, re)
	re.Add(re, im.Mul(im, im))
	return fzF64(new(big.Float).SetPrec(fzElemPrec).Sqrt(re))
}

func fzBMul(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(fzElemPrec).Mul(a, b) }

func fzBAdd(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(fzElemPrec).Add(a, b) }

func fzBSub(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(fzElemPrec).Sub(a, b) }

func fzBQuo(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(fzElemPrec).Quo(a, b) }

func fzBSqrt(a *big.Float) *big.Float { return new(big.Float).SetPrec(fzElemPrec).Sqrt(a) }

// fzBigMatMul returns the n×n row-major product a·b.
func fzBigMatMul(a, b []*big.Float, n int) []*big.Float {
	c := make([]*big.Float, n*n)
	for i := range n {
		for j := range n {
			s := fzBig(0)
			for k := range n {
				s.Add(s, fzBMul(a[i*n+k], b[k*n+j]))
			}
			c[i*n+j] = s
		}
	}
	return c
}

// fzBigInverse inverts the n×n row-major a by Gauss–Jordan with partial
// pivoting; ok is false for a zero pivot.
func fzBigInverse(a []*big.Float, n int) (inv []*big.Float, ok bool) {
	w := make([]*big.Float, n*2*n)
	for i := range n {
		for j := range n {
			w[i*2*n+j] = new(big.Float).SetPrec(fzElemPrec).Set(a[i*n+j])
			w[i*2*n+n+j] = fzBig(0)
		}
		w[i*2*n+n+i] = fzBig(1)
	}
	row := func(i int) []*big.Float { return w[i*2*n : (i+1)*2*n] }
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if new(big.Float).Abs(w[i*2*n+k]).Cmp(new(big.Float).Abs(w[piv*2*n+k])) > 0 {
				piv = i
			}
		}
		if w[piv*2*n+k].Sign() == 0 {
			return nil, false
		}
		rk, rp := row(k), row(piv)
		for j := range rk {
			rk[j], rp[j] = rp[j], rk[j]
		}
		d := new(big.Float).Set(rk[k])
		for j := range rk {
			rk[j].Quo(rk[j], d)
		}
		for i := range n {
			if i == k || w[i*2*n+k].Sign() == 0 {
				continue
			}
			ri, f := row(i), new(big.Float).Set(w[i*2*n+k])
			for j := range ri {
				ri[j].Sub(ri[j], fzBMul(f, rk[j]))
			}
		}
	}
	inv = make([]*big.Float, n*n)
	for i := range n {
		copy(inv[i*n:(i+1)*n], w[i*2*n+n:(i+1)*2*n])
	}
	return inv, true
}

// fzBigCond2 returns the 2-norm condition number of the n×n row-major t
// from the eigenvalues of tᵀt (cyclic Jacobi).
func fzBigCond2(t []*big.Float, n int) float64 {
	a := make([]*big.Float, n*n)
	for i := range n {
		for j := range n {
			s := fzBig(0)
			for k := range n {
				s.Add(s, fzBMul(t[k*n+i], t[k*n+j]))
			}
			a[i*n+j] = s
		}
	}
	one := fzBig(1)
	for range 60 {
		off, diag := fzBig(0), fzBig(0)
		for i := range n {
			for j := range n {
				if i == j {
					diag.Add(diag, fzBMul(a[i*n+j], a[i*n+j]))
				} else {
					off.Add(off, fzBMul(a[i*n+j], a[i*n+j]))
				}
			}
		}
		if off.Cmp(new(big.Float).SetMantExp(diag, -2*fzElemPrec+40)) <= 0 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if apq.Sign() == 0 {
					continue
				}
				theta := fzBQuo(fzBSub(a[q*n+q], a[p*n+p]), fzBMul(fzBig(2), apq))
				tn := fzBQuo(one, fzBAdd(new(big.Float).Abs(theta), fzBSqrt(fzBAdd(fzBMul(theta, theta), one))))
				if theta.Sign() < 0 {
					tn.Neg(tn)
				}
				c := fzBQuo(one, fzBSqrt(fzBAdd(fzBMul(tn, tn), one)))
				s := fzBMul(tn, c)
				for k := range n {
					if k == p || k == q {
						continue
					}
					akp, akq := a[k*n+p], a[k*n+q]
					np, nq := fzBSub(fzBMul(c, akp), fzBMul(s, akq)), fzBAdd(fzBMul(s, akp), fzBMul(c, akq))
					a[k*n+p], a[p*n+k], a[k*n+q], a[q*n+k] = np, np, nq, nq
				}
				tp := fzBMul(tn, apq)
				a[p*n+p], a[q*n+q] = fzBSub(a[p*n+p], tp), fzBAdd(a[q*n+q], tp)
				a[p*n+q], a[q*n+p] = fzBig(0), fzBig(0)
			}
		}
	}
	lo, hi := a[0], a[0]
	for i := range n {
		if d := a[i*n+i]; d.Cmp(lo) < 0 {
			lo = d
		} else if d.Cmp(hi) > 0 {
			hi = d
		}
	}
	if lo.Sign() <= 0 {
		return math.Inf(1)
	}
	return fzF64(fzBSqrt(fzBQuo(hi, lo)))
}
