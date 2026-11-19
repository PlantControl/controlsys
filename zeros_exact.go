package controlsys

import (
	"math"
	"math/big"
)

// bigC is a complex number of big.Float parts with an inclusion radius.
type bigC struct {
	re, im *big.Float
	rad    float64
}

func (z bigC) complex() complex128 {
	re, _ := z.re.Float64()
	im, _ := z.im.Float64()
	return complex(re, im)
}

// aberthRoots returns the roots of the ascending rational polynomial p of
// degree ≥ 1 by Aberth–Ehrlich iteration in big.Float, from 256 bits
// doubling up to 512 (300 sweeps each) until every correction is below 2^(40−prec) of its
// root.
func aberthRoots(p []*big.Rat) ([]bigC, bool) {
	for prec := uint(256); prec <= 512; prec *= 2 {
		if z, ok := aberth(p, prec); ok {
			return z, true
		}
	}
	return nil, false
}

func aberth(p []*big.Rat, prec uint) ([]bigC, bool) {
	d := len(p) - 1
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	a := make([]*big.Float, d+1)
	la := make([]float64, d+1)
	for i, c := range p {
		a[i] = nf().SetRat(c)
		la[i] = math.Inf(-1)
		if a[i].Sign() != 0 {
			la[i] = bigLog2(a[i])
		}
	}
	var hull []int
	for i := 0; i <= d; i++ {
		if math.IsInf(la[i], -1) {
			continue
		}
		for len(hull) >= 2 {
			u, v := hull[len(hull)-2], hull[len(hull)-1]
			if (la[v]-la[u])*float64(i-u) <= (la[i]-la[u])*float64(v-u) {
				hull = hull[:len(hull)-1]
			} else {
				break
			}
		}
		hull = append(hull, i)
	}
	zr, zi := make([]*big.Float, 0, d), make([]*big.Float, 0, d)
	for h := 0; h+1 < len(hull); h++ {
		u, v := hull[h], hull[h+1]
		lr := (la[u] - la[v]) / float64(v-u)
		for t := range v - u {
			th := 2*math.Pi*float64(t)/float64(v-u) + 2*math.Pi*float64(u)/float64(d) + 0.4
			e := math.Floor(lr)
			mag := nf().SetMantExp(nf().SetFloat64(math.Exp2(lr-e)), int(e))
			zr = append(zr, nf().Mul(mag, nf().SetFloat64(math.Cos(th))))
			zi = append(zi, nf().Mul(mag, nf().SetFloat64(math.Sin(th))))
		}
	}
	if len(zr) != d {
		return nil, false
	}
	done := make([]bool, d)
	lim := 40 - float64(prec)
	pr, pi, dr, di := nf(), nf(), nf(), nf()
	for range 300 {
		all := true
		for i := range d {
			if done[i] {
				continue
			}
			zeroPolyHorner(a, zr[i], zi[i], pr, pi, dr, di, prec)
			if pr.Sign() == 0 && pi.Sign() == 0 {
				done[i] = true
				continue
			}
			wr, wi := bigCDiv(pr, pi, dr, di, prec)
			sr, si := nf(), nf()
			for j := range d {
				if j == i {
					continue
				}
				xr, xi := nf().Sub(zr[i], zr[j]), nf().Sub(zi[i], zi[j])
				ur, ui := bigCDiv(nf().SetInt64(1), nf(), xr, xi, prec)
				sr.Add(sr, ur)
				si.Add(si, ui)
			}
			mr := nf().Sub(nf().Mul(wr, sr), nf().Mul(wi, si))
			mi := nf().Add(nf().Mul(wr, si), nf().Mul(wi, sr))
			stR, stI := bigCDiv(wr, wi, nf().Sub(nf().SetInt64(1), mr), nf().Neg(mi), prec)
			zr[i].Sub(zr[i], stR)
			zi[i].Sub(zi[i], stI)
			if bigLog2Abs(stR, stI)-bigLog2Abs(zr[i], zi[i]) < lim {
				done[i] = true
			} else {
				all = false
			}
		}
		if all {
			out := make([]bigC, d)
			for i := range out {
				out[i] = bigC{re: zr[i], im: zi[i]}
			}
			return out, true
		}
	}
	return nil, false
}

func bigLog2(x *big.Float) float64 {
	m := new(big.Float)
	e := x.MantExp(m)
	f, _ := m.Float64()
	return float64(e) + math.Log2(math.Abs(f))
}

func bigLog2Abs(re, im *big.Float) float64 {
	switch {
	case re.Sign() == 0 && im.Sign() == 0:
		return math.Inf(-1)
	case re.Sign() == 0:
		return bigLog2(im)
	case im.Sign() == 0:
		return bigLog2(re)
	}
	a, b := bigLog2(re), bigLog2(im)
	hi := math.Max(a, b)
	return hi + 0.5*math.Log2(math.Exp2(2*(a-hi))+math.Exp2(2*(b-hi)))
}

// zeroPolyHorner sets p and dp to the values at z of the polynomial with
// ascending coefficients a and of its derivative.
func zeroPolyHorner(a []*big.Float, zr, zi, pr, pi, dr, di *big.Float, prec uint) {
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	d := len(a) - 1
	pr.SetPrec(prec).Set(a[d])
	pi.SetPrec(prec).SetInt64(0)
	dr.SetPrec(prec).SetInt64(0)
	di.SetPrec(prec).SetInt64(0)
	t1, t2 := nf(), nf()
	for k := d - 1; k >= 0; k-- {
		nr := nf().Sub(t1.Mul(dr, zr), t2.Mul(di, zi))
		ni := nf().Add(t1.Mul(dr, zi), t2.Mul(di, zr))
		dr.Add(nr, pr)
		di.Add(ni, pi)
		qr := nf().Sub(t1.Mul(pr, zr), t2.Mul(pi, zi))
		qi := nf().Add(t1.Mul(pr, zi), t2.Mul(pi, zr))
		pr.Add(qr, a[k])
		pi.Set(qi)
	}
}

func bigCDiv(ar, ai, br, bi *big.Float, prec uint) (*big.Float, *big.Float) {
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	d := nf().Add(nf().Mul(br, br), nf().Mul(bi, bi))
	x := nf().Add(nf().Mul(ar, br), nf().Mul(ai, bi))
	y := nf().Sub(nf().Mul(ai, br), nf().Mul(ar, bi))
	return x.Quo(x, d), y.Quo(y, d)
}
