package controlsys

import (
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffSplitDC is G(0) = D − C·A⁻¹·B of the SISO pencil sys at 320 bits
// and the descriptor bar u(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖) at s = 0,
// M = A⁻¹, u = 4(n+5)ε, Frobenius norms of the stored data.
func stiffSplitDC(t *testing.T, sys *System) (g0, bar float64) {
	t.Helper()
	n, _, _ := sys.Dims()
	rhs := newHbMat(n, n+1)
	for i := range n {
		rhs.at(i, i).SetInt64(1)
		rhs.at(i, n).SetFloat64(sys.B.At(i, 0))
	}
	x := hbSolve(hbFrom(sys.A), rhs)
	mf := 0.0
	for i := range n {
		for j := range n {
			v, _ := x.at(i, j).Float64()
			mf += v * v
		}
	}
	mf = math.Sqrt(mf)
	g := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.D.At(0, 0))
	p := new(big.Float).SetPrec(hbPrec)
	for i := range n {
		g.Sub(g, p.Mul(new(big.Float).SetPrec(hbPrec).SetFloat64(sys.C.At(0, i)), x.at(i, n)))
	}
	g0, _ = g.Float64()
	u := 4 * float64(n+5) * eps()
	fc, fb, fa := mat.Norm(sys.C, 2), mat.Norm(sys.B, 2), mat.Norm(sys.A, 2)
	bar = u * (fc*mf*fa*mf*fb + fc*mf*fb + math.Abs(sys.D.At(0, 0)))
	return g0, bar
}

// stiffSplitDiscreteDC is Gd(1) = Cd(I − Ad)⁻¹Bd + Dd of the explicit
// discrete SISO d at 320 bits.
func stiffSplitDiscreteDC(t *testing.T, d *System) float64 {
	t.Helper()
	n, _, _ := d.Dims()
	m := newHbMat(n, n)
	for i := range n {
		for j := range n {
			v := -d.A.At(i, j)
			if i == j {
				v++
			}
			m.at(i, j).SetFloat64(v)
		}
	}
	x := hbSolve(m, hbFrom(d.B))
	g := new(big.Float).SetPrec(hbPrec).SetFloat64(d.D.At(0, 0))
	p := new(big.Float).SetPrec(hbPrec)
	for i := range n {
		g.Add(g, p.Mul(new(big.Float).SetPrec(hbPrec).SetFloat64(d.C.At(0, i)), x.at(i, 0)))
	}
	v, _ := g.Float64()
	return v
}

// stiffSplitExpm is e^X at 320 bits by scaling and squaring with a 70-term
// Taylor series.
func stiffSplitExpm(x hbMat) hbMat {
	nrm := 0.0
	for j := range x.c {
		s := 0.0
		for i := range x.r {
			v, _ := x.at(i, j).Float64()
			s += math.Abs(v)
		}
		nrm = math.Max(nrm, s)
	}
	sq := 0
	if nrm > 0.5 {
		sq = int(math.Ceil(math.Log2(nrm / 0.5)))
	}
	xs := newHbMat(x.r, x.c)
	for i := range x.d {
		xs.d[i].SetMantExp(x.d[i], -sq)
	}
	res, term := newHbMat(x.r, x.c), newHbMat(x.r, x.c)
	for i := range x.r {
		res.at(i, i).SetInt64(1)
		term.at(i, i).SetInt64(1)
	}
	for k := 1; k <= 70; k++ {
		term = hbMul(term, xs)
		f := new(big.Float).SetPrec(hbPrec).SetFloat64(1 / float64(k))
		for i := range term.d {
			term.d[i].Mul(term.d[i], f)
		}
		res = hbAdd(res, term, 1)
	}
	for range sq {
		res = hbMul(res, res)
	}
	return res
}

// stiffSplitResponse is the exact response at t of the SISO pencil sys at
// 320 bits: the step response C∫₀ᵗe^{Fτ}dτ·G + D (kind "step"), the
// impulse response C·e^{Ft}·G ("impulse"), or the free response C·e^{Ft}·x0
// ("initial"), F = E⁻¹A, G = E⁻¹B.
func stiffSplitResponse(t *testing.T, sys *System, kind string, x0 []float64, at float64) float64 {
	t.Helper()
	n, _, _ := sys.Dims()
	rhs := newHbMat(n, n+1)
	for i := range n {
		for j := range n {
			rhs.at(i, j).SetFloat64(sys.A.At(i, j))
		}
		rhs.at(i, n).SetFloat64(sys.B.At(i, 0))
	}
	fg := hbSolve(hbFrom(sys.E), rhs)
	aug := newHbMat(n+1, n+1)
	tt := new(big.Float).SetPrec(hbPrec).SetFloat64(at)
	for i := range n {
		for j := range n + 1 {
			aug.at(i, j).Mul(fg.at(i, j), tt)
		}
	}
	x := stiffSplitExpm(aug)
	y := new(big.Float).SetPrec(hbPrec)
	p := new(big.Float).SetPrec(hbPrec)
	for i := range n {
		var v *big.Float
		switch kind {
		case "step":
			v = x.at(i, n)
		case "impulse":
			v = new(big.Float).SetPrec(hbPrec)
			for j := range n {
				v.Add(v, p.Mul(x.at(i, j), fg.at(j, n)))
			}
		default:
			v = new(big.Float).SetPrec(hbPrec)
			for j := range n {
				v.Add(v, p.Mul(x.at(i, j), new(big.Float).SetPrec(hbPrec).SetFloat64(x0[j])))
			}
		}
		y.Add(y, p.Mul(new(big.Float).SetPrec(hbPrec).SetFloat64(sys.C.At(0, i)), v))
	}
	out, _ := y.Float64()
	if kind == "step" {
		out += sys.D.At(0, 0)
	}
	return out
}

// stiffSplitWithinRel reports whether the samples Y[0, k] of resp are within
// rel·max|y| of the exact response at a few sample times.
func stiffSplitWithinRel(t *testing.T, sys *System, kind string, x0 []float64, resp *TimeResponse, rel float64) (worst float64, ok bool) {
	t.Helper()
	nt := len(resp.T)
	ymax := 0.0
	var errs []float64
	for _, k := range []int{1, nt / 8, nt / 3, nt - 1} {
		ref := stiffSplitResponse(t, sys, kind, x0, resp.T[k])
		ymax = math.Max(ymax, math.Abs(ref))
		errs = append(errs, math.Abs(resp.Y.At(0, k)-ref))
	}
	for _, e := range errs {
		worst = math.Max(worst, e/ymax)
	}
	return worst, worst <= rel
}
