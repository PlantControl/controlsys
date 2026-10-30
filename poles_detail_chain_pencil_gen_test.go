package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// rvbCase is a structured descriptor U·(F, W)·V: an upper triangular finite
// block (poles f_ii/w_ii, optionally a 2×2 complex block w·I) beside
// nilpotent chains. oracle holds the exact poles of (F, W); rank is the
// structural rank of W. Generator of the IP4PC3/ELT76F/L4BCTU rvb families
// (L4BCTU design Appendix A.1, X7O5UZ), PCG(seed, 44044).
type rvbCase struct {
	sys    *System
	oracle []complex128
	rank   int
}

func rvbBuild(t *testing.T, fam string, seed uint64) rvbCase {
	rng := rand.New(rand.NewPCG(seed, 44044))
	var fd, wd []float64
	cpx := false
	var ca, cb, cw float64
	stiffW := func() float64 { return math.Pow(10, -6.5-3.5*rng.Float64()) }
	reg := func() (float64, float64) {
		return rng.NormFloat64() * math.Pow(10, rng.Float64()*2-1), 0.5 + rng.Float64()
	}
	nreg := rng.IntN(3)
	switch fam {
	case "cpx":
		cpx = true
		cw = stiffW()
		ca, cb = rng.NormFloat64(), math.Abs(rng.NormFloat64())+0.1
		if rng.IntN(3) == 0 {
			ca = -math.Abs(ca)
		}
	case "two":
		for range 2 {
			fd = append(fd, rng.NormFloat64())
			wd = append(wd, stiffW())
		}
	case "smallLam":
		w := stiffW()
		fd, wd = append(fd, w*rng.NormFloat64()*math.Pow(10, 3*rng.Float64())), append(wd, w)
		nreg = 1 + rng.IntN(2)
	case "tri":
		fd, wd = append(fd, rng.NormFloat64()), append(wd, stiffW())
		nreg = 1 + rng.IntN(3)
	case "bigA":
		fd, wd = append(fd, rng.NormFloat64()), append(wd, stiffW())
		fd, wd = append(fd, math.Copysign(math.Pow(10, 2+4*rng.Float64()), rng.NormFloat64())), append(wd, 1)
	case "unstS":
		fd, wd = append(fd, math.Abs(rng.NormFloat64())), append(wd, stiffW())
		for range 1 + rng.IntN(2) {
			fd, wd = append(fd, -math.Abs(rng.NormFloat64())-0.05), append(wd, 0.5+rng.Float64())
		}
		nreg = 0
	case "edge":
		// stiff singular value near the stiff-band boundaries
		fd, wd = append(fd, rng.NormFloat64()), append(wd, math.Pow(10, -5.5-1.5*rng.Float64()-5*float64(rng.IntN(2))*rng.Float64()))
	default:
		t.Fatal(fam)
	}
	for range nreg {
		f, w := reg()
		fd, wd = append(fd, f), append(wd, w)
	}
	nf := len(fd)
	if cpx {
		nf += 2
	}
	chains := [][]int{{2}, {3}, {4}, {5}, {2, 2}, {2, 3}, {1, 2}, {3, 1}, {6}}[rng.IntN(9)]
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	var oracle []complex128
	o := 0
	if cpx {
		F.Set(0, 0, ca)
		F.Set(0, 1, cb)
		F.Set(1, 0, -cb)
		F.Set(1, 1, ca)
		W.Set(0, 0, cw)
		W.Set(1, 1, cw)
		oracle = append(oracle, complex(ca/cw, cb/cw), complex(ca/cw, -cb/cw))
		o = 2
	}
	tri := fam == "tri" || rng.IntN(2) == 0
	for i := range fd {
		F.Set(o+i, o+i, fd[i])
		W.Set(o+i, o+i, wd[i])
		oracle = append(oracle, complex(fd[i]/wd[i], 0))
		if tri {
			for j := o + i + 1; j < nf; j++ {
				F.Set(o+i, j, rng.NormFloat64())
				W.Set(o+i, j, 0.3*rng.NormFloat64())
			}
		}
	}
	o = nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
		}
		if rng.IntN(2) == 0 {
			for j := o; j < o+k; j++ {
				for i := range nf {
					F.Set(i, j, 0.5*rng.NormFloat64())
				}
			}
		}
		o += k
	}
	logCond := 2 * rng.Float64()
	mk := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		var qr mat.QR
		qr.Factorize(g)
		var q mat.Dense
		qr.QTo(&q)
		for j := range n {
			s := math.Pow(10, -logCond*float64(j)/float64(max(1, n-1)))
			for i := range n {
				q.Set(i, j, q.At(i, j)*s)
			}
		}
		g2 := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g2.Set(i, j, rng.NormFloat64())
			}
		}
		qr.Factorize(g2)
		var q2, out mat.Dense
		qr.QTo(&q2)
		out.Mul(&q, &q2)
		return &out
	}
	U, V := mk(), mk()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	b := mat.NewDense(n, 1, nil)
	c := mat.NewDense(1, n, nil)
	for i := range n {
		b.Set(i, 0, rng.NormFloat64())
		c.Set(0, i, rng.NormFloat64())
	}
	B.Mul(U, b)
	C.Mul(c, V)
	sys, err := NewDescriptor(&A, &B, &C, mat.NewDense(1, 1, []float64{0.1}), &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return rvbCase{sys, oracle, n - len(chains)}
}
