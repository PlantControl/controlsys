package controlsys

import (
	"math/big"

	"plantcontrol.org/v1/gonum/mat"
)

// hasExactZeroLine reports whether some row or column of E is exactly zero.
func hasExactZeroLine(E *mat.Dense) bool {
	raw := E.RawMatrix()
	colZero := make([]bool, raw.Cols)
	for j := range colZero {
		colZero[j] = true
	}
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		rowZero := true
		for j, v := range row {
			if v != 0 {
				rowZero = false
				colZero[j] = false
			}
		}
		if rowZero {
			return true
		}
	}
	for _, z := range colZero {
		if z {
			return true
		}
	}
	return false
}

// exactProperness decides properness of the descriptor sys (no internal
// delays) in exact rational arithmetic on its float64 data: G is proper iff
// every channel's num_ij(s) = det([sE−A, −B_j; C_i, D_ij]), which is
// det(sE−A)·G_ij(s), has degree at most that of den(s) = det(sE−A), i.e.
// the infinite structure carries no s^k, k ≥ 1, through B and C. Both have
// degree ≤ n, so their degrees are the highest nonzero forward difference of
// their values at s = 0..n. ok is false when den is identically zero.
func exactProperness(sys *System) (proper, ok bool) {
	n, m, p := sys.Dims()
	rat := func(m *mat.Dense) [][]*big.Rat {
		r, c := m.Dims()
		raw := m.RawMatrix()
		out := make([][]*big.Rat, r)
		for i := range r {
			out[i] = make([]*big.Rat, c)
			for j := range c {
				out[i][j] = new(big.Rat).SetFloat64(raw.Data[i*raw.Stride+j])
			}
		}
		return out
	}
	A, E, B, C, D := rat(sys.A), rat(sys.E), rat(sys.B), rat(sys.C), rat(sys.D)
	pencil := func(s int64, extra int) [][]*big.Rat {
		M := make([][]*big.Rat, n+extra)
		sr := new(big.Rat).SetInt64(s)
		for i := range n {
			M[i] = make([]*big.Rat, n+extra)
			for j := range n {
				v := new(big.Rat).Mul(sr, E[i][j])
				M[i][j] = v.Sub(v, A[i][j])
			}
		}
		return M
	}
	den := make([]*big.Rat, n+1)
	num := make([][][]*big.Rat, p)
	for i := range p {
		num[i] = make([][]*big.Rat, m)
		for j := range m {
			num[i][j] = make([]*big.Rat, n+1)
		}
	}
	for k := range n + 1 {
		den[k] = ratDet(pencil(int64(k), 0))
		for i := range p {
			for j := range m {
				M := pencil(int64(k), 1)
				for r := range n {
					M[r][n] = new(big.Rat).Neg(B[r][j])
				}
				M[n] = make([]*big.Rat, n+1)
				for c := range n {
					M[n][c] = new(big.Rat).Set(C[i][c])
				}
				M[n][n] = new(big.Rat).Set(D[i][j])
				num[i][j][k] = ratDet(M)
			}
		}
	}
	dd := ratDegree(den)
	if dd < 0 {
		return false, false
	}
	for i := range p {
		for j := range m {
			if ratDegree(num[i][j]) > dd {
				return false, true
			}
		}
	}
	return true, true
}

// ratDet returns det(M) by Gaussian elimination with the first nonzero
// pivot; M is overwritten.
func ratDet(M [][]*big.Rat) *big.Rat {
	n := len(M)
	det := big.NewRat(1, 1)
	t := new(big.Rat)
	for k := range n {
		piv := -1
		for r := k; r < n; r++ {
			if M[r][k].Sign() != 0 {
				piv = r
				break
			}
		}
		if piv < 0 {
			return new(big.Rat)
		}
		if piv != k {
			M[k], M[piv] = M[piv], M[k]
			det.Neg(det)
		}
		det.Mul(det, M[k][k])
		for r := k + 1; r < n; r++ {
			if M[r][k].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Quo(M[r][k], M[k][k])
			for c := k + 1; c < n; c++ {
				M[r][c].Sub(M[r][c], t.Mul(f, M[k][c]))
			}
		}
	}
	return det
}

// ratDegree returns the degree of the polynomial of degree ≤ len(v)−1 whose
// values at 0, 1, … are v (−1 for the zero polynomial): its highest nonzero
// forward difference at 0.
func ratDegree(v []*big.Rat) int {
	d := make([]*big.Rat, len(v))
	for i, x := range v {
		d[i] = new(big.Rat).Set(x)
	}
	deg := -1
	for k := range len(d) {
		if d[0].Sign() != 0 {
			deg = k
		}
		for i := 0; i+1 < len(d)-k; i++ {
			d[i].Sub(d[i+1], d[i])
		}
	}
	return deg
}
