package controlsys

import (
	"math"
	"math/big"

	"plantcontrol.org/v1/gonum/mat"
)

// closedLoopStateMatrix returns the state matrix, states [x; xk], of the
// plant of gp closed with u = K·y, K = (Ak, Bk, Ck, Dk):
//
//	[A + B2·L·Dk·C2,   B2·L·Ck;   Bk·(C2 + D22·L·Dk·C2),   Ak + Bk·D22·L·Ck],
//
// L = (I − Dk·D22)⁻¹, exactly in rationals from the float64 data of gp and
// K. ok is false when I − Dk·D22 is exactly singular.
func closedLoopStateMatrix(gp *generalizedPlantPartition, K *System) ([][]*big.Rat, bool) {
	n := gp.n
	nk, _, _ := K.Dims()
	A, B2, C2, D22 := ratMat(gp.A), ratMat(gp.B2), ratMat(gp.C2), ratMat(gp.D22)
	Dk := ratMat(K.D)
	var Ak, Bk, Ck [][]*big.Rat
	if nk > 0 {
		Ak, Bk, Ck = ratMat(K.A), ratMat(K.B), ratMat(K.C)
	}
	Ux, Uk := ratMul(Dk, C2), Ck
	if !ratZero(Dk) {
		W := ratMul(Dk, D22)
		for i, row := range W {
			for j, w := range row {
				w.Neg(w)
				if i == j {
					w.Add(w, big.NewRat(1, 1))
				}
			}
		}
		L, ok := ratInverse(W)
		if !ok {
			return nil, false
		}
		Ux = ratMul(L, Ux)
		if nk > 0 {
			Uk = ratMul(L, Ck)
		}
	}
	N := n + nk
	out := make([][]*big.Rat, N)
	for i := range out {
		out[i] = make([]*big.Rat, N)
	}
	put := func(r0, c0 int, M, add [][]*big.Rat) {
		for i, row := range M {
			for j, v := range row {
				if add != nil {
					v.Add(v, add[i][j])
				}
				out[r0+i][c0+j] = v
			}
		}
	}
	put(0, 0, ratMul(B2, Ux), A)
	if nk > 0 {
		put(0, n, ratMul(B2, Uk), nil)
		Yx := ratMul(D22, Ux)
		for i, row := range Yx {
			for j, v := range row {
				v.Add(v, C2[i][j])
			}
		}
		put(n, 0, ratMul(Bk, Yx), nil)
		put(n, n, ratMul(Bk, ratMul(D22, Uk)), Ak)
	}
	return out, true
}

// roundRat rounds each entry of M to the nearest float64, which is within
// u·|entry| (u = eps/2) plus the subnormal spacing of it.
func roundRat(M [][]*big.Rat) *mat.Dense {
	out := mat.NewDense(len(M), len(M), nil)
	for i, row := range M {
		for j, v := range row {
			f, _ := v.Float64()
			out.Set(i, j, f)
		}
	}
	return out
}

// exactStable reports whether the rational matrix M is Hurwitz (Schur when
// discrete), decided exactly on the integer characteristic polynomial of
// δ·M, δ the least common denominator of its entries (intHurwitz,
// intSchur).
func exactStable(M [][]*big.Rat, discrete bool) bool {
	delta := big.NewInt(1)
	var g big.Int
	for _, row := range M {
		for _, v := range row {
			d := v.Denom()
			g.GCD(nil, nil, delta, d)
			delta.Mul(delta, new(big.Int).Quo(d, &g))
		}
	}
	m := make([][]*big.Int, len(M))
	for i, row := range M {
		m[i] = make([]*big.Int, len(row))
		for j, v := range row {
			t := new(big.Int).Mul(v.Num(), delta)
			m[i][j] = t.Quo(t, v.Denom())
		}
	}
	p := intCharPoly(m)
	if !discrete {
		return intHurwitz(p)
	}
	intScaleRoots(p, delta, big.NewInt(1))
	return intSchur(p)
}

func ratMat(m *mat.Dense) [][]*big.Rat {
	r, c := m.Dims()
	out := make([][]*big.Rat, r)
	raw := m.RawMatrix()
	for i := range r {
		out[i] = make([]*big.Rat, c)
		for j := range c {
			out[i][j] = new(big.Rat).SetFloat64(raw.Data[i*raw.Stride+j])
		}
	}
	return out
}

func ratZero(m [][]*big.Rat) bool {
	for _, row := range m {
		for _, v := range row {
			if v.Sign() != 0 {
				return false
			}
		}
	}
	return true
}

func ratMul(a, b [][]*big.Rat) [][]*big.Rat {
	out := make([][]*big.Rat, len(a))
	var t big.Rat
	for i, row := range a {
		out[i] = make([]*big.Rat, len(b[0]))
		for j := range out[i] {
			s := new(big.Rat)
			for k, v := range row {
				if v.Sign() != 0 && b[k][j].Sign() != 0 {
					s.Add(s, t.Mul(v, b[k][j]))
				}
			}
			out[i][j] = s
		}
	}
	return out
}

// ratInverse is exact Gauss–Jordan elimination; ok is false when M is
// singular.
func ratInverse(M [][]*big.Rat) ([][]*big.Rat, bool) {
	n := len(M)
	a := make([][]*big.Rat, n)
	inv := make([][]*big.Rat, n)
	for i := range n {
		a[i] = make([]*big.Rat, n)
		inv[i] = make([]*big.Rat, n)
		for j := range n {
			a[i][j] = new(big.Rat).Set(M[i][j])
			inv[i][j] = new(big.Rat)
		}
		inv[i][i].SetInt64(1)
	}
	var t big.Rat
	for c := range n {
		p := c
		for p < n && a[p][c].Sign() == 0 {
			p++
		}
		if p == n {
			return nil, false
		}
		a[c], a[p] = a[p], a[c]
		inv[c], inv[p] = inv[p], inv[c]
		piv := new(big.Rat).Inv(a[c][c])
		for j := range n {
			a[c][j].Mul(a[c][j], piv)
			inv[c][j].Mul(inv[c][j], piv)
		}
		for r := range n {
			if r == c || a[r][c].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(a[r][c])
			for j := range n {
				a[r][j].Sub(a[r][j], t.Mul(f, a[c][j]))
				inv[r][j].Sub(inv[r][j], t.Mul(f, inv[c][j]))
			}
		}
	}
	return inv, true
}

// lyapunovStable reports whether every matrix A = Â + E with
// |E| ≤ u·|Â| + η elementwise (u = eps/2, η the smallest subnormal), that
// is every exact matrix Â rounds from, is stable: Hurwitz, or Schur when
// discrete. It solves ÂᵀP + PÂ = −I (ÂᵀPÂ − P = −I) in float64 and
// certifies, with every rounding of the check bounded by γ_k = k·u/(1−k·u)
// (Higham, Accuracy and Stability of Numerical Algorithms, §3.1), that P
// is positive definite and AᵀP + PA (AᵀPA − P) negative definite for every
// such A (Lyapunov). P ≻ 0 is Rump's test: a float64 Cholesky
// factorization of P − c·I completing with c = γ_{N+3}·tr(P) proves it.
// A false result says only that the certificate fails, not that A is
// unstable.
func lyapunovStable(Ahat *mat.Dense, discrete bool) bool {
	N, _ := Ahat.Dims()
	At := mat.DenseCopyOf(Ahat.T())
	var P *mat.Dense
	var err error
	if discrete {
		P, err = DLyap(At, eyeDense(N), nil)
	} else {
		P, err = Lyap(At, eyeDense(N), nil)
	}
	if err != nil {
		return false
	}
	pr := P.RawMatrix()
	for i := range N {
		for j := i + 1; j < N; j++ {
			pr.Data[j*pr.Stride+i] = pr.Data[i*pr.Stride+j]
		}
	}
	u := eps() / 2
	gam := func(k int) float64 { return float64(k) * u / (1 - float64(k)*u) }
	up := func(x float64, k int) float64 { return x * (1 + gam(k)) }

	tr := 0.0
	for i := range N {
		tr += pr.Data[i*pr.Stride+i]
	}
	if !(tr > 0) || math.IsInf(tr, 0) {
		return false
	}
	c := up(gam(N+3)*up(tr, N), 1)
	S := mat.NewSymDense(N, nil)
	for i := range N {
		for j := i; j < N; j++ {
			v := pr.Data[i*pr.Stride+j]
			if i == j {
				v -= c
			}
			S.SetSym(i, j, v)
		}
	}
	var ch mat.Cholesky
	if !ch.Factorize(S) {
		return false
	}

	nA, nP := up(denseNormScaled(Ahat.RawMatrix()), N*N+2), up(denseNormScaled(P.RawMatrix()), N*N+2)
	e := up(gam(1)*nA+float64(N)*math.SmallestNonzeroFloat64, 2)
	sqrtN := math.Sqrt(float64(N))
	R := mat.NewDense(N, N, nil)
	var bound, perturb float64
	if discrete {
		var PA mat.Dense
		PA.Mul(P, Ahat)
		R.Mul(At, &PA)
		R.Sub(R, P)
		bound = gam(2*N+2) * up(2*nA*nA*nP+nP+sqrtN, 4)
		perturb = up(nP*e*(2*nA+e), 4)
	} else {
		var PA mat.Dense
		PA.Mul(P, Ahat)
		R.Add(&PA, PA.T())
		bound = gam(2*N+2) * up(2*nA*nP+sqrtN, 3)
		perturb = up(2*e*nP, 2)
	}
	for i := range N {
		R.Set(i, i, R.At(i, i)+1)
	}
	res := up(denseNormScaled(R.RawMatrix()), N*N+2) + bound
	return up(res+perturb, 2) < 1
}
