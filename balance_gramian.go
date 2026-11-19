package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

const (
	gramianRefineSteps = 6
	// gramianRefineDone ends refinement: the correction is at rounding level.
	gramianRefineDone = 0x1p-46
	// gramianRefineAccept is the largest last relative correction accepted
	// when refinement stops early; above it the gramian is not certified.
	gramianRefineAccept = 1e-10
	// gramianDDSteps, gramianDDDone and gramianDDAccept are the same for
	// the double-double continuation.
	gramianDDSteps  = 10
	gramianDDDone   = 0x1p-100
	gramianDDAccept = 0x1p-56
)

// refinedGramian is the solution X of A·X + X·Aᵀ + F·Fᵀ = 0 (continuous) or
// A·X·Aᵀ − X + F·Fᵀ = 0 (discrete), F n×k of row stride k, refined with
// residuals evaluated in double-double from the stored A and F, all
// corrections reusing one Schur form of A.
type refinedGramian struct {
	a        *mat.Dense
	f        []float64
	n, k     int
	discrete bool
	schur    schurLyapunov
	x        []float64
	r, d     []float64
	ax       []dd
}

// newRefinedGramian solves and refines X in float64. With transposeOf set,
// A is the transpose of transposeOf's matrix and its Schur form is derived
// from transposeOf's instead of computed. ok is false when the corrections
// do not fall below gramianRefineAccept relative to X: the equation is then
// too ill-conditioned for X to be accurate in float64.
func newRefinedGramian(A *mat.Dense, f []float64, k int, discrete bool, transposeOf *schurLyapunov) (g *refinedGramian, ok bool, err error) {
	n, _ := A.Dims()
	nn := n * n
	slab := make([]float64, 6*nn+2*n)
	take := func(k int) []float64 { t := slab[:k:k]; slab = slab[k:]; return t }
	var s schurLyapunov
	if transposeOf != nil {
		s = transposeOf.transposed(take(nn), take(nn), take(nn))
	} else {
		ws := &LyapunovWorkspace{aData: take(nn), wr: take(n), wi: take(n), vs: take(nn), tmp: take(nn)}
		if s, err = newSchurLyapunov(A, n, discrete, ws); err != nil {
			return nil, false, err
		}
	}
	g = &refinedGramian{a: A, f: f, n: n, k: k, discrete: discrete, schur: s,
		x: take(nn), r: take(nn), d: take(nn)}
	if discrete {
		g.ax = make([]dd, n*n)
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: k, Stride: k, Data: f},
		blas64.General{Rows: n, Cols: k, Stride: k, Data: f},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: g.r})
	symmetrize(g.r, n, n)
	if err := s.solve(g.r, n, g.x); err != nil {
		return nil, false, err
	}
	corr := math.Inf(1)
	for range gramianRefineSteps {
		next, err := g.correction(nil)
		if err != nil {
			return nil, false, err
		}
		for i, v := range g.d {
			g.x[i] += v
		}
		if next <= gramianRefineDone {
			return g, true, nil
		}
		if !(next < corr/2) {
			corr = next
			break
		}
		corr = next
	}
	return g, corr <= gramianRefineAccept, nil
}

// transposed returns the Schur form of Aᵀ from A = U·T·Uᵀ: with J the
// reversal permutation, Aᵀ = (U·J)·(J·Tᵀ·J)·(U·J)ᵀ, and J·Tᵀ·J is upper
// quasi-triangular in standard form. t, u and tmp are n×n storage.
func (s *schurLyapunov) transposed(t, u, tmp []float64) schurLyapunov {
	n := s.n
	for i := range n {
		for j := range n {
			t[i*n+j] = s.t[(n-1-j)*n+(n-1-i)]
			u[i*n+j] = s.u[i*n+(n-1-j)]
		}
	}
	return schurLyapunov{n: n, discrete: s.discrete, t: t, u: u, tmp: tmp}
}

// correction solves for the correction d of X (plus lo when not nil) and
// returns ‖d‖/‖X‖.
func (g *refinedGramian) correction(lo []float64) (float64, error) {
	if lo == nil {
		lyapResidualDot2(g.a, g.x, g.f, g.k, g.discrete, g.ax, g.r)
	} else {
		lyapResidualDD(g.a, g.x, lo, g.f, g.k, g.discrete, g.ax, g.r)
	}
	if err := g.schur.solve(g.r, g.n, g.d); err != nil {
		return 0, err
	}
	dn := scaledNorm2(g.d)
	if dn == 0 {
		return 0, nil
	}
	return dn / scaledNorm2(g.x), nil
}

// ddCholesky continues the refinement with X held in double-double and
// returns its lower Cholesky factor in double-double. ok is false when the
// refinement stalls before X is accurate beyond float64. A pivot below
// 2⁻²¹²·max diag(X), a direction X determines only to double-double
// rounding, gets that floor and a zero column.
func (g *refinedGramian) ddCholesky() (l []dd, ok bool, err error) {
	n := g.n
	hi := append([]float64(nil), g.x...)
	lo := make([]float64, n*n)
	saved := g.x
	g.x = hi
	defer func() { g.x = saved }()
	corr := math.Inf(1)
	for range gramianDDSteps {
		next, err := g.correction(lo)
		if err != nil {
			return nil, false, err
		}
		for i, v := range g.d {
			s := dd{hi[i], lo[i]}.add(dd{v, 0})
			hi[i], lo[i] = s.hi, s.lo
		}
		if next <= gramianDDDone {
			corr = next
			break
		}
		if !(next < corr/2) {
			corr = next
			break
		}
		corr = next
	}
	if corr > gramianDDAccept {
		return nil, false, nil
	}
	top := 0.0
	for j := range n {
		top = math.Max(top, hi[j*n+j])
	}
	floor := math.Ldexp(top, -212)
	l = make([]dd, n*n)
	for j := range n {
		s := dd{hi[j*n+j], lo[j*n+j]}
		for k := range j {
			s = s.add(l[j*n+k].mul(l[j*n+k]).neg())
		}
		if !(s.hi > floor) {
			l[j*n+j] = dd{math.Sqrt(floor), 0}
			continue
		}
		r := math.Sqrt(s.hi)
		root := quickTwoSum(r, s.add(ddProd(r, r).neg()).hi/(2*r))
		l[j*n+j] = root
		for i := j + 1; i < n; i++ {
			t := dd{hi[i*n+j], lo[i*n+j]}
			for k := range j {
				t = t.add(l[i*n+k].mul(l[j*n+k]).neg())
			}
			l[i*n+j] = t.div(root)
		}
	}
	return l, true, nil
}

// lyapResidualDD writes into r the Lyapunov residual A·X + X·Aᵀ + F·Fᵀ
// (continuous) or A·X·Aᵀ − X + F·Fᵀ (discrete) of X = x + lo (lo nil for
// zero), accumulated in double-double and rounded once; ax is n×n scratch
// for the discrete case.
func lyapResidualDD(A *mat.Dense, x, lo, f []float64, k int, discrete bool, ax []dd, r []float64) {
	n, _ := A.Dims()
	aRaw := A.RawMatrix()
	a, lda := aRaw.Data, aRaw.Stride
	xAt := func(i, j int) dd {
		if lo == nil {
			return dd{x[i*n+j], 0}
		}
		return dd{x[i*n+j], lo[i*n+j]}
	}
	if discrete {
		for i := range n {
			for j := range n {
				var s dd
				for l := range n {
					s = s.add(ddProd(a[i*lda+l], x[l*n+j]))
					if lo != nil {
						s = s.add(ddProd(a[i*lda+l], lo[l*n+j]))
					}
				}
				ax[i*n+j] = s
			}
		}
	}
	for i := range n {
		for j := i; j < n; j++ {
			var s dd
			if discrete {
				for l := range n {
					s = s.add(ax[i*n+l].mul(dd{a[j*lda+l], 0}))
				}
				s = s.add(xAt(i, j).neg())
			} else {
				for l := range n {
					s = s.add(ddProd(a[i*lda+l], x[l*n+j]))
					s = s.add(ddProd(x[i*n+l], a[j*lda+l]))
					if lo != nil {
						s = s.add(ddProd(a[i*lda+l], lo[l*n+j]))
						s = s.add(ddProd(lo[i*n+l], a[j*lda+l]))
					}
				}
			}
			for l := range k {
				s = s.add(ddProd(f[i*k+l], f[j*k+l]))
			}
			v := s.hi + s.lo
			r[i*n+j] = v
			r[j*n+i] = v
		}
	}
}

// lyapResidualDot2 is lyapResidualDD of X = x, accumulated as compensated
// dot products (Ogita–Rump–Oishi Dot2: FMA TwoProduct and TwoSum, errors
// summed in float64), accurate as if computed in twice the working
// precision. The float64 conversions keep products from fusing.
func lyapResidualDot2(A *mat.Dense, x, f []float64, k int, discrete bool, ax []dd, r []float64) {
	n, _ := A.Dims()
	aRaw := A.RawMatrix()
	a, lda := aRaw.Data, aRaw.Stride
	if discrete {
		for i := range n {
			ai := a[i*lda : i*lda+n]
			for j := range n {
				var p, e float64
				for l, av := range ai {
					xv := x[l*n+j]
					h := float64(av * xv)
					lo := math.FMA(av, xv, -h)
					s := p + h
					bb := s - p
					e += ((p - (s - bb)) + (h - bb)) + lo
					p = s
				}
				ax[i*n+j] = dd{p, e}
			}
		}
		for i := range n {
			axi := ax[i*n : i*n+n]
			fi := f[i*k : i*k+k]
			for j := i; j < n; j++ {
				aj := a[j*lda : j*lda+n]
				var p, e float64
				for l, y := range axi {
					av := aj[l]
					h := float64(y.hi * av)
					lo := math.FMA(y.hi, av, -h)
					s := p + h
					bb := s - p
					e += ((p - (s - bb)) + (h - bb)) + lo + y.lo*av
					p = s
				}
				{
					h := -x[i*n+j]
					s := p + h
					bb := s - p
					e += (p - (s - bb)) + (h - bb)
					p = s
				}
				fj := f[j*k : j*k+k]
				for l, fv := range fi {
					gv := fj[l]
					h := float64(fv * gv)
					lo := math.FMA(fv, gv, -h)
					s := p + h
					bb := s - p
					e += ((p - (s - bb)) + (h - bb)) + lo
					p = s
				}
				v := p + e
				r[i*n+j] = v
				r[j*n+i] = v
			}
		}
		return
	}
	for i := range n {
		ai := a[i*lda : i*lda+n]
		xi := x[i*n : i*n+n]
		fi := f[i*k : i*k+k]
		for j := i; j < n; j++ {
			aj := a[j*lda : j*lda+n]
			xj := x[j*n : j*n+n]
			var p, e float64
			for l, av := range ai {
				xv := xj[l]
				h := float64(av * xv)
				lo := math.FMA(av, xv, -h)
				s := p + h
				bb := s - p
				e += ((p - (s - bb)) + (h - bb)) + lo
				p = s
				av, xv = aj[l], xi[l]
				h = float64(av * xv)
				lo = math.FMA(av, xv, -h)
				s = p + h
				bb = s - p
				e += ((p - (s - bb)) + (h - bb)) + lo
				p = s
			}
			fj := f[j*k : j*k+k]
			for l, fv := range fi {
				gv := fj[l]
				h := float64(fv * gv)
				lo := math.FMA(fv, gv, -h)
				s := p + h
				bb := s - p
				e += ((p - (s - bb)) + (h - bb)) + lo
				p = s
			}
			v := p + e
			r[i*n+j] = v
			r[j*n+i] = v
		}
	}
}

// semidefiniteFactor returns L (n×n) with X ≈ L·Lᵀ for a symmetric X that
// is positive definite in exact arithmetic but not to working precision:
// the Cholesky factor when it exists, else V·|Λ|^½ from X = V·Λ·Vᵀ with each
// |λ| kept at least 2⁻¹⁰⁶·max|λ| so that the factor stays nonsingular.
func semidefiniteFactor(x []float64, n int) ([]float64, error) {
	l := append([]float64(nil), x...)
	if impl.Dpotrf(blas.Lower, n, l, n) {
		for i := range n {
			clear(l[i*n+i+1 : i*n+n])
		}
		return l, nil
	}
	var eig mat.EigenSym
	if !eig.Factorize(mat.NewSymDense(n, append([]float64(nil), x...)), true) {
		return nil, ErrSchurFailed
	}
	vals := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	top := 0.0
	for _, v := range vals {
		top = math.Max(top, math.Abs(v))
	}
	floor := math.Ldexp(top, -106)
	vRaw := V.RawMatrix()
	for j, v := range vals {
		s := math.Sqrt(math.Max(math.Abs(v), floor))
		for i := range n {
			l[i*n+j] = vRaw.Data[i*vRaw.Stride+j] * s
		}
	}
	return l, nil
}

func scaledNorm2(x []float64) float64 {
	return blas64.Nrm2(blas64.Vector{N: len(x), Inc: 1, Data: x})
}
