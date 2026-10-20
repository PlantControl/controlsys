package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

type complexSVDWorkspace struct {
	block  []float64
	eig    []float64
	work   []float64
	bidiag []complex128
	diag   []float64
	super  []float64
	qwork  []float64
	scale  float64
	gramN  int
	blockN int
	useCol bool
}

func newComplexSVDWorkspace(p, m int) *complexSVDWorkspace {
	gramN := min(p, m)
	blockN := 2 * gramN
	block := make([]float64, blockN*blockN)
	eig := make([]float64, blockN)
	workQuery := make([]float64, 1)
	impl.Dsyev(lapack.EVNone, blas.Upper, blockN, block, blockN, eig, workQuery, -1)
	work := make([]float64, int(workQuery[0]))
	return &complexSVDWorkspace{
		block:  block,
		eig:    eig,
		work:   work,
		bidiag: make([]complex128, p*m),
		diag:   make([]float64, gramN),
		super:  make([]float64, gramN),
		qwork:  make([]float64, 4*gramN),
		gramN:  gramN,
		blockN: blockN,
		useCol: m <= p,
	}
}

func (ws *complexSVDWorkspace) singularValuesFromFlat(dst []float64, data []complex128, base, p, m int) error {
	ws.fillBidiag(func(i, j int) complex128 {
		return data[base+i*m+j]
	}, p, m)
	return ws.singularValues(dst, p, m)
}

func (ws *complexSVDWorkspace) singularValuesFromNested(dst []float64, data [][]complex128, p, m int) error {
	ws.fillBidiag(func(i, j int) complex128 {
		return data[i][j]
	}, p, m)
	return ws.singularValues(dst, p, m)
}

// fillBidiag stores G/‖G‖_max, or its conjugate transpose when p < m, so
// the stored matrix is rows×cols with rows ≥ cols.
func (ws *complexSVDWorkspace) fillBidiag(at func(i, j int) complex128, p, m int) {
	ws.scale = 0
	for i := range p {
		for j := range m {
			magnitude := cmplx.Abs(at(i, j))
			if math.IsNaN(magnitude) || magnitude > ws.scale {
				ws.scale = magnitude
			}
		}
	}
	if ws.scale == 0 || math.IsInf(ws.scale, 1) || math.IsNaN(ws.scale) {
		return
	}
	scale := ws.scale
	a := ws.bidiag
	if m <= p {
		for i := range p {
			for j := range m {
				v := at(i, j)
				a[i*m+j] = complex(real(v)/scale, imag(v)/scale)
			}
		}
		return
	}
	for i := range p {
		for j := range m {
			v := at(i, j)
			a[j*p+i] = complex(real(v)/scale, -imag(v)/scale)
		}
	}
}

// bidiagonalize reduces the stored rows×cols matrix by complex Householder
// reflections to upper bidiagonal form and keeps the moduli of its entries.
// A bidiagonal matrix has the singular values of its entrywise modulus, and
// the reduction is backward stable, so σᵢ get absolute error O(eps·σ₁)
// rather than the O(√eps·σ₁) of Gram eigenvalues.
func (ws *complexSVDWorkspace) bidiagonalize(rows, cols int) {
	a := ws.bidiag
	for k := range cols {
		ws.diag[k] = householderColumn(a, rows, cols, k)
		if k+1 < cols {
			ws.super[k] = householderRow(a, rows, cols, k)
		}
	}
}

// householderColumn annihilates a[k+1:, k], applies the reflection to
// a[k:, k+1:] and returns the modulus of the new a[k,k].
func householderColumn(a []complex128, rows, cols, k int) float64 {
	alpha := a[k*cols+k]
	var sumSq float64
	for i := k + 1; i < rows; i++ {
		sumSq += complexMagnitudeSquared(a[i*cols+k])
	}
	if sumSq == 0 {
		return cmplx.Abs(alpha)
	}
	beta := complex(-math.Copysign(math.Sqrt(complexMagnitudeSquared(alpha)+sumSq), real(alpha)), 0)
	tau := cmplx.Conj((beta - alpha) / beta)
	inv := 1 / (alpha - beta)
	a[k*cols+k] = 1
	for i := k + 1; i < rows; i++ {
		a[i*cols+k] *= inv
	}
	for j := k + 1; j < cols; j++ {
		var w complex128
		for i := k; i < rows; i++ {
			w += cmplx.Conj(a[i*cols+k]) * a[i*cols+j]
		}
		w *= tau
		for i := k; i < rows; i++ {
			a[i*cols+j] -= a[i*cols+k] * w
		}
	}
	return math.Abs(real(beta))
}

// householderRow annihilates a[k, k+2:], applies the reflection to
// a[k+1:, k+1:] and returns the modulus of the new a[k,k+1].
func householderRow(a []complex128, rows, cols, k int) float64 {
	row := a[k*cols : (k+1)*cols]
	alpha := row[k+1]
	var sumSq float64
	for j := k + 2; j < cols; j++ {
		sumSq += complexMagnitudeSquared(row[j])
	}
	if sumSq == 0 {
		return cmplx.Abs(alpha)
	}
	beta := complex(-math.Copysign(math.Sqrt(complexMagnitudeSquared(alpha)+sumSq), real(alpha)), 0)
	tau := cmplx.Conj((beta - alpha) / beta)
	inv := 1 / (alpha - beta)
	row[k+1] = 1
	for j := k + 2; j < cols; j++ {
		row[j] *= inv
	}
	for i := k + 1; i < rows; i++ {
		r := a[i*cols : (i+1)*cols]
		var w complex128
		for j := k + 1; j < cols; j++ {
			w += r[j] * cmplx.Conj(row[j])
		}
		w *= tau
		for j := k + 1; j < cols; j++ {
			r[j] -= w * row[j]
		}
	}
	return math.Abs(real(beta))
}

func (ws *complexSVDWorkspace) fillBlock(at func(i, j int) complex128, p, m int) {
	n := ws.gramN
	stride := ws.blockN
	clear(ws.block)
	ws.scale = 0
	for i := range p {
		for j := range m {
			magnitude := cmplx.Abs(at(i, j))
			if math.IsNaN(magnitude) || magnitude > ws.scale {
				ws.scale = magnitude
			}
		}
	}
	if ws.scale == 0 {
		return
	}
	invScale := complex(1/ws.scale, 0)

	for a := range n {
		for b := range n {
			var g complex128
			if ws.useCol {
				for row := range p {
					left := at(row, a) * invScale
					right := at(row, b) * invScale
					g += cmplx.Conj(left) * right
				}
			} else {
				for col := range m {
					left := at(a, col) * invScale
					right := at(b, col) * invScale
					g += left * cmplx.Conj(right)
				}
			}

			re, im := real(g), imag(g)
			ws.block[a*stride+b] = re
			ws.block[a*stride+n+b] = -im
			ws.block[(n+a)*stride+b] = im
			ws.block[(n+a)*stride+n+b] = re
		}
	}
}

// singularValues writes the len(dst) largest singular values. A block with
// an infinite entry, a response at a pole, has σ₁ = +Inf, but σ₂, … depend on
// the pole's residue and are not determined by the sampled response, so
// asking for more than σ₁ there returns ErrInvalidArgument.
func (ws *complexSVDWorkspace) singularValues(dst []float64, p, m int) error {
	if math.IsInf(ws.scale, 1) {
		if len(dst) > 1 {
			return fmt.Errorf("response is infinite (frequency at a pole), singular values below the largest are undetermined: %w", ErrInvalidArgument)
		}
		dst[0] = math.Inf(1)
		return nil
	}
	if ws.scale == 0 {
		clear(dst)
		return nil
	}
	if math.IsNaN(ws.scale) {
		return fmt.Errorf("singular values: response is NaN: %w", ErrSchurFailed)
	}
	cols := min(p, m)
	ws.bidiagonalize(max(p, m), cols)
	if impl.Dlasq1(cols, ws.diag, ws.super, ws.qwork) != 0 {
		return fmt.Errorf("singular values: bidiagonal SVD did not converge: %w", ErrSchurFailed)
	}
	for i := range dst {
		dst[i] = ws.scale * ws.diag[i]
	}
	return nil
}

func (ws *complexSVDWorkspace) maximumFromFlat(data []complex128, base, p, m int) (float64, error) {
	if p == 0 || m == 0 {
		return 0, nil
	}
	values := data[base : base+p*m]
	if slices.ContainsFunc(values, cmplx.IsInf) {
		return math.Inf(1), nil
	}
	if p == 1 || m == 1 {
		var norm float64
		for _, value := range values {
			norm = math.Hypot(norm, cmplx.Abs(value))
		}
		return norm, nil
	}
	if p == 2 && m == 2 {
		return maximumComplex2x2SingularValue(values), nil
	}
	ws.fillBlock(func(i, j int) complex128 {
		return data[base+i*m+j]
	}, p, m)
	if !ws.factorize() {
		return 0, fmt.Errorf("maximum singular value: eigenvalue iteration did not converge: %w", ErrSchurFailed)
	}
	scale := 1.0
	for _, lambda := range ws.eig {
		scale = max(scale, math.Abs(lambda))
	}
	lambda := nonnegativeGramEigenvalue(ws.eig[ws.blockN-1], scale)
	return ws.scale * math.Sqrt(lambda), nil
}

func (ws *complexSVDWorkspace) factorize() bool {
	return impl.Dsyev(lapack.EVNone, blas.Upper, ws.blockN, ws.block, ws.blockN, ws.eig, ws.work, len(ws.work))
}

func maximumComplex2x2SingularValue(data []complex128) float64 {
	scale := 0.0
	for _, value := range data[:4] {
		scale = math.Max(scale, cmplx.Abs(value))
	}
	if scale == 0 {
		return 0
	}
	a00 := data[0] / complex(scale, 0)
	a01 := data[1] / complex(scale, 0)
	a10 := data[2] / complex(scale, 0)
	a11 := data[3] / complex(scale, 0)
	p := complexMagnitudeSquared(a00) + complexMagnitudeSquared(a01)
	r := complexMagnitudeSquared(a10) + complexMagnitudeSquared(a11)
	q := a00*cmplx.Conj(a10) + a01*cmplx.Conj(a11)
	halfDiff := (p - r) / 2
	radius := math.Sqrt(halfDiff*halfDiff + complexMagnitudeSquared(q))
	return scale * math.Sqrt((p+r)/2+radius)
}

func complexMagnitudeSquared(value complex128) float64 {
	return real(value)*real(value) + imag(value)*imag(value)
}

func nonnegativeGramEigenvalue(lambda, scale float64) float64 {
	if lambda >= 0 {
		return lambda
	}
	if lambda > -1e-12*max(1, scale) {
		return 0
	}
	return lambda
}
