package controlsys

import (
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

type complexSVDWorkspace struct {
	block  []float64
	eig    []float64
	work   []float64
	scale  float64
	nSV    int
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
		nSV:    gramN,
		gramN:  gramN,
		blockN: blockN,
		useCol: m <= p,
	}
}

func (ws *complexSVDWorkspace) singularValuesFromFlat(dst []float64, data []complex128, base, p, m int) {
	ws.fillBlock(func(i, j int) complex128 {
		return data[base+i*m+j]
	}, p, m)
	ws.singularValues(dst)
}

func (ws *complexSVDWorkspace) singularValuesFromNested(dst []float64, data [][]complex128, p, m int) {
	ws.fillBlock(func(i, j int) complex128 {
		return data[i][j]
	}, p, m)
	ws.singularValues(dst)
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

func (ws *complexSVDWorkspace) singularValues(dst []float64) {
	if !ws.factorize() {
		for i := range dst {
			dst[i] = math.NaN()
		}
		return
	}
	scale := 1.0
	for _, lambda := range ws.eig {
		scale = max(scale, math.Abs(lambda))
	}
	for i := range dst {
		lambda := nonnegativeGramEigenvalue(ws.eig[ws.blockN-1-2*i], scale)
		dst[i] = ws.scale * math.Sqrt(lambda)
	}
}

func (ws *complexSVDWorkspace) maximumFromFlat(data []complex128, base, p, m int) (float64, bool) {
	if p == 0 || m == 0 {
		return 0, true
	}
	values := data[base : base+p*m]
	if p == 1 || m == 1 {
		var norm float64
		for _, value := range values {
			norm = math.Hypot(norm, cmplx.Abs(value))
		}
		return norm, true
	}
	if p == 2 && m == 2 {
		return maximumComplex2x2SingularValue(values), true
	}
	ws.fillBlock(func(i, j int) complex128 {
		return data[base+i*m+j]
	}, p, m)
	if !ws.factorize() {
		return 0, false
	}
	scale := 1.0
	for _, lambda := range ws.eig {
		scale = max(scale, math.Abs(lambda))
	}
	lambda := nonnegativeGramEigenvalue(ws.eig[ws.blockN-1], scale)
	return ws.scale * math.Sqrt(lambda), true
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
