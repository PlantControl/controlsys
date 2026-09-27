package controlsys

import (
	"fmt"
	"math"
	"math/rand"

	"plantcontrol.org/v1/gonum/mat"
)

func Rss(n, p, m int) (*System, error) {
	if n < 0 || p < 1 || m < 1 {
		return nil, fmt.Errorf("controlsys: invalid dimensions n=%d p=%d m=%d", n, p, m)
	}
	return randomSS(randomStableModelSpec{states: n, outputs: p, inputs: m, dt: 0, continuous: true})
}

func Drss(n, p, m int, dt float64) (*System, error) {
	if n < 0 || p < 1 || m < 1 {
		return nil, fmt.Errorf("controlsys: invalid dimensions n=%d p=%d m=%d", n, p, m)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}
	return randomSS(randomStableModelSpec{states: n, outputs: p, inputs: m, dt: dt, continuous: false})
}

type randomSource interface {
	Float64() float64
	NormFloat64() float64
}

type packageRandomSource struct{}

func (packageRandomSource) Float64() float64     { return rand.Float64() }
func (packageRandomSource) NormFloat64() float64 { return rand.NormFloat64() }

type randomStableModelSpec struct {
	states     int
	outputs    int
	inputs     int
	dt         float64
	continuous bool
}

func randomSS(spec randomStableModelSpec) (*System, error) {
	return randomSSWithSource(spec, packageRandomSource{})
}

func randomSSWithSource(spec randomStableModelSpec, rng randomSource) (*System, error) {
	if spec.states == 0 {
		dData := make([]float64, spec.outputs*spec.inputs)
		for i := range dData {
			dData[i] = rng.NormFloat64()
		}
		return NewGain(mat.NewDense(spec.outputs, spec.inputs, dData), spec.dt)
	}

	A := randomStableAWithSource(spec.states, spec.continuous, rng)

	bData := make([]float64, spec.states*spec.inputs)
	for i := range bData {
		bData[i] = rng.NormFloat64()
	}
	B := mat.NewDense(spec.states, spec.inputs, bData)

	cData := make([]float64, spec.outputs*spec.states)
	for i := range cData {
		cData[i] = rng.NormFloat64()
	}
	C := mat.NewDense(spec.outputs, spec.states, cData)

	D := mat.NewDense(spec.outputs, spec.inputs, nil)

	return newNoCopy(A, B, C, D, spec.dt)
}

func randomStableA(n int, continuous bool) *mat.Dense {
	return randomStableAWithSource(n, continuous, packageRandomSource{})
}

func randomStableAWithSource(n int, continuous bool, rng randomSource) *mat.Dense {
	diagData := make([]float64, n*n)
	i := 0
	for i < n {
		if i+1 < n && rng.Float64() < 0.5 {
			if continuous {
				sigma := -rng.Float64()*4 - 0.1
				omega := rng.Float64()*4 + 0.1
				diagData[i*n+i] = sigma
				diagData[i*n+i+1] = omega
				diagData[(i+1)*n+i] = -omega
				diagData[(i+1)*n+i+1] = sigma
			} else {
				r := rng.Float64()*0.9 + 0.01
				theta := rng.Float64() * math.Pi
				diagData[i*n+i] = r * math.Cos(theta)
				diagData[i*n+i+1] = r * math.Sin(theta)
				diagData[(i+1)*n+i] = -r * math.Sin(theta)
				diagData[(i+1)*n+i+1] = r * math.Cos(theta)
			}
			i += 2
		} else {
			if continuous {
				diagData[i*n+i] = -rng.Float64()*4 - 0.1
			} else {
				diagData[i*n+i] = rng.Float64()*1.8 - 0.9
			}
			i++
		}
	}
	D := mat.NewDense(n, n, diagData)

	Q := randomOrthogonalWithSource(n, rng)

	// A = Q * D * Q'
	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(Q, D)
	A := mat.NewDense(n, n, nil)
	A.Mul(tmp, Q.T())
	return A
}

func randomOrthogonal(n int) *mat.Dense {
	return randomOrthogonalWithSource(n, packageRandomSource{})
}

func randomOrthogonalWithSource(n int, rng randomSource) *mat.Dense {
	data := make([]float64, n*n)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	M := mat.NewDense(n, n, data)
	var qr mat.QR
	qr.Factorize(M)
	Q := mat.NewDense(n, n, nil)
	qr.QTo(Q)
	return Q
}
