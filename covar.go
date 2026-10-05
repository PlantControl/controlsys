package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// Covar computes the steady-state output covariance of sys driven by white
// noise with intensity W. For continuous models, entries fed by white noise
// through D are infinite.
func Covar(sys *System, W *mat.Dense) (*mat.Dense, error) {
	if err := requireStandardCovarianceSystem(sys, "Covar"); err != nil {
		return nil, err
	}
	_, m, _ := sys.Dims()
	if err := validateCovarianceRole("Covar", covarianceInputNoise, W, m); err != nil {
		return nil, err
	}
	sys, err := absorbEnergyInternalDelay(sys, "Covar")
	if err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
	}

	P := mat.NewDense(p, p, nil)
	if n > 0 {
		Q := inputNoiseIntensity(sys.B, W, n, m)

		// symmetrize Q for Lyapunov solver
		qRaw := Q.RawMatrix()
		for i := range n {
			for j := i + 1; j < n; j++ {
				avg := (qRaw.Data[i*qRaw.Stride+j] + qRaw.Data[j*qRaw.Stride+i]) / 2
				qRaw.Data[i*qRaw.Stride+j] = avg
				qRaw.Data[j*qRaw.Stride+i] = avg
			}
		}

		var X *mat.Dense
		if sys.IsContinuous() {
			X, err = Lyap(sys.A, Q, nil)
		} else {
			X, err = DLyap(sys.A, Q, nil)
		}
		if err != nil {
			return nil, err
		}

		var CX mat.Dense
		CX.Mul(sys.C, X)
		P.Mul(&CX, sys.C.T())
	}
	if m == 0 {
		return P, nil
	}

	var DW, DWDt mat.Dense
	DW.Mul(sys.D, W)
	DWDt.Mul(&DW, sys.D.T())
	if sys.IsDiscrete() {
		P.Add(P, &DWDt)
		return P, nil
	}
	for i := range p {
		for j := range p {
			if v := DWDt.At(i, j); v != 0 {
				P.Set(i, j, math.Inf(int(math.Copysign(1, v))))
			}
		}
	}
	return P, nil
}
