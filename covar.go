package controlsys

import "plantcontrol.org/v1/gonum/mat"

func Covar(sys *System, W *mat.Dense) (*mat.Dense, error) {
	n, m, p := sys.Dims()

	if err := requireStandardCovarianceSystem(sys, "Covar"); err != nil {
		return nil, err
	}
	if err := validateCovarianceRole("Covar", covarianceInputNoise, W, m); err != nil {
		return nil, err
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
	}

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

	// P = C*X*C' + D*W*D'
	var CX mat.Dense
	CX.Mul(sys.C, X)
	P := mat.NewDense(p, p, nil)
	P.Mul(&CX, sys.C.T())

	var DW mat.Dense
	DW.Mul(sys.D, W)
	var DWDt mat.Dense
	DWDt.Mul(&DW, sys.D.T())
	P.Add(P, &DWDt)

	return P, nil
}
