package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

type covarianceRole string

const (
	covarianceInputNoise       covarianceRole = "input noise covariance"
	covarianceProcessNoise     covarianceRole = "process noise covariance"
	covarianceMeasurementNoise covarianceRole = "measurement noise covariance"
	covarianceStateEstimate    covarianceRole = "state estimate covariance"
)

func validateCovarianceRole(context string, role covarianceRole, cov *mat.Dense, dim int) error {
	if cov == nil {
		return fmt.Errorf("%s: nil %s: %w", context, role, ErrDimensionMismatch)
	}
	r, c := cov.Dims()
	if r != dim || c != dim {
		return fmt.Errorf("%s: %s is %dx%d, want %dx%d: %w", context, role, r, c, dim, dim, ErrDimensionMismatch)
	}
	return nil
}

func requireStandardCovarianceSystem(sys *System, context string) error {
	return newDescriptorPolicy(sys).requireStandard(context)
}

func requireStandardEstimatorSystem(sys *System, context string) error {
	return newDescriptorPolicy(sys).requireRiccatiStandard(context)
}

func inputNoiseIntensity(B, W *mat.Dense, n, m int) *mat.Dense {
	if m == 0 {
		return mat.NewDense(n, n, nil)
	}
	bRaw := B.RawMatrix()
	wRaw := W.RawMatrix()

	bwData := make([]float64, n*m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: m, Data: wRaw.Data, Stride: wRaw.Stride},
		0, blas64.General{Rows: n, Cols: m, Data: bwData, Stride: m})

	qData := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.Trans,
		1, blas64.General{Rows: n, Cols: m, Data: bwData, Stride: m},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		0, blas64.General{Rows: n, Cols: n, Data: qData, Stride: n})

	return mat.NewDense(n, n, qData)
}
