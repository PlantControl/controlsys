package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type SsbalResult struct {
	Sys *System
	T   *mat.Dense
}

func Ssbal(sys *System) (*SsbalResult, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Ssbal"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Ssbal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		eye := &mat.Dense{}
		return &SsbalResult{Sys: policy.zeroOrderCopy(), T: eye}, nil
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	d := make([]float64, n)
	for i := range d {
		d[i] = 1.0
	}

	// Iterative diagonal balancing
	const maxIter = 20
	const beta = 2.0
	for range maxIter {
		changed := false
		for i := range n {
			rowNorm := 0.0
			colNorm := 0.0

			for j := range n {
				rowNorm += math.Abs(aRaw.Data[i*aRaw.Stride+j]) * d[j]
				colNorm += math.Abs(aRaw.Data[j*aRaw.Stride+i]) * d[j]
			}

			for j := range m {
				rowNorm += math.Abs(bRaw.Data[i*bRaw.Stride+j])
			}

			for j := range p {
				colNorm += math.Abs(cRaw.Data[j*cRaw.Stride+i])
			}

			if rowNorm == 0 || colNorm == 0 {
				continue
			}

			f := 1.0
			s := rowNorm + colNorm
			// Scale by powers of beta to find best balance
			for rowNorm < colNorm/beta {
				rowNorm *= beta
				colNorm /= beta
				f *= beta
			}
			for rowNorm > colNorm*beta {
				rowNorm /= beta
				colNorm *= beta
				f /= beta
			}

			if rowNorm+colNorm < 0.95*s {
				changed = true
				d[i] *= f
			}
		}
		if !changed {
			break
		}
	}

	Anew := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			Anew.Set(i, j, (d[i]/d[j])*aRaw.Data[i*aRaw.Stride+j])
		}
	}

	Bnew := mat.NewDense(n, m, nil)
	for i := range n {
		for j := range m {
			Bnew.Set(i, j, d[i]*bRaw.Data[i*bRaw.Stride+j])
		}
	}

	Cnew := mat.NewDense(p, n, nil)
	for i := range p {
		for j := range n {
			Cnew.Set(i, j, cRaw.Data[i*cRaw.Stride+j]/d[j])
		}
	}

	Dnew := denseCopy(sys.D)

	T := mat.NewDense(n, n, nil)
	for i := range n {
		T.Set(i, i, d[i])
	}

	newSys, err := policy.result(Anew, Bnew, Cnew, Dnew)
	if err != nil {
		return nil, err
	}
	return &SsbalResult{Sys: newSys, T: T}, nil
}
