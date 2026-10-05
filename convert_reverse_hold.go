package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"plantcontrol.org/v1/gonum/mat"
)

func (sys *System) d2cZOHRobust() (*System, error) {
	if sys.HasInternalDelay() {
		return undiscretizeInternal(sys, (*System).d2cZOHRobust)
	}
	n, m, _ := sys.Dims()
	if n == 0 {
		out := sys.Copy()
		out.Dt = 0
		d2cPropagateDelays(out, sys, sys.Dt)
		return out, nil
	}
	if fast, err := sys.d2cZOHFast(); err == nil {
		return fast, nil
	}
	augmented := mat.NewDense(n+m, n+m, nil)
	ar, br, aug := sys.A.RawMatrix(), sys.B.RawMatrix(), augmented.RawMatrix()
	for i := range n {
		copy(aug.Data[i*aug.Stride:i*aug.Stride+n], ar.Data[i*ar.Stride:i*ar.Stride+n])
		copy(aug.Data[i*aug.Stride+n:i*aug.Stride+n+m], br.Data[i*br.Stride:i*br.Stride+m])
	}
	for i := range m {
		aug.Data[(n+i)*aug.Stride+n+i] = 1
	}
	negative := false
	var values []complex128
	if !matrixRightHalfPlane(sys.A) {
		var eig mat.Eigen
		if !eig.Factorize(sys.A, mat.EigenNone) {
			return nil, fmt.Errorf("D2C zoh: eigenvalue computation failed: %w", ErrSchurFailed)
		}
		values = eig.Values(nil)
		for _, v := range values {
			if v == 0 {
				return nil, fmt.Errorf("D2C zoh: pole at z=0 has no finite continuous logarithm: %w", ErrSingularTransform)
			}
			negative = negative || (imag(v) == 0 && real(v) < 0)
		}
	}

	var out *System
	if negative {
		var err error
		out, err = sys.d2cZOHRealExtension(augmented, values)
		if err != nil {
			return nil, err
		}
	} else {
		logarithm, err := matrixLogISS(augmented)
		if err != nil {
			return nil, fmt.Errorf("D2C zoh: %w", err)
		}
		a, b := mat.NewDense(n, n, nil), newDense(n, m)
		lr, acr, bcr := logarithm.RawMatrix(), a.RawMatrix(), b.RawMatrix()
		for i := range n {
			for j := range n {
				acr.Data[i*acr.Stride+j] = lr.Data[i*lr.Stride+j] / sys.Dt
			}
			for j := range m {
				bcr.Data[i*bcr.Stride+j] = lr.Data[i*lr.Stride+n+j] / sys.Dt
			}
		}
		out = &System{A: a, B: b, C: denseCopy(sys.C), D: denseCopy(sys.D)}
		propagateNames(out, sys)
	}
	d2cPropagateDelays(out, sys, sys.Dt)
	return out, nil
}

// Only the negative spectral subspace needs conjugate alias states. Original
// states stay first so [x_d;0] maps initial conditions without caller sequencing.
func (sys *System) d2cZOHRealExtension(augmented *mat.Dense, values []complex128) (*System, error) {
	n, m, _ := sys.Dims()
	q := n + m
	theta := math.Pi / 4
	negativeCount := 0
	for _, v := range values {
		if imag(v) == 0 && real(v) < 0 {
			negativeCount++
			continue
		}
		if imag(v) != 0 {
			theta = math.Min(theta, (math.Pi-math.Abs(cmplx.Phase(v)))/2)
		}
	}
	if theta <= 32*(math.Nextafter(1, 2)-1) {
		return nil, fmt.Errorf("D2C zoh: conjugate poles too close to the logarithm branch cut: %w", ErrSingularTransform)
	}
	rotated := mat.NewDense(2*q, 2*q, nil)
	rr, ar := rotated.RawMatrix(), augmented.RawMatrix()
	co, si := math.Cos(theta), -math.Sin(theta)
	for i := range q {
		for j := range q {
			v := ar.Data[i*ar.Stride+j]
			rr.Data[i*rr.Stride+j] = co * v
			rr.Data[i*rr.Stride+q+j] = -si * v
			rr.Data[(q+i)*rr.Stride+j] = si * v
			rr.Data[(q+i)*rr.Stride+q+j] = co * v
		}
	}
	logarithm, err := matrixLogISS(rotated)
	if err != nil {
		return nil, fmt.Errorf("D2C zoh: real continuous extension: %w", err)
	}
	re, im := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	rb, ib := newDense(n, m), newDense(n, m)
	lr, rer, imr, rbr, ibr := logarithm.RawMatrix(), re.RawMatrix(), im.RawMatrix(), rb.RawMatrix(), ib.RawMatrix()
	for i := range n {
		for j := range n {
			rer.Data[i*rer.Stride+j] = lr.Data[i*lr.Stride+j] / sys.Dt
			v := lr.Data[(q+i)*lr.Stride+j]
			if i == j {
				v += theta
			}
			imr.Data[i*imr.Stride+j] = v / sys.Dt
		}
		for j := range m {
			rbr.Data[i*rbr.Stride+j] = lr.Data[i*lr.Stride+n+j] / sys.Dt
			ibr.Data[i*ibr.Stride+j] = lr.Data[(q+i)*lr.Stride+n+j] / sys.Dt
		}
	}
	return sys.compressZOHRealExtension(re, im, rb, ib, negativeCount)
}

// discretizeModifiedFOH uses piecewise-linear interpolation between input samples.
// The discrete state is x_d[k] = x_c(k*dt) - Gamma1*u[k].
func (sys *System) discretizeModifiedFOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeFOH: model already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 || math.IsInf(dt, 0) || math.IsNaN(dt) {
		return nil, ErrInvalidSampleTime
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("DiscretizeFOH: modified FOH with internal delays not supported: %w", ErrFeedbackDelay)
	}
	n, m, p := sys.Dims()
	out := sys.Copy()
	out.Dt = dt
	if n > 0 {
		ad, gamma0, gamma1 := fohInputIntegrals(sys.A, sys.B, dt)
		out.A = ad
		if m > 0 {
			var extra mat.Dense
			extra.Mul(ad, gamma1)
			extra.Sub(&extra, gamma1)
			out.B = newDense(n, m)
			out.B.Add(gamma0, &extra)
			if p > 0 {
				var feedthrough mat.Dense
				feedthrough.Mul(sys.C, gamma1)
				out.D.Add(out.D, &feedthrough)
			}
		}

	}
	policy := newDelayConversionPolicy(dt, 0, 0)
	var err error
	if out, err = policy.applyDiscreteDelayFields(out, sys); err != nil {
		return nil, err
	}
	return out, nil
}

func (sys *System) d2cFOH() (*System, error) {
	if sys.HasInternalDelay() {
		return undiscretizeInternal(sys, (*System).d2cFOH)
	}
	n, m, p := sys.Dims()
	if n == 0 {
		out := sys.Copy()
		out.Dt = 0
		d2cPropagateDelays(out, sys, sys.Dt)
		return out, nil
	}
	logarithm, err := matLog(sys.A)
	if err != nil {
		return nil, fmt.Errorf("D2C foh: requires nonzero poles off the negative real axis: %w", err)
	}
	ac := mat.NewDense(n, n, nil)
	ac.Scale(1/sys.Dt, logarithm)
	out := &System{A: ac, B: denseCopy(sys.B), C: denseCopy(sys.C), D: denseCopy(sys.D)}
	if m > 0 {
		ad, k0, k1 := fohKernel(ac, sys.Dt)
		var kernel, ak1 mat.Dense
		ak1.Mul(ad, k1)
		ak1.Sub(&ak1, k1)
		kernel.Add(k0, &ak1)
		var lu mat.LU
		lu.Factorize(&kernel)
		if err := lu.SolveTo(out.B, false, sys.B); err != nil {
			return nil, fmt.Errorf("D2C foh: input reconstruction is singular: %w", ErrSingularTransform)
		}
		if p > 0 {
			var gamma1, feedthrough mat.Dense
			gamma1.Mul(k1, out.B)
			feedthrough.Mul(sys.C, &gamma1)
			out.D.Sub(sys.D, &feedthrough)
		}
	}
	propagateNames(out, sys)
	d2cPropagateDelays(out, sys, sys.Dt)
	return out, nil
}

func fohKernel(a *mat.Dense, dt float64) (ad, k0, k1 *mat.Dense) {
	n, _ := a.Dims()
	eye := mat.NewDense(n, n, nil)
	for i := range n {
		eye.Set(i, i, 1)
	}
	return fohInputIntegrals(a, eye, dt)
}

func fohInputIntegrals(a, b *mat.Dense, dt float64) (ad, gamma0, gamma1 *mat.Dense) {
	n, _ := a.Dims()
	_, m := b.Dims()
	q := n + 2*m
	augmented := mat.NewDense(q, q, nil)
	aug, ar, br := augmented.RawMatrix(), a.RawMatrix(), b.RawMatrix()
	for i := range n {
		for j := range n {
			aug.Data[i*aug.Stride+j] = dt * ar.Data[i*ar.Stride+j]
		}
		for j := range m {
			aug.Data[i*aug.Stride+n+j] = dt * br.Data[i*br.Stride+j]
		}
	}
	for i := range m {
		aug.Data[(n+i)*aug.Stride+n+m+i] = 1
	}
	var exponential mat.Dense
	exponential.Exp(augmented)
	er := exponential.RawMatrix()
	ad, gamma0, gamma1 = mat.NewDense(n, n, nil), newDense(n, m), newDense(n, m)
	adr, g0r, g1r := ad.RawMatrix(), gamma0.RawMatrix(), gamma1.RawMatrix()
	for i := range n {
		copy(adr.Data[i*adr.Stride:i*adr.Stride+n], er.Data[i*er.Stride:i*er.Stride+n])
		copy(g0r.Data[i*g0r.Stride:i*g0r.Stride+m], er.Data[i*er.Stride+n:i*er.Stride+n+m])
		copy(g1r.Data[i*g1r.Stride:i*g1r.Stride+m], er.Data[i*er.Stride+n+m:i*er.Stride+n+2*m])
	}
	return ad, gamma0, gamma1
}

func (sys *System) d2cZOHFast() (*System, error) {
	n, m, _ := sys.Dims()
	var lu mat.LU
	if m > 0 {
		minus := denseCopy(sys.A)
		raw := minus.RawMatrix()
		for i := range n {
			raw.Data[i*raw.Stride+i] -= 1
		}
		lu.Factorize(minus)
		if luNearSingular(&lu) {
			return nil, ErrSingularTransform
		}
	}
	logarithm, err := matLogSpectral(sys.A)
	if err != nil {
		return nil, err
	}
	logarithm.Scale(1/sys.Dt, logarithm)
	bc := denseCopy(sys.B)
	if m > 0 {
		var ab mat.Dense
		ab.Mul(logarithm, sys.B)
		if err := lu.SolveTo(bc, false, &ab); err != nil {
			return nil, ErrSingularTransform
		}
	}
	out := &System{A: logarithm, B: bc, C: denseCopy(sys.C), D: denseCopy(sys.D)}
	propagateNames(out, sys)
	d2cPropagateDelays(out, sys, sys.Dt)
	return out, nil
}
