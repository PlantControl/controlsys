package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type ReduceMode int

const (
	ReduceAll ReduceMode = iota
	ReduceUncontrollable
	ReduceUnobservable
)

type ReduceOpts struct {
	Mode     ReduceMode
	Tol      float64
	Equalize bool
}

type ReduceResult struct {
	Sys        *System
	Order      int
	BlockSizes []int
}

func (sys *System) Reduce(opts *ReduceOpts) (*ReduceResult, error) {
	if opts == nil {
		opts = &ReduceOpts{}
	}

	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Reduce"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		return &ReduceResult{Sys: policy.zeroOrderCopy(), Order: 0}, nil
	}

	if m == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable) {
		return zeroOrderResult(sys, m, p), nil
	}
	if p == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable) {
		return zeroOrderResult(sys, m, p), nil
	}

	A := mat.DenseCopyOf(sys.A)
	B := mat.DenseCopyOf(sys.B)
	C := mat.DenseCopyOf(sys.C)

	if opts.Equalize {
		equalize(A, B, C, n, m, p)
	}

	ncont := n
	var blockSizes []int

	if opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable {
		res := ControllabilityStaircase(A, B, C, opts.Tol)
		A = res.A
		B = res.B
		C = res.C
		ncont = res.NCont
		blockSizes = res.BlockSizes
	}

	nr := ncont

	if opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable {
		if ncont == 0 {
			nr = 0
		} else {
			ac := extractSubmatrix(A, 0, ncont, 0, ncont)
			bc := extractSubmatrix(B, 0, ncont, 0, m)
			cc := extractSubmatrix(C, 0, p, 0, ncont)

			slab1 := make([]float64, ncont*ncont+ncont*p+m*ncont)
			acT := transposeDenseInto(slab1[:ncont*ncont], ac)
			ccT := transposeDenseInto(slab1[ncont*ncont:ncont*ncont+ncont*p], cc)
			bcT := transposeDenseInto(slab1[ncont*ncont+ncont*p:], bc)

			dualRes := ControllabilityStaircase(acT, ccT, bcT, opts.Tol)
			nobs := dualRes.NCont

			if nobs == 0 {
				nr = 0
			} else {
				nr = nobs

				aObsDual := extractSubmatrix(dualRes.A, 0, nobs, 0, nobs)
				slab2 := make([]float64, nobs*nobs+nobs*m+p*nobs)
				aObs := transposeDenseInto(slab2[:nobs*nobs], aObsDual)
				bObs := transposeDenseInto(slab2[nobs*nobs:nobs*nobs+nobs*m], extractSubmatrix(dualRes.C, 0, m, 0, nobs))
				cObs := transposeDenseInto(slab2[nobs*nobs+nobs*m:], extractSubmatrix(dualRes.B, 0, nobs, 0, p))

				// Pertranspose for upper block Hessenberg: P*A*P, P*B, C*P
				pertransposeSquare(aObs, nobs)
				reverseRows(bObs, nobs)
				reverseCols(cObs, nobs)

				A = aObs
				B = bObs
				C = cObs
			}
		}
	}

	if nr == 0 {
		res := zeroOrderResult(sys, m, p)
		res.BlockSizes = blockSizes
		return res, nil
	}

	ar := extractSubmatrix(A, 0, nr, 0, nr)
	br := extractSubmatrix(B, 0, nr, 0, m)
	cr := extractSubmatrix(C, 0, p, 0, nr)
	dr := denseCopySafe(sys.D, p, m)

	reduced, err := policy.result(ar, br, cr, dr)
	if err != nil {
		return nil, err
	}

	return &ReduceResult{
		Sys:        reduced,
		Order:      nr,
		BlockSizes: blockSizes,
	}, nil
}

func (sys *System) MinimalRealization() (*ReduceResult, error) {
	return sys.Reduce(nil)
}

func copySys(sys *System, n, m, p int) *System {
	cp := &System{
		A:     denseCopySafe(sys.A, n, n),
		B:     denseCopySafe(sys.B, n, m),
		C:     denseCopySafe(sys.C, p, n),
		D:     denseCopySafe(sys.D, p, m),
		Delay: copyDelayOrNil(sys.Delay),
		Dt:    sys.Dt,
	}
	propagateIONames(cp, sys)
	return cp
}

func zeroOrderResult(sys *System, m, p int) *ReduceResult {
	return &ReduceResult{
		Sys:   newRealizationTransformPolicy(sys).zeroOrderOriginalFeedthrough(),
		Order: 0,
	}
}

func extractSubmatrix(m *mat.Dense, r0, r1, c0, c1 int) *mat.Dense {
	return extractBlock(m, r0, c0, r1-r0, c1-c0)
}

func pertransposeSquare(a *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := a.RawMatrix()
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride:]
		rj := raw.Data[j*raw.Stride:]
		for k := range n {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
	for k := range n {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

func reverseRows(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	cols := raw.Cols
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride : i*raw.Stride+cols]
		rj := raw.Data[j*raw.Stride : j*raw.Stride+cols]
		for k := range ri {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
}

func reverseCols(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	for k := 0; k < raw.Rows; k++ {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

// equalize balances A by a diagonal similarity D such that A → D^{-1} A D,
// B → D^{-1} B, C → C D, following LAPACK DGEBAL with the SLICOT TB01ID
// extension that includes B rows and C columns in the row/column 1-norms.
func equalize(A, B, C *mat.Dense, n, m, p int) {
	const sclfac = 10.0
	const factor = 0.95
	sfmin1 := math.SmallestNonzeroFloat64 / eps()
	sfmax1 := 1.0 / sfmin1
	sfmin2 := sfmin1 * sclfac
	sfmax2 := 1.0 / sfmin2

	scale := make([]float64, n)
	for i := range scale {
		scale[i] = 1.0
	}

	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()

	for {
		noconv := false
		for i := range n {
			c := 0.0
			r := 0.0
			ca := 0.0
			ra := 0.0

			aRow := aRaw.Data[i*aRaw.Stride:]
			for j := range n {
				if j == i {
					continue
				}
				v := math.Abs(aRaw.Data[j*aRaw.Stride+i])
				c += v
				if v > ca {
					ca = v
				}
				v = math.Abs(aRow[j])
				r += v
				if v > ra {
					ra = v
				}
			}
			bRow := bRaw.Data[i*bRaw.Stride:]
			for j := range m {
				v := math.Abs(bRow[j])
				r += v
				if v > ra {
					ra = v
				}
			}
			for j := range p {
				v := math.Abs(cRaw.Data[j*cRaw.Stride+i])
				c += v
				if v > ca {
					ca = v
				}
			}

			if c == 0 || r == 0 {
				continue
			}

			g := r / sclfac
			f := 1.0
			s := c + r

			for c < g && max(f, c, ca) < sfmax2 && min(r, g, ra) > sfmin2 {
				f *= sclfac
				c *= sclfac
				ca *= sclfac
				g /= sclfac
				r /= sclfac
				ra /= sclfac
			}

			g = c / sclfac
			for g >= r && max(r, ra) < sfmax2 && min(f, c, g, ca) > sfmin2 {
				f /= sclfac
				c /= sclfac
				g /= sclfac
				ca /= sclfac
				r *= sclfac
				ra *= sclfac
			}

			if c+r >= factor*s {
				continue
			}
			if f < 1 && scale[i] < 1 && f*scale[i] <= sfmin1 {
				continue
			}
			if f > 1 && scale[i] > 1 && scale[i] >= sfmax1/f {
				continue
			}

			scale[i] *= f
			noconv = true

			fi := 1.0 / f
			for j := range n {
				aRow[j] *= fi
			}
			for j := range m {
				bRow[j] *= fi
			}
			for j := range n {
				aRaw.Data[j*aRaw.Stride+i] *= f
			}
			for j := range p {
				cRaw.Data[j*cRaw.Stride+i] *= f
			}
		}
		if !noconv {
			break
		}
	}
}
