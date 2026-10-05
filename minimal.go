package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// ReduceMode selects which states Reduce removes.
type ReduceMode int

const (
	// ReduceAll removes uncontrollable and unobservable states (MATLAB minreal).
	ReduceAll ReduceMode = iota
	// ReduceUncontrollable removes only uncontrollable states.
	ReduceUncontrollable
	// ReduceUnobservable removes only unobservable states.
	ReduceUnobservable
)

// ReduceOpts configures Reduce; the zero value is MATLAB minreal(sys).
type ReduceOpts struct {
	Mode ReduceMode
	// Tol is the staircase rank tolerance (MATLAB minreal tol); 0 selects
	// n²·eps relative to the data.
	Tol float64
	// Equalize balances A, B, C before the staircase reductions.
	Equalize bool
}

// ReduceResult is a reduced realization and its state count.
type ReduceResult struct {
	Sys   *System
	Order int
}

// Reduce removes uncontrollable and/or unobservable states of sys by
// orthogonal staircase reductions, like MATLAB minreal(sys,tol); the result
// has the same transfer function and keeps D and the delays. A nil
// opts is ReduceAll with the default tolerance. A descriptor model returns
// ErrDescriptorUnsupported; an unknown Mode, a negative or non-finite Tol,
// and a nil or non-finite model return ErrInvalidArgument.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.minreal.html.
func (sys *System) Reduce(opts *ReduceOpts) (*ReduceResult, error) {
	if err := requireFiniteSystem("Reduce", sys); err != nil {
		return nil, err
	}
	if opts == nil {
		opts = &ReduceOpts{}
	}
	switch opts.Mode {
	case ReduceAll, ReduceUncontrollable, ReduceUnobservable:
	default:
		return nil, fmt.Errorf("Reduce: unknown mode %d: %w", opts.Mode, ErrInvalidArgument)
	}
	if !(opts.Tol >= 0) || math.IsInf(opts.Tol, 1) {
		return nil, fmt.Errorf("Reduce: Tol is %g, want finite and non-negative: %w", opts.Tol, ErrInvalidArgument)
	}

	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Reduce"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		return &ReduceResult{Sys: policy.zeroOrderCopy(), Order: 0}, nil
	}
	if sys.internalDelayCount() > 0 {
		return reduceInternalDelay(sys, policy, opts)
	}

	if m == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable) {
		return zeroOrderResult(sys)
	}
	if p == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable) {
		return zeroOrderResult(sys)
	}

	A := mat.DenseCopyOf(sys.A)
	B := mat.DenseCopyOf(sys.B)
	C := mat.DenseCopyOf(sys.C)

	if opts.Equalize {
		equalize(A, B, C, n, m, p)
	}

	ncont := n

	if opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable {
		res, err := controllabilityStaircase(A, B, C, opts.Tol, false)
		if err != nil {
			return nil, fmt.Errorf("Reduce: %w", err)
		}
		A = res.A
		B = res.B
		C = res.C
		ncont = res.NCont
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

			dualRes, err := controllabilityStaircase(acT, ccT, bcT, opts.Tol, false)
			if err != nil {
				return nil, fmt.Errorf("Reduce: %w", err)
			}
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
		return zeroOrderResult(sys)
	}

	ar := extractSubmatrix(A, 0, nr, 0, nr)
	br := extractSubmatrix(B, 0, nr, 0, m)
	cr := extractSubmatrix(C, 0, p, 0, nr)
	dr := denseCopySafe(sys.D, p, m)

	reduced, err := policy.result(ar, br, cr, dr)
	if err != nil {
		return nil, err
	}

	return &ReduceResult{Sys: reduced, Order: nr}, nil
}

// reduceInternalDelay reduces the delay-free augmented model
// [A, [B B2]; [C; C2], 0]: the delays close around its I/O channels, so any
// state it does not need is not needed by the delayed system either.
func reduceInternalDelay(sys *System, policy realizationTransformPolicy, opts *ReduceOpts) (*ReduceResult, error) {
	n, m, p := policy.n, policy.m, policy.p
	N := sys.internalDelayCount()
	B := newDense(n, m+N)
	C := newDense(p+N, n)
	setBlock(B, 0, 0, sys.B)
	setBlock(B, 0, m, sys.LFT.B2)
	setBlock(C, 0, 0, sys.C)
	setBlock(C, p, 0, sys.LFT.C2)
	aug, err := newNoCopy(denseCopy(sys.A), B, C, newDense(p+N, m+N), sys.Dt)
	if err != nil {
		return nil, err
	}
	red, err := aug.Reduce(opts)
	if err != nil {
		return nil, err
	}
	nr := red.Order
	block := func(src *mat.Dense, r0, r1, c0, c1 int) *mat.Dense {
		if r1 == r0 || c1 == c0 {
			return &mat.Dense{}
		}
		return extractSubmatrix(src, r0, r1, c0, c1)
	}
	reduced, err := policy.resultWithInternalDelay(
		denseCopy(red.Sys.A),
		nonEmptyDense(block(red.Sys.B, 0, nr, 0, m)),
		nonEmptyDense(block(red.Sys.C, 0, p, 0, nr)),
		denseCopySafe(sys.D, p, m),
		block(red.Sys.B, 0, nr, m, m+N),
		block(red.Sys.C, p, p+N, 0, nr),
	)
	if err != nil {
		return nil, err
	}
	return &ReduceResult{Sys: reduced, Order: nr}, nil
}

// MinimalRealization is Reduce(nil), MATLAB minreal(sys) with the default
// tolerance; use Reduce to set Tol.
func (sys *System) MinimalRealization() (*ReduceResult, error) {
	return sys.Reduce(nil)
}

func zeroOrderResult(sys *System) (*ReduceResult, error) {
	g, err := newRealizationTransformPolicy(sys).zeroOrderOriginalFeedthrough()
	if err != nil {
		return nil, fmt.Errorf("Reduce: %w", err)
	}
	return &ReduceResult{Sys: g, Order: 0}, nil
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
