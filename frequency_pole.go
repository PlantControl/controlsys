package controlsys

import (
	"math"
	"math/cmplx"
)

// poleResponse is the response entry at one of its poles. Like MATLAB's 1/0
// it is real +Inf; multiplying by a unit-modulus delay factor keeps
// cmplx.IsInf true.
var poleResponse = complex(math.Inf(1), 0)

const (
	poleStepFraction = 1e-5
	poleResidueTol   = 1e-6
	poleResidueFloor = 1e-10
)

// evalWithPoleLimit sets dst to eval(s), or to its pole limit when s is a
// numerical pole of the realization.
func evalWithPoleLimit(eval func(complex128, []complex128) error, sys *System, s complex128, dst []complex128) error {
	err := eval(s, dst)
	if err == nil {
		return nil
	}
	return poleLimitInto(eval, s, poleSpacing(sys, s), dst, err)
}

// poleLimitInto sets dst to the response at s, a numerical pole of the
// realization at which eval fails. Each entry is the quadratic extrapolation
// to s of eval at s+kδ, k = 1, 2, 3, unless kδ·G(s+kδ) extrapolates to a
// nonzero residue, in which case the entry is poleResponse. A residue below
// poleResidueFloor times the largest one is rounding from the solve's large
// state and marks a channel the pole does not reach. δ is a fraction of rho,
// the distance from s to the nearest other singularity, so finite entries
// carry an extrapolation error of order (δ/rho)³ plus rounding of order
// eps·(rho/δ)^q from a pole of order q; for q > 1 they are instead quartic
// extrapolations from five points with δ = rho·eps^{1/(q+5)}, which balances
// the two. When eval also fails off s the model is singular everywhere and
// singularErr is returned.
func poleLimitInto(eval func(complex128, []complex128) error, s complex128, rho float64, dst []complex128, singularErr error) error {
	k := len(dst)
	g := make([]complex128, 5*k)
	sample := func(points int, delta float64) error {
		for i := range points {
			if err := eval(s+complex(float64(i+1)*delta, 0), g[i*k:(i+1)*k]); err != nil {
				return singularErr
			}
		}
		return nil
	}
	delta := poleStepFraction * rho
	if err := sample(3, delta); err != nil {
		return err
	}
	d := complex(delta, 0)
	residue := make([]float64, k)
	maxResidue := 0.0
	for j := range dst {
		residue[j] = cmplx.Abs(3 * d * (g[j] - 2*g[k+j] + g[2*k+j]))
		maxResidue = max(maxResidue, residue[j])
	}
	pole := make([]bool, k)
	order := 1
	for j := range dst {
		scale := max(cmplx.Abs(d*g[j]), cmplx.Abs(2*d*g[k+j]), cmplx.Abs(3*d*g[2*k+j]))
		if residue[j] > poleResidueTol*scale && residue[j] > poleResidueFloor*maxResidue {
			pole[j] = true
			order = max(order, int(math.Round(math.Log2(cmplx.Abs(g[j])/cmplx.Abs(g[k+j])))))
		}
	}
	weights := []complex128{3, -3, 1}
	if order > 1 {
		weights = []complex128{5, -10, 10, -5, 1}
		if err := sample(len(weights), rho*math.Pow(eps(), 1/float64(order+5))); err != nil {
			return err
		}
	}
	for j := range dst {
		if pole[j] {
			dst[j] = poleResponse
			continue
		}
		var v complex128
		for i, w := range weights {
			v += w * g[i*k+j]
		}
		dst[j] = v
	}
	return nil
}

// poleSpacing returns the distance from s to the nearest pole of the
// delay-free plant (A, E) that is distinct from s, bounded by the overall
// spectral scale and, for continuous internal delays, by 1/τ, over which
// e^{-sτ} varies.
func poleSpacing(sys *System, s complex128) float64 {
	n, _, _ := sys.Dims()
	var poles []complex128
	if n > 0 {
		poles, _ = newDescriptorPolicy(sys).poles("poleSpacing", sys.A, n)
	}
	scale := cmplx.Abs(s)
	for _, p := range poles {
		if !cmplx.IsInf(p) && !cmplx.IsNaN(p) {
			scale = max(scale, cmplx.Abs(p))
		}
	}
	if scale == 0 {
		scale = 1
	}
	rho := scale
	for _, p := range poles {
		if d := cmplx.Abs(p - s); d > 1e-6*scale && d < rho {
			rho = d
		}
	}
	if sys.IsContinuous() && sys.LFT != nil {
		for _, tau := range sys.LFT.Tau {
			if tau > 0 {
				rho = min(rho, 1/tau)
			}
		}
	}
	return rho
}

// evalFrLFTFrozenInto sets ws.g to G(s) by closing the delay channels with
// their values at s, K = Δ(I-D22Δ)⁻¹ = (I-ΔD22)⁻¹Δ, and solving the complex
// realization (sE - A - B2·K·C2, B1 + B2·K·D21, C1 + D12·K·C2, D11 + D12·K·D21)
// once. Unlike the H-then-LFT form it stays regular at poles of the
// delay-free plant that the delay loop moves, such as an integrator inside it.
func evalFrLFTFrozenInto(ws *lftWorkspace, s complex128, N, p, m int) error {
	br := &ws.bd.balancedRealization
	n, mN := br.n, m+N
	K := make([]complex128, N*N)
	lhs := make([]complex128, N*N)
	for i := range N {
		K[i*N+i] = ws.delta[i]
		for j := range N {
			lhs[i*N+j] = -ws.delta[i] * complex(br.d[(p+i)*br.dStride+m+j], 0)
		}
		lhs[i*N+i] += 1
	}
	if err := cSolveInPlace(lhs, K, N, N); err != nil {
		return err
	}
	nm := n + m
	T := make([]complex128, N*nm)
	for i := range N {
		for k := range N {
			kik := K[i*N+k]
			if kik == 0 {
				continue
			}
			row := p + k
			for j := range n {
				T[i*nm+j] += kik * complex(br.c[row*n+j], 0)
			}
			for j := range m {
				T[i*nm+n+j] += kik * complex(br.d[row*br.dStride+j], 0)
			}
		}
	}
	pencil := make([]complex128, n*n)
	x := make([]complex128, n*m)
	for i := range n {
		for j := range n {
			v := -complex(br.a[i*n+j], 0)
			if br.e != nil {
				v += s * complex(br.e[i*n+j], 0)
			} else if i == j {
				v += s
			}
			pencil[i*n+j] = v
		}
		for j := range m {
			x[i*m+j] = complex(br.b[i*mN+j], 0)
		}
		for k := range N {
			b2 := complex(br.b[i*mN+m+k], 0)
			if b2 == 0 {
				continue
			}
			for j := range n {
				pencil[i*n+j] -= b2 * T[k*nm+j]
			}
			for j := range m {
				x[i*m+j] += b2 * T[k*nm+n+j]
			}
		}
	}
	if n > 0 {
		if err := cSolveInPlace(pencil, x, n, m); err != nil {
			return err
		}
	}
	for i := range p {
		for j := range m {
			v := complex(br.d[i*br.dStride+j], 0)
			for k := range N {
				v += complex(br.d[i*br.dStride+m+k], 0) * T[k*nm+n+j]
			}
			for l := range n {
				c := complex(br.c[i*n+l], 0)
				for k := range N {
					c += complex(br.d[i*br.dStride+m+k], 0) * T[k*nm+l]
				}
				v += c * x[l*m+j]
			}
			ws.g[i*m+j] = v
		}
	}
	return nil
}
