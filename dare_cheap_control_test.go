package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dareDoublingBig returns the stabilizing solution of the Dare of
// (A, B, Q, R) by the structure-preserving doubling algorithm in hbPrec
// bits: A₊ = A(I+GH)⁻¹A, G₊ = G + A(I+GH)⁻¹GAᵀ, H₊ = H + AᵀH(I+GH)⁻¹A
// from G = BR⁻¹Bᵀ, H = Q, with H → X quadratically. It shares no step
// with the pencil or Hewer's iteration that Dare uses. A cross term S
// (nil for none) is eliminated first: A − BR⁻¹Sᵀ, Q − SR⁻¹Sᵀ.
func dareDoublingBig(A, B, Q, R, S *mat.Dense) *mat.Dense {
	h, _ := dareDoublingHB(hbFrom(A), hbFrom(B), hbFrom(Q), hbFrom(R), hbFrom(S), 80)
	return hbDense(h)
}

// dareDoublingHB is dareDoublingBig on 320-bit matrices (s.r == 0 for no
// cross term), at most steps doublings; it reports convergence.
func dareDoublingHB(a, b, h, r, s hbMat, steps int) (hbMat, bool) {
	n := a.r
	if s.r > 0 {
		rs := hbSolve(r, s.t())
		a = hbAdd(a, hbMul(b, rs), -1)
		h = hbAdd(h, hbMul(s, rs), -1)
	}
	g := hbMul(b, hbSolve(r, b.t()))
	eye := newHbMat(n, n)
	for i := range n {
		eye.at(i, i).SetInt64(1)
	}
	tol := new(big.Float).SetPrec(hbPrec).SetFloat64(1e-80)
	for range steps {
		w := hbAdd(eye, hbMul(g, h), 1)
		wa, wg := hbSolve(w, a), hbSolve(w, g)
		hn := hbAdd(h, hbMul(hbMul(a.t(), h), wa), 1)
		gn := hbAdd(g, hbMul(hbMul(a, wg), a.t()), 1)
		an := hbMul(a, wa)
		d := hbAdd(hn, h, -1)
		a, g, h = an, gn, hn
		dm, hm := new(big.Float), new(big.Float)
		for i := range d.d {
			if v := new(big.Float).Abs(d.d[i]); v.Cmp(dm) > 0 {
				dm = v
			}
			if v := new(big.Float).Abs(h.d[i]); v.Cmp(hm) > 0 {
				hm = v
			}
		}
		if dm.Cmp(new(big.Float).Mul(hm, tol)) <= 0 {
			return h, true
		}
	}
	return h, false
}

// dareCheapUnstableCase returns the k-th draw of a cheap-control sweep of
// strongly unstable plants: n = 4–5 states, A ~ 300·N(0,1) (ρ(A) ≈
// 300–900), m = 1–2 inputs, Q = 1e14·CᵀC of rank 1–2, R = 1e-12·I.
func dareCheapUnstableCase(k int) (A, B, Q, R *mat.Dense) {
	rng := rand.New(rand.NewPCG(1, 0))
	for i := 0; ; i++ {
		n, m, p := 4+rng.IntN(2), 1+rng.IntN(2), 1+rng.IntN(2)
		a := make([]float64, n*n)
		for j := range a {
			a[j] = 300 * rng.NormFloat64()
		}
		b := make([]float64, n*m)
		for j := range b {
			b[j] = rng.NormFloat64()
		}
		c := make([]float64, p*n)
		for j := range c {
			c[j] = rng.NormFloat64()
		}
		if i < k {
			continue
		}
		C := mat.NewDense(p, n, c)
		var q mat.Dense
		q.Mul(C.T(), C)
		q.Scale(1e14, &q)
		r := mat.NewDense(m, m, nil)
		for j := range m {
			r.Set(j, j, 1e-12)
		}
		return mat.NewDense(n, n, a), mat.NewDense(n, m, b), &q, r
	}
}

// dareCheapCross returns a cross term S = γ·Q·U (U with unit-norm
// columns) of norm ≈ √(‖Q‖‖R‖) for R = 1e-12·I, with γ small enough that
// R − SᵀQ⁺S = R − γ²UᵀQU stays positive definite: [Q S; Sᵀ R] ⪰ 0.
func dareCheapCross(Q *mat.Dense, m int) *mat.Dense {
	n, _ := Q.Dims()
	u := mat.NewDense(n, m, nil)
	for i := range n {
		for j := range m {
			u.Set(i, j, math.Cos(float64(i+2*j+1))/math.Sqrt(float64(n)))
		}
	}
	gamma := math.Sqrt(0.25e-12 / mat.Norm(Q, 2))
	var s mat.Dense
	s.Mul(Q, u)
	s.Scale(gamma, &s)
	return &s
}

// dareCheapDescriptor returns a nonsymmetric, well-conditioned E.
func dareCheapDescriptor(n int) *mat.Dense {
	e := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			e.Set(i, j, 0.3*math.Cos(float64(3*i+j)))
		}
		e.Set(i, i, 2+0.1*float64(i))
	}
	return e
}

// TV3THH: every cost scale of the extended pencil misplaced its stable
// split or gave an X whose gain destabilized (‖X‖ 1e14–1e30, closed loop
// ρ 0.002–0.04), so Dare returned ErrNoStabilizing although the
// stabilizing solution exists and its rounded K stabilizes.
func TestDare_CheapControlOfUnstablePlantIsCertified(t *testing.T) {
	for _, k := range []int{6, 19, 25, 29} {
		A0, B0, Q, R := dareCheapUnstableCase(k)
		n, m := B0.Dims()
		want := dareDoublingBig(A0, B0, Q, R, nil)
		S := dareCheapCross(Q, m)
		wantS := dareDoublingBig(A0, B0, Q, R, S)
		E := dareCheapDescriptor(n)
		var Ae, Be mat.Dense
		Ae.Mul(E, A0)
		Be.Mul(E, B0)
		for _, tc := range []struct {
			name string
			A, B *mat.Dense
			opts *RiccatiOpts
			want *mat.Dense
		}{
			{"explicit", A0, B0, nil, want},
			{"NoScaling", A0, B0, &RiccatiOpts{NoScaling: true}, want},
			{"descriptor", &Ae, &Be, &RiccatiOpts{E: E}, want},
			{"cross term", A0, B0, &RiccatiOpts{S: S}, wantS},
		} {
			res, err := Dare(tc.A, tc.B, Q, R, tc.opts)
			if err != nil {
				t.Errorf("case %d %s: %v", k, tc.name, err)
				continue
			}
			if res.Accuracy != 0 {
				t.Errorf("case %d %s: Accuracy %d, want 0", k, tc.name, res.Accuracy)
			}
			if !(res.Rcnd > 0 && res.Rcnd <= 1) {
				t.Errorf("case %d %s: Rcnd %g outside (0, 1]", k, tc.name, res.Rcnd)
			}
			X := res.X
			if tc.opts != nil && tc.opts.E != nil {
				var ex mat.Dense
				ex.Mul(E.T(), res.X)
				ex.Mul(&ex, E)
				X = &ex
			}
			var d mat.Dense
			d.Sub(X, tc.want)
			if rel := mat.Norm(&d, 2) / mat.Norm(tc.want, 2); !(rel <= 1e-8) {
				t.Errorf("case %d %s: ‖X − X*‖/‖X*‖ = %.3g (‖X*‖ %.3g)", k, tc.name, rel, mat.Norm(tc.want, 2))
			}
			var acl mat.Dense
			acl.Mul(B0, res.K)
			acl.Sub(A0, &acl)
			var eig mat.Eigen
			if !eig.Factorize(&acl, mat.EigenNone) {
				t.Fatalf("case %d %s: eig(A − BK) failed", k, tc.name)
			}
			for _, v := range eig.Values(nil) {
				if !(cmplx.Abs(v) < 1) {
					t.Errorf("case %d %s: closed-loop eigenvalue %v not stable", k, tc.name, v)
				}
			}
			// The closed loop clusters near 0 like a deadbeat Jordan block
			// of norm 1e3, so single eigenvalues move by 1e-4 under
			// rounding; their sum (the trace) does not.
			var sum complex128
			for _, e := range res.Eig {
				if !(cmplx.Abs(e) < 1) {
					t.Errorf("case %d %s: Eig %v not stable", k, tc.name, e)
				}
				sum += e
			}
			var bk mat.Dense
			bk.Mul(B0, res.K)
			if d := cmplx.Abs(sum - complex(mat.Trace(&acl), 0)); !(d <= 1e-9*float64(n)*(mat.Norm(A0, 2)+mat.Norm(&bk, 2))) {
				t.Errorf("case %d %s: Σ Eig = %v, trace(A − BK) = %v", k, tc.name, sum, mat.Trace(&acl))
			}
		}
	}
}

// An unstable mode no input reaches has no stabilizing solution: the
// placement fallback must not turn the pencil's ErrNoStabilizing into a
// value.
func TestDare_CheapControlUnstabilizableModeIsError(t *testing.T) {
	A0, B0, Q0, R := dareCheapUnstableCase(25)
	n, m := B0.Dims()
	A := mat.NewDense(n+1, n+1, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(A0)
	A.Set(n, n, 5)
	for i := range n {
		A.Set(i, n, 10*math.Sin(float64(i+1)))
	}
	B := mat.NewDense(n+1, m, nil)
	B.Slice(0, n, 0, m).(*mat.Dense).Copy(B0)
	Q := mat.NewDense(n+1, n+1, nil)
	Q.Slice(0, n, 0, n).(*mat.Dense).Copy(Q0)
	Q.Set(n, n, 1e14)
	for _, opts := range []*RiccatiOpts{nil, {NoScaling: true}} {
		if res, err := Dare(A, B, Q, R, opts); !errors.Is(err, ErrNoStabilizing) {
			t.Errorf("NoScaling %v: got %v, %v; want ErrNoStabilizing", opts != nil, res, err)
		}
	}
}
