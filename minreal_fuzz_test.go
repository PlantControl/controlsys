package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzMinimalRealization: Reduce in every mode (and MinimalRealization)
// returns at most n states, keeps D and the delays, and its exact rational
// response equals the original's within the first-order bound for
// orthogonal staircase backward errors of size K·eps·‖[A B; C 0]‖
// (staircase truncation at the default tolerance is of the same size).
// Dropping a state of a clearly minimal model breaks the response at
// points near its pole, so the response check also bounds the order.
func FuzzMinimalRealization(f *testing.F) {
	for _, s := range fzConvSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m, p := s.N, s.M, s.P
		sys := s.System(t, fzSysOpts{Delays: true})
		mr, err := sys.MinimalRealization()
		fzContract(t, s, "MinimalRealization", mr != nil, err)
		fzErr(t, s, "MinimalRealization", err)
		nA, nB, nC := fzFrob(s.A), fzFrob(s.B), fzFrob(s.C)
		const eps = 0x1p-52
		kk := 256 * float64((n+1)*(n+1))
		dA := kk * eps * math.Max(nA, math.Max(nB, nC))
		dB := kk * eps * math.Max(nA, nB)
		dC := kk * eps * math.Max(nA, nC)
		pts := fzConvPoints(s)
		if s.Discrete {
			pts = append(pts, 1)
		} else {
			pts = append(pts, 0)
		}
		want := make([][]complex128, len(pts))
		for k, x := range pts {
			if g, ok := fzExactResponse(n, m, p, s.A, s.B, s.C, s.D, x); ok {
				want[k] = g
			}
		}
		for _, mode := range []struct {
			name string
			opts *ReduceOpts
		}{
			{"MinimalRealization", nil},
			{"ReduceUncontrollable", &ReduceOpts{Mode: ReduceUncontrollable}},
			{"ReduceUnobservable", &ReduceOpts{Mode: ReduceUnobservable}},
			{"Reduce(Equalize)", &ReduceOpts{Equalize: true}},
		} {
			r := mr
			if mode.opts != nil {
				r, err = sys.Reduce(mode.opts)
				fzContract(t, s, mode.name, r != nil, err)
				fzErr(t, s, mode.name, err)
			}
			rn, rm, rp := r.Sys.Dims()
			if rn != r.Order || r.Order > n || rm != m || rp != p || r.Sys.Dt != s.Dt {
				t.Fatalf("%s: order %d dims %d×%d×%d dt=%g, original n=%d\n%s", mode.name, r.Order, rn, rm, rp, r.Sys.Dt, n, s)
			}
			a, b, c, d := fzRaw(r.Sys)
			if !slices.Equal(d, s.D) {
				t.Fatalf("%s: D = %v, want %v\n%s", mode.name, d, s.D, s)
			}
			fzSameDelays(t, s, mode.name, r.Sys, sys)
			for k, x := range pts {
				if want[k] == nil {
					continue
				}
				rinv := fzResolventNorm(s, x)
				if !(dA*rinv < 0.1) {
					continue
				}
				tol := (dC*rinv*nB + nC*rinv*dB + nC*rinv*dA*rinv*nB) / (1 - dA*rinv)
				g, ok := fzExactResponse(rn, m, p, a, b, c, d, x)
				if !ok {
					t.Fatalf("%s: reduced model has an exact pole at %v, original does not\n%s", mode.name, x, s)
				}
				for e := range g {
					if diff := cmplx.Abs(g[e] - want[k][e]); !(diff <= tol) {
						t.Fatalf("%s (order %d): response(%v)[%d] = %v, original %v (|Δ| %g, tol %g)\n%s", mode.name, r.Order, x, e, g[e], want[k][e], diff, tol, s)
					}
				}
			}
		}
	})
}

// fzResolventNorm returns ‖(xI − A)⁻¹‖₂ (+Inf when numerically singular).
func fzResolventNorm(s *fzSpec, x complex128) float64 {
	n := s.N
	if n == 0 {
		return 0
	}
	g := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			g[i*n+j] = complex(-s.A[i*n+j], 0)
		}
		g[i*n+i] += x
	}
	sv := fzSingularValues(g, n, n)
	return 1 / sv[n-1]
}

func fzSameDelays(t *testing.T, s *fzSpec, op string, got, want *System) {
	t.Helper()
	if !slices.Equal(got.InputDelay, want.InputDelay) || !slices.Equal(got.OutputDelay, want.OutputDelay) {
		t.Fatalf("%s: delays in=%v out=%v, want %v %v\n%s", op, got.InputDelay, got.OutputDelay, want.InputDelay, want.OutputDelay, s)
	}
	if (got.Delay == nil) != (want.Delay == nil) {
		t.Fatalf("%s: IODelay presence changed\n%s", op, s)
	}
	if want.Delay != nil {
		for i := range s.P {
			for j := range s.M {
				if got.Delay.At(i, j) != want.Delay.At(i, j) {
					t.Fatalf("%s: IODelay[%d,%d] = %g, want %g\n%s", op, i, j, got.Delay.At(i, j), want.Delay.At(i, j), s)
				}
			}
		}
	}
}
