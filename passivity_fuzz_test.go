package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzSampledPassive (task O5FYRC): on the returned grid (the default one
// and a pole-clustered custom one), MinHermitianPart is the minimum over ω
// of λmin((H+Hᴴ)/2), H the exact rational response C(xI−A)⁻¹B+D at
// x = jω (e^{jωDt}), its Frequency attains it, and the verdict is the sign
// of that minimum against Tol. Non-square models must return
// ErrDimensionMismatch (or ErrUnstable, checked first) and decidably
// unstable ones ErrUnstable.

func FuzzSampledPassive(f *testing.F) {
	for _, s := range fzPassSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		res, err := SampledPassive(sys, nil)
		fzContract(t, s, "SampledPassive", res != nil, err)
		if s.M != s.P {
			unstable := s.Unstable() > 0 || !s.Decidable()
			if !errors.Is(err, ErrDimensionMismatch) && !(unstable && errors.Is(err, ErrUnstable)) {
				t.Fatalf("SampledPassive %dx%d: err %v, want ErrDimensionMismatch\n%s", s.P, s.M, err, s)
			}
			return
		}
		if !s.Decidable() {
			fzSkip(t, "passivity: stability undecidable")
		}
		if s.Unstable() > 0 {
			if !errors.Is(err, ErrUnstable) {
				t.Fatalf("SampledPassive unstable: err %v, want ErrUnstable\n%s", err, s)
			}
			return
		}
		fzErr(t, s, "SampledPassive", err)
		upper := 100.0
		if s.Discrete {
			upper = math.Pi / s.Dt
		}
		if len(res.Omega) != 121 || res.Omega[0] != 0 || res.Omega[120] != upper {
			t.Fatalf("default grid: %d points [%g … %g], want 121 on [0, %g]\n%s", len(res.Omega), res.Omega[0], res.Omega[len(res.Omega)-1], upper, s)
		}
		fzPassCheck(t, s, res)

		grid := fzPassGrid(s)
		res, err = SampledPassive(sys, &PassivityOptions{Omega: grid})
		fzContract(t, s, "SampledPassive(grid)", res != nil, err)
		fzErr(t, s, "SampledPassive(grid)", err)
		if !slices.Equal(res.Omega, grid) {
			t.Fatalf("Omega = %v, want the requested grid %v\n%s", res.Omega, grid, s)
		}
		fzPassCheck(t, s, res)
	})
}

func fzPassCheck(t *testing.T, s *fzSpec, res *PassivityResult) {
	t.Helper()
	best, bestW, slack := math.Inf(1), math.NaN(), 0.0
	at := map[float64][2]float64{}
	for _, w := range res.Omega {
		x := complex(0, w)
		if s.Discrete {
			x = cmplx.Exp(complex(0, w*s.Dt))
		}
		h, ok := fzExactResponse(s.N, s.M, s.P, s.A, s.B, s.C, s.D, x)
		if !ok {
			fzSkip(t, "passivity: grid point on a pole")
		}
		v, scale := fzPassMinHerm(h, s.P)
		if math.IsNaN(v) {
			fzSkip(t, "passivity: response overflow")
		}
		tol := 1e-8 * scale * math.Pow(s.CondT, 2) * math.Max(1, s.NormJ*math.Max(1, s.Dt)/fzPassDist(s, x))
		at[w] = [2]float64{v, tol}
		slack = math.Max(slack, tol)
		if v < best {
			best, bestW = v, w
		}
	}
	if math.Abs(res.MinHermitianPart-best) > slack {
		t.Fatalf("MinHermitianPart = %.17g at %g, oracle %.17g at %g (tol %g)\n%s", res.MinHermitianPart, res.Frequency, best, bestW, slack, s)
	}
	o, ok := at[res.Frequency]
	if !ok {
		t.Fatalf("Frequency %g is not a grid point\n%s", res.Frequency, s)
	}
	if math.Abs(o[0]-res.MinHermitianPart) > o[1] {
		t.Fatalf("MinHermitianPart = %.17g at %g, oracle there %.17g (tol %g)\n%s", res.MinHermitianPart, res.Frequency, o[0], o[1], s)
	}
	switch {
	case best < -res.Tolerance-slack && res.Status != PassivityViolated:
		t.Fatalf("Status %s, oracle min Hermitian part %g < −Tol\n%s", res.Status, best, s)
	case best > -res.Tolerance+slack && res.Status != PassivitySampled:
		t.Fatalf("Status %s, oracle min Hermitian part %g ≥ −Tol\n%s", res.Status, best, s)
	}
}

// fzPassMinHerm returns λmin((H+Hᴴ)/2) of the p×p H via its real symmetric
// embedding, and max |Hij| as the error scale.
func fzPassMinHerm(h []complex128, p int) (float64, float64) {
	scale := 0.0
	for _, v := range h {
		if cmplx.IsInf(v) || cmplx.IsNaN(v) {
			return math.NaN(), 0
		}
		scale = math.Max(scale, cmplx.Abs(v))
	}
	e := mat.NewSymDense(2*p, nil)
	for i := range p {
		for j := i; j < p; j++ {
			hm := (h[i*p+j] + cmplx.Conj(h[j*p+i])) / 2
			e.SetSym(i, j, real(hm))
			e.SetSym(p+i, p+j, real(hm))
			e.SetSym(i, p+j, -imag(hm))
			e.SetSym(j, p+i, imag(hm))
		}
	}
	var eig mat.EigenSym
	if !eig.Factorize(e, false) {
		return math.NaN(), 0
	}
	return slices.Min(eig.Values(nil)), scale
}

// fzPassDist is the distance from x to the nearest intended eigenvalue.
func fzPassDist(s *fzSpec, x complex128) float64 {
	d := math.Inf(1)
	for _, e := range s.Eig {
		d = math.Min(d, cmplx.Abs(x-e))
	}
	return math.Max(d, 1e-300)
}

// fzPassGrid is 0, the pole frequencies and their half-bandwidth
// neighbours, and 24 log points, capped at π/Dt.
func fzPassGrid(s *fzSpec) []float64 {
	wmax := 1e6
	if s.Discrete {
		wmax = math.Pi / s.Dt
	}
	out := []float64{0, wmax}
	for _, p := range s.S {
		w, d := math.Abs(imag(p)), math.Abs(real(p))
		out = append(out, w, w-d, w+d, cmplx.Abs(p))
	}
	for i := range 24 {
		out = append(out, math.Pow(10, -4+10*float64(i)/23))
	}
	out = slices.DeleteFunc(out, func(w float64) bool { return !(w >= 0 && w <= wmax) })
	slices.Sort(out)
	return slices.Compact(out)
}

func fzPassSeeds() []fzSeed {
	none := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10}
	mk := func(f func(*fzSeed)) fzSeed {
		x := none
		f(&x)
		return x
	}
	return []fzSeed{
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.DScal = 1, [3]float64{0}, [3]float64{1}, 6, 1 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.DScal = 2, [3]float64{0.5}, [3]float64{2.5}, 6, 2 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.DScal = 2, [3]float64{0.5}, [3]float64{3.5}, 6, 0.1 }),
		mk(func(x *fzSeed) {
			x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.DScal = 3, 2, 2, [3]float64{0, 0.3}, [3]float64{2, 1}, 5, 3
		}),
		mk(func(x *fzSeed) {
			x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.DScal, x.Shear = 4, 3, 3, [3]float64{0, 1}, [3]float64{1.5, 2.5}, 4, 1, -1
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.Mag, x.Damp, x.Gain, x.DScal = true, -1, 2, [3]float64{0.5}, [3]float64{2}, 5, 2
		}),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain = 2, [3]float64{0}, [3]float64{-2}, 6 }),
		mk(func(x *fzSeed) { x.N, x.M, x.Mag, x.Damp, x.Gain = 2, 2, [3]float64{0}, [3]float64{2}, 6 }),
	}
}
