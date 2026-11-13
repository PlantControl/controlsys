package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// zpkEvalBoundedFrexp is zpkEvalBounded with math.Frexp and math.Ldexp as
// its power-of-two scaling, the reference the fast scaling must match.
func zpkEvalBoundedFrexp(zs, ps []complex128, k float64, s complex128) (complex128, float64, bool) {
	if k == 0 {
		return 0, 0, true
	}
	acc, ex := complex(k, 0), 0
	norm := func() {
		mx := math.Max(math.Abs(real(acc)), math.Abs(imag(acc)))
		if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
			return
		}
		_, e := math.Frexp(mx)
		acc = complex(math.Ldexp(real(acc), -e), math.Ldexp(imag(acc), -e))
		ex += e
	}
	for i := 0; i < len(zs) || i < len(ps); i++ {
		if i < len(zs) {
			acc *= s - zs[i]
			norm()
		}
		if i < len(ps) {
			w := s - ps[i]
			a, b := real(w), imag(w)
			mx := math.Max(math.Abs(a), math.Abs(b))
			if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
				return 0, 0, false
			}
			_, e := math.Frexp(mx)
			a, b = math.Ldexp(a, -e), math.Ldexp(b, -e)
			q := a*a + b*b
			acc *= complex(a/q, -b/q)
			ex -= e
			norm()
		}
	}
	v := complex(math.Ldexp(real(acc), ex), math.Ldexp(imag(acc), ex))
	if cmplx.IsNaN(v) || cmplx.IsInf(v) {
		return 0, 0, false
	}
	g := gammaN(4*len(zs) + 7*len(ps))
	return v, g*cmplx.Abs(v)/(1-g) + 2*math.SmallestNonzeroFloat64, true
}

func TestFrexpExpLdexpNegMatchMath(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 11))
	xs := []float64{math.SmallestNonzeroFloat64, 0x1p-1022, 0x1p-1023, math.MaxFloat64, 1, 0.5, 0.75, 3}
	for range 20000 {
		xs = append(xs, math.Ldexp(rng.Float64()+0.5, rng.IntN(2150)-1075))
	}
	for _, x := range xs {
		if !(x > 0) || math.IsInf(x, 0) {
			continue
		}
		_, want := math.Frexp(x)
		if got := frexpExp(x); got != want {
			t.Fatalf("frexpExp(%g) = %d, math.Frexp %d", x, got, want)
		}
	}
	for range 200000 {
		x := math.Ldexp(rng.Float64()-0.5, rng.IntN(2150)-1075)
		e := rng.IntN(2200) - 1100
		if got, want := ldexpNeg(x, e), math.Ldexp(x, -e); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("ldexpNeg(%g, %d) = %g, math.Ldexp %g", x, e, got, want)
		}
	}
}

func TestZPKEvalBoundedMatchesFrexpScaling(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 13))
	val := func() float64 {
		switch rng.IntN(6) {
		case 0:
			return 0
		case 1:
			return math.Ldexp(rng.Float64()-0.5, rng.IntN(2100)-1070)
		case 2:
			return math.Ldexp(rng.Float64()-0.5, rng.IntN(40)-1080)
		case 3:
			return math.Ldexp(rng.Float64()-0.5, rng.IntN(40)+990)
		default:
			return rng.NormFloat64() * math.Pow(10, 4*rng.NormFloat64())
		}
	}
	roots := func(k int) []complex128 {
		r := make([]complex128, k)
		for i := range r {
			r[i] = complex(val(), val())
		}
		return r
	}
	same := func(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }
	for i := range 100000 {
		zs, ps := roots(rng.IntN(8)), roots(rng.IntN(8))
		s := complex(val(), val())
		if rng.IntN(5) == 0 && len(ps) > 0 {
			s = ps[0] + complex(math.Ldexp(1, -rng.IntN(1074)), 0)
		}
		k := val()
		v, b, ok := zpkEvalBounded(zs, ps, k, s)
		wv, wb, wok := zpkEvalBoundedFrexp(zs, ps, k, s)
		if ok != wok || !same(real(v), real(wv)) || !same(imag(v), imag(wv)) || !same(b, wb) {
			t.Fatalf("case %d: zpkEvalBounded(%v, %v, %g, %v) = %v, %g, %v; reference %v, %g, %v", i, zs, ps, k, s, v, b, ok, wv, wb, wok)
		}
	}
}
