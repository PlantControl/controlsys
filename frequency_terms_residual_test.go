package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// FKYKOG: the compensated pencil residual's error grows as γₖ² in its
// term count k (Dot2), not linearly; pencilResidualRow's bound must hold
// against the exact residual for long rows with heavy cancellation, at
// plain and unit-circle points.
func TestPencilResidualRowBoundExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	rat := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	type cr struct{ re, im *big.Rat }
	mul := func(a, b cr) cr {
		return cr{
			new(big.Rat).Sub(new(big.Rat).Mul(a.re, b.re), new(big.Rat).Mul(a.im, b.im)),
			new(big.Rat).Add(new(big.Rat).Mul(a.re, b.im), new(big.Rat).Mul(a.im, b.re)),
		}
	}
	cx := func(v complex128) cr { return cr{rat(real(v)), rat(imag(v))} }
	worst := 0.0
	for trial := range 400 {
		n := 40 + rng.IntN(200)
		plain := trial%2 == 0
		var pt frequencyPoint
		if plain {
			pt = frequencyPoint{p: complex(0, rng.Float64()*4), q: -1}
		} else {
			th := rng.Float64() * math.Pi
			p := complex(math.Cos(th), math.Sin(th))
			pt = frequencyPoint{p: p, q: -complex(real(p), -imag(p))}
		}
		kappa := complex(1, 0)
		if !plain {
			kappa = pt.scale()
		}
		row := make([]float64, n)
		x := make([]complex128, n)
		for k := range n {
			if rng.IntN(8) == 0 {
				continue
			}
			row[k] = (1 + rng.Float64()) * math.Ldexp(1, rng.IntN(40)-20)
			if trial%3 == 0 && rng.IntN(2) == 0 {
				row[k] = -row[k]
			}
			x[k] = complex(1+rng.Float64(), 1+rng.Float64()) * complex(math.Ldexp(1, rng.IntN(20)-10), 0)
		}
		i := rng.IntN(n)
		var ax complex128
		for k, a := range row {
			ax += complex(a, 0) * x[k]
		}
		b := real((pt.p*x[i] + pt.q*ax) / kappa)
		r, e := pencilResidualRow(x, row, i, b, pt.p, pt.q, kappa)

		exact := cx(complex(b, 0))
		exact = mul(cx(kappa), exact)
		px := mul(cx(pt.p), cx(x[i]))
		exact.re.Sub(exact.re, px.re)
		exact.im.Sub(exact.im, px.im)
		axe := cr{new(big.Rat), new(big.Rat)}
		for k, a := range row {
			axe.re.Add(axe.re, new(big.Rat).Mul(rat(a), rat(real(x[k]))))
			axe.im.Add(axe.im, new(big.Rat).Mul(rat(a), rat(imag(x[k]))))
		}
		qax := mul(cx(pt.q), axe)
		exact.re.Sub(exact.re, qax.re)
		exact.im.Sub(exact.im, qax.im)
		dre, _ := new(big.Rat).Sub(exact.re, rat(real(r))).Float64()
		dim, _ := new(big.Rat).Sub(exact.im, rat(imag(r))).Float64()
		dist := math.Abs(dre) + math.Abs(dim)
		if bound := e + termsEps*cabs1(r); !(dist <= bound) {
			t.Fatalf("trial %d (n=%d plain=%v): |exact − r| = %g above bound %g (e = %g)", trial, n, plain, dist, bound, e)
		}
		if e > 0 {
			worst = max(worst, dist/(e+termsEps*cabs1(r)))
		}
	}
	t.Logf("largest |exact − r|/bound: %.3g", worst)
}

func TestDot2ErrGuard(t *testing.T) {
	if g := dot2Err(1 << 52); !math.IsInf(g, 1) {
		t.Errorf("dot2Err(2⁵²) = %g, want +Inf", g)
	}
	if g, lin := dot2Err(200), 2*float64(200)*termsEps*termsEps; !(g > lin) {
		t.Errorf("dot2Err(200) = %g not above the linear 2kε² = %g", g, lin)
	}
}
