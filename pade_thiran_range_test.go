package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"testing"
)

const delayBigPrec = 256

// exactPadeResponse is the order-n Padé approximant of e^{-x} at x = jw,
// P(-x)/P(x) with P(x) = sum_k (2n-k)! n! / ((2n)! k! (n-k)!) x^k, in big
// arithmetic.
func exactPadeResponse(n int, w float64) complex128 {
	fact := func(k int) *big.Int { return new(big.Int).MulRange(1, int64(max(k, 1))) }
	re, im := new(big.Float).SetPrec(delayBigPrec), new(big.Float).SetPrec(delayBigPrec)
	wk := new(big.Float).SetPrec(delayBigPrec).SetInt64(1)
	bw := new(big.Float).SetPrec(delayBigPrec).SetFloat64(w)
	for k := 0; k <= n; k++ {
		num := new(big.Int).Mul(fact(2*n-k), fact(n))
		den := new(big.Int).Mul(fact(2*n), new(big.Int).Mul(fact(k), fact(n-k)))
		q := new(big.Float).SetPrec(delayBigPrec).SetRat(new(big.Rat).SetFrac(num, den))
		term := new(big.Float).SetPrec(delayBigPrec).Mul(q, wk)
		switch k % 4 {
		case 0:
			re.Add(re, term)
		case 1:
			im.Add(im, term)
		case 2:
			re.Sub(re, term)
		case 3:
			im.Sub(im, term)
		}
		wk.Mul(wk, bw)
	}
	// conj(P)/P = conj(P)^2/|P|^2
	r2 := new(big.Float).SetPrec(delayBigPrec).Mul(re, re)
	i2 := new(big.Float).SetPrec(delayBigPrec).Mul(im, im)
	mag := new(big.Float).SetPrec(delayBigPrec).Add(r2, i2)
	hr := new(big.Float).SetPrec(delayBigPrec).Sub(r2, i2)
	hi := new(big.Float).SetPrec(delayBigPrec).Mul(re, im)
	hi.Mul(hi, big.NewFloat(-2))
	hr.Quo(hr, mag)
	hi.Quo(hi, mag)
	fr, _ := hr.Float64()
	fi, _ := hi.Float64()
	return complex(fr, fi)
}

func TestPadeDelayResponseAcrossTauScales(t *testing.T) {
	taus := []float64{1e-280, 1e-100, 1e-25, 1e-3, 1, 7.3, 1e20, 1e23, 1e24, 1e31, 1e100, 1e150}
	for n := 1; n <= 10; n++ {
		for _, tau := range taus {
			sys, err := PadeDelay(tau, n)
			if err != nil {
				t.Errorf("PadeDelay(%g, %d): %v", tau, n, err)
				continue
			}
			for _, w := range []float64{0.1, 1, 3, 30} {
				h, err := sys.EvalFr(complex(0, w/tau))
				if err != nil {
					t.Errorf("PadeDelay(%g, %d).EvalFr: %v", tau, n, err)
					continue
				}
				want := exactPadeResponse(n, w)
				if d := cmplx.Abs(h[0][0] - want); !(d <= 1e-12) {
					t.Errorf("PadeDelay(%g, %d) at w·tau=%g: got %v, want %v (|err| %g)", tau, n, w, h[0][0], want, d)
				}
			}
		}
	}
	sys, err := PadeDelay(1e155, 2)
	if err != nil {
		t.Fatalf("PadeDelay(1e155, 2): %v", err)
	}
	h, err := sys.EvalFr(complex(0, 1/1e155))
	if err != nil {
		t.Fatalf("PadeDelay(1e155, 2).EvalFr: %v", err)
	}
	if want := exactPadeResponse(2, 1); !(cmplx.Abs(h[0][0]-want) <= 1e-12) {
		t.Errorf("PadeDelay(1e155, 2): got %v, want %v", h[0][0], want)
	}
}

func TestPadeDelayCoefficientRangeErrors(t *testing.T) {
	for _, c := range []struct {
		tau   float64
		order int
	}{{1e-300, 10}, {5e-324, 1}, {math.MaxFloat64, 2}} {
		sys, err := PadeDelay(c.tau, c.order)
		if !errors.Is(err, ErrOverflow) || sys != nil {
			t.Errorf("PadeDelay(%g, %d) = %v, %v; want nil, ErrOverflow", c.tau, c.order, sys, err)
		}
	}
}

// exactThiranCoeffs evaluates a_k = (-1)^k C(N,k) prod_{n=0}^{N}
// (D-N+n)/(D-N+k+n) directly in big arithmetic.
func exactThiranCoeffs(D float64, N int) []*big.Float {
	bD := new(big.Float).SetPrec(delayBigPrec).SetFloat64(D)
	out := make([]*big.Float, N+1)
	for k := 0; k <= N; k++ {
		binom := new(big.Float).SetPrec(delayBigPrec).SetInt(new(big.Int).Binomial(int64(N), int64(k)))
		prod := new(big.Float).SetPrec(delayBigPrec).SetInt64(1)
		for n := 0; n <= N; n++ {
			num := new(big.Float).SetPrec(delayBigPrec).SetInt64(int64(n - N))
			num.Add(num, bD)
			den := new(big.Float).SetPrec(delayBigPrec).SetInt64(int64(k + n - N))
			den.Add(den, bD)
			prod.Mul(prod, num.Quo(num, den))
		}
		if k%2 != 0 {
			binom.Neg(binom)
		}
		out[k] = binom.Mul(binom, prod)
	}
	return out
}

func TestThiranCoeffsMatchClosedForm(t *testing.T) {
	for _, c := range []struct {
		D float64
		N int
	}{{0.3, 1}, {math.Pi, 4}, {49.7, 50}, {1029.5, 1030}} {
		got := thiranCoeffs(c.D, c.N)
		want := exactThiranCoeffs(c.D, c.N)
		for k, w := range want {
			wf, _ := w.Float64()
			tol := 1e-11 * math.Abs(wf)
			if math.Abs(wf) < 1e-290 {
				tol = 1e-300
			}
			if d := math.Abs(got[k] - wf); !(d <= tol) {
				t.Errorf("thiranCoeffs(%g, %d)[%d] = %g, want %g", c.D, c.N, k, got[k], wf)
				break
			}
		}
	}
}

func TestThiranDelayHighOrderFinite(t *testing.T) {
	if testing.Short() {
		t.Skip("1031-state frequency evaluation")
	}
	for _, D := range []float64{1029.5, 1030.5} {
		sys, err := ThiranDelay(D, 1)
		if err != nil {
			t.Fatalf("ThiranDelay(%g, 1): %v", D, err)
		}
		for _, v := range sys.A.RawMatrix().Data {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("ThiranDelay(%g, 1): non-finite A entry %v", D, v)
			}
		}
		for _, w := range []float64{1e-4, 0.5, 2} {
			h, err := sys.EvalFr(cmplx.Exp(complex(0, w)))
			if err != nil {
				t.Fatalf("ThiranDelay(%g, 1).EvalFr: %v", D, err)
			}
			if m := cmplx.Abs(h[0][0]); math.Abs(m-1) > 1e-8 {
				t.Errorf("ThiranDelay(%g, 1) |H(e^{j%g})| = %v, want 1", D, w, m)
			}
			if w == 1e-4 {
				if ph := cmplx.Phase(h[0][0]); math.Abs(ph+D*w) > 1e-8 {
					t.Errorf("ThiranDelay(%g, 1) phase at w=%g = %v, want %v", D, w, ph, -D*w)
				}
			}
		}
	}
}
