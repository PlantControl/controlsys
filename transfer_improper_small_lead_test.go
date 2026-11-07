package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

func bigHornerC(p []float64, s complex128) (re, im *big.Float) {
	const prec = 256
	re, im = new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)
	sr, si := new(big.Float).SetPrec(prec).SetFloat64(real(s)), new(big.Float).SetPrec(prec).SetFloat64(imag(s))
	t1, t2 := new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)
	for _, c := range p {
		nr := new(big.Float).SetPrec(prec).Sub(t1.Mul(re, sr), t2.Mul(im, si))
		ni := new(big.Float).SetPrec(prec).Add(t1.Mul(re, si), t2.Mul(im, sr))
		re, im = nr.Add(nr, new(big.Float).SetFloat64(c)), ni
	}
	return re, im
}

func bigRatioC(num, den []float64, s complex128) complex128 {
	nr, ni := bigHornerC(num, s)
	dr, di := bigHornerC(den, s)
	q := new(big.Float).SetPrec(256)
	mag := new(big.Float).SetPrec(256).Add(q.Mul(dr, dr), new(big.Float).SetPrec(256).Mul(di, di))
	re := new(big.Float).SetPrec(256).Add(new(big.Float).SetPrec(256).Mul(nr, dr), new(big.Float).SetPrec(256).Mul(ni, di))
	im := new(big.Float).SetPrec(256).Sub(new(big.Float).SetPrec(256).Mul(ni, dr), new(big.Float).SetPrec(256).Mul(nr, di))
	re.Quo(re, mag)
	im.Quo(im, mag)
	fr, _ := re.Float64()
	fi, _ := im.Float64()
	return complex(fr, fi)
}

func polyEvalCond(p []float64, s complex128) float64 {
	abs := 0.0
	for k, c := range p {
		abs += math.Abs(c) * math.Pow(cmplx.Abs(s), float64(len(p)-1-k))
	}
	return abs / cmplx.Abs(hornerC(p, s))
}

// RBVVVP: an improper channel whose denominator has a small leading
// coefficient has a quotient with coefficients ~lead^-(q+1); realizing it as
// D plus a chain beside the remainder cancelled ~1e12 to O(1) at moderate
// |s|. The pencil realization must keep EvalFr within a modest multiple of
// cond(num/den)·eps of the exact big.Float value.
func TestTransferFuncStateSpace_ImproperSmallLeadNoCancellation(t *testing.T) {
	const condMax = 1e6
	// Observed max err/(cond·eps) is 1.6: EvalFr's pencil solve on an
	// O(1)-scaled (N+1)-state pencil is Horner-level backward stable; 64
	// leaves headroom for LU growth. The quotient realization gave 5e11.
	const k = 64
	skipped, total := 0, 0
	points := []complex128{0.37 + 1.1i, 0.01i, 1e-3 + 0.02i, 3i, -2.5, 1e3i, 1e5i, -40 + 7e3i}
	check := func(label string, tf *TransferFunc) float64 {
		t.Helper()
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		worst := 0.0
		for _, s := range points {
			got, err := res.Sys.EvalFr(s)
			if err != nil {
				t.Fatalf("%s: EvalFr(%v): %v", label, s, err)
			}
			for i := range tf.Num {
				for j := range tf.Num[i] {
					num, den := tf.Num[i][j], tf.Den[i]
					total++
					cond := polyEvalCond(num, s) + polyEvalCond(den, s)
					if !(cond < condMax) {
						skipped++
						continue
					}
					want := bigRatioC(num, den, s)
					ratio := cmplx.Abs(got[i][j]-want) / (cmplx.Abs(want) * cond * eps())
					worst = max(worst, ratio)
					if !(ratio <= k) {
						t.Fatalf("%s H[%d][%d](%v) = %v, want %v: err/(cond·eps) = %.3g (cond %.3g)", label, i, j, s, got[i][j], want, ratio, cond)
					}
				}
			}
		}
		return worst
	}
	repro := &TransferFunc{
		Num: [][][]float64{{{0.9362843810729338, -0.8725836744109265, 1.3242523804032609, -0.6149587035413703, 0.2467593523583006}}},
		Den: [][]float64{{0.0010299449048600806, 1.2992484844637944}},
	}
	worst := check("repro", repro)
	rng := rand.New(rand.NewPCG(0x4bd5, 0x0e1))
	coef := func() float64 {
		c := rng.Float64()*2 - 1
		if math.Abs(c) < 0.1 {
			c = math.Copysign(0.1, c)
		}
		return c
	}
	poly := func(lead float64, deg int) []float64 {
		c := []float64{lead}
		for range deg {
			c = append(c, coef())
		}
		return c
	}
	leads := []float64{1e-2, 1e-3, 1e-4}
	for trial := range 60 {
		lead := leads[trial%3] * (1 + rng.Float64())
		if rng.IntN(2) == 0 {
			lead = -lead
		}
		d := 1 + rng.IntN(2)
		q := 1 + (trial/3)%3
		tf := &TransferFunc{Num: [][][]float64{{poly(coef(), d+q)}}, Den: [][]float64{poly(lead, d)}}
		worst = max(worst, check("siso", tf))
	}
	for trial := range 12 {
		lead := leads[trial%3]
		tf := &TransferFunc{
			Num: [][][]float64{
				{poly(coef(), 1+1+trial%3), poly(coef(), 1)},
				{poly(coef(), 2), poly(coef(), 2+1+(trial+1)%3)},
			},
			Den: [][]float64{poly(lead, 1), poly(-lead, 2)},
		}
		worst = max(worst, check("mimo", tf))
	}
	t.Logf("max err/(cond·eps) = %.3g; %d of %d points skipped as ill-conditioned", worst, skipped, total)
}
