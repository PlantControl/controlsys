package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// kalmanCase is a model in Kalman form x = [co, cō, c̄o] (sizes r, u1, u2),
// optionally under a random orthogonal similarity and I/O scaling. The core
// co is stable and the hidden blocks are unstable, so a kept hidden state
// shows up as an unstable pole.
type kalmanCase struct {
	sys          *System
	r, u1, u2    int
	rotated      bool
	kB, kC       int
	continuousDt bool
}

func kalmanShiftedBlock(rng *rand.Rand, k int, dt float64, unstable bool, lo, hi float64) *mat.Dense {
	g := mat.NewDense(k, k, nil)
	for i := range k {
		for j := range k {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var e mat.Eigen
	e.Factorize(g, mat.EigenNone)
	vals := e.Values(nil)
	target := lo + (hi-lo)*rng.Float64()
	if dt == 0 {
		mx := math.Inf(-1)
		for _, v := range vals {
			mx = math.Max(mx, real(v))
		}
		mn := math.Inf(1)
		for _, v := range vals {
			mn = math.Min(mn, real(v))
		}
		shift := -target - mx
		if unstable {
			shift = target - mn
		}
		for i := range k {
			g.Set(i, i, g.At(i, i)+shift)
		}
		return g
	}
	mx, mn := 0.0, math.Inf(1)
	for _, v := range vals {
		mx = math.Max(mx, cmplx.Abs(v))
		mn = math.Min(mn, cmplx.Abs(v))
	}
	if !unstable {
		g.Scale(target/mx, g)
	} else {
		g.Scale(target/math.Max(mn, 1e-3), g)
	}
	return g
}

func newKalmanCase(rng *rand.Rand) kalmanCase {
	return newKalmanCaseScaled(rng, 0)
}

// newKalmanCaseScaled also applies a diagonal state similarity with
// entries 10^U(-diagDecades, diagDecades) after the rotation.
func newKalmanCaseScaled(rng *rand.Rand, diagDecades float64) kalmanCase {
	dt := 0.0
	if rng.IntN(2) == 1 {
		dt = 0.1
	}
	r := 1 + rng.IntN(4)
	u1 := rng.IntN(3)
	u2 := rng.IntN(3)
	m := 1 + rng.IntN(2)
	p := 1 + rng.IntN(2)
	n := r + u1 + u2
	stableLo, stableHi := 0.2, 2.0
	unstLo, unstHi := 0.2, 2.0
	if dt != 0 {
		stableLo, stableHi = 0.3, 0.9
		unstLo, unstHi = 1.2, 2.0
	}
	a := mat.NewDense(n, n, nil)
	setBlock(a, 0, 0, kalmanShiftedBlock(rng, r, dt, false, stableLo, stableHi))
	if u1 > 0 {
		setBlock(a, r, r, kalmanShiftedBlock(rng, u1, dt, true, unstLo, unstHi))
	}
	if u2 > 0 {
		setBlock(a, r+u1, r+u1, kalmanShiftedBlock(rng, u2, dt, true, unstLo, unstHi))
	}
	gauss := func(i0, i1, j0, j1 int) {
		for i := i0; i < i1; i++ {
			for j := j0; j < j1; j++ {
				a.Set(i, j, rng.NormFloat64())
			}
		}
	}
	gauss(r, r+u1, 0, r)
	gauss(0, r, r+u1, n)
	gauss(r, r+u1, r+u1, n)
	b := mat.NewDense(n, m, nil)
	for i := range r + u1 {
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
	}
	c := mat.NewDense(p, n, nil)
	for i := range p {
		for j := range r {
			c.Set(i, j, rng.NormFloat64())
		}
		for j := r + u1; j < n; j++ {
			c.Set(i, j, rng.NormFloat64())
		}
	}
	rotated := rng.IntN(4) != 0
	if rotated {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		var qr mat.QR
		qr.Factorize(g)
		var q mat.Dense
		qr.QTo(&q)
		var t1, t2 mat.Dense
		t1.Mul(q.T(), a)
		t2.Mul(&t1, &q)
		a = mat.DenseCopyOf(&t2)
		var tb, tc mat.Dense
		tb.Mul(q.T(), b)
		tc.Mul(c, &q)
		b, c = mat.DenseCopyOf(&tb), mat.DenseCopyOf(&tc)
	}
	if diagDecades > 0 {
		for i := range n {
			d := math.Pow(10, diagDecades*(2*rng.Float64()-1))
			for j := range n {
				a.Set(i, j, a.At(i, j)/d)
				a.Set(j, i, a.At(j, i)*d)
			}
			for j := range m {
				b.Set(i, j, b.At(i, j)/d)
			}
			for j := range p {
				c.Set(j, i, c.At(j, i)*d)
			}
		}
	}
	exps := []int{0, 0, 2, -2, 4, -4, 8, -8}
	kB, kC := exps[rng.IntN(len(exps))], exps[rng.IntN(len(exps))]
	b.Scale(math.Pow(10, float64(kB)), b)
	c.Scale(math.Pow(10, float64(kC)), c)
	sys, err := New(a, b, c, mat.NewDense(p, m, nil), dt)
	if err != nil {
		panic(err)
	}
	return kalmanCase{sys: sys, r: r, u1: u1, u2: u2, rotated: rotated, kB: kB, kC: kC, continuousDt: dt == 0}
}

func kalmanUnstable(vals []complex128, continuous bool) bool {
	for _, v := range vals {
		if continuous && real(v) > 1e-6 {
			return true
		}
		if !continuous && cmplx.Abs(v) > 1+1e-6 {
			return true
		}
	}
	return false
}

func kalmanRespErr(sys *System, eval func(complex128) ([][]complex128, error)) float64 {
	worst := 0.0
	for _, w := range []float64{0.3, 1.7} {
		z := complex(0, w)
		if sys.Dt != 0 {
			z = cmplx.Exp(complex(0, w*sys.Dt*3))
		}
		want, err := sys.EvalFr(z)
		if err != nil {
			return math.Inf(1)
		}
		got, err := eval(z)
		if err != nil {
			return math.Inf(1)
		}
		var num, den float64
		for i := range want {
			for j := range want[i] {
				d := cmplx.Abs(got[i][j] - want[i][j])
				num += d * d
				den += cmplx.Abs(want[i][j]) * cmplx.Abs(want[i][j])
			}
		}
		if den > 0 {
			worst = math.Max(worst, math.Sqrt(num/den))
		}
	}
	return worst
}

type kalmanTally struct {
	cases, raOver, raUnder, raUnst, raResp int
	cfWrong, ofWrong                       int
	tfOver, tfUnder, tfUnst, tfResp        int
	failed                                 []int
	lines                                  []string
	plain, plainBad                        int
}

func (k *kalmanTally) run(id int, kc kalmanCase) {
	k.cases++
	sys := kc.sys
	bad := false
	plain := !kc.rotated && kc.kB == 0 && kc.kC == 0
	plainBad := false
	if plain {
		k.plain++
		defer func() {
			if plainBad {
				k.plainBad++
			}
		}()
	}
	line := strconv.Itoa(id) + " kB=" + strconv.Itoa(kc.kB) + " kC=" + strconv.Itoa(kc.kC) + " rot=" + strconv.FormatBool(kc.rotated)
	defer func() { k.lines = append(k.lines, line) }()
	if red, err := sys.Reduce(nil); err == nil {
		line += " ra=" + strconv.Itoa(red.Order-kc.r)
		if red.Order != kc.r {
			plainBad = true
		}
		if red.Order > kc.r {
			k.raOver++
			bad = true
		}
		if red.Order < kc.r {
			k.raUnder++
			bad = true
		}
		if red.Order > 0 {
			var e mat.Eigen
			if e.Factorize(red.Sys.A, mat.EigenNone) && kalmanUnstable(e.Values(nil), kc.continuousDt) {
				k.raUnst++
			}
		}
		rerr := kalmanRespErr(sys, red.Sys.EvalFr)
		line += " rr=" + strconv.FormatFloat(rerr, 'g', 3, 64)
		if rerr > 1e-6 {
			k.raResp++
			bad = true
		}
	}
	if cf, err := CtrbF(sys.A, sys.B, sys.C, 0); err == nil {
		s := 0
		for _, v := range cf.K {
			s += v
		}
		if s != kc.r+kc.u1 {
			k.cfWrong++
			plainBad = true
		}
		line += " cf=" + strconv.Itoa(s-kc.r-kc.u1)
	}
	if of, err := ObsvF(sys.A, sys.B, sys.C, 0); err == nil {
		s := 0
		for _, v := range of.K {
			s += v
		}
		if s != kc.r+kc.u2 {
			k.ofWrong++
			plainBad = true
		}
		line += " of=" + strconv.Itoa(s-kc.r-kc.u2)
	}
	if tf, err := sys.TransferFunction(nil); err == nil {
		over, under, unst := false, false, false
		for i, d := range tf.RowDegrees {
			if d > kc.r {
				over = true
			}
			if d < kc.r {
				under = true
			}
			if roots, err := Poly(tf.TF.Den[i]).Roots(); err == nil && kalmanUnstable(roots, kc.continuousDt) {
				unst = true
			}
		}
		line += " tf=" + strconv.FormatBool(over) + "," + strconv.FormatBool(under) + "," + strconv.FormatBool(unst)
		if over {
			k.tfOver++
		}
		if under {
			k.tfUnder++
		}
		if unst {
			k.tfUnst++
		}
		if kalmanRespErr(sys, tf.TF.Eval) > 1e-6 {
			k.tfResp++
		}
	}
	if bad {
		k.failed = append(k.failed, id)
	}
}

// TestKalmanStructuredSweep checks Reduce, CtrbF, ObsvF and TransferFunction
// against the structural orders of seeded Kalman-form models. Env:
// KALMAN_SWEEP_N (cases), KALMAN_SWEEP_DIAG (diagonal state scaling
// decades), KALMAN_SWEEP_OUT (per-case results file for base comparisons).
func TestKalmanStructuredSweep(t *testing.T) {
	n := 300
	if s := os.Getenv("KALMAN_SWEEP_N"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	var tally kalmanTally
	decades := 0.0
	if s := os.Getenv("KALMAN_SWEEP_DIAG"); s != "" {
		decades, _ = strconv.ParseFloat(s, 64)
	}
	for id := range n {
		rng := rand.New(rand.NewPCG(uint64(id), 0x9e3779b97f4a7c15))
		tally.run(id, newKalmanCaseScaled(rng, decades))
	}
	t.Logf("cases %d | Reduce over %d under %d unstable %d resp %d | CtrbF %d ObsvF %d | TF over %d under %d unstable %d resp %d",
		tally.cases, tally.raOver, tally.raUnder, tally.raUnst, tally.raResp, tally.cfWrong, tally.ofWrong,
		tally.tfOver, tally.tfUnder, tally.tfUnst, tally.tfResp)
	t.Logf("plain (unrotated, unit I/O scale) %d, wrong %d", tally.plain, tally.plainBad)
	if out := os.Getenv("KALMAN_SWEEP_OUT"); out != "" {
		if err := os.WriteFile(out, []byte(strings.Join(tally.lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if tally.raUnder+tally.raResp+tally.tfUnder+tally.tfResp > 0 {
		t.Errorf("dropped needed states or wrong response: Reduce under %d resp %d, TF under %d resp %d",
			tally.raUnder, tally.raResp, tally.tfUnder, tally.tfResp)
	}
	// Base 0c02feb over-counted 59 Reduce, 20 CtrbF, 24 ObsvF and 67 TF of
	// the default 300 by keeping rounding noise as rank.
	if n == 300 && decades == 0 && (tally.raOver > 50 || tally.cfWrong > 17 || tally.ofWrong > 20 || tally.tfOver > 60) {
		t.Errorf("noise counted as rank: Reduce over %d (≤50), CtrbF %d (≤17), ObsvF %d (≤20), TF over %d (≤60)",
			tally.raOver, tally.cfWrong, tally.ofWrong, tally.tfOver)
	}
	if len(tally.failed) > 0 && os.Getenv("KALMAN_SWEEP_N") != "" {
		t.Logf("failed ids %v", tally.failed[:min(len(tally.failed), 60)])
	}
}
