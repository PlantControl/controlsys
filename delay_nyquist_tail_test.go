package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// companionDelayQuasiLoop is k·(-31.29)(s+0.09685)/(s(s²+14.84s+367.9)(s²+7.963s+36.18))·e^{-0.5042s}
// in controllable-canonical form: ‖A‖₂ ≈ 1.4e4 against max|pole| ≈ 19.2.
func companionDelayQuasiLoop(k float64) quasiLoop {
	den := polyMul([]float64{1, 0}, polyMul([]float64{1, 14.84, 367.9}, []float64{1, 7.963, 36.18}))
	num := polyScale([]float64{-31.29, -31.29 * 0.09685}, k)
	return quasiLoop{name: "companion", num: num, den: den, tau: 0.5042, box: 60, wmax: 60}
}

// The given-realization tail bound is +Inf below ‖A‖₂ ≈ 1.4e4, which put
// ~18k points on the linear delay grid of a loop with |L| < 0.3 beyond
// ~50 rad/s. The balanced realization bounds it near the pole magnitudes.
// Oracles: RHP roots of χ by the argument principle, dense peak sweep.
func TestDelayLoopCompanionTailUsesBalancedRealization(t *testing.T) {
	for _, k := range []float64{-0.3, -1, 1} {
		q := companionDelayQuasiLoop(k)
		sys := q.system(t)
		if a := spectralNorm(sys.A); a < 1e4 {
			t.Fatalf("‖A‖₂ = %g, want companion form > 1e4", a)
		}
		want := oracleRHPRoots(t, q.chi, q.box)
		l, err := delayLoopFromSystem(sys, "delay loop")
		if err != nil {
			t.Fatal(err)
		}
		if b := l.tail(100); !(b < 0.01) {
			t.Fatalf("k=%g: tail(100) = %g, want < 0.01", k, b)
		}
		n := 0
		at := l.at
		l.at = func(w float64) complex128 { n++; return at(w) }
		shifts := []float64{0, -0.5}
		stable, peaks, err := l.sensitivityPeaks(shifts...)
		if err != nil {
			t.Fatalf("k=%g: %v", k, err)
		}
		if stable != (want == 0) {
			t.Fatalf("k=%g: stable=%v, oracle roots %d", k, stable, want)
		}
		if n > 5000 {
			t.Errorf("k=%g: %d evaluations, want <= 5000", k, n)
		}
		if !stable {
			continue
		}
		Lp := func(w float64) complex128 { return q.L(math.Max(w, 1e-9)) }
		for i, c := range shifts {
			peak, _ := oraclePeakShifted(Lp, q.wmax, c)
			if math.Abs(peaks[i].peak-peak) > 1e-7*peak {
				t.Errorf("k=%g shift %g: peak=%.12g, oracle %.12g", k, c, peaks[i].peak, peak)
			}
		}
	}
}

// The internal-delay stability test of Feedback(companion loop, 1) bounds
// H22 = −L in balanced coordinates too: ~32k grid points with the given
// ‖A‖₂ ≈ 1.4e4. Oracle: RHP roots of χ by the argument principle.
func TestInternalDelayStabilityCompanionTail(t *testing.T) {
	one, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []float64{-0.3, 1} {
		q := companionDelayQuasiLoop(k)
		cl, err := Feedback(q.system(t), one, -1)
		if err != nil {
			t.Fatal(err)
		}
		want := oracleRHPRoots(t, q.chi, q.box)
		e, err := newDelayLFT(cl)
		if err != nil {
			t.Fatal(err)
		}
		l, res, err := e.stability()
		if err != nil {
			t.Fatalf("k=%g: %v", k, err)
		}
		for _, w := range []float64{45, 60, 100, 1e3} {
			bound := l.tail(w)
			for j := 0; j <= 64; j++ {
				s := cmplx.Rect(w, math.Pi/2*(1-float64(j)/32))
				if g := cmplx.Abs(polyEval(q.num, s) / polyEval(q.den, s)); g > bound*(1+1e-9) {
					t.Fatalf("k=%g: |H22(%v)| = %g > tail %g", k, s, g, bound)
				}
			}
		}
		if res.stable != (want == 0) {
			t.Fatalf("k=%g: stable=%v, oracle roots %d", k, res.stable, want)
		}
		if len(res.w) > 3000 {
			t.Errorf("k=%g: %d grid points, want <= 3000", k, len(res.w))
		}
		stable, err := internalDelayStable(cl)
		if err != nil || stable != (want == 0) {
			t.Errorf("k=%g: internalDelayStable=%v err=%v, oracle roots %d", k, stable, err, want)
		}
	}
}

// rationalTailBound must bound |L(s)| on |s| = w, Re s >= 0 for a badly
// scaled realization with two internal delays (H22 nilpotent), and the
// jointly balanced realization must make it finite well below the given
// ‖A‖₂. Oracle: L(s) from complex solves with sI−A closed over
// Δ = diag(e^{-sτ}).
func TestRationalTailBoundBalancedLFTBoundsArc(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 17))
	for trial := range 40 {
		n := 2 + trial%4
		a := make([]float64, n*n)
		for i := range a {
			a[i] = 2 * (rng.Float64() - 0.5)
		}
		for i := range n {
			a[i*n+i] -= 1 + rng.Float64()
		}
		scale := make([]float64, n)
		for i := range n {
			scale[i] = math.Pow(10, 6*float64(i)/float64(n-1)-3) * (1 + rng.Float64())
		}
		for i := range n {
			for j := range n {
				a[i*n+j] *= scale[i] / scale[j]
			}
		}
		rnd := func(r, c int, rowScale, colScale bool) *mat.Dense {
			m := mat.NewDense(r, c, nil)
			for i := range r {
				for j := range c {
					v := 2*rng.Float64() - 1
					if rowScale {
						v *= scale[i]
					}
					if colScale {
						v /= scale[j]
					}
					m.Set(i, j, v)
				}
			}
			return m
		}
		B, C := rnd(n, 1, true, false), rnd(1, n, false, true)
		B2, C2 := rnd(n, 2, true, false), rnd(2, n, false, true)
		for i := range n {
			B2.Set(i, 1, 0)
			C2.Set(0, i, 0)
		}
		D := mat.NewDense(1, 1, []float64{0.2 * (2*rng.Float64() - 1)})
		D12 := mat.NewDense(1, 2, []float64{0.3, -0.2})
		D21 := mat.NewDense(2, 1, []float64{0.4, 0.1})
		D22 := mat.NewDense(2, 2, []float64{0, 0, 0.5, 0})
		sys, err := New(mat.NewDense(n, n, a), B, C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		tau := []float64{0.3, 0.7}
		if err := sys.SetInternalDelay(tau, B2, C2, D12, D21, D22); err != nil {
			t.Fatal(err)
		}
		r, err := newRationalTailBound(sys)
		if err != nil {
			t.Fatal(err)
		}
		given, bal := r.reals[0].normA, r.reals[1].normA
		if bal > given*(1+1e-12) {
			t.Fatalf("trial %d: balanced ‖A‖₂ %g > given %g", trial, bal, given)
		}
		if !(given > 10*bal) {
			t.Fatalf("trial %d: given ‖A‖₂ %g not ≫ balanced %g", trial, given, bal)
		}
		for _, f := range []float64{1.01, 1.3, 2, 5, 30, 1e3} {
			w := f * bal
			bound := r.at(w)
			if math.IsInf(bound, 1) {
				t.Fatalf("trial %d: bound +Inf at w = %g > balanced ‖A‖₂", trial, w)
			}
			for k := 0; k <= 64; k++ {
				s := cmplx.Rect(w, math.Pi/2*(1-float64(k)/32))
				if g := cmplx.Abs(lftResponseAt(sys, s)); g > bound*(1+1e-9) {
					t.Fatalf("trial %d: |L(%v)| = %g > bound %g", trial, s, g, bound)
				}
			}
		}
	}
}

// lftResponseAt evaluates the SISO LFT D + C X B with (sI−A)X = [B B2],
// closed over the internal delays.
func lftResponseAt(sys *System, s complex128) complex128 {
	n, _, _ := sys.Dims()
	lft := sys.LFT
	nd := len(lft.Tau)
	k := 1 + nd
	rows := make([][]complex128, n)
	for i := range n {
		rows[i] = make([]complex128, n+k)
		for j := range n {
			rows[i][j] = complex(-sys.A.At(i, j), 0)
		}
		rows[i][i] += s
		rows[i][n] = complex(sys.B.At(i, 0), 0)
		for j := range nd {
			rows[i][n+1+j] = complex(lft.B2.At(i, j), 0)
		}
	}
	for c := range n {
		p := c
		for i := c + 1; i < n; i++ {
			if cmplx.Abs(rows[i][c]) > cmplx.Abs(rows[p][c]) {
				p = i
			}
		}
		rows[c], rows[p] = rows[p], rows[c]
		for i := c + 1; i < n; i++ {
			f := rows[i][c] / rows[c][c]
			for j := c; j < n+k; j++ {
				rows[i][j] -= f * rows[c][j]
			}
		}
	}
	x := make([][]complex128, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = make([]complex128, k)
		for j := range k {
			v := rows[i][n+j]
			for l := i + 1; l < n; l++ {
				v -= rows[i][l] * x[l][j]
			}
			x[i][j] = v / rows[i][i]
		}
	}
	cRow := func(r int) func(int) float64 {
		if r == 0 {
			return func(i int) float64 { return sys.C.At(0, i) }
		}
		return func(i int) float64 { return lft.C2.At(r-1, i) }
	}
	dAt := func(r, c int) float64 {
		switch {
		case r == 0 && c == 0:
			return sys.D.At(0, 0)
		case r == 0:
			return lft.D12.At(0, c-1)
		case c == 0:
			return lft.D21.At(r-1, 0)
		}
		return lft.D22.At(r-1, c-1)
	}
	h := make([][]complex128, k)
	for r := range k {
		h[r] = make([]complex128, k)
		cr := cRow(r)
		for c := range k {
			v := complex(dAt(r, c), 0)
			for i := range n {
				v += complex(cr(i), 0) * x[i][c]
			}
			h[r][c] = v
		}
	}
	d0, d1 := cmplx.Exp(-s*complex(lft.Tau[0], 0)), cmplx.Exp(-s*complex(lft.Tau[1], 0))
	m00, m01, m10, m11 := 1-h[1][1]*d0, -h[1][2]*d1, -h[2][1]*d0, 1-h[2][2]*d1
	det := m00*m11 - m01*m10
	y0 := (m11*h[1][0] - m01*h[2][0]) / det
	y1 := (m00*h[2][0] - m10*h[1][0]) / det
	return h[0][0] + h[0][1]*d0*y0 + h[0][2]*d1*y1
}
