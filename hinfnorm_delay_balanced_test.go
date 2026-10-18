package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func unitFeedback(t *testing.T, sys *System) *System {
	t.Helper()
	one, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(sys, one, -1)
	if err != nil {
		t.Fatal(err)
	}
	return cl
}

// The H∞ peak tail of Feedback(companion loop, 1) is bounded in the
// balanced realization too: with the given ‖A‖₂ ≈ 1.4e4 tailGain is +Inf up
// to there, and the peak grid regrew to it. Oracle: dense sweep of
// |L/(1+L)| refined by golden section.
func TestHinfNormDelayCompanionTailGainBalanced(t *testing.T) {
	q := companionDelayQuasiLoop(-0.3)
	cl := unitFeedback(t, q.system(t))
	e, err := newDelayLFT(cl)
	if err != nil {
		t.Fatal(err)
	}
	if !(e.normA > 1e4) || e.bal == nil || !(e.bal.normA < 100) {
		t.Fatalf("‖A‖₂ given %g, balanced %v", e.normA, e.bal)
	}
	if g := e.tailGain(0, 100); !(g < 1) {
		t.Fatalf("tailGain(0, 100) = %g, want < 1", g)
	}
	l, res, err := e.stability()
	if err != nil || !res.stable {
		t.Fatalf("stable=%v err=%v", res.stable, err)
	}
	peak, _, err := e.peakOrZero(l, res)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := oraclePeakShifted(q.L, q.wmax, -1)
	if math.Abs(peak-want) > 1e-7*want {
		t.Errorf("peak %.12g, oracle %.12g", peak, want)
	}
	if e.sigmaCalls > 2000 {
		t.Errorf("%d gain samples, want <= 2000", e.sigmaCalls)
	}
}

// A stiff companion loop whose given ‖A‖₂ put the H∞ tail past the delay
// grid budget although the closed loop is stable. Oracles: RHP roots of χ by
// the argument principle, dense sweep of |L/(1+L)|.
func TestHinfNormDelayStiffCompanionLoop(t *testing.T) {
	q := quasiLoop{
		num: polyScale([]float64{4.35518229454678e+06, 7.216831436462314e+06}, 0.05),
		den: []float64{1, 155.944003732878, 9955.032050307716, 303390.20384392294, 3.7420871694087246e+06},
		tau: 0.1821465736750913, box: 400,
	}
	if roots := oracleRHPRoots(t, q.chi, q.box, q.den, q.num); roots != 0 {
		t.Fatalf("oracle RHP roots %d, want 0", roots)
	}
	cl := unitFeedback(t, q.system(t))
	got, _, err := HinfNorm(cl)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := oraclePeakShifted(q.L, 2000, -1)
	if math.Abs(got-want) > 1e-7*want {
		t.Errorf("HinfNorm %.12g, oracle %.12g", got, want)
	}
}

// tailGain must bound ‖G(jω) − G∞(jω)‖ for ω >= w with the jointly
// balanced realization of a badly scaled MIMO model, both in norm and
// entrywise (‖D22‖ = 0.5), also with a zero row of C2 and column of B2, and
// be finite well below the given ‖A‖₂, never NaN. Oracle:
// G and G∞ closed over Δ = diag(e^{-jωτ}) from complex solves with jωI − A.
func TestDelayTailGainBalancedMIMOBoundsGap(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 5))
	const p, m, nd = 2, 3, 2
	for trial := range 30 {
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
		rnd := func(r, c int, rowScale bool) *mat.Dense {
			out := mat.NewDense(r, c, nil)
			for i := range r {
				for j := range c {
					v := 2*rng.Float64() - 1
					if rowScale {
						v *= scale[i]
					} else {
						v /= scale[j]
					}
					out.Set(i, j, v)
				}
			}
			return out
		}
		B, C := rnd(n, m, true), rnd(p, n, false)
		B2, C2 := rnd(n, nd, true), rnd(nd, n, false)
		D := rnd(p, m, true)
		D.Scale(0.1/scale[0], D)
		D12 := mat.NewDense(p, nd, []float64{0.3, -0.2, 0.1, 0.4})
		D21 := mat.NewDense(nd, m, []float64{0.4, 0.1, -0.3, 0.2, 0, 0.5})
		d22 := 0.5
		if trial%2 == 1 {
			d22 = 0.3
		}
		D22 := mat.NewDense(nd, nd, []float64{0, 0, d22, 0})
		if trial%3 == 2 {
			for i := range n {
				C2.Set(0, i, 0)
				B2.Set(i, 1, 0)
			}
		}
		sys, err := New(mat.NewDense(n, n, a), B, C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		tau := []float64{0.3, 0.7}
		if err := sys.SetInternalDelay(tau, B2, C2, D12, D21, D22); err != nil {
			t.Fatal(err)
		}
		e, err := newDelayLFT(sys)
		if err != nil {
			t.Fatal(err)
		}
		if e.entrywise != (d22 == 0.5) {
			t.Fatalf("trial %d: entrywise=%v with ‖D22‖ = %g", trial, e.entrywise, d22)
		}
		bal := e.bal.normA
		if !(e.normA > 10*bal) {
			t.Fatalf("trial %d: given ‖A‖₂ %g not ≫ balanced %g", trial, e.normA, bal)
		}
		for _, f := range []float64{2, 5, 30, 1e3} {
			w := f * bal
			bound := e.tailGain(0, w)
			if math.IsNaN(bound) {
				t.Fatalf("trial %d: tailGain NaN at w = %g·balanced ‖A‖₂", trial, f)
			}
			if f >= 5 && math.IsInf(bound, 1) {
				t.Fatalf("trial %d: tailGain +Inf at w = %g·balanced ‖A‖₂", trial, f)
			}
			for k := 0; k <= 200; k++ {
				om := w * math.Pow(100, float64(k)/200)
				if gap := lftGapNorm(sys, om); gap > bound*(1+1e-9) {
					t.Fatalf("trial %d: ‖G − G∞‖(%g) = %g > tailGain(0, %g) = %g", trial, om, gap, w, bound)
				}
			}
		}
	}
}

// A realization's undefined bound (0·∞ below its ‖A‖₂) must not mask
// another's finite one.
func TestMinBoundSkipsNaN(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range []struct{ a, b, want float64 }{
		{inf, math.NaN(), inf}, {2, math.NaN(), 2}, {inf, 3, 3}, {2, 3, 2},
	} {
		if got := minBound(c.a, c.b); got != c.want {
			t.Errorf("minBound(%g, %g) = %g, want %g", c.a, c.b, got, c.want)
		}
	}
}

// lftGapNorm returns σ_max(G(jω) − G∞(jω)) for a 2-output model with two
// internal delays.
func lftGapNorm(sys *System, w float64) float64 {
	n, m, p := sys.Dims()
	lft := sys.LFT
	nd := len(lft.Tau)
	s := complex(0, w)
	cols := m + nd
	lu := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			lu[i*n+j] = complex(-sys.A.At(i, j), 0)
		}
		lu[i*n+i] += s
	}
	x := make([]complex128, n*cols)
	for i := range n {
		for j := range m {
			x[i*cols+j] = complex(sys.B.At(i, j), 0)
		}
		for j := range nd {
			x[i*cols+m+j] = complex(lft.B2.At(i, j), 0)
		}
	}
	piv := make([]int, n)
	cLUFactor(lu, piv, n)
	cLUSolve(lu, piv, x, n, cols)
	rows := p + nd
	cAt := func(r, k int) float64 {
		if r < p {
			return sys.C.At(r, k)
		}
		return lft.C2.At(r-p, k)
	}
	dAt := func(r, c int) float64 {
		switch {
		case r < p && c < m:
			return sys.D.At(r, c)
		case r < p:
			return lft.D12.At(r, c-m)
		case c < m:
			return lft.D21.At(r-p, c)
		}
		return lft.D22.At(r-p, c-m)
	}
	h := make([]complex128, rows*cols)
	d := make([]complex128, rows*cols)
	for r := range rows {
		for c := range cols {
			var v complex128
			for k := range n {
				v += complex(cAt(r, k), 0) * x[k*cols+c]
			}
			d[r*cols+c] = complex(dAt(r, c), 0)
			h[r*cols+c] = d[r*cols+c] + v
		}
	}
	delta := []complex128{cmplx.Exp(-s * complex(lft.Tau[0], 0)), cmplx.Exp(-s * complex(lft.Tau[1], 0))}
	closeLoop := func(h []complex128) []complex128 {
		at := func(r, c int) complex128 { return h[r*cols+c] }
		i00, i01 := 1-at(p, m)*delta[0], -at(p, m+1)*delta[1]
		i10, i11 := -at(p+1, m)*delta[0], 1-at(p+1, m+1)*delta[1]
		det := i00*i11 - i01*i10
		inv := [2][2]complex128{{i11 / det, -i01 / det}, {-i10 / det, i00 / det}}
		g := make([]complex128, p*m)
		for r := range p {
			for c := range m {
				v := at(r, c)
				for a := range nd {
					for b := range nd {
						v += at(r, m+a) * delta[a] * inv[a][b] * at(p+b, c)
					}
				}
				g[r*m+c] = v
			}
		}
		return g
	}
	g, ginf := closeLoop(h), closeLoop(d)
	var g00, g01, g11 complex128
	for c := range m {
		u, v := g[c]-ginf[c], g[m+c]-ginf[m+c]
		g00 += u * cmplx.Conj(u)
		g01 += u * cmplx.Conj(v)
		g11 += v * cmplx.Conj(v)
	}
	tr, det := real(g00+g11), real(g00*g11-g01*cmplx.Conj(g01))
	return math.Sqrt((tr + math.Sqrt(math.Max(tr*tr-4*det, 0))) / 2)
}
