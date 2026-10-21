package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"testing"
)

// Nyquist checks for FuzzFreqResponse (task MOI6LA).
//
// Contour must be the FreqResponse data on the same grid and ContourN its
// conjugate mirror. The counts are checked against exact root counts of the
// float64 model: RHPPoles against Routh / Schur–Cohn on det(xI−A), RHPZerosCL
// against den·z^lag + num (delay-free continuous and discrete) or the
// argument principle on den + num·e^{−τs} (continuous delay). Counts are only
// compared when every intended pole is either exactly on the boundary or
// clear of the library's boundary band (cbrt(eps)·max|p|) by the eigenvalue
// error, and every closed-loop root is clear of a 1e-6 relative band.

const fzNyqBand = 1e-6

// fzNyqClosedLoopRHP counts the closed-loop roots of pl under unit negative
// feedback outside the stability boundary; ok is false when a root lies
// within the fzNyqBand of it or the oracle is undecided.
func fzNyqClosedLoopRHP(pl *fzPidPlant) (int, bool) {
	den, num := pl.den, fzTrimRat(pl.num)
	if pl.dt > 0 && pl.lag > 0 {
		zk := make([]*big.Rat, pl.lag+1)
		zk[0] = big.NewRat(1, 1)
		for i := 1; i <= pl.lag; i++ {
			zk[i] = new(big.Rat)
		}
		den = pidtuneRatMul(den, zk)
	}
	if pl.dt == 0 && pl.tau > 0 && len(num) > 0 {
		if pl.tau*fzNyqGainRadius(fzPidRatFloats(num), fzPidRatFloats(den), 0.5) > 200 {
			return 0, false
		}
		sigma := fzNyqBand * math.Max(1, fzPidFujiwara(fzPidRatFloats(den)))
		shifted := func(sg float64) (int, bool) {
			r := fzRat(sg)
			sn := fzPidRatScale(fzTaylorShift(num, r), fzRat(math.Exp(-pl.tau*sg)))
			return fzQuasiRHP(fzPidRatFloats(sn), fzPidRatFloats(fzTaylorShift(den, r)), pl.tau)
		}
		lo, okLo := shifted(-sigma)
		hi, okHi := shifted(sigma)
		return hi, okLo && okHi && lo == hi
	}
	cl := pidtuneRatAdd(den, num)
	if len(cl) > 0 && cl[0].Sign() == 0 {
		return 0, false
	}
	if len(cl) <= 1 {
		return 0, true
	}
	tol := fzNyqBand * math.Max(1, fzPidFujiwara(fzPidRatFloats(cl)))
	if pl.dt > 0 {
		if len(cl) > 21 {
			return 0, false
		}
		lo, okLo := fzSchurCohnOutside(cl, 1-fzNyqBand)
		hi, okHi := fzSchurCohnOutside(cl, 1+fzNyqBand)
		return hi, okLo && okHi && lo == hi
	}
	lo, okLo := fzRouthRight(cl, -tol)
	hi, okHi := fzRouthRight(cl, tol)
	return hi, okLo && okHi && lo == hi
}

// fzNyqGainRadius returns a radius r ≥ 1 with |num/den| < level on |s| ≥ r
// (the bound of sweepOracle), or +Inf when none is found below 1e8.
func fzNyqGainRadius(num, den []float64, level float64) float64 {
	bound := func(r float64) float64 {
		nv, dv := 0.0, math.Pow(r, float64(len(den)-1))*math.Abs(den[0])
		for i, c := range num {
			nv += math.Abs(c) * math.Pow(r, float64(len(num)-1-i))
		}
		for i, c := range den[1:] {
			dv -= math.Abs(c) * math.Pow(r, float64(len(den)-2-i))
		}
		return nv / dv
	}
	for r := 1.0; r < 1e8; r *= 2 {
		if b := bound(r); b > 0 && b < level {
			return r
		}
	}
	return math.Inf(1)
}

// fzNyqPoleClass reports whether some intended pole is exactly on the
// stability boundary (boundary) and whether one lies where the library's
// classification may differ from the intended one (ambiguous).
func fzNyqPoleClass(s *fzSpec) (boundary, ambiguous bool) {
	const eps = 0x1p-52
	maxAbs := 1.0
	for _, e := range s.Eig {
		maxAbs = math.Max(maxAbs, cmplx.Abs(e))
	}
	ee := s.EigErr()
	clear := 4 * math.Max(math.Cbrt(eps)*maxAbs, ee)
	for _, e := range s.Eig {
		switch m := s.Margin(e); {
		case m > clear:
		case m <= 1e-12*maxAbs && ee <= math.Sqrt(eps)*maxAbs/4:
			boundary = true
		default:
			ambiguous = true
		}
	}
	return boundary, ambiguous
}

func fzFRCheckNyquist(t *testing.T, s *fzSpec, sys *System, ws []float64, fr *FreqResponseMatrix) {
	t.Helper()
	res, err := sys.Nyquist(ws, 0)
	fzContract(t, s, "Nyquist", res != nil, err)
	if s.P != 1 || s.M != 1 {
		if !errors.Is(err, ErrNotSISO) {
			t.Fatalf("Nyquist on %dx%d: err %v, want ErrNotSISO\n%s", s.P, s.M, err, s)
		}
		return
	}
	boundary, ambiguous := fzNyqPoleClass(s)
	pl := fzPidPlantOf(t, sys)
	z, zok := fzNyqClosedLoopRHP(pl)
	closure := math.Abs(1 + s.D[0])
	decided := !boundary && !ambiguous && zok && closure > fzNyqBand
	if err != nil {
		if errors.Is(err, ErrSingularTransform) && !decided {
			fzSkip(t, "Nyquist: closed loop or poles on the stability boundary")
		}
		if errors.Is(err, ErrDelayUnsupported) && !decided {
			fzSkip(t, "Nyquist: delay winding beyond the sample budget")
		}
		t.Fatalf("Nyquist: %v (closed-loop count %d decided)\n%s", err, z, s)
	}
	if !fzFRBitEqual(res.Contour, fr.Data) {
		t.Fatalf("Nyquist Contour %v, FreqResponse %v\n%s", res.Contour, fr.Data, s)
	}
	nc := len(res.Contour)
	if len(res.Omega) != len(ws) || len(res.ContourN) != nc {
		t.Fatalf("Nyquist lengths Omega %d ContourN %d, want %d\n%s", len(res.Omega), len(res.ContourN), len(ws), s)
	}
	for k, w := range ws {
		if res.Omega[k] != w {
			t.Fatalf("Nyquist Omega[%d] = %g, want %g\n%s", k, res.Omega[k], w, s)
		}
		if !fzFRBitEqual([]complex128{res.ContourN[k]}, []complex128{cmplx.Conj(res.Contour[nc-1-k])}) {
			t.Fatalf("Nyquist ContourN[%d] = %v, want conj(Contour[%d]) = %v\n%s", k, res.ContourN[k], nc-1-k, cmplx.Conj(res.Contour[nc-1-k]), s)
		}
	}
	if res.RHPZerosCL != res.Encirclements+res.RHPPoles {
		t.Fatalf("Nyquist RHPZerosCL %d != Encirclements %d + RHPPoles %d\n%s", res.RHPZerosCL, res.Encirclements, res.RHPPoles, s)
	}
	auto, err := sys.Nyquist(nil, 0)
	fzErr(t, s, "Nyquist(auto)", err)
	if auto.Encirclements != res.Encirclements || auto.RHPPoles != res.RHPPoles {
		t.Fatalf("Nyquist counts depend on omega: auto %d/%d, grid %d/%d\n%s", auto.Encirclements, auto.RHPPoles, res.Encirclements, res.RHPPoles, s)
	}
	nyq := math.Inf(1)
	if s.Discrete {
		nyq = math.Pi / s.Dt * (1 + 1e-12)
	}
	for k, w := range auto.Omega {
		if !(w >= 0 && w <= nyq) || k > 0 && !(w > auto.Omega[k-1]) {
			t.Fatalf("Nyquist auto grid Omega[%d] = %g not increasing in [0, %g]\n%s", k, w, nyq, s)
		}
	}
	if ambiguous {
		return
	}
	var p int
	var pok bool
	if s.Discrete {
		p, pok = fzSchurCohnOutside(pl.den, 1)
	} else {
		p, pok = fzRouthRight(pl.den, 0)
	}
	if !pok && !boundary {
		return
	}
	if !boundary && res.RHPPoles != p {
		t.Fatalf("Nyquist RHPPoles %d, exact %d\n%s", res.RHPPoles, p, s)
	}
	if boundary && res.RHPPoles != s.Unstable()-fzNyqOnBoundary(s) {
		t.Fatalf("Nyquist RHPPoles %d, intended %d off-boundary unstable\n%s", res.RHPPoles, s.Unstable()-fzNyqOnBoundary(s), s)
	}
	if !decided {
		return
	}
	if res.RHPZerosCL != z {
		t.Fatalf("Nyquist RHPZerosCL %d (Encirclements %d, RHPPoles %d), exact closed-loop count %d\n%s", res.RHPZerosCL, res.Encirclements, res.RHPPoles, z, s)
	}
}

// fzNyqOnBoundary counts intended poles that Unstable counts as unstable
// but lie exactly on the boundary.
func fzNyqOnBoundary(s *fzSpec) int {
	k := 0
	for _, e := range s.Eig {
		if s.Margin(e) <= 1e-12*math.Max(1, cmplx.Abs(e)) && (s.Discrete && cmplx.Abs(e) >= 1 || !s.Discrete && real(e) >= 0) {
			k++
		}
	}
	return k
}

func TestFzNyqOracles(t *testing.T) {
	plant := func(a, b, c, d []float64, dt, tau float64, lag int) *fzPidPlant {
		n := len(a)
		if n > 0 {
			n = int(math.Sqrt(float64(len(a))))
		}
		return &fzPidPlant{n: n, a: a, b: b, c: c, d: d, dt: dt, tau: tau, lag: lag,
			den: fzCharPolyExact(a, n), num: fzPencilDetRat(n, 1, a, b, c, d)}
	}
	for _, c := range []struct {
		name string
		pl   *fzPidPlant
		want int
		ok   bool
	}{
		// 10/(s+1)^3: 1+L = (s+1)^3+10, roots −1−∛10·{1, e^{±jπ/3}} → 2 RHP.
		{"cubic high gain", plant([]float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}, []float64{0, 0, 1}, []float64{10, 0, 0}, []float64{0}, 0, 0, 0), 2, true},
		{"cubic low gain", plant([]float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}, []float64{0, 0, 1}, []float64{1, 0, 0}, []float64{0}, 0, 0, 0), 0, true},
		// 2/(s−1) closes to s+1: unstable open loop, stable closed loop.
		{"unstable plant", plant([]float64{1}, []float64{1}, []float64{2}, []float64{0}, 0, 0, 0), 0, true},
		// 2e^{−2s}/(s+1): quasi-polynomial with 2 RHP roots.
		{"delay", plant([]float64{-1}, []float64{1}, []float64{2}, []float64{0}, 0, 2, 0), 2, true},
		// 1.5/(z−0.5)·z^{−1}: z²−0.5z+1.5, |roots|² = 1.5.
		{"discrete lag", plant([]float64{0.5}, []float64{1}, []float64{1.5}, []float64{0}, 0.1, 0, 1), 2, true},
		{"discrete no lag", plant([]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 0.1, 0, 0), 0, true},
		// 1/s²: 1+L = (s²+1)/s², closed-loop poles on the boundary.
		{"boundary", plant([]float64{0, 1, 0, 0}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0, 0, 0), 0, false},
		{"D = -1", plant([]float64{-1}, []float64{1}, []float64{1}, []float64{-1}, 0, 0, 0), 0, false},
	} {
		if z, ok := fzNyqClosedLoopRHP(c.pl); ok != c.ok || ok && z != c.want {
			t.Errorf("%s: closed-loop RHP %d,%v, want %d,%v", c.name, z, ok, c.want, c.ok)
		}
	}
}
