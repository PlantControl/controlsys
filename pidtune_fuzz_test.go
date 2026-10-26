package controlsys

// PID tuning fuzz targets (QNALQV): Pidtune, TunePID and TunePIDFRD on
// SISO fzSpec plants. A nil error must give a closed loop that is stable by
// the exact oracle wherever the library claims a check (pole test, Nyquist),
// and reported evidence margins must match the directly evaluated loop.
// Robustness policy (|S| bound, margin floors, signed vs classical margins)
// is not checked.

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var fzPidFamilies = [...]PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF, PidtunePI}

// fzPidCase is a decoded PID fuzz input: kind bits 1-3 pick the family
// (7: PI with an explicit 45° margin), bit 4 an explicit crossover at the
// first block's natural frequency; the rest is fzDecode's SISO spec.
type fzPidCase struct {
	spec   *fzSpec
	family PidtuneType
	wc, pm float64
}

func fzPidFuzz(f *testing.F, body func(t *testing.T, c *fzPidCase)) {
	fzStats(f)
	f.Fuzz(func(t *testing.T, kind uint8, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 float64) {
		skipSlowUnderRace(t)
		s := fzDecode(t, kind&0xe1, [fzNVals]float64{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15})
		defer fzRecover(t, s)
		defer func() {
			if t.Failed() {
				t.Logf("spec:\n%s", s)
			}
		}()
		c := &fzPidCase{spec: s, family: fzPidFamilies[(kind>>1)&7]}
		if (kind>>1)&7 == 7 {
			c.pm = 45
		}
		if kind&0x10 != 0 {
			c.wc = 1
			if s.N > 0 && cmplx.Abs(s.S[0]) > 0 {
				c.wc = cmplx.Abs(s.S[0])
			}
			if s.Discrete {
				c.wc = math.Min(c.wc, 0.3*math.Pi/s.Dt)
			}
		}
		body(t, c)
	})
}

func fzPidAddSeed(f *testing.F, family int, explicitWC bool, s fzSeed) {
	s.M, s.P = 1, 1
	k, v := s.encode()
	k |= uint8(family&7) << 1
	if explicitWC {
		k |= 0x10
	}
	f.Add(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])
}

func fzPidSeeds(f *testing.F) {
	nd := float64(fzNoDelay)
	base := []fzSeed{
		{N: 2, Mag: [3]float64{0, 0.5}, Gain: 6, DScal: 0, LogDt: -1, Shear: -10, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 3, Mag: [3]float64{0, 0.3, -0.5}, Damp: [3]float64{1.5, 0, 0}, Gain: 6.5, LogDt: -1, Shear: -2, Rot: 0.4, Phase: 0.3, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 1, Mag: [3]float64{-40}, Gain: 6, LogDt: -1, Shear: -10, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 2, Mag: [3]float64{-40, 0}, Gain: 6, LogDt: -1, Shear: -10, Rot: 0.3, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 3, Mag: [3]float64{0.2, 0.4, 0}, Damp: [3]float64{-1.5, 0, 0}, Gain: 6.2, LogDt: -1.5, Shear: -1, Rot: 0.7, Phase: 1.1, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 4, Mag: [3]float64{0, 0.3}, Damp: [3]float64{11, 0.5}, Gain: 6, LogDt: -1, Shear: -3, Rot: 0.2, Phase: 0.5, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 2, Mag: [3]float64{0, -1}, Damp: [3]float64{1.2}, Gain: 6, DScal: 0.3, LogDt: -1, Shear: -10, Rot: 0.5, Phase: 2, In: nd, Out: nd, IO: nd, Tau: nd},
		{N: 2, Mag: [3]float64{0.1}, Damp: [3]float64{1.7}, Gain: 6.3, LogDt: -1, Shear: -10, In: -0.3, Out: nd, IO: nd, Tau: nd},
		{N: 3, Mag: [3]float64{-0.3, -0.1}, Damp: [3]float64{1.4}, Gain: 6.1, LogDt: -1, Shear: -2, Rot: 0.3, Phase: 0.9, In: -0.8, Out: -1.2, IO: nd, Tau: nd},
		{N: 0, Gain: 6, DScal: 1.5, LogDt: -1, In: nd, Out: nd, IO: nd, Tau: nd},
	}
	for i, s := range base {
		for fam := range 8 {
			fzPidAddSeed(f, fam, (i+fam)%3 == 0, s)
			if fam%2 == 0 {
				d := s
				d.Discrete = true
				fzPidAddSeed(f, fam, (i+fam)%4 == 1, d)
			}
		}
	}
	dl := base[7]
	dl.Discrete, dl.In = true, 0.3
	fzPidAddSeed(f, 5, false, dl)
	fzPidAddSeed(f, 2, true, dl)
}

// fzPidCtrl is the feedback path Kp + Ki·I + Kd·D of a PID or PID2.
type fzPidCtrl struct {
	kp, ki, kd, tf, dt float64
	wi, wd             float64
}

func fzPidWeight(f PIDFormula) float64 {
	switch f {
	case BackwardEuler:
		return 1
	case Trapezoidal:
		return 0.5
	}
	return 0
}

func fzPidCtrlOf(pid *PID) fzPidCtrl {
	return fzPidCtrl{pid.Kp, pid.Ki, pid.Kd, pid.Tf, pid.Dt, fzPidWeight(pid.IFormula), fzPidWeight(pid.DFormula)}
}

func fzPidCtrlOf2(pid *PID2) fzPidCtrl {
	return fzPidCtrl{pid.Kp, pid.Ki, pid.Kd, pid.Tf, pid.Dt, fzPidWeight(pid.IFormula), fzPidWeight(pid.DFormula)}
}

func (c fzPidCtrl) finite() bool {
	for _, x := range []float64{c.kp, c.ki, c.kd, c.tf} {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return c.tf >= 0
}

// lead is the z coefficient of the discrete derivative denominator
// Tf(z−1) + Dt(wd·z + 1 − wd).
func (c fzPidCtrl) lead() float64 { return c.tf + c.wd*c.dt }

func fzPidRatScale(p []*big.Rat, k *big.Rat) []*big.Rat {
	out := make([]*big.Rat, len(p))
	for i, x := range p {
		out[i] = new(big.Rat).Mul(x, k)
	}
	return out
}

// poly returns the exact controller numerator and denominator in s
// (continuous) or z (discrete), descending, without common factors from
// absent terms.
func (c fzPidCtrl) poly() (num, den []*big.Rat) {
	r := fzRat
	x := []*big.Rat{r(1), r(0)}
	iNum := []*big.Rat{r(c.ki)}
	dd := []*big.Rat{r(c.tf), r(1)}
	if c.tf == 0 {
		dd = dd[1:]
	}
	if c.dt > 0 {
		dt := r(c.dt)
		x = []*big.Rat{r(1), r(-1)}
		kdt := new(big.Rat).Mul(r(c.ki), dt)
		iNum = []*big.Rat{new(big.Rat).Mul(kdt, r(c.wi)), new(big.Rat).Mul(kdt, r(1-c.wi))}
		lead := new(big.Rat).Add(r(c.tf), new(big.Rat).Mul(r(c.wd), dt))
		dd = []*big.Rat{lead, new(big.Rat).Sub(dt, lead)}
		if lead.Sign() == 0 {
			dd = dd[1:]
		}
	}
	kp, kd := []*big.Rat{r(c.kp)}, []*big.Rat{r(c.kd)}
	add, mul := pidtuneRatAdd, pidtuneRatMul
	switch {
	case c.ki != 0 && c.kd != 0:
		return add(add(mul(mul(kp, x), dd), mul(iNum, dd)), mul(mul(kd, x), x)), mul(x, dd)
	case c.ki != 0:
		return add(mul(kp, x), iNum), x
	case c.kd != 0:
		return add(mul(kp, dd), mul(kd, x)), dd
	}
	return kp, []*big.Rat{r(1)}
}

// resp evaluates the controller at s = jω or z = e^{jωDt}.
func (c fzPidCtrl) resp(w float64) complex128 {
	s := complex(0, w)
	if c.dt == 0 {
		return complex(c.kp, 0) + complex(c.ki, 0)/s + complex(c.kd, 0)*s/(complex(c.tf, 0)*s+1)
	}
	z := cmplx.Exp(s * complex(c.dt, 0))
	dt := complex(c.dt, 0)
	i := dt * (complex(c.wi, 0)*z + complex(1-c.wi, 0)) / (z - 1)
	d := (z - 1) / (complex(c.lead(), 0)*z + dt - complex(c.lead(), 0))
	return complex(c.kp, 0) + complex(c.ki, 0)*i + complex(c.kd, 0)*d
}

// fzPidPlant is a SISO realization with its total I/O delay: tau seconds
// (continuous) or lag samples (discrete).
type fzPidPlant struct {
	n          int
	a, b, c, d []float64
	dt, tau    float64
	lag        int
	num, den   []*big.Rat
	numF, denF []float64
	// eig are the intended poles (s or z) when known, eigTol their
	// rounding radius: the response within it of a pole is undefined.
	eig    []complex128
	eigTol float64
}

func fzPidSpecPlant(t *testing.T, s *fzSpec, sys *System) *fzPidPlant {
	t.Helper()
	pl := fzPidPlantOf(t, sys)
	pl.eig = s.Eig
	if len(s.Eig) > 0 {
		pl.eigTol = 1e3 * s.EigErr()
	}
	return pl
}

// nearPole reports x(ω) within eigTol (or 1e-8 relative) of an intended pole.
func (pl *fzPidPlant) nearPole(w float64) bool {
	x := pl.x(w)
	for _, e := range pl.eig {
		if cmplx.Abs(x-e) <= math.Max(pl.eigTol, 1e-8*cmplx.Abs(e)) {
			return true
		}
	}
	return false
}

func fzPidPlantOf(t *testing.T, sys *System) *fzPidPlant {
	t.Helper()
	n, m, p := sys.Dims()
	if m != 1 || p != 1 || sys.HasInternalDelay() || sys.IsDescriptor() {
		t.Fatalf("fzPidPlantOf: need a SISO standard model without internal delays")
	}
	raw := func(x *mat.Dense) []float64 {
		if x == nil || x.IsEmpty() {
			return nil
		}
		return mat.DenseCopyOf(x).RawMatrix().Data
	}
	pl := &fzPidPlant{n: n, a: raw(sys.A), b: raw(sys.B), c: raw(sys.C), d: raw(sys.D), dt: sys.Dt}
	if n == 0 {
		pl.a, pl.b, pl.c = nil, nil, nil
	}
	if pl.d == nil {
		pl.d = []float64{0}
	}
	if sys.InputDelay != nil {
		pl.tau += sys.InputDelay[0]
	}
	if sys.OutputDelay != nil {
		pl.tau += sys.OutputDelay[0]
	}
	if sys.Delay != nil {
		pl.tau += sys.Delay.At(0, 0)
	}
	if pl.dt > 0 {
		pl.lag = int(math.Round(pl.tau))
		pl.tau = 0
	}
	pl.den = fzCharPolyExact(pl.a, n)
	pl.num = fzPencilDetRat(n, 1, pl.a, pl.b, pl.c, pl.d)
	return pl
}

// x is s = jω or z = e^{jωDt}.
func (pl *fzPidPlant) x(w float64) complex128 {
	if pl.dt > 0 {
		return cmplx.Exp(complex(0, w*pl.dt))
	}
	return complex(0, w)
}

// respExact is G(jω) from exact C(xI−A)⁻¹B+D times the delay factor.
func (pl *fzPidPlant) respExact(w float64) (complex128, bool) {
	x := pl.x(w)
	g, ok := fzExactResponse(pl.n, 1, 1, pl.a, pl.b, pl.c, pl.d, x)
	if !ok {
		return 0, false
	}
	return g[0] * pl.delay(w), true
}

func (pl *fzPidPlant) delay(w float64) complex128 {
	if pl.dt > 0 {
		return cmplx.Pow(pl.x(w), complex(-float64(pl.lag), 0))
	}
	return cmplx.Exp(complex(0, -w*pl.tau))
}

func fzPidRatFloats(p []*big.Rat) []float64 {
	out := make([]float64, len(p))
	for i, x := range p {
		out[i], _ = x.Float64()
	}
	return out
}

// respFast evaluates G from the exact num/den polynomials in big.Float;
// NaN when a coefficient overflows float64.
func (pl *fzPidPlant) respFast(w float64) complex128 {
	x := pl.x(w)
	num, den := fzPidRatFloats(pl.num), fzPidRatFloats(pl.den)
	for _, c := range append(num, den...) {
		if math.IsInf(c, 0) {
			return cmplx.NaN()
		}
	}
	return fzBigPolyEval(num, x) / fzBigPolyEval(den, x) * pl.delay(w)
}

// respHorner is respFast in complex128, for scanning only.
func (pl *fzPidPlant) respHorner(w float64) (complex128, bool) {
	if pl.numF == nil {
		pl.numF, pl.denF = fzPidRatFloats(pl.num), fzPidRatFloats(pl.den)
	}
	x := pl.x(w)
	return polyEval(pl.numF, x) / polyEval(pl.denF, x) * pl.delay(w), true
}

// fzPidVerdict is the exact closed-loop stability verdict.
type fzPidVerdict int

const (
	fzPidStable fzPidVerdict = iota
	fzPidUnstable
	fzPidBoundary
	fzPidUndecided
)

// fzPidFujiwara bounds the root moduli of the descending polynomial c.
func fzPidFujiwara(c []float64) float64 {
	b := 0.0
	for i := 1; i < len(c); i++ {
		b = math.Max(b, math.Pow(math.Abs(c[i]/c[0]), 1/float64(i)))
	}
	return 2 * b
}

// fzPidClosedLoop decides stability of the negative-feedback loop of pl and
// c: den·z^lag·denC + num·numC exactly for delay-free continuous and all
// discrete loops, the argument principle on the quasi-polynomial for a
// continuous delay. Roots within the boundary band (1e-9 relative) are
// fzPidBoundary.
func fzPidClosedLoop(pl *fzPidPlant, c fzPidCtrl) (fzPidVerdict, string) {
	cn, cd := c.poly()
	den := pidtuneRatMul(pl.den, cd)
	if pl.dt > 0 && pl.lag > 0 {
		zk := make([]*big.Rat, pl.lag+1)
		zk[0] = big.NewRat(1, 1)
		for i := 1; i <= pl.lag; i++ {
			zk[i] = new(big.Rat)
		}
		den = pidtuneRatMul(den, zk)
	}
	num := pidtuneRatMul(pl.num, cn)
	if pl.dt == 0 && pl.tau > 0 {
		num, den = fzTrimRat(num), fzTrimRat(den)
		k0, ok0 := fzQuasiRHP(fzPidRatFloats(num), fzPidRatFloats(den), pl.tau)
		if ok0 && k0 == 0 {
			return fzPidStable, ""
		}
		sigma := 1e-9 * math.Max(1, fzPidFujiwara(fzPidRatFloats(den)))
		sr := fzRat(sigma)
		sn := fzPidRatScale(fzTaylorShift(num, sr), fzRat(math.Exp(-pl.tau*sigma)))
		kt, okt := fzQuasiRHP(fzPidRatFloats(sn), fzPidRatFloats(fzTaylorShift(den, sr)), pl.tau)
		if okt && kt > 0 {
			return fzPidUnstable, fmt.Sprintf("%d roots of the delayed loop with Re s > %.3g", kt, sigma)
		}
		if !ok0 && !okt {
			return fzPidUndecided, "quasi-polynomial contour undecided"
		}
		return fzPidBoundary, fmt.Sprintf("delayed-loop roots near Re s = 0 (%d,%v / %d,%v)", k0, ok0, kt, okt)
	}
	cp := fzTrimRat(pidtuneRatAdd(den, num))
	if len(cp) == 0 || cp[0].Sign() == 0 {
		return fzPidUndecided, "characteristic polynomial vanishes"
	}
	if len(cp) == 1 {
		return fzPidStable, ""
	}
	if pl.dt > 0 {
		k0, ok0 := fzSchurCohnOutside(cp, 1)
		if ok0 && k0 == 0 {
			return fzPidStable, ""
		}
		kt, okt := fzSchurCohnOutside(cp, 1+1e-9)
		if okt && kt > 0 {
			return fzPidUnstable, fmt.Sprintf("%d roots with |z| > 1+1e-9", kt)
		}
		return fzPidBoundary, fmt.Sprintf("roots on or near |z| = 1 (%d,%v / %d,%v)", k0, ok0, kt, okt)
	}
	k0, ok0 := fzRouthRight(cp, 0)
	if ok0 && k0 == 0 {
		return fzPidStable, ""
	}
	tol := 1e-9 * math.Max(1, fzPidFujiwara(fzPidRatFloats(cp)))
	kt, okt := fzRouthRight(cp, tol)
	if okt && kt > 0 {
		return fzPidUnstable, fmt.Sprintf("%d roots with Re s > %.3g", kt, tol)
	}
	return fzPidBoundary, fmt.Sprintf("roots on or near Re s = 0 (%d,%v / %d,%v)", k0, ok0, kt, okt)
}

func fzPidWrap(deg float64) float64 {
	deg = math.Mod(deg, 360)
	if deg > 180 {
		deg -= 360
	}
	if deg <= -180 {
		deg += 360
	}
	return deg
}

func fzPidAngleDiff(a, b float64) float64 { return math.Abs(fzPidWrap(a - b)) }

func fzPidMaxPoleAbs(pl *fzPidPlant) float64 {
	m := 0.0
	for _, e := range pl.eig {
		m = math.Max(m, cmplx.Abs(e))
	}
	return m
}

// fzPidCheckStable fails when a design the library certified is not
// stable by the exact oracle; boundary roots are a precision skip.
func fzPidCheckStable(t *testing.T, name string, pl *fzPidPlant, c fzPidCtrl) {
	t.Helper()
	switch v, why := fzPidClosedLoop(pl, c); v {
	case fzPidUnstable:
		if pl.dt > 0 && fzPidMaxPoleAbs(pl) > 0x1p52 {
			fzSkip(t, name+": discrete pole beyond 1/eps, no float64 gain resolves |z| < 1")
		}
		t.Fatalf("%s returned a closed loop that is not stable: %s; ctrl %+v", name, why, c)
	case fzPidBoundary:
		fzSkip(t, name+": closed-loop root within the boundary band")
	case fzPidUndecided:
		fzSkip(t, name+": "+why)
	}
}

// fzPidLoopTol is the tolerance on log|L| and on phase (rad) at a reported
// crossover: the library evaluates its own realization, the oracle the
// exact one, so both track rounding of the response, scaled by the plant's
// sensitivity there.
const fzPidLoopTol = 1e-6

// fzPidCheckCrossover checks |L(jω)| = 1 and, when pm is finite, that the
// wrapped 180° + ∠L equals pm.
func fzPidCheckCrossover(t *testing.T, name string, pl *fzPidPlant, g func(float64) (complex128, bool), c fzPidCtrl, w, pm float64) {
	t.Helper()
	gw, ok := g(w)
	if !ok || pl.nearPole(w) {
		fzSkip(t, "crossover within rounding of a plant pole")
	}
	l := gw * c.resp(w)
	if l == 0 || cmplx.IsNaN(l) || cmplx.IsInf(l) {
		t.Fatalf("%s: loop %v at reported crossover %g; ctrl %+v", name, l, w, c)
	}
	tol := fzPidLoopTol * fzPidSlope(pl, g, c, w)
	if e := math.Abs(math.Log(cmplx.Abs(l))); e > tol {
		t.Fatalf("%s: |L(%g)| = %.12g at a reported crossover (log error %.3g > %.3g); ctrl %+v", name, w, cmplx.Abs(l), e, tol, c)
	}
	if math.IsNaN(pm) {
		return
	}
	got := 180 + cmplx.Phase(l)*180/math.Pi
	if d := fzPidAngleDiff(got, pm); d > tol*180/math.Pi+1e-9 {
		t.Fatalf("%s: margin %.9g° reported at %g, loop gives %.9g° (diff %.3g°); ctrl %+v", name, pm, w, got, d, c)
	}
}

// fzPidSlope bounds how far rounding moves log L near ω: 1 plus the
// sensitivity |d log L / d log ω| from a centred difference.
func fzPidSlope(pl *fzPidPlant, g func(float64) (complex128, bool), c fzPidCtrl, w float64) float64 {
	const h = 1e-4
	lo, ok1 := g(w * (1 - h))
	hi, ok2 := g(w * (1 + h))
	if !ok1 || !ok2 {
		return 1e6
	}
	dl := cmplx.Log(hi*c.resp(w*(1+h))) - cmplx.Log(lo*c.resp(w*(1-h)))
	return 1 + 10*float64(pl.n+2) + cmplx.Abs(dl)/(2*h)
}

// fzPidSystem builds the plant with its I/O delays. A continuous delay
// whose phase spans more than 1e3 rad over the plant's modes makes the
// library's Nyquist screen take ~1 min per call (followup), so such inputs
// are skipped.
func fzPidSystem(t *testing.T, s *fzSpec) *System {
	t.Helper()
	if !s.Discrete {
		tau := 0.0
		if s.In != nil {
			tau += s.In[0]
		}
		if s.Out != nil {
			tau += s.Out[0]
		}
		for _, e := range s.S {
			if tau*cmplx.Abs(e) > 1e3 {
				fzSkip(t, "delay phase over the plant modes too wide for a fast Nyquist screen")
			}
		}
	}
	return s.System(t, fzSysOpts{Delays: true})
}

// fzPidSkipDocumented skips documented errors other than unattainable,
// and known limits:
//   - ErrInvalidArgument wrapping ErrSingularTransform (ill-conditioned
//     realization; TunePID returns the bare ErrSingularTransform for the
//     same plants, a numerical failure it documents as aborting);
//   - a static plant with automatic wc (nothing to infer it from), and an
//     explicit wc where |G| is 0 or infinite;
//   - automatic wc on a plant with an intended pole within rounding of the
//     integrator ("no gain crossover, bandwidth or nonzero pole", or a
//     nonfinite response at ω ≈ 1e-16 for TunePID), which
//     rounding of A moves just inside the stability region (precision
//     limit: the exact integrator takes the integrating path); the same
//     error with poles ~1e-8 off the integrator and |G| > 1 everywhere
//     (followup);
//   - TunePID evaluating a plant whose float A has an eigenvalue exactly
//     on the imaginary axis at that pole's frequency (followup);
//   - TunePID P/I with automatic wc on a zero-DC-gain plant whose
//     one-gain search lands where |G| ≈ 0 and returns ErrInvalidArgument
//     (followup).
//
// fzPidNearIntegrator reports an intended pole within rounding (or the
// library's 1e-10 boundary band) of the integrator s = 0 / z = 1.
func fzPidNearIntegrator(s *fzSpec) bool {
	tol := math.Max(4*s.EigErr(), 1e-10)
	for _, e := range s.Eig {
		x := e
		if s.Discrete {
			x = e - 1
		}
		if cmplx.Abs(x) <= tol {
			return true
		}
	}
	return false
}

func fzPidSkipDocumented(t *testing.T, s *fzSpec, name string, wc float64, err error) {
	t.Helper()
	if errors.Is(err, ErrSingularTransform) {
		fzSkip(t, name+": ill-conditioned plant")
	}
	if !errors.Is(err, ErrInvalidArgument) {
		return
	}
	msg := err.Error()
	if strings.Contains(msg, "nonfinite plant response") && !s.Decidable() {
		fzSkip(t, name+": search sampled an imaginary-axis pole exactly (followup)")
	}
	if wc != 0 {
		if strings.Contains(msg, "at crossover is zero or not finite") {
			fzSkip(t, "explicit wc at a plant pole or zero")
		}
		return
	}
	switch {
	case s.N == 0:
		fzSkip(t, "static plant without crossover")
	case fzPidNearIntegrator(s):
		fzSkip(t, "rounded integrator without crossover (precision limit)")
	case strings.Contains(msg, "no gain crossover, bandwidth or nonzero pole"):
		fzSkip(t, "near-integrator plant with |G| > 1 everywhere (followup)")
	case strings.Contains(msg, "at crossover is zero or not finite"):
		fzSkip(t, name+": one-gain search at |G| = 0 (followup)")
	}
}

func fzPidCheckPID(t *testing.T, name string, pl *fzPidPlant, pid *PID) fzPidCtrl {
	t.Helper()
	if pid.Dt != pl.dt {
		t.Fatalf("%s: controller Dt %g, plant %g", name, pid.Dt, pl.dt)
	}
	c := fzPidCtrlOf(pid)
	if !c.finite() {
		t.Fatalf("%s: non-finite or negative-Tf gains %+v", name, pid)
	}
	if _, err := pid.System(); err != nil && !(c.kd != 0 && c.tf == 0 && errors.Is(err, ErrImproperModel)) {
		t.Fatalf("%s: controller System: %v (%+v)", name, err, pid)
	}
	return c
}

// fzPidCheckPidtune runs Pidtune on sys and checks the oracles.
func fzPidCheckPidtune(t *testing.T, s *fzSpec, sys *System, family PidtuneType, wc, pm float64) {
	t.Helper()
	pl := fzPidSpecPlant(t, s, sys)
	name := fmt.Sprintf("Pidtune(%s, wc=%g, pm=%g)", family, wc, pm)
	var opts *PidtuneOptions
	if pm > 0 {
		opts = &PidtuneOptions{PhaseMargin: pm}
	}
	pid, err := Pidtune(sys, family, wc, opts)
	fzContract(t, s, name, pid != nil, err)
	if err != nil {
		fzPidSkipDocumented(t, s, name, wc, err)
		fzErr(t, s, name, err, ErrPIDTuningTargetUnattainable)
	}
	c := fzPidCheckPID(t, name, pl, pid)
	if wc > 0 {
		fzPidCheckCrossover(t, name, pl, pl.respExact, c, wc, math.NaN())
	}
	fzPidCheckStable(t, name, pl, c)
}

func fzPidCheckEvidence(t *testing.T, name string, pl *fzPidPlant, g, scan func(float64) (complex128, bool), grid []float64, missedSkip string, c fzPidCtrl, ev PIDTuningEvidence) {
	t.Helper()
	if len(ev.GainCrossovers) != len(ev.PhaseMargins) {
		t.Fatalf("%s: %d crossovers, %d margins", name, len(ev.GainCrossovers), len(ev.PhaseMargins))
	}
	if ev.AchievedCrossover != ev.RequestedCrossover {
		t.Fatalf("%s: achieved crossover %g != requested %g", name, ev.AchievedCrossover, ev.RequestedCrossover)
	}
	fzPidCheckCrossover(t, name+" at wc", pl, g, c, ev.AchievedCrossover, math.NaN())
	for i, w := range ev.GainCrossovers {
		if !(w >= ev.FrequencyBand[0]*(1-1e-12) && w <= ev.FrequencyBand[1]*(1+1e-12)) {
			t.Fatalf("%s: crossover %g outside band %v", name, w, ev.FrequencyBand)
		}
		fzPidCheckCrossover(t, fmt.Sprintf("%s crossing %d", name, i), pl, g, c, w, ev.PhaseMargins[i])
	}
	fzPidCheckMissed(t, name, g, scan, grid, missedSkip, c, ev)
	wcMargin := fzPidWrap(180 + cmplx.Phase(fzPidMust(g(ev.AchievedCrossover))*c.resp(ev.AchievedCrossover))*180/math.Pi)
	if tol := fzPidLoopTol * fzPidSlope(pl, g, c, ev.AchievedCrossover) * 180 / math.Pi; ev.AchievedPhaseMargin > wcMargin+tol {
		t.Fatalf("%s: achieved margin %.9g above the wrapped margin %.9g at wc", name, ev.AchievedPhaseMargin, wcMargin)
	}
}

// fzPidCheckMissed fails when a sweep finds a clean gain crossing
// (|log|L|| > 1e-3 on both sides of a step, confirmed by g) that is neither
// wc nor reported. scan is a cheap evaluator used only to locate candidates;
// grid nil means a dense log grid of the band. Tangent grazes and crossing
// pairs within one step are not resolved. A nonempty missedSkip turns a
// miss into that counted skip. More than 16 candidate brackets means the cheap scan is
// unreliable (rounded coefficients); the check then stops.
func fzPidCheckMissed(t *testing.T, name string, g, scan func(float64) (complex128, bool), grid []float64, missedSkip string, c fzPidCtrl, ev PIDTuningEvidence) {
	t.Helper()
	lo, hi := ev.FrequencyBand[0], ev.FrequencyBand[1]
	if !(lo > 0 && hi > lo) {
		return
	}
	if grid == nil {
		const k = 3000
		for i := range k {
			grid = append(grid, lo*math.Pow(hi/lo, float64(i)/(k-1)))
		}
	}
	logL := func(f func(float64) (complex128, bool), w float64) float64 {
		gw, ok := f(w)
		if !ok {
			return math.NaN()
		}
		return math.Log(cmplx.Abs(gw * c.resp(w)))
	}
	clean := func(a, b float64) bool { return math.Abs(a) > 1e-3 && math.Abs(b) > 1e-3 && (a < 0) != (b < 0) }
	reported := append([]float64{ev.AchievedCrossover}, ev.GainCrossovers...)
	pw, pl := math.NaN(), math.NaN()
	confirms := 0
	for _, w := range grid {
		if w < lo || w > hi {
			continue
		}
		l := logL(scan, w)
		if clean(pl, l) {
			if confirms++; confirms > 16 {
				return
			}
			found := false
			for _, r := range reported {
				if r >= pw*(1-1e-9) && r <= w*(1+1e-9) {
					found = true
				}
			}
			if a, b := logL(g, pw), logL(g, w); !found && clean(a, b) {
				if missedSkip != "" {
					fzSkip(t, missedSkip)
				}
				t.Fatalf("%s: unreported gain crossing in [%.9g, %.9g] (log|L| %.3g -> %.3g); reported %v; ctrl %+v", name, pw, w, a, b, reported, c)
			}
		}
		pw, pl = w, l
	}
}

func fzPidMust(v complex128, ok bool) complex128 {
	if !ok {
		return cmplx.NaN()
	}
	return v
}

// fzPidCheckTune runs TunePID and checks stability when the evidence names
// a certifying check, and the reported margins.
func fzPidCheckTune(t *testing.T, s *fzSpec, sys *System, family PidtuneType, wc, pm float64) {
	t.Helper()
	pl := fzPidSpecPlant(t, s, sys)
	name := fmt.Sprintf("TunePID(%s, wc=%g, pm=%g)", family, wc, pm)
	res, err := TunePID(context.Background(), sys, family, PIDTuningOptions{CrossoverFrequency: wc, PhaseMargin: pm, MaxEvaluations: 512})
	fzContract(t, s, name, res != nil, err)
	if err != nil {
		fzPidSkipDocumented(t, s, name, wc, err)
		fzErr(t, s, name, err, ErrPIDTuningTargetUnattainable)
	}
	if res.Controller == nil {
		t.Fatalf("%s: nil controller with nil error", name)
	}
	if res.Controller.Dt != pl.dt {
		t.Fatalf("%s: controller Dt %g, plant %g", name, res.Controller.Dt, pl.dt)
	}
	c := fzPidCtrlOf2(res.Controller)
	if !c.finite() {
		t.Fatalf("%s: non-finite gains %+v", name, res.Controller)
	}
	if _, err := res.Controller.System(); err != nil && !(c.kd != 0 && c.tf == 0 && errors.Is(err, ErrImproperModel)) {
		t.Fatalf("%s: controller System: %v", name, err)
	}
	fzPidCheckEvidence(t, name, pl, pl.respExact, pl.respHorner, nil, "", c, res.Evidence)
	switch res.Evidence.Stability {
	case PIDStabilityClosedLoopPoles, PIDStabilityNyquist:
		fzPidCheckStable(t, name, pl, c)
	case PIDStabilitySampledOnly:
	default:
		t.Fatalf("%s: unknown stability evidence %q", name, res.Evidence.Stability)
	}
}

// fzPidFRDOf samples pl's response on a log grid spanning its modes; the
// data, not the model, is the oracle's truth.
func fzPidFRDOf(t *testing.T, s *fzSpec, pl *fzPidPlant) *FRD {
	t.Helper()
	lo, hi := 1e-3, 1e3
	for _, e := range s.S {
		if a := cmplx.Abs(e); a > 0 {
			lo, hi = math.Min(lo, a/100), math.Max(hi, a*100)
		}
	}
	if pl.dt > 0 {
		hi = math.Min(hi, 0.95*math.Pi/pl.dt)
		lo = math.Min(lo, hi/1e4)
	}
	const k = 600
	frd := &FRD{Dt: pl.dt}
	for i := range k {
		w := lo * math.Pow(hi/lo, float64(i)/(k-1))
		g := pl.respFast(w)
		if cmplx.IsNaN(g) || cmplx.IsInf(g) || g == 0 {
			fzSkip(t, "FRD sample on a pole")
		}
		frd.Omega = append(frd.Omega, w)
		frd.Response = append(frd.Response, [][]complex128{{g}})
	}
	return frd
}

// fzPidInterp is the documented log-frequency linear interpolant of frd.
func fzPidInterp(frd *FRD) func(float64) (complex128, bool) {
	return func(w float64) (complex128, bool) {
		om := frd.Omega
		if w <= om[0] {
			return frd.Response[0][0][0], true
		}
		k := len(om) - 1
		if w >= om[k] {
			return frd.Response[k][0][0], true
		}
		j := 1
		for om[j] < w {
			j++
		}
		a := math.Log(w/om[j-1]) / math.Log(om[j]/om[j-1])
		return complex(1-a, 0)*frd.Response[j-1][0][0] + complex(a, 0)*frd.Response[j][0][0], true
	}
}

func fzPidCheckFRD(t *testing.T, s *fzSpec, sys *System, family PidtuneType, wc, pm float64) {
	t.Helper()
	pl := fzPidSpecPlant(t, s, sys)
	frd := fzPidFRDOf(t, s, pl)
	if wc > 0 && (wc <= frd.Omega[0] || wc >= frd.Omega[len(frd.Omega)-1]) {
		wc = 0
	}
	name := fmt.Sprintf("TunePIDFRD(%s, wc=%g, pm=%g)", family, wc, pm)
	res, err := TunePIDFRD(context.Background(), frd, family, PIDTuningOptions{CrossoverFrequency: wc, PhaseMargin: pm, MaxEvaluations: 512})
	fzContract(t, s, name, res != nil, err)
	if err != nil {
		if errors.Is(err, ErrInvalidArgument) && strings.Contains(err.Error(), "at crossover is zero or not finite") {
			fzSkip(t, "FRD response below 1e-15 at the crossover")
		}
		fzErr(t, s, name, err, ErrPIDTuningTargetUnattainable)
	}
	if res.Evidence.Stability != PIDStabilitySampledOnly {
		t.Fatalf("%s: stability evidence %q for sampled data", name, res.Evidence.Stability)
	}
	c := fzPidCtrlOf2(res.Controller)
	if !c.finite() || res.Controller.Dt != pl.dt {
		t.Fatalf("%s: bad controller %+v", name, res.Controller)
	}
	g := fzPidInterp(frd)
	ev := res.Evidence
	if p, certify := pidTuningFRDStrictPlant(frd, nil); certify {
		// Crossings read along the data's resonance fits, or along the cubic
		// across a smooth step, are not the chord's.
		fit, _ := pidTuningFRDFitCrossings(p, *res.Controller)
		ev.GainCrossovers, ev.PhaseMargins = nil, nil
		for i, w := range res.Evidence.GainCrossovers {
			v, _, smooth := pidTuningFRDSmoothCrossing(p, *res.Controller, w)
			if !slices.ContainsFunc(fit, func(x float64) bool { return math.Abs(math.Log(w/x)) <= 1e-6 }) && !(smooth && math.Abs(math.Log(w/v)) <= 1e-6) {
				ev.GainCrossovers, ev.PhaseMargins = append(ev.GainCrossovers, w), append(ev.PhaseMargins, res.Evidence.PhaseMargins[i])
			}
		}
	}
	fzPidCheckEvidence(t, name, pl, g, g, frd.Omega, "", c, ev)
	for i, m := range res.Evidence.PhaseMargins {
		if res.Evidence.AchievedPhaseMargin > m+1e-9 {
			t.Fatalf("%s: achieved margin %.9g above %.9g at crossing %g", name, res.Evidence.AchievedPhaseMargin, m, res.Evidence.GainCrossovers[i])
		}
	}
}

func FuzzPidtune(f *testing.F) {
	fzPidSeeds(f)
	fzPidFuzz(f, func(t *testing.T, c *fzPidCase) {
		fzPidCheckPidtune(t, c.spec, fzPidSystem(t, c.spec), c.family, c.wc, c.pm)
	})
}

// fzPidTuneSlow reports a continuous delay τ with a near-undamped mode
// (ζ < 1e-3), or a lightly damped (ζ < 0.2) or unstable one with τ·|p| > 10:
// TunePID then runs 1–3 min per call whatever
// MaxEvaluations is (followup), so the fuzz body skips it.
func fzPidTuneSlow(s *fzSpec) bool {
	if s.Discrete {
		return false
	}
	tau := 0.0
	if s.In != nil {
		tau += s.In[0]
	}
	if s.Out != nil {
		tau += s.Out[0]
	}
	for _, e := range s.S {
		a := cmplx.Abs(e)
		if tau > 0 && a > 0 && (math.Abs(real(e)) < 1e-3*a || (math.Abs(real(e)) < 0.2*a || real(e) > 0) && tau*a > 10) {
			return true
		}
	}
	return false
}

func FuzzTunePID(f *testing.F) {
	fzPidSeeds(f)
	fzPidFuzz(f, func(t *testing.T, c *fzPidCase) {
		if fzPidTuneSlow(c.spec) {
			fzSkip(t, "TunePID slow on a delayed lightly damped plant (followup)")
		}
		fzPidCheckTune(t, c.spec, fzPidSystem(t, c.spec), c.family, c.wc, c.pm)
	})
}

func FuzzTunePIDFRD(f *testing.F) {
	fzPidSeeds(f)
	fzPidFuzz(f, func(t *testing.T, c *fzPidCase) {
		fzPidCheckFRD(t, c.spec, fzPidSystem(t, c.spec), c.family, c.wc, c.pm)
	})
}

// TestFzPidOracles checks the closed-loop builder against the existing
// exact Routh and Schur–Cohn PID oracles.
func TestFzPidOracles(t *testing.T) {
	num, den := Poly{1, 2}, Poly{1, 0.5, 3, -1}
	sys := pidtuneBackoffPlant(t, num, den, 0, 0)
	pl := fzPidPlantOf(t, sys)
	cont, disc := map[bool]bool{}, map[bool]bool{}
	for _, pid := range []*PID{
		{Kp: 1, Ki: 0.5, Kd: 0.2, Tf: 0.05},
		{Kp: 0.1, Ki: 2},
		{Kp: 3, Kd: 0.4, Tf: 0.01},
		{Kp: 0.6},
		{Kp: 2, Ki: 0.3, Kd: 0.5},
	} {
		want := pidtuneExactHurwitz(pid, num, den)
		cont[want] = true
		v, why := fzPidClosedLoop(pl, fzPidCtrlOf(pid))
		if (v == fzPidStable) != want || v > fzPidUnstable {
			t.Errorf("continuous %+v: verdict %d (%s), exact Hurwitz %v", pid, v, why, want)
		}
	}
	dnum, dden := Poly{0.2, -0.1}, Poly{1, -1.6, 0.7}
	dsys := pidtuneBackoffPlant(t, dnum, dden, 0.1, 0)
	dpl := fzPidPlantOf(t, dsys)
	for _, pid := range []*PID{
		{Kp: 0.5, Ki: 0.4, Kd: 0.05, Tf: 0.2, Dt: 0.1, IFormula: ForwardEuler, DFormula: BackwardEuler},
		{Kp: 1.5, Ki: 3, Dt: 0.1, IFormula: Trapezoidal},
		{Kp: 4, Ki: 9, Kd: 0.3, Tf: 0.02, Dt: 0.1, IFormula: BackwardEuler, DFormula: Trapezoidal},
		{Kp: 0.3, Kd: 0.1, Dt: 0.1, DFormula: BackwardEuler},
	} {
		want := pidtuneExactSchur(pid, dnum, dden, 0.1)
		disc[want] = true
		v, why := fzPidClosedLoop(dpl, fzPidCtrlOf(pid))
		if (v == fzPidStable) != want || v > fzPidUnstable {
			t.Errorf("discrete %+v: verdict %d (%s), exact Schur %v", pid, v, why, want)
		}
	}
	if len(cont) != 2 || len(disc) != 2 {
		t.Errorf("oracle cases need stable and unstable loops in both domains: %v %v", cont, disc)
	}
	for _, w := range []float64{0.1, 1, 7} {
		g, ok := pl.respExact(w)
		want := polyEval(num, complex(0, w)) / polyEval(den, complex(0, w))
		if !ok || cmplx.Abs(g-want) > 1e-12*cmplx.Abs(want) {
			t.Errorf("plant response at %g: %v, want %v", w, g, want)
		}
		if f := pl.respFast(w); cmplx.Abs(f-want) > 1e-12*cmplx.Abs(want) {
			t.Errorf("fast plant response at %g: %v, want %v", w, f, want)
		}
		c := fzPidCtrl{kp: 1, ki: 0.5, kd: 0.2, tf: 0.05}
		cn, cd := c.poly()
		pw := fzBigPolyEval(fzPidRatFloats(cn), complex(0, w)) / fzBigPolyEval(fzPidRatFloats(cd), complex(0, w))
		if r := c.resp(w); cmplx.Abs(r-pw) > 1e-12*cmplx.Abs(pw) {
			t.Errorf("controller response at %g: %v, poly %v", w, r, pw)
		}
		dc := fzPidCtrl{kp: 1, ki: 0.5, kd: 0.2, tf: 0.05, dt: 0.1, wi: 0.5, wd: 1}
		cn, cd = dc.poly()
		z := cmplx.Exp(complex(0, w*0.1))
		pw = fzBigPolyEval(fzPidRatFloats(cn), z) / fzBigPolyEval(fzPidRatFloats(cd), z)
		if r := dc.resp(w); cmplx.Abs(r-pw) > 1e-12*cmplx.Abs(pw) {
			t.Errorf("discrete controller response at %g: %v, poly %v", w, r, pw)
		}
	}
}

// TestPIDTune_FuzzRegressionCases runs the fuzz oracles over closed PID task repros
// lifted from the pidtune regression tests (source noted per row).
func TestPIDTune_FuzzRegressionCases(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	for i, c := range []struct {
		f         PidtuneType
		num, den  Poly
		dt, delay float64
	}{
		{PidtuneP, Poly{-0.20029805788732397, -0.032588497683437787, -8.730269241365936e-10}, Poly{1, 15.674185099429454, 28.120045177542792}, 0, 0},                                                                                      // pidtune_crossover_test.go:146
		{PidtunePD, Poly{0.018507331344365975, 0.009412563890440635, 1.5960285944590718e-07}, Poly{1, 0.6317501655870971, -0.17009008889471883}, 0, 0},                                                                                    // pidtune_crossover_test.go:147
		{PidtuneP, Poly{-0.24371488599761515, 0, 0, 0}, Poly{1, 5.3999877579765405, 7.271135289167493, 2.7160209747907498}, 0, 0},                                                                                                         // pidtune_crossover_test.go:212
		{PidtunePDF, Poly{0.4680649732178014, 0, 0, 0}, Poly{1, 4.609720722243954, 4.49026010353689, 1.2399444016467684}, 0, 0},                                                                                                           // pidtune_crossover_test.go:213
		{PidtuneI, Poly{0.8638233193779494, 0, 0}, Poly{1, 3.6152996784214597, 3.1165143190980205}, 0, 0},                                                                                                                                 // pidtune_crossover_test.go:214
		{PidtuneI, Poly{0.8296, 0.008296}, Poly{1, 0.999, 0.1257}, 0, 0},                                                                                                                                                                  // pidtune_delay_gate_test.go:183
		{PidtunePD, Poly{0.9259305881240456, -0.25001768378306766, -14.09549687784304, -4.875339983688054}, Poly{1, 10.913817523403575, 51.69190733555752, 143.2440252325901, 183.43503385908775}, 0, 0},                                  // pidtune_delay_gate_test.go:184
		{PidtunePD, Poly{0.8001772693191576, -0.930259502360965}, Poly{1, 7.820477719021238, 22.734552161124903, 29.06460406628826, 13.753654621908568}, 0, 0},                                                                            // pidtune_delay_gate_test.go:185
		{PidtunePDF, Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0, 0.11879022516031652},                                                                                    // pidtune_delay_gate_test.go:280
		{PidtunePDF, Poly{0.1043562311628774, 0}, Poly{1, 2.157756223719102, 12.93308866944033}, 0, 0.06098188011777296},                                                                                                                  // pidtune_delay_gate_test.go:281
		{PidtunePD, Poly{-0.014478772994582018, 0}, Poly{1, 17.775626209671092, 25.212313873525357}, 0, 0.08062746075097056},                                                                                                              // pidtune_delay_gate_test.go:282
		{PidtunePDF, Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0, 0.11879022516031652},                                                                                    // pidtune_delay_gate_test.go:315
		{PidtuneP, Poly{-0.7103680010513562, 0}, Poly{1, 7.438667224341966, 36.6377214599462}, 0, 0.66426273141541514},                                                                                                                    // pidtune_delay_gate_test.go:316
		{PidtunePD, Poly{-0.46375204436937656, -0.36621853843466834, 0}, Poly{1, 11.51148942688912, 43.55983162707241, 54.30021224234672}, 0, 0.485905856269508},                                                                          // pidtune_delay_gate_test.go:317
		{PidtunePID, Poly{1, 2}, Poly{1, 0.4, 4, 1}, 0, 0},                                                                                                                                                                                // pidtune_delay_gate_test.go:318
		{PidtunePID, Poly{0.0526145298369273, 0.09850694040108975, 6.479336680488852e-05}, Poly{1, 0.05549916906689037, 1.312022785519552}, 0, 0},                                                                                         // pidtune_backoff_level_cap_test.go:64
		{PidtunePD, Poly{-0.05081777337581659, -0.024330319418974466, -2.906628966763354e-09}, Poly{1, 0.060832011111875334, 2.042918947643126}, 0, 0},                                                                                    // pidtune_backoff_level_cap_test.go:65
		{PidtunePD, Poly{0.06293130656568592, 0.1269834378653455, 0.019629540046287263, 3.1895484290748856e-09}, Poly{1, 1.0877309346904058, 2.1584038875316245, 2.3120037925637735}, 0, 0},                                               // pidtune_backoff_level_cap_test.go:66
		{PidtunePI, Poly{0.06605624869768673, 0.03409642515501172, 6.837066608086198e-09}, Poly{1, 0.008011001940457408, 8.102832839220882}, 0, 0},                                                                                        // pidtune_backoff_level_cap_test.go:67
		{PidtuneI, Poly{0.27517413517840106, 3.6669924390083567e-05}, Poly{1, 6.391643798250351, -30.351456187775952}, 0, 0},                                                                                                              // pidtune_ungated_retry_test.go:26
		{PidtuneP, Poly{3.24364602965127, 0.0021262777284359194}, Poly{1, 13.719866434906919, 22.960516396640905, -141.22246804852216}, 0, 0},                                                                                             // pidtune_ungated_retry_test.go:27
		{PidtunePID, Poly{-1.8326830463596728, -4.165603628320087e-05}, Poly{1, 0.7334709862442443, 0.25518924748370436, 0.03900616993539339}, 0, 0},                                                                                      // pidtune_ungated_retry_test.go:28
		{PidtunePID, Poly{0.05153281511550831, 0.13713526593255349, 0.07056564212558677, 1.7258480953641303e-05}, Poly{1, 1.6146986643307246, 1.9381866559399348, 2.7956778334415695}, 0, 0},                                              // pidtune_ungated_retry_test.go:29
		{PidtunePD, Poly{0.6095365102040942, -0.3200683340297962, -0.1152823784131911}, Poly{1, 5.393739050975377, 9.53892918533701, 5.541668306872566}, 0, 0},                                                                            // pidtune_ungated_retry_test.go:30
		{PidtunePI, Poly{0.0009165950240917041, 0.00037102544443059034, 0.032239853096689565}, Poly{1, 0.40478666660694174, -13.978966685379245}, 0, 0},                                                                                   // pidtune_ungated_retry_test.go:31
		{PidtunePID, Poly{1.2654745408440329, 1.8998645549384214, 1.938148022288258e-06}, Poly{1, 1.2955994761823357, 1.078580429900587}, 0, 0},                                                                                           // pidtune_ungated_retry_test.go:59
		{PidtunePID, Poly{1.4754169078841284, 1.9475037635085144, 1.86207705614513, 3.085670387747623e-07}, Poly{1, 1.9884706490264086, 0.7760771899540704, 0.09700570403707065}, 0, 0},                                                   // pidtune_ungated_retry_test.go:60
		{PidtunePID, Poly{1.6393781502237057, 1.5729744617276629, 1.4422833102656818, 5.005710706969215e-07}, Poly{1, 0.9945214998051222, 0.5848509925742207, 0.15398262905361126}, 0, 0},                                                 // pidtune_ungated_retry_test.go:61
		{PidtunePID, Poly{1.7018722825622308, 0.7912405885497165, 1.1656162153765089e-06}, Poly{1, 0.31138563121400886, 0.008977261787331159}, 0, 0},                                                                                      // pidtune_ungated_retry_test.go:62
		{PidtunePID, Poly{0.7441606413430202, -1.4348718019413627, -8.909880207643177e-07}, Poly{1, 0.9530571064061941, 1.7469511168498217}, 0, 0},                                                                                        // pidtune_ungated_retry_test.go:63
		{PidtunePD, Poly{1.869847039800924, 1.9460197960801928, 0.33618542251074346, 0.8662899479933694}, Poly{1, 0.37809636237642263, 1.2251180342422012, 0.10118488017248772}, 0, 0},                                                    // pidtune_ungated_retry_test.go:64
		{PidtunePD, Poly{1.4070934529826, 0.01569665533397635, 1.7315628492954016}, Poly{1, 0.13987154850957495, 1.498413308783041}, 0, 0},                                                                                                // pidtune_ungated_retry_test.go:65
		{PidtunePD, Poly{-1.3812490475523809, 0.3049081517775649, -0.679036258551609}, Poly{1, 0.8507156397529578, 0.6749494711240132}, 0, 0},                                                                                             // pidtune_ungated_retry_test.go:66
		{PidtunePID, Poly{0.44274738180524537, 0.8774349592894404, 5.105628467814613e-09}, Poly{1, 0.44104297875501014, 0.6743286112634963}, 0, 0},                                                                                        // pidtune_ungated_retry_test.go:67
		{PidtunePID, Poly{0.298926677490796, 1.2071203877758425, 1.2110010334103394, 1.5931056551339456e-09}, Poly{1, 1.3266358687227324, 0.845646116236277, 0.4893392826128204}, 0, 0},                                                   // pidtune_ungated_retry_test.go:68
		{PidtuneP, Poly{2.645988078729179}, Poly{1, 0.6994623329224331, -1.129423843040615}, 0, 0},                                                                                                                                        // pidtune_ungated_retry_test.go:252
		{PidtunePI, Poly{2.645988078729179}, Poly{1, 0.6994623329224331, -1.129423843040615}, 0, 0},                                                                                                                                       // pidtune_ungated_retry_test.go:253
		{PidtunePI, Poly{1.9331897426073496e-15, -1.0091665836801221e-15}, Poly{1, -6.853180411596403, 20.125372937931182, -32.82904851475664, 32.126073810329295, -18.859953544489116, 6.150097496943598, -0.8593617743619164}, 0.01, 0}, // pidtune_ungated_retry_test.go:254
	} {
		t.Run(fmt.Sprintf("%d_%s", i, c.f), func(t *testing.T) {
			sys := pidtuneBackoffPlant(t, c.num, c.den, c.dt, c.delay)
			n, _, _ := sys.Dims()
			s := &fzSpec{N: n, M: 1, P: 1, Discrete: c.dt > 0, Dt: c.dt}
			t.Run("Pidtune", func(t *testing.T) { fzPidCheckPidtune(t, s, sys, c.f, 0, 0) })
			t.Run("TunePID", func(t *testing.T) { fzPidCheckTune(t, s, sys, c.f, 0, 0) })
		})
	}
}
