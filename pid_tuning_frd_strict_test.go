package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// pidFRDStrictData samples num/den, continuous or at dt (in z), at n log
// points on [lo, hi].
func pidFRDStrictData(t *testing.T, num, den []float64, dt, lo, hi float64, n int) *FRD {
	t.Helper()
	omega := make([]float64, n)
	values := make([][][]complex128, n)
	for k := range omega {
		omega[k] = math.Exp(math.Log(lo) + float64(k)*math.Log(hi/lo)/float64(n-1))
		x := complex(0, omega[k])
		if dt > 0 {
			x = cmplx.Exp(complex(0, omega[k]*dt))
		}
		values[k] = [][]complex128{{pidPolyAt(num, x) / pidPolyAt(den, x)}}
	}
	frd, err := NewFRD(values, omega, dt)
	if err != nil {
		t.Fatal(err)
	}
	return frd
}

// pidFRDStrictWithDC is frd with the exact sample of num/den at ω = 0
// prepended.
func pidFRDStrictWithDC(t *testing.T, frd *FRD, num, den []float64) *FRD {
	t.Helper()
	x := complex128(0)
	if frd.Dt > 0 {
		x = 1
	}
	out, err := NewFRD(append([][][]complex128{{{pidPolyAt(num, x) / pidPolyAt(den, x)}}}, frd.Response...), append([]float64{0}, frd.Omega...), frd.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// pidFRDStrictStable is the exact closed-loop stability of c on num/den:
// companion roots (pidClosedLoopMaxRe) or exact Schur–Cohn.
func pidFRDStrictStable(num, den []float64, c PID2) bool {
	if c.Dt > 0 {
		return pidtuneExactSchur(&PID{IFormula: c.IFormula, DFormula: c.DFormula, Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt}, num, den, c.Dt)
	}
	return pidClosedLoopMaxRe(num, den, c) < 0
}

const pidFRDLooserWarning = "The strict sampled Nyquist count certified no design"

type pidFRDStrictCase struct {
	name       string
	num, den   []float64
	dt, lo, hi float64
	n          int
	family     PidtuneType
	o          PIDTuningOptions
	k          PID2
}

// PN7T63: loops, all unstable, that TunePIDFRD returned and whose ends the
// data do not fix. The looser count certified the first three: a low end
// still turning from DC taken for none, a filtered derivative on a strictly
// proper plant ended at 1 + Re L although it rolls off, and a discrete
// |L| < 1 rising short of Nyquist. Then: discrete data stopping at 0.3π
// with |L| rising past the band; a discrete falling |L| in [1, 2) at 0.7π
// that a discrete L, ending at the real L(z = −1), keeps above 1; an
// integrator hidden behind an unstable zero just below the band, whose
// flat look was taken for a DC gain; a forward-Euler derivative pole at
// −0.82 that lifts |L| above 1 before Nyquist; an ideal derivative on
// continuous data stopping a decade below a far zero; and forward-Euler
// ideal derivatives on biproper discrete plants, whose pole at z = ∞ the
// data cannot show; an ideal PD on an integrator behind a nonminimum-phase
// zero 80 times below the band, whose flat look was taken for a DC gain
// (#11); and a PIDF on data stopping 1.5 times below a zero, whose top was
// taken to roll off (#12). The strict count must not certify any of them; each
// may still be returned, uncertified and so warned, as the looser count
// did. Oracles: companion roots and exact Schur–Cohn.
func TestPIDTuningFRDStrictCountRejectsUnknownEnds(t *testing.T) {
	for _, c := range []pidFRDStrictCase{
		{"low end turning from DC", []float64{-1.6397335053538995, 0.022678154196576948}, []float64{1, 0.05699814057448621, 0.0010336161545953615, 5.660971928156559e-06}, 0, 1e-3, 1e3, 100, PidtunePI, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -0.43546409142353076, Ki: 0.4189570650373774}},
		{"filtered derivative top", []float64{2.166140865041497, 16.891340677621397, 6.110303450674674, 0.4018892635929858}, []float64{1, 23.952741735010143, 226.10414419796365, 543.5529399666149, 325.67512854605485}, 0, 1e-2, 10, 100, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: -113.26339200852624, Ki: -83.09718274255565, Kd: -545.7322006479607, Tf: 0.1918018355416451}},
		{"discrete rising at 0.7π", []float64{1.4351105419474985}, []float64{1, 0.3657232065834215, 0}, 1, 1e-3, 0.7 * math.Pi, 500, PidtunePID, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -0.5597812346809283, Ki: 0.03546479912098162, Dt: 1}},
		{"discrete rising at 0.3π", []float64{0.5285655144705834}, []float64{1, 0.27082365510092343, 0.13777847481081767}, 0.05, 0.02, 0.3 * math.Pi / 0.05, 50, PidtunePI, PIDTuningOptions{PhaseMargin: 45, IFormula: Trapezoidal}, PID2{Kp: -1.682462918585547, Ki: 1.2686294737064991, Dt: 0.05, IFormula: Trapezoidal}},
		{"discrete falling above 1", []float64{1.1272406328814324}, []float64{1, 1.1684544028120818, 1.0260321132310026, 0.29572027153985536, -0.0373213055162959}, 1, 0.038707392544896815, 0.7 * math.Pi, 100, PidtunePI, PIDTuningOptions{PhaseMargin: 45, DFormula: BackwardEuler}, PID2{Kp: 0.7105312658025903, Ki: 0.8655482384549156, Dt: 1, DFormula: BackwardEuler}},
		{"hidden integrator", []float64{-0.24436543075354913, 0.009229922316894113}, []float64{1, 4.734640851796328, 16.2011199053105, 0}, 0, 0.04353021199383071, 1.4803666719297164, 100, PidtunePIDF, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: 59.347708325267654, Ki: -18.199284260310943, Kd: -125.57441253934684, Tf: 1.7654754475491596}},
		{"derivative pole near −1", []float64{-1.3448981227532175}, []float64{1, -1.5811862552962634, 0.8405266355392914, -0.2593403802430281}, 0.1, 0.5184154210153508, 17.34960121464623, 422, PidtunePIDF, PIDTuningOptions{PhaseMargin: 60, IFormula: BackwardEuler}, PID2{Kp: -0.13939, Ki: -0.030937, Kd: -0.015729, Tf: 0.054975, Dt: 0.1, IFormula: BackwardEuler}},
		{"ideal derivative below a far zero", []float64{-0.9043089425627748, -1256.883521757146, -389.6467771092301}, []float64{1, 6.820830968278832, 34.526396140064016}, 0, 1.3086639769123847, 147.87950751653452, 271, PidtunePD, PIDTuningOptions{PhaseMargin: 30}, PID2{Kp: -0.000531, Kd: 0.00076}},
		{"forward-Euler ideal PD", []float64{3.393044303813305, -0.27295347658179514, -3.3082691822417116, 0.16753894363629457}, []float64{1, -1.6457107028224587, 0.3553134944360514, 0.29046578031903075}, 0.1, 0.2162, 11.67, 365, PidtunePD, PIDTuningOptions{PhaseMargin: 45, IFormula: Trapezoidal}, PID2{Kp: 0.012539442669882746, Kd: -0.018758771811151466, Dt: 0.1, IFormula: Trapezoidal}},
		{"forward-Euler ideal PID", []float64{1.1165872865935105, -0.8967939326027657, 0.05725857765328797}, []float64{1, -1.099536361782705, 0.16610723397810698}, 1, 0.0013434142960855107, 3.0216239262734725, 218, PidtunePID, PIDTuningOptions{}, PID2{Kp: 0.002589, Ki: 0.018529, Kd: -0.059611, Dt: 1}},
		pidFRDHiddenNMPIntegrator,
		pidFRDShortTop,
	} {
		frd := pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n)
		k := c.k
		k.B, k.C = 1, 1
		if pidFRDStrictStable(c.num, c.den, k) {
			t.Fatalf("%s: oracle finds %+v stable", c.name, k)
		}
		p, _ := pidTuningFRDStrictPlant(frd, nil)
		if stable, known := pidTuningFRDNyquistCount(k, p); stable && known {
			t.Errorf("%s: strict count certifies unstable %+v", c.name, k)
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			continue
		}
		if !pidFRDStrictStable(c.num, c.den, *r.Controller) {
			if stable, known := pidTuningFRDNyquistCount(*r.Controller, p); stable && known || !pidHasWarning(r, "not certified") {
				t.Errorf("%s: unstable %+v returned as certified", c.name, *r.Controller)
			}
		}
	}
}

// PN7T63 redo: on these data the strict count certifies no design, so
// base's designs, oracle-stable and robust, must come back unchanged from
// the looser count, with its warning, not as the error the fail-closed
// count gave (wave 17): a discrete P whose |L| rises at 0.53 of Nyquist,
// a discrete PID whose |L(z = −1)| is 1.19 past the data, and a continuous
// ideal PD on a relative-degree-one plant whose data end before lim sG
// shows. Oracles: exact Schur–Cohn or companion roots, and a dense |S|
// sweep of the exact plant.
func TestTunePIDFRDKeepsLooserDesigns(t *testing.T) {
	for _, c := range []pidFRDStrictCase{
		{"discrete P", []float64{0.006977023683431606, -0.005548470227242547, 0.00783097344631761, -0.0038408396703643382}, []float64{1, -1.3506485258631935, 0.17990440470775687, 0.17329519001830493}, 0.1, 0.006737870932604683, 16.738137189669747, 184, PidtuneP, PIDTuningOptions{IFormula: Trapezoidal}, PID2{Kp: 2.9448}},
		{"discrete PID", []float64{3.220831505570417}, []float64{1, 0.2825729648756968}, 0.1, 0.2449532382066765, 24.49532382066765, 117, PidtunePID, PIDTuningOptions{}, PID2{Kp: -0.0579, Ki: 0.9845, Kd: 0.00791}},
		{"continuous ideal PD", []float64{-0.0003520644395038245, 0.004000505073661339, -0.025073389440655926}, []float64{1, 57.281250811462336, 6.403694511504273, 0.13273578080962684}, 0, 1.696e-4, 74.906, 143, PidtunePD, PIDTuningOptions{PhaseMargin: 50.23}, PID2{Kp: -37.18, Kd: -22.01}},
	} {
		r, err := TunePIDFRD(context.Background(), pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.family, c.o)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Ki, c.k.Ki}, {k.Kd, c.k.Kd}} {
			if math.Abs(g[0]-g[1]) > 1e-3*math.Max(math.Abs(g[1]), 1e-3) {
				t.Errorf("%s: %+v, base gave %+v", c.name, k, c.k)
				break
			}
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", c.name, k)
		}
		if s := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo); s > 4*1.0125 {
			t.Errorf("%s: %+v |S| %g", c.name, k, s)
		}
		if !pidHasWarning(r, pidFRDLooserWarning) {
			t.Errorf("%s: no looser-count warning in %q", c.name, r.Evidence.Warnings)
		}
	}
}

// pidFRDStrictPeak is the dense |S| peak of c on num/den from lo/1e4 to
// 1e4·lo·1e4 (continuous) or Nyquist (discrete).
func pidFRDStrictPeak(num, den []float64, dt float64, c PID2, lo float64) float64 {
	a, b := lo*1e-4, lo*1e8
	if dt > 0 {
		b = math.Pi / dt * (1 - 1e-12)
	}
	peak := 0.
	for i := 0; i <= 20000; i++ {
		w := a * math.Pow(b/a, float64(i)/20000)
		x := complex(0, w)
		if dt > 0 {
			x = cmplx.Exp(complex(0, w*dt))
		}
		peak = math.Max(peak, 1/cmplx.Abs(1+pidPolyAt(num, x)/pidPolyAt(den, x)*pidTuningFeedback(c, w)))
	}
	return peak
}

// #11 and #12: data on which the strict count certified unstable loops.
var (
	pidFRDHiddenNMPIntegrator = pidFRDStrictCase{"hidden NMP integrator", []float64{5.887909685908595, -0.22177996919354104}, []float64{1, 19.084760080327086, 0}, 0, 2.9975772244616192, 110.92050700065798, 141, PidtunePD, PIDTuningOptions{}, PID2{Kp: 3.597588165638652, Kd: -0.11915608313294526}}
	pidFRDShortTop            = pidFRDStrictCase{"top short of a zero", []float64{-0.4156058126006063, -0.7551548371911957}, []float64{1, 114.20727027161044, 4604.154939900912}, 0, 0.01042111217302736, 1.1785809567972114, 300, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: 3347.8068011369137, Ki: -672.1489827804135, Kd: -8158.033287328667, Tf: 2.9586115757568936}}
)

// Certification needs anchored ends: a DC gain sampled at ω = 0, not
// extrapolated from the band, and data reaching the loop's end above it.
// #11's integrator has no finite DC sample, and #12 stops below its zero
// with or without one, so neither may come back certified; a design they
// return carries the warning, or the call fails. Oracle: companion roots.
func TestTunePIDFRDCertificationNeedsAnchoredEnds(t *testing.T) {
	for _, c := range []pidFRDStrictCase{pidFRDHiddenNMPIntegrator, pidFRDShortTop} {
		frds := []*FRD{pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n)}
		if c.den[len(c.den)-1] != 0 {
			frds = append(frds, pidFRDStrictWithDC(t, frds[0], c.num, c.den))
		}
		for i, frd := range frds {
			k := c.k
			k.B, k.C = 1, 1
			p, _ := pidTuningFRDStrictPlant(frd, nil)
			if p.lowAnchored != (i == 1) {
				t.Errorf("%s %d: low anchored %v", c.name, i, p.lowAnchored)
			}
			if stable, known := pidTuningFRDNyquistCount(k, p); stable && known {
				t.Errorf("%s %d: strict count certifies unstable %+v", c.name, i, k)
			}
			r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
			if err != nil {
				continue
			}
			got := *r.Controller
			stable, known := pidTuningFRDNyquistCount(got, p)
			if !pidHasWarning(r, "not certified") != (stable && known) {
				t.Errorf("%s %d: count %v %v, warnings %q", c.name, i, stable, known, r.Evidence.Warnings)
			}
			if !pidHasWarning(r, "not certified") && !pidFRDStrictStable(c.num, c.den, got) {
				t.Errorf("%s %d: unstable %+v returned as certified", c.name, i, got)
			}
		}
	}
}

// The looser count's designs here, which base returned, the strict count
// finds unstable, and certifies none: a PD on a stable fourth-order plant
// and a PI on an integrator. The call fails rather than return them
// warned. Oracle: companion roots.
func TestTunePIDFRDFailsProvenUnstableFallback(t *testing.T) {
	for _, c := range []pidFRDStrictCase{
		{"stable PD", []float64{1.9069932277433254e-05}, []float64{1, 0.025199971874419462, 0.0011256420978643122, 1.0718595210231957e-05, 1.8479606353999822e-07}, 0, 0.0031957940883252326, 0.7111971289538478, 131, PidtunePD, PIDTuningOptions{}, PID2{Kp: -0.17017809052904345, Kd: -1.628104076156004}},
		{"integrator PI", []float64{572.2142240918271}, []float64{1, 60.73564696214835, 0}, 0, 3.267325748040892, 567.6764400557765, 162, PidtunePI, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: 5.543879041148438, Ki: -34.96722874113392}},
	} {
		k := c.k
		k.B, k.C = 1, 1
		if pidFRDStrictStable(c.num, c.den, k) {
			t.Fatalf("%s: oracle finds %+v stable", c.name, k)
		}
		r, err := TunePIDFRD(context.Background(), pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.family, c.o)
		if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s: %v, %v; want ErrPIDTuningTargetUnattainable", c.name, r, err)
		}
	}
}

// On these data the looser count returned unstable loops; the strict
// count's designs are stable: a forward-Euler derivative pole near z = −1
// that lifts |L| past the data (base returned Kp −0.139 unstable) and a
// biproper discrete plant on which base returned a forward-Euler ideal PID
// with a closed-loop root at z = 15.8. Without a DC sample they come back
// warned; with one, the PID, whose data reach 0.96π, is certified, as is
// a PI on a continuous plant whose data show its −2 slope. The first
// plant integrates (a pole at z = 1 to rounding), so it has no DC sample.
// Oracle: exact Schur–Cohn and companion roots.
func TestTunePIDFRDStrictCertifiesStableDesigns(t *testing.T) {
	for _, c := range []struct {
		pidFRDStrictCase
		certified bool
	}{
		{pidFRDStrictCase{"derivative pole near −1", []float64{-1.3448981227532175}, []float64{1, -1.5811862552962634, 0.8405266355392914, -0.2593403802430281}, 0.1, 0.5184154210153508, 17.34960121464623, 422, PidtunePIDF, PIDTuningOptions{PhaseMargin: 60, IFormula: BackwardEuler}, PID2{}}, false},
		{pidFRDStrictCase{"biproper PID", []float64{1.1165872865935105, -0.8967939326027657, 0.05725857765328797}, []float64{1, -1.099536361782705, 0.16610723397810698}, 1, 0.0013434142960855107, 3.0216239262734725, 218, PidtunePID, PIDTuningOptions{}, PID2{}}, true},
		{pidFRDStrictCase{"continuous PI", []float64{1}, []float64{1, 2, 1}, 0, 1e-2, 1e2, 200, PidtunePI, PIDTuningOptions{}, PID2{}}, true},
	} {
		sampled := pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n)
		frds := []*FRD{sampled}
		if c.certified {
			frds = append(frds, pidFRDStrictWithDC(t, sampled, c.num, c.den))
		}
		for i, frd := range frds {
			r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
			if err != nil {
				t.Errorf("%s %d: %v", c.name, i, err)
				continue
			}
			k := *r.Controller
			p, _ := pidTuningFRDStrictPlant(frd, nil)
			stable, known := pidTuningFRDNyquistCount(k, p)
			want := i == 1 && c.certified
			if (stable && known) != want || pidHasWarning(r, "not certified") == want {
				t.Errorf("%s %d: %+v count %v %v, warnings %q; want certified %v", c.name, i, k, stable, known, r.Evidence.Warnings, want)
			}
			if !pidFRDStrictStable(c.num, c.den, k) {
				t.Errorf("%s %d: %+v unstable", c.name, i, k)
			}
		}
	}
}

// pidFRDStrictSweepCase is a random stable plant, continuous or discrete,
// sampled on a band that may stop short of its poles at either end, with
// a family and options.
func pidFRDStrictSweepCase(r *rand.Rand) pidFRDStrictCase {
	dt := 0.
	if r.IntN(3) == 0 {
		dt = []float64{0.01, 0.1, 1}[r.IntN(3)]
	}
	np := 1 + r.IntN(4)
	var poles, zeros []complex128
	freq := func() float64 {
		w := math.Pow(10, -2+4*r.Float64())
		if dt > 0 {
			w = math.Min(w, 2/dt)
		}
		return w
	}
	minW, maxW := math.Inf(1), 0.
	for len(poles) < np {
		w := freq()
		minW, maxW = math.Min(minW, w), math.Max(maxW, w)
		if len(poles)+2 <= np && r.IntN(2) == 0 {
			z := 0.05 + 0.85*r.Float64()
			poles = append(poles, complex(-z*w, w*math.Sqrt(1-z*z)), complex(-z*w, -w*math.Sqrt(1-z*z)))
		} else {
			poles = append(poles, complex(-w, 0))
		}
	}
	nz := r.IntN(np)
	if dt > 0 {
		nz = r.IntN(np + 1)
	}
	for range nz {
		w := freq()
		if r.IntN(4) == 0 {
			zeros = append(zeros, complex(w, 0))
		} else {
			zeros = append(zeros, complex(-w, 0))
		}
	}
	x := func(w float64) complex128 { return complex(0, w) }
	if dt > 0 {
		for i := range poles {
			poles[i] = cmplx.Exp(poles[i] * complex(dt, 0))
		}
		for i := range zeros {
			zeros[i] = cmplx.Exp(zeros[i] * complex(dt, 0))
		}
		x = func(w float64) complex128 { return cmplx.Exp(complex(0, w*dt)) }
	}
	poly := func(roots []complex128) []float64 {
		p := []complex128{1}
		for _, z := range roots {
			q := make([]complex128, len(p)+1)
			for i, c := range p {
				q[i] += c
				q[i+1] -= c * z
			}
			p = q
		}
		out := make([]float64, len(p))
		for i, c := range p {
			out[i] = real(c)
		}
		return out
	}
	num, den := poly(zeros), poly(poles)
	lo := minW * math.Pow(10, -2+2.5*r.Float64())
	hi := maxW * math.Pow(10, -0.5+2*r.Float64())
	if dt > 0 {
		lo = math.Min(lo, 0.1/dt)
		hi = math.Min(hi, (0.3+0.69*r.Float64())*math.Pi/dt)
	}
	if hi < lo*30 {
		hi = lo * 30
		if dt > 0 && hi >= math.Pi/dt {
			hi = 0.99 * math.Pi / dt
			lo = hi / 30
		}
	}
	wm := math.Sqrt(lo * hi)
	k := math.Pow(10, -1+2*r.Float64()) / cmplx.Abs(pidPolyAt(num, x(wm))/pidPolyAt(den, x(wm)))
	if r.IntN(3) == 0 {
		k = -k
	}
	for i := range num {
		num[i] *= k
	}
	fams := []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	var o PIDTuningOptions
	if r.IntN(2) == 0 {
		o.PhaseMargin = []float64{30, 45, 60}[r.IntN(3)]
	}
	if dt > 0 {
		forms := []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal}
		o.IFormula, o.DFormula = forms[r.IntN(3)], forms[r.IntN(3)]
	}
	return pidFRDStrictCase{num: num, den: den, dt: dt, lo: lo, hi: hi, n: 50 + r.IntN(350), family: fams[r.IntN(len(fams))], o: o}
}

// Tripwire for the strict count: on random stable plants (no integrators,
// whose cancellation by a zero below the band the data cannot show) with
// their DC sample, every design TunePIDFRD returns as certified is stable, and every design kept
// from the looser count says so; the floor on certified designs guards
// against a strict count that certifies nothing. Oracles: companion roots
// and exact Schur–Cohn.
func TestTunePIDFRDCertifiedDesignsStable(t *testing.T) {
	n := 150
	if testing.Short() || raceEnabled {
		n = 30
	}
	r := rand.New(rand.NewPCG(18, 6373))
	certified := 0
	for i := range n {
		c := pidFRDStrictSweepCase(r)
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		res, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			continue
		}
		k := *res.Controller
		p, _ := pidTuningFRDStrictPlant(frd, nil)
		stable, known := pidTuningFRDNyquistCount(k, p)
		looser := pidHasWarning(res, "not certified")
		if looser {
			continue
		}
		if !stable || !known {
			t.Errorf("case %d: count %v %v without the looser warning", i, stable, known)
		}
		certified++
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("case %d %s num %v den %v dt %v band [%v, %v] n %d %+v: certified %+v unstable", i, c.family, c.num, c.den, c.dt, c.lo, c.hi, c.n, c.o, k)
		}
	}
	if certified < n/12 {
		t.Errorf("only %d of %d designs certified", certified, n)
	}
}
