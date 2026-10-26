package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// pidFRDDenseMargin is the least phase margin of c on the continuous
// num/den over [lo, hi], from a dense log sweep: +Inf without a crossing.
func pidFRDDenseMargin(num, den []float64, c PID2, lo, hi float64) float64 {
	L := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * pidTuningFeedback(c, w)
	}
	least := math.Inf(1)
	const n = 200000
	prev := cmplx.Abs(L(lo)) - 1
	for k := 1; k <= n; k++ {
		w := lo * math.Exp(float64(k)*math.Log(hi/lo)/n)
		v := cmplx.Abs(L(w)) - 1
		if prev*v < 0 || v == 0 {
			least = math.Min(least, 180-math.Abs(cmplx.Phase(L(w)))*180/math.Pi)
		}
		prev = v
	}
	return least
}

// CTA5F5: designs TunePIDFRD returns whose reported margin is the chord's.
// Across a lightly damped mode the chord between samples cuts inside the
// mode's circle, so a crossing in a step turning 38-59° read 8-26° high,
// and one a resonance lifts between samples was missed. Each design is
// unchanged, as only the reading is, and its reported margin now takes
// the crossings along the step's fits: within 1° of the exact plant's
// where two fits agree (resolved), and where the samples leave the mode
// unresolved, below the 58.5° bar it misses and uncertified. The last
// case's lone fit (σ 0.345 against 0.241) keeps |L| under 1, so its
// margin, 63° against 31°, is the data's to leave open: it is only
// uncertified, its warning saying the margins are not certified there.
// So is the PD on 19 samples over five decades (IJCIOS, sparse sweep 11)
// whose lone fit, pole 31.96+1.34j, keeps |L| under 0.37 across a step
// where the plant's ζ 0.012 pair at 33.4 rad/s crosses |L| = 1 twice, at
// 51.5° against the 68.0° reported: a fit σ shrunk by that 0.37 still
// leaves ζ above 0.001, so the step cannot bound the crossings.
// Oracles: companion roots and dense margin sweeps of the exact plant.
func TestTunePIDFRDFitMargins(t *testing.T) {
	const resolved, unresolved, open = 0, 1, 2
	for _, c := range []struct {
		pidFRDStrictCase
		kind int
	}{
		{pidFRDStrictCase{"resolved PID", []float64{0.00021970886469083835}, []float64{1, 0.9776769452222025, 0.1027180463208214, 0.018029569371016227, 0.0012495681804130616}, 0, 0.00437729377687368, 257.2570683909091, 215, PidtunePID, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: 0.3327909585719677, Ki: 0.21737968463068763, Kd: 23.760792318236128}}, resolved},
		{pidFRDStrictCase{"resolved PI", []float64{-28014.479965573824}, []float64{1, 3.9214130518411494, 584.1902033222893, 1630.3239500202471, 30828.914640377556}, 0, 0.049197713463672056, 31647.558055718375, 120, PidtunePI, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: -0.24866648396064797, Ki: -2.225808268882998}}, resolved},
		{pidFRDStrictCase{"unresolved PI", []float64{-0.0038222756367263635, -0.0012677818618067795}, []float64{1, 0.06464407243270755, 0.10005760257112094, 0.006401267418325189}, 0, 0.009316179985551253, 42.7362164944245, 24, PidtunePI, PIDTuningOptions{}, PID2{Kp: 0.7813664543073326, Ki: -0.15030598936538894}}, unresolved},
		{pidFRDStrictCase{"unresolved PID", []float64{-0.009282208062694386, 0.04876889443040703}, []float64{1, 0.07603633177863955, 308.805799363479, 11.21177666667732, 0.09525859590094857}, 0, 0.0012085278713223453, 240.56717347654157, 56, PidtunePID, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: 8.922063202887225, Ki: 0.12976245348251073, Kd: 197.33952686919986}}, unresolved},
		{pidFRDStrictCase{"lone fit PIDF", []float64{-78.74148934979667, -861.811951445858}, []float64{1, 1.6360010805908205, 115.34333096043075, 132.496771148978}, 0, 0.0011441884089918987, 226.26584361638064, 30, PidtunePIDF, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: 0.01802552622194053, Ki: -0.09105004128568404, Kd: -0.022092043152332708, Tf: 0.2651911956500729}}, open},
		{pidFRDStrictCase{"lone fit below unity PD", []float64{-3.615594739522033, -192.1747327166767, -3385.2820999049263, -19775.660643644802}, []float64{1, 4.79275327213227, 1118.14081692987, 4419.945850325421, 122.65938818334538, 70.46075872422402}, 0, 0.0035825926993535953, 398.3837886475481, 19, PidtunePD, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: -0.09989555702680614, Kd: -0.25882559180853604}}, open},
	} {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Ki, c.k.Ki}, {k.Kd, c.k.Kd}, {k.Tf, c.k.Tf}} {
			if math.Abs(g[0]-g[1]) > 1e-3*math.Abs(g[1]) {
				t.Errorf("%s: %+v, base gave %+v", c.name, k, c.k)
				break
			}
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", c.name, k)
		}
		least := pidFRDDenseMargin(c.num, c.den, k, c.lo/1e3, c.hi*1e3)
		pm, certified := r.Evidence.AchievedPhaseMargin, !pidHasWarning(r, "not certified")
		switch c.kind {
		case resolved:
			if !certified || math.Abs(pm-least) > 1 {
				t.Errorf("%s: certified %v, achieved margin %g, dense %g", c.name, certified, pm, least)
			}
		case unresolved:
			if certified || !(least < 58.5) || !(pm < 58.5) {
				t.Errorf("%s: certified %v, achieved margin %g, dense %g", c.name, certified, pm, least)
			}
		case open:
			if certified || !pidHasWarning(r, "nor its margins") || !(least < 58.5) {
				t.Errorf("%s: certified %v, warnings %q, dense %g", c.name, certified, r.Evidence.Warnings, least)
			}
		}
	}
}

// pidFRDResonanceSweepCase is a random stable continuous plant with one or
// two lightly damped pole pairs (ζ 0.01-0.3), up to two real poles and
// fewer zeros, a quarter of them unstable, sampled on a band past its
// poles at 20-249 points, with a family and options.
func pidFRDResonanceSweepCase(r *rand.Rand) pidFRDStrictCase {
	var poles, zeros []complex128
	minW, maxW := math.Inf(1), 0.
	for range 1 + r.IntN(2) {
		w := math.Pow(10, -2+4*r.Float64())
		z := math.Pow(10, -2+1.5*r.Float64())
		poles = append(poles, complex(-z*w, w*math.Sqrt(1-z*z)), complex(-z*w, -w*math.Sqrt(1-z*z)))
		minW, maxW = math.Min(minW, w), math.Max(maxW, w)
	}
	for range r.IntN(3) {
		w := math.Pow(10, -2+4*r.Float64())
		poles = append(poles, complex(-w, 0))
		minW, maxW = math.Min(minW, w), math.Max(maxW, w)
	}
	for range r.IntN(len(poles)) {
		w := math.Pow(10, -2+4*r.Float64())
		if r.IntN(4) == 0 {
			zeros = append(zeros, complex(w, 0))
		} else {
			zeros = append(zeros, complex(-w, 0))
		}
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
	lo := minW * math.Pow(10, -2+1.5*r.Float64())
	hi := maxW * math.Pow(10, 0.5+1.5*r.Float64())
	wm := math.Sqrt(lo * hi)
	k := math.Pow(10, -1+2*r.Float64()) / cmplx.Abs(pidPolyAt(num, complex(0, wm))/pidPolyAt(den, complex(0, wm)))
	if r.IntN(3) == 0 {
		k = -k
	}
	for i := range num {
		num[i] *= k
	}
	fams := []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	var o PIDTuningOptions
	if r.IntN(2) == 0 {
		o.PhaseMargin = []float64{45, 60, 70}[r.IntN(3)]
	}
	return pidFRDStrictCase{num: num, den: den, lo: lo, hi: hi, n: 20 + r.IntN(230), family: fams[r.IntN(len(fams))], o: o}
}

// CTA5F5 sweep: of 1200 resonant plants (pidFRDResonanceSweepCase, seed
// PCG(2525, i)), base certified stable designs whose reported margin was
// the chord's, 11-18° above the exact plant's, and one (422) whose |S|
// reached 5.6 along an unresolved resonance its samples left |S| under 4.
// The first now report within 2° of the dense margin, still certified;
// 422 is uncertified. Designs whose loop stays above |L| = 1 across an
// unresolved step (446, 573, 1133) stay certified, or read their steps
// as resolved where an ideal derivative's end above the band leaves the
// design uncertified (573, QHMGQL). Oracles: companion roots and dense
// margin and |S| sweeps of the exact plant.
func TestTunePIDFRDFitMarginsSweep(t *testing.T) {
	t.Parallel()
	const reported, uncertified, kept = 0, 1, 2
	for _, c := range []struct{ seed, kind int }{
		{161, reported}, {165, reported}, {531, reported}, {560, reported}, {600, reported}, {842, reported},
		{903, reported}, {920, reported}, {1011, reported}, {1154, reported}, {1190, reported},
		{422, uncertified},
		{446, kept}, {573, kept}, {1133, kept},
	} {
		p := pidFRDResonanceSweepCase(rand.New(rand.NewPCG(2525, uint64(c.seed))))
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, p.num, p.den, 0, p.lo, p.hi, p.n), p.num, p.den)
		r, err := TunePIDFRD(context.Background(), frd, p.family, p.o)
		if err != nil {
			t.Errorf("case %d: %v", c.seed, err)
			continue
		}
		k := *r.Controller
		if !pidFRDStrictStable(p.num, p.den, k) {
			t.Errorf("case %d: %+v unstable", c.seed, k)
		}
		certified, pm := !pidHasWarning(r, "not certified"), r.Evidence.AchievedPhaseMargin
		switch c.kind {
		case reported:
			if least := pidFRDDenseMargin(p.num, p.den, k, p.lo/1e3, p.hi*1e3); !certified || math.Abs(pm-least) > 2 {
				t.Errorf("case %d: certified %v, achieved margin %g, dense %g", c.seed, certified, pm, least)
			}
		case uncertified:
			if peak := pidFRDStrictPeak(p.num, p.den, 0, k, p.lo); certified || !(peak > 4) {
				t.Errorf("case %d: certified %v, dense |S| %g", c.seed, certified, peak)
			}
		case kept:
			if !certified && k.Tf == 0 && k.Kd != 0 {
				strict, _ := pidTuningFRDStrictPlant(frd, p.o.UnstablePoles)
				certified = pidHasWarning(r, "loop's end above the band is established") && pidTuningFRDFitMargins(r, strict)
			}
			if peak := pidFRDStrictPeak(p.num, p.den, 0, k, p.lo); !certified || !(peak <= 4) {
				t.Errorf("case %d: certified %v, dense |S| %g", c.seed, certified, peak)
			}
		}
	}
}
