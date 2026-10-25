package controlsys

import (
	"context"
	"math"
	"testing"
)

// N4MOKA: designs base certified whose closed loop is unstable. In each, a
// step of the data spans one or two lightly damped pole pairs that its
// Möbius fits miss or misplace (σ 18 times off, a pole on a sample, a
// single fit, none), and 1 + L winds clockwise round 0 between the samples,
// a whole turn the count's chord or fit omits. Such a loop winds only
// clockwise, so it hides 0 only on the chord's left. The count is now
// unknown there, and TunePIDFRD returns none of them as certified. 23N2LA:
// a step with no fit, its samples lifted above both neighbours as by a pole
// pair, may turn 1 + L by a whole turn on either side of a chord pointing
// at 0, so the count is unknown there unless |L| < 1/4 at both samples.
// Oracle: companion roots.
var pidFRDHiddenLoopCases = []pidFRDStrictCase{
	{"single unpeaked fit", []float64{2062.1976477818785, 95894.4886540121}, []float64{1, 95.7185776975706, 1420.5099165305878, 6014.416930097244, 24228.74835678858, 74743.87826038703}, 0, 0.014349330730324742, 11758.877956972425, 53, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: -0.10570812779034199, Ki: 0.9287026223339061, Kd: 0.06039806259245215, Tf: 0.13570665265529955}},
	{"I across a mode", []float64{599.9337231846489, -159.14521677684016}, []float64{1, 7.090886370482792, 29.89349950809612, 93.78254217522755, 213.62850727128034, 355.5991911324361, 447.36078792140376, 293.3248819375879, 129.48269631512647}, 0, 0.0030654955277600445, 319.28957363335036, 80, PidtuneI, PIDTuningOptions{}, PID2{Ki: -0.06711596679742844}},
	{"lightly damped pair", []float64{195.04176652497574}, []float64{1, 4.6929151968810565, 10.538090212621537, 48.77063403507675}, 0, 0.027849440633924225, 6694.226712295209, 42, PidtunePID, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: 0.002448811986326135, Ki: 0.4402769631965608, Kd: 0.04122176480656958}},
	{"fit σ off", []float64{195.1087467856993, 149.4501885574735}, []float64{1, 1.1249102676183718, 34.84815874687926, 27.69698975596252, 286.3502560328987, 81.06564787697926}, 0, 0.0033216640857289905, 1379.064168849248, 73, PidtunePD, PIDTuningOptions{}, PID2{Kp: 0.04870706953733432, Kd: 0.017016895853668352}},
	{"pole on a sample", []float64{-347.19846759911667, 11406.557951898125}, []float64{1, 51.20132926743227, 17.61234172897028, 820.0230221411465, 82.54258245350434, 3259.7580407623036, 48.288106380418554}, 0, 0.0010795219976801114, 1999.1494070358347, 104, PidtunePDF, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: 0.009049307831128717, Kd: -0.6122667635397638, Tf: 40.52247475253112}},
	{"two modes, no fit", []float64{-12505.756771008935, -1028787.5793204512, -15447337.992057707, -19571916.22781113, -3696877.4568146323}, []float64{1, 3.944436452372062, 15344.660767433672, 42629.59467122873, 57366279.82362354, 100751684.60760896, 2190524.6539016543}, 0, 9.591749284931512e-05, 2980.8411637540753, 63, PidtunePID, PIDTuningOptions{}, PID2{Kp: 2.009202774917678, Ki: -2.675150509159182}},
	{"P across a mode", []float64{0.00970854841538643, -0.011141568401819483, -0.19741337834125497, -0.2534753830285813, 0.01982621725700642}, []float64{1, 0.610372592528112, 135.75173816654333, 0.10343880640886427, 0.44592809533042876, 0.00012757506173720644, 0.0001503290019930676}, 0, 0.0002135729753304453, 127.71857789197442, 36, PidtuneP, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: 0.38088183459717423}},
	{"sparse, small |L| at samples", []float64{1338.2302683628454, 11823.193520075822, 26392.151008555666, 1360.6663842286257}, []float64{1, 31.092275323248685, 309.38892628527253, 3004.6511982298607, 20282.780775236362, 5739.81810943598, 10981.514949157812, 2076.1294507146254}, 0, 0.002280940077023028, 148.79438108535723, 20, PidtunePD, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: 0.1046652378625318, Kd: 0.0909849386070258}},
	{"chord points at 0", []float64{541.9345096301029, 41.62417449410552}, []float64{1, 92.57052625149991, 46.89794663487194, 123.96419354430567, 55.59009726316367, 41.21394348555669, 16.261385576967008}, 0, 0.004865497756727794, 666.0911760211138, 21, PidtunePID, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: -0.7431524411694643, Ki: 1.616785815773071, Kd: -0.0012746377994637761}},
	{"two close modes", []float64{5.025468768587509, 0.5035648540116158, 0.009892507382930122}, []float64{1, 2.0542670962178504, 24.123969807830978, 39.5573695311615, 0.1470554161814978, 0.05448336926448866, 8.934845166519797e-05, 1.8455040902742028e-05}, 0, 0.0029356348925410553, 75.88587666492003, 35, PidtunePD, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: -3.4589500798522264e-05, Kd: -0.0019974758385031426}},
}

func TestPIDTuningFRDCountHiddenLoop(t *testing.T) {
	for _, c := range pidFRDHiddenLoopCases {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		k := c.k
		k.B, k.C = 1, 1
		if pidFRDStrictStable(c.num, c.den, k) {
			t.Fatalf("%s: oracle finds %+v stable", c.name, k)
		}
		p, _ := pidTuningFRDStrictPlant(frd, nil)
		if pidTuningFRDCertified(k, p) {
			t.Errorf("%s: strict count certifies unstable %+v", c.name, k)
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			continue
		}
		got := *r.Controller
		certified := !pidHasWarning(r, "not certified")
		if certified != pidTuningFRDCertified(got, p) {
			t.Errorf("%s: %+v certified %v, warnings %q", c.name, got, certified, r.Evidence.Warnings)
		}
		if certified && !pidFRDStrictStable(c.num, c.den, got) {
			t.Errorf("%s: unstable %+v returned as certified", c.name, got)
		}
	}
}

// Wave 24 blockers: base's designs here are oracle-stable and robust, one
// uncertified on data whose strict count certifies none and one that only
// the strict search at a moved crossover finds, and whose strict verdict a
// hidden loop now leaves unknown. Each must still come back. Oracles:
// companion roots, dense |S| and dense margin sweeps of the exact plant.
func TestTunePIDFRDHiddenLoopKeepsDesigns(t *testing.T) {
	for _, c := range []struct {
		pidFRDStrictCase
		pm float64
	}{
		{pidFRDStrictCase{"PIDF PM70", []float64{565.5592668433621, 2506.6374579726407, 804.3539912461409}, []float64{1, 14.51539550040114, 725.6241918271446, 5241.563446031468}, 0, 0.7267839745619823, 36.812007783819055, 28, PidtunePIDF, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: 0.003192642175697842, Ki: 0.4845107633535718, Tf: 0.0034176500170738588}}, 62},
		{pidFRDStrictCase{"PDF moved", []float64{18608.21363927388, 9351.762842261767}, []float64{1, 7.670368014352613, 6112.424090355229, 113.30099159896533, 1761.349788653694, 1.2126873308642636, 5.9384559332623015}, 0, 0.004755982546556745, 327.1851651627443, 23, PidtunePDF, PIDTuningOptions{}, PID2{Kp: 2.5233608900272682e-05, Kd: 0.0004807209293351548, Tf: 0.0708146591711532}}, 58.5},
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
		if s := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo); s > 4*1.0125 {
			t.Errorf("%s: %+v |S| %g", c.name, k, s)
		}
		if _, _, least := pidFRDDenseAngles(c.num, c.den, 0, k, c.lo/1e3, c.hi*1e3); least < c.pm {
			t.Errorf("%s: %+v dense margin %g", c.name, k, least)
		}
	}
}

// 23N2LA: the two samples about a smooth, densely sampled maximum of |G|
// exceed both neighbours, as about a pole pair, though only to second
// order in the step. The count stays known there, and base's oracle-stable
// designs stay certified. Oracles: companion roots, dense |S|.
func TestTunePIDFRDSmoothPeakStaysCertified(t *testing.T) {
	for _, c := range []pidFRDStrictCase{
		{"P at a flat maximum", []float64{-16.65166441248027, -0.2656008763515839}, []float64{1, 14.896440526427224, 102.3050809895444, 2.647216720152547}, 0, 0.0043993174278432546, 198.42290124282408, 110, PidtuneP, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: -6.125875260846174}},
		{"PD at a flat maximum", []float64{3.0657746670401274, 0.09640741963422544}, []float64{1, 3.7931016977404304, 0.35301221155991774}, 0, 0.002011814205805227, 26.3524093461653, 71, PidtunePD, PIDTuningOptions{}, PID2{Kp: 3.7117693349502523, Kd: -0.22214239134900643}},
	} {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Kd, c.k.Kd}} {
			if math.Abs(g[0]-g[1]) > 1e-3*math.Abs(g[1]) {
				t.Errorf("%s: %+v, base gave %+v", c.name, k, c.k)
				break
			}
		}
		if pidHasWarning(r, "not certified") {
			t.Errorf("%s: %+v not certified", c.name, k)
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", c.name, k)
		}
		if s := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo); s > 4 {
			t.Errorf("%s: %+v |S| %g", c.name, k, s)
		}
	}
}
