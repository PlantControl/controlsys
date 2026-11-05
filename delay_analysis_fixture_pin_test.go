package controlsys

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// delayPinFixture is an existing delay test model with the analyses that
// take it through the delay loops: h HinfNorm (sigma, scan, peak,
// peakPastLimit, certify, torus scans), l Norm(·, Inf) (linfNormDelayed),
// s IsStable (refine, winding, nyquistGrid, descriptor balance), d DiskMargin
// (sensitivityPeaks, certifyPeaks, descriptor sensitivity), n Nyquist
// (neutralLoopRoots), p TunePID (pidLoopCertifier), c IsStable of the unit
// feedback loop (the cluster indentations behind unresolved samples).
type delayPinFixture struct {
	name  string
	ops   string
	build func(t *testing.T) *System
}

// delayPinQuasiLoops are the near-axis pole crowds of the delay Nyquist
// cluster, double-double and spread tests, as L = num/den·e^{−sτ}.
var delayPinQuasiLoops = []struct {
	name string
	q    quasiLoop
}{
	{"cluster/YJINK6-A", quasiLoop{num: []float64{4, 8, 8, 4, 0}, den: []float64{1, 1, 2, 2, 1, 1}, tau: 0.01}},
	{"cluster/5IT7DX-A", quasiLoop{num: []float64{2.068398001856945, 0.8594606031323779, 0.06137818585641332}, den: []float64{1, -0.0002458302753966347, -9.793006777603344e-11, 0}, tau: 0.030620080660797796}},
	{"cluster/RMVV47", quasiLoop{num: []float64{1.0220019540306915, 2.4805485191105365, 1.1305200275357294, 0.01821221818505486}, den: []float64{1, -0.017718929845052005, -1.4494897693008775e-10, 0, 0}, tau: 0.0044056653976761095}},
	{"ddreach/0", quasiLoop{num: []float64{0.11076720035335656, -6.130423072572409, -1.025734097506373, 2.740715007718803}, den: []float64{1, 2.511832216255788e-06, 6.992159985979351e-07, 8.781546547116625e-13, 1.2222520172891677e-13}, tau: 0.26598736421075958}},
	{"ddreach/2", quasiLoop{num: []float64{1.7881654623960874, 7.365322715526643, 0.7011603266679345, 1.4299568358509152}, den: []float64{1, 5.993352121789086e-07, 4.527910987579485e-07, 1.3568679771305505e-13, 5.1254924448136614e-14}, tau: 0.018411330449251636}},
	{"ddorigin/stable", quasiLoop{num: []float64{3.2352121915287078, 2.9318462185373737, 0.9209310942430203, 0.09073512716732975}, den: []float64{1, 7.916822512875715e-07, 2.23193650784524e-09, 8.409041323629466e-16, 1.2424568960202456e-18}, tau: 0.001993944126195067}},
	{"ddorigin/3-rhp", quasiLoop{num: []float64{0.0035412403357989973, 0.0009290551691596314, 0.01490447771022342, 0.00925057234567085, -0.03453728719690581}, den: []float64{1, 0.03877321282183848, 5.692903947267971e-07, 3.946154617055747e-09, 2.6374292206713098e-14, 1.003382448129943e-16}, tau: 0.007897857148837936}},
	{"ddorigin/pair-cluster-4-rhp", quasiLoop{num: []float64{-0.10001631017511914, 0.17353666104041401, 0.20980088519890508, -0.10933698566813194, 0.01366866251438389}, den: []float64{1, 0.09448176700595785, 3.759758798052076e-08, 1.308643397253839e-08, -2.1963128386391888e-15, 4.53142795006062e-16}, tau: 0.0026622700140613922}},
	{"pole-limit", quasiLoop{num: []float64{1.9188317375876882, 1.5083785090172463, 0.32156836056367194, 0.015658860719559062}, den: []float64{1, 0.46404121934763415, -6.209067747296065e-08, 0, 0}, tau: 0.001181949427158492}},
	{"spread/0", quasiLoop{num: []float64{0.7194876744332475, 0.48307710020589417, 0.9006210999510125, 0.8689376494791426}, den: []float64{1, -2.2904812444629556e-10, 3.930218025243091e-06, -4.501249932868734e-16, 3.86165339109627e-12}, tau: 0.0005760929572876901}},
	{"spread/1", quasiLoop{num: []float64{1.1516330598529883, 0.40152334289343644, 0.04128879625877322, 0.0012938485649076784}, den: []float64{1, 1.7245639736907134e-11, 5.342746792140293e-10, 4.646274483085076e-21, 7.135535394238958e-20}, tau: 0.00607265601266062}},
	{"spread/2", quasiLoop{num: []float64{0.02627760588501345, 0.2207918587498755, 0.09123646473331047, 0.1267845430927261}, den: []float64{1, -4.604291177027741e-14, 4.522955059349833e-12, -1.0375621120906069e-25, 5.00821720153443e-24}, tau: 0.00018902756700667477}},
	{"spread/3", quasiLoop{num: []float64{1, 1, 0.1, 0.01}, den: []float64{1, -4.604291177027741e-14, 4.522955059349833e-12, -1.0375621120906069e-25, 5.00821720153443e-24}, tau: 0.01}},
	{"tiny-real/0", quasiLoop{num: []float64{5.52461747222729, 10.25395216896589, 7.411393861456306, 2.08816698988852, 0.22553859559127404, 0.007420390418043597}, den: []float64{1, -0.008139247275845123, -7.936569881325126e-08, -2.288908801286239e-14, 0, 0, 0}, tau: 0.007702370618739055}},
	{"ddtrust/0", quasiLoop{num: []float64{6.840024404727002, 17.378659843517983, 20.769470510544636, 12.205225117024748, 3.4300126522593968, 0.41430325974383475, 0.02126540360186989, 0.00038621074628372373}, den: []float64{1, -0.02234011552778956, 6.512873463748606e-08, -1.904256343145472e-10, 6.010100728674183e-16, -6.602226110472291e-23, 4.120430178343998e-31, 0, 0}, tau: 0.019739683857353685}},
}

func delayPinFixtures() []delayPinFixture {
	first := func(s *System, _ ...any) *System { return s }
	var quasi []delayPinFixture
	for _, c := range delayPinQuasiLoops {
		quasi = append(quasi, delayPinFixture{c.name, "cd", func(t *testing.T) *System {
			p := makePlant(t, c.q.num, c.q.den)
			if err := p.SetInputDelay([]float64{c.q.tau}); err != nil {
				t.Fatal(err)
			}
			return p
		}})
	}
	return append(quasi, []delayPinFixture{
		{"repro-internal", "hls", func(t *testing.T) *System { return delayNonFiniteModel(t) }},
		{"first-order-loop/k1.5", "d", func(t *testing.T) *System {
			return delayedLoop(t, []float64{-1}, []float64{1}, []float64{1.5}, []float64{0}, 1, .5)
		}},
		{"first-order-loop/k-1.2", "d", func(t *testing.T) *System {
			return delayedLoop(t, []float64{-1}, []float64{1}, []float64{-1.2}, []float64{0}, 1, .5)
		}},
		{"integrator-loop", "d", func(t *testing.T) *System {
			return delayedLoop(t, []float64{0}, []float64{1}, []float64{.5}, []float64{0}, 1, .4)
		}},
		{"light-resonance/io", "d", func(t *testing.T) *System { s, _ := lightResonanceLoop(t, 1e-3, false); return s }},
		{"light-resonance/internal", "d", func(t *testing.T) *System { s, _ := lightResonanceLoop(t, 1e-3, true); return s }},
		{"mimo-resonances", "h", func(t *testing.T) *System { s, _ := mimoResonances(t); return s }},
		{"mimo-sensitivity/1", "hs", func(t *testing.T) *System { s, _ := mimoDelayedSensitivity(t, 1); return s }},
		{"mimo-sensitivity/1.8", "hs", func(t *testing.T) *System { s, _ := mimoDelayedSensitivity(t, 1.8); return s }},
		{"mimo3-sensitivity", "hs", func(t *testing.T) *System { return mimo3DelayedSensitivity(t) }},
		{"partly-commensurate/no-loop", "h", func(t *testing.T) *System {
			s, _ := partlyCommensurateLoop(t, .05, []float64{1, 1, -1, 1}, []float64{0, 0, 0, 0})
			return s
		}},
		{"partly-commensurate/neutral", "h", func(t *testing.T) *System {
			s, _ := partlyCommensurateLoop(t, .15, []float64{1, .6, -.8, .5}, []float64{.2, 0, -.3, .4})
			return s
		}},
		{"constant-difference/diagonal-scaling", "hs", func(t *testing.T) *System {
			s, _ := constantDifferenceLoop(t, []float64{.6, 5, -.05, .6}, []float64{.4, .9})
			return s
		}},
		{"constant-difference/equal-stein", "hs", func(t *testing.T) *System {
			s, _ := constantDifferenceLoop(t, []float64{.9, 1, -1, -.9}, []float64{.7, .7})
			return s
		}},
		{"neutral-pair/k3", "hs", func(t *testing.T) *System {
			return neutralPair(t, [2]neutralBlock{{2, 1, .6}, {1.5, -1, -.7}}, 3, []float64{.5, .5 * math.Sqrt2})
		}},
		{"sharp-resonance/0.999", "h", func(t *testing.T) *System { s, _, _ := sharpResonanceModel(t, .999); return s }},
		{"feedthrough-difference/commensurate", "h", func(t *testing.T) *System {
			s, _, _ := feedthroughDifferenceLoop(t, []float64{.54, .6, -.6, -.54}, []float64{1, -1}, []float64{1, -1}, []float64{.7, 1.4}, [2]float64{1, 2})
			return s
		}},
		{"expansion/near-90", "h", func(t *testing.T) *System { s, _, _ := expansionFinite[0].build(t); return s }},
		{"expansion/limit", "h", func(t *testing.T) *System { s, _, _ := expansionLimit.build(t); return s }},
		{"torus-cluster-siso", "h", func(t *testing.T) *System { return torusClusterModelSISO(t, expansionFinite[0], -365, 100) }},
		{"torus-seed/near-three-15", "h", func(t *testing.T) *System {
			return torusSeedModel(t, [2]float64{1, nearThree}, [2]float64{1, 3}, 15)
		}},
		{"far-channel", "h", func(t *testing.T) *System { s, _ := farChannel(t, .37, .8, .4, -1e-3, -10, 3); return s }},
		{"delay-resonance", "h", func(t *testing.T) *System {
			s, _ := delayResonanceLoop(t, []float64{.37}, .05, []float64{.8}, []float64{.4})
			return s
		}},
		{"companion-feedback", "h", func(t *testing.T) *System { return unitFeedback(t, companionDelayQuasiLoop(-.3).system(t)) }},
		{"neutral-parallel", "n", func(t *testing.T) *System {
			return neutralParallel(t, []neutralBranch{{d: .6, tau: 1}, {d: .6, tau: 2}, {d: .3, tau: 3}})
		}},
		{"indent/double-integrator", "hs", func(t *testing.T) *System {
			p := indentTestPlant(t, []float64{5.910980332296372, 27.806410726825444, 37.431038620035174, 11.565340194450041}, []float64{1, 0.20389057161026952, -0.07104652183880648, 0, 0})
			if err := p.SetInputDelay([]float64{0.022715415198538882}); err != nil {
				t.Fatal(err)
			}
			return first(Feedback(p, nil, -1))
		}},
		{"indent/triple-integrator", "s", func(t *testing.T) *System {
			p := indentTestPlant(t, []float64{0.46603599034106363, 1.1507662521895636, 0.8492300943505945, 0.24647378460270886, 0.024431190753480315}, []float64{1, 1.0954836376994372, 0, 0, 0})
			if err := p.SetOutputDelay([]float64{0.02290693554361851}); err != nil {
				t.Fatal(err)
			}
			k := indentTestDelayedGains(t, 0.9575000297606748, 0.019511878680882565, -0.022781002697111294, 0.26994534441905227)
			return first(Feedback(p, k, -1))
		}},
		{"static-internal-loop", "nsh", func(t *testing.T) *System { return staticInternalDelayLoop(t, 0) }},
		{"delayed-first-order", "pd", func(t *testing.T) *System { return delayedFirstOrder(t, 2.1) }},
		{"pid-delay-loop/io", "d", func(t *testing.T) *System { return benchPIDDelayLoop(t, false) }},
		{"pid-delay-loop/internal", "d", func(t *testing.T) *System { return benchPIDDelayLoop(t, true) }},
	}...)
}

func delayPinDigest(t *testing.T, sys *System, ops string) string {
	t.Helper()
	if sys == nil {
		t.Fatal("fixture model is nil")
	}
	var b strings.Builder
	for _, op := range ops {
		switch op {
		case 'h':
			n, w, err := HinfNorm(sys)
			fmt.Fprintf(&b, "h:%x,%x,%v;", n, w, err)
		case 'l':
			n, err := Norm(sys, math.Inf(1))
			fmt.Fprintf(&b, "l:%x,%v;", n, err)
		case 's':
			ok, err := sys.IsStable()
			fmt.Fprintf(&b, "s:%v,%v;", ok, err)
		case 'c':
			cl, err := Feedback(sys, nil, -1)
			if err != nil {
				t.Fatal(err)
			}
			ok, err := cl.IsStable()
			fmt.Fprintf(&b, "c:%v,%v;", ok, err)
		case 'd':
			dm, err := DiskMargin(sys)
			if err != nil {
				fmt.Fprintf(&b, "d:%v;", err)
				continue
			}
			w, ok := dm.Frequency()
			pw, _ := dm.PeakFreq()
			fmt.Fprintf(&b, "d:%x,%x,%x,%x,%x,%x,%v;", dm.Alpha, dm.GainMargin, dm.PhaseMargin, dm.PeakSensitivity, w, pw, ok)
		case 'n':
			r, err := sys.Nyquist(nil, 500)
			if err != nil {
				fmt.Fprintf(&b, "n:%v;", err)
				continue
			}
			fmt.Fprintf(&b, "n:%d,%d,%d;", r.Encirclements, r.RHPPoles, r.RHPZerosCL)
		case 'p':
			r, err := TunePID(context.Background(), sys, PidtunePID, PIDTuningOptions{})
			if err != nil {
				fmt.Fprintf(&b, "p:%v;", err)
				continue
			}
			c := r.Controller
			fmt.Fprintf(&b, "p:%x,%x,%x,%x;", c.Kp, c.Ki, c.Kd, c.Tf)
		}
	}
	return b.String()
}

// TestDelayAnalysisFixtureResultsPinned pins the results of the existing
// delay fixtures that reach the loops guarded against non-finite values to
// their values on 3f1ba16, before the guards (OAIXFO). Head matched base bit
// for bit on arm64 and amd64 when the guards landed; the pin allows the
// rounding spread across architectures and kernel selections.
func TestDelayAnalysisFixtureResultsPinned(t *testing.T) {
	skipSlowUnderRace(t)
	for _, f := range delayPinFixtures() {
		t.Run(f.name, func(t *testing.T) {
			got := delayPinDigest(t, f.build(t), f.ops)
			want := delayPinGolden[f.name]
			if msg := delayPinMismatch(got, want); msg != "" {
				t.Errorf("%s\ngot  %q\nwant %q", msg, got, want)
			}
		})
	}
}

const (
	delayPinValueTol = 1e-11
	// Peak and crossing locations sit on flat maxima, so their rounding
	// spread is near sqrt(eps): 6.2e-7 measured between arm64 and amd64.
	delayPinLocationTol = 1e-5
)

var delayPinFloat = regexp.MustCompile(`-?0x[0-9a-f.]+p[+-][0-9]+`)

// delayPinMismatch compares digests: everything except hex floats (errors,
// booleans, counts, infinities) exactly, values to delayPinValueTol and the
// HinfNorm and DiskMargin frequencies to delayPinLocationTol.
func delayPinMismatch(got, want string) string {
	gs, ws := strings.Split(got, ";"), strings.Split(want, ";")
	if len(gs) != len(ws) {
		return "different result count"
	}
	for i := range gs {
		g, w := gs[i], ws[i]
		if delayPinFloat.ReplaceAllString(g, "#") != delayPinFloat.ReplaceAllString(w, "#") {
			return fmt.Sprintf("segment %d differs outside floats", i)
		}
		gf, wf := delayPinFloat.FindAllString(g, -1), delayPinFloat.FindAllString(w, -1)
		for k := range gf {
			x, err1 := strconv.ParseFloat(gf[k], 64)
			y, err2 := strconv.ParseFloat(wf[k], 64)
			if err1 != nil || err2 != nil {
				return fmt.Sprintf("segment %d float %d unparsable", i, k)
			}
			tol := delayPinValueTol
			if (g[0] == 'h' && k == 1) || (g[0] == 'd' && k >= 5) {
				tol = delayPinLocationTol
			}
			if d := math.Abs(x - y); d > tol*math.Max(math.Abs(x), math.Abs(y)) {
				return fmt.Sprintf("segment %d float %d: %v vs %v (rel %.3g > %.0e)", i, k, x, y, d/math.Max(math.Abs(x), math.Abs(y)), tol)
			}
		}
	}
	return ""
}

// delayPinGolden holds the arm64 results on 3f1ba16.
var delayPinGolden = map[string]string{
	"cluster/YJINK6-A":                     "c:true,<nil>;d:0x1.632d1ed477ec9p+00,[0x1.725e6a58512ep-03 0x1.61e5591911302p+02],0x1.15fe54bf65fecp+06,0x1.09e8c0ae9bf65p+00,0x1.4418b6836c407p+01,0x1.8dfd866af7693p+05,true;",
	"cluster/5IT7DX-A":                     "c:true,<nil>;d:0x1.6f972e1969a2fp+00,[0x1.503c93a332848p-03 0x1.85d219dcca706p+02],0x1.1d6967d311947p+06,0x1.0fa0d90bbc4d3p+00,0x1.e636a52b70c3cp-01,0x1.3271a0093b6f9p+04,true;",
	"cluster/RMVV47":                       "c:true,<nil>;d:0x1.4950477b9f2b9p-02,[0x1.7226eb296f529p-01 0x1.621a6856750d1p+00],0x1.24508c9d7a4fp+04,0x1.7a232d9dd5986p+01,0x1.7f606a81df9ccp+00,0x1.8a138018d3981p+00,true;",
	"ddreach/0":                            "c:false,<nil>;d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"ddreach/2":                            "c:true,<nil>;d:0x1.1a3087c36ae09p-01,[0x1.22c655acfedaep-01 0x1.c2c4a6e11f6cp+00],0x1.ed0573f385ce7p+04,0x1.d034decadf877p+00,0x1.5a2ea96a1193p+01,0x1.7dc18cfef4935p+01,true;",
	"ddorigin/stable":                      "c:true,<nil>;d:0x1.47f03c7870cdap+00,[0x1.c0cae4d903da3p-03 0x1.240e047df0332p+02],0x1.051e3768c9abp+06,0x1.019d48aeaf6a3p+00,0x1.5ebb5ac86c85bp+00,0x1.3176694e43972p+07,true;",
	"ddorigin/3-rhp":                       "c:false,<nil>;d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"ddorigin/pair-cluster-4-rhp":          "c:false,<nil>;d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"pole-limit":                           "c:true,<nil>;d:0x1.868244da5935p+00,[0x1.13b0f9845b692p-03 0x1.db6e27ea39549p+02],0x1.2aaa70f0682fep+06,0x1.0091c3b8f04b3p+00,0x1.6d445fdacd163p-01,0x1.b7813deeeef85p+07,true;",
	"spread/0":                             "c:false,<nil>;d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"spread/1":                             "c:true,<nil>;d:0x1.42f8c42c0dfb9p+00,[0x1.cfa4ef4992b88p-03 0x1.1ab311b723f7cp+02],0x1.01f382241cc84p+06,0x1.01c0880bc1fc6p+00,0x1.0fdd936fb5a14p-01,0x1.8e6c70aaac146p+05,true;",
	"spread/2":                             "c:false,<nil>;d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"spread/3":                             "c:true,<nil>;d:0x1.9ed597e3eab33p-01,[0x1.b189100d8eadap-02 0x1.2e5536a6f88dp+01],0x1.60dae7ea5ecf7p+05,0x1.37c9696fe1552p+00,0x1.e89578f94f489p-01,0x1.4654b12de5388p+00,true;",
	"tiny-real/0":                          "c:true,<nil>;d:0x1.28952bc38d618p+00,[0x1.10ceaf87229c7p-02 0x1.e074cbd8f261p+01],0x1.e1506695a6764p+05,0x1.0ab68b43e5534p+00,0x1.494aa4dc0343bp+01,0x1.d93aad7ffe658p+05,true;",
	"ddtrust/0":                            "c:true,<nil>;d:0x1.072ca429121eep+00,[0x1.48b269acee93ep-02 0x1.8ec331b1f6b7ap+01],0x1.b342790269a2cp+05,0x1.247e44c700182p+00,0x1.af96e5fa67308p+01,0x1.d7ced80926625p+04,true;",
	"repro-internal":                       "h:0x1.f507507507508p+00,0x0p+00,<nil>;l:0x1.f507507507508p+00,<nil>;s:true,<nil>;",
	"first-order-loop/k1.5":                "d:0x1.9e8c9a94e3488p-01,[0x1.b1d303f90fdd8p-02 0x1.2e21acf730ed3p+01],0x1.60a2c4d9bf098p+05,0x1.b805a9d09e02cp+00,0x1.91a951d6089f9p+01,0x1.9db44eb5a5151p+01,true;",
	"first-order-loop/k-1.2":               "d:0x0p+00,[0x1p+00 0x1p+00],0x0p+00,+Inf,0x0p+00,0x0p+00,false;",
	"integrator-loop":                      "d:0x1.6726fd778a886p+00,[0x1.6754c0318d412p-03 0x1.6cc445b3ed59dp+02],0x1.18634aedf227fp+06,0x1.31b56638292e6p+00,0x1.b34ac67224364p+00,0x1.1f7168905a679p+01,true;",
	"light-resonance/io":                   "d:0x1.0013ecde77168p-01,[0x1.332672d7a3252p-01 0x1.aabc613b64ab1p+00],0x1.c14a7bbd2576fp+04,0x1.3feb15a46f09dp+01,0x1.2a67766e1ef8ap+05,0x1.2a6784fa3201fp+05,true;",
	"light-resonance/internal":             "d:0x1.0013ecde77168p-01,[0x1.332672d7a3252p-01 0x1.aabc613b64ab1p+00],0x1.c14a7bbd2576fp+04,0x1.3feb15a46f09dp+01,0x1.2a67766e1ef8ap+05,0x1.2a6784fa3201fp+05,true;",
	"mimo-resonances":                      "h:0x1.db0a8722d9496p+11,0x1.7199994c67d16p+04,<nil>;",
	"mimo-sensitivity/1":                   "h:0x1.26dfbc80772c6p+00,0x1.712faae1f8912p+01,<nil>;s:true,<nil>;",
	"mimo-sensitivity/1.8":                 "h:0x1.4e4965700e27cp+00,0x1.768caf5251cdbp+01,<nil>;s:true,<nil>;",
	"mimo3-sensitivity":                    "h:0x1.29688a34a2b4fp+00,0x1.7cff23eaa9dap+01,<nil>;s:true,<nil>;",
	"partly-commensurate/no-loop":          "h:0x1.b71a904ca1d1p+01,0x1.4dddfd4e9989ap+00,<nil>;",
	"partly-commensurate/neutral":          "h:0x1.b68515a3f9fbfp+01,0x1.4c956351d2489p+00,<nil>;",
	"constant-difference/diagonal-scaling": "h:0x1.663dbe70f1309p+01,0x1.4feff2dcb5e06p-01,<nil>;s:true,<nil>;",
	"constant-difference/equal-stein":      "h:0x1.47ecbf0631342p+00,0x1.00480f2b13d02p+00,<nil>;s:true,<nil>;",
	"neutral-pair/k3":                      "h:0x1.54f31d9452359p+02,0x0p+00,<nil>;s:true,<nil>;",
	"sharp-resonance/0.999":                "h:0x1.1eb245ff85dc9p+09,+Inf,<nil>;",
	"feedthrough-difference/commensurate":  "h:0x1.7dfccd2596d16p+02,0x1.908b557e122ccp+00,<nil>;",
	"expansion/near-90":                    "h:0x1.1df44a4b09495p+02,0x1.66cb46fece102p+06,<nil>;",
	"expansion/limit":                      "h:0x1.de56d3a776b44p+00,+Inf,<nil>;",
	"torus-cluster-siso":                   "h:0x1.29f15249f27e4p+02,0x1.ee18c7a8723b3p+07,<nil>;",
	"torus-seed/near-three-15":             "h:0x1.9b33dd4936d8cp+00,+Inf,<nil>;",
	"far-channel":                          "h:0x1.555555c03ac9ap+00,0x1.0fb48f3bf5cbcp+07,<nil>;",
	"delay-resonance":                      "h:0x1.74626e5fc644fp+00,0x1.45808126322a9p+00,<nil>;",
	"companion-feedback":                   "h:0x1p+00,0x0p+00,<nil>;",
	"neutral-parallel":                     "n:0,0,0;",
	"indent/double-integrator":             "h:0x1.c6f8cb7e66abbp+00,0x1.1abb69930ce87p+02,<nil>;s:true,<nil>;",
	"indent/triple-integrator":             "s:true,<nil>;",
	"static-internal-loop":                 "n:0,0,0;s:true,<nil>;h:0x1p-01,0x0p+00,<nil>;",
	"delayed-first-order":                  "p:0x1.3efe7372b9b15p+02,0x1.1ab6688283131p+02,0x1.66e284330ad11p-02,0x0p+00;d:0x1.a3c2334d3e17dp+00,[0x1.957dfd5ec5f8dp-04 0x1.433de4bac5e1bp+03],0x1.3ac53ba19a3ep+06,0x1.1c20b664d8dbp+00,0x1.6cf3ee79ec633p+00,0x1.6d02fda93871dp+00,true;",
	"pid-delay-loop/io":                    "d:0x1.e894cf17cd784p-01,[0x1.6a78e659c313fp-02 0x1.699b0873f2fdbp+01],0x1.981d8fabe9912p+05,0x1.80c103082c868p+00,0x1.6ac505f268784p-01,0x1.89904ba65dee4p-01,true;",
	"pid-delay-loop/internal":              "d:0x1.e894cf17cd782p-01,[0x1.6a78e659c3142p-02 0x1.699b0873f2fd8p+01],0x1.981d8fabe991p+05,0x1.80c103082c868p+00,0x1.6ac505eba690bp-01,0x1.89904ba65dee4p-01,true;",
}

func TestDelayPinMismatchDetectsChanges(t *testing.T) {
	want := "c:true,<nil>;d:0x1p+00,[0x1p+01 0x1p+02],0x1p+03,0x1p+00,0x1p+00,0x1p+00,true;"
	for _, c := range []struct {
		name, got string
		equal     bool
	}{
		{"same", want, true},
		{"location within spread", "c:true,<nil>;d:0x1p+00,[0x1p+01 0x1p+02],0x1p+03,0x1p+00,0x1.00001p+00,0x1.00001p+00,true;", true},
		{"value 1e-10 off", "c:true,<nil>;d:0x1.000000006dp+00,[0x1p+01 0x1p+02],0x1p+03,0x1p+00,0x1p+00,0x1p+00,true;", false},
		{"location 5e-4 off", "c:true,<nil>;d:0x1p+00,[0x1p+01 0x1p+02],0x1p+03,0x1p+00,0x1.002p+00,0x1p+00,true;", false},
		{"bool flipped", "c:false,<nil>;d:0x1p+00,[0x1p+01 0x1p+02],0x1p+03,0x1p+00,0x1p+00,0x1p+00,true;", false},
		{"error appeared", "c:true,<nil>;d:controlsys: overflow;", false},
	} {
		if got := delayPinMismatch(c.got, want) == ""; got != c.equal {
			t.Errorf("%s: equal = %v, want %v", c.name, got, c.equal)
		}
	}
}
