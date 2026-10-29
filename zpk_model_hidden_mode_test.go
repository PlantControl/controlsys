package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkHiddenModeClustered returns stable discrete SISO models with five
// poles clustered near z = 1 and one hidden mode, all mixed by an
// orthogonal similarity so that Sminreal keeps six states (ZAR4NK, rv8
// seeds 73 and 87).
func zpkHiddenModeClustered(t *testing.T) []struct {
	name string
	sys  *System
} {
	t.Helper()
	cases := []struct {
		name    string
		a, b, c []float64
		dt      float64
	}{
		{
			name: "uncontrollable",
			a: []float64{
				1.1358126790060064, 0.09648917456530205, -0.05741187672279773, -0.05904075957926131, 0.011868135419337428, -0.03792132183129446,
				0.025420134265492476, 1.0179630324802333, -0.010738847809258996, -0.011053513882006817, 0.0022466912847786814, -0.007145355305782679,
				0.053528324621306626, 0.03800158639069512, 0.9772732475677156, -0.02320566734125002, 0.004618256371628698, -0.014974851823109071,
				0.18614322073306525, 0.13215071919121313, -0.07865894959209287, 0.9189539883416278, 0.01630206739837113, -0.05197996818367307,
				0.1768769263120668, 0.12549894351924892, -0.07460820616630337, -0.07683097230363184, 1.0153622954326849, -0.04939642921029058,
				0.23666020544380342, 0.1680307415546318, -0.09986963436239649, -0.10277107137799506, 0.02074989175052766, 0.9338484281579731,
			},
			b:  []float64{0.3012043125863701, 0.07173928908845786, -1.4124024012700453, 1.3372795641569404, 0.06205290623513738, 1.3352217818820111},
			c:  []float64{0.8268525367274729, 1.227683334350517, 1.4212790162223448, 0.6184556327490655, -0.8602942622208246, 0.844120237823973},
			dt: 1e-4,
		},
		{
			name: "unobservable",
			a: []float64{
				0.9841841244179377, -0.027513825997221287, -0.0341427588558871, 0.022726884600243566, 0.018675537694266286, 0.025258493619547732,
				-0.056483820743678165, 0.8971292011565452, -0.12559757640738095, 0.08110510447372468, 0.0670242365718116, 0.09369537308177989,
				0.054964691290695075, 0.09997535194417936, 1.1218802770519867, -0.0787645934861993, -0.06552810919398054, -0.09127406997503076,
				0.03256019503327083, 0.05942863772269774, 0.07313731047628037, 0.9519753647440349, -0.03991964383405736, -0.05324776418544562,
				-0.0159449939622054, -0.028090553287099626, -0.0344366043498952, 0.022140121738591725, 1.0175119995540711, 0.026407385606660956,
				-0.01405955844206707, -0.02714622048289104, -0.03274070089997452, 0.02021506719934149, 0.01704892697673855, 1.0233815136282627,
			},
			b:  []float64{-0.22132610470976224, -1.2281051298203074, -0.4417288579014369, -0.4251424899203741, -0.0474450073372063, -0.2210965175903602},
			c:  []float64{-0.18066958000958028, 1.1227985905141118, 0.32449677570342333, 0.5143710565831464, -0.771761339676508, -0.9481606692421359},
			dt: 1e-3,
		},
	}
	out := make([]struct {
		name string
		sys  *System
	}, len(cases))
	for k, tc := range cases {
		sys, err := New(mat.NewDense(6, 6, tc.a), mat.NewDense(6, 1, tc.b), mat.NewDense(1, 6, tc.c), mat.NewDense(1, 1, nil), tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		out[k].name, out[k].sys = tc.name, sys
	}
	return out
}

// TestZPKModelHiddenModeClusteredPolesStayStable checks ZPKModel of
// clustered discrete models with a hidden mode against a 320-bit oracle of
// the stored matrices. MATLAB zpk(ss) cancels the hidden mode and takes the
// remaining poles from eig of the minimal part; roots of the
// transfer-function polynomials put a pole outside the unit circle (ZAR4NK).
func TestZPKModelHiddenModeClusteredPolesStayStable(t *testing.T) {
	for _, tc := range zpkHiddenModeClustered(t) {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys
			res, err := sys.ZPKModel(nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.MinimalOrder != 5 || len(res.RowDegrees) != 1 || res.RowDegrees[0] != 5 {
				t.Fatalf("MinimalOrder %d RowDegrees %v, want 5 [5]", res.MinimalOrder, res.RowDegrees)
			}
			z := res.ZPK
			poles, zeros, gain := z.Poles[0][0], z.Zeros[0][0], z.Gain[0][0]
			if len(poles) != 5 || len(zeros) != 4 {
				t.Fatalf("got %d poles and %d zeros, want 5 and 4", len(poles), len(zeros))
			}
			for _, p := range poles {
				if !(cmplx.Abs(p) < 1) {
					t.Errorf("pole %v outside the unit circle", p)
				}
			}

			o := newStepOraclePencil(sys)
			starts, err := sys.Poles()
			if err != nil {
				t.Fatal(err)
			}
			modes, ok := newStepOracleModes(sys, starts)
			if !ok {
				t.Fatal("oracle poles unresolved")
			}
			for _, p := range poles {
				if d := nearestRootDistance(modes.poles, p); !(d <= 1e-9) {
					t.Errorf("pole %v: nearest oracle eigenvalue %.3g away", p, d)
				}
			}

			for k, want := range matchedOracleZeros(t, o, zeros) {
				if d := cmplx.Abs(zeros[k] - want); !(d <= 1e-6*math.Max(1, cmplx.Abs(want))) {
					t.Errorf("zero %v: oracle %v (%.3g away)", zeros[k], want, d)
				}
			}

			markov := mat.Dot(sys.C.RowView(0), sys.B.ColView(0))
			if rel := math.Abs(gain-markov) / math.Abs(markov); !(rel <= 1e-9) {
				t.Errorf("gain %v, want C·B = %v (rel %.3g)", gain, markov, rel)
			}

			for _, s := range []complex128{cmplx.Exp(1e-4i), cmplx.Exp(1e-3i), cmplx.Exp(1e-2i), cmplx.Exp(0.1i), cmplx.Exp(1i), cmplx.Exp(3i), 0.999 + 0.001i} {
				got, err := z.Eval(s)
				if err != nil {
					t.Fatal(err)
				}
				want := matchedOracleTransfer(o, s)
				if rel := cmplx.Abs(got[0][0]-want) / cmplx.Abs(want); !(rel <= 1e-6) {
					t.Errorf("H(%.6g) = %v, oracle %v (rel %.3g)", s, got[0][0], want, rel)
				}
			}
		})
	}
}
