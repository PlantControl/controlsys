package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDlqrDescriptorStalledRefinementNotCertified: rotated E of condition
// 1e9 and 1e10 (O2MVDD; rv40Gen seed 6299, r2bGen seed 21936 NoScaling).
// The double-double Hewer refinement of X ~ 1e20 stalls at a noise floor
// of EᵀXE steps ~3e-9 and met newtonRefineTol once: EᵀXE was 1.3e-8 and
// 2.7e-8 off the 320-bit oracle at Accuracy 0 (base: Accuracy 1).
func TestDlqrDescriptorStalledRefinementNotCertified(t *testing.T) {
	for i, m := range []struct {
		n, m          int
		a, b, e, q, r []float64
		noScaling     bool
	}{
		{4, 1,
			[]float64{0.3485508549286703, 0.11770551141869293, 0.10448508308596577, 0.1923847739190919, -0.33789839064234223, 0.40840393513772544, -0.03859713013726715, -0.13561516597754408, -0.4979520064893839, 0.20549533564107147, -0.2685494513011114, -0.4956918565895243, 0.23278212274164803, -0.9203564433908897, 1.021212646609918, -0.6993620829910131},
			[]float64{0.1550663925419704, -2.1406600096971715, 0.5146673469144831, 1.461102571554982},
			[]float64{-0.17017187452487698, 1.9445601210686252, -0.5038847276067643, 0.6704101732539725, -0.2584633791783249, 0.7039656032666773, -0.8202121358485249, -0.035934026909570586, -1.0937956300042222, 0.9079103291066428, 0.49645974665114156, 0.47188901270831957, 1.2240603166505628, -0.8206782120173114, -0.051130263871029, -0.2382351371336621},
			nil, nil, false},
		{3, 1,
			[]float64{-0.37888674180449616, -1.0564868525010445, 0.296419528205748, 0.0854169452297714, -1.2290397106043665, 0.22491365198650545, -1.1729333419894443, 0.15034138888378432, -0.05980508970924318},
			[]float64{-0.7796290442607897, 1.8879580680322232, -1.2076595645870352},
			[]float64{0.5258729484709597, 0.016342513049244767, -0.4855146037087744, 0.09020999825674433, 1.2691998313223773, -0.18866592031869156, -0.5702314831313473, 0.4913046416891816, 0.4841118786669185},
			[]float64{902.8076154497909, -148.7519177487837, 795.2550321036234, -148.7519177487837, 175.64638317518853, -598.2888275070236, 795.2550321036234, -598.2888275070236, 2145.1023714968433},
			[]float64{25.680547128486573}, true},
	} {
		A, B, E := mat.NewDense(m.n, m.n, m.a), mat.NewDense(m.n, m.m, m.b), mat.NewDense(m.n, m.n, m.e)
		Q, R := eyeDense(m.n), eyeDense(m.m)
		if m.q != nil {
			Q, R = mat.NewDense(m.n, m.n, m.q), mat.NewDense(m.m, m.m, m.r)
		}
		res, err := Dlqr(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: m.noScaling})
		if err != nil {
			t.Fatalf("model %d Dlqr: %v", i, err)
		}
		Y, _, Ko, ok := nearSingularOracle(false, A, B, Q, R, nil, E, res.K)
		if !ok {
			t.Fatalf("model %d: oracle did not converge", i)
		}
		eX, eK := relErr(res.X, Y), relErr(res.K, Ko)
		t.Logf("model %d: Accuracy %d, EᵀXE off the oracle by %.3g, K by %.3g", i, res.Accuracy, eX, eK)
		if res.Accuracy == 0 && (eX > newtonTol || eK > newtonTol) {
			t.Errorf("model %d: Accuracy 0 with EᵀXE off the oracle by %.3g, K by %.3g", i, eX, eK)
		}
	}
}
