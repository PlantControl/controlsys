package controlsys

import (
	"errors"
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Cheap control with a cross term S and nearly collinear inputs (σmin/σmax
// of B 1.9e-9, 1e-8, 2.9e-8 and 1.5e-7): the solve in B's singular-vector
// inputs either failed, or its gain mapped back to u left A − BK unstable
// in float64, and Dare then solved in the given inputs, where Newton's
// R + BᵀXB has lost its smallest eigenvalue and certified X ≈ Q off by
// 4.5e-2 (n = m = 3), X off by 1.7e-3 (n = 4, m = 2, E), on amd64 from the
// placed fallback X off by 0.16 (n = m = 2), and X off by 2.6e-3
// (n = m = 3 given-input gain), all with Accuracy 0 (D3WPCL; the doubling
// oracle moves 6.8e-9, 9.3e-7, 6.6e-8 and 2e-8 under eps perturbations).
// A given-input X must now pass the same Newton check in the inputs of B's
// singular vectors, from its X or from the Hewer step of its K, or is
// Accuracy 1 (the first case natively, the third on amd64), and the other
// scaling mode's and the placed gains are tried before the given inputs.
// The last case's K once certified its own cost as X: that shows only
// X = cost(K), not K = gain(X). Any K returned must stabilize: Hewer from
// it reaches the oracle. noStabilizing marks a case where both base and
// fix return ErrNoStabilizing on amd64.
func TestDare_CollinearCrossTermGivenInputXIsNotCertifiedWrong(t *testing.T) {
	cases := []struct {
		name             string
		n, m             int
		a, b, q, r, s, e []float64
		noStabilizing    bool
	}{
		{"n=m=3", 3, 3,
			[]float64{14.709308328720477, 22.029180781782163, 14.34126889213351, -11.23171695506349, -9.4374591791453124, 21.738358860804802, -37.622418463278542, 28.700305746106388, 1.8178780847653089},
			[]float64{0.25909557847906678, 1.7624663776759129, 1.7624663709718, -0.11109258602281202, -0.73733403364247341, -0.73733404057487917, -1.3909602237549459, -1.1726908269551395, -1.1726908303687378},
			[]float64{76746576563117.656, -24432091016600.645, 10245214450161.965, -24432091016600.645, 7777898352932.2031, -3261539773374.6284, 10245214450161.965, -3261539773374.6284, 1367675586721.2275},
			[]float64{1.0827374498323734e-12, -5.8274765268520017e-14, -2.713286584750832e-14, -5.8274765268520017e-14, 1.0661429680892122e-12, -1.0120969940174057e-13, -2.713286584750832e-14, -1.0120969940174057e-13, 1.6069801643513065e-12},
			[]float64{-1.0667956669820327, 1.1939492205041986, -1.8955822712572745, 0.33961188379887569, -0.38009091700851599, 0.60345409096610569, -0.14241091851485974, 0.15938516656558821, -0.2530490265801274}, nil, false},
		{"n=4 m=2 E", 4, 2,
			[]float64{-0.48240735387882944, 0.94266002707410135, 1.8606339272567889, -0.28964763498131291, -0.49090160180590781, 0.064263785142072427, -0.26535207615476508, -0.51329621851311436, -1.5642186748695732, -0.79299454150183268, -0.44347152634129128, 1.2235854147793686, -0.8114179568672264, -1.1987613404236039, -1.1961718564464709, -0.53081521571611423},
			[]float64{-0.67549550082189036, -0.67549551496200377, 0.79703175144653771, 0.79703166743881471, -0.62594380604344479, -0.62594379689059043, 1.0903390885854725, 1.0903390993682323},
			[]float64{10877587254763.496, -68787567109525.617, 38890420326017.047, 12955354644213.082, -68787567109525.617, 434998063267695.56, -245934813983868.31, -81926929763422.766, 38890420326017.047, -245934813983868.31, 139044142576005.91, 46319020549839.609, 12955354644213.082, -81926929763422.766, 46319020549839.609, 15430003917811.158},
			[]float64{1.0165663748013665e-12, 3.2184072635856789e-14, 3.2184072635856789e-14, 1.0958449220694888e-12},
			[]float64{-0.093649750623230485, 0.10089435827614486, 0.59222128479271097, -0.63803463155151585, -0.33482413059945976, 0.36072556652275389, -0.11153813177957646, 0.12016655465742458},
			[]float64{2, 0.16209069176044191, -0.12484405096414272, -0.29699774898013359, -0.29699774898013359, 2.1000000000000001, 0.085098655638967888, 0.28805108599510981, 0.28805108599510981, 0.22617067630299137, 2.2000000000000002, -0.27333907856540307, -0.27333907856540307, -0.25172145872293572, 0.0013277093964152358, 2.2999999999999998}, false},
		{"n=m=2", 2, 2,
			[]float64{-0.067128624834135675, 0.33157324552917489, -0.39076065696643097, 0.5236438798406271},
			[]float64{-0.75489091043558654, -0.75489097130355443, -0.67337283991977559, -0.67337281549035566},
			[]float64{124478879260.39685, -2949024693220.0093, -2949024693220.0093, 69865238929639.477},
			[]float64{1.2067434011042394e-12, 8.378628083754024e-14, 8.378628083754024e-14, 1.0789895580252643e-12},
			[]float64{-0.21827250029614315, -0.027193214291023157, 5.1710860275222901, 0.64423347491110239}, nil, false},
		{"n=m=3 given-input gain", 3, 3,
			[]float64{-2.8399534788419056, 1.5747770850316118, -1.1172601108100324, 0.40990242286723844, -0.7897453843666609, -0.6862802170755049, 0.27505235960400676, 0.7226820118056213, 1.7081471449059176},
			[]float64{1.6118533073409205, 0.5543013708721011, 0.5543010447780001, 0.3120603088150442, 2.039767049005779, 2.039766856587498, -0.82081151650893, -1.1343223586380966, -1.134322299534502},
			[]float64{1.8054843166805974e12, -4.772769635834572e12, 6.238166596675897e11, -4.772769635834572e12, 1.261674210420422e13, -1.6490496118310688e12, 6.238166596675897e11, -1.6490496118310688e12, 2.1553619784096542e11},
			[]float64{1.4616753743848063e-12, -3.9283477219729515e-14, 6.0428705914753066e-15, -3.9283477219729515e-14, 1.0922534435077715e-12, 1.1694541948936462e-13, 6.0428705914753066e-15, 1.1694541948936462e-13, 1.5585283669069314e-12},
			[]float64{-0.056802161936153, 0.08481541338045245, -0.02242232941606692, 0.15015562928680343, -0.22420822263204046, 0.05927307815026371, -0.019625833685475883, 0.02930475073890086, -0.00774718589858107}, nil, true},
	}
	for _, c := range cases {
		A, B := mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b)
		Q, R, S := mat.NewDense(c.n, c.n, c.q), mat.NewDense(c.m, c.m, c.r), mat.NewDense(c.n, c.m, c.s)
		var E *mat.Dense
		Ae, Be := A, B
		if c.e != nil {
			E = mat.NewDense(c.n, c.n, c.e)
			var lu mat.LU
			lu.Factorize(E)
			var ea, eb mat.Dense
			if err := lu.SolveTo(&ea, false, A); err != nil {
				t.Fatal(err)
			}
			if err := lu.SolveTo(&eb, false, B); err != nil {
				t.Fatal(err)
			}
			Ae, Be = &ea, &eb
		}
		want := dareDoublingBig(Ae, Be, Q, R, S)
		for _, noScaling := range []bool{false, true} {
			name := fmt.Sprintf("%s NoScaling=%v", c.name, noScaling)
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: noScaling})
			if err != nil {
				if !c.noStabilizing || !errors.Is(err, ErrNoStabilizing) {
					t.Errorf("%s: %v", name, err)
				}
				continue
			}
			X := res.X
			if E != nil {
				var ex mat.Dense
				ex.Mul(E.T(), res.X)
				ex.Mul(&ex, E)
				X = &ex
			}
			if rel := relErr(X, want); res.Accuracy == 0 && !(rel <= 1e-8) {
				t.Errorf("%s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", name, rel)
			}
			if rel := relErr(hewerDareBig(Ae, Be, Q, R, S, res.K, 30), want); !(rel <= 1e-12) {
				t.Errorf("%s: Hewer from K reaches %.3g of X*", name, rel)
			}
		}
	}
}

// Under GOARCH=amd64 the given-input solve of r1 (TestDare_NearlyCollinear
// InputsAreCertified) has Newton certify X off by 1.04e-5 (5.46e-7 with
// NoScaling) at Accuracy 0 while the oracle moves 1.9e-15 under eps
// perturbations (D3WPCL item 4). Dare solves r1 in the rotated inputs, but
// were that to fail, its given-input fallback must not keep such an X:
// recertified in B's singular-vector inputs, from X or from K, it is
// refined to the oracle or rejected.
func TestDare_ReducedInputsCertifyRejectsGivenInputNewtonX(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.4620991642104912, -0.8784792442575562, -1.2071382595750249, 0.16268284639648423})
	B := mat.NewDense(2, 2, []float64{0.7873745903512255, -0.20096102363846285, -0.8592447854231653, 0.21930438011983636})
	Q := mat.NewDense(2, 2, []float64{6.441166398562846e13, 4.786743416308757e13, 4.786743416308757e13, 1.614517464168949e14})
	R := mat.NewDense(2, 2, []float64{1.0193412194411262e-12, 2.999265069241783e-14, 2.999265069241783e-14, 6.403938765630009e-13})
	S := mat.NewDense(2, 2, []float64{2.9394250706403358, 0.8088194867064437, 2.625832947633263, 2.2414533619527712})
	want := dareDoublingBig(A, B, Q, R, S)
	for _, noScaling := range []bool{false, true} {
		o := &RiccatiOpts{S: S, NoScaling: noScaling}
		res, err := dareSolve(A, B, Q, R, o, true)
		if err != nil {
			t.Fatalf("NoScaling=%v: %v", noScaling, err)
		}
		red, _ := reduceInputs(A, B, Q, R, o)
		if red == nil {
			t.Fatalf("NoScaling=%v: no input reduction", noScaling)
		}
		for _, k0 := range []*mat.Dense{nil, res.K} {
			c, ok := red.certify(A, Q, o, res, k0)
			if !ok {
				continue
			}
			if rel := relErr(c.X, want); !(rel <= 1e-8) {
				t.Errorf("NoScaling=%v k0=%v: certified ‖X − X*‖/‖X*‖ = %.3g", noScaling, k0 != nil, rel)
			}
		}
	}
}
