package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Under cheap control (R ≈ 1e-10·I) with a rank-deficient Q, Dare kept
// a K at Accuracy 0 as long as it was within 1e-6 of the gain of its X
// and of X one Newton step on: with X off by 1.6e-14 (inside newtonTol)
// the gain of X was off by 2.8e-7, 500 times its 5e-10 move per ulp of
// X, while the gain of X one step on was exact (7UDJQH, row of seed
// 6443). An Accuracy 0 K must lie within max(newtonTol, 4× the move of
// the 320-bit gain per ulp of the 320-bit doubling X) of that gain. The
// other two rows are resolved only to that move (3e-6 and 4e-7) since
// their K follows Q's rounding by as much: perturbing Q componentwise by
// eps moves the oracle K by 3e-6 and 8e-7.
func TestDare_CheapControlGainWithinItsUlpMove(t *testing.T) {
	cases := []struct {
		name       string
		n, m       int
		a, b, q, r []float64
	}{
		{"n=3 m=3", 3, 3,
			[]float64{0.17197346906216776, 0.6113183071610844, 0.14834599750738167, 0.18570044896504476, 0.3401904819149403, 0.3152400290077037, 0.24503370813828884, 0.3938150698094831, -0.1401672646841102},
			[]float64{0.2716812159269904, 0.2076395798363685, 0.07997427407116771, -1.136648694429242, 0.9249323471485567, -1.0398537891783404, 0.5341449813524776, -0.676570342075872, 0.5837798723513619},
			[]float64{0.00019322720219529536, -4.479257927821696e-05, -0.000279184271493441, -4.479257927821696e-05, 1.0383502610401105e-05, 6.471854620894408e-05, -0.000279184271493441, 6.471854620894408e-05, 0.0004033793201153182},
			[]float64{5.833577892753313e-11, 2.2305307343431172e-13, 2.830864400525519e-13, 2.2305307343431172e-13, 5.872220601618206e-11, -1.8068106796522542e-14, 2.830864400525519e-13, -1.8068106796522542e-14, 5.81649695238782e-11}},
		{"n=4 m=6", 4, 6,
			[]float64{-0.20450111787054473, -0.009227954767809536, 0.32504254254826104, 0.20319554179223695, 0.18525043808861372, -0.3597521597324114, 0.11329323909913377, -0.845083804250841, -0.16321216044152267, -0.2189942706282515, -0.4150199624845433, 0.6626780372083774, 0.34578561391313334, 0.4988663868897769, -0.5667321539042591, 0.3495976307577379},
			[]float64{0.5560080013187935, -1.5405414065919774, -2.153552439944834, 0.4247862460550443, 0.7702707032959887, -0.21239312302752214, 0.5929310203802683, -0.5424110275024778, 1.8488987158558672, 0.8453027066642307, 0.2712055137512389, -0.42265135333211534, 0.8836412865185461, -1.0145381103063271, -1.1007414797367387, 0.8136582186946879, 0.5072690551531636, -0.40682910934734395, 0.6592986669546026, 0.7627402197924883, -0.5548588544251669, 0.29350360934678876, -0.38137010989624415, -0.14675180467339438},
			[]float64{0.1189763326204459, 0.02854302857962945, -0.017440919798109654, -0.09650227436106233, 0.02854302857962945, 0.06199624140343476, -0.03225310336853232, -0.009071707236742086, -0.017440919798109654, -0.03225310336853232, 0.04293650171094039, 0.0023668290244816086, -0.09650227436106233, -0.009071707236742086, 0.0023668290244816086, 0.08268374187948734},
			[]float64{1.0508724348598854e-12, 3.189109323031854e-14, 1.8750038657803724e-14, -2.8775907155431518e-15, 1.639242840432385e-14, 7.0192792756480675e-15, 3.189109323031854e-14, 1.0645842617082708e-12, 1.8827405215447475e-14, -7.520597966220918e-15, -1.6699959167429525e-14, 2.8236343212330698e-14, 1.8750038657803724e-14, 1.8827405215447475e-14, 1.0593187222748133e-12, 1.5268930614757913e-14, 2.362734973587235e-14, -2.2137293678208834e-14, -2.8775907155431518e-15, -7.520597966220918e-15, 1.5268930614757913e-14, 1.0258242894220598e-12, 3.7614253765232004e-14, -9.680795275530988e-15, 1.639242840432385e-14, -1.6699959167429525e-14, 2.362734973587235e-14, 3.7614253765232004e-14, 1.0726442407057839e-12, -2.19689878925607e-14, 7.0192792756480675e-15, 2.8236343212330698e-14, -2.2137293678208834e-14, -9.680795275530988e-15, -2.19689878925607e-14, 1.034333425746392e-12}},
		{"n=3 m=4", 3, 4,
			[]float64{0.1790980321553196, 0.05894003604559097, 0.36228340737344567, -0.09426968563092339, 0.4328001871065155, 0.09361459362577322, 0.28840201747839195, 0.09310763648063938, 0.08559497752949735},
			[]float64{1.3415902524206402, -0.05964009772428536, 0.10023943494643327, 0.24236076013094232, 0.013778239951501514, -1.3603488779742472, -1.5799087352792451, -0.05952921555503632, 2.244461078614149, -1.0327099346004487, 0.6419108346138522, -0.32725254237049606},
			[]float64{0.05698145566745329, 0.05718913257352608, 0.1361393890117277, 0.05718913257352608, 1.7008845758750402, 1.0795668221035828, 0.1361393890117277, 1.0795668221035828, 0.8662582104692268},
			[]float64{1.4826365750802243e-10, 1.465475701993126e-12, -1.3756597570359592e-12, 3.013439715293845e-12, 1.465475701993126e-12, 1.3931979934673492e-10, 1.1157530186348366e-12, -9.29159031481137e-13, -1.3756597570359592e-12, 1.1157530186348366e-12, 1.4631889497442423e-10, -6.0899383641363576e-12, 3.013439715293845e-12, -9.29159031481137e-13, -6.0899383641363576e-12, 1.427854573317115e-10}},
	}
	for _, c := range cases {
		A, B := mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b)
		Q, R := mat.NewDense(c.n, c.n, c.q), mat.NewDense(c.m, c.m, c.r)
		wantX := dareDoublingBig(A, B, Q, R, nil)
		wantK := dareGainBig(A, B, R, mat.NewDense(c.n, c.m, nil), wantX)
		kTol := max(newtonTol, 4*dareGainUlpMove(A, B, R, nil, wantX, wantK))
		for _, ns := range []bool{false, true} {
			for _, solve := range []struct {
				name string
				f    func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
			}{{"Dare", Dare}, {"Dlqr", Dlqr}} {
				name := fmt.Sprintf("%s %s NoScaling=%v", solve.name, c.name, ns)
				res, err := solve.f(A, B, Q, R, &RiccatiOpts{NoScaling: ns})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				for _, e := range res.Eig {
					if !(cmplx.Abs(e) < 1) {
						t.Errorf("%s: Eig %v not stable", name, e)
					}
				}
				if res.Accuracy != 0 {
					continue
				}
				if rel := relErr(res.X, wantX); !(rel <= newtonTol) {
					t.Errorf("%s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", name, rel)
				}
				if rel := relErr(res.K, wantK); !(rel <= kTol) {
					t.Errorf("%s: Accuracy 0 with ‖K − K*‖/‖K*‖ = %.3g > %.3g", name, rel, kTol)
				}
			}
		}
	}
}

// A tolerance of gainTol on the distance of K from the gain of its X
// must not cost a K resolved only more coarsely. With S (n = 2, m = 3),
// the gain of an X off by 1.3e-9 was exact to 1.8e-9 while that of its
// Newton iterate x1 moved 1.7e-8 per ulp: judged by the moves about X
// alone, K was traded for the gain of X one step on, off by 1.8e-8. With
// E (n = 3), K one Newton step on was exact to 3.4e-7 (its
// eps-sensitivity 4.5e-7) while the gain of X a 2-ulp step from x1 moved
// 8e-7, 30 times its move under the ±eps perturbations of X: without the
// former 1e-6 floor as a fallback it came out at Accuracy 1 with the
// float64 gain off by 1.5 (7UDJQH review, draws 4175 and 6547 of the
// wave-39 reviewer sweep generator; amd64 leaves the E case at Accuracy 1
// on base as well). K must lie within max(newtonTol, 4× the move of the
// 320-bit gain per ulp of its X) of the 320-bit gain.
func TestDare_GainResolvedAtNewtonIterate(t *testing.T) {
	cases := []struct {
		name             string
		n, m             int
		a, b, q, r, s, e []float64
		kTol             float64
		mustCertify      bool
	}{
		{"E", 3, 3,
			[]float64{0.09881904021718858, 0.02842688923022688, -0.05073982054164841, -0.005430769410857691, 0.10001562226283889, -0.22968762772515325, -0.06504657668826087, 0.09597690435726397, 0.3049505863517156},
			[]float64{2.2626824076834735, 1.0174746318631867, 1.946458452559602, -0.15589531306336307, 0.35554626801235023, -1.851057747075576, -0.24110004138593077, 0.530520574494675, -0.33455149408030804},
			[]float64{3.011420285188916e+11, -1.3566965611288037e+11, -2.8291213103592995e+10, -1.3566965611288037e+11, 6.309696983950444e+10, 1.0008938260376373e+10, -2.8291213103592995e+10, 1.0008938260376373e+10, 6.449249774739811e+09},
			[]float64{7.816504827950286e-12, 3.1373991650654315e-14, -2.7957092369262666e-15, 3.1373991650654315e-14, 7.958716553401623e-12, 6.498686388958827e-14, -2.7957092369262666e-15, 6.498686388958827e-14, 7.915925302592696e-12},
			nil,
			[]float64{1.023034257867305, 0.0022117036130786722, -0.11645319695619356, -0.009589437543928887, 1.0739279582333912, 0.2167625748607436, 0.06248697628725246, -0.013532548546664679, 1.042329470990784},
			0, false},
		{"S", 2, 3,
			[]float64{0.04049315759658805, 0.13106055730326033, 0.06526289018949011, -0.2214261484702837},
			[]float64{2.053732806819227, 0.126339080840286, 2.053732806819227, 1.0487698057844066, -0.6931311091821057, 1.0487698057844066},
			[]float64{2.0470498244915469e+12, -2.4767447025600527e+12, -2.4767447025600527e+12, 2.996636548982355e+12},
			[]float64{2.928452535666982e-12, 4.7961226422024686e-14, -1.220370773730529e-14, 4.7961226422024686e-14, 3.0727116573265732e-12, -3.71752471965101e-14, -1.220370773730529e-14, -3.71752471965101e-14, 2.963337309232291e-12},
			[]float64{-0.00021246718919245902, -0.003693732197035752, -0.003577528872264808, 0.0002570660366954934, 0.004469081085486584, 0.004328485597480918},
			nil,
			newtonTol, true},
	}
	for _, c := range cases {
		A, B := mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b)
		Q, R := mat.NewDense(c.n, c.n, c.q), mat.NewDense(c.m, c.m, c.r)
		var S, E *mat.Dense
		if c.s != nil {
			S = mat.NewDense(c.n, c.m, c.s)
		}
		if c.e != nil {
			E = mat.NewDense(c.n, c.n, c.e)
		}
		var wantX, wantK *mat.Dense
		kTol := c.kTol
		if S == nil {
			var Y *mat.Dense
			wantX, Y, wantK = dareDescriptorGainBig(A, B, Q, R, E)
			kTol = max(newtonTol, 4*dareGainUlpMove(A, B, R, E, Y, wantK))
		} else {
			wantX = dareDoublingBig(A, B, Q, R, S)
			wantK = dareGainBig(A, B, R, S, wantX)
		}
		for _, ns := range []bool{false, true} {
			name := fmt.Sprintf("%s NoScaling=%v", c.name, ns)
			res, err := Dare(A, B, Q, R, &RiccatiOpts{NoScaling: ns, S: S, E: E})
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			if res.Accuracy != 0 {
				if c.mustCertify {
					t.Errorf("%s: Accuracy %d, want 0 (‖K − K*‖/‖K*‖ = %.3g)", name, res.Accuracy, relErr(res.K, wantK))
				}
				continue
			}
			if rel := relErr(res.X, wantX); !(rel <= newtonTol) {
				t.Errorf("%s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", name, rel)
			}
			if rel := relErr(res.K, wantK); !(rel <= kTol) {
				t.Errorf("%s: Accuracy 0 with ‖K − K*‖/‖K*‖ = %.3g > %.3g", name, rel, kTol)
			}
		}
	}
}
