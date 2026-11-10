package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// assertReducedUnchanged fails unless res is sys unreduced: the order and
// the entries of A, B and C.
func assertReducedUnchanged(t *testing.T, name string, sys *System, res *ReduceResult) {
	t.Helper()
	n, _, _ := sys.Dims()
	if res.Order != n {
		t.Fatalf("%s: order %d, want %d (unreduced)", name, res.Order, n)
	}
	for _, pair := range [][2]*mat.Dense{{sys.A, res.Sys.A}, {sys.B, res.Sys.B}, {sys.C, res.Sys.C}} {
		if !mat.Equal(pair[0], pair[1]) {
			t.Fatalf("%s: unreduced result changed the realization", name)
		}
	}
}

// Y5H2MH seeds 403, 62, 188, 301, 406: the Equalize staircase cut removed
// the planted state with an error of 1e-21 at the matched-C2D zero z = -1,
// where the model's response is 1e-42..1e-92 and determined to 1e-27
// (certified wrong by 1e21..1e67). The cut is not verified at enough
// points, so the model comes back unreduced.
func TestReduceEqualizeRefusesCutWrongAtNyquistZero(t *testing.T) {
	lits := reduceVerifyLiterals(t)
	for _, name := range []string{"403-unc-false", "62-unc-false", "188-unc-false", "301-unc-false", "406-unc-false"} {
		sys := lits[name]
		for _, opts := range []*ReduceOpts{{}, {Equalize: true}} {
			res, err := sys.Reduce(opts)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			assertReducedUnchanged(t, name, sys, res)
		}
	}
}

// Y5H2MH fz 2244: eigenvalues to 3e21 at dt = 100 leave the response
// undetermined by the data at every probe; no cut may be accepted on
// that (with radix-2 Equalize the staircase cut to order 3 was certified
// wrong by 0.7).
func TestReduceEqualizeKeepsModelWithUndeterminedResponse(t *testing.T) {
	sys := reduceVerifyLiterals(t)["fz2244"]
	res, err := sys.Reduce(&ReduceOpts{Equalize: true})
	if err != nil {
		t.Fatal(err)
	}
	assertReducedUnchanged(t, "fz2244", sys, res)
}

// Y5H2MH fz 17305: the Equalize cut to order 2 is off by 1.1e-4 relative,
// so it is refused at the default tolerance and accepted when Tol admits
// that change.
func TestReduceTolAdmitsCutRefusedAtDefault(t *testing.T) {
	sys := reduceVerifyLiterals(t)["fz17305"]
	res, err := sys.Reduce(&ReduceOpts{Equalize: true})
	if err != nil {
		t.Fatal(err)
	}
	assertReducedUnchanged(t, "fz17305 default", sys, res)
	res, err = sys.Reduce(&ReduceOpts{Equalize: true, Tol: 1e-4})
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 2 {
		t.Fatalf("Tol 1e-4: order %d, want 2", res.Order)
	}
}

// Y5H2MH fz 10878, one of the ~1220 census cuts that matched the model
// only within the model's own rounding bound (here 93% off): unreduced with
// a nil error.
func TestReduceReturnsModelWhenCutUnverifiable(t *testing.T) {
	sys := reduceVerifyLiterals(t)["fz10878"]
	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}
	assertReducedUnchanged(t, "fz10878", sys, res)
}

// An exact axis pole, kept or removed, puts probes where both responses
// are rounding noise (a kept integrator that the rotated cut carries only
// to rounding); the correct cut must still be accepted.
func TestReduceAcceptsCutBesideExactAxisPole(t *testing.T) {
	q := mat.NewDense(3, 3, []float64{7.0 / 9, -4.0 / 9, -4.0 / 9, -4.0 / 9, 1.0 / 9, -8.0 / 9, -4.0 / 9, -8.0 / 9, 1.0 / 9})
	for _, tc := range []struct {
		name string
		d    [3]float64
		dt   float64
	}{
		{"discrete kept integrator, removed 0.5", [3]float64{1, 0.6, 0.5}, 0.1},
		{"discrete kept integrator, removed -0.5", [3]float64{1, 0.6, -0.5}, 0.1},
		{"discrete kept integrator, removed -1", [3]float64{1, 0.6, -1}, 0.1},
		{"discrete removed integrator", [3]float64{0.5, 0.6, 1}, 0.1},
		{"continuous kept integrator, removed -3", [3]float64{0, -1, -3}, 0},
		{"continuous removed integrator", [3]float64{-2, -1, 0}, 0},
	} {
		for _, rot := range []bool{false, true} {
			// The third state is uncontrollable: no input reaches it.
			A := mat.NewDense(3, 3, []float64{tc.d[0], 0.7, 0.2, 0, tc.d[1], 0.3, 0, 0, tc.d[2]})
			B := mat.NewDense(3, 1, []float64{1, 1, 0})
			C := mat.NewDense(1, 3, []float64{1, -0.5, 2})
			if rot {
				var tmp, a, b, c mat.Dense
				tmp.Mul(q, A)
				a.Mul(&tmp, q.T())
				b.Mul(q, B)
				c.Mul(C, q.T())
				A, B, C = &a, &b, &c
			}
			sys, err := New(A, B, C, mat.NewDense(1, 1, nil), tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			pts := []complex128{0.37i, 2.1i}
			if tc.dt != 0 {
				pts = []complex128{cmplx.Rect(1, 0.7), cmplx.Rect(1, 2)}
			}
			for _, opts := range []*ReduceOpts{{}, {Equalize: true}} {
				res, err := sys.Reduce(opts)
				if err != nil {
					t.Fatalf("%s rot=%v: %v", tc.name, rot, err)
				}
				if res.Order != 2 {
					t.Fatalf("%s rot=%v eq=%v: order %d, want 2", tc.name, rot, opts.Equalize, res.Order)
				}
				for _, s := range pts {
					g, _ := sys.EvalFr(s)
					r, _ := res.Sys.EvalFr(s)
					if d := cmplx.Abs(g[0][0] - r[0][0]); d > 1e-10*cmplx.Abs(g[0][0]) {
						t.Errorf("%s rot=%v eq=%v: |G-R| = %g at %v, |G| = %g", tc.name, rot, opts.Equalize, d, s, cmplx.Abs(g[0][0]))
					}
				}
			}
		}
	}
}

// reduceVerifyLiterals are frozen models of the Reduce verification
// census (Y5H2MH): fz rows decoded before QS7HIC, and matched-hid rows,
// a matched C2D model with one planted hidden state (zzCefHidden); the
// -amd64 rows were generated on amd64, where their C2D differs.
func reduceVerifyLiterals(t *testing.T) map[string]*System {
	t.Helper()
	type lit struct {
		dt         float64
		n, m, p    int
		a, b, c, d []float64
	}
	lits := map[string]lit{
		"fz2244": {dt: 100, n: 6, m: 1, p: 2,
			a: []float64{
				-6.076573937518291e+22, 5.248283159199702e+22, -1.8956265882896455e+22, 8.758981327717419e+22, -2.246048492285362e+23, 3.1951706125006143e+21,
				-6.41428999107562e+22, 5.5399655273731375e+22, -2.000979298056389e+22, 9.245776788053272e+22, -2.3708765023964e+23, 3.372747717756205e+21,
				2.7614075899125623e+22, -2.3850033092402757e+22, 8.614389789982418e+21, -3.9803872654167774e+22, 1.0206829403677928e+23, -1.4519972061803302e+21,
				1.7270995840921097, -1.4916806336356943, 0.3356173008894234, -2.6918282558414135, 6.157613309945193, -2.007308029346734,
				-0.03664996377922845, 0.03165424952705512, -0.007121976077451126, 0.07697965481396914, -0.3034401969083944, -0.5786224809025085,
				0.7424037881498827, -0.6412075848556832, 0.14426704623400485, -1.5593463526627285, 6.146667784414401, 1.730737323238318,
			},
			b: []float64{
				-0.902422060833894,
				-0.6565489093423328,
				0.5511699324662456,
				0.9514235319579598,
				-0.042160571817001564,
				-0.9739793391083408,
			},
			c: []float64{
				-0.15677561169918452, -0.3890912423072425, 0.4083538900909663, 0.1250581377981722, -0.48921387249402115, 0.1912573440862459,
				-0.4192966514917397, -0.10523030535940484, 0.4873363711728522, -0.20987122334720054, -0.35163801738679207, 0.43723302801463493,
			},
			d: []float64{
				4.656393986181331,
				-3.845113835012964,
			},
		},
		"fz17305": {dt: 100, n: 3, m: 3, p: 1,
			a: []float64{
				7.09770963042327e+11, -7.160113346743568e+11, 8.82547179016819e+09,
				3.7198323877663794e+11, -3.752537496480638e+11, 4.625333736064431e+09,
				7.680446813604221e+11, -7.74797400887224e+11, 9.55006195221161e+09,
			},
			b: []float64{
				0.2995279776401208, 0.6723617487801199, -0.9784063971473038,
				0.9994422200290952, -0.5333916830027213, -0.4608807932493233,
				0.2351712686313591, -0.9577250495827699, 0.7318362524877744,
			},
			c: []float64{
				1.3436484559284246e-06, -2.785244970153597e-06, 4.5723282409099066e-07,
			},
			d: []float64{
				-0.19319638154967314, -0.5412482581652248, -0.8038489357863109,
			},
		},
		"fz10878": {dt: 100, n: 3, m: 3, p: 1,
			a: []float64{
				1.1956112818154686e+80, 3.316266815704275e+79, -2.064561468528301e+80,
				2.6960468872085387e+79, 7.478024807574887e+78, -4.6554884562685985e+79,
				-2.153002482898052e+80, -5.971782632664805e+79, 3.7177685050676245e+80,
			},
			b: []float64{
				-0.8041337788322646, 0.9190972076253325, -0.12387151120424594,
				0.35768074975452635, 0.6255290921531041, -0.9892726007292874,
				0.9954921419940651, -0.5844406087822195, -0.4053870125667214,
			},
			c: []float64{
				0.363750964680997, -0.06083850857285808, -0.3244140545107153,
			},
			d: []float64{
				-2.234434632410065, -1.5405064481482724, -0.6033661683885425,
			},
		},
		"403-unc-false": {dt: 0.001, n: 6, m: 1, p: 1,
			a: []float64{
				0.9992718351676878, -0.000121172368855893, 1, -0, 0, -1.1970254150404576,
				0, 0.9990201532489162, 1, -0, 0, -1.4282313739088437,
				0, 0, 0.9981368868248504, -0.5961539521459532, 1.9995461812752986, -0.9499810610547582,
				0, 0, 0, 0.9995980353929118, 0.0001739779794643965, 0.7311332993988631,
				0, 0, 0, -0.0001739779794643965, 0.9995980353929118, 1.2125041161418564,
				0, 0, 0, 0, 0, 0.9988357087664415,
			},
			b: []float64{
				0,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				6.912130290475256e-10, -4.1893212638951056e-14, 3.4573238960751454e-10, -0, 0, -0.4249513059760568,
			},
			d: []float64{
				0,
			},
		},
		"62-unc-false": {dt: 0.01, n: 6, m: 1, p: 1,
			a: []float64{
				0.9847755341576271, 1225.5779738939873, 3.9784373097804946, 243.3624021913897, 3.9797370909359016, 0.8816296841567932,
				0, 0.9892186548902473, 0.0032286647744775953, 0, 0, -0.3233191646539401,
				0, -0.0032286647744775953, 0.9892186548902473, 243.3624021913897, 3.9797370909359016, 0.8040991516892575,
				0, 0, 0, 0.9898685454679508, 0.01626920225001366, 1.1593174580727887,
				0, 0, 0, -0.01626920225001366, 0.9898685454679508, 0.4049221561745098,
				0, 0, 0, 0, 0, 0.997625675408712,
			},
			b: []float64{
				1,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				6.584266481261553e-12, 0, 0, 0, 0, 0.6926883367138308,
			},
			d: []float64{
				0,
			},
		},
		"188-unc-false": {dt: 0.01, n: 8, m: 1, p: 1,
			a: []float64{
				0.9971670943308685, -0.0020465794771237222, 1.9856297798046678, 1.9843404987370603, 1, 0, 0, 1.0232664577594073,
				0, 0.9883954188738843, 1.9856297798046678, 1.9843404987370603, 1, 0, 0, -0.11487245560861822,
				0, 0, 0.9856297798046678, 1.9843404987370603, 1, 0, 0, -0.8563772993939605,
				0, 0, 0, 0.9843404987370603, 1, 0, 0, 0.3447690068799303,
				0, 0, 0, 0, 0.9799685211387175, 206.23460709231367, 3.9873447288390014, 1.3936966692970376,
				0, 0, 0, 0, 0, 0.9936723644195007, 0.019271053385198758, 1.6592409100753478,
				0, 0, 0, 0, 0, -0.019271053385198758, 0.9936723644195007, -0.45337571034539004,
				0, 0, 0, 0, 0, 0, 0, 0.989728368717362,
			},
			b: []float64{
				0,
				0,
				0,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				1.0450170841209183e-13, -1.0708720986223237e-16, 1.0389801877495629e-13, 1.0383055718169713e-13, 5.232497005820342e-14, 0, 0, -0.09605895767730577,
			},
			d: []float64{
				0,
			},
		},
		"301-unc-false": {dt: 0.001, n: 8, m: 1, p: 1,
			a: []float64{
				0.9997168046964238, 13290.796169034027, 1, 0, 0, 0, 0, 0.1709664279987535,
				0, 0.998307702236476, 0.00015035274612760184, 0, 0, 0, 0, 0.7617141143639816,
				0, -0.00015035274612760184, 0.998307702236476, 2461.466667835915, 3.9961860577003723, 2228.3917940795563, 3.9991290316905204, -0.873518333411397,
				0, 0, 0, 0.9980930288501861, 0.0016219488865683303, 0, 0, -1.2859046956674398,
				0, 0, 0, -0.0016219488865683303, 0.9980930288501861, 2228.3917940795563, 3.9991290316905204, 0.5867843384865203,
				0, 0, 0, 0, 0, 0.9995645158452603, 0.001794233421778242, 0.6287390400073067,
				0, 0, 0, 0, 0, -0.001794233421778242, 0.9995645158452603, -0.9617190796380695,
				0, 0, 0, 0, 0, 0, 0, 0.9994417385318478,
			},
			b: []float64{
				0,
				0,
				1,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				1.533641379893842e-23, 5.609946470430354e-16, 4.220925818951961e-20, 0, 0, 0, 0, -0.6141831967249345,
			},
			d: []float64{
				0,
			},
		},
		"406-unc-false": {dt: 0.001, n: 9, m: 1, p: 1,
			a: []float64{
				0.9996183404208603, 1, 0, 0, 0, 0, 0, 0, 0.1641248663725789,
				0, 0.9996153752978781, 3240.244504482043, 3.996750081953068, 3942.468248230786, 3.997305638414697, 34414.335794244216, 3.9992646480773546, 1.4264852429843917,
				0, 0, 0.9983750409765341, 0.001232469117652652, 0, 0, 0, 0, -0.15230315386510362,
				0, 0, -0.001232469117652652, 0.9983750409765341, 3942.468248230786, 3.997305638414697, 34414.335794244216, 3.9992646480773546, 1.0514702265395213,
				0, 0, 0, 0, 0.9986528192073485, 0.00101322618562389, 0, 0, -0.5582573169465819,
				0, 0, 0, 0, -0.00101322618562389, 0.9986528192073485, 34414.335794244216, 3.9992646480773546, -0.37553480303407794,
				0, 0, 0, 0, 0, 0, 0.9996323240386774, 0.0001161879003490588, 0.8245491754244918,
				0, 0, 0, 0, 0, 0, -0.0001161879003490588, 0.9996323240386774, 0.7135592053089225,
				0, 0, 0, 0, 0, 0, 0, 0, 0.9994851877627321,
			},
			b: []float64{
				0,
				1,
				0,
				1,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				1.814068992942322e-26, 9.072076187101356e-27, 0, 0, 0, 0, 0, 0, -0.10117853385667885,
			},
			d: []float64{
				0,
			},
		},
		"148-unc-false": {dt: 0.01, n: 6, m: 1, p: 1,
			a: []float64{
				0.9897565788385125, 272.26481910896115, 3.9726393875482016, 1.753496284392368, 1, 0.27454851795027113,
				0, 0.9863196937741009, 0.014490509511733961, 0, 0, 0.307734264387422,
				0, -0.014490509511733961, 0.9863196937741009, 1.753496284392368, 1, 2.76919057858152,
				0, 0, 0, 0.9930396265104477, 0.005890387678995963, 2.400666515610661,
				0, 0, 0, -0.005890387678995963, 0.9930396265104477, 0.8377409970430088,
				0, 0, 0, 0, 0, 0.9908783533241214,
			},
			b: []float64{
				0,
				0,
				0,
				0,
				1,
				0,
			},
			c: []float64{
				-9.56544202259722e-10, 0.00011357054299594634, 1.6571175587337691e-06, 7.314405357679245e-07, 4.171326408150261e-07, -1.0448031552312158,
			},
			d: []float64{
				0,
			},
		},
		"221-unc-false-amd64": {dt: 0.01, n: 6, m: 1, p: 1,
			a: []float64{
				0.9793608907337193, 206.92597443602762, 3.9582987572645107, 0.004276123252322091, 0.0027558785884693737, 1.007412696553358,
				0, 0.9791493786322553, 0.01892789925597794, 0, 0, -0.8322032582357499,
				0, -0.01892789925597794, 0.9791493786322553, 0.004276123252322091, 0.0027558785884693737, -1.988368171197713,
				0, 0, 0, 0.9980608132337387, 6.969850193979593e-05, -0.9727735970398861,
				0, 0, 0, -6.969850193979593e-05, 0.9980608132337387, 0.4005503081635751,
				0, 0, 0, 0, 0, 0.9944797244924881,
			},
			b: []float64{
				1,
				0,
				1,
				0,
				1,
				0,
			},
			c: []float64{
				4.2188592861086976e-07, 0, 0, 0, 0, -0.14161120933668375,
			},
			d: []float64{
				0,
			},
		},
		"246-unc-false-amd64": {dt: 0.001, n: 6, m: 1, p: 1,
			a: []float64{
				0.9996424574182071, 3282.0863635864443, 3.99673529003694, 0.5866633034438116, 1, 0.921981271894043,
				0, 0.99836764501847, 0.001216747922445697, 0, 0, 0.503385462616573,
				0, -0.001216747922445697, 0.99836764501847, 0.5866633034438116, 1, -0.20119638596239997,
				0, 0, 0, 0.9990643903352582, 0.00047210639344921545, 0.9751765862471118,
				0, 0, 0, -0.00047210639344921545, 0.9990643903352582, -0.380590908626684,
				0, 0, 0, 0, 0, 0.9998723435415622,
			},
			b: []float64{
				0,
				0,
				0,
				0,
				1,
				0,
			},
			c: []float64{
				-7.008227081302481e-14, 1.0794324549624958e-06, 1.3144705251587412e-09, 1.9294538282069254e-10, 3.288860606894464e-10, 0.16574654254472998,
			},
			d: []float64{
				0,
			},
		},
	}
	out := make(map[string]*System, len(lits))
	for name, l := range lits {
		sys, err := New(mat.NewDense(l.n, l.n, l.a), mat.NewDense(l.n, l.m, l.b), mat.NewDense(l.p, l.n, l.c), mat.NewDense(l.p, l.m, l.d), l.dt)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out[name] = sys
	}
	return out
}

// Y5H2MH matched-hid 148, 221, 246 unc-false: the planted state is
// structurally disconnected; the rotated staircase cut is verified at
// nearly every probe but deviates at the matched-C2D zero z = -1 by more
// than the model's own bound there (certified wrong by 3..737 on main).
// Every mode returns the model unreduced.
func TestReduceRefusesCutBeyondModelBoundAtNyquistZero(t *testing.T) {
	lits := reduceVerifyLiterals(t)
	for _, name := range []string{"148-unc-false", "221-unc-false-amd64", "246-unc-false-amd64"} {
		sys := lits[name]
		for _, opts := range []*ReduceOpts{{}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}, {Equalize: true}} {
			res, err := sys.Reduce(opts)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			assertReducedUnchanged(t, name, sys, res)
		}
	}
}

// A kept exact integrator with kept modes -1, -30 and an uncontrollable
// -2, as given and rotated: the rotated cut carries the integrator only to
// rounding, and its bound near s = 0 is a constant multiple of the
// model's; the cut must still be accepted (4 -> 3).
func TestReduceAcceptsRotatedCutOfKeptIntegrator(t *testing.T) {
	v := []float64{1.0 / 5, 2.0 / 5, 2.0 / 5, 4.0 / 5}
	q := mat.NewDense(4, 4, nil)
	for i := range 4 {
		for j := range 4 {
			q.Set(i, j, -2*v[i]*v[j])
		}
		q.Set(i, i, q.At(i, i)+1)
	}
	for _, rot := range []bool{false, true} {
		A := mat.NewDense(4, 4, []float64{0, 0.7, 0.2, 0.4, 0, -1, 0.3, -0.6, 0, 0, -30, 0.5, 0, 0, 0, -2})
		B := mat.NewDense(4, 1, []float64{1, 1, 1, 0})
		C := mat.NewDense(1, 4, []float64{1, -0.5, 0.3, 2})
		if rot {
			var tmp, a, b, c mat.Dense
			tmp.Mul(q, A)
			a.Mul(&tmp, q.T())
			b.Mul(q, B)
			c.Mul(C, q.T())
			A, B, C = &a, &b, &c
		}
		sys, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, opts := range []*ReduceOpts{{}, {Equalize: true}} {
			res, err := sys.Reduce(opts)
			if err != nil {
				t.Fatal(err)
			}
			if res.Order != 3 {
				t.Fatalf("rot=%v eq=%v: order %d, want 3", rot, opts.Equalize, res.Order)
			}
			for _, s := range []complex128{0.37i, 2.1i, 40i} {
				g, _ := sys.EvalFr(s)
				r, _ := res.Sys.EvalFr(s)
				if d := cmplx.Abs(g[0][0] - r[0][0]); d > 1e-10*cmplx.Abs(g[0][0]) {
					t.Errorf("rot=%v eq=%v: |G-R| = %g at %v, |G| = %g", rot, opts.Equalize, d, s, cmplx.Abs(g[0][0]))
				}
			}
		}
	}
}
