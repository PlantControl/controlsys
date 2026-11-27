package controlsys

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// c2dMapLFTSystem is the frozen row name with one internal delay of 0.05 s
// whose delay channel is excited (C2 ≠ 0) and fed back (B2 ≠ 0).
func c2dMapLFTSystem(t *testing.T, name string) *System {
	t.Helper()
	s := c2dMapRowSystem(t, name).Copy()
	n, _, _ := s.Dims()
	b2, c2 := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		b2.Set(i, 0, []float64{0.3, -0.2, 0.5, 0.1, -0.4}[i%5])
		c2.Set(0, i, []float64{0.25, 0.5, -0.4, 0.2, -0.1}[i%5])
	}
	if err := s.SetInternalDelay([]float64{0.05}, b2, c2, mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return s
}

func bitsEqualLFT(a, b *System) bool {
	if a.LFT == nil || b.LFT == nil {
		return a.LFT == b.LFT
	}
	x, y := a.LFT, b.LFT
	return bitsEqualDense(x.B2, y.B2) && bitsEqualDense(x.C2, y.C2) && bitsEqualDense(x.D12, y.D12) &&
		bitsEqualDense(x.D21, y.D21) && bitsEqualDense(x.D22, y.D22) && fmt.Sprint(x.Tau) == fmt.Sprint(y.Tau)
}

// C2DMap of a stiff internal-delay descriptor (LU condition estimate of E
// 1e6 or more) returns C2D's model and verdict bitwise, on the stiff LFT
// split (BWVNRE; T3IVBX ruling Q2), or C2D's error where C2D refuses the
// split (stfr/3408 on arm64) or returns no model; G is nd×(n+m+N). At
// dt = 1 the delay is under one sample and no exact map exists. cluster's
// fast part is dense, so its real Schur form changes coordinates.
func TestC2DMapStiffLFTMatchesC2D(t *testing.T) {
	routed := 0
	for _, name := range []string{"cluster", "stfr/399", "stfr/2880", "cond/567", "stfr/3408"} {
		sys := c2dMapLFTSystem(t, name)
		if _, ok, _ := newStiffLFTSplitWithin(sys, nil, 1); ok {
			routed++
		}
		n, m, _ := sys.Dims()
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				key := fmt.Sprintf("%s@%g/%s", name, dt, meth)
				want, werr := sys.C2D(dt, C2DOptions{Method: meth})
				got, g, err := sys.C2DMap(dt, C2DOptions{Method: meth})
				if want == nil {
					if got != nil || g != nil || err == nil || strings.TrimPrefix(err.Error(), "C2DMap: ") != strings.TrimPrefix(werr.Error(), "C2D: ") {
						t.Errorf("%s: C2DMap (%v), want C2D's error (%v) with no model", key, err, werr)
					}
					continue
				}
				if dt == 1 && errors.Is(err, ErrOptionUnsupported) && got == nil {
					continue
				}
				if wc, gc := errClass(want, werr), errClass(got, err); wc != gc {
					t.Errorf("%s: C2DMap %s (%v), C2D %s (%v)", key, gc, err, wc, werr)
					continue
				}
				if !bitsEqualSystem(got, want) || !bitsEqualLFT(got, want) {
					t.Errorf("%s: C2DMap model differs from C2D's", key)
				}
				nd, _, _ := got.Dims()
				if r, c := g.Dims(); r != nd || c != n+m+1 {
					t.Errorf("%s: G is %d×%d, want %d×%d", key, r, c, nd, n+m+1)
				}
			}
		}
	}
	if routed < 3 {
		t.Errorf("%d rows take the stiff LFT split, want at least 3", routed)
	}
}

// c2dMapLFTResponseRefs are the 320-bit responses of the exact method on
// the exact augmented core of c2dMapLFTSystem (inputs [u; w], outputs
// [y; z]) from [x0; 0] (free) and from [0; u] with w0 = u_w[0] ≠ 0
// (forced), at c2dMapRefKs, with the 1-ulp data floor (census/bwvnre).
var c2dMapLFTResponseRefs = []c2dMapResponseRef{
	{row: "cluster", dt: 0.0001, method: "zoh", forced: false, floor: 3.52e-16, y: [][]float64{{2.1749999999999998, 1.625719520829364, 1.6256144446128697, 1.6254043127133342, 1.62498413103544, 1.6241440960633118, 1.6237242427113536}, {-1.5250000000000001, -0.63029327445580663, -0.63025252106754692, -0.63017102225948107, -0.63000805651225589, -0.62968225245418397, -0.62951941412091617}}},
	{row: "cluster", dt: 0.0001, method: "zoh", forced: true, floor: 2.28e-16, y: [][]float64{{0.52622064772118449, 1.8067621331822199, 1.9512498957990925, 1.4972214130056083, 1.0908171781313807, 0.84983581823659793, 1.9517373611479938}, {0, 0.81229677214416562, 1.0768003857786379, 1.1704443577172852, 0.41256081011526219, 0.45667422643033967, 1.0871604328415196}}},
	{row: "cluster", dt: 0.0001, method: "foh", forced: false, floor: 3.52e-16, y: [][]float64{{2.1749999999999998, 1.625719520829364, 1.6256144446128697, 1.6254043127133342, 1.62498413103544, 1.6241440960633118, 1.6237242427113536}, {-1.5250000000000001, -0.63029327445580663, -0.63025252106754692, -0.63017102225948107, -0.63000805651225589, -0.62968225245418397, -0.62951941412091617}}},
	{row: "cluster", dt: 0.0001, method: "foh", forced: true, floor: 8.37e-14, y: [][]float64{{0.52622064772118449, 1.9815993709331028, 1.930062207481082, 1.201386310068272, 1.3596708742656018, 0.97822983073525771, 1.9205907601348537}, {0, 1.0768001779161542, 1.2183827736234085, 0.95554841185558193, 0.55449509024043575, 0.41380866250617399, 1.220509570940715}}},
	{row: "cluster", dt: 0.0001, method: "tustin", forced: false, floor: 3.25e-11, y: [][]float64{{2.1749999999999998, 1.0765761810543168, 2.1747257297131437, 2.1744514934420693, 2.1739031229304153, 2.1728067899111676, 2.1722588273362096}, {-1.5250000000000001, 0.26433172152843504, -1.5248365455705284, -1.5246731099018755, -1.5243462948378883, -1.5236928897392477, -1.523366299668067}}},
	{row: "cluster", dt: 0.0001, method: "tustin", forced: true, floor: 1.56e-11, y: [][]float64{{0.52622064772118449, 3.2072709556637711, 0.70450832907584526, -0.023943069902211876, 0.13479202752164379, -0.245745090015377, 0.69707323110239883}, {2.0856868160153744e-93, 1.8890158559293881, 0.40625008255034206, 0.14357473289388323, -0.25716214417042504, -0.39721030380531602, 0.40981415337210225}}},
	{row: "cluster", dt: 0.01, method: "zoh", forced: false, floor: 3.55e-16, y: [][]float64{{2.1749999999999998, 1.6153501080540043, 1.6049436180199048, 1.5843328710963589, 1.543907986199764, 1.4661485023573257, 1.42876059018526}, {-1.5250000000000001, -0.62627154684332054, -0.62223545416831194, -0.61424174873704929, -0.59856347100109863, -0.56840610423788851, -0.55390630959492282}}},
	{row: "cluster", dt: 0.01, method: "zoh", forced: true, floor: 3.65e-16, y: [][]float64{{0.52622064772118449, 1.8079009556490178, 1.9540079811051343, 1.5038295305312137, 1.1012635969209719, 0.87041832665427787, 1.97588077402217}, {0, 0.81185746450086416, 1.0757353701232735, 1.1678891870798909, 0.40852020638614639, 0.4487131678769179, 1.0778255918569366}}},
	{row: "cluster", dt: 0.01, method: "foh", forced: false, floor: 3.55e-16, y: [][]float64{{2.1749999999999998, 1.6153501080540043, 1.6049436180199048, 1.5843328710963589, 1.543907986199764, 1.4661485023573257, 1.42876059018526}, {-1.5250000000000001, -0.62627154684332054, -0.62223545416831194, -0.61424174873704929, -0.59856347100109863, -0.56840610423788851, -0.55390630959492282}}},
	{row: "cluster", dt: 0.01, method: "foh", forced: true, floor: 1e-15, y: [][]float64{{0.52622064772118449, 1.9829871112239608, 1.9332234629941765, 1.2082276765575914, 1.3699080960651071, 0.99850748514655474, 1.9451316914372534}, {0, 1.0762718496209267, 1.217163938128988, 0.95289416569807783, 0.55054143077135087, 0.4059659133332722, 1.2110228794615752}}},
	{row: "cluster", dt: 0.01, method: "tustin", forced: false, floor: 3.27e-13, y: [][]float64{{2.1749999999999998, 1.0661749951568615, 2.1541182987665617, 2.1335068384600473, 2.093080532574199, 2.0153182287708691, 1.9779289170309189}, {-1.5250000000000001, 0.2683940274219952, -1.5169005753500342, -1.5089060223713286, -1.4932260517886751, -1.4630653079004869, -1.4485638287469811}}},
	{row: "cluster", dt: 0.01, method: "tustin", forced: true, floor: 1.57e-13, y: [][]float64{{0.52622064772118449, 3.2087756533448601, 0.70743538782459192, -0.01755799672596418, 0.14412747819814645, -0.22726396937615642, 0.71936407124575108}, {3.7204439288995056e-91, 1.8885723192111392, 0.4048645746659445, 0.14059633317705833, -0.26175344899764996, -0.40632261993457913, 0.39873787692838875}}},
	{row: "stfr/399", dt: 0.0001, method: "zoh", forced: false, floor: 7.48e-15, y: [][]float64{{0.3876658107167682, 0.85539346868345911, 0.85787282151348587, 0.86285718267995437, 0.8729294710497989, 0.89349597623625365, 0.90399465075363539}, {-1.325, 1.4298979197581323, 1.4371567444106748, 1.4517585993354156, 1.4813022329916818, 1.5417743748108037, 1.5727175277287619}}},
	{row: "stfr/399", dt: 0.0001, method: "zoh", forced: true, floor: 4.79e-15, y: [][]float64{{-0.4275671333071982, -0.76087556618053498, -0.92940701418685634, -0.77029295714079138, -0.29369230838238103, -0.21015020695444367, -0.94180472365675105}, {0, -0.47540493521544808, -0.32847628860032363, 0.11399507251162525, -0.32270499416406379, -0.10363396899442058, -0.34504094347867625}}},
	{row: "stfr/399", dt: 0.0001, method: "foh", forced: false, floor: 7.48e-15, y: [][]float64{{0.3876658107167682, 0.85539346868345911, 0.85787282151348587, 0.86285718267995437, 0.8729294710497989, 0.89349597623625365, 0.90399465075363539}, {-1.325, 1.4298979197581323, 1.4371567444106748, 1.4517585993354156, 1.4813022329916818, 1.5417743748108037, 1.5727175277287619}}},
	{row: "stfr/399", dt: 0.0001, method: "foh", forced: true, floor: 2.12e-13, y: [][]float64{{-0.4275671333071982, -0.73559135942363196, -0.88951113297327944, -0.75533361161135992, -0.32014751336462466, -0.24996788829596839, -0.90164566895864229}, {0, -0.32613231966603912, -0.093122368924416291, 0.20159590938014532, -0.47908645061355765, -0.33919214799273056, -0.10817400529277588}}},
	{row: "stfr/399", dt: 0.0001, method: "tustin", forced: false, floor: 2.23e-11, y: [][]float64{{0.3876658107167682, 1.3206502837250726, 0.39261601494060677, 0.39760038199312076, 0.40767268240304655, 0.42823921277142313, 0.43873790044807726}, {-1.325, 4.1775649453657673, -1.3105102530899593, -1.2959083782580554, -1.2663647039070018, -1.205892477078564, -1.1749492797888061}}},
	{row: "stfr/399", dt: 0.0001, method: "tustin", forced: true, floor: 1.72e-11, y: [][]float64{{-0.4275671333071982, -0.81568571304576742, -0.80941670620507933, -0.67523906634809461, -0.24005311739747248, -0.16987343268545735, -0.82155124306407068}, {1.4808680126826021e-87, -0.79914554921845027, 0.3798910994820634, 0.67460976738790068, -0.0060730823488777804, 0.13382141604555184, 0.36483945935206052}}},
	{row: "stfr/399", dt: 0.01, method: "zoh", forced: false, floor: 2.59e-14, y: [][]float64{{0.3876658107167682, 1.1480783635401874, 1.5702737141168999, 3.0691479507663533, 13.246562802013152, 293.90505271595612, 1409.3648188503385}, {-1.325, 2.3035527297408058, 3.601533534591316, 8.3468775494436898, 41.42215854110124, 962.7641741322102, 4627.1412830294203}}},
	{row: "stfr/399", dt: 0.01, method: "zoh", forced: true, floor: 2.39e-14, y: [][]float64{{-0.4275671333071982, -0.84926208133909675, -1.1443105235575017, -1.3768284340948655, -3.406153496025405, -75.377611484522618, -362.73283101051845}, {0, -0.76503205799761187, -1.0335163616749719, -1.8793335097873791, -10.554269212283801, -247.08317640426958, -1189.0141196414218}}},
	{row: "stfr/399", dt: 0.01, method: "foh", forced: false, floor: 2.59e-14, y: [][]float64{{0.3876658107167682, 1.1480783635401874, 1.5702737141168999, 3.0691479507663533, 13.246562802013152, 293.90505271595612, 1409.3648188503385}, {-1.325, 2.3035527297408058, 3.601533534591316, 8.3468775494436898, 41.42215854110124, 962.7641741322102, 4627.1412830294203}}},
	{row: "stfr/399", dt: 0.01, method: "foh", forced: true, floor: 2.38e-14, y: [][]float64{{-0.4275671333071982, -0.82195214393278027, -1.0923878078232647, -1.309037361583741, -3.212713504023418, -70.272034084846581, -338.0022016903323}, {0, -0.60927309716452738, -0.75909304637187291, -1.61897446739693, -9.9882755547262505, -230.41425848449686, -1107.6592415908931}}},
	{row: "stfr/399", dt: 0.01, method: "tustin", forced: false, floor: 2.66e-14, y: [][]float64{{0.3876658107167682, 1.6175422682820311, 1.1175068181342622, 2.6589328269134835, 13.315803089579322, 318.64414118514782, 1561.8881387061069}, {-1.325, 5.0650413797540264, 0.89490027479605594, 5.7800438932491351, 40.430536104338415, 1042.8233952987289, 5127.0255526297351}}},
	{row: "stfr/399", dt: 0.01, method: "tustin", forced: true, floor: 2.43e-14, y: [][]float64{{-0.4275671333071982, -0.90327597965949424, -1.0153392433334758, -1.240578203233176, -3.2530992539235708, -76.050252777334762, -373.69031227454178}, {-1.3727266067819753e-85, -1.0863311020613728, -0.29609536503320322, -1.1841928598790388, -9.9110858934203954, -249.1882664816529, -1224.6998649330897}}},
}

// C2DMap's G maps [x0; u0; w0] of a stiff internal-delay descriptor into
// the coordinates of C2D's split model (BWVNRE): from G·[x0; u0; w0] the
// response of the model's augmented core matches the exact method on the
// exact core within 1000 times its 1-ulp floor or 10 times the model's own
// forced ZOH error, where G plays no part. At t = 0, [C; C2]_d·G_x
// reproduces [C; C2], and the ZOH u0 and w0 columns are zero.
func TestC2DMapStiffLFTInitialResponse(t *testing.T) {
	type cell struct {
		row string
		dt  float64
	}
	errs := make([]float64, len(c2dMapLFTResponseRefs))
	model := map[cell]float64{}
	for i, ref := range c2dMapLFTResponseRefs {
		sys := c2dMapLFTSystem(t, ref.row)
		d, g, err := sys.C2DMap(ref.dt, C2DOptions{Method: ref.method})
		key := fmt.Sprintf("%s@%g/%s forced=%t", ref.row, ref.dt, ref.method, ref.forced)
		if err != nil {
			t.Fatalf("%s: %v", key, err)
		}
		aug := conversionAugmentedRational(d)
		n, _, _ := sys.Dims()
		_, ma, p := aug.Dims()
		x0, u := c2dMapRefX0(n), (*mat.Dense)(nil)
		if ref.forced {
			x0, u = make([]float64, n), c2dMapRefU(ma)
		}
		y := mapResponse(aug, g, x0, u, 21)
		got, want := mat.NewDense(p, len(c2dMapRefKs), nil), mat.NewDense(p, len(c2dMapRefKs), nil)
		for o := range p {
			for k, kk := range c2dMapRefKs {
				got.Set(o, k, y.At(o, kk))
				want.Set(o, k, ref.y[o][k])
			}
		}
		errs[i] = responseError(got, want)
		if ref.method == C2DMethodZOH && ref.forced {
			model[cell{ref.row, ref.dt}] = errs[i]
		}
		nd, _, _ := aug.Dims()
		ca := conversionAugmentedRational(sys)
		var cw mat.Dense
		cw.Mul(aug.C, g.Slice(0, nd, 0, n))
		cw.Sub(&cw, ca.C)
		if e := mat.Norm(&cw, 2) / mat.Norm(ca.C, 2); !(e <= 1e-9) {
			t.Errorf("%s: ‖[C; C2]_d·G_x − [C; C2]‖/‖[C; C2]‖ = %.3g", key, e)
		}
		if ref.method == C2DMethodZOH && mat.Norm(g.Slice(0, nd, n, n+ma), 1) != 0 {
			t.Errorf("%s: G has nonzero u0 or w0 columns", key)
		}
	}
	for i, ref := range c2dMapLFTResponseRefs {
		if ref.method == C2DMethodZOH && ref.forced {
			continue
		}
		tol := math.Max(1e3*ref.floor, 10*model[cell{ref.row, ref.dt}])
		t.Logf("%s@%g/%s forced=%t: %.3g bar %.3g", ref.row, ref.dt, ref.method, ref.forced, errs[i], tol)
		if !(errs[i] <= tol) {
			t.Errorf("%s@%g/%s forced=%t: response from G %.3g off, bar %.3g (floor %.3g)", ref.row, ref.dt, ref.method, ref.forced, errs[i], tol, ref.floor)
		}
	}
}
