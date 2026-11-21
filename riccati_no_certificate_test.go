package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// gateCounter counts dareGate runs that reach the certificate and keeps the
// X and K of the last result it was handed.
type gateCounter struct {
	calls int
	x, k  *mat.Dense
}

func countGate(t *testing.T) *gateCounter {
	t.Helper()
	g := &gateCounter{}
	dareGateHook = func(r *RiccatiResult) {
		g.calls++
		g.x, g.k = mat.DenseCopyOf(r.X), mat.DenseCopyOf(r.K)
	}
	t.Cleanup(func() { dareGateHook = nil })
	return g
}

func (g *gateCounter) reset() { g.calls, g.x, g.k = 0, nil, nil }

func dareCertRows() []dareFloorCase {
	return append(append([]dareFloorCase{}, dareNoiseFloorCases...),
		dareEnclosureCtrlCase, dareEnclosureBigCase, dareEnclosureWCase, dareEnclosureWBoundaryCase, dareEnclosureAtXCase)
}

// TestDare_NoCertificateAccuracy (EPYIVF): on every row whose default Dare
// runs the gate and returns a nil error (the gate certified X at X, at
// its centre, or replaced it in tier 2), opts.NoCertificate runs no gate
// work and returns the X and K the gate was handed, with
// ErrAccuracyUnverified. Which rows gate depends on the build, so only
// that some row does is required.
func TestDare_NoCertificateAccuracy(t *testing.T) {
	g := countGate(t)
	gated, replaced := 0, 0
	for _, c := range dareCertRows() {
		A, B, Q, R, S, E := c.dense()
		for _, ns := range []bool{false, true} {
			g.reset()
			def, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: ns})
			if err != nil || g.calls == 0 {
				continue
			}
			gated++
			preX, preK := g.x, g.k
			if !mat.Equal(def.X, preX) {
				replaced++
			}
			g.reset()
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: ns, NoCertificate: true})
			if !errors.Is(err, ErrAccuracyUnverified) || res == nil || res.X == nil {
				t.Errorf("%s ns=%v: NoCertificate: result %v err %v, want a result with ErrAccuracyUnverified", c.name, ns, res != nil, err)
				continue
			}
			if g.calls != 0 {
				t.Errorf("%s ns=%v: NoCertificate: %d gate runs, want 0", c.name, ns, g.calls)
			}
			if !mat.Equal(res.X, preX) || !mat.Equal(res.K, preK) {
				t.Errorf("%s ns=%v: NoCertificate: X or K differs from the pre-gate candidate", c.name, ns)
			}
		}
	}
	if gated == 0 {
		t.Fatal("no row ran the gate with a nil default error in this build")
	}
	t.Logf("%d gated solves, %d with X replaced by the gate", gated, replaced)
}

// TestDare_NoCertificateExemptRow (EPYIVF): near-singular descriptor rows
// (nearSingularTriangularE seeds 163, 213, 415, 500, dt 1, Q = I, R = I)
// whose X the extended solve confirmed never reach the gate, so
// opts.NoCertificate leaves their result, and its nil error, unchanged.
// Which rows are exempt depends on the build, so only that some row is
// is required.
func TestDare_NoCertificateExemptRow(t *testing.T) {
	g := countGate(t)
	rows := []struct{ a, b, e []float64 }{
		{[]float64{-0.13724618874059225, 1.217088645560901, 0.6539759935733573, 0.625550292051194},
			[]float64{1.0056269676565766, -0.43721297890158445},
			[]float64{0.8136281888350105, -0.45758472346960766, 0, 3.8390252196819285e-16}},
		{[]float64{0.5000278208714111, -0.07733055775849312, -0.4888167933205462, 1.3129070029607885},
			[]float64{0.3112712539078186, 1.219531204968224},
			[]float64{0.8602200064118372, 0.06309803400898362, 0, 2.64158237466957e-16}},
		{[]float64{-2.3221006124296615, -1.0818055667933515, -1.021140292167516, 0.7023646645229246},
			[]float64{0.8694884592755641, 2.527025463345394, -0.22663117073751365, -1.131236581183983},
			[]float64{0.7047884850050473, -0.07412186793679755, 0, 3.275877501116736e-16}},
		{[]float64{-0.7312715912385475, -0.054319723233813816, 0.132396174077426, -0.5991796548980122},
			[]float64{0.6561074877388584, -0.5688892842798058, 0.6471839580069148, 0.33716000964540804},
			[]float64{0.6097791941584733, -0.3393214326084397, 0, 3.188503216119319e-16}},
	}
	exempt := 0
	for i, r := range rows {
		m := len(r.b) / 2
		A, B, E := mat.NewDense(2, 2, r.a), mat.NewDense(2, m, r.b), mat.NewDense(2, 2, r.e)
		Q, R := eyeDense(2), eyeDense(m)
		g.reset()
		def, err := Dare(A, B, Q, R, &RiccatiOpts{E: E})
		if err != nil || g.calls != 0 {
			continue
		}
		exempt++
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoCertificate: true})
		if err != nil {
			t.Errorf("row %d: NoCertificate: %v, want nil", i, err)
			continue
		}
		if g.calls != 0 || !mat.Equal(res.X, def.X) || !mat.Equal(res.K, def.K) {
			t.Errorf("row %d: NoCertificate: %d gate runs, X equal %v, K equal %v; want 0, true, true", i, g.calls, mat.Equal(res.X, def.X), mat.Equal(res.K, def.K))
		}
	}
	if exempt == 0 {
		t.Fatal("no row was exempt from the gate with a nil default error in this build")
	}
	t.Logf("%d of %d rows exempt", exempt, len(rows))
}

// TestDare_NoCertificateDefaultUnchanged (EPYIVF): NoCertificate false is
// nil opts bit for bit, with the same gate runs.
func TestDare_NoCertificateDefaultUnchanged(t *testing.T) {
	g := countGate(t)
	for _, c := range dareCertRows() {
		if c.s != nil || c.e != nil {
			continue
		}
		A, B, Q, R, _, _ := c.dense()
		g.reset()
		want, werr := Dare(A, B, Q, R, nil)
		wantCalls := g.calls
		g.reset()
		got, gerr := Dare(A, B, Q, R, &RiccatiOpts{NoCertificate: false})
		if (werr == nil) != (gerr == nil) || g.calls != wantCalls {
			t.Errorf("%s: err %v calls %d, want err %v calls %d", c.name, gerr, g.calls, werr, wantCalls)
			continue
		}
		if want != nil && (!mat.Equal(got.X, want.X) || !mat.Equal(got.K, want.K)) {
			t.Errorf("%s: X or K differs from nil opts", c.name)
		}
	}
}

// dlqrXECertifiedGateCases are near-singular-E Dlqr rows (BN4LWB frozen
// census: zrwctrl, zw46ctrl, zw46mid) whose result reaches the gate with
// xeCertified set and X not confirmed in extended precision: the first
// four on native arm64 and amd64, the next four on amd64, the last four
// natively.
var dlqrXECertifiedGateCases = []dareFloorCase{
	{name: "zw46ctrl:1076", n: 2, m: 1, a: []float64{-0.5855947120342042, 1.3387998690779592, -0.8046034412435987, -0.14414255809578996}, b: []float64{1.7020235628064342, 0.4778024538201253}, q: []float64{3.742642538817651, 0.7932826402006268, 0.7932826402006268, 1.719370369555857}, r: []float64{6.291088131968703}, s: []float64{-2.0486374268637793, -1.5822613999356854}, e: []float64{0.011343860981665791, -0.0959239100625107, 0.1314085624464087, -1.1111933692315994}},
	{name: "zw46ctrl:1713", n: 2, m: 1, a: []float64{-0.29172665557165334, 1.9700101280133628, 0.6293894289968237, 0.674564783974045}, b: []float64{2.048297933489893, 0.22490069268210544}, q: []float64{7.937189308083653, 0.9794154538379635, 0.9794154538379635, 7.80269450116074}, r: []float64{7.095742848513051}, s: []float64{-0.12966020367035028, -4.521670568709842}, e: []float64{0.062421712592621525, -0.0036982998644142424, -0.895377355669893, 0.0530484317780834}},
	{name: "zw46mid:1646", n: 3, m: 1, a: []float64{0.9470531256949372, -0.553880924602379, 0.7897538572334435, -0.7492782172335031, 0.7899750703359367, 2.170891941325952, -0.20074028421962667, -1.2087396686828933, 0.7881221605953292}, b: []float64{-1.165681162654708, 0.4334352585246565, 0.44089992629922514}, q: []float64{1.3483008018064806, -1.0727913542946683, -1.3966688010017674, -1.0727913542946683, 1.483099559184335, 1.1574603855587442, -1.3966688010017674, 1.1574603855587442, 6.30573384627416}, r: []float64{0.5370344929230644}, s: []float64(nil), e: []float64{0.8688915844559565, 0.05890637843244142, 0.1759399325891008, -0.04411683315009132, 0.5911993677136238, 0.13813376170104313, -0.8407661009636253, -0.14610824443186174, -0.19229996853883186}},
	{name: "zw46mid:1880", n: 4, m: 1, a: []float64{1.304614173870719, -0.10254517169821298, 0.41247235180762576, -0.553127679743858, 1.660162071855229, -0.668322944675577, -0.3471029518179516, -2.0311885295626997, 0.146610341764438, 0.04255953685655123, 0.3144258868721697, 0.8047938591434873, 0.9768491541835551, -1.417007189372168, -0.9076581832373678, -1.2725518304089738}, b: []float64{0.2805975986540834, -0.4379298215045566, -0.902097787402492, 1.8933638540103725}, q: []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, r: []float64{1}, s: []float64(nil), e: []float64{-0.12227391420439505, 0.7167928612484048, -0.15907817227356552, 0.6169884408094568, 0.05829523750841348, 0.058463545814478754, 0.014912743867307957, -0.4087384651693422, -0.19408665337669834, -1.0652070401720564, -0.24762489923100683, 0.2202452494821431, 0.17919340408058607, -0.15152531726319396, 0.5882506563893345, 0.9072554409649451}},
	{name: "zrwctrl:14839", n: 2, m: 1, a: []float64{0.5200221221988941, -1.7599671952491514, -1.924156237772368, -1.021924772736714}, b: []float64{-0.6020002394152005, 0.4522790070333368}, q: []float64{1.9765072622320112, 1.5305142896768984, 1.5305142896768984, 1.6060046570324509}, r: []float64{0.9551339965523218}, s: []float64(nil), e: []float64{0.05993817329192091, 0.08775285237803432, 0.3103707133000417, 0.4544001576602553}},
	{name: "zrwctrl:17307", n: 2, m: 1, a: []float64{0.05180439154026484, -0.8776392547363798, 0.06750723911824474, 0.7080822039205079}, b: []float64{-0.10106672954554613, 0.006661725235461713}, q: []float64{2.0476942507260554, -2.122127556722604, -2.122127556722604, 2.763344697084188}, r: []float64{0.8330556335014635}, s: []float64(nil), e: []float64{0.42275056561176416, 1.2200698195983122, 0.055326148042577065, 0.1596727928554774}},
	{name: "zw46ctrl:1011", n: 2, m: 1, a: []float64{-0.24183972884542215, 0.24484551305919688, -0.5244516155318022, -0.09129134646094605}, b: []float64{0.48170059816854127, 0.07931175580898286}, q: []float64{1.023051280479656, 1.6007865820255989, 1.6007865820255989, 5.064434113697244}, r: []float64{1.2399417205498393}, s: []float64(nil), e: []float64{0.22599946760436254, 0.1455132001493176, 0.43456291233788613, 0.2797999513897147}},
	{name: "zw46mid:1649", n: 4, m: 1, a: []float64{0.9988595479856077, -0.4391899077867851, -0.8791595448254504, -2.2224343805350504, 0.15636126587706967, -0.020320638335828534, -1.4311545370302396, -1.1309443804047437, -1.4761398321344625, -0.3135735740858264, -0.14431679299493894, -0.3296023556321064, -0.828052568398942, -1.726609655001972, -0.11753476216062458, -1.2912020286774109}, b: []float64{-0.5602369331287269, -1.1258146713266772, -0.18677917533958777, -0.6169905587282808}, q: []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, r: []float64{1}, s: []float64(nil), e: []float64{-0.5559337668034753, -0.30996100239037006, -0.018594438972973325, 0.2970172614751996, -0.5324570986500226, 0.17662997972493527, -0.09929683460348435, 0.02357966970147244, 0.44912250314475244, 0.19284001763573, -0.6175837292060566, 0.3634820906853742, -0.12989324251227605, 0.6418742743170002, 0.007359494015664951, -0.4439689165322344}},
	{name: "zrwctrl:11220", n: 2, m: 1, a: []float64{-0.13780214247478897, 1.9511839117112353, -1.7158279481668792, -0.4092400127223139}, b: []float64{0.011614074052167744, -0.2819127378337477}, q: []float64{1.4366812645557285, 1.1924860697307282, 1.1924860697307282, 1.1719048496563087}, r: []float64{1.648924307693482}, s: []float64(nil), e: []float64{1.2598270886426528, 0.09724589437219147, 0.02464227734325914, 0.0019021343060126195}},
	{name: "zrwctrl:16859", n: 2, m: 1, a: []float64{2.0167912454133474, -0.10710496199400732, 0.3527812566534412, 2.1384709294373536}, b: []float64{1.8252235969392914, 1.845688453086316}, q: []float64{1, 0, 0, 1}, r: []float64{1}, s: []float64(nil), e: []float64{0.010712060010389846, -0.003522469895827285, 0.6248473963355181, -0.20546992225865765}},
	{name: "zrwctrl:20356", n: 2, m: 1, a: []float64{-0.48472435838558736, 0.5099697847724264, -1.555749927472771, 1.4205578402526369}, b: []float64{-1.6180881067253607, -1.0960180099312686}, q: []float64{0.9178104151644892, -1.0023823435396597, -1.0023823435396597, 2.144213315416552}, r: []float64{0.5207629034716447}, s: []float64(nil), e: []float64{1.0779967422046484, 0.13217657120860934, -0.02463443045221868, -0.003020504994397966}},
	{name: "zrwctrl:16688", n: 2, m: 1, a: []float64{-0.7471544113790201, -0.3321141629455049, 0.013021819406051433, -0.017665229963921142}, b: []float64{-0.6599921737759707, 1.5433509585633884}, q: []float64{2.302626067521486, -0.1779190508997251, -0.1779190508997251, 0.4524730032436649}, r: []float64{0.6608214047924337}, s: []float64(nil), e: []float64{0.8343212317506309, -0.7331113289036647, 0.06904879217535712, -0.06067261607690165}},
}

// TestDlqr_NoCertificateDescriptor (EPYIVF): descriptor rows whose default
// Dlqr runs the gate, rv49 3641 (tier 2) and the census rows reaching it
// with a certified EᵀXE (xeCertified). Under opts.NoCertificate they run
// no gate work and keep ErrAccuracyUnverified: the cleared xeCertified is
// not promoted back to a nil error by Dlqr's explicit EᵀXE. Which rows
// gate, and with xeCertified, depends on the build.
func TestDlqr_NoCertificateDescriptor(t *testing.T) {
	g := countGate(t)
	var xeCert bool
	dareGateHook = func(r *RiccatiResult) { g.calls++; xeCert = xeCert || r.xeCertified }
	A, B, Q, R, S, E := dlqrCorrectedEnclosureRow()
	rows := append([]dareFloorCase{{name: "rv49:3641", n: 5, m: 8, a: A.RawMatrix().Data, b: B.RawMatrix().Data,
		q: Q.RawMatrix().Data, r: R.RawMatrix().Data, s: S.RawMatrix().Data, e: E.RawMatrix().Data}}, dlqrXECertifiedGateCases...)
	gated, withXE := 0, 0
	for _, c := range rows {
		A, B, Q, R, S, E := c.dense()
		g.calls, xeCert = 0, false
		if _, err := Dlqr(A, B, Q, R, &RiccatiOpts{S: S, E: E}); g.calls == 0 {
			t.Logf("%s: default Dlqr (err %v) ran no gate in this build", c.name, err)
			continue
		}
		gated++
		if xeCert {
			withXE++
		}
		g.calls = 0
		res, err := Dlqr(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoCertificate: true})
		if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
			t.Errorf("%s: NoCertificate: err %v, want a result with ErrAccuracyUnverified", c.name, err)
		}
		if g.calls != 0 {
			t.Errorf("%s: NoCertificate: %d gate runs, want 0", c.name, g.calls)
		}
	}
	if withXE == 0 {
		t.Fatalf("no row reached the gate with xeCertified in this build (%d gated)", gated)
	}
	t.Logf("%d gated rows, %d reaching the gate with xeCertified", gated, withXE)
}

// TestNoCertificate_Inheritance (EPYIVF): Lqi of a discrete descriptor
// plant, Lqrd, discrete Kalman and Kalmd pass opts.NoCertificate to their
// Dare, which then runs no gate work.
func TestNoCertificate_Inheritance(t *testing.T) {
	g := countGate(t)
	A := mat.NewDense(3, 3, []float64{0.5, 0.3, -0.1, -0.2, 0.4, 0.25, 0.1, -0.35, 0.6})
	Ac := mat.NewDense(3, 3, []float64{-1, 0.7, 0.2, -0.4, -0.5, 0.9, 0.3, -0.6, -1.2})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.3, 0.8, 0.5, -0.6})
	C := mat.NewDense(2, 3, []float64{1, -0.4, 0.3, 0.2, 0.9, -0.5})
	D := mat.NewDense(2, 2, nil)
	E := mat.NewDense(3, 3, []float64{1.2, 0.1, -0.2, 0, 0.9, 0.3, 0.1, 0, 1.1})
	desc, err := NewDescriptor(A, B, C, D, E, 1)
	if err != nil {
		t.Fatal(err)
	}
	disc, err := New(A, B, C, D, 1)
	if err != nil {
		t.Fatal(err)
	}
	cont, err := New(Ac, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func(o *RiccatiOpts) error
	}{
		{"Lqi descriptor", func(o *RiccatiOpts) error { _, err := Lqi(desc, eyeDense(5), eyeDense(2), o); return err }},
		{"Lqrd", func(o *RiccatiOpts) error { _, err := Lqrd(Ac, B, eyeDense(3), eyeDense(2), 0.1, o); return err }},
		{"Kalman discrete", func(o *RiccatiOpts) error { _, err := Kalman(disc, eyeDense(2), eyeDense(2), nil, o); return err }},
		{"Kalmd", func(o *RiccatiOpts) error { _, err := Kalmd(cont, eyeDense(2), eyeDense(2), 0.1, o); return err }},
	} {
		g.reset()
		if err := tc.run(&RiccatiOpts{}); g.calls == 0 {
			t.Fatalf("%s: precondition: default (err %v) ran no gate", tc.name, err)
		}
		g.reset()
		err := tc.run(&RiccatiOpts{NoCertificate: true})
		if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: NoCertificate: %v", tc.name, err)
		}
		if g.calls != 0 {
			t.Errorf("%s: NoCertificate: %d gate runs, want 0", tc.name, g.calls)
		}
	}
}

// TestCare_NoCertificateIgnored (EPYIVF): Care has no enclosure gate and
// ignores opts.NoCertificate.
func TestCare_NoCertificateIgnored(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.7, 0.2, -0.4, -0.5, 0.9, 0.3, -0.6, -1.2})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.3, 0.8, 0.5, -0.6})
	want, err := Care(A, B, eyeDense(3), eyeDense(2), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Care(A, B, eyeDense(3), eyeDense(2), &RiccatiOpts{NoCertificate: true})
	if err != nil {
		t.Fatalf("NoCertificate: %v, want nil", err)
	}
	if !mat.Equal(got.X, want.X) || !mat.Equal(got.K, want.K) {
		t.Error("NoCertificate: X or K differs from nil opts")
	}
}
