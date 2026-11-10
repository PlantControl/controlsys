package controlsys

import (
	"errors"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type dcGainInfSignRow struct {
	name       string
	n, m, p    int
	a, b, c, d []float64
	dt         float64
}

func checkDCGainInfSignRows(t *testing.T, rows []dcGainInfSignRow, wantErr error) {
	t.Helper()
	for _, tc := range rows {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		g, err := sys.DCGain()
		if !errors.Is(err, wantErr) {
			t.Errorf("%s: DCGain error %v, want %v", tc.name, err, wantErr)
		}
		if g == nil {
			t.Fatalf("%s: nil gain", tc.name)
		}
		checkDCGainExactLimitEntries(t, tc.name, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, g.RawMatrix().Data, tc.dt)
	}
}

// VNWHVK: at an exact pole at DC, a route that reads a pole residue as
// rounding (couplings below the decoupled-mode tolerance 100ε‖A‖, or a
// Laurent coefficient below 100nε of its terms) can return the Inf sign of
// the next order. Where the residue is clear of ε of its own exact terms,
// DCGain gives the entry the exact limit's sign with a nil error. Fuzz rows
// frozen as decoded on arm64.
func TestDCGainInfSignFromResidueReadAsRounding(t *testing.T) {
	checkDCGainInfSignRows(t, []dcGainInfSignRow{
		{"decoupled modes, five-fold z = 1 (fuzz 0xa3)", 5, 2, 1,
			[]float64{1, 0, 0, 0, -6.938893903907228e-18, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
			[]float64{-0.9928294791619291, 0.6044134701545401, -0.15039772802928933, 0.9293185617772851, 0.9123670470106603, -0.1072302167655213, 0.6385119607313409, -0.98668647653316, -0.5707646438938166, -0.42064473661916085},
			[]float64{823.3903805643572, 364.11928583243446, -1058.822312971188, 320.49312805930276, 851.5981438651463},
			[]float64{-12.418206970610514, -11.34203853322024}, 1},
		{"Laurent expansion, triple s = 0 (fuzz 0xa0)", 5, 1, 1,
			[]float64{0, 0, -3.350644874697394, 13.948632095891256, -33.51377823233131, 0, 0, -3.350644874697394, 13.948632095891256, -33.51377823233131, 0, 0, -3.3514372938198327, 11.659267594801127, -26.271666165263092, 0, 0, -1.0599334177091932, 3.35217024235557, -7.248689302394539, 0, 0, 0, 0, 0},
			[]float64{-0.48136868822172835, -0.9733418635898161, -0.03936692850136947, 0.952280649068488, 0.5488348447967344},
			[]float64{93.2162351132746, -317.55048379097644, 112.10528161088675, 245.06554792078705, -270.55955129184355},
			[]float64{-3.9205570977554167}, 0},
		{"decoupled modes beside a stiff pole", 3, 1, 1,
			[]float64{0, 0.4, -1.3, 0, -0.5, 0.9, 0, 0, -3.8e15},
			[]float64{0.7, -1.2, 0.4}, []float64{1.1, 0.3, -0.8}, []float64{0.5}, 0},
	}, nil)
}

// VNWHVK: with no pole at DC but eigenvalues near 1e113, the transfer
// coefficients overflow; DCGain returns the exact limit, as it does when
// the DC solve itself overflows. Fuzz 0xbf frozen as decoded on arm64.
func TestDCGainTransferCoefficientOverflowExactLimit(t *testing.T) {
	const n, m, p, dt = 5, 3, 3, 1
	a := []float64{0.6935312516398303, 0.7204263599708978, -4.374388811737615e+106, 4.882973583183672e+106, -1.415391020646735e+99, -0.7204263599708419, 0.6935316526290295, -4.374388811737615e+106, 4.882973583183672e+106, -1.415391020646735e+99, 0, 0, 9.137282084882717e+111, 1.663198667360618e+113, -4.882971077630522e+106, 0, 0, -1.6631986673604896e+113, 9.137374658494493e+111, 4.374387523576169e+106, 0, 0, 0, 0, 0.9999999920562479}
	b := []float64{-0.3808377475205685, 0.9904239867122461, -0.6191856356673074, 0.7890726026827783, 0.1319092404465915, -0.9222603350831519, 0.8029897413553648, -0.9198528521038003, 0.12577851702398002, -0.3594749722625363, -0.6240283613360986, 0.9895515470232578, -0.9953080093555333, 0.5859991407319469, 0.4036292423607582}
	c := []float64{-0.06036340031800291, 0.042336305391199605, 0.032989628652959097, -0.06366671090760545, 0.00817593813292827, -0.030632198062876134, 0.0641568086525815, -0.010850255698399406, -0.05714125972239396, 0.047796601909774646, 0.0135058055825191, 0.05580336232686611, -0.049587095254878484, -0.02374138123292823, 0.06493777696594777}
	d := []float64{-0.4297948959555326, -0.08261067339622216, 0.2776159580483855, 0.6097702622422342, 0.2975196748273661, -0.061702727378413626, -0.7543279340234785, -0.49514764173634523, -0.15779442412824884}
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("DCGain: %v", err)
	}
	checkDCGainExactLimitEntries(t, "transfer overflow", n, m, p, a, b, c, d, g.RawMatrix().Data, dt)
}

// VNWHVK: dcGainAgainstExactLimit keeps a route's gain unless an entry is
// the opposite infinity of the strict exact limit; it then takes the exact
// sign when the rounded limit agrees, and returns the strict limit with
// ErrAccuracyUnverified when the sign rests on a residue below ε.
func TestDCGainAgainstExactLimitInfSign(t *testing.T) {
	inf := math.Inf(1)
	d := func(v ...float64) *mat.Dense { return mat.NewDense(1, len(v), v) }
	for _, tc := range []struct {
		name           string
		g, strict, rnd *mat.Dense
		want           []float64
		unverified     bool
	}{
		{"agree", d(inf, 2), d(inf, 3), d(inf, 3), []float64{inf, 2}, false},
		{"finite route, Inf strict", d(4, 2), d(-inf, 3), d(5, 3), []float64{4, 2}, false},
		{"residue clear of rounding", d(inf, 2), d(-inf, 3), d(-inf, 3), []float64{-inf, 2}, false},
		{"residue below rounding", d(inf, 2), d(-inf, 3), d(7, 3), []float64{-inf, 3}, true},
	} {
		g, err := dcGainAgainstExactLimit(tc.g, &dcExactLimits{tc.rnd, tc.strict}, "test route")
		if errors.Is(err, ErrAccuracyUnverified) != tc.unverified || !tc.unverified && err != nil {
			t.Errorf("%s: error %v, unverified %v", tc.name, err, tc.unverified)
		}
		if got := g.RawMatrix().Data; !slices.Equal(got, tc.want) {
			t.Errorf("%s: gain %v, want %v", tc.name, got, tc.want)
		}
	}
	if g, err := dcGainAgainstExactLimit(d(inf), nil, "test route"); err != nil || g.At(0, 0) != inf {
		t.Errorf("nil limits: %v, %v", g, err)
	}
}

// VNWHVK: the strict reading of exactDCGainLimitRounded is the float model's
// exact limit, by the independent rational oracle.
func TestExactDCGainLimitRoundedStrictIsExactLimit(t *testing.T) {
	for _, tc := range []dcGainInfSignRow{
		{"stiff decoupled", 3, 1, 1, []float64{0, 0.4, -1.3, 0, -0.5, 0.9, 0, 0, -3.8e15}, []float64{0.7, -1.2, 0.4}, []float64{1.1, 0.3, -0.8}, []float64{0.5}, 0},
		{"double integrator, cancelled residue", 3, 2, 2, []float64{1, 0.25, -0.5, 0, 1, 0.75, 0, 0, 0.4}, []float64{1, -2, 0, 1, 0.5, 0.3}, []float64{0.25, 0, 1, 0, 1, -1}, []float64{0, 1, -1, 0.5}, 1},
	} {
		x0 := 0.0
		if tc.dt > 0 {
			x0 = 1
		}
		A, B, C, D := mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b), mat.NewDense(tc.p, tc.n, tc.c), mat.NewDense(tc.p, tc.m, tc.d)
		_, strict, overflow := exactDCGainLimitRounded(A, B, C, D, x0)
		if overflow {
			t.Fatalf("%s: overflow", tc.name)
		}
		checkDCGainExactLimitEntries(t, tc.name, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, strict.RawMatrix().Data, tc.dt)
	}
}
