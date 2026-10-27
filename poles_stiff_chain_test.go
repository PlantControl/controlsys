package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffChainOraclePoles returns the finite poles of the structured oracle
// (F, W) of tc, f_ii/w_ii over its leading diagonal block (the rows of W
// before the nilpotent chains), each refined to the root of det(A − λE) of
// the stored data at pencilOraclePrec bits.
func stiffChainOraclePoles(t *testing.T, tc stiffChainCase, sys *System) []float64 {
	t.Helper()
	n := tc.n
	A, E := pencilOracleMatrix(sys.A), pencilOracleMatrix(sys.E)
	var poles []float64
	for i := 0; i < n && tc.w[i*n+i] != 0; i++ {
		x0 := tc.f[i*n+i] / tc.w[i*n+i]
		r, ok := pencilOracleRealRoot(A, E, x0)
		if !ok || math.Abs(r-x0) > 1e-6*math.Abs(x0) {
			t.Fatalf("%s: structured pole %v not a root of the data pencil (%v, %v)", tc.name, x0, r, ok)
		}
		poles = append(poles, r)
	}
	return poles
}

// missingPoles returns the oracle poles want without a pole of got within
// rel relative, matched one to one.
func missingPoles(got []complex128, want []float64, rel float64) []float64 {
	used := make([]bool, len(got))
	var miss []float64
	for _, w := range want {
		best, bestD := -1, math.Inf(1)
		for j, g := range got {
			if d := math.Hypot(real(g)-w, imag(g)) / math.Abs(w); !used[j] && d < bestD {
				best, bestD = j, d
			}
		}
		if bestD > rel {
			miss = append(miss, w)
			continue
		}
		used[best] = true
	}
	return miss
}

// TestPoles_StiffChainDeflatedFiniteSet: IP4PC3. An index ≥ 2 chain beside
// a finite pole small in E (E singular value 1.4e-9..9.7e-9 relative, MATLAB
// rank keeps it): QZ on the whole pencil smears the chain to |λ| ~ ε^(−1/k)
// and gives the stiff pole β = 0, so base Poles kept 2–3 chain members and
// lost the stiff pole (condChain 955 read unstable). The finite block of
// deflateStiffChain, the split the transfer function uses, has exactly the
// oracle poles.
func TestPoles_StiffChainDeflatedFiniteSet(t *testing.T) {
	for _, tc := range append(append([]stiffChainCase{}, stiffChainResplitCases...), stiffChainClosedCases[:2]...) {
		sys, _ := tc.systems(t)
		want := stiffChainOraclePoles(t, tc, sys)
		got, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", tc.name, err)
		}
		if miss := missingPoles(got, want, 1e-5); len(got) != len(want) || len(miss) > 0 {
			t.Errorf("%s: Poles %v, want %v (missing %v)", tc.name, got, want, miss)
		}
		wantStable := true
		for _, p := range want {
			wantStable = wantStable && p < 0
		}
		if st, err := sys.IsStable(); err != nil || st != wantStable {
			t.Errorf("%s: IsStable %v, %v; want %v", tc.name, st, err, wantStable)
		}
	}
}

// smallEPoleCase is a descriptor pencil (A, E), row-major, beside the
// finite poles of its structured oracle.
type smallEPoleCase struct {
	name  string
	n     int
	poles []float64
	a, e  []float64
}

// system returns the pencil of tc as a descriptor model and its oracle
// poles refined to the roots of det(A − λE) of the stored data at
// pencilOraclePrec bits.
func (tc smallEPoleCase) system(t *testing.T) (*System, []float64) {
	t.Helper()
	n := tc.n
	sys, err := NewDescriptor(mat.NewDense(n, n, tc.a), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), mat.NewDense(n, n, tc.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	A, E := pencilOracleMatrix(sys.A), pencilOracleMatrix(sys.E)
	want := make([]float64, len(tc.poles))
	for i, p := range tc.poles {
		r, ok := pencilOracleRealRoot(A, E, p)
		if !ok || math.Abs(r-p) > 1e-6*math.Abs(p) {
			t.Fatalf("%s: oracle pole %v not a root of the data pencil (%v, %v)", tc.name, p, r, ok)
		}
		want[i] = r
	}
	return sys, want
}

// TestPoles_SmallEPoleBesideChainKept: F4OPPD. A finite pole small in E
// (E singular value 1.2e-8..1.9e-8 relative, not stiff) beside an index ≥ 2
// chain and a pole large in A: base Poles carries every oracle pole (beside
// chain members QZ smears); a staircase deflation took the small singular
// value for a chain zero and dropped the pole. Every oracle pole must stay.
func TestPoles_SmallEPoleBesideChainKept(t *testing.T) {
	for _, tc := range smallEPoleCases {
		sys, want := tc.system(t)
		got, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", tc.name, err)
		}
		if miss := missingPoles(got, want, 1e-6); len(miss) > 0 {
			t.Errorf("%s: Poles %v miss oracle poles %v", tc.name, got, miss)
		}
	}
}

// TestPoles_StiffChainPoleKeepsQZValue: IP4PC3. An index-2 chain beside a
// pole small in E (stiff E singular value 1.6e-10..2.7e-10 relative): base
// Poles has the stiff pole to 1e-8 beside two smeared chain members; the
// deflateStiffChain block moves it by 1.8e-4 and 3.5e-4 (staircase turns
// of order n·ε·σ₁/σs). Poles must be the oracle set with the stiff pole at
// QZ's accuracy.
func TestPoles_StiffChainPoleKeepsQZValue(t *testing.T) {
	for _, tc := range stiffChainQZValueCases {
		sys, want := tc.system(t)
		got, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", tc.name, err)
		}
		if miss := missingPoles(got, want, 1e-6); len(got) != len(want) || len(miss) > 0 {
			t.Errorf("%s: Poles %v, want %v (missing %v)", tc.name, got, want, miss)
		}
	}
}

var stiffChainQZValueCases = []smallEPoleCase{
	{
		name: "mix PCG(42, 6061)", n: 4,
		poles: []float64{-110.24405344196043, 1426975220.5355685},
		a: []float64{
			-23.106543215970053, 26.002790343470835, 35.757950299280651, 20.899203316283334,
			-9.8368227716379781, 11.013802543342766, 15.228156645973307, 8.9984981828187216,
			-1.6987991003005194, 1.9347190349710712, 2.6107792790428759, 1.3334863573509488,
			-15.690670579924049, 17.622076816556124, 24.26106770312801, 14.258021201303722,
		},
		e: []float64{
			0.14983465593715778, -0.15529859791120029, -0.25088841264367689, -0.14321903087362439,
			0.097448814326347694, -0.11093077513704654, -0.14827943242706709, -0.08828716555919279,
			0.022083126147350928, -0.026077541926508036, -0.032193154049582895, -0.019547313344893704,
			0.13469238478882428, -0.14945408305221422, -0.21075898010470376, -0.1239246984354008,
		},
	},
	{
		name: "mix PCG(638, 6061)", n: 4,
		poles: []float64{25123.134512833632, -347302838.20609295},
		a: []float64{
			-365.26320251301769, -741.71229932414053, -523.07740522709196, 936.53675527126961,
			-891.6658386136362, -1812.970269573515, -1277.6703093333258, 2288.460846217969,
			-717.8024736285995, -1459.4271440651298, -1028.500413950642, 1842.1189246642202,
			34.067931111083112, 69.042928683454946, 48.721836299836511, -87.200364415902868,
		},
		e: []float64{
			0.018220446061089805, -0.033935369717634306, 0.0017722608972167246, 0.01703894068486999,
			-0.3484645451368138, -0.030009873022821109, -0.26669340299643218, 0.28443699774497894,
			-0.16451684314921763, -0.039780794963953169, -0.1346922468328943, 0.15730881135520475,
			-0.012112116879360429, 0.0045621614357077671, -0.0073481354762158831, 0.004848594703902552,
		},
	},
}

var smallEPoleCases = []smallEPoleCase{
	{
		name: "mix PCG(752, 6061)", n: 5,
		poles: []float64{502753.39346874005, -13176208.013643585},
		a: []float64{
			1055.2301336995099, 6592.5768592030345, 4500.9302511026981, 6893.7415837356211, 3096.9479847385091,
			-1643.0545147426844, -10265.347656076938, -7008.4620447774569, -10734.06658041025, -4822.1066986694386,
			-233.46903650496139, -1458.1703061546934, -995.50482960420447, -1524.7828370679158, -684.9933024004888,
			-14868.195899805305, -92888.418161772017, -63418.998637900957, -97132.643235881333, -43634.540167667292,
			5889.8294160972118, 36796.46869645502, 25122.666425275085, 38477.884601023339, 17285.312015357846,
		},
		e: []float64{
			0.030325502041212048, -0.033842056195514975, -0.039877924961924709, 0.011120741614785851, -0.053532331868511737,
			-0.054028700910368703, 0.013178193201513484, 0.064176571455683937, -0.018417117703534446, 0.085883670763539727,
			-0.025491466476166597, -0.21410113339962394, -0.006902007448322727, -0.0093188796887229922, -0.0074654714206144899,
			-0.0084837579794308556, -0.46868013893666666, -0.2100829651193821, -0.20358491077268071, -0.18948642071725946,
			-0.017853613141084159, 0.24056339062245141, 0.12366152505649931, 0.083260764892139155, 0.12450428448738249,
		},
	},
	{
		name: "mix PCG(247, 6061)", n: 7,
		poles: []float64{-677620.4384695953, -51827009.03272943, 0.4851087063084701},
		a: []float64{
			15806.835173014806, -20157.757223482204, -17798.161688320368, 34822.035175036952, -11842.478240547853, 6519.4924350726451, -1629.4540216760031,
			43425.506859867426, -55378.493234413014, -48895.367691291816, 95664.327163306589, -32533.444173786545, 17910.325348275433, -4477.2257507316644,
			-75295.048084264432, 96019.988228254733, 84779.348829442111, -165872.38886336962, 56409.776037150339, -31054.012572108564, 7762.9545252372291,
			-48490.386880596867, 61836.768499413643, 54597.64256108542, -106821.36383732541, 36328.108186327925, -19999.032421598513, 4999.0548390741933,
			57629.029310691869, -73490.947446225342, -64887.863892327987, 126953.28492607431, -43174.514570762978, 23767.871705553876, -5941.2786423152193,
			-41247.703218897877, 52601.215155019308, 46442.573909661151, -90866.411889094932, 30902.010156341079, -17011.879940605839, 4252.7159008865601,
			10409.735531744282, -13275.321817376023, -11721.241477969341, 22932.682342588087, -7798.8071518638772, 4293.2670207548763, -1072.9524171406526,
		},
		e: []float64{
			-0.037597949830544299, 0.046903309550126518, 0.098089906122569889, -0.058998966695979226, -0.20151742259317384, -0.15183090813059172, 0.10381896059052478,
			-0.033750535925555832, 0.14195121694938156, 0.25878312770964013, 0.032211720494584338, 0.14712555573363362, 0.23107998144416464, 0.10291678140144048,
			-0.047816380809702623, -0.27216627167562857, -0.090526339431873576, 0.24741907108978087, -0.0027901605055824022, 0.30687475577524526, -0.17531022195071577,
			-0.00065891369648328436, -0.14617790128520772, -0.068409122285348567, 0.13423244361217579, -0.10115871389225987, 0.058014267876168105, -0.054458019882740377,
			-0.085108259307463505, 0.093328270967931115, 0.01359964249695403, -0.26107454433159649, 0.015929933296124337, -0.16478168676144259, -0.019253984630374334,
			0.22124071419139302, 0.036406612455397981, -0.12421224250543461, 0.17177181834038785, 0.086999693001367229, -0.038453563977531149, 0.073295507949440841,
			-0.093727633314202766, -0.081103253725402905, -0.10938333850862009, -0.11877706135776403, 0.13118675808851402, 0.06388224559178475, -0.18512540546264064,
		},
	},
}
