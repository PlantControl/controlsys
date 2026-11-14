package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

type dcEndRow struct {
	name    string
	a, b, c []float64
	d       float64
}

func (r dcEndRow) sys(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(len(r.b), 1, 1, r.a, r.b, r.c, []float64{r.d}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sameCrossings(got, want []float64, rel float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !(math.Abs(got[i]-want[i]) <= rel*want[i]) {
			return false
		}
	}
	return true
}

// Discrete models with a pole within the dc rounding disk whose exact DC
// gain DCGain certifies (A2OX65 census): the gain search reads DC as that
// certified gain, and the −3 dB crossing, inside the disk, closes on the
// stored model's exact responses there (exactResponseAt, Q1 ruling (c)):
// nil and the 320-bit oracle bandwidth. +Inf with nil, a false claim of no
// crossing, and later ErrAccuracyUnverified under the ddAtPole veto were
// the defects. x-dplant-pcg 63.0 is exactly −3 dB at 4.593653946133242e-16
// (Fable, big.Rat); the F5 rows put a stiff block beside the disk, whose
// snapped responses the scan steps over to the disk edge (diskScan).
func TestBandwidthCertifiedDCBesideDiskPole(t *testing.T) {
	for _, tc := range []struct {
		row    dcEndRow
		oracle float64
	}{
		{dcEndRow{"x-dplant-pcg/63.0",
			[]float64{0.1222395759887928, 0, 0, 1, 0, -0.16369789843386065, 0, 1, 1.1636978984338606},
			[]float64{-0.15181541674771168, 0.17295769163747726, 0},
			[]float64{0, 0, 1}, 0}, 4.593653946133242e-16},
		{dcEndRow{"F5illCondBlockBesideStiff/1153",
			[]float64{1.6047273271825544, 0.011557314468999233, 0.33764257328723901, -1.6672380779179998, 0.44222904703767885, 1.8450843712564799,
				0.79620332791320148, 1.0152167296371886, -0.78988407264605431, -0.23584642219195984, 0.35773176321460376, -0.13012063671865282,
				0, 0, -2990.0471424811426, -0.81636192711873745, -0.5606837624217067, -0.90735798986995908,
				0, 0, 0, -35227.16915542425, -0.45014899719054963, -0.27101159076258707,
				0, 0, 0, 0, -1458.202847708204, 1.0446044724074832,
				0, 0, 0, 0, 0, -3797330.1974188583},
			[]float64{0.85022087961274417, 0.55098165264345766, 0.23764371146879237, -0.86401793032592211, 0.16598809038627202, 1.2677387988387023},
			[]float64{-0.5854712849769611, -1.5314164143798918, 1.453054601124079, -0.54505085925890562, -0.22494677576004041, -1.7345954209318781},
			0.64768970089697575}, 2.7443693792881339e-16},
		{dcEndRow{"F5illCondBlockBesideStiff/1247",
			[]float64{1.0814873722372307, -0.36359838445478371, -1.1432719629011732, 0.20294002630687921, 0.094477295309330467, -0.62913263974126488, 0, 0, -43682.370037232271},
			[]float64{0.08309229483413616, 0.25488454514509673, -0.082889472256297836},
			[]float64{1.1147821766727759, 0.43943754373356042, 0.0292435661761411},
			-1.9675371545937157}, 1.0319742088215906e-15},
		{dcEndRow{"F5illCondBlockBesideStiff/1451",
			[]float64{1.0202772955301354, -0.099726361452630474, -0.42379452766092851, -0.17124391520770199,
				-0.19821776631274307, 1.9748606060537013, -1.0287620745100745, 0.7079277761981122,
				0, 0, -56.489030449549084, 0.68077591464673581,
				0, 0, 0, -14337235.65274117},
			[]float64{0.070094069706672646, 0.247278202096129, -2.0621198456192573, 1.0914670784624292},
			[]float64{-0.18273878471815808, -0.50609503542266987, -0.96740847071126812, -0.91348414940165934},
			-0.015322836081546276}, 1.02588222212745e-15},
		{dcEndRow{"F8rotatedIntegrator/1912",
			[]float64{0.84256297171049266, -0.033884630535426633, -0.033884630535426612, 0.99270712742106004},
			[]float64{-0.035337132193933793, 0.42273462585819843},
			[]float64{-0.26562500647029985, 0.73676120216240215},
			0.42639565184676087}, 1.4429677368741009e-15},
	} {
		t.Run(tc.row.name, func(t *testing.T) {
			sys := tc.row.sys(t)
			if _, err := sys.DCGain(); err != nil {
				t.Fatalf("DCGain error %v; want a certified gain", err)
			}
			bw, err := Bandwidth(sys, -3)
			if err != nil || !(math.Abs(bw-tc.oracle) <= 1e-12*tc.oracle) {
				t.Fatalf("Bandwidth = %g, %v; want %g, nil", bw, err, tc.oracle)
			}
		})
	}
}

// FreqResponse off DC inside the dc rounding disk of a discrete model with
// an exact pole at z = 1 (A2OX65 census; dseries 0.0, x-d2-pcg 0.1): the
// point is no pole of the stored model, so the response is its exact value
// (exactResponseAt), oracle |G| from a 6000-bit solve, never Inf with nil.
// At 1e-300 the x-d2-pcg double pole puts |G| near 3.65e584: the exact real
// part overflows float64, the one place Inf stands off DC.
func TestFreqResponseOffDCInDiskExact(t *testing.T) {
	for _, tc := range []struct {
		name    string
		a, b, c []float64
		w       float64
		oracle  float64
	}{
		{"dseries/0.0", []float64{0.59039115052557856, 0, 1, 1}, []float64{-21.148952980220074, 51.632070467610212}, []float64{0, 1}, 1e-16, 321.27826653843408},
		{"dseries/0.0", []float64{0.59039115052557856, 0, 1, 1}, []float64{-21.148952980220074, 51.632070467610212}, []float64{0, 1}, 1e-300, 2.9551748428583678e+286},
		{"x-d2-pcg/0.1", []float64{0, -1, 0, 0, 1, 2, 0, 0, 0, 0.042981864298618298, 0, -0.065197226619843326, 0, -0.065858764449553867, 1, 0.56765273764980684},
			[]float64{1, 0, 0, 0}, []float64{0, 0.045979607166974933, 0, 1}, 1e-14, 3650720762530.4204},
		{"x-d2-pcg/0.1", []float64{0, -1, 0, 0, 1, 2, 0, 0, 0, 0.042981864298618298, 0, -0.065197226619843326, 0, -0.065858764449553867, 1, 0.56765273764980684},
			[]float64{1, 0, 0, 0}, []float64{0, 0.045979607166974933, 0, 1}, 1e-300, math.Inf(1)},
	} {
		sys, err := NewFromSlices(len(tc.b), 1, 1, tc.a, tc.b, tc.c, []float64{0}, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		fr, err := sys.FreqResponse([]float64{tc.w})
		if err != nil {
			t.Fatalf("%s ω=%g: FreqResponse error %v; want |G| %g", tc.name, tc.w, err, tc.oracle)
		}
		g := cmplx.Abs(fr.At(0, 0, 0))
		if math.IsInf(tc.oracle, 1) {
			if !math.IsInf(real(fr.At(0, 0, 0)), 0) || math.IsInf(imag(fr.At(0, 0, 0)), 0) {
				t.Fatalf("%s ω=%g: G = %v; want an overflowed real part beside a finite imaginary one", tc.name, tc.w, fr.At(0, 0, 0))
			}
			continue
		}
		if !(math.Abs(g-tc.oracle) <= 1e-12*tc.oracle) {
			t.Fatalf("%s ω=%g: |G| = %g; oracle %g", tc.name, tc.w, g, tc.oracle)
		}
	}
}

// AllMargin gain crossings next to DC of discrete loops with a pole inside
// the dc rounding disk (A2OX65 census; oracle: 320-bit scan of |L| on the
// stored model).
//   - dover-grid a=0.2,b=0.6,k=1.0: the pole at z = 1 is exactly cancelled,
//     DCGain certifies 0, and the pole-limit snap +Inf at DC is no bracket
//     end: no crossing below 4.30, where a resolve that kept the snap
//     claimed one (ErrAccuracyUnverified).
//   - x-dplant-pcg 4.0: certified finite DC 0.00584 and a crossing at
//     1.794e-15 inside the disk, closed on the double-double responses the
//     certified pole limit returns there; base missed it.
//   - dover-pcg 6.1: DC is a pole whose residue DCGain cannot certify; the
//     crossing at 4.0994e-16 does not close on resolved responses, so the
//     search returns ErrAccuracyUnverified, never [1.4376] nil.
func TestAllMarginGainCrossingDCEnd(t *testing.T) {
	for _, tc := range []struct {
		row  dcEndRow
		want []float64
		unv  bool
	}{
		{dcEndRow{"dover-grid/a=0.2,b=0.6,k=1.0",
			[]float64{0, -0.12, 0, 1, 0.8, 0, 0, 1, 1},
			[]float64{0.88, -1.2, 1},
			[]float64{0, 0, 1}, 0},
			[]float64{4.3011069084067906}, false},
		{dcEndRow{"x-dplant-pcg/4.0",
			[]float64{0.648787497656813, 0, 0, 1, 0, -0.5318105712028457, 0, 1, 1.5318105712028456},
			[]float64{-0.09571666678411585, 0.272532060064839, 0},
			[]float64{0, 0, 1}, 0},
			[]float64{1.7940723717410463e-15, 4.2255968855426795}, false},
		{dcEndRow{"dover-pcg/6.1",
			[]float64{1, 0, 0, 3.4785626920727757, 0, -0.06162486527476235, -5.428986306868575, 1, 0.5354796770692531},
			[]float64{1, 0, 0},
			[]float64{3.7070064661200997, 0, 1}, 0},
			[]float64{4.0994234703721725e-16, 1.4375576481858239}, true},
	} {
		t.Run(tc.row.name, func(t *testing.T) {
			res, err := AllMargin(tc.row.sys(t))
			if err != nil {
				if tc.unv && errors.Is(err, ErrAccuracyUnverified) {
					return
				}
				t.Fatalf("AllMargin error %v; want %v", err, tc.want)
			}
			if !sameCrossings(res.GainCrossFreqs, tc.want, 1e-6) {
				t.Fatalf("GainCrossFreqs = %v; oracle %v", res.GainCrossFreqs, tc.want)
			}
		})
	}
}
