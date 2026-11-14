package controlsys

import (
	"errors"
	"math"
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

// A discrete model with a pole within the dc rounding disk whose exact DC
// gain DCGain certifies (A2OX65 census, x-dplant-pcg 63.0): the gain
// search reads DC as that certified gain, not as no sign at all. The
// −3 dB crossing (320-bit oracle 4.593653946133242e-16) lies where
// ddAtPole vetoes the double-double response, so Bandwidth returns
// ErrAccuracyUnverified; +Inf with nil, which claims no crossing, was the
// defect.
func TestBandwidthCertifiedDCBesideDiskPole(t *testing.T) {
	for _, tc := range []struct {
		row    dcEndRow
		oracle float64
	}{
		{dcEndRow{"x-dplant-pcg/63.0",
			[]float64{0.1222395759887928, 0, 0, 1, 0, -0.16369789843386065, 0, 1, 1.1636978984338606},
			[]float64{-0.15181541674771168, 0.17295769163747726, 0},
			[]float64{0, 0, 1}, 0}, 4.593653946133242e-16},
	} {
		t.Run(tc.row.name, func(t *testing.T) {
			sys := tc.row.sys(t)
			if _, err := sys.DCGain(); err != nil {
				t.Fatalf("DCGain error %v; want a certified gain", err)
			}
			bw, err := Bandwidth(sys, -3)
			switch {
			case err == nil && !(math.Abs(bw-tc.oracle) <= 1e-6*tc.oracle):
				t.Fatalf("Bandwidth = %g with nil error; oracle %g", bw, tc.oracle)
			case err != nil && !errors.Is(err, ErrAccuracyUnverified):
				t.Fatalf("Bandwidth error %v; want ErrAccuracyUnverified or %g", err, tc.oracle)
			}
		})
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
