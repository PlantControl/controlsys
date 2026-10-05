package controlsys

import (
	"errors"
	"math/cmplx"
	"strings"
	"testing"
)

func TestSeriesDomainMismatchNamesSampleTimes(t *testing.T) {
	d := makeSISO(0.5, 1, 1, 0)
	d.Dt = 0.1
	_, err := Series(d, makeSISO(-1, 1, 1, 0))
	if !errors.Is(err, ErrDomainMismatch) || !strings.Contains(err.Error(), "sample times 0.1 and 0") {
		t.Errorf("err = %v, want sample-time detail wrapping ErrDomainMismatch", err)
	}
}

// Next to a pole within 1e-9 of the unit circle, s − v is about 1e-9 while
// its terms are O(1); pencilAt must keep it to a few ε relative, against
// a 200-bit z − v at the exact on-circle point.
func TestFrequencyPointPencilAtNearCircle(t *testing.T) {
	const dt = 0.1
	for _, th := range []float64{1e-4, 0.003, 1.4, 3.11, -2.2} {
		w := th / dt
		pt := newTimeDomain(dt).frequencyPoint(w)
		z := frExactPoint(w, dt)
		for _, r := range []float64{1 - 1e-9, 1 + 1e-9, 0.5} {
			v := cmplx.Rect(r, th)
			want := z.sub(frBigC(v)).complex()
			got := pt.pencilAt(v) / pt.scale()
			if e := cmplx.Abs(got-want) / cmplx.Abs(want); !(e <= 1e-15) {
				t.Errorf("θ=%g r=%v: s−v = %v, exact %v, rel err %.3g", th, r, got, want, e)
			}
		}
	}
}
