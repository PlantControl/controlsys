package controlsys

import (
	"errors"
	"runtime"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

// withIterationCap sets an extended-precision iteration cap for one test.
func withIterationCap(t *testing.T, limit *int, v int) {
	t.Helper()
	old := *limit
	*limit = v
	t.Cleanup(func() { *limit = old })
}

// boundaryError is the error base Dare returned (before the extended
// solve existed) on the stabilizable near-singular-E plants below.
const boundaryError = "Dare: controlsys: no stabilizing solution exists: a mode on or beyond the stability boundary is unstabilizable"

// A near-singular-E Dare, Dlqr or Kalman whose extended-precision solve
// hits an iteration cap (SDA doubling or its exponent spread, stableLoop
// squaring or the Smith doubling of the perturbed Hewer step) returns exactly the float64
// pencil's error, boundary verdict included, as before the extended solve
// existed (O2YDYA round 3).
func TestDare_NearSingularEIterationCapKeepsFloatError(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0.3431284077914542, -0.7926130901776904})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, -2.5367599785728485, -0.7294612311954138})
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	KA := mat.NewDense(2, 2, []float64{0.923010477433099, 2.0820636924475, -0.9200537881286719, -1.361812134364599})
	KB := mat.NewDense(2, 3, []float64{-0.7649133317207344, -0.0015390678071096064, 1.2371736568788594, 0.6699795793910823, 0.37405306155207996, 1.3833774115150517})
	KE := mat.NewDense(2, 2, []float64{0.5004371458498397, 2.921975405177471, 0, 8.520053948604121e-13})
	Rn := mat.NewDense(3, 3, []float64{4.7486349226800515, -1.1064545236921663, 1.512245701955511, -1.1064545236921663, 2.42778007790058, 0.6521422583478569, 1.512245701955511, 0.6521422583478569, 3.6215308801989963})
	sys, err := NewDescriptor(KA, KB, mat.DenseCopyOf(KB.T()), mat.NewDense(3, 3, nil), KE, 1)
	if err != nil {
		t.Fatal(err)
	}
	check := func(t *testing.T, api string, err error, want string) {
		t.Helper()
		if err == nil || err.Error() != want || !errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("%s: %v, want %q", api, err, want)
		}
	}
	for _, c := range []struct {
		name  string
		limit *int
		v     int
	}{{"doubling", &extendedDoublings, 2}, {"spread", &extendedSpreadBits, 0}, {"stableLoop", &stableSquarings, 1}, {"smithStein", &steinDoublings, 1}} {
		t.Run(c.name, func(t *testing.T) {
			withIterationCap(t, c.limit, c.v)
			for _, ns := range []bool{false, true} {
				opts := &RiccatiOpts{E: E, NoScaling: ns}
				_, err := Dare(A, B, eyeDense(2), eyeDense(2), opts)
				check(t, "Dare", err, boundaryError)
				_, err = Dlqr(A, B, eyeDense(2), eyeDense(2), opts)
				check(t, "Dlqr", err, boundaryError)
			}
			_, err := Kalman(sys, eyeDense(3), Rn, nil, nil)
			check(t, "Kalman", err, "Kalman: "+boundaryError)
		})
	}
}

// bnd generator seed 82 (O2YDYA round-1 review): a stable uncontrollable
// mode at −(1 − 1e-9) beside fast modes, σ_min(E) 4.3e-14. Repeated
// squaring drove the fast entries to exponents near −2³⁰, and each
// big.Float sum then shifted its mantissa by that gap: 21.9 GB allocated,
// 4.8 GB resident. Base errBoundaryUnstabilizable; now Accuracy 1.
func TestDare_NearSingularENearUnitModeBoundedCost(t *testing.T) {
	skipSlowUnderRace(t)
	A := mat.NewDense(4, 4, []float64{-0.5665014768372381, 0, 0, 0, -1.5980210024765213, -0.06465621784083286, -0.0828969416783879, 0.9847150508953924, 0.7217324705211728, 0.53862200268466, -0.17110939638464817, 1.1214081883588474, -1.0400461037212216, -1.5146521616137787, 0.5536448758199233, 0.8495075925372486})
	B := mat.NewDense(4, 3, []float64{0, 0, 0, -0.32533712585683494, -0.3850771055627178, 0.819380445871456, -0.9629967805389414, 0.5603125968042595, 1.6017953055098246, -0.2470353648529452, -0.5959107444303582, -0.840210045986464})
	E := mat.NewDense(4, 4, []float64{0.5665014774037396, 0, 0, 0, 0, 0.749141452362936, 0.9986910131853461, -0.40126864570400955, 0, 0, 1.1127408060359611, 0.27638968643835526, 0, 0, 0, 4.331142713859568e-14})
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	t0 := time.Now()
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if _, err := Dare(A, B, eyeDense(4), eyeDense(3), opts); err != nil {
			t.Errorf("Dare NoScaling %v: %v", ns, err)
		}
		if _, err := Dlqr(A, B, eyeDense(4), eyeDense(3), opts); err != nil {
			t.Errorf("Dlqr NoScaling %v: %v", ns, err)
		}
	}
	elapsed := time.Since(t0)
	runtime.ReadMemStats(&after)
	if elapsed > 20*time.Second {
		t.Errorf("4 calls took %v", elapsed)
	}
	if raceEnabled {
		return
	}
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 1<<28 {
		t.Errorf("4 calls allocated %d MB", alloc>>20)
	}
}
