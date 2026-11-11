package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type boundaryBandCase struct {
	A, B, Q, R, E *mat.Dense
}

// A descriptor Dare/Dlqr whose mode at −1 is controllable through B at
// 7.3e-12 (zw46Gen reg seed 217, native literals): at the computed λ
// σmin[c(A − λE), B] is 3.7× the rounding plus first-order bound and
// 0.38× the 10× band (MZTFJJ). Within the band is no proof that no
// stabilizing solution exists: ErrNotConverged, not ErrNoStabilizing.
func TestDareBoundaryBandNotConverged(t *testing.T) {
	c := boundaryBandCase{
		A: mat.NewDense(5, 5, []float64{-1.148126828319257, -0.33206126831551636, 1.3493607790684883, -1.1502058236026327, 0.804770324522356, 0.08870217078122497, 0.10238696791449557, -0.17775028471912155, 0.7749520337121794, 0.19357016907681687, 1.4543543138346486, 0.2924933502091493, -0.012680323767917, 0.49999393136227216, -0.5532181835841907, 0.567879809677987, 0.3052859411288899, 1.9691911586582882, 0.507163016485248, 0.5025463947322835, 0.7824660646490744, -0.21354396577274673, -0.20711271740531817, 0.9253740125861831, -0.831648716833636}),
		B: mat.NewDense(5, 1, []float64{1.064826657717095, -0.7463895675722435, 1.611195710399849, 1.1388206372520882, -0.3089855341670672}),
		Q: mat.NewDense(5, 5, []float64{1.8281976579746781, -0.2322665255903235, -1.2644670683968116, -1.4221386939063194, -0.012758820876249414, -0.2322665255903235, 1.783408984782593, -0.3834799025640537, 0.29005106270033004, -1.2500575415322723, -1.2644670683968116, -0.3834799025640537, 2.554320314526758, 2.431042190668749, 2.1440679743420046, -1.4221386939063194, 0.29005106270033004, 2.431042190668749, 3.25223206120906, 1.8195565366665152, -0.012758820876249414, -1.2500575415322723, 2.1440679743420046, 1.8195565366665152, 3.1687836919771017}),
		R: mat.NewDense(1, 1, []float64{0.5112482085051272}),
		E: mat.NewDense(5, 5, []float64{0.3656876882533476, 0.431450765813746, 0.14406143835346588, 1.2568630807013539, -1.7838793677317362, 0.005140259297570443, 1.1594377649504082, 1.2031055051691395, 0.4688355382506739, -0.926797177149353, -0.2781396095093954, -1.6777093519050088, -0.8335654838878862, 0.4964539064562031, 0.5359699171357478, -0.29081473868867036, 0.737041786383285, -1.2103077873247714, 0.8696624116717665, 2.068534881554315, -0.28896662651448585, 0.6431603704653701, -0.19078431430823353, -0.4039301036220846, 1.9913395122806623}),
	}
	for _, api := range []string{"Dare", "Dlqr"} {
		solve := Dare
		if api == "Dlqr" {
			solve = Dlqr
		}
		res, err := solve(c.A, c.B, c.Q, c.R, &RiccatiOpts{E: c.E})
		if errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) {
			t.Fatalf("%s: %v, want no unstabilizability verdict", api, err)
		}
		if err == nil {
			continue
		}
		if !errors.Is(err, ErrNotConverged) && !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("%s: %v, want ErrNotConverged or ErrAccuracyUnverified", api, err)
		}
		if res != nil && !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("%s: result with %v", api, err)
		}
	}
}

// The same generator with B's row at the mode exactly zero (zw46Gen reg
// seed 1): the mode at 1 is uncontrollable to rounding, within the
// first-order bound, and the verdict stays ErrNoStabilizing.
func TestDareBoundaryExactNoStabilizing(t *testing.T) {
	c := boundaryBandCase{
		A: mat.NewDense(2, 2, []float64{0.5163343114905039, -0.7932662655761806, -0.749294371375942, 1.260690523687388}),
		B: mat.NewDense(2, 1, []float64{0.07965388629890349, 1.0890826823072717}),
		Q: mat.NewDense(2, 2, []float64{5.330284686478571, 2.155372416973004, 2.155372416973004, 2.610689342594011}),
		R: mat.NewDense(1, 1, []float64{1.1609607581887609}),
		E: mat.NewDense(2, 2, []float64{0.6172382778963376, -0.8579368965978433, 0.6303340121083366, 0.37646920595257893}),
	}
	for _, api := range []string{"Dare", "Dlqr"} {
		solve := Dare
		if api == "Dlqr" {
			solve = Dlqr
		}
		if _, err := solve(c.A, c.B, c.Q, c.R, &RiccatiOpts{E: c.E}); !errors.Is(err, ErrNoStabilizing) {
			t.Fatalf("%s: %v, want ErrNoStabilizing", api, err)
		}
	}
}
