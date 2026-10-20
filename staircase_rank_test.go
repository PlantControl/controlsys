package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func staircaseRankSum(k []int) int {
	s := 0
	for _, v := range k {
		s += v
	}
	return s
}

// 65BPNH/GHNJYJ: a 1e-13 coupling against ‖A‖ = 1e4 is rounding noise to
// MATLAB ctrbf's absolute default (10·n·‖A‖₁·eps ≈ 9e-11); state 3 then is
// uncontrollable, exactly as with the coupling set to 0.
func TestStaircaseNoiseCouplingNotRank(t *testing.T) {
	for _, c := range []float64{1e-13, 0} {
		A := mat.NewDense(4, 4, []float64{-1e4, 0, 0, 0, 0, -1, 0, 0, 1, 0, -2, 0, 0, c, 0, -3})
		B := mat.NewDense(4, 2, []float64{1, 0, 0, 1, 0, 0, 0, 0})
		C := mat.NewDense(1, 4, []float64{1, 1, 1, 1})
		f, err := CtrbF(A, B, C, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got := staircaseRankSum(f.K); got != 3 {
			t.Errorf("coupling %g: CtrbF K %v, want sum 3", c, f.K)
		}
		sys, err := New(A, B, C, mat.NewDense(1, 2, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := sys.Reduce(&ReduceOpts{Mode: ReduceUncontrollable})
		if err != nil {
			t.Fatal(err)
		}
		if r.Order != 3 {
			t.Errorf("coupling %g: ReduceUncontrollable order %d, want 3", c, r.Order)
		}
	}
}

// 65BPNH/w12f#41: state 3 is exactly unobservable (zero C column and zero A
// column outside its diagonal); a later staircase block of size ~1e-5 must
// not count its rounding noise as rank.
func TestStaircaseSmallBlockNoiseNotRank(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{
		197.11151535898398, -125.58165054554455, 129.30370945971754, 0, 59.76947800347428, -5.265470231350326,
		-173.82085079604235, 110.9846205585125, -114.44113190250626, 0, -9.172863226049667, 4.351756247287568,
		136.7917714371087, -87.3782153938056, 89.80570195554552, 0, 198.7113470934537, -4.207586106182046,
		65.96476980969965, 168.26520163148763, 20.78475297333022, -0.31697863849222285, -106.39718338430359, 188.40590032108176,
		0, 0, 0, 0, -0.15848931924611143, 0,
		-35.79801621875614, 23.572121758723704, -24.270765434267762, 0, -196.34760124624893, -5.173930199602784})
	B := mat.NewDense(6, 1, []float64{0.5260775173811046, -0.01238866346289056, -0.9667981925794609, 0.3422733332214033, 0, 0.9601702866503661})
	C := mat.NewDense(2, 6, []float64{
		-4.3137684497061916e-07, -2.1079579943077972e-07, -8.568887533689471e-07, 0, 0.44660305575466125, 9.931849187581927e-07,
		-9.111302618846769e-07, 4.685166713003763e-07, -9.87479769908865e-07, 0, 0.4537233907250981, 6.845466664428065e-07})
	f, err := ObsvF(A, B, C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := staircaseRankSum(f.K); got != 5 {
		t.Errorf("ObsvF K %v, want sum 5", f.K)
	}
	sys, err := New(A, B, C, mat.NewDense(2, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := sys.Reduce(&ReduceOpts{Mode: ReduceUnobservable})
	if err != nil {
		t.Fatal(err)
	}
	if r.Order != 5 {
		t.Errorf("ReduceUnobservable order %d, want 5", r.Order)
	}
}
