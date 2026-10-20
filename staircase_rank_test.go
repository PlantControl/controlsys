package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
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

func reduceTestSys(t *testing.T, n, m, p int, a, b, c []float64, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, m, p, a, b, c, make([]float64, p*m), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// JBKTCQ: ‖B‖ ≈ 1e-15 against ‖A‖ ≈ 30 is a tiny but exact input gain of a
// 6-state model; minreal must not drop it as uncontrollable. Oracle: the
// transfer function it was realized from.
func TestReduceKeepsTinyInputGain(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{-8.70643320022765e-16, 2.9186581655941266e-16, 1.3545850066477854e-16}}},
		Den: [][]float64{{1, -5.609886783435264, 13.099973168748164, -16.29841103348229, 11.394314088807377, -4.243853507030152, 0.6578640663921657}},
		Dt:  0.01,
	}
	ss, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []int{0, -30, 30} {
		b := mat.DenseCopyOf(ss.Sys.B)
		b.Scale(math.Ldexp(1, k), b)
		sys := reduceTestSys(t, 6, 1, 1, ss.Sys.A.RawMatrix().Data, b.RawMatrix().Data, ss.Sys.C.RawMatrix().Data, ss.Sys.Dt)
		r, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if r.Order != 6 {
			t.Fatalf("B·2^%d: order %d, want 6", k, r.Order)
		}
		for _, w := range []float64{0.1, 0.7, 2.0} {
			z := cmplx.Exp(complex(0, w))
			got, err := r.Sys.EvalFr(z)
			if err != nil {
				t.Fatal(err)
			}
			want, err := tf.Eval(z)
			if err != nil {
				t.Fatal(err)
			}
			wv := want[0][0] * complex(math.Ldexp(1, k), 0)
			if d := cmplx.Abs(got[0][0] - wv); d > 1e-8*cmplx.Abs(wv) {
				t.Errorf("B·2^%d w=%g: got %v, want %v", k, w, got[0][0], wv)
			}
		}
	}
}

// w13 review: states 1-2 are never driven and states 3-4 never reach the
// output (exact zeros), so the minimal order is 1 with the stable pole
// 0.868; the hidden poles 1.885 and -2.106 must not survive. Scaling B or C
// by a large power of ten must not change that.
func TestReduceExactKalmanStructureScaled(t *testing.T) {
	a := []float64{0.8684323205674875, -0.6049616753433141, -0.817025610149589, 0, 0, 0, 0.5112361371882209, -1.665579513685, 0, 0, 0, 0.4426963353960824, 2.2287867502570378, 0, 0, 1.450755586734131, 0.5926180571361175, -1.003801815806515, -0.7392334214012152, 2.598303245115053, 0.019516845706728403, 0.7670269935617764, 2.019276442042413, 1.380736914734296, 0.5174218017098235}
	b := []float64{0.7453432469816224, -0.1349637189736732, 0, 0, 0, 0, 0.2391746712711118, 1.270569808451821, -1.6807270924517255, 1.013393545638635}
	c := []float64{0.10373234420450061, 1.1189760895011407, -0.6387064209174134, 0, 0, 0.010176619760267125, -0.40735317004013655, 0.15355429034927204, 0, 0}
	for _, sc := range [][2]float64{{1, 1}, {1e8, 1}, {1, 1e8}, {1e-8, 1e8}} {
		bs, cs := slices.Clone(b), slices.Clone(c)
		for i := range bs {
			bs[i] *= sc[0]
		}
		for i := range cs {
			cs[i] *= sc[1]
		}
		sys := reduceTestSys(t, 5, 2, 2, a, bs, cs, 0.1)
		r, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if r.Order != 1 || math.Abs(r.Sys.A.At(0, 0)-0.8684323205674875) > 1e-12 {
			t.Errorf("B·%g C·%g: order %d A %v, want 1 [0.8684]", sc[0], sc[1], r.Order, mat.Formatted(r.Sys.A))
		}
	}
}
