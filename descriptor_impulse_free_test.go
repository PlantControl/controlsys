package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func idealPDDescriptor(t *testing.T, kp, kd float64) *System {
	t.Helper()
	c, err := NewDescriptor(eyeDense(2), mat.NewDense(2, 1, []float64{0, -1}), mat.NewDense(1, 2, []float64{kd, 0}), mat.NewDense(1, 1, []float64{kp}), mat.NewDense(2, 2, []float64{0, 1, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func transposeDescriptor(t *testing.T, sys *System) *System {
	t.Helper()
	var a, b, c, d, e mat.Dense
	a.CloneFrom(sys.A.T())
	b.CloneFrom(sys.C.T())
	c.CloneFrom(sys.B.T())
	d.CloneFrom(sys.D.T())
	e.CloneFrom(sys.E.T())
	out, err := NewDescriptor(&a, &b, &c, &d, &e, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// RMGVA3: Feedback(Series(ideal PD descriptor, 1/(s(s−1)))) is proper, but
// its impulsive mode is unobservable, so C∞ is rounding noise; the
// impulse-free test scaled by that noise rejected it. Oracle: the explicit
// companion form of (Kd·s + Kp)/(s² + (Kd−1)·s + Kp), MATLAB
// step(feedback(pid(Kp,0,Kd)*G,1)).
func TestDescriptorProperClosedLoopIdealPD(t *testing.T) {
	const kp, kd = 1.9019237886466818, 3.0980762113533165
	g, err := New(mat.NewDense(2, 2, []float64{0, 1, 0, 1}), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	loop, err := Series(idealPDDescriptor(t, kp, kd), g)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(loop, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.E == nil {
		t.Fatal("closed loop lost E")
	}
	oracle, err := New(mat.NewDense(2, 2, []float64{0, 1, -kp, 1 - kd}), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{kp, kd}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Step(oracle, 3)
	if err != nil {
		t.Fatal(err)
	}
	wantTF := func(s complex128) complex128 {
		return (complex(kd, 0)*s + complex(kp, 0)) / (s*s + complex(kd-1, 0)*s + complex(kp, 0))
	}
	for name, sys := range map[string]*System{"cl": cl, "transpose": transposeDescriptor(t, cl)} {
		if proper, err := sys.IsProper(); err != nil || !proper {
			t.Fatalf("%s: IsProper = %v, %v, want true", name, proper, err)
		}
		got, err := Step(sys, 3)
		if err != nil {
			t.Fatalf("%s: Step: %v", name, err)
		}
		if len(got.T) != len(want.T) {
			t.Fatalf("%s: %d samples, oracle %d", name, len(got.T), len(want.T))
		}
		for k := range got.T {
			if d := math.Abs(got.Y.At(0, k) - want.Y.At(0, k)); d > 1e-10 {
				t.Fatalf("%s: y(%g) = %v, oracle %v", name, got.T[k], got.Y.At(0, k), want.Y.At(0, k))
			}
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", name, err)
		}
		for _, w := range []float64{0, 0.3, 1, 7} {
			s := complex(0, w)
			got := polyEvalC(res.TF.Num[0][0], s) / polyEvalC(res.TF.Den[0], s)
			fr, err := sys.EvalFr(s)
			if err != nil {
				t.Fatal(err)
			}
			if d := cmplx.Abs(got - wantTF(s)); d > 1e-12*cmplx.Abs(wantTF(s)) {
				t.Errorf("%s: tf(%gj) = %v, oracle %v", name, w, got, wantTF(s))
			}
			if d := cmplx.Abs(fr[0][0] - wantTF(s)); d > 1e-12*cmplx.Abs(wantTF(s)) {
				t.Errorf("%s: EvalFr(%gj) = %v, oracle %v", name, w, fr[0][0], wantTF(s))
			}
		}
		if len(res.TF.Den[0]) != 3 {
			t.Errorf("%s: den %v, want degree 2", name, res.TF.Den[0])
		}
	}
}

// RMGVA3 guard: an observable derivative stays improper, also for a small
// Kd in a loop with a biproper plant (s+2)/(s+1).
func TestDescriptorObservableDerivativeStaysImproper(t *testing.T) {
	g, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, kd := range []float64{1, 1e-5} {
		pd := idealPDDescriptor(t, 2, kd)
		loop, err := Series(pd, g)
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"pd": pd, "loop": loop, "transpose": transposeDescriptor(t, loop)} {
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Errorf("Kd %g %s: Step err = %v, want ErrImproperModel", kd, name, err)
			}
			if _, err := sys.TransferFunction(nil); !errors.Is(err, ErrImproperModel) || !errors.Is(err, ErrDescriptorUnsupported) {
				t.Errorf("Kd %g %s: TransferFunction err = %v, want ErrImproperModel", kd, name, err)
			}
		}
	}
}

func smallDerivativeDescriptor(t *testing.T, c *mat.Dense) *System {
	t.Helper()
	p, _ := c.Dims()
	sys, err := NewDescriptor(mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 1, 0, 0, 0, 1}), mat.NewDense(3, 1, []float64{1, 0, -1}), c, mat.NewDense(p, 1, nil), mat.NewDense(3, 3, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Review of RMGVA3: y = c1/(s+1)·u + kd·s·u with kd ≤ 1e-8·c1 passed the
// 1e-8-relative impulse test, so Step and TransferFunction dropped kd·s
// (TF(1e6j) = −0.001i, EvalFr +0.999i) while IsProper reported false. Base
// and MATLAB step reject them. The test is now rounding level and IsProper
// uses it, so all three agree; a 1e-15 s-term is rounding noise.
func TestDescriptorSmallDerivativeStaysImproper(t *testing.T) {
	cases := []struct {
		name   string
		c      *mat.Dense
		proper bool
	}{
		{"c1=1e3 kd=1e-6", mat.NewDense(1, 3, []float64{1e3, 1e-6, 0}), false},
		{"c1=1e2 kd=1e-6", mat.NewDense(1, 3, []float64{1e2, 1e-6, 0}), false},
		{"c1=1 kd=1e-9", mat.NewDense(1, 3, []float64{1, 1e-9, 0}), false},
		{"c1=1e4 kd=1e-5", mat.NewDense(1, 3, []float64{1e4, 1e-5, 0}), false},
		{"c1=1 kd=1e-11", mat.NewDense(1, 3, []float64{1, 1e-11, 0}), false},
		{"two outputs", mat.NewDense(2, 3, []float64{1e4, 0, 0, 0, 1e-5, 0}), false},
		{"c1=1 kd=1e-15", mat.NewDense(1, 3, []float64{1, 1e-15, 0}), true},
	}
	for _, tc := range cases {
		sys := smallDerivativeDescriptor(t, tc.c)
		for name, s := range map[string]*System{"sys": sys, "transpose": transposeDescriptor(t, sys)} {
			label := tc.name + " " + name
			proper, err := s.IsProper()
			if err != nil {
				t.Fatalf("%s: IsProper: %v", label, err)
			}
			if proper != tc.proper {
				t.Errorf("%s: IsProper = %v, want %v", label, proper, tc.proper)
			}
			_, stepErr := Step(s, 1)
			res, tfErr := s.TransferFunction(nil)
			if (stepErr == nil) != proper || (tfErr == nil) != proper {
				t.Errorf("%s: IsProper %v but Step err %v, TransferFunction err %v", label, proper, stepErr, tfErr)
			}
			if !tc.proper {
				if !errors.Is(stepErr, ErrImproperModel) {
					t.Errorf("%s: Step err = %v, want ErrImproperModel", label, stepErr)
				}
				if !errors.Is(tfErr, ErrImproperModel) || !errors.Is(tfErr, ErrDescriptorUnsupported) {
					t.Errorf("%s: TransferFunction err = %v, want ErrImproperModel", label, tfErr)
				}
				continue
			}
			if tfErr != nil {
				continue
			}
			for _, w := range []float64{0, 1, 1e3} {
				z := complex(0, w)
				got := polyEvalC(res.TF.Num[0][0], z) / polyEvalC(res.TF.Den[0], z)
				want := 1 / (z + 1)
				if cmplx.Abs(got-want) > 1e-12 {
					t.Errorf("%s: tf(%gj) = %v, want %v", label, w, got, want)
				}
			}
		}
	}
}

// Review of 70e1da6, seeds of the RV sweep (PCG(seed, 2020)). Seed 821: two
// length-1 infinite chains (index 1, N = 0), yet T∞ kept a 1.5e-12
// off-diagonal just above the zeroing bound, N = S∞⁻¹T∞ was noise and
// IsProper/Step/TF called the model impulsive. Seed 1031: E with a stiff
// singular value 1.9e-13·σ₁ (MATLAB rank 4); the flat |β| cut put the stiff
// pole −2.6e12 in the infinite block. Base IsProper and branch Poles were
// right on both. Oracle: EvalFr on the pencil and the finite pole count;
// rounding of σs (ε·σ₁ against 1.9e-13·σ₁) fixes the stiff pole, whose
// residue is O(1), only to ~2e-3 (Poles alike), so seed 1031 matches to 5e-3.
func TestDescriptorIndexOneSplitMatchesPoles(t *testing.T) {
	cases := []struct {
		name       string
		a, e, b, c []float64
		poles      int
		tol        float64
	}{
		{"seed 821",
			[]float64{-2.372071276847989, -0.8274230772696413, -4.207730361609846, 4.4677135725951915, -1.393089161672643, -7.819405538753489, 0.9477346170996906, -8.34068457461149, -2.7286378394679858, 13.448601736969728, 0.10709308735312477, -2.8216349927260658, -2.390418783340044, -3.562240743108193, 3.014486519287966, -5.093131059101724, 6.804983540881573, 0.7036506344493302, -0.9073953694587538, 6.736427681067049, 1.3756396631805647, -0.2880200768757621, 1.0352050286947077, 5.311614441988355, -3.4807856684472385},
			[]float64{-0.1706328983806659, -1.3900967816084753, -1.7738935813331496, 1.87092775151033, -0.9283901542802125, 0.6016466108267258, -2.835465425929605, -1.8390769008170833, -5.081409930075303, -0.14015753074276754, 3.205764814320218, 0.5005130672894684, 4.447057970087615, -0.5292553345200064, -4.721504051366027, -2.6518372829484504, -1.5919251273681057, -4.683432043818324, -2.564643576304631, 4.572215940451496, -2.095355976522846, -0.690282546026149, -3.4349788884369103, 2.5266357776401107, 2.1518489797404885},
			[]float64{2.2115520578348393, -7.395417584688467, 1.9871093411752423, -9.842132440343107, 2.227196305213222},
			[]float64{3.498420558067308, 1.1767129762273012, 5.276380677925465, 3.3727095364658597, -5.311456511591591},
			3, 1e-10},
		{"seed 1031",
			[]float64{-3.0320097643340316, -1.2097889157028203, -0.8118756935585082, -2.296330091852601, 7.33519999588112, -4.3625668949910645, 0.6897017086958513, 1.569859765118842, -0.9744942674126708, 1.6793355547668494, -12.43277187997851, -7.6256207439631885, -8.503203901843852, -10.181750027561371, 23.14788666588983, 0.6944551026374722, 2.676756982301263, 3.514756538420735, 0.6514005160002918, -2.081172133754693, 7.499658010958484, 1.976020948039258, 2.2886989020152093, 7.542851365309246, -12.303335665970259},
			[]float64{0.4975122634745897, -0.30399177539407857, -0.39494386961094274, -0.16859905944501516, -4.572522497732277, 0.6411539221320162, -0.13907260421450904, -0.42290151983697827, -1.0279415786608082, 4.2779757414458635, 4.732895504818696, -0.0047377567155343974, -0.4535315349384543, 1.3909877527467511, -2.8737913289091317, 1.7820935466106065, -0.3524974727822892, -0.7705426329790079, -0.8885643882363503, 0.44859198620927077, -9.536228466560283, 0.28296742918968226, 1.6155128352184633, -0.46483132171345676, -3.0294046427957806},
			[]float64{1.3915218686778894, -0.13426334543570556, 5.688078872260182, -1.284425007108326, -1.9459010334648315},
			[]float64{0.41322346924031406, -0.2168449926860463, -0.5520401217846703, -1.480631026385153, -0.7760527052136756},
			4, 5e-3},
	}
	for _, tc := range cases {
		sys, err := NewDescriptor(mat.NewDense(5, 5, tc.a), mat.NewDense(5, 1, tc.b), mat.NewDense(1, 5, tc.c), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(5, 5, tc.e), 0)
		if err != nil {
			t.Fatal(err)
		}
		if proper, err := sys.IsProper(); err != nil || !proper {
			t.Fatalf("%s: IsProper = %v, %v, want true", tc.name, proper, err)
		}
		poles, err := sys.Poles()
		if err != nil || len(poles) != tc.poles {
			t.Fatalf("%s: Poles = %v, %v, want %d", tc.name, poles, err, tc.poles)
		}
		if _, err := Step(sys, 2); err != nil {
			t.Fatalf("%s: Step: %v", tc.name, err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", tc.name, err)
		}
		if got := len(res.TF.Den[0]) - 1; got != tc.poles {
			t.Errorf("%s: tf degree %d, want %d", tc.name, got, tc.poles)
		}
		for _, w := range []float64{0.37, 1.3, 5.1} {
			s := complex(0, w)
			got := polyEvalC(res.TF.Num[0][0], s) / polyEvalC(res.TF.Den[0], s)
			fr, err := sys.EvalFr(s)
			if err != nil {
				t.Fatal(err)
			}
			if d := cmplx.Abs(got - fr[0][0]); d > tc.tol*math.Max(1, cmplx.Abs(fr[0][0])) {
				t.Errorf("%s: tf(%gj) = %v, EvalFr %v", tc.name, w, got, fr[0][0])
			}
		}
	}
}
