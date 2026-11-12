package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A slow pole −8.83e-8 of the float model, 300× outside its first-order
// rounding disk 2.96e-10 (120× the γ_n-scaled bound 7.4e-10), beside ‖A‖ = 5.5e5 and cond₁ 8.8e13: the
// double solve at s = 0 fails and the pole limit took the slow pole for an
// integrator, +Inf, at ω = 0 only; ω = 5e-324 read −284.9069183 (fz census
// row 12571, NSYMIV). The response at s = 0 is the certified −CA⁻¹B + D.
func TestEvalFrOriginResolvedSlowPoleFinite(t *testing.T) {
	a := []float64{-0x1.053651b347954p+17, -0x1.3d5f27abe429dp+18, 0x1.301aade8911cap+18, 0x1.611d3ee69673ep+16, -0x1.b403be4598927p+17, 0x1.574b59aa941bdp+14, 0x1.9f52a5baef0fdp+15, -0x1.8f1bbb0ecadedp+15, -0x1.cee0b7e01adb9p+13, 0x1.1dc60d9d4ba08p+15, -0x1.d57eebd26163ap+14, -0x1.1e078e6465b8cp+16, 0x1.118c5e8ccffb4p+16, 0x1.3de2bcf21c574p+14, -0x1.8883f830d225fp+15, -0x1.5386b62f56bf9p-10, -0x1.9db27461eef66p-09, 0x1.c51c6510e77d2p-09, 0x1.65d85e90fd3f6p-10, -0x1.2e1a90b02e5b4p-09, 0x1.987ed3c9dc4eap-12, 0x1.f1bb9eb196717p-11, -0x1.1093954fd310fp-10, -0x1.8fdae67b14b47p-12, 0x1.779f36abbb3f3p-11}
	b := []float64{-0x1.c70385196a53cp-05, -0x1.f42ffb31e6e4ep-01, -0x1.de524cca020a6p-02, 0x1.743cab5f8f06dp-01, 0x1.b64e813f7e6a6p-01}
	c := []float64{0x1.19cad07eba302p-16, -0x1.7c8ec13ae1efbp-16, -0x1.1dda5a2ab48ecp-19, 0x1.93a93449e44d3p-16, -0x1.c288c59f3f4ebp-17}
	sys, err := New(mat.NewDense(5, 5, a), mat.NewDense(5, 1, b), mat.NewDense(1, 5, c), mat.NewDense(1, 1, []float64{-0x1.55a6e837610d9p-05}), 0)
	if err != nil {
		t.Fatal(err)
	}
	want := bigExplicitDCGain(t, sys)
	if math.Abs(want[0][0]+284.9069183) > 1e-6 {
		t.Fatalf("oracle G(0) = %v", want[0][0])
	}
	assertDCResponse(t, "slow pole", sys, want, 1e-12)
	bode, err := sys.Bode([]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if g, w := bode.MagDBAt(0, 0, 0), 20*math.Log10(math.Abs(want[0][0])); !(math.Abs(g-w) <= 1e-9) {
		t.Errorf("Bode(0) magnitude = %v dB, want %v", g, w)
	}
}

// A pole inside its own rounding disk keeps the pole limit at s = 0: the
// stored model's exact root nearest 0 is 1.652e-10, inside its first-order
// disk c = 1.28e-9 (det(sI − A) changes sign on [−c, c]), so rounding of A
// can carry it to 0 and EvalFr(0) is +Inf as for a rounded integrator
// (W45MGH), although the stored model's G(0) = 2.384283283e6 is finite and
// DCGain returns it (fz census row 13418, NSYMIV ruling A; AVKYUV).
func TestEvalFrOriginPoleInsideRoundingDiskStaysPole(t *testing.T) {
	a := []float64{0x1.453f7c319b18fp+16, 0x1.18e9ef834be13p+16, 0x1.f9a0e714742a7p+13, 0x1.5b647040326cbp+17, 0x1.4c87569af3c3fp+19, -0x1.419d42820d1e9p+18, 0x1.886d9725e3ea5p+11, 0x1.52efc979737eep+11, 0x1.31087c5a8e5ecp+09, 0x1.532451117226cp+12, 0x1.e65bc804b65e8p+14, -0x1.f28d06a664effp+12, 0x1.e5c3081b509c6p+15, 0x1.a38c5742bbfb3p+15, 0x1.7994c8c648706p+13, 0x1.0516e8258db84p+17, 0x1.ed1391ce4f7dap+18, -0x1.e622b1c515df2p+17, 0x1.4909dbca0d707p+14, 0x1.1c3009dd86763p+14, 0x1.ff85707064f21p+11, 0x1.546e64667077cp+15, 0x1.582e3c957d639p+17, -0x1.36cf2ada3005p+16, 0x1.2248db526c849p+14, 0x1.f56eb6cae0994p+13, 0x1.c3464f8a3f88ep+11, 0x1.2fbbf5c79b059p+15, 0x1.2ce001865be0ep+17, -0x1.1779bf56fdaf4p+16, 0x1.2cf8eb9f6ec9fp+15, 0x1.03f2789429066p+15, 0x1.d3e3c74b1f4b2p+12, 0x1.3c14d9c9052dp+16, 0x1.377bfb5bcdfa5p+18, -0x1.22a980a00a4fp+17}
	b := []float64{0x1.de66f5b235fc9p-01, 0x1.2fbd634d66a85p-01, -0x1.3be6f07be269ep-01, -0x1.d8bf26a8cc686p-01, 0x1.f7deb26ec98a4p-04, 0x1.fa715db8cd433p-01}
	c := []float64{0x1.f1a2ae2e09c89p-18, 0x1.d281dc9064a4bp-17, -0x1.1339ce1eedbfep-16, -0x1.ba6312c503d4dp-19, 0x1.36fb071be5f0bp-16, -0x1.238ce3f1f7ce6p-17}
	sys, err := New(mat.NewDense(6, 6, a), mat.NewDense(6, 1, b), mat.NewDense(1, 6, c), mat.NewDense(1, 1, []float64{0x1.9909f1fcb57b5p+01}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if exactCharPolySign(sys.A, -1.28e-9)*exactCharPolySign(sys.A, 1.28e-9) >= 0 {
		t.Fatal("no exact root in [−c, c]")
	}
	ev, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	fr, err := sys.FreqResponse([]float64{0})
	if err != nil {
		t.Fatal(err)
	}
	if !cmplx.IsInf(ev[0][0]) || !cmplx.IsInf(fr.At(0, 0, 0)) {
		t.Errorf("EvalFr(0) %v, FreqResponse(0) %v, want Inf", ev[0][0], fr.At(0, 0, 0))
	}
}

// Integrators keep the pole limit at DC: an exact one of a non-symmetric
// triangular A beside a fast pole, and the same rotated by an orthogonal
// similarity, which rounds it off 0; continuous at s = 0 and discrete at
// z = 1 (NSYMIV control).
func TestEvalFrOriginIntegratorStaysPole(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 4093))
	Q := randomOrthogonal(rng, 4)
	rotate := func(A *mat.Dense) *mat.Dense {
		var tmp, R mat.Dense
		tmp.Mul(Q, A)
		R.Mul(&tmp, Q.T())
		return &R
	}
	cont := mat.NewDense(4, 4, []float64{0, 0.7, -1.3, 0.2, 0, -0.5, 2.1, -0.4, 0, 0, -3, 1.1, 0, 0, 0, -2e6})
	disc := mat.NewDense(4, 4, []float64{1, 0.3, -0.6, 0.2, 0, 0.5, 0.9, -0.4, 0, 0, -0.3, 0.8, 0, 0, 0, 0.1})
	b := mat.NewDense(4, 1, []float64{0.9, -1.1, 0.4, 0.6})
	c := mat.NewDense(1, 4, []float64{1.2, 0.3, -0.7, 0.5})
	d := mat.NewDense(1, 1, []float64{0.25})
	for _, tc := range []struct {
		name string
		A    *mat.Dense
		dt   float64
	}{
		{"continuous exact", cont, 0},
		{"continuous rotated", rotate(cont), 0},
		{"discrete exact", disc, 0.1},
		{"discrete rotated", rotate(disc), 0.1},
	} {
		B, C := b, c
		if tc.A != cont && tc.A != disc {
			var qb, cq mat.Dense
			qb.Mul(Q, b)
			cq.Mul(c, Q.T())
			B, C = &qb, &cq
		}
		sys, err := New(tc.A, B, C, d, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		at := complex(0, 0)
		if tc.dt > 0 {
			at = 1
		}
		ev, err := sys.EvalFr(at)
		if err != nil {
			t.Fatalf("%s: EvalFr: %v", tc.name, err)
		}
		fr, err := sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatalf("%s: FreqResponse: %v", tc.name, err)
		}
		if !cmplx.IsInf(ev[0][0]) || !cmplx.IsInf(fr.At(0, 0, 0)) {
			t.Errorf("%s: EvalFr %v, FreqResponse(0) %v, want Inf", tc.name, ev[0][0], fr.At(0, 0, 0))
		}
	}
}

// The floating-point Rouché bound decides the root-free disk of the slow
// pole model (fz 12571, disk 1.48e-9 against a pole −8.83e-8) and the
// root inside that of a pole rounded off 0 by a dense similarity, as the
// exact test does; at the Rouché boundary of s² + s + 0.75 on |s| = 0.5
// (q = pR + R², which the strict test rejects) its error bounds overlap
// and it defers to the exact test (NSYMIV).
func TestOriginDiskRootBoundDecidesOrDefers(t *testing.T) {
	slow := mat.NewDense(5, 5, []float64{-0x1.053651b347954p+17, -0x1.3d5f27abe429dp+18, 0x1.301aade8911cap+18, 0x1.611d3ee69673ep+16, -0x1.b403be4598927p+17, 0x1.574b59aa941bdp+14, 0x1.9f52a5baef0fdp+15, -0x1.8f1bbb0ecadedp+15, -0x1.cee0b7e01adb9p+13, 0x1.1dc60d9d4ba08p+15, -0x1.d57eebd26163ap+14, -0x1.1e078e6465b8cp+16, 0x1.118c5e8ccffb4p+16, 0x1.3de2bcf21c574p+14, -0x1.8883f830d225fp+15, -0x1.5386b62f56bf9p-10, -0x1.9db27461eef66p-09, 0x1.c51c6510e77d2p-09, 0x1.65d85e90fd3f6p-10, -0x1.2e1a90b02e5b4p-09, 0x1.987ed3c9dc4eap-12, 0x1.f1bb9eb196717p-11, -0x1.1093954fd310fp-10, -0x1.8fdae67b14b47p-12, 0x1.779f36abbb3f3p-11})
	rng := rand.New(rand.NewPCG(5, 2026))
	T := mat.NewDense(3, 3, nil)
	for i := range 9 {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti, tmp, rounded mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	tmp.Mul(T, mat.NewDense(3, 3, []float64{0, 0.4, 0, 0, -1, 0.7, 0, 0, -3}))
	rounded.Mul(&tmp, &Ti)
	pe, ok := newPoleEigen(&rounded)
	if !ok {
		t.Fatal("eigen")
	}
	move := 0.0
	for k := range pe.poles {
		move = max(move, pe.movement(&rounded, k))
	}
	for _, tc := range []struct {
		name          string
		A             *mat.Dense
		r             float64
		free, decided bool
	}{
		{"slow pole", slow, 1.48e-9, true, true},
		{"rounded integrator", &rounded, move, false, true},
		{"Rouché boundary", mat.NewDense(2, 2, []float64{0, 1, -0.75, -1}), 0.5, false, false},
	} {
		free, decided := originDiskRootBound(tc.A, tc.r)
		if free != tc.free || decided != tc.decided {
			t.Errorf("%s: float bound free %v decided %v, want %v %v", tc.name, free, decided, tc.free, tc.decided)
		}
		if exact := originDiskRootFree(tc.A, tc.r); exact != tc.free {
			t.Errorf("%s: exact test free %v, want %v", tc.name, exact, tc.free)
		}
	}
}
