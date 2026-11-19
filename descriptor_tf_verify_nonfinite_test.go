package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func nonFiniteJudgeDescriptor(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 0.5, -0.3, -2}), mat.NewDense(2, 1, []float64{1, 0.4}),
		mat.NewDense(1, 2, []float64{0.7, -1}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, []float64{1, 0.2, 0, 1.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// 5CFETF: an Inf response, comp or terms, or a row of the model that
// overflows, made Inf ≤ Inf pass and verified a wrong model there.
func TestDescriptorTFJudgeNonFinitePointUndetermined(t *testing.T) {
	sys := nonFiniteJudgeDescriptor(t)
	wrong := &TransferFunc{Num: [][][]float64{{{3, 1}}}, Den: [][]float64{{1, 2, 5}}}
	pts := []complex128{1i}
	full := newPencilBoundedResponse(sys, pts)
	if !full.ok[0] || !(full.eta[0] < 1) {
		t.Fatalf("point not determined: ok %v η %g", full.ok[0], full.eta[0])
	}
	inf := math.Inf(1)
	for _, tc := range []struct {
		name string
		set  func(r *pencilBoundedResp)
	}{
		{"g, comp and terms", func(r *pencilBoundedResp) {
			r.g, r.comp, r.terms = []complex128{complex(inf, 0)}, []float64{inf}, []float64{inf}
		}},
		{"terms", func(r *pencilBoundedResp) { r.terms = []float64{inf} }},
		{"comp", func(r *pencilBoundedResp) { r.comp = []float64{inf} }},
		{"g NaN", func(r *pencilBoundedResp) { r.g = []complex128{cmplx.NaN()} }},
	} {
		r := full
		r.g, r.comp, r.terms = []complex128{full.g[0]}, []float64{full.comp[0]}, []float64{full.terms[0]}
		tc.set(&r)
		jp := judgeDescriptorCached(sys, tfRows{wrong}, pts, r, tfWindows{m: 1}, nil)[0]
		if jp.verified || jp.refuted {
			t.Errorf("%s non-finite: %+v, want not verified, not refuted", tc.name, jp)
		}
	}
	overflow := &TransferFunc{Num: [][][]float64{{{1e308}}}, Den: [][]float64{{1e-300, 1e-300}}}
	at0 := []complex128{0}
	full0 := newPencilBoundedResponse(sys, at0)
	if jp := judgeDescriptorCached(sys, tfRows{overflow}, at0, full0, tfWindows{m: 1}, nil)[0]; jp.verified || jp.refuted {
		t.Errorf("overflowing model row: %+v, want not verified, not refuted", jp)
	}
}

// 5CFETF: on the XBVOCH model (E⁻¹B overflows, G ≈ 2⁹⁹⁶) every point
// was flagged ok with η < 1 and NaN g, comp and terms. Power-of-two
// scaling now evaluates them; any point left non-finite is not ok.
func TestPencilBoundedResponseOverflowNotOK(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := explicitOverflowDescriptor(t, dt)
		oracle := explicitOverflowOracle(t, dt)
		ws := []float64{1e-3, 0.1, 1, 3, 1e4, 1e12}
		if dt != 0 {
			ws = ws[:4]
		}
		var pts []complex128
		for _, w := range ws {
			pts = append(pts, axisPoint(dt != 0, w))
		}
		full := newPencilBoundedResponse(sys, pts)
		for k, s := range pts {
			fin := isFinite(real(full.g[k])) && isFinite(imag(full.g[k])) &&
				finiteAll(full.comp[k], full.data[k], full.terms[k], full.dTer[k], full.eta[k])
			if full.ok[k] && !fin {
				t.Errorf("dt %g s %v: ok with g %v comp %g terms %g η %g", dt, s, full.g[k], full.comp[k], full.terms[k], full.eta[k])
			}
			if !full.ok[k] || !(full.eta[k] < 1) {
				t.Errorf("dt %g s %v: not determined (ok %v, η %g)", dt, s, full.ok[k], full.eta[k])
				continue
			}
			w, err := oracle.EvalFr(s)
			if err != nil {
				t.Fatal(err)
			}
			if d := cmplx.Abs(full.g[k] - w[0][0]); !(d <= full.comp[k]+1e-12*cmplx.Abs(w[0][0])) {
				t.Errorf("dt %g s %v: g %v, want %v (comp %g)", dt, s, full.g[k], w[0][0], full.comp[k])
			}
		}
	}
}

// 5CFETF: with the overflowing points evaluated, the XBVOCH model's
// zero-pole-gain form certifies and matches the scaled explicit oracle.
func TestZPKModel_DescriptorWithOverflowingEinvB(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	got, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	oracle := explicitOverflowOracle(t, 0)
	for _, s := range []complex128{0, 1i, 3i, 0.5 + 2i} {
		g, err := got.ZPK.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		w, err := oracle.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(g[0][0]-w[0][0]) / cmplx.Abs(w[0][0]); !(d <= 1e-12) {
			t.Fatalf("s=%v: Z=%v, want %v", s, g[0][0], w[0][0])
		}
	}
}

// 5CFETF: with C raised to 2⁻¹⁰⁰ the XBVOCH response itself overflows
// (G ≈ 2¹⁰⁶²): no scaling evaluates it, so no point is ok.
func TestPencilBoundedResponseOverflowingResponseNotOK(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	c := math.Ldexp(1, -100)
	sys.C = mat.NewDense(1, 2, []float64{c, c})
	pts := []complex128{1e-3i, 1i, 1e4i}
	full := newPencilBoundedResponse(sys, pts)
	for k, s := range pts {
		if full.ok[k] {
			t.Errorf("s %v: ok with g %v comp %g terms %g", s, full.g[k], full.comp[k], full.terms[k])
		}
	}
}

// 5CFETF: E and A near 2⁻⁶⁷⁷, B near 2⁴¹⁹ and C near 2⁻⁸⁹⁶ (A = E·M,
// non-symmetric M). GEPP on sE − A gave NaN below ~1e60j, flagged ok
// with η < 1, so only the high band was judged and a ZPK with a zero at
// −1.47e60 instead of the stored response (≈ −4.7e61 at s = 0) verified.
func TestZPKModel_DescriptorOverflowingSolveNotVerifiedWrong(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(3, 3, []float64{-2.406778576576046e-204, -2.0386439418247506e-204, -3.8705757921820436e-204,
			-1.297283944103912e-204, -2.017219779670027e-204, -1.4962877978536035e-204,
			-1.5054053306422805e-204, -2.137409448301117e-204, -2.5352075714330445e-204}),
		mat.NewDense(3, 1, []float64{-2.3781269517608835e+126, -2.2646106411394616e+125, 3.885259238431645e+126}),
		mat.NewDense(1, 3, []float64{2.9715400540113267e-270, -1.8730043667115604e-271, 2.5650412465376597e-271}),
		mat.NewDense(1, 1, []float64{-0.8536619411158632}),
		mat.NewDense(3, 3, []float64{4.671877514131595e-204, -1.7561249072122202e-205, -1.2839898695507485e-206,
			2.2094128189347175e-205, 3.284567646985096e-204, 1.1056984210698517e-205,
			3.7533090962241964e-205, 2.1129124653890444e-205, 4.341030006087057e-204}), 0)
	if err != nil {
		t.Fatal(err)
	}
	z, err := sys.ZPKModel(nil)
	if err != nil {
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatal(err)
		}
		return
	}
	for _, s := range []complex128{0, 1e-3i, 0.1i, 1i, 10i} {
		v, err := z.ZPK.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		w := storedResponse320(sys, s)[0][0]
		if d := cmplx.Abs(v[0][0]-w) / cmplx.Abs(w); !(d <= 1e-6) {
			t.Fatalf("nil error but s=%v: Z=%v, stored response %v", s, v[0][0], w)
		}
	}
}

// 5CFETF: the reduction of this badly scaled pencil (C subnormal, B near
// 2³⁷²) returns a +Inf numerator coefficient; the certificate's 320-bit
// evaluation panicked on it (0·Inf in big.Float) instead of treating the
// row as not finite.
func TestTransferFunction_DescriptorNonFiniteCoefficientNoPanic(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{2.2860803651350193e-207, -1.0259374858937401e-207, -8.159771144236081e-208, -6.593255930627217e-207}),
		mat.NewDense(2, 1, []float64{-1.7964723570647378e+112, 1.7696249782607216e+112}),
		mat.NewDense(1, 2, []float64{math.Copysign(0, -1), 1.5e-323}),
		mat.NewDense(1, 1, []float64{-1.023279493509505}),
		mat.NewDense(2, 2, []float64{1.915707240418936e-207, -1.2457803970606594e-208, -3.15283643138896e-208, 2.0502829130113785e-209}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		return
	}
	for i := range res.TF.Num {
		if !finiteAll(res.TF.Den[i]...) {
			t.Fatalf("nil error with denominator %v", res.TF.Den[i])
		}
		for _, num := range res.TF.Num[i] {
			if !finiteAll(num...) {
				t.Fatalf("nil error with numerator %v", num)
			}
		}
	}
}

func TestDescriptorTFJudgeNonFiniteCoefficientUndetermined(t *testing.T) {
	sys := nonFiniteJudgeDescriptor(t)
	pts := []complex128{1i}
	full := newPencilBoundedResponse(sys, pts)
	for _, tf := range []*TransferFunc{
		{Num: [][][]float64{{{math.Inf(1)}}}, Den: [][]float64{{1}}},
		{Num: [][][]float64{{{1}}}, Den: [][]float64{{math.NaN(), 1}}},
	} {
		if jp := judgeDescriptorCached(sys, tfRows{tf}, pts, full, tfWindows{m: 1}, nil)[0]; jp.verified || jp.refuted {
			t.Errorf("%v: %+v, want not verified, not refuted", tf, jp)
		}
	}
}

func nonFiniteJudgeDescriptor22(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 0.5, -0.3, -2}), mat.NewDense(2, 2, []float64{1, 0.4, -0.2, 0.8}),
		mat.NewDense(2, 2, []float64{0.7, -1, 0.3, 0.9}), mat.NewDense(2, 2, nil), mat.NewDense(2, 2, []float64{1, 0.2, 0, 1.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// 5CFETF ruling (2b): a row that cannot be evaluated finitely keeps the
// point from verifying but does not discard the refutation of a finite
// wrong row, before or after it.
func TestDescriptorTFJudgeNonFiniteRowKeepsFiniteRefutation(t *testing.T) {
	sys := nonFiniteJudgeDescriptor22(t)
	pts := []complex128{1i}
	full := newPencilBoundedResponse(sys, pts)
	wrong := [][]float64{{1e3}, {-2e3}}
	bad := [][]float64{{math.Inf(1)}, {1}}
	for _, tc := range []struct {
		name string
		num  [][][]float64
		want int
	}{
		{"wrong row first", [][][]float64{wrong, bad}, 0},
		{"wrong row second", [][][]float64{bad, wrong}, 2},
	} {
		tf := &TransferFunc{Num: tc.num, Den: [][]float64{{1}, {1}}}
		jp := judgeDescriptorCached(sys, tfRows{tf}, pts, full, tfWindows{m: 2}, nil)[0]
		if jp.verified || !jp.refuted || len(jp.refutedEntries) != 2 || jp.refutedEntries[0] != tc.want || jp.refutedEntries[1] != tc.want+1 {
			t.Errorf("%s: %+v, want refuted entries [%d %d]", tc.name, jp, tc.want, tc.want+1)
		}
	}
}

// 5CFETF ruling (2b): a non-finite quantity of one entry does not discard
// the refutation of the other entry of the same row.
func TestDescriptorTFJudgeNonFiniteEntryKeepsFiniteRefutation(t *testing.T) {
	sys := nonFiniteJudgeDescriptor22(t)
	pts := []complex128{1i}
	full := newPencilBoundedResponse(sys, pts)
	g, err := sys.EvalFr(1i)
	if err != nil {
		t.Fatal(err)
	}
	for bad := range 2 {
		wrongEntry := 1 - bad
		num := [][][]float64{{{real(g[0][0])}, {real(g[0][1])}}, {{real(g[1][0])}, {real(g[1][1])}}}
		num[0][wrongEntry] = []float64{1e3}
		tf := &TransferFunc{Num: num, Den: [][]float64{{1}, {1}}}
		r := full
		r.comp = slices.Clone(full.comp)
		r.comp[bad] = math.Inf(1)
		jp := judgeDescriptorCached(sys, tfRows{tf}, pts, r, tfWindows{m: 2}, nil)[0]
		if jp.verified || !jp.refuted || !slices.Contains(jp.refutedEntries, wrongEntry) || slices.Contains(jp.refutedEntries, bad) {
			t.Errorf("non-finite comp on entry %d: %+v, want entry %d refuted", bad, jp, wrongEntry)
		}
	}
}
