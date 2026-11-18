package controlsys

import (
	"math"
	"math/cmplx"
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
			t.Errorf("%s non-finite: %+v, want neither verified nor refuted", tc.name, jp)
		}
	}
	overflow := &TransferFunc{Num: [][][]float64{{{1e308}}}, Den: [][]float64{{1e-300, 1e-300}}}
	at0 := []complex128{0}
	full0 := newPencilBoundedResponse(sys, at0)
	if jp := judgeDescriptorCached(sys, tfRows{overflow}, at0, full0, tfWindows{m: 1}, nil)[0]; jp.verified || jp.refuted {
		t.Errorf("overflowing model row: %+v, want neither verified nor refuted", jp)
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
