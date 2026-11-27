package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// matchedDC320 returns the DC gain of the delay-free SISO sys at x0 (0 or 1)
// and the descriptor bar u·(‖C‖‖M‖·(‖A‖ + x0‖E‖)·‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + |D|),
// M = (x0·E − A)⁻¹, u = 4(n+5)ε, by 320-bit solves of the stored entries.
func matchedDC320(t *testing.T, sys *System, x0 float64) (g, bar float64) {
	t.Helper()
	n, _, _ := sys.Dims()
	if n == 0 {
		return sys.D.At(0, 0), 4 * 5 * 0x1p-52 * math.Abs(sys.D.At(0, 0))
	}
	e := mat.NewDense(n, n, nil)
	normE := math.Sqrt(float64(n))
	if sys.E != nil {
		e.Copy(sys.E)
		normE = mat.Norm(sys.E, 2)
	} else {
		for i := range n {
			e.Set(i, i, 1)
		}
	}
	k := newHbMat(n, n)
	a, eh := hbFrom(sys.A), hbFrom(e)
	for i := range k.d {
		v := new(big.Float).SetPrec(hbPrec).Mul(eh.d[i], new(big.Float).SetFloat64(x0))
		k.d[i].Sub(v, a.d[i])
	}
	rhs := newHbMat(n, n+1)
	for i := range n {
		rhs.at(i, i).SetInt64(1)
		rhs.at(i, n).SetFloat64(sys.B.At(i, 0))
	}
	x := hbSolve(k, rhs)
	gb := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.D.At(0, 0))
	normM := 0.0
	for i := range n {
		p := new(big.Float).SetPrec(hbPrec).Mul(new(big.Float).SetFloat64(sys.C.At(0, i)), x.at(i, n))
		gb.Add(gb, p)
		for j := range n {
			v, _ := x.at(i, j).Float64()
			normM += v * v
		}
	}
	normM = math.Sqrt(normM)
	g, _ = gb.Float64()
	cm := mat.Norm(sys.C, 2) * normM
	u := 4 * float64(n+5) * 0x1p-52
	bar = u * (cm*(mat.Norm(sys.A, 2)+x0*normE)*normM*mat.Norm(sys.B, 2) + cm*mat.Norm(sys.B, 2) + math.Abs(sys.D.At(0, 0)))
	return g, bar
}

func matchedFixture(t *testing.T, name string) *System {
	t.Helper()
	f, ok := matchedDCFixtures[name]
	if !ok {
		t.Fatalf("no fixture %q", name)
	}
	sys, err := f.system()
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// matchedDCInBars is |DC(d) − DC(sys)| in descriptor bars of sys, d the
// discrete result of matched C2D, both exact for their stored entries.
func matchedDCInBars(t *testing.T, sys, d *System) float64 {
	t.Helper()
	g0, bar := matchedDC320(t, sys, 0)
	free := *d
	free.InputDelay, free.OutputDelay, free.Delay = nil, nil, nil
	free.Dt = 1
	h, _ := matchedDC320(t, &free, 1)
	return math.Abs(h-g0) / bar
}

func requireMatchedUnverifiedModel(t *testing.T, d *System, err error) {
	t.Helper()
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("err %v, want ErrAccuracyUnverified", err)
	}
	if d == nil {
		t.Fatal("ErrAccuracyUnverified without the model")
	}
	if ferr := finiteSystemMatrices("result", d, ErrOverflow); ferr != nil {
		t.Fatal(ferr)
	}
	if d.D.At(0, 0) == 0 && mat.Norm(d.B, 2) == 0 {
		t.Fatal("unverified model has zero gain")
	}
}

// SWKJW4: the transfer-function roots of these stiff descriptors lose
// their small roots to 1e-7, and their DC product missed DCGain by up to
// 8e5 descriptor bars; matched now takes its gain from DCGain and returns
// nil only when the result's DC is inside the bar.
func TestC2DMatchedStiffDescriptorDCGain(t *testing.T) {
	for _, name := range []string{"stiff root product 1", "stiff root product 2", "stiff root product 3", "stiff root product 4"} {
		sys := matchedFixture(t, name)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if err != nil {
				t.Fatalf("%s dt=%g: %v", name, dt, err)
			}
			if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
				t.Errorf("%s dt=%g: DC off by %.3g descriptor bars", name, dt, bars)
			}
		}
	}
}

// SWKJW4: the pencil roots of g3/fz.7 pass the ZPK bar at DC, but their
// DC product misses DCGain by 35 descriptor bars; the gain now comes from
// DCGain.
func TestC2DMatchedPencilRootsDCGain(t *testing.T) {
	sys := matchedFixture(t, "pencil root product")
	for _, dt := range []float64{1e-4, 1e-2, 1} {
		d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
			t.Errorf("dt=%g: DC off by %.3g descriptor bars", dt, bars)
		}
	}
}

// SWKJW4: at dt = 1e-4 the cascade's DC cancels near z = 1, and the gain
// rounded into C lost it (21 bars); the gain now sits on B and D.
func TestC2DMatchedSmallSampleTimeDCGain(t *testing.T) {
	for _, name := range []string{"small sample time 1", "small sample time 2"} {
		sys := matchedFixture(t, name)
		d, err := sys.C2D(1e-4, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
			t.Errorf("%s: DC off by %.3g descriptor bars", name, bars)
		}
	}
}

// SWKJW4: the section pairing (strad2/388 at 1e-4) and the large poles of
// an unstable model at dt = 1 (stfr/4317) leave the cascade's DC outside
// the bar; the model comes with ErrAccuracyUnverified, never nil outside.
func TestC2DMatchedIllConditionedRealizationUnverified(t *testing.T) {
	for _, tc := range []struct {
		name string
		dt   float64
	}{{"section pairing", 1e-4}, {"unstable unit sample time", 1}} {
		sys := matchedFixture(t, tc.name)
		d, err := sys.C2D(tc.dt, C2DOptions{Method: C2DMethodMatched})
		if err == nil {
			if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
				t.Errorf("%s: nil with DC off by %.3g descriptor bars", tc.name, bars)
			}
			continue
		}
		requireMatchedUnverifiedModel(t, d, err)
	}
}

// SWKJW4: on the standard-form route the transfer function of these
// models can drop modes (order 1 of 6 on arm64), and its roots' DC product
// then misses DCGain by 1 and 85 relative; the root gate refutes them. A
// nil result keeps every finite pole of the model.
func TestC2DMatchedStandardFormSourceRefuted(t *testing.T) {
	for _, name := range []string{"standard form 1", "standard form 2"} {
		sys := matchedFixture(t, name)
		d, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			requireMatchedUnverifiedModel(t, d, err)
			continue
		}
		poles, perr := sys.Poles()
		if perr != nil {
			t.Fatal(perr)
		}
		if n, _, _ := d.Dims(); n != len(poles) {
			t.Errorf("%s: nil result of order %d, model has %d finite poles", name, n, len(poles))
		}
		if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
			t.Errorf("%s: nil with DC off by %.3g descriptor bars", name, bars)
		}
	}
}

// SWKJW4: DCGain unverified (g3/rlrep.427): the model with the
// root-product gain and ErrAccuracyUnverified.
func TestC2DMatchedDCGainUnavailable(t *testing.T) {
	sys := matchedFixture(t, "dc gain unverified")
	d, err := sys.C2D(1e-2, C2DOptions{Method: C2DMethodMatched})
	requireMatchedUnverifiedModel(t, d, err)
}

func TestMatchedRootGateBoundary(t *testing.T) {
	r := matchedReference{g0: 1, terms: 0.5}
	tol := math.Sqrt(0x1p-52)
	for _, tc := range []struct {
		v    complex128
		want bool
	}{
		{1, true},
		{complex(1+0.5*tol, 0), true},
		{complex(1-tol, 0), true},
		{complex(1+4*tol, 0), false},
		{complex(1-4*tol, 0), false},
		{complex(1, 4*tol), false},
	} {
		if got := r.rootsAgree(tc.v); got != tc.want {
			t.Errorf("rootsAgree(%v) = %v, want %v", tc.v, got, tc.want)
		}
	}
	big := matchedReference{g0: 1e-3, terms: 1}
	if !big.rootsAgree(complex(1e-3+0.5*tol, 0)) || big.rootsAgree(complex(1e-3+2*tol, 0)) {
		t.Error("root gate does not scale with the terms")
	}
}

func TestMatchedDCGateBoundary(t *testing.T) {
	r := matchedReference{g0: 1, terms: 1, bar: 1e-10}
	if !r.dcAgrees(mddFloat(1+0.5e-10), true) {
		t.Error("DC inside the bar refuted")
	}
	if r.dcAgrees(mddFloat(1+2e-10), true) {
		t.Error("DC outside the bar accepted")
	}
	if r.dcAgrees(matchedDC{hi: 1 + 0.5e-10, err: 1e-10}, true) {
		t.Error("rounding bound of the result ignored")
	}
	if r.dcAgrees(mddFloat(1), false) {
		t.Error("undecided DC accepted")
	}
	// B along the large singular direction of M: ‖X‖/‖B‖ is 1 and ‖M‖ is
	// 1000, so barLow decides only DC errors far inside the bar.
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 0, 0, -1e-3}), mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	ref := newMatchedReference(sys)
	if ref.err != nil {
		t.Fatal(ref.err)
	}
	bar := ref.fullBar()
	if !(ref.barLow < bar/1e3) {
		t.Fatalf("barLow %g not below the bar %g", ref.barLow, bar)
	}
	for _, tc := range []struct {
		off  float64
		want bool
	}{{0.5 * ref.barLow, true}, {0.5 * bar, true}, {2 * bar, false}} {
		if got := ref.dcAgrees(mddFloat(ref.g0+tc.off), true); got != tc.want {
			t.Errorf("DC off by %g (barLow %g, bar %g): agrees %v, want %v", tc.off, ref.barLow, bar, got, tc.want)
		}
	}
}

// R5a: with the gain on B and D, every other cascade entry is a copy of a
// section entry, so the stored model is gain·ΠSᵢ exactly.
func TestMatchedCascadeGainOnInputKeepsSections(t *testing.T) {
	zeros := []complex128{0.9, complex(0.2, 0.5), complex(0.2, -0.5), -1}
	poles := []complex128{0.95, 0.5, complex(0.7, 0.3), complex(0.7, -0.3), -0.2}
	const gain = 0.37
	sys := matchedCascadeFinite(zeros, poles, gain, 0.1)
	if sys == nil {
		t.Fatal("matchedCascadeFinite = nil on a finite model")
	}
	var entries []float64
	for _, s := range matchedSections(zeros, poles) {
		a, b, c, d := s.realizeAtOne()
		entries = slices.Concat(entries, a, b, c, []float64{d})
	}
	entries = append(entries, 0, 1)
	n, _, _ := sys.Dims()
	for i := range n {
		for j := range n {
			if v := sys.A.At(i, j); !slices.Contains(entries, v) {
				t.Errorf("A[%d,%d] = %v is not a section entry", i, j, v)
			}
		}
		if v := sys.C.At(0, i); !slices.Contains(entries, v) {
			t.Errorf("C[%d] = %v is not a section entry", i, v)
		}
		if v := sys.B.At(i, 0); v != 0 && v != gain {
			t.Errorf("B[%d] = %v, want 0 or the gain", i, v)
		}
	}
	if v := sys.D.At(0, 0); v != 0 && v != gain {
		t.Errorf("D = %v, want 0 or the gain", v)
	}
	_, dc, ok, err := matchedCascadeDC(zeros, poles, gain, 0.1)
	if err != nil || !ok {
		t.Fatalf("matchedCascadeDC: dc %v, ok %v, err %v", dc, ok, err)
	}
	g, _ := matchedDC320(t, sys, 1)
	if e := math.Abs(dc.hi + dc.lo - g); e > dc.err+0x1p-51*math.Abs(g) || dc.err > 1e-25*math.Abs(g) {
		t.Errorf("section-product DC %v ± %g, 320-bit DC of the cascade %v", dc.hi+dc.lo, dc.err, g)
	}
}

func TestMatchedPolishRoots(t *testing.T) {
	coef := []float64{1, -(0.2 + 1 + 1e15), 0.2 + 1e15 + 0.2e15, -0.2e15}
	start := []complex128{0.2 * (1 + 1e-9), 1 - 3e-10, 1e15 * (1 + 1e-12)}
	got := matchedPolish(coef, start)
	for i, x := range got {
		if imag(x) != 0 {
			t.Errorf("root %d left the real axis: %v", i, x)
		}
		p, _, mu := matchedHorner(coef, x)
		if p0, _, _ := matchedHorner(coef, start[i]); cmplx.Abs(p) > mu && cmplx.Abs(p) >= cmplx.Abs(p0) {
			t.Errorf("root %d: residual %g over the Horner bound %g and not lowered", i, cmplx.Abs(p), mu)
		}
	}
	if d := math.Abs(real(got[0])-0.2) / 0.2; d > 1e-14 {
		t.Errorf("small root %v off by %.3g", got[0], d)
	}
	double := []float64{1, -2, 1}
	cluster := []complex128{1 - 1e-8, 1 + 1e-8}
	if moved := matchedPolish(double, cluster); moved[0] != cluster[0] && moved[1] != cluster[1] && moved[0] == moved[1] {
		t.Errorf("cluster collapsed: %v", moved)
	}
	pair := []complex128{complex(-1, 2+1e-9), complex(-1, -2-1e-9)}
	pp := matchedPolish([]float64{1, 2, 5}, pair)
	if pp[1] != cmplx.Conj(pp[0]) {
		t.Errorf("pair %v not conjugate", pp)
	}
	if d := cmplx.Abs(pp[0] - complex(-1, 2)); d > 1e-14 {
		t.Errorf("pair root %v off by %.3g", pp[0], d)
	}
}

// SWKJW4: D2C matched took its gain from the discrete roots' DC product,
// 265, 333 and 535 descriptor bars (at z = 1) off on these ZOH models; it now
// matches DCGain of the discrete model. The repeated-root model's roots
// can fail the root gate (amd64), and then come with ErrAccuracyUnverified.
func TestD2CMatchedDCGain(t *testing.T) {
	for _, name := range []string{"zoh of an explicit model", "zoh of a repeated-root model", "zoh of a stiff descriptor"} {
		sys := matchedFixture(t, name)
		c, err := sys.D2C(D2COptions{Method: C2DMethodMatched})
		if err != nil && name == "zoh of a repeated-root model" {
			requireMatchedUnverifiedModel(t, c, err)
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		g1, bar := matchedDC320(t, sys, 1)
		free := *c
		free.InputDelay, free.OutputDelay, free.Delay = nil, nil, nil
		h, _ := matchedDC320(t, &free, 0)
		if bars := math.Abs(h-g1) / bar; !(bars <= 1) {
			t.Errorf("%s: DC off by %.3g descriptor bars", name, bars)
		}
	}
}

// D2D resamples by ZOH or Tustin only, as MATLAB d2d does; matched C2D's
// unverified model has no D2D route.
func TestD2DRejectsMatched(t *testing.T) {
	sys := conversionSISO(t, []float64{0.1}, []float64{1, -0.9}, 0.1)
	if _, err := sys.D2D(0.2, D2DOptions{Method: C2DMethodMatched}); !errors.Is(err, ErrInvalidConversionOptions) {
		t.Fatalf("err %v, want ErrInvalidConversionOptions", err)
	}
}

// An exactly zero transfer function has no root set to refute: matched
// returns the zero model with a nil error (SWKJW4 ruling, 2026-11-25).
func TestC2DMatchedExactZeroModel(t *testing.T) {
	zeroGain, err := NewGain(mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	noOutput, err := New(mat.NewDense(2, 2, []float64{-1, 0.5, 0.2, -2}), mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{0, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	unreachable, err := New(mat.NewDense(2, 2, []float64{-1, 0, 0.5, -2}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"static zero": zeroGain, "C = 0": noOutput, "CAᵏB = 0": unreachable} {
		d, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if g, _ := matchedDC320(t, &System{A: d.A, B: d.B, C: d.C, D: d.D, Dt: 1}, 1); g != 0 || !zeroDense(d.D) {
			t.Errorf("%s: discrete model not zero: DC %g, D %v", name, g, d.D.At(0, 0))
		}
	}
	dz, err := NewGain(mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	c, err := dz.D2C(D2COptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatalf("D2C: %v", err)
	}
	if !zeroDense(c.D) {
		t.Errorf("D2C: D = %v, want 0", c.D.At(0, 0))
	}
}

// The float LU of [3 1; 1 fl(1/3)] meets an exact zero pivot, but its exact
// determinant 3·fl(1/3) − 1 is not zero: the reference keeps a bar from the
// singular values. An exactly singular DC pencil has no bar (AQU6M7).
func TestMatchedReferenceSingularPencil(t *testing.T) {
	third := 1.0 / 3
	ill, err := New(mat.NewDense(2, 2, []float64{-3, -1, -1, -third}), mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0.5}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	u := 4 * 7 * 0x1p-52
	if r := matchedSingularLU(ill, 1, 0, u, math.Sqrt2, 0); r.err != nil || !(r.bar > 0) || math.IsInf(r.bar, 0) || !(r.terms > 0) {
		t.Errorf("ill-conditioned pencil: err %v, bar %g", r.err, r.bar)
	}
	sing, err := New(mat.NewDense(2, 2, []float64{-2, -1, -4, -2}), mat.NewDense(2, 1, []float64{1, -2}),
		mat.NewDense(1, 2, []float64{2, -1}), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if r := matchedSingularLU(sing, 1, 0, u, math.Sqrt2, 1); r.err == nil {
		t.Errorf("singular pencil: reference without error, bar %g", r.bar)
	}
}
