package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"strings"
	"testing"
)

const pidTol = 1e-10

func pidPoles(t *testing.T, sys *System) []complex128 {
	t.Helper()
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(poles, func(i, j int) bool {
		if real(poles[i]) != real(poles[j]) {
			return real(poles[i]) < real(poles[j])
		}
		return imag(poles[i]) < imag(poles[j])
	})
	return poles
}

func pidZeros(t *testing.T, sys *System) []complex128 {
	t.Helper()
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(zeros, func(i, j int) bool {
		if real(zeros[i]) != real(zeros[j]) {
			return real(zeros[i]) < real(zeros[j])
		}
		return imag(zeros[i]) < imag(zeros[j])
	})
	return zeros
}

func pidDCGain(t *testing.T, sys *System) float64 {
	t.Helper()
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	return g.At(0, 0)
}

func assertComplexSlice(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d; got %v", name, len(got), len(want), got)
	}
	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > pidTol {
			t.Errorf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestPIDCopyIsIndependent(t *testing.T) {
	pid := mustPID(t, 1, 2, 3, 4, 0.1)

	cp := pid.Copy()
	pid.Kp = 99

	if cp.Kp != 1 || cp.Ki != 2 || cp.Kd != 3 || cp.Tf != 4 || cp.Dt != 0.1 {
		t.Fatalf("Copy = %+v, want original controller values", cp)
	}
}

func TestPID2CopyIsIndependent(t *testing.T) {
	pid := mustPID2(t, 1, 2, 3, 4, 5, 6, 0.1)

	cp := pid.Copy()
	pid.Kp = 99

	if cp.Kp != 1 || cp.Ki != 2 || cp.Kd != 3 || cp.Tf != 4 || cp.B != 5 || cp.C != 6 || cp.Dt != 0.1 {
		t.Fatalf("Copy = %+v, want original controller values", cp)
	}
}

func TestPID_PureP(t *testing.T) {
	c := mustPID(t, 2, 0, 0, 0, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 0 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (0,1,1)", n, m, p)
	}
	if d := sys.D.At(0, 0); math.Abs(d-2) > pidTol {
		t.Errorf("D = %v, want 2", d)
	}
}

func TestPID_PI(t *testing.T) {
	c := mustPID(t, 1, 2, 0, 0, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 1 {
		t.Fatalf("states = %d, want 1", n)
	}
	if v := sys.A.At(0, 0); math.Abs(v) > pidTol {
		t.Errorf("A = %v, want 0", v)
	}
	if v := sys.B.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("B = %v, want 1", v)
	}
	if v := sys.C.At(0, 0); math.Abs(v-2) > pidTol {
		t.Errorf("C = %v, want 2", v)
	}
	if v := sys.D.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("D = %v, want 1", v)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{0})
}

func TestPID_PDFiltered(t *testing.T) {
	c := mustPID(t, 1, 0, 3, 0.5, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 1 {
		t.Fatalf("states = %d, want 1", n)
	}
	if v := sys.A.At(0, 0); math.Abs(v-(-2)) > pidTol {
		t.Errorf("A = %v, want -2", v)
	}
	if v := sys.B.At(0, 0); math.Abs(v-2) > pidTol {
		t.Errorf("B = %v, want 2", v)
	}
	if v := sys.C.At(0, 0); math.Abs(v-(-6)) > pidTol {
		t.Errorf("C = %v, want -6", v)
	}
	if v := sys.D.At(0, 0); math.Abs(v-7) > pidTol {
		t.Errorf("D = %v, want 7", v)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{-2})
	g := pidDCGain(t, sys)
	if math.Abs(g-1) > pidTol {
		t.Errorf("DCGain = %v, want 1 (Kp)", g)
	}
}

func TestPID_PIDFiltered(t *testing.T) {
	c := mustPID(t, 1, 2, 3, 0.5, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 2 {
		t.Fatalf("states = %d, want 2", n)
	}
	if v := sys.A.At(0, 0); math.Abs(v) > pidTol {
		t.Errorf("A[0,0] = %v, want 0", v)
	}
	if v := sys.A.At(0, 1); math.Abs(v) > pidTol {
		t.Errorf("A[0,1] = %v, want 0", v)
	}
	if v := sys.A.At(1, 0); math.Abs(v) > pidTol {
		t.Errorf("A[1,0] = %v, want 0", v)
	}
	if v := sys.A.At(1, 1); math.Abs(v-(-2)) > pidTol {
		t.Errorf("A[1,1] = %v, want -2", v)
	}
	if v := sys.B.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("B[0] = %v, want 1", v)
	}
	if v := sys.B.At(1, 0); math.Abs(v-2) > pidTol {
		t.Errorf("B[1] = %v, want 2", v)
	}
	if v := sys.C.At(0, 0); math.Abs(v-2) > pidTol {
		t.Errorf("C[0] = %v, want 2", v)
	}
	if v := sys.C.At(0, 1); math.Abs(v-(-6)) > pidTol {
		t.Errorf("C[1] = %v, want -6", v)
	}
	if v := sys.D.At(0, 0); math.Abs(v-7) > pidTol {
		t.Errorf("D = %v, want 7", v)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{-2, 0})

	tfr, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	s := complex(0, 1.0)
	h := mustEval(t, tfr.TF, s)[0][0]
	want := complex(1, 0) + complex(2, 0)/s + complex(3, 0)*s/(complex(0.5, 0)*s+1)
	if cmplx.Abs(h-want) > 1e-8 {
		t.Errorf("TF at s=j = %v, want %v", h, want)
	}
}

func TestPID_PDNoFilter_Error(t *testing.T) {
	c := mustPID(t, 1, 0, 3, 0, 0)
	_, err := c.System()
	if err == nil {
		t.Fatal("expected error for PD without filter")
	}
}

func TestPID_UnfilteredDerivativeWithIntegral_Error(t *testing.T) {
	c := mustPID(t, 1, 2, 3, 0, 0)
	_, err := c.System()
	if err == nil {
		t.Fatal("expected error for unfiltered derivative term")
	}
}

func TestPID_DiscretePI(t *testing.T) {
	c := mustPID(t, 1, 2, 0, 0, 0.1)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 1 {
		t.Fatalf("states = %d, want 1", n)
	}
	if !sys.IsDiscrete() {
		t.Fatal("expected discrete system")
	}
	if v := sys.A.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("A = %v, want 1", v)
	}
	if v := sys.B.At(0, 0); math.Abs(v-0.1) > pidTol {
		t.Errorf("B = %v, want 0.1", v)
	}
	if v := sys.C.At(0, 0); math.Abs(v-2) > pidTol {
		t.Errorf("C = %v, want 2", v)
	}
	if v := sys.D.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("D = %v, want 1", v)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{1})

	zeros := pidZeros(t, sys)
	assertComplexSlice(t, "zeros", zeros, []complex128{0.8})
}

func TestPID_DiscretePIDFiltered(t *testing.T) {
	c := mustPID(t, 1, 2, 3, 0.5, 0.1)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 2 {
		t.Fatalf("states = %d, want 2", n)
	}
	if !sys.IsDiscrete() {
		t.Fatal("expected discrete system")
	}
	if v := sys.A.At(0, 0); math.Abs(v-1) > pidTol {
		t.Errorf("A[0,0] = %v, want 1", v)
	}
	if v := sys.A.At(1, 1); math.Abs(v-0.8) > pidTol {
		t.Errorf("A[1,1] = %v, want 0.8", v)
	}
	if v := sys.B.At(0, 0); math.Abs(v-0.1) > pidTol {
		t.Errorf("B[0] = %v, want 0.1", v)
	}
	if v := sys.B.At(1, 0); math.Abs(v-0.2) > pidTol {
		t.Errorf("B[1] = %v, want 0.2", v)
	}
	if v := sys.D.At(0, 0); math.Abs(v-7) > pidTol {
		t.Errorf("D = %v, want 7", v)
	}
}

func TestPID_PIDFilteredTransferFunction(t *testing.T) {
	c := mustPID(t, 2, 5, 0.5, 0.1, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	tfr, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.01, 0.1, 1, 10, 100}
	for _, w := range freqs {
		s := complex(0, w)
		got := mustEval(t, tfr.TF, s)[0][0]
		want := complex(2, 0) + complex(5, 0)/s + 0.5*s/(0.1*s+1)
		if cmplx.Abs(got-want) > 1e-6*cmplx.Abs(want) {
			t.Errorf("w=%g: got %v, want %v", w, got, want)
		}
	}
}

func TestPID_StandardForm(t *testing.T) {
	c := mustPID(t, 4, 0, 0, 0, 0)
	c.Form = PIDStandard
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 0 {
		t.Fatalf("states = %d, want 0", n)
	}
	if v := sys.D.At(0, 0); math.Abs(v-4) > pidTol {
		t.Errorf("D = %v, want 4", v)
	}
}

func TestPID_WithFilterAndTs(t *testing.T) {
	c := mustPID(t, 1, 0, 0, 0.5, 0.01)
	if c.Tf != 0.5 {
		t.Errorf("Tf = %v, want 0.5", c.Tf)
	}
	if c.Dt != 0.01 {
		t.Errorf("Dt = %v, want 0.01", c.Dt)
	}
}

func TestPID_KdWithIButNoFilter_Works(t *testing.T) {
	c := mustPID(t, 1, 2, 3, 0.5, 0)
	_, err := c.System()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPIDStd_PI(t *testing.T) {
	c, err := NewPIDStd(2, 5, 0, math.Inf(1), 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(c.Kp-2) > pidTol {
		t.Errorf("Kp = %v, want 2", c.Kp)
	}
	if math.Abs(c.Ki-0.4) > pidTol {
		t.Errorf("Ki = Kp/Ti = %v, want 0.4", c.Ki)
	}
	if math.Abs(c.Kd) > pidTol {
		t.Errorf("Kd = %v, want 0", c.Kd)
	}
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{0})
}

func TestPIDStd_PIDFiltered(t *testing.T) {
	c, err := NewPIDStd(1, 2, 0.5, (0.5)/(0.1), 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(c.Ki-0.5) > pidTol {
		t.Errorf("Ki = Kp/Ti = %v, want 0.5", c.Ki)
	}
	if math.Abs(c.Kd-0.5) > pidTol {
		t.Errorf("Kd = Kp*Td = %v, want 0.5", c.Kd)
	}
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	poles := pidPoles(t, sys)
	assertComplexSlice(t, "poles", poles, []complex128{-10, 0})
}

func TestPIDStd_ZeroTi_Error(t *testing.T) {
	_, err := NewPIDStd(1, 0, 0.5, math.Inf(1), 0)
	if err == nil {
		t.Fatal("expected error for Ti=0")
	}
}

func TestPIDStd_ConvertRoundTrip(t *testing.T) {
	orig := mustPID(t, 3, 1.5, 3, 0.2, 0)
	std, err := orig.Standard()
	if err != nil {
		t.Fatal(err)
	}
	if ti, ok := std.Ti(); !ok || math.Abs(ti-2) > pidTol {
		t.Errorf("Ti = %v, %v, want 2", ti, ok)
	}
	if td, ok := std.Td(); !ok || math.Abs(td-1) > pidTol {
		t.Errorf("Td = %v, %v, want 1", td, ok)
	}
	par := std.Parallel()
	if math.Abs(par.Kp-3) > pidTol || math.Abs(par.Ki-1.5) > pidTol || math.Abs(par.Kd-3) > pidTol {
		t.Errorf("round-trip failed: Kp=%v Ki=%v Kd=%v", par.Kp, par.Ki, par.Kd)
	}
}

func TestPID2_MatchesPID1DOF(t *testing.T) {
	c1 := mustPID(t, 1, 2, 3, 0.5, 0)
	sys1, err := c1.System()
	if err != nil {
		t.Fatal(err)
	}

	c2 := mustPID2(t, 1, 2, 3, 0.5, 1, 1, 0)
	sys2, err := c2.System()
	if err != nil {
		t.Fatal(err)
	}

	_, m2, _ := sys2.Dims()
	if m2 != 2 {
		t.Fatalf("PID2 inputs = %d, want 2", m2)
	}

	freqs := []float64{0.01, 0.1, 1, 10}
	for _, w := range freqs {
		s := complex(0, w)
		h1 := pidEvalSS(sys1, s)
		h2 := pidEvalSS(sys2, s)
		hR := h2[0][0]
		hY := h2[0][1]
		if cmplx.Abs(hR-h1[0][0]) > 1e-8 {
			t.Errorf("w=%g: H_r = %v, want PID1 = %v", w, hR, h1[0][0])
		}
		if cmplx.Abs(hY+h1[0][0]) > 1e-8 {
			t.Errorf("w=%g: H_y = %v, want -PID1 = %v", w, hY, -h1[0][0])
		}
	}
}

func TestPID2_SetpointWeight(t *testing.T) {
	c := mustPID2(t, 2, 1, 0, 0, 0.5, 0, 0)
	sys, err := c.System()
	if err != nil {
		t.Fatal(err)
	}
	_, m, _ := sys.Dims()
	if m != 2 {
		t.Fatalf("inputs = %d, want 2", m)
	}

	s := complex(0, 1.0)
	h := pidEvalSS(sys, s)
	wantR := complex(2*0.5, 0) + complex(1, 0)/s
	wantY := complex(-2, 0) + complex(-1, 0)/s
	if cmplx.Abs(h[0][0]-wantR) > 1e-8 {
		t.Errorf("H_r at s=j = %v, want %v", h[0][0], wantR)
	}
	if cmplx.Abs(h[0][1]-wantY) > 1e-8 {
		t.Errorf("H_y at s=j = %v, want %v", h[0][1], wantY)
	}
}

func pidEvalSS(sys *System, s complex128) [][]complex128 {
	h, _ := sys.EvalFr(s)
	return h
}

func mustPID(tb testing.TB, kp, ki, kd, tf, ts float64, opts ...PIDOption) *PID {
	tb.Helper()
	p, err := NewPID(kp, ki, kd, tf, ts, opts...)
	if err != nil {
		tb.Fatal(err)
	}
	return p
}

func mustPID2(tb testing.TB, kp, ki, kd, tf, b, c, ts float64, opts ...PIDOption) *PID2 {
	tb.Helper()
	p, err := NewPID2(kp, ki, kd, tf, b, c, ts, opts...)
	if err != nil {
		tb.Fatal(err)
	}
	return p
}

func TestPIDConstructorsMatchMATLABShapes(t *testing.T) {
	p2, err := NewPID2(1, 2, 0.5, 0.1, 0.7, 0.3, 0.05, WithPIDFormulas(Trapezoidal, BackwardEuler))
	if err != nil {
		t.Fatal(err)
	}
	if want := (PID2{Kp: 1, Ki: 2, Kd: 0.5, Tf: 0.1, B: 0.7, C: 0.3, Dt: 0.05, IFormula: Trapezoidal, DFormula: BackwardEuler}); *p2 != want {
		t.Fatalf("NewPID2 = %+v, want %+v", *p2, want)
	}
	if _, err := NewPID2(1, 0, 0, 0, math.NaN(), 1, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NaN b err = %v", err)
	}

	kp, ti, td, n := 2.0, 0.5, 0.25, 8.0
	std, err := NewPIDStd(kp, ti, td, n, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := std.System()
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0.3, 2, 40} {
		s := complex(0, w)
		want := complex(kp, 0) * (1 + 1/(complex(ti, 0)*s) + complex(td, 0)*s/(complex(td/n, 0)*s+1))
		resp, err := sys.FreqResponse([]float64{w})
		if err != nil {
			t.Fatal(err)
		}
		if got := resp.At(0, 0, 0); cmplx.Abs(got-want) > 1e-10*cmplx.Abs(want) {
			t.Fatalf("pidstd C(j%g) = %v, want %v", w, got, want)
		}
	}
	noI, err := NewPIDStd(2, math.Inf(1), 0, math.Inf(1), 0)
	if err != nil || noI.Ki != 0 || noI.Kd != 0 || noI.Tf != 0 {
		t.Fatalf("pidstd(2, Inf, 0, Inf) = %+v, %v", noI, err)
	}
	for _, a := range [][4]float64{{1, 0, 0, 1}, {1, -1, 0, 1}, {1, math.NaN(), 0, 1}, {1, 1, -1, 1}, {1, 1, 1, 0}, {1, 1, 1, -2}, {math.Inf(1), 1, 0, 1}} {
		if _, err := NewPIDStd(a[0], a[1], a[2], a[3], 0); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "NewPIDStd: ") {
			t.Fatalf("NewPIDStd%v err = %v, want ErrInvalidArgument", a, err)
		}
	}
	for _, call := range []func() error{
		func() error { _, err := NewPIDStd(1, 1, 0, 1, -1); return err },
		func() error { _, err := NewPIDStd(1, 1, 0, 1, 0, WithPIDFormulas(9, 0)); return err },
	} {
		if err := call(); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "NewPIDStd: ") || strings.Contains(err.Error(), "NewPID:") {
			t.Fatalf("NewPIDStd delegated err = %v, want one NewPIDStd prefix", err)
		}
	}
}

func TestPIDStandardTimesCommaOk(t *testing.T) {
	cases := []struct {
		kp, ki, kd float64
		ti, td     float64
		tiOK, tdOK bool
	}{
		{0, 1, 1, 0, 0, false, false},
		{0, 0, 1, math.Inf(1), 0, false, false},
		{0, 0, 0, math.Inf(1), 0, true, true},
		{2, 0, 0, math.Inf(1), 0, true, true},
		{2, 4, 1, 0.5, 0.5, true, true},
	}
	for _, c := range cases {
		p := mustPID(t, c.kp, c.ki, c.kd, 0.1, 0)
		ti, okI := p.Ti()
		td, okD := p.Td()
		if okI != c.tiOK || okD != c.tdOK || (okI && ti != c.ti) || (okD && td != c.td) {
			t.Errorf("%+v: Ti=%g,%v Td=%g,%v", c, ti, okI, td, okD)
		}
	}
	var nilPID *PID
	if nilPID.Parallel() != nil {
		t.Fatal("nil Parallel")
	}
	if _, err := mustPID(t, 0, 1, 0, 0, 0).Standard(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Standard err = %v", err)
	}
	if _, err := mustPID(t, 1, 0, 1, 0, 0).System(); !errors.Is(err, ErrImproperModel) || !strings.HasPrefix(err.Error(), "PID.System: ") {
		t.Fatalf("ideal derivative err = %v, want ErrImproperModel", err)
	}
	if _, err := (&PID2{Kp: 1, Kd: 1, B: 1, C: 1}).System(); !errors.Is(err, ErrImproperModel) || !strings.HasPrefix(err.Error(), "PID2.System: ") {
		t.Fatalf("PID2 ideal derivative err = %v", err)
	}
}
