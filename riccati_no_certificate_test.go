package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// gateCounter counts dareGate runs that reach the certificate and keeps the
// X and K of the last result it was handed.
type gateCounter struct {
	calls int
	x, k  *mat.Dense
}

func countGate(t *testing.T) *gateCounter {
	t.Helper()
	g := &gateCounter{}
	dareGateHook = func(r *RiccatiResult) {
		g.calls++
		g.x, g.k = mat.DenseCopyOf(r.X), mat.DenseCopyOf(r.K)
	}
	t.Cleanup(func() { dareGateHook = nil })
	return g
}

func (g *gateCounter) reset() { g.calls, g.x, g.k = 0, nil, nil }

func dareFloorRow(t *testing.T, name string) dareFloorCase {
	t.Helper()
	for _, c := range append(dareNoiseFloorCases, dareEnclosureAtXCase) {
		if c.name == name {
			return c
		}
	}
	t.Fatalf("no row %s", name)
	return dareFloorCase{}
}

// TestDare_NoCertificateAccuracy (EPYIVF): on rows whose X the gate
// certifies (rv39 1404, at X) or replaces (zrwreg 20196, by the Hewer or
// extended X of tier 2), opts.NoCertificate runs no gate work and returns
// the X and K the gate was handed, with ErrAccuracyUnverified.
func TestDare_NoCertificateAccuracy(t *testing.T) {
	g := countGate(t)
	for _, tc := range []struct {
		name     string
		replaced bool
	}{{"rv39:1404", false}, {"zrwreg:20196", true}} {
		A, B, Q, R, S, E := dareFloorRow(t, tc.name).dense()
		g.reset()
		def, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E})
		if err != nil || g.calls == 0 {
			t.Fatalf("%s: precondition: default Dare err %v, %d gate runs; want nil and a gate run", tc.name, err, g.calls)
		}
		preX, preK := g.x, g.k
		if replaced := !mat.Equal(def.X, preX); replaced != tc.replaced {
			t.Fatalf("%s: precondition: gate replaced X %v, want %v", tc.name, replaced, tc.replaced)
		}
		g.reset()
		res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoCertificate: true})
		if !errors.Is(err, ErrAccuracyUnverified) || res == nil || res.X == nil {
			t.Fatalf("%s: NoCertificate: res %v err %v, want a result with ErrAccuracyUnverified", tc.name, res != nil, err)
		}
		if g.calls != 0 {
			t.Errorf("%s: NoCertificate: %d gate runs, want 0", tc.name, g.calls)
		}
		if !mat.Equal(res.X, preX) || !mat.Equal(res.K, preK) {
			t.Errorf("%s: NoCertificate: X or K differs from the pre-gate candidate", tc.name)
		}
	}
}

// TestDare_NoCertificateExemptRow (EPYIVF): a descriptor row whose X the
// extended solve confirmed never reaches the gate, so opts.NoCertificate
// leaves its result, and its nil error, unchanged.
func TestDare_NoCertificateExemptRow(t *testing.T) {
	g := countGate(t)
	A := mat.NewDense(2, 2, []float64{-0.13724618874059225, 1.217088645560901, 0.6539759935733573, 0.625550292051194})
	B := mat.NewDense(2, 1, []float64{1.0056269676565766, -0.43721297890158445})
	E := mat.NewDense(2, 2, []float64{0.8136281888350105, -0.45758472346960766, 0, 3.8390252196819285e-16})
	Q, R := eyeDense(2), eyeDense(1)
	def, err := Dare(A, B, Q, R, &RiccatiOpts{E: E})
	if err != nil || g.calls != 0 {
		t.Fatalf("precondition: default Dare err %v, %d gate runs; want nil and none", err, g.calls)
	}
	res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoCertificate: true})
	if err != nil {
		t.Fatalf("NoCertificate: %v, want nil", err)
	}
	if g.calls != 0 || !mat.Equal(res.X, def.X) || !mat.Equal(res.K, def.K) {
		t.Errorf("NoCertificate: %d gate runs, X equal %v, K equal %v; want 0, true, true", g.calls, mat.Equal(res.X, def.X), mat.Equal(res.K, def.K))
	}
}

// TestDare_NoCertificateDefaultUnchanged (EPYIVF): NoCertificate false is
// nil opts bit for bit, with the same gate runs.
func TestDare_NoCertificateDefaultUnchanged(t *testing.T) {
	g := countGate(t)
	for _, c := range append(dareNoiseFloorCases, dareEnclosureAtXCase) {
		if c.s != nil || c.e != nil {
			continue
		}
		A, B, Q, R, _, _ := c.dense()
		g.reset()
		want, werr := Dare(A, B, Q, R, nil)
		wantCalls := g.calls
		g.reset()
		got, gerr := Dare(A, B, Q, R, &RiccatiOpts{NoCertificate: false})
		if (werr == nil) != (gerr == nil) || g.calls != wantCalls {
			t.Errorf("%s: err %v calls %d, want err %v calls %d", c.name, gerr, g.calls, werr, wantCalls)
			continue
		}
		if want != nil && (!mat.Equal(got.X, want.X) || !mat.Equal(got.K, want.K)) {
			t.Errorf("%s: X or K differs from nil opts", c.name)
		}
	}
}

// TestDlqr_NoCertificateDescriptor (EPYIVF): the descriptor row whose
// default Dlqr takes the gate's tier 2 (rv49 3641) runs no gate work under
// opts.NoCertificate and is not promoted back to a nil error by the
// explicit EᵀXE.
func TestDlqr_NoCertificateDescriptor(t *testing.T) {
	g := countGate(t)
	A, B, Q, R, S, E := dlqrCorrectedEnclosureRow()
	if _, err := Dlqr(A, B, Q, R, &RiccatiOpts{S: S, E: E}); g.calls == 0 {
		t.Fatalf("precondition: default Dlqr (err %v) ran no gate", err)
	}
	g.reset()
	res, err := Dlqr(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoCertificate: true})
	if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
		t.Fatalf("NoCertificate: err %v, want a result with ErrAccuracyUnverified", err)
	}
	if g.calls != 0 {
		t.Errorf("NoCertificate: %d gate runs, want 0", g.calls)
	}
}

// TestNoCertificate_Inheritance (EPYIVF): Lqi of a discrete descriptor
// plant, Lqrd, discrete Kalman and Kalmd pass opts.NoCertificate to their
// Dare, which then runs no gate work.
func TestNoCertificate_Inheritance(t *testing.T) {
	g := countGate(t)
	A := mat.NewDense(3, 3, []float64{0.5, 0.3, -0.1, -0.2, 0.4, 0.25, 0.1, -0.35, 0.6})
	Ac := mat.NewDense(3, 3, []float64{-1, 0.7, 0.2, -0.4, -0.5, 0.9, 0.3, -0.6, -1.2})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.3, 0.8, 0.5, -0.6})
	C := mat.NewDense(2, 3, []float64{1, -0.4, 0.3, 0.2, 0.9, -0.5})
	D := mat.NewDense(2, 2, nil)
	E := mat.NewDense(3, 3, []float64{1.2, 0.1, -0.2, 0, 0.9, 0.3, 0.1, 0, 1.1})
	desc, err := NewDescriptor(A, B, C, D, E, 1)
	if err != nil {
		t.Fatal(err)
	}
	disc, err := New(A, B, C, D, 1)
	if err != nil {
		t.Fatal(err)
	}
	cont, err := New(Ac, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func(o *RiccatiOpts) error
	}{
		{"Lqi descriptor", func(o *RiccatiOpts) error { _, err := Lqi(desc, eyeDense(5), eyeDense(2), o); return err }},
		{"Lqrd", func(o *RiccatiOpts) error { _, err := Lqrd(Ac, B, eyeDense(3), eyeDense(2), 0.1, o); return err }},
		{"Kalman discrete", func(o *RiccatiOpts) error { _, err := Kalman(disc, eyeDense(2), eyeDense(2), nil, o); return err }},
		{"Kalmd", func(o *RiccatiOpts) error { _, err := Kalmd(cont, eyeDense(2), eyeDense(2), 0.1, o); return err }},
	} {
		g.reset()
		if err := tc.run(&RiccatiOpts{}); g.calls == 0 {
			t.Fatalf("%s: precondition: default (err %v) ran no gate", tc.name, err)
		}
		g.reset()
		err := tc.run(&RiccatiOpts{NoCertificate: true})
		if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: NoCertificate: %v", tc.name, err)
		}
		if g.calls != 0 {
			t.Errorf("%s: NoCertificate: %d gate runs, want 0", tc.name, g.calls)
		}
	}
}

// TestCare_NoCertificateIgnored (EPYIVF): Care has no enclosure gate and
// ignores opts.NoCertificate.
func TestCare_NoCertificateIgnored(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.7, 0.2, -0.4, -0.5, 0.9, 0.3, -0.6, -1.2})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.3, 0.8, 0.5, -0.6})
	want, err := Care(A, B, eyeDense(3), eyeDense(2), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Care(A, B, eyeDense(3), eyeDense(2), &RiccatiOpts{NoCertificate: true})
	if err != nil {
		t.Fatalf("NoCertificate: %v, want nil", err)
	}
	if !mat.Equal(got.X, want.X) || !mat.Equal(got.K, want.K) {
		t.Error("NoCertificate: X or K differs from nil opts")
	}
}
