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

func dareCertRows() []dareFloorCase {
	return append(append([]dareFloorCase{}, dareNoiseFloorCases...),
		dareEnclosureCtrlCase, dareEnclosureBigCase, dareEnclosureWCase, dareEnclosureWBoundaryCase, dareEnclosureAtXCase)
}

// TestDare_NoCertificateAccuracy (EPYIVF): on every row whose default Dare
// runs the gate and returns a nil error (the gate certified X at X, at
// its centre, or replaced it in tier 2), opts.NoCertificate runs no gate
// work and returns the X and K the gate was handed, with
// ErrAccuracyUnverified. Which rows gate depends on the build, so only
// that some row does is required.
func TestDare_NoCertificateAccuracy(t *testing.T) {
	g := countGate(t)
	gated, replaced := 0, 0
	for _, c := range dareCertRows() {
		A, B, Q, R, S, E := c.dense()
		for _, ns := range []bool{false, true} {
			g.reset()
			def, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: ns})
			if err != nil || g.calls == 0 {
				continue
			}
			gated++
			preX, preK := g.x, g.k
			if !mat.Equal(def.X, preX) {
				replaced++
			}
			g.reset()
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: ns, NoCertificate: true})
			if !errors.Is(err, ErrAccuracyUnverified) || res == nil || res.X == nil {
				t.Errorf("%s ns=%v: NoCertificate: result %v err %v, want a result with ErrAccuracyUnverified", c.name, ns, res != nil, err)
				continue
			}
			if g.calls != 0 {
				t.Errorf("%s ns=%v: NoCertificate: %d gate runs, want 0", c.name, ns, g.calls)
			}
			if !mat.Equal(res.X, preX) || !mat.Equal(res.K, preK) {
				t.Errorf("%s ns=%v: NoCertificate: X or K differs from the pre-gate candidate", c.name, ns)
			}
		}
	}
	if gated == 0 {
		t.Fatal("no row ran the gate with a nil default error in this build")
	}
	t.Logf("%d gated solves, %d with X replaced by the gate", gated, replaced)
}

// TestDare_NoCertificateExemptRow (EPYIVF): near-singular descriptor rows
// (nearSingularTriangularE seeds 163, 213, 415, 500, dt 1, Q = I, R = I)
// whose X the extended solve confirmed never reach the gate, so
// opts.NoCertificate leaves their result, and its nil error, unchanged.
// Which rows are exempt depends on the build, so only that some row is
// is required.
func TestDare_NoCertificateExemptRow(t *testing.T) {
	g := countGate(t)
	rows := []struct{ a, b, e []float64 }{
		{[]float64{-0.13724618874059225, 1.217088645560901, 0.6539759935733573, 0.625550292051194},
			[]float64{1.0056269676565766, -0.43721297890158445},
			[]float64{0.8136281888350105, -0.45758472346960766, 0, 3.8390252196819285e-16}},
		{[]float64{0.5000278208714111, -0.07733055775849312, -0.4888167933205462, 1.3129070029607885},
			[]float64{0.3112712539078186, 1.219531204968224},
			[]float64{0.8602200064118372, 0.06309803400898362, 0, 2.64158237466957e-16}},
		{[]float64{-2.3221006124296615, -1.0818055667933515, -1.021140292167516, 0.7023646645229246},
			[]float64{0.8694884592755641, 2.527025463345394, -0.22663117073751365, -1.131236581183983},
			[]float64{0.7047884850050473, -0.07412186793679755, 0, 3.275877501116736e-16}},
		{[]float64{-0.7312715912385475, -0.054319723233813816, 0.132396174077426, -0.5991796548980122},
			[]float64{0.6561074877388584, -0.5688892842798058, 0.6471839580069148, 0.33716000964540804},
			[]float64{0.6097791941584733, -0.3393214326084397, 0, 3.188503216119319e-16}},
	}
	exempt := 0
	for i, r := range rows {
		m := len(r.b) / 2
		A, B, E := mat.NewDense(2, 2, r.a), mat.NewDense(2, m, r.b), mat.NewDense(2, 2, r.e)
		Q, R := eyeDense(2), eyeDense(m)
		g.reset()
		def, err := Dare(A, B, Q, R, &RiccatiOpts{E: E})
		if err != nil || g.calls != 0 {
			continue
		}
		exempt++
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoCertificate: true})
		if err != nil {
			t.Errorf("row %d: NoCertificate: %v, want nil", i, err)
			continue
		}
		if g.calls != 0 || !mat.Equal(res.X, def.X) || !mat.Equal(res.K, def.K) {
			t.Errorf("row %d: NoCertificate: %d gate runs, X equal %v, K equal %v; want 0, true, true", i, g.calls, mat.Equal(res.X, def.X), mat.Equal(res.K, def.K))
		}
	}
	if exempt == 0 {
		t.Fatal("no row was exempt from the gate with a nil default error in this build")
	}
}

// TestDare_NoCertificateDefaultUnchanged (EPYIVF): NoCertificate false is
// nil opts bit for bit, with the same gate runs.
func TestDare_NoCertificateDefaultUnchanged(t *testing.T) {
	g := countGate(t)
	for _, c := range dareCertRows() {
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
		t.Skipf("default Dlqr (err %v) ran no gate in this build", err)
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
