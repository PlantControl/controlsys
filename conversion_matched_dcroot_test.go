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

// matchedRationalLaurent is the order of the lowest term in σ of the
// transfer function of the explicit delay-free SISO sys at x0 + σ, and its
// coefficient, by exact rational arithmetic independent of the library.
func matchedRationalLaurent(t *testing.T, sys *System, x0 float64) (int, *big.Rat) {
	t.Helper()
	n, _, _ := sys.Dims()
	if sys.E != nil {
		t.Fatal("explicit model expected")
	}
	if n == 0 {
		return 0, fzRat(sys.D.At(0, 0))
	}
	a := mat.DenseCopyOf(sys.A).RawMatrix().Data
	b := mat.DenseCopyOf(sys.B).RawMatrix().Data
	c := mat.DenseCopyOf(sys.C).RawMatrix().Data
	low := func(desc []*big.Rat) (int, *big.Rat) {
		asc := fzTaylorShift(desc, fzRat(x0))
		for k, v := range slices.Backward(asc) {
			if v.Sign() != 0 {
				return len(asc) - 1 - k, v
			}
		}
		t.Fatal("identically zero polynomial")
		return 0, nil
	}
	on, nv := low(fzPencilDetRat(n, 1, a, b, c, []float64{sys.D.At(0, 0)}))
	od, dv := low(fzCharPolyExact(a, n))
	return on - od, new(big.Rat).Quo(nv, dv)
}

// checkMatchedDCRoot reports an error unless the DC gain of d, exactly for
// its stored entries, is that of sys with its root at DC: at a pole, the
// order of the lowest term at z = 1 (C2D) or s = 0 (D2C) and the sign of
// its coefficient; at a zero, a gain of exactly 0 or within the descriptor
// bar of sys. Where the orders agree, that coefficient times dt^order
// (C2D) or dt^−order (D2C) must be within 1e-12 of sys's when withLead is set.
func checkMatchedDCRoot(t *testing.T, name string, sys, d *System, dt float64, withLead bool) {
	t.Helper()
	x0, y0, p := 0.0, 1.0, float64(1)
	if sys.IsDiscrete() {
		x0, y0, p = 1, 0, -1
	}
	order, lead := matchedRationalLaurent(t, sys, x0)
	free := *d
	free.InputDelay, free.OutputDelay, free.Delay = nil, nil, nil
	got, glead := matchedRationalLaurent(t, &free, y0)
	switch {
	case order < 0 && (got != order || glead.Sign() != lead.Sign()):
		t.Errorf("%s: result has order %d (sign %d) at DC, model %d (sign %d)", name, got, glead.Sign(), order, lead.Sign())
		return
	case order > 0 && got < 0:
		t.Errorf("%s: result has a pole at DC, model a zero of order %d", name, order)
		return
	case order > 0 && got == 0:
		g, _ := glead.Float64()
		if bar, ok := matchedZeroBar(sys); !ok || math.Abs(g) > bar {
			t.Errorf("%s: DC gain %g of the result, model 0 (bar %g)", name, g, bar)
		}
		return
	case got != order || !withLead:
		return
	}
	g, _ := glead.Float64()
	w, _ := lead.Float64()
	if g *= math.Pow(dt, p*float64(order)); math.Abs(g-w) > 1e-12*math.Abs(w) {
		t.Errorf("%s: leading coefficient %g, want %g", name, g, w)
	}
}

// AQU6M7: a pole or zero at DC is the stored data's exact limit (U-3), not
// a root within 1e-9 of z = 1 or 1e-9/dt of s = 0; matched puts the roots
// nearest DC there, matches the exact leading coefficient, and certifies
// that the result's exact expansion holds the order and sign.
func TestC2DMatchedDCRootStoredExact(t *testing.T) {
	cases := map[string]*System{
		"integrator":                 conversionSISO(t, []float64{2}, []float64{1, 0}, 0),
		"double integrator":          conversionSISO(t, []float64{1, 3}, []float64{1, 4, 0, 0}, 0),
		"zero at DC":                 conversionSISO(t, []float64{1, 0}, []float64{1, 1}, 0),
		"double zero at DC":          conversionSISO(t, []float64{-3, 0, 0}, []float64{1, 2, 5, 1}, 0),
		"integrator, slow":           conversionSISO(t, []float64{1}, []float64{1, 1e-6, 0}, 0),
		"window dc pole":             matchedFixture(t, "window dc pole"),
		"window dc zero":             matchedFixture(t, "window dc zero"),
		"window dc pole cancelled":   matchedFixture(t, "window dc pole cancelled"),
		"window dc pole kept copies": matchedFixture(t, "window dc pole kept copies"),
		"zero at DC, D and C":        nil,
	}
	zd, err := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	cases["zero at DC, D and C"] = zd
	// The pole at −1e-6 (the zero at 9.8e-12 of near1/73, the roots near DC
	// rlrep.169 keeps) has |p·dt| ≤ 1e-6: a stored e^{p·dt} holds
	// 1 − e^{p·dt} only to ε/|p·dt|, so the result's leading coefficient (not
	// certified) is off by that much while its DC gain is exact.
	unheld := map[string]bool{"integrator, slow": true, "window dc zero": true, "window dc pole kept copies": true}
	for name, sys := range cases {
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if err != nil {
				t.Errorf("%s @%g: %v", name, dt, err)
				continue
			}
			checkMatchedDCRoot(t, name, sys, d, dt, !unheld[name])
		}
	}
}

// AQU6M7: roots near DC that the stored model does not have at DC (census
// rows the 1e-9 windows took for DC roots, nil with the DC gain wrong) map
// like any other: nil only with the DC gain within the descriptor bar.
func TestC2DMatchedNearDCRootNotAtDC(t *testing.T) {
	for _, name := range []string{"near dc pole 1", "near dc pole 2", "near dc pole 3", "near dc root 4"} {
		sys := matchedFixture(t, name)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if err != nil {
				requireMatchedUnverifiedModel(t, d, err)
				continue
			}
			if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
				t.Errorf("%s @%g: nil with DC off by %.3g descriptor bars", name, dt, bars)
			}
		}
	}
}

// AQU6M7: the stored model has an exact DC pole, but the transfer function
// drops it (g3/rotZ.331: residue 5.7e-14) or its roots near DC miss the
// leading coefficient by 1e6 (near1/128): the roots mapped as given, with
// ErrAccuracyUnverified.
func TestC2DMatchedDCRootUnresolved(t *testing.T) {
	for _, name := range []string{"missed dc pole 1", "missed dc pole 2"} {
		sys := matchedFixture(t, name)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			requireMatchedUnverifiedModel(t, d, err)
			theta := 1e-3
			g := mustEvalFr(t, sys, complex(0, theta/dt))
			h := mustEvalFr(t, d, cmplx.Exp(complex(0, theta)))
			if e := math.Abs(cmplx.Abs(h)/cmplx.Abs(g) - 1); e > 1e-3 {
				t.Errorf("%s @%g: unverified model's gain at θ = %g off by %.3g", name, dt, theta, e)
			}
		}
	}
}

// AQU6M7: a root whose image e^{p·dt} rounds to exactly z = 1 cannot be
// held off DC in float64; the model comes with ErrAccuracyUnverified.
func TestC2DMatchedRootHeldAtDCUnverified(t *testing.T) {
	sys := conversionSISO(t, []float64{1}, []float64{1, 1e-13}, 0)
	d, err := sys.C2D(1e-4, C2DOptions{Method: C2DMethodMatched})
	requireMatchedUnverifiedModel(t, d, err)
}

func TestD2CMatchedDCRootStoredExact(t *testing.T) {
	dt := 0.1
	for name, tf := range map[string][2][]float64{
		"integrator":        {{dt}, {1, -1}},
		"double integrator": {{dt * dt / 2, dt * dt / 2}, {1, -2, 1}},
		"zero at DC":        {{1, -1}, {1, -0.5}},
	} {
		sys := conversionSISO(t, tf[0], tf[1], dt)
		c, err := sys.D2C(D2COptions{Method: C2DMethodMatched})
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		checkMatchedDCRoot(t, name, sys, c, dt, true)
	}
}

func TestMatchedExactLaurent(t *testing.T) {
	for _, tc := range []struct {
		name       string
		a, b, c, d []float64
		x0         float64
		order      int
	}{
		{"regular", []float64{-1, 2, 0, -3}, []float64{1, 1}, []float64{1, 0}, []float64{0}, 0, 0},
		{"pole", []float64{0, 1, 0, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0, -1},
		{"zero by D", []float64{-2}, []float64{1}, []float64{-2}, []float64{1}, 0, 1},
		{"cancelled off DC", []float64{0, 1, 0, -3}, []float64{1, 0}, []float64{1, 0}, []float64{0}, 0, -1},
		{"pole at one", []float64{1, 0.5, 0, 0.25}, []float64{1, 1}, []float64{1, -1}, []float64{0.5}, 1, -1},
		{"uncontrollable", []float64{0, 0, 1, -2}, []float64{0, 1}, []float64{1, 1}, []float64{0}, 0, 0},
	} {
		n := len(tc.b)
		sys, err := NewFromSlices(n, 1, 1, tc.a, tc.b, tc.c, tc.d, 0)
		if err != nil {
			t.Fatal(err)
		}
		order, lead, ok := matchedExactLaurent(nil, sys.A, sys.B, sys.C, sys.D, tc.x0)
		wo, wl := matchedRationalLaurent(t, sys, tc.x0)
		w, _ := wl.Float64()
		if !ok || order != tc.order || order != wo || lead != w {
			t.Errorf("%s: order %d lead %g ok %v, want %d (oracle %d) %g", tc.name, order, lead, ok, tc.order, wo, w)
		}
	}
	e := mat.NewDense(2, 2, []float64{2, 1, 0, 0.5})
	a := mat.NewDense(2, 2, []float64{0, 1, 0, -1})
	order, lead, ok := matchedExactLaurent(e, a, mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if !ok || order != -1 || lead != 0.5 {
		t.Errorf("descriptor: order %d lead %g ok %v, want −1 0.5", order, lead, ok)
	}
	if _, _, ok := matchedExactLaurent(nil, mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0); ok {
		t.Error("zero transfer function reported an order")
	}
}

func TestMatchedMarkDC(t *testing.T) {
	roots := []complex128{-3, complex(-1e-12, 1e-12), complex(-1e-12, -1e-12), 1e-14, -2}
	for _, tc := range []struct {
		k    int
		want []bool
		ok   bool
	}{
		{1, []bool{false, false, false, true, false}, true},
		{2, []bool{false, false, false, true, true}, true},
		{3, []bool{false, true, true, true, false}, true},
		{6, nil, false},
	} {
		got, ok := matchedMarkDC(roots, tc.k)
		if ok != tc.ok || (ok && !slices.Equal(got, tc.want)) {
			t.Errorf("k=%d: %v %v, want %v %v", tc.k, got, ok, tc.want, tc.ok)
		}
	}
}

// AQU6M7: away from DC (not certified) a nil result keeps the gain at low
// frequency (the phase differs by the half sample of the zero at −1): g3/rotH.188 has an exact zero at DC that DCGain returns as
// 2.5e-32, which the gain must not be matched to; near1/100 has a pole at
// 9.7e-10 whose stored e^{p·dt} at dt = 1e-4 cannot hold 1 − e^{p·dt},
// which the gain must not be matched through.
func TestC2DMatchedLowFrequencyResponse(t *testing.T) {
	for _, tc := range []struct {
		name string
		dt   float64
	}{{"dc zero gain noise", 1e-2}, {"near dc cluster", 1e-4}} {
		sys := matchedFixture(t, tc.name)
		d, err := sys.C2D(tc.dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		theta := 1e-3
		g := mustEvalFr(t, sys, complex(0, theta/tc.dt))
		h := mustEvalFr(t, d, cmplx.Exp(complex(0, theta)))
		if e := math.Abs(cmplx.Abs(h)/cmplx.Abs(g) - 1); e > 1e-5 {
			t.Errorf("%s: gain at θ = %g off by %.3g", tc.name, theta, e)
		}
	}
}

// matchedSingularPencilModel is an explicit model with det(A) = 0 exactly and
// a finite DC limit: rows 0 and 1 of A, and of B, differ by the factor k, so
// the mode at 0 is uncontrollable; the entries are float32 values, so k·x is
// exact, and the LU need not meet a zero pivot.
func matchedSingularPencilModel(t *testing.T, a, b float32, k float64, tail []float64) *System {
	t.Helper()
	m := len(tail)
	n := 2 + m
	A := mat.NewDense(n, n, nil)
	A.Set(0, 0, float64(a))
	A.Set(0, 1, float64(b))
	A.Set(1, 0, k*float64(a))
	A.Set(1, 1, k*float64(b))
	B := mat.NewDense(n, 1, nil)
	B.Set(0, 0, float64(a))
	B.Set(1, 0, k*float64(a))
	C := mat.NewDense(1, n, nil)
	C.Set(0, 0, 0.5)
	C.Set(0, 1, -0.25)
	for i, l := range tail {
		A.Set(2+i, 0, 1+float64(i))
		A.Set(2+i, 1, 2-float64(i))
		if i > 0 {
			A.Set(2+i, 1+i, 0.75)
		}
		A.Set(2+i, 2+i, -l)
		B.Set(2+i, 0, 1)
		C.Set(0, 2+i, 2-0.5*float64(i))
	}
	sys, err := New(A, B, C, mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// AQU6M7 (F1, U-A): an exactly singular DC pencil with a finite exact limit
// takes the bar u·(2|g0 − D| + |D|), whether or not its LU meets a zero
// pivot; a rounding-size pivot gave a bar of 1e20 that passed any DC (the
// arbiter's model, DC off by 1.25e-8 at dt = 1e-8 with nil).
func TestC2DMatchedSingularPencilBar(t *testing.T) {
	models := map[string]*System{"arbiter": matchedSingularPencilModel(t, 0.47, 0.81, 3, []float64{1.5})}
	for i, c := range []struct {
		a, b float32
		k    float64
		tail []float64
	}{
		{0.3, -1.7, 5, []float64{2}}, {1.1, 0.13, 7, []float64{0.5, 3}}, {-0.62, 2.9, 3, []float64{1, 4, 0.25}},
		{0.07, 0.91, 9, []float64{6}}, {2.3, -0.4, 11, []float64{0.1, 1.3}},
	} {
		models[string(rune('a'+i))] = matchedSingularPencilModel(t, c.a, c.b, c.k, c.tail)
	}
	models["g3/rotZ.7"] = matchedFixture(t, "singular pencil zero pivot")
	nils := 0
	for name, sys := range models {
		n, _, _ := sys.Dims()
		order, lead := matchedRationalLaurent(t, sys, 0)
		if order != 0 {
			t.Fatalf("%s: order %d at DC, want a finite limit", name, order)
		}
		if exactDetIsZero(sys.A.RawMatrix()) != true {
			t.Fatalf("%s: A not exactly singular", name)
		}
		g0, _ := lead.Float64()
		bar := 4 * float64(n+5) * 0x1p-52 * 2 * math.Abs(g0)
		for _, dt := range []float64{1e-2, 1e-4, 1e-6, 1e-8} {
			d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if err != nil {
				requireMatchedUnverifiedModel(t, d, err)
				continue
			}
			nils++
			free := *d
			free.InputDelay, free.OutputDelay, free.Delay = nil, nil, nil
			free.Dt = 1
			h, _ := matchedDC320(t, &free, 1)
			if e := math.Abs(h - g0); !(e <= bar) {
				t.Errorf("%s @%g: nil with DC off by %.3g, bar %.3g", name, dt, e, bar)
			}
		}
	}
	if nils == 0 {
		t.Error("no singular pencil returned nil")
	}
}

// AQU6M7 (U-B): where the gain through 1 − e^{p·dt} misses the DC bar and
// the stored discrete roots' gain is taken, their DC product rescaled to
// DCGain must be within terms (R2), else the DC-matched model comes with
// ErrAccuracyUnverified: g3/rotZ.144 and near1/72 at dt = 1e-4, gain off
// by up to 3e-4.
func TestC2DMatchedStoredRootsRescaleUnverified(t *testing.T) {
	for _, name := range []string{"stored roots refuted", "near dc pole 3"} {
		sys := matchedFixture(t, name)
		d, err := sys.C2D(1e-4, C2DOptions{Method: C2DMethodMatched})
		requireMatchedUnverifiedModel(t, d, err)
		if bars := matchedDCInBars(t, sys, d); !(bars <= 1) {
			t.Errorf("%s: DC of the unverified model off by %.3g bars", name, bars)
		}
	}
}

func TestMatchedLaurentAgrees(t *testing.T) {
	ref := conversionSISO(t, []float64{1}, []float64{1, 1, 0}, 0)
	zref := conversionSISO(t, []float64{1, 0}, []float64{1, 1}, 0)
	for _, tc := range []struct {
		name     string
		num, den []float64
		ref      *System
		root     matchedDCRoot
		want     bool
	}{
		{"pole", []float64{1}, []float64{1, -1}, ref, matchedDCRoot{order: -1, lead: 1}, true},
		{"pole, sign", []float64{-1}, []float64{1, -1}, ref, matchedDCRoot{order: -1, lead: 1}, false},
		{"pole, order", []float64{1}, []float64{1, -2, 1}, ref, matchedDCRoot{order: -1, lead: 1}, false},
		{"pole missing", []float64{1}, []float64{1, -0.5}, ref, matchedDCRoot{order: -1, lead: 1}, false},
		{"zero", []float64{1, -1}, []float64{1, -0.5}, zref, matchedDCRoot{order: 1, lead: 1}, true},
		{"zero within bar", []float64{1, -1 + 0x1p-52}, []float64{1, -0.5}, zref, matchedDCRoot{order: 1, lead: 1}, true},
		{"zero off", []float64{1, -0.999}, []float64{1, -0.5}, zref, matchedDCRoot{order: 1, lead: 1}, false},
		{"zero, pole", []float64{1}, []float64{1, -1}, zref, matchedDCRoot{order: 1, lead: 1}, false},
	} {
		out := conversionSISO(t, tc.num, tc.den, 0.1)
		if got := matchedLaurentAgrees(out, tc.ref, tc.root, 1); got != tc.want {
			t.Errorf("%s: %v, want %v", tc.name, got, tc.want)
		}
	}
}

var _ = errors.Is
