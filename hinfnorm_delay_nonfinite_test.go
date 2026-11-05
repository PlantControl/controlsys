package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

// OAIXFO: the delay peak search and the delay Nyquist test hung on
// non-finite internal-delay data reached past the public entry checks.

const delayNonFiniteLimit = 10 * time.Second

func delayNonFiniteModel(t *testing.T) *System {
	t.Helper()
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, -.5, -3}), mat.NewDense(2, 1, []float64{1, .5}),
		mat.NewDense(1, 2, []float64{1, .3}), mat.NewDense(1, 1, []float64{.1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{.3}, mat.NewDense(2, 1, []float64{.2, .1}), mat.NewDense(1, 2, []float64{.5, -.4}),
		mat.NewDense(1, 1, []float64{.3}), mat.NewDense(1, 1, []float64{.7}), mat.NewDense(1, 1, []float64{.2})); err != nil {
		t.Fatal(err)
	}
	return sys
}

func delayNonFiniteBlock(sys *System, name string) *mat.Dense {
	switch name {
	case "A":
		return sys.A
	case "B":
		return sys.B
	case "C":
		return sys.C
	case "D":
		return sys.D
	case "B2":
		return sys.LFT.B2
	case "C2":
		return sys.LFT.C2
	case "D12":
		return sys.LFT.D12
	case "D21":
		return sys.LFT.D21
	}
	return sys.LFT.D22
}

func TestHinfNormDelayedNonFiniteModel(t *testing.T) {
	entries := map[string]func(*System) error{
		"hinfNormDelayed": func(s *System) error { _, _, _, err := hinfNormDelayed(s); return err },
		"linfNormDelayed": func(s *System) error { _, _, err := linfNormDelayed(s); return err },
		"delayPeakSetup":  func(s *System) error { _, _, _, err := delayPeakSetup(s); return err },
		"internalDelayStable": func(s *System) error {
			_, err := internalDelayStable(s)
			return err
		},
	}
	for _, blk := range []string{"A", "B", "C", "D", "B2", "C2", "D12", "D21", "D22"} {
		for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			for name, f := range entries {
				sys := delayNonFiniteModel(t)
				delayNonFiniteBlock(sys, blk).Set(0, 0, v)
				var err error
				finishesWithin(t, delayNonFiniteLimit, func() { err = f(sys) })
				if !errors.Is(err, ErrInvalidArgument) {
					t.Errorf("%s with %s[0,0]=%g: err = %v, want ErrInvalidArgument", name, blk, v, err)
				}
			}
		}
	}
}

func TestHinfNormDelayedNonFiniteStaticModel(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		sys, err := NewGain(mat.NewDense(1, 1, []float64{.1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInternalDelay([]float64{.3}, nil, nil, mat.NewDense(1, 1, []float64{.3}),
			mat.NewDense(1, 1, []float64{.7}), mat.NewDense(1, 1, []float64{.5})); err != nil {
			t.Fatal(err)
		}
		sys.LFT.D22.Set(0, 0, v)
		var errH, errS error
		finishesWithin(t, delayNonFiniteLimit, func() {
			_, _, _, errH = hinfNormDelayed(sys)
			_, errS = internalDelayStable(sys)
		})
		if !errors.Is(errH, ErrInvalidArgument) || !errors.Is(errS, ErrInvalidArgument) {
			t.Errorf("D22=%g: hinfNormDelayed err = %v, internalDelayStable err = %v, want ErrInvalidArgument", v, errH, errS)
		}
	}
}

// The model turns non-finite after newDelayLFT checked it, so the loops
// themselves meet the NaN/Inf: the descriptor balance spun forever on it.
func TestDelayLFTNonFiniteAfterSetup(t *testing.T) {
	for _, blk := range []string{"B2", "C2", "D22"} {
		for _, v := range []float64{math.NaN(), math.Inf(1)} {
			sys := delayNonFiniteModel(t)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			delayNonFiniteBlock(sys, blk).Set(0, 0, v)
			var errStab, errNyq error
			finishesWithin(t, delayNonFiniteLimit, func() {
				_, _, errStab = e.stability()
				_, _, errNyq = e.nyquist(true)
			})
			if !errors.Is(errStab, ErrOverflow) || !errors.Is(errNyq, ErrOverflow) {
				t.Errorf("%s=%g: stability err = %v, nyquist err = %v, want ErrOverflow", blk, v, errStab, errNyq)
			}

			sys = delayNonFiniteModel(t)
			e, err = newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			l, res, err := e.stability()
			if err != nil {
				t.Fatal(err)
			}
			delayNonFiniteBlock(sys, blk).Set(0, 0, v)
			finishesWithin(t, delayNonFiniteLimit, func() { _, _, err = e.peak(l, res.w) })
			if !errors.Is(err, ErrOverflow) {
				t.Errorf("%s=%g: peak err = %v, want ErrOverflow", blk, v, err)
			}
		}
	}
}

func delayNonFiniteEval(f func(frequencyPoint, []complex128) error, v complex128, from float64) func(frequencyPoint, []complex128) error {
	return func(pt frequencyPoint, dst []complex128) error {
		err := f(pt, dst)
		if imag(pt.value()) >= from {
			for i := range dst {
				dst[i] = v
			}
		}
		return err
	}
}

// A non-finite H22 or G response without an evaluation failure: the
// Nyquist test took a NaN 1+L for a closed-loop root on the axis and
// answered unstable with a nil error, and the peak scan skipped NaN gains.
func TestDelayLFTNonFiniteResponse(t *testing.T) {
	vals := []complex128{cmplx.NaN(), cmplx.Inf(), complex(math.Inf(-1), 0)}
	for _, v := range vals {
		for _, from := range []float64{0, 1} {
			e, err := newDelayLFT(delayNonFiniteModel(t))
			if err != nil {
				t.Fatal(err)
			}
			e.h22 = delayNonFiniteEval(e.h22, v, from)
			var errStab, errNyq error
			finishesWithin(t, delayNonFiniteLimit, func() {
				_, _, errStab = e.stability()
				_, _, errNyq = e.nyquist(true)
			})
			if !errors.Is(errStab, ErrOverflow) || !errors.Is(errNyq, ErrOverflow) {
				t.Errorf("H22=%v from ω=%g: stability err = %v, nyquist err = %v, want ErrOverflow", v, from, errStab, errNyq)
			}

			e, err = newDelayLFT(delayNonFiniteModel(t))
			if err != nil {
				t.Fatal(err)
			}
			l, res, err := e.stability()
			if err != nil {
				t.Fatal(err)
			}
			e.g = delayNonFiniteEval(e.g, v, from)
			finishesWithin(t, delayNonFiniteLimit, func() { _, _, err = e.peak(l, res.w) })
			if err == nil {
				t.Errorf("G=%v from ω=%g: peak returned no error", v, from)
			}
			if cmplx.IsNaN(v) && !errors.Is(err, ErrOverflow) {
				t.Errorf("G=NaN from ω=%g: peak err = %v, want ErrOverflow", from, err)
			}
		}
	}
}

func TestDelaySensitivityPeaksNonFiniteLoop(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, -.5, -3}), mat.NewDense(2, 1, []float64{1, .5}),
		mat.NewDense(1, 2, []float64{1, .3}), mat.NewDense(1, 1, []float64{.1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{.3}); err != nil {
		t.Fatal(err)
	}
	for _, v := range []complex128{cmplx.NaN(), cmplx.Inf()} {
		l, err := delayLoopFromSystem(sys, "test")
		if err != nil {
			t.Fatal(err)
		}
		at := l.at
		l.at = func(w float64) complex128 {
			if w >= 1 {
				return v
			}
			return at(w)
		}
		finishesWithin(t, delayNonFiniteLimit, func() { _, _, err = l.sensitivityPeaks(0, -.5) })
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("L=%v: sensitivityPeaks err = %v, want ErrOverflow", v, err)
		}
	}
}

func TestDescriptorResponseNonFinite(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		sys := delayNonFiniteModel(t)
		sys.LFT.B2.Set(1, 0, v)
		var err error
		finishesWithin(t, delayNonFiniteLimit, func() { _, err = newDescriptorResponse(sys) })
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("B2=%g: newDescriptorResponse err = %v, want ErrOverflow", v, err)
		}

		r, err := newDescriptorResponse(delayNonFiniteModel(t))
		if err != nil {
			t.Fatal(err)
		}
		r.m0[1] = complex(v, 0)
		finishesWithin(t, delayNonFiniteLimit, r.balance)
	}
}

func TestTorusPassagesIndicesNonFinite(t *testing.T) {
	p := &torusPassages{h: []float64{1}, scale: []float64{1}, r0: 1}
	for _, c := range [][3]float64{{math.NaN(), 10, 1}, {0, math.NaN(), 1}, {0, 10, math.NaN()}} {
		var ks []int
		finishesWithin(t, delayNonFiniteLimit, func() { ks = p.indices(c[0], c[1], c[2], nil) })
		if len(ks) != 0 {
			t.Errorf("indices(%g, %g, %g) = %v, want none", c[0], c[1], c[2], ks)
		}
	}
	if ks := p.indices(0, 2.5, 1, nil); len(ks) != 2 || ks[0] != 1 || ks[1] != 2 {
		t.Errorf("indices(0, 2.5, 1) = %v, want [1 2]", ks)
	}
}

func TestHinfNormDelayedFiniteUnchanged(t *testing.T) {
	sys := delayNonFiniteModel(t)
	norm, w, stable, err := hinfNormDelayed(sys)
	if err != nil || !stable {
		t.Fatalf("hinfNormDelayed: stable = %v, err = %v", stable, err)
	}
	const want = 1.9571428571428573
	if math.Abs(norm-want) > 1e-12*want || w != 0 {
		t.Errorf("hinfNormDelayed = %.17g at ω=%g, want %.17g at 0", norm, w, want)
	}
	linf, _, err := linfNormDelayed(sys)
	if err != nil || math.Abs(linf-want) > 1e-12*want {
		t.Errorf("linfNormDelayed = %.17g, %v, want %.17g", linf, err, want)
	}
	if ok, err := internalDelayStable(sys); err != nil || !ok {
		t.Errorf("internalDelayStable = %v, %v, want true", ok, err)
	}
}
