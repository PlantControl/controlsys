package controlsys

import (
	"errors"
	"math/cmplx"
	"math/rand"
	"strings"
	"testing"
)

func TestRss_Basic(t *testing.T) {
	sys, err := Rss(5, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 5 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (5,3,2)", n, m, p)
	}

	if !sys.IsContinuous() {
		t.Error("expected continuous system")
	}

	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := sys.Poles()
		t.Errorf("system not stable, poles = %v", poles)
	}
}

func TestRss_StrictStability(t *testing.T) {
	for i := range 20 {
		sys, err := Rss(8, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		poles, _ := sys.Poles()
		for j, p := range poles {
			if real(p) >= 0 {
				t.Errorf("trial %d: pole[%d] = %v has non-negative real part", i, j, p)
			}
		}
	}
}

func TestRss_ZeroStates(t *testing.T) {
	sys, err := Rss(0, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 0 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (0,3,2)", n, m, p)
	}
}

func TestDrss_Basic(t *testing.T) {
	sys, err := Drss(4, 1, 1, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 4 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (4,1,1)", n, m, p)
	}

	if !sys.IsDiscrete() || sys.Dt != 0.1 {
		t.Errorf("expected discrete with dt=0.1, got dt=%g", sys.Dt)
	}

	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("discrete system not stable")
	}
}

func TestDrss_StrictStability(t *testing.T) {
	for i := range 20 {
		sys, err := Drss(6, 2, 2, 0.05)
		if err != nil {
			t.Fatal(err)
		}
		poles, _ := sys.Poles()
		for j, p := range poles {
			if cmplx.Abs(p) >= 1 {
				t.Errorf("trial %d: pole[%d] = %v has |p| >= 1", i, j, p)
			}
		}
	}
}

func TestDrss_ZeroStates(t *testing.T) {
	sys, err := Drss(0, 1, 1, 0.5)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	if n != 0 {
		t.Fatalf("n = %d, want 0", n)
	}
}

func TestDrss_InvalidDt(t *testing.T) {
	_, err := Drss(4, 1, 1, 0)
	if err == nil {
		t.Error("expected error for dt=0")
	}

	_, err = Drss(4, 1, 1, -1)
	if err == nil {
		t.Error("expected error for dt<0")
	}
}

func TestRandomSS_DeterministicAdapter(t *testing.T) {
	rng1 := rand.New(rand.NewSource(42))
	rng2 := rand.New(rand.NewSource(42))

	sys1, err := randomSSWithSource(randomStableModelSpec{states: 3, outputs: 1, inputs: 1, dt: 0, continuous: true}, rng1)
	if err != nil {
		t.Fatal(err)
	}
	sys2, err := randomSSWithSource(randomStableModelSpec{states: 3, outputs: 1, inputs: 1, dt: 0, continuous: true}, rng2)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(sys1.A, sys2.A, 0) || !matEqual(sys1.B, sys2.B, 0) || !matEqual(sys1.C, sys2.C, 0) {
		t.Fatal("same seed produced different systems")
	}
}

func TestRss_InvalidDims(t *testing.T) {
	_, err := Rss(-1, 1, 1)
	if err == nil {
		t.Error("expected error for n<0")
	}

	_, err = Rss(1, 0, 1)
	if err == nil {
		t.Error("expected error for p=0")
	}

	_, err = Rss(1, 1, 0)
	if err == nil {
		t.Error("expected error for m=0")
	}
}

func TestRssDrssErrorSentinels(t *testing.T) {
	for _, d := range [][3]int{{-1, 1, 1}, {2, 0, 1}, {2, 1, 0}} {
		if _, err := Rss(d[0], d[1], d[2]); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "Rss: ") {
			t.Errorf("Rss%v err = %v", d, err)
		}
		if _, err := Drss(d[0], d[1], d[2], 0.1); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "Drss: ") {
			t.Errorf("Drss%v err = %v", d, err)
		}
	}
	if _, err := Drss(2, 1, 1, 0); !errors.Is(err, ErrInvalidSampleTime) || !strings.HasPrefix(err.Error(), "Drss: ") {
		t.Errorf("Drss dt=0 err = %v", err)
	}
}
