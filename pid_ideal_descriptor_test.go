package controlsys

import (
	"math"
	"math/cmplx"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// MATLAB ss(pid(99,49,29)) as printed in MATLAB Answers 1984559.
func TestPIDIdealDerivativeMatchesMATLABDescriptor(t *testing.T) {
	sys, err := mustPID(t, 99, 49, 29, 0, 0).System()
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		got  *mat.Dense
		want []float64
	}{
		"A": {sys.A, []float64{0, 0, 0, 0, 1, 0, 0, 0, 1}},
		"B": {sys.B, []float64{1, 0, -1}},
		"C": {sys.C, []float64{49, 29, 0}},
		"D": {sys.D, []float64{99}},
		"E": {sys.E, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0}},
	}
	for name, m := range want {
		if m.got == nil {
			t.Fatalf("%s is nil", name)
		}
		if got := m.got.RawMatrix().Data; !equalFloats(got, m.want) {
			t.Fatalf("%s = %v, want %v", name, got, m.want)
		}
	}
}

func equalFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPIDIdealDerivativeDescriptorResponse(t *testing.T) {
	omega := []float64{0.03, 0.7, 2, 13, 400}
	cases := []struct {
		name       string
		kp, ki, kd float64
		n          int
		poles      int
	}{
		{"PID", 1.7, -0.4, 2.3, 3, 1},
		{"PD", -0.6, 0, 3.1, 2, 0},
	}
	for _, c := range cases {
		sys, err := mustPID(t, c.kp, c.ki, c.kd, 0, 0).System()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if n, _, _ := sys.Dims(); n != c.n || sys.E == nil {
			t.Fatalf("%s: n = %d, E = %v", c.name, n, sys.E)
		}
		if proper, err := sys.IsProper(); err != nil || proper {
			t.Fatalf("%s: IsProper = %v, %v", c.name, proper, err)
		}
		poles, err := sys.Poles()
		if err != nil || len(poles) != c.poles || (c.poles == 1 && cmplx.Abs(poles[0]) > 1e-12) {
			t.Fatalf("%s: poles = %v, %v", c.name, poles, err)
		}
		h, err := sys.FreqResponse(omega)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		for k, w := range omega {
			s := complex(0, w)
			want := complex(c.kp, 0) + complex(c.kd, 0)*s
			if c.ki != 0 {
				want += complex(c.ki, 0) / s
			}
			if got := h.At(k, 0, 0); cmplx.Abs(got-want) > 1e-10*cmplx.Abs(want) {
				t.Fatalf("%s: H(j%g) = %v, want %v", c.name, w, got, want)
			}
		}
	}
}

// The 2-DOF state order is derived from the 1-DOF MATLAB printout; only the
// transfer from (r, y) is checked against the pid2 formula.
func TestPID2IdealDerivativeDescriptorResponse(t *testing.T) {
	kp, ki, kd, b, c := 1.3, 0.8, 0.45, 0.6, -0.35
	sys, err := mustPID2(t, kp, ki, kd, 0, b, c, 0).System()
	if err != nil {
		t.Fatal(err)
	}
	if n, m, p := sys.Dims(); n != 3 || m != 2 || p != 1 || sys.E == nil {
		t.Fatalf("dims %d,%d,%d E=%v", n, m, p, sys.E)
	}
	omega := []float64{0.05, 1.1, 9, 250}
	h, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		s := complex(0, w)
		want := []complex128{
			complex(kp*b, 0) + complex(ki, 0)/s + complex(kd*c, 0)*s,
			-(complex(kp, 0) + complex(ki, 0)/s + complex(kd, 0)*s),
		}
		for j := range want {
			if got := h.At(k, 0, j); cmplx.Abs(got-want[j]) > 1e-10*cmplx.Abs(want[j]) {
				t.Fatalf("H[%d](j%g) = %v, want %v", j, w, got, want[j])
			}
		}
	}
	if _, err := (&PID2{Kp: 1, Kd: 1, B: 1, C: 1}).explicitSystem(); err == nil || !strings.Contains(err.Error(), "Tf=0") {
		t.Fatalf("explicit 2-DOF ideal derivative err = %v", err)
	}
}

// Unity feedback of pid(99,49,29) around 1/((s+1)(s+2)) has characteristic
// polynomial s³ + 32s² + 101s + 49.
func TestPIDIdealDerivativeDescriptorClosedLoop(t *testing.T) {
	ctrl, err := mustPID(t, 99, 49, 29, 0, 0).System()
	if err != nil {
		t.Fatal(err)
	}
	plant, err := New(mat.NewDense(2, 2, []float64{0, 1, -2, -3}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	loop, err := Series(ctrl, plant)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(loop, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := cl.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 3 {
		t.Fatalf("closed-loop poles = %v, want 3", poles)
	}
	for _, p := range poles {
		r := ((p+32)*p+101)*p + 49
		scale := math.Pow(cmplx.Abs(p), 3) + 32*cmplx.Abs(p)*cmplx.Abs(p) + 101*cmplx.Abs(p) + 49
		if real(p) >= 0 || cmplx.Abs(r) > 1e-9*scale {
			t.Fatalf("pole %v: residual %v", p, r)
		}
	}
}
