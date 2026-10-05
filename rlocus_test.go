package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestRootLocus_Integrator(t *testing.T) {
	// G(s) = 1/s => pole at 0, no zeros
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RootLocus(sys, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Branches) != 1 {
		t.Fatalf("expected 1 branch, got %d", len(res.Branches))
	}

	// pole at K=0 should be at origin
	p0 := res.Branches[0][0]
	if cmplx.Abs(p0) > 0.01 {
		t.Errorf("pole at K=0 should be ~0, got %v", p0)
	}

	// as K increases, pole moves left on real axis
	last := res.Branches[0][len(res.Gains)-1]
	if real(last) >= 0 {
		t.Errorf("pole should move left for large K, got %v", last)
	}

	if len(res.AsymptoteAngles) != 1 {
		t.Fatalf("expected 1 asymptote, got %d", len(res.AsymptoteAngles))
	}
	if math.Abs(res.AsymptoteAngles[0]-math.Pi) > 0.01 {
		t.Errorf("asymptote should be pi, got %f", res.AsymptoteAngles[0])
	}
}

func TestRootLocus_SecondOrder(t *testing.T) {
	// G(s) = 1/(s^2+2s+1) = 1/(s+1)^2 => poles at -1,-1
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RootLocus(sys, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Branches) != 2 {
		t.Fatalf("expected 2 branches, got %d", len(res.Branches))
	}

	// breakaway at -1 (double pole)
	if len(res.Breakaway) == 0 {
		t.Fatal("expected breakaway point")
	}
	foundBreakaway := false
	for _, bp := range res.Breakaway {
		if math.Abs(real(bp)-(-1)) < 0.05 {
			foundBreakaway = true
			break
		}
	}
	if !foundBreakaway {
		t.Errorf("expected breakaway near -1, got %v", res.Breakaway)
	}

	if len(res.AsymptoteAngles) != 2 {
		t.Fatalf("expected 2 asymptotes, got %d", len(res.AsymptoteAngles))
	}
	// asymptotes at pi/2 and 3pi/2
	angles := make([]float64, len(res.AsymptoteAngles))
	copy(angles, res.AsymptoteAngles)
	for i := range angles {
		angles[i] = math.Mod(angles[i], 2*math.Pi)
	}
	if math.Abs(angles[0]-math.Pi/2) > 0.01 {
		t.Errorf("first asymptote should be pi/2, got %f", angles[0])
	}
	if math.Abs(angles[1]-3*math.Pi/2) > 0.01 {
		t.Errorf("second asymptote should be 3pi/2, got %f", angles[1])
	}
}

func TestRootLocus_WithZero(t *testing.T) {
	// G(s) = (s+3)/(s+1)^2 => zero at -3, poles at -1,-1
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RootLocus(sys, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Branches) != 2 {
		t.Fatalf("expected 2 branches, got %d", len(res.Branches))
	}

	// 1 zero => 1 arrival angle
	if len(res.ArrivalAngles) != 1 {
		t.Fatalf("expected 1 arrival angle, got %d", len(res.ArrivalAngles))
	}

	// nPoles - nZeros = 1, so 1 asymptote
	if len(res.AsymptoteAngles) != 1 {
		t.Fatalf("expected 1 asymptote, got %d", len(res.AsymptoteAngles))
	}
}

func TestRootLocus_ThirdOrder(t *testing.T) {
	// G(s) = 1/(s(s+1)(s+2)) => poles at 0,-1,-2
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RootLocus(sys, nil)
	if err != nil {
		t.Fatal(err)
	}

	// breakaway near -0.4226
	if len(res.Breakaway) == 0 {
		t.Fatal("expected breakaway points")
	}
	foundBreakaway := false
	for _, bp := range res.Breakaway {
		if math.Abs(real(bp)-(-0.4226)) < 0.05 {
			foundBreakaway = true
			break
		}
	}
	if !foundBreakaway {
		t.Errorf("expected breakaway near -0.4226, got %v", res.Breakaway)
	}

	// 3 asymptotes at 60, 180, 300 degrees
	if len(res.AsymptoteAngles) != 3 {
		t.Fatalf("expected 3 asymptotes, got %d", len(res.AsymptoteAngles))
	}
	expectedAngles := []float64{math.Pi / 3, math.Pi, 5 * math.Pi / 3}
	for i, ea := range expectedAngles {
		if math.Abs(res.AsymptoteAngles[i]-ea) > 0.01 {
			t.Errorf("asymptote[%d] expected %f, got %f", i, ea, res.AsymptoteAngles[i])
		}
	}

	// centroid at -1
	if c, ok := res.AsymptoteCentroid(); !ok || math.Abs(c-(-1)) > 0.01 {
		t.Errorf("centroid = %f, %v, want -1, true", c, ok)
	}
}

func TestRootLocus_CustomGains(t *testing.T) {
	// G(s) = 1/s
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	gains := []float64{0, 1, 5, 10}
	res, err := RootLocus(sys, gains)
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Gains) != 4 {
		t.Fatalf("expected 4 gains, got %d", len(res.Gains))
	}

	// For 1/s: closed-loop pole at -K
	for gi, K := range gains {
		p := res.Branches[0][gi]
		expected := complex(-K, 0)
		if cmplx.Abs(p-expected) > 0.01 {
			t.Errorf("K=%f: expected pole at %v, got %v", K, expected, p)
		}
	}
}

func TestRootLocus_NotSISO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = RootLocus(sys, nil)
	if !errors.Is(err, ErrNotSISO) {
		t.Errorf("expected ErrNotSISO, got %v", err)
	}
}

func TestRootLocus_NonZeroD(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0.5}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{2})
	if err != nil {
		t.Fatal(err)
	}
	// (s²+3s+2)(1+K/2) + K = 0 at K=2 -> s² + 3s + 3 = 0.
	want := []complex128{complex(-1.5, math.Sqrt(3)/2), complex(-1.5, -math.Sqrt(3)/2)}
	assertRootSet(t, []complex128{res.Branches[0][0], res.Branches[1][0]}, want, 1e-9)
	if len(res.AsymptoteAngles) != 0 {
		t.Errorf("AsymptoteAngles = %v, want none", res.AsymptoteAngles)
	}
	if _, ok := res.AsymptoteCentroid(); ok {
		t.Error("AsymptoteCentroid ok = true with no asymptotes")
	}

	sys.D.Set(0, 0, -0.5)
	if _, err := RootLocus(sys, []float64{2}); !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("1+KD=0 error = %v, want ErrAlgebraicLoop", err)
	}
}

func TestRootLocus_Descriptor(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	for k, want := range []complex128{-0.5, -1} {
		if got := res.Branches[0][k]; cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("K=%g pole = %v, want %v", res.Gains[k], got, want)
		}
	}

	sys2, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(2, 2, []float64{2, 0, 1, 1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err = RootLocus(sys2, []float64{0, 10})
	if err != nil {
		t.Fatal(err)
	}
	// det(sE - (A-KBC)) = 2s² + 7s + 2 + K.
	for k, K := range res.Gains {
		disc := complex(49-8*(2+K), 0)
		sq := cmplx.Sqrt(disc)
		want := []complex128{(-7 + sq) / 4, (-7 - sq) / 4}
		assertRootSet(t, []complex128{res.Branches[0][k], res.Branches[1][k]}, want, 1e-9)
	}
}

func TestRootLocus_Delays(t *testing.T) {
	cont, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	cont.InputDelay = []float64{1}
	if _, err := RootLocus(cont, nil); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("continuous delay error = %v, want ErrDelayUnsupported", err)
	}

	disc, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	disc.InputDelay = []float64{2}
	K := 0.1
	res, err := RootLocus(disc, []float64{K})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Branches) != 3 {
		t.Fatalf("branches = %d, want 3 (1 state + 2 delay samples)", len(res.Branches))
	}
	// z²(z-0.5) + K = 0.
	for b := range res.Branches {
		z := res.Branches[b][0]
		if r := z*z*z - 0.5*z*z + complex(K, 0); cmplx.Abs(r) > 1e-12 {
			t.Errorf("branch %d pole %v residual %g", b, z, cmplx.Abs(r))
		}
	}
}

func TestRootLocus_InvalidInput(t *testing.T) {
	siso := func() *System {
		s, err := New(
			mat.NewDense(1, 1, []float64{-1}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{0}),
			0,
		)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	nanA := siso()
	nanA.A.Set(0, 0, math.NaN())
	cases := []struct {
		name  string
		sys   *System
		gains []float64
		want  error
	}{
		{"nil", nil, nil, ErrInvalidArgument},
		{"static gain", gain, nil, ErrDimensionMismatch},
		{"empty gains", siso(), []float64{}, ErrInvalidArgument},
		{"NaN gain", siso(), []float64{0, math.NaN()}, ErrInvalidArgument},
		{"Inf gain", siso(), []float64{math.Inf(1)}, ErrInvalidArgument},
		{"NaN A", nanA, nil, ErrInvalidArgument},
	}
	for _, tc := range cases {
		if _, err := RootLocus(tc.sys, tc.gains); !errors.Is(err, tc.want) {
			t.Errorf("%s: error = %v, want %v", tc.name, err, tc.want)
		}
	}
}

func assertRootSet(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()
	used := make([]bool, len(got))
	for _, w := range want {
		found := false
		for i, g := range got {
			if !used[i] && cmplx.Abs(g-w) <= tol {
				used[i] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("roots %v missing %v", got, w)
		}
	}
}

func TestRootLocus_BranchContinuity(t *testing.T) {
	// G(s) = 1/(s^2+2s+1) => two branches
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	gains := make([]float64, 100)
	for i := range gains {
		gains[i] = float64(i) * 0.1
	}

	res, err := RootLocus(sys, gains)
	if err != nil {
		t.Fatal(err)
	}

	for b := 0; b < len(res.Branches); b++ {
		for i := 1; i < len(gains); i++ {
			d := cmplx.Abs(res.Branches[b][i] - res.Branches[b][i-1])
			if d > 1.0 {
				t.Errorf("branch %d: jump of %f between gain[%d]=%f and gain[%d]=%f",
					b, d, i-1, gains[i-1], i, gains[i])
			}
		}
	}
}
