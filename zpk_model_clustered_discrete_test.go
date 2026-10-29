package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkClusteredDiscreteMatched is the matched C2D (dt 1e-3) of an 8-pole,
// 2-zero continuous ZPK (clustered generator seed 688): every discrete pole
// lies within 2.1e-3 of z = 1 and five zeros sit at z = −1 (HP3RZO).
func zpkClusteredDiscreteMatched(t *testing.T) *System {
	t.Helper()
	A := []float64{
		0.9994408400786775, 1.99859903389979, 1.9984648788466934, 1, -0, 0, 0, 0,
		0, 0.9985990338997901, 1.9984648788466934, 1, -0, 0, 0, 0,
		0, 0, 0.9984648788466934, 1, -0, 0, 0, 0,
		0, 0, 0, 0.9983765657776308, -0.5924299350651948, 1.9978357456324733, 3257.175532775565, 3.9991107369635106,
		0, 0, 0, 0, 0.9979297400924433, 0.0003171597283494937, 0, 0,
		0, 0, 0, 0, -0.0003171597283494937, 0.9979297400924433, 3257.175532775565, 3.9991107369635106,
		0, 0, 0, 0, 0, 0, 0.9995553684817553, 0.0012275114204339129,
		0, 0, 0, 0, 0, 0, -0.0012275114204339129, 0.9995553684817553,
	}
	B := []float64{0, 0, 0, 1, 0, 1, 0, 1}
	C := []float64{-1.0911483762999689e-23, 6.817678665203774e-20, 6.81722103161754e-20, 3.411228840584746e-20, -0, 0, 0, 0}
	sys, err := New(mat.NewDense(8, 8, A), mat.NewDense(8, 1, B), mat.NewDense(1, 8, C), mat.NewDense(1, 1, nil), 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestZPKModelClusteredDiscretePolesStayStable checks ZPKModel of a stable
// discrete model whose poles cluster at z = 1 against a 320-bit oracle of
// the stored matrices: roots of the transfer-function polynomials put four
// poles outside the unit circle and the zeros O(1) off (HP3RZO), while
// MATLAB zpk(ss) takes poles from eig(A) and zeros from tzero.
func TestZPKModelClusteredDiscretePolesStayStable(t *testing.T) {
	sys := zpkClusteredDiscreteMatched(t)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 8 || len(res.RowDegrees) != 1 || res.RowDegrees[0] != 8 {
		t.Fatalf("MinimalOrder %d RowDegrees %v, want 8 [8]", res.MinimalOrder, res.RowDegrees)
	}
	z := res.ZPK
	poles, zeros, gain := z.Poles[0][0], z.Zeros[0][0], z.Gain[0][0]
	if len(poles) != 8 || len(zeros) != 7 {
		t.Fatalf("got %d poles and %d zeros, want 8 and 7", len(poles), len(zeros))
	}
	for _, p := range poles {
		if !(cmplx.Abs(p) < 1) {
			t.Errorf("pole %v outside the unit circle", p)
		}
	}

	o := newStepOraclePencil(sys)
	starts, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, starts)
	if !ok {
		t.Fatal("oracle poles unresolved")
	}
	for _, want := range modes.poles {
		if d := nearestRootDistance(poles, want); !(d <= 1e-12) {
			t.Errorf("no pole near oracle %v: nearest %.3g away", want, d)
		}
	}

	pencil, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	var simple []complex128
	for _, q := range pencil {
		if d := nearestRootDistance(zeros, q); !(d <= 1e-9*cmplx.Abs(q)) {
			t.Errorf("System.Zeros %v: nearest ZPK zero %.3g away", q, d)
		}
		if cmplx.Abs(q+1) > 0.1 {
			simple = append(simple, q)
		}
	}
	if len(simple) != 2 {
		t.Fatalf("System.Zeros %v: want 2 zeros away from the five-fold zero at -1", pencil)
	}
	for _, want := range matchedOracleZeros(t, o, simple) {
		if d := nearestRootDistance(zeros, want); !(d <= 1e-9) {
			t.Errorf("no zero near oracle %v: nearest %.3g away", want, d)
		}
	}

	// Relative degree one: the gain is C·B, here fitted at low frequency to
	// within the tzero spread of the five-fold zero at −1.
	markov := 3.411228840584746e-20
	if rel := math.Abs(gain-markov) / markov; !(rel <= 1e-8) {
		t.Errorf("gain %v, want C·B = %v (rel %.3g)", gain, markov, rel)
	}

	for _, s := range []complex128{cmplx.Exp(1e-3i), cmplx.Exp(1e-2i), cmplx.Exp(0.1i), cmplx.Exp(1i), cmplx.Exp(2i), 0.999 + 0.001i} {
		got, err := z.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want := matchedOracleTransfer(o, s)
		if rel := cmplx.Abs(got[0][0]-want) / cmplx.Abs(want); !(rel <= 1e-6) {
			t.Errorf("H(%.6g) = %v, oracle %v (rel %.3g)", s, got[0][0], want, rel)
		}
	}
}

func nearestRootDistance(roots []complex128, x complex128) float64 {
	best := math.Inf(1)
	for _, r := range roots {
		best = math.Min(best, cmplx.Abs(r-x))
	}
	return best
}
