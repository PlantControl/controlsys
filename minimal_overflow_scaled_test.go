package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// overflowScaledKalman is a discrete Kalman-decomposition model with B and
// C scaled by 2^≈430 and 2^≈590: |G| ≈ 2^1020 on the unit circle, so the
// band's responses and bounds leave the double range at some probes
// (census row kalman-d4/18/3.1.2, ovf scaling; QMVNHR).
func overflowScaledKalman(t *testing.T) *System {
	t.Helper()
	a := []float64{
		-0x1.2c5c515d81f12p-04, 0x1.f9c35806e750dp+07, -0x1.5794401200526p-07, -0x1.58dca3366ddfdp-10, -0x1.12a992910ec0fp-02, -0x1.1c7affb066b6ap-07,
		-0x1.3938741c02332p-10, -0x1.298f3a62dc418p-02, -0x1.b6286b0a6e7cp-16, 0x1.4e2adfa1e0d47p-22, 0x1.96ae276d1786bp-14, 0x1.7ad214c7c2158p-15,
		-0x1.acc4f406cf1b5p+04, 0x1.4337455bafb68p+09, 0x1.68c18ec114584p-02, 0x1.6607b19f0c658p-05, 0x1.1802d55587e0bp+03, 0x1.9898b7971e30fp+00,
		-0x1.0cb43e34a4a57p+11, -0x1.c9b90b79e8602p+15, -0x1.3111ca7cda31ap+04, 0x1.d9db8a3a0cfc9p-01, 0x1.9902a21786b3cp+08, -0x1.e0bf5ade2b98ap+01,
		-0x1.041658a6be75ap+03, 0x1.30799a0851c11p+09, -0x1.e8c00c9137f79p-04, 0x1.00f8889d992d3p-08, 0x1.07e2ef168f97ap-02, -0x1.851121e9723c3p-03,
		-0x1.6a1f03f3b83c5p+05, -0x1.4c0573284f149p+10, -0x1.008ec5eed2301p-02, 0x1.73cc845b785ap-07, 0x1.dd33286e56b33p+00, -0x1.6c9a4e3150972p-04,
	}
	b := []float64{0x1.23a54fb1e2df6p+430, 0x1.95278a182c958p+421, -0x1.b5bda10648165p+435, 0x1.f0c2796a4b903p+439, -0x1.b495be8cae74ap+431, -0x1.40ba31aaf0e01p+433}
	c := []float64{-0x1.fd470721f55fep+593, -0x1.64ce4fd6f41f7p+600, -0x1.13375bb18c15fp+589, 0x1.0ffefd7ffc4c1p+585, -0x1.4886c300b65f3p+592, -0x1.774ffac48d848p+588}
	sys, err := New(mat.NewDense(6, 6, a), mat.NewDense(6, 1, b), mat.NewDense(1, 6, c), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Reduce judges an overflow-scaled model in exactly rescaled units: it
// keeps the cut of the same model with B and C scaled back into range,
// rather than losing it to the non-finite probes, which count as
// undetermined (QMVNHR).
func TestReduceOverflowScaledIOKeepsCut(t *testing.T) {
	sys := overflowScaledKalman(t)
	down := func(m *mat.Dense, e int) *mat.Dense {
		out := mat.DenseCopyOf(m)
		out.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, -e) }, out)
		return out
	}
	ref, err := New(sys.A, down(sys.B, 430), down(sys.C, 590), sys.D, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	for _, opts := range []*ReduceOpts{nil, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}} {
		want, err := ref.Reduce(opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.Reduce(opts)
		if err != nil {
			t.Fatal(err)
		}
		if want.Order >= 6 {
			t.Fatalf("opts %+v: in-range model not reduced (order %d)", opts, want.Order)
		}
		if got.Order != want.Order {
			t.Errorf("opts %+v: overflow-scaled order %d, in-range %d", opts, got.Order, want.Order)
		}
	}
}
