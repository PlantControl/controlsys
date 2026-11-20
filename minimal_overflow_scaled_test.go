package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A rotated discrete model with a kept integrator at z = 1, B scaled by
// 2^577 and C by 2^447 (|G| ≈ 2^1024 on the unit circle; census row
// twin/1-true, edge scaling; QMVNHR). With the overflowing probes judged on
// a rescaled realization, the band granted the integrator an axis window
// from pole resolutions inflated by the unscaled B and C, and accepted
// under Equalize an order-2 cut that is wrong beside z = 1 by 7·10⁷·tol
// outside every window of the stored model. The band refuses a cut whose
// responses leave the double range (bandResponses).
func TestReduceOverflowScaledIORefusesCut(t *testing.T) {
	a := []float64{
		0x1.7dc9a3b6ad32p-02, -0x1.106f3fd78bb1ap-01, -0x1.218050e89cc2bp-01,
		-0x1.a3b6ad31ef90cp-02, -0x1.6708379febc5bp-03, 0x1.d5046cb892a58p-02,
		0x1.b2e24a9670835p-03, 0x1.e06522c3f35bap-02, 0x1.ce106f3fd78b9p-01,
	}
	b := []float64{0x1.5555555555556p+575, -0x1.5555555555555p+575, -0x1.5555555555555p+577}
	c := []float64{0x1.c71c71c71c72p+443, -0x1.238e38e38e38ep+448, 0x1.c71c71c71c71cp+444}
	sys, err := New(mat.NewDense(3, 3, a), mat.NewDense(3, 1, b), mat.NewDense(1, 3, c), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.Reduce(&ReduceOpts{Equalize: true})
	if err != nil {
		t.Fatal(err)
	}
	if got.Order != 3 {
		t.Fatalf("overflow-scaled model reduced to order %d, want 3 (unreduced)", got.Order)
	}
}

// Reduce under Equalize judged an underflow-scaled model (B·2^−531,
// C·2^−589: |G| ≈ 2^−1076, below the double range) by responses, bounds
// and terms that all underflow to exact zeros, so 0 ≤ tol·0 verified the
// order-0 cut of a model whose exact response is nonzero. The band judges
// it in exactly rescaled units (census row matched/8, unf scaling; QMVNHR).
func TestReduceUnderflowScaledIONoZeroCut(t *testing.T) {
	a := []float64{
		0x1.fd0b31135a628p-01, 0x1.fd2e6d8452546p+00, 0x1.fcaee57366e44p+00, -0x1.62ff9e02622p-10, 0x1p+00, 0, 0,
		0, 0x1.fa5cdb08a4a8bp-01, 0x1.fcaee57366e44p+00, -0x1.62ff9e02622p-10, 0x1p+00, 0, 0,
		0, 0, 0x1.f95dcae6cdc89p-01, -0x1.62ff9e02622p-10, 0x1p+00, 0, 0,
		0, 0, 0, 0x1.f7e8728e5a619p-01, 0x1p+00, 0, 0,
		0, 0, 0, 0, 0x1.f6c955edc9022p-01, 0x1.3689cf2f46abfp+08, 0x1.fb164ee06b725p+01,
		0, 0, 0, 0, 0, 0x1.f62c9dc0d6e4ap-01, 0x1.9e007c8ce8ea6p-07,
		0, 0, 0, 0, 0, -0x1.9e007c8ce8ea6p-07, 0x1.f62c9dc0d6e4ap-01,
	}
	b := []float64{0, 0, 0, 0, 0x1p-531, 0, 0x1p-531}
	c := []float64{0x1.f5af98510313dp-589, 0x1.f45e5dca9fd7dp-589, 0x1.f3e10ac98420cp-589, -0x1.5cdab40d7149p-599, 0x1.f7237328f0838p-590, 0, 0}
	sys, err := New(mat.NewDense(7, 7, a), mat.NewDense(7, 1, b), mat.NewDense(1, 7, c), mat.NewDense(1, 1, nil), 0.01)
	if err != nil {
		t.Fatal(err)
	}
	up := func(m *mat.Dense, e int) *mat.Dense {
		out := mat.DenseCopyOf(m)
		out.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, e) }, out)
		return out
	}
	ref, err := New(sys.A, up(sys.B, 531), up(sys.C, 589), sys.D, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	opts := &ReduceOpts{Equalize: true}
	want, err := ref.Reduce(opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.Reduce(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Order != want.Order {
		t.Fatalf("underflow-scaled order %d, in-range %d", got.Order, want.Order)
	}
}

// An underflow-scaled discrete model (B·2^−424, C·2^−593: |G| ≈ 2^−1015)
// whose terms fall below 2^−969 at every probe: the band judges it on B̂
// and Ĉ scaled up by exact powers of two, which tiny B̂ and Ĉ leave the
// band's pole resolutions unchanged for, and keeps the in-range model's
// correct cut under Equalize (census row axis-hid/2-unobs, unf scaling;
// QMVNHR ruling B′).
func TestReduceUnderflowScaledIOKeepsCut(t *testing.T) {
	a := []float64{
		0x1.0e577b9dfedep+00, 0x1.8e01a23cdf608p-02, -0x1.2345739318bd6p-04,
		-0x1.0f166e636c1e3p-02, -0x1.4a07a66a6fec6p-02, 0x1.aa59a605018dap-03,
		-0x1.19c58c849e1d2p+00, -0x1.622a83bfa97c5p-02, 0x1.2b8ac51a53086p-01,
	}
	b := []float64{0x1.416912ad0e8f8p-424, -0x1.55a8ccd139d74p-424, -0x1.a6720538b9d1ep-423}
	c := []float64{-0x1.d1b2ab292bc4dp-593, 0x1.cd610064c0719p-594, -0x1.0d47f647143b8p-593}
	sys, err := New(mat.NewDense(3, 3, a), mat.NewDense(3, 1, b), mat.NewDense(1, 3, c), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	up := func(m *mat.Dense, e int) *mat.Dense {
		out := mat.DenseCopyOf(m)
		out.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, e) }, out)
		return out
	}
	ref, err := New(sys.A, up(sys.B, 424), up(sys.C, 593), sys.D, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	opts := &ReduceOpts{Equalize: true}
	want, err := ref.Reduce(opts)
	if err != nil {
		t.Fatal(err)
	}
	if want.Order >= 3 {
		t.Fatalf("in-range model not reduced (order %d)", want.Order)
	}
	got, err := sys.Reduce(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Order != want.Order {
		t.Fatalf("underflow-scaled order %d, in-range %d", got.Order, want.Order)
	}
}
