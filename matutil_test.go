package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func assertMatClose(t *testing.T, name string, got, want *mat.Dense, tol float64) {
	t.Helper()
	r1, c1 := got.Dims()
	r2, c2 := want.Dims()
	if r1 != r2 || c1 != c2 {
		t.Fatalf("%s: dim mismatch got %dx%d want %dx%d", name, r1, c1, r2, c2)
	}
	for i := range r1 {
		for j := range c1 {
			if diff := math.Abs(got.At(i, j) - want.At(i, j)); diff > tol {
				t.Errorf("%s[%d,%d] = %v, want %v (diff %v)", name, i, j, got.At(i, j), want.At(i, j), diff)
			}
		}
	}
}

func TestDenseCopyTo(t *testing.T) {
	src := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	dst := mat.NewDense(2, 3, nil)

	denseCopyTo(dst, src)
	assertMatClose(t, "copy", dst, src, 0)

	src.Set(0, 0, 99)
	if dst.At(0, 0) == 99 {
		t.Fatal("dst should not alias src backing data")
	}

	denseCopyTo(dst, nil)
	assertMatClose(t, "nil-zero", dst, mat.NewDense(2, 3, nil), 0)
}

func matEqual(a, b *mat.Dense, tol float64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	if ra != rb || ca != cb {
		return false
	}
	for i := range ra {
		for j := range ca {
			if math.Abs(a.At(i, j)-b.At(i, j)) > tol {
				return false
			}
		}
	}
	return true
}

func TestExtractBlockZeroDim(t *testing.T) {
	M := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	blk := extractBlock(M, 0, 0, 0, 0)
	r, c := blk.Dims()
	if r != 0 || c != 0 {
		t.Fatalf("extractBlock(0,0,0,0) dims = %dx%d, want 0x0", r, c)
	}

	blk = extractBlock(M, 1, 1, 2, 2)
	want := mat.NewDense(2, 2, []float64{5, 6, 8, 9})
	assertMatClose(t, "2x2 block", blk, want, 0)
}

func TestLUNearSingular(t *testing.T) {
	wellCond := mat.NewDense(2, 2, []float64{
		2, 0,
		0, 3,
	})
	var lu mat.LU
	lu.Factorize(wellCond)
	if luNearSingular(&lu) {
		t.Fatal("expected well-conditioned matrix to be accepted")
	}

	singular := mat.NewDense(2, 2, []float64{
		1, 2,
		2, 4,
	})
	lu.Factorize(singular)
	if !luNearSingular(&lu) {
		t.Fatal("expected singular matrix to be rejected")
	}
}

// TestLUNearSingularThreshold exercises the cond*eps >= 1 boundary with
// diagonal matrices whose condition numbers are deterministically far from
// the threshold on each side, making the test robust against LAPACK estimator
// variance.
func TestLUNearSingularThreshold(t *testing.T) {
	machEps := eps()
	// threshold = 1/eps ≈ 4.5e15

	cases := []struct {
		name     string
		small    float64 // off-diagonal scale; cond ≈ 1/small
		wantNear bool
	}{
		// cond ≈ 1e2, cond*eps ≈ 2e-14 — clearly well-conditioned
		{"well_conditioned", 1e-2, false},
		// cond ≈ 1e12, cond*eps ≈ 2e-4 — high cond but 4 orders below threshold
		{"high_cond_below_threshold", 1e-12, false},
		// cond ≈ 1e17, cond*eps ≈ 20 — 2 orders above threshold
		{"ill_conditioned_above_threshold", 1e-17, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := mat.NewDense(2, 2, []float64{
				1, 0,
				0, tc.small,
			})
			var lu mat.LU
			lu.Factorize(m)
			cond := lu.Cond()
			got := luNearSingular(&lu)
			if got != tc.wantNear {
				t.Errorf("luNearSingular=%v want=%v (cond=%.3e, cond*eps=%.3e, threshold=%.3e)",
					got, tc.wantNear, cond, cond*machEps, 1/machEps)
			}
		})
	}
}
