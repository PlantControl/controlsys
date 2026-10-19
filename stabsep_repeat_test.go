package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// A moved 2x2 block with real eigenvalues is split by Dtrexc's
// standardization; both halves must be placed.
func TestOrderSchurGroupFirstSplitBlock(t *testing.T) {
	const n = 3
	a := []float64{
		1, 0.3, -0.7,
		0, -9.5, 1,
		0, 0.25, -9.5,
	}
	tm := append([]float64(nil), a...)
	z := make([]float64, n*n)
	for i := range n {
		z[i*n+i] = 1
	}
	got, err := orderSchurGroupFirst(tm, z, n, func(e complex128) bool { return real(e) < 0 })
	if err != nil {
		t.Fatal(err)
	}
	if got != 2 {
		t.Fatalf("placed %d, want 2; T=%v", got, tm)
	}
	ev := schurEigenvaluesRaw(tm, n)
	for i, e := range ev {
		want := real(e) < 0
		if (i < got) != want {
			t.Fatalf("eigenvalue %d = %v on wrong side of %d; T=%v", i, e, got, tm)
		}
	}
	if math.Abs(real(ev[2])-1) > 1e-12 {
		t.Fatalf("trailing eigenvalue %v, want 1", ev[2])
	}
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				for l := range n {
					s += z[k*n+i] * a[k*n+l] * z[l*n+j]
				}
			}
			if math.Abs(s-tm[i*n+j]) > 1e-12 {
				t.Fatalf("ZᵀAZ[%d,%d] = %v, T = %v", i, j, s, tm[i*n+j])
			}
		}
	}
}

// Double stable poles that Dgees returns as a near-real 2x2 block must both
// land in Stable.
func TestStabsepRepeatedPoleSchurBlock(t *testing.T) {
	cases := []struct {
		kind uint8
		v    [fzNVals]float64
	}{
		{0xb2, [fzNVals]float64{0, -1.5, 1, 2, -4, -3, 0, 2, -17, -1, 44.9, 0, -10, -10, -10, -47}},
		{0xb2, [fzNVals]float64{0, -1.5, 1, 2, -4, -3, 0, 2, 0, -1, 9, 0, -50, 0, -10, -10}},
		{0xcb, [fzNVals]float64{0.6, -2, 37.7, 3.5999999999999996, 0.2, 1, 0, 82, -0.06989700043360188, 38.2, 1.7, 0, -10, -10, 76, -50}},
	}
	for _, c := range cases {
		s := fzDecode(t, c.kind, c.v)
		if !s.Decidable() {
			t.Fatalf("not decidable\n%s", s)
		}
		sys := s.System(t, fzSysOpts{})
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("Stabsep: %v\n%s", err, s)
		}
		nu := s.Unstable()
		ns, _, _ := res.Stable.Dims()
		nun, _, _ := res.Unstable.Dims()
		if ns != s.N-nu || nun != nu {
			t.Fatalf("orders stable=%d unstable=%d, want %d/%d\n%s", ns, nun, s.N-nu, nu, s)
		}
		sp, _ := res.Stable.Poles()
		for _, p := range sp {
			if modalPoleUnstable(p, s.Dt) {
				t.Fatalf("Stable pole %v\n%s", p, s)
			}
		}
		up, _ := res.Unstable.Poles()
		for _, p := range up {
			if !modalPoleUnstable(p, s.Dt) {
				t.Fatalf("Unstable pole %v\n%s", p, s)
			}
		}
		for _, w := range []float64{0.03, 0.7, 2.5} {
			pt := complex(0, w)
			if s.Discrete {
				pt = cmplx.Exp(complex(0, w))
			}
			g, err := sys.EvalFr(pt)
			if err != nil {
				t.Fatal(err)
			}
			g1, _ := res.Stable.EvalFr(pt)
			g2, _ := res.Unstable.EvalFr(pt)
			for i := range g {
				for j := range g[i] {
					d := cmplx.Abs(g1[i][j] + g2[i][j] - g[i][j])
					if d > 1e-8*(1+cmplx.Abs(g[i][j])) {
						t.Fatalf("Stable+Unstable at %v [%d,%d] off by %g\n%s", pt, i, j, d, s)
					}
				}
			}
		}
	}
}
