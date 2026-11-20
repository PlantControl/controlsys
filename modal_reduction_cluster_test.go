package controlsys

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func sameComplexSet(got, want []complex128, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	used := make([]bool, len(got))
	for _, w := range want {
		found := false
		for i, g := range got {
			if !used[i] && cmplx.Abs(g-w) <= tol {
				used[i], found = true, true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func assertModalTruncateKeeps(t *testing.T, sys *System, order, wantOrder int, want []complex128, tol float64) {
	t.Helper()
	res, err := ModalTruncate(sys, &ModalTruncateOptions{Order: order})
	if err != nil {
		t.Fatalf("Order %d: %v", order, err)
	}
	if res.Order != wantOrder || !sameComplexSet(res.KeptPoles, want, tol) {
		t.Fatalf("Order %d: got order %d poles %v, want order %d poles %v", order, res.Order, res.KeptPoles, wantOrder, want)
	}
	n, _, _ := sys.Dims()
	if wantOrder == n {
		return
	}
	var identity mat.Dense
	identity.Mul(res.Projection, res.Basis)
	if !mat.EqualApprox(&identity, eyeDense(wantOrder), 1e-9) {
		t.Fatalf("Order %d: projection·basis =\n%v", order, mat.Formatted(&identity))
	}
	assertPolesMatch(t, "reduced", res.Sys, want, tol)
}

// Census rows (gonum-3407, IURGLT) whose Order fell between the copies of
// an exactly repeated complex pair: the cut solved or failed by ulp ties.
func TestModalTruncateRoundsOrderToRepeatedPairCluster(t *testing.T) {
	tests := []struct {
		name        string
		a, b, c, d  []float64
		m, p        int
		dt          float64
		order, kept int
		poles       []complex128
	}{
		{
			name: "discrete repeated pair behind unstable modes",
			a: []float64{
				0.5, 0.25, 0, -0.5, -0.25, -0.25, -0.5, -0.25,
				-0.25, 0.5, 0.5, 0, -0.25, -0.5, -0.5, 0.25,
				0, 0, 0.5, 0.25, 0.5, -0.25, 0.25, 0.5,
				0, 0, -0.25, 0.5, -0.5, 0.25, 0.25, 0,
				0, 0, 0, 0, 0.25, 0, -0.5, 0,
				0, 0, 0, 0, 0, -1, 0.25, -0.5,
				0, 0, 0, 0, 0, 0, 1, 0.25,
				0, 0, 0, 0, 0, 0, -0.25, 1,
			},
			b: []float64{1, 0.5, 2, 0, -1, 0.5, 1, 2}, c: []float64{2, 2, -1, 2, -1, 2, 0.5, 0.5, 0, 1, 2, -1, 2, -1, 2, -1},
			d: []float64{0, 0}, m: 1, p: 2, dt: 0.5, order: 4, kept: 7,
			poles: []complex128{1 + 0.25i, 1 - 0.25i, -1, 0.5 + 0.25i, 0.5 - 0.25i, 0.5 + 0.25i, 0.5 - 0.25i},
		},
		{
			name: "discrete repeated pair is the whole remainder",
			a: []float64{
				0.75, 0, 0.25, 0.125, 0.25, 0,
				0, 0.75, 0, 0.25, 0, -0.25,
				-0.25, 0.125, 0.75, 0.25, -0.25, 0,
				0, -0.25, 0, 0.75, 0, 0,
				0, 0, 0, 0, 0.875, -0.25,
				0, 0, 0, 0, 0.25, 0.875,
			},
			b: []float64{2, 0, 1, 0.5, 1, 2}, c: []float64{0.5, 0, 0.5, 0.5, -1, 0}, d: []float64{0},
			m: 1, p: 1, dt: 0.5, order: 4, kept: 6,
			poles: []complex128{0.875 + 0.25i, 0.875 - 0.25i, 0.75 + 0.25i, 0.75 - 0.25i, 0.75 + 0.25i, 0.75 - 0.25i},
		},
		{
			name: "continuous repeated pair after a real mode",
			a: []float64{
				-1, 1, 0.5, 0, -0.5, 1, -0.5,
				-1, -1, 0, 1, 0.5, -1, 1,
				0, 0, -1, 1, 1, 0.5, 0.5,
				0, 0, -1, -1, 0, 0, 0,
				0, 0, 0, 0, -2, 2, -0.5,
				0, 0, 0, 0, -2, -2, -0.5,
				0, 0, 0, 0, 0, 0, -1,
			},
			b: []float64{1, 0.5, 1, -1, 0, 0.5, -1}, c: []float64{1, 2, 0.5, 2, 0.5, 1, 0, -1, 1, 0.5, 0, -1, 0, 1},
			d: []float64{0.5, 0}, m: 1, p: 2, order: 3, kept: 5,
			poles: []complex128{-1, -1 + 1i, -1 - 1i, -1 + 1i, -1 - 1i},
		},
		{
			name: "continuous real mode tied with a repeated pair",
			a: []float64{
				-2, 0.5, -0.5, 0, -0.5, 0, 0, 0.5,
				-0.5, -2, 0.5, 1, 0, 0.5, 1, 1,
				0, 0, -2, 0.5, -0.5, -0.5, 0, 0,
				0, 0, -0.5, -2, -0.5, -0.5, 1, -0.5,
				0, 0, 0, 0, -2, 0, 0.5, 1,
				0, 0, 0, 0, 0, -0x1p-31, 0.5, 0.5,
				0, 0, 0, 0, 0, -0.5, -0x1p-31, 1,
				0, 0, 0, 0, 0, 0, 0, -0x1p-30,
			},
			b: []float64{0, 0, 0.5, 1, 0, 0, 2, 0.5}, c: []float64{0.5, 0, 0.5, 0, 0.5, 0, 0, 1}, d: []float64{0},
			m: 1, p: 1, order: 4, kept: 4,
			poles: []complex128{-0x1p-30, complex(-0x1p-31, 0.5), complex(-0x1p-31, -0.5), -2},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.b) / tc.m
			sys, err := New(mat.NewDense(n, n, tc.a), mat.NewDense(n, tc.m, tc.b), mat.NewDense(tc.p, n, tc.c), mat.NewDense(tc.p, tc.m, tc.d), tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			assertModalTruncateKeeps(t, sys, tc.order, tc.kept, tc.poles, 1e-12)
		})
	}
}

// An exactly repeated pair rotated by random orthogonal Q: whichever copy
// rounding makes dominant, an Order between the copies keeps both, for a
// semisimple pair (copies within rounding, one cluster) and a defective
// one (copies √ε apart, inseparable by Sylvester; an Order inside either
// copy splits a pair).
func TestModalTruncateKeepsExactRepeatedPairWhole(t *testing.T) {
	for _, tc := range []struct {
		name     string
		coupling float64
		tol      float64
		orders   []int
	}{
		{"semisimple", 0, 1e-12, []int{3, 4}},
		{"defective", 1, 1e-6, []int{3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			T := mat.NewDense(6, 6, []float64{
				-0.5, 0.3, -0.2, 0.1, 0.4, -0.3,
				0, -1, 2, tc.coupling, 0, 0.2,
				0, -2, -1, 0, tc.coupling, -0.1,
				0, 0, 0, -1, 2, 0.3,
				0, 0, 0, -2, -1, 0.5,
				0, 0, 0, 0, 0, -5,
			})
			want := []complex128{-0.5, -1 + 2i, -1 - 2i, -1 + 2i, -1 - 2i}
			for seed := range uint64(16) {
				q := randomOrthogonal(rand.New(rand.NewPCG(seed, 11)), 6)
				var tmp, a mat.Dense
				tmp.Mul(q, T)
				a.Mul(&tmp, q.T())
				sys, err := New(&a, mat.NewDense(6, 1, []float64{1, -0.5, 2, 1, 0.5, -1}),
					mat.NewDense(1, 6, []float64{0.5, 1, -1, 2, 0, 1}), mat.NewDense(1, 1, nil), 0)
				if err != nil {
					t.Fatal(err)
				}
				for _, order := range tc.orders {
					assertModalTruncateKeeps(t, sys, order, 5, want, tc.tol)
				}
			}
		})
	}
}
