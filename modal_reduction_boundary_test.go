package controlsys

import (
	"errors"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func modalTestRows(t *testing.T, a, b, c, d []float64, m, p int, dt float64) *System {
	t.Helper()
	n := len(b) / m
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Census rows (GJ7WMQ, IURGLT) with a defective double on the stability
// boundary, split by rounding to either side of it: the double counts as
// unstable whichever way it splits, so it is never dropped.
func TestModalTruncateKeepsSplitDoubleOnStabilityBoundary(t *testing.T) {
	t.Run("discrete double at z=1", func(t *testing.T) {
		sys := modalTestRows(t, []float64{
			0.18688255781258878, -0.34038578228664146, 0.6413208313993699, 0.2970125965991074, 0.17610221555517003, -0.37104837439621136, -0.3276532352775292,
			-0.4607323335189294, -0.23826248714959652, 0.5248904583415813, -0.37990577992952024, -1.030767759081846, -0.21803969467192416, 0.10054894121937841,
			0.17426030115413757, 0.24927540402782794, -0.33517154471747046, -0.27950655339921576, -0.21023226478753693, -0.6060415942087083, -0.2604936981492648,
			-0.3176159645313922, 0.24198599034577808, -0.3617212578669971, -0.647483415894785, 0.18642973250027745, 0.11345209404470308, -0.7852608455913901,
			0.5044446674475827, -0.42150420895420876, -0.01336316327291218, -0.5322131874620039, -0.3687251079573724, 0.41575451675222874, -0.5821212238286957,
			-0.29587033161578913, -0.9377408466077032, -0.9837586166018988, 0.028598923835987604, -0.39439233445733723, -0.3426240732114672, -0.20526188036450058,
			-0.4688889625359904, 0.5517450101115301, 0.5016081327583802, 0.4724977662394129, 0.08909451933880362, -0.02701165631693371, -0.7546159288818974,
		}, []float64{0.5, 2, 1, -1, 2, 0, -1}, []float64{0.5, -1, -1, 0, 0, -1, 0.5}, []float64{-1}, 1, 1, 0.5)
		if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 4}); !errors.Is(err, ErrInvalidOrder) {
			t.Fatalf("Order 4 err = %v, want ErrInvalidOrder (unstable double at z=1)", err)
		}
		assertModalTruncateKeeps(t, sys, 6, 6, []complex128{-1 + 1i, -1 - 1i, -1 + 0.5i, -1 - 0.5i, 1, 1}, 1e-6)
	})
	t.Run("discrete double at z=-1", func(t *testing.T) {
		sys := modalTestRows(t, []float64{
			-0.062499999999999986, 0.062499999999999986, 0.31249999999999994, -0.31249999999999994, -0.0625, 0.06249999999999999, -0.31249999999999994, -0.6874999999999999,
			-0.4374999999999999, -0.18749999999999997, -0.1875, -0.18749999999999997, -0.4374999999999999, 0.8124999999999999, -0.0625, -0.31249999999999994,
			0.25, -0.8749999999999999, -0.125, -0.24999999999999992, -0.49999999999999994, 0.12499999999999997, 0.49999999999999994, -0.125,
			-0.7499999999999998, 0.24999999999999997, 0.49999999999999994, 2.770348727867446e-17, -0.4999999999999998, -0.24999999999999997, -0.12499999999999997, -0.12499999999999997,
			0.37499999999999994, 0.25, -0.37499999999999994, -1.247959135964212e-17, -1.3981964481723364e-18, -0.6249999999999999, -0.12499999999999992, -0.4999999999999999,
			1.247959135964212e-17, 1.3981964481723364e-18, -0.37499999999999994, -0.37499999999999994, -0.37499999999999994, 0.12499999999999997, -0.37499999999999994, 0.37499999999999994,
			0.06249999999999999, -0.5624999999999999, -0.18749999999999994, -0.0625, 0.18749999999999997, -0.6874999999999999, 0.0625, 0.18749999999999994,
			0.06250000000000006, -0.43749999999999994, -0.06249999999999997, 0.6874999999999999, -0.8124999999999999, -0.06249999999999992, -0.06249999999999997, -0.31249999999999994,
		}, []float64{0.5, 0.5, 2, 0.5, 0.5, 0, 2, 0.5}, []float64{0.5, 2, 1, 2, 0.5, -1, 1, 1, 1, 0.5, -1, 0, -1, 2, 0, -1}, []float64{0, 0}, 1, 2, 0.5)
		if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 4}); !errors.Is(err, ErrInvalidOrder) {
			t.Fatalf("Order 4 err = %v, want ErrInvalidOrder (unstable double at z=-1)", err)
		}
		assertModalTruncateKeeps(t, sys, 5, 6, []complex128{1 + 0.25i, 1 - 0.25i, -0.25 + 1i, -0.25 - 1i, -1, -1}, 1e-6)
	})
}

// A census row (IURGLT) whose close stable pair near the imaginary axis
// rounds to a split with one copy at Re > 0: the pair stays stable, so the
// slower stable modes are kept ahead of it.
func TestModalTruncateKeepsCloseStablePairNearAxisStable(t *testing.T) {
	sys := modalTestRows(t, []float64{
		-2.9802322387695312e-08, -2, -0.5, 0, 0, 0, 0,
		2, -2.9802322387695312e-08, -1, 0, 0, 1, 0,
		0, 0, -1.8189894035458565e-12, 0, 0, -0.5, 0,
		0, 1, 0, -1.1920928955078125e-07, -1, 0, -0.5,
		0.5, 0, -0.5, 0, -9.313225746154785e-10, -1, 0,
		0, 0, 0.5, 0, 0, -1.8189894035458565e-12, 0,
		0, -1, -0.5, 0.5, 0, -1, -1.1920928955078125e-07,
	}, []float64{1, 0, 0.5, 2, 0, 1, -1}, []float64{-1, 2, 1, 1, -1, 2, 0.5}, []float64{0}, 1, 1, 0)
	assertModalTruncateKeeps(t, sys, 3, 3, []complex128{-0x1p-30, complex(-0x1p-25, 2), complex(-0x1p-25, -2)}, 1e-12)
}

// Exact integrators rotated by random orthogonal Q: a simple pole at 0 is
// kept by Order 1, and a double (Jordan) integrator is kept whole, whichever
// sign rounding gives the computed poles.
func TestModalTruncateKeepsIntegratorsWhateverRounding(t *testing.T) {
	for _, tc := range []struct {
		name        string
		t           []float64
		order, kept int
		want        []complex128
		tol         float64
	}{
		{"simple", []float64{0, 0.4, -0.3, 0.2, 0, -1, 0.5, 0.1, 0, 0, -2, 0.7, 0, 0, 0, -3}, 1, 1, []complex128{0}, 1e-12},
		{"double", []float64{0, 1, -0.3, 0.2, 0, 0, 0.5, 0.1, 0, 0, -2, 0.7, 0, 0, 0, -3}, 1, 2, []complex128{0, 0}, 1e-6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for seed := range uint64(16) {
				q := randomOrthogonal(rand.New(rand.NewPCG(seed, 13)), 4)
				var tmp, a mat.Dense
				tmp.Mul(q, mat.NewDense(4, 4, tc.t))
				a.Mul(&tmp, q.T())
				sys, err := New(&a, mat.NewDense(4, 1, []float64{1, -0.5, 2, 1}), mat.NewDense(1, 4, []float64{0.5, 1, -1, 2}), mat.NewDense(1, 1, nil), 0)
				if err != nil {
					t.Fatal(err)
				}
				assertModalTruncateKeeps(t, sys, tc.order, tc.kept, tc.want, tc.tol)
			}
		})
	}
}
