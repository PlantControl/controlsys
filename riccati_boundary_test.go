package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// exactBoundaryUncontrollable returns integer A = P·J·P⁻¹, B = P·B̃ and
// Q = MMᵀ + I, P unimodular, J lower block triangular with a leading
// boundary block Jb (0, 1, [0 ω; −ω 0] or a 2×2 Jordan block) whose rows of B̃ are zero: Jb's
// modes are exactly uncontrollable (left eigenvector [w; 0] of J), so no
// stabilizing solution exists. The trailing 3×3 block is random and may
// share Jb's modes (a Jordan block across the two parts).
func exactBoundaryUncontrollable(rng *rand.Rand, jb [][]int64) (A, B, Q *mat.Dense) {
	const ns = 3
	nb := len(jb)
	n := nb + ns
	J := make([][]int64, n)
	for i := range J {
		J[i] = make([]int64, n)
	}
	for i := range nb {
		copy(J[i], jb[i])
	}
	bt := make([]int64, n)
	for i := nb; i < n; i++ {
		for k := range n {
			if k < nb {
				J[i][k] = int64(rng.IntN(5) - 2)
			} else {
				J[i][k] = int64(rng.IntN(7) - 3)
			}
		}
		if bt[i] = int64(rng.IntN(7) - 3); bt[i] == 0 {
			bt[i] = 1
		}
	}
	P, Pi := make([][]int64, n), make([][]int64, n)
	for i := range n {
		P[i], Pi[i] = make([]int64, n), make([]int64, n)
		P[i][i], Pi[i][i] = 1, 1
	}
	for range 2 * n {
		a, b := rng.IntN(n), rng.IntN(n)
		if a == b {
			continue
		}
		c := int64(rng.IntN(5) - 2)
		for k := range n {
			P[a][k] += c * P[b][k]
			Pi[k][b] -= c * Pi[k][a]
		}
	}
	A, B, Q = mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(n, n, nil)
	M := make([][]int64, n)
	for i := range M {
		M[i] = make([]int64, n)
		for k := range n {
			M[i][k] = int64(rng.IntN(7) - 3)
		}
	}
	for r := range n {
		var bs int64
		for k := range n {
			bs += P[r][k] * bt[k]
		}
		B.Set(r, 0, float64(bs))
		for c := range n {
			var s, q int64
			for k := range n {
				q += M[r][k] * M[c][k]
				for l := range n {
					s += P[r][k] * J[k][l] * Pi[l][c]
				}
			}
			if r == c {
				q++
			}
			A.Set(r, c, float64(s))
			Q.Set(r, c, float64(q))
		}
	}
	return A, B, Q
}

// TestRiccati_ExactBoundaryUncontrollable (K72KOR review): (A, B) of exact
// integers with an exactly uncontrollable mode at 0 or ±iω (Care) or 1
// (Dare) in a dense realization has no stabilizing solution (MATLAB
// icare/idare Report 3). Its computed mode is off by eps·‖A‖·κ, which a
// band of n·eps·|λ| missed (Accuracy 1 X on about a quarter of these, its
// exact closed loop not stable); the componentwise eigenvalue error band
// and the Jordan cluster mean catch them. The two review repros are named.
// boundaryUnstabilizable is asserted directly too, unscaled and scaled:
// whether Care reaches it depends on rounding (zero1 returned Accuracy 1
// under -race on amd64 CI), and its verdict must not. A Jordan block at
// the boundary (MPWOJM) splits by √(eps·κ); its mean, off by eps·‖A‖ times
// the cluster condition, missed a band without that factor, and a pass
// with no Accuracy 1 X returned before the check (base: 60 of 800 Accuracy
// 1 X).
func TestRiccati_ExactBoundaryUncontrollable(t *testing.T) {
	type tcase struct {
		name       string
		continuous bool
		A, B, Q    *mat.Dense
		r          float64
	}
	cases := []tcase{
		{"care0", true,
			mat.NewDense(4, 4, []float64{-1, 2, -1, 2, -6, 8, -4, 8, -6, 9, -5, 8, 1, -1, 0, -2}),
			mat.NewDense(4, 1, []float64{2, 3, 4, -1}),
			mat.NewDense(4, 4, []float64{13, 4, -7, -6, 4, 6, -2, -3, -7, -2, 4, 3, -6, -3, 3, 5}), 1e-11},
		{"dare1", false,
			mat.NewDense(5, 5, []float64{3, 8, 18, -8, 4, 8, 12, 38, -16, -4, 3, 4, 14, -6, -2, 15, 23, 73, -31, -6, -1, 4, -2, -2, 2}),
			mat.NewDense(5, 1, []float64{4, 5, 1, 9, 1}),
			mat.NewDense(5, 5, []float64{14, 1, -3, 11, 4, 1, 9, 10, 4, 6, -3, 10, 17, 3, 0, 11, 4, 3, 14, 2, 4, 6, 0, 2, 13}), 1e-7},
	}
	kinds := []struct {
		name       string
		continuous bool
	}{{"zero", true}, {"one", false}, {"imag", true}, {"jordan0", true}, {"jordan1", false}}
	for kind, k := range kinds {
		for seed := range 50 {
			rng := rand.New(rand.NewPCG(uint64(seed), uint64(kind)))
			jb := [][]int64{{0}}
			switch kind {
			case 1:
				jb = [][]int64{{1}}
			case 2:
				w := int64(1 + rng.IntN(4))
				jb = [][]int64{{0, w}, {-w, 0}}
			case 3:
				jb = [][]int64{{0, 1}, {0, 0}}
			case 4:
				jb = [][]int64{{1, 1}, {0, 1}}
			}
			A, B, Q := exactBoundaryUncontrollable(rng, jb)
			cases = append(cases, tcase{fmt.Sprintf("%s%d", k.name, seed), k.continuous, A, B, Q, math.Pow(10, -12+12*rng.Float64())})
		}
	}
	for _, tc := range cases {
		for _, ns := range []bool{false, true} {
			R := mat.NewDense(1, 1, []float64{tc.r})
			opts := &RiccatiOpts{NoScaling: ns}
			var res *RiccatiResult
			var err error
			if tc.continuous {
				res, err = Care(tc.A, tc.B, tc.Q, R, opts)
			} else {
				res, err = Dare(tc.A, tc.B, tc.Q, R, opts)
			}
			if problem, perr := newRiccatiProblem(tc.A, tc.B, tc.Q, R, opts); perr != nil {
				t.Errorf("%s NoScaling %v: %v", tc.name, ns, perr)
			} else if !problem.boundaryUnstabilizable(tc.continuous) {
				t.Errorf("%s NoScaling %v: boundaryUnstabilizable false", tc.name, ns)
			} else if problem.scale(opts, true); !problem.boundaryUnstabilizable(tc.continuous) {
				t.Errorf("%s NoScaling %v: boundaryUnstabilizable false once scaled", tc.name, ns)
			}
			if err == nil {
				t.Errorf("%s NoScaling %v: Accuracy %d X, want ErrNoStabilizing", tc.name, ns, res.Accuracy)
			} else if !errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
				t.Errorf("%s NoScaling %v: %v", tc.name, ns, err)
			}
		}
	}
}

// TestCare_StableJordanUncontrollableMode (gonum v0.20.9 bump sweep): A =
// P·J·P⁻¹ with an uncontrollable mode at −1.2e-5 or −6.7e-7 in a Jordan
// block with a controllable one, cheap control. Correct Dggev eigenvectors
// put x ⊥ y there, so the first-order band τ (1.6e-5) swallowed the mode
// and Care returned ErrNoStabilizing; the computed pair's split, not τ,
// bounds it, and its mean is resolved as stable. X is within 1e-6 of the
// 256-bit Newton oracle at Accuracy 0, 1e-3 at Accuracy 1 (amd64: 2.8e-4).
func TestCare_StableJordanUncontrollableMode(t *testing.T) {
	cases := []struct {
		name    string
		a, b, q []float64
		r       float64
	}{
		{"delta=-1.2e-5",
			[]float64{-1.4453226759754887, 2.779361002952737, -2.5272690637142685, -1.456422025518266, -3.981815825286485, 4.839842889751445, -6.963577168239024, -2.3966507695570507, -1.5250997781990823, 2.178132112704622, -3.3469016433967123, -0.6822586821517838, -4.2409578277559055, 2.6881957904403624, -5.240518209479649, -2.488539344996856},
			[]float64{0.21300832558460095, -6.1748980397572275, -4.376807008123427, -3.5613792332682905},
			[]float64{3.7034555093118704, -0.8456430331679469, 1.3922707856469327, -2.2588290036849514, -0.8456430331679469, 2.207555796595078, -0.7000977282621735, 0.19949110621559002, 1.3922707856469327, -0.7000977282621735, 4.245015893090951, -1.273159328382354, -2.2588290036849514, 0.19949110621559002, -1.273159328382354, 3.5243518782749126},
			3.911826880790316e-12},
		{"delta=-6.7e-7",
			[]float64{-1.119439515984305, 0.5406937956029283, 0.890471322925628, 0.20116639948390957, 1.6979453898656534, -0.23517482400293632, -0.6022666161659568, -0.5996275654081964, 0.04017216223750807, -0.08310823387680955, -0.0980079908217437, 0.5747377816786886, -1.8926930615264648, 1.1476767230873344, 1.8061774070463725, 0.2890975492568851},
			[]float64{0.9143324096042801, 1.5564909335724222, -2.205031022108274, 2.4967841798835244},
			[]float64{3.7215406257608903, 2.3297165035539353, -3.412908189188079, 1.1675675327551236, 2.3297165035539353, 7.73480995686668, -0.2681220056133864, 2.12772689645334, -3.412908189188079, -0.2681220056133864, 7.046441196696745, -0.6235613597168616, 1.1675675327551236, 2.12772689645334, -0.6235613597168616, 2.092852213281822},
			5.377188308450666e-12},
	}
	for _, tc := range cases {
		for _, ns := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/NoScaling=%v", tc.name, ns), func(t *testing.T) {
				A, B, Q := mat.NewDense(4, 4, tc.a), mat.NewDense(4, 1, tc.b), mat.NewDense(4, 4, tc.q)
				res, err := Care(A, B, Q, mat.NewDense(1, 1, []float64{tc.r}), &RiccatiOpts{NoScaling: ns})
				if err != nil {
					t.Fatal(err)
				}
				Xo := bigRiccatiNewton(t, true, A, B, Q, tc.r, res.X)
				var d mat.Dense
				d.Sub(res.X, Xo)
				want := 1e-6
				if res.Accuracy == 1 {
					want = 1e-3
				}
				if rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2); !(rel <= want) {
					t.Errorf("Accuracy %d X off the oracle by %.3g relative", res.Accuracy, rel)
				}
			})
		}
	}
}
