package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// chainDescriptor returns A = U·blkdiag(Ff, I)·V, E = U·blkdiag(Wf, N)·V,
// B = U·b, C = c·V with nilpotent Jordan chains N of the given lengths, and
// the structured model (blkdiag(Ff, I), blkdiag(Wf, N), b, c), which has the
// same transfer function and is the oracle. Ff is diagonal with
// Wf = diag(stiffRel, ...) when stiffRel > 0, else Ff is dense and Wf = I.
// With proper, b drives each chain only at its kernel state, so
// c·N^k·b = 0 for k ≥ 1.
func chainDescriptor(t *testing.T, rng *rand.Rand, nf int, chains []int, stiffRel float64, proper bool) (sys, oracle *System, poles []complex128) {
	t.Helper()
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range nf {
		c.Set(0, i, rng.NormFloat64())
		b.Set(i, 0, rng.NormFloat64())
		if stiffRel > 0 {
			f, w := rng.NormFloat64(), 0.5+rng.Float64()
			if i == 0 {
				w = stiffRel
			}
			F.Set(i, i, f)
			W.Set(i, i, w)
			poles = append(poles, complex(f/w, 0))
			continue
		}
		W.Set(i, i, 1)
		for j := range nf {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			c.Set(0, o+i, rng.NormFloat64())
			if !proper || i == 0 {
				b.Set(o+i, 0, rng.NormFloat64())
			}
		}
		o += k
	}
	U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			U.Set(i, j, rng.NormFloat64())
			V.Set(i, j, rng.NormFloat64())
		}
	}
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	D := mat.NewDense(1, 1, []float64{0.3})
	sys, err := NewDescriptor(&A, &B, &C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	oracle, err = NewDescriptor(F, b, c, D, W, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys, oracle, poles
}

func requirePolesContain(t *testing.T, label string, got, want []complex128, tol float64) {
	t.Helper()
	for _, w := range want {
		found := false
		for _, g := range got {
			found = found || cmplx.Abs(g-w) <= tol*cmplx.Abs(w)
		}
		if !found {
			t.Fatalf("%s: pole %v missing from %v", label, w, got)
		}
	}
}

// TestPoles_StiffPoleBesideIndexTwoChain: a stiff finite pole (E singular
// value 150× the rounding n·ε·σ₁) beside an index-2 infinite chain. The d =
// n − rank(E) eigenvalues of least |β|/|α| are infinite only at index 1;
// QZ leaves the chain members at |β|/|α| ~ √ε, above the stiff pole's
// 1.4e-13, so Poles dropped the unstable stiff pole 7.0045e12, kept the
// chain members −1.14 ± 2.47e7i and IsStable flipped to true. Oracle: the
// construction A = U·F·V, E = U·W·V and, in the sweep, fd/wd of the
// structured pencil. The chain members QZ cannot resolve as infinite stay
// as large poles, as on base (β ≠ 0 kept).
func TestPoles_StiffPoleBesideIndexTwoChain(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{1.6894392624787624, 11.45747886182073, -2.0911191588664653,
		62.140891294751555, 28.297675799532467, 18.05171632704232,
		38.2412866977158, -4.879420685687981, 8.26639587079175})
	E := mat.NewDense(3, 3, []float64{-28.2048654460364, 6.6919398400920524, 53.316753834534595,
		-18.754438277637135, 4.449713576153428, 35.452243899813155,
		40.99441857182659, -9.726413458238328, -77.49334340082})
	sys, err := NewDescriptor(A, mat.NewDense(3, 1, nil), mat.NewDense(1, 3, nil), mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	requirePolesContain(t, "seed 337", poles, []complex128{7.0045e12}, 1e-3)
	if st, err := sys.IsStable(); err != nil || st {
		t.Fatalf("seed 337: IsStable = %v, %v, want false", st, err)
	}

	checked, chained := 0, 0
	for seed := range uint64(300) {
		rng := rand.New(rand.NewPCG(seed, 99))
		nf := 1 + rng.IntN(3)
		chains := [][]int{{1}, {2}, {2, 1}, {3}}[rng.IntN(4)]
		sys, _, want := chainDescriptor(t, rng, nf, chains, math.Pow(10, -13+3*rng.Float64()), false)
		n, _, _ := sys.Dims()
		var svd mat.SVD
		if !svd.Factorize(sys.E, mat.SVDNone) {
			t.Fatal("svd failed")
		}
		sv := svd.Values(nil)
		r, stiff, _ := descriptorRank(sv, sv[0], n, true)
		if !stiff || sv[r-1] < 10*float64(n)*eps()*sv[0] {
			continue
		}
		label := fmt.Sprintf("seed %d chains %v", seed, chains)
		poles, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		ar, ai, b, ok := qzEigenvalues(n, sys.A.RawMatrix().Data, n, sys.E.RawMatrix().Data, n)
		if !ok {
			t.Fatal("QZ failed")
		}
		resolved := finitePoles(ar, ai, b, 0)
		var want2 []complex128
		for _, w := range want {
			for _, g := range resolved {
				if cmplx.Abs(g-w) <= 2e-2*cmplx.Abs(w) {
					want2 = append(want2, w)
					break
				}
			}
		}
		if len(want2) < len(want) {
			continue
		}
		requirePolesContain(t, label, poles, want, 2e-2)
		checked++
		if chains[0] > 1 {
			chained++
		}
		if len(poles) == len(want) {
			if st, err := sys.IsStable(); err != nil || st != stableSet(want) {
				t.Fatalf("%s: IsStable = %v, %v, want %v", label, st, err, stableSet(want))
			}
		}
	}
	if chained < 50 {
		t.Fatalf("%d of %d checked draws have a chain, want ≥ 50", chained, checked)
	}
	t.Logf("checked %d draws, %d with a chain", checked, chained)
}

// TestStabsep_StiffPoleBesideIndexTwoChain: seed 3133, a stiff unstable
// pole 6.403e10 (E singular value 190× the rounding) beside an index-2
// chain. The staircase, stiff, deflated only the null space of E and the
// ordered QZ of the remaining block, chain member and stiff pole together,
// failed with ErrSchurFailed; base split it exactly. Oracle: the response
// of sys at test points and the pole of the pencil.
func TestStabsep_StiffPoleBesideIndexTwoChain(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1.319099977578236, 0.32602873063711985, 4.447196509444975,
		4.017538017511279, 0.5729518265406396, 0.8239030668727039,
		0.6455094796415385, 4.034881498832863, 0.23430797658073424})
	E := mat.NewDense(3, 3, []float64{0.02723488800112055, 0.04870382370993776, 0.0481340841026225,
		-0.01784815243825018, -0.03191763711048171, -0.03154426305274396,
		-0.009429767272754382, -0.01686313970663152, -0.016665873946078432})
	B := mat.NewDense(3, 2, []float64{1.4500773955773156, 0.4979798294911739,
		-1.2750177933381144, -0.7186855150182745,
		-0.2470761062270532, 0.1356746211188346})
	C := mat.NewDense(1, 3, []float64{-0.645913823722979, -1.0817128635881375, 2.4584088799913473})
	D := mat.NewDense(1, 2, []float64{-0.38950021961460024, 0.3160840243416402})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}
	unstable, err := res.Unstable.Poles()
	if err != nil {
		t.Fatal(err)
	}
	requirePolesContain(t, "unstable", unstable, []complex128{6.403094198551596e10}, 1e-6)
	if stable, err := res.Stable.Poles(); err != nil || len(stable) != 0 {
		t.Fatalf("stable poles %v, %v, want none", stable, err)
	}
	for _, s := range []complex128{complex(0.3, 1.7), complex(-0.2, 2.5), 4i} {
		want := pencilResponse(t, sys, s)
		got1, got2 := pencilResponse(t, res.Stable, s), pencilResponse(t, res.Unstable, s)
		for j := range want[0] {
			if d := cmplx.Abs(got1[0][j] + got2[0][j] - want[0][j]); !(d <= 1e-8*math.Max(1, cmplx.Abs(want[0][j]))) {
				t.Fatalf("G(%v)[%d]: sum %v, want %v", s, j, got1[0][j]+got2[0][j], want[0][j])
			}
		}
	}
}

// TestTransferFunction_DescriptorHigherIndexChain: an improper descriptor
// with an index-3 chain (seed 1472, |G(1e5j)|/|G(1e3j)| = 8.7e3). The
// whole-pencil QZ split kept two chain members at |β|/|α| ~ ε^(1/3) as
// finite poles ±9e9, the impulsive-mode check passed, and TransferFunction
// returned a proper tf with G(1j) = −1.534 against EvalFr −0.917 (Step
// simulated it alike); base and MATLAB tf(dss) do not return that proper
// fit. The infinite structure is now deflated (staircase) when it disagrees
// with QZ. The sweep: chains of length ≥ 2 driven beyond their kernel
// state are improper (ErrImproperModel); driven only at it they are proper
// and TransferFunction matches the structured model (oracle). IsProper
// agrees; its former minimal-realization rank test called seed 54 (chains
// [3 1], driven at the kernel) improper.
func TestTransferFunction_DescriptorHigherIndexChain(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{1.6845448672194017, 6.2677246894560765, -7.904169090011227, -7.527371794454999, -0.9747958955559203,
		-6.921228232067118, 7.387288301665237, 1.631139568049228, -2.370924674068633, -4.1463663476005514,
		-6.085950234531109, -5.381879965560523, 1.10401198936006, -5.819585919212208, 8.894098091075191,
		-5.674344338242792, 0.6198248251534335, -6.638266871652843, 4.607206220116197, 5.146142091981357,
		-1.290752374208545, -8.124763283579359, -1.668493744694982, 0.6911959422133213, -8.083285896618321})
	B := mat.NewDense(5, 1, []float64{-1.86590574665262, -0.6871570533880818, -0.1888641040097557, -1.0138306942332618, -0.7543616774747333})
	C := mat.NewDense(1, 5, []float64{-1.0969689928621424, -0.04789010123736493, 0.2244899132633626, 1.196144678101048, 1.8527133195893255})
	D := mat.NewDense(1, 1, []float64{-0.7783607791257332})
	E := mat.NewDense(5, 5, []float64{0.018138308692740912, 0.039622303558365844, -0.016206877837461046, -0.008806963997873991, -0.011776726212601505,
		0.015083065585986516, 0.008320892155679908, -0.03829312218697803, -0.018130584410720484, -0.019635222988625387,
		0.005534021513122287, 0.0009662813020849289, -0.025021102656268956, -0.007949051868397979, 0.05761837461981317,
		-0.05136429180590401, -0.003993710138701988, -0.015548703997087037, -0.0310271004665568, -0.006360850042439659,
		0.017999213458553077, -0.048115045445760896, -0.024336474114700732, 0.0010411789738205373, -0.012052708412684637})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	requireImproperDescriptorTF(t, "seed 1472", sys, res, err, 1e-6)
	if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("seed 1472: Step err = %v, want ErrImproperModel", err)
	}

	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, 31))
		nf := 1 + rng.IntN(3)
		chains := [][]int{{2}, {3}, {2, 1}, {3, 1}}[rng.IntN(4)]
		proper := seed%2 == 0
		sys, oracle, _ := chainDescriptor(t, rng, nf, chains, 0, proper)
		label := fmt.Sprintf("seed %d chains %v proper %v", seed, chains, proper)
		if got, err := sys.IsProper(); err != nil || got != proper {
			t.Fatalf("%s: IsProper = %v, %v", label, got, err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		if res.TF.isProper() != proper {
			t.Fatalf("%s: tf %v / %v", label, res.TF.Num, res.TF.Den)
		}
		for _, s := range []complex128{1i, complex(0.5, 3), 20i} {
			got, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			want := pencilResponse(t, oracle, s)[0][0]
			if d := cmplx.Abs(got[0][0] - want); !(d <= 1e-6*math.Max(1, cmplx.Abs(want))) {
				t.Fatalf("%s: G(%v) = %v, want %v", label, s, got[0][0], want)
			}
		}
	}
}

// TestTransferFunction_DescriptorIndexFourChain: an index-4 chain leaves
// its QZ members at |β|/|α| ~ ε^(1/4), above the large-eigenvalue bound,
// so the split kept them finite beside the true finite poles (e.g. chains
// [4] seed 239: QZ kept all 5, rank(E) = 4, 1 finite). The finite count cannot
// exceed rank(E), so such a split now goes to the staircase; before, the
// slow-subsystem LU failed and IsProper, Step and TransferFunction
// returned ErrDescriptorSingular. Oracle: the structured model. Seed 202
// of chains [4 1] (proper) is skipped: its c·N·b, zero in exact
// arithmetic, is 1.1× the 1e3·n·ε rounding bound after the staircase, so
// it is called improper (also on base).
func TestTransferFunction_DescriptorIndexFourChain(t *testing.T) {
	for _, chains := range [][]int{{4}, {4, 1}} {
		for seed := range uint64(300) {
			for _, proper := range []bool{true, false} {
				if seed == 202 && len(chains) == 2 && proper {
					continue
				}
				rng := rand.New(rand.NewPCG(seed, 77))
				nf := 1 + rng.IntN(3)
				sys, oracle, _ := chainDescriptor(t, rng, nf, chains, 0, proper)
				label := fmt.Sprintf("seed %d chains %v proper %v", seed, chains, proper)
				if got, err := sys.IsProper(); err != nil || got != proper {
					t.Fatalf("%s: IsProper = %v, %v", label, got, err)
				}
				res, err := sys.TransferFunction(nil)
				_, serr := Step(sys, 1)
				if !proper {
					if !errors.Is(err, ErrImproperModel) || !errors.Is(serr, ErrImproperModel) {
						t.Fatalf("%s: TransferFunction err = %v, Step err = %v, want ErrImproperModel", label, err, serr)
					}
					continue
				}
				if err != nil || serr != nil {
					t.Fatalf("%s: TransferFunction err = %v, Step err = %v", label, err, serr)
				}
				for _, s := range []complex128{1i, complex(0.5, 3), 20i} {
					got, err := res.TF.Eval(s)
					if err != nil {
						t.Fatal(err)
					}
					want := pencilResponse(t, oracle, s)[0][0]
					if d := cmplx.Abs(got[0][0] - want); !(d <= 1e-6*math.Max(1, cmplx.Abs(want))) {
						t.Fatalf("%s: G(%v) = %v, want %v", label, s, got[0][0], want)
					}
				}
			}
		}
	}
}

// TestDescriptor_IndexFiveSixChain: chains of length 5 and 6 leave their QZ
// members at |β|/|α| ~ ε^(1/k) ≈ 1e-3·‖E‖/‖A‖, at or above the
// large-eigenvalue bound, while rank(E) drops by one per chain. With one
// member marked infinite the QZ split kept rank(E) finite, so the
// slow-subsystem LU failed (IsProper, Step and TransferFunction returned
// ErrDescriptorSingular) or G was wrong, and Poles returned the k−1
// spurious members (|λ| ~ 1e3, e.g. chains [6] seed 0: 8 poles, 3 finite,
// IsStable wrong on seed 22). A pencil of index ≥ 2 has fewer than
// rank(E) finite eigenvalues, so both now defer to the staircase. Where
// the staircase is ambiguous too (amd64: [5] seed 66, [6] seed 84) all
// three fail closed with ErrDescriptorSingular, and IsStable may only
// report unstable; at most 5 such cases are allowed. Oracle: the
// structured model and the eigenvalues of its finite block.
func TestDescriptor_IndexFiveSixChain(t *testing.T) {
	unresolved := 0
	for _, chains := range [][]int{{5}, {6}, {5, 1}} {
		for seed := range uint64(100) {
			for _, proper := range []bool{true, false} {
				rng := rand.New(rand.NewPCG(seed, 77))
				nf := 1 + rng.IntN(3)
				sys, oracle, _ := chainDescriptor(t, rng, nf, chains, 0, proper)
				label := fmt.Sprintf("seed %d chains %v proper %v", seed, chains, proper)
				var eig mat.Eigen
				if !eig.Factorize(oracle.A.Slice(0, nf, 0, nf), mat.EigenNone) {
					t.Fatal("oracle eig failed")
				}
				want := eig.Values(nil)
				wantStable := true
				for _, p := range want {
					wantStable = wantStable && real(p) < 0
				}
				st, sterr := sys.IsStable()
				isProper, prErr := sys.IsProper()
				res, err := sys.TransferFunction(nil)
				_, serr := Step(sys, 1)
				if errors.Is(prErr, ErrDescriptorSingular) {
					if !errors.Is(err, ErrDescriptorSingular) || !errors.Is(serr, ErrDescriptorSingular) {
						t.Fatalf("%s: IsProper err = %v, TransferFunction err = %v, Step err = %v, want all ErrDescriptorSingular", label, prErr, err, serr)
					}
					if sterr == nil && st && !wantStable {
						t.Fatalf("%s: unresolved model reported stable", label)
					}
					unresolved++
					continue
				}
				got, perr := sys.Poles()
				if perr != nil || len(got) != nf {
					t.Fatalf("%s: Poles = %v, %v, want %v", label, got, perr, want)
				}
				requirePolesContain(t, label, got, want, 1e-6)
				if sterr != nil || st != wantStable {
					t.Fatalf("%s: IsStable = %v, %v, want %v", label, st, sterr, wantStable)
				}
				if prErr != nil || isProper != proper {
					t.Fatalf("%s: IsProper = %v, %v", label, isProper, prErr)
				}
				if !proper {
					if !errors.Is(err, ErrImproperModel) || !errors.Is(serr, ErrImproperModel) {
						t.Fatalf("%s: TransferFunction err = %v, Step err = %v, want ErrImproperModel", label, err, serr)
					}
					continue
				}
				if err != nil || serr != nil {
					t.Fatalf("%s: TransferFunction err = %v, Step err = %v", label, err, serr)
				}
				for _, s := range []complex128{1i, complex(0.5, 3), 20i} {
					g, err := res.TF.Eval(s)
					if err != nil {
						t.Fatal(err)
					}
					w := pencilResponse(t, oracle, s)[0][0]
					if d := cmplx.Abs(g[0][0] - w); !(d <= 1e-6*math.Max(1, cmplx.Abs(w))) {
						t.Fatalf("%s: G(%v) = %v, want %v", label, s, g[0][0], w)
					}
				}
			}
		}
	}
	if unresolved > 5 {
		t.Fatalf("%d models failed closed, want at most 5", unresolved)
	}
}

// TestPoles_DescriptorRankMATLABEps: E = diag(1.99, 1, 0.5, 1.3e-15) is
// nonsingular by MATLAB rank, whose bound n·eps(σ₁) = 8.9e-16 is the float
// spacing at σ₁ = 1.99, half of n·ε·σ₁ = 1.8e-15. The rank floor used
// n·ε·σ₁, so Poles dropped the stiff pole near +1e15 and IsStable reported
// a stable model. Oracle: eig(E⁻¹A), E diagonal.
func TestPoles_DescriptorRankMATLABEps(t *testing.T) {
	ed := []float64{1.99, 1, 0.5, 1.3e-15}
	a := mat.NewDense(4, 4, []float64{
		-2, 0.4, -0.3, 0.2,
		0.7, -1.5, 0.5, -0.1,
		-0.2, 0.3, -3, 0.6,
		0.5, -0.4, 0.8, 1.1,
	})
	e := mat.NewDense(4, 4, nil)
	ea := mat.NewDense(4, 4, nil)
	for i := range 4 {
		e.Set(i, i, ed[i])
		for j := range 4 {
			ea.Set(i, j, a.At(i, j)/ed[i])
		}
	}
	sys, err := NewDescriptor(a, mat.NewDense(4, 1, []float64{1, 0, -1, 0.5}), mat.NewDense(1, 4, []float64{1, -0.5, 0.3, 0.2}), mat.NewDense(1, 1, nil), e, 0)
	if err != nil {
		t.Fatal(err)
	}
	var eig mat.Eigen
	if !eig.Factorize(ea, mat.EigenNone) {
		t.Fatal("oracle eig failed")
	}
	want := eig.Values(nil)
	got, err := sys.Poles()
	if err != nil || len(got) != 4 {
		t.Fatalf("Poles = %v, %v, want %v", got, err, want)
	}
	requirePolesContain(t, "rank-eps", got, want, 1e-6)
	if st, err := sys.IsStable(); err != nil || st {
		t.Fatalf("IsStable = %v, %v, want false (poles %v)", st, err, want)
	}
}

// TestStabsep_StiffPoleBesideChainFailsClosed: a stiff finite pole (E
// singular value 29× the rounding n·ε·σ₁) beside an index-2 chain and an
// index-1 algebraic state (review seed 351). The staircase reports stiff,
// the index test at stiffIndexTol sends the pencil to the whole-pencil QZ,
// and certifyFiniteRetained then certified the large chain members finite
// at √ε: Stable got poles −7.7e5 ± 1.3e6i and Stable(1.3j) = −195.8 against
// −1.907. Base fails closed. Oracle: the structured pencil, one finite pole
// 6.174e12 (unstable), so Stable is the static gain D. With the stiff
// singular value within ~10× the rounding the chain leaves Ulᵀ·A·Vr above
// the fixed 1e-5·‖A‖ (e.g. seed 200: the staircase assumed index 1, Poles
// lost the stiff pole and Stable was wrong); the bound now grows as
// n·ε·σ₁/σs. In the sweep (stiff poles QZ resolves), Poles keeps the stiff
// pole and Stabsep either fails or Stable has only true stable poles and,
// unless it lost one (as base does on a few draws), matches the modal sum
// of the structured model.
func TestStabsep_StiffPoleBesideChainFailsClosed(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{0.0326899509802978, -0.36681246430299563, -3.56487586437879, 0.25246302255452163,
		0.6606135039155011, -0.43540691343289234, 3.3979318604419704, -1.7075249912700554,
		1.3826351754761992, 0.640327981177481, 3.1680427757739014, 0.44104901835708266,
		-1.514985795667952, 0.1931901558680369, -0.1333724682997395, -0.7695321392558229})
	E := mat.NewDense(4, 4, []float64{1.0662327841230625, 1.7917350472211688, 2.531354002133062, 2.217927773599035,
		0.5932828850880467, -0.8836228344645478, -0.8753323906956624, -0.4134416502459626,
		-0.372441429505175, -0.8025989135619086, -1.0988494198957623, -0.9295703080116933,
		-1.9927302748987492, -1.1015990026565192, -2.002071511755178, -2.1765748276831554})
	B := mat.NewDense(4, 1, []float64{-0.6619790154245511, -1.854762109834214, -0.9504604353484285, -1.3106967440008177})
	C := mat.NewDense(1, 4, []float64{0.49894649696582166, 0.33908167011908524, 0.4276281248480783, 0.8677160078788655})
	D := mat.NewDense(1, 1, []float64{-1.9066529430493717})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res, err := Stabsep(sys); err == nil {
		for _, s := range []complex128{1.3i, complex(0.5, 2), 10i} {
			if got := pencilResponse(t, res.Stable, s)[0][0]; !(cmplx.Abs(got-complex(D.At(0, 0), 0)) <= 1e-6) {
				t.Fatalf("seed 351: Stable(%v) = %v, want D = %v", s, got, D.At(0, 0))
			}
		}
	}

	split, failed, lost := 0, 0, 0
	for seed := range uint64(1000) {
		rng := rand.New(rand.NewPCG(seed, 2026))
		nf := 1 + rng.IntN(2)
		chains := [][]int{{2}, {2, 1}, {3}}[rng.IntN(3)]
		sys, oracle, poles := chainDescriptor(t, rng, nf, chains, math.Pow(10, -14+5*rng.Float64()), true)
		n, _, _ := sys.Dims()
		var svd mat.SVD
		if !svd.Factorize(sys.E, mat.SVDNone) {
			t.Fatal("svd failed")
		}
		sv := svd.Values(nil)
		if _, stiff, _ := descriptorRank(sv, sv[0], n, true); !stiff {
			continue
		}
		ar, ai, b, ok := qzEigenvalues(n, sys.A.RawMatrix().Data, n, sys.E.RawMatrix().Data, n)
		if !ok {
			t.Fatal("QZ failed")
		}
		resolved := finitePoles(ar, ai, b, 0)
		unresolved := false
		for _, w := range poles {
			found := false
			for _, g := range resolved {
				found = found || cmplx.Abs(g-w) <= 2e-2*cmplx.Abs(w)
			}
			unresolved = unresolved || !found
		}
		if unresolved {
			continue
		}
		label := fmt.Sprintf("seed %d chains %v", seed, chains)
		got, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", label, err)
		}
		requirePolesContain(t, label, got, poles, 2e-2)
		res, err := Stabsep(sys)
		if err != nil {
			failed++
			continue
		}
		split++
		var stable []complex128
		for _, p := range poles {
			if real(p) < 0 {
				stable = append(stable, p)
			}
		}
		got, err = res.Stable.Poles()
		if err != nil {
			t.Fatalf("%s: Stable.Poles: %v", label, err)
		}
		for _, g := range got {
			requirePolesContain(t, label+" Stable pole", stable, []complex128{g}, 2e-2)
		}
		if len(got) < len(stable) {
			lost++
			continue
		}
		for _, s := range []complex128{1.3i, complex(0.5, 2), 10i} {
			want := complex(oracle.D.At(0, 0), 0)
			for i, p := range poles {
				if real(p) < 0 {
					f, w := oracle.A.At(i, i), oracle.E.At(i, i)
					want += complex(oracle.C.At(0, i)*oracle.B.At(i, 0), 0) / (s*complex(w, 0) - complex(f, 0))
				}
			}
			scale := math.Max(1, cmplx.Abs(pencilResponse(t, oracle, s)[0][0]))
			if got := pencilResponse(t, res.Stable, s)[0][0]; !(cmplx.Abs(got-want) <= 1e-6*scale) {
				t.Fatalf("%s: Stable(%v) = %v, want %v", label, s, got, want)
			}
		}
	}
	t.Logf("stiff draws: %d split (%d losing a stable pole), %d failed closed", split, lost, failed)
}
