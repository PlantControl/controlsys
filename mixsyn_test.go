package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func mixsynTF(t testing.TB, num, den []float64, dt float64) *System {
	t.Helper()
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	r, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	return r.Sys
}

func mixsynWeight(t testing.TB, dc float64, fm []float64, hf float64) *System {
	t.Helper()
	W, err := Makeweight(dc, fm, hf, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	return W
}

// mixsynFR evaluates C(sI−A)⁻¹B + D by complex Gaussian elimination with
// partial pivoting, independent of the library's frequency evaluators.
func mixsynFR(sys *System, s complex128) [][]complex128 {
	n, m, p := sys.Dims()
	M := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n+m)
		for j := range n {
			M[i][j] = complex(-sys.A.At(i, j), 0)
		}
		M[i][i] += s
		for j := range m {
			M[i][n+j] = complex(sys.B.At(i, j), 0)
		}
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(M[i][k]) > cmplx.Abs(M[piv][k]) {
				piv = i
			}
		}
		M[k], M[piv] = M[piv], M[k]
		for i := k + 1; i < n; i++ {
			f := M[i][k] / M[k][k]
			for j := k; j < n+m; j++ {
				M[i][j] -= f * M[k][j]
			}
		}
	}
	X := make([][]complex128, n)
	for i := n - 1; i >= 0; i-- {
		X[i] = make([]complex128, m)
		for j := range m {
			v := M[i][n+j]
			for k := i + 1; k < n; k++ {
				v -= M[i][k] * X[k][j]
			}
			X[i][j] = v / M[i][i]
		}
	}
	H := make([][]complex128, p)
	for i := range p {
		H[i] = make([]complex128, m)
		for j := range m {
			v := complex(sys.D.At(i, j), 0)
			for k := range n {
				v += complex(sys.C.At(i, k), 0) * X[k][j]
			}
			H[i][j] = v
		}
	}
	return H
}

func cmatMul(A, B [][]complex128) [][]complex128 {
	C := make([][]complex128, len(A))
	for i := range A {
		C[i] = make([]complex128, len(B[0]))
		for j := range B[0] {
			for k := range B {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

func cmatScale(A [][]complex128, f complex128) [][]complex128 {
	C := make([][]complex128, len(A))
	for i := range A {
		C[i] = make([]complex128, len(A[i]))
		for j := range A[i] {
			C[i][j] = f * A[i][j]
		}
	}
	return C
}

func cmatEye(n int) [][]complex128 {
	I := make([][]complex128, n)
	for i := range n {
		I[i] = make([]complex128, n)
		I[i][i] = 1
	}
	return I
}

// weightFR is W(s), or W(s)·I for a SISO W of a size-n channel group.
func weightFR(W *System, s complex128, n int) [][]complex128 {
	h := mixsynFR(W, s)
	if len(h) == 1 && n > 1 {
		return cmatScale(cmatEye(n), h[0][0])
	}
	return h
}

// augwOracle assembles [W1 −W1G; 0 W2; 0 W3G; I −G] at s from the block
// responses, scaled by the plant input-delay factor.
func augwOracle(G, W1, W2, W3 *System, s complex128, delay complex128) [][]complex128 {
	_, nu, ny := G.Dims()
	g := cmatScale(mixsynFR(G, s), delay)
	negG := cmatScale(g, -1)
	type row struct{ left, right [][]complex128 }
	var rows []row
	zero := func(r, c int) [][]complex128 { return cmatScale(make2(r, c), 0) }
	if W1 != nil {
		w := weightFR(W1, s, ny)
		rows = append(rows, row{w, cmatMul(w, negG)})
	}
	if W2 != nil {
		rows = append(rows, row{zero(nu, ny), weightFR(W2, s, nu)})
	}
	if W3 != nil {
		rows = append(rows, row{zero(ny, ny), cmatMul(weightFR(W3, s, ny), g)})
	}
	rows = append(rows, row{cmatEye(ny), negG})
	var P [][]complex128
	for _, r := range rows {
		for i := range r.left {
			P = append(P, append(slices.Clone(r.left[i]), r.right[i]...))
		}
	}
	return P
}

func make2(r, c int) [][]complex128 {
	M := make([][]complex128, r)
	for i := range M {
		M[i] = make([]complex128, c)
	}
	return M
}

func assertFRClose(t *testing.T, label string, got, want [][]complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) || len(got[0]) != len(want[0]) {
		t.Fatalf("%s: %d×%d, want %d×%d", label, len(got), len(got[0]), len(want), len(want[0]))
	}
	for i := range want {
		for j := range want[i] {
			if d := cmplx.Abs(got[i][j] - want[i][j]); d > tol*math.Max(1, cmplx.Abs(want[i][j])) {
				t.Errorf("%s[%d][%d] = %v, want %v", label, i, j, got[i][j], want[i][j])
			}
		}
	}
}

func mimoMixsynPlant(t testing.TB, d float64) *System {
	t.Helper()
	G, err := New(
		mat.NewDense(3, 3, []float64{-1, 0.4, 0, -0.3, -2, 0.5, 0.2, 0, -3}),
		mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, 0, -0.7}),
		mat.NewDense(2, 3, []float64{1, -0.2, 0.3, 0, 1, 0.8}),
		mat.NewDense(2, 2, []float64{d, 0, 0.3 * d, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return G
}

func diagWeight(t testing.TB, ws ...*System) *System {
	t.Helper()
	W, err := BlkDiag(ws...)
	if err != nil {
		t.Fatal(err)
	}
	return W
}

func TestAugwPartitionAndNames(t *testing.T) {
	siso := mixsynTF(t, []float64{1, -1}, []float64{1, 2, 1}, 0)
	w1 := mixsynWeight(t, 10, []float64{1, 0.1}, 0.01)
	w2 := mixsynWeight(t, 0.1, []float64{32, 0.32}, 1)
	w3 := mixsynWeight(t, 0.01, []float64{1, 0.1}, 10)
	mimo := mimoMixsynPlant(t, 0.5)
	w2b := mixsynWeight(t, 0.2, []float64{10, 0.5}, 2)
	W2full := diagWeight(t, w2, w2b)
	cases := []struct {
		name          string
		G, W1, W2, W3 *System
		in, out       []string
	}{
		{"siso all", siso, w1, w2, w3, []string{"w", "u"}, []string{"z1", "z2", "z3", "e"}},
		{"siso no W2", siso, w1, nil, w3, []string{"w", "u"}, []string{"z1", "z3", "e"}},
		{"siso W2 only", siso, nil, w2, nil, []string{"w", "u"}, []string{"z2", "e"}},
		{"mimo scalar weights", mimo, w1, w2, w3,
			[]string{"w(1)", "w(2)", "u(1)", "u(2)"},
			[]string{"z1(1)", "z1(2)", "z2(1)", "z2(2)", "z3(1)", "z3(2)", "e(1)", "e(2)"}},
		{"mimo square W2", mimo, w1, W2full, nil,
			[]string{"w(1)", "w(2)", "u(1)", "u(2)"},
			[]string{"z1(1)", "z1(2)", "z2(1)", "z2(2)", "e(1)", "e(2)"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			P, err := Augw(tc.G, tc.W1, tc.W2, tc.W3)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(P.InputName, tc.in) || !slices.Equal(P.OutputName, tc.out) {
				t.Fatalf("names in %v out %v, want %v %v", P.InputName, P.OutputName, tc.in, tc.out)
			}
			nG, _, _ := tc.G.Dims()
			_, nu, ny := tc.G.Dims()
			wantStates := nG
			for _, w := range []struct {
				W    *System
				size int
			}{{tc.W1, ny}, {tc.W2, nu}, {tc.W3, ny}} {
				if w.W == nil {
					continue
				}
				nw, _, pw := w.W.Dims()
				if pw == 1 {
					nw *= w.size
				}
				wantStates += nw
			}
			if n, _, _ := P.Dims(); n != wantStates {
				t.Fatalf("P has %d states, want %d", n, wantStates)
			}
			for _, w := range []float64{0, 0.07, 1, 3.3, 40} {
				s := complex(0, w)
				assertFRClose(t, fmt.Sprintf("P(j%g)", w), mixsynFR(P, s), augwOracle(tc.G, tc.W1, tc.W2, tc.W3, s, 1), 1e-9)
			}
		})
	}
}

func TestAugwDiscrete(t *testing.T) {
	const dt = 0.1
	G := mixsynTF(t, []float64{0.5, -0.2}, []float64{1, -1.2, 0.5}, dt)
	W1, err := Makeweight(10, []float64{1, 0.5}, 0.1, dt, 2)
	if err != nil {
		t.Fatal(err)
	}
	W2, err := Makeweight(0.1, []float64{5}, 3, dt, 1)
	if err != nil {
		t.Fatal(err)
	}
	P, err := Augw(G, W1, W2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if P.Dt != dt {
		t.Fatalf("Dt = %g", P.Dt)
	}
	for _, th := range []float64{0, 0.4, 2, 3.1} {
		z := cmplx.Exp(complex(0, th))
		assertFRClose(t, fmt.Sprintf("P(e^j%g)", th), mixsynFR(P, z), augwOracle(G, W1, W2, nil, z, 1), 1e-9)
	}
}

func TestAugwInputDelay(t *testing.T) {
	G := mixsynTF(t, []float64{2}, []float64{1, 3}, 0)
	G.InputDelay = []float64{0.4}
	W1 := mixsynWeight(t, 10, []float64{1}, 0.1)
	W2 := mixsynWeight(t, 0.1, []float64{10}, 2)
	W3 := mixsynWeight(t, 0.1, []float64{5}, 10)
	P, err := Augw(G, W1, W2, W3)
	if err != nil {
		t.Fatal(err)
	}
	G0 := G.Copy()
	G0.InputDelay = nil
	for _, w := range []float64{0, 0.5, 2, 7} {
		s := complex(0, w)
		got, err := P.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		assertFRClose(t, fmt.Sprintf("P(j%g)", w), got, augwOracle(G0, W1, W2, W3, s, cmplx.Exp(-0.4*s)), 1e-9)
	}
	if _, err := Mixsyn(G, W1, W2, W3); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("Mixsyn delay err = %v, want ErrDelayUnsupported", err)
	}
}

// W3 = s + 1 as a singular-E descriptor: improper alone, proper after a
// strictly proper G.
func improperW3(t *testing.T) *System {
	t.Helper()
	W, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 1, []float64{0, -1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{0, 1, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return W
}

func TestAugwImproperW3(t *testing.T) {
	W3 := improperW3(t)
	W1 := mixsynWeight(t, 10, []float64{1}, 0.1)
	W2 := mixsynWeight(t, 0.1, []float64{10}, 2)
	G := mixsynTF(t, []float64{1}, []float64{1, 2}, 0)
	P, err := Augw(G, W1, W2, W3)
	if err != nil {
		t.Fatal(err)
	}
	if P.IsDescriptor() {
		t.Fatal("P is a descriptor model")
	}
	for _, w := range []float64{0, 1, 3, 30} {
		s := complex(0, w)
		want := augwOracle(G, W1, W2, nil, s, 1)
		z3 := (s + 1) / (s + 2)
		want = slices.Insert(want, 2, []complex128{0, z3})
		assertFRClose(t, fmt.Sprintf("P(j%g)", w), mixsynFR(P, s), want, 1e-9)
	}

	Gbi := mixsynTF(t, []float64{1, 3}, []float64{1, 2}, 0)
	if _, err := Augw(Gbi, W1, W2, W3); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("improper W3·G err = %v, want ErrImproperModel", err)
	}
	if _, err := Mixsyn(Gbi, W1, W2, W3); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("Mixsyn improper W3·G err = %v, want ErrImproperModel", err)
	}
}

func TestAugwErrors(t *testing.T) {
	G := mimoMixsynPlant(t, 0)
	w := mixsynWeight(t, 10, []float64{1}, 0.1)
	w3x3, err := BlkDiag(w, w, w)
	if err != nil {
		t.Fatal(err)
	}
	wd, err := Makeweight(10, []float64{1}, 0.1, 0.1, 1)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name          string
		G, W1, W2, W3 *System
		want          error
	}{
		{"nil G", nil, w, w, w, ErrInvalidArgument},
		{"W1 wrong size", G, w3x3, nil, nil, ErrDimensionMismatch},
		{"W2 wrong size", G, nil, w3x3, nil, ErrDimensionMismatch},
		{"W3 wrong size", G, nil, nil, w3x3, ErrDimensionMismatch},
		{"domain mismatch", G, wd, nil, nil, ErrDomainMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			P, err := Augw(tc.G, tc.W1, tc.W2, tc.W3)
			if !errors.Is(err, tc.want) || P != nil {
				t.Fatalf("Augw = %v, %v; want %v", P, err, tc.want)
			}
			r, err := Mixsyn(tc.G, tc.W1, tc.W2, tc.W3)
			if !errors.Is(err, tc.want) || r != nil {
				t.Fatalf("Mixsyn = %v, %v; want %v", r, err, tc.want)
			}
		})
	}
}

func TestMixsynErrors(t *testing.T) {
	G := mixsynTF(t, []float64{200}, []float64{0.025, 1.01, 10.1, 1}, 0)
	W1 := mixsynTF(t, []float64{1}, []float64{1, 1}, 0)
	W1bi := mixsynWeight(t, 10, []float64{1}, 0.1)
	W3 := mixsynWeight(t, 0.1, []float64{5}, 10)
	cases := []struct {
		name       string
		W1, W2, W3 *System
	}{
		{"W2 nil strictly proper W1", W1, nil, nil},
		{"W2 nil biproper W1", W1bi, nil, nil},
		{"W2 nil with W3", W1bi, nil, W3},
		{"no weights", nil, nil, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Mixsyn(G, tc.W1, tc.W2, tc.W3)
			if !errors.Is(err, ErrInvalidPartition) || r != nil {
				t.Fatalf("Mixsyn = %v, %v; want ErrInvalidPartition", r, err)
			}
		})
	}
}

// mixsynHandLoop builds [W1·e; W2·u; W3·y] driven by r, with u = K·e,
// e = r − y and y = G·u, directly from the component realizations; weights
// must be full-size (no SISO expansion).
func mixsynHandLoop(t *testing.T, G, K, W1, W2, W3 *System) (A, B, C, D *mat.Dense) {
	t.Helper()
	nG, nu, ny := G.Dims()
	nK, _, _ := K.Dims()
	n0 := nG + nK
	Mi := mat.NewDense(nu, nu, nil)
	Mi.Mul(K.D, G.D)
	for i := range nu {
		Mi.Set(i, i, Mi.At(i, i)+1)
	}
	var M mat.Dense
	if err := M.Inverse(Mi); err != nil {
		t.Fatal(err)
	}
	Cu := mat.NewDense(nu, n0, nil)
	var KDCg mat.Dense
	KDCg.Mul(K.D, G.C)
	KDCg.Scale(-1, &KDCg)
	Cu.Slice(0, nu, 0, nG).(*mat.Dense).Copy(&KDCg)
	Cu.Slice(0, nu, nG, n0).(*mat.Dense).Copy(K.C)
	var CuM, DuM mat.Dense
	CuM.Mul(&M, Cu)
	DuM.Mul(&M, K.D)
	Cy := mat.NewDense(ny, n0, nil)
	Cy.Slice(0, ny, 0, nG).(*mat.Dense).Copy(G.C)
	var tmp mat.Dense
	tmp.Mul(G.D, &CuM)
	Cy.Add(Cy, &tmp)
	var Dy mat.Dense
	Dy.Mul(G.D, &DuM)
	Ce := mat.DenseCopyOf(Cy)
	Ce.Scale(-1, Ce)
	De := mat.DenseCopyOf(&Dy)
	De.Scale(-1, De)
	for i := range ny {
		De.Set(i, i, De.At(i, i)+1)
	}

	type sig struct {
		W    *System
		C, D *mat.Dense
	}
	sigs := []sig{{W1, Ce, De}, {W2, &CuM, &DuM}, {W3, Cy, &Dy}}
	n, p := n0, 0
	for _, s := range sigs {
		if s.W != nil {
			nw, _, pw := s.W.Dims()
			n += nw
			p += pw
		}
	}
	A, B = mat.NewDense(n, n, nil), mat.NewDense(n, ny, nil)
	C, D = mat.NewDense(p, n, nil), mat.NewDense(p, ny, nil)
	var bu, bue, bke mat.Dense
	bu.Mul(G.B, &CuM)
	A.Slice(0, nG, 0, n0).(*mat.Dense).Copy(&bu)
	ag := A.Slice(0, nG, 0, nG).(*mat.Dense)
	ag.Add(ag, G.A)
	bue.Mul(G.B, &DuM)
	B.Slice(0, nG, 0, ny).(*mat.Dense).Copy(&bue)
	bke.Mul(K.B, Ce)
	A.Slice(nG, n0, 0, n0).(*mat.Dense).Copy(&bke)
	ak := A.Slice(nG, n0, nG, n0).(*mat.Dense)
	ak.Add(ak, K.A)
	var bkd mat.Dense
	bkd.Mul(K.B, De)
	B.Slice(nG, n0, 0, ny).(*mat.Dense).Copy(&bkd)
	off, row := n0, 0
	for _, s := range sigs {
		if s.W == nil {
			continue
		}
		nw, _, pw := s.W.Dims()
		var bc, bd, dc, dd mat.Dense
		if nw > 0 {
			bc.Mul(s.W.B, s.C)
			A.Slice(off, off+nw, 0, n0).(*mat.Dense).Copy(&bc)
			A.Slice(off, off+nw, off, off+nw).(*mat.Dense).Copy(s.W.A)
			bd.Mul(s.W.B, s.D)
			B.Slice(off, off+nw, 0, ny).(*mat.Dense).Copy(&bd)
			C.Slice(row, row+pw, off, off+nw).(*mat.Dense).Copy(s.W.C)
		}
		dc.Mul(s.W.D, s.C)
		C.Slice(row, row+pw, 0, n0).(*mat.Dense).Copy(&dc)
		dd.Mul(s.W.D, s.D)
		D.Slice(row, row+pw, 0, ny).(*mat.Dense).Copy(&dd)
		off += nw
		row += pw
	}
	return A, B, C, D
}

// hamiltonianHasImagEig reports whether γ ≤ ‖C(sI−A)⁻¹B + D‖∞ for stable A,
// by the imaginary-axis eigenvalues of the Hamiltonian with R = γ²I − DᵀD.
func hamiltonianHasImagEig(A, B, C, D *mat.Dense, gamma float64) bool {
	n, _ := A.Dims()
	_, m := B.Dims()
	p, _ := C.Dims()
	R := mat.NewDense(m, m, nil)
	R.Mul(D.T(), D)
	R.Scale(-1, R)
	for i := range m {
		R.Set(i, i, R.At(i, i)+gamma*gamma)
	}
	var Ri mat.Dense
	if err := Ri.Inverse(R); err != nil {
		return true
	}
	var BRi, F, G, DRi, Sm, Q mat.Dense
	BRi.Mul(B, &Ri)
	var DtC mat.Dense
	DtC.Mul(D.T(), C)
	F.Mul(&BRi, &DtC)
	F.Add(&F, A)
	G.Mul(&BRi, B.T())
	DRi.Mul(D, &Ri)
	Sm.Mul(&DRi, D.T())
	for i := range p {
		Sm.Set(i, i, Sm.At(i, i)+1)
	}
	var SC mat.Dense
	SC.Mul(&Sm, C)
	Q.Mul(C.T(), &SC)
	H := mat.NewDense(2*n, 2*n, nil)
	H.Slice(0, n, 0, n).(*mat.Dense).Copy(&F)
	H.Slice(0, n, n, 2*n).(*mat.Dense).Copy(&G)
	Qn := H.Slice(n, 2*n, 0, n).(*mat.Dense)
	Qn.Scale(-1, &Q)
	Ft := H.Slice(n, 2*n, n, 2*n).(*mat.Dense)
	Ft.Scale(-1, F.T())
	var eig mat.Eigen
	if !eig.Factorize(H, mat.EigenNone) {
		return true
	}
	for _, ev := range eig.Values(nil) {
		if math.Abs(real(ev)) <= 1e-7*cmplx.Abs(ev) {
			return true
		}
	}
	return false
}

// frSigmaMax is the largest singular value of a complex matrix H, the
// square root of the largest eigenvalue of the real 2p×2p embedding of HᴴH.
func frSigmaMax(H [][]complex128) float64 {
	p, m := len(H), len(H[0])
	R := mat.NewDense(2*p, 2*m, nil)
	for i := range p {
		for j := range m {
			re, im := real(H[i][j]), imag(H[i][j])
			R.Set(i, j, re)
			R.Set(i, m+j, -im)
			R.Set(p+i, j, im)
			R.Set(p+i, m+j, re)
		}
	}
	var svd mat.SVD
	svd.Factorize(R, mat.SVDNone)
	return svd.Values(nil)[0]
}

// hinfNormBisection is the H∞ norm of a stable model by bisection on the
// Hamiltonian imaginary-eigenvalue test, to relative 1e-9.
func hinfNormBisection(t *testing.T, A, B, C, D *mat.Dense) float64 {
	t.Helper()
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	var svd mat.SVD
	svd.Factorize(D, mat.SVDNone)
	lo := svd.Values(nil)[0]
	for k := -40; k <= 40; k++ {
		lo = math.Max(lo, frSigmaMax(mixsynFR(sys, complex(0, math.Pow(10, float64(k)/10)))))
	}
	lo = math.Max(lo, frSigmaMax(mixsynFR(sys, 0)))
	hi := math.Max(2*lo, 1)
	for hamiltonianHasImagEig(A, B, C, D, hi) {
		if hi *= 2; hi > 1e12 {
			t.Fatal("no upper bound for the H∞ norm")
		}
	}
	for hi-lo > 1e-9*hi {
		mid := (lo + hi) / 2
		if hamiltonianHasImagEig(A, B, C, D, mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return hi
}

func assertMixsynOracle(t *testing.T, G, W1, W2, W3 *System) *MixsynResult {
	t.Helper()
	r, err := Mixsyn(G, W1, W2, W3)
	if err != nil {
		t.Fatal(err)
	}
	if r.K.InputName[0] != "e" && r.K.InputName[0] != "e(1)" {
		t.Errorf("K input names %v", r.K.InputName)
	}
	A, B, C, D := mixsynHandLoop(t, G, r.K, W1, W2, W3)
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		t.Fatal("closed-loop eigenvalues failed")
	}
	for _, ev := range eig.Values(nil) {
		if real(ev) >= 0 {
			t.Fatalf("closed loop unstable: pole %v", ev)
		}
	}
	hand, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0, 0.01, 0.3, 1, 4, 50} {
		s := complex(0, w)
		assertFRClose(t, fmt.Sprintf("CL(j%g)", w), mixsynFR(r.CL, s), mixsynFR(hand, s), 1e-7)
	}
	norm := hinfNormBisection(t, A, B, C, D)
	if math.Abs(r.Gamma-norm) > 1e-6*norm {
		t.Fatalf("Gamma = %.12g, bisection ‖CL‖∞ = %.12g", r.Gamma, norm)
	}
	if r.Gamma > r.Info.GammaOpt*(1+1e-9) {
		t.Fatalf("Gamma %.12g exceeds HinfSyn bound %.12g", r.Gamma, r.Info.GammaOpt)
	}
	if r.Info.GammaOpt > r.Gamma*(1+2e-4) {
		t.Fatalf("HinfSyn bound %.12g not near achieved %.12g", r.Info.GammaOpt, r.Gamma)
	}
	return r
}

// MathWorks mixsyn doc example: G = (s−1)/(s+1)² with makeweight weights.
func TestMixsynMATLABDocExample(t *testing.T) {
	G := mixsynTF(t, []float64{1, -1}, []float64{1, 2, 1}, 0)
	W1 := mixsynWeight(t, 10, []float64{1, 0.1}, 0.01)
	W2 := mixsynWeight(t, 0.1, []float64{32, 0.32}, 1)
	W3 := mixsynWeight(t, 0.01, []float64{1, 0.1}, 10)
	r := assertMixsynOracle(t, G, W1, W2, W3)
	for _, w := range []float64{1e-3, 0.1, 1, 10, 100} {
		s := complex(0, w)
		g := mixsynFR(G, s)[0][0]
		k := mixsynFR(r.K, s)[0][0]
		S := 1 / (1 + g*k)
		if bound := r.Gamma / cmplx.Abs(mixsynFR(W1, s)[0][0]); cmplx.Abs(S) > bound*(1+1e-9) {
			t.Errorf("|S(j%g)| = %g exceeds γ/|W1| = %g", w, cmplx.Abs(S), bound)
		}
		if bound := r.Gamma / cmplx.Abs(mixsynFR(W3, s)[0][0]); cmplx.Abs(g*k*S) > bound*(1+1e-9) {
			t.Errorf("|T(j%g)| = %g exceeds γ/|W3| = %g", w, cmplx.Abs(g*k*S), bound)
		}
	}
}

// Skogestad & Postlethwaite, Multivariable Feedback Control (2nd ed.),
// §2.8.3: G = 200/((10s+1)(0.05s+1)²), W_P = (s/M + ω_B)/(s + ω_B·A) with
// M = 1.5, ω_B = 10, A = 1e-4, W_u = 1, reported γ = 1.37.
func TestMixsynSkogestadPostlethwaite(t *testing.T) {
	G := mixsynTF(t, []float64{200}, []float64{0.025, 1.01, 10.1, 1}, 0)
	WP := mixsynTF(t, []float64{1 / 1.5, 10}, []float64{1, 10 * 1e-4}, 0)
	Wu, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	r := assertMixsynOracle(t, G, WP, Wu, nil)
	if math.Abs(r.Gamma-1.37) > 0.005 {
		t.Fatalf("Gamma = %g, published 1.37", r.Gamma)
	}
}

func TestMixsynMIMO(t *testing.T) {
	G := mimoMixsynPlant(t, 0.5)
	W1 := diagWeight(t, mixsynWeight(t, 20, []float64{0.5}, 0.2), mixsynWeight(t, 10, []float64{1}, 0.3))
	W2 := diagWeight(t, mixsynWeight(t, 0.1, []float64{20, 0.5}, 2), mixsynWeight(t, 0.2, []float64{10, 0.5}, 1))
	W3 := diagWeight(t, mixsynWeight(t, 0.05, []float64{8}, 5), mixsynWeight(t, 0.05, []float64{8}, 5))
	assertMixsynOracle(t, G, W1, W2, W3)

	scalar := mixsynWeight(t, 10, []float64{1}, 0.3)
	rs, err := Mixsyn(G, scalar, W2, nil)
	if err != nil {
		t.Fatal(err)
	}
	rf := assertMixsynOracle(t, G, diagWeight(t, scalar, scalar), W2, nil)
	if math.Abs(rs.Gamma-rf.Gamma) > 1e-6*rf.Gamma {
		t.Fatalf("scalar W1 gamma %g, diagonal W1 gamma %g", rs.Gamma, rf.Gamma)
	}
}

// A biproper G fills D12 through W3·D_G, so W2 may be nil.
func TestMixsynBiproperPlantWithoutW2(t *testing.T) {
	G := mixsynTF(t, []float64{0.5, 1, 2}, []float64{1, 3, 2}, 0)
	W1 := mixsynWeight(t, 10, []float64{1}, 0.1)
	W3 := mixsynWeight(t, 0.1, []float64{10}, 2)
	assertMixsynOracle(t, G, W1, nil, W3)
}

func BenchmarkMixsyn(b *testing.B) {
	W1 := mixsynWeight(b, 100, []float64{0.5}, 0.1)
	W2 := mixsynWeight(b, 0.1, []float64{20, 0.5}, 2)
	W3 := mixsynWeight(b, 0.01, []float64{10}, 10)
	for _, n := range []int{2, 10, 30} {
		G := randomHinfTestPlant(b, n, 1, int64(n))
		b.Run(fmt.Sprintf("order%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := Mixsyn(G, W1, W2, W3); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestAugwNonsingularDescriptorWeight(t *testing.T) {
	W1d, err := NewDescriptor(
		mat.NewDense(1, 1, []float64{-0.2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	W1, err := W1d.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	G := mimoMixsynPlant(t, 0.5)
	W2 := mixsynWeight(t, 0.1, []float64{10}, 2)
	P, err := Augw(G, W1d, W2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if P.IsDescriptor() {
		t.Fatal("P is a descriptor model")
	}
	for _, w := range []float64{0, 0.3, 5} {
		s := complex(0, w)
		assertFRClose(t, fmt.Sprintf("P(j%g)", w), mixsynFR(P, s), augwOracle(G, W1, W2, nil, s, 1), 1e-9)
	}
}

// assertDiscreteMixsyn checks a discrete Mixsyn design against the hand-built
// weighted loop: Schur stable, CL equal to the hand loop on the unit circle,
// and Gamma equal to its independent unit-circle peak.
func assertDiscreteMixsyn(t *testing.T, G, W1, W2, W3 *System) *MixsynResult {
	t.Helper()
	r, err := Mixsyn(G, W1, W2, W3)
	if err != nil {
		t.Fatal(err)
	}
	if r.K.Dt != G.Dt || r.CL.Dt != G.Dt {
		t.Fatalf("K.Dt = %g, CL.Dt = %g, want %g", r.K.Dt, r.CL.Dt, G.Dt)
	}
	A, B, C, D := mixsynHandLoop(t, G, r.K, W1, W2, W3)
	assertSchurStable(t, A)
	hand, err := New(A, B, C, D, G.Dt)
	if err != nil {
		t.Fatal(err)
	}
	for _, th := range []float64{0, 0.05, 0.4, 1.5, 3} {
		z := cmplx.Exp(complex(0, th))
		assertFRClose(t, fmt.Sprintf("CL(e^j%g)", th), mixsynFR(r.CL, z), mixsynFR(hand, z), 1e-7)
	}
	norm := discreteHinfNormSweep(hand)
	if math.Abs(r.Gamma-norm) > 1e-6*norm {
		t.Fatalf("Gamma = %.12g, unit-circle ‖CL‖∞ = %.12g", r.Gamma, norm)
	}
	if r.Gamma > r.Info.GammaOpt*(1+1e-9) || r.Info.GammaOpt > r.Gamma*(1+2e-4) {
		t.Fatalf("Gamma %.12g, HinfSyn bound %.12g", r.Gamma, r.Info.GammaOpt)
	}
	return r
}

func TestMixsynDiscrete(t *testing.T) {
	const dt = 0.1
	G := mixsynTF(t, []float64{0.5, -0.2}, []float64{1, -1.2, 0.5}, dt)
	W1, err := Makeweight(10, []float64{1, 0.5}, 0.1, dt, 2)
	if err != nil {
		t.Fatal(err)
	}
	W2, err := Makeweight(0.1, []float64{5}, 3, dt, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertDiscreteMixsyn(t, G, W1, W2, nil)

	// D12 = 0 here, but P12(−1) = [−W1(−1)·G(−1); W3(−1)·G(−1)] ≠ 0 satisfies
	// the discrete rank condition, so W2 may be nil for a strictly proper G.
	W3, err := Makeweight(0.1, []float64{5}, 2, dt, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertDiscreteMixsyn(t, G, W1, nil, W3)
}

func TestMixsynDiscreteMIMO(t *testing.T) {
	const dt = 0.05
	Gc := mimoMixsynPlant(t, 0.5)
	G, err := Gc.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	dw := func(dc float64, fm []float64, hf float64) *System {
		W, err := Makeweight(dc, fm, hf, dt, 1)
		if err != nil {
			t.Fatal(err)
		}
		return W
	}
	W1 := diagWeight(t, dw(20, []float64{0.5}, 0.2), dw(10, []float64{1}, 0.3))
	W2 := diagWeight(t, dw(0.1, []float64{20, 0.5}, 2), dw(0.2, []float64{10, 0.5}, 1))
	W3 := diagWeight(t, dw(0.05, []float64{8}, 5), dw(0.05, []float64{8}, 5))
	assertDiscreteMixsyn(t, G, W1, W2, W3)
}

// With W2 = W3 = nil and a biproper G the problem is square one-block: the
// central controller has Dk = G(∞)⁻¹, so I + D22·Dk = I − G(∞)·Dk = 0 and its
// loop shift is ill-posed. HinfSyn returns a non-central controller instead.
func TestMixsynW1OnlyBiproperPlant(t *testing.T) {
	W1 := mixsynWeight(t, 10, []float64{1}, 0.1)
	t.Run("nonminimum phase", func(t *testing.T) {
		G := mixsynTF(t, []float64{0.5, 1, -2}, []float64{1, 3, 2}, 0)
		assertMixsynOracle(t, G, W1, nil, nil)
	})
	t.Run("MIMO", func(t *testing.T) {
		// G has a transmission zero at s ≈ 1.19.
		G, err := New(
			mat.NewDense(3, 3, []float64{-1, 0.4, 0, -0.3, -2, 0.5, 0.2, 0, -3}),
			mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, 0, -0.7}),
			mat.NewDense(2, 3, []float64{1, -0.2, 0.3, 0, -1, 0.8}),
			mat.NewDense(2, 2, []float64{0.5, 0.1, -0.2, 0.4}),
			0,
		)
		if err != nil {
			t.Fatal(err)
		}
		W := diagWeight(t, W1, mixsynWeight(t, 5, []float64{0.5}, 0.2))
		assertMixsynOracle(t, G, W, nil, nil)
	})
	t.Run("discrete", func(t *testing.T) {
		const dt = 0.1
		Gc := mixsynTF(t, []float64{0.5, 1, -2}, []float64{1, 3, 2}, 0)
		G, err := Gc.C2D(dt, C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		Wd, err := Makeweight(10, []float64{1}, 0.1, dt, 1)
		if err != nil {
			t.Fatal(err)
		}
		assertDiscreteMixsyn(t, G, Wd, nil, nil)
	})
	// A minimum-phase G admits S → 0 (K → ∞), so the infimum is 0 and
	// unattained (6WORAF); the controller must still be stabilizing and meet
	// the γ it reports.
	t.Run("minimum phase", func(t *testing.T) {
		assertZeroInfimumMixsyn(t, mixsynTF(t, []float64{0.5, 1, 2}, []float64{1, 3, 2}, 0), W1)
	})
	t.Run("minimum phase MIMO", func(t *testing.T) {
		G, W := minimumPhaseMIMOW1Only(t)
		assertZeroInfimumMixsyn(t, G, W)
	})
	t.Run("minimum phase discrete", func(t *testing.T) {
		const dt = 0.1
		G, err := mixsynTF(t, []float64{0.5, 1, 2}, []float64{1, 3, 2}, 0).C2D(dt, C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		Wd, err := Makeweight(10, []float64{1}, 0.1, dt, 1)
		if err != nil {
			t.Fatal(err)
		}
		assertZeroInfimumMixsyn(t, G, Wd)
	})
	t.Run("minimum phase discrete MIMO", func(t *testing.T) {
		const dt = 0.05
		Gc, _ := minimumPhaseMIMOW1Only(t)
		G, err := Gc.C2D(dt, C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		W1d, err := Makeweight(10, []float64{1}, 0.1, dt, 1)
		if err != nil {
			t.Fatal(err)
		}
		W2d, err := Makeweight(5, []float64{0.5}, 0.2, dt, 1)
		if err != nil {
			t.Fatal(err)
		}
		assertZeroInfimumMixsyn(t, G, diagWeight(t, W1d, W2d))
	})
}

// minimumPhaseMIMOW1Only is the 6WORAF repro: a biproper G with transmission
// zeros −2.33 and −2.85 ± 2.03i and a diagonal W1.
func minimumPhaseMIMOW1Only(t *testing.T) (G, W1 *System) {
	t.Helper()
	G, err := New(
		mat.NewDense(3, 3, []float64{-1, 0.4, 0, -0.3, -2, 0.5, 0.2, 0, -3}),
		mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, 0, -0.7}),
		mat.NewDense(2, 3, []float64{1, -0.2, 0.3, 0, 1, 0.8}),
		mat.NewDense(2, 2, []float64{0.5, 0.1, -0.2, 0.4}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return G, diagWeight(t, mixsynWeight(t, 10, []float64{1}, 0.1), mixsynWeight(t, 5, []float64{0.5}, 0.2))
}

// assertZeroInfimumMixsyn checks Mixsyn(G, W1, nil, nil) for a problem whose
// infimum is 0 and unattained: the hand-built loop is stable, its H∞ norm (by
// Hamiltonian bisection, after an exact bilinear map for a discrete G) is
// Gamma and at most HinfSyn's bound, and that bound is small.
func assertZeroInfimumMixsyn(t *testing.T, G, W1 *System) {
	t.Helper()
	r, err := Mixsyn(G, W1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	A, B, C, D := mixsynHandLoop(t, G, r.K, W1, nil, nil)
	if G.IsDiscrete() {
		assertSchurStable(t, A)
		A, B, C, D = bilinearToContinuous(t, A, B, C, D)
	}
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		t.Fatal("closed-loop eigenvalues failed")
	}
	for _, ev := range eig.Values(nil) {
		if real(ev) >= 0 {
			t.Fatalf("closed loop unstable: pole %v", ev)
		}
	}
	norm := hinfNormBisection(t, A, B, C, D)
	if math.Abs(r.Gamma-norm) > 1e-6*norm || norm > r.Info.GammaOpt*(1+1e-9) {
		t.Fatalf("Gamma = %.12g, bisection ‖CL‖∞ = %.12g, HinfSyn bound %.12g", r.Gamma, norm, r.Info.GammaOpt)
	}
	if r.Info.GammaOpt > 1e-3 {
		t.Fatalf("HinfSyn bound %g, want near the zero infimum", r.Info.GammaOpt)
	}
}

// bilinearToContinuous maps the discrete (A, B, C, D) by z = (1+s)/(1−s),
// which preserves stability and the H∞ norm.
func bilinearToContinuous(t *testing.T, A, B, C, D *mat.Dense) (Ac, Bc, Cc, Dc *mat.Dense) {
	t.Helper()
	n, _ := A.Dims()
	IpA := mat.DenseCopyOf(A)
	for i := range n {
		IpA.Set(i, i, IpA.At(i, i)+1)
	}
	var inv mat.Dense
	if err := inv.Inverse(IpA); err != nil {
		t.Fatal(err)
	}
	Ac, Bc, Cc, Dc = new(mat.Dense), new(mat.Dense), new(mat.Dense), new(mat.Dense)
	Ac.Mul(&inv, A)
	Ac.Sub(Ac, &inv)
	Bc.Mul(&inv, B)
	Bc.Scale(math.Sqrt2, Bc)
	Cc.Mul(C, &inv)
	Cc.Scale(math.Sqrt2, Cc)
	var CB mat.Dense
	CB.Mul(Cc, B)
	CB.Scale(1/math.Sqrt2, &CB)
	Dc.Sub(D, &CB)
	return Ac, Bc, Cc, Dc
}

// In the zero-optimum regime (minimum-phase G, W1 only) Mixsyn's CL and
// Gamma are HinfSyn's, which match LFT + HinfNorm and are attained at
// PeakFrequency.
func TestMixsynReusesHinfSynClosedLoop(t *testing.T) {
	G := mixsynTF(t, []float64{0.5, 1, 2}, []float64{1, 3, 2}, 0)
	W1 := mixsynWeight(t, 10, []float64{1}, 0.1)
	Gd, err := G.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	W1d, err := Makeweight(10, []float64{1}, 0.1, 0.1, 1)
	if err != nil {
		t.Fatal(err)
	}
	Gm, Wm := minimumPhaseMIMOW1Only(t)
	for _, tc := range []struct {
		name  string
		G, W1 *System
	}{
		{"continuous", G, W1},
		{"discrete", Gd, W1d},
		{"MIMO", Gm, Wm},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Mixsyn(tc.G, tc.W1, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if r.CL != r.Info.CL || r.Gamma != r.Info.Gamma {
				t.Fatalf("Mixsyn CL/Gamma not HinfSyn's: Gamma %v, Info.Gamma %v", r.Gamma, r.Info.Gamma)
			}
			if r.Info.GammaOpt > 1e-3 {
				t.Fatalf("GammaOpt %g, want near the zero infimum", r.Info.GammaOpt)
			}
			P, err := Augw(tc.G, tc.W1, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			_, nu, ny := tc.G.Dims()
			assertHinfSynOutputs(t, P, ny, nu, r.Info)
		})
	}
}
