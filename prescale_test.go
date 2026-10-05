package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPrescale_Simple(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleBodePreserved(t, sys, pr.Sys, 1e-6)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-10)

	for i, s := range pr.Info.StateScale {
		if math.Abs(s-1.0) > 1e-10 {
			t.Errorf("StateScale[%d] = %g, want ~1", i, s)
		}
	}
}

func TestPrescale_LargeGainSpread(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -1000}),
		mat.NewDense(2, 1, []float64{1, 1000}),
		mat.NewDense(1, 2, []float64{1000, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-6)
	checkPrescaleBodePreserved(t, sys, pr.Sys, 1e-6)

	if len(pr.Info.InputScale) != 1 {
		t.Fatalf("InputScale length = %d, want 1", len(pr.Info.InputScale))
	}
	if len(pr.Info.OutputScale) != 1 {
		t.Fatalf("OutputScale length = %d, want 1", len(pr.Info.OutputScale))
	}
}

func TestPrescale_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 100, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 100}),
		mat.NewDense(2, 2, []float64{100, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-6)
}

func TestPrescale_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := pr.Sys.Dims()
	if n != 0 {
		t.Errorf("expected pure gain, got n=%d", n)
	}
}

func TestPrescale_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-10)
}

func checkPrescalePolesPreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	p1, err := orig.Poles()
	if err != nil {
		t.Fatalf("orig poles: %v", err)
	}
	p2, err := scaled.Poles()
	if err != nil {
		t.Fatalf("scaled poles: %v", err)
	}
	if len(p1) != len(p2) {
		t.Fatalf("pole count %d != %d", len(p1), len(p2))
	}
	sort.Slice(p1, func(i, j int) bool {
		if real(p1[i]) != real(p1[j]) {
			return real(p1[i]) < real(p1[j])
		}
		return imag(p1[i]) < imag(p1[j])
	})
	sort.Slice(p2, func(i, j int) bool {
		if real(p2[i]) != real(p2[j]) {
			return real(p2[i]) < real(p2[j])
		}
		return imag(p2[i]) < imag(p2[j])
	})
	for i := range p1 {
		if cmplx.Abs(p1[i]-p2[i]) > tol {
			t.Errorf("pole[%d]: %v != %v", i, p1[i], p2[i])
		}
	}
}

func checkPrescaleBodePreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	freqs := []float64{0.01, 0.1, 1, 10, 100}
	for _, w := range freqs {
		var s complex128
		if orig.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*orig.Dt))
		}
		g1, err1 := orig.EvalFr(s)
		g2, err2 := scaled.EvalFr(s)
		if err1 != nil || err2 != nil {
			t.Errorf("EvalFr error at w=%g", w)
			continue
		}
		_, m1, p1 := orig.Dims()
		for i := range p1 {
			for j := range m1 {
				diff := cmplx.Abs(g1[i][j] - g2[i][j])
				mag := cmplx.Abs(g1[i][j])
				relTol := tol
				if mag > 1e-10 {
					relTol = tol * mag
				}
				if diff > relTol {
					t.Errorf("w=%g: G[%d,%d] diff=%g > tol=%g", w, i, j, diff, relTol)
				}
			}
		}
	}
}

func checkPrescaleDCGainPreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	dc1, err := orig.DCGain()
	if err != nil {
		t.Fatalf("orig DCGain: %v", err)
	}
	dc2, err := scaled.DCGain()
	if err != nil {
		t.Fatalf("scaled DCGain: %v", err)
	}
	r, c := dc1.Dims()
	for i := range r {
		for j := range c {
			diff := math.Abs(dc1.At(i, j) - dc2.At(i, j))
			mag := math.Abs(dc1.At(i, j))
			relTol := tol
			if mag > 1e-10 {
				relTol = tol * mag
			}
			if diff > relTol {
				t.Errorf("DCGain[%d,%d]: %g != %g (diff=%g)", i, j, dc2.At(i, j), dc1.At(i, j), diff)
			}
		}
	}
}

func prescaleOracleFr(A, B, C, D *mat.Dense, s complex128) [][]complex128 {
	n, m := B.Dims()
	p, _ := C.Dims()
	M := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n+m)
		for j := range n {
			M[i][j] = complex(-A.At(i, j), 0)
		}
		M[i][i] += s
		for j := range m {
			M[i][n+j] = complex(B.At(i, j), 0)
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
	G := make([][]complex128, p)
	for i := range p {
		G[i] = make([]complex128, m)
		for j := range m {
			g := complex(D.At(i, j), 0)
			for k := range n {
				g += complex(C.At(i, k), 0) * X[k][j]
			}
			G[i][j] = g
		}
	}
	return G
}

func checkPrescaleContract(t *testing.T, sys *System, pr *PrescaleResult) {
	t.Helper()
	n, m, p := sys.Dims()
	ss := pr.Info.StateScale
	if len(ss) != n {
		t.Fatalf("len(StateScale) = %d, want %d", len(ss), n)
	}
	for i, v := range ss {
		if !(v > 0) || math.IsInf(v, 0) {
			t.Fatalf("StateScale[%d] = %g, want positive finite", i, v)
		}
	}
	const tol = 1e-12
	near := func(name string, got, want float64) {
		if math.Abs(got-want) > tol*math.Max(1, math.Abs(want)) {
			t.Errorf("%s = %g, want %g", name, got, want)
		}
	}
	for i := range n {
		for j := range n {
			near("As", pr.Sys.A.At(i, j), sys.A.At(i, j)*ss[j]/ss[i])
		}
		for j := range m {
			near("Bs", pr.Sys.B.At(i, j), sys.B.At(i, j)/ss[i])
		}
	}
	for i := range p {
		for j := range n {
			near("Cs", pr.Sys.C.At(i, j), sys.C.At(i, j)*ss[j])
		}
		for j := range m {
			near("Ds", pr.Sys.D.At(i, j), sys.D.At(i, j))
		}
	}
	for _, w := range []float64{0.01, 0.3, 1, 7, 100} {
		s := complex(0, w)
		if sys.IsDiscrete() {
			s = cmplx.Exp(complex(0, w*sys.Dt))
		}
		want := prescaleOracleFr(sys.A, sys.B, sys.C, sys.D, s)
		got, err := pr.Sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range p {
			for j := range m {
				if d := cmplx.Abs(got[i][j] - want[i][j]); d > 1e-9*math.Max(1, cmplx.Abs(want[i][j])) {
					t.Errorf("w=%g G[%d,%d] = %v, want %v", w, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

func TestPrescale_TriangularNoPermutation(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 100, 0.001, 0, -2, 1000, 0, 0, -3}),
		mat.NewDense(3, 1, []float64{1, 2, 3}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.StateName = []string{"x1", "x2", "x3"}
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}
	checkPrescaleContract(t, sys, pr)
	for i, nm := range []string{"x1", "x2", "x3"} {
		if i >= len(pr.Sys.StateName) || pr.Sys.StateName[i] != nm {
			t.Fatalf("StateName = %v, want [x1 x2 x3]", pr.Sys.StateName)
		}
	}
	if len(pr.Sys.InputName) != 1 || pr.Sys.InputName[0] != "u" || len(pr.Sys.OutputName) != 1 || pr.Sys.OutputName[0] != "y" {
		t.Errorf("IO names lost: %v %v", pr.Sys.InputName, pr.Sys.OutputName)
	}
}

func TestPrescale_MIMONonSymmetricFeedthrough(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-2, 3e3, 0,
		1e-3, -5, 40,
		0.5, -2e-2, -1,
	})
	B := mat.NewDense(3, 2, []float64{1, 0, 1e3, 2, 0, 1e-2})
	C := mat.NewDense(2, 3, []float64{1e-2, 1, 0, 0, 3, 1e2})
	D := mat.NewDense(2, 2, []float64{0.5, 0, -1, 2})
	for _, dt := range []float64{0, 0.05} {
		Ad := A
		if dt > 0 {
			Ad = mat.NewDense(3, 3, []float64{
				0.5, 2e3, 0,
				1e-4, 0.2, 30,
				0, 1e-3, -0.4,
			})
		}
		sys, err := New(Ad, B, C, D, dt)
		if err != nil {
			t.Fatal(err)
		}
		pr, err := Prescale(sys)
		if err != nil {
			t.Fatal(err)
		}
		checkPrescaleContract(t, sys, pr)
	}
}

func TestPrescale_RejectsDescriptorAndDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	desc := sys.Copy()
	desc.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	if _, err := Prescale(desc); err == nil {
		t.Error("descriptor: want error")
	}
	del := sys.Copy()
	del.InputDelay = []float64{0.3}
	if _, err := Prescale(del); err == nil {
		t.Error("input delay: want error")
	}
}
