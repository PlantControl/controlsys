package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestH2Syn_Simple(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	P.InputName = []string{"disturbance", "control"}
	P.OutputName = []string{"performance1", "performance2", "measurement"}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	kn, km, kp := res.K.Dims()
	if kn != 2 {
		t.Errorf("controller states: got %d, want 2", kn)
	}
	if km != 1 {
		t.Errorf("controller inputs: got %d, want 1", km)
	}
	if kp != 1 {
		t.Errorf("controller outputs: got %d, want 1", kp)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

	if res.X == nil {
		t.Error("X is nil")
	}
	if res.Y == nil {
		t.Error("Y is nil")
	}
	if !stringSlicesEqual(res.K.InputName, []string{"measurement"}) {
		t.Fatalf("controller input names = %v, want [measurement]", res.K.InputName)
	}
	if !stringSlicesEqual(res.K.OutputName, []string{"control"}) {
		t.Fatalf("controller output names = %v, want [control]", res.K.OutputName)
	}
}

func TestH2Syn_Stability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := res.K.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("controller is not stable")
	}

	for _, p := range res.CLPoles {
		if real(p) >= -1e-10 {
			t.Errorf("closed-loop pole not in open left half-plane: %v", p)
		}
	}
}

func TestH2Syn_DimensionErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, nil)

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name  string
		nmeas int
		ncont int
	}{
		{"nmeas=0", 0, 1},
		{"ncont=0", 1, 0},
		{"nmeas too large", 4, 1},
		{"ncont too large", 1, 3},
		{"nmeas=p", 3, 1},
		{"ncont=m", 1, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := H2Syn(P, tt.nmeas, tt.ncont)
			if !errors.Is(err, ErrInvalidPartition) {
				t.Errorf("got %v, want ErrInvalidPartition", err)
			}
		})
	}
}

func TestH2Syn_D11NonZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		0, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = H2Syn(P, 1, 1)
	if !errors.Is(err, ErrNoFiniteH2Norm) {
		t.Errorf("got %v, want ErrNoFiniteH2Norm", err)
	}
}

func TestH2Syn_D22NonZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0.25,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	closedLoop(t, P, res.K, 1, 1)
}

func TestH2Syn_Unstabilizable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = H2Syn(P, 1, 1)
	if !errors.Is(err, ErrNotStabilizable) {
		t.Errorf("got %v, want ErrNotStabilizable", err)
	}
}

func TestH2Syn_SecondOrder(t *testing.T) {
	// Plant: x' = Ax + B1*w + B2*u, z = C1*x + D12*u, y = C2*x + D21*w
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	}) // B1=2×2, B2=2×1
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	}) // C1=2×2, C2=1×2
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0, // D11=0
		0, 0, 1, // D12=[0;1]
		0.1, 0.1, 0, // D21=[0.1 0.1], D22=0
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	// nmeas=1 (1 measurement), ncont=1 (1 control)
	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) > 1e-8 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

}

func TestH2Syn_D22LoopShiftMixedSensitivity(t *testing.T) {
	P := mixedSensitivityFeedthroughPlant(t)
	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	norm, err := H2Norm(closedLoop(t, P, res.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}

	P0 := withoutD22(t, P, 1, 1)
	ref, err := H2Syn(P0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	refNorm, err := H2Norm(closedLoop(t, P0, ref.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-refNorm) > 1e-8*refNorm {
		t.Fatalf("closed-loop H2 norm %v, want D22-free optimum %v", norm, refNorm)
	}
}

// discreteH2NormSmith is the H2 norm of a Schur-stable discrete model from
// the observability Gramian X = AᵀXA + CᵀC by Smith doubling, independent of
// the library's Lyapunov solvers: ‖G‖₂² = tr(BᵀXB) + ‖D‖_F².
func discreteH2NormSmith(t *testing.T, sys *System) float64 {
	t.Helper()
	n, m, p := sys.Dims()
	X := mat.NewDense(n, n, nil)
	X.Mul(sys.C.T(), sys.C)
	Ak := mat.DenseCopyOf(sys.A)
	for range 80 {
		var AX, AXA, A2 mat.Dense
		AX.Mul(Ak.T(), X)
		AXA.Mul(&AX, Ak)
		X.Add(X, &AXA)
		A2.Mul(Ak, Ak)
		Ak = &A2
	}
	if mat.Norm(Ak, 1) > 1e-12 {
		t.Fatalf("Smith iteration did not converge: ‖A^(2^k)‖ = %g", mat.Norm(Ak, 1))
	}
	var BX, BXB mat.Dense
	BX.Mul(sys.B.T(), X)
	BXB.Mul(&BX, sys.B)
	j := mat.Trace(&BXB)
	for i := range p {
		for k := range m {
			j += sys.D.At(i, k) * sys.D.At(i, k)
		}
	}
	return math.Sqrt(j)
}

// assertDiscreteH2Optimal checks the closed loop is Schur stable and that K
// is a stationary minimum of the closed-loop H2 norm: random perturbations of
// (Ak, Bk, Ck, Dk) never lower it and its central-difference slope vanishes.
func assertDiscreteH2Optimal(t *testing.T, P *System, nmeas, ncont int) *H2SynResult {
	t.Helper()
	res, err := H2Syn(P, nmeas, ncont)
	if err != nil {
		t.Fatal(err)
	}
	if res.K.Dt != P.Dt {
		t.Fatalf("K.Dt = %g, want %g", res.K.Dt, P.Dt)
	}
	cl := handLFT(t, P, res.K, nmeas, ncont)
	assertSchurStable(t, cl.A)
	j0 := discreteH2NormSmith(t, cl)
	if lib, err := H2Norm(cl); err != nil || math.Abs(lib-j0) > 1e-9*j0 {
		t.Fatalf("H2Norm = %v, %v; Smith = %v", lib, err, j0)
	}
	rng := rand.New(rand.NewSource(7))
	perturb := func(M *mat.Dense, eps float64, dir *mat.Dense) *mat.Dense {
		out := mat.DenseCopyOf(dir)
		out.Scale(eps, out)
		out.Add(out, M)
		return out
	}
	random := func(M *mat.Dense) *mat.Dense {
		r, c := M.Dims()
		d := mat.NewDense(r, c, nil)
		for i := range r {
			for j := range c {
				d.Set(i, j, rng.NormFloat64())
			}
		}
		return d
	}
	const eps = 1e-4
	for k := range 12 {
		K := res.K
		dA, dB, dC, dD := random(K.A), random(K.B), random(K.C), random(K.D)
		j := func(e float64) float64 {
			Kp, err := New(perturb(K.A, e, dA), perturb(K.B, e, dB), perturb(K.C, e, dC), perturb(K.D, e, dD), K.Dt)
			if err != nil {
				t.Fatal(err)
			}
			return discreteH2NormSmith(t, handLFT(t, P, Kp, nmeas, ncont))
		}
		jp, jm := j(eps), j(-eps)
		if jp < j0*(1-1e-12) || jm < j0*(1-1e-12) {
			t.Fatalf("direction %d: perturbed norm %v / %v below optimum %v", k, jp, jm, j0)
		}
		if slope := (jp - jm) / (2 * eps); math.Abs(slope) > 1e-5*j0 {
			t.Fatalf("direction %d: slope %g, want 0 at the optimum", k, slope)
		}
	}
	return res
}

func TestH2Syn_DiscreteMIMO(t *testing.T) {
	for _, d11 := range []float64{0, 0.4} {
		t.Run(fmt.Sprintf("D11=%g", d11), func(t *testing.T) {
			assertDiscreteH2Optimal(t, discreteHinfTestPlant(t, d11), 2, 1)
		})
	}
}

func TestH2Syn_DiscreteUnstablePlant(t *testing.T) {
	P, err := New(
		mat.NewDense(2, 2, []float64{1.1, 0.4, -0.2, 0.7}),
		mat.NewDense(2, 3, []float64{1, 0, 0.5, 0.3, 0, 1}),
		mat.NewDense(3, 2, []float64{1, 0.2, 0, 0, 0.5, 1}),
		mat.NewDense(3, 3, []float64{0.1, 0, 0, 0, 0, 1, 0, 1, 0.2}),
		0.05,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertDiscreteH2Optimal(t, P, 1, 1)
}

// continuousH2NormKron is the H2 norm of a Hurwitz, strictly proper
// continuous model from the controllability Gramian solved as the Kronecker
// system (I⊗A + A⊗I)·vec(W) = −vec(BBᵀ): ‖G‖₂² = tr(CWCᵀ).
func continuousH2NormKron(t *testing.T, sys *System) float64 {
	t.Helper()
	n, _, _ := sys.Dims()
	M := mat.NewDense(n*n, n*n, nil)
	for i := range n {
		for j := range n {
			for k := range n {
				M.Set(i*n+j, k*n+j, M.At(i*n+j, k*n+j)+sys.A.At(i, k))
				M.Set(i*n+j, i*n+k, M.At(i*n+j, i*n+k)+sys.A.At(j, k))
			}
		}
	}
	var BBt mat.Dense
	BBt.Mul(sys.B, sys.B.T())
	rhs := mat.NewVecDense(n*n, nil)
	for i := range n {
		for j := range n {
			rhs.SetVec(i*n+j, -BBt.At(i, j))
		}
	}
	var w mat.VecDense
	if err := w.SolveVec(M, rhs); err != nil {
		t.Fatal(err)
	}
	var CW, CWCt mat.Dense
	CW.Mul(sys.C, mat.NewDense(n, n, w.RawVector().Data))
	CWCt.Mul(&CW, sys.C.T())
	return math.Sqrt(mat.Trace(&CWCt))
}

// H2Syn returns MATLAB h2syn's CL and gamma = ‖CL‖₂.
func TestH2Syn_ClosedLoopGamma(t *testing.T) {
	continuous := func(d22 float64) func(*testing.T) *System {
		return func(t *testing.T) *System {
			P, err := New(
				mat.NewDense(2, 2, []float64{0, 1, -2, -1}),
				mat.NewDense(2, 2, []float64{0, 0, 1, 1}),
				mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0}),
				mat.NewDense(3, 2, []float64{0, 0, 0, 1, 0.1, d22}),
				0,
			)
			if err != nil {
				t.Fatal(err)
			}
			return P
		}
	}
	for _, tc := range []struct {
		name         string
		plant        func(*testing.T) *System
		nmeas, ncont int
	}{
		{"continuous", continuous(0), 1, 1},
		{"continuous D22", continuous(0.25), 1, 1},
		{"discrete D11 D22", func(t *testing.T) *System { return discreteHinfTestPlant(t, 0.4) }, 2, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			P := tc.plant(t)
			res, err := H2Syn(P, tc.nmeas, tc.ncont)
			if err != nil {
				t.Fatal(err)
			}
			cl, err := LFT(P, res.K, LFTFeedback{Nu: tc.ncont, Ny: tc.nmeas})
			if err != nil {
				t.Fatal(err)
			}
			norm, err := H2Norm(cl)
			if err != nil {
				t.Fatal(err)
			}
			if res.Gamma != norm {
				t.Fatalf("Gamma = %v, H2Norm(LFT(P, K)) = %v", res.Gamma, norm)
			}
			var want float64
			if P.IsDiscrete() {
				want = discreteH2NormSmith(t, res.CL)
			} else {
				want = continuousH2NormKron(t, res.CL)
			}
			if math.Abs(res.Gamma-want) > 1e-9*want {
				t.Fatalf("Gamma = %.15g, Gramian oracle %.15g", res.Gamma, want)
			}
			for _, w := range []float64{0.001, 0.3, 1, 2.7, 10, 31} {
				s := complex(0, w)
				if P.IsDiscrete() {
					if w*P.Dt > math.Pi {
						continue
					}
					s = cmplx.Exp(complex(0, w*P.Dt))
				}
				assertFRClose(t, fmt.Sprintf("CL(%v)", s), mixsynFR(res.CL, s), lowerLFTOracle(P, res.K, tc.nmeas, tc.ncont, s), 1e-9)
			}
		})
	}
}
