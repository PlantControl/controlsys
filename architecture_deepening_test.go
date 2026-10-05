package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestArchitectureLocalApproximationJacobiansDriveLinearizeAndEKF(t *testing.T) {
	x0 := mat.NewVecDense(2, []float64{0.4, -0.2})
	u0 := mat.NewVecDense(1, []float64{0.3})

	state := func(x, u *mat.VecDense) *mat.VecDense {
		x1, x2, u1 := x.AtVec(0), x.AtVec(1), u.AtVec(0)
		return mat.NewVecDense(2, []float64{
			math.Sin(x1) + x2*u1,
			x1*x1 - 0.5*x2 + u1,
		})
	}
	output := func(x, u *mat.VecDense) *mat.VecDense {
		return mat.NewVecDense(1, []float64{x.AtVec(0)*x.AtVec(1) + 2*u.AtVec(0)})
	}

	lin, err := Linearize(&NonlinearModel{F: state, H: output, N: 2, M: 1, P: 1}, x0, u0)
	if err != nil {
		t.Fatal(err)
	}

	explicitA := mat.NewDense(2, 2, []float64{
		math.Cos(x0.AtVec(0)), u0.AtVec(0),
		2 * x0.AtVec(0), -0.5,
	})
	explicitC := mat.NewDense(1, 2, []float64{x0.AtVec(1), x0.AtVec(0)})
	assertDenseApprox(t, lin.A, explicitA, 1e-7)
	assertDenseApprox(t, lin.C, explicitC, 1e-7)

	P0 := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 2})
	Q := mat.NewDense(2, 2, []float64{0.01, 0, 0, 0.02})
	R := mat.NewDense(1, 1, []float64{0.1})
	ekf, err := NewEKF(&EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense { return state(x, u) },
		H: func(x *mat.VecDense) *mat.VecDense { return output(x, u0) },
		FJac: func(x, u *mat.VecDense) *mat.Dense {
			return mat.NewDense(2, 2, []float64{
				math.Cos(x.AtVec(0)), u.AtVec(0),
				2 * x.AtVec(0), -0.5,
			})
		},
		HJac: func(x *mat.VecDense) *mat.Dense {
			return mat.NewDense(1, 2, []float64{x.AtVec(1), x.AtVec(0)})
		},
		Q: Q,
		R: R,
	}, x0, P0)
	if err != nil {
		t.Fatal(err)
	}
	if err := ekf.Predict(u0); err != nil {
		t.Fatal(err)
	}

	var ap, want mat.Dense
	ap.Mul(lin.A, P0)
	want.Mul(&ap, lin.A.T())
	want.Add(&want, Q)
	assertDenseApprox(t, ekf.P, &want, 1e-7)
}

func TestArchitectureLocalApproximationRejectsBadCallbackShapes(t *testing.T) {
	x0 := mat.NewVecDense(2, nil)
	u0 := mat.NewVecDense(1, nil)
	if _, err := Linearize(&NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense { return nil },
		H: func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
		N: 2, M: 1, P: 1,
	}, x0, u0); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("Linearize nil state callback error = %v, want ErrDimensionMismatch", err)
	}

	P0 := mat.NewDense(2, 2, nil)
	Q := mat.NewDense(2, 2, nil)
	R := mat.NewDense(1, 1, nil)
	ekf, err := NewEKF(&EKFModel{
		F:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(2, nil) },
		H:    func(x *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) },
		Q:    Q,
		R:    R,
	}, x0, P0)
	if err != nil {
		t.Fatal(err)
	}
	if err := ekf.Predict(u0); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("EKF wrong Jacobian error = %v, want ErrDimensionMismatch", err)
	}
}

func TestArchitectureFreqRespEstHandlesStridedSISOInput(t *testing.T) {
	dt := 0.01
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}

	n := 2048
	uData := make([]float64, n)
	backingU := mat.NewDense(2, n, nil)
	uRow := backingU.Slice(1, 2, 0, n).(*mat.Dense)
	uRaw := uRow.RawMatrix()
	for k := range n {
		uData[k] = math.Sin(0.17*float64(k)) + 0.5*math.Sin(0.43*float64(k))
		uRaw.Data[k] = uData[k]
	}
	uContig := mat.NewDense(1, n, uData)
	resp, err := sys.Simulate(uContig, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	backingY := mat.NewDense(2, n, nil)
	yRow := backingY.Slice(1, 2, 0, n).(*mat.Dense)
	yRow.Copy(resp.Y)

	contig, err := FreqRespEst(uContig, resp.Y, dt, &FreqRespEstOpts{NFFT: 512})
	if err != nil {
		t.Fatal(err)
	}
	strided, err := FreqRespEst(uRow, yRow, dt, &FreqRespEstOpts{NFFT: 512})
	if err != nil {
		t.Fatal(err)
	}
	if len(contig.Omega) != len(strided.Omega) {
		t.Fatalf("strided frequency count = %d, want %d", len(strided.Omega), len(contig.Omega))
	}
	for f := range contig.Omega {
		want := contig.H.At(f, 0, 0)
		got := strided.H.At(f, 0, 0)
		if cmplx.Abs(got-want) > 1e-12 {
			t.Fatalf("freq %d: strided estimate = %v, want %v", f, got, want)
		}
	}
}

func TestArchitecturePolynomialChannelAlgebraCoversLeadingZerosAndZeroGain(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{
			{{0, 0, 1, 2}, {0}},
		},
		Den: [][]float64{{1, 3, 2}},
		Dt:  0,
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	if len(z.Zeros[0][0]) != 1 || cmplx.Abs(z.Zeros[0][0][0]+2) > 1e-12 {
		t.Fatalf("leading-zero channel zeros = %v, want [-2]", z.Zeros[0][0])
	}
	if len(z.Zeros[0][1]) != 0 || z.Gain[0][1] != 0 {
		t.Fatalf("zero-gain channel = zeros %v gain %v, want no zeros and zero gain", z.Zeros[0][1], z.Gain[0][1])
	}

	roundtrip, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []complex128{1i, 2 + 0.5i} {
		want := tf.Eval(s)
		got := roundtrip.Eval(s)
		for j := range want[0] {
			if cmplx.Abs(got[0][j]-want[0][j]) > 1e-10 {
				t.Fatalf("channel %d at %v = %v, want %v", j, s, got[0][j], want[0][j])
			}
		}
	}
}

func TestArchitectureZerosUseInvariantZeroSemantics(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{
			0, 1,
			-6, -5,
		},
		[]float64{0, 1},
		[]float64{2, 1},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if num, _ := Poly(tf.TF.Num[0][0]).Roots(); len(num) != 0 {
		t.Fatalf("minimal TF numerator roots %v, want none ((s+2) cancels)", num)
	}
	got, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if want := []complex128{-2}; !complexSetsApprox(got, want, 1e-10) {
		t.Fatalf("zeros = %v, want invariant zeros %v (MATLAB zero/tzero)", got, want)
	}
}

func complexSetsApprox(got, want []complex128, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	used := make([]bool, len(want))
	for _, g := range got {
		found := false
		for i, w := range want {
			if !used[i] && cmplx.Abs(g-w) <= tol {
				used[i] = true
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func TestArchitectureControllerObserverAssemblyPreservesDiscreteMIMOBehavior(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0.7, 0.2, 0,
			-0.1, 0.6, 0.15,
			0.05, -0.2, 0.5,
		},
		[]float64{
			0.2, 0,
			0.1, 0.3,
			0, 0.2,
		},
		[]float64{
			1, 0.1, 0,
			0, 1, -0.2,
		},
		[]float64{0, 0, 0, 0},
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Lqg(sys, eye(5), eye(5), nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Controller.Dt != sys.Dt {
		t.Fatalf("controller sample time = %v, want %v", res.Controller.Dt, sys.Dt)
	}
	n, m, p := res.Controller.Dims()
	if n != 3 || m != 2 || p != 2 {
		t.Fatalf("controller dims = (%d,%d,%d), want (3,2,2)", n, m, p)
	}
	kr, kc := res.K.Dims()
	if kr != 2 || kc != 3 {
		t.Fatalf("state-feedback gain dims = (%d,%d), want (2,3)", kr, kc)
	}
	lr, lc := res.L.Dims()
	if lr != 3 || lc != 2 {
		t.Fatalf("observer gain dims = (%d,%d), want (3,2)", lr, lc)
	}
	cl, err := Feedback(sys, res.Controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Fatalf("observer-based closed-loop poles = %v, want stable", poles)
	}
}

func TestArchitectureControllerObserverAssemblyRejectsDescriptorModel(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	_, err = Lqg(sys, eye(3), eye(3), nil)
	if !errors.Is(err, ErrDescriptorRiccati) {
		t.Fatalf("Lqg descriptor error = %v, want ErrDescriptorRiccati", err)
	}
}
