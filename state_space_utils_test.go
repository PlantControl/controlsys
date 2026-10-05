package controlsys

import (
	"errors"
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestDescriptorConstructionAccessAndExplicitConversion(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-2, 1, -3, -4})
	B := mat.NewDense(2, 1, []float64{2, -4})
	C := mat.NewDense(1, 2, []float64{1, -1})
	D := mat.NewDense(1, 1, []float64{0.5})
	E := mat.NewDense(2, 2, []float64{2, 0, 0, 4})

	desc, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatalf("NewDescriptor: %v", err)
	}
	if !desc.IsDescriptor() {
		t.Fatal("expected descriptor model")
	}
	gotE := desc.DescriptorE()
	gotE.Set(0, 0, 99)
	if desc.E.At(0, 0) == 99 {
		t.Fatal("DescriptorE returned aliased matrix")
	}

	explicit, err := desc.ToExplicit()
	if err != nil {
		t.Fatalf("ToExplicit: %v", err)
	}
	wantA := mat.NewDense(2, 2, []float64{-1, 0.5, -0.75, -1})
	wantB := mat.NewDense(2, 1, []float64{1, -1})
	if explicit.IsDescriptor() {
		t.Fatal("explicit model still reports descriptor")
	}
	if !matEqual(explicit.A, wantA, 1e-12) || !matEqual(explicit.B, wantB, 1e-12) {
		t.Fatalf("explicit matrices mismatch\nA=%v\nB=%v", mat.Formatted(explicit.A), mat.Formatted(explicit.B))
	}

	singular, err := NewDescriptor(A, B, C, D, mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := singular.ToExplicit(); !errors.Is(err, ErrDescriptorSingular) {
		t.Fatalf("singular ToExplicit err = %v, want ErrDescriptorSingular", err)
	}
}

func TestStateSpaceUtilityWrappersPreserveBehaviorAndMetadata(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.InputName = []string{"left", "right"}
	sys.OutputName = []string{"position"}
	sys.StateName = []string{"fast", "slow"}

	truncated, err := sys.EliminateStates([]int{1}, Truncate)
	if err != nil {
		t.Fatalf("EliminateStates: %v", err)
	}
	if n, _, _ := truncated.Dims(); n != 1 {
		t.Fatalf("truncated order = %d, want 1", n)
	}
	if !sameStrings(truncated.InputName, sys.InputName) || !sameStrings(truncated.OutputName, sys.OutputName) {
		t.Fatalf("metadata lost input=%v output=%v", truncated.InputName, truncated.OutputName)
	}

	T := mat.NewDense(2, 2, []float64{1, 2, 0, 1})
	equivalent, err := sys.StateTransform(T)
	if err != nil {
		t.Fatalf("StateTransform: %v", err)
	}
	omega := []float64{0.1, 1.0, 3.0}
	baseResp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	equivResp, err := equivalent.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		for i := 0; i < baseResp.P; i++ {
			for j := 0; j < baseResp.M; j++ {
				if cmplx.Abs(baseResp.At(k, i, j)-equivResp.At(k, i, j)) > 1e-10 {
					t.Fatalf("transformed response mismatch at (%d,%d,%d)", k, i, j)
				}
			}
		}
	}
}

func TestFixedInputReductionAddsOffsetChannel(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.InputName = []string{"free", "bias"}
	sys.OutputName = []string{"y"}

	reduced, err := sys.FixedInputReduction(map[int]float64{1: 2.5}, "bias_offset")
	if err != nil {
		t.Fatalf("FixedInputReduction: %v", err)
	}
	if _, m, p := reduced.Dims(); m != 2 || p != 1 {
		t.Fatalf("reduced dims = (_, %d, %d), want (_, 2, 1)", m, p)
	}
	if !sameStrings(reduced.InputName, []string{"free", "bias_offset"}) {
		t.Fatalf("input names = %v", reduced.InputName)
	}

	uOriginal := mat.NewDense(4, 2, []float64{
		1, 2.5,
		0.5, 2.5,
		-0.25, 2.5,
		0, 2.5,
	})
	uReduced := mat.NewDense(4, 2, []float64{
		1, 1,
		0.5, 1,
		-0.25, 1,
		0, 1,
	})
	tGrid := []float64{0, 0.1, 0.2, 0.3}
	origResp, err := Lsim(sys, uOriginal, tGrid, nil)
	if err != nil {
		t.Fatal(err)
	}
	redResp, err := Lsim(reduced, uReduced, tGrid, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(origResp.Y, redResp.Y, 1e-12) {
		t.Fatalf("fixed-input response mismatch\norig=%v\nred=%v", mat.Formatted(origResp.Y), mat.Formatted(redResp.Y))
	}
}

func TestAugmentInternalDelayOutputs(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.OutputName = []string{"plant_y"}
	if err := sys.SetInternalDelay(
		[]float64{0.2},
		mat.NewDense(2, 1, []float64{0.5, -0.25}),
		mat.NewDense(1, 2, []float64{0.3, -0.4}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 2, []float64{0.2, 0.1}),
		mat.NewDense(1, 1, []float64{0}),
	); err != nil {
		t.Fatalf("SetInternalDelay: %v", err)
	}

	aug, err := sys.AugmentInternalDelayOutputs("delay")
	if err != nil {
		t.Fatalf("AugmentInternalDelayOutputs: %v", err)
	}
	if _, _, p := aug.Dims(); p != 2 {
		t.Fatalf("outputs = %d, want 2", p)
	}
	if !sameStrings(aug.OutputName, []string{"plant_y", "delay1"}) {
		t.Fatalf("output names = %v", aug.OutputName)
	}
	if !aug.HasInternalDelay() || aug.LFT.Tau[0] != 0.2 {
		t.Fatalf("internal delay not preserved: %v", aug.LFT)
	}
	if aug.LFT.C2.At(0, 0) != 0.3 || aug.LFT.D21.At(0, 1) != 0.1 {
		t.Fatal("internal-delay output matrices not preserved")
	}
}

func utilityTestSystem(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, -2, -3}),
		mat.NewDense(2, 2, []float64{1, -1, 0.5, 2}),
		mat.NewDense(1, 2, []float64{2, -0.5}),
		mat.NewDense(1, 2, []float64{0.25, -0.75}),
		0,
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return sys
}

func fixedInputPlant(t *testing.T, dt float64, descriptor, lft bool) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1, 0.5, 0.2, -2, -3, 0.4, 0.3, -0.6, -1.5})
	if dt > 0 {
		A.Scale(0.2, A)
	}
	sys, err := New(
		A,
		mat.NewDense(3, 3, []float64{1, -1, 0.3, 0.5, 2, -0.7, 0.2, 0.4, 1.1}),
		mat.NewDense(2, 3, []float64{2, -0.5, 0.3, 0.1, 1, -0.8}),
		mat.NewDense(2, 3, []float64{0.25, -0.75, 0.4, 0.6, 0.1, -0.3}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	if descriptor {
		sys.E = mat.NewDense(3, 3, []float64{2, 0.3, 0, 0.1, 1.5, 0.2, 0, -0.4, 1.2})
	}
	if lft {
		tau := 0.7
		if dt > 0 {
			tau = 3
		}
		if err := sys.SetInternalDelay([]float64{tau},
			mat.NewDense(3, 1, []float64{1, 0.5, -0.3}),
			mat.NewDense(1, 3, []float64{0.3, 1, -0.2}),
			mat.NewDense(2, 1, []float64{0.2, -0.4}),
			mat.NewDense(1, 3, []float64{0.5, 0.1, -0.6}),
			mat.NewDense(1, 1, []float64{0.1})); err != nil {
			t.Fatal(err)
		}
	}
	return sys
}

func fixedInputDelayFactor(sys *System, s complex128, tau float64) complex128 {
	if sys.Dt == 0 {
		return cmplx.Exp(-s * complex(tau, 0))
	}
	return cmplx.Pow(s, complex(-tau, 0))
}

func fixedInputAt(m *mat.Dense, i, j int) float64 {
	if m == nil {
		return 0
	}
	return m.At(i, j)
}

// fixedInputBlock returns C(sE−A)⁻¹B + D by dense complex elimination.
func fixedInputBlock(sys *System, s complex128, B *mat.Dense, cols int, C *mat.Dense, rows int, D *mat.Dense) [][]complex128 {
	n, _, _ := sys.Dims()
	M := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			M[i*n+j] = s*complex(e, 0) - complex(sys.A.At(i, j), 0)
		}
	}
	X := make([][]complex128, cols)
	for j := range cols {
		b := make([]complex128, n)
		for i := range n {
			b[i] = complex(B.At(i, j), 0)
		}
		X[j] = complexSolve(M, b, n)
	}
	G := make([][]complex128, rows)
	for i := range rows {
		G[i] = make([]complex128, cols)
		for j := range cols {
			v := complex(fixedInputAt(D, i, j), 0)
			for k := range n {
				v += complex(C.At(i, k), 0) * X[j][k]
			}
			G[i][j] = v
		}
	}
	return G
}

// fixedInputOracle evaluates the full delayed (descriptor, LFT) response at ω.
func fixedInputOracle(sys *System, w float64) [][]complex128 {
	_, m, p := sys.Dims()
	s := complex(0, w)
	if sys.Dt > 0 {
		s = cmplx.Exp(complex(0, w*sys.Dt))
	}
	G := fixedInputBlock(sys, s, sys.B, m, sys.C, p, sys.D)
	if sys.HasInternalDelay() {
		N := len(sys.LFT.Tau)
		H12 := fixedInputBlock(sys, s, sys.LFT.B2, N, sys.C, p, sys.LFT.D12)
		H21 := fixedInputBlock(sys, s, sys.B, m, sys.LFT.C2, N, sys.LFT.D21)
		H22 := fixedInputBlock(sys, s, sys.LFT.B2, N, sys.LFT.C2, N, sys.LFT.D22)
		th := make([]complex128, N)
		for k := range N {
			th[k] = fixedInputDelayFactor(sys, s, sys.LFT.Tau[k])
		}
		IM := make([]complex128, N*N)
		for i := range N {
			for j := range N {
				IM[i*N+j] = -H22[i][j] * th[j]
			}
			IM[i*N+i] += 1
		}
		for j := range m {
			rhs := make([]complex128, N)
			for k := range N {
				rhs[k] = H21[k][j]
			}
			X := complexSolve(IM, rhs, N)
			for i := range p {
				for k := range N {
					G[i][j] += H12[i][k] * th[k] * X[k]
				}
			}
		}
	}
	for i := range p {
		for j := range m {
			tau := fixedInputAt(sys.Delay, i, j)
			if sys.InputDelay != nil {
				tau += sys.InputDelay[j]
			}
			if sys.OutputDelay != nil {
				tau += sys.OutputDelay[i]
			}
			G[i][j] *= fixedInputDelayFactor(sys, s, tau)
		}
	}
	return G
}

func assertFixedInputReduction(t *testing.T, label string, sys *System, fixed map[int]float64) *System {
	t.Helper()
	_, m, p := sys.Dims()
	var keep []int
	for j := range m {
		if _, ok := fixed[j]; !ok {
			keep = append(keep, j)
		}
	}
	red, err := sys.FixedInputReduction(fixed, "offset")
	if err != nil {
		t.Fatalf("%s: FixedInputReduction: %v", label, err)
	}
	if err := red.Validate(); err != nil {
		t.Fatalf("%s: result invalid: %v", label, err)
	}
	if _, mr, pr := red.Dims(); mr != len(keep)+1 || pr != p {
		t.Fatalf("%s: dims m=%d p=%d, want m=%d p=%d", label, mr, pr, len(keep)+1, p)
	}
	omega := []float64{0.05, 0.4, 1.3, 2.9}
	got, err := red.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	for k, w := range omega {
		H := fixedInputOracle(sys, w)
		for i := range p {
			for c, j := range keep {
				if d := cmplx.Abs(got.At(k, i, c) - H[i][j]); d > 1e-9 {
					t.Fatalf("%s: ω=%g H(%d,%d) differs by %g", label, w, i, c, d)
				}
			}
			var off complex128
			for j, v := range fixed {
				off += complex(v, 0) * H[i][j]
			}
			if d := cmplx.Abs(got.At(k, i, len(keep)) - off); d > 1e-9 {
				t.Fatalf("%s: ω=%g offset(%d) = %v, want %v", label, w, i, got.At(k, i, len(keep)), off)
			}
		}
	}
	return red
}

func TestFixedInputReductionMatchesOracle(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, descriptor := range []bool{false, true} {
			for _, lft := range []bool{false, true} {
				for _, fixed := range []map[int]float64{{1: 0}, {1: 0.7}, {0: -1.3, 2: 0.6}} {
					label := fmt.Sprintf("dt=%g/desc=%v/lft=%v/%v", dt, descriptor, lft, fixed)
					assertFixedInputReduction(t, label, fixedInputPlant(t, dt, descriptor, lft), fixed)
				}
			}
		}
	}
}

func TestFixedInputReductionPropagatesSharedDelays(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		scale := 0.1
		if dt > 0 {
			scale = 1
		}
		sys := fixedInputPlant(t, dt, false, true)
		if err := sys.SetInputDelay([]float64{1 * scale, 3 * scale, 3 * scale}); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{2 * scale, 0}); err != nil {
			t.Fatal(err)
		}
		assertFixedInputReduction(t, fmt.Sprintf("dt=%g/input", dt), sys, map[int]float64{1: 0.7, 2: -0.4})

		sys = fixedInputPlant(t, dt, true, false)
		if err := sys.SetInputDelay([]float64{0, 2 * scale, 4 * scale}); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetDelay(mat.NewDense(2, 3, []float64{1 * scale, 2 * scale, 2 * scale, 0, 1 * scale, 1 * scale})); err != nil {
			t.Fatal(err)
		}
		assertFixedInputReduction(t, fmt.Sprintf("dt=%g/io/zero-valued", dt), sys, map[int]float64{1: 0.7, 2: 0})
		sys.InputDelay[2] = 2 * scale
		assertFixedInputReduction(t, fmt.Sprintf("dt=%g/io", dt), sys, map[int]float64{1: 0.7, 2: -0.4})
	}
}

func TestFixedInputReductionRejectsMismatchedDelays(t *testing.T) {
	sys := fixedInputPlant(t, 0, false, false)
	if err := sys.SetInputDelay([]float64{0, 0.2, 0.3}); err != nil {
		t.Fatal(err)
	}
	if _, err := sys.FixedInputReduction(map[int]float64{1: 1, 2: 1}, "off"); !errors.Is(err, ErrFixedInputDelayMismatch) {
		t.Fatalf("input delays: err = %v, want ErrFixedInputDelayMismatch", err)
	}
	sys = fixedInputPlant(t, 0, false, false)
	if err := sys.SetDelay(mat.NewDense(2, 3, []float64{0, 0.2, 0.2, 0, 0.1, 0.3})); err != nil {
		t.Fatal(err)
	}
	if _, err := sys.FixedInputReduction(map[int]float64{1: 1, 2: 1}, "off"); !errors.Is(err, ErrFixedInputDelayMismatch) {
		t.Fatalf("io delays: err = %v, want ErrFixedInputDelayMismatch", err)
	}
}

func TestFixedInputReductionInputNames(t *testing.T) {
	sys := fixedInputPlant(t, 0, false, false)
	red, err := sys.FixedInputReduction(map[int]float64{1: 1}, "")
	if err != nil {
		t.Fatal(err)
	}
	if red.InputName != nil {
		t.Fatalf("unnamed: InputName = %q, want nil", red.InputName)
	}
	red, err = sys.FixedInputReduction(map[int]float64{1: 1}, "off")
	if err != nil {
		t.Fatal(err)
	}
	if !sameStrings(red.InputName, []string{"", "", "off"}) {
		t.Fatalf("offset only: InputName = %q", red.InputName)
	}
}

func TestDescriptorEExplicitIsIdentity(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	E := sys.DescriptorE()
	if E == nil || !mat.Equal(E, mat.NewDense(2, 2, []float64{1, 0, 0, 1})) {
		t.Errorf("DescriptorE = %v, want I2", E)
	}
	E.Set(0, 0, 5)
	if sys.E != nil {
		t.Error("DescriptorE result aliases the model")
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if E := gain.DescriptorE(); E == nil || !E.IsEmpty() {
		t.Errorf("static gain DescriptorE = %v, want empty", E)
	}
}

func TestFixedInputReductionInvalidIndices(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sys.FixedInputReduction(map[int]float64{0: 1, 5: 2}, "o"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("index beyond m error = %v, want ErrInvalidArgument", err)
	}
	var nilSys *System
	if _, err := nilSys.FixedInputReduction(map[int]float64{0: 1}, "o"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil error = %v, want ErrInvalidArgument", err)
	}
	if _, err := nilSys.AugmentInternalDelayOutputs("z"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil AugmentInternalDelayOutputs error = %v, want ErrInvalidArgument", err)
	}
	if _, err := nilSys.ToExplicit(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil ToExplicit error = %v, want ErrInvalidArgument", err)
	}
}
