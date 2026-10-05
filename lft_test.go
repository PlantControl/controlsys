package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func evalMIMOTF(sys *System, s complex128) [][]complex128 {
	tf, err := sys.TransferFunction(nil)
	if err != nil {
		panic(err)
	}
	return tf.TF.Eval(s)
}

func TestLFT_NilM(t *testing.T) {
	delta, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	_, err := LFT(nil, delta, 1, 1)
	if err == nil {
		t.Fatal("expected error for nil M")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestLFT_DimMismatch(t *testing.T) {
	M, _ := NewGain(mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}), 0)

	delta, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	// pM-ny=3-1=2 != mD=1
	_, err := LFT(M, delta, 1, 1)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestLFT_InvalidPartition(t *testing.T) {
	M, _ := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)

	t.Run("nu>mM", func(t *testing.T) {
		_, err := LFT(M, nil, 3, 1)
		if err == nil {
			t.Fatal("expected error")
		}
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("got %v, want ErrDimensionMismatch", err)
		}
	})

	t.Run("ny>pM", func(t *testing.T) {
		_, err := LFT(M, nil, 1, 3)
		if err == nil {
			t.Fatal("expected error")
		}
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("got %v, want ErrDimensionMismatch", err)
		}
	})
}

func TestLFT_DomainMismatch(t *testing.T) {
	M, _ := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	delta, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.01)

	_, err := LFT(M, delta, 1, 1)
	if err == nil {
		t.Fatal("expected error for domain mismatch")
	}
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("got %v, want ErrDomainMismatch", err)
	}
}

func TestLFT_Lower_SISO(t *testing.T) {
	a, b, c, d := 1.0, 2.0, 3.0, 0.5
	M, _ := NewGain(mat.NewDense(2, 2, []float64{a, b, c, d}), 0)

	// Delta = 1/(s+1)
	Delta, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	// F_l(M, Delta) = a + b*Delta*(1 - d*Delta)^-1 * c
	// Delta(s) = 1/(s+1)
	// = a + b*c/(s+1) / (1 - d/(s+1))
	// = a + b*c / (s+1-d)
	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		deltaVal := 1 / (s + 1)
		want := complex(a, 0) + complex(b*c, 0)*deltaVal/(1-complex(d, 0)*deltaVal)
		got := evalMIMOTF(result, s)
		if cmplx.Abs(got[0][0]-want) > 1e-10 {
			t.Errorf("s=%v: got %v, want %v (diff=%v)", s, got[0][0], want, cmplx.Abs(got[0][0]-want))
		}
	}
}

func TestLFT_Lower_PureGainDelta(t *testing.T) {
	// M is 2x2 dynamic, Delta is scalar gain k=0.4
	// Non-symmetric A
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0.5, 0.5, 0}),
		0,
	)
	k := 0.4
	Delta, _ := NewGain(mat.NewDense(1, 1, []float64{k}), 0)

	result, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	// M partitioned as 1x1 blocks: M11=D(0,0)=0, M12=D(0,1)=0.5, M21=D(1,0)=0.5, M22=D(1,1)=0
	// Compute reference via full state-space formula
	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hM := evalMIMOTF(M, s)
		m11 := hM[0][0]
		m12 := hM[0][1]
		m21 := hM[1][0]
		m22 := hM[1][1]
		want := m11 + m12*complex(k, 0)/(1-m22*complex(k, 0))*m21

		got := evalMIMOTF(result, s)
		if cmplx.Abs(got[0][0]-want) > 1e-10 {
			t.Errorf("s=%v: got %v, want %v (diff=%v)", s, got[0][0], want, cmplx.Abs(got[0][0]-want))
		}
	}
}

func TestLFT_BothPureGain(t *testing.T) {
	M, _ := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	Delta, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	result, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	// M11=1, M12=2, M21=3, M22=4, Delta=0.5
	// F_l = 1 + 2*0.5*(1 - 4*0.5)^-1 * 3 = 1 + 1*(-1)^-1*3 = 1 - 3 = -2
	n, m, p := result.Dims()
	if n != 0 {
		t.Errorf("expected static gain, got n=%d", n)
	}
	if m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (0,1,1)", n, m, p)
	}
	got := result.D.At(0, 0)
	if got != -2 {
		t.Errorf("D = %v, want -2", got)
	}
}

func TestLFT_DeltaNil(t *testing.T) {
	// 3x3 system, nu=2, ny=1 -> extract 1x2 top-left channel
	// Non-symmetric A
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 3, 0, -2}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0, 0}),
		mat.NewDense(3, 3, []float64{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}),
		0,
	)

	result, err := LFT(M, nil, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (2,2,1)", n, m, p)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hM := evalMIMOTF(M, s)
		hR := evalMIMOTF(result, s)
		for j := range 2 {
			if cmplx.Abs(hR[0][j]-hM[0][j]) > 1e-10 {
				t.Errorf("s=%v col %d: got %v, want %v", s, j, hR[0][j], hM[0][j])
			}
		}
	}
}

func TestLFT_RecoversFeedback(t *testing.T) {
	// Plant P: 1/(s+2), controller K: 3/(s+5)
	P, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	K, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	ref, err := Feedback(P, K, -1)
	if err != nil {
		t.Fatal(err)
	}

	// Build M = [P, -P; I, -I] so F_l(M, K) = P - P*K*(I+K)^-1*I ... too complex
	// Instead: M = [I, P; I, P] with Delta=-K
	// F_l = I + P*(-K)*(I - P*(-K))^-1 * I = I - PK*(I+PK)^-1 = (I+PK)^-1 = S
	// That gives sensitivity. We want complementary sensitivity T = PK/(1+PK).
	//
	// Use M11=0, M12=P, M21=I, M22=P, Delta=-K:
	// F_l = 0 + P*(-K)*(I - P*(-K))^-1 * I = -PK/(1+PK) = -T
	//
	// Or just verify at frequency level:
	// Feedback(P, K, -1) = P*(I+K*P)^-1
	// Using M = [P, P; I, I], Delta = -K:
	// F_l = P + P*(-K)*(I - I*(-K))^-1*I = P - PK*(I+K)^-1 = P*(1 - K/(1+K)) = P*1/(1+K)
	// That's P/(1+K), not P/(1+KP).
	//
	// Standard formulation: M11=0, M12=I, M21=P, M22=0, Delta=K:
	// F_l = 0 + I*K*(I-0)^-1*P = KP (series). Not helpful.
	//
	// For negative feedback closed-loop T = P/(1+KP):
	// M = [[0, I]; [I, P]], nu=1, ny=1 with appropriate dimensions
	// But M has 2 outputs, 2 inputs; ny=1 external outputs, nu=1 external inputs
	// M11=0(1x1), M12=I(1x1), M21=I(1x1), M22=P
	// Need M as 2x2 system: top-left 0, top-right I, bottom-left I, bottom-right P
	//
	// Build M by augmenting P with gain routing using Append
	eye1, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	zero1, _ := NewGain(mat.NewDense(1, 1, []float64{0}), 0)

	// Top row: [0, I] = [zero1, eye1] => 1 output, 2 inputs
	topRow, _ := Append(zero1, eye1)
	// Bottom row: [I, P] - need to build as 1 output, 2 inputs
	// P has 1 state, so we need to combine I and P side by side
	// Append gives block-diagonal. We need [I | P] as a single row system.
	// Instead build M directly as a gain+state system.

	// M has state from P plus routing.
	// M: 2 outputs, 2 inputs
	// States: same as P (1 state, A=[-2])
	// B = [b_u1 b_u2] for the 2 inputs
	// u1 is external, u2 goes to Delta
	// y1 is external (top), y2 goes to Delta (bottom)
	//
	// M11=0: y1 has no direct path from u1
	// M12=I: y1 = u2 (passthrough)
	// M21=I: y2 = u1 (passthrough) + P*u1... wait.
	//
	// For y = P/(1+KP)*r:
	// e = r - K*y, y = P*e => y = P*(r - K*y) => y(1+PK) = P*r => y = P/(1+PK)*r
	//
	// LFT: y1 = M11*u1 + M12*w, w = Delta*y2, y2 = M21*u1 + M22*w
	// Substituting: w = K*y2, y2 = M21*u1 + M22*K*y2
	// y2 = (I - M22*K)^-1*M21*u1
	// y1 = M11*u1 + M12*K*(I-M22*K)^-1*M21*u1
	//
	// We want y1 = P/(1+KP)*u1
	// Try: M11=P, M12=-P, M21=I, M22=0
	// y1 = P*u1 + (-P)*K*(I-0)^-1*I*u1 = P*u1 - PK*u1 = P*(1-K)*u1. No.
	//
	// Try: M11=0, M12=P, M21=I, M22=0, Delta=K
	// y1 = 0 + P*K*I*u1 = PK*u1. Series, not feedback.
	//
	// Try: M11=P, M12=-P, M21=I, M22=-I, Delta=K
	// y1 = P + (-P)*K*(I-(-I)*K)^-1*I = P - PK*(I+K)^-1 = P*(1+K-K)/(1+K) = P/(1+K). Not P/(1+KP).
	//
	// The standard plant is: M = [[I, P]; [I, P]] and Delta = -K gives
	// S = (I+PK)^-1 at y1. We want T at plant output.
	//
	// Actually: M11=P, M12=P, M21=I, M22=I, Delta=-K
	// F_l = P + P*(-K)*(I-(-K))^-1*I = P - PK/(1+K) = P/(1+K). Still wrong.
	//
	// The issue is SISO: let's just verify numerically by direct comparison.
	// Build M as a 2-input, 2-output system where
	// [y1; y2] = M * [u1; u2] and w=u2, z=y2, then Delta connects z->w.
	//
	// We want: y1 = P*e, e = u1 + sign*u2, y2 = y1
	// So: y1 = P*(u1 + sign*u2), y2 = P*(u1 + sign*u2)
	// M = [P, sign*P; P, sign*P]
	// Using Delta = K, sign = -1:
	// F_l = P + (-P)*K*(I - (-P)*K)^-1*P = P + (-PK)*(1+PK)^-1*P = P - P^2K/(1+PK)
	// = P*(1+PK-PK)/(1+PK) = P/(1+PK). Yes!

	// Build M = [P, -P; P, -P] as a single 2-in, 2-out system
	M_sys, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, -1}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	_ = topRow
	_ = eye1
	_ = zero1

	lftResult, err := LFT(M_sys, K, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hRef := evalMIMOTF(ref, s)
		hLFT := evalMIMOTF(lftResult, s)
		if cmplx.Abs(hLFT[0][0]-hRef[0][0]) > 1e-10 {
			t.Errorf("s=%v: LFT=%v, Feedback=%v (diff=%v)", s, hLFT[0][0], hRef[0][0], cmplx.Abs(hLFT[0][0]-hRef[0][0]))
		}
	}
}

func TestLFT_AlgebraicLoop(t *testing.T) {
	// D22=2, D_Delta=0.5 => I - D22*D_Delta = 1 - 1 = 0 => singular
	M, _ := NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 2}), 0)
	Delta, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	_, err := LFT(M, Delta, 1, 1)
	if err == nil {
		t.Fatal("expected algebraic loop error")
	}
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("got %v, want ErrAlgebraicLoop", err)
	}
}

func TestLFT_NonSymmetricA(t *testing.T) {
	// Both M and Delta have non-symmetric A matrices
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 3, 0, -4}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 0.0}),
		0,
	)
	Delta, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hM := evalMIMOTF(M, s)
		m11, m12, m21, m22 := hM[0][0], hM[0][1], hM[1][0], hM[1][1]
		deltaVal := evalMIMOTF(Delta, s)[0][0]

		want := m11 + m12*deltaVal/(1-m22*deltaVal)*m21
		got := evalMIMOTF(result, s)
		if cmplx.Abs(got[0][0]-want) > 1e-10 {
			t.Errorf("s=%v: got %v, want %v (diff=%v)", s, got[0][0], want, cmplx.Abs(got[0][0]-want))
		}
	}
}

func TestLFT_MIMO_2x2(t *testing.T) {
	// M is 4x4 with nu=2, ny=2, so 2x2 external and 2x2 to Delta
	// Non-symmetric A
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 4, []float64{
			1, 0, 0.5, 0,
			0, 1, 0, 0.5,
		}),
		mat.NewDense(4, 2, []float64{
			1, 0,
			0, 1,
			0.3, 0,
			0, 0.3,
		}),
		mat.NewDense(4, 4, []float64{
			0.1, 0, 0.2, 0,
			0, 0.1, 0, 0.2,
			0.3, 0, 0, 0,
			0, 0.3, 0, 0,
		}),
		0,
	)
	Delta, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -4}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	result, err := LFT(M, Delta, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := result.Dims()
	if m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (_,2,2)", n, m, p)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hM := evalMIMOTF(M, s)
		hD := evalMIMOTF(Delta, s)

		// M11 = hM[:2][:2], M12 = hM[:2][2:], M21 = hM[2:][:2], M22 = hM[2:][2:]
		// F_l = M11 + M12*Delta*(I - M22*Delta)^-1*M21  (all 2x2)
		var m11, m12, m21, m22 [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				m11[i][j] = hM[i][j]
				m12[i][j] = hM[i][j+2]
				m21[i][j] = hM[i+2][j]
				m22[i][j] = hM[i+2][j+2]
			}
		}

		// I - M22*Delta
		var m22d [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					m22d[i][j] += m22[i][k] * hD[k][j]
				}
			}
		}
		var imd [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				imd[i][j] = -m22d[i][j]
			}
			imd[i][i] += 1
		}

		// Invert 2x2: [a b; c d]^-1 = 1/det * [d -b; -c a]
		det := imd[0][0]*imd[1][1] - imd[0][1]*imd[1][0]
		var inv [2][2]complex128
		inv[0][0] = imd[1][1] / det
		inv[0][1] = -imd[0][1] / det
		inv[1][0] = -imd[1][0] / det
		inv[1][1] = imd[0][0] / det

		// Delta * inv * M21
		var di [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					di[i][j] += hD[i][k] * inv[k][j]
				}
			}
		}
		var dim21 [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					dim21[i][j] += di[i][k] * m21[k][j]
				}
			}
		}

		// M12 * dim21
		var m12dim21 [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					m12dim21[i][j] += m12[i][k] * dim21[k][j]
				}
			}
		}

		got := evalMIMOTF(result, s)
		for i := range 2 {
			for j := range 2 {
				want := m11[i][j] + m12dim21[i][j]
				if cmplx.Abs(got[i][j]-want) > 1e-10 {
					t.Errorf("s=%v [%d][%d]: got %v, want %v (diff=%v)", s, i, j, got[i][j], want, cmplx.Abs(got[i][j]-want))
				}
			}
		}
	}
}

func TestLFT_DeltaNil_GainOnly(t *testing.T) {
	M, _ := NewGain(mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}), 0)

	result, err := LFT(M, nil, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := result.Dims()
	if n != 0 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (0,2,2)", n, m, p)
	}
	for i := range 2 {
		for j := range 2 {
			got := result.D.At(i, j)
			want := M.D.At(i, j)
			if got != want {
				t.Errorf("D[%d][%d] = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestLFT_BothDynamic(t *testing.T) {
	// Both M and Delta have dynamics, non-symmetric A
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0.5, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.5, 1}),
		mat.NewDense(2, 2, []float64{0, 0.3, 0.3, 0}),
		0,
	)
	Delta, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.1}),
		0,
	)

	result, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		hM := evalMIMOTF(M, s)
		m11, m12, m21, m22 := hM[0][0], hM[0][1], hM[1][0], hM[1][1]
		deltaVal := evalMIMOTF(Delta, s)[0][0]

		want := m11 + m12*deltaVal/(1-m22*deltaVal)*m21
		got := evalMIMOTF(result, s)
		if cmplx.Abs(got[0][0]-want) > 1e-10 {
			t.Errorf("s=%v: got %v, want %v (diff=%v)", s, got[0][0], want, cmplx.Abs(got[0][0]-want))
		}
	}
}

func TestLFT_DelayedMWithDynamicDeltaMatchesFrequencyLFT(t *testing.T) {
	M, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.4, -0.3, -2}),
		mat.NewDense(2, 2, []float64{1, 0.2, -0.5, 0.8}),
		mat.NewDense(2, 2, []float64{0.6, -0.1, 0.2, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0.05, -0.2, 0.15}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := M.SetInputDelay([]float64{0.3, 0.1}); err != nil {
		t.Fatal(err)
	}
	if err := M.SetOutputDelay([]float64{0, 0.2}); err != nil {
		t.Fatal(err)
	}
	Delta, err := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{0.2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LFT(M, Delta, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0.1, 0.7, 2.5}
	fm, err := M.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	fd, err := Delta.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	fg, err := got.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		d := fd.At(k, 0, 0)
		want := fm.At(k, 0, 0) + fm.At(k, 0, 1)*d*fm.At(k, 1, 0)/(1-fm.At(k, 1, 1)*d)
		if diff := cmplx.Abs(want - fg.At(k, 0, 0)); diff > 1e-12 {
			t.Fatalf("w=%g: differs by %g", omega[k], diff)
		}
	}
}

func TestLFTEmptyLoopKeepsUpperChannels(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		M := emptyIOFixture(t, 2, 2, 3, dt)
		empty, err := NewGain(&mat.Dense{}, dt)
		if err != nil {
			t.Fatal(err)
		}
		autonomous := emptyIOFixture(t, 1, 0, 0, dt)
		for name, delta := range map[string]*System{"gain": empty, "states": autonomous} {
			got, err := LFT(M, delta, 2, 3)
			if err != nil {
				t.Fatalf("dt=%g %s: %v", dt, name, err)
			}
			nD, _, _ := delta.Dims()
			if n, m, p := got.Dims(); n != 2+nD || m != 2 || p != 3 {
				t.Fatalf("dt=%g %s Dims = (%d,%d,%d)", dt, name, n, m, p)
			}
			for _, w := range []float64{0.3, 2} {
				s := complex(0, w)
				if dt > 0 {
					s = cmplx.Exp(complex(0, w*dt))
				}
				want, err := M.EvalFr(s)
				if err != nil {
					t.Fatal(err)
				}
				have, err := got.EvalFr(s)
				if err != nil {
					t.Fatal(err)
				}
				for i := range 3 {
					for j := range 2 {
						if cmplx.Abs(have[i][j]-want[i][j]) > 1e-12 {
							t.Errorf("dt=%g %s w=%g G[%d][%d] = %v, want %v", dt, name, w, i, j, have[i][j], want[i][j])
						}
					}
				}
			}
		}
	}
}

func lftZeroWidthPlants(t *testing.T, dt float64, nu, ny int) (M, Delta *System) {
	t.Helper()
	const z, w = 2, 2
	M = emptyIOFixture(t, 3, nu+z, ny+w, dt)
	Delta, err := New(
		mat.NewDense(2, 2, []float64{-0.4, 0.7, -0.3, -0.9}),
		mat.NewDense(2, w, []float64{1, -0.5, 0.25, 2}),
		mat.NewDense(z, 2, []float64{0.6, -1, 1.5, 0.2}),
		mat.NewDense(z, w, []float64{0.3, -0.2, 0.1, 0.4}),
		dt)
	if err != nil {
		t.Fatal(err)
	}
	if dt > 0 {
		Delta.A.Scale(0.5, Delta.A)
	}
	return M, Delta
}

// lftLoopOracle evaluates the interconnection of M and Delta at one point by
// solving the algebraic loop directly; it returns the state update [xM'; xD']
// and the closed-loop output y.
func lftLoopOracle(M, Delta *System, nu, ny int, x, u []float64) (dx, y []float64) {
	nM, mM, pM := M.Dims()
	nD, _, _ := Delta.Dims()
	z, w := mM-nu, pM-ny
	K := mat.NewDense(w+z, w+z, nil)
	rhs := mat.NewVecDense(w+z, nil)
	for i := range w + z {
		K.Set(i, i, 1)
	}
	for i := range w {
		for j := range z {
			K.Set(i, w+j, -M.D.At(ny+i, nu+j))
		}
		v := 0.0
		for j := range nM {
			v += M.C.At(ny+i, j) * x[j]
		}
		for j := range nu {
			v += M.D.At(ny+i, j) * u[j]
		}
		rhs.SetVec(i, v)
	}
	for i := range z {
		for j := range w {
			K.Set(w+i, j, -Delta.D.At(i, j))
		}
		v := 0.0
		for j := range nD {
			v += Delta.C.At(i, j) * x[nM+j]
		}
		rhs.SetVec(w+i, v)
	}
	var sig mat.VecDense
	if err := sig.SolveVec(K, rhs); err != nil {
		panic(err)
	}
	dx = make([]float64, nM+nD)
	for i := range nM {
		for j := range nM {
			dx[i] += M.A.At(i, j) * x[j]
		}
		for j := range nu {
			dx[i] += M.B.At(i, j) * u[j]
		}
		for j := range z {
			dx[i] += M.B.At(i, nu+j) * sig.AtVec(w+j)
		}
	}
	for i := range nD {
		for j := range nD {
			dx[nM+i] += Delta.A.At(i, j) * x[nM+j]
		}
		for j := range w {
			dx[nM+i] += Delta.B.At(i, j) * sig.AtVec(j)
		}
	}
	y = make([]float64, ny)
	for i := range ny {
		for j := range nM {
			y[i] += M.C.At(i, j) * x[j]
		}
		for j := range nu {
			y[i] += M.D.At(i, j) * u[j]
		}
		for j := range z {
			y[i] += M.D.At(i, nu+j) * sig.AtVec(w+j)
		}
	}
	return dx, y
}

func ssPointEval(sys *System, x, u []float64) (dx, y []float64) {
	n, m, p := sys.Dims()
	dx = make([]float64, n)
	y = make([]float64, p)
	for i := range n {
		for j := range n {
			dx[i] += sys.A.At(i, j) * x[j]
		}
		for j := range m {
			dx[i] += sys.B.At(i, j) * u[j]
		}
	}
	for i := range p {
		for j := range n {
			y[i] += sys.C.At(i, j) * x[j]
		}
		for j := range m {
			y[i] += sys.D.At(i, j) * u[j]
		}
	}
	return dx, y
}

func TestLFTZeroWidthPartitionsMatchLoopOracle(t *testing.T) {
	x := []float64{0.7, -1.2, 0.4, 0.9, -0.6}
	uAll := []float64{1.1, -0.8}
	for _, dt := range []float64{0, 0.1} {
		for _, part := range [][2]int{{0, 2}, {2, 0}, {0, 0}, {1, 2}} {
			nu, ny := part[0], part[1]
			M, Delta := lftZeroWidthPlants(t, dt, nu, ny)
			tag := fmt.Sprintf("dt=%g nu=%d ny=%d", dt, nu, ny)
			got, err := LFT(M, Delta, nu, ny)
			if err != nil {
				t.Fatalf("%s: %v", tag, err)
			}
			if err := got.Validate(); err != nil {
				t.Fatalf("%s: Validate: %v", tag, err)
			}
			if n, m, p := got.Dims(); n != 5 || m != nu || p != ny {
				t.Fatalf("%s: Dims = (%d,%d,%d), want (5,%d,%d)", tag, n, m, p, nu, ny)
			}
			u := uAll[:nu]
			wantDx, wantY := lftLoopOracle(M, Delta, nu, ny, x, u)
			gotDx, gotY := ssPointEval(got, x, u)
			for i := range wantDx {
				if math.Abs(gotDx[i]-wantDx[i]) > 1e-12 {
					t.Errorf("%s: dx[%d] = %v, want %v", tag, i, gotDx[i], wantDx[i])
				}
			}
			for i := range wantY {
				if math.Abs(gotY[i]-wantY[i]) > 1e-12 {
					t.Errorf("%s: y[%d] = %v, want %v", tag, i, gotY[i], wantY[i])
				}
			}
		}
	}
}

func TestLFTZeroWidthUpperChannelsSimulate(t *testing.T) {
	const steps = 6
	M, Delta := lftZeroWidthPlants(t, 0.1, 0, 2)
	got, err := LFT(M, Delta, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	x := []float64{0.7, -1.2, 0.4, 0.9, -0.6}
	resp, err := got.Simulate(nil, mat.NewVecDense(5, append([]float64(nil), x...)), &SimulateOpts{Steps: steps})
	if err != nil {
		t.Fatal(err)
	}
	if r, c := resp.Y.Dims(); r != 2 || c != steps {
		t.Fatalf("Y dims = %dx%d, want 2x%d", r, c, steps)
	}
	for k := range steps {
		dx, y := lftLoopOracle(M, Delta, 0, 2, x, nil)
		for i := range y {
			if math.Abs(resp.Y.At(i, k)-y[i]) > 1e-10 {
				t.Errorf("k=%d y[%d] = %v, want %v", k, i, resp.Y.At(i, k), y[i])
			}
		}
		x = dx
	}
	for i := range x {
		if math.Abs(resp.XFinal.AtVec(i)-x[i]) > 1e-10 {
			t.Errorf("XFinal[%d] = %v, want %v", i, resp.XFinal.AtVec(i), x[i])
		}
	}
}

func TestLFTZeroWidthStaticGain(t *testing.T) {
	M, err := NewGain(mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), 0)
	if err != nil {
		t.Fatal(err)
	}
	Delta, err := NewGain(mat.NewDense(2, 2, []float64{0.1, 0.2, -0.3, 0.4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LFT(M, Delta, 0, 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("LFT gain nu=0 ny=1: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := LFT(M, nil, 0, 2); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("LFT gain nil Delta nu=0 ny=2: err = %v, want ErrDimensionMismatch", err)
	}
	M2, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LFT(M2, Delta, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n, m, p := got.Dims(); n != 0 || m != 0 || p != 0 {
		t.Errorf("Dims = (%d,%d,%d), want (0,0,0)", n, m, p)
	}
}

// TestLFTZeroWidthDelayedDeltaMatchesFullPartition checks the internal-delay
// LFT path: closing the loop with nu=0 (ny=0) must reproduce the full
// partition's free response (state trajectory) with the upper inputs zeroed
// (outputs dropped).
func TestLFTZeroWidthDelayedDeltaMatchesFullPartition(t *testing.T) {
	const steps = 8
	full, Delta := lftZeroWidthPlants(t, 0.1, 2, 2)
	Delta.InputDelay = []float64{1, 2}
	fullLFT, err := LFT(full, Delta, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := fullLFT.Dims()
	x0 := mat.NewVecDense(n, nil)
	for i := range n {
		x0.SetVec(i, 0.3*float64(i+1)-0.7*float64(i%2))
	}
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)))
		u.Set(1, k, 0.5-0.1*float64(k))
	}

	noInputs, err := New(full.A, mat.DenseCopyOf(full.B.Slice(0, 3, 2, 4)), full.C, mat.DenseCopyOf(full.D.Slice(0, 4, 2, 4)), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LFT(noInputs, Delta, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	gotR, err := got.Simulate(nil, x0, &SimulateOpts{Steps: steps})
	if err != nil {
		t.Fatal(err)
	}
	wantR, err := fullLFT.Simulate(mat.NewDense(2, steps, nil), x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(gotR.Y, wantR.Y, 1e-10) {
		t.Errorf("nu=0 Y =\n%v\nwant\n%v", mat.Formatted(gotR.Y), mat.Formatted(wantR.Y))
	}

	noOutputs, err := New(full.A, full.B, mat.DenseCopyOf(full.C.Slice(2, 4, 0, 3)), mat.DenseCopyOf(full.D.Slice(2, 4, 0, 4)), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	got, err = LFT(noOutputs, Delta, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	gotR, err = got.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantR, err = fullLFT.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotR.Y != nil {
		t.Errorf("ny=0 Y = %v, want nil", gotR.Y)
	}
	if !vecEqual(gotR.XFinal, wantR.XFinal, 1e-10) {
		t.Errorf("ny=0 XFinal = %v, want %v", mat.Formatted(gotR.XFinal.T()), mat.Formatted(wantR.XFinal.T()))
	}
}
