package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// freeOracle returns y_i(t) = C·e^{E⁻¹A·t}·x0 of the pencil for t > 0 at
// stepOraclePrec bits: the residues of C(sE − A)⁻¹E, applied to x0.
func freeOracle(t *testing.T, sys *System, modes stepOracleModes, x0 *mat.VecDense) func(i int, tt float64) float64 {
	t.Helper()
	n, _, p := sys.Dims()
	fm := sys.Copy()
	fm.B = mat.DenseCopyOf(sys.E)
	fm.D = mat.NewDense(p, n, nil)
	fm.InputName = nil
	fe, ok := newStepOracleModes(fm, modes.poles)
	if !ok {
		t.Fatal("free oracle did not resolve the poles")
	}
	var y0 mat.VecDense
	y0.MulVec(sys.C, x0)
	for i := range p {
		var s complex128
		for q := range fe.poles {
			for j := range n {
				s += fe.residues[q][i][j] * complex(x0.AtVec(j), 0)
			}
		}
		if d := cmplx.Abs(s - complex(y0.AtVec(i), 0)); d > 1e-9*math.Max(1, math.Abs(y0.AtVec(i))) {
			t.Fatalf("free oracle: Σ residues·x0 = %v, C·x0 = %v", s, y0.AtVec(i))
		}
	}
	return func(i int, tt float64) float64 {
		var y complex128
		for q, l := range fe.poles {
			if real(l)*tt < -700 {
				continue
			}
			var r complex128
			for j := range n {
				r += fe.residues[q][i][j] * complex(x0.AtVec(j), 0)
			}
			y += r * cmplx.Exp(l*complex(tt, 0))
		}
		return real(y)
	}
}

func onesVec(n int) *mat.VecDense {
	x := mat.NewVecDense(n, nil)
	for i := range n {
		x.SetVec(i, 1)
	}
	return x
}

// Initial of a full-rank E whose fold E⁻¹A loses the slow poles follows
// the pencil: after t = 0 every sample matches C·e^{E⁻¹A·t}·x0 of the
// 320-bit generalized eigenvalues (stepFoldLossCases seed 2210, AZUMSC:
// the fold simulated a stable model as diverging), and y(0) = C·x0.
func TestInitialFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		n, _, p := sys.Dims()
		x0 := onesVec(n)
		want := freeOracle(t, sys, modes[k], x0)
		r, err := Initial(sys, x0, 5)
		if err != nil {
			t.Errorf("%s: Initial: %v", names[k], err)
			continue
		}
		worst := 0.0
		_, cols := r.Y.Dims()
		for s := 1; s < cols; s++ {
			for i := range p {
				worst = math.Max(worst, stepOracleRel(r.Y.At(i, s), want(i, r.T[s])))
			}
		}
		var y0 mat.VecDense
		y0.MulVec(sys.C, x0)
		for i := range p {
			worst = math.Max(worst, stepOracleRel(r.Y.At(i, 0), y0.AtVec(i)))
		}
		if worst > 1e-7 {
			t.Errorf("%s: Initial off the pencil by %.3g", names[k], worst)
		}
	}
}

// Lsim from x0 of the same models is the pencil's step response of input 0
// plus its free response, and with u = 0 it is Initial on the same grid.
func TestLsimInitialStateFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		n, m, p := sys.Dims()
		x0 := onesVec(n)
		free := freeOracle(t, sys, modes[k], x0)
		in, err := Initial(sys, x0, 5)
		if err != nil {
			t.Fatalf("%s: Initial: %v", names[k], err)
		}
		cols := len(in.T)
		u := mat.NewDense(cols, m, nil)
		for s := range cols {
			u.Set(s, 0, 1)
		}
		ls, err := Lsim(sys, u, in.T, x0)
		if err != nil {
			t.Errorf("%s: Lsim: %v", names[k], err)
			continue
		}
		zero, err := Lsim(sys, mat.NewDense(cols, m, nil), in.T, x0)
		if err != nil {
			t.Errorf("%s: Lsim u = 0: %v", names[k], err)
			continue
		}
		worst, same := 0.0, 0.0
		for s := 1; s < cols; s++ {
			for i := range p {
				worst = math.Max(worst, stepOracleRel(ls.Y.At(i, s), modes[k].step(i, 0, in.T[s])+free(i, in.T[s])))
			}
		}
		for s := range cols {
			for i := range p {
				same = math.Max(same, stepOracleRel(zero.Y.At(i, s), in.Y.At(i, s)))
			}
		}
		if worst > 1e-7 {
			t.Errorf("%s: Lsim from x0 off the pencil by %.3g", names[k], worst)
		}
		if same > 1e-9 {
			t.Errorf("%s: Lsim(u = 0, x0) differs from Initial by %.3g", names[k], same)
		}
	}
}

// The pencil route keeps the argument checks: a wrong x0 length is
// ErrDimensionMismatch, a wrong u shape the Lsim shape error.
func TestInitialStateFullRankIllConditionedEArgumentErrors(t *testing.T) {
	_, systems, _ := stiffDescriptorModels(t)
	sys := systems[0]
	n, m, _ := sys.Dims()
	if _, err := Initial(sys, onesVec(n+1), 5); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Initial x0 length: err = %v, want ErrDimensionMismatch", err)
	}
	tt := []float64{0, 0.1, 0.2}
	if _, err := Lsim(sys, mat.NewDense(3, m, nil), tt, onesVec(n+1)); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Lsim x0 length: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := Lsim(sys, mat.NewDense(2, m, nil), tt, onesVec(n)); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Lsim u shape: err = %v, want ErrDimensionMismatch", err)
	}
}
