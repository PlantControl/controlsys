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

const fieldTol = 1e-9

func fieldSolve(M, R [][]complex128) [][]complex128 {
	n := len(M)
	if n == 0 {
		return R
	}
	k := len(R[0])
	a := make([][]complex128, n)
	for i := range n {
		a[i] = append(append([]complex128{}, M[i]...), R[i]...)
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if cmplx.Abs(a[r][c]) > cmplx.Abs(a[piv][c]) {
				piv = r
			}
		}
		a[c], a[piv] = a[piv], a[c]
		for r := range n {
			if r == c {
				continue
			}
			f := a[r][c] / a[c][c]
			for j := c; j < n+k; j++ {
				a[r][j] -= f * a[c][j]
			}
		}
	}
	X := make([][]complex128, n)
	for i := range n {
		X[i] = make([]complex128, k)
		for j := range k {
			X[i][j] = a[i][n+j] / a[i][i]
		}
	}
	return X
}

func fieldAt(m *mat.Dense, i, j int) float64 {
	if m == nil {
		return 0
	}
	if r, c := m.Dims(); r == 0 || c == 0 {
		return 0
	}
	return m.At(i, j)
}

// fieldBlock evaluates Cblk (sE-A)⁻¹ Bblk + Dblk by dense complex elimination.
func fieldBlock(sys *System, s complex128, B *mat.Dense, cols int, C *mat.Dense, rows int, D *mat.Dense) [][]complex128 {
	n, _, _ := sys.Dims()
	M := make([][]complex128, n)
	R := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n)
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = fieldAt(sys.E, i, j)
			} else if i == j {
				e = 1
			}
			M[i][j] = s*complex(e, 0) - complex(fieldAt(sys.A, i, j), 0)
		}
		R[i] = make([]complex128, cols)
		for j := range cols {
			R[i][j] = complex(fieldAt(B, i, j), 0)
		}
	}
	X := fieldSolve(M, R)
	G := make([][]complex128, rows)
	for i := range rows {
		G[i] = make([]complex128, cols)
		for j := range cols {
			v := complex(fieldAt(D, i, j), 0)
			for k := range n {
				v += complex(fieldAt(C, i, k), 0) * X[k][j]
			}
			G[i][j] = v
		}
	}
	return G
}

func fieldDelay(dt float64, s complex128, tau float64) complex128 {
	if tau == 0 {
		return 1
	}
	if dt == 0 {
		return cmplx.Exp(-s * complex(tau, 0))
	}
	return cmplx.Pow(s, complex(-tau, 0))
}

// fieldOracle evaluates the full delayed response of sys at s independently
// of the library: descriptor E, LFT internal delays and all external delays.
func fieldOracle(sys *System, s complex128) [][]complex128 {
	_, m, p := sys.Dims()
	G := fieldBlock(sys, s, sys.B, m, sys.C, p, sys.D)
	if N := sys.internalDelayCount(); N > 0 {
		L := sys.LFT
		H12 := fieldBlock(sys, s, L.B2, N, sys.C, p, L.D12)
		H21 := fieldBlock(sys, s, sys.B, m, L.C2, N, L.D21)
		H22 := fieldBlock(sys, s, L.B2, N, L.C2, N, L.D22)
		dl := make([]complex128, N)
		for k := range N {
			dl[k] = fieldDelay(sys.Dt, s, L.Tau[k])
		}
		IM := make([][]complex128, N)
		for i := range N {
			IM[i] = make([]complex128, N)
			for j := range N {
				IM[i][j] = -H22[i][j] * dl[j]
			}
			IM[i][i]++
		}
		X := fieldSolve(IM, H21)
		for i := range p {
			for j := range m {
				for k := range N {
					G[i][j] += H12[i][k] * dl[k] * X[k][j]
				}
			}
		}
	}
	for i := range p {
		for j := range m {
			tau := fieldAt(sys.Delay, i, j)
			if sys.InputDelay != nil {
				tau += sys.InputDelay[j]
			}
			if sys.OutputDelay != nil {
				tau += sys.OutputDelay[i]
			}
			G[i][j] *= fieldDelay(sys.Dt, s, tau)
		}
	}
	return G
}

func fieldPoints(dt float64) []complex128 {
	if dt == 0 {
		return []complex128{complex(0.3, 1.1), complex(0, 0.4), complex(-0.2, 3)}
	}
	return []complex128{cmplx.Exp(complex(0, 0.4)), complex(1.1, 0.3), cmplx.Exp(complex(0, 2.2))}
}

func fieldSub(G [][]complex128, outs, ins []int) [][]complex128 {
	out := make([][]complex128, len(outs))
	for a, i := range outs {
		out[a] = make([]complex128, len(ins))
		for b, j := range ins {
			out[a][b] = G[i][j]
		}
	}
	return out
}

func fieldMaxDiff(a, b [][]complex128) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	d := 0.0
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return math.Inf(1)
		}
		for j := range a[i] {
			d = math.Max(d, cmplx.Abs(a[i][j]-b[i][j]))
		}
	}
	return d
}

func assertFieldResponse(t *testing.T, label string, got *System, want func(complex128) [][]complex128) {
	t.Helper()
	for _, s := range fieldPoints(got.Dt) {
		if d := fieldMaxDiff(fieldOracle(got, s), want(s)); d > fieldTol {
			t.Errorf("%s: response at s=%v differs by %.3g", label, s, d)
			return
		}
	}
}

// fieldPlant is 3 states, 2 inputs, 3 outputs, non-symmetric A, D != 0.
// State 3 is uncontrollable from u and from the internal-delay channel.
func fieldPlant(t *testing.T, dt float64) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1.2, 0.7, 0.1, -0.3, -2.1, 0.4, 0, 0, -3.3})
	if dt > 0 {
		A.Scale(0.2, A)
		for i := range 3 {
			A.Set(i, i, A.At(i, i)+0.9)
		}
	}
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1.3, 0, 0})
	C := mat.NewDense(3, 3, []float64{1, 0.5, -0.2, 0, 1.1, 0.3, 0.7, -0.6, 1})
	D := mat.NewDense(3, 2, []float64{0.1, 0, 0, -0.2, 0.05, 0.3})
	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func fieldScale(dt float64) float64 {
	if dt > 0 {
		return 10
	}
	return 0.1
}

func fieldIODelay(t *testing.T, dt float64) *System {
	t.Helper()
	sys := fieldPlant(t, dt)
	k := fieldScale(dt)
	if err := sys.SetDelay(mat.NewDense(3, 2, []float64{k, 0, 2 * k, 3 * k, 0, k})); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{4 * k, k}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{0, 2 * k, 5 * k}); err != nil {
		t.Fatal(err)
	}
	return sys
}

func fieldLFT(t *testing.T, dt float64) *System {
	t.Helper()
	sys := fieldPlant(t, dt)
	k := fieldScale(dt)
	if err := sys.SetInternalDelay([]float64{7 * k},
		mat.NewDense(3, 1, []float64{0.3, -0.5, 0}),
		mat.NewDense(1, 3, []float64{0.4, 0.2, -0.7}),
		mat.NewDense(3, 1, []float64{0.2, -0.1, 0.3}),
		mat.NewDense(1, 2, []float64{0.1, -0.3}),
		mat.NewDense(1, 1, []float64{0.05})); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{2 * k, 0}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{k, 0, 3 * k}); err != nil {
		t.Fatal(err)
	}
	return sys
}

func fieldDescriptor(t *testing.T, dt float64) *System {
	t.Helper()
	sys := fieldIODelay(t, dt)
	sys.E = mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, 0.2, 0, 0.4, 1.5})
	return sys
}

func TestRealizationPolicyResultRefusesInternalDelay(t *testing.T) {
	sys := fieldLFT(t, 0)
	policy := newRealizationTransformPolicy(sys)
	if _, err := policy.result(sys.A, sys.B, sys.C, sys.D); !errors.Is(err, ErrDelayNotRepresentable) {
		t.Fatalf("result with internal delay: err = %v, want ErrDelayNotRepresentable", err)
	}
	io := fieldIODelay(t, 0)
	got, err := newRealizationTransformPolicy(io).result(io.A, io.B, io.C, io.D)
	if err != nil {
		t.Fatal(err)
	}
	assertFieldResponse(t, "result keeps external delays", got, func(s complex128) [][]complex128 { return fieldOracle(io, s) })
}

// TestStructuralOpsPreserveDelayedResponse sweeps structural operations over
// delayed, internal-delay and descriptor fixtures: each op either reproduces
// the source response or refuses with a documented error (descriptor models
// may always refuse with ErrDescriptorUnsupported).
func TestStructuralOpsPreserveDelayedResponse(t *testing.T) {
	T := mat.NewDense(3, 3, []float64{2, 0.3, -0.1, 0.4, 1.5, 0.2, -0.3, 0.1, 1.2})
	ins, outs := []int{1, 0}, []int{2, 0, 1}
	ops := []struct {
		name   string
		op     func(*System) (*System, error)
		outs   []int
		ins    []int
		refuse error
	}{
		{"SS2SS", func(s *System) (*System, error) { return SS2SS(s, T) }, nil, nil, nil},
		{"StateTransform", func(s *System) (*System, error) { return s.StateTransform(T) }, nil, nil, nil},
		{"Xperm", func(s *System) (*System, error) { return Xperm(s, []int{2, 0, 1}) }, nil, nil, nil},
		{"SelectByIndex", func(s *System) (*System, error) { return s.SelectByIndex(ins, outs) }, outs, ins, nil},
		{"MinimalRealization", func(s *System) (*System, error) {
			r, err := s.MinimalRealization()
			if err != nil {
				return nil, err
			}
			return r.Sys, nil
		}, nil, nil, nil},
		{"ReduceUnobservable", func(s *System) (*System, error) {
			r, err := s.Reduce(&ReduceOpts{Mode: ReduceUnobservable})
			if err != nil {
				return nil, err
			}
			return r.Sys, nil
		}, nil, nil, nil},
		{"TF->SS", func(s *System) (*System, error) {
			r, err := s.TransferFunction(nil)
			if err != nil {
				return nil, err
			}
			ss, err := r.TF.StateSpace(nil)
			if err != nil {
				return nil, err
			}
			return ss.Sys, nil
		}, nil, nil, ErrDelayNotRepresentable},
	}
	fixtures := []struct {
		name string
		mk   func(*testing.T, float64) *System
	}{{"io", fieldIODelay}, {"lft", fieldLFT}, {"descriptor", fieldDescriptor}}
	for _, dt := range []float64{0, 0.1} {
		for _, fx := range fixtures {
			for _, op := range ops {
				label := fmt.Sprintf("%s/%s/dt=%g", fx.name, op.name, dt)
				orig := fx.mk(t, dt)
				got, err := op.op(orig)
				if err != nil {
					refused := op.refuse != nil && errors.Is(err, op.refuse)
					if fx.name == "descriptor" && errors.Is(err, ErrDescriptorUnsupported) {
						refused = true
					}
					if !refused {
						t.Errorf("%s: unexpected error %v", label, err)
					}
					continue
				}
				want := func(s complex128) [][]complex128 {
					G := fieldOracle(orig, s)
					if op.outs != nil {
						return fieldSub(G, op.outs, op.ins)
					}
					return G
				}
				assertFieldResponse(t, label, got, want)
			}
		}
	}
}

// gridSingularSolver fails exactly on the sweep grid, as at a pole, and
// evaluates normally elsewhere.
type gridSingularSolver struct {
	inner frequencyPointSolver
	grid  []complex128
}

func (g gridSingularSolver) evalInto(pt frequencyPoint, dst []complex128) error {
	if slices.Contains(g.grid, pt.value()) {
		return ErrSingularTransform
	}
	return g.inner.evalInto(pt, dst)
}

func TestFrequencySweepPoleLimitAppliesExternalDelaysOnce(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := fieldIODelay(t, dt)
		e := newFrequencyEvaluator(sys)
		omega := []float64{0.4, 1.3, 2.2}
		grid := make([]complex128, len(omega))
		for k, w := range omega {
			grid[k] = e.pointAt(w).value()
		}
		data := make([]complex128, len(omega)*e.p*e.m)
		if err := e.sweepInto(omega, data, gridSingularSolver{e.pointSolver(1), grid}); err != nil {
			t.Fatal(err)
		}
		for k, w := range omega {
			want := fieldOracle(sys, e.pointAt(w).value())
			got := make([][]complex128, e.p)
			for i := range e.p {
				got[i] = data[(k*e.p+i)*e.m : (k*e.p+i+1)*e.m]
			}
			if d := fieldMaxDiff(got, want); d > fieldTol {
				t.Errorf("dt=%v w=%v: pole-limit response off by %.3g", dt, w, d)
			}
		}
	}
}
