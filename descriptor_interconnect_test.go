package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const descriptorInterconnectTol = 1e-12

// descriptorTwin eliminates an invertible E, keeping state coordinates, so
// every interconnection of twins must match the interconnection of the
// descriptor models.
func descriptorTwin(t *testing.T, sys *System) *System {
	t.Helper()
	out, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

type descriptorFixture struct {
	P, K, S, Delta *System
}

func newDescriptorFixture(t *testing.T, dt float64, delays string) descriptorFixture {
	t.Helper()
	P := absorbScopePlant(t, dt, delays == "internal", true)
	kA := []float64{0.4, 0.1, -0.2, 0.3}
	dA := []float64{0.3}
	sA := []float64{0.5, -0.1, 0.2, 0.1}
	if dt == 0 {
		kA = []float64{-1, 0.3, -0.2, -1.7}
		dA = []float64{-2}
		sA = []float64{-0.5, -0.1, 0.2, -3}
	}
	K, err := NewDescriptor(
		mat.NewDense(2, 2, kA),
		mat.NewDense(2, 2, []float64{0.5, 0.1, -0.2, 0.4}),
		mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.25}),
		mat.NewDense(2, 2, []float64{0.05, 0, 0.02, -0.04}),
		mat.NewDense(2, 2, []float64{0.9, 0.2, -0.1, 1.3}), dt)
	if err != nil {
		t.Fatal(err)
	}
	S, err := New(
		mat.NewDense(2, 2, sA),
		mat.NewDense(2, 2, []float64{1, 0.2, 0, 0.7}),
		mat.NewDense(2, 2, []float64{0.4, 0.1, -0.3, 1}),
		mat.NewDense(2, 2, []float64{0, 0.1, 0, 0}), dt)
	if err != nil {
		t.Fatal(err)
	}
	Delta, err := NewDescriptor(
		mat.NewDense(1, 1, dA),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1.7}), dt)
	if err != nil {
		t.Fatal(err)
	}
	scale := 1.0
	if dt == 0 {
		scale = 0.1
	}
	switch delays {
	case "io", "internal":
		absorbScopeCases[0].apply(t, P, scale)
	case "residual":
		absorbScopeCases[1].apply(t, P, scale)
	}
	if delays != "none" {
		if err := K.SetInputDelay([]float64{scale, 0}); err != nil {
			t.Fatal(err)
		}
	}
	return descriptorFixture{P: P, K: K, S: S, Delta: Delta}
}

func (f descriptorFixture) twin(t *testing.T) descriptorFixture {
	t.Helper()
	return descriptorFixture{P: descriptorTwin(t, f.P), K: descriptorTwin(t, f.K), S: descriptorTwin(t, f.S), Delta: descriptorTwin(t, f.Delta)}
}

type descriptorOp struct {
	name       string
	continuous bool
	run        func(f descriptorFixture) (*System, error)
}

func descriptorOps() []descriptorOp {
	ops := []descriptorOp{
		{name: "Series(P,K)", run: func(f descriptorFixture) (*System, error) { return Series(f.P, f.K) }},
		{name: "Series(S,P)", run: func(f descriptorFixture) (*System, error) { return Series(f.S, f.P) }},
		{name: "Series(K,S)", run: func(f descriptorFixture) (*System, error) { return Series(f.K, f.S) }},
		{name: "Parallel(P,K)", run: func(f descriptorFixture) (*System, error) { return Parallel(f.P, f.K) }},
		{name: "Parallel(S,K)", run: func(f descriptorFixture) (*System, error) { return Parallel(f.S, f.K) }},
		{name: "Feedback(P,K)", run: func(f descriptorFixture) (*System, error) { return Feedback(f.P, f.K, -1) }},
		{name: "Feedback(S,K)", run: func(f descriptorFixture) (*System, error) { return Feedback(f.S, f.K, 1) }},
		{name: "Feedback(P,nil)", run: func(f descriptorFixture) (*System, error) { return Feedback(f.P, nil, -1) }},
		{name: "FeedbackApprox(P,K)", run: func(f descriptorFixture) (*System, error) {
			cl, err := Feedback(f.P, f.K, -1)
			if err != nil {
				return nil, err
			}
			return cl.AbsorbDelay()
		}},
		{name: "Append(P,K)", run: func(f descriptorFixture) (*System, error) { return Append(f.P, f.K) }},
		{name: "BlkDiag(P,S,K)", run: func(f descriptorFixture) (*System, error) { return BlkDiag(f.P, f.S, f.K) }},
		{name: "Connect", run: func(f descriptorFixture) (*System, error) {
			aug, err := Append(f.P, f.K)
			if err != nil {
				return nil, err
			}
			Q := mat.NewDense(4, 4, nil)
			Q.Set(1, 2, 0.5)
			Q.Set(2, 0, -0.3)
			return connectGain("Connect", aug, Q, []int{0, 3}, []int{1, 2})
		}},
		{name: "LFT(P,Delta)", run: func(f descriptorFixture) (*System, error) { return LFT(f.P, f.Delta) }},
		{name: "SelectByIndex", run: func(f descriptorFixture) (*System, error) { return f.P.SelectByIndex([]int{1}, []int{0, 1}) }},
		{name: "Augstate", run: func(f descriptorFixture) (*System, error) { return Augstate(f.P) }},
		{name: "DelayModel", run: func(f descriptorFixture) (*System, error) {
			H, tau, err := f.P.GetDelayModel()
			if err != nil {
				return nil, err
			}
			return SetDelayModel(H, tau)
		}},
		{name: "Pade(P)", continuous: true, run: func(f descriptorFixture) (*System, error) { return f.P.Pade(3) }},
		{name: "Pade(Series(P,K))", continuous: true, run: func(f descriptorFixture) (*System, error) {
			s, err := Series(f.P, f.K)
			if err != nil {
				return nil, err
			}
			return s.Pade(4)
		}},
	}
	for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbInternal, AbsorbAll} {
		ops = append(ops, descriptorOp{name: "AbsorbDelay(" + string(scope) + ")", run: func(f descriptorFixture) (*System, error) {
			return f.P.AbsorbDelay(scope)
		}})
	}
	return ops
}

func assertDescriptorMatchesTwin(t *testing.T, label string, want, got *System) {
	t.Helper()
	if n, _, _ := got.Dims(); n > 0 && got.E == nil {
		t.Fatalf("%s: descriptor E dropped", label)
	}
	if got.E != nil {
		n, _, _ := got.Dims()
		if r, c := got.E.Dims(); r != n || c != n {
			t.Fatalf("%s: E %dx%d for %d states", label, r, c, n)
		}
	}
	fw, err := want.FreqResponsePointwise(absorbScopeOmega)
	if err != nil {
		t.Fatalf("%s: twin FreqResponse: %v", label, err)
	}
	candidates := map[string]*System{"explicit": descriptorTwin(t, got)}
	if !got.HasInternalDelay() {
		candidates["pencil"] = got
	}
	for kind, sys := range candidates {
		fg, err := sys.FreqResponsePointwise(absorbScopeOmega)
		if err != nil {
			t.Fatalf("%s/%s: FreqResponse: %v", label, kind, err)
		}
		for k := range fw.Data {
			if d := cmplx.Abs(fw.Data[k] - fg.Data[k]); d > descriptorInterconnectTol {
				t.Fatalf("%s/%s: freq response entry %d differs by %g", label, kind, k, d)
			}
		}
	}
}

func TestDescriptorInterconnectionsMatchExplicitTwin(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		for _, delays := range []string{"none", "io", "residual", "internal"} {
			for _, op := range descriptorOps() {
				if op.continuous && dt != 0 {
					continue
				}
				label := fmt.Sprintf("dt=%g/%s/%s", dt, delays, op.name)
				f := newDescriptorFixture(t, dt, delays)
				tw := f.twin(t)
				want, errWant := op.run(tw)
				got, errGot := op.run(f)
				if errWant != nil || errGot != nil {
					if (errWant == nil) != (errGot == nil) {
						t.Fatalf("%s: twin err %v, descriptor err %v", label, errWant, errGot)
					}
					continue
				}
				assertDescriptorMatchesTwin(t, label, want, got)
			}
		}
	}
}

// singularDescriptorLag realizes 1/(s+a) with an algebraic second state:
// x1' = -a x1 + u, 0 = x1 - x2, y = x2.
func singularDescriptorLag(t *testing.T, a, dt float64) (desc, ref *System) {
	t.Helper()
	var err error
	desc, err = NewDescriptor(
		mat.NewDense(2, 2, []float64{-a, 0, 1, -1}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}), dt)
	if err != nil {
		t.Fatal(err)
	}
	ref, err = New(
		mat.NewDense(1, 1, []float64{-a}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, nil), dt)
	if err != nil {
		t.Fatal(err)
	}
	return desc, ref
}

func TestSingularDescriptorInterconnectionsMatchReducedReference(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		a, b := 1.3, 0.4
		if dt != 0 {
			a, b = -0.5, 0.2
		}
		d1, r1 := singularDescriptorLag(t, a, dt)
		d2, r2 := singularDescriptorLag(t, b, dt)
		delay := 0.2
		if dt != 0 {
			delay = 2
		}
		if err := d1.SetInputDelay([]float64{delay}); err != nil {
			t.Fatal(err)
		}
		if err := r1.SetInputDelay([]float64{delay}); err != nil {
			t.Fatal(err)
		}
		gain := func(sys *System) *System {
			g, err := NewGain(mat.NewDense(1, 1, []float64{0.7}), sys.Dt)
			if err != nil {
				t.Fatal(err)
			}
			return g
		}
		ops := []struct {
			name string
			run  func(x, y *System) (*System, error)
		}{
			{"Series", func(x, y *System) (*System, error) { return Series(x, y) }},
			{"Parallel", func(x, y *System) (*System, error) { return Parallel(x, y) }},
			{"Feedback", func(x, y *System) (*System, error) { return Feedback(x, y, -1) }},
			{"FeedbackGain", func(x, y *System) (*System, error) { return Feedback(y, gain(y), -1) }},
			{"Append", func(x, y *System) (*System, error) { return Append(x, y) }},
			{"AbsorbAll", func(x, y *System) (*System, error) {
				s, err := Series(x, y)
				if err != nil {
					return nil, err
				}
				return s.AbsorbDelay()
			}},
			{"FeedbackApprox", func(x, y *System) (*System, error) {
				cl, err := Feedback(x, y, -1)
				if err != nil {
					return nil, err
				}
				return cl.AbsorbDelay()
			}},
		}
		for _, op := range ops {
			label := fmt.Sprintf("dt=%g/%s", dt, op.name)
			want, err := op.run(r1, r2)
			if err != nil {
				t.Fatalf("%s: reference: %v", label, err)
			}
			got, err := op.run(d1, d2)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			if !got.IsDescriptor() {
				t.Fatalf("%s: descriptor E dropped", label)
			}
			if got.HasInternalDelay() {
				if got, err = got.AbsorbDelay(); err != nil {
					t.Fatalf("%s: absorb: %v", label, err)
				}
				if want, err = want.AbsorbDelay(); err != nil {
					t.Fatalf("%s: absorb reference: %v", label, err)
				}
			}
			fw, err := want.FreqResponsePointwise(absorbScopeOmega)
			if err != nil {
				t.Fatalf("%s: reference FreqResponse: %v", label, err)
			}
			fg, err := got.FreqResponsePointwise(absorbScopeOmega)
			if err != nil {
				t.Fatalf("%s: FreqResponse: %v", label, err)
			}
			for k := range fw.Data {
				if d := cmplx.Abs(fw.Data[k] - fg.Data[k]); d > descriptorInterconnectTol {
					t.Fatalf("%s: freq response entry %d differs by %g", label, k, d)
				}
			}
		}
	}
}

func TestAbsorbDelayContinuousDescriptorPadeEIsBlockIdentity(t *testing.T) {
	sys := absorbScopePlant(t, 0, false, true)
	if err := sys.SetInputDelay([]float64{0.2, 0}); err != nil {
		t.Fatal(err)
	}
	got, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := got.Dims()
	if got.E == nil || n != 3+DefaultPadeOrder {
		t.Fatalf("states %d, E nil %v", n, got.E == nil)
	}
	want := mat.NewDense(n, n, nil)
	setBlock(want, DefaultPadeOrder, DefaultPadeOrder, sys.E)
	for i := range DefaultPadeOrder {
		want.Set(i, i, 1)
	}
	if !mat.Equal(want, got.E) {
		t.Fatalf("E = %v, want %v", mat.Formatted(got.E), mat.Formatted(want))
	}
}
