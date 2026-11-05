package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var nonFiniteValues = []struct {
	name string
	v    float64
}{{"NaN", math.NaN()}, {"+Inf", math.Inf(1)}, {"-Inf", math.Inf(-1)}}

type nonFiniteBlocks struct {
	A, B, C, D, E, B2, C2, D12, D21, D22 *mat.Dense
}

func finiteBlocks() nonFiniteBlocks {
	return nonFiniteBlocks{
		A:   mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3}),
		B:   mat.NewDense(2, 1, []float64{1, 0.5}),
		C:   mat.NewDense(1, 2, []float64{0.3, -1}),
		D:   mat.NewDense(1, 1, []float64{0.2}),
		E:   mat.NewDense(2, 2, []float64{1, 0.1, 0, 2}),
		B2:  mat.NewDense(2, 1, []float64{0.4, -0.2}),
		C2:  mat.NewDense(1, 2, []float64{0.7, 0.1}),
		D12: mat.NewDense(1, 1, []float64{0.05}),
		D21: mat.NewDense(1, 1, []float64{0.3}),
		D22: mat.NewDense(1, 1, []float64{0.1}),
	}
}

func (b *nonFiniteBlocks) field(name string) *mat.Dense {
	return map[string]*mat.Dense{
		"A": b.A, "B": b.B, "C": b.C, "D": b.D, "E": b.E,
		"B2": b.B2, "C2": b.C2, "D12": b.D12, "D21": b.D21, "D22": b.D22,
	}[name]
}

func poison(m *mat.Dense, v float64) {
	r, c := m.Dims()
	m.Set(r-1, c-1, v)
}

func assertNonFiniteRejected(t *testing.T, err error, name string) {
	t.Helper()
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("err = %v, want ErrInvalidArgument", err)
	}
	if !strings.Contains(err.Error(), name+"[") {
		t.Fatalf("err = %v, want it to name %s", err, name)
	}
}

func TestValidate_NonFiniteEntries(t *testing.T) {
	for _, name := range []string{"A", "B", "C", "D", "E", "B2", "C2", "D12", "D21", "D22"} {
		for _, nf := range nonFiniteValues {
			t.Run(name+"/"+nf.name, func(t *testing.T) {
				b := finiteBlocks()
				poison(b.field(name), nf.v)
				sys := &System{
					A: b.A, B: b.B, C: b.C, D: b.D, E: b.E,
					LFT: &LFTDelay{Tau: []float64{0.5}, B2: b.B2, C2: b.C2, D12: b.D12, D21: b.D21, D22: b.D22},
				}
				err := sys.Validate()
				assertNonFiniteRejected(t, err, name)
				if !strings.HasPrefix(err.Error(), "Validate: ") {
					t.Fatalf("err = %v, want Validate prefix", err)
				}
			})
		}
	}
}

func TestValidate_FiniteDescriptorWithInternalDelay(t *testing.T) {
	b := finiteBlocks()
	sys := &System{
		A: b.A, B: b.B, C: b.C, D: b.D, E: b.E,
		LFT: &LFTDelay{Tau: []float64{0.5}, B2: b.B2, C2: b.C2, D12: b.D12, D21: b.D21, D22: b.D22},
	}
	if err := sys.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestNew_NonFiniteEntries(t *testing.T) {
	for _, name := range []string{"A", "B", "C", "D"} {
		for _, nf := range nonFiniteValues {
			t.Run(name+"/"+nf.name, func(t *testing.T) {
				b := finiteBlocks()
				poison(b.field(name), nf.v)
				sys, err := New(b.A, b.B, b.C, b.D, 0)
				if sys != nil {
					t.Fatal("non-nil system with error")
				}
				assertNonFiniteRejected(t, err, name)
			})
		}
	}
}

func TestNewWithDelay_NonFiniteEntries(t *testing.T) {
	for _, nf := range nonFiniteValues {
		b := finiteBlocks()
		poison(b.A, nf.v)
		_, err := NewWithDelay(b.A, b.B, b.C, b.D, mat.NewDense(1, 1, []float64{1}), 0)
		assertNonFiniteRejected(t, err, "A")
	}
}

func TestNewDescriptor_NonFiniteEntries(t *testing.T) {
	for _, name := range []string{"A", "B", "C", "D", "E"} {
		for _, nf := range nonFiniteValues {
			t.Run(name+"/"+nf.name, func(t *testing.T) {
				b := finiteBlocks()
				poison(b.field(name), nf.v)
				sys, err := NewDescriptor(b.A, b.B, b.C, b.D, b.E, 0)
				if sys != nil {
					t.Fatal("non-nil system with error")
				}
				assertNonFiniteRejected(t, err, name)
			})
		}
	}
}

func TestNewGain_NonFiniteEntries(t *testing.T) {
	for _, nf := range nonFiniteValues {
		D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
		poison(D, nf.v)
		sys, err := NewGain(D, 0.1)
		if sys != nil {
			t.Fatal("non-nil system with error")
		}
		assertNonFiniteRejected(t, err, "D")
	}
}

func TestNewFromSlices_NonFiniteEntries(t *testing.T) {
	for _, name := range []string{"A", "B", "C", "D"} {
		for _, nf := range nonFiniteValues {
			t.Run(name+"/"+nf.name, func(t *testing.T) {
				a := []float64{-1, 2, -0.5, -3}
				bv := []float64{1, 0.5}
				c := []float64{0.3, -1}
				d := []float64{0.2}
				map[string][]float64{"A": a, "B": bv, "C": c, "D": d}[name][0] = nf.v
				sys, err := NewFromSlices(2, 1, 1, a, bv, c, d, 0)
				if sys != nil {
					t.Fatal("non-nil system with error")
				}
				assertNonFiniteRejected(t, err, name)
			})
		}
	}
	for _, nf := range nonFiniteValues {
		_, err := NewFromSlices(0, 1, 1, nil, nil, nil, []float64{nf.v}, 0)
		assertNonFiniteRejected(t, err, "D")
	}
}

func TestSetInternalDelay_NonFiniteBlocks(t *testing.T) {
	for _, name := range []string{"B2", "C2", "D12", "D21", "D22"} {
		for _, nf := range nonFiniteValues {
			t.Run(name+"/"+nf.name, func(t *testing.T) {
				b := finiteBlocks()
				sys, err := New(b.A, b.B, b.C, b.D, 0)
				if err != nil {
					t.Fatal(err)
				}
				if err := sys.SetInternalDelay([]float64{0.5}, b.B2, b.C2, b.D12, b.D21, b.D22); err != nil {
					t.Fatal(err)
				}
				prev := sys.LFT
				poison(b.field(name), nf.v)
				err = sys.SetInternalDelay([]float64{0.7}, b.B2, b.C2, b.D12, b.D21, b.D22)
				assertNonFiniteRejected(t, err, name)
				if sys.LFT != prev {
					t.Fatal("rejected SetInternalDelay changed sys.LFT")
				}
			})
		}
	}
}
