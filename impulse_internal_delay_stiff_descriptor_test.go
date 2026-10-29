package controlsys

import (
	"errors"
	"math"
	"testing"
)

// Impulse of a continuous descriptor with internal delays whose E is too
// ill-conditioned for E⁻¹A returns ErrInternalDelayImpulse even with
// D21 = 0 (43H73S), as MATLAB's impulse rejects every continuous
// internal-delay model: the stiff modes pass the delays a near-Dirac, so the
// sampled response was off by 0.04 to 0.67.
func TestImpulseInternalDelayStiffDescriptorRejected(t *testing.T) {
	names, systems := stiffInternalDelayCases(t)
	names, systems = append(names, "cond(E) 1e10"), append(systems, stiffGridCase(t, 10))
	for k, sys := range systems {
		sys.LFT.D21.Zero()
		if _, err := Impulse(sys, 1); !errors.Is(err, ErrInternalDelayImpulse) {
			t.Errorf("%s: Impulse error %v, want ErrInternalDelayImpulse", names[k], err)
		}
	}
}

// Impulse of a well-conditioned descriptor with an internal delay and
// D21 = 0 is still sampled: it matches the impulse of its explicit form.
func TestImpulseInternalDelayWellConditionedDescriptorSampled(t *testing.T) {
	sys := stiffGridCase(t, 0)
	sys.LFT.D21.Zero()
	got, err := Impulse(sys, 1)
	if err != nil {
		t.Fatal(err)
	}
	ex, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	want, err := Impulse(ex, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.T) != len(want.T) {
		t.Fatalf("%d samples, want %d", len(got.T), len(want.T))
	}
	worst := 0.0
	for s := range got.T {
		worst = math.Max(worst, stepOracleRel(got.Y.At(0, s), want.Y.At(0, s)))
	}
	if worst > 1e-12 {
		t.Errorf("Impulse off the explicit form by %.3g", worst)
	}
}
