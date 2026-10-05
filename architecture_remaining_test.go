package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func remainingArchitectureMIMO(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0,
			-2, -3, 1,
			0.5, -0.25, -1.5,
		},
		[]float64{
			1, 0,
			0, 1,
			1, -1,
		},
		[]float64{
			1, 0, 0.5,
			0, 1, -0.25,
		},
		[]float64{
			0.1, -0.2,
			0.05, 0.3,
		},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"u-fast", "u-slow"}
	sys.OutputName = []string{"y-top", "y-bottom"}
	sys.StateName = []string{"x1", "x2", "x3"}
	sys.Notes = "remaining architecture regression model"
	return sys
}

func TestRemainingArchitectureDelayTopologyPublicOperations(t *testing.T) {
	base := remainingArchitectureMIMO(t, 0)

	split := base.Copy()
	if err := split.SetInputDelay([]float64{0.1, 0.2}); err != nil {
		t.Fatal(err)
	}
	if err := split.SetOutputDelay([]float64{0.05, 0.15}); err != nil {
		t.Fatal(err)
	}
	if err := split.SetDelay(mat.NewDense(2, 2, []float64{
		0.0, 0.05,
		0.1, 0.15,
	})); err != nil {
		t.Fatal(err)
	}

	collapsed := base.Copy()
	if err := collapsed.SetDelay(mustTotalDelay(t, split)); err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.2, 1.0, 3.5}
	assertFreqResponseApprox(t, split, collapsed, omega, 1e-10)

	splitLFT, err := split.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	collapsedLFT, err := collapsed.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	assertFreqResponseApprox(t, splitLFT, collapsedLFT, omega, 1e-10)

	controller, err := NewGain(mat.NewDense(2, 2, []float64{0.2, 0.05, -0.1, 0.15}), 0)
	if err != nil {
		t.Fatal(err)
	}
	safeSplit, err := Feedback(split, controller, -1, WithPadeOrder(2))
	if err != nil {
		t.Fatal(err)
	}
	manualPlant, err := replaceExplicitIODelaysWithPade(split, 2)
	if err != nil {
		t.Fatal(err)
	}
	manualSafe, err := Feedback(manualPlant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	assertFreqResponseApprox(t, safeSplit, manualSafe, omega, 1e-8)
}

func replaceExplicitIODelaysWithPade(sys *System, padeOrder int) (*System, error) {
	_, m, p := sys.Dims()
	cur := sys.Copy()
	if cur.Delay != nil {
		inDel, outDel, residual := DecomposeIODelay(cur.Delay)
		if delayMatrixHasNonzero(residual) {
			return nil, ErrFeedbackDelay
		}
		cur.Delay = nil
		cur.InputDelay = mergeDelays(cur.InputDelay, inDel)
		cur.OutputDelay = mergeDelays(cur.OutputDelay, outDel)
	}

	result := &System{A: cur.A, B: cur.B, C: cur.C, D: cur.D, Dt: cur.Dt}
	for j := m - 1; j >= 0; j-- {
		if cur.InputDelay == nil || cur.InputDelay[j] == 0 {
			continue
		}
		pade, err := PadeDelay(cur.InputDelay[j], padeOrder)
		if err != nil {
			return nil, err
		}
		diag, err := buildDiagWithPade(j, m, pade)
		if err != nil {
			return nil, err
		}
		result, err = Series(diag, result)
		if err != nil {
			return nil, err
		}
	}
	for i := p - 1; i >= 0; i-- {
		if cur.OutputDelay == nil || cur.OutputDelay[i] == 0 {
			continue
		}
		pade, err := PadeDelay(cur.OutputDelay[i], padeOrder)
		if err != nil {
			return nil, err
		}
		diag, err := buildDiagWithPade(i, p, pade)
		if err != nil {
			return nil, err
		}
		result, err = Series(result, diag)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

func TestRemainingArchitectureConversionPlannerBehavior(t *testing.T) {
	sys := remainingArchitectureMIMO(t, 0)
	sys.InputDelay = []float64{0.2, 0.3}
	sys.OutputDelay = []float64{0.1, 0.0}

	discDefault, err := sys.DiscretizeWithOpts(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	discZOH, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	assertDenseApprox(t, discDefault.A, discZOH.A, 1e-12)
	assertDenseApprox(t, discDefault.B, discZOH.B, 1e-12)
	if !stringSlicesEqual(discDefault.InputName, sys.InputName) || !stringSlicesEqual(discDefault.OutputName, sys.OutputName) {
		t.Fatalf("names not preserved through default conversion: inputs=%v outputs=%v", discDefault.InputName, discDefault.OutputName)
	}
	if !floatSlicesEqual(discDefault.InputDelay, []float64{2, 3}) {
		t.Fatalf("InputDelay = %v, want [2 3]", discDefault.InputDelay)
	}

	resampled, err := discDefault.D2D(0.05, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if resampled.Dt != 0.05 {
		t.Fatalf("D2D Dt = %v, want 0.05", resampled.Dt)
	}
	if !stringSlicesEqual(resampled.StateName, sys.StateName) {
		t.Fatalf("state names = %v, want %v", resampled.StateName, sys.StateName)
	}

	if _, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethod("unknown")}); err == nil {
		t.Fatal("expected unknown C2D method to fail")
	}
	if _, err := discDefault.D2C(C2DMethod("unknown")); err == nil {
		t.Fatal("expected unknown D2C method to fail")
	}
}

func TestRemainingArchitectureTimeDomainPolicyPublicConsumers(t *testing.T) {
	cont := remainingArchitectureMIMO(t, 0)
	disc, err := cont.DiscretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := disc.SetDelay(mat.NewDense(2, 2, []float64{1, 2, 0, 3})); err != nil {
		t.Fatal(err)
	}

	otherDt := disc.Copy()
	otherDt.Dt = 0.2
	if _, err := Series(disc, otherDt); !errors.Is(err, ErrDomainMismatch) {
		t.Fatalf("Series domain error = %v, want ErrDomainMismatch", err)
	}

	omega := []float64{0.2, 1.2}
	resp, err := disc.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*disc.Dt))
		eval, err := disc.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < resp.P; i++ {
			for j := 0; j < resp.M; j++ {
				assertComplexApprox(t, resp.At(k, i, j), eval[i][j], 1e-10)
			}
		}
	}

	contAgain, err := disc.D2C(C2DMethodTustin)
	if err != nil {
		t.Fatal(err)
	}
	delay := mustTotalDelay(t, contAgain)
	if delay == nil {
		t.Fatal("continuous delay is nil")
	}
	if got := delay.At(0, 1); math.Abs(got-0.2) > 1e-12 {
		t.Fatalf("delay(0,1) = %v, want 0.2", got)
	}
}

func TestRemainingArchitectureSignalMetadataMappings(t *testing.T) {
	sys := remainingArchitectureMIMO(t, 0)
	selected, err := sys.SelectByName([]string{"u-slow"}, []string{"y-bottom"})
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(selected.InputName, []string{"u-slow"}) {
		t.Fatalf("selected inputs = %v", selected.InputName)
	}
	if !stringSlicesEqual(selected.OutputName, []string{"y-bottom"}) {
		t.Fatalf("selected outputs = %v", selected.OutputName)
	}
	if !stringSlicesEqual(selected.StateName, sys.StateName) {
		t.Fatalf("selected states = %v, want %v", selected.StateName, sys.StateName)
	}

	foh, err := sys.DiscretizeFOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(foh.StateName, sys.StateName) {
		t.Fatalf("FOH state names = %v, want %v", foh.StateName, sys.StateName)
	}

	series, err := Series(sys, sys)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(series.InputName, sys.InputName) || !stringSlicesEqual(series.OutputName, sys.OutputName) {
		t.Fatalf("series names inputs=%v outputs=%v", series.InputName, series.OutputName)
	}
}

func floatSlicesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
