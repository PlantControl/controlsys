package controlsys

import (
	"context"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// benchSysNonSym builds a stable system with non-symmetric A for benchmark workloads.
// A has sub/superdiagonal coupling so transposition bugs surface.
func benchSysNonSym(tb testing.TB, n, m, p int) *System {
	tb.Helper()
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1.0)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1*float64(i+1))
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < n && i < m; i++ {
		B.Set(i, i, 1)
	}
	if n > m {
		for i := m; i < n; i++ {
			B.Set(i, i%m, 0.5)
		}
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p && i < n; i++ {
		C.Set(i, i, 1)
	}
	if p > 1 && n > 1 {
		C.Set(0, 1, 0.3)
	}
	D := mat.NewDense(p, m, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		tb.Fatal(err)
	}
	return sys
}

func benchFRDFromSys(tb testing.TB, sys *System, nw int) *FRD {
	tb.Helper()
	omega := logspace(-2, 3, nw)
	f, err := sys.FRD(omega)
	if err != nil {
		tb.Fatal(err)
	}
	return f
}

func benchIntegratorMIMOSystem(tb testing.TB, n, m, p int) *System {
	tb.Helper()
	A := mat.NewDense(n, n, nil)
	for i := 1; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.25)
		if i > 1 {
			A.Set(i, i-1, 0.2)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.05)
		}
	}
	B := mat.NewDense(n, m, nil)
	for j := range m {
		B.Set(0, j, float64(j+1))
	}
	for i := 1; i < n; i++ {
		B.Set(i, i%m, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := range p {
		if i%2 == 0 {
			C.Set(i, 0, 1)
		}
		if n > 1 {
			C.Set(i, 1+i%(n-1), 1)
		}
	}
	D := mat.NewDense(p, m, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		tb.Fatal(err)
	}
	return sys
}

func benchModelArray(b *testing.B, count, n, m, p int) *ModelArray {
	b.Helper()
	models := make([]*System, count)
	for i := range models {
		if i%7 == 3 {
			continue
		}
		sys := benchSysNonSym(b, n, m, p)
		sys.A.Set(0, 0, sys.A.At(0, 0)-float64(i)*0.01)
		models[i] = sys
	}
	arr, err := NewModelArray([]int{count}, models)
	if err != nil {
		b.Fatal(err)
	}
	return arr
}

// --------------- DC gain ---------------

func BenchmarkDCGain_StableMIMO_N10_M4_P6(b *testing.B) {
	sys := benchSysNonSym(b, 10, 4, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.DCGain(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDCGain_IntegratorMIMO_N10_M4_P6(b *testing.B) {
	sys := benchIntegratorMIMOSystem(b, 10, 4, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.DCGain(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDCGain_IntegratorMIMO_N40_M8_P8(b *testing.B) {
	sys := benchIntegratorMIMOSystem(b, 40, 8, 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.DCGain(); err != nil {
			b.Fatal(err)
		}
	}
}

// --------------- FRD stack ---------------

func BenchmarkSystemFRD_SISO_200(b *testing.B)   { benchSystemFRD(b, 2, 1, 1, 200) }
func BenchmarkSystemFRD_SISO_2000(b *testing.B)  { benchSystemFRD(b, 2, 1, 1, 2000) }
func BenchmarkSystemFRD_SISO_10000(b *testing.B) { benchSystemFRD(b, 2, 1, 1, 10000) }
func BenchmarkSystemFRD_MIMO_200(b *testing.B)   { benchSystemFRD(b, 10, 3, 4, 200) }
func BenchmarkSystemFRD_MIMO_2000(b *testing.B)  { benchSystemFRD(b, 10, 3, 4, 2000) }
func BenchmarkSystemFRD_MIMO_10000(b *testing.B) { benchSystemFRD(b, 10, 3, 4, 10000) }
func BenchmarkSystemFRD_Large_2000(b *testing.B) { benchSystemFRD(b, 50, 4, 6, 2000) }

func benchSystemFRD(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(b, n, m, p)
	omega := logspace(-2, 3, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FRD(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDSigma_SISO_200(b *testing.B)   { benchFRDSigma(b, 2, 1, 1, 200) }
func BenchmarkFRDSigma_SISO_2000(b *testing.B)  { benchFRDSigma(b, 2, 1, 1, 2000) }
func BenchmarkFRDSigma_MIMO_200(b *testing.B)   { benchFRDSigma(b, 10, 3, 4, 200) }
func BenchmarkFRDSigma_MIMO_2000(b *testing.B)  { benchFRDSigma(b, 10, 3, 4, 2000) }
func BenchmarkFRDSigma_MIMO_10000(b *testing.B) { benchFRDSigma(b, 10, 3, 4, 10000) }
func BenchmarkFRDSigma_Large_2000(b *testing.B) { benchFRDSigma(b, 50, 4, 6, 2000) }

func benchFRDSigma(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(b, n, m, p)
	f := benchFRDFromSys(b, sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := f.Sigma(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDMargin_200(b *testing.B)   { benchFRDMargin(b, 4, 200) }
func BenchmarkFRDMargin_2000(b *testing.B)  { benchFRDMargin(b, 4, 2000) }
func BenchmarkFRDMargin_10000(b *testing.B) { benchFRDMargin(b, 4, 10000) }

func BenchmarkFRDMargin_NegGM_2000(b *testing.B) {
	A := mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1})
	B := mat.NewDense(3, 1, []float64{0, 0, 10})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	f := benchFRDFromSys(b, sys, 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDMargin(f); err != nil {
			b.Fatal(err)
		}
	}
}

func benchFRDMargin(b *testing.B, n, nw int) {
	sys := benchSysNonSym(b, n, 1, 1)
	f := benchFRDFromSys(b, sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDMargin(f); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDSigma_MIMO5x5_2000(b *testing.B) { benchFRDSigma(b, 20, 5, 5, 2000) }

func BenchmarkFRDSeries_SISO_200(b *testing.B)   { benchFRDSeries(b, 2, 1, 1, 200) }
func BenchmarkFRDSeries_SISO_2000(b *testing.B)  { benchFRDSeries(b, 2, 1, 1, 2000) }
func BenchmarkFRDSeries_MIMO_200(b *testing.B)   { benchFRDSeries(b, 10, 3, 3, 200) }
func BenchmarkFRDSeries_MIMO_2000(b *testing.B)  { benchFRDSeries(b, 10, 3, 3, 2000) }
func BenchmarkFRDSeries_MIMO_10000(b *testing.B) { benchFRDSeries(b, 10, 3, 3, 10000) }

func benchFRDSeries(b *testing.B, n, m, p, nw int) {
	s1 := benchSysNonSym(b, n, m, p)
	s2 := benchSysNonSym(b, n, p, m)
	f1 := benchFRDFromSys(b, s1, nw)
	f2 := benchFRDFromSys(b, s2, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDSeries(f1, f2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDParallel_SISO_2000(b *testing.B) { benchFRDParallel(b, 2, 1, 1, 2000) }
func BenchmarkFRDParallel_MIMO_2000(b *testing.B) { benchFRDParallel(b, 10, 3, 3, 2000) }

func benchFRDParallel(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(b, n, m, p)
	f1 := benchFRDFromSys(b, sys, nw)
	f2 := benchFRDFromSys(b, sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDParallel(f1, f2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDFeedback_SISO_200(b *testing.B)   { benchFRDFeedback(b, 2, 1, 1, 200) }
func BenchmarkFRDFeedback_SISO_2000(b *testing.B)  { benchFRDFeedback(b, 2, 1, 1, 2000) }
func BenchmarkFRDFeedback_MIMO_200(b *testing.B)   { benchFRDFeedback(b, 10, 3, 3, 200) }
func BenchmarkFRDFeedback_MIMO_2000(b *testing.B)  { benchFRDFeedback(b, 10, 3, 3, 2000) }
func BenchmarkFRDFeedback_MIMO_10000(b *testing.B) { benchFRDFeedback(b, 10, 3, 3, 10000) }
func BenchmarkFRDFeedback_MIMO8x8_2000(b *testing.B) {
	benchFRDFeedback(b, 20, 8, 8, 2000)
}

func benchFRDFeedback(b *testing.B, n, m, p, nw int) {
	plant := benchSysNonSym(b, n, m, p)
	ctrl := benchSysNonSym(b, n/2+1, p, m)
	fp := benchFRDFromSys(b, plant, nw)
	fc := benchFRDFromSys(b, ctrl, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDFeedback(fp, fc, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDAbs_MIMO_2000(b *testing.B) {
	frd := benchFRDFromSys(b, benchSysNonSym(b, 10, 3, 4), 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frd.Abs()
	}
}

func BenchmarkFRDSelectFrequencies_MIMO_2000(b *testing.B) {
	frd := benchFRDFromSys(b, benchSysNonSym(b, 10, 3, 4), 2000)
	indices := make([]int, 0, 1000)
	for i := 0; i < 2000; i += 2 {
		indices = append(indices, i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := frd.SelectFrequencies(indices); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDPeakGain_MIMO_2000(b *testing.B) {
	frd := benchFRDFromSys(b, benchSysNonSym(b, 10, 3, 4), 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := frd.PeakGain(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModelArrayFreqResponse_MIMO_16x200(b *testing.B) {
	arr := benchModelArray(b, 16, 10, 3, 4)
	omega := logspace(-2, 3, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := arr.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModelArrayFreqResponseManualLoop_MIMO_16x200(b *testing.B) {
	arr := benchModelArray(b, 16, 10, 3, 4)
	models := arr.models
	omega := logspace(-2, 3, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, sys := range models {
			if sys == nil {
				continue
			}
			if _, err := sys.FreqResponse(omega); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// --------------- Time-response wrappers ---------------

func BenchmarkStep_SISO_N2(b *testing.B)  { benchStep(b, 2, 1, 1) }
func BenchmarkStep_SISO_N10(b *testing.B) { benchStep(b, 10, 1, 1) }
func BenchmarkStep_SISO_N50(b *testing.B) { benchStep(b, 50, 1, 1) }
func BenchmarkStep_MIMO_N10(b *testing.B) { benchStep(b, 10, 3, 4) }

func benchStep(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(b, n, m, p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Step(sys, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStepInfo_SISO_N10(b *testing.B) { benchStepInfo(b, 10, 1, 1) }
func BenchmarkStepInfo_MIMO_N10(b *testing.B) { benchStepInfo(b, 10, 3, 4) }

func benchStepInfo(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(b, n, m, p)
	resp, err := Step(sys, 10)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := StepInfo(resp, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModelArrayStep_MIMO_16(b *testing.B) {
	arr := benchModelArray(b, 16, 10, 3, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := arr.Step(10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModelArrayStepManualLoop_MIMO_16(b *testing.B) {
	arr := benchModelArray(b, 16, 10, 3, 4)
	models := arr.models
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, sys := range models {
			if sys == nil {
				continue
			}
			if _, err := Step(sys, 10); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkImpulse_SISO_N2(b *testing.B)  { benchImpulse(b, 2, 1, 1) }
func BenchmarkImpulse_SISO_N10(b *testing.B) { benchImpulse(b, 10, 1, 1) }
func BenchmarkImpulse_SISO_N50(b *testing.B) { benchImpulse(b, 50, 1, 1) }
func BenchmarkImpulse_MIMO_N10(b *testing.B) { benchImpulse(b, 10, 3, 4) }

func benchImpulse(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(b, n, m, p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Impulse(sys, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLsim_SISO_N2(b *testing.B)  { benchLsimB(b, 2, 1, 1, 500) }
func BenchmarkLsim_SISO_N10(b *testing.B) { benchLsimB(b, 10, 1, 1, 500) }
func BenchmarkLsim_SISO_N50(b *testing.B) { benchLsimB(b, 50, 1, 1, 500) }
func BenchmarkLsim_MIMO_N10(b *testing.B) { benchLsimB(b, 10, 3, 4, 500) }
func BenchmarkLsim_SISO_1e4(b *testing.B) { benchLsimB(b, 4, 1, 1, 10000) }
func BenchmarkLsim_MIMO_1e4(b *testing.B) { benchLsimB(b, 10, 4, 4, 10000) }

func BenchmarkLsim_Discrete_SISO(b *testing.B) {
	sys := benchSysNonSym(b, 4, 1, 1)
	dsys, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		b.Fatal(err)
	}
	steps := 1000
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * 0.01
	}
	u := mat.NewDense(steps, 1, nil)
	for k := range steps {
		u.Set(k, 0, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lsim(dsys, u, t, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func benchLsimB(b *testing.B, n, m, p, steps int) {
	sys := benchSysNonSym(b, n, m, p)
	dt := 0.01
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * dt
	}
	u := mat.NewDense(steps, m, nil)
	for k := range steps {
		for j := range m {
			u.Set(k, j, math.Sin(float64(k)*dt*float64(j+1)))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lsim(sys, u, t, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// --------------- Loopsens ---------------

func BenchmarkLoopsens_SISO_N2(b *testing.B)  { benchLoopsens(b, 2, 1, 1) }
func BenchmarkLoopsens_SISO_N10(b *testing.B) { benchLoopsens(b, 10, 1, 1) }
func BenchmarkLoopsens_MIMO_N10(b *testing.B) { benchLoopsens(b, 10, 3, 3) }

func benchLoopsens(b *testing.B, n, m, p int) {
	P := benchSysNonSym(b, n, m, p)
	C := benchSysNonSym(b, n/2+1, p, m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Loopsens(P, C); err != nil {
			b.Fatal(err)
		}
	}
}

// --------------- Transform / decomposition ---------------

func BenchmarkCanon_Modal_N2(b *testing.B)   { benchCanon(b, CanonModal, 2) }
func BenchmarkCanon_Modal_N10(b *testing.B)  { benchCanon(b, CanonModal, 10) }
func BenchmarkCanon_Modal_N50(b *testing.B)  { benchCanon(b, CanonModal, 50) }
func BenchmarkCanon_Modal_N100(b *testing.B) { benchCanon(b, CanonModal, 100) }
func BenchmarkCanon_Comp_N2(b *testing.B)    { benchCanon(b, CanonCompanion, 2) }
func BenchmarkCanon_Comp_N10(b *testing.B)   { benchCanon(b, CanonCompanion, 10) }

func benchCanon(b *testing.B, form CanonForm, n int) {
	sys := benchSysNonSym(b, n, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Canon(sys, form); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDescriptorToExplicit_N10(b *testing.B) {
	sys := benchDescriptorSystem(b, 10, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.ToExplicit(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDescriptorFreqResponse_MIMO_N10x200(b *testing.B) {
	benchDescriptorFreqResponse(b, 10, 3, 3, 200)
}

func BenchmarkDescriptorFreqResponse_MIMO_N50x200(b *testing.B) {
	benchDescriptorFreqResponse(b, 50, 5, 5, 200)
}

func benchDescriptorFreqResponse(b *testing.B, n, m, p, nw int) {
	sys := benchDescriptorSystem(b, n, m, p)
	omega := logspace(-2, 3, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFixedInputReduction_N10(b *testing.B) {
	sys := benchSysNonSym(b, 10, 4, 2)
	fixed := map[int]float64{1: 0.5, 3: -2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FixedInputReduction(fixed, "offset"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTunableGainCurrentSystem_4x4(b *testing.B) {
	block := benchTunableGain(b, 4, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := block.CurrentSystem(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTunableGainSampleCurrentSystem_4x4(b *testing.B) {
	block := benchTunableGain(b, 4, 4)
	values := map[string]float64{"p_0_0": 2.5, "p_3_3": -1.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampled, err := block.Sample(values)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := sampled.CurrentSystem(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeneralizedCurrentSystem_SISO(b *testing.B) {
	k, err := newBoundedReal("K", 2, 0, 10)
	if err != nil {
		b.Fatal(err)
	}
	block := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))
	gm, err := NewGeneralizedModel("loop", block)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gm.CurrentSystem(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeneralizedClosedLoop_SISO(b *testing.B) {
	plant := benchSysNonSym(b, 4, 1, 1)
	k, err := newBoundedReal("K", 2, 0, 10)
	if err != nil {
		b.Fatal(err)
	}
	block := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))
	gm, err := NewGeneralizedClosedLoop("loop", plant, block, "u")
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gm.ComplementarySensitivity("u"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTuningGoalWeightedGain_SISO(b *testing.B) {
	sys := benchSysNonSym(b, 4, 1, 1)
	sys.InputName, sys.OutputName = []string{"r"}, []string{"y"}
	goal := mustOK(NewGainGoal("r", "y", 10))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := goal.Evaluate(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTuningGoalWeightedGain_MIMO(b *testing.B) {
	sys := benchSysNonSym(b, 8, 3, 3)
	sys.InputName, sys.OutputName = expandName("r", 3), expandName("y", 3)
	goal := mustOK(NewGainGoal("r", "y", 10))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := goal.Evaluate(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTuningGoalDynamicWeightedGain_MIMO(b *testing.B) {
	sys := benchSysNonSym(b, 8, 3, 3)
	sys.InputName, sys.OutputName = expandName("r", 3), expandName("y", 3)
	goal, err := NewWeightedGainGoal("r", "y", benchSysNonSym(b, 2, 3, 3), benchSysNonSym(b, 2, 3, 3))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := goal.Evaluate(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSystune_SISO(b *testing.B) {
	plant := benchSysNonSym(b, 2, 1, 1)
	k, err := newBoundedReal("K", 0.5, 0.1, 3)
	if err != nil {
		b.Fatal(err)
	}
	controller := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))
	model, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		b.Fatal(err)
	}
	goals := []TuningGoal{mustOK(NewGainGoal("u", "u", 10))}
	opts := &SystuneOptions{GridPoints: 5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Systune(context.Background(), model, goals, nil, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSystune_MIMO(b *testing.B) {
	plant := benchSysNonSym(b, 2, 2, 2)
	k1, err := newBoundedReal("K1", 0.5, 0.1, 2)
	if err != nil {
		b.Fatal(err)
	}
	k2, err := newBoundedReal("K2", 0.5, 0.1, 2)
	if err != nil {
		b.Fatal(err)
	}
	controller := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k1, fixedBenchReal(b, "z12", 0)}, {fixedBenchReal(b, "z21", 0), k2}}))
	model, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		b.Fatal(err)
	}
	goals := []TuningGoal{mustOK(NewGainGoal("u", "u", 10))}
	opts := &SystuneOptions{GridPoints: 3}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Systune(context.Background(), model, goals, nil, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPhysicalAssembly_8Components(b *testing.B) {
	components := make([]PhysicalComponent, 8)
	for i := range components {
		sys := benchDescriptorSystem(b, 4, 1, 1)
		sys.InputName = []string{"force"}
		sys.OutputName = []string{"position"}
		sys.StateName = autoLabel("x", 4)
		c, err := NewPhysicalComponent(
			fmt.Sprintf("c%d", i),
			sys,
			[]PhysicalPort{{Name: "mount", Kind: PhysicalPortDisplacement, Dimension: 1}},
		)
		if err != nil {
			b.Fatal(err)
		}
		components[i] = c
	}
	connections := make([]PhysicalConnection, 0, len(components)-1)
	for i := 0; i < len(components)-1; i++ {
		connections = append(connections, PhysicalConnection{
			FromComponent: fmt.Sprintf("c%d", i),
			FromPort:      "mount",
			ToComponent:   fmt.Sprintf("c%d", i+1),
			ToPort:        "mount",
		})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AssemblePhysical("chain", components, connections); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModalTruncate_N50_Order10(b *testing.B) {
	sys := benchSysNonSym(b, 50, 2, 2)
	opts := &ModalTruncateOptions{Order: 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ModalTruncate(sys, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPassivity_SISO(b *testing.B) {
	sys := makeSISO(-1, 1, 1, 0)
	opts := &PassivityOptions{Omega: logspace(-2, 2, 200)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Passive(sys, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFRDPassivity_MIMO(b *testing.B) {
	sys := benchSysNonSym(b, 4, 2, 2)
	frd, err := sys.FRD(logspace(-2, 2, 200))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := FRDPassive(frd, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func benchDescriptorSystem(b *testing.B, n, m, p int) *System {
	b.Helper()
	sys := benchSysNonSym(b, n, m, p)
	E := mat.NewDense(n, n, nil)
	for i := range n {
		E.Set(i, i, 1+0.1*float64(i+1))
	}
	sys.E = E
	return sys
}

func fixedBenchReal(tb testing.TB, name string, value float64) *TunableReal {
	tb.Helper()
	param, err := NewTunableReal(name, value)
	if err != nil {
		tb.Fatal(err)
	}
	param.SetFixed(true)
	return param
}

func benchTunableGain(b *testing.B, p, m int) *TunableGain {
	b.Helper()
	params := make([][]*TunableReal, p)
	for i := range params {
		params[i] = make([]*TunableReal, m)
		for j := range params[i] {
			param, err := newBoundedReal(fmt.Sprintf("p_%d_%d", i, j), float64(i-j), -10, 10)
			if err != nil {
				b.Fatal(err)
			}
			params[i][j] = param
		}
	}
	return mustOK(tunableGainWith("gain", params))
}

func BenchmarkStabsep_N2(b *testing.B)   { benchStabsep(b, 2) }
func BenchmarkStabsep_N10(b *testing.B)  { benchStabsep(b, 10) }
func BenchmarkStabsep_N50(b *testing.B)  { benchStabsep(b, 50) }
func BenchmarkStabsep_N100(b *testing.B) { benchStabsep(b, 100) }

func benchStabsep(b *testing.B, n int) {
	A := mat.NewDense(n, n, nil)
	for i := range n {
		if i%2 == 0 {
			A.Set(i, i, -float64(i+1)*0.3)
		} else {
			A.Set(i, i, float64(i)*0.2)
		}
		if i > 0 {
			A.Set(i, i-1, 0.5)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, 2, nil)
	B.Set(0, 0, 1)
	if n > 1 {
		B.Set(1, 1, 1)
	}
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	if n > 1 {
		C.Set(1, 1, 1)
	}
	D := mat.NewDense(2, 2, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Stabsep(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModsep_N10(b *testing.B) { benchModsep(b, 10) }
func BenchmarkModsep_N50(b *testing.B) { benchModsep(b, 50) }

func benchModsep(b *testing.B, n int) {
	sys := benchSysNonSym(b, n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Modsep(sys, 1.0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSsbal_N2(b *testing.B)  { benchSsbalB(b, 2) }
func BenchmarkSsbal_N10(b *testing.B) { benchSsbalB(b, 10) }
func BenchmarkSsbal_N50(b *testing.B) { benchSsbalB(b, 50) }

func benchSsbalB(b *testing.B, n int) {
	sys := benchSysNonSym(b, n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Ssbal(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPrescale_N2(b *testing.B)  { benchPrescaleB(b, 2) }
func BenchmarkPrescale_N10(b *testing.B) { benchPrescaleB(b, 10) }
func BenchmarkPrescale_N50(b *testing.B) { benchPrescaleB(b, 50) }

func benchPrescaleB(b *testing.B, n int) {
	sys := benchSysNonSym(b, n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Prescale(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSminreal_N2(b *testing.B)  { benchSminrealB(b, 2) }
func BenchmarkSminreal_N10(b *testing.B) { benchSminrealB(b, 10) }
func BenchmarkSminreal_N50(b *testing.B) { benchSminrealB(b, 50) }

func benchSminrealB(b *testing.B, n int) {
	sys := benchSysNonSym(b, n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Sminreal(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInv_SISO_N2(b *testing.B)  { benchInvB(b, 2, 1, 1) }
func BenchmarkInv_SISO_N10(b *testing.B) { benchInvB(b, 10, 1, 1) }
func BenchmarkInv_MIMO_N10(b *testing.B) { benchInvB(b, 10, 3, 3) }

func benchInvB(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(b, n, m, p)
	sys.D.Set(0, 0, 1)
	for i := 1; i < m && i < p; i++ {
		sys.D.Set(i, i, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Inv(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCovar_SISO_N2(b *testing.B)   { benchCovarB(b, 2, 1, 1) }
func BenchmarkCovar_SISO_N10(b *testing.B)  { benchCovarB(b, 10, 1, 1) }
func BenchmarkCovar_SISO_N50(b *testing.B)  { benchCovarB(b, 50, 1, 1) }
func BenchmarkCovar_SISO_N100(b *testing.B) { benchCovarB(b, 100, 1, 1) }
func BenchmarkCovar_MIMO_N10(b *testing.B)  { benchCovarB(b, 10, 3, 4) }

func benchCovarB(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(b, n, m, p)
	W := mat.NewDense(m, m, nil)
	for i := range m {
		W.Set(i, i, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Covar(sys, W); err != nil {
			b.Fatal(err)
		}
	}
}

// --------------- Pidtune ---------------

func BenchmarkPidtune_PI(b *testing.B) {
	sys := benchSysNonSym(b, 4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Pidtune(sys, PidtunePI, 0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPidtune_PID(b *testing.B) {
	sys := benchSysNonSym(b, 4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Pidtune(sys, PidtunePID, 0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPidtune_PIDF(b *testing.B) {
	sys := benchSysNonSym(b, 4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Pidtune(sys, PidtunePIDF, 0, nil); err != nil {
			b.Fatal(err)
		}
	}
}
