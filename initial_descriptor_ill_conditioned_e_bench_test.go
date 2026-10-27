package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// illConditionedEBenchModel is largeX0StiffDescriptor (cond(E) 7.7e7):
// routed at foldAccurateECond, and its σ_min(E) is far above the MATLAB
// rank tolerance. The stepFoldLossCases fixtures sit within 1.5x of that
// tolerance, so their rank decision flips with the BLAS backend.
func illConditionedEBenchModel(tb testing.TB) (*System, *mat.VecDense) {
	return largeX0StiffDescriptor(tb)
}

// The benchmark fixture must take the routed full-rank path on every
// backend: E is rank full with margin, the LU condition estimate reaches
// foldAccurateECond, and Initial, Lsim with x0 and delayed C2D return values.
func TestIllConditionedEBenchModelFullRankOnEveryBackend(t *testing.T) {
	sys, x0 := illConditionedEBenchModel(t)
	n, _ := sys.E.Dims()
	var svd mat.SVD
	if !svd.Factorize(sys.E, mat.SVDNone) {
		t.Fatal("SVD of E failed")
	}
	sv := svd.Values(nil)
	tol := float64(n) * (math.Nextafter(sv[0], math.Inf(1)) - sv[0])
	if margin := sv[n-1] / tol; margin < 1e6 {
		t.Fatalf("σ_min(E) only %.3g x the rank tolerance", margin)
	}
	if _, ok := foldInaccurateParts(sys); !ok {
		t.Fatal("fixture not routed at foldAccurateECond")
	}
	if _, err := Initial(sys, x0, 5); err != nil {
		t.Fatalf("Initial: %v", err)
	}
	tv := makeTimeVector(51, 0.1)
	if _, err := Lsim(sys, mat.NewDense(len(tv), 1, nil), tv, x0); err != nil {
		t.Fatalf("Lsim: %v", err)
	}
	d := sys.Copy()
	d.InputDelay = []float64{0.2}
	if _, err := d.C2D(0.1, C2DOptions{}); err != nil {
		t.Fatalf("C2D: %v", err)
	}
}

func BenchmarkInitialIllConditionedE(b *testing.B) {
	sys, x0 := illConditionedEBenchModel(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Initial(sys, x0, 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLsimInitialStateIllConditionedE(b *testing.B) {
	sys, x0 := illConditionedEBenchModel(b)
	t := makeTimeVector(501, 0.01)
	u := mat.NewDense(len(t), 1, nil)
	for k := range t {
		u.Set(k, 0, 1)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Lsim(sys, u, t, x0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkC2DInputDelayIllConditionedE(b *testing.B) {
	sys, _ := illConditionedEBenchModel(b)
	sys.InputDelay = []float64{0.2}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.C2D(0.1, C2DOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}
