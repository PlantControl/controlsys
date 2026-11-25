package controlsys

import "testing"

func benchmarkC2DMatchedFixture(b *testing.B, name string, dt float64) {
	f, ok := matchedDCFixtures[name]
	if !ok {
		b.Fatalf("no fixture %q", name)
	}
	sys, err := f.system()
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if d, _ := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched}); d == nil {
			b.Fatal("no model")
		}
	}
}

func BenchmarkC2DMatchedStiffDescriptor(b *testing.B) {
	benchmarkC2DMatchedFixture(b, "stiff root product 1", 0.01)
}

func BenchmarkC2DMatchedSmallSampleTime(b *testing.B) {
	benchmarkC2DMatchedFixture(b, "small sample time 1", 1e-4)
}

func BenchmarkC2DMatchedPencilRoots(b *testing.B) {
	benchmarkC2DMatchedFixture(b, "pencil root product", 0.01)
}

func BenchmarkC2DMatchedDCPoleNearOne(b *testing.B) {
	benchmarkC2DMatchedFixture(b, "missed dc pole 2", 0.01)
}
