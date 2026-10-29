package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zerosWorkModel draws a model for the zeros workspace tests: SISO (D = 0
// in two of four draws) and non-square MIMO, sparse entries, block scales
// over six decades, zero, generic and rank-one D.
func zerosWorkModel(rng *rand.Rand) *System {
	n := 1 + rng.IntN(10)
	m, p := 1+rng.IntN(4), 1+rng.IntN(4)
	if rng.IntN(3) == 0 {
		m, p = 1, 1
	}
	dens := []float64{1, 0.6, 0.3}[rng.IntN(3)]
	sc := func() float64 { return math.Pow(10, float64(rng.IntN(7)-3)) }
	fill := func(r, c int, s float64) *mat.Dense {
		d := mat.NewDense(r, c, nil)
		for i := range r {
			for j := range c {
				if rng.Float64() < dens {
					d.Set(i, j, s*rng.NormFloat64())
				}
			}
		}
		return d
	}
	var A, B, C *mat.Dense
	if n > 0 {
		A, B, C = fill(n, n, sc()), fill(n, m, sc()), fill(p, n, sc())
	}
	D := mat.NewDense(p, m, nil)
	switch rng.IntN(4) {
	case 2:
		D = fill(p, m, sc())
	case 3:
		D.Mul(fill(p, 1, 1), fill(1, m, 1))
	}
	sys, err := New(A, B, C, D, []float64{0, 0.1}[rng.IntN(2)])
	if err != nil {
		return nil
	}
	return sys
}

func sameZeros(a, b *ZerosResult) bool {
	if a.Rank != b.Rank || len(a.Zeros) != len(b.Zeros) {
		return false
	}
	for i, z := range a.Zeros {
		w := b.Zeros[i]
		if math.Float64bits(real(z)) != math.Float64bits(real(w)) || math.Float64bits(imag(z)) != math.Float64bits(imag(w)) {
			return false
		}
	}
	return true
}

// A workspace reused across models of changing size (XSOA7K) must give the
// zeros of a fresh one bit for bit, including after larger models ran in
// it, on every route: eig(A−BD⁻¹C), the staircase with the raw reduction
// competing, and the SISO cap.
func TestMimoZerosWorkReusedMatchesFresh(t *testing.T) {
	rng := rand.New(rand.NewPCG(53, 7))
	var w zerosWork
	var models []*System
	var want []*ZerosResult
	arbitrated, capped := 0, 0
	for len(models) < 3000 {
		sys := zerosWorkModel(rng)
		if sys == nil {
			continue
		}
		ref, err := mimoZeros(sys)
		if err != nil {
			continue
		}
		models = append(models, sys)
		want = append(want, ref)
		n, m, p := sys.Dims()
		if n > 0 && reduceStaircase(zerosPrescale(sys)).margin < rankMarginMin {
			arbitrated++
		}
		if m == 1 && p == 1 && sys.D.At(0, 0) == 0 && len(ref.Zeros) > 0 {
			capped++
		}
		got, err := mimoZerosWork(sys, &w)
		if err != nil || !sameZeros(got, ref) {
			t.Fatalf("model %d: reused workspace %v, %v; fresh %v", len(models)-1, got, err, ref)
		}
	}
	if arbitrated == 0 || capped == 0 {
		t.Fatalf("sweep misses a route: %d arbitrated, %d SISO D = 0 with zeros", arbitrated, capped)
	}
	for k := len(models) - 1; k >= 0; k -= 7 {
		got, err := mimoZerosWork(models[k], &w)
		if err != nil || !sameZeros(got, want[k]) {
			t.Fatalf("model %d again: reused workspace %v, %v; fresh %v", k, got, err, want[k])
		}
	}
}

// sortZeros must order ties in modulus (conjugate pairs, equal real and
// imaginary magnitudes) exactly as sort.Slice does.
func TestSortZerosMatchesSortSlice(t *testing.T) {
	rng := rand.New(rand.NewPCG(53, 8))
	for range 20000 {
		n := rng.IntN(20)
		z := make([]complex128, 0, n)
		for len(z) < n {
			v := complex(float64(rng.IntN(5)-2), float64(rng.IntN(5)-2))
			if rng.IntN(2) == 0 {
				v = complex(rng.NormFloat64(), rng.NormFloat64())
			}
			z = append(z, v)
			if imag(v) != 0 && len(z) < n && rng.IntN(2) == 0 {
				z = append(z, cmplx.Conj(v))
			}
		}
		rng.Shuffle(len(z), func(i, j int) { z[i], z[j] = z[j], z[i] })
		want := slices.Clone(z)
		sort.Slice(want, func(i, j int) bool { return cmplx.Abs(want[i]) < cmplx.Abs(want[j]) })
		sortZeros(z)
		for i := range z {
			if math.Float64bits(real(z[i])) != math.Float64bits(real(want[i])) || math.Float64bits(imag(z[i])) != math.Float64bits(imag(want[i])) {
				t.Fatalf("sortZeros %v, sort.Slice %v", z, want)
			}
		}
	}
}

func zerosWorkSISOChannel(tb testing.TB) *System {
	tb.Helper()
	sys := benchSys(tb, 8, 1, 1)
	b := sys.B.RawMatrix()
	for i := range 8 {
		b.Data[i*b.Stride] = 1 / float64(i+1)
	}
	return sys
}

// A warm workspace cuts the allocations of the SISO staircase route by at
// least a third; the rest are the result, matrix headers and the pivoted
// QR/RQ factors.
func TestMimoZerosWorkSISOAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("allocation counts differ under -race")
	}
	sys := zerosWorkSISOChannel(t)
	var w zerosWork
	if _, err := mimoZerosWork(sys, &w); err != nil {
		t.Fatal(err)
	}
	fresh := testing.AllocsPerRun(50, func() { _, _ = mimoZeros(sys) })
	warm := testing.AllocsPerRun(50, func() { _, _ = mimoZerosWork(sys, &w) })
	if !(3*warm <= 2*fresh) {
		t.Fatalf("warm workspace %v allocs, fresh %v", warm, fresh)
	}
}

func BenchmarkMimoZerosWorkSISO(b *testing.B) {
	sys := zerosWorkSISOChannel(b)
	var w zerosWork
	b.ReportAllocs()
	for b.Loop() {
		if _, err := mimoZerosWork(sys, &w); err != nil {
			b.Fatal(err)
		}
	}
}
