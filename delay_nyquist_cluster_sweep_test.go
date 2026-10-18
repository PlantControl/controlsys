package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"slices"
	"testing"
)

// TestDelayNyquistClusterSweep compares IsStable of Feedback(P,nil,-1) and
// the DiskMargin verdict of P = num/den·e^{-τs} with the argument-principle
// oracle on χ over five families of near-axis pole clusters. Set
// NYQ_CLUSTER_SWEEP to a file to record two lines per case (name, oracle,
// verdict) for comparing revisions.
func TestDelayNyquistClusterSweep(t *testing.T) {
	out := os.Getenv("NYQ_CLUSTER_SWEEP")
	if out == "" {
		t.Skip("set NYQ_CLUSTER_SWEEP=<file> to run the sweep")
	}
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	n := 400
	wrong, errs := 0, 0
	for fam := range 5 {
		rng := rand.New(rand.NewPCG(8, uint64(fam)))
		for i := range n {
			q := clusterSweepCase(rng, fam)
			name := fmt.Sprintf("f%d/%03d", fam, i)
			want, ok := sweepOracle(q)
			if !ok {
				fmt.Fprintf(f, "%s skip\n", name)
				continue
			}
			for _, v := range []string{sweepVerdict(t, q), sweepDiskVerdict(t, q)} {
				fmt.Fprintf(f, "%s %d %s\n", name, want, v)
				switch {
				case v == "E":
					errs++
				case (v == "S") != (want == 0):
					wrong++
				}
			}
		}
	}
	t.Logf("wrong %d, errors %d", wrong, errs)
}

func sweepVerdict(t *testing.T, q quasiLoop) string {
	p := makePlant(t, q.num, q.den)
	if err := p.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(p, nil, -1)
	if err != nil {
		return "E"
	}
	st, err := cl.IsStable()
	switch {
	case err != nil:
		return "E"
	case st:
		return "S"
	}
	return "U"
}

func sweepDiskVerdict(t *testing.T, q quasiLoop) string {
	p := makePlant(t, q.num, q.den)
	if err := p.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	dm, err := DiskMargin(p)
	switch {
	case err != nil:
		return "E"
	case dm.Alpha > 0:
		return "S"
	}
	return "U"
}

func logUniform(rng *rand.Rand, lo, hi float64) float64 {
	return math.Exp(math.Log(lo) + rng.Float64()*(math.Log(hi)-math.Log(lo)))
}

func signed(rng *rand.Rand, v float64) float64 {
	if rng.IntN(2) == 0 {
		return -v
	}
	return v
}

func clusterSweepCase(rng *rand.Rand, fam int) quasiLoop {
	var den []float64
	switch fam {
	case 0: // Jordan pair(s), exact double
		w := logUniform(rng, 0.1, 10)
		pair := []float64{1, 0, w * w}
		den = polyMul(polyMul(pair, pair), []float64{1, logUniform(rng, 0.1, 10)})
	case 1: // near-double pair of either damping
		w := logUniform(rng, 0.1, 10)
		w2 := w * (1 + signed(rng, logUniform(rng, 1e-9, 1e-5)))
		z1, z2 := signed(rng, logUniform(rng, 1e-10, 1e-6)), signed(rng, logUniform(rng, 1e-10, 1e-6))
		den = polyMul(polyMul([]float64{1, 2 * z1 * w, w * w}, []float64{1, 2 * z2 * w2, w2 * w2}), []float64{1, logUniform(rng, 0.1, 10)})
	case 2: // integrators + small real poles
		den = []float64{1}
		for range 1 + rng.IntN(3) {
			den = polyMul(den, []float64{1, 0})
		}
		for range 1 + rng.IntN(2) {
			den = polyMul(den, []float64{1, signed(rng, logUniform(rng, 1e-9, 1e-3))})
		}
		if rng.IntN(2) == 0 {
			den = polyMul(den, []float64{1, logUniform(rng, 0.1, 10)})
		}
	case 3: // triple axis poles
		if rng.IntN(2) == 0 {
			den = polyMul([]float64{1, 0, 0, 0}, []float64{1, logUniform(rng, 0.1, 10)})
		} else {
			w := logUniform(rng, 0.1, 10)
			pair := []float64{1, 0, w * w}
			den = polyMul(polyMul(pair, pair), pair)
		}
	case 4: // distinct near-axis pairs + fast pole
		w1 := logUniform(rng, 0.1, 10)
		w2 := w1 * (1 + logUniform(rng, 1e-4, 1e-2))
		z1, z2 := signed(rng, logUniform(rng, 1e-10, 1e-6)), signed(rng, logUniform(rng, 1e-10, 1e-6))
		den = polyMul(polyMul([]float64{1, 2 * z1 * w1, w1 * w1}, []float64{1, 2 * z2 * w2, w2 * w2}), []float64{1, logUniform(rng, 1e2, 1e4)})
	}
	nd := len(den) - 1
	if rng.IntN(2) == 0 {
		// den + num is a stable target polynomial, so small delays keep
		// many loops stable.
		target := []float64{1}
		for range nd {
			target = polyMul(target, []float64{1, logUniform(rng, 0.05, 5)})
		}
		num := make([]float64, nd)
		for i := range num {
			num[i] = target[i+1] - den[i+1]
		}
		return quasiLoop{num: num, den: den, tau: logUniform(rng, 1e-3, 0.3)}
	}
	num := make([]float64, nd)
	for i := range num {
		num[i] = rng.Float64()
	}
	num = polyScale(num, logUniform(rng, 1e-2, 10))
	return quasiLoop{num: num, den: den, tau: logUniform(rng, 1e-3, 1)}
}

// sweepOracle counts the RHP roots of χ = den + num·e^{-τs} by the argument
// principle on [0, B]×[-B, B], with |num/den| < 1/2 for Re s >= 0, |s| >= B;
// ok is false when the contour passes too close to a root.
func sweepOracle(q quasiLoop) (int, bool) {
	lead := q.den[0]
	bound := func(r float64) float64 {
		nv, dv := 0.0, math.Pow(r, float64(len(q.den)-1))*math.Abs(lead)
		for i, c := range q.num {
			nv += math.Abs(c) * math.Pow(r, float64(len(q.num)-1-i))
		}
		for i, c := range q.den[1:] {
			dv -= math.Abs(c) * math.Pow(r, float64(len(q.den)-2-i))
		}
		return nv / dv
	}
	B := 1.0
	for !(bound(B) > 0 && bound(B) < 0.5) {
		B *= 2
		if B > 1e7 {
			return 0, false
		}
	}
	corners := []complex128{complex(0, -B), complex(B, -B), complex(B, B), complex(0, B), complex(0, -B)}
	total := 0.0
	ok := true
	var walk func(a, b, fa, fb complex128, depth int)
	walk = func(a, b, fa, fb complex128, depth int) {
		d := cmplx.Phase(fb / fa)
		if math.Abs(d) <= 0.2 {
			total += d
			return
		}
		if depth > 60 {
			ok = false
			return
		}
		m := (a + b) / 2
		fm := q.chi(m)
		walk(a, m, fa, fm, depth+1)
		walk(m, b, fm, fb, depth+1)
	}
	const n = 4000
	axis := make([]float64, 0, n+1)
	for i := range n + 1 {
		axis = append(axis, B*(1-2*float64(i)/n))
	}
	roots, err := Poly(q.den).Roots()
	if err != nil {
		return 0, false
	}
	for _, r := range append(roots, 0) {
		w := math.Abs(imag(r))
		for d := 1e-13; d < 0.5; d *= 1.5 {
			for _, v := range []float64{w + d*max(1, w), w - d*max(1, w)} {
				if math.Abs(v) < B {
					axis = append(axis, v, -v)
				}
			}
		}
	}
	slices.Sort(axis)
	slices.Reverse(axis)
	for e := range 4 {
		a, b := corners[e], corners[e+1]
		prev, fprev := a, q.chi(a)
		step := func(z complex128) {
			fz := q.chi(z)
			walk(prev, z, fprev, fz, 0)
			prev, fprev = z, fz
		}
		if e == 3 {
			for _, w := range axis[1:] {
				step(complex(0, w))
			}
			continue
		}
		for i := 1; i <= n && ok; i++ {
			step(a + (b-a)*complex(float64(i)/n, 0))
		}
	}
	turns := total / (2 * math.Pi)
	if !ok || math.Abs(turns-math.Round(turns)) > 1e-6 {
		return 0, false
	}
	return int(math.Round(turns)), true
}
