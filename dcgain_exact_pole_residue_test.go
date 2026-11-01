package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dcGainRationalLimit is the σ → 0 limit of each entry of the float model at
// x0, by exact rational arithmetic independent of the library's solver.
func dcGainRationalLimit(n, m, p int, a, b, c, d []float64, x0 float64) []float64 {
	den, at := fzCharPolyExact(a, n), fzRat(x0)
	out := make([]float64, p*m)
	bj := make([]float64, n)
	for i := range p {
		for j := range m {
			for k := range n {
				bj[k] = b[k*m+j]
			}
			num := fzPencilDetRat(n, 1, a, bj, c[i*n:(i+1)*n], []float64{d[i*m+j]})
			out[i*m+j] = fzLimitAt(num, den, at)
		}
	}
	return out
}

func checkDCGainAgainstRationalLimit(t *testing.T, name string, n, m, p int, a, b, c, d []float64, dt float64) {
	t.Helper()
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	x0 := 0.0
	if dt > 0 {
		x0 = 1
	}
	want := dcGainRationalLimit(n, m, p, a, b, c, d, x0)
	for k, v := range g.RawMatrix().Data {
		w := want[k]
		if math.IsInf(w, 0) || w == 0 {
			if v != w {
				t.Errorf("%s: entry %d = %v, exact limit %v", name, k, v, w)
			}
		} else if !(math.Abs(v-w) <= 1e-9*math.Abs(w)) {
			t.Errorf("%s: entry %d = %.17g, exact limit %.17g", name, k, v, w)
		}
	}
}

// 2TUAD6: a residue below the rounding of its own terms reads as cancelled,
// though the float model's exact limit is Inf: c2 = c1·k/ak rounds, and the
// intended gain, by hand with the residue gone, is want. A residue cancelled
// exactly stays finite too.
func TestDCGainIntegratorResidueCancellation(t *testing.T) {
	a, b, c, d := []float64{-3, 10, 0, 0}, []float64{0.5, 1.5}, []float64{1, 10.0 / -3.0}, []float64{0.25}
	if w := dcGainRationalLimit(2, 1, 1, a, b, c, d, 0)[0]; !math.IsInf(w, 0) {
		t.Fatalf("oracle limit %v, want a rounded residue", w)
	}
	sys, err := NewFromSlices(2, 1, 1, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	want := 0.5/3 - 10*1.5/9 + 0.25
	if !(math.Abs(g.At(0, 0)-want) <= 1e-12) {
		t.Errorf("rounded residue: DCGain = %.17g, want %.17g", g.At(0, 0), want)
	}
	checkDCGainAgainstRationalLimit(t, "exact cancellation", 2, 1, 1, []float64{-4, 8, 0, 0}, []float64{0.5, 1.5}, []float64{1, -2}, []float64{0.25}, 0)
}

// 2TUAD6: an integrator cancelled by a zero at DC through Series leaves a
// sub-rounding residue. DCGain is the intended k/a, not ±Inf, and Bandwidth is
// that of the first-order lag a·√(10^0.3 − 1).
func TestDCGainSeriesCancelledIntegratorFinite(t *testing.T) {
	lagFactor := math.Sqrt(math.Pow(10, 0.3) - 1)
	check := func(t *testing.T, name string, sys *System, gain, bw float64) {
		t.Helper()
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !(math.Abs(g.At(0, 0)-gain) <= 1e-9*math.Abs(gain)) {
			t.Errorf("%s: DCGain = %.17g, want %.17g", name, g.At(0, 0), gain)
		}
		b, err := Bandwidth(sys, -3)
		if err != nil {
			t.Fatalf("%s: Bandwidth: %v", name, err)
		}
		if !(math.Abs(b-bw) <= 1e-3*bw) {
			t.Errorf("%s: Bandwidth = %.17g, want %.17g", name, b, bw)
		}
	}
	for _, tc := range []struct {
		name       string
		a, b, c    []float64
		gain, band float64
	}{
		{"integrator then zero", []float64{-0.026882469217874124, 0, 1, -0}, []float64{-0.0091835259382100666, 0.34161764917427861}, []float64{0, 1}, 12.707822574093656, 0.026818713278324691},
		{"zero then integrator", []float64{-0, 0, -0.0017626337936102773, -0.10190534737851512}, []float64{1, 0}, []float64{0.017296774300402381, 1}, 0.16973372590699878, 0.10166366306318779},
	} {
		sys, err := NewFromSlices(2, 1, 1, tc.a, tc.b, tc.c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		check(t, tc.name, sys, tc.gain, tc.band)
	}

	tfSys := func(t *testing.T, num, den []float64) *System {
		t.Helper()
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	for _, p := range []struct{ a, k float64 }{
		{0.026882469217874124, 0.34161764917427861},
		{0.10190534737851512, 0.017296774300402381},
		{1.0 / 3, 7},
		{0.7, 0.1},
		{37.1, 2.9},
	} {
		zero := tfSys(t, []float64{p.k, 0}, []float64{1, p.a})
		integ := tfSys(t, []float64{1}, []float64{1, 0})
		for _, order := range []struct {
			name   string
			g1, g2 *System
		}{{"zero then integrator", zero, integ}, {"integrator then zero", integ, zero}} {
			sys, err := Series(order.g1, order.g2)
			if err != nil {
				t.Fatal(err)
			}
			check(t, order.name, sys, p.k/p.a, p.a*lagFactor)
		}
	}
}

// 2TUAD6: σmin(x0·I − A) against τ = n·ε·σ1 (descriptorRank's tolerance). The
// float model's exact limit is finite on both sides, but inside τ it is not the
// truth standard: the rounded-off integrator stays +Inf as FreqResponse(0) has
// it, so Bandwidth keeps its error. Outside τ the gain is the exact limit.
func TestDCGainOffPoleResolutionAroundRankTolerance(t *testing.T) {
	shiftedSigmas := func(n int, a []float64, x0 float64) (sigmaMin, tau float64) {
		ms := make([]float64, n*n)
		for i := range ms {
			ms[i] = -a[i]
			if i%(n+1) == 0 {
				ms[i] += x0
			}
		}
		var svd mat.SVD
		if !svd.Factorize(mat.NewDense(n, n, ms), mat.SVDNone) {
			t.Fatal("SVD failed")
		}
		sv := svd.Values(nil)
		return sv[n-1], float64(n) * 0x1p-52 * sv[0]
	}

	tf := &TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, 10, 21, 0}}}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	d, err := res.Sys.C2D(0.1, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := d.Dims()
	raw := d.A.RawMatrix().Data
	if s, tau := shiftedSigmas(n, raw, 1); !(s <= tau) {
		t.Fatalf("C2D ZOH: σmin %.17g outside τ %.17g", s, tau)
	}
	if w := dcGainRationalLimit(n, m, p, raw, d.B.RawMatrix().Data, d.C.RawMatrix().Data, d.D.RawMatrix().Data, 1); math.IsInf(w[0], 0) {
		t.Fatalf("C2D ZOH: exact limit %v, want a rounded-off pole", w[0])
	}
	g, err := d.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Errorf("C2D ZOH inside τ: DCGain = %.17g, want +Inf", g.At(0, 0))
	}
	if bw, err := Bandwidth(d, -3); err == nil {
		t.Errorf("C2D ZOH inside τ: Bandwidth = %v, want an error", bw)
	}

	for _, tc := range []struct {
		name string
		a    []float64
		dt   float64
	}{
		{"continuous", []float64{1, 2, 3, 6 - 1e-9}, 0},
		{"discrete", []float64{0, -2, -3, -5 + 1e-9}, 0.1},
	} {
		x0 := 0.0
		if tc.dt > 0 {
			x0 = 1
		}
		if s, tau := shiftedSigmas(2, tc.a, x0); !(s > tau) {
			t.Fatalf("%s: σmin %.17g inside τ %.17g", tc.name, s, tau)
		}
		b, c, dd := []float64{1, -1}, []float64{2, 1}, []float64{0.5}
		sys, err := NewFromSlices(2, 1, 1, tc.a, b, c, dd, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		w := dcGainRationalLimit(2, 1, 1, tc.a, b, c, dd, x0)[0]
		if !(math.Abs(g.At(0, 0)-w) <= 1e-4*math.Abs(w)) {
			t.Errorf("%s outside τ: DCGain = %.17g, exact limit %.17g", tc.name, g.At(0, 0), w)
		}
	}
}

// 2TUAD6: a dense companion block with a pole within rounding of DC stays
// with the transfer limit's integrator reading: its intended pole is not
// resolved by the coefficients.
func TestDCGainDenseBlockRoundedPoleKeepsIntegrator(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{-8.70643320022765e-16, 2.9186581655941266e-16, 1.3545850066477854e-16}}},
		Den: [][]float64{{1, -5.609886783435264, 13.099973168748164, -16.29841103348229, 11.394314088807377, -4.243853507030152, 0.6578640663921657}},
		Dt:  0.01,
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	g, err := res.Sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), -1) {
		t.Errorf("DCGain = %v, want -Inf", g.At(0, 0))
	}
}

// 2TUAD6: integrators cancelled by zeros at DC through Series, in both orders,
// keep the intended k/(a·b); the second output channel of a SIMO integrator
// with no state coupling stays 0 beside the +Inf of the first.
func TestDCGainSeriesCancelledDoubleIntegratorFinite(t *testing.T) {
	for _, tc := range []struct {
		name    string
		a, b, c []float64
		want    float64
	}{
		{"integrators first", []float64{0, -2.0923021674681676e-06, 0, 0, 1, -0.10913131676372737, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0}, []float64{-1.3703096518060768e-05, -0.71473279051562744, 6.5492913648521789, 0}, []float64{0, 0, 0, 1}, 3130184.2853689152},
		{"stiff, integrators first", []float64{0, -0.00047207290661579793, 0, 0, 1, -17.026263864335586, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0}, []float64{-2.0836938504235868e-05, -0.75152631749483545, 0.044139238266419419, 0}, []float64{0, 0, 0, 1}, 93.500892865988291},
		{"zeros first", []float64{0, -0, 0, 0, 1, -0, 0, 0, 0, -1.3703096518060768e-05, 0, -2.0923021674681676e-06, 0, -0.71473279051562744, 1, -0.10913131676372737}, []float64{1, 0, 0, 0}, []float64{0, 6.5492913648521789, 0, 1}, 3130184.2853689152},
		{"stiff, zeros first", []float64{0, -0, 0, 0, 1, -0, 0, 0, 0, -2.0836938504235868e-05, 0, -0.00047207290661579793, 0, -0.75152631749483545, 1, -17.026263864335586}, []float64{1, 0, 0, 0}, []float64{0, 0.044139238266419419, 0, 1}, 93.500892865988291},
	} {
		sys, err := NewFromSlices(4, 1, 1, tc.a, tc.b, tc.c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !(math.Abs(g.At(0, 0)-tc.want) <= 1e-9*tc.want) {
			t.Errorf("%s: DCGain = %.17g, want %.17g", tc.name, g.At(0, 0), tc.want)
		}
	}
}

func TestDCGainSIMOIntegratorBesideZeroChannel(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 2, []float64{0}, []float64{1}, []float64{1, 0}, []float64{0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) || g.At(1, 0) != 0 {
		t.Errorf("DCGain = [%v %v], want [+Inf 0]", g.At(0, 0), g.At(1, 0))
	}
}
