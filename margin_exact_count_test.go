package controlsys

import (
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCrossingCounterFirstOrderLoop(t *testing.T) {
	cases := []struct {
		name    string
		a, c, d float64
		dt      float64
		want    []float64
	}{
		{"2/(s+1)", -1, 2, 0, 0, []float64{math.Sqrt(3)}},
		{"discrete 0.5/(z−0.8)", 0.8, 0.5, 0, 0.1, []float64{math.Acos((1+0.64-0.25)/1.6) / 0.1}},
	}
	for _, c := range cases {
		cc, ok := newCrossingCounter(1, []float64{c.a}, []float64{1}, []float64{c.c}, c.d, c.dt, 1)
		if !ok {
			t.Fatalf("%s: no counter", c.name)
		}
		top := math.Inf(1)
		if c.dt > 0 {
			top = math.Pi / c.dt
		}
		ws, ok := cc.crossings(top)
		if !ok || len(ws) != len(c.want) {
			t.Fatalf("%s: crossings = %v, %v; want %v", c.name, ws, ok, c.want)
		}
		for i, w := range c.want {
			if math.Abs(ws[i]-w) > 1e-14*w {
				t.Errorf("%s: crossing %.17g, want %.17g", c.name, ws[i], w)
			}
		}
	}
}

// Loops whose gain pencil carries a double pole at DC (z = 1) in a
// cluster whose rounding swallows a pair of low crossings: the pencil
// reports neither, the end signs below its crossings agree (or the scan
// from above stops at the first), and main returned the rest with nil
// (IYBP6K, TE2TLR; A2OX65 census). The exact crossing count of the stored
// model finds the pair: the 320-bit oracle crossings with nil. series2 29.0
// has a pair 7.8e-7 apart; F9extremeScale 2129 an infinite pencil bound
// where main claimed the whole axis empty.
func TestAllMarginExactCountFindsLowCrossingPairs(t *testing.T) {
	cases := []struct {
		name    string
		n       int
		a, b, c []float64
		dt      float64
		want    []float64
	}{
		{"series2 230.1", 4,
			[]float64{0, -0, 0, 0, 1, -0, 0, 0, 0, -8.9315115819252515, 0, -2.8703484659694563, 0, -8500.7437684794077, 1, -2731.9112349171542},
			[]float64{1, 0, 0, 0}, []float64{0, 3.1116471354667548, 0, 1}, 0,
			[]float64{1.1717018938522901e-08, 5.8339501913171108e-08, 0.00043977859471744829}},
		{"series2 62.0", 4,
			[]float64{0, -0.66151500186250389, 0, 0, 1, -6.1134257107392438, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0},
			[]float64{-0.50422817905416373, -4.6598512584445775, 0.76223241745765846, 0}, []float64{0, 0, 0, 1}, 0,
			[]float64{6.205538015604519e-09, 2.3331552002427507e-08, 0.06306472219332121}},
		{"series2 29.0", 4,
			[]float64{0, -3.952394265696193e-05, 0, 0, 1, -0.35126221218700088, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0},
			[]float64{-0.0018255518806746631, -16.224276956210232, 46.188506458454292, 0}, []float64{0, 0, 0, 1}, 0,
			[]float64{3.8474389599175687e-11, 3.8474419518485586e-11, 6.7916783315443778}},
		{"x-d2-pcg 100.0", 4,
			[]float64{0, -0.13968356033341023, 0, 0, 1, 0.81235212980056903, 0, 0, 0, 1, 0, -1, 0, 0, 1, 2},
			[]float64{2.1012781450149327, -2.9007681343282727, 2.442447973944645, 0}, []float64{0, 0, 0, 1}, 0.1,
			[]float64{9.33220555494485e-08, 1.0679224980296815e-07}},
		{"dpar a=0.9,b=0.1,k=5", 3,
			[]float64{1, 0, 0, 0, 0, -0.90000000000000002, 0, 1, 1.8999999999999999},
			[]float64{5, -0.49999999999999989, 0}, []float64{1, 0, 1}, 0.1,
			[]float64{1.0432899624317958e-07, 1.064368656703404e-07}},
		{"F9extremeScale 2129", 2,
			[]float64{-1.0000000000000003e+227, 71352925505794.219, 0, -9.9999999999999974e-167},
			[]float64{0.41610901187470573, 0.74918488119353732}, []float64{-1.2337032365139637, -0.034149176799357472}, 0,
			[]float64{0.03013537376520486}},
	}
	for _, c := range cases {
		d := 0.0
		if c.name == "F9extremeScale 2129" {
			d = 0.52844004919507892
		}
		sys, err := NewFromSlices(c.n, 1, 1, c.a, c.b, c.c, []float64{d}, c.dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(sys)
		if err != nil {
			t.Fatalf("%s: AllMargin: %v", c.name, err)
		}
		got := res.GainCrossFreqs
		if len(got) != len(c.want) {
			t.Fatalf("%s: GainCrossFreqs = %.17g, want %.17g", c.name, got, c.want)
		}
		for i, w := range c.want {
			if math.Abs(got[i]-w) > 1e-8*w {
				t.Errorf("%s: GainCrossFreqs[%d] = %.17g, want %.17g", c.name, i, got[i], w)
			}
		}
	}
}

// A DC gain certified exactly at the level is no crossing by itself: dpar
// a=0.5,b=0.1,k=0.5 has G(1) = 1, and the stored model's |L| stays below
// 1 beside DC (exact count; 640-bit evaluation down to ω = 2e-39, where a
// 320-bit evaluation flips sign on rounding of the double pole at z = 1).
func TestAllMarginAtLevelDCNoCrossing(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1, []float64{1, 0, 0, 0, 0, -0.5, 0, 1, 1.5}, []float64{0.5, -0.25, 0}, []float64{1, 0, 1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	res, err := AllMargin(sys)
	if err != nil {
		t.Fatalf("AllMargin: %v", err)
	}
	if len(res.GainCrossFreqs) != 0 {
		t.Fatalf("GainCrossFreqs = %.17g, want none", res.GainCrossFreqs)
	}
}

// States coupled into the loop's core but unreachable from the input
// (b = 0, no edge from the core into them) leave the transfer function
// exactly the core's: structuralCore drops them bit for bit, and AllMargin's
// crossings agree with the core's.
func TestStructuralCoreDropsUnreachableCoupledStates(t *testing.T) {
	core := []float64{0, -0.66151500186250389, 0, 0, 1, -6.1134257107392438, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0}
	cb, cc := []float64{-0.50422817905416373, -4.6598512584445775, 0.76223241745765846, 0}, []float64{0, 0, 0, 1}
	const n = 7
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := range 4 {
		copy(a[i*n:i*n+4], core[i*4:i*4+4])
		b[i], c[i] = cb[i], cc[i]
	}
	for k := 4; k < n; k++ {
		a[k*n+k] = -0.3 - 0.2*float64(k)
		a[1*n+k] = 0.7
		if k > 4 {
			a[k*n+k-1] = 0.4
		}
		c[k] = 1.5
	}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := structuralCore(sys)
	want, err := NewFromSlices(4, 1, 1, core, cb, cc, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !mat.Equal(got.A, want.A) || !mat.Equal(got.B, want.B) || !mat.Equal(got.C, want.C) || !mat.Equal(got.D, want.D) {
		t.Fatalf("structuralCore = %v %v %v; want the order-4 core", got.A, got.B, got.C)
	}
	rp, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	rc, err := AllMargin(want)
	if err != nil {
		t.Fatal(err)
	}
	if len(rp.GainCrossFreqs) != len(rc.GainCrossFreqs) {
		t.Fatalf("padded %v, core %v", rp.GainCrossFreqs, rc.GainCrossFreqs)
	}
	for i, w := range rc.GainCrossFreqs {
		if math.Abs(rp.GainCrossFreqs[i]-w) > 1e-9*w {
			t.Errorf("crossing %d: padded %.17g, core %.17g", i, rp.GainCrossFreqs[i], w)
		}
	}
}

func TestSquarefreeModPrime(t *testing.T) {
	ip := func(cs ...int64) ipoly {
		p := make(ipoly, len(cs))
		for i, c := range cs {
			p[i] = big.NewInt(c)
		}
		return p
	}
	if !squarefreeModPrime(ip(-2, 1, 1)) {
		t.Error("(x−1)(x+2) not proved squarefree")
	}
	if squarefreeModPrime(ip(2, -3, 0, 1)) {
		t.Error("(x−1)²(x+2) proved squarefree")
	}
}
