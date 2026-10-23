package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// companionShape returns (A, b, c) of the SISO observable companion
// (observableCompanion(den), b, c) in shape t of companionRowTF: 0 as is,
// 1 reversed (superdiagonal, first column), 2 transposed (superdiagonal,
// last row), 3 both (subdiagonal, first row, as MATLAB tf2ss).
func companionShape(den, b, c []float64, t int) (a, bt, ct []float64) {
	n := len(den) - 1
	ao := observableCompanion(den)
	a, bt, ct = make([]float64, n*n), make([]float64, n), make([]float64, n)
	rev := func(k int) int { return n - 1 - k }
	for i := range n {
		for j := range n {
			switch t {
			case 0:
				a[i*n+j] = ao[i*n+j]
			case 1:
				a[i*n+j] = ao[rev(i)*n+rev(j)]
			case 2:
				a[i*n+j] = ao[j*n+i]
			default:
				a[i*n+j] = ao[rev(j)*n+rev(i)]
			}
		}
		switch t {
		case 0:
			bt[i], ct[i] = b[i], c[i]
		case 1:
			bt[i], ct[i] = b[rev(i)], c[rev(i)]
		case 2:
			bt[i], ct[i] = c[i], b[i]
		default:
			bt[i], ct[i] = c[rev(i)], b[rev(i)]
		}
	}
	return a, bt, ct
}

func convertSISOCompanion(t *testing.T, a, b, c []float64, n int, noDirect bool) *TransferFunc {
	t.Helper()
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	cv := newRowRealizationConverter(sys, &TransferFuncOpts{})
	cv.noDirect = noDirect
	res, err := cv.convert()
	if err != nil {
		t.Fatal(err)
	}
	return res.TF
}

// N2JIRH: a slow observable companion with dense B and C has no unit row
// or input for directRowTF, so its rows went through the staircase, whose
// rotations lost the low den coefficients (Osborne: den[5] 7.9e-17 for an
// exact 9.4e-20, response error 1) and the check points above the poles
// could not tell. The den is A's last column, so it must come back exact,
// in each of the four companion shapes.
func TestTransferFunctionCompanionDenseBC(t *testing.T) {
	den6 := []float64{1, 0.00075531005859375, 2.2356107365339994e-07, 3.2775782088378946e-11, 2.48872997100772e-15, 9.360271606279925e-20, 1.3795732784857206e-24}
	b6 := []float64{0, 480, -6.4373016357421875e-06, 0, 0, 0.0126953125}
	den7 := []float64{1, -9.182366262945676e-05, -2.0336609409945156e-08, -3.3328008306331357e-12, -2.0074265317820019e-16, 5.3864724729292865e-21, -9.426549067309936e-25, -5.165772441499083e-29}
	cases := []struct {
		name string
		cs   mimoCompanionCase
	}{
		{"c1", newObservableMIMO(den6, b6, []float64{-0.5, 0.75, -0.75, 1, -2, 0.25}, 1, 1)},
		{"c2", newObservableMIMO(den6, b6, []float64{2, -0.25, -1, -1, 1.25, -2}, 1, 1)},
		{"mimo", newObservableMIMO(den6, b6, []float64{0, 0, 0, 0, 0, 1, -0.5, 0.75, -0.75, 1, -2, 0.25, 2, -0.25, -1, -1, 1.25, -2}, 1, 3)},
		{"n7", newObservableMIMO(den7,
			[]float64{0, 4.5284158111454793e-13, 2.322615756567058e-09, 4.323514018864433e-06, 0.0034318854034692584, 0.963951699238913, 0},
			[]float64{0.8016887941906009, 0.8379864884716763, 0.1594996044769284, -0.5670130929083917, 1.4606774350628802, -0.5313814629958058, -0.3927162478543107}, 1, 1)},
		// The staircase num of entry (1, 1) led with 5546.80 for an exact 5543.25.
		{"473", newObservableMIMO([]float64{1, 0.01373291015625, 8.013099431991577e-05, 2.5520193958072923e-07, 4.621997923281995e-10, 4.52900172775311e-13, 2.392364643532841e-16, 8.69746111271906e-20},
			[]float64{4.492176230996847e-08, -6656, 0.0025234222412109375, 544, 47.25, 0, 294912, 13.5, 0, -0.5625, 0, -0.0078125, 0, -0.0107421875},
			[]float64{0, 0, 0, 0, 0, 0, 1, -1, -2, -2, -1.75, 2, 0.25, 0}, 2, 2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := tc.cs
			ex := newMIMOExact(cs)
			check := func(tf *TransferFunc, label string) {
				t.Helper()
				for i := range tf.Den {
					got := tf.Den[i]
					if len(got) != cs.n+1 {
						t.Fatalf("%s row %d den %v, want %v", label, i, got, cs.den)
					}
					for k, want := range cs.den {
						if math.Abs(got[k]-want) > 1e-12*math.Abs(want) {
							t.Errorf("%s row %d den[%d] = %.17g, want %.17g", label, i, k, got[k], want)
						}
					}
				}
				if e := ex.respErr(tf, cs.m); e > 1e-12 {
					t.Errorf("%s response err %.3g", label, e)
				}
			}
			sys, err := NewFromSlices(cs.n, cs.m, cs.p, cs.a, cs.b, cs.c, make([]float64, cs.p*cs.m), 0)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			check(res.TF, "observable")
			if cs.p != 1 {
				return
			}
			for shape := 1; shape < 4; shape++ {
				a, b, c := companionShape(cs.den, cs.b, cs.c, shape)
				check(convertSISOCompanion(t, a, b, c, cs.n, false), []string{"", "reversed", "transposed", "tf2ss"}[shape])
			}
		})
	}
}

// TestTransferFunctionCompanionDenseBCSweep converts seeded SISO companions
// of graded poles in the four shapes of companionRowTF, with dense b and c,
// against the exact big.Rat response: the given-basis rows must never be
// worse than the staircase ones, nor flip the right-half-plane root count,
// and a row keeping all states must be exact.
func TestTransferFunctionCompanionDenseBCSweep(t *testing.T) {
	skipSlowUnderRace(t)
	r := rand.New(rand.NewPCG(2210, 4711))
	const trials = 800
	worst := 0.0
	for trial := range trials {
		n := 2 + r.IntN(9)
		den := ratPolyFromRoots(hessenbergSweepRoots(r, n, math.Ldexp(1, -20+r.IntN(28))))
		b, c := make([]float64, n), make([]float64, n)
		for k := range n {
			if r.IntN(4) > 0 {
				b[k] = math.Ldexp(float64(r.IntN(129)-64), r.IntN(41)-25)
			}
			c[k] = float64(r.IntN(17)-8) / 4
		}
		cs := newObservableMIMO(den, b, c, 1, 1)
		ex := newMIMOExact(cs)
		shape := r.IntN(4)
		a, bt, ct := companionShape(den, b, c, shape)
		tfNew := convertSISOCompanion(t, a, bt, ct, n, false)
		tfBase := convertSISOCompanion(t, a, bt, ct, n, true)
		eNew, eBase := ex.respErr(tfNew, 1), ex.respErr(tfBase, 1)
		if eNew > 2*eBase && eNew > 1e-12 {
			t.Errorf("trial %d shape %d: err %.3g, without the path %.3g; den %v b %v c %v", trial, shape, eNew, eBase, den, b, c)
		}
		got := tfNew.Den[0]
		if len(got) != n+1 {
			continue
		}
		worst = max(worst, eNew)
		if eNew > 1e-12 {
			t.Errorf("trial %d shape %d: full-order err %.3g; den %v b %v c %v", trial, shape, eNew, den, b, c)
		}
		for k, want := range den {
			if math.Abs(got[k]-want) > 1e-12*math.Abs(want) {
				t.Errorf("trial %d shape %d: den[%d] = %.17g, want %.17g", trial, shape, k, got[k], want)
			}
		}
		if rhp := rhpRoots(den); rhpRoots(got) != rhp && rhpRoots(tfBase.Den[0]) == rhp {
			t.Errorf("trial %d shape %d: den %v flips the RHP root count of %v", trial, shape, got, den)
		}
	}
	t.Logf("worst full-order response err %.3g", worst)
}

func BenchmarkTransferFunctionCompanionDense(b *testing.B) {
	den := []float64{1, 0.00075531005859375, 2.2356107365339994e-07, 3.2775782088378946e-11, 2.48872997100772e-15, 9.360271606279925e-20, 1.3795732784857206e-24}
	cs := newObservableMIMO(den, []float64{0, 480, -6.4373016357421875e-06, 0, 0, 0.0126953125}, []float64{-0.5, 0.75, -0.75, 1, -2, 0.25}, 1, 1)
	sys, err := NewFromSlices(cs.n, 1, 1, cs.a, cs.b, cs.c, []float64{0}, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.TransferFunction(nil); err != nil {
			b.Fatal(err)
		}
	}
}
