package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type dcGainLiteral struct {
	name       string
	n, m, p    int
	a, b, c, d []float64
	dt         float64
}

func (r dcGainLiteral) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(r.n, r.m, r.p, r.a, r.b, r.c, r.d, r.dt)
	if err != nil {
		t.Fatalf("%s: %v", r.name, err)
	}
	return sys
}

func (r dcGainLiteral) x0() float64 {
	if r.dt > 0 {
		return 1
	}
	return 0
}

// N4GQHT: where a route reads a pole at DC from rounded data and no
// coefficient fixes it, DCGain's nil error still means the float model's
// exact limit, finite entries included. Rows frozen from the INGGF2 census
// (arm64), each nil and off the exact limit on base: a discrete block with
// a pole 1.4e6 from DC beside an ill-conditioned pair (F5 #1150, MIMO,
// base off by 10x), and a Series row whose exact limit is finite at the
// pole of the second factor rounded off z = 1 (x-dplant-pcg #0.0: the
// intended 0.0924 assumes the pole and zero at z = 1 cancel, the float
// model's limit is −0.0315).
func TestDCGainNilIsExactLimitOffPole(t *testing.T) {
	for _, r := range []dcGainLiteral{
		{"ill-conditioned block beside stiff pole", 3, 2, 2, []float64{0.12493445278191539, -0.47085209210468409, 0.21980275346936351, 0.098685978921089462, 1.0531005931882367, -0.25142711167051041, 0, 0, -1408147.6547531674}, []float64{-0.28785107210982896, -0.046146257240553018, 0.26922911530114912, -1.5711754038701913, 0.68309735721723275, -0.29462280792704121}, []float64{0.10369895868125401, 1.5060534999125765, 0.34459156407271885, -0.42796522491003353, 1.1495849426158391, -0.23137049619064587}, []float64{-0.94816595301829376, -0.028887162890477214, -1.5090577335728821, -0.69967077187488846}, 0.1},
		{"plant pole rounded off z = 1", 3, 1, 1, []float64{0.15989017144703166, 0, 0, 1, 0, -0.40776256620277523, 0, 1, 1.4077625662027753}, []float64{-0.038627919893980144, 0.045979607166974933, 0}, []float64{0, 0, 1}, []float64{0}, 0.1},
	} {
		checkDCGainAgainstRationalLimit(t, r.name, r.n, r.m, r.p, r.a, r.b, r.c, r.d, r.dt)
	}
}

// N4GQHT (ruling C): x-dplant-pcg #0.0, Series(k(z−1)/(z−a), 1/((z−1)(z−b)))
// with the second factor's pole rounded off z = 1, is a nonsingular pencil
// whose exact DC gain is −0.0315, returned with a nil error: an intentional
// change from the intended 0.0924, which assumes the pole and zero at z = 1
// cancel.
func TestDCGainSeriesPoleRoundedOffDCExactValue(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1, []float64{0.15989017144703166, 0, 0, 1, 0, -0.40776256620277523, 0, 1, 1.4077625662027753}, []float64{-0.038627919893980144, 0.045979607166974933, 0}, []float64{0, 0, 1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("DCGain: %v", err)
	}
	if want := -0.031452715889969438; !(math.Abs(g.At(0, 0)-want) <= 1e-12*math.Abs(want)) {
		t.Errorf("DCGain = %.17g, want %.17g", g.At(0, 0), want)
	}
}

// N4GQHT: a pole at DC whose residue is below rounding of its own terms
// returns ErrAccuracyUnverified with the finite regular part where every
// float reading is finite: Series(k/(s+a)-type factors, integrators) rows
// frozen from the INGGF2 census (arm64). series3 #108.1 and dover-pcg
// #180.0 were nil on base and matched neither the exact limit (±Inf) nor the
// intended value; the regular part is the intended k/(a·b·c), and the
// over-cancelled discrete Series' intended 0 to rounding.
func TestDCGainSubRoundingResidueUnverifiedRegularPart(t *testing.T) {
	for _, tc := range []struct {
		dcGainLiteral
		want, tol float64
	}{
		{dcGainLiteral{"integrator then zero", 2, 1, 1, []float64{-0.079322793258357296, 0, 1, -0}, []float64{-4.0249182639151666, 50.741005183791977}, []float64{0, 1}, []float64{0}, 0}, 639.67748864474083, 1e-12},
		{dcGainLiteral{"triple integrator cancelled", 6, 1, 1, []float64{0, 0, -0, 0, 0, 0, 1, 0, -0, 0, 0, 0, 0, 1, -0, 0, 0, 0, 0, 0, -182.07146009139097, 0, 0, -9.0859752664405438, 0, 0, -128087121.15572563, 1, 0, -6391976.0636089118, 0, 0, -131857.01941173122, 0, 1, -6580.1066047374925}, []float64{1, 0, 0, 0, 0, 0}, []float64{0, 0, 20.038736046737885, 0, 0, 1}, []float64{0}, 0}, 2.2054579127847567, 1e-12},
		{dcGainLiteral{"discrete over-cancelled Series", 3, 1, 1, []float64{0, -0.82303606596215528, 0, 1, 1.8150869958131495, 0, 0, 1, 1}, []float64{0.96812587861227395, -1.0116132737360233, 5.4707524664615272}, []float64{0, 0, 1}, []float64{0}, 0.1}, 0, 1e-12},
	} {
		x0 := tc.x0()
		if w := dcGainRationalLimit(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, x0)[0]; !math.IsInf(w, 0) {
			t.Fatalf("%s: exact limit %v, want a pole", tc.name, w)
		}
		g, err := tc.system(t).DCGain()
		requireDCGainUnverified(t, tc.name, g, err)
		if v := g.At(0, 0); !(math.Abs(v-tc.want) <= tc.tol*math.Max(1, math.Abs(tc.want))) {
			t.Errorf("%s: DCGain = %.17g, want %.17g", tc.name, v, tc.want)
		}
	}
}

// N4GQHT: an exactly cancelled integrator is the finite exact limit and a
// true integrator the exact signed Inf, both with a nil error, on
// non-symmetric A, continuous and discrete, SISO and MIMO.
func TestDCGainExactPoleCertified(t *testing.T) {
	for _, r := range []dcGainLiteral{
		{"exact cancellation", 2, 1, 1, []float64{-4, 8, 0, 0}, []float64{0.5, 1.5}, []float64{1, -2}, []float64{0.25}, 0},
		{"integrator through a lag, negative residue", 2, 1, 1, []float64{0, 0.75, 0, -2}, []float64{0, 1}, []float64{-1.5, 0.5}, []float64{0}, 0},
		{"discrete integrator beside a coupled pair", 3, 2, 2, []float64{1, 0.25, -0.5, 0, 0.5, 0.75, 0, -0.25, 0.4}, []float64{1, -2, 0, 1, 0.5, 0.3}, []float64{-0.25, 0, 1, 0, 1, -1}, []float64{0, 1, -1, 0.5}, 0.1},
	} {
		checkDCGainAgainstRationalLimit(t, r.name, r.n, r.m, r.p, r.a, r.b, r.c, r.d, r.dt)
		g, err := r.system(t).DCGain()
		if err != nil {
			t.Fatal(err)
		}
		want := dcGainRationalLimit(r.n, r.m, r.p, r.a, r.b, r.c, r.d, r.x0())
		inf := false
		for k, v := range g.RawMatrix().Data {
			inf = inf || math.IsInf(v, 0) && v == want[k]
		}
		if r.name != "exact cancellation" && !inf {
			t.Errorf("%s: DCGain %v has no exact integrator entry", r.name, g.RawMatrix().Data)
		}
	}
}

// N4GQHT: a pole within rounding of DC that a route reads as an integrator
// (the float model's exact limit finite) keeps the MATLAB Inf with
// ErrAccuracyUnverified; the exact limit beyond float64 range with no
// route reading a pole there is ErrOverflow. Rows frozen from the INGGF2
// census (arm64): pipeInf #107.modal3, F4nearDCTriangle #851 (base +Inf and
// −Inf with a nil error) and F9extremeScale #2122 (base D with a nil error).
func TestDCGainPoleWithinRoundingOfDC(t *testing.T) {
	for _, tc := range []struct {
		dcGainLiteral
		want []float64
	}{
		{dcGainLiteral{"modal rounded integrator", 3, 1, 1, []float64{5.8598012900453521e-32, 4.1540689902272088e-18, -1.3111349255063842e-16, -1.5764728921279438e-17, -0.16045152059710999, -9.8603053547527615e-17, 1.6541851305564131e-16, -3.1891247444509092e-17, -0.62922020791865096}, []float64{1.2781917802031404, 1.5858935485510073, -0.34666091737457755}, []float64{0.78235521107878858, -0.84638904859094288, -0.9873709857946118}, []float64{0}, 0}, []float64{math.Inf(1)}},
		{dcGainLiteral{"discrete triangle near z = 1", 3, 1, 2, []float64{0.73279578492275921, 0, -76.279436143689509, 2.8346520637232264, 0.99999999999999389, 0.049210126272714066, 0, 0, 0.99999999848547438}, []float64{1.3984991791951509, 0.35183235648865996, 1.1917553534802914}, []float64{0.48528070172499849, 1.3482529999699995, 2.9504717102270752, -0.31272776821111037, 0.20704052794468897, -1.5419357711756572}, []float64{-0.5735536809899684, -0.75984816552287349}, 0.1}, []float64{math.Inf(-1), math.Inf(-1)}},
	} {
		w := dcGainRationalLimit(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.x0())
		for k := range w {
			if math.IsInf(w[k], 0) {
				t.Fatalf("%s: exact limit %v, want finite", tc.name, w)
			}
		}
		g, err := tc.system(t).DCGain()
		requireDCGainUnverified(t, tc.name, g, err)
		for k, v := range g.RawMatrix().Data {
			if v != tc.want[k] {
				t.Errorf("%s: entry %d = %v, want %v", tc.name, k, v, tc.want[k])
			}
		}
	}
	overflow := dcGainLiteral{"exact limit beyond range", 3, 1, 2, []float64{-9.9999999999999973e-102, -4.6870336953354353e+38, 2.7000262584272997e+127, 0, -9.9999999999999971e-228, 5.5811239678783426e+173, 0, 0, -1.0000000000000001e-15}, []float64{0.086545845273014066, -0.32053165874752843, -1.828899759986589}, []float64{1.9230441056069847, 0.1577326354291807, -0.44092565157271751, -1.0309109257140241, 1.0747716684450626, -0.68398129521792117}, []float64{0.76867187315043006, 0.85718803677325262}, 0}
	if g, err := overflow.system(t).DCGain(); !errors.Is(err, ErrOverflow) {
		t.Errorf("%s: DCGain = %v, %v; want ErrOverflow", overflow.name, g, err)
	}
}

// N4GQHT: FreqResponse at ω = 0 against DCGain. Where FreqResponse reads
// the float model's pole (+Inf, unsigned, as MATLAB 1/0), as on the doubly
// cancelled Series integrators (INGGF2 seriesguard #0), DCGain's finite
// regular part comes with ErrAccuracyUnverified, as does its Inf on a pole
// within rounding of DC; a true integrator is Inf in both with a nil error,
// and where DCGain is certified finite the two agree.
func TestDCGainFreqResponseAtDCConsistency(t *testing.T) {
	for _, r := range []dcGainLiteral{
		{"sub-rounding residue", 4, 1, 1, []float64{0, -2.0923021674681676e-06, 0, 0, 1, -0.10913131676372737, 0, 0, 0, 1, 0, -0, 0, 0, 1, -0}, []float64{-1.3703096518060768e-05, -0.71473279051562744, 6.5492913648521789, 0}, []float64{0, 0, 0, 1}, []float64{0}, 0},
		{"integrator through a lag", 2, 1, 1, []float64{0, 0.75, 0, -2}, []float64{0, 1}, []float64{-1.5, 0.5}, []float64{0}, 0},
		{"modal rounded integrator", 3, 1, 1, []float64{5.8598012900453521e-32, 4.1540689902272088e-18, -1.3111349255063842e-16, -1.5764728921279438e-17, -0.16045152059710999, -9.8603053547527615e-17, 1.6541851305564131e-16, -3.1891247444509092e-17, -0.62922020791865096}, []float64{1.2781917802031404, 1.5858935485510073, -0.34666091737457755}, []float64{0.78235521107878858, -0.84638904859094288, -0.9873709857946118}, []float64{0}, 0},
		{"exact cancellation", 2, 1, 1, []float64{-4, 8, 0, 0}, []float64{0.5, 1.5}, []float64{1, -2}, []float64{0.25}, 0},
	} {
		sys := r.system(t)
		g, err := sys.DCGain()
		if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("%s: DCGain: %v", r.name, err)
		}
		fr, ferr := sys.FreqResponse([]float64{0})
		if ferr != nil {
			t.Fatalf("%s: FreqResponse: %v", r.name, ferr)
		}
		h := fr.At(0, 0, 0)
		strict := dcGainRationalLimit(r.n, r.m, r.p, r.a, r.b, r.c, r.d, r.x0())[0]
		switch {
		case err != nil:
			if !cmplx.IsInf(h) {
				t.Errorf("%s: FreqResponse(0) = %v beside DCGain %v with %v, want Inf", r.name, h, g.At(0, 0), err)
			}
		case math.IsInf(strict, 0):
			if !cmplx.IsInf(h) || g.At(0, 0) != strict {
				t.Errorf("%s: FreqResponse(0) = %v, DCGain = %v, exact limit %v", r.name, h, g.At(0, 0), strict)
			}
		default:
			if !(cmplx.Abs(h-complex(g.At(0, 0), 0)) <= 1e-9*math.Abs(strict)) {
				t.Errorf("%s: FreqResponse(0) = %v, DCGain = %v", r.name, h, g.At(0, 0))
			}
		}
	}
}

// N4GQHT: beyond the exact check (n > exactDetMaxOrder) a nil error comes
// only from an exactly decoupled integrator or a structurally exact
// expansion; an integrator rounded by a dense similarity keeps its reading
// with ErrAccuracyUnverified.
func TestDCGainHighOrderPoleAtDC(t *testing.T) {
	const n = exactDetMaxOrder + 2
	rng := rand.New(rand.NewPCG(17, 4))
	diag := mat.NewDense(n, n, nil)
	for k := 1; k < n; k++ {
		diag.Set(k, k, -float64(k))
		if k+1 < n {
			diag.Set(k, k+1, 0.5)
		}
	}
	b, c := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for k := range n {
		b.Set(k, 0, rng.NormFloat64())
		c.Set(0, k, rng.NormFloat64())
	}
	exact, err := New(diag, b, c, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := exact.DCGain()
	if err != nil {
		t.Fatalf("decoupled integrator: %v", err)
	}
	if want := math.Inf(signInt(b.At(0, 0) * c.At(0, 0))); g.At(0, 0) != want {
		t.Errorf("decoupled integrator: DCGain = %v, want %v", g.At(0, 0), want)
	}

	T := mat.NewDense(n, n, nil)
	for i := range n * n {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti, tmp, A mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	tmp.Mul(T, diag)
	A.Mul(&tmp, &Ti)
	rounded, err := New(&A, b, c, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err = rounded.DCGain()
	requireDCGainUnverified(t, "dense similarity", g, err)
}
