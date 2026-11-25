package controlsys

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// h2ExactReference returns √(tr(Bᵀ·Wo·B) + ‖D‖_F²) (‖D‖_F only when
// discrete) from the observability gramian with its right-hand side, solve
// and quadratic form in hbPrec bits.
func h2ExactReference(sys *System) (float64, bool) {
	disc := sys.Dt > 0
	b, c := hbFrom(sys.B), hbFrom(sys.C)
	wo, ok := hsvExactLyap(hbFrom(mat.DenseCopyOf(sys.A.T())), hbMul(c.t(), c), disc)
	if !ok {
		return 0, false
	}
	q := hbMul(hbMul(b.t(), wo), b)
	tr := new(big.Float).SetPrec(hbPrec)
	for i := 0; i < q.r; i++ {
		tr.Add(tr, q.at(i, i))
	}
	f, _ := tr.Float64()
	if disc {
		d := mat.Norm(sys.D, 2)
		f += d * d
	}
	return math.Sqrt(math.Max(f, 0)), true
}

// h2WeakCoupling: rotated decoupled halves joined by a coupling 2⁻ᵏ whose
// plain-path H2 norm is wrong by 0.5% to 150× (D5BGNA sweep rows
// sw0k32r1/86, sw0k30r1/122, sw1k28r1/733, sw0k23r1/186, arm64).
var h2WeakCoupling = []hsvCase{
	{name: "sw0k32r1/86 (arm64)", n: 3, m: 1, p: 2, dt: 0,
		a:   []float64{-3.7184780771947867, -0.7229225894776641, 0.3198204965384432, -0.7229225895770389, -0.74534537127658, 0.7872896393992797, 0.31982049674621477, 0.7872896394334216, -1.7861765515286305},
		b:   []float64{-1.8740908206623992, -0.5271076721420149, 0.458193297504404},
		c:   []float64{-0.6102049271299785, 2.1510926848437824, -0.021218108253635237, 1.8542646342548859, -5.034879855775046, 1.7921181613002608},
		ref: []float64{1.4246000706639738e-10, 2.2461850660336087e-11, 7.4505953210005621e-17}, T: 4.5708813536309183},
	{name: "sw0k30r1/122 (arm64)", n: 3, m: 2, p: 1, dt: 0,
		a:   []float64{-1.8938070039712618, 1.4537143525744525, -0.1491112628094831, 1.4537143557305585, -2.3767944046646345, 0.9415641586320876, -0.14911126327374996, 0.9415641567082275, -2.229398591364105},
		b:   []float64{0.13321260709786276, -1.065700856782902, -0.18192900187997227, 1.4554320150397781, 0.10796406617549889, -0.8637125294039911},
		c:   []float64{0.8242661787579955, 1.4735480041930042, 1.4660291763449003},
		ref: []float64{4.09428261909764e-09, 3.4000139651045349e-10, 3.1448670699090527e-19}, T: 1.4542183460462881},
	{name: "sw1k28r1/733 (arm64)", n: 2, m: 1, p: 2, dt: 0.5,
		a:   []float64{-0.25000000009101825, 1.8899760908295123e-10, -4.3833034570918403e-11, -0.24999999990898164},
		b:   []float64{0.867781900758041, -1.8019307902127542},
		c:   []float64{0.9009653951063771, 0.4338909503790206, -1.8019307902127542, -0.8677819007580412},
		ref: []float64{1.5173499720554779e-09, 9.2499480426589202e-10}, T: 4.7702783519995506},
	{name: "sw0k23r1/186 (arm64)", n: 3, m: 1, p: 2, dt: 0,
		a:   []float64{-0.5000000106039604, -3.1859378140735496e-09, -1.359706750422552e-08, -9.989279787189711e-09, -0.5000000030012582, -1.2808885311019072e-08, 1.0610317481205215e-08, 3.1878478076759114e-09, -0.4999999863947811},
		b:   []float64{-0.15118322657790545, -0.0454226862573523, -0.19385667792124114},
		c:   []float64{0.9702946888739374, -4.357597909899477, 0.2643268287229673, 0.7793349656493281, -1.9016645305938515, -0.16220056842571917},
		ref: []float64{3.7081753860413737e-08, 6.3622231236687402e-09, 5.5959183072216666e-25}, T: 1.2268986996756475},
}

func TestH2Norm_WeakCouplingMatchesExact(t *testing.T) {
	for _, c := range h2WeakCoupling {
		sys := c.sys()
		want, ok := h2ExactReference(sys)
		if !ok {
			t.Fatalf("%s: exact reference failed", c.name)
		}
		h, err := H2Norm(sys)
		if err != nil {
			t.Fatalf("%s: H2Norm err %v, want nil", c.name, err)
		}
		if d := math.Abs(h-want) / want; d > hsvTau/8 {
			t.Fatalf("%s: H2Norm = %.17g, exact %.17g (relative %.3g > τ/8)", c.name, h, want, d)
		}
	}
}

func TestH2Norm_BelowResolutionUnverified(t *testing.T) {
	for _, c := range hsvBelowResolution {
		h, err := H2Norm(c.sys())
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("%s: H2Norm = %v, err %v, want ErrAccuracyUnverified", c.name, h, err)
		}
		if h == 0 {
			t.Fatalf("%s: H2Norm = 0 for a nonzero transfer function", c.name)
		}
	}
}

func TestH2Norm_ExactlyZeroTransferFunction(t *testing.T) {
	cancelA, cancelB, cancelC := zeroCancelSystem(false)
	cancel, _ := New(cancelA, cancelB, cancelC, mat.NewDense(1, 1, nil), 0)
	structural := hsvLiteral(3, 2, 1,
		[]float64{-1, 0.5, 0.25, 0, -2, 0, 0, 0.75, -3},
		[]float64{1, -2, 0, 0, 0, 0},
		[]float64{0, 1, 4}, 0)
	discD, err := NewFromSlices(2, 1, 1, []float64{-0.75, 0.25, -0.25, -0.25}, []float64{1, 1}, []float64{1, -1}, []float64{0.75}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		sys  *System
		want float64
	}{{"structural", structural, 0}, {"cancellation", cancel, 0}, {"cancellation discrete with D", discD, 0.75}} {
		h, err := H2Norm(c.sys)
		if err != nil || math.Abs(h-c.want) > hsvTau*c.want/8 || c.want == 0 && h != 0 {
			t.Fatalf("%s: H2Norm = %v, %v; want %v, nil", c.name, h, err, c.want)
		}
	}
	ulpA, ulpB, ulpC := zeroCancelSystem(true)
	ulp, _ := New(ulpA, ulpB, ulpC, mat.NewDense(1, 1, nil), 0)
	h, err := H2Norm(ulp)
	if h == 0 && err == nil {
		t.Fatal("one ulp from a zero transfer function: H2Norm = 0 with nil error")
	}
	want, ok := h2ExactReference(ulp)
	if !ok || !(want > 0) {
		t.Fatalf("one-ulp model: exact reference %v, ok %v", want, ok)
	}
	if err == nil && math.Abs(h-want) > hsvTau*want/8 {
		t.Fatalf("one-ulp model: nil H2Norm %v, exact %v", h, want)
	}
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("one-ulp model: H2Norm err %v, want nil or ErrAccuracyUnverified", err)
	}
}

// rlrepd/591 (arm64) with D = 1: the state part ≈ 1.6·10⁻¹⁶ is below the
// double-double resolution, but the norm is ‖D‖_F to within 10⁻³¹.
func TestH2Norm_DiscreteFeedthroughDominates(t *testing.T) {
	c := hsvBelowResolution[1]
	sys, err := NewFromSlices(c.n, c.m, c.p, c.a, c.b, c.c, []float64{1}, c.dt)
	if err != nil {
		t.Fatal(err)
	}
	want, ok := h2ExactReference(sys)
	if !ok {
		t.Fatal("exact reference failed")
	}
	h, err := H2Norm(sys)
	if err != nil {
		t.Fatalf("H2Norm err %v, want nil", err)
	}
	if math.Abs(h-want) > hsvTau*want/8 || math.Abs(h-1) > hsvTau/8 {
		t.Fatalf("H2Norm = %.17g, exact %.17g, ‖D‖_F = 1", h, want)
	}
}

func TestNorm_H2MatchesH2Norm(t *testing.T) {
	cancelA, cancelB, cancelC := zeroCancelSystem(false)
	zero, _ := New(cancelA, cancelB, cancelC, mat.NewDense(1, 1, nil), 0)
	for _, c := range []struct {
		name string
		sys  *System
		unv  bool
	}{{"nil", h2WeakCoupling[0].sys(), false}, {"unverified", hsvBelowResolution[0].sys(), true}, {"exact zero", zero, false}} {
		h, err := H2Norm(c.sys)
		n, nerr := Norm(c.sys, NormH2)
		if h != n || (err == nil) != (nerr == nil) || err != nil && err.Error() != nerr.Error() {
			t.Fatalf("%s: Norm(2) = %v, %v; H2Norm = %v, %v", c.name, n, nerr, h, err)
		}
		if errors.Is(err, ErrAccuracyUnverified) != c.unv {
			t.Fatalf("%s: err %v, want unverified %v", c.name, err, c.unv)
		}
	}
}

// analysisGenPlant's H2-optimal closed loop has an exact H2 norm of
// 2.2·10⁻¹⁷ (a near cancellation in the stored K); the plain path returned
// 6.7·10⁻⁹ with nil (D5BGNA).
func TestH2Syn_ClosedLoopBelowResolutionUnverified(t *testing.T) {
	res, err := H2Syn(analysisGenPlant(), 1, 1)
	if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
		t.Fatalf("H2Syn = %v, err %v, want the full result with ErrAccuracyUnverified", res != nil, err)
	}
	want, ok := h2ExactReference(res.CL)
	if !ok {
		t.Fatal("exact reference of the closed loop failed")
	}
	if !(res.Gamma < 1e-15) || want >= 1e-15 {
		t.Fatalf("Gamma = %.3g, exact closed-loop H2 norm %.3g; want both below 1e-15", res.Gamma, want)
	}
}
