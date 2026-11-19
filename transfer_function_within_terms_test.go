package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestTransferFunctionExplicitCancellationWithinTerms: an explicit model's
// TransferFunction carries the within-terms claim (SL352H, user decision
// 2026-11-18). The frozen model is the E⁻¹ fold of seed 199 of
// TestTransferFunctionBadlyScaledDiagonalEStaysExact: non-symmetric A whose
// rows span 1e-6..1e6, so ss→tf cancels the denominator's constant term
// (1.7e3 against coefficients near 6e11) and the TF misses the response at
// s = 0 by 2.4e-5, 1.95× the threshold. Base returned it with nil.
func TestTransferFunctionExplicitCancellationWithinTerms(t *testing.T) {
	a := mat.NewDense(6, 6, []float64{
		-2.1526395573943165e-06, -1.6098102592741983e-07, 1.024187842785151e-06, 8.554534076887711e-07, 1.5968979345070293e-06, 6.980546990031616e-07,
		-1.2186441756691468, 1.383274645855612, -0.1781166681523655, -1.6584758680694145, -1.514016320795996, -1.3835385171315928,
		0.7454617967349538, 0.8249511188117254, -1.5331568453433537, -0.3354773247379212, -0.11770012767961613, 0.9352781717674695,
		-0.03178332463999581, 0.06849066153601797, -0.02386892811350999, -0.05774464709923803, 0.012611823850270121, -0.038910639857539196,
		636487.1613737826, 307727.44956641813, -305997.62464879156, -51128.43174224292, 557620.2286870152, -32646.9990874342,
		33272.12074610738, 221871.68802671094, 153573.13904404672, 140149.38067890162, 210763.12867068683, -47918.109564502454,
	})
	b := mat.NewDense(6, 1, []float64{8.414884089738574e-07, 3.615669355867621, -1.706035551377568, 0.009281780791814468, 694956.8244158281, -34223.756628359646})
	c := mat.NewDense(1, 6, []float64{-1.5472765647609799, -0.8102724218481375, 0.9620129068061639, 1.3209050785395942, -0.13558472630240903, -1.2751911458719805})
	d := mat.NewDense(1, 1, []float64{-0.4739316006621122})
	sys, err := New(a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	requireTFWithinTermsOrUnverified(t, sys, res, err)
}

// TestTransferFunctionFoldUnderflowingEinvBUnverified: the E⁻¹ fold of a
// descriptor at 2^±600 underflows E⁻¹B to 0, so its TF is ≡ 0 against a
// nonzero response; base returned it with nil (SL352H; frozen ovf/14 of
// the 5CFETF census). The fold's residual E·B̄ − B = −B is no rounding of
// B, the pencil certificate refutes it, and it comes back unverified.
func TestTransferFunctionFoldUnderflowingEinvBUnverified(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{-1.3076812206422687e+179, 1.217251857475639e+178, 1.1162048192073216e+179, -8.790741366365397e+178})
	b := mat.NewDense(2, 2, []float64{-7.025645444127569e-166, 1.5071368202610764e-166, -2.169155599180122e-166, 1.6285480104683778e-166})
	c := mat.NewDense(1, 2, []float64{-1.212546455738041e+293, -1.5385790206657246e+294})
	e := mat.NewDense(2, 2, []float64{1.3338663855694357e+179, -3.1383411901755512e+177, -1.1312463736812494e+177, 2.027122678256875e+179})
	sys, err := NewDescriptor(a, b, c, mat.NewDense(1, 2, nil), e, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	requireTFWithinTermsOrUnverified(t, sys, res, err)
}

// TestExplicitFastTierVerifiedPointsWithinTerms: at every point the
// explicit fast tier verifies (explicitFastTier.verify) the transfer
// function, divided at 320 bits, is within √ε·max(scale, T) of the exact
// response (exactPencilAt), T the entry's exact terms, on random dense
// non-symmetric models, continuous and discrete, with and without a badly
// scaled diagonal similarity.
func TestExplicitFastTierVerifiedPointsWithinTerms(t *testing.T) {
	tol := math.Sqrt(eps())
	verified := 0
	for seed := range uint64(24) {
		rng := rand.New(rand.NewPCG(seed, 0x5f7))
		n, m, p := 2+rng.IntN(6), 1+rng.IntN(2), 1+rng.IntN(2)
		a, b, c, d := make([]float64, n*n), make([]float64, n*m), make([]float64, p*n), make([]float64, p*m)
		for i := range a {
			a[i] = rng.NormFloat64() / math.Sqrt(float64(n))
		}
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		for i := range c {
			c[i] = rng.NormFloat64()
		}
		for i := range d {
			d[i] = rng.NormFloat64()
		}
		if seed%2 == 1 {
			sc := make([]float64, n)
			for i := range sc {
				sc[i] = math.Pow(10, 6*(2*rng.Float64()-1))
			}
			for i := range n {
				for j := range n {
					a[i*n+j] *= sc[i] / sc[j]
				}
				for j := range m {
					b[i*m+j] *= sc[i]
				}
				for j := range p {
					c[j*n+i] /= sc[i]
				}
			}
		}
		dt := 0.0
		if seed%3 == 2 {
			dt = 0.1
			for i := range a {
				a[i] *= 0.5
			}
		}
		sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if res == nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		pencil := identityPencil(sys)
		roots, _, _, _, ok := tfShape(res.TF)
		if !ok {
			continue
		}
		ft := newExplicitFastTier(sys, newTFRows(res.TF))
		for _, s := range descriptorTFPoints(pencil, roots) {
			if !ft.verify(s) {
				continue
			}
			verified++
			ur, ui := exactPencilAt(pencil, s)
			if ur == nil {
				t.Fatalf("seed %d: verified a pole at %v", seed, s)
			}
			for i := range p {
				r, g, T := make([]complex128, m), make([]complex128, m), make([]float64, m)
				scale := 0.0
				for j := range m {
					r[j], _ = tfEntryAt(res.TF.Num[i][j], res.TF.Den[i], s)
					g[j], T[j] = exactEntry(pencil, ur, ui, i, j)
					scale = math.Max(scale, math.Max(cmplx.Abs(r[j]), cmplx.Abs(g[j])))
				}
				for j := range m {
					if q := (cmplx.Abs(r[j]-g[j]) - 4*eps()*(cmplx.Abs(r[j])+cmplx.Abs(g[j]))) / (tol * math.Max(scale, T[j])); q > 1 {
						t.Fatalf("seed %d: verified %v at %v, %.3g× √ε·max(scale, T) off", seed, r[j], s, q)
					}
				}
			}
		}
	}
	if verified == 0 {
		t.Fatal("no point verified")
	}
}

// TestTransferFunctionZeroCScaledDescriptorExact: C scaled by 2⁻⁶⁰⁰ down
// to zero leaves a static model, G ≡ D, while U = (sE − A)⁻¹B overflows
// (E ~ 1e-200, B ~ 1e123), so the pencil judge has no determined point.
// The TF D/1 is exact and must come back nil (frozen sc-gen/113 of the
// SL352H census).
func TestTransferFunctionZeroCScaledDescriptorExact(t *testing.T) {
	a := mat.NewDense(5, 5, []float64{
		-6.482423315255261e-201, -7.250941381473191e-201, 5.155605915063175e-201, -1.2434300787758613e-200, -1.83533648919143e-200,
		-1.0402641242474691e-201, -4.214240534898938e-201, -2.4816805079909427e-200, -2.0144040984927038e-203, -9.801294119610885e-201,
		2.0357266285170537e-200, 1.2565499319135401e-200, 7.758038980908553e-201, -4.765353965737153e-202, 1.520672725799147e-200,
		1.2045186531148942e-200, -6.4255718592128355e-201, -1.6420397995601e-201, -1.544348677424788e-200, -7.007907380410447e-201,
		-2.4049249727189326e-200, 2.2199916002675208e-200, 1.2823443819231273e-200, -2.5246436946572457e-200, -3.2318749969325064e-200,
	})
	b := mat.NewDense(5, 1, []float64{-8.12919484825432e+123, 3.6995818016749145e+123, 1.2392441003926717e+123, 7.834799232951917e+123, -1.394379773316501e+123})
	e := mat.NewDense(5, 5, []float64{
		2.5170928702310347e-200, -5.924589804764107e-203, -5.382197770318307e-203, -1.0188940353240161e-202, 2.3345487718398027e-202,
		5.540693782708861e-204, 2.4277706590083818e-200, 1.053590388266575e-202, 1.1016820817092456e-202, 7.20880380189886e-204,
		8.103077018562128e-203, -4.834222709103202e-203, 1.3352154878628966e-200, -5.95195524544102e-203, 1.4765322776866952e-202,
		-1.4681517269163384e-202, -2.5488814121823532e-203, 2.5467787872254975e-202, 2.1455705121684678e-200, 1.3550959087921906e-202,
		8.538773842545555e-203, -1.3228385392385017e-202, 1.3979997042035615e-203, 7.718123013228834e-203, 1.3229212666202935e-200,
	})
	d := mat.NewDense(2, 1, []float64{-0.4159759885164252, -0.3049029936182224})
	sys, err := NewDescriptor(a, b, mat.NewDense(2, 5, nil), d, e, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		if v, _ := tfEntryAt(res.TF.Num[i][0], res.TF.Den[i], 1i); v != complex(d.At(i, 0), 0) {
			t.Fatalf("row %d: TF(j) = %v, want D = %v", i, v, d.At(i, 0))
		}
	}
}

// TestDescriptorExactEntryTermsBeyondFloatRange: at s = 1e10j the exact U
// of this 2^±600 model is ~1e-324, below float64, while C ~ 1e295 brings
// C·U back to ~1e-30. exactEntry rounded each |U| first and returned the
// terms T = 0 < |G|, so the pencil judge refuted a TF within 0.21× of the
// threshold at 2.07× (frozen ovf/988 of the SL352H census).
func TestDescriptorExactEntryTermsBeyondFloatRange(t *testing.T) {
	a := mat.NewDense(3, 3, []float64{3.8348409356454316e+188, 6.943510581867232e+187, 9.190267969656211e+187, 5.997618646428171e+188, 6.29897391141415e+187, 9.846928918499896e+187, 2.935262921736679e+188, 1.4695591477171504e+189, -4.95373929706092e+188})
	b := mat.NewDense(3, 1, []float64{1.3505940664608903e-126, 1.0275053260571309e-126, 9.127785212276266e-127})
	c := mat.NewDense(2, 3, []float64{-5.629540717275584e+294, -5.911275284317662e+294, 2.379019455318654e+295, -5.301639358489369e+294, -2.4805886895083824e+295, -1.0879334013493183e+295})
	e := mat.NewDense(3, 3, []float64{5.942926229080493e+188, -5.58589092223062e+186, 4.116237123017461e+187, 8.486583687200944e+185, 9.49149520197582e+188, -9.750041938651018e+187, 3.108555377628159e+187, 5.495528316163908e+187, 1.0328571138978448e+189})
	sys, err := NewDescriptor(a, b, c, mat.NewDense(2, 1, nil), e, 0)
	if err != nil {
		t.Fatal(err)
	}
	ur, ui := exactPencilAt(sys, 1e10i)
	if ur == nil {
		t.Fatal("singular pencil")
	}
	for i := range 2 {
		g, T := exactEntry(sys, ur, ui, i, 0)
		if !(T >= cmplx.Abs(g)*(1-4*eps())) || g == 0 {
			t.Fatalf("row %d: terms %g below |G| = %g", i, T, cmplx.Abs(g))
		}
	}
	res, err := sys.TransferFunction(nil)
	requireTFWithinTermsOrUnverified(t, sys, res, err)
}
