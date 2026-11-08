package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func lqgCurrentDescriptor(t *testing.T, A, B, C, Q, R, S, E, Rn *mat.Dense) (*LqgResult, error) {
	t.Helper()
	n, m := B.Dims()
	p, _ := C.Dims()
	sys, err := NewDescriptor(A, B, C, mat.NewDense(p, m, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	qxu := mat.NewDense(n+m, n+m, nil)
	qxu.Slice(0, n, 0, n).(*mat.Dense).Copy(Q)
	qxu.Slice(n, n+m, n, n+m).(*mat.Dense).Copy(R)
	if S != nil {
		qxu.Slice(0, n, n, n+m).(*mat.Dense).Copy(S)
		qxu.Slice(n, n+m, 0, n).(*mat.Dense).Copy(S.T())
	}
	qwv := mat.NewDense(n+p, n+p, nil)
	for i := range n {
		qwv.Set(i, i, 1)
	}
	qwv.Slice(n, n+p, n, n+p).(*mat.Dense).Copy(Rn)
	return acceptUnverifiedLqg(Lqg(sys, qxu, qwv, &LqgOpts{Current: true}))
}

// JGGEN2 census rv46 bnd 227: regulator closed loop at ρ 0.999998, ‖X‖
// 1e19, descriptorNoiseGain fails and G = R + BᵀXB is singular to float64
// (cond 2e16-9e16), so floatNoiseGain returned ErrSingularR (its Kw 0.8
// off when forced). Kw now comes from the extended Dare solve, matching
// the 320-bit (R + BᵀXB)⁻¹BᵀX, at Accuracy 1.
func TestLqg_CurrentBoundaryNoiseGainFromExtendedSolve(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.93830111079836132, 1.4032431681908395, -0.455146716312609, 1.1252396255691615})
	B := mat.NewDense(2, 2, []float64{0.21701201680315724, 0.4663895654239687, 0.3543098173700151, 0.76146198849823876})
	C := mat.NewDense(1, 2, []float64{0.52206489499407926, -1.0732990489317276})
	Q := mat.NewDense(2, 2, []float64{0.99999999999999956, -1.2275579728358081e-16, -1.2275579728358081e-16, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	E := mat.NewDense(2, 2, []float64{0.88505095275930279, 0.51924830385956677, -0.54208684816388109, -0.31803556056139609})
	Rn := mat.NewDense(1, 1, []float64{1.1740993830164248})
	want := mat.NewDense(2, 2, []float64{5.721442728230012, -3.0020285656130601, -2.0772697575662029, 2.3518485946453822})

	res, err := lqgCurrentDescriptor(t, A, B, C, Q, R, nil, E, Rn)
	if err != nil {
		t.Fatalf("Lqg: %v", err)
	}
	if res.accuracy != 1 {
		t.Errorf("Accuracy = %d, want 1", res.accuracy)
	}
	kw, _ := res.Kw()
	if d := relFrob(kw, want); d > 1e-9 {
		t.Errorf("Kw %.3g off the 320-bit gain", d)
	}
}

// JGGEN2 census rv46 edge 919: every Riccati certified, but S = CPCᵀ + Rn
// has cond 1.1e14 (P ~ 1/σ_min(E)²), so Mx = PCᵀS⁻¹ is 2.8e-3 off the
// 320-bit gain; it was returned at Accuracy 0. edge 92 has cond(S) 2.7e16
// and keeps ErrSingularEquation.
func TestLqg_CurrentIllConditionedInnovationIsUncertified(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.56953369244727525, -0.65775888940867411, 1.3728968265214614, 0.39562010574080331, -1.3507810615315679, -1.6243737370404396, -0.60016448954579038, -0.32912663943122067, -0.52668565839310466})
	B := mat.NewDense(3, 2, []float64{1.4148994829950834, 0.30979585314651981, 1.1165661392410542, -0.56760563073155224, -1.4976217452268268, 0.22425543812900492})
	C := mat.NewDense(2, 3, []float64{0.7968717977453772, -0.0038775330579481349, -0.94049376192270606, 0.82050354482078425, 2.0214491626049904, -1.0560618286911965})
	Q := mat.NewDense(3, 3, []float64{4.5422750570014987, 0.93197983427198383, -1.5960731999357591, 0.93197983427198383, 1.2452058063003397, -0.41993013315607536, -1.5960731999357591, -0.41993013315607536, 1.7191563666175504})
	R := mat.NewDense(2, 2, []float64{2.3528007331779026, -0.10509253124602419, -0.10509253124602419, 1.0081641293154475})
	S := mat.NewDense(3, 2, []float64{-2.1890619667404172, 0.17005761269712202, -0.57594669418495525, 0.044742506764088918, 0.98634439220273296, -0.076624314516301195})
	E := mat.NewDense(3, 3, []float64{0.89298261004396806, 0.078941422622581858, 0.43425110379535869, 0, 0.73798108422397024, -1.4627241630543975, 0, 0, 9.3479805826880943e-08})
	Rn := mat.NewDense(2, 2, []float64{1.1602888773090823, -1.4389951320438528, -1.4389951320438528, 3.7173510842700126})
	res, err := lqgCurrentDescriptor(t, A, B, C, Q, R, S, E, Rn)
	if err != nil {
		t.Fatalf("Lqg: %v", err)
	}
	if res.accuracy != 1 {
		t.Errorf("Accuracy = %d, want 1", res.accuracy)
	}

	A92 := mat.NewDense(2, 2, []float64{-0.090074153033864235, -0.0045558063145923564, 0.16258408844141831, 0.86435830695651372})
	B92 := mat.NewDense(2, 1, []float64{-0.026372021295875375, -1.09779572873084})
	C92 := mat.NewDense(2, 2, []float64{1.1668000910336576, 0.34571089408734301, -1.2818980063272178, -1.0125681116346612})
	Q92 := mat.NewDense(2, 2, []float64{2.6745731022600308, -0.17867586016639794, -0.17867586016639794, 0.025002577384781375})
	R92 := mat.NewDense(1, 1, []float64{4.187915760250533})
	E92 := mat.NewDense(2, 2, []float64{0.042667561936902992, 1.0004956426373759, -0.035049159346558195, -0.82185423298958682})
	Rn92 := mat.NewDense(2, 2, []float64{3.6869102066077715, -1.9453082496964869, -1.9453082496964869, 3.2010341771583319})
	if _, err := lqgCurrentDescriptor(t, A92, B92, C92, Q92, R92, nil, E92, Rn92); !errors.Is(err, ErrSingularEquation) {
		t.Errorf("edge 92: err = %v, want ErrSingularEquation", err)
	}
}
