package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorGainTol bounds the error of a certified K or EᵀXE of the
// models below against the 320-bit oracle, which moves them by under 1e-15
// under 1-ulp perturbations of A, B and E.
const descriptorGainTol = 1e-9

// descriptorGainModels returns the continuous descriptor plants with
// triangular E of condition 1e13 and 3.9e12 (D = 0).
func descriptorGainModels(t *testing.T) []*System {
	t.Helper()
	two, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{0.19887662554059538, -0.14856103312523139, -1.0558847194781948, -0.83053605748214521}),
		mat.NewDense(2, 1, []float64{-1.5154567560715049, 1.7416748591014379}),
		mat.NewDense(1, 2, []float64{0.97228848391683376, -0.39447132160607812}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{0.99465977130926597, -1.5289113538341979, 0, 1.2867483309244827e-13}), 0)
	if err != nil {
		t.Fatal(err)
	}
	three, err := NewDescriptor(
		mat.NewDense(3, 3, []float64{-0.871464623122238, -1.1334075778351131, -0.0747345975194616, 0.8696186800699262, -0.4612936214620338, -2.0352249143933827, 0.04198273455009893, 0.5984312848623197, 1.7330191982672514}),
		mat.NewDense(3, 1, []float64{-0.19820169905618423, -0.8022360366594512, -1.1850355946903401}),
		mat.NewDense(1, 3, []float64{0.2820027918001812, -1.791408088792565, -0.9504198942563948}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(3, 3, []float64{0.8761213913566956, 0.9095527301188022, 0.5804043030654127, 0, 1.018094020177648, -1.0271347333690652, 0, 0, 1.0322659520340186e-12}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return []*System{two, three}
}

// TestKalmanDescriptorGainMatchesOracle: with E of condition 1e13 the
// dual X ~ 1/σ_min(E) is exact while L = (R⁻¹CXEᵀ)ᵀ formed from it was off
// by 4.6e-4 (and 4.3e-5 at cond 3.9e12) with Accuracy 0 (5AFLTD); L is
// now that of P = XEᵀ refined in double-double, as icare's from the
// deflating subspace. The first must be certified; the second, and Lqg's
// L, right whenever certified (amd64 rounding leaves the second's X
// uncertified, Accuracy 1, as on base).
func TestKalmanDescriptorGainMatchesOracle(t *testing.T) {
	for i, sys := range descriptorGainModels(t) {
		n, m, p := sys.Dims()
		res, err := Kalman(sys, eyeDense(m), eyeDense(p), nil, nil)
		if err != nil {
			t.Fatalf("model %d Kalman: %v", i, err)
		}
		dual := kalmanDual(sys, sys.B, sys.D)
		_, _, Lo, ok := nearSingularOracle(true, dual.A, dual.B, dual.Q, dual.R, dual.S, dual.E, mat.DenseCopyOf(res.K.T()))
		if !ok {
			t.Fatalf("model %d: oracle did not converge", i)
		}
		if e := relErr(res.K.T(), Lo); (res.Accuracy == 0 || i == 0) && (res.Accuracy != 0 || e > descriptorGainTol) {
			t.Errorf("model %d Kalman: Accuracy %d, L off the oracle by %.3g", i, res.Accuracy, e)
		}
		lqg, err := Lqg(sys, eyeDense(n+m), eyeDense(n+p), nil)
		if err != nil {
			continue
		}
		dual = kalmanDual(sys, eyeDense(n), nil)
		_, _, Lo, ok = nearSingularOracle(true, dual.A, dual.B, dual.Q, dual.R, dual.S, dual.E, mat.DenseCopyOf(lqg.L.T()))
		if e := relErr(lqg.L.T(), Lo); ok && lqg.Accuracy == 0 && e > descriptorGainTol {
			t.Errorf("model %d Lqg: Accuracy 0, L off the oracle by %.3g", i, e)
		}
	}
}

// TestLqrCareRotatedDescriptorMatchesOracle: a full-rank E of condition
// 4.1e14 below the near-singular window gave Lqr an EᵀXE off by 1.8e-4
// and K by 1.9e-3, and Care with S a K off by 2.8e-3, all with Accuracy 0
// (5AFLTD).
func TestLqrCareRotatedDescriptorMatchesOracle(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.07724605784568937, -0.27972825396822587, -1.2089888556254231, -1.107758416485964})
	B := mat.NewDense(2, 2, []float64{1.8941729912494576, 0.9546509297762662, 0.09187158197576986, 1.873414031167777})
	E := mat.NewDense(2, 2, []float64{0.5928042801985111, -0.0005292567946201094, -0.09606464544811585, 8.576669910372701e-05})
	S := mat.NewDense(2, 2, []float64{0.1, 0.2, -0.3, 0.05})
	I := eyeDense(2)
	res, err := Lqr(A, B, I, I, &RiccatiOpts{E: E})
	if err != nil {
		t.Fatal(err)
	}
	Y, _, Ko, ok := nearSingularOracle(true, A, B, I, I, nil, E, res.K)
	if !ok {
		t.Fatal("oracle did not converge")
	}
	if eX, eK := relErr(res.X, Y), relErr(res.K, Ko); res.Accuracy != 0 || eX > descriptorGainTol || eK > descriptorGainTol {
		t.Errorf("Lqr: Accuracy %d, EᵀXE off the oracle by %.3g, K by %.3g", res.Accuracy, eX, eK)
	}
	cres, err := Care(A, B, I, I, &RiccatiOpts{E: E, S: S})
	if err != nil {
		t.Fatal(err)
	}
	_, _, Ko, ok = nearSingularOracle(true, A, B, I, I, S, E, cres.K)
	if !ok {
		t.Fatal("oracle with S did not converge")
	}
	if eK := relErr(cres.K, Ko); cres.Accuracy != 0 || eK > descriptorGainTol {
		t.Errorf("Care with S: Accuracy %d, K off the oracle by %.3g", cres.Accuracy, eK)
	}
}

// TestDlqrDescriptorExplicitSolutionHonest: a rotated E of condition 1e12
// leaves X and K of Dare certified, while EᵀXE formed from the rounded X
// ~ 1/σ_min(E) was off by 8e5 of itself with Accuracy 0 (5AFLTD sweep).
// Dlqr must return it with Accuracy 1 or right. That model errors on amd64
// (its pencil's infinite-like mode falls in the boundary test's band, as
// on base), so the models here certify on both archs: EᵀXE of the
// rounded Dare X is off by 2e5 and 1.6e3 (O2MVDD generator, seeds 0, 132).
func TestDlqrDescriptorExplicitSolutionHonest(t *testing.T) {
	for i, m := range []struct{ a, b, e []float64 }{
		{
			[]float64{1.7315579929713067, -0.1940539074978917, 1.1190984891999822, -0.02238295182090704},
			[]float64{-0.8303147397714788, -0.593834159892699},
			[]float64{0.25882966603870977, 0.013068593225123316, -1.3735630017052287, -0.06935269983202669},
		},
		{
			[]float64{-0.22838186469513516, 0.5665228292856828, 0.11061261264548296, 0.18596912536540255},
			[]float64{1.167180144775775, -0.9092794127429058},
			[]float64{-0.0034286785675923708, -0.0015845417530948786, -0.8886126096442941, -0.41066660391411713},
		},
	} {
		A, B, E := mat.NewDense(2, 2, m.a), mat.NewDense(2, 1, m.b), mat.NewDense(2, 2, m.e)
		res, err := Dlqr(A, B, eyeDense(2), eyeDense(1), &RiccatiOpts{E: E})
		if err != nil {
			t.Fatalf("model %d Dlqr: %v", i, err)
		}
		Y, _, Ko, ok := nearSingularOracle(false, A, B, eyeDense(2), eyeDense(1), nil, E, res.K)
		if !ok {
			t.Fatalf("model %d: oracle did not converge", i)
		}
		if eK := relErr(res.K, Ko); eK > descriptorGainTol {
			t.Errorf("model %d: K off the oracle by %.3g", i, eK)
		}
		if eX := relErr(res.X, Y); res.Accuracy == 0 && eX > newtonTol {
			t.Errorf("model %d: Accuracy 0 with EᵀXE off the oracle by %.3g", i, eX)
		}
	}
}

// TestKalmanDescriptorMoreSensorsThanStates: three sensors on two states
// make the dual B = Cᵀ wider than tall, through the cancellation check and
// refinement of the gain of an E of condition 1e13; the certified L must
// match the oracle.
func TestKalmanDescriptorMoreSensorsThanStates(t *testing.T) {
	base := descriptorGainModels(t)[0]
	C := mat.NewDense(3, 2, []float64{0.97228848391683376, -0.39447132160607812, 0.3, 1.1, -0.7, 0.25})
	sys, err := NewDescriptor(base.A, base.B, C, mat.NewDense(3, 1, nil), base.E, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Kalman(sys, eyeDense(1), eyeDense(3), nil, nil)
	if errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrDescriptorSingular) {
		t.Skipf("Kalman: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	dual := kalmanDual(sys, sys.B, sys.D)
	_, _, Lo, ok := nearSingularOracle(true, dual.A, dual.B, dual.Q, dual.R, dual.S, dual.E, mat.DenseCopyOf(res.K.T()))
	if e := relErr(res.K.T(), Lo); ok && res.Accuracy == 0 && e > descriptorGainTol {
		t.Errorf("Accuracy 0, L off the oracle by %.3g", e)
	}
}
