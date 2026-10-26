package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// HHNPGV: A = T·J·T⁻¹ with J diagonal holding the real eigenvalue
// 1.0001638027849067 twice, so its eigenspace is 2-D and no single input can
// reach both copies (PBH). Before the Schur-block cutoff scaled with ‖A‖, the
// rounding-level coupling of the last 2×2 block passed the block-local test on
// arm64, giving a ~1e14 gain and ErrSchurFailed from the next reordering,
// while amd64 returned ErrUncontrollable.
func TestPlaceSingleInputRepeatedEigenvalueUncontrollable(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{
		-5.440838630790492, -2.9900407334641947, 4.866034709813673, 13.240443415177351, -65.32006374711894,
		7.519097530386641, 4.2677702235681005, -6.141058597027019, -17.35490306808208, 76.85969451329804,
		-2.5093715705151727, -1.0672777459608007, 2.8193807104661253, 4.303634108914683, -23.980623411974708,
		-0.08057044468725783, -0.06958798038653725, -0.01565678981482277, 0.8309334849162636, -0.6670760099019413,
		-0.00345410133452515, -0.002983276769205442, -0.0006712155903281473, -0.0072550011243097965, 0.9715658703175387,
	})
	B := mat.NewDense(5, 1, []float64{
		0.15249615320336748, -0.9114959283620048, -0.6401443394691997, 0.5690202064446649, 0.9445688168445349,
	})
	poles := []complex128{
		complex(0.4169930697776802, 0.5660576860116876), complex(0.4169930697776802, -0.5660576860116876),
		complex(-0.049884331202635364, 0.5321308975841257), complex(-0.049884331202635364, -0.5321308975841257),
		0.04232921962320497,
	}

	if _, err := Acker(A, B, poles); !errors.Is(err, ErrUncontrollable) {
		t.Fatalf("Acker oracle: err = %v, want ErrUncontrollable", err)
	}
	K, err := Place(A, B, poles)
	if !errors.Is(err, ErrUncontrollable) {
		t.Fatalf("Place: err = %v, want ErrUncontrollable", err)
	}
	if K != nil {
		t.Fatalf("Place: K = %v, want nil", mat.Formatted(K))
	}
}

// HHNPGV follow-up: A = T·diag(λ, λ, …)·T⁻¹ (T random), so λ has a 2-D
// eigenspace and no single input can reach both copies (PBH). The Schur
// path's rounding-level cutoffs let these through to an inaccurate gain on
// both arches, returning ErrPoleAccuracy on base 5a28e97; the orthogonal
// controllability staircase now names the cause.
func TestPlaceSingleInputRepeatedEigenvaluePoleErrorIsUncontrollable(t *testing.T) {
	cases := []struct {
		a, b  []float64
		poles []complex128
	}{
		{[]float64{-1.1100066578086105, -1.2862585453959245e-18, -1.2966562769565667e-15, -1.1100066578086105},
			[]float64{0.5860811670113159, -0.18889404786803454},
			[]complex128{-0.24653458180338841, -0.5231409419426759}},
		{[]float64{-0.9700061947882448, -0.5096429417719341, -0.8507562776025975, 0.8058964591586829, 0.9198019393181583,
			0.4182516812888321, -0.4745254474549538, -0.14752947830864704, 0.4229758963900185},
			[]float64{0.7744099061837675, -0.9266454336321067, -2.592432782466831},
			[]complex128{-0.3386628277364704, -0.312996782508005, -0.578019332586389}},
		{[]float64{-0.20481067418314225, 0.6571342937780202, -0.7473166681635425, -0.06817459085133244, 0.4134444408391492,
			-0.6077699058253904, -0.19140497909245252, 1.5004424621807453, -1.8273398224314152},
			[]float64{-0.4769656336951772, 1.4738675991193686, 0.16037314426374516},
			[]complex128{-0.5369375000458507, -0.17674737141272304, -0.12806484784073283}},
	}
	for ci, tc := range cases {
		n := len(tc.b)
		A := mat.NewDense(n, n, tc.a)
		B := mat.NewDense(n, 1, tc.b)
		if d := pbhDistance(A, B); !(d < 1e-10) {
			t.Fatalf("case %d: PBH oracle: min σ([A−λI, B])/‖[A B]‖ = %g, want rounding level", ci, d)
		}
		K, err := Place(A, B, tc.poles)
		if !errors.Is(err, ErrUncontrollable) {
			t.Errorf("case %d: Place err = %v, want ErrUncontrollable", ci, err)
		}
		if K != nil {
			t.Errorf("case %d: K = %v, want nil", ci, mat.Formatted(K))
		}
	}
}

// pbhDistance returns min over real eigenvalues λ of A of
// σ_min([A − λI, B]) / ‖[A B]‖₂, the PBH controllability measure.
func pbhDistance(A, B *mat.Dense) float64 {
	n, m := B.Dims()
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		return math.NaN()
	}
	ab := mat.NewDense(n, n+m, nil)
	ab.Slice(0, n, 0, n).(*mat.Dense).Copy(A)
	ab.Slice(0, n, n, n+m).(*mat.Dense).Copy(B)
	scale := mat.Norm(ab, 2)
	best := math.Inf(1)
	for _, l := range eig.Values(nil) {
		if imag(l) != 0 {
			continue
		}
		p := mat.DenseCopyOf(ab)
		for i := range n {
			p.Set(i, i, p.At(i, i)-real(l))
		}
		var svd mat.SVD
		if !svd.Factorize(p, mat.SVDNone) {
			return math.NaN()
		}
		sv := svd.Values(nil)
		best = math.Min(best, sv[n-1]/scale)
	}
	return best
}
