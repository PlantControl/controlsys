package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A weakly controllable unstable mode (s2/s1 of B 2e-13) needs ‖X‖ so large
// that eps·|b₁|ᵀ|X||b₁| ≈ 5e9 buries R₁₁ + b₁ᵀXb₁ ≈ 4 in the strong channel:
// float64 X cannot hold it, Newton converges to the Dare with that input
// removed and its step certifies it, X 0.036 and K 0.062 (first case) and
// X 0.046 and K 0.29 (second) off the 320-bit solution. Each literal
// reproduces on one arch only, arm64 then amd64 (AND6B2); the first errors on
// amd64, with and without the check, and an error certifies nothing.
func TestDarePlacedLostChannelNotCertified(t *testing.T) {
	const maxX = 1e-8
	for _, tc := range []struct {
		name          string
		a, b, q, r, e []float64
	}{
		{
			name: "strong channel lost arm64",
			a:    []float64{1.197652873208657, 0.037222031646009776, 1.2192983082486777, 0.6052821305691662},
			b:    []float64{-0.32984715595698716, -0.36884046676840876, 0.52553686790058285, 0.58766389268461072},
			e:    []float64{1.0553186807588606, -0.21382970543431437, -0.2141832612860147, 0.65247344005070285},
			q:    []float64{1.0213596485996295, -0.13613582407552849, -0.13613582407552849, 1.7461404059375563},
			r:    []float64{1.7458126501168427, 1.0218654324542988, 1.0218654324542988, 1.4037727720511952},
		},
		{
			name: "strong channel lost amd64",
			a:    []float64{1.1611818764293993, 0.86722531120390123, 0.69821370808011674, 1.6398898343168307},
			b:    []float64{-0.34464523738961195, 0.3108590065777361, 0.060261569219816666, -0.05435401250377007},
			e:    []float64{0.53601192273672238, -1.0937104517484695, 0.35616191682546128, 1.0855286673206457},
			q:    []float64{0.87851177651913959, -0.60320438076280514, -0.60320438076280514, 0.58045797284646217},
			r:    []float64{1.716560173451851, 1.1702736440637245, 1.1702736440637245, 8.8136381636112056},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			A, B, Q, R, E := mat.NewDense(2, 2, tc.a), mat.NewDense(2, 2, tc.b), mat.NewDense(2, 2, tc.q), mat.NewDense(2, 2, tc.r), mat.NewDense(2, 2, tc.e)
			res, err := acceptUnverified(Dare(A, B, Q, R, &RiccatiOpts{E: E}))
			if err != nil {
				t.Logf("error, certifies nothing: %v", err)
				return
			}
			oracle := descriptorOracle(false, A, B, Q, R, nil, E, res.K, 60)
			prev := descriptorOracle(false, A, B, Q, R, nil, E, res.K, 59)
			if e := relErr(prev, oracle); !(e <= 1e-14) {
				t.Fatalf("oracle not converged: %.3g", e)
			}
			e := relErr(res.X, oracle)
			t.Logf("Accuracy %d, X rel err %.3g", res.accuracy, e)
			if res.accuracy == 0 && !(e <= maxX) {
				t.Errorf("certified X %.3g off the 320-bit solution, want ≤ %g or Accuracy 1", e, maxX)
			}
		})
	}
}

// A channel counts as lost by column of B, with R on its diagonal: b₁ = (1, 1)
// cancels in X, b₂ = (1, −1) does not.
func TestInputChannelLostByColumn(t *testing.T) {
	B := mat.NewDense(2, 2, []float64{1, 1, 1, -1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	for _, tc := range []struct {
		x    float64
		want bool
	}{{1e17, true}, {1e14, false}} {
		X := mat.NewDense(2, 2, []float64{tc.x, -tc.x, -tc.x, tc.x})
		if got := inputChannelLost(B, R, X); got != tc.want {
			t.Errorf("x=%g: lost %v, want %v", tc.x, got, tc.want)
		}
	}
	X := mat.NewDense(2, 2, []float64{1e17, 1e17, 1e17, 1e17})
	if !inputChannelLost(mat.NewDense(2, 2, []float64{2, 1, 2, -1}), R, X) {
		t.Error("second column cancelling: not lost")
	}
	if inputChannelLost(mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 1, []float64{1}), X) {
		t.Error("no cancellation: lost")
	}
}
