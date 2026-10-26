package controlsys

import (
	"errors"
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
