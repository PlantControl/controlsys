package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDescriptorIndexOneStableNoRoundingPole: a stable index-1 pencil
// (structured poles -2593.67, -1.103, -0.682) whose QZ leaves a
// rounding-born infinite eigenvalue near +4e9 on amd64. Poles must not
// keep it and IsStable must stay true (w23 7TQIAU b171669 regression).
func TestDescriptorIndexOneStableNoRoundingPole(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		-0.029600478280985564, -0.040905845100694394, -0.035062821899181314, 0.12188216889236919,
		-0.04826048563228318, -0.06707192430646268, -0.057219746133352754, 0.19993808762409027,
		0.015817601940601157, 0.021519395633665804, 0.018719037701902187, -0.06397153299453241,
		-0.03320636099633181, -0.04514549486738244, -0.039386490677616785, 0.13399804758085299,
	})
	E := mat.NewDense(4, 4, []float64{
		0.0020392861058072514, 0.0016055261667342058, 0.001860445777394094, -0.005351090861532362,
		0.003758762022830788, 0.005125335579950575, 0.003464245910524407, -0.01739746895290442,
		-0.0007856095750793041, 0.0007961738432276134, -0.0006938150639150325, -0.002859483501673768,
		0.0018958103432696986, 8.469870710556271e-7, 0.0017055405084022927, 0.0002145828534767802,
	})
	B := mat.NewDense(4, 1, []float64{0.6670682934324924, 1.0464131264270067, -0.3832789254483039, 0.7767107760823396})
	C := mat.NewDense(1, 4, []float64{0.002125641106288566, -0.08427009554416104, 0.01470999604365903, 0.3231430923333386})
	D := mat.NewDense(1, 1, []float64{0.3})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range poles {
		if real(p) > 0 {
			t.Fatalf("Poles = %v: spurious unstable pole %v", poles, p)
		}
	}
	if st, err := sys.IsStable(); err != nil || !st {
		t.Fatalf("IsStable = %v, %v; want true", st, err)
	}
}
