package controlsys

import (
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// RBVVVP: improper small-lead TF realizations, their series with a proper
// plant and their feedback loops keep E's exact zero rows. Where the
// Weierstrass reduction refutes its own improper split (25 of the 72
// realizations and the repro, 11 series) IsProper errored; it now decides properness
// exactly from the data. Series G·P with P of relative degree 1 is proper
// iff the quotient degree q is 1; the feedback loops are all proper.
func TestIsProper_ImproperTFSmallLeadExactE(t *testing.T) {
	repro := &TransferFunc{
		Num: [][][]float64{{{-0.2347143836851664, 0.6471986071804068, -0.6781041698782371, -0.18908743177153076}}},
		Den: [][]float64{{1e-4, -0.2284209051246615}},
	}
	cases := append(smallLeadImproperTFs(), smallLeadTF{"repro", repro})
	for _, c := range cases {
		res, err := c.tf.StateSpace()
		if err != nil {
			t.Fatalf("%s: %v", c.label, err)
		}
		if pr, err := res.Sys.IsProper(); err != nil || pr {
			t.Errorf("%s: IsProper = %v, %v; want false", c.label, pr, err)
		}
		p := len(c.tf.Den)
		P := smallLeadPlant(t, p)
		ser, err := Series(P, res.Sys)
		if err != nil {
			t.Fatal(err)
		}
		want := p == 1 && len(c.tf.Num[0][0])-len(c.tf.Den[0]) == 1
		if pr, err := ser.IsProper(); err != nil || pr != want {
			t.Errorf("%s: Series IsProper = %v, %v; want %v", c.label, pr, err, want)
		}
		fb, err := Feedback(P, res.Sys, -1)
		if err != nil {
			t.Fatal(err)
		}
		if pr, err := fb.IsProper(); err != nil || !pr {
			t.Errorf("%s: Feedback IsProper = %v, %v; want true", c.label, pr, err)
		}
	}
}

// RBVVVP: exactProperness against structured descriptors proper or improper
// by construction (exact zero rows in E), and an identically singular pencil.
func TestExactProperness_StructuredDescriptors(t *testing.T) {
	for seed := range uint64(40) {
		for _, chains := range [][]int{{2}, {3}, {5}, {2, 3}} {
			for _, proper := range []bool{true, false} {
				rng := rand.New(rand.NewPCG(seed, 77))
				_, o, _ := chainDescriptor(t, rng, 1+rng.IntN(3), chains, 0, proper)
				if got, ok := exactProperness(o); !ok || got != proper {
					t.Fatalf("seed %d chains %v: exactProperness = %v, %v; want %v", seed, chains, got, ok, proper)
				}
			}
		}
	}
	for seed := range uint64(20) {
		_, o := refutedChainDescriptor(t, seed)
		if got, ok := exactProperness(o); !ok || !got {
			t.Fatalf("refuted chain seed %d: exactProperness = %v, %v; want true", seed, got, ok)
		}
	}
	s, err := NewDescriptor(mat.NewDense(2, 2, []float64{0, -1, 1, 0}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{0, 1}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := exactProperness(s); !ok || got {
		t.Fatalf("s: exactProperness = %v, %v; want false", got, ok)
	}
	sing, err := NewDescriptor(mat.NewDense(2, 2, []float64{1, 2, 0, 0}), mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := exactProperness(sing); ok {
		t.Fatal("singular pencil: exactProperness decided")
	}
}
