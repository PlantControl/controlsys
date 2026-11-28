package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func requireFiniteOrNoResponse(t *testing.T, api string, resp *TimeResponse, err error) {
	t.Helper()
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		if resp != nil {
			t.Errorf("%s: error %v came with a response", api, err)
		}
		return
	}
	if resp == nil || resp.Y == nil {
		t.Fatalf("%s: err %v without a response", api, err)
	}
	r, c := resp.Y.Dims()
	for i := range r {
		for k := range c {
			if v := resp.Y.At(i, k); math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s: sample (%d, %d) = %g with err %v", api, i, k, v, err)
			}
		}
	}
}

// responseValueErr is valueErr that also passes ErrOverflow: the stored
// model's response leaves float64 range (an unstable stiff mode).
func responseValueErr(err error) error {
	if errors.Is(err, ErrOverflow) {
		return nil
	}
	return valueErr(err)
}

func requireOverflow(t *testing.T, api string, resp *TimeResponse, err error) {
	t.Helper()
	if !errors.Is(err, ErrOverflow) {
		t.Fatalf("%s: err = %v, want ErrOverflow", api, err)
	}
	if resp != nil {
		t.Fatalf("%s: response returned with ErrOverflow", api)
	}
}

// Ordinary unstable explicit model: the automatic horizon (from the slow
// pole) and a long Lsim grid carry the unstable mode beyond float64 range.
func TestTimeResponseOverflowUnstableExplicit(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-0.005, 0.3, 0, 3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Step(sys, 0)
	requireOverflow(t, "Step", resp, err)
	resp, err = Impulse(sys, 0)
	requireOverflow(t, "Impulse", resp, err)
	resp, err = Initial(sys, mat.NewVecDense(2, []float64{1, -1}), 0)
	requireOverflow(t, "Initial", resp, err)
	tt := makeTimeVector(401, 1)
	u := mat.NewDense(len(tt), 1, nil)
	for k := range tt {
		u.Set(k, 0, 1)
	}
	resp, err = Lsim(sys, u, tt, nil)
	requireOverflow(t, "Lsim", resp, err)

	resp, err = Step(sys, 5)
	requireFiniteOrNoResponse(t, "Step tFinal=5", resp, err)
	if err != nil {
		t.Fatalf("Step tFinal=5: %v", err)
	}
}

// Census row desc/201 (5EKNHC): E singular within the rank floor, the
// reduced model keeps unstable poles 0.0054 and 2.58 rad/s; the automatic
// horizon of 1309 s overflows from about 275 s.
func TestTimeResponseOverflowSlowUnstableDescriptor(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(4, 4, []float64{0.30531923614101975, 0.5893103783680249, -0.7237386800118472, 0.4980908890943772, -0.6666832788317991, 0.8831214977180099, -0.7484070048945094, 1.8277980199835273, -0.5473678931105155, -0.1387701688126084, -1.401576730159471, -1.7330186849089708, 0.10366324071425707, -0.14813287506702427, 0.5302419639512178, 0.18315084124582867}),
		mat.NewDense(4, 1, []float64{-0.9743926569825044, 1.911406753652208, -0.283478613863633, 1.0041763300948245}),
		mat.NewDense(1, 4, []float64{1.7283752819217106, 0.11424472155635353, 1.6403029671956275, 0.051240803064873434}),
		mat.NewDense(1, 1, []float64{0.35644397692457397}),
		mat.NewDense(4, 4, []float64{0.06989853715238017, -0.05216100459747966, -0.11320073375696321, 0.2883878414371537, 0.43624312034277163, -0.8869516647809851, -0.43524462087751126, 0.6148241628269207, 0.03762325120255357, -0.6876229063779306, -0.0267931057749995, -0.5453316614406385, 0.4010035667638762, 0.4684066560743674, 0.36398363113570514, -0.09008158842834013}),
		0)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Step(sys, 0)
	requireOverflow(t, "Step", resp, err)
	resp, err = Impulse(sys, 0)
	requireOverflow(t, "Impulse", resp, err)
}

// Census rows stfr/714 (literal regenerated on arm64) and stfr/243 (on
// amd64; 5EKNHC): E singular within the rank floor; where the reduction
// keeps the stiff unstable mode (about 1e14 rad/s) every sample after t = 0
// is NaN. Which arches keep it is not asserted: a nil or unverified error
// must come with finite samples, any other error without a response.
func TestTimeResponseOverflowStiffUnstableMode(t *testing.T) {
	for _, c := range []struct {
		name          string
		a, b, c, d, e *mat.Dense
	}{
		{"stfr/714",
			mat.NewDense(3, 3, []float64{-0.6806284625252985, -0.37410853321184073, 0.9152193673851281, -0.1840589232166251, 0.3383235068858649, 0.5889930082078766, 0.45666005304728235, -0.9405898512091189, -0.4430204489055137}),
			mat.NewDense(3, 1, []float64{0.4783591667295819, -1.090219982199198, -0.5154525941854181}),
			mat.NewDense(1, 3, []float64{0.8337673074717253, -0.4530905746489745, 0.6485233754204445}),
			mat.NewDense(1, 1, []float64{-0.48954030417559186}),
			mat.NewDense(3, 3, []float64{0.24370082547719854, 0.6252340402655436, 0.47294023117098943, -0.8703763348502936, -0.17372501746087166, 0.16849535734882953, 0.11069477398056876, -0.04316891887818562, -0.08030045785457147})},
		{"stfr/243",
			mat.NewDense(4, 4, []float64{0.8667121914531961, -0.03207805341177322, 1.1495581128710775, -0.07892276580642732, -0.06753656522391988, -0.708683226096992, 0.16126974754646847, -1.0785763439184521, -0.5166832697169743, 0.0006026642409087124, -0.46951474150483286, 1.2901046865809624, -0.4698328416905223, -0.1799775265077841, 0.47448423404256634, -0.14049236805976995}),
			mat.NewDense(4, 1, []float64{0.5260059128759885, -0.8538969561955401, 0.25061577456056644, 0.5510530873322523}),
			mat.NewDense(1, 4, []float64{0.6313119978632703, -0.5007033626400759, -1.6068521121279586, 0.44806059736590576}),
			mat.NewDense(1, 1, []float64{0.31194600392720956}),
			mat.NewDense(4, 4, []float64{0.21149668066626234, -0.25708219265407173, -0.15575991458749638, -0.05468605141385357, -0.004020840544857761, 0.37802083139679177, 0.13451648414194256, 0.026066503497004415, -0.44426197173794024, 0.5241798857749524, -0.5265031019806824, -0.2851375482823841, -0.2638521468937871, -0.3936863472953933, 0.2556820051961284, 0.1676555199927509})},
	} {
		sys, err := NewDescriptor(c.a, c.b, c.c, c.d, c.e, 0)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := Step(sys, 0)
		requireFiniteOrNoResponse(t, c.name+" Step", resp, err)
		resp, err = Impulse(sys, 0)
		requireFiniteOrNoResponse(t, c.name+" Impulse", resp, err)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			tt := makeTimeVector(201, dt)
			u := mat.NewDense(len(tt), 1, nil)
			for k := range tt {
				u.Set(k, 0, math.Sin(1.3*tt[k]/(dt*20))+0.5)
			}
			resp, err = Lsim(sys, u, tt, nil)
			requireFiniteOrNoResponse(t, c.name+" Lsim", resp, err)
		}
	}
}

// ModelArray.Step and StepInfoForSystem pass a model's ErrOverflow on
// without a value; StepInfoForSystem reaches Step only past its stability
// gate, so the unstable model stops there with ErrUnstable.
func TestTimeResponseOverflowCallers(t *testing.T) {
	unstable, err := New(
		mat.NewDense(2, 2, []float64{-0.005, 0.3, 0, 3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	arr, err := NewModelArray([]int{2}, []*System{stable, unstable})
	if err != nil {
		t.Fatal(err)
	}
	if out, err := arr.Step(0); !errors.Is(err, ErrOverflow) || out != nil {
		t.Errorf("ModelArray.Step = %v, %v; want nil, ErrOverflow", out, err)
	}
	if info, err := StepInfoForSystem(unstable, 0, nil); !errors.Is(err, ErrUnstable) || info != nil {
		t.Errorf("StepInfoForSystem = %v, %v; want nil, ErrUnstable", info, err)
	}
}
