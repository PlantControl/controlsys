package controlsys

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func feedbackChannelPlant(t *testing.T, dt float64, delayed bool) *System {
	t.Helper()
	A := []float64{-1, 2, 0.5, 0.3, -3, 1, -0.7, 0.2, -2}
	if dt > 0 {
		A = []float64{0.5, 0.2, -0.1, 0.1, 0.7, 0.3, -0.2, 0.05, 0.4}
	}
	P, err := New(mat.NewDense(3, 3, A),
		mat.NewDense(3, 3, []float64{1, 0.3, 0, 0, 1, -0.5, 0.2, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0.4, 0, 1, 0.2, -0.3, 0.6, 1}),
		mat.NewDense(3, 3, []float64{0.2, 0.1, 0, 0, 0.3, 0.1, 0.05, 0, -0.1}), dt)
	if err != nil {
		t.Fatal(err)
	}
	P.InputName = []string{"u1", "u2", "u3"}
	P.OutputName = []string{"y1", "y2", "y3"}
	P.StateName = []string{"p1", "p2", "p3"}
	if delayed {
		sc := 1.0
		if dt > 0 {
			sc = 10
		}
		P.InputDelay = []float64{0.1 * sc, 0, 0.2 * sc}
		P.OutputDelay = []float64{0, 0.3 * sc, 0}
	}
	return P
}

func feedbackChannelController(t *testing.T, dt float64, delayed bool) *System {
	t.Helper()
	a := -2.0
	if dt > 0 {
		a = 0.3
	}
	K, err := New(mat.NewDense(1, 1, []float64{a}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{0.3, -0.2}), mat.NewDense(2, 1, []float64{0.5, 0.1}), dt)
	if err != nil {
		t.Fatal(err)
	}
	K.StateName = []string{"k1"}
	if delayed {
		sc := 1.0
		if dt > 0 {
			sc = 10
		}
		K.InputDelay = []float64{0.1 * sc}
		K.OutputDelay = []float64{0, 0.2 * sc}
	}
	return K
}

// channelLoopOracle evaluates (I - sign*G*Kf)^{-1} G, where Kf places
// controller output a on plant input in[a] and reads plant output out[b]
// into controller input b. A nil controller is the identity.
func channelLoopOracle(t *testing.T, plant, ctrl *System, sign float64, in, out []int, s complex128) [][]complex128 {
	t.Helper()
	f := exactDelayFactor(plant.Dt)
	G := evalDelaySystem(t, plant, s, f)
	_, m, p := plant.Dims()
	K := make([][]complex128, len(in))
	for a := range in {
		K[a] = make([]complex128, len(out))
		if ctrl == nil {
			K[a][a] = 1
		}
	}
	if ctrl != nil {
		K = evalDelaySystem(t, ctrl, s, f)
	}
	Kf := make([][]complex128, m)
	for i := range Kf {
		Kf[i] = make([]complex128, p)
	}
	for a, i := range in {
		for b, j := range out {
			Kf[i][j] = K[a][b]
		}
	}
	GK := cmul(G, Kf)
	loop := make([][]complex128, p)
	for i := range loop {
		loop[i] = make([]complex128, p)
		for j := range loop[i] {
			loop[i][j] = -complex(sign, 0) * GK[i][j]
		}
		loop[i][i] += 1
	}
	return csolve(t, loop, G)
}

func TestFeedbackChannelsMatchOracle(t *testing.T) {
	in, out := []int{2, 0}, []int{1}
	for _, dt := range []float64{0, 0.1} {
		for _, delayed := range []bool{false, true} {
			for _, sign := range []float64{-1, 1} {
				label := fmt.Sprintf("dt=%v delayed=%v sign=%v", dt, delayed, sign)
				P := feedbackChannelPlant(t, dt, delayed)
				K := feedbackChannelController(t, dt, delayed)
				cl, err := Feedback(P, K, sign, FeedbackChannels{FeedIn: in, FeedOut: out})
				if err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				if n, m, p := cl.Dims(); m != 3 || p != 3 || n != 4 {
					t.Fatalf("%s: dims n=%d m=%d p=%d, want 4,3,3", label, n, m, p)
				}
				assertResponseOracle(t, label, cl, func(s complex128) [][]complex128 {
					return channelLoopOracle(t, P, K, sign, in, out, s)
				})
				if !slices.Equal(cl.InputName, P.InputName) || !slices.Equal(cl.OutputName, P.OutputName) {
					t.Errorf("%s: names in=%v out=%v, want plant names", label, cl.InputName, cl.OutputName)
				}
				if !slices.Equal(cl.StateName, []string{"p1", "p2", "p3", "k1"}) {
					t.Errorf("%s: StateName = %v, want plant then controller", label, cl.StateName)
				}
			}
		}
	}
}

func TestFeedbackChannelsNilControllerIsUnitGain(t *testing.T) {
	in, out := []int{1, 2}, []int{0, 2}
	for _, dt := range []float64{0, 0.1} {
		P := feedbackChannelPlant(t, dt, true)
		cl, err := Feedback(P, nil, -1, FeedbackChannels{FeedIn: in, FeedOut: out})
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		assertResponseOracle(t, fmt.Sprintf("dt=%v", dt), cl, func(s complex128) [][]complex128 {
			return channelLoopOracle(t, P, nil, -1, in, out, s)
		})
	}
}

func TestFeedbackChannelsAllChannelsEqualsPlainFeedback(t *testing.T) {
	P := feedbackDelayPlant(t, 0.1, "mixed")
	K := feedbackDelayController(t, 0.1, "in")
	cl, err := Feedback(P, K, -1, FeedbackChannels{FeedIn: []int{0, 1}, FeedOut: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	assertResponseOracle(t, "identity channels", cl, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, P, K, -1, s)
	})
}

func TestFeedbackChannelsErrors(t *testing.T) {
	P := feedbackChannelPlant(t, 0, false)
	K := feedbackChannelController(t, 0, false)
	Kd := feedbackChannelController(t, 0.1, false)
	cases := []struct {
		name string
		ctrl *System
		ch   []FeedbackChannels
		want error
	}{
		{"two specs", K, []FeedbackChannels{{FeedIn: []int{0, 1}, FeedOut: []int{0}}, {FeedIn: []int{0, 1}, FeedOut: []int{0}}}, ErrInvalidArgument},
		{"empty feedin", K, []FeedbackChannels{{FeedOut: []int{0}}}, ErrInvalidArgument},
		{"empty feedout", K, []FeedbackChannels{{FeedIn: []int{0, 1}}}, ErrInvalidArgument},
		{"feedin out of range", K, []FeedbackChannels{{FeedIn: []int{0, 3}, FeedOut: []int{0}}}, ErrInvalidArgument},
		{"negative feedout", K, []FeedbackChannels{{FeedIn: []int{0, 1}, FeedOut: []int{-1}}}, ErrInvalidArgument},
		{"repeated feedin", K, []FeedbackChannels{{FeedIn: []int{1, 1}, FeedOut: []int{0}}}, ErrInvalidArgument},
		{"controller size", K, []FeedbackChannels{{FeedIn: []int{0}, FeedOut: []int{0}}}, ErrDimensionMismatch},
		{"nil controller not square", nil, []FeedbackChannels{{FeedIn: []int{0, 1}, FeedOut: []int{0}}}, ErrDimensionMismatch},
		{"domain", Kd, []FeedbackChannels{{FeedIn: []int{0, 1}, FeedOut: []int{0}}}, ErrDomainMismatch},
	}
	for _, tc := range cases {
		if _, err := Feedback(P, tc.ctrl, -1, tc.ch...); !errors.Is(err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, err, tc.want)
		}
	}
}

// MATLAB feedback has no delay-approximation option: approximating the exact
// closed loop with pade/absorbDelay is the supported workflow, and equals
// approximating the plant and controller before closing the loop.
func TestFeedbackThenApproximateDelays(t *testing.T) {
	for _, pk := range []string{"in", "out", "iod", "mixed", "lft"} {
		P := feedbackDelayPlant(t, 0.1, pk)
		K := feedbackDelayController(t, 0.1, "in")
		cl, err := Feedback(P, K, -1)
		if err != nil {
			t.Fatal(err)
		}
		abs, err := cl.AbsorbDelay()
		if err != nil {
			t.Fatalf("discrete %s: %v", pk, err)
		}
		if abs.HasDelay() {
			t.Fatalf("discrete %s: AbsorbDelay left delays", pk)
		}
		assertResponseOracle(t, "discrete "+pk, abs, func(s complex128) [][]complex128 {
			return closedLoopOracle(t, P, K, -1, s)
		})

		Pc := feedbackDelayPlant(t, 0, pk)
		Kc := feedbackDelayController(t, 0, "in")
		clc, err := Feedback(Pc, Kc, -1)
		if err != nil {
			t.Fatal(err)
		}
		pade, err := clc.Pade(3)
		if err != nil {
			t.Fatalf("continuous %s: %v", pk, err)
		}
		Pp, err := Pc.Pade(3)
		if err != nil {
			t.Fatal(err)
		}
		Kp, err := Kc.Pade(3)
		if err != nil {
			t.Fatal(err)
		}
		assertResponseOracle(t, "continuous "+pk, pade, func(s complex128) [][]complex128 {
			return closedLoopOracle(t, Pp, Kp, -1, s)
		})
	}
}
