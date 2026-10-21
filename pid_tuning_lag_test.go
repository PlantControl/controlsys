package controlsys

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 5LB3LS (QNALQV): a first-order discrete plant with its pole at 3.7e-44
// and a 20-sample input delay. Its absorbed 21-state model's
// TransferFunction collapsed to 0/(z−3e-33), so the ideal-derivative
// screen tested den(c)·(z−a) alone and certified the forward-Euler PD
// TunePID returned (Kd/Dt ≈ 0.097, Tf = 0), which has 19 roots outside
// the unit circle. The screen now tests the undelayed plant's
// characteristic shifted by z^20, and TunePID PD reports the target
// unattainable: with |G| ≈ 58 a gain crossover of Kp + Kd(z−1)/Dt puts
// |L| > 1 at high frequency against the 20-sample delay. Oracle: exact
// Schur–Cohn on (D·z + CB)/((z−a)·z^20).
func TestPIDStabilityScreenIdealDelayAbsorbed(t *testing.T) {
	const a, b, c, d = 3.720075976020836e-44, -0.9667981925794609, -0.8568887533689471, 57.7675665396686
	const lag = 20
	num, den := Poly{d, c*b - d*a}, Poly{1, -a}
	for _, dt := range []float64{100, 1, 0.1} {
		p, err := New(mat.NewDense(1, 1, []float64{a}), mat.NewDense(1, 1, []float64{b}), mat.NewDense(1, 1, []float64{c}), mat.NewDense(1, 1, []float64{d}), dt)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.SetInputDelay([]float64{lag}); err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(p, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range []PID2{
			{Kp: 0.008945254131813402, Kd: 0.9728762924556705 * dt / 10},
			{Kp: 0.008945254131813402 / 100, Kd: 0.9728762924556705 * dt / 1e3},
			{Kp: 0.001, Ki: 1e-6 / dt, Kd: 1e-5 * dt},
		} {
			k.B, k.C, k.Dt = 1, 1, dt
			want := pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag)
			if got, err := screen.closedLoopPoles(&k); err != nil || got != want {
				t.Errorf("dt %g %+v: screen stable %v %v, oracle %v", dt, k, got, err, want)
			}
			if got, ok := pidTuningIdealExact(screen.rational, &k, screen.lag); !ok || got != want {
				t.Errorf("dt %g %+v: exact stable %v %v, oracle %v", dt, k, got, ok, want)
			}
		}
		for _, f := range []PidtuneType{PidtunePD, PidtunePID} {
			name := fmt.Sprintf("dt %g %s", dt, f)
			if k, err := Pidtune(p, f, 0, nil); err == nil && k.Kd != 0 && k.Tf == 0 && !pidtuneExactSchurStableLag(t, k, num, den, dt, 1, lag) {
				t.Errorf("%s: Pidtune %+v unstable", name, *k)
			}
			r, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
			if f == PidtunePD && errors.Is(err, ErrPIDTuningTargetUnattainable) {
				continue
			}
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			k := r.Controller
			if r.Evidence.Stability == PIDStabilitySampledOnly {
				continue
			}
			if !pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}, num, den, dt, 1, lag) {
				t.Errorf("%s: %+v certified %s, unstable", name, *k, r.Evidence.Stability)
			}
		}
	}
}

// 5LB3LS (IP3XIC k=1): TunePID returned Kp 105572.5, Ki −38.475,
// Kd 483.4 (Tf = 0) on this near-unit-pole plant and certified it by
// closed-loop poles; a negative Ki on a plant of DC gain +0.208 puts a real
// root above z = 1. Fixed by the exact retest of 632b44a; pinned here.
// Oracle: exact Schur–Cohn.
func TestPIDStabilityScreenIdealIP3XICUnit(t *testing.T) {
	const dt = 0.01
	num := Poly{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}
	den := Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}
	screen, err := newPIDStabilityScreen(pidtuneBackoffPlant(t, num, den, dt, 0), nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	k := PID2{Kp: 105572.5, Ki: -38.475, Kd: 483.4, B: 1, C: 1, Dt: dt}
	if pidtuneExactSchurStable(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt) {
		t.Fatal("oracle stable")
	}
	if got, err := screen.closedLoopPoles(&k); err != nil || got {
		t.Errorf("screen stable %v %v, oracle unstable", got, err)
	}
}

// 5LB3LS: delayed discrete plants with near-zero poles. The undelayed
// plant's TransferFunction drops or corrupts the tiny-pole terms (#3089:
// 0.0551(z+2e-30)/(z+2e-30) for a degree-2 plant), so the float screen on
// it certified unstable PD/PID designs as closed-loop-poles stable and
// rejected stable ones (#3018). Oracle: exact Schur–Cohn with z^lag.
func TestPIDStabilityScreenIdealDelayTinyPoles(t *testing.T) {
	for _, tc := range []struct {
		name     string
		dt       float64
		lag      int
		num, den Poly
		f        PidtuneType
		pm       float64
		stable   *PID2
	}{
		{"3089", 0.1, 15, Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}, PidtunePID, 45, nil},
		{"3036", 0.1, 35, Poly{-0.014784959385975079, -1.8019347038001476, 0.11477038906105119, -1.8748869417529457}, Poly{1, -5.00447263254577e-10, 1.5650841538727049e-31, -1.0087484683502245e-60}, PidtunePID, 45, nil},
		{"3078", 0.1, 49, Poly{0.09812833503587619, 1.650140140621783, 0.1979807726277072, 1.2158999453157313}, Poly{1, -9.011102747110696e-30, -1.5166629168579831e-64, -8.978278833426846e-102}, PidtunePD, 45, nil},
		{"3018", 1, 11, Poly{0.263564432656593, -0.2987087982355384, 0.7955358290524017, 0.16793332407135853}, Poly{1, -0.9332191934662787, 1.974877418821016e-13, 1.6307831686022736e-31}, PidtunePD, 0, &PID2{Kp: 0.12300171178088162, Kd: 0.11028146728202622}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := pidtuneBackoffPlant(t, tc.num, tc.den, tc.dt, float64(tc.lag))
			if k := tc.stable; k != nil {
				k.B, k.C, k.Dt = 1, 1, tc.dt
				if !pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Kd: k.Kd, Dt: tc.dt}, tc.num, tc.den, tc.dt, 1, tc.lag) {
					t.Fatal("oracle unstable")
				}
				screen, err := newPIDStabilityScreen(p, nil, 1)
				if err != nil {
					t.Fatal(err)
				}
				if got, err := screen.closedLoopPoles(k); err != nil || !got {
					t.Errorf("screen stable %v %v, oracle stable", got, err)
				}
			}
			r, err := TunePID(context.Background(), p, tc.f, PIDTuningOptions{PhaseMargin: tc.pm})
			if errors.Is(err, ErrPIDTuningTargetUnattainable) {
				if tc.stable != nil {
					t.Errorf("unattainable, stable %+v exists", *tc.stable)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			k := r.Controller
			if r.Evidence.Stability == PIDStabilitySampledOnly {
				return
			}
			if !pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}, tc.num, tc.den, tc.dt, 1, tc.lag) {
				t.Errorf("%+v certified %s, unstable", *k, r.Evidence.Stability)
			}
		})
	}
}

// 5LB3LS #3089: the collapsed transfer function TransferFunction returned
// for the undelayed degree-2 plant, 0.0551(z+2e-30)/(z+2e-30), made the
// float screen certify these gains, whose loop has a root at |z| ≈ 1.17.
func TestPIDTuningIdealStableCollapsedTransfer(t *testing.T) {
	const dt, lag = 0.1, 15
	num, den := Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}
	plant := pidtuneBackoffPlant(t, num, den, dt, 0)
	transfer := &TransferFuncResult{TF: &TransferFunc{Num: [][][]float64{{{0.05508874359571571, 1.1e-31}}}, Den: [][]float64{{1, 2.0625805448700525e-30}}, Dt: dt}}
	k := PID2{Kp: 0.5706220540940695, Ki: 1.0390660705947083, Kd: 0.3000376123041945, B: 1, C: 1, Dt: dt}
	if pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag) {
		t.Fatal("oracle stable")
	}
	if got, err := pidTuningIdealStable(plant, func() (*TransferFuncResult, error) { return transfer, nil }, &k, lag); err != nil || got {
		t.Errorf("stable %v %v, oracle unstable", got, err)
	}
}

// H2RXWD: at lag 70, n+lag exceeds pidTuningExactLagOrder, and the screen
// tested the collapsed transfer of TestPIDTuningIdealStableCollapsedTransfer
// in float, certifying loops with roots outside the unit circle. The exact
// characteristic is now built at any lag, its float verdict deciding; the
// transfer function is not consulted, so its failure does not matter.
// Oracle: exact Schur–Cohn with z^70.
func TestPIDTuningIdealStableCollapsedTransferLongLag(t *testing.T) {
	const dt, lag = 0.1, 70
	num, den := Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}
	plant := pidtuneBackoffPlant(t, num, den, dt, 0)
	collapsed := &TransferFuncResult{TF: &TransferFunc{Num: [][][]float64{{{0.05508874359571571, 1.1e-31}}}, Den: [][]float64{{1, 2.0625805448700525e-30}}, Dt: dt}}
	for _, tc := range []struct {
		k    PID2
		want bool
	}{
		{PID2{Kp: 0.5706220540940695, Ki: 1.0390660705947083, Kd: 0.3000376123041945}, false},
		{PID2{Kp: 0.02, Ki: 0.01, Kd: 0.005}, true},
	} {
		k := tc.k
		k.B, k.C, k.Dt = 1, 1, dt
		if pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag) != tc.want {
			t.Fatalf("%+v: oracle not %v", k, tc.want)
		}
		for name, transfer := range map[string]func() (*TransferFuncResult, error){
			"collapsed": func() (*TransferFuncResult, error) { return collapsed, nil },
			"failed":    func() (*TransferFuncResult, error) { return nil, ErrOverflow },
		} {
			if got, err := pidTuningIdealStable(plant, transfer, &k, lag); err != nil || got != tc.want {
				t.Errorf("%s %+v: stable %v %v, oracle %v", name, k, got, err, tc.want)
			}
		}
	}
}

// H2RXWD: above pidTuningExactOrder the screen took the float verdict on
// the transfer's characteristic, so the collapsed transfer of
// TestPIDTuningIdealStableCollapsedTransfer, in series with 16 stable
// pole-zero pairs (n = 18), certified unstable loops, and a failed transfer
// aborted. A transfer of lower degree than n, or a failure, now leaves the
// exact characteristic to decide. Oracle: exact Schur–Cohn with z^15.
func TestPIDTuningIdealStableCollapsedTransferHighOrder(t *testing.T) {
	const dt, lag = 0.1, 15
	numG, denG := Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}
	numH, denH := Poly{1}, Poly{1}
	for i := range 16 {
		b := 0.1 + 0.03*float64(i)
		numH, denH = numH.Mul(Poly{1, -(b + 0.005)}), denH.Mul(Poly{1, -b})
	}
	num, den := numG.Mul(numH), denG.Mul(denH)
	plant := pidtuneBackoffPlant(t, num, den, dt, 0)
	if n, _, _ := plant.Dims(); n <= pidTuningExactOrder {
		t.Fatalf("order %d within pidTuningExactOrder", n)
	}
	collapsed := &TransferFuncResult{TF: &TransferFunc{Num: [][][]float64{{Poly{0.05508874359571571, 1.1e-31}.Mul(numH)}}, Den: [][]float64{Poly{1, 2.0625805448700525e-30}.Mul(denH)}, Dt: dt}}
	for _, tc := range []struct {
		k    PID2
		want bool
	}{
		{PID2{Kp: 0.5706220540940695, Ki: 1.0390660705947083, Kd: 0.3000376123041945}, false},
		{PID2{Kp: 0.02, Ki: 0.01, Kd: 0.005}, true},
	} {
		k := tc.k
		k.B, k.C, k.Dt = 1, 1, dt
		if pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag) != tc.want {
			t.Fatalf("%+v: oracle not %v", k, tc.want)
		}
		for name, transfer := range map[string]func() (*TransferFuncResult, error){
			"collapsed": func() (*TransferFuncResult, error) { return collapsed, nil },
			"failed":    func() (*TransferFuncResult, error) { return nil, ErrOverflow },
		} {
			if got, err := pidTuningIdealStable(plant, transfer, &k, lag); err != nil || got != tc.want {
				t.Errorf("%s %+v: stable %v %v, oracle %v", name, k, got, err, tc.want)
			}
		}
	}
}

// H2RXWD: closedLoopPoles extracted the transfer function before the ideal
// test and returned its error, even where the exact characteristic decides.
// Oracles: exact Schur–Cohn with z^15 for the delayed plant; for the
// undelayed one, the slow-root analysis of TestPIDTuningIdealStableBand.
func TestPIDStabilityScreenIdealTransferError(t *testing.T) {
	const dt, lag = 0.1, 15
	num, den := Poly{0.05508874359571571, 1.4671058437019866, -0.6931435149725129}, Poly{1, 2.0625805448700525e-30, 4.842253744153856e-61}
	delayed := pidtuneBackoffPlant(t, num, den, dt, lag)
	for _, k := range []PID2{
		{Kp: 0.5706220540940695, Ki: 1.0390660705947083, Kd: 0.3000376123041945, B: 1, C: 1, Dt: dt},
		{Kp: 0.005, Ki: 0.001, Kd: 0.003, B: 1, C: 1, Dt: dt},
	} {
		want := pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag)
		screen, err := newPIDStabilityScreen(delayed, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		screen.transferErr = ErrOverflow
		if got, err := screen.closedLoopPoles(&k); err != nil || got != want {
			t.Errorf("lag %d %+v: stable %v %v, oracle %v", lag, k, got, err, want)
		}
	}
	c := &PID2{Kp: 7.954440137905823, Ki: 121.18255736538495, Kd: 1.2450662060726023, B: 1, C: 1}
	for _, tc := range []struct {
		z    float64
		want bool
	}{{1e-10, false}, {1e-8, true}} {
		screen, err := newPIDStabilityScreen(makePlant(t, Poly{0.8296, 0.8296 * tc.z}, Poly{1, 0.999, 0.1257}), nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		screen.transferErr = ErrOverflow
		if got, err := screen.closedLoopPoles(c); err != nil || got != tc.want {
			t.Errorf("z %g: stable %v %v, want %v", tc.z, got, err, tc.want)
		}
	}
}
