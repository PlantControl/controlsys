package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// checkDiscreteDescriptorKalman runs Kalman (scaled and unscaled), the
// Dare of its dual with E, and Lqg's L on dc, and fails on an error or
// on an Accuracy 0 pair (X, K) off the 320-bit oracle: K by more than
// descriptorGainTol, the generalized X by more than newtonTol.
func checkDiscreteDescriptorKalman(t *testing.T, dc descriptorRefinedCase) {
	t.Helper()
	sys := dc.system(t)
	kd := kalmanDual(sys, sys.B, sys.D)
	Im, Ip := eyeDense(dc.m), eyeDense(dc.p)
	check := func(label string, pr nearSingularRiccati, acc int, X, K *mat.Dense) {
		t.Helper()
		_, Xo, Ko, ok := nearSingularOracle(false, pr.A, pr.B, pr.Q, pr.R, pr.S, pr.E, K)
		if !ok {
			t.Fatalf("%s: oracle did not converge", label)
		}
		if acc != 0 {
			return
		}
		if e := relErr(K, Ko); e > descriptorGainTol {
			t.Errorf("%s: Accuracy 0, K off the oracle by %.3g", label, e)
		}
		if X != nil {
			if e := relErr(X, Xo); e > newtonTol {
				t.Errorf("%s: Accuracy 0, X off the oracle by %.3g", label, e)
			}
		}
	}
	for _, ns := range []bool{false, true} {
		label := dc.name
		if ns {
			label += " NoScaling"
		}
		res, err := Kalman(sys, Im, Ip, nil, &RiccatiOpts{NoScaling: ns})
		if err != nil {
			t.Fatalf("%s Kalman: %v", label, err)
		}
		check(label+" Kalman", kd, res.Accuracy, res.X, mat.DenseCopyOf(res.K.T()))
		res, err = Dare(kd.A, kd.B, kd.Q, kd.R, &RiccatiOpts{E: kd.E, NoScaling: ns})
		if err != nil {
			t.Fatalf("%s Dare of the dual: %v", label, err)
		}
		check(label+" Dare of the dual", kd, res.Accuracy, res.X, res.K)
	}
	lq, err := Lqg(sys, eyeDense(dc.n+dc.m), eyeDense(dc.n+dc.p), nil)
	if err != nil {
		t.Fatalf("%s Lqg: %v", dc.name, err)
	}
	check(dc.name+" Lqg L", kalmanDual(sys, eyeDense(dc.n), nil), lq.Accuracy, nil, mat.DenseCopyOf(lq.L.T()))
}

// TestKalmanDescriptorRefinedGainKept: with E's last pivot 2.4e-6 the
// double-double refinement of the discrete descriptor Dare matched the
// 320-bit L to 1e-17, but the cheap-control gain check (rLost) then
// replaced it by the gain of the rounded X one Newton step on: Kalman's
// L off by 5.4e-6 (unscaled 1.4e-6) and Lqg's by 1.9e-5 at Accuracy 0,
// against an eps-sensitivity of 4e-16 (seed 3559, Y6BJEX amendment 3).
// The refined K is terminal.
func TestKalmanDescriptorRefinedGainKept(t *testing.T) {
	checkDiscreteDescriptorKalman(t, descriptorRefinedCase{"triangular n3 pivot 2.4e-6", 3, 2, 2, false,
		[]float64{-1.579308016545376, 0.8591618135983426, -0.5890359147997083, 0.566157946089093, -0.31556496875403056, 0.785999490689384, 0.32739677164585274, 0.43382622425773854, -1.205726782954539},
		[]float64{1.4095936814894143, 1.2213515253617466, -0.7432104341331951, 1.3072170899411903, 0.09095106738082054, -0.26698198955439784},
		[]float64{-1.2936894763979772, -0.039310751632049, -0.3807261384635414, 0.06346737187035156, 0.9874068805259106, -0.6747946198550068},
		[]float64{0.5725307957548333, 0.44522490984410945, -0.6457519400653848, 0, 0.8382172166570938, -0.00026510629966991495, 0, 0, 2.3910429915618444e-06},
		nil})
}

// TestKalmanDescriptorUnrefinedXHonest: with a rotated E of smallest
// singular value 1.2e-6 the refined gain matched the 320-bit L exactly,
// but the certified float64 X was 3e-8 off the double-double iterate it
// came from, and off the oracle by as much (sensitivity 9e-11): keeping
// the refined gain must not certify that X (seed 2333).
func TestKalmanDescriptorUnrefinedXHonest(t *testing.T) {
	checkDiscreteDescriptorKalman(t, descriptorRefinedCase{"rotated n3 sigma 1.2e-6", 3, 2, 2, false,
		[]float64{0.7177557553360906, -0.7856729543492139, -0.24939807025805608, 0.46769506753940654, 0.41681523075194316, -0.015664356293194733, -0.6326431762856172, 0.3767835858455136, 0.04656369493267445},
		[]float64{-0.40597588814052454, -0.3113308189294068, -2.6180891017485397, -1.6479297414189569, 1.1652458309838107, 0.34377900258842903},
		[]float64{0.22998118095077658, -1.4921640220140868, 0.3230267275881077, -0.21302662354736024, 0.14174581564434047, -0.962876771974801},
		[]float64{-0.15344643658423204, -0.5102932898115922, 0.015784043308831257, 0.4877439941748117, 1.1426352450844655, 1.0407458451455662, -0.5418879949847201, -1.8434289032487157, 0.14983518335695376},
		nil})
}
