package controlsys

import (
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A weakly controllable unstable mode with a well-conditioned E: X is
// certified and within its eps band of the 320-bit X, but the gain of the
// rounded X moves 9e-6 per ulp of X, so neither the pencil K (3.99e-6 off)
// nor the double-double gain of X (4.67e-6 off) is K to gainFloor while
// Dare certified it (3GARV6). The second case is a 5x5 descriptor Dare
// with S whose K was 1.77e-6 off at Accuracy 0 (zw46Gen reg 38, cross).
// Measured on arm64; on amd64 the first case's pencil X is not certified
// (Accuracy 1, X 9e-4 off) and the second's K was 9e-7 off, so a K at
// Accuracy 0 is held to newtonTol, the precision of the refined gain.
func TestDare_GainUnresolvedByXRefined(t *testing.T) {
	cases := []struct {
		name             string
		A, B, Q, R, S, E *mat.Dense
	}{
		{
			name: "n2m1",
			A:    mat.NewDense(2, 2, []float64{-1.3607438571715633, 1.1640905296091977, 1.7446646742450758, -0.3594044433712067}),
			B:    mat.NewDense(2, 1, []float64{0.05690913508829608, 0.29571587001271443}),
			Q:    mat.NewDense(2, 2, []float64{1.0000000000000002, 1.9189786719054797e-16, 1.9189786719054797e-16, 1}),
			R:    mat.NewDense(1, 1, []float64{1}),
			E:    mat.NewDense(2, 2, []float64{1.2332640457815616, -0.7681307689138523, 1.5931093767749318, -0.4909957319394508}),
		},
		{
			name: "n5m1S",
			A:    mat.NewDense(5, 5, []float64{0.44778124293409222, 1.2825806700158586, -0.3153424393803112, -1.3730462842051028, -0.14825386528987097, -0.88463946448459296, 1.2512142101337451, -1.2685681059546354, -0.60053698277622303, 0.14095665644904631, -0.92761662639959241, -1.1729735398340047, 1.2752809638515221, -2.5474582982823883, 0.14258918321938235, -0.18807673644951389, 0.54083038018159035, -0.3495743770604538, 0.35109426188587156, 1.6149283175802374, 0.68988456407278942, -0.30705274102357094, -0.3138911658775525, -0.14286164352000741, 0.29677728328760278}),
			B:    mat.NewDense(5, 1, []float64{1.5729633848483371, 0.97920678055415833, -0.20972268161830487, 1.5240222837955253, 0.51863494075016803}),
			Q:    mat.NewDense(5, 5, []float64{7.984019906678923, -0.24038460807885897, 4.6816212482835793, -6.2270653885347116, -4.4419744385401536, -0.24038460807885897, 11.51332008747991, -1.4086575343659362, 3.4687625719646999, 5.4628916980843787, 4.6816212482835793, -1.4086575343659362, 4.4971575211240129, -5.5346964586866294, -2.0918747278161565, -6.2270653885347116, 3.4687625719646999, -5.5346964586866294, 10.875363042722196, 5.3055838015615135, -4.4419744385401536, 5.4628916980843787, -2.0918747278161565, 5.3055838015615135, 9.058251918135511}),
			R:    mat.NewDense(1, 1, []float64{4.6173719144331713}),
			S:    mat.NewDense(5, 1, []float64{-0.18530673415391288, -1.4029429271484626, -0.33901640635372737, -0.094134855892670691, -0.7978541114543295}),
			E:    mat.NewDense(5, 5, []float64{0.84891872238850441, -0.21841375237250443, 0.85711906958171025, -0.41829005992096491, -0.73169945474760545, 0.11414665351631229, 0.32797850380540294, -1.2130778843578647, 0.49253507376410804, -0.36216100200660062, -0.30044819213744717, -0.43367496503590758, 0.29518827719206864, -1.3315171843415496, 0.84255169623787951, -0.88366935285237458, 1.369803568400664, -0.77170925330311657, 1.143323995098104, -0.045350862027905892, 0.43726946902173469, 0.40335800820799145, -0.78771136749893456, 0.48524544275648701, -0.14681521610579107}),
		},
	}
	for _, c := range cases {
		ref := riccatiExactReference(false, c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: 320-bit reference failed", c.name)
		}
		for _, ns := range []bool{false, true} {
			opts := &RiccatiOpts{S: c.S, E: c.E, NoScaling: ns}
			res, err := Dare(c.A, c.B, c.Q, c.R, opts)
			if err != nil {
				t.Fatalf("%s NoScaling %v: Dare: %v", c.name, ns, err)
			}
			eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
			if runtime.GOARCH == "arm64" && res.Accuracy != 0 {
				if raceEnabled {
					// -race rounds this pencil differently; Accuracy 1 is honest.
					t.Logf("%s NoScaling %v: Dare Accuracy %d under race", c.name, ns, res.Accuracy)
				} else {
					t.Errorf("%s NoScaling %v: Dare Accuracy %d on arm64, want 0", c.name, ns, res.Accuracy)
				}
			}
			if res.Accuracy == 0 && (eX > newtonTol || eK > newtonTol) {
				t.Errorf("%s NoScaling %v: Dare Accuracy 0 with X %.3g, K %.3g off the 320-bit solution; want ≤ %g", c.name, ns, eX, eK, newtonTol)
			}
			if !closedLoopStable(c.A, c.B, res.K, c.E, make([]complex128, len(res.Eig))) {
				t.Errorf("%s NoScaling %v: A − BK not stable", c.name, ns)
			}
			lq, err := Dlqr(c.A, c.B, c.Q, c.R, opts)
			if err != nil {
				t.Fatalf("%s NoScaling %v: Dlqr: %v", c.name, ns, err)
			}
			if eK := relErr(lq.K, ref.K); lq.Accuracy == 0 && eK > newtonTol {
				t.Errorf("%s NoScaling %v: Dlqr Accuracy 0 with K %.3g off; want ≤ %g", c.name, ns, eK, newtonTol)
			}
		}
	}
}

// TestDare_GainUnresolvedByXNotCertifiedWrong: on amd64 the gain of X
// moved 2.2e-7 per ulp of X and the pencil K was 1.21e-5 off the 320-bit
// gain at Accuracy 0 (data sensitivity 1e-7), while the double-double
// refinement converged but its K was not the gain of the rounded X to
// its tolerance and the Hewer gain failed; Dare now returns it at
// Accuracy 1 (zw46Gen reg 1920 on amd64; 3GARV6). Whatever the platform's
// rounding, a K at Accuracy 0 must be within gainFloor of the 320-bit gain.
func TestDare_GainUnresolvedByXNotCertifiedWrong(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-0.34589172149790254, 0.41271439154337652, 1.2616956867432305, 0.60865897650811118, 0.8533727817532224, 1.303228884273236, -1.7342142391706266, 0.29384774706837691, -3.3046660772696059})
	B := mat.NewDense(3, 1, []float64{-0.44198869945481079, 1.0019916991309084, 0.83680753210707759})
	Q := mat.NewDense(3, 3, []float64{1, -5.5511151231257827e-17, 0, -5.5511151231257827e-17, 1.0000000000000002, -1.3877787807814457e-16, 0, -1.3877787807814457e-16, 1.0000000000000002})
	R := mat.NewDense(1, 1, []float64{1})
	E := mat.NewDense(3, 3, []float64{-1.0357288926804353, 0.63447798102919284, 0.28077630042058566, 1.0410869627585242, -0.0086802341549025441, -0.026291133284471163, -0.28738180914378531, -0.081741622197320507, -1.0069164227403546})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: ns})
		if err != nil {
			// arm64 rounds the pencil to ErrNoStabilizing here, as before.
			t.Logf("NoScaling %v: %v", ns, err)
			continue
		}
		if eX := relErr(res.X, ref.X); eX > newtonTol {
			t.Errorf("NoScaling %v: X %.3g off the 320-bit X", ns, eX)
		}
		if eK := relErr(res.K, ref.K); res.Accuracy == 0 && eK > gainFloor {
			t.Errorf("NoScaling %v: Accuracy 0 with K %.3g off the 320-bit gain, above %g", ns, eK, gainFloor)
		}
	}
}
