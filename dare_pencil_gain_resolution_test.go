package controlsys

import (
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// darePencilGainCases are Dares of the given inputs (no reduced inputs) at
// which BᵀXB does not round R away (rLost false), drawn on arm64: a
// stable mode near the unit circle that B does not reach, under a state
// transform, so that ‖X‖ is 1e6 to 1e8 while the gain is 1e-7 to 1e-4
// (HFGUNG). Dare returned the pencil K there at Accuracy 0 with no
// resolution check: 2e-5 to 3.5e-4 off the 320-bit gain against a band of
// 1.5e-6 to 3.1e-5. In "cancelling" the pencil K is 2.1e-5 from the gain
// of its X while the rounding of X alone resolves that gain (gainResolved
// with miss 0 passes); in the others the rounding of X does not resolve
// it either.
var darePencilGainCases = []struct {
	name       string
	n, m       int
	A, B, Q, R []float64
}{
	{
		name: "cancelling", n: 3, m: 2,
		A: []float64{1.2397492734047182, -0.18335820487393248, 0.3046453493078671, 0.7722263366854194, -0.2617537873694807, 0.5502730979102666, -0.8055287716951083, 0.5490861307507068, 0.05656541164805932},
		B: []float64{0.18918649728612982, 1.2505642085854247, -2.951246805566693, 1.7010562954102384, -0.5734859345179992, -3.7559849152093747},
		Q: []float64{101369.84289297072, -55.47092335331715, 33726.2133567795, -55.47092335331715, 0.03355005019124774, -18.456217026703964, 33726.2133567795, -18.456217026703964, 11220.86814753847},
		R: []float64{93.50496364150716, -0.034685921610246744, -0.034685921610246744, 93.41038979020534},
	},
	{
		name: "unresolved n4", n: 4, m: 1,
		A: []float64{1.435181462018049, -0.2644822083867526, 0.2920096989382238, -0.18352996866862437, 0.1374026883736259, -0.3080629061882962, -0.20215207067676288, 0.07938664550840625, -1.4804027690040913, -0.2688566776768537, -0.19141422050595924, 0.8477685386325666, 0.7908063611649715, -0.39204983724675224, 0.2028866410935319, 0.2671569514930164},
		B: []float64{0.3427417570492545, 0.3345741229533868, -1.566087627501197, 3.2032369306837727},
		Q: []float64{1703.7461408518764, -488.81269560774786, 521.6050371398841, 123.75223832287475, -488.81269560774786, 140.30175223774984, -149.6436487160401, -35.50147584946009, 521.6050371398841, -149.6436487160401, 159.70317060825204, 37.8906732906695, 123.75223832287475, -35.50147584946009, 37.8906732906695, 9.008374891182386},
		R: []float64{55990.114237120935},
	},
	{
		name: "unresolved n3", n: 3, m: 1,
		A: []float64{0.8337010265092233, -0.16421745507175792, -0.1699338771643311, -0.14554394208158322, 0.6978573906842083, -0.07749280263569237, -0.1531250866756581, -0.35302616981538754, 0.46690702246994836},
		B: []float64{0.5023690362379318, -0.8235867590990339, 1.9161233480677156},
		Q: []float64{103785.08763597686, -20166.527092806773, -35878.32654522326, -20166.527092806773, 3918.567020276628, 6971.533811166202, -35878.32654522326, 6971.533811166202, 12403.075976638693},
		R: []float64{10.683476465478314},
	},
}

// TestDarePencilGainUnresolvedTakesHewerGain: the pencil K of the cases is
// held to gainResolved, with its distance from the gain of its X, before
// Dare returns it, and takes the Hewer gain, exact here. On arm64, where
// they were drawn, Dare, Dlqr and Kalman (on the dual plant) return
// Accuracy 0 with that gain; elsewhere none certifies a K beyond its band,
// max(gainFloor, 4× its move under eps perturbations of the data).
func TestDarePencilGainUnresolvedTakesHewerGain(t *testing.T) {
	for _, tc := range darePencilGainCases {
		t.Run(tc.name, func(t *testing.T) {
			n, m := tc.n, tc.m
			A, B := mat.NewDense(n, n, tc.A), mat.NewDense(n, m, tc.B)
			Q, R := mat.NewDense(n, n, tc.Q), mat.NewDense(m, m, tc.R)
			ref := riccatiExactReference(false, A, B, Q, R, nil, nil, riccatiExactPrec)
			sens, ok := riccatiExactSensitivity(false, A, B, Q, R, nil, nil, ref, "ABQR")
			if !ref.OK || !ok {
				t.Fatal("reference not certified")
			}
			if red, _ := reduceInputs(A, B, Q, R, nil); red != nil || rLost(B, R, ref.X) {
				t.Fatal("case left the non-rLost exit of the given inputs")
			}
			kTol := max(gainFloor, 4*sens.K)
			native := runtime.GOARCH == "arm64"
			sys, err := New(mat.DenseCopyOf(A.T()), eyeDense(n), mat.DenseCopyOf(B.T()), mat.NewDense(m, n, nil), 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, solve := range []struct {
				name string
				f    func() (*RiccatiResult, error)
				gain func(*RiccatiResult) mat.Matrix
			}{
				{"Dare", func() (*RiccatiResult, error) { return Dare(A, B, Q, R, nil) }, func(r *RiccatiResult) mat.Matrix { return r.K }},
				{"Dlqr", func() (*RiccatiResult, error) { return Dlqr(A, B, Q, R, nil) }, func(r *RiccatiResult) mat.Matrix { return r.K }},
				{"Kalman", func() (*RiccatiResult, error) { return Kalman(sys, Q, R, nil, nil) }, func(r *RiccatiResult) mat.Matrix { return r.K.T() }},
			} {
				res, err := acceptUnverified(solve.f())
				if err != nil {
					t.Errorf("%s: %v", solve.name, err)
					continue
				}
				kErr := relErr(solve.gain(res), ref.K)
				t.Logf("%s on %s: Accuracy %d, K err %.3g (tol %.3g)", solve.name, runtime.GOARCH, res.accuracy, kErr, kTol)
				switch {
				case native && res.accuracy != 0:
					t.Errorf("%s: Accuracy %d, want 0", solve.name, res.accuracy)
				case native && !(kErr <= 1e-12):
					t.Errorf("%s: K %.3g off the 320-bit gain, want the exact Hewer gain", solve.name, kErr)
				case res.accuracy == 0 && !(kErr <= kTol):
					t.Errorf("%s: Accuracy 0 with K %.3g off the 320-bit gain, want within %.3g", solve.name, kErr, kTol)
				}
			}
		})
	}
}

// TestPencilGainResolved: the 320-bit gain of a well-conditioned Dare with
// cross term and non-symmetric A is resolved from its rounded X and
// iterate, a K 2·gainFloor off it is not, nor is the 320-bit gain of the
// cases, which the rounding of their X does not resolve.
func TestPencilGainResolved(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.9, 0.4, -0.2, -0.3, 0.7, 0.5, 0.1, -0.6, 0.8})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1, 0.3, -0.5})
	Q := mat.NewDense(3, 3, []float64{2, 0.3, 0.1, 0.3, 1.5, -0.2, 0.1, -0.2, 1})
	R := mat.NewDense(2, 2, []float64{1, 0.1, 0.1, 2})
	S := mat.NewDense(3, 2, []float64{0.1, 0, -0.2, 0.1, 0, 0.3})
	ref := riccatiExactReference(false, A, B, Q, R, S, nil, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("reference not certified")
	}
	opts := &RiccatiOpts{S: S}
	if !pencilGainResolved(A, B, R, opts, &RiccatiResult{X: ref.X, K: ref.K, x1: ref.X}) {
		t.Error("exact gain not resolved")
	}
	var off mat.Dense
	off.Scale(1+2*gainFloor, ref.K)
	if pencilGainResolved(A, B, R, opts, &RiccatiResult{X: ref.X, K: &off}) {
		t.Error("K 2·gainFloor off its gain resolved")
	}
	for _, tc := range darePencilGainCases {
		n, m := tc.n, tc.m
		A, B := mat.NewDense(n, n, tc.A), mat.NewDense(n, m, tc.B)
		Q, R := mat.NewDense(n, n, tc.Q), mat.NewDense(m, m, tc.R)
		ref := riccatiExactReference(false, A, B, Q, R, nil, nil, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: reference not certified", tc.name)
		}
		if tc.name != "cancelling" && pencilGainResolved(A, B, R, nil, &RiccatiResult{X: ref.X, K: ref.K}) {
			t.Errorf("%s: gain resolved from the rounded X", tc.name)
		}
	}
}
