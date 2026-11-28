package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// singularWithinFloorCase is a frozen descriptor whose E is singular within
// the rank floor and whose reduction verifies and removes the algebraic part
// (GM22S5, census desc/184 with a stable stored fast mode, desc/14 with an
// unstable one). References are 320-bit responses of the nearest singular-E
// model E − σ_n·u·vᵀ: Lsim of bandLsimInput by sample index (dt 0.01,
// lsimPeak the largest |y|), Step at t = 1 and 2, and stepY0 = D_eff.
type singularWithinFloorCase struct {
	bandDescriptorCase
	stepY0 float64
}

var singularWithinFloorCases = []singularWithinFloorCase{
	{
		name:     "desc/184",
		a:        reductionDescriptorCases[1].a,
		b:        reductionDescriptorCases[1].b,
		c:        reductionDescriptorCases[1].c,
		d:        reductionDescriptorCases[1].d,
		e:        reductionDescriptorCases[1].e,
		lsimPeak: 6.18038209467616,
		lsimRef:  map[int]float64{0: 2.1932281633928303, 1: 2.4708129920969166, 10: 4.730456719025148, 50: 0.8276858692978837, 100: 2.3470881294170405, 150: -0.7589497495029063, 200: 2.718173478905465},
		stepRef:  [2]float64{2.799233795562193, 1.5868123825898177},
		stepY0:   4.386456326785661,
	},
	{
		name:     "desc/14",
		a:        mat.NewDense(5, 5, []float64{0.38206887954221996, -0.701153451107689, -0.014997110110361938, 0.04963186839247708, 0.18014134075803265, -0.0755478263108408, 0.6176621278744601, 0.8796664537771676, -0.31214752067482443, 1.2815799963677572, -0.8973567877703581, 0.6053553526004881, -0.3746337531552232, 0.1768129219619257, -0.5238135917203337, -0.39127111417304544, 0.12927055689109795, 0.39967835108397726, 0.14575587417134708, 0.8684111636103172, 0.5041346090763745, -0.7187992021248136, -0.448422887619753, 0.10387646496738386, 0.13246299130697947}),
		b:        mat.NewDense(5, 1, []float64{-1.7915916935784826, -0.30996547692419774, 1.8029557537178744, 0.038010825413523526, -0.7888760647477894}),
		c:        mat.NewDense(1, 5, []float64{-0.13350121972700563, -0.46827126260841806, -1.4007059835318232, 0.016893319936047244, 0.8353490893907163}),
		d:        mat.NewDense(1, 1, []float64{-1.3785892351630769}),
		e:        mat.NewDense(5, 5, []float64{-0.3348520555758538, -0.4571687628326559, -0.2794612483526051, 1.1187527320147512, -0.17142638009772143, 0.9129144870098662, 0.5443059965236016, 0.36660774348704617, 0.17739903407921814, 0.07543444672278002, 0.09472947954583003, -0.03307915726691498, -1.1415911357876547, -0.3205294119548239, -0.26175900041363265, 0.5023951420791498, -1.0092924857941707, 0.1666408665260477, -0.060114642191265416, -0.5462407638448332, -0.45132224706597274, -0.2593364660703733, 0.5078763455112241, -0.40639880762671626, 0.10179676219452088}),
		lsimPeak: 0.32916458443001917,
		lsimRef:  map[int]float64{0: -0.1256435320181748, 1: -0.1409644410201299, 10: -0.26235126666236974, 50: -0.002710858218889456, 100: -0.13467707434042167, 150: 0.03275773660368987, 200: -0.29165989532010855},
		stepRef:  [2]float64{-0.11718309740001565, -0.22468938915460723},
		stepY0:   -0.2512870640363496,
	},
}

// TestStepLsimDescriptorSingularWithinFloorResponse: on an E singular within
// the rank floor whose reduction removes the algebraic part, Step and Lsim
// return nil and the response of the singular-E model, D_eff acting at
// once: Step starts at D_eff, and Lsim follows input jumps without lag
// (GM22S5, option A).
func TestStepLsimDescriptorSingularWithinFloorResponse(t *testing.T) {
	const line = 1e-7
	for _, k := range singularWithinFloorCases {
		sys := k.sys(t)
		tt, u := bandLsimInput()
		resp, err := Lsim(sys, u, tt, nil)
		if err != nil || resp == nil {
			t.Fatalf("%s Lsim: err = %v", k.name, err)
		}
		for i, ref := range k.lsimRef {
			if e := math.Abs(resp.Y.At(0, i) - ref); !(e <= line*k.lsimPeak) {
				t.Errorf("%s Lsim sample %d: %v, singular-E model %v", k.name, i, resp.Y.At(0, i), ref)
			}
		}
		peak := math.Max(math.Abs(k.stepY0), math.Max(math.Abs(k.stepRef[0]), math.Abs(k.stepRef[1])))
		for j, tf := range []float64{1, 2} {
			resp, err := Step(sys, tf)
			if err != nil || resp == nil {
				t.Fatalf("%s Step(%g): err = %v", k.name, tf, err)
			}
			last := len(resp.T) - 1
			if resp.T[last] != tf {
				t.Fatalf("%s Step: last sample at %g, want %g", k.name, resp.T[last], tf)
			}
			if e := math.Abs(resp.Y.At(0, 0) - k.stepY0); !(e <= line*peak) {
				t.Errorf("%s Step(%g) y(0) = %v, want D_eff %v", k.name, tf, resp.Y.At(0, 0), k.stepY0)
			}
			if e := math.Abs(resp.Y.At(0, last) - k.stepRef[j]); !(e <= line*peak) {
				t.Errorf("%s Step(%g) y(%g) = %v, singular-E model %v", k.name, tf, tf, resp.Y.At(0, last), k.stepRef[j])
			}
		}
	}
}
