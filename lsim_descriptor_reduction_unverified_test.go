package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// reductionDescriptorCases are frozen continuous descriptors whose E is
// singular to the rank floor of eRankDeficient (LU condition estimate over
// 1e15), so Step and Lsim take the Weierstrass reduction (ZC4XHI census:
// desc/183, desc/184, stfr/2008). References are the 320-bit ZOH response
// of the stored model to bandLsimInput (dt 0.01), whose regular E lags the
// algebraic part one sample; the reduction does not.
var reductionDescriptorCases = []bandDescriptorCase{
	{
		name:     "desc/183",
		a:        mat.NewDense(2, 2, []float64{-0.7377595619325386, 1.0889315484480036, -0.6650599619859114, -0.43336234534810625}),
		b:        mat.NewDense(2, 1, []float64{-0.41139271604713523, 1.3728381729722905}),
		c:        mat.NewDense(1, 2, []float64{0.13595115074396996, -1.7665896794152722}),
		d:        mat.NewDense(1, 1, []float64{1.2816239477770353}),
		e:        mat.NewDense(2, 2, []float64{0.7035909294624929, 0.1384379215303479, 1.008446139588914, 0.1984209598134604}),
		lsimPeak: 0.7540102871082272,
		lsimRef:  map[int]float64{1: -0.12659150700834654, 10: -0.3880410591669846, 50: -0.376179111329086, 100: -0.25257734294291, 150: -0.32426943647884754, 200: -0.3778381953669821},
	},
	{
		name:     "desc/184",
		a:        mat.NewDense(4, 4, []float64{-0.5376327454280856, -0.327611936747689, 0.04727272880753575, -0.8392813928526531, -0.41650628788928556, -1.2450167451752432, 0.25034395859712144, -0.7076803498404549, 1.7228408157644506, 1.1560967115837828, -0.3288257622561135, 1.2530848078688857, 0.7793092964775613, 1.3820808250051324, -0.10838999044546536, 0.9042783937938308}),
		b:        mat.NewDense(4, 1, []float64{2.41593023726111, -0.06940219499058763, 0.49191929937087886, -1.06266091065886}),
		c:        mat.NewDense(1, 4, []float64{-1.6171867874531969, -0.572495534263099, -1.188171952983225, 1.4523811527226131}),
		d:        mat.NewDense(1, 1, []float64{-1.0435291414590195}),
		e:        mat.NewDense(4, 4, []float64{0.1496063868265561, 0.0008825862634884663, 0.689428739320381, 0.17559369738857203, -0.5161104484561543, 0.6658783188198815, -0.5727986190255383, 0.4199020367630874, -0.18572718113485168, -0.030441406512082866, -0.7397409127181286, -0.6705991399136254, 0.3769988008301966, -0.6224840655931512, -0.422655581577703, 0.47308529981997954}),
		lsimPeak: 6.153825399057868,
		lsimRef:  map[int]float64{1: 2.118112419123153, 10: 4.4427379519260475, 50: 1.1795566282986987, 100: 2.0001783750554085, 150: -0.4210739450383076, 200: 2.3932985051713995},
	},
	{
		name:     "stfr/2008",
		a:        mat.NewDense(6, 6, []float64{0.348846228548942, -0.02537149802228711, -0.28241905564498393, -0.3381976544297973, -0.3691320389584049, -0.29709119636367604, -0.06309290916817555, 0.20156183715032164, -0.4139019127219865, 0.172275055203429, -0.07227312195728586, 0.44216903847667277, -0.2644231240784809, -0.3092958062774014, 0.4441680572743194, 0.5043485899095734, 0.2682135656693601, -0.7357369666746794, -0.38689336382386225, 0.1796093220369523, -0.03895338426207931, -0.7144818996700659, -0.020875526205213248, 0.701080403424398, -0.3791533201862418, -0.7501650920818539, -1.050598359050154, -0.7938561460002402, -0.04852154645997864, -1.2693186057845045, -0.5301635576649653, -0.15506040315084924, 1.0654728415346804, 0.778782812866041, 1.02151852447051, -0.5827513713282613}),
		b:        mat.NewDense(6, 1, []float64{0.48833112986177085, 0.2523784671072571, 1.1225392051035565, 0.20336583472721292, -0.16045703471015504, -1.466862395243905}),
		c:        mat.NewDense(1, 6, []float64{1.1251691192260265, -0.5192868598346269, -1.1577127418639273, -1.2492080131528736, 2.1963593690730514, 0.4209013071883502}),
		d:        mat.NewDense(1, 1, []float64{0.9026697272024868}),
		e:        mat.NewDense(6, 6, []float64{0.35356475683714295, 0.412778528991719, 0.23946409689325932, -0.29013490496359035, -0.13418660882202513, -0.023515407207877047, -0.7226037442491287, 0.09321214155819355, 0.18609903157865468, 0.25050526073436985, -0.18380739338222954, 0.1670698793536958, 0.04990568979163882, -0.0629920529606038, 0.3078039451794603, 0.06416012901817543, -0.2149175390527253, 0.30702060580629986, -0.1277674374340725, -0.0582458350725807, 0.04961667650843985, 0.379162199765939, -0.09064667214348653, -0.1602703816859438, 0.18798093064898827, 0.17280038683895949, 0.4258226199862251, -0.054247444738664975, 0.268515329381923, -0.07112578456470443, -0.05585156734306361, 0.20241970205688095, 0.2529955021168475, 0.18735649106422253, -0.26135212264811003, -0.02904517024783787}),
		lsimPeak: 3.214601565188774,
		lsimRef:  map[int]float64{1: -3.214601565188774, 10: -1.5764389492451114, 50: -0.5949895687590003, 100: -2.928317928804716, 150: 0.07561840351622261, 200: 2.3486107970932992},
	},
}

// TestLsimDescriptorReductionUnverifiedAsStep: Lsim of a singular-E
// descriptor returns ErrAccuracyUnverified exactly where Step does (an
// unverified reduction TransferFunction), with the samples of the
// unflagged simulation bit for bit (ZC4XHI). Which rows are unverified
// depends on the arch; the parity does not.
func TestLsimDescriptorReductionUnverifiedAsStep(t *testing.T) {
	tt, u := bandLsimInput()
	for _, k := range reductionDescriptorCases {
		sys := k.sys(t)
		unv := sys.reductionUnverified()
		step, serr := Step(sys, 0)
		if step == nil || (serr != nil) != unv || (serr != nil && !errors.Is(serr, ErrAccuracyUnverified)) {
			t.Fatalf("%s: Step response %t, err %v; reduction unverified %t", k.name, step != nil, serr, unv)
		}
		resp, err := Lsim(sys, u, tt, nil)
		if resp == nil || (err != nil) != unv || (err != nil && !errors.Is(err, ErrAccuracyUnverified)) {
			t.Fatalf("%s: Lsim response %t, err %v; reduction unverified %t", k.name, resp != nil, err, unv)
		}
		raw, rerr := lsim(sys, u, tt, nil)
		if rerr != nil {
			t.Fatalf("%s: lsim: %v", k.name, rerr)
		}
		for i := range len(tt) {
			if math.Float64bits(resp.Y.At(0, i)) != math.Float64bits(raw.Y.At(0, i)) {
				t.Fatalf("%s: sample %d %v, unflagged %v", k.name, i, resp.Y.At(0, i), raw.Y.At(0, i))
			}
		}
	}
}

// TestLsimDescriptorReductionAlgebraicLag: the reduction's response is the
// stored model's 320-bit ZOH response with the one-sample lag of its
// algebraic part removed, y_k = ref_k + (D_eff − D)(u_k − u_{k−1}), O(1)
// apart from ref itself (ZC4XHI residual: kept as is).
func TestLsimDescriptorReductionAlgebraicLag(t *testing.T) {
	tt, u := bandLsimInput()
	for _, k := range reductionDescriptorCases {
		sys := k.sys(t)
		red, err := sys.properExplicitForm()
		if err != nil {
			t.Fatalf("%s: %v", k.name, err)
		}
		lag := red.D.At(0, 0) - sys.D.At(0, 0)
		resp, _ := Lsim(sys, u, tt, nil)
		if resp == nil {
			t.Fatalf("%s: no response", k.name)
		}
		worst, raw := 0.0, 0.0
		for i, ref := range k.lsimRef {
			y := resp.Y.At(0, i)
			worst = math.Max(worst, math.Abs(y-lag*(u.At(i, 0)-u.At(i-1, 0))-ref))
			raw = math.Max(raw, math.Abs(y-ref))
		}
		if !(worst <= 1e-9*k.lsimPeak) {
			t.Errorf("%s: lag-corrected error %g over 1e-9·%g", k.name, worst, k.lsimPeak)
		}
		if !(raw > 1e-3*k.lsimPeak) {
			t.Errorf("%s: uncorrected error %g, want the O(1) lag", k.name, raw)
		}
	}
}

// TestLsimDescriptorReductionInitialStateRejected: with x0 ≠ 0 these
// singular-E rows are rejected before any reduction runs.
func TestLsimDescriptorReductionInitialStateRejected(t *testing.T) {
	tt, u := bandLsimInput()
	for _, k := range reductionDescriptorCases {
		sys := k.sys(t)
		n, _, _ := sys.Dims()
		x0 := mat.NewVecDense(n, nil)
		x0.SetVec(0, 1)
		resp, err := Lsim(sys, u, tt, x0)
		if resp != nil || !errors.Is(err, ErrDescriptorInitialState) {
			t.Errorf("%s: response %t, err %v; want ErrDescriptorInitialState", k.name, resp != nil, err)
		}
	}
}

// TestLsimReductionAndRefusedSplitMarkersDistinct: the reduction flag and
// the refused band split marker (5ZCX6J) never cover the same model, and
// each Lsim error names its own cause, so neither hides the other.
func TestLsimReductionAndRefusedSplitMarkersDistinct(t *testing.T) {
	const refusedMark, reductionMark = "pencil split refused", "descriptor reduction"
	tt, u := bandLsimInput()
	for _, k := range bandRefusedCases {
		sys := k.sys(t)
		if _, _, refused := sys.zeroStateRoute(); refused == nil || sys.reductionUnverified() {
			t.Fatalf("%s: refused %v, reduction unverified %t; want refused only", k.name, refused, sys.reductionUnverified())
		}
		resp, err := Lsim(sys, u, tt, nil)
		if resp != nil && (!errors.Is(err, ErrAccuracyUnverified) || !strings.Contains(err.Error(), refusedMark) || strings.Contains(err.Error(), reductionMark)) {
			t.Errorf("%s: err %v; want the refused split marker only", k.name, err)
		}
	}
	for _, k := range reductionDescriptorCases {
		sys := k.sys(t)
		if _, _, refused := sys.zeroStateRoute(); refused != nil {
			t.Fatalf("%s: refused %v on a reduced descriptor", k.name, refused)
		}
		_, err := Lsim(sys, u, tt, nil)
		if err != nil && (!strings.Contains(err.Error(), reductionMark) || strings.Contains(err.Error(), refusedMark)) {
			t.Errorf("%s: err %v; want the reduction marker only", k.name, err)
		}
	}
}
