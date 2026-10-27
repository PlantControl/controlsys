package controlsys

import (
	"cmp"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffFullRankECases are U·(F, W)·V models with W[n-1,n-1] = 1.5·k·ε
// whose rounded E has σ₃ at 1.09–1.26 × n·eps(σ₁), full MATLAB rank, yet
// QZ returns β = 0 for the stiff mode (EJYYFD: window seeds 7878, 70276,
// 121514, 274104, k drawn after F and W).
var stiffFullRankECases = []struct {
	name          string
	a, b, c, d, e []float64
	p             int
}{
	{"k=0.444", []float64{-1.1902035657131833, 0.2313460641067257, -0.1908007515341377, -0.36649261055144033, 0.3609446152655081, -0.12011715719003468, 0.4386134467549775, 0.046639522609797306, -0.9574575350229291},
		[]float64{-0.9915270525822623, -0.3639378966851768, 0.3241065208025237},
		[]float64{1.2501521258706652, -0.6319602243174343, -1.100726530362802},
		[]float64{1.7326201243228754},
		[]float64{-0.01402995176645566, 0.8394203486921765, -0.28167262602453164, -0.537279348478248, 0.2901924814914731, -0.1121290255218737, -0.8022854408941439, -0.24312382497507473, 0.05923842870018078}, 1},
	{"k=0.839", []float64{-0.7541743814673876, 0.8134016104792965, 0.5979070630116079, -1.6792343198167865, -0.25300172392251896, -1.305677990594461, -0.29906871186440637, -0.34329963928255325, 0.22370019651400724},
		[]float64{0.3352430980416986, 1.2550683860640892, 1.054455164932288},
		[]float64{-0.6017195716491115, -1.4349288561067157, -0.974891604133274},
		[]float64{-0.0159289577942906},
		[]float64{-0.31396564050788694, 0.15702307420568062, -0.13011492280012613, -0.7888720497951793, 0.03380250948878309, -0.003130061739464068, 0.12257844429082965, 0.6418687212337923, -0.5803732827275608}, 1},
	{"k=0.939 unstable stiff", []float64{-0.32146007676274246, 0.42439417619586967, -0.4205803872325719, -0.011185093038736865, 0.1979888431325947, 0.33366355647544943, 0.15759492926205135, 1.2688376437864912, 0.19779699393503405},
		[]float64{1.7837828720280384, -0.7509223060906908, -1.3417205077165355},
		[]float64{-1.4708582515873392, 0.7089190691997246, -1.410367940489857, -2.5614752591805483, 0.6894863875138618, -0.5062399881888265},
		[]float64{0.4297225812922495, 1.2374607638088848},
		[]float64{0.8576495396944448, -0.25799007003136576, 0.406786946981527, 0.24005654242368868, 0.1733977916367387, -0.4327919900548725, -0.37495383564072793, -0.2649713557228225, 0.6629400798702162}, 2},
	{"k=0.936", []float64{-0.19192447130815163, -0.26968134922268466, 0.9181201500614193, -0.6803605704196104, 0.16245873992425938, -0.11174594082177579, 1.5145221272922038, 0.38614069422758185, 0.3153891593967021},
		[]float64{-1.7807276234624982, -1.0712476700105285, 0.37454715145989925},
		[]float64{-1.2223141400912394, -0.045403234421407965, -0.212212069577098},
		[]float64{-0.9015151423679206},
		[]float64{-0.0912678462740003, 0.17429976184431567, 0.04316637678469383, -0.7644778029201222, 0.5049875882995569, -0.0008443900121799322, 0.505805127326457, 0.7018174999312362, 0.39369474494074364}, 1},
}

func stiffFullRankEModel(t *testing.T, i int) *System {
	t.Helper()
	c := stiffFullRankECases[i]
	sys, err := NewDescriptor(mat.NewDense(3, 3, c.a), mat.NewDense(3, 1, c.b), mat.NewDense(c.p, 3, c.c),
		mat.NewDense(c.p, 1, c.d), mat.NewDense(3, 3, c.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func largestPole(p []complex128) complex128 {
	return slices.MaxFunc(p, func(a, b complex128) int { return cmp.Compare(cmplx.Abs(a), cmplx.Abs(b)) })
}

// With E of full MATLAB rank, Poles dropped the stiff mode QZ gives β = 0:
// IsStable was true for "k=0.939 unstable stiff" while ToExplicit has a
// pole at +2.5e15 and Step diverges. Poles and IsStable must agree with
// the explicit model, as ToExplicit, IsProper and Step do (RL6IT7).
func TestPolesFullRankEKeepsStiffModeOfExplicitModel(t *testing.T) {
	full := 0
	for i, c := range stiffFullRankECases {
		sys := stiffFullRankEModel(t, i)
		if !matlabRankFull(t, sys.E) {
			continue
		}
		full++
		ex, err := sys.ToExplicit()
		if err != nil {
			t.Fatalf("%s: ToExplicit: %v", c.name, err)
		}
		pol, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", c.name, err)
		}
		want, err := ex.Poles()
		if err != nil {
			t.Fatal(err)
		}
		if len(pol) != 3 {
			t.Errorf("%s: Poles = %v, want 3 finite as ToExplicit %v", c.name, pol, want)
			continue
		}
		if g, w := largestPole(pol), largestPole(want); cmplx.Abs(g-w) > 1e-9*cmplx.Abs(w) {
			t.Errorf("%s: stiff pole %v, ToExplicit %v", c.name, g, w)
		}
		got, err := sys.IsStable()
		if err != nil {
			t.Fatal(err)
		}
		if ws, _ := ex.IsStable(); got != ws {
			t.Errorf("%s: IsStable = %v, ToExplicit model %v (poles %v)", c.name, got, ws, want)
		}
	}
	if full == 0 {
		t.Fatal("no model has full MATLAB rank E on this platform")
	}
}

// Stabsep folded the stiff mode, its β = 0 turned ~1e-32 by the reorder,
// into the explicit G1: G1 + G2 was off the pencil by 0.35 and 0.56.
func TestStabsepFullRankEStiffModeMatchesPencil(t *testing.T) {
	for i, c := range stiffFullRankECases {
		sys := stiffFullRankEModel(t, i)
		sep, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if rel := maxPencilRel(t, sys, []complex128{0.05i, 0.37i, complex(0.5, 3), 10i}, func(s complex128) [][]complex128 {
			g, q := pencilResponse(t, sep.Stable, s), pencilResponse(t, sep.Unstable, s)
			for i := range g {
				for j := range g[i] {
					g[i][j] += q[i][j]
				}
			}
			return g
		}); rel > 1e-9 || math.IsNaN(rel) {
			t.Errorf("%s: Stable + Unstable off the pencil by %.3g", c.name, rel)
		}
		for _, part := range []*System{sep.Stable, sep.Unstable} {
			if pol, err := part.Poles(); err != nil || slices.ContainsFunc(pol, func(p complex128) bool { return cmplx.Abs(p) > 1e20 }) {
				t.Errorf("%s: part poles %v, %v", c.name, pol, err)
			}
		}
	}
}
