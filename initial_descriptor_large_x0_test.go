package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// largeX0StiffDescriptor is reviewer generator gen 1331 of
// scratchpad/w45review/rev_w45_sweep_test.go.txt (w45Case "gen",
// PCG(1331, 454545)): cond(E) 7.7e7, poles −2.0e7, −2.5e3, −1.68,
// −2.06 ± 1.28i, and x0 drawn at |x0| ~ 3e3 with C·x0 ≈ 9908.
func largeX0StiffDescriptor(t *testing.T) (*System, *mat.VecDense) {
	t.Helper()
	A := mat.NewDense(5, 5, []float64{
		-3.2016767179805368, 8.7501185032415005, -0.056943677446508641, 1.9351087706555996, -2.8496837416942662,
		0.34686840422846582, -5.4626121795159479, -2.0815994572090095, 1.9159388178107093, -0.5298844674134372,
		-2.326241009489499, 9.8121753310865163, -3.3171125651010716, 1.8710805455308512, 2.2017642580213104,
		-1.2591889646363761, -3.3619454645852298, 0.41923335291893982, -2.3217936928464402, 1.2598903579740672,
		-0.35071228037275881, -3.2232360810239502, -0.059538470744501873, -1.7016299619531747, -4.561237193956031,
	})
	B := mat.NewDense(5, 1, []float64{
		-1.9742842179783129, 2.4348679439413821, 1.2227400505125556, -0.59655219308418228, 0.79017162301743915,
	})
	C := mat.NewDense(1, 5, []float64{
		-1.4925422142361997, 4.4628455294224825, 1.271261500148996, -3.1279459265977629, -1.3074948997718288,
	})
	E := mat.NewDense(5, 5, []float64{
		3.8221522963479062, -5.3425032310438993, 1.4390046679133046, 0.24383147009439318, 0.3617381061311411,
		-0.040056627339405612, 7.9571324762774012, 0.52087420797933726, 0.82690777548025751, -0.54073201724955267,
		0.41107721147171405, 0.43635325565832339, 1.1912043015685561, -0.57406748523232465, -0.034124394050653123,
		0.45337090861280677, -1.9313230583447341, -0.088921861510047259, 0.018480200524839837, 0.13186820000055374,
		-0.27754521916299107, 2.2400904704614599, -0.016171802401539707, 0.20415610869596743, -0.15196380570478904,
	})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	x0 := mat.NewVecDense(5, []float64{455.7236572789702, 1435.7321694033053, 2565.82870612144, -790.8545333888403, 1189.4051162773342})
	return sys, x0
}

// A large x0 of a routed stiff descriptor keeps the decaying tail of
// Initial and Lsim on the 320-bit pencil: the free response is not the
// step response of A·x0 plus C·x0, whose 1e4 terms cancel to 1e-5 absolute
// in a tail of size 1 (AZUMSC review, gen 1331).
func TestInitialStateLargeX0StiffDescriptorTail(t *testing.T) {
	sys, x0 := largeX0StiffDescriptor(t)
	pol, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, pol)
	if !ok {
		t.Fatal("oracle did not resolve the poles")
	}
	free := freeOracle(t, sys, modes, x0)
	var y0 mat.VecDense
	y0.MulVec(sys.C, x0)

	r, err := Initial(sys, x0, 5)
	if err != nil {
		t.Fatalf("Initial: %v", err)
	}
	worst := stepOracleRel(r.Y.At(0, 0), y0.AtVec(0))
	for s := 1; s < len(r.T); s++ {
		worst = math.Max(worst, stepOracleRel(r.Y.At(0, s), free(0, r.T[s])))
	}
	if worst > 1e-7 {
		t.Errorf("Initial off the pencil by %.3g", worst)
	}

	tt := makeTimeVector(101, 0.05)
	const amp = 0.7
	u := mat.NewDense(len(tt), 1, nil)
	for s := range tt {
		u.Set(s, 0, amp)
	}
	l, err := Lsim(sys, u, tt, x0)
	if err != nil {
		t.Fatalf("Lsim: %v", err)
	}
	worst = stepOracleRel(l.Y.At(0, 0), y0.AtVec(0))
	for s := 1; s < len(tt); s++ {
		worst = math.Max(worst, stepOracleRel(l.Y.At(0, s), free(0, tt[s])+amp*modes.step(0, 0, tt[s])))
	}
	if worst > 1e-7 {
		t.Errorf("Lsim off the pencil by %.3g", worst)
	}
}
