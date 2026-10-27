package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// largeX0StiffDescriptor is reviewer generator gen 1331 of
// scratchpad/w45review/rev_w45_sweep_test.go.txt (w45Case "gen",
// PCG(1331, 454545)): cond(E) 7.7e7, poles −2.0e7, −2.5e3, −1.68,
// −2.06 ± 1.28i, and x0 drawn at |x0| ~ 3e3 with C·x0 ≈ 9908.
func largeX0StiffDescriptor(t testing.TB) (*System, *mat.VecDense) {
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

// Lsim from x0 on a grid that resolves the fast pole (dt = 0.2/|λ_fast|)
// keeps the fast part's share of x0 on the pencil: taken from E·x0, which
// projects x0 onto the nearly null directions of E, it carried cond(E)·ε
// relative error (2.8e-7 here), while from A·x0 it is accurate. Reviewer
// generator gen 229 of scratchpad/w45review/rev_w45_sweep_test.go.txt
// (PCG(229, 454545), cond(E) 2.2e10, |λ_fast| 6.2e9; AZUMSC review).
func TestLsimInitialStateStiffDescriptorFastGrid(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{
		-20.92193676320473, 7.5710963870004493, 10.178472416004963, -0.61210886914353912, 9.6624818420711662, 8.211412247662258,
		-5.8776079579807119, -22.283094863857613, -10.508547318893488, 15.947596509521928, -10.159822766041623, -8.0135603854710862,
		-7.614681945661145, 16.268257361115701, -28.39499942538718, 6.4823865377718279, -3.3224911304532347, -1.841826911615648,
		0.88741135604255372, -8.2368627226890681, 11.877888190047681, -0.54787748409886705, -2.5581311981325459, 15.392611223374029,
		-22.92640633024271, -7.7790344688429691, -17.052740642532111, 23.301986110678136, -26.798359322113978, 6.8853956660428501,
		-16.742902929404003, -5.5373747006232117, 7.5803358450507847, 9.6292135334703097, -4.6546419245941877, 4.0707730073573547,
	})
	B := mat.NewDense(6, 1, []float64{
		0.37767784060338194, -3.2054957759580436, 3.5124177045472527, 0.55487876962653671, 1.8879224494925892, 0.89889703865169801,
	})
	C := mat.NewDense(1, 6, []float64{
		1.9367509547619997, -2.3981137356763713, 1.7343555005797284, 1.31598518113679, 0.34322044459024842, 1.5514171207017835,
	})
	E := mat.NewDense(6, 6, []float64{
		3.4056271052572153, 0.12240280292364843, 0.10664311228932497, 0.96590049825467339, 0.24771335757976304, -0.6208538972176042,
		-1.7975039134060302, 7.4063687382391263, -1.1150180377831298, -1.3336419120719585, -0.19085876646808103, -1.2506379057425407,
		0.41422208876747724, -0.22764583421054849, 1.5473741404670298, -0.057713759212410398, -0.13946288935777193, -0.055156006168660315,
		0.080746099913851482, 1.5800297475510006, -2.599008872583938, 4.4537170766782825, -0.7757041474180788, 1.225539045030769,
		1.3573847462649957, 0.3418929928284743, 0.16354734810129504, -1.6325375616139191, 3.135715595631094, 1.3458576894210235,
		1.3697237130616686, 0.96560109107943759, -0.55568112775698064, 0.31883847817976613, 0.9821537061370943, 0.31180797264681764,
	})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	x0 := mat.NewVecDense(6, []float64{1.979653950810202, 2.122624355443024, 0.35953887072096885, 0.11961900558070837, 0.5626117257850443, -0.7368971446921588})
	pol, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, pol)
	if !ok {
		t.Fatal("oracle did not resolve the poles")
	}
	fast := 0.0
	for _, q := range modes.poles {
		fast = math.Max(fast, cmplx.Abs(q))
	}
	free := freeOracle(t, sys, modes, x0)
	tt := makeTimeVector(30, 0.2/fast)
	r, err := Lsim(sys, mat.NewDense(len(tt), 1, nil), tt, x0)
	if err != nil {
		t.Fatalf("Lsim: %v", err)
	}
	worst := 0.0
	for s := 1; s < len(tt); s++ {
		worst = math.Max(worst, stepOracleRel(r.Y.At(0, s), free(0, tt[s])))
	}
	if worst > 1e-7 {
		t.Errorf("Lsim from x0 on the fast grid off the pencil by %.3g", worst)
	}
}
