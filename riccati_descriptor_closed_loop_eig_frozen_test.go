package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Descriptor models frozen at full precision from the BGKVXR censuses
// (generators differ in the last bits between arm64 and amd64, so each
// is frozen from the arch where its row was found). Eig of a descriptor
// result must hold the closed-loop eigenvalues within the oracle bound of
// assertClosedLoopEigExact, or the result carries ErrAccuracyUnverified.
// "Old" is the solver's E⁻¹(A − BK) Eig before the exit recompute, "new"
// the recomputed one, ratios to the oracle bound; λ∞ is the mode near
// 1/σ_min(E).

type frozenDescriptor struct {
	A, B, C, D, E *mat.Dense
	dt            float64
}

func (f *frozenDescriptor) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(f.A, f.B, f.C, f.D, f.E, f.dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// assertEigCertifiedOrUnverified checks the descriptor Eig contract on
// the pencil (A − BK, E), and with certified that the error is nil.
func assertEigCertifiedOrUnverified(t *testing.T, name string, res *RiccatiResult, err error, certified bool, A, B, K, E func(*RiccatiResult) *mat.Dense) {
	t.Helper()
	switch {
	case err == nil:
		assertClosedLoopEigExact(t, name, A(res), B(res), K(res), E(res), res.Eig)
	case errors.Is(err, ErrAccuracyUnverified) && res != nil && !certified:
	default:
		t.Fatalf("%s: %v", name, err)
	}
}

func fixed(M *mat.Dense) func(*RiccatiResult) *mat.Dense {
	return func(*RiccatiResult) *mat.Dense { return M }
}

func gain(r *RiccatiResult) *mat.Dense { return r.K }

// Lqr/Dlqr on near-singular E. z5a seed 561 (dt 0, E[4][4] = 5e-15,
// λ∞ ≈ −5.6e14): old nil, four O(1) modes up to 22% off (1.1e12× the
// bound; amd64 36%, 4.1e11×); the first two-stage rework went to
// Accuracy 1 (β = 0 on the balanced pencil). Now the unbalanced stage
// certifies −7.55497, −0.554973±0.805579i, −0.83895 (oracle to 8e-3 of the
// bound, both arches). Random seed 95 (dt 1, modes 0.340264, 2.41355e-9,
// 6.18684e-7): the balanced-only recompute put 2.41355e-9 at 2.39e-9 (23×
// the bound, nil); now unbalanced, 2.41343e-9 native (0.12×), 2.41356e-9
// amd64 (0.04×); old 0.09× / 0.03×. z5a seed 689 (dt 1, oracle
// 0.124994±0.0319542i native, 0.235816 and −0.0419018 amd64): balanced
// stage, 0.02× / 0.07×.
func TestLqrDescriptorClosedLoopEigNearSingularE(t *testing.T) {
	for _, c := range []struct {
		name string
		f    *frozenDescriptor
	}{
		{"z5a/561", mod561},
		{"rand/95", mod95},
		{"z5a/689", mod689},
	} {
		sys := c.f.system(t)
		n, m, _ := sys.Dims()
		solve := Lqr
		if !sys.IsContinuous() {
			solve = Dlqr
		}
		res, err := solve(sys.A, sys.B, eyeDense(n), eyeDense(m), &RiccatiOpts{E: sys.E})
		assertEigCertifiedOrUnverified(t, c.name, res, err, true, fixed(sys.A), fixed(sys.B), gain, fixed(sys.E))
	}
}

// Lqi on near-singular E. Rotated seed 123 (dt 0, amd64 model; native
// does not converge): old nil, −1.70242 and −0.441838 against oracle
// −1.76337 and −0.60539 (27%, 3.4e13× the bound); the two-stage rework
// went to Accuracy 1 (β = 0); now unbalanced certifies −1.76337,
// −0.60539 (0.11×). Random seed 108 (dt 0, κ(E) ~ 1e8, λ∞ = −1.63742e8,
// oracle −0.403108, −0.723006, −4.59874): balanced QZ put −0.403108 at
// −0.3849 (4.5%, nil); now unbalanced, −0.403109 (0.01×) native,
// −0.403098 (0.06×) amd64; old 0.003× / 0.08×.
func TestLqiDescriptorClosedLoopEigNearSingularE(t *testing.T) {
	for _, c := range []struct {
		name      string
		f         *frozenDescriptor
		certified bool
	}{
		{"rot/123", modrot123, false},
		{"rand/108", mod108, true},
	} {
		sys := c.f.system(t)
		n, m, p := sys.Dims()
		res, err := Lqi(sys, eyeDense(n+p), eyeDense(m), nil)
		if !c.certified && errors.Is(err, ErrNotConverged) {
			continue
		}
		Aa, Ba := lqiAugmentation(sys)
		Ea := augmentedDescriptor(sys.E, p)
		assertEigCertifiedOrUnverified(t, c.name, res, err, c.certified, fixed(Aa), fixed(Ba), gain, fixed(Ea))
	}
}

// Lqr on Lqi's augmented pencil with the integrator rows scaled by 2^k
// (amd64 models; native does not converge). Rotated seed 123, k = 0: as
// TestLqiDescriptorClosedLoopEigNearSingularE. Rotated seed 176, k = −4:
// old nil, −1.1861±0.346792i against −1.18637±0.394353i (3.8%, 5.8e12×);
// two-stage rework Accuracy 1 (β = 0); now unbalanced, −1.18637±0.394353i
// (0.12×).
func TestLqrAugmentedDescriptorClosedLoopEigRowScaled(t *testing.T) {
	for _, c := range []struct {
		name string
		f    *frozenDescriptor
		k    int
	}{
		{"rot/123/k0", modrot123, 0},
		{"rot/176/k-4", modrot176, -4},
	} {
		sys := c.f.system(t)
		n, m, p := sys.Dims()
		Aa, Ba := lqiAugmentation(sys)
		Ea := augmentedDescriptor(sys.E, p)
		for _, M := range []*mat.Dense{Ea, Aa, Ba} {
			raw := M.RawMatrix()
			for i := n; i < raw.Rows; i++ {
				for j := range raw.Cols {
					raw.Data[i*raw.Stride+j] = math.Ldexp(raw.Data[i*raw.Stride+j], c.k)
				}
			}
		}
		res, err := Lqr(Aa, Ba, eyeDense(n+p), eyeDense(m), &RiccatiOpts{E: Ea, plantE: sys.E, augmentedE: true})
		if errors.Is(err, ErrNotConverged) {
			continue
		}
		assertEigCertifiedOrUnverified(t, c.name, res, err, false, fixed(Aa), fixed(Ba), gain, fixed(Ea))
	}
}

// Kalman Eig on the primal pencil (A − LC, E). zw46 mid seed 1879, cross
// (dt 1, E[3][3] = 5.7e-9, oracle 9.40262e-8, 0.269426±0.256244i,
// 0.646441): QZ of the dual (Aᵀ − CᵀLᵀ, Eᵀ), balanced, put the O(1) modes
// 2e-3 off and the small one at 1.1e-3, nil; the primal is within 1.5e-6.
// Now unbalanced primal: 5.35949e-6, 0.269424±0.256239i, 0.646439 native
// (0.03×), 2.33827e-6 … amd64 (0.01×); old 0.01× / 0.006×.
func TestKalmanDescriptorClosedLoopEigPrimalPencil(t *testing.T) {
	sys := modzw.system(t)
	_, m, _ := sys.Dims()
	R := rzw
	res, err := Kalman(sys, eyeDense(m), R, nil, nil)
	assertEigCertifiedOrUnverified(t, "zw46mid/1879/1", res, err, true, fixed(sys.A), gain, fixed(sys.C), fixed(sys.E))
}

// Lqg's Kalman stage on z5a seed 689 (dt 1, E = [[0.856, 1.49], [0,
// 8.1e-15]], modes conditioned ~1e13; oracle 0.0307485, −0.153853 amd64).
// Old nil, 0.0270776 (12% off, 0.003× the bound at that conditioning); the
// two-stage rework's balanced dual QZ gave [−0.41, 6.08], unstable, so
// Lqg's Xc and Xf went to Accuracy 1 (amd64). Now the primal unbalanced
// stage certifies 0.025974, −0.147588 amd64 (0.004×); native keeps the
// extended solve's Eig.
func TestLqgDescriptorNearSingularEKalmanEig(t *testing.T) {
	sys := mod689.system(t)
	n, m, p := sys.Dims()
	if _, err := Lqg(sys, eyeDense(n+m), eyeDense(n+p), nil); err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("Lqg: %v", err)
	}
	k, err := NewDescriptor(sys.A, eyeDense(n), sys.C, mat.NewDense(p, n, nil), sys.E, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Kalman(k, eyeDense(n), eyeDense(p), nil, nil)
	assertEigCertifiedOrUnverified(t, "z5a/689", res, err, false, fixed(sys.A), gain, fixed(sys.C), fixed(sys.E))
}

var (
	mod561 = &frozenDescriptor{
		A: mat.NewDense(5, 5, []float64{
			-0.9522843891900381, -0.42279674759203034, 0.9857024476352265, -1.9341957649682797, -0.4783124229796969,
			-0.5097353553052281, -0.45135053683151827, 2.3091684742429353, 0.939748966928529, -0.17765990881242064,
			-0.48470305124329327, -0.06780517934011243, 2.1768295178108157, -2.4068935555916133, 0.7500797374740426,
			0.5740357674523373, -1.575299581715216, 0.26054924175694494, 0.29567642426230833, -0.26908491323217865,
			-0.0940533902952085, 1.112402807055542, -0.6312564943223631, 1.166075898794151, 0.14153750621318817,
		}),
		B: mat.NewDense(5, 1, []float64{
			-0.7472995658419674,
			0.48307652721041583,
			0.15912531279965475,
			-1.499233280795596,
			0.27057360729865165,
		}),
		C: mat.NewDense(1, 5, []float64{
			1.4805974415747225, 0.4456768910753349, -0.29750675569286633, 0.12510876695599438, -0.642248422233061,
		}),
		D: mat.NewDense(1, 1, []float64{
			0,
		}),
		E: mat.NewDense(5, 5, []float64{
			0.9749386684172331, 1.1540815926566266, 1.3710941527795304, 0.19819896834698358, 0.12089447015021132,
			0, 0.870587809849001, -0.6169319638177064, -0.34601397671189804, -1.618256447538893,
			0, 0, 1.4693448870021046, 0.9640793260077957, -1.2446423904489523,
			0, 0, 0, 0.8340008725711803, 0.19672024666602228,
			0, 0, 0, 0, 4.9647292629631965e-15,
		}),
		dt: 0,
	}
	mod95 = &frozenDescriptor{
		A: mat.NewDense(3, 3, []float64{
			-0.00018634393033704815, 0.00023302025584597422, -0.00022728307002223858,
			0.0016862041118490816, -0.00031615007969770136, 0.001885753956330798,
			-9.52829734251095e-05, 0.0013231111975992712, 0.00029376596376147647,
		}),
		B: mat.NewDense(3, 2, []float64{
			-0.06254209413086631, 1.876203672518086,
			0.670491515893858, 0.5068333395090372,
			0.9366333545346344, -1.913913519897476,
		}),
		C: mat.NewDense(1, 3, []float64{
			-0.051565914902721, -0.5434658566706543, 0.07132951142513239,
		}),
		D: mat.NewDense(1, 2, []float64{
			0, 0,
		}),
		E: mat.NewDense(3, 3, []float64{
			0.0006706359764841194, 0.00030445774671806965, 0.0006660435483508736,
			0.0014653236226527717, 0.0007258273963364031, 0.0011962410897346759,
			0.0003451940242926899, 0.00013707301841872304, -0.0021687301170549337,
		}),
		dt: 1,
	}
	mod689 = &frozenDescriptor{
		A: mat.NewDense(2, 2, []float64{
			0.7291047655906884, -0.33541885843612806,
			0.5020957065984277, -0.801902475718303,
		}),
		B: mat.NewDense(2, 1, []float64{
			-0.04153662685277393,
			0.43336686389019774,
		}),
		C: mat.NewDense(1, 2, []float64{
			-2.380093258999346, -0.6115356250660309,
		}),
		D: mat.NewDense(1, 1, []float64{
			0,
		}),
		E: mat.NewDense(2, 2, []float64{
			0.8557346100140317, 1.490249054984718,
			0, 8.1369069928416e-15,
		}),
		dt: 1,
	}
	modrot123 = &frozenDescriptor{
		A: mat.NewDense(2, 2, []float64{
			-0.8516035809129043, -0.6372623746825746,
			1.745971558139388, 0.23475937561828353,
		}),
		B: mat.NewDense(2, 1, []float64{
			-0.009389915208062427,
			-1.0558982368056928,
		}),
		C: mat.NewDense(1, 2, []float64{
			-1.769169374040846, -0.28710968980944385,
		}),
		D: mat.NewDense(1, 1, []float64{
			0,
		}),
		E: mat.NewDense(2, 2, []float64{
			0.443064967623597, 0.4531709872692627,
			-0.5853160155518413, -0.5986666877654804,
		}),
		dt: 0,
	}
	modrot176 = &frozenDescriptor{
		A: mat.NewDense(2, 2, []float64{
			0.595042694472534, 1.1081591138009375,
			-0.1563077776512692, 1.027758503461913,
		}),
		B: mat.NewDense(2, 2, []float64{
			0.7127203744031145, -0.032225678415617354,
			0.5744247157341817, 1.2006860703891702,
		}),
		C: mat.NewDense(1, 2, []float64{
			0.1307761023039179, -1.7041854940991366,
		}),
		D: mat.NewDense(1, 2, []float64{
			0, 0,
		}),
		E: mat.NewDense(2, 2, []float64{
			0.2197650201772829, -0.1627386721885619,
			0.8937963173993386, -0.6618670513772227,
		}),
		dt: 0,
	}
	mod108 = &frozenDescriptor{
		A: mat.NewDense(3, 3, []float64{
			-2.53740033612679e-08, -6.46068356274542e-09, -1.658147647891154e-09,
			6.767729903837605e-10, -2.5382457602390475e-09, 2.4185386422263756e-09,
			-3.4874882405317915e-09, 2.1957420806551397e-09, 2.3011307261784842e-08,
		}),
		B: mat.NewDense(3, 1, []float64{
			-0.3794297893607512,
			-0.6738466923307098,
			1.2123103515203646,
		}),
		C: mat.NewDense(1, 3, []float64{
			-0.25756507261448014, 0.4663671198262326, -0.7848981525458362,
		}),
		D: mat.NewDense(1, 1, []float64{
			0,
		}),
		E: mat.NewDense(3, 3, []float64{
			-1.1065456111466143e-08, -5.787712235718012e-09, -1.849635215445077e-08,
			1.4842611146098729e-08, 4.870646605331075e-09, 8.214100328390268e-09,
			-4.069992652709058e-09, -5.999497913765765e-09, 2.663894318584155e-09,
		}),
		dt: 0,
	}
	modzw = &frozenDescriptor{
		A: mat.NewDense(4, 4, []float64{
			0.6095818089334863, -0.715670297430663, -0.6664781517777136, -1.9857108422098215,
			-0.5566957595712289, 1.6689537341172098, 0.3521774391087912, 0.6227340789709337,
			1.6274587290065299, 0.5027848268712358, 2.366255124162958, 0.16081160762093832,
			0.18418526055216594, 0.6448778561531627, -0.4155196217488999, -0.9047911409367848,
		}),
		B: mat.NewDense(4, 1, []float64{
			0.08038028118954599,
			0.2357836023914004,
			-0.9493059293957877,
			0.2710946106537836,
		}),
		C: mat.NewDense(1, 4, []float64{
			0.08038028118954599, 0.2357836023914004, -0.9493059293957877, 0.2710946106537836,
		}),
		D: mat.NewDense(1, 1, []float64{
			0,
		}),
		E: mat.NewDense(4, 4, []float64{
			0.8146207763634995, -0.5752870431374131, 0.34771749522917106, -1.2720423137220065,
			0, 0.6501829710063823, 1.3491947690689827, 1.1878925504828524,
			0, 0, 1.39761103279937, 1.7798408870175066,
			0, 0, 0, 5.65697099137665e-09,
		}),
		dt: 1,
	}
	rzw = mat.NewDense(1, 1, []float64{
		2.4129949005810807,
	})
)
