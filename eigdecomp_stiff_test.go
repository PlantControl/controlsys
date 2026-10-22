package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// eigAinvEPoles returns 1/μ for the eigenvalues μ of A⁻¹E, the poles of
// the pencil (A, E) for nonsingular A and E.
func eigAinvEPoles(t *testing.T, A, E *mat.Dense) []complex128 {
	t.Helper()
	var ae mat.Dense
	if err := ae.Solve(A, E); err != nil {
		t.Fatal(err)
	}
	var eig mat.Eigen
	if !eig.Factorize(&ae, mat.EigenNone) {
		t.Fatal("eig failed")
	}
	mu := eig.Values(nil)
	out := make([]complex128, len(mu))
	for i, m := range mu {
		out[i] = 1 / m
	}
	return out
}

func stableSet(poles []complex128) bool {
	for _, p := range poles {
		if real(p) >= 0 {
			return false
		}
	}
	return true
}

// TestPoles_NonsingularEStiffPoleKept: a nonsingular E with
// σ_min/σ₁ ~ 1e-13, below the staircase rank tolerance 1e3·n·ε·σ₁ but far
// above the rounding n·ε·σ₁, has no infinite mode; its stiff finite pole
// must stay a pole and keep IsStable false. Deflating it as infinite
// returned {-0.74433} and IsStable = true for the first 2×2 case. Oracles:
// eig(A⁻¹E) and the construction fd/wd. IOAGOI.
func TestPoles_NonsingularEStiffPoleKept(t *testing.T) {
	cases := []struct {
		a, e []float64
		want []complex128
	}{
		{
			a:    []float64{-0.07919577482027212, 0.36720899300886656, 0.14872717269986616, -0.010044452932383471},
			e:    []float64{0.12984082748909265, -0.48709217347530465, -0.01661451681212843, 0.06232863161637024},
			want: []complex128{1.022572247629947e12, -0.7443309751704783},
		},
		{
			a:    []float64{-0.13059864906094776, -0.3549041694151974, 0.033823735072008206, 0.5508408001369759},
			e:    []float64{-0.7951516267061715, 0.1226714777537877, -0.5363160965815704, 0.08273980194564767},
			want: []complex128{-1.7101e12, 0.093168},
		},
	}
	for i, c := range cases {
		A, E := mat.NewDense(2, 2, c.a), mat.NewDense(2, 2, c.e)
		label := fmt.Sprintf("case %d", i)
		oracle := eigAinvEPoles(t, A, E)
		assertEigSetNear(t, label+" oracle", oracle, c.want, 1e-3)
		sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{0.3, -0.8}), mat.NewDense(1, 2, []float64{1.1, 0.4}), mat.NewDense(1, 1, nil), E, 0)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, label, poles, oracle, 1e-3)
		if st, err := sys.IsStable(); err != nil || st {
			t.Errorf("%s: IsStable = %v, %v, want false", label, st, err)
		}
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatal(err)
		}
		sp, err := res.Stable.Poles()
		if err != nil {
			t.Fatal(err)
		}
		up, err := res.Unstable.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, label+" Stabsep", append(sp, up...), oracle, 1e-3)
		if !stableSet(sp) || len(up) != 1 || real(up[0]) < 0 {
			t.Errorf("%s: Stabsep stable %v unstable %v", label, sp, up)
		}
	}

	for seed := range uint64(2000) {
		rng := rand.New(rand.NewPCG(seed, 11))
		n := 2 + rng.IntN(4)
		fd, wd := make([]float64, n), make([]float64, n)
		for i := range n {
			fd[i], wd[i] = rng.NormFloat64(), 1+rng.Float64()
		}
		wd[0] = math.Pow(10, -14+2*rng.Float64())
		U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				U.Set(i, j, rng.NormFloat64())
				V.Set(i, j, rng.NormFloat64())
			}
		}
		var T, A, E mat.Dense
		T.Mul(U, mat.NewDiagDense(n, fd))
		A.Mul(&T, V)
		T.Mul(U, mat.NewDiagDense(n, wd))
		E.Mul(&T, V)
		var svd mat.SVD
		if !svd.Factorize(&E, mat.SVDNone) {
			t.Fatal("svd failed")
		}
		sv := svd.Values(nil)
		if sv[n-1] <= 100*float64(n)*eps()*sv[0] {
			continue
		}
		want := make([]complex128, n)
		for i := range n {
			want[i] = complex(fd[i]/wd[i], 0)
		}
		oracle := eigAinvEPoles(t, &A, &E)
		label := fmt.Sprintf("seed %d σ_min/σ₁ %.2g", seed, sv[n-1]/sv[0])
		assertEigSetNear(t, label+" oracle", oracle, want, 0.2)
		sys, err := NewDescriptor(&A, mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), &E, 0)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		assertEigSetNear(t, label, poles, oracle, 0.2)
		if st, err := sys.IsStable(); err != nil || st != stableSet(oracle) {
			t.Fatalf("%s: IsStable = %v, %v, want %v", label, st, err, stableSet(oracle))
		}
	}
}

// TestPoles_SingularEStiffPoleNearRounding: singular E (index 1) with a
// stiff singular value only 10–30× the rounding n·ε·σ₁(E). The staircase
// mixed the stiff direction with the null space of E: the deflated block's
// least singular value fell below the rounding and the stiff pole was
// deflated as a second infinite one (σ₃ of the 4×4 block of case 0 was
// 6.7e-15 for the stiff pole −2.43e12), and for case 1 IsStable flipped to
// true. Oracle: the construction A = U·diag(fd)·V, E = U·diag(wd)·V
// (fd/wd), which the rounding of the stored data moves by up to 18% for
// the stiff pole; eig(A⁻¹E) cannot resolve the stiff pole from the
// infinite one here (both μ ~ 5e-13).
func TestPoles_SingularEStiffPoleNearRounding(t *testing.T) {
	cases := []struct {
		a, e []float64
		want []complex128
	}{
		{
			a: []float64{0.1504492256275744, 0.1377258460818396, -1.2317343679578239, -0.3741054825398291, 0.2873747457356886,
				0.2662097129176385, 0.027093495871395357, -1.1059106554651639, 0.04599782619700756, 0.40670503227182914,
				2.0265780779417257, -4.395559947663698, -1.516136645303072, 3.777062128610673, 1.3463278922787691,
				-0.04535859949670607, 0.17054162228026792, 0.10674723493971029, -0.04948417629874526, -0.07379105092426315,
				-0.6160081623053199, 2.5087596469598052, -0.2941622636595208, -1.0872747407671977, 0.40779093300884933},
			e: []float64{-0.5632784077157112, 1.7373019741672144, -0.19024283017603316, -0.11835988290709326, 2.0869804584540472,
				-0.4754543806471195, 3.28751057800169, 0.6096661638550409, 0.9687121572918135, 2.014052382830856,
				-0.24756325741402566, -1.3223035364711537, -1.0979189881674478, -1.388677361140673, 0.6189440099556822,
				-0.4713535828032018, 4.482796018731243, 0.9441485032138269, 1.5267021165016628, 2.154040671232813,
				-0.5042394935945068, 4.599619872489651, 0.2375335506907937, 0.9299306520742701, 2.229538117716564},
			want: []complex128{-2.4353296521794673e12, 0.06304554897610047, 0.4414607622361663, -0.11570019305424666},
		},
		{
			a: []float64{-1.87324232855353, 0.8355088473298266, 1.0289295223401886, -0.04352035662486524,
				-2.1473110475871473, -0.16922702854556096, 0.2693878010009497, -0.3782483585468946,
				-0.9355163200304798, -1.2190795921802768, -0.13917858077207795, 0.4656069838491571,
				0.7388528850757813, -1.3782573973667014, -1.2682364112291156, -0.3186657317875352},
			e: []float64{-0.2037063922772386, -2.150249955861688, -0.7208190389052863, 0.7627882776441802,
				0.17985685159903508, 1.3072910439788994, 0.4869594535561671, -0.36774886728064565,
				1.0959303037840304, 3.506736968678482, 1.83989843274947, 0.06508450966712317,
				1.1937881551048521, 4.607379358750597, 2.2032836302614403, -0.3363538465129209},
			want: []complex128{2.2147498658254742e13, -0.5440012751152693, -0.3605282243041494},
		},
	}
	for i, c := range cases {
		n := int(math.Sqrt(float64(len(c.a))))
		A, E := mat.NewDense(n, n, c.a), mat.NewDense(n, n, c.e)
		label := fmt.Sprintf("case %d", i)
		sys, err := NewDescriptor(A, mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), E, 0)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, label, poles, c.want, 0.3)
		if st, err := sys.IsStable(); err != nil || st {
			t.Errorf("%s: IsStable = %v, %v, want false", label, st, err)
		}
	}
}
