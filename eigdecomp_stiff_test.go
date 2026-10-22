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
// stiff singular value 28× the rounding n·ε·σ₁(E). The staircase mixed the
// stiff direction with the null space of E: the deflated 4×4 block's least
// singular value (6.7e-15) fell below the rounding and the stiff pole
// −2.43e12 was deflated as a second infinite one. Oracle: the construction
// A = U·diag(fd)·V, E = U·diag(wd)·V (fd/wd), which the rounding of the
// stored data moves by 0.4% for the stiff pole; eig(A⁻¹E) cannot resolve
// it from the infinite one (both μ ~ 5e-13).
func TestPoles_SingularEStiffPoleNearRounding(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{0.1504492256275744, 0.1377258460818396, -1.2317343679578239, -0.3741054825398291, 0.2873747457356886,
		0.2662097129176385, 0.027093495871395357, -1.1059106554651639, 0.04599782619700756, 0.40670503227182914,
		2.0265780779417257, -4.395559947663698, -1.516136645303072, 3.777062128610673, 1.3463278922787691,
		-0.04535859949670607, 0.17054162228026792, 0.10674723493971029, -0.04948417629874526, -0.07379105092426315,
		-0.6160081623053199, 2.5087596469598052, -0.2941622636595208, -1.0872747407671977, 0.40779093300884933})
	E := mat.NewDense(5, 5, []float64{-0.5632784077157112, 1.7373019741672144, -0.19024283017603316, -0.11835988290709326, 2.0869804584540472,
		-0.4754543806471195, 3.28751057800169, 0.6096661638550409, 0.9687121572918135, 2.014052382830856,
		-0.24756325741402566, -1.3223035364711537, -1.0979189881674478, -1.388677361140673, 0.6189440099556822,
		-0.4713535828032018, 4.482796018731243, 0.9441485032138269, 1.5267021165016628, 2.154040671232813,
		-0.5042394935945068, 4.599619872489651, 0.2375335506907937, 0.9299306520742701, 2.229538117716564})
	want := []complex128{-2.4353296521794673e12, 0.06304554897610047, 0.4414607622361663, -0.11570019305424666}
	sys, err := NewDescriptor(A, mat.NewDense(5, 1, nil), mat.NewDense(1, 5, nil), mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, "stiff near rounding", poles, want, 0.3)
}
