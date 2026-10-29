package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffGridCase is a 3-state descriptor with one internal delay τ = 0.3
// whose pencil, in generalized Schur form, has the slow poles −0.6 and −1.75
// and the stiff pole −0.9·10^k, so cond(E) grows as 10^k while the slow
// dynamics stay put. The orthogonal Q and Z are fixed rotations.
func stiffGridCase(t *testing.T, k float64) *System {
	t.Helper()
	rot := func(i, j int, th float64) *mat.Dense {
		r := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
		c, s := math.Cos(th), math.Sin(th)
		r.Set(i, i, c)
		r.Set(j, j, c)
		r.Set(i, j, -s)
		r.Set(j, i, s)
		return r
	}
	orth := func(a, b, c float64) *mat.Dense {
		var q, tmp mat.Dense
		tmp.Mul(rot(0, 1, a), rot(1, 2, b))
		q.Mul(&tmp, rot(0, 2, c))
		return &q
	}
	Q, Z := orth(0.7, -0.4, 1.1), orth(-0.3, 0.9, 0.5)
	Ta := mat.NewDense(3, 3, []float64{-0.6, 0.3, -0.2, 0, -1.4, 0.25, 0, 0, -0.9})
	Te := mat.NewDense(3, 3, []float64{1, 0.2, -0.1, 0, 0.8, 0.3, 0, 0, math.Pow(10, -k)})
	var A, E, tmp mat.Dense
	tmp.Mul(Q, Ta)
	A.Mul(&tmp, Z)
	tmp.Mul(Q, Te)
	E.Mul(&tmp, Z)
	sys, err := NewDescriptor(&A, mat.NewDense(3, 1, []float64{1, -0.5, 0.7}), mat.NewDense(1, 3, []float64{0.6, -0.4, 0.9}),
		mat.NewDense(1, 1, []float64{0.1}), &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.3}, mat.NewDense(3, 1, []float64{0.4, -0.3, 0.2}), mat.NewDense(1, 3, []float64{0.3, 0.5, -0.4}),
		mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{0.2}), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

// stiffGridExponents are the exponents k of stiffGridCase, cond(E) from
// about 1e6 to 1e15.
func stiffGridExponents() []float64 {
	var ks []float64
	for k := 6.0; k <= 15; k += 0.5 {
		ks = append(ks, k)
	}
	return ks
}

// The stiff internal-delay route moves with the grid (43H73S): Lsim takes
// the ZOH discretization of the pencil's parts only where the LU fold's
// cond(E)·ε error exceeds that discretization's dt·ρ, ρ the largest slow
// pole modulus, so the crossover cond(E) grows a hundredfold from dt = 1e-4
// to dt = 1e-2. A fixed cond(E) gate does not move.
func TestLsimInternalDelayStiffDescriptorRouteMovesWithGridSpacing(t *testing.T) {
	crossover := func(dt float64) float64 {
		for _, k := range stiffGridExponents() {
			sys := stiffGridCase(t, k)
			if _, _, ok := stiffLFTSampled(sys, nil, func(*System) float64 { return dt }); ok {
				var lu mat.LU
				lu.Factorize(sys.E)
				return lu.Cond()
			}
		}
		return math.Inf(1)
	}
	fine, coarse := crossover(1e-4), crossover(1e-2)
	t.Logf("crossover cond(E): %.3g at dt 1e-4, %.3g at dt 1e-2", fine, coarse)
	if shift := math.Log10(coarse / fine); !(shift >= 1.5 && shift <= 2.5) {
		t.Errorf("crossover moved by 10^%.3g from dt 1e-4 to 1e-2, want about 10^2", shift)
	}
}

// Lsim of stiffGridCase from cond(E) 1e6 to 1e15 at dt 1e-2 and 1e-4 keeps
// the better of the two routes against the 320-bit method of steps
// (43H73S): below the crossover the exact chain of the fold keeps its
// cond(E)·ε accuracy (1e-6 where the ZOH discretization of the parts was
// off by 1e-3 at dt 1e-2), above it the parts stay within their dt·ρ
// (lftStepTol on the fine grid) where the fold was off by up to 0.1.
func TestLsimInternalDelayStiffDescriptorKeepsBetterRoute(t *testing.T) {
	skipSlowUnderRace(t)
	times := []float64{0.11, 0.29, 0.43, 0.61, 0.79, 0.97, 1.13}
	for _, k := range stiffGridExponents() {
		sys := stiffGridCase(t, k)
		if !matlabRankFull(t, sys.E) {
			continue
		}
		var lu mat.LU
		lu.Factorize(sys.E)
		cond := lu.Cond()
		o := newLFTStepOracle(t, sys)
		want := make([]float64, len(times))
		for i, tt := range times {
			want[i] = o.response(0, nil, tt)[0]
		}
		for _, dt := range []float64{1e-2, 1e-4} {
			n := int(math.Round(1.2 / dt))
			tt, u := lsimStepGrid(n, 1, 1.2, []lsimStep{{0, 0, 1}})
			r, err := Lsim(sys, u, tt, nil)
			if err != nil {
				t.Fatalf("k %g dt %g: Lsim: %v", k, dt, err)
			}
			worst := 0.0
			for i, at := range times {
				worst = math.Max(worst, stepOracleRel(r.Y.At(0, int(math.Round(at/dt))), want[i]))
			}
			tol := math.Max(lftStepTol, 1.75*dt)
			if fold := cond * 0x1p-52; fold < 1.75*dt {
				tol = math.Max(1e-6, 10*fold)
			}
			if worst > tol {
				t.Errorf("k %g (cond(E) %.3g) dt %g: Lsim off the pencil by %.3g, want <= %.3g", k, cond, dt, worst, tol)
			}
		}
	}
}

// Step and Initial of stiffGridCase at cond(E) about 1e8 sample E⁻¹A by
// the exact chain on their automatic grids, as before the stiff route
// (43H73S, PJIDAF): the fold's cond(E)·ε is far below the dt·ρ of the ZOH
// discretization of the parts, which was off by 1e-6.
func TestStepInternalDelayModeratelyStiffDescriptorKeepsChainAccuracy(t *testing.T) {
	sys := stiffGridCase(t, 8)
	o := newLFTStepOracle(t, sys)
	st, err := Step(sys, 1)
	if err != nil {
		t.Fatal(err)
	}
	x0 := []float64{0.7, -0.4, 1.1}
	in, err := Initial(sys, mat.NewVecDense(3, x0), 1)
	if err != nil {
		t.Fatal(err)
	}
	worstStep, worstInitial := 0.0, 0.0
	for _, s := range lftOracleSamples(len(st.T)) {
		if !nearDelayGrid(st.T[s], o.tau) {
			worstStep = math.Max(worstStep, stepOracleRel(st.Y.At(0, s), o.response(0, nil, st.T[s])[0]))
		}
	}
	for _, s := range lftOracleSamples(len(in.T)) {
		if !nearDelayGrid(in.T[s], o.tau) {
			worstInitial = math.Max(worstInitial, stepOracleRel(in.Y.At(0, s), o.response(-1, x0, in.T[s])[0]))
		}
	}
	t.Logf("Step %.3g, Initial %.3g", worstStep, worstInitial)
	if worstStep > 1e-7 || worstInitial > 1e-7 {
		t.Errorf("Step off the pencil by %.3g, Initial by %.3g, want <= 1e-7", worstStep, worstInitial)
	}
}
