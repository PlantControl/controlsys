package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// cheapFeedthroughPlant is the generalized plant of a random non-symmetric
// plant (A, B, C): w = [process noise (n); sensor noise (p)], z = [x; ρu],
// y = Cx + σv.
func cheapFeedthroughPlant(t *testing.T, A, B, C *mat.Dense, rho, sig float64) *System {
	t.Helper()
	n, m := B.Dims()
	p, _ := C.Dims()
	Bp := mat.NewDense(n, n+p+m, nil)
	setBlock(Bp, 0, 0, eyeDense(n))
	setBlock(Bp, 0, n+p, B)
	Cp := mat.NewDense(n+m+p, n, nil)
	setBlock(Cp, 0, 0, eyeDense(n))
	setBlock(Cp, n+m, 0, C)
	Dp := mat.NewDense(n+m+p, n+p+m, nil)
	for i := range m {
		Dp.Set(n+i, n+p+i, rho)
	}
	for i := range p {
		Dp.Set(n+m+i, n+i, sig)
	}
	P, err := New(A, Bp, Cp, Dp, 0)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// TestHinfSynCheapFeedthroughBelowH2Loop covers a nearly noise-free
// measurement (σ = 2.4e-8, D21D21ᵀ ≈ 6e-16) and cheap control (ρ = 2.7e-8)
// (BMNGOS, census rows rnd 94 and rnd 268). The Hamiltonian forms R⁻¹ ≈
// 1e15 in one direction, its stable subspace was garbage at every γ (Y
// asymmetric by 7e5, or n+1 stable eigenvalues) and HinfSyn returned
// ErrGammaNotAchievable, though the H2Syn controller already gives a
// finite ‖T_zw‖∞; the extended pencil solves it.
func TestHinfSynCheapFeedthroughBelowH2Loop(t *testing.T) {
	cases := []struct {
		name     string
		A, B, C  []float64
		n, m, p  int
		rho, sig float64
	}{
		{"cheap sensor", []float64{-0.8818239319995558, 1.0272090190053553, 0.17080353744374277, 1.0199632631482856, 0.5471366288591906, -0.6962689818825372, 1.6532722336356085, -1.6694170964510877, 0.2683103440482473},
			[]float64{2.191148570418685, -0.012940693613284565, -0.7743770444890247},
			[]float64{-0.05789755742046765, -0.5792002632564541, 0.5428072282660892},
			3, 1, 1, 1, 2.4082739736756504e-08},
		{"cheap control", []float64{-0.6120352698081422, -0.3059705360037235, 0.7573781845944288, -0.8874187404739076, 0.6832272508708469, 0.48756825167810547, -1.1707321915840678, 0.914789396591708, 0.4628241523030431, -0.3230200407739814, -0.7566740797651874, -0.03170667286827128, 0.19676755251267175, -0.2727247459842097, -0.23622401793167214, 0.9235663569834602, 0.19516097947306826, -0.6431538109588982, 1.1330859404689786, 0.6052751334814033, 0.026904410045013316, -0.41297938314337507, 0.28277871669490917, 0.002279668920271427, -1.3836730041213707},
			[]float64{-1.1500095822745635, 1.3587484086660022, 0.345254018411797, 1.1077977465664468, -0.43658659303098446, 0.821067610061941, -0.04364339699426634, 0.4937138562428575, 0.1401389675645014, 0.7439031718177748},
			[]float64{-0.7520738484167243, 2.456818810171617, -0.14803479313482126, 1.0531630896803783, 0.22262326276930278},
			5, 2, 1, 2.7466024550722615e-08, 2.126560824222357e-06},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			P := cheapFeedthroughPlant(t, mat.NewDense(tc.n, tc.n, tc.A), mat.NewDense(tc.n, tc.m, tc.B), mat.NewDense(tc.p, tc.n, tc.C), tc.rho, tc.sig)
			h2, err := H2Syn(P, tc.p, tc.m)
			if err != nil {
				t.Fatal(err)
			}
			h2Norm, _, err := HinfNorm(h2.CL)
			if err != nil {
				t.Fatal(err)
			}
			res, err := HinfSyn(P, tc.p, tc.m)
			if err != nil {
				t.Fatalf("HinfSyn: %v (H2Syn loop ‖·‖∞ = %g)", err, h2Norm)
			}
			if !(res.GammaOpt <= h2Norm) {
				t.Errorf("GammaOpt = %.10g above the H2Syn loop's %.10g", res.GammaOpt, h2Norm)
			}
			cl, err := LFT(P, res.K, LFTFeedback{Nu: tc.m, Ny: tc.p})
			if err != nil {
				t.Fatal(err)
			}
			if ok, err := cl.IsStable(); err != nil || !ok {
				t.Fatalf("closed loop not stable (%v)", err)
			}
			norm, _, err := HinfNorm(cl)
			if err != nil {
				t.Fatal(err)
			}
			if !(norm <= res.GammaOpt*(1+1e-9)) {
				t.Errorf("‖CL‖∞ = %.10g above GammaOpt = %.10g", norm, res.GammaOpt)
			}
		})
	}
}

// TestHinfPencilSolveMatchesHamiltonian checks the extended pencil against
// the Hamiltonian solve on a well-conditioned plant, X and Y at two γ.
func TestHinfPencilSolveMatchesHamiltonian(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 2, 0.5, -0.3, -0.7, 1.2, 0.4, -1.1, -2})
	B := mat.NewDense(3, 1, []float64{1, -0.5, 0.3})
	C := mat.NewDense(1, 3, []float64{0.2, 1, -0.4})
	P := cheapFeedthroughPlant(t, A, B, C, 0.5, 0.3)
	gp, err := partitionGeneralizedPlant("HinfSyn", P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	r := newHinfD11ZeroRiccatis(gp)
	for _, g := range []float64{20, 1e3} {
		X, Y, err := r.solve(g)
		if err != nil {
			t.Fatalf("γ = %g: %v", g, err)
		}
		Xp, err := r.px.atGamma(g, r.m1).pencilSolve()
		if err != nil {
			t.Fatalf("γ = %g: pencil X: %v", g, err)
		}
		Yp, err := r.py.atGamma(g, r.p1).pencilSolve()
		if err != nil {
			t.Fatalf("γ = %g: pencil Y: %v", g, err)
		}
		for _, c := range []struct {
			name string
			h, p *mat.Dense
		}{{"X", X, Xp}, {"Y", Y, Yp}} {
			var d mat.Dense
			d.Sub(c.h, c.p)
			if e := mat.Norm(&d, 1) / mat.Norm(c.h, 1); !(e < 1e-10) {
				t.Errorf("γ = %g: %s pencil vs Hamiltonian relative difference %g", g, c.name, e)
			}
		}
	}
}
