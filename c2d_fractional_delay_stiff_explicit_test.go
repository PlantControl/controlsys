package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fohDelayed returns the exact triangle-hold (FOH) discretization of
// e^{−sτ}·G_ij(s) at z, τ = N·T + ρ, 0 ≤ ρ < T: the modified z-transform
// z^{−N}·[G(∞)·(1 − ρ(z−1)/(T·z)) + Σ Rᵢ·(−1/λᵢ + ρ(z−1)/(λᵢT·z) +
// (z−1)/(λᵢ²T·z)·(z·expm1(λᵢ(T−ρ)) + e^{λᵢ(T−ρ)}·expm1(λᵢρ))/(z − e^{λᵢT}))]
// of the sampled response to the delayed triangle input. ρ = 0 gives
// z^{−N}·foh.
func (m stepOracleModes) fohDelayed(i, j int, dt, tau float64, z complex128) complex128 {
	k := math.Floor(tau/dt + 1e-12)
	rho := tau - k*dt
	if rho <= 1e-12*dt {
		return m.foh(i, j, dt, z) * cmplx.Pow(z, complex(-k, 0))
	}
	T, r := complex(dt, 0), complex(rho, 0)
	g := m.gInf(i, j) * (1 - r*(z-1)/(T*z))
	for q, l := range m.poles {
		e, ef := cmplx.Exp(l*T), cmplx.Exp(l*complex(dt-rho, 0))
		hold := (z*complexExpm1(l*complex(dt-rho, 0)) + ef*complexExpm1(l*r)) / (z - e)
		g += m.residues[q][i][j] * (-1/l + r*(z-1)/(l*T*z) + (z-1)/(l*l*T*z)*hold)
	}
	return g * cmplx.Pow(z, complex(-k, 0))
}

// complexExpm1 is e^x − 1 without the cancellation near x = 0.
func complexExpm1(x complex128) complex128 {
	if cmplx.Abs(x) >= 0.5 {
		return cmplx.Exp(x) - 1
	}
	s, term := complex(0, 0), complex(1, 0)
	for k := 1; k < 40; k++ {
		term *= x / complex(float64(k), 0)
		s += term
	}
	return s
}

// delayedHoldOracleGap is the worst relative gap max|G_d − want|/max(1,
// |want|) of the discrete d against the exact delayed ZOH or FOH of modes,
// channel delay tau[i][j], at z = e^{jθ}, θ ∈ {0.05, 0.5, 2, 3}.
func delayedHoldOracleGap(t *testing.T, d *System, modes stepOracleModes, method C2DMethod, dt float64, tau [][]float64) float64 {
	t.Helper()
	_, m, p := d.Dims()
	worst := 0.0
	for _, th := range []float64{0.05, 0.5, 2, 3} {
		z := cmplx.Exp(complex(0, th))
		g, err := d.EvalFr(z)
		if err != nil {
			t.Fatalf("EvalFr: %v", err)
		}
		for i := range p {
			for j := range m {
				want := modes.zohDelayed(i, j, dt, tau[i][j], z)
				if method == C2DMethodFOH {
					want = modes.fohDelayed(i, j, dt, tau[i][j], z)
				}
				worst = math.Max(worst, cmplx.Abs(g[i][j]-want)/math.Max(1, cmplx.Abs(want)))
			}
		}
	}
	return worst
}

func mustOracleModes(t *testing.T, sys *System) stepOracleModes {
	t.Helper()
	pol, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, pol)
	if !ok {
		t.Fatal("oracle modes did not converge")
	}
	return modes
}

// heldPulseResponse returns y[k], k < K, of x' = A·x + b·u, y = c·x + d·u
// for the unit pulse u[0] = 1 held by ZOH (rectangle on [τ, τ+T)) or FOH
// (triangle on [τ−T, τ+T]), by Simpson quadrature of the convolution with
// e^{A·t} = V·diag(e^{λt})·V⁻¹ of the given modes: independent of the
// partial fractions of the oracles and of C2D.
func heldPulseResponse(lambda []float64, v, vinv *mat.Dense, b, c []float64, d float64, foh bool, dt, tau float64, K int) []float64 {
	n := len(lambda)
	w := make([]float64, n)
	for i := range n {
		cv, wb := 0., 0.
		for l := range n {
			cv += c[l] * v.At(l, i)
			wb += vinv.At(i, l) * b[l]
		}
		w[i] = cv * wb
	}
	lo, hi := tau, tau+dt
	breaks := []float64{tau, tau + dt}
	if foh {
		lo = tau - dt
		breaks = []float64{tau - dt, tau, tau + dt}
	}
	// hold on the piece around mid, so the quadrature never samples the
	// other side of a jump or kink at the piece ends.
	hold := func(s, mid float64) float64 {
		switch {
		case !foh && mid >= tau && mid < tau+dt:
			return 1
		case !foh:
			return 0
		case mid < tau:
			return 1 + (s-tau)/dt
		}
		return 1 - (s-tau)/dt
	}
	simpson := func(f func(float64) float64, a, b float64) float64 {
		const steps = 400
		h := (b - a) / steps
		sum := f(a) + f(b)
		for q := 1; q < steps; q++ {
			if q%2 == 1 {
				sum += 4 * f(a+float64(q)*h)
			} else {
				sum += 2 * f(a+float64(q)*h)
			}
		}
		return sum * h / 3
	}
	y := make([]float64, K)
	for k := range K {
		tk := float64(k) * dt
		edges := []float64{lo}
		for _, bp := range breaks {
			if bp > edges[0] && bp < math.Min(hi, tk) {
				edges = append(edges, bp)
			}
		}
		edges = append(edges, math.Min(hi, tk))
		for q := 0; q+1 < len(edges); q++ {
			a, e := edges[q], edges[q+1]
			if !(e > a) {
				continue
			}
			mid := (a + e) / 2
			for i := range n {
				y[k] += w[i] * simpson(func(s float64) float64 {
					return math.Exp(lambda[i]*(tk-s)) * hold(s, mid)
				}, a, e)
			}
		}
		if tk >= lo && tk < hi {
			y[k] += d * math.Max(0, hold(tk, tk))
		}
	}
	return y
}

// The delayed hold oracles are checked against brute force: on a non-stiff
// non-diagonal model with known modes, zohDelayed and fohDelayed match the
// z-transform of the sampled pulse response computed by quadrature, and
// C2D (one augmented exponential, accurate here) matches both. Simpson
// with 400 steps per piece is exact to ~1e-15 here (measured 1e-15).
func TestC2DFractionalInputDelayHoldOraclesMatchQuadrature(t *testing.T) {
	const dt, K = 0.1, 500
	lambda := []float64{-1.3, -2.7}
	v := mat.NewDense(2, 2, []float64{1, 0.6, -0.4, 1})
	var vinv mat.Dense
	if err := vinv.Inverse(v); err != nil {
		t.Fatal(err)
	}
	var a mat.Dense
	a.Mul(v, mat.NewDense(2, 2, []float64{lambda[0], 0, 0, lambda[1]}))
	a.Mul(&a, &vinv)
	b, c, d := []float64{0.7, -1.1}, []float64{0.9, 0.35}, 0.12
	sys, err := New(&a, mat.NewDense(2, 1, b), mat.NewDense(1, 2, c), mat.NewDense(1, 1, []float64{d}), 0)
	if err != nil {
		t.Fatal(err)
	}
	modes := mustOracleModes(t, sys)
	for _, tau := range []float64{0.037, 0.137, 0.25} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
			y := heldPulseResponse(lambda, v, &vinv, b, c, d, method == C2DMethodFOH, dt, tau, K)
			sd := sys.Copy()
			sd.InputDelay = []float64{tau}
			disc, err := sd.C2D(dt, C2DOptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			g := delayedHoldOracleGap(t, disc, modes, method, dt, [][]float64{{tau}})
			worst := 0.0
			for _, th := range []float64{0.05, 0.5, 2, 3} {
				z := cmplx.Exp(complex(0, th))
				var brute complex128
				for k := K - 1; k >= 0; k-- {
					brute = brute/z + complex(y[k], 0)
				}
				want := modes.zohDelayed(0, 0, dt, tau, z)
				if method == C2DMethodFOH {
					want = modes.fohDelayed(0, 0, dt, tau, z)
				}
				worst = math.Max(worst, cmplx.Abs(brute-want)/math.Max(1, cmplx.Abs(want)))
			}
			if !(worst <= 1e-12) || !(g <= 1e-13) {
				t.Errorf("τ=%g %s: oracle off quadrature by %.3g, C2D off oracle by %.3g", tau, method, worst, g)
			}
		}
	}
}

// RR43HG: C2D of stiff triangular explicit models with a fractional input
// delay keeps the slow mode. Before the fractional-delay ZOH kernel took
// the stiff modal hold (d728012), the single exponential of [A B; 0 0]·h
// rounded the slow mode to ε·|λ_fast|·h: on 4140671 the ZOH literal was
// 2.4e-6 (ZOH) and 1.2e-3 (FOH) off its slow twin, the FOH literal 6.9e-7
// (FOH) off the oracle. The stiff model differs from the twin 1/(s+1) by
// 1e-14 relative, and the oracle uses the 320-bit modes of the float64
// data, so 1e-12 leaves two decades over rounding and four under the
// smallest old error.
func TestC2DFractionalInputDelayStiffKeepsSlowMode(t *testing.T) {
	const dt = 0.1
	t.Run("zoh literal", func(t *testing.T) {
		stiff, err := New(mat.NewDense(2, 2, []float64{-1, 1, 0, -1e14}), mat.NewDense(2, 1, []float64{0, 1e14}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		twin, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		modes := mustOracleModes(t, stiff)
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
			for _, tau := range []float64{0.23, 0.037} {
				stiff.InputDelay, twin.InputDelay = []float64{tau}, []float64{tau}
				ds, err := stiff.C2D(dt, C2DOptions{Method: method})
				if err != nil {
					t.Fatal(err)
				}
				dw, err := twin.C2D(dt, C2DOptions{Method: method})
				if err != nil {
					t.Fatal(err)
				}
				worst := 0.0
				for _, th := range []float64{0.05, 0.5, 2, 3} {
					z := cmplx.Exp(complex(0, th))
					gs, err := ds.EvalFr(z)
					if err != nil {
						t.Fatal(err)
					}
					gw, err := dw.EvalFr(z)
					if err != nil {
						t.Fatal(err)
					}
					worst = math.Max(worst, cmplx.Abs(gs[0][0]-gw[0][0])/math.Max(1, cmplx.Abs(gw[0][0])))
				}
				if !(worst <= 1e-12) {
					t.Errorf("%s τ=%g: C2D off the slow twin by %.3g", method, tau, worst)
				}
				if g := delayedHoldOracleGap(t, ds, modes, method, dt, [][]float64{{tau}}); !(g <= 1e-12) {
					t.Errorf("%s τ=%g: C2D off the exact delayed hold by %.3g", method, tau, g)
				}
			}
		}
	})
	t.Run("foh literal", func(t *testing.T) {
		sys, err := New(
			mat.NewDense(2, 2, []float64{-1.2168945090432504, 2.2056203198763917, 0, -2697344797704.7061}),
			mat.NewDense(2, 1, []float64{0.32541996060729217, -0.21088919050045629}),
			mat.NewDense(1, 2, []float64{0.60226010288964349, -0.14290415842693283}),
			mat.NewDense(1, 1, []float64{-0.041154831037892858}), 0)
		if err != nil {
			t.Fatal(err)
		}
		modes := mustOracleModes(t, sys)
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
			for _, tau := range []float64{0.37 * dt, 1.37 * dt} {
				sys.InputDelay = []float64{tau}
				d, err := sys.C2D(dt, C2DOptions{Method: method})
				if err != nil {
					t.Fatal(err)
				}
				if g := delayedHoldOracleGap(t, d, modes, method, dt, [][]float64{{tau}}); !(g <= 1e-12) {
					t.Errorf("%s τ=%g: C2D off the exact delayed hold by %.3g", method, tau, g)
				}
			}
		}
	})
}

// stiffTriangularDelayModel is a seeded two-input, two-output explicit
// model with n = 2 or 3 states whose upper quasi-triangular A has O(1) slow
// modes, |λ| in [0.3, 3] (a complex pair for seed ≡ 1 mod 4), above a stiff
// pole −10^(6..14).
func stiffTriangularDelayModel(t *testing.T, seed uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 4343))
	n := 2 + int(seed%2)
	a := mat.NewDense(n, n, nil)
	for i := range n {
		for j := i + 1; j < n; j++ {
			a.Set(i, j, 2*rng.NormFloat64())
		}
		a.Set(i, i, -(0.3 + 2.7*rng.Float64()))
	}
	if n == 3 && seed%4 == 1 {
		a.Set(1, 0, -(0.5 + rng.Float64()))
		a.Set(0, 1, 1+rng.Float64())
		a.Set(1, 1, a.At(0, 0))
	}
	a.Set(n-1, n-1, -math.Pow(10, 6+8*rng.Float64()))
	b, c := mat.NewDense(n, 2, nil), mat.NewDense(2, n, nil)
	for i := range n {
		for k := range 2 {
			b.Set(i, k, rng.NormFloat64())
			c.Set(k, i, rng.NormFloat64())
		}
	}
	d := mat.NewDense(2, 2, nil)
	for i := range 4 {
		d.RawMatrix().Data[i] = 0.1 * rng.NormFloat64()
	}
	sys, err := New(a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// RR43HG: every fractional-delay hold route of C2D (ZOH and FOH: input,
// output and I/O delays, the IODelay matrix, DelayModeling state, and the
// LFT path of a model with an internal delay) keeps the slow modes of
// seeded stiff triangular 2- and 3-state models, MIMO 2×2. Before d728012
// 240 of these checks failed, up to 5.2e-4 off. The slow poles are kept off
// z = 1 (|λ| ≥ 0.3), where the oracle and C2D both lose ε/|1 − e^{λT}|;
// measured worst is 4e-14, so 1e-11 is rounding with margin.
func TestC2DFractionalDelayStiffTriangularRoutesKeepSlowModes(t *testing.T) {
	const dt = 0.1
	for seed := uint64(1); seed <= 24; seed++ {
		sys := stiffTriangularDelayModel(t, seed)
		n, _, _ := sys.Dims()
		modes := mustOracleModes(t, sys)
		rng := rand.New(rand.NewPCG(seed, 77))
		in := []float64{(0.05 + 0.9*rng.Float64()) * dt, (1.05 + 0.9*rng.Float64()) * dt}
		out := []float64{(0.05 + 0.9*rng.Float64()) * dt, 2 * dt}
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
			for _, route := range []string{"input", "state", "output", "io", "iodelay", "lft"} {
				sd := sys.Copy()
				opts := C2DOptions{Method: method}
				switch route {
				case "input", "state", "lft":
					sd.InputDelay = append([]float64(nil), in...)
				case "output":
					sd.OutputDelay = append([]float64(nil), out...)
				case "io":
					sd.InputDelay = append([]float64(nil), in...)
					sd.OutputDelay = append([]float64(nil), out...)
				case "iodelay":
					if err := sd.SetDelay(mat.NewDense(2, 2, []float64{in[0], out[0], in[1], 0.3 * dt})); err != nil {
						t.Fatal(err)
					}
				}
				if route == "state" {
					opts.DelayModeling = C2DDelayModelingState
				}
				if route == "lft" {
					if err := sd.SetInternalDelay([]float64{2.5 * dt}, newDense(n, 1), newDense(1, n), newDense(2, 1), newDense(1, 2), newDense(1, 1)); err != nil {
						t.Fatal(err)
					}
				}
				tau := [][]float64{{0, 0}, {0, 0}}
				for i := range 2 {
					for j := range 2 {
						if len(sd.InputDelay) > 0 {
							tau[i][j] += sd.InputDelay[j]
						}
						if len(sd.OutputDelay) > 0 {
							tau[i][j] += sd.OutputDelay[i]
						}
						if sd.Delay != nil {
							tau[i][j] += sd.Delay.At(i, j)
						}
					}
				}
				d, err := sd.C2D(dt, opts)
				if err != nil {
					t.Errorf("seed %d %s %s: C2D: %v", seed, method, route, err)
					continue
				}
				if g := delayedHoldOracleGap(t, d, modes, method, dt, tau); !(g <= 1e-11) {
					t.Errorf("seed %d %s %s: C2D off the exact delayed hold by %.3g", seed, method, route, g)
				}
			}
		}
	}
}
