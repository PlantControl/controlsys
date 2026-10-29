package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffInternalDelayCase is seed 2210 of stepFoldLossCases (cond(E) ~ 1e16,
// fold slow pole +0.099 for the pencil's stable one) with one internal
// delay τ = 0.3 fed back through (B2, C2, D12, D21), D22 = 0 (RT73OO).
func stiffInternalDelayCase(t *testing.T) *System {
	t.Helper()
	sys := stepFoldLossCases[2].system(t)
	err := sys.SetInternalDelay([]float64{0.3},
		mat.NewDense(4, 1, []float64{0.3, -0.2, 0.5, 0.1}),
		mat.NewDense(1, 4, []float64{0.4, -0.3, 0.2, 0.6}),
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 2, []float64{0.2, -0.1}),
		mat.NewDense(1, 1, []float64{0}))
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// lftAugmentedDescriptor is G = (A, E, [B B2], [C; C2], [D D12; D21 D22]).
func lftAugmentedDescriptor(sys *System) *System {
	g := conversionAugmentedRational(sys)
	g.E = denseCopy(sys.E)
	return g
}

// stiffInternalDelayCases are the stepFoldLossCases, each with one
// internal delay τ = 0.3 fed back through fixed (B2, C2, D12, D21), D22 =
// 0, and seed 2210 of stiffInternalDelayCase with input and output delays;
// models whose E reads rank deficient on this platform (seed 2210 under
// amd64 gonum) are left out.
func stiffInternalDelayCases(t *testing.T) (names []string, systems []*System) {
	t.Helper()
	add := func(name string, sys *System) {
		if matlabRankFull(t, sys.E) {
			names, systems = append(names, name), append(systems, sys)
		}
	}
	for _, c := range stepFoldLossCases {
		sys := c.system(t)
		b2, c2, d21, d12 := make([]float64, c.n), make([]float64, c.n), make([]float64, c.m), make([]float64, c.p)
		for i := range c.n {
			b2[i] = 0.4 * math.Cos(float64(2*i+1))
			c2[i] = 0.5 * math.Sin(float64(3*i+2))
		}
		for j := range c.m {
			d21[j] = 0.2 - 0.3*float64(j)
		}
		for i := range c.p {
			d12[i] = 0.5
		}
		if err := sys.SetInternalDelay([]float64{0.3}, mat.NewDense(c.n, 1, b2), mat.NewDense(1, c.n, c2),
			mat.NewDense(c.p, 1, d12), mat.NewDense(1, c.m, d21), mat.NewDense(1, 1, nil)); err != nil {
			t.Fatal(err)
		}
		add(c.name, sys)
	}
	sys := stiffInternalDelayCase(t)
	sys.InputDelay = []float64{0.2, 0}
	sys.OutputDelay = []float64{0.1}
	add("seed 2210 with I/O delays", sys)
	if len(systems) < 3 {
		t.Fatalf("only %d models have full-rank E", len(systems))
	}
	return names, systems
}

// C2D (ZOH) of a stiff descriptor with internal delays of whole samples is
// lft(c2d(G), z^-k), MATLAB's c2d of the delay model: its frequency
// response matches the exact ZOH of the 320-bit modes of the pencil of G
// closed by z^-k, times the I/O delays. The LU fold E⁻¹A put it off by up
// to 0.47 (RT73OO).
func TestC2DZOHInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	const dt = 0.1
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		g := lftAugmentedDescriptor(sys)
		pol, err := g.Poles()
		if err != nil {
			t.Fatal(err)
		}
		modes, ok := newStepOracleModes(g, pol)
		if !ok {
			t.Fatalf("%s: oracle does not resolve the pencil", names[k])
		}
		d, err := sys.C2D(dt, C2DOptions{})
		if err != nil {
			t.Errorf("%s: C2D: %v", names[k], err)
			continue
		}
		_, m, p := sys.Dims()
		worst := 0.0
		for _, th := range []float64{0.05, 0.5, 2, 3} {
			z := cmplx.Exp(complex(0, th))
			want := lftClose(m, p, func(i, j int) complex128 { return modes.zoh(i, j, dt, z) }, []float64{3}, z)
			got := lftClose(m, p, lftAugmentedAt(t, d, z), d.LFT.Tau, z)
			for i := range p {
				for j := range m {
					shift := 0.0
					if sys.InputDelay != nil {
						shift += sys.InputDelay[j]
					}
					if sys.OutputDelay != nil {
						shift += sys.OutputDelay[i]
					}
					w := want[i][j] * cmplx.Pow(z, complex(-math.Round(shift/dt), 0))
					g := got[i][j] * discreteIODelayAt(d, i, j, z)
					worst = math.Max(worst, cmplx.Abs(g-w)/math.Max(1, cmplx.Abs(w)))
				}
			}
		}
		if !(worst <= 1e-9) {
			t.Errorf("%s: C2D frequency response off the pencil by %.3g", names[k], worst)
		}
	}
}

// discreteIODelayAt is z^-(k_in + k_out + k_ij) of channel (i, j) of the
// discrete sys.
func discreteIODelayAt(sys *System, i, j int, z complex128) complex128 {
	k := 0.0
	if sys.InputDelay != nil {
		k += sys.InputDelay[j]
	}
	if sys.OutputDelay != nil {
		k += sys.OutputDelay[i]
	}
	if sys.Delay != nil {
		k += sys.Delay.At(i, j)
	}
	return cmplx.Pow(z, complex(-k, 0))
}

func lftAugmentedAt(t *testing.T, sys *System, z complex128) func(i, j int) complex128 {
	g := pencilResponse(t, conversionAugmentedRational(sys), z)
	return func(i, j int) complex128 { return g[i][j] }
}

// lftClose returns H = G11 + G12·Δ(I − G22·Δ)⁻¹·G21, Δ = diag(z^-k).
func lftClose(m, p int, g func(i, j int) complex128, k []float64, z complex128) [][]complex128 {
	q := len(k)
	I := make([][]complex128, q)
	G21 := make([][]complex128, q)
	for a := range q {
		I[a] = make([]complex128, q)
		G21[a] = make([]complex128, m)
		for b := range q {
			I[a][b] = -g(p+a, m+b) * cmplx.Pow(z, complex(-k[b], 0))
		}
		I[a][a] += 1
		for j := range m {
			G21[a][j] = g(p+a, j)
		}
	}
	X := csolveNoT(I, G21)
	H := make([][]complex128, p)
	for i := range p {
		H[i] = make([]complex128, m)
		for j := range m {
			H[i][j] = g(i, j)
			for a := range q {
				H[i][j] += g(i, m+a) * cmplx.Pow(z, complex(-k[a], 0)) * X[a][j]
			}
		}
	}
	return H
}

func csolveNoT(a, b [][]complex128) [][]complex128 {
	n := len(a)
	k := len(b[0])
	W := make([][]complex128, n)
	for i := range n {
		W[i] = append(append([]complex128{}, a[i]...), b[i]...)
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if cmplx.Abs(W[r][c]) > cmplx.Abs(W[piv][c]) {
				piv = r
			}
		}
		W[c], W[piv] = W[piv], W[c]
		for r := range n {
			if r == c {
				continue
			}
			f := W[r][c] / W[c][c]
			for j := c; j < n+k; j++ {
				W[r][j] -= f * W[c][j]
			}
		}
	}
	out := make([][]complex128, n)
	for i := range n {
		out[i] = make([]complex128, k)
		for j := range k {
			out[i][j] = W[i][n+j] / W[i][i]
		}
	}
	return out
}

// A stiff descriptor (cond(E) 2e15, pencil poles −1.4e15 and 0.304, fold
// slow pole 0.364) closed by two internal delays of 0.25 s (2.5 samples at
// dt = 0.1) has two closed-loop roots in the right half-plane: the argument
// principle on det(sE − A − e^{−sτ}·B2·C2), whose s² coefficient det(E) is
// formed exactly. Its ZOH and FOH discretizations are then unstable; the
// fold made them stable (RT73OO review sweep, nearSingularEContinuousK
// seed 116).
func TestC2DFractionalInternalDelayStiffDescriptorKeepsInstability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.16651325993927746, -0.18881816152361589, -0.7812443071724737, -0.5105728073358802})
	E := mat.NewDense(2, 2, []float64{0.05744448278498728, -0.062185957118855655, -0.3464693904136969, 0.3750670144582298})
	B2 := mat.NewDense(2, 2, []float64{-0.4122687632518598, 0.02628900878011349, 0.05255512500346221, -0.05734616494225475})
	C2 := mat.NewDense(2, 2, []float64{-0.6942289693736933, -0.6542020932766299, -0.5569197866445883, -0.028610092985908483})
	sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{-0.5096244121191215, -0.45477962922031856}),
		mat.NewDense(1, 2, []float64{1.7357434392386313, 0.7945681613231941}), mat.NewDense(1, 1, []float64{-0.7933357460691265}), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !matlabRankFull(t, E) {
		t.Skip("E reads rank deficient on this platform")
	}
	const tau = 0.25
	if err := sys.SetInternalDelay([]float64{tau, tau}, B2, C2, mat.NewDense(1, 2, []float64{-0.12170991249970453, 0.9483805251588965}),
		mat.NewDense(2, 1, []float64{-0.13280317748780845, -0.5976750171300884}), mat.NewDense(2, 2, nil)); err != nil {
		t.Fatal(err)
	}

	var K mat.Dense
	K.Mul(B2, C2)
	detE, _ := new(big.Float).SetPrec(200).Sub(
		new(big.Float).SetPrec(200).Mul(big.NewFloat(E.At(0, 0)), big.NewFloat(E.At(1, 1))),
		new(big.Float).SetPrec(200).Mul(big.NewFloat(E.At(0, 1)), big.NewFloat(E.At(1, 0)))).Float64()
	char := func(s complex128) complex128 {
		ex := cmplx.Exp(-s * tau)
		M := func(i, j int) complex128 { return complex(A.At(i, j), 0) + ex*complex(K.At(i, j), 0) }
		e := func(i, j int) complex128 { return complex(E.At(i, j), 0) }
		c1 := -e(0, 0)*M(1, 1) - M(0, 0)*e(1, 1) + e(0, 1)*M(1, 0) + M(0, 1)*e(1, 0)
		return s*s*complex(detE, 0) + s*c1 + M(0, 0)*M(1, 1) - M(0, 1)*M(1, 0)
	}
	const R, N = 200.0, 200000
	wind, prev := 0.0, char(complex(0, -R))
	for k := 1; k <= 2*N; k++ {
		var s complex128
		if k <= N {
			s = complex(0, -R+2*R*float64(k)/N)
		} else {
			th := math.Pi/2 - math.Pi*float64(k-N)/N
			s = complex(R*math.Cos(th), R*math.Sin(th))
		}
		cur := char(s)
		wind += cmplx.Phase(cur / prev)
		prev = cur
	}
	if rhp := -math.Round(wind / (2 * math.Pi)); rhp != 2 {
		t.Fatalf("argument principle counts %g right half-plane roots, want 2", rhp)
	}

	for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		d, err := sys.C2D(0.1, C2DOptions{Method: meth})
		if err != nil {
			t.Fatalf("%s: %v", meth, err)
		}
		stable, err := d.IsStable()
		if err != nil {
			t.Fatalf("%s: IsStable: %v", meth, err)
		}
		if stable {
			t.Errorf("%s: C2D is stable; the closed loop has two right half-plane roots", meth)
		}
	}
}
