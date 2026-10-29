package controlsys

import (
	"math"
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
