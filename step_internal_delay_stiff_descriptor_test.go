package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type stepOracleMat [][]*big.Float

func stepOracleZeros(r, c int) stepOracleMat {
	out := make(stepOracleMat, r)
	for i := range r {
		out[i] = make([]*big.Float, c)
		for j := range c {
			out[i][j] = stepOracleNew()
		}
	}
	return out
}

func (a stepOracleMat) mul(b stepOracleMat) stepOracleMat {
	out := stepOracleZeros(len(a), len(b[0]))
	t := stepOracleNew()
	for i := range a {
		for k := range b {
			if a[i][k].Sign() == 0 {
				continue
			}
			for j := range b[0] {
				out[i][j].Add(out[i][j], t.Mul(a[i][k], b[k][j]))
			}
		}
	}
	return out
}

// stepOracleExpm returns e^{M·t} by scaling and squaring of the Taylor
// series at stepOraclePrec bits.
func stepOracleExpm(M stepOracleMat, t float64) stepOracleMat {
	n := len(M)
	norm := 0.0
	for i := range n {
		row := 0.0
		for j := range n {
			v, _ := M[i][j].Float64()
			row += math.Abs(v)
		}
		norm = math.Max(norm, row)
	}
	s := 0
	if x := norm * t; x > 0.5 {
		s = int(math.Ceil(math.Log2(x))) + 1
	}
	scale := new(big.Float).SetPrec(stepOraclePrec).SetMantExp(stepOracleFloat(t), -s)
	X := stepOracleZeros(n, n)
	for i := range n {
		for j := range n {
			X[i][j].Mul(M[i][j], scale)
		}
	}
	R, term := stepOracleZeros(n, n), stepOracleZeros(n, n)
	for i := range n {
		R[i][i].SetInt64(1)
		term[i][i].SetInt64(1)
	}
	for k := 1; k < 120; k++ {
		term = term.mul(X)
		inv := stepOracleNew().Quo(stepOracleFloat(1), stepOracleFloat(float64(k)))
		big0 := true
		for i := range n {
			for j := range n {
				term[i][j].Mul(term[i][j], inv)
				R[i][j].Add(R[i][j], term[i][j])
				if term[i][j].Sign() != 0 && term[i][j].MantExp(nil) > -stepOraclePrec-20 {
					big0 = false
				}
			}
		}
		if big0 {
			break
		}
	}
	for range s {
		R = R.mul(R)
	}
	return R
}

// lftStepOracle is the continuous response with zero history of a
// descriptor sys with internal delays of one length τ and D22 = 0, at
// stepOraclePrec bits by the method of steps: on [kτ, (k+1)τ] the stack
// [x(t); x(t−τ); …; x(t−kτ); 1] obeys a block-bidiagonal linear ODE in the
// pencil's E⁻¹A (formed at stepOraclePrec bits), exponentiated exactly.
// Column j ≥ 0 is the step response of input j from x(0) = 0; j < 0 the
// free response from x0.
type lftStepOracle struct {
	n, m, p, q         int
	tau                float64
	f, bu, bw, c2, d21 stepOracleMat
	c, d12             stepOracleMat
	d                  *mat.Dense
}

func newLFTStepOracle(t *testing.T, sys *System) lftStepOracle {
	t.Helper()
	n, m, p := sys.Dims()
	q := len(sys.LFT.Tau)
	for _, v := range sys.LFT.Tau {
		if v != sys.LFT.Tau[0] {
			t.Fatal("oracle needs one common internal delay")
		}
	}
	if !allZeroDense(sys.LFT.D22) {
		t.Fatal("oracle needs D22 = 0")
	}
	conv := func(x *mat.Dense, r, c int) stepOracleMat {
		out := stepOracleZeros(r, c)
		for i := range r {
			for j := range c {
				out[i][j] = stepOracleFloat(x.At(i, j))
			}
		}
		return out
	}
	rhs := stepOracleZeros(n, n+m+q)
	for i := range n {
		for j := range n {
			rhs[i][j] = stepOracleFloat(sys.A.At(i, j))
		}
		for j := range m {
			rhs[i][n+j] = stepOracleFloat(sys.B.At(i, j))
		}
		for j := range q {
			rhs[i][n+m+j] = stepOracleFloat(sys.LFT.B2.At(i, j))
		}
	}
	sol := stepOracleSolve(conv(sys.E, n, n), rhs)
	if sol == nil {
		t.Fatal("singular E")
	}
	o := lftStepOracle{n: n, m: m, p: p, q: q, tau: sys.LFT.Tau[0], d: sys.D,
		f: stepOracleZeros(n, n), bu: stepOracleZeros(n, m), bw: stepOracleZeros(n, q),
		c2: conv(sys.LFT.C2, q, n), d21: conv(sys.LFT.D21, q, m), c: conv(sys.C, p, n), d12: conv(sys.LFT.D12, p, q)}
	for i := range n {
		copy(o.f[i], sol[i][:n])
		copy(o.bu[i], sol[i][n:n+m])
		copy(o.bw[i], sol[i][n+m:])
	}
	return o
}

// response returns y(t) (p values) for input j (j < 0: free from x0).
func (o lftStepOracle) response(j int, x0 []float64, t float64) []float64 {
	n := o.n
	bwc2 := o.bw.mul(o.c2)
	src := stepOracleZeros(n, 1)
	srcDelay := stepOracleZeros(n, 1)
	if j >= 0 {
		bwd := o.bw.mul(o.d21)
		for i := range n {
			src[i][0].Set(o.bu[i][j])
			srcDelay[i][0].Add(o.bu[i][j], bwd[i][j])
		}
	}
	grid := []stepOracleMat{stepOracleZeros(n, 1)}
	if j < 0 {
		for i := range n {
			grid[0][i][0] = stepOracleFloat(x0[i])
		}
	}
	for k := 0; ; k++ {
		N := (k+1)*n + 1
		G := stepOracleZeros(N, N)
		for b := 0; b <= k; b++ {
			for r := range n {
				for c := range n {
					G[b*n+r][b*n+c].Set(o.f[r][c])
					if b < k {
						G[b*n+r][(b+1)*n+c].Set(bwc2[r][c])
					}
				}
				if b < k {
					G[b*n+r][N-1].Set(srcDelay[r][0])
				} else {
					G[b*n+r][N-1].Set(src[r][0])
				}
			}
		}
		xi := stepOracleZeros(N, 1)
		for b := 0; b <= k; b++ {
			for r := range n {
				xi[b*n+r][0].Set(grid[k-b][r][0])
			}
		}
		xi[N-1][0].SetInt64(1)
		end := math.Min(t, float64(k+1)*o.tau)
		X := stepOracleExpm(G, end-float64(k)*o.tau).mul(xi)
		if t <= float64(k+1)*o.tau {
			y := make([]float64, o.p)
			for i := range o.p {
				acc := stepOracleNew()
				if j >= 0 {
					acc.SetFloat64(o.d.At(i, j))
				}
				for r := range n {
					acc.Add(acc, stepOracleNew().Mul(o.c[i][r], X[r][0]))
				}
				if k >= 1 {
					for l := range o.q {
						w := stepOracleNew()
						if j >= 0 {
							w.Set(o.d21[l][j])
						}
						for r := range n {
							w.Add(w, stepOracleNew().Mul(o.c2[l][r], X[n+r][0]))
						}
						acc.Add(acc, stepOracleNew().Mul(o.d12[i][l], w))
					}
				}
				y[i], _ = acc.Float64()
			}
			return y
		}
		next := stepOracleZeros(n, 1)
		for r := range n {
			next[r][0].Set(X[r][0])
		}
		grid = append(grid, next)
	}
}

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

// C2D (ZOH) of a stiff descriptor with an internal delay of whole samples
// is lft(c2d(G), z^-k), MATLAB's c2d of the delay model: its frequency
// response matches the exact ZOH of the 320-bit modes of the pencil of G
// closed by z^-k.
func TestC2DZOHInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	sys := stiffInternalDelayCase(t)
	g := lftAugmentedDescriptor(sys)
	pol, err := g.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(g, pol)
	if !ok {
		t.Fatal("oracle does not resolve the pencil")
	}
	const dt = 0.1
	d, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	_, m, p := sys.Dims()
	worst := 0.0
	for _, th := range []float64{0.05, 0.5, 2, 3} {
		z := cmplx.Exp(complex(0, th))
		want := lftClose(m, p, func(i, j int) complex128 { return modes.zoh(i, j, dt, z) }, []float64{3}, z)
		got := lftClose(m, p, lftAugmentedAt(t, d, z), d.LFT.Tau, z)
		for i := range p {
			for j := range m {
				worst = math.Max(worst, cmplx.Abs(got[i][j]-want[i][j])/math.Max(1, cmplx.Abs(want[i][j])))
			}
		}
	}
	if !(worst <= 1e-9) {
		t.Errorf("C2D frequency response off the pencil by %.3g", worst)
	}
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

// lftOracleSamples picks 12 sample indices spread over a grid of n.
func lftOracleSamples(n int) []int {
	var out []int
	for k := 1; k <= 12; k++ {
		out = append(out, k*(n-1)/12)
	}
	return out
}

func nearDelayGrid(t, tau float64) bool {
	r := t / tau
	return math.Abs(r-math.Round(r)) < 1e-6
}

// Step of the stiff descriptor with an internal delay follows the pencil:
// every sample off the delay grid matches the method-of-steps response of
// E⁻¹A formed at 320 bits.
func TestStepInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	sys := stiffInternalDelayCase(t)
	o := newLFTStepOracle(t, sys)
	st, err := Step(sys, 1)
	if err != nil {
		t.Fatal(err)
	}
	_, m, p := sys.Dims()
	worst := 0.0
	for _, s := range lftOracleSamples(len(st.T)) {
		if nearDelayGrid(st.T[s], o.tau) {
			continue
		}
		for j := range m {
			want := o.response(j, nil, st.T[s])
			for i := range p {
				worst = math.Max(worst, stepOracleRel(st.Y.At(j*p+i, s), want[i]))
			}
		}
	}
	if worst > 1e-7 {
		t.Errorf("Step off the pencil by %.3g", worst)
	}
}

// Initial from x0 with zero delay history follows the pencil as Step does.
func TestInitialInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	sys := stiffInternalDelayCase(t)
	o := newLFTStepOracle(t, sys)
	x0 := []float64{0.7, -0.4, 0.2, 1.1}
	in, err := Initial(sys, mat.NewVecDense(4, x0), 1)
	if err != nil {
		t.Fatal(err)
	}
	_, _, p := sys.Dims()
	worst := 0.0
	for _, s := range lftOracleSamples(len(in.T)) {
		if nearDelayGrid(in.T[s], o.tau) {
			continue
		}
		want := o.response(-1, x0, in.T[s])
		for i := range p {
			worst = math.Max(worst, stepOracleRel(in.Y.At(i, s), want[i]))
		}
	}
	if worst > 1e-7 {
		t.Errorf("Initial off the pencil by %.3g", worst)
	}
}

// The method-of-steps oracle agrees with Step and Initial on a
// well-conditioned descriptor with the same delay feedback, where the fold
// E⁻¹A is accurate.
func TestLFTStepOracleMatchesWellConditionedDescriptor(t *testing.T) {
	sys := stiffInternalDelayCase(t)
	sys.E = mat.NewDense(4, 4, []float64{1.2, 0.1, 0, -0.2, 0, 0.9, 0.3, 0, 0.1, 0, 1.1, 0.2, 0, -0.1, 0, 0.8})
	o := newLFTStepOracle(t, sys)
	st, err := Step(sys, 1)
	if err != nil {
		t.Fatal(err)
	}
	x0 := []float64{0.7, -0.4, 0.2, 1.1}
	in, err := Initial(sys, mat.NewVecDense(4, x0), 1)
	if err != nil {
		t.Fatal(err)
	}
	_, m, p := sys.Dims()
	worst := 0.0
	for _, s := range lftOracleSamples(len(st.T)) {
		if nearDelayGrid(st.T[s], o.tau) {
			continue
		}
		for j := range m {
			want := o.response(j, nil, st.T[s])
			for i := range p {
				worst = math.Max(worst, stepOracleRel(st.Y.At(j*p+i, s), want[i]))
			}
		}
	}
	for _, s := range lftOracleSamples(len(in.T)) {
		if nearDelayGrid(in.T[s], o.tau) {
			continue
		}
		want := o.response(-1, x0, in.T[s])
		for i := range p {
			worst = math.Max(worst, stepOracleRel(in.Y.At(i, s), want[i]))
		}
	}
	if worst > 1e-10 {
		t.Errorf("oracle off Step/Initial by %.3g", worst)
	}
}
