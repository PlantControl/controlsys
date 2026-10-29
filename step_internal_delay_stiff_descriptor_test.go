package controlsys

import (
	"math"
	"math/big"
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
	skipSlowUnderRace(t)
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		if sys.InputDelay != nil {
			continue
		}
		o := newLFTStepOracle(t, sys)
		st, err := Step(sys, 1)
		if err != nil {
			t.Errorf("%s: Step: %v", names[k], err)
			continue
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
		t.Logf("%s: worst %.3g dt %g", names[k], worst, st.T[1])
		if worst > lftStepTol {
			t.Errorf("%s: Step off the pencil by %.3g", names[k], worst)
		}
	}
}

// lftStepTol bounds the O(dt) error of the ZOH discretization of the delay
// channels that Step and Initial of these models take.
const lftStepTol = 5e-5

// Initial from x0 with zero delay history follows the pencil as Step does.
func TestInitialInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	skipSlowUnderRace(t)
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		if sys.InputDelay != nil {
			continue
		}
		o := newLFTStepOracle(t, sys)
		n, _, p := sys.Dims()
		x0 := make([]float64, n)
		for i := range n {
			x0[i] = 0.7 - 0.45*float64(i)
		}
		in, err := Initial(sys, mat.NewVecDense(n, x0), 1)
		if err != nil {
			t.Errorf("%s: Initial: %v", names[k], err)
			continue
		}
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
		t.Logf("%s: worst %.3g", names[k], worst)
		if worst > lftStepTol {
			t.Errorf("%s: Initial off the pencil by %.3g", names[k], worst)
		}
	}
}

// Initial of the seed 2210 model without inputs (an autonomous model) takes
// the same route as with them.
func TestInitialNoInputInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	c := stepFoldLossCases[2]
	sys, err := NewDescriptor(mat.NewDense(4, 4, c.a), &mat.Dense{}, mat.NewDense(1, 4, c.c), &mat.Dense{}, mat.NewDense(4, 4, c.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !matlabRankFull(t, sys.E) {
		t.Skip("E reads rank deficient on this platform")
	}
	if err := sys.SetInternalDelay([]float64{0.3}, mat.NewDense(4, 1, []float64{0.3, -0.2, 0.5, 0.1}),
		mat.NewDense(1, 4, []float64{0.4, -0.3, 0.2, 0.6}), mat.NewDense(1, 1, []float64{0.5}), nil, mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	o := newLFTStepOracle(t, sys)
	x0 := []float64{0.7, -0.4, 0.2, 1.1}
	in, err := Initial(sys, mat.NewVecDense(4, x0), 1)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for _, s := range lftOracleSamples(len(in.T)) {
		if nearDelayGrid(in.T[s], o.tau) {
			continue
		}
		worst = math.Max(worst, stepOracleRel(in.Y.At(0, s), o.response(-1, x0, in.T[s])[0]))
	}
	if worst > lftStepTol {
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
