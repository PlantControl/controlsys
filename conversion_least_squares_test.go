package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// lsqModalSource is a dense realization T·blkdiag(J, complex pairs, real
// poles)·T⁻¹ of a model whose response has the closed form
// D + Σ C_b (sI - A_b)⁻¹ B_b, with J an integrator Jordan chain.
type lsqModalSource struct {
	sys      *System
	blocks   [][2]float64
	sizes    []int
	b0, c0   *mat.Dense
	d        *mat.Dense
	chain    int
	residues *mat.Dense
}

func newLSQModalSource(t *testing.T, rng *rand.Rand, chain, pairs, reals, m, p int, spread float64, uncontrollable bool) *lsqModalSource {
	t.Helper()
	n := chain + 2*pairs + reals
	if uncontrollable {
		n++
	}
	a0 := mat.NewDense(n, n, nil)
	src := &lsqModalSource{chain: chain, b0: mat.NewDense(n, m, nil), c0: mat.NewDense(p, n, nil), d: mat.NewDense(p, m, nil)}
	for i := range chain - 1 {
		a0.Set(i, i+1, 1)
	}
	at := chain
	for range pairs {
		sigma, omega := -.5-4.5*rng.Float64(), 10*rng.Float64()
		a0.Set(at, at, sigma)
		a0.Set(at+1, at+1, sigma)
		a0.Set(at, at+1, omega)
		a0.Set(at+1, at, -omega)
		src.blocks = append(src.blocks, [2]float64{sigma, omega})
		src.sizes = append(src.sizes, 2)
		at += 2
	}
	for range reals {
		pole := -.5 - 4.5*rng.Float64()
		a0.Set(at, at, pole)
		src.blocks = append(src.blocks, [2]float64{pole, 0})
		src.sizes = append(src.sizes, 1)
		at++
	}
	for i := range n {
		for j := range m {
			src.b0.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			src.c0.Set(j, i, rng.NormFloat64())
		}
	}
	if uncontrollable {
		for j := range m {
			src.b0.Set(n-1, j, 0)
		}
	}
	for i := range p {
		for j := range m {
			src.d.Set(i, j, rng.NormFloat64())
		}
	}
	src.residues = mat.NewDense(p, m, nil)
	if chain > 0 {
		for i := range p {
			for j := range m {
				src.residues.Set(i, j, src.c0.At(i, 0)*src.b0.At(chain-1, j))
			}
		}
	}
	g := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(g)
	var q mat.Dense
	qr.QTo(&q)
	tm, tinv := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for j := range n {
		s := math.Pow(10, spread*(2*rng.Float64()-1))
		for i := range n {
			tm.Set(i, j, q.At(i, j)*s)
			tinv.Set(j, i, q.At(i, j)/s)
		}
	}
	var a, tmp, b, c mat.Dense
	tmp.Mul(tm, a0)
	a.Mul(&tmp, tinv)
	b.Mul(tm, src.b0)
	c.Mul(src.c0, tinv)
	sys, err := New(&a, &b, &c, mat.DenseCopyOf(src.d), 0)
	if err != nil {
		t.Fatal(err)
	}
	src.sys = sys
	return src
}

func (src *lsqModalSource) eval(s complex128, i, j int) complex128 {
	g := complex(src.d.At(i, j), 0)
	for a := range src.chain {
		for b := a; b < src.chain; b++ {
			g += complex(src.c0.At(i, a)*src.b0.At(b, j), 0) / cmplx.Pow(s, complex(float64(b-a+1), 0))
		}
	}
	at := src.chain
	for k, blk := range src.blocks {
		sigma, omega := complex(blk[0], 0), complex(blk[1], 0)
		ci, bj := src.c0.At(i, at), src.b0.At(at, j)
		if src.sizes[k] == 1 {
			g += complex(ci*bj, 0) / (s - sigma)
			at++
			continue
		}
		c2, b2 := src.c0.At(i, at+1), src.b0.At(at+1, j)
		det := (s-sigma)*(s-sigma) + omega*omega
		x1 := ((s-sigma)*complex(bj, 0) + omega*complex(b2, 0)) / det
		x2 := (-omega*complex(bj, 0) + (s-sigma)*complex(b2, 0)) / det
		g += complex(ci, 0)*x1 + complex(c2, 0)*x2
		at += 2
	}
	return g
}

func lsqChannel(t *testing.T, sys *System, i, j int) *System {
	t.Helper()
	n, _, _ := sys.Dims()
	b := mat.NewDense(n, 1, nil)
	c := mat.NewDense(1, n, nil)
	for k := range n {
		b.Set(k, 0, sys.B.At(k, j))
		c.Set(0, k, sys.C.At(i, k))
	}
	channel, err := New(mat.DenseCopyOf(sys.A), b, c, mat.NewDense(1, 1, []float64{sys.D.At(i, j)}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return channel
}

func TestLeastSquaresHighOrderSources(t *testing.T) {
	const dt, order = .05, 20
	tests := []struct {
		name                      string
		chain, pairs, reals, m, p int
		spread                    float64
		uncontrollable            bool
		row, col                  int
	}{
		{name: "n40 stable", pairs: 20, m: 1, p: 1},
		{name: "n100 stable", pairs: 48, reals: 4, m: 1, p: 1, spread: 1},
		{name: "n41 one integrator", chain: 1, pairs: 20, m: 1, p: 1},
		{name: "n100 two integrators", chain: 2, pairs: 47, reals: 4, m: 1, p: 1, spread: 1},
		{name: "n43 three integrators", chain: 3, pairs: 20, m: 1, p: 1, spread: 1},
		{name: "n60 MIMO channel", pairs: 28, reals: 4, m: 2, p: 3, row: 2, col: 1},
		{name: "n100 MIMO channel integrator", chain: 1, pairs: 47, reals: 5, m: 2, p: 3, row: 1, col: 0, spread: 1},
		{name: "n41 uncontrollable pole at zero", pairs: 20, m: 1, p: 1, uncontrollable: true},
	}
	// The case index seeds its model; reordering cases changes the models.
	for seed, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(uint64(seed), 7))
			src := newLSQModalSource(t, rng, tc.chain, tc.pairs, tc.reals, tc.m, tc.p, tc.spread, tc.uncontrollable)
			source := src.sys
			if tc.m > 1 || tc.p > 1 {
				source = lsqChannel(t, src.sys, tc.row, tc.col)
			}
			if source.D.At(0, 0) == 0 {
				t.Fatal("want D != 0")
			}
			k, residue, err := leastSquaresIntegratorLimit(source)
			if err != nil {
				t.Fatal(err)
			}
			want := src.residues.At(tc.row, tc.col)
			if k != tc.chain || math.Abs(residue-want) > 1e-8*math.Abs(want) {
				t.Fatalf("integrators=%d residue=%g, want %d %g", k, residue, tc.chain, want)
			}
			if tc.chain > 1 {
				_, _, err := source.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: tc.chain - 1})
				if !errors.Is(err, ErrInvalidOrder) || !strings.Contains(err.Error(), fmt.Sprintf("retain %d integrators", tc.chain)) {
					t.Fatalf("order %d err=%v", tc.chain-1, err)
				}
			}
			resultSys, result, err := source.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: order})
			if err != nil {
				t.Fatal(err)
			}
			nd, _, _ := resultSys.Dims()
			samples := max(513, 32*order+1)
			residual, magnitude, maxError, maxMagnitude := 0.0, 0.0, 0.0, 0.0
			for q := 0; q <= samples; q++ {
				theta := math.Pi * (float64(q) + .5) / float64(samples+1)
				want := src.eval(complex(0, theta/dt), tc.row, tc.col)
				got := oracleResponse(t, resultSys, cmplx.Exp(complex(0, theta)), nd, 1, 1)[0]
				delta := cmplx.Abs(got - want)
				residual, magnitude = math.Hypot(residual, delta), math.Hypot(magnitude, cmplx.Abs(want))
				maxError, maxMagnitude = math.Max(maxError, delta), math.Max(maxMagnitude, cmplx.Abs(want))
			}
			rms, worst := residual/magnitude, maxError/maxMagnitude
			t.Logf("reported rms=%g max=%g, oracle rms=%g max=%g", result.RMSRelativeError, result.MaxRelativeError, rms, worst)
			if math.Abs(rms-result.RMSRelativeError) > 1e-6 || math.Abs(worst-result.MaxRelativeError) > 1e-6 {
				t.Fatalf("reported rms=%g max=%g, oracle rms=%g max=%g", result.RMSRelativeError, result.MaxRelativeError, rms, worst)
			}
			if rms > .01 {
				t.Fatalf("oracle rms=%g", rms)
			}
			poles, err := resultSys.Poles()
			if err != nil {
				t.Fatal(err)
			}
			exact := 0
			for _, pole := range poles {
				if pole == 1 {
					exact++
				}
			}
			if exact != tc.chain {
				t.Fatalf("%d poles exactly at z=1, want %d", exact, tc.chain)
			}
			if tc.chain > 0 {
				for _, w := range []float64{1e-4, 1e-3, 1e-2} {
					want := src.eval(complex(0, w), tc.row, tc.col)
					got := bigSISOResponse(resultSys, cmplx.Exp(complex(0, w*dt)))
					if cmplx.Abs(got-want) > 3e-3*cmplx.Abs(want) {
						t.Fatalf("w=%g fitted=%v want=%v", w, got, want)
					}
				}
			}
		})
	}
}

type lsqBigComplex struct{ re, im *big.Float }

func bigReal(x float64) *big.Float { return new(big.Float).SetPrec(400).SetFloat64(x) }
func bigMul(a, b lsqBigComplex) lsqBigComplex {
	r1 := new(big.Float).SetPrec(400).Mul(a.re, b.re)
	r2 := new(big.Float).SetPrec(400).Mul(a.im, b.im)
	i1 := new(big.Float).SetPrec(400).Mul(a.re, b.im)
	i2 := new(big.Float).SetPrec(400).Mul(a.im, b.re)
	return lsqBigComplex{r1.Sub(r1, r2), i1.Add(i1, i2)}
}
func bigSub(a, b lsqBigComplex) lsqBigComplex {
	return lsqBigComplex{new(big.Float).SetPrec(400).Sub(a.re, b.re), new(big.Float).SetPrec(400).Sub(a.im, b.im)}
}
func bigDiv(a, b lsqBigComplex) lsqBigComplex {
	d := new(big.Float).SetPrec(400).Mul(b.re, b.re)
	d.Add(d, new(big.Float).SetPrec(400).Mul(b.im, b.im))
	conj := lsqBigComplex{b.re, new(big.Float).SetPrec(400).Neg(b.im)}
	n := bigMul(a, conj)
	return lsqBigComplex{n.re.Quo(n.re, d), n.im.Quo(n.im, d)}
}

// bigSISOResponse evaluates C(zI-A)⁻¹B+D exactly enough by 400-bit Gaussian
// elimination, for frequencies where float64 solves of the fitted
// realization lose accuracy next to its poles at z = 1.
func bigSISOResponse(sys *System, z complex128) complex128 {
	n, _, _ := sys.Dims()
	M := make([][]lsqBigComplex, n)
	x := make([]lsqBigComplex, n)
	for i := range n {
		M[i] = make([]lsqBigComplex, n)
		for j := range n {
			M[i][j] = lsqBigComplex{bigReal(-sys.A.At(i, j)), bigReal(0)}
		}
		M[i][i] = lsqBigComplex{new(big.Float).SetPrec(400).Add(M[i][i].re, bigReal(real(z))), bigReal(imag(z))}
		x[i] = lsqBigComplex{bigReal(sys.B.At(i, 0)), bigReal(0)}
	}
	for k := range n {
		p := k
		for i := k; i < n; i++ {
			if M[i][k].re.Sign() != 0 || M[i][k].im.Sign() != 0 {
				p = i
				break
			}
		}
		M[k], M[p] = M[p], M[k]
		x[k], x[p] = x[p], x[k]
		for i := k + 1; i < n; i++ {
			f := bigDiv(M[i][k], M[k][k])
			for j := k; j < n; j++ {
				M[i][j] = bigSub(M[i][j], bigMul(f, M[k][j]))
			}
			x[i] = bigSub(x[i], bigMul(f, x[k]))
		}
	}
	for k := n - 1; k >= 0; k-- {
		for j := k + 1; j < n; j++ {
			x[k] = bigSub(x[k], bigMul(M[k][j], x[j]))
		}
		x[k] = bigDiv(x[k], M[k][k])
	}
	re, im := bigReal(sys.D.At(0, 0)), bigReal(0)
	for k := range n {
		t := bigMul(lsqBigComplex{bigReal(sys.C.At(0, k)), bigReal(0)}, x[k])
		re.Add(re, t.re)
		im.Add(im, t.im)
	}
	r, _ := re.Float64()
	i, _ := im.Float64()
	return complex(r, i)
}

func TestLeastSquaresIntegratorCount(t *testing.T) {
	tests := []struct {
		name     string
		num, den []float64
		count    int
		residue  float64
	}{
		{"slow stable pole", []float64{1}, []float64{1, 1 + 1e-7, 1e-7}, 0, 0},
		{"one integrator", []float64{3}, []float64{1, 2, 0}, 1, 1.5},
		{"pole-zero cancellation at origin", []float64{1, 0}, []float64{1, 1, 0}, 0, 0},
		{"triple integrator with dynamics", []float64{1, 2}, []float64{1, 3, 3, 1, 0, 0, 0}, 3, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			count, residue, err := leastSquaresIntegratorLimit(conversionSISO(t, tc.num, tc.den, 0))
			if err != nil {
				t.Fatal(err)
			}
			if count != tc.count || math.Abs(residue-tc.residue) > 1e-9*math.Max(1, math.Abs(tc.residue)) {
				t.Fatalf("count=%d residue=%g, want %d %g", count, residue, tc.count, tc.residue)
			}
		})
	}
}

func TestLeastSquaresDenseSweepSourceN100(t *testing.T) {
	rng := rand.New(rand.NewPCG(100, uint64(sweepDense)))
	source, _ := randomSweepRealization(rng, sweepDense, 0, false, 100, 1, 1)
	_, result, err := source.C2DFit(.05, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 8})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result.RMSRelativeError) || math.IsInf(result.RMSRelativeError, 0) || result.RMSRelativeError > .05 {
		t.Fatalf("rms=%g", result.RMSRelativeError)
	}
}
