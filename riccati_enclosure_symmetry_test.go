package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"sync/atomic"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var encSymCalls, encSymBad atomic.Int64

func init() {
	encSymCheck = func(sym bool) {
		encSymCalls.Add(1)
		if !sym {
			encSymBad.Add(1)
		}
	}
}

// encSymSites returns a check that symT call sites ran since and met only
// bitwise symmetric middle factors.
func encSymSites(t *testing.T) func() {
	t.Helper()
	c0, b0 := encSymCalls.Load(), encSymBad.Load()
	return func() {
		t.Helper()
		if encSymCalls.Load() == c0 {
			t.Error("no symT call site ran")
		}
		if b := encSymBad.Load() - b0; b != 0 {
			t.Errorf("%d symT middle factors not bitwise symmetric", b)
		}
	}
}

// TestDareEnclosureSymTMatchesMul: symT is mul's upper triangle bitwise,
// mirrored, in double-double and in encPrec bits.
func TestDareEnclosureSymTMatchesMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	const n = 6
	a := mat.NewDense(n, n, nil)
	dm := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			a.Set(i, j, rng.NormFloat64())
			if j >= i {
				v := rng.NormFloat64()
				dm.Set(i, j, v)
				dm.Set(j, i, v)
			}
		}
	}
	ad := encDDOf(a)
	dd := encSym(encDDOf(dm).add(encDDOf(dm).scale(1.0/3), 1))
	if !dd.bitSym() {
		t.Fatal("middle factor not symmetric")
	}
	m := dd.mul(ad, false)
	got, want := ad.symT(m), ad.mul(m, true)
	for i := range n {
		for j := i; j < n; j++ {
			for _, p := range []int{i*n + j, j*n + i} {
				if math.Float64bits(got.h[p]) != math.Float64bits(want.h[i*n+j]) || math.Float64bits(got.l[p]) != math.Float64bits(want.l[i*n+j]) {
					t.Fatalf("dd (%d,%d): symT %v %v, mul upper %v %v", p/n, p%n, got.h[p], got.l[p], want.h[i*n+j], want.l[i*n+j])
				}
			}
		}
	}
	ar := encArenas.Get().(*encArena)
	defer func() { ar.reset(); encArenas.Put(ar) }()
	ab, db := encBigOf(ar, a), encBigDD(ar, dd)
	mb := db.mul(ab, false)
	gb, wb := ab.symT(mb), ab.mul(mb, true)
	for i := range n {
		for j := i; j < n; j++ {
			for _, p := range []int{i*n + j, j*n + i} {
				if gb.d[p].Cmp(wb.d[i*n+j]) != 0 {
					t.Fatalf("encPrec (%d,%d): symT %v, mul upper %v", p/n, p%n, gb.d[p], wb.d[i*n+j])
				}
			}
		}
	}
}

// TestDareEnclosureSymmetricMiddleFactors: on the noise-floor and tier
// rows every symT middle factor is bitwise symmetric.
func TestDareEnclosureSymmetricMiddleFactors(t *testing.T) {
	done := encSymSites(t)
	for _, c := range append(append([]dareFloorCase(nil), dareNoiseFloorCases...), dareEnclosureCtrlCase, dareEnclosureBigCase, dareEnclosureWCase) {
		A, B, Q, R, S, E := c.dense()
		for _, ns := range []bool{false, true} {
			o := &RiccatiOpts{S: S, E: E, NoScaling: ns}
			if res, err := dareInputs(A, B, Q, R, o); err == nil && res.accuracy == 0 {
				var dx *mat.Dense
				if res.newtonDelta != nil && mat.Equal(res.X, res.newtonX) {
					dx = res.newtonDelta
				}
				encloseDare(nil, A, B, Q, R, S, E, res.X, res.K, dx)
			}
			Dare(A, B, Q, R, o)
		}
	}
	done()
}

// TestDareZeroSolutionExact: with Q = 0 and a stable pencil the
// stabilizing solution is exactly 0, certified (zeroExact).
func TestDareZeroSolutionExact(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{2, 0.1, 0, 1})
	for _, c := range []struct {
		name string
		A, B *mat.Dense
		E    *mat.Dense
	}{
		{"stable", mat.NewDense(2, 2, []float64{0.5, 2, 0, 0.3}), mat.NewDense(2, 1, []float64{1, 0.3}), nil},
		{"stable E", mat.NewDense(2, 2, []float64{0.5, 2, 0.1, 0.3}), mat.NewDense(2, 1, []float64{1, 0.3}), E},
		{"near unit circle, B = I", mat.NewDense(2, 2, []float64{0.5, 0, 0, 1 - 1e-12}), eye(2), nil},
	} {
		_, m := c.B.Dims()
		res, err := Dare(c.A, c.B, mat.NewDense(2, 2, nil), eye(m), &RiccatiOpts{E: c.E})
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !allZeroDense(res.X) {
			t.Errorf("%s: X = %v, want exactly 0", c.name, res.X.RawMatrix().Data)
		}
	}
}

// rotationOutsideUnitCircle returns the stored cos θ, sin θ of the first
// θ = 0.1k whose c² + s², evaluated exactly, exceeds 1: the stored
// rotation is unstable however close to the unit circle.
func rotationOutsideUnitCircle(t *testing.T) (float64, float64) {
	t.Helper()
	one := big.NewFloat(1)
	for k := 1; k < 64; k++ {
		c, s := math.Cos(0.1*float64(k)), math.Sin(0.1*float64(k))
		bc, bs := new(big.Float).SetPrec(256).SetFloat64(c), new(big.Float).SetPrec(256).SetFloat64(s)
		sum := new(big.Float).SetPrec(256).Mul(bc, bc)
		sum.Add(sum, new(big.Float).SetPrec(256).Mul(bs, bs))
		if sum.Cmp(one) > 0 {
			return c, s
		}
	}
	t.Fatal("no stored rotation with c² + s² > 1")
	return 0, 0
}

// TestDareZeroExactNeverOnNonStablePencil: the gate never certifies X = 0
// (X = K = Q = 0) for a pencil that is not Schur stable, over B and R
// scalings.
func TestDareZeroExactNeverOnNonStablePencil(t *testing.T) {
	c, s := rotationOutsideUnitCircle(t)
	type pencil struct {
		name string
		a, e []float64
	}
	cases := []pencil{
		{"rho 1", []float64{0.5, 0, 0, 1}, nil},
		{"Jordan rho 1", []float64{1, 1, 0, 1}, nil},
		{"1+4e-16", []float64{0.5, 0, 0, 1 + 4e-16}, nil},
		{"rotation c²+s²>1", []float64{c, -s, s, c}, nil},
		{"E unstable", []float64{0.5, 0, 0, 1 + 3e-12}, []float64{1, 0, 0, 1 + 1e-12}},
		{"E singular", []float64{0.5, 0, 0, 0.5}, []float64{1, 0, 0, 0}},
	}
	tiny := math.Ldexp(1, -480)
	cases = append(cases, pencil{"1+4e-16 scaled 2^-480", []float64{0.5 * tiny, 0, 0, (1 + 4e-16) * tiny}, []float64{tiny, 0, 0, tiny}})
	for _, p := range cases {
		for _, b := range [][]float64{{1, 0, 0, 1}, {1, 0, 0, 1e-8}, {0, 0, 0, 0}, {1e-150, 0, 0, 1e-150}} {
			for _, r := range []float64{1, 1e-12, 1e300} {
				A := mat.NewDense(2, 2, append([]float64(nil), p.a...))
				var E *mat.Dense
				if p.e != nil {
					E = mat.NewDense(2, 2, append([]float64(nil), p.e...))
				}
				B := mat.NewDense(2, 2, append([]float64(nil), b...))
				R := mat.NewDense(2, 2, []float64{r, 0, 0, r})
				Z := mat.NewDense(2, 2, nil)
				if out, _, _, _ := encloseDare(nil, A, B, Z, R, nil, E, Z, mat.NewDense(2, 2, nil), nil); out != encFailed {
					t.Errorf("%s B=%v R=%g: enclosure %v of X = 0, want failed", p.name, b, r, out)
				}
			}
		}
	}
}
