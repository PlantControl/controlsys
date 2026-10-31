package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fohBigResponse evaluates C(zI−A)⁻¹B + D of a SISO explicit system in
// fzDiscPrec arithmetic so that only the float64 coefficients contribute
// error.
func fohBigResponse(sys *System, z complex128) complex128 {
	n, _, _ := sys.Dims()
	k := 2 * n
	m := make([][]*big.Float, k)
	for i := range k {
		m[i] = make([]*big.Float, k+1)
		for j := range k + 1 {
			m[i][j] = fzDiscBig()
		}
	}
	for i := range n {
		for j := range n {
			m[i][j].SetFloat64(-sys.A.At(i, j))
			m[n+i][n+j].SetFloat64(-sys.A.At(i, j))
		}
		m[i][i].Add(m[i][i], fzDiscBig().SetFloat64(real(z)))
		m[n+i][n+i].Add(m[n+i][n+i], fzDiscBig().SetFloat64(real(z)))
		m[i][n+i].SetFloat64(-imag(z))
		m[n+i][i].SetFloat64(imag(z))
		m[i][k].SetFloat64(sys.B.At(i, 0))
	}
	for c := range k {
		p := c
		for r := c + 1; r < k; r++ {
			if new(big.Float).Abs(m[r][c]).Cmp(new(big.Float).Abs(m[p][c])) > 0 {
				p = r
			}
		}
		m[c], m[p] = m[p], m[c]
		for r := c + 1; r < k; r++ {
			f := fzDiscBig().Quo(m[r][c], m[c][c])
			for j := c; j <= k; j++ {
				m[r][j].Sub(m[r][j], fzDiscBig().Mul(f, m[c][j]))
			}
		}
	}
	x := make([]*big.Float, k)
	for r := k - 1; r >= 0; r-- {
		s := fzDiscBig().Set(m[r][k])
		for j := r + 1; j < k; j++ {
			s.Sub(s, fzDiscBig().Mul(m[r][j], x[j]))
		}
		x[r] = fzDiscBig().Quo(s, m[r][r])
	}
	gr := fzDiscBig().SetFloat64(sys.D.At(0, 0))
	gi := fzDiscBig()
	for i := range n {
		c := fzDiscBig().SetFloat64(sys.C.At(0, i))
		gr.Add(gr, fzDiscBig().Mul(c, x[i]))
		gi.Add(gi, fzDiscBig().Mul(c, x[n+i]))
	}
	re, _ := gr.Float64()
	im, _ := gi.Float64()
	return complex(re, im)
}

// TestC2DFOHFastUnstablePoleAtRealizationFloor pins K5SBGD as a precision
// limit: with poles +152.7, −1.33±1.50i, −0.50 and dt = 0.1 the MATLAB FOH
// realization Ad = e^{A·dt}, Bd = Γ₀ + (Ad−I)Γ₁, Dd = D + CΓ₁ has
// ‖Bd‖ ≈ 1e12 and Dd ≈ 3.8e5, so rounding the exact 320-bit Bd/Dd to float64
// already costs ≈ 4e-4 against the pencil-mode oracle while ZOH is ≈ 1e-9.
// With (Ad, C) fixed, Bd and Dd are unique, so the gap is the realization's,
// not the algorithm's; only a non-MATLAB state basis (e.g. modal, ≈ 5e-11)
// gets below it. The test checks the computed Bd/Dd equal the exact ones to
// float64 and the response stays at the floor.
func TestC2DFOHFastUnstablePoleAtRealizationFloor(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{0.4772379316723846, -0.34242695208978235, 3.0044535097805705, 1.0637520253071637, -1.2420874299569429, -0.20003188408120062, 0.11473274499822719, 0.16959349801662824, -1.0047774264355196, -1.3851841703443815, 0.27218107556972498, -0.44627300141694909, -1.4307811032381565, 0.41811665441134566, -0.018699223791052828, -0.035841990052581166})
	B := mat.NewDense(4, 1, []float64{1.3199170244111194, -0.44278772947457523, 1.2622973904592818, 0.78160796332474647})
	C := mat.NewDense(1, 4, []float64{-0.080985129462101277, -1.1052854667425092, 0.78622926236261204, 0.27242214673026977})
	D := mat.NewDense(1, 1, []float64{-0.88535189884772292})
	E := mat.NewDense(4, 4, []float64{0.77541913578182631, 0.26888702257843089, -0.69373623189213407, -0.46151752108035754, 0.19316038868578628, 0.2419377602738316, 0.12900620121465473, 0.36675083878649944, 0.070269656365002778, 0.58950665549240566, 0.63545224088440533, 0.61267005648051132, 0.087500165625870027, -0.61686056653614862, -0.31852948589661106, 0.82033178766149961})
	desc, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := desc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(desc, poles)
	if !ok {
		t.Fatal("oracle modes unavailable")
	}
	ex, err := desc.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	const dt = 0.1
	const n = 4
	phi, gam, g := fzDiscHold(n, 1, ex.A.RawMatrix().Data, ex.B.RawMatrix().Data, dt, true)
	pg := fzDiscMul(phi, g, n, n, 1)
	bd := make([]*big.Float, n)
	for i := range n {
		bd[i] = fzDiscBig().Add(gam[i], pg[i])
		bd[i].Sub(bd[i], g[i])
	}
	cg := fzDiscMul(fzDiscBigOf(ex.C.RawMatrix().Data), g, 1, n, 1)
	dd, _ := fzDiscBig().Add(cg[0], fzDiscBig().SetFloat64(ex.D.At(0, 0))).Float64()
	exact := &System{A: mat.NewDense(n, n, fzDiscRound(phi)), B: mat.NewDense(n, 1, fzDiscRound(bd)), C: ex.C, D: mat.NewDense(1, 1, []float64{dd}), Dt: dt}

	thetas := []float64{0.05, 0.5, 2, 3}
	relErr := func(got, want complex128) float64 {
		return cmplx.Abs(got-want) / math.Max(1, cmplx.Abs(want))
	}
	floor := 0.0
	for _, th := range thetas {
		z := cmplx.Exp(complex(0, th))
		floor = math.Max(floor, relErr(fohBigResponse(exact, z), modes.foh(0, 0, dt, z)))
	}
	for _, tc := range []struct {
		name string
		sys  *System
	}{{"descriptor", desc}, {"explicit", ex}} {
		t.Run(tc.name, func(t *testing.T) {
			foh, err := tc.sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
			if err != nil {
				t.Fatal(err)
			}
			zoh, err := tc.sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
			if err != nil {
				t.Fatal(err)
			}
			if foh.E != nil || zoh.E != nil {
				t.Fatal("discrete result is descriptor; big-float evaluator assumes explicit")
			}
			fohWorst, zohWorst := 0.0, 0.0
			for _, th := range thetas {
				z := cmplx.Exp(complex(0, th))
				fohWorst = math.Max(fohWorst, relErr(fohBigResponse(foh, z), modes.foh(0, 0, dt, z)))
				zohWorst = math.Max(zohWorst, relErr(fohBigResponse(zoh, z), modes.zoh(0, 0, dt, z)))
			}
			t.Logf("foh %.3g floor %.3g zoh %.3g", fohWorst, floor, zohWorst)
			if zohWorst > 1e-8 {
				t.Errorf("ZOH response error %.3g > 1e-8", zohWorst)
			}
			if floor < 100*zohWorst {
				t.Errorf("realization floor %.3g not above 100·ZOH %.3g", floor, zohWorst)
			}
			if fohWorst > 4*floor {
				t.Errorf("FOH response error %.3g > 4·realization floor %.3g", fohWorst, floor)
			}
			var diff mat.Dense
			diff.Sub(foh.B, exact.B)
			if r := mat.Norm(&diff, math.Inf(1)) / mat.Norm(exact.B, math.Inf(1)); r > 1e-12 {
				t.Errorf("Bd relative error %.3g > 1e-12", r)
			}
			if r := math.Abs(foh.D.At(0, 0)-dd) / math.Abs(dd); r > 1e-12 {
				t.Errorf("Dd relative error %.3g > 1e-12", r)
			}
		})
	}
}
