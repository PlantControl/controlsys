package controlsys

import (
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func lsimDelayMatrixCase() (*System, *mat.Dense, []float64) {
	A := mat.NewDense(6, 6, []float64{-0.67221666560589832, -1.5330301353727724, -0.57185401605934694, -0.13307448791135901, -0.34294421675919579, -0.78450680936632111, 0.1965449597835473, -0.22907707453875259, 1.3908260717454115, 1.1575291455355019, -0.29446013180857339, 0.31432888826307376, -0.7936825880108509, 2.2943756522009209, -3.3813574483965985, -2.2051204611144275, 1.3476963600941261, 0.50615806232807503, 0.90848283765569482, -0.44100413087577395, 0.55472762742062243, -0.49208732780041281, -0.54617080362854475, -0.5297573351460213, -0.47550175048114923, 2.4663470799999487, -1.613414036930408, -0.23157372662615028, -0.22687312660822856, 0.27988131206146183, 0.50430924599867921, -1.1350099357473917, 2.1448636976409095, 1.5119818488183119, -0.73748640081604588, -0.62019488419973168})
	B := mat.NewDense(6, 2, []float64{0.57232968759273095, -1.5255834112700393, -0.63973185165523572, 0.44086973098121884, -2.5351807588390569, 1.8641605274682151, -0.098514964564473576, 0.074141731713248504, -1.2303635891686617, 0.7947655014758217, -0.12417169296963168, -0.30633171301824202})
	C := mat.NewDense(3, 6, []float64{-2.2913679207349937, 1.6444097562115003, 0.13359364806660243, 0.28165129490319446, 1.8440236891970927, -0.78369728362234703, 0.81542072654223141, -0.3647208661709434, -1.5170873325374958, 0.11007323788681621, 0.51509539529324921, 1.1467018509327598, 1.7470966995210397, -1.8521467936593028, 1.5675576667591671, 0.1164403510395224, 0.36581721779889825, 1.0114275340796992})
	D := mat.NewDense(3, 2, []float64{0.69545869669340166, 1.0745833686707513, 0.0096813096767178197, 0.37283961538383598, -0.47382433909252386, 0.86693463951900451})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		panic(err)
	}
	sys.Delay = mat.NewDense(3, 2, []float64{0.65000000000000002, 0.34999999999999998, 0.55910822365319968, 0.25910822365319963, 0.70000000000000007, 0.40000000000000002})
	blocks := [2][]float64{
		{0.60644760174327716, -0.33806324787336667, -0.85052597473191049, -0.87784878122269105, 1.0842252942246855, -1.3779844802368417, 0.38547450647841153, -0.67944295299363366, 1.293310231878952},
		{-0.14344867620339063, -0.30682709535364422, 1.2307689773723436, -0.65959858660732784, 1.274960307381936, -1.1974920049475746, -0.86616291818304414, 0.30822704842433934, -1.3212803059939797},
	}
	t := make([]float64, 121)
	u := mat.NewDense(121, 2, nil)
	for s := range t {
		t[s] = float64(s) * 0.05
		for j := range 2 {
			u.Set(s, j, blocks[j][s/15])
		}
	}
	return sys, u, t
}

func lsimSISOChannelSum(t *testing.T, sys *System, u *mat.Dense, tt []float64) *mat.Dense {
	t.Helper()
	n, m, p := sys.Dims()
	want := mat.NewDense(p, len(tt), nil)
	for i := range p {
		for j := range m {
			b := mat.NewDense(n, 1, nil)
			b.Copy(sys.B.Slice(0, n, j, j+1))
			c := mat.NewDense(1, n, nil)
			c.Copy(sys.C.Slice(i, i+1, 0, n))
			ch, err := New(sys.A, b, c, mat.NewDense(1, 1, []float64{sys.D.At(i, j)}), sys.Dt)
			if err != nil {
				t.Fatal(err)
			}
			if sys.E != nil {
				ch.E = mat.DenseCopyOf(sys.E)
			}
			ch.InputDelay = []float64{sys.Delay.At(i, j)}
			uj := mat.NewDense(len(tt), 1, nil)
			uj.Copy(u.Slice(0, len(tt), j, j+1))
			r, err := Lsim(ch, uj, tt, nil)
			if err != nil {
				t.Fatal(err)
			}
			for s := range tt {
				want.Set(i, s, want.At(i, s)+r.Y.At(0, s))
			}
		}
	}
	return want
}

func assertLsimMatches(t *testing.T, label string, got, want *mat.Dense, tol float64) {
	t.Helper()
	p, steps := want.Dims()
	scale := 1.0
	for i := range p {
		for s := range steps {
			scale = math.Max(scale, math.Abs(want.At(i, s)))
		}
	}
	bad := 0
	for i := range p {
		for s := range steps {
			if g, w := got.At(i, s), want.At(i, s); math.Abs(g-w) > tol*scale {
				if bad++; bad <= 5 {
					t.Errorf("%s: out %d sample %d: Lsim %.17g, want %.17g", label, i, s, g, w)
				}
			}
		}
	}
}

// descriptorForm returns T·A, T·B with E = T for a fixed non-symmetric T.
func descriptorForm(sys *System) *System {
	n, _, _ := sys.Dims()
	T := mat.NewDense(n, n, nil)
	for i := range n {
		T.Set(i, i, 1.5+0.25*float64(i))
		for j := i + 1; j < n; j++ {
			T.Set(i, j, 0.3/float64(j-i))
		}
		if i > 0 {
			T.Set(i, i-1, -0.2)
		}
	}
	out := sys.Copy()
	out.A.Mul(T, sys.A)
	out.B.Mul(T, sys.B)
	out.E = T
	return out
}

// H6IH3S: channel (2,1) delay 0.40 = 8 samples; u(:,1) switches at sample 45,
// so sample 53 must carry the post-switch value. The Delay matrix splits into
// input and output delays whose fractions sum to one sample, and rounding of
// the split moved such channels by a sample.
func TestLsimDelayMatrixMatchesSISOSuperposition(t *testing.T) {
	sys, u, tt := lsimDelayMatrixCase()
	got, err := Lsim(sys, u, tt, nil)
	if err != nil {
		t.Fatal(err)
	}
	oracle := lsimShiftedZOHOracle(sys, u, 0.05)
	assertLsimMatches(t, "SISO sum", got.Y, lsimSISOChannelSum(t, sys, u, tt), 1e-10)
	assertLsimMatches(t, "shifted ZOH", got.Y, oracle, 1e-10)
	if w := oracle.At(2, 53); math.Abs(w-0.86464593885308316) > 1e-10 {
		t.Errorf("oracle out 2 sample 53 = %.17g", w)
	}
}

func TestLsimDescriptorDelayMatrixMatchesShiftedZOH(t *testing.T) {
	base, u, tt := lsimDelayMatrixCase()
	got, err := Lsim(descriptorForm(base), u, tt, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertLsimMatches(t, "shifted ZOH", got.Y, lsimShiftedZOHOracle(base, u, 0.05), 1e-9)
}

// Delay[i][j] = in[j] + out[i] with fractional parts summing to whole samples
// on some channels, beside extra InputDelay and OutputDelay.
func TestLsimDelayMatrixWithIODelaysMatchesShiftedZOH(t *testing.T) {
	base, u, tt := lsimDelayMatrixCase()
	in := []float64{0.13, 0.07}
	out := []float64{0.02, 0.03, 0.12}
	delay := mat.NewDense(3, 2, nil)
	for i := range 3 {
		for j := range 2 {
			delay.Set(i, j, in[j]+out[i])
		}
	}
	cases := []struct {
		name        string
		inputDelay  []float64
		outputDelay []float64
	}{
		{"delay only", nil, nil},
		{"input", []float64{0.0137, 0.05}, nil},
		{"output", nil, []float64{0.03, 0, 0.0263}},
		{"both", []float64{0.02, 0.0412}, []float64{0.03, 0.0091, 0.1}},
	}
	for _, c := range cases {
		for _, descriptor := range []bool{false, true} {
			sys := base.Copy()
			sys.Delay = delay
			sys.InputDelay, sys.OutputDelay = c.inputDelay, c.outputDelay
			ref := sys.Copy()
			label := c.name
			if descriptor {
				sys, label = descriptorForm(sys), label+" descriptor"
			}
			got, err := Lsim(sys, u, tt, nil)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			assertLsimMatches(t, label, got.Y, lsimShiftedZOHOracle(ref, u, 0.05), 1e-9)
		}
	}
}

// xbHoldKernel returns e^{A h} and ∫₀ʰ e^{A s} ds·b by a scaled Taylor series
// of the augmented exponential [A b; 0 0].
func xbHoldKernel(a, b xbMat, h *big.Float) (phi, gamma xbMat) {
	n, prec := a.r, a.prec
	m := xbNew(n+1, n+1, prec)
	for i := range n {
		for j := range n {
			m.at(i, j).Mul(a.at(i, j), h)
		}
		m.at(i, n).Mul(b.at(i, 0), h)
	}
	norm, _ := xbMaxAbs(m).Float64()
	squarings := 0
	for norm*float64(n+1) > 0.5 {
		norm /= 2
		squarings++
	}
	m = xbScale(new(big.Float).SetPrec(prec).SetMantExp(big.NewFloat(1), -squarings), m)
	exp, term := xbEye(n+1, prec), xbEye(n+1, prec)
	for k := 1; k <= 80; k++ {
		term = xbScale(new(big.Float).SetPrec(prec).Quo(big.NewFloat(1), big.NewFloat(float64(k))), xbMul(term, m))
		exp = xbAxpy(exp, 1, term)
	}
	for range squarings {
		exp = xbMul(exp, exp)
	}
	phi, gamma = xbNew(n, n, prec), xbNew(n, 1, prec)
	for i := range n {
		for j := range n {
			phi.at(i, j).Set(exp.at(i, j))
		}
		gamma.at(i, 0).Set(exp.at(i, n))
	}
	return phi, gamma
}

// lsimShiftedZOHOracle samples, at 320 bits, Σ_j of channel (i, j) driven by
// the ZOH input u(:, j) shifted by its total delay InputDelay[j] + Delay[i][j]
// + OutputDelay[i]; delays within 1e-9 samples of the grid snap to it as in
// MATLAB. E must be nonsingular.
func lsimShiftedZOHOracle(sys *System, u *mat.Dense, dt float64) *mat.Dense {
	const prec = 320
	n, m, p := sys.Dims()
	steps, _ := u.Dims()
	a, b := xbFrom(sys.A, n, n, prec), xbFrom(sys.B, n, m, prec)
	if sys.E != nil {
		a, _, _ = xbSolve(xbFrom(sys.E, n, n, prec), a)
		b, _, _ = xbSolve(xbFrom(sys.E, n, n, prec), b)
	}
	bigDt := new(big.Float).SetPrec(prec).SetFloat64(dt)
	y := mat.NewDense(p, steps, nil)
	for i := range p {
		ci := xbNew(1, n, prec)
		for k := range n {
			ci.at(0, k).SetFloat64(sys.C.At(i, k))
		}
		acc := make([]*big.Float, steps)
		for s := range acc {
			acc[s] = new(big.Float).SetPrec(prec)
		}
		for j := range m {
			tau := new(big.Float).SetPrec(prec)
			if sys.InputDelay != nil {
				tau.Add(tau, big.NewFloat(sys.InputDelay[j]))
			}
			if sys.OutputDelay != nil {
				tau.Add(tau, big.NewFloat(sys.OutputDelay[i]))
			}
			if sys.Delay != nil {
				tau.Add(tau, big.NewFloat(sys.Delay.At(i, j)))
			}
			q := new(big.Float).SetPrec(prec).Quo(tau, bigDt)
			qf, _ := q.Float64()
			whole := math.Floor(qf)
			frac := new(big.Float).SetPrec(prec).Sub(q, big.NewFloat(whole))
			if math.Abs(qf-math.Round(qf)) < 1e-9 {
				whole = math.Round(qf)
				frac.SetInt64(0)
			}
			bj := xbNew(n, 1, prec)
			for k := range n {
				bj.at(k, 0).Set(b.at(k, j))
			}
			first := new(big.Float).SetPrec(prec).Mul(frac, bigDt)
			second := new(big.Float).SetPrec(prec).Sub(bigDt, first)
			phiF, gamF := xbHoldKernel(a, bj, first)
			phiS, gamS := xbHoldKernel(a, bj, second)
			in := func(s int) *big.Float {
				s -= int(whole)
				if s < 0 {
					return new(big.Float)
				}
				return big.NewFloat(u.At(s, j))
			}
			x := xbNew(n, 1, prec)
			dij := big.NewFloat(sys.D.At(i, j))
			v := new(big.Float).SetPrec(prec)
			for s := range steps {
				hold := in(s)
				if frac.Sign() != 0 {
					hold = in(s - 1)
				}
				acc[s].Add(acc[s], xbMul(ci, x).at(0, 0))
				acc[s].Add(acc[s], v.Mul(dij, hold))
				x = xbAdd(xbMul(phiF, x), xbScale(hold, gamF))
				x = xbAdd(xbMul(phiS, x), xbScale(in(s), gamS))
			}
		}
		for s := range steps {
			f, _ := acc[s].Float64()
			y.Set(i, s, f)
		}
	}
	return y
}

func xbAdd(a, b xbMat) xbMat {
	o := xbNew(a.r, a.c, a.prec)
	for i := range o.d {
		o.d[i].Add(a.d[i], b.d[i])
	}
	return o
}
