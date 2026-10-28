package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// matchedOracleZeros refines starts to zeros of the SISO G(s) of o at
// stepOraclePrec bits by Newton with a 1e-40 relative difference quotient.
func matchedOracleZeros(t *testing.T, o stepOraclePencil, starts []complex128) []complex128 {
	t.Helper()
	g := func(s stepOracleComplex) stepOracleComplex {
		re, im := o.transfer(s)
		if re == nil {
			t.Fatalf("oracle transfer singular at %v", s.complex128())
		}
		return stepOracleComplex{re[0][0], im[0][0]}
	}
	out := make([]complex128, len(starts))
	for k, x0 := range starts {
		s := stepOracleComplex{stepOracleFloat(real(x0)), stepOracleFloat(imag(x0))}
		for range 60 {
			h := stepOracleFloat(1e-40 * math.Max(1, cmplx.Abs(s.complex128())))
			g0 := g(s)
			g1 := g(stepOracleComplex{stepOracleNew().Add(s.re, h), stepOracleNew().Set(s.im)})
			dg := stepOracleComplex{stepOracleNew().Sub(g1.re, g0.re), stepOracleNew().Sub(g1.im, g0.im)}
			if dg.re.Sign() == 0 && dg.im.Sign() == 0 {
				break
			}
			q := stepOracleQuo(g0, dg)
			step := stepOracleComplex{stepOracleNew().Mul(q.re, h), stepOracleNew().Mul(q.im, h)}
			s.re.Sub(s.re, step.re)
			s.im.Sub(s.im, step.im)
			if cmplx.Abs(step.complex128()) <= 1e-60*math.Max(1, cmplx.Abs(s.complex128())) {
				break
			}
		}
		out[k] = s.complex128()
		if cmplx.Abs(out[k]-x0) > 1e-3*math.Max(1, cmplx.Abs(x0)) {
			t.Fatalf("oracle zero from %v moved to %v", x0, out[k])
		}
	}
	return out
}

// matchedOracleResponse is MATLAB c2d(G,T,'matched') at z: zeros e^{zT},
// poles e^{pT}, zeros at −1 up to relative degree one, gain matching G(0).
func matchedOracleResponse(zeros, poles []complex128, dc, dt float64, z complex128) complex128 {
	T := complex(dt, 0)
	h := complex(dc, 0)
	for _, q := range zeros {
		e := cmplx.Exp(q * T)
		h *= (z - e) / (1 - e)
	}
	for _, p := range poles {
		e := cmplx.Exp(p * T)
		h *= (1 - e) / (z - e)
	}
	for range max(len(poles)-len(zeros)-1, 0) {
		h *= (z + 1) / 2
	}
	return h
}

func matchedOracleTransfer(o stepOraclePencil, z complex128) complex128 {
	re, im := o.transfer(stepOracleComplex{stepOracleFloat(real(z)), stepOracleFloat(imag(z))})
	if re == nil {
		return cmplx.NaN()
	}
	return stepOracleComplex{re[0][0], im[0][0]}.complex128()
}

// checkMatchedAgainstOracle compares matched C2D d of sys with the oracle
// model: every pencil pole maps to a pole of d at e^{λT}, and the exact
// transfer function of d's float64 matrices matches the matched ZPK.
func checkMatchedAgainstOracle(t *testing.T, name string, sys, d *System, dt float64) {
	t.Helper()
	cont := newStepOraclePencil(sys)
	starts, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, starts)
	if !ok {
		t.Fatalf("%s: oracle poles unresolved", name)
	}
	zstarts, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	zeros := matchedOracleZeros(t, cont, zstarts)
	dc, _ := new(big.Float).Set(func() *big.Float {
		re, _ := cont.transfer(stepOracleComplex{stepOracleFloat(0), stepOracleFloat(0)})
		return re[0][0]
	}()).Float64()

	got, err := d.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range modes.poles {
		e := cmplx.Exp(l * complex(dt, 0))
		best := math.Inf(1)
		for _, q := range got {
			best = math.Min(best, cmplx.Abs(q-e))
		}
		if !(best <= 1e-9*cmplx.Abs(e)+1e-14) {
			t.Errorf("%s dt=%g: no pole of d near e^{λT} = %v (λ = %v): nearest %.3g away", name, dt, e, l, best)
		}
	}
	disc := newStepOraclePencil(d)
	for _, z := range []complex128{1, cmplx.Exp(1e-3i), cmplx.Exp(1e-2i), cmplx.Exp(0.1i), cmplx.Exp(1i), cmplx.Exp(3i), -0.5} {
		want := matchedOracleResponse(zeros, modes.poles, dc, dt, z)
		g := matchedOracleTransfer(disc, z)
		if rel := cmplx.Abs(g-want) / math.Max(cmplx.Abs(want), 1e-12); !(rel <= 1e-7) {
			t.Errorf("%s dt=%g: H(%.6g) = %v, matched ZPK %v (rel %.3g)", name, dt, z, g, want, rel)
		}
	}
}

// matchedClusterDiagonal is the EGHOKQ literal: eight equally spaced real
// poles whose discrete images e^{pT} crowd at 1 for small T.
func matchedClusterDiagonal(t *testing.T) *System {
	t.Helper()
	const n = 8
	A, B, C := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		A.Set(i, i, -0.2*float64(i+1))
		B.Set(i, 0, 1)
		C.Set(0, i, float64(i%3)-0.7)
	}
	sys, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Matched C2D poles are e^{pT} to rounding, as MATLAB c2d(...,'matched');
// a companion realization of the discrete ZPK lost 4e-4 to 2e-2 here
// (EGHOKQ).
func TestC2DMatchedClusteredPolesMapExactly(t *testing.T) {
	sys := matchedClusterDiagonal(t)
	for _, dt := range []float64{0.1, 0.01, 0.001} {
		d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatal(err)
		}
		checkMatchedAgainstOracle(t, "diag cluster", sys, d, dt)
		if stable, err := d.IsStable(); err != nil || !stable {
			t.Errorf("dt=%g: IsStable = %v, %v; want stable", dt, stable, err)
		}
	}
}

// matchedStiffDescriptor is a full-rank stiff descriptor (sweep model
// multi:984, EGHOKQ): two stiff poles, six slow ones clustered near 1 after
// matching at T = 0.01, complex zeros over real poles.
func matchedStiffDescriptor(t *testing.T) *System {
	t.Helper()
	a := []float64{0.099678404134092694, -0.29817703588070338, -0.73002630414450143, 0.32551045392582417, -0.29052629786625905, 0.20626850583265294, -0.90955187586410124, -0.60161682863754351, -0.068422659202509908, 0.0024292012598285751, 1.473572251074359, -0.18978226300792569, 0.28331725695560922, -1.3002205019557951, 0.10948129896196887, 0.18900775397842293, 0.1323695508259698, 0.30738780888817174, -0.26989464423823739, -0.23840481183872078, -0.18085018970663908, -0.37100735068214546, -0.0019047841467216158, 0.28379022799160947, -0.87296792603976403, -0.074613106647409805, 0.85685441763246628, -0.084107885048652853, 0.12074363964550444, 0.72870812148849629, 1.353394828865915, -0.31358975479570589, -0.084878863827736434, -0.18679526963799711, -0.99102904936400149, 0.25487919624853372, -0.87920117957699395, 0.056110414643104227, -0.42675578679216153, -0.17310905572520188, 0.32156319907341091, 0.061060142069507738, -0.52538997286088551, 0.0011399196362768744, 0.63287251143901424, 1.5003951080346465, 1.2677189262676456, 1.2575179306789377, -1.5392285585013399, -0.89282298169668151, -0.41301466308150792, 0.40059358604733514, -0.18899800917850673, 0.38759021322889875, 1.9328194277243369, 1.734004853431387, -0.59243861952422627, 0.32199102101658755, 1.7364347053044178, -0.61554446960881692, 1.8138686595489042, -0.95700385853439218, 1.4413089733789104, 1.7334225847682956}
	e := []float64{-0.024779906300603206, -0.017961725644720285, -0.030770767715695153, -0.37111043077484307, 0.074161582818630042, 0.11470590389526469, 0.47452755449835432, -0.2903522531042147, -0.065705630718624952, 0.38061385272181225, 0.053304113293433139, 0.48057557177198912, -0.066298104795596086, -0.14967941124142783, -0.1606617287241148, 0.21827429801174741, 0.17641477910902917, 0.22067517300199424, 0.1758975346998288, 0.67611793694454791, 0.33122263849099831, 0.55177355132251849, -0.29287352509631048, 0.51409269375593103, -0.24454905632031065, -0.76356927724325674, -0.048368496987983345, -0.24507501865466952, 0.037575555884217481, -0.32804938615695084, -0.44927578469597812, 0.67032245014308012, 0.87274666648084231, 0.16934947801483735, -0.34390844092496303, -0.000531905691105492, -0.059499380031447641, -0.72921357797288167, 0.0090358591723668415, 0.029079411743154142, 0.54463964529488562, -0.16303301636798803, 0.13947261688645848, -0.47922802007284077, -0.06089130921039064, 0.66552145810116592, -0.39164721853169338, -0.084511310363256778, 0.0086861133363712734, 0.32106207999896613, 0.27427228431310741, -0.67305697404632114, -0.39695882230656832, 0.10972213802097409, -0.265818218397188, 0.46951489577493677, -0.46603158573736447, 0.50038150690547678, 0.30909126116574226, 0.086375851374910176, -0.54771312871860323, 0.076634320003227924, -0.46874987355582132, -0.024661531075382669}
	b := []float64{0.42636471330159714, -0.42867301892500748, 0.0741494476202849, 0.8029744225718245, 1.2114381884040244, 0.25913392616800585, -0.74243710206226576, 0.7815761821827838}
	c := []float64{-0.083582869370514692, -1.1317997139268827, 0.32616361164631497, -1.4855200086578249, -1.0068372131417613, -0.40089287882720304, -0.30427619019457353, 1.2908215996805175}
	sys, err := NewDescriptor(mat.NewDense(8, 8, a), mat.NewDense(8, 1, b), mat.NewDense(1, 8, c), mat.NewDense(1, 1, []float64{1.546413191770422}), mat.NewDense(8, 8, e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestC2DMatchedStiffDescriptorFollowsMatchedZPK(t *testing.T) {
	const dt = 0.01
	sys := matchedStiffDescriptor(t)
	d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	checkMatchedAgainstOracle(t, "multi:984", sys, d, dt)
	if stable, err := d.IsStable(); err != nil || stable {
		t.Errorf("IsStable = %v, %v; the pencil has the unstable pole 0.74", stable, err)
	}
}

// With an input delay the stiff descriptor keeps the delay-free matched
// route and gains z^-k (RL6IT7 am.9).
func TestC2DMatchedDelayedStiffDescriptorFollowsMatchedZPK(t *testing.T) {
	const dt = 0.01
	sys := matchedStiffDescriptor(t)
	sys.InputDelay = []float64{3 * dt}
	d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	core := matchedStiffDescriptor(t)
	for _, z := range []complex128{cmplx.Exp(1e-3i), cmplx.Exp(0.1i), cmplx.Exp(2i)} {
		want, err := core.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatal(err)
		}
		w := matchedOracleTransfer(newStepOraclePencil(want), z) / (z * z * z)
		g, err := d.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		if rel := cmplx.Abs(g[0][0]-w) / cmplx.Abs(w); !(rel <= 1e-9) {
			t.Errorf("H(%.6g) = %v, delay-free matched times z^-3 = %v (rel %.3g)", z, g[0][0], w, rel)
		}
	}
	undelayed := *d
	undelayed.InputDelay, undelayed.OutputDelay, undelayed.Delay = nil, nil, nil
	checkMatchedAgainstOracle(t, "multi:984 delayed", core, &undelayed, dt)
}

// Matched C2D section shapes (complex pole pairs, complex zeros over real
// poles, repeated poles, zeros at −1 for relative degree > 1) follow the
// matched ZPK of the given continuous roots.
func TestC2DMatchedSectionShapesFollowMatchedZPK(t *testing.T) {
	const dt = 0.1
	cases := []struct {
		name         string
		zeros, poles []complex128
		gain         float64
		distinct     bool
	}{
		{"complex poles, relative degree 2", []complex128{-3}, []complex128{-1 + 2i, -1 - 2i, -0.5}, 2, true},
		{"complex zeros over complex poles", []complex128{-0.3 + 4i, -0.3 - 4i}, []complex128{-1 + 2i, -1 - 2i, -2 + 1i, -2 - 1i}, 1, true},
		{"complex zeros over real poles", []complex128{-1 + 1i, -1 - 1i}, []complex128{-1, -2, -3}, -3, true},
		{"nonminimum phase mixed", []complex128{2, -4}, []complex128{-0.1 + 5i, -0.1 - 5i, -1}, 0.5, true},
		{"biproper", []complex128{-5, -0.2 + 1i, -0.2 - 1i}, []complex128{-1, -0.5 + 3i, -0.5 - 3i}, 1.5, true},
		{"repeated poles", nil, []complex128{-1, -1, -1}, 1, false},
	}
	for _, c := range cases {
		z, err := NewZPK(c.zeros, c.poles, c.gain, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := z.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		d, err := res.Sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		dc := complex(c.gain, 0)
		for _, q := range c.zeros {
			dc *= -q
		}
		for _, p := range c.poles {
			dc /= -p
		}
		disc := newStepOraclePencil(d)
		for _, s := range []complex128{1, cmplx.Exp(0.05i), cmplx.Exp(1i), cmplx.Exp(3i), -0.5} {
			want := matchedOracleResponse(c.zeros, c.poles, real(dc), dt, s)
			g := matchedOracleTransfer(disc, s)
			if rel := cmplx.Abs(g-want) / math.Max(cmplx.Abs(want), 1e-12); !(rel <= 1e-8) {
				t.Errorf("%s: H(%.4g) = %v, matched ZPK %v (rel %.3g)", c.name, s, g, want, rel)
			}
		}
		if !c.distinct {
			continue
		}
		got, err := d.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range c.poles {
			e := cmplx.Exp(p * dt)
			best := math.Inf(1)
			for _, q := range got {
				best = math.Min(best, cmplx.Abs(q-e))
			}
			if !(best <= 1e-12) {
				t.Errorf("%s: no pole of d near %v: nearest %.3g away", c.name, e, best)
			}
		}
	}
}

// A discrete pole e^{pT} near 1e188 overflows the cascade's C; matched C2D
// keeps the companion route's answer there and never returns Inf entries.
func TestC2DMatchedOverflowingSectionKeepsFiniteOrError(t *testing.T) {
	z, err := NewZPK([]complex128{-1.0241915648397435, -10356.095644626927}, []complex128{-0.5217603941516756, 4320.677636617519}, 0.9231445700524119, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := z.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	d, err := res.Sys.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		return
	}
	for name, m := range map[string]*mat.Dense{"A": d.A, "B": d.B, "C": d.C, "D": d.D} {
		if !finiteSlice(m.RawMatrix().Data) {
			t.Errorf("%s = %v is not finite", name, m.RawMatrix().Data)
		}
	}
}
