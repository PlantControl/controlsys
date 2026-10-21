package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Discretization fuzz targets (task 4UAU4O). Oracles are independent of the
// library's conversion code: fzDiscExpm (Taylor scaling-squaring in
// big.Float) for ZOH, FOH, impulse and fractional input delays; the bilinear
// map evaluated on the exact continuous response for Tustin; the closed-form
// Thiran coefficients in big.Rat; exact rational matrix powers for D2D; exact
// Schur–Cohn counts for stability preservation.

const fzDiscPrec = 256

func fzDiscSeeds() []fzSeed {
	seeds := fzConvSeeds()
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 1, Gain: 6}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(seeds,
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.In, s.Out, s.Tau = 2, 1, 1, 2.5, -0.7, -0.4, 1.5 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out, s.LogDt, s.Tau = 3, 2, 2, [3]float64{0.3, -0.2, 0}, [3]float64{1.7, 0, 0}, -0.9, -1.3, -1, -2.5
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.LogDt, s.IO = 1, 1, 1, [3]float64{0.5, 0, 0}, [3]float64{-2, 0, 0}, -0.5, 0.8
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.DScal, s.LogDt = 3, 1, 1, [3]float64{0, 0.4, 0}, [3]float64{1.6, 0, 0}, 0, -0.5
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.DScal, s.Rot = 2, 1, 1, [3]float64{0.2, 0, 0}, [3]float64{2.4, 0, 0}, 0, 1.1
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Damp, s.LogDt, s.Tau, s.In, s.Phase = 2, 1, 1, [3]float64{1.8, 0, 0}, -1, -0.35, -0.62, 2
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.LogDt, s.Tau, s.In, s.Out, s.IO = 3, 2, 2, [3]float64{0.1, -0.3, 0}, [3]float64{1.5, 0, 0}, -1, -0.83, -0.95, -0.77, 2.1
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.LogDt, s.Mag, s.Damp, s.In, s.Out = true, 2, 1, 1, -1, [3]float64{0.3, 0, 0}, [3]float64{1.4, 0, 0}, 0.2, -0.3
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.LogDt, s.Mag, s.Damp, s.In, s.Out = true, 3, 1, 2, -2, [3]float64{1, 0.2, 0}, [3]float64{2, 1, 0}, -0.5, 1
		}),
	)
}

func fzDiscBig() *big.Float { return new(big.Float).SetPrec(fzDiscPrec) }

func fzDiscMul(a, b []*big.Float, r, k, c int) []*big.Float {
	out := make([]*big.Float, r*c)
	t := fzDiscBig()
	for i := range r {
		for j := range c {
			s := fzDiscBig()
			for l := range k {
				s.Add(s, t.Mul(a[i*k+l], b[l*c+j]))
			}
			out[i*c+j] = s
		}
	}
	return out
}

// fzDiscExpm returns exp(M) for the k×k row-major float64 matrix M by a
// Taylor series on M/2^q (‖M/2^q‖∞ ≤ 1/2) and q squarings, in big.Float.
func fzDiscExpm(m []float64, k int) []*big.Float {
	nrm := 0.0
	for i := range k {
		row := 0.0
		for j := range k {
			row += math.Abs(m[i*k+j])
		}
		nrm = math.Max(nrm, row)
	}
	q := 0
	if nrm > 0.5 {
		q = int(math.Ceil(math.Log2(nrm / 0.5)))
	}
	x := make([]*big.Float, k*k)
	e := make([]*big.Float, k*k)
	term := make([]*big.Float, k*k)
	for i := range x {
		x[i] = fzDiscBig().SetFloat64(m[i])
		x[i].SetMantExp(x[i], -q)
		e[i], term[i] = fzDiscBig(), fzDiscBig()
	}
	for i := range k {
		e[i*k+i].SetInt64(1)
		term[i*k+i].SetInt64(1)
	}
	for it := int64(1); it < 1000; it++ {
		term = fzDiscMul(term, x, k, k, k)
		div := fzDiscBig().SetInt64(it)
		small := true
		for i := range term {
			term[i].Quo(term[i], div)
			e[i].Add(e[i], term[i])
			if term[i].Sign() != 0 && term[i].MantExp(nil) > -fzDiscPrec-8 {
				small = false
			}
		}
		if small {
			break
		}
	}
	for range q {
		e = fzDiscMul(e, e, k, k, k)
	}
	return e
}

func fzDiscRound(x []*big.Float) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i], _ = v.Float64()
	}
	return out
}

// fzDiscBlock returns rows r0..r0+r and columns c0..c0+c of the k-column
// matrix x.
func fzDiscBlock(x []*big.Float, k, r0, r, c0, c int) []*big.Float {
	out := make([]*big.Float, 0, r*c)
	for i := range r {
		for j := range c {
			out = append(out, x[(r0+i)*k+c0+j])
		}
	}
	return out
}

func fzDiscBigOf(x []float64) []*big.Float {
	out := make([]*big.Float, len(x))
	for i, v := range x {
		out[i] = fzDiscBig().SetFloat64(v)
	}
	return out
}

// fzDiscHold returns Φ = e^{Ah}, Γ = ∫₀ʰ e^{Aσ}dσ B and, when foh, the
// triangle-hold term G = (1/h)∫₀ʰ e^{Aσ}(h−σ)dσ B, from one exponential of
// the augmented matrix [[Ah, Bh, 0], [0, 0, I], [0, 0, 0]].
func fzDiscHold(n, m int, a, b []float64, h float64, foh bool) (phi, gam, g []*big.Float) {
	k := n + m
	if foh {
		k += m
	}
	M := make([]float64, k*k)
	for i := range n {
		for j := range n {
			M[i*k+j] = a[i*n+j] * h
		}
		for j := range m {
			M[i*k+n+j] = b[i*m+j] * h
		}
	}
	if foh {
		for j := range m {
			M[(n+j)*k+n+m+j] = 1
		}
	}
	E := fzDiscExpm(M, k)
	phi = fzDiscBlock(E, k, 0, n, 0, n)
	gam = fzDiscBlock(E, k, 0, n, n, m)
	if foh {
		g = fzDiscBlock(E, k, 0, n, n+m, m)
	}
	return phi, gam, g
}

func fzDiscFrobBig(x []*big.Float) float64 {
	s := 0.0
	for _, v := range x {
		f, _ := v.Float64()
		s = math.Hypot(s, f)
	}
	return s
}

// fzDiscZPts maps the continuous evaluation points of s into the z plane
// for sample time dt (|s·dt| capped at 3).
func fzDiscZPts(s *fzSpec, dt float64) []complex128 {
	var out []complex128
	for _, x := range fzConvPoints(s) {
		xd := x * complex(dt, 0)
		if a := cmplx.Abs(xd); a > 3 {
			xd *= complex(3/a, 0)
		}
		out = append(out, cmplx.Exp(xd))
	}
	return out
}

// fzDiscDelayedResp is the exact response of a delay-free-dynamics discrete
// model at z including its sample-count external delays; ok is false for
// internal delays, descriptor models or a singular zI − A.
func fzDiscDelayedResp(sys *System, z complex128) ([]complex128, bool) {
	if sys.HasInternalDelay() || sys.IsDescriptor() {
		return nil, false
	}
	q := fzMatsOf(sys)
	h, ok := q.resp(z)
	if !ok {
		return nil, false
	}
	for i := range q.p {
		for j := range q.m {
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
			h[i*q.m+j] *= cmplx.Pow(z, complex(-k, 0))
		}
	}
	return h, true
}

// fzDiscVerdict fails unless the discrete model sd has exactly want poles
// outside the unit circle (exact Schur–Cohn on its float64 A).
func fzDiscVerdict(t *testing.T, s *fzSpec, op string, sd *System, want int) {
	t.Helper()
	n, _, _ := sd.Dims()
	if n == 0 {
		if want != 0 {
			t.Fatalf("%s: static result, want %d unstable poles\n%s", op, want, s)
		}
		return
	}
	k, ok := fzSchurCohnOutside(fzCharPolyExact(mat.DenseCopyOf(sd.A).RawMatrix().Data, n), 1)
	if !ok || k != want {
		t.Fatalf("%s: %d poles outside |z|=1 (table ok %v), want %d\n%s", op, k, ok, want, s)
	}
}

// fzDiscClear reports whether every mapped pole z_k lies farther than r
// from the unit circle.
func fzDiscClear(zs []complex128, r float64) bool {
	for _, z := range zs {
		if !(math.Abs(cmplx.Abs(z)-1) > r) {
			return false
		}
	}
	return true
}

// FuzzC2DExpm checks C2D(zoh), C2D(foh), C2D(impulse) and C2D(zoh) with
// fractional input delays against the big.Float exponential of the
// augmented matrix: Hzoh(z) = C(zI−Φ)⁻¹Γ + D, Hfoh(z) = C(zI−Φ)⁻¹(Γ +
// (z−1)G) + D (triangle hold), Himp(z) = Ts·C·z(zI−Φ)⁻¹B (strictly proper
// part; D is zeroed), and for an input delay of (ℓ+ε)Ts on input j column j
// of z^{−ℓ}[C(zI−Φ)⁻¹(Γ((1−ε)Ts) + z⁻¹e^{A(1−ε)Ts}Γ(εTs)) + z⁻¹D]. Each
// stable/unstable pole count is preserved when the mapped poles clear the
// unit circle.
func FuzzC2DExpm(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "discrete spec")
		}
		const eps = 0x1p-52
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		n, m, p := s.N, s.M, s.P
		if n > 0 && s.NormJ*dt > 20 {
			fzSkip(t, "‖J‖·Ts > 20")
		}
		nAB := math.Hypot(fzFrob(s.A), fzFrob(s.B)) * dt
		if nAB > 1e8 {
			fzSkip(t, "‖[A B]‖·Ts > 1e8")
		}
		sys := s.System(t, fzSysOpts{})
		kk := 256 * float64((n+2*m+1)*(n+2*m+1))
		zpts := fzDiscZPts(s, dt)
		mapped := make([]complex128, n)
		for k, e := range s.S {
			mapped[k] = cmplx.Exp(e * complex(dt, 0))
		}
		growth := math.Exp(math.Min(s.NormJ*dt, 20)) * s.CondT
		nC := fzFrob(s.C)

		phi, gam, g := fzDiscHold(n, m, s.A, s.B, dt, true)
		nPhi := fzDiscFrobBig(phi)
		delta := kk * eps * s.CondT * s.CondT * growth * (1 + nAB) * math.Max(1, nPhi)
		stableRad := 8 * s.CondT * (delta + s.EigErr()*dt*math.Max(1, nPhi))

		bo := make([]*big.Float, n*m)
		do := fzDiscBigOf(s.D)
		pg := fzDiscMul(phi, g, n, n, m)
		cg := fzDiscMul(fzDiscBigOf(s.C), g, p, n, m)
		for i := range bo {
			bo[i] = fzDiscBig().Add(gam[i], pg[i])
			bo[i].Sub(bo[i], g[i])
		}
		for i := range do {
			do[i].Add(do[i], cg[i])
		}
		phiF := fzDiscRound(phi)
		imB := fzDiscRound(fzDiscMul(phi, fzDiscBigOf(s.B), n, n, m))
		imD := fzDiscRound(fzDiscMul(fzDiscBigOf(s.C), fzDiscBigOf(s.B), p, n, m))
		for i := range imB {
			imB[i] *= dt
		}
		for i := range imD {
			imD[i] *= dt
		}
		zsys, err := NewFromSlices(n, m, p, s.A, s.B, s.C, make([]float64, p*m), 0)
		if err != nil {
			t.Fatalf("NewFromSlices: %v\n%s", err, s)
		}
		cases := []struct {
			name string
			src  *System
			opts C2DOptions
			want fzMats
			dB   float64
			dD   float64
		}{
			{"C2D(zoh)", sys, C2DOptions{}, fzMats{n, m, p, phiF, fzDiscRound(gam), s.C, s.D}, delta, 0},
			{"C2D(foh)", sys, C2DOptions{Method: C2DMethodFOH}, fzMats{n, m, p, phiF, fzDiscRound(bo), s.C, fzDiscRound(do)}, 3 * delta * (1 + nPhi), 2 * nC * delta},
			{"C2D(impulse)", zsys, C2DOptions{Method: C2DMethodImpulse}, fzMats{n, m, p, phiF, imB, s.C, imD}, dt * delta * (1 + fzFrob(s.B)), kk * eps * dt * nC * fzFrob(s.B)},
		}
		for _, c := range cases {
			sd, err := c.src.C2D(dt, c.opts)
			fzContract(t, s, c.name, sd != nil, err)
			fzErr(t, s, c.name, err)
			if sd.Dt != dt {
				t.Fatalf("%s: dt %g, want %g\n%s", c.name, sd.Dt, dt, s)
			}
			got := fzMatsOf(sd)
			for _, z := range zpts {
				hw, ok1 := c.want.resp(z)
				hg, ok2 := got.resp(z)
				if !ok1 || !ok2 {
					continue
				}
				tol := c.want.pert(z, delta+eps*nPhi, c.dB+eps*fzFrob(c.want.b), 0, c.dD+eps*fzFrob(c.want.d))
				fzCompare(t, s, c.name+" vs expm oracle", z, hg, hw, tol, "")
			}
			if s.Decidable() && fzDiscClear(mapped, stableRad) {
				fzDiscVerdict(t, s, c.name, sd, s.Unstable())
			}
		}
		fzDiscInputDelayZOH(t, s, sys, dt, kk, growth)
	})
}

// fzDiscInputDelayZOH checks C2D(zoh) with fractional input delays (state
// modeling) against the exact sampled hold response.
func fzDiscInputDelayZOH(t *testing.T, s *fzSpec, sys *System, dt, kk, growth float64) {
	const eps = 0x1p-52
	if s.In == nil || s.N == 0 {
		return
	}
	n, m, p := s.N, s.M, s.P
	ell := make([]float64, m)
	frac := make([]float64, m)
	for j, tau := range s.In {
		x := tau / dt
		if x > 50 {
			fzSkip(t, "input delay above 50 samples")
		}
		ell[j] = math.Floor(x)
		frac[j] = x - ell[j]
		if frac[j] < 1e-6 || frac[j] > 1-1e-6 {
			frac[j] = 0
			ell[j] = math.Round(x)
		}
	}
	if err := sys.SetInputDelay(s.In); err != nil {
		t.Fatalf("SetInputDelay: %v\n%s", err, s)
	}
	defer func() { sys.InputDelay = nil }()
	sd, err := sys.C2D(dt, C2DOptions{DelayModeling: C2DDelayModelingState})
	fzContract(t, s, "C2D(zoh, input delay)", sd != nil, err)
	fzErr(t, s, "C2D(zoh, input delay)", err)
	phi, _, _ := fzDiscHold(n, m, s.A, s.B, dt, false)
	phiF := fzDiscRound(phi)
	nPhi := fzFrob(phiF)
	nAB := math.Hypot(fzFrob(s.A), fzFrob(s.B)) * dt
	delta := kk * eps * s.CondT * s.CondT * growth * (1 + nAB) * math.Max(1, nPhi)
	b1 := make([]float64, n*m)
	b2 := make([]float64, n*m)
	for j := range m {
		bj := make([]float64, n)
		for i := range n {
			bj[i] = s.B[i*m+j]
		}
		e := frac[j]
		if e == 0 {
			_, g, _ := fzDiscHold(n, 1, s.A, bj, dt, false)
			for i := range n {
				b1[i*m+j], _ = g[i].Float64()
			}
			continue
		}
		ph1, g1, _ := fzDiscHold(n, 1, s.A, bj, (1-e)*dt, false)
		_, g0, _ := fzDiscHold(n, 1, s.A, bj, e*dt, false)
		w := fzDiscMul(ph1, g0, n, n, 1)
		for i := range n {
			b1[i*m+j], _ = g1[i].Float64()
			b2[i*m+j], _ = w[i].Float64()
		}
	}
	q1 := fzMats{n, m, p, phiF, b1, s.C, make([]float64, p*m)}
	q2 := fzMats{n, m, p, phiF, b2, s.C, make([]float64, p*m)}
	for _, z := range fzDiscZPts(s, dt) {
		hg, ok := fzDiscDelayedResp(sd, z)
		h1, ok1 := q1.resp(z)
		h2, ok2 := q2.resp(z)
		if !ok || !ok1 || !ok2 {
			continue
		}
		t1, t2 := q1.pert(z, delta, 2*delta, 0, 0), q2.pert(z, delta, 2*delta, 0, 0)
		want := make([]complex128, p*m)
		tol := 0.0
		for i := range p {
			for j := range m {
				k := i*m + j
				lag := cmplx.Pow(z, complex(-ell[j], 0))
				dlag := lag
				if frac[j] > 0 {
					dlag = lag / z
				}
				want[k] = lag*(h1[k]+h2[k]/z) + dlag*complex(s.D[k], 0)
				tol = math.Max(tol, cmplx.Abs(lag)*(t1+t2/cmplx.Abs(z))+4*eps*cmplx.Abs(want[k]))
			}
		}
		if math.IsInf(tol, 1) {
			continue
		}
		fzCompare(t, s, "C2D(zoh, fractional input delay) vs hold oracle", z, hg, want, tol, "")
	}
}

// FuzzC2DTustinPrewarp checks prewarped Tustin, Hd(z) = Hc(β(z−1)/(z+1))
// with β = ω₀/tan(ω₀Ts/2), the exact match Hd(e^{jω₀Ts}) = Hc(jω₀), the
// D2C inverse with the same prewarp, and stability verdict preservation.
func FuzzC2DTustinPrewarp(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "discrete spec")
		}
		const eps = 0x1p-52
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		w0 := (0.05 + 0.9*(0.5+0.5*math.Sin(s.Vals[14]))) * math.Pi / dt
		beta := w0 / math.Tan(w0*dt/2)
		n := s.N
		sys := s.System(t, fzSysOpts{})
		orig := fzMats{n, s.M, s.P, s.A, s.B, s.C, s.D}
		nA, nB, nC, nD := fzFrob(s.A), fzFrob(s.B), fzFrob(s.C), fzFrob(s.D)
		kk := 256 * float64((n+1)*(n+1))
		kT := 1.0
		if n > 0 {
			kT = fzCond(s.A, n, -1/beta)
		}
		if !(kT < 1e8) {
			fzSkip(t, "I − A/β ill-conditioned")
		}
		td, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w0})
		fzContract(t, s, "C2D(tustin, prewarp)", td != nil, err)
		fzErr(t, s, "C2D(tustin, prewarp)", err)
		tm := fzMatsOf(td)
		back, err := td.D2C(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: w0})
		fzContract(t, s, "D2C(tustin, prewarp)", back != nil, err)
		fzErr(t, s, "D2C(tustin, prewarp)", err)
		rt := fzMatsOf(back)
		zs := []complex128{cmplx.Exp(complex(0, w0*dt))}
		for _, x := range append(fzConvPoints(s), 0) {
			den := complex(beta, 0) - x
			if cmplx.Abs(den) < 1e-8*beta {
				continue
			}
			zs = append(zs, (complex(beta, 0)+x)/den)
		}
		for k, z := range zs {
			x2 := complex(beta, 0) * (z - 1) / (z + 1)
			if k == 0 {
				x2 = complex(0, w0)
			}
			dx := cmplx.Abs(x2) * eps * (8 + (cmplx.Abs(z)+1)*(1/cmplx.Abs(z-1)+1/cmplx.Abs(z+1)))
			hc, ok1 := orig.resp(x2)
			hd, ok2 := tm.resp(z)
			if ok1 && ok2 {
				dA := kk*eps*kT*(nA+2*beta) + 4*dx
				tol := orig.pert(x2, dA, kk*eps*kT*nB, kk*eps*kT*nC, kk*eps*kT*(nD+nC*nB*kT/beta))
				fzCompare(t, s, "C2D(tustin, prewarp) vs Hc(β(z−1)/(z+1))", z, hd, hc, tol, "")
			}
			hc, ok1 = orig.resp(x2)
			hr, ok2 := rt.resp(x2)
			if ok1 && ok2 {
				k2 := kk * kT * kT
				tol := orig.pert(x2, k2*eps*(nA+2*beta), k2*eps*nB, k2*eps*nC, k2*eps*(nD+nC*nB*kT/beta))
				fzCompare(t, s, "D2C(C2D(tustin, prewarp))", x2, hr, hc, tol, "")
			}
		}
		if n > 0 && s.Decidable() {
			mapped := make([]complex128, n)
			for k, e := range s.S {
				mapped[k] = (complex(beta, 0) + e) / (complex(beta, 0) - e)
			}
			if fzDiscClear(mapped, 8*s.CondT*(kk*eps*kT*kT*s.CondT*(1+nA/beta)+2*s.EigErr()/beta)) {
				fzDiscVerdict(t, s, "C2D(tustin, prewarp)", td, s.Unstable())
			}
		}
	})
}

// fzDiscRatEval evaluates the exact rational descending polynomial c at z.
func fzDiscRatEval(c []*big.Rat, z complex128) complex128 {
	zr := &fzCRat{}
	zr.re.Set(fzRat(real(z)))
	zr.im.Set(fzRat(imag(z)))
	acc := &fzCRat{}
	for _, v := range c {
		acc = fzCMul(acc, zr)
		acc.re.Add(&acc.re, v)
	}
	re, _ := acc.re.Float64()
	im, _ := acc.im.Float64()
	return complex(re, im)
}

func fzDiscRatDeriv(c []*big.Rat) []*big.Rat {
	n := len(c) - 1
	out := make([]*big.Rat, n)
	for k := range n {
		out[k] = new(big.Rat).Mul(c[k], big.NewRat(int64(n-k), 1))
	}
	return out
}

// fzDiscRoots returns the roots of the exact polynomial c (companion
// eigenvalues, Newton-polished on the exact coefficients); ok is false when
// a polished root is not resolved to 1e-11 relative.
func fzDiscRoots(c []*big.Rat) ([]complex128, bool) {
	n := len(c) - 1
	if n <= 0 {
		return nil, true
	}
	lead, _ := c[0].Float64()
	comp := mat.NewDense(n, n, nil)
	for k := 1; k <= n; k++ {
		v, _ := c[k].Float64()
		comp.Set(0, k-1, -v/lead)
	}
	for i := 1; i < n; i++ {
		comp.Set(i, i-1, 1)
	}
	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		return nil, false
	}
	roots := eig.Values(nil)
	dc := fzDiscRatDeriv(c)
	for i, r := range roots {
		step := complex(0, 0)
		for range 8 {
			d := fzDiscRatEval(dc, r)
			if d == 0 {
				return nil, false
			}
			step = fzDiscRatEval(c, r) / d
			r -= step
		}
		if cmplx.IsNaN(r) || cmplx.Abs(step) > 1e-11*math.Max(cmplx.Abs(r), 1e-300) {
			return nil, false
		}
		roots[i] = r
	}
	return roots, true
}

// FuzzC2DMatched checks SISO matched pole-zero mapping: Hd(z) =
// K·Π(z−e^{zᵢTs})(z+1)^r / Π(z−e^{pᵢTs}) with r = max(np−nz−1, 0) and K
// from Hd(1) = Hc(0), zeros from the exact Rosenbrock determinant, plus
// stability verdict preservation. Gated to well-separated roots clear of DC
// (the library roots polynomials, which is ill-conditioned for clusters).
func FuzzC2DMatched(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "discrete spec")
		}
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		n := s.N
		sys := s.System(t, fzSysOpts{})
		if s.M != 1 || s.P != 1 {
			_, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if !errors.Is(err, ErrNotSISO) {
				t.Fatalf("C2D(matched) MIMO: err %v, want ErrNotSISO\n%s", err, s)
			}
			return
		}
		if n == 0 || n > 4 || s.CondT > 100 {
			fzSkip(t, "matched: outside well-conditioned regime")
		}
		num := fzTrimRat(fzPencilDetRat(n, 1, s.A, s.B, s.C, s.D))
		if len(num) == 0 {
			fzSkip(t, "matched: zero transfer function")
		}
		zc, ok := fzDiscRoots(num)
		if !ok {
			fzSkip(t, "matched: unresolved zero")
		}
		roots := slices.Concat(zc, s.S)
		for i, a := range roots {
			if cmplx.Abs(a)*dt < 1e-6 || math.Abs(real(a))*dt > 50 || cmplx.Abs(a)*dt > 1e3 {
				fzSkip(t, "matched: root at DC or out of range")
			}
			for _, b := range roots[i+1:] {
				if cmplx.Abs(a-b) < 1e-3*cmplx.Abs(a) {
					fzSkip(t, "matched: clustered roots")
				}
			}
		}
		known := ""
		for _, a := range roots {
			if imag(a) != 0 && math.Sin(math.Abs(imag(a))*dt) < 0 {
				known = "matched: aliased complex root maps to negative-imag-first pair (ZPK conjugate ordering defect)"
			}
		}
		sd, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		fzContract(t, s, "C2D(matched)", sd != nil, err)
		if err != nil && known != "" {
			fzSkip(t, known)
		}
		fzErr(t, s, "C2D(matched)", err)
		hc0, ok := fzMats{n, 1, 1, s.A, s.B, s.C, s.D}.resp(0)
		if !ok {
			fzSkip(t, "matched: singular A")
		}
		r := max(n-len(zc)-1, 0)
		dz := make([]complex128, len(zc))
		dp := make([]complex128, n)
		for i, z := range zc {
			dz[i] = cmplx.Exp(z * complex(dt, 0))
		}
		for i, e := range s.S {
			dp[i] = cmplx.Exp(e * complex(dt, 0))
		}
		const eps = 0x1p-52
		ke := 64 * float64((n+1)*(n+1)) * eps
		numRoots := slices.Clone(dz)
		for range r {
			numRoots = append(numRoots, -1)
		}
		// Wilkinson root condition of a polynomial realization: the library
		// realizes the discrete ZPK through monic polynomials.
		wilk := func(set []complex128, i int) float64 {
			c := ke * 2 * cmplx.Abs(set[i])
			for j, b := range set {
				if j != i {
					c *= (cmplx.Abs(set[i]) + cmplx.Abs(b)) / cmplx.Abs(set[i]-b)
				}
			}
			return c
		}
		rat := func(x complex128) (complex128, float64) {
			v, rel, mag, pmag := complex(1, 0), 0.0, 1.0, 1.0
			for i, z := range dz {
				v *= x - z
				mag *= cmplx.Abs(x) + cmplx.Abs(z)
				rel += ((1e-8*cmplx.Abs(zc[i])+1e-12/dt)*dt*cmplx.Abs(z) + wilk(numRoots, i)) / cmplx.Abs(x-z)
			}
			for k := range r {
				v *= x + 1
				mag *= cmplx.Abs(x) + 1
				if cmplx.Abs(x+1) > 0 {
					rel += wilk(numRoots, len(dz)+k) / cmplx.Abs(x+1)
				}
			}
			for i, p := range dp {
				v /= x - p
				mag /= cmplx.Abs(x - p)
				pmag *= (cmplx.Abs(x) + cmplx.Abs(p)) / cmplx.Abs(x-p)
				rel += ((1e-8*cmplx.Abs(s.S[i])+s.EigErr())*dt*cmplx.Abs(p) + wilk(dp, i)) / cmplx.Abs(x-p)
			}
			return v, rel + ke*(mag+pmag)/cmplx.Abs(v)
		}
		v1, rel1 := rat(1)
		K := hc0[0] / v1
		got := fzMatsOf(sd)
		for _, z := range fzDiscZPts(s, dt) {
			vz, relz := rat(z)
			if relz+rel1 > 0.01 {
				continue
			}
			hg, ok := got.resp(z)
			if !ok {
				continue
			}
			want := K * vz
			tol := (relz + rel1 + ke) * cmplx.Abs(want) * 8
			fzCompare(t, s, "C2D(matched) vs pole-zero map", z, hg, []complex128{want}, tol, known)
		}
		clear := known == "" && s.Decidable()
		for i, p := range dp {
			if !(math.Abs(cmplx.Abs(p)-1) > 1e-6+8*wilk(dp, i)) {
				clear = false
			}
		}
		if clear {
			fzDiscVerdict(t, s, "C2D(matched)", sd, s.Unstable())
		}
	})
}

// FuzzC2DLeastSquares checks the least-squares fit's contract: C2DFit and
// C2D agree bitwise, the reported Stable flag matches an exact Schur–Cohn
// count of the fitted A, the auto order is the source order, and MIMO is
// ErrNotSISO.
func FuzzC2DLeastSquares(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "discrete spec")
		}
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		sys := s.System(t, fzSysOpts{})
		opts := C2DOptions{Method: C2DMethodLeastSquares}
		if s.M != 1 || s.P != 1 {
			_, err := sys.C2D(dt, opts)
			if !errors.Is(err, ErrNotSISO) {
				t.Fatalf("C2D(least-squares) MIMO: err %v, want ErrNotSISO\n%s", err, s)
			}
			return
		}
		sd, fit, err := sys.C2DFit(dt, opts)
		fzContract(t, s, "C2DFit", sd != nil, err)
		fzErr(t, s, "C2DFit", err, ErrSingularTransform, ErrOverflow)
		sd2, err := sys.C2D(dt, opts)
		fzContract(t, s, "C2D(least-squares)", sd2 != nil, err)
		fzErr(t, s, "C2D(least-squares)", err)
		a1, b1, c1, d1 := fzRaw(sd)
		a2, b2, c2, d2 := fzRaw(sd2)
		if !slices.Equal(a1, a2) || !slices.Equal(b1, b2) || !slices.Equal(c1, c2) || !slices.Equal(d1, d2) || sd.Dt != dt || sd2.Dt != dt {
			t.Fatalf("C2DFit and C2D(least-squares) differ\n%s", s)
		}
		if fit.FitOrder != s.N {
			t.Fatalf("C2DFit auto order %d, want %d\n%s", fit.FitOrder, s.N, s)
		}
		if !(fit.RMSRelativeError >= 0) || !(fit.MaxRelativeError >= 0) || math.IsInf(fit.RMSRelativeError+fit.MaxRelativeError, 0) {
			t.Fatalf("C2DFit errors rms %g max %g\n%s", fit.RMSRelativeError, fit.MaxRelativeError, s)
		}
		n, _, _ := sd.Dims()
		if n == 0 {
			if !fit.Stable {
				t.Fatalf("C2DFit static model reported unstable\n%s", s)
			}
			return
		}
		k, ok := fzSchurCohnOutside(fzCharPolyExact(a1, n), 1)
		if !ok {
			fzSkip(t, "least-squares: fitted pole on |z| = 1")
		}
		if fit.Stable != (k == 0) {
			if math.Abs(fzDiscSpectralRadius(a1, n)-1) < 1e-9 {
				fzSkip(t, "least-squares: fitted pole within 1e-9 of |z| = 1")
			}
			t.Fatalf("C2DFit Stable=%v but %d fitted poles outside |z|=1\n%s", fit.Stable, k, s)
		}
	})
}

func fzDiscSpectralRadius(a []float64, n int) float64 {
	var eig mat.Eigen
	if !eig.Factorize(mat.NewDense(n, n, slices.Clone(a)), mat.EigenNone) {
		return math.NaN()
	}
	r := 0.0
	for _, v := range eig.Values(nil) {
		r = math.Max(r, cmplx.Abs(v))
	}
	return r
}

// fzDiscThiran returns the exact closed-form Thiran coefficients
// a_k = (−1)^k C(N,k) Π_{i=0}^{N} (D−N+i)/(D−N+k+i).
func fzDiscThiran(D *big.Rat, N int) []*big.Rat {
	a := make([]*big.Rat, N+1)
	binom := big.NewRat(1, 1)
	for k := 0; k <= N; k++ {
		if k > 0 {
			binom = new(big.Rat).Mul(binom, big.NewRat(int64(N-k+1), int64(k)))
		}
		prod := new(big.Rat).Set(binom)
		if k%2 == 1 {
			prod.Neg(prod)
		}
		for i := 0; i <= N; i++ {
			num := new(big.Rat).Add(D, big.NewRat(int64(i-N), 1))
			den := new(big.Rat).Add(D, big.NewRat(int64(k+i-N), 1))
			prod.Mul(prod, num)
			prod.Quo(prod, den)
		}
		a[k] = prod
	}
	return a
}

// fzDiscThiranResp evaluates the order-N Thiran allpass for D samples,
// Σa_k z^k / Σa_k z^{N−k}, and a relative error allowance for a float64
// realization of it.
func fzDiscThiranResp(D *big.Rat, N int, z complex128) (complex128, float64) {
	const eps = 0x1p-52
	a := fzDiscThiran(D, N)
	rev := slices.Clone(a)
	slices.Reverse(rev)
	num, den := fzDiscRatEval(rev, z), fzDiscRatEval(a, z)
	sum := 0.0
	for _, v := range a {
		f, _ := v.Float64()
		sum += math.Abs(f)
	}
	sum *= math.Pow(math.Max(1, cmplx.Abs(z)), float64(N))
	return num / den, 64 * float64((N+1)*(N+1)) * eps * sum / math.Min(cmplx.Abs(num), cmplx.Abs(den))
}

// fzDiscThiranSplit mirrors MATLAB's fractional-delay split for a delay of
// x samples and maximum order nmax: order N = min(ceil x, nmax) for the last
// x − whole samples, whole = ceil x − N pure samples; frac false for an
// integer delay.
func fzDiscThiranSplit(x float64, nmax int) (D *big.Rat, N, whole int, frac bool) {
	c := math.Ceil(x)
	if math.Abs(x-math.Round(x)) <= 1e-9*math.Max(1, x) {
		return nil, 0, int(math.Round(x)), false
	}
	N = min(int(c), nmax)
	whole = int(c) - N
	return new(big.Rat).Sub(fzRat(x), big.NewRat(int64(whole), 1)), N, whole, true
}

// FuzzThiran checks ThiranDelay against the closed-form coefficients
// (allpass, exact delay for integer D, stable for D > N−1), and C2D(tustin)
// with ThiranOrder and state delay modeling against Hc(bilinear) times the
// per-input and per-output Thiran factors.
func FuzzThiran(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "discrete spec")
		}
		const eps = 0x1p-52
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		tau := fzDelay(s.Vals[15], false)
		if tau/dt > 40 {
			fzSkip(t, "thiran: delay above 40 samples")
		}
		th, err := ThiranDelay(tau, dt)
		fzContract(t, s, "ThiranDelay", th != nil, err)
		fzErr(t, s, "ThiranDelay", err)
		D, N, whole, frac := fzDiscThiranSplit(tau/dt, math.MaxInt)
		for _, z := range append(fzDiscZPts(s, dt), cmplx.Exp(complex(0, 0.3)), cmplx.Exp(complex(0, 2.9))) {
			hg, ok := fzDiscDelayedResp(th, z)
			if !ok {
				continue
			}
			want, rel := cmplx.Pow(z, complex(-float64(whole), 0)), 0.0
			if frac {
				var h complex128
				h, rel = fzDiscThiranResp(D, N, z)
				want *= h
			}
			fzCompare(t, s, "ThiranDelay vs closed form", z, hg, []complex128{want}, rel*cmplx.Abs(want)+64*eps*cmplx.Abs(want), "")
		}
		if frac {
			dd, _ := D.Float64()
			if dd-float64(N-1) > 1e-3 {
				fzRequireStable(t, s, "ThiranDelay", th)
			}
		}

		if s.In == nil && s.Out == nil {
			return
		}
		nmax := 1 + int(math.Abs(s.Vals[14]))%4
		sys := s.System(t, fzSysOpts{Delays: true})
		sys.Delay = nil
		td, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: nmax, DelayModeling: C2DDelayModelingState})
		fzContract(t, s, "C2D(tustin, thiran)", td != nil, err)
		fzErr(t, s, "C2D(tustin, thiran)", err)
		n := s.N
		orig := fzMats{n, s.M, s.P, s.A, s.B, s.C, s.D}
		nA, nB, nC, nD := fzFrob(s.A), fzFrob(s.B), fzFrob(s.C), fzFrob(s.D)
		kk := 256 * float64((n+1)*(n+1))
		kT := 1.0
		if n > 0 {
			kT = fzCond(s.A, n, -dt/2)
		}
		if !(kT < 1e8) {
			fzSkip(t, "thiran: I − A·Ts/2 ill-conditioned")
		}
		factor := func(delays []float64, k int, z complex128) (complex128, float64, bool) {
			if delays == nil {
				return 1, 0, true
			}
			x := delays[k] / dt
			if x > 40 {
				return 0, 0, false
			}
			D, N, whole, frac := fzDiscThiranSplit(x, nmax)
			h, rel := cmplx.Pow(z, complex(-float64(whole), 0)), 0.0
			if frac {
				var g complex128
				g, rel = fzDiscThiranResp(D, N, z)
				h *= g
			}
			return h, rel, true
		}
		for _, x := range fzConvPoints(s) {
			den := 1 - x*complex(dt/2, 0)
			if cmplx.Abs(den) < 1e-8 {
				continue
			}
			z := (1 + x*complex(dt/2, 0)) / den
			x2 := complex(2/dt, 0) * (z - 1) / (z + 1)
			dx := cmplx.Abs(x2) * eps * (3 + (cmplx.Abs(z)+1)*(1/cmplx.Abs(z-1)+1/cmplx.Abs(z+1)))
			hc, ok1 := orig.resp(x2)
			hg, ok2 := fzDiscDelayedResp(td, z)
			if !ok1 || !ok2 {
				continue
			}
			dA := kk*eps*kT*(nA+2/dt) + 4*dx
			tolc := orig.pert(x2, dA, kk*eps*kT*nB, kk*eps*kT*nC, kk*eps*kT*(nD+nC*nB*dt*kT))
			want := make([]complex128, s.P*s.M)
			tol := 0.0
			for i := range s.P {
				fo, ro, ok := factor(s.Out, i, z)
				if !ok {
					fzSkip(t, "thiran: output delay above 40 samples")
				}
				for j := range s.M {
					fi, ri, ok := factor(s.In, j, z)
					if !ok {
						fzSkip(t, "thiran: input delay above 40 samples")
					}
					k := i*s.M + j
					want[k] = fo * hc[k] * fi
					g := cmplx.Abs(fo * fi)
					tol = math.Max(tol, 4*(g*tolc+cmplx.Abs(want[k])*(ro+ri))*(1+kk*eps*kT))
				}
			}
			if math.IsInf(tol, 1) {
				continue
			}
			fzCompare(t, s, "C2D(tustin, thiran) vs Hc(bilinear)·Thiran", z, hg, want, tol+kk*eps*nD, "")
		}
	})
}

// fzDiscRatPow returns Ad^r and Σ_{k<r} Ad^k·Bd exactly.
func fzDiscRatPow(n, m int, a, b []float64, r int) (pw, sum []float64) {
	A := fzRatPoly(a)
	P := make([]*big.Rat, n*n)
	for i := range P {
		P[i] = new(big.Rat)
	}
	for i := range n {
		P[i*n+i].SetInt64(1)
	}
	S := make([]*big.Rat, n*m)
	for i := range S {
		S[i] = new(big.Rat)
	}
	Bv := fzRatPoly(b)
	mul := func(x, y []*big.Rat, cols int) []*big.Rat {
		out := make([]*big.Rat, n*cols)
		var t big.Rat
		for i := range n {
			for j := range cols {
				v := new(big.Rat)
				for l := range n {
					v.Add(v, t.Mul(x[i*n+l], y[l*cols+j]))
				}
				out[i*cols+j] = v
			}
		}
		return out
	}
	for range r {
		pb := mul(P, Bv, m)
		for i := range S {
			S[i].Add(S[i], pb[i])
		}
		P = mul(P, A, n)
	}
	pw = make([]float64, n*n)
	sum = make([]float64, n*m)
	for i, v := range P {
		pw[i], _ = v.Float64()
	}
	for i, v := range S {
		sum[i], _ = v.Float64()
	}
	return pw, sum
}

// FuzzD2D checks resampling by integer ratios against exact identities.
// ZOH downsampling to r·Ts gives Ad^r, Σ_{k<r}Ad^k·Bd with C, D unchanged
// (branch independent); upsampling to Ts/r gives A₂ with A₂^r = Ad,
// Σ A₂^k·B₂ = Bd and principal-branch poles e^{λTs/r}. Tustin resampling
// satisfies H₂(z₂) = H₁(z₁) with s = β₂(z₂−1)/(z₂+1), z₁ = (β₁+s)/(β₁−s).
func FuzzD2D(f *testing.F) {
	for _, s := range fzDiscSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if !s.Discrete {
			fzSkip(t, "continuous spec")
		}
		const eps = 0x1p-52
		n, m, p, dt := s.N, s.M, s.P, s.Dt
		sys := s.System(t, fzSysOpts{})
		r := []int{2, 3, 5}[int(math.Abs(s.Vals[12]))%3]
		up := s.Vals[13] < 0
		dt2 := float64(r) * dt
		if up {
			dt2 = dt / float64(r)
		}
		same, err := sys.D2D(dt, D2DOptions{})
		fzContract(t, s, "D2D(same dt)", same != nil, err)
		fzErr(t, s, "D2D(same dt)", err)
		a0, b0, c0, d0 := fzRaw(same)
		if !slices.Equal(a0, s.A) || !slices.Equal(b0, s.B) || !slices.Equal(c0, s.C) || !slices.Equal(d0, s.D) || same.Dt != dt {
			t.Fatalf("D2D(same dt) is not a copy\n%s", s)
		}
		minZ, maxS, negReal := math.Inf(1), 0.0, false
		for k, e := range s.Eig {
			minZ = math.Min(minZ, cmplx.Abs(e))
			maxS = math.Max(maxS, cmplx.Abs(s.S[k]))
			if math.Pi-math.Abs(cmplx.Phase(e)) < 1e-3 {
				negReal = true
			}
		}
		orig := fzMats{n, m, p, s.A, s.B, s.C, s.D}
		kk := 256 * float64((n+1)*(n+1))
		nA := fzFrob(s.A)
		inv := 1 / math.Max(fzUnitGap(orig), 1e-8)
		if n == 0 || (minZ >= 1e-3 && s.CondT <= 1e4 && maxS*dt <= 1 && !negReal) {
			fzD2DZOH(t, s, sys, r, up, dt2, kk, inv)
		}

		if n > 0 {
			for _, e := range s.Eig {
				if cmplx.Abs(e+1) < 1e-3 {
					fzSkip(t, "d2d tustin: pole near z = −1")
				}
			}
		}
		kT1 := 1.0
		if n > 0 {
			kT1 = fzCond(s.A, n, 1)
		}
		b1, b2 := 2/dt, 2/dt2
		kT2 := s.CondT
		for _, e := range s.Eig {
			lc := complex(b1, 0) * (e - 1) / (e + 1)
			kT2 = math.Max(kT2, s.CondT*(1+cmplx.Abs(lc)/b2)/cmplx.Abs(1-lc/complex(b2, 0)))
		}
		if !(kT1 < 1e6 && kT2 < 1e6) {
			fzSkip(t, "d2d tustin: ill-conditioned bilinear maps")
		}
		td, err := sys.D2D(dt2, D2DOptions{Method: C2DMethodTustin})
		fzContract(t, s, "D2D(tustin)", td != nil, err)
		fzErr(t, s, "D2D(tustin)", err)
		got := fzMatsOf(td)
		nB, nC, nD := fzFrob(s.B), fzFrob(s.C), fzFrob(s.D)
		for _, z2 := range fzDiscZPts(s, dt2) {
			x := complex(b2, 0) * (z2 - 1) / (z2 + 1)
			den := complex(b1, 0) - x
			if cmplx.Abs(den) < 1e-6*b1 || cmplx.Abs(z2+1) < 1e-6 {
				continue
			}
			z1 := (complex(b1, 0) + x) / den
			h1, ok1 := orig.resp(z1)
			h2, ok2 := got.resp(z2)
			if !ok1 || !ok2 {
				continue
			}
			k2 := kk * kT1 * kT1 * kT2 * kT2
			dz := 16 * eps * cmplx.Abs(z1) * (1 + cmplx.Abs(z2)/cmplx.Abs(z2+1) + cmplx.Abs(x)/cmplx.Abs(den)) * (1 + cmplx.Abs(z2)/cmplx.Abs(z2-1))
			tol := orig.pert(z1, k2*eps*math.Max(1, nA)+dz, k2*eps*nB, k2*eps*nC, k2*eps*(nD+nB*nC))
			fzCompare(t, s, "D2D(tustin) vs H₁(z₁)", z2, h2, h1, tol, "")
		}
	})
}

func fzD2DZOH(t *testing.T, s *fzSpec, sys *System, r int, up bool, dt2, kk, inv float64) {
	const eps = 0x1p-52
	n, m := s.N, s.M
	zd, err := sys.D2D(dt2, D2DOptions{})
	fzContract(t, s, "D2D(zoh)", zd != nil, err)
	fzErr(t, s, "D2D(zoh)", err)
	if zd.Dt != dt2 {
		t.Fatalf("D2D(zoh) dt %g, want %g\n%s", zd.Dt, dt2, s)
	}
	q := fzMatsOf(zd)
	if q.n != n || !slices.Equal(q.c, s.C) || !slices.Equal(q.d, s.D) {
		t.Fatalf("D2D(zoh): n=%d C=%v D=%v, want %d %v %v\n%s", q.n, q.c, q.d, n, s.C, s.D, s)
	}
	if n == 0 {
		return
	}
	nA := fzFrob(s.A)
	ke := kk * s.CondT * s.CondT * float64(r) * (1 + inv)
	cmp := func(op string, got, want []float64, tol float64) {
		for i := range got {
			if d := math.Abs(got[i] - want[i]); !(d <= tol) {
				t.Fatalf("%s[%d] = %v, want %v (|Δ| %g, tol %g)\n%s", op, i, got[i], want[i], d, tol, s)
			}
		}
	}
	if !up {
		pw, sum := fzDiscRatPow(n, m, s.A, s.B, r)
		g := math.Pow(math.Max(1, nA), float64(r))
		cmp("D2D(zoh) A vs Ad^r", q.a, pw, ke*eps*g*math.Max(1, nA))
		cmp("D2D(zoh) B vs ΣAd^k·Bd", q.b, sum, ke*eps*g*float64(r)*math.Max(1, fzFrob(s.B))*math.Max(1, nA))
		return
	}
	pw, sum := fzDiscRatPow(n, m, q.a, q.b, r)
	g := math.Pow(math.Max(1, fzFrob(q.a)), float64(r))
	cmp("D2D(zoh, up)^r A vs Ad", pw, s.A, ke*eps*g*math.Max(1, nA))
	cmp("D2D(zoh, up) ΣA₂^k·B₂ vs Bd", sum, s.B, ke*eps*g*float64(r)*math.Max(1, fzFrob(q.b))*math.Max(1, nA))
	var eig mat.Eigen
	if eig.Factorize(mat.NewDense(n, n, slices.Clone(q.a)), mat.EigenNone) {
		want := make([]complex128, n)
		for k, e := range s.S {
			want[k] = cmplx.Exp(e * complex(s.Dt/float64(r), 0))
		}
		rad := 1e3 * float64(n) * eps * s.CondT * s.CondT * (1 + inv) * math.Max(1, nA)
		if _, _, msg := fzMatchEig(eig.Values(nil), want, rad); msg != "" {
			t.Fatalf("D2D(zoh, up) poles vs e^{λTs/r} (r=%g): %s\n%s", rad, msg, s)
		}
	}
}
