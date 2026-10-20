package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzC2DD2C: continuous models discretize and discrete models
// undiscretize consistently with exact identities. Tustin satisfies
// Hd(z) = Hc((2/Ts)(z−1)/(z+1)) at every point; ZOH and FOH keep the DC
// gain (Hd(1) = Hc(0)) and ZOH keeps D bitwise and maps the poles to
// e^{λTs}; D2C inverts C2D for ZOH, FOH and Tustin on the principal
// logarithm branch (|λ|·Ts ≤ 1). Each comparison is between exact rational
// responses of the float64 realizations, within first-order bounds for
// blockwise backward errors scaled by cond(I − A·Ts/2) (Tustin) or the
// eigenvector condition (ZOH/FOH). C2D of a discrete model and D2C of a
// continuous one return ErrWrongDomain.
func FuzzC2DD2C(f *testing.F) {
	for _, s := range fzConvSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		if s.Discrete {
			_, err := sys.C2D(s.Dt, C2DOptions{})
			if !errors.Is(err, ErrWrongDomain) {
				t.Fatalf("C2D of a discrete model: err %v, want ErrWrongDomain\n%s", err, s)
			}
			fzD2CChecks(t, s, sys)
			return
		}
		_, err := sys.D2C(D2COptions{})
		if !errors.Is(err, ErrWrongDomain) {
			t.Fatalf("D2C of a continuous model: err %v, want ErrWrongDomain\n%s", err, s)
		}
		dt := math.Pow(10, fzClamp(s.Vals[8], -4, 2))
		for _, e := range s.S {
			if math.Abs(real(e))*dt > 300 {
				fzSkip(t, "exp(s·Dt) out of range")
			}
		}
		fzC2DChecks(t, s, sys, dt)
	})
}

type fzMats struct {
	n, m, p    int
	a, b, c, d []float64
}

func fzMatsOf(sys *System) fzMats {
	n, m, p := sys.Dims()
	a, b, c, d := fzRaw(sys)
	return fzMats{n, m, p, a, b, c, d}
}

func (q fzMats) resp(x complex128) ([]complex128, bool) {
	return fzExactResponse(q.n, q.m, q.p, q.a, q.b, q.c, q.d, x)
}

// pert bounds ‖ΔH(x)‖ (entrywise) to first order for blockwise
// perturbations of size dA, dB, dC, dD; +Inf when not first-order safe.
func (q fzMats) pert(x complex128, dA, dB, dC, dD float64) float64 {
	if q.n == 0 {
		return dD
	}
	g := make([]complex128, q.n*q.n)
	for i := range q.n {
		for j := range q.n {
			g[i*q.n+j] = complex(-q.a[i*q.n+j], 0)
		}
		g[i*q.n+i] += x
	}
	sv := fzSingularValues(g, q.n, q.n)
	r := 1 / sv[q.n-1]
	if !(dA*r < 0.1) {
		return math.Inf(1)
	}
	nB, nC := fzFrob(q.b), fzFrob(q.c)
	return (dC*r*nB+nC*r*dB+nC*r*dA*r*nB)/(1-dA*r) + dD
}

func fzCond(a []float64, n int, alpha float64) float64 {
	g := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			g[i*n+j] = complex(alpha*a[i*n+j], 0)
		}
		g[i*n+i] += 1
	}
	sv := fzSingularValues(g, n, n)
	return sv[0] / sv[n-1]
}

// fzCompare fails when got and want differ by more than tol; a non-empty
// known names a reported library defect whose regime the input is in, and
// turns the mismatch into a counted skip.
func fzCompare(t *testing.T, s *fzSpec, op string, x complex128, got, want []complex128, tol float64, known string) {
	t.Helper()
	for e := range got {
		if d := cmplx.Abs(got[e] - want[e]); !(d <= tol) {
			if known != "" {
				fzSkip(t, known)
			}
			t.Fatalf("%s at %v: [%d] = %v, want %v (|Δ| %g, tol %g)\n%s", op, x, e, got[e], want[e], d, tol, s)
		}
	}
}

func fzC2DChecks(t *testing.T, s *fzSpec, sys *System, dt float64) {
	const eps = 0x1p-52
	n := s.N
	orig := fzMats{n, s.M, s.P, s.A, s.B, s.C, s.D}
	nA, nB, nC, nD := fzFrob(s.A), fzFrob(s.B), fzFrob(s.C), fzFrob(s.D)
	kk := 256 * float64((n+1)*(n+1))
	pts := append(fzConvPoints(s), 0)

	td, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	fzContract(t, s, "C2D(tustin)", td != nil, err)
	fzErr(t, s, "C2D(tustin)", err)
	if td.Dt != dt {
		t.Fatalf("C2D(tustin) dt = %g, want %g\n%s", td.Dt, dt, s)
	}
	tm := fzMatsOf(td)
	kT := 1.0
	if n > 0 {
		kT = fzCond(s.A, n, -dt/2)
	}
	if kT < 1e8 {
		var rt fzMats
		back, err := td.D2C(D2COptions{Method: C2DMethodTustin})
		fzContract(t, s, "D2C(tustin)", back != nil, err)
		fzErr(t, s, "D2C(tustin)", err)
		rt = fzMatsOf(back)
		for _, x := range pts {
			den := 1 - x*complex(dt/2, 0)
			if cmplx.Abs(den) < 1e-8 {
				continue
			}
			z := (1 + x*complex(dt/2, 0)) / den
			x2 := complex(2/dt, 0) * (z - 1) / (z + 1)
			dx := cmplx.Abs(x2) * eps * (3 + (cmplx.Abs(z)+1)*(1/cmplx.Abs(z-1)+1/cmplx.Abs(z+1)))
			hc, ok1 := orig.resp(x2)
			hd, ok2 := tm.resp(z)
			if ok1 && ok2 {
				dA := kk*eps*kT*(nA+2/dt) + 4*dx
				tol := orig.pert(x2, dA, kk*eps*kT*nB, kk*eps*kT*nC, kk*eps*kT*(nD+nC*nB*dt*kT))
				fzCompare(t, s, "C2D(tustin) vs Hc(bilinear)", z, hd, hc, tol, "")
			}
			hc, ok1 = orig.resp(x)
			hr, ok2 := rt.resp(x)
			if ok1 && ok2 {
				k2 := kk * kT * kT
				tol := orig.pert(x, k2*eps*(nA+2/dt), k2*eps*nB, k2*eps*nC, k2*eps*(nD+nC*nB*dt*kT))
				fzCompare(t, s, "D2C(C2D(tustin))", x, hr, hc, tol, "")
			}
		}
	}

	if n > 0 && s.NormJ*dt > 20 {
		return
	}
	zd, err := sys.C2D(dt, C2DOptions{})
	fzContract(t, s, "C2D(zoh)", zd != nil, err)
	fzErr(t, s, "C2D(zoh)", err)
	zm := fzMatsOf(zd)
	if zd.Dt != dt || zm.n != n || !slices.Equal(zm.d, s.D) {
		t.Fatalf("C2D(zoh): dt=%g n=%d D=%v, want %g %d %v\n%s", zd.Dt, zm.n, zm.d, dt, n, s.D, s)
	}
	fd, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	fzContract(t, s, "C2D(foh)", fd != nil, err)
	fzErr(t, s, "C2D(foh)", err)
	fm := fzMatsOf(fd)
	growth := math.Exp(math.Min(s.NormJ*dt, 20)) * s.CondT
	ke := kk * s.CondT * s.CondT
	nAB := math.Hypot(nA, nB)
	if hc, ok := orig.resp(0); ok {
		for _, c := range []struct {
			name string
			q    fzMats
		}{{"C2D(zoh)", zm}, {"C2D(foh)", fm}} {
			hd, ok := c.q.resp(1)
			if !ok {
				continue
			}
			tol := orig.pert(0, ke*eps*nAB, ke*eps*nAB, ke*eps*nC, ke*eps*nD)
			qa, qb, qc, qd := fzFrob(c.q.a), fzFrob(c.q.b), fzFrob(c.q.c), fzFrob(c.q.d)
			tol += c.q.pert(1, kk*eps*growth*math.Max(1, qa), kk*eps*growth*qb, kk*eps*growth*qc, kk*eps*growth*qd)
			fzCompare(t, s, c.name+" DC gain", 1, hd, hc, tol, "")
		}
	}
	if n > 0 {
		var eig mat.Eigen
		if eig.Factorize(mat.NewDense(n, n, slices.Clone(zm.a)), mat.EigenNone) {
			want := make([]complex128, n)
			maxRe := math.Inf(-1)
			for k, e := range s.S {
				want[k] = cmplx.Exp(e * complex(dt, 0))
				maxRe = math.Max(maxRe, real(e))
			}
			r := 1e3 * float64(n) * eps * s.CondT * s.CondT * (1 + s.NormJ*dt) * math.Max(1, math.Exp(maxRe*dt))
			if _, _, msg := fzMatchEig(eig.Values(nil), want, r); msg != "" {
				t.Fatalf("C2D(zoh) poles vs e^{λTs} (r=%g): %s\n%s", r, msg, s)
			}
		}
	}

	if n > 0 && (s.NormJ*dt > 1 || s.CondT > 1e4) {
		return
	}
	for _, c := range []struct {
		name  string
		sd    *System
		opts  D2COptions
		q     fzMats
		known string
	}{{"D2C(C2D(zoh))", zd, D2COptions{}, zm, ""}, {"D2C(C2D(foh))", fd, D2COptions{Method: C2DMethodFOH}, fm, ""}} {
		back, err := c.sd.D2C(c.opts)
		fzContract(t, s, c.name, back != nil, err)
		fzErr(t, s, c.name, err)
		bm := fzMatsOf(back)
		if bm.n != n {
			t.Fatalf("%s order %d, want %d\n%s", c.name, bm.n, n, s)
		}
		inv := 1 / math.Max(fzUnitGap(c.q), 1e-8)
		kr := ke * growth
		dA := kr * eps * (nA + 1/dt)
		for _, x := range pts {
			hc, ok1 := orig.resp(x)
			hr, ok2 := bm.resp(x)
			if !ok1 || !ok2 {
				continue
			}
			tol := orig.pert(x, dA, kr*eps*nB*(1+inv), kr*eps*nC*(1+inv), kr*eps*(nD+nC*nB*dt*(1+inv)))
			fzCompare(t, s, c.name, x, hr, hc, tol, c.known)
		}
	}
}

func fzD2CChecks(t *testing.T, s *fzSpec, sys *System) {
	const eps = 0x1p-52
	n, dt := s.N, s.Dt
	orig := fzMats{n, s.M, s.P, s.A, s.B, s.C, s.D}
	nA, nB, nC, nD := fzFrob(s.A), fzFrob(s.B), fzFrob(s.C), fzFrob(s.D)
	kk := 256 * float64((n+1)*(n+1))
	ke := kk * s.CondT * s.CondT
	cont, err := sys.D2C(D2COptions{})
	fzContract(t, s, "D2C(zoh)", cont != nil, err)
	if !fzErr(t, s, "D2C(zoh)", err, ErrSingularTransform) {
		return
	}
	cm := fzMatsOf(cont)
	if cont.Dt != 0 || !slices.Equal(cm.d, s.D) {
		t.Fatalf("D2C(zoh): dt=%g D=%v, want 0, %v\n%s", cont.Dt, cm.d, s.D, s)
	}
	minZ, maxS := math.Inf(1), 0.0
	for k, e := range s.Eig {
		minZ = math.Min(minZ, cmplx.Abs(e))
		maxS = math.Max(maxS, cmplx.Abs(s.S[k]))
	}
	if n > 0 && (minZ < 1e-3 || s.CondT > 1e4) {
		return
	}
	ca, cb, cc, cd := fzFrob(cm.a), fzFrob(cm.b), fzFrob(cm.c), fzFrob(cm.d)
	if hd, ok := orig.resp(1); ok {
		if hc, ok := cm.resp(0); ok {
			inv := 1 / math.Max(fzUnitGap(orig), 1e-8)
			tol := orig.pert(1, ke*eps*math.Max(1, nA), ke*eps*nB*(1+inv), ke*eps*nC, ke*eps*nD)
			tol += cm.pert(0, ke*eps*(ca+1/dt), ke*eps*cb*(1+inv), ke*eps*cc, ke*eps*cd)
			fzCompare(t, s, "D2C(zoh) DC gain", 0, hc, hd, tol, "")
		}
	}
	if n > 0 && maxS*dt > 1 {
		return
	}
	inv := 1 / math.Max(fzUnitGap(orig), 1e-8)
	back, err := cont.C2D(dt, C2DOptions{})
	fzContract(t, s, "C2D(D2C(zoh))", back != nil, err)
	fzErr(t, s, "C2D(D2C(zoh))", err)
	bm := fzMatsOf(back)
	for _, x := range fzConvPoints(s) {
		hd, ok1 := orig.resp(x)
		hr, ok2 := bm.resp(x)
		if !ok1 || !ok2 {
			continue
		}
		tol := orig.pert(x, ke*eps*math.Max(1, nA)*(1+inv), ke*eps*nB*(1+inv), ke*eps*nC*(1+inv), ke*eps*nD)
		fzCompare(t, s, "C2D(D2C(zoh))", x, hr, hd, tol, "")
	}
}

// fzUnitGap is σmin(Ad − I)/max(1, ‖Ad‖) (+Inf for a static model).
func fzUnitGap(q fzMats) float64 {
	if q.n == 0 {
		return math.Inf(1)
	}
	g := make([]complex128, q.n*q.n)
	for i := range q.n {
		for j := range q.n {
			g[i*q.n+j] = complex(q.a[i*q.n+j], 0)
		}
		g[i*q.n+i] -= 1
	}
	sv := fzSingularValues(g, q.n, q.n)
	return sv[q.n-1] / math.Max(1, fzFrob(q.a))
}
