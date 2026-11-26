package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func bitsEqualDense(a, b *mat.Dense) bool {
	if a == nil || b == nil {
		return a == b
	}
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := range ar {
		for j := range ac {
			if math.Float64bits(a.At(i, j)) != math.Float64bits(b.At(i, j)) {
				return false
			}
		}
	}
	return true
}

func bitsEqualSystem(a, b *System) bool {
	return bitsEqualDense(a.A, b.A) && bitsEqualDense(a.B, b.B) && bitsEqualDense(a.C, b.C) && bitsEqualDense(a.D, b.D) &&
		a.Dt == b.Dt && slices.Equal(a.InputDelay, b.InputDelay) && slices.Equal(a.OutputDelay, b.OutputDelay) &&
		bitsEqualDense(a.Delay, b.Delay) && slices.Equal(a.StateName, b.StateName)
}

func c2dMapRowSystem(t *testing.T, name string) *System {
	t.Helper()
	for _, rows := range [][]descriptorDCGateRow{descriptorDCGateRows, c2dMapRows} {
		for _, r := range rows {
			if r.name == name {
				sys, err := r.system()
				if err != nil {
					t.Fatal(err)
				}
				return sys
			}
		}
	}
	t.Fatalf("no row %s", name)
	return nil
}

// errClass is nil, unv (model with ErrAccuracyUnverified) or err.
func errClass(out *System, err error) string {
	switch {
	case err == nil:
		return "nil"
	case out != nil && errors.Is(err, ErrAccuracyUnverified):
		return "unv"
	}
	return "err"
}

// C2DMap of a full-rank continuous descriptor returns C2D's model bitwise,
// with C2D's verdict, on every route C2D takes (T3IVBX; EJDS3U ruling R8):
// the band and stiff pencil parts, a refused band split (fold, unverified),
// the real Schur retry and the plain fold. cond/567's split certificate
// changes when [E A] is appended to B: the riders must not enter it.
func TestC2DMapDescriptorMatchesC2D(t *testing.T) {
	names := []string{"cond/13", "cond/111", "cond/644", "strad1m/47", "strad1m/283", "stfr/3408", "strad1m/17", "stfr/399", "cond/567", "stfr/2880", "strad1m/212"}
	for _, name := range names {
		sys := c2dMapRowSystem(t, name)
		n, m, _ := sys.Dims()
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse} {
				key := fmt.Sprintf("%s@%g/%s", name, dt, meth)
				want, werr := sys.C2D(dt, C2DOptions{Method: meth})
				got, g, err := sys.C2DMap(dt, C2DOptions{Method: meth})
				if wc, gc := errClass(want, werr), errClass(got, err); wc != gc {
					t.Errorf("%s: C2DMap %s (%v), C2D %s (%v)", key, gc, err, wc, werr)
					continue
				}
				if want == nil {
					continue
				}
				if !bitsEqualSystem(got, want) {
					t.Errorf("%s: C2DMap model differs from C2D's", key)
				}
				nd, _, _ := got.Dims()
				if r, c := g.Dims(); r != nd || c != n+m {
					t.Errorf("%s: G is %d×%d, want %d×%d", key, r, c, nd, n+m)
				}
			}
		}
	}
}

// mapResponse returns y[k], k < K, of discrete d from x[0] = G·[x0; u[0]]
// under the inputs u (m × K+1; nil for none).
func mapResponse(d *System, G *mat.Dense, x0 []float64, u *mat.Dense, K int) *mat.Dense {
	nd, m, p := d.Dims()
	n := len(x0)
	z := make([]float64, n+m)
	copy(z, x0)
	if u != nil {
		for j := range m {
			z[n+j] = u.At(j, 0)
		}
	}
	x := mat.NewVecDense(nd, nil)
	x.MulVec(G.Slice(0, nd, 0, n+m), mat.NewVecDense(n+m, z))
	Y := mat.NewDense(p, K, nil)
	uk := mat.NewVecDense(m, nil)
	for k := range K {
		for j := range m {
			if u != nil {
				uk.SetVec(j, u.At(j, k))
			}
		}
		var y mat.VecDense
		y.MulVec(d.C, x)
		var du mat.VecDense
		du.MulVec(d.D, uk)
		y.AddVec(&y, &du)
		for i := range p {
			Y.Set(i, k, y.AtVec(i))
		}
		var nx, bu mat.VecDense
		nx.MulVec(d.A, x)
		bu.MulVec(d.B, uk)
		nx.AddVec(&nx, &bu)
		x = &nx
	}
	return Y
}

// responseError is max over outputs of max_k |got − ref| / max_k |ref|.
func responseError(got, ref *mat.Dense) float64 {
	p, K := ref.Dims()
	worst := 0.0
	for i := range p {
		scale, diff := 0.0, 0.0
		for k := range K {
			scale = math.Max(scale, math.Abs(ref.At(i, k)))
			d := math.Abs(got.At(i, k) - ref.At(i, k))
			if math.IsNaN(d) {
				d = math.Inf(1)
			}
			diff = math.Max(diff, d)
		}
		if diff > 0 {
			worst = math.Max(worst, diff/scale)
		}
	}
	return worst
}

// clusterStiffDescriptor is a stable 5-state descriptor (E symmetric
// positive definite, A symmetric negative definite) whose E has three stiff
// singular values within a factor 1.7: its fast modes form one part that
// splitFastPencil does not split, dense after svdExplicit.
func clusterStiffDescriptor(t *testing.T) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(0, 99))
	const n = 5
	orth := func() *mat.Dense {
		a := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				a.Set(i, j, rng.NormFloat64())
			}
		}
		var qr mat.QR
		qr.Factorize(a)
		var q mat.Dense
		qr.QTo(&q)
		return &q
	}
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range i + 1 {
			v := 0.2 * rng.NormFloat64()
			A.Set(i, j, v)
			A.Set(j, i, v)
		}
		A.Set(i, i, A.At(i, i)-1)
	}
	q := orth()
	var T, E mat.Dense
	T.Mul(q, mat.NewDiagDense(n, []float64{1, 0.7, 1e-9, 1.3e-9, 1.7e-9}))
	E.Mul(&T, q.T())
	sys, err := NewDescriptor(A, mat.NewDense(n, 1, []float64{1, 0.3, -0.7, 0.5, 0.2}),
		mat.NewDense(1, n, []float64{0.6, -0.9, 0.4, 0.2, 0.1}), mat.NewDense(1, 1, []float64{0.1}), &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// With an input delay of whole samples, C2D of a stiff descriptor holds the
// generalized Schur parts in their real Schur forms (stiffDescriptorExplicit)
// and C2DMap's G carries Zᵀ·W into them: the free response from G·x0 is the
// delay-free one, which the input delay does not act on. The cluster row's
// fast part is dense, so its Schur form changes coordinates.
func TestC2DMapDescriptorInputDelaySchurParts(t *testing.T) {
	cluster := clusterStiffDescriptor(t)
	sp, ok := foldInaccurateParts(cluster)
	dense := false
	for _, part := range sp.parts {
		dense = dense || !upperQuasiTriangular(part.A)
	}
	if !ok || sp.err != nil || !dense {
		t.Fatal("cluster row: no certified split with a dense part; the case no longer tests Zᵀ")
	}
	for name, sys := range map[string]*System{"stfr/399": c2dMapRowSystem(t, "stfr/399"), "cluster": cluster} {
		n, _, _ := sys.Dims()
		x0 := c2dMapRefX0(n)
		for _, dt := range []float64{1e-4, 1e-2} {
			delayed := sys.Copy()
			delayed.InputDelay = []float64{3 * dt}
			d, g, err := delayed.C2DMap(dt, C2DOptions{})
			if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
				t.Fatalf("%s@%g: %v", name, dt, err)
			}
			d0, g0, err := sys.C2DMap(dt, C2DOptions{})
			if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
				t.Fatalf("%s@%g core: %v", name, dt, err)
			}
			if e := responseError(mapResponse(d, g, x0, nil, 21), mapResponse(d0, g0, x0, nil, 21)); !(e <= 1e-9) {
				t.Errorf("%s@%g: free response from G·x0 %.3g off the delay-free one", name, dt, e)
			}
		}
	}
}

// A descriptor whose E is singular by the MATLAB rule is reduced; its
// algebraic states have no image (EJDS3U ruling R8).
func TestC2DMapDescriptorReducedKeepsInitialStateError(t *testing.T) {
	sys := c2dMapRowSystem(t, "stfr/45")
	for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse} {
		if d, g, err := sys.C2DMap(0.01, C2DOptions{Method: meth}); !errors.Is(err, ErrDescriptorInitialState) || d != nil || g != nil {
			t.Errorf("%s: %v, want ErrDescriptorInitialState with no model", meth, err)
		}
	}
}

var c2dMapRefKs = []int{0, 1, 2, 4, 8, 16, 20}

func c2dMapRefX0(n int) []float64 {
	x := make([]float64, n)
	for i := range n {
		x[i] = (1 + 0.25*float64(i)) * float64(1-2*(i%2))
	}
	return x
}

func c2dMapRefU(m int) *mat.Dense {
	u := mat.NewDense(m, 22, nil)
	for j := range m {
		for k := range 22 {
			u.Set(j, k, 1+0.5*math.Sin(0.7*float64(k)+float64(j)))
		}
	}
	return u
}

// c2dMapResponseErrors returns, per reference, the error of the response
// of C2DMap's model from G·[x0; u0] (responseError at c2dMapRefKs) and its
// class; NaN without a model.
func c2dMapResponseErrors(t *testing.T) ([]float64, []string) {
	errs, classes := make([]float64, len(c2dMapResponseRefs)), make([]string, len(c2dMapResponseRefs))
	for i, ref := range c2dMapResponseRefs {
		sys := c2dMapRowSystem(t, ref.row)
		n, m, p := sys.Dims()
		d, g, err := sys.C2DMap(ref.dt, C2DOptions{Method: ref.method})
		classes[i] = errClass(d, err)
		if d == nil {
			errs[i] = math.NaN()
			continue
		}
		x0, u := c2dMapRefX0(n), (*mat.Dense)(nil)
		if ref.forced {
			x0, u = make([]float64, n), c2dMapRefU(m)
		}
		y := mapResponse(d, g, x0, u, 21)
		got, want := mat.NewDense(p, len(c2dMapRefKs), nil), mat.NewDense(p, len(c2dMapRefKs), nil)
		for o := range p {
			for k, kk := range c2dMapRefKs {
				got.Set(o, k, y.At(o, kk))
				want.Set(o, k, ref.y[o][k])
			}
		}
		errs[i] = responseError(got, want)
	}
	return errs, classes
}

// C2DMap's G maps x0 and u0 into the coordinates of C2D's model (T3IVBX):
// from G·[x0; u0], the response of a nil model matches the exact method on
// the exact pencil (frozen 320-bit references) within 1000 times its own
// 1-ulp data floor or 10 times the model's own error, the forced ZOH
// response, where G plays no part (its u0 columns are zero). At t = 0,
// C_d·G_x reproduces C. A nil model is certified at DC only, so the bar
// leaves the model's off-DC error to the model.
func TestC2DMapDescriptorInitialResponse(t *testing.T) {
	errs, classes := c2dMapResponseErrors(t)
	type cell struct {
		row string
		dt  float64
	}
	model := map[cell]float64{}
	for i, ref := range c2dMapResponseRefs {
		if ref.method == C2DMethodZOH && ref.forced {
			model[cell{ref.row, ref.dt}] = errs[i]
		}
	}
	checked := 0
	for i, ref := range c2dMapResponseRefs {
		key := fmt.Sprintf("%s@%g/%s forced=%t", ref.row, ref.dt, ref.method, ref.forced)
		if classes[i] != "nil" || ref.method == C2DMethodImpulse || ref.method == C2DMethodZOH && ref.forced {
			continue
		}
		tol := 1e3 * ref.floor
		if e, ok := model[cell{ref.row, ref.dt}]; ok {
			tol = math.Max(tol, 10*e)
		}
		if !(errs[i] <= tol) {
			t.Errorf("%s: response from G %.3g off, bar %.3g (floor %.3g)", key, errs[i], tol, ref.floor)
		}
		checked++
	}
	if checked < 40 {
		t.Errorf("%d nil cells checked, want at least 40", checked)
	}
	for _, name := range []string{"cond/13", "stfr/399", "cond/644", "strad1m/17", "strad1m/212", "stfr/2880", "cond/567"} {
		sys := c2dMapRowSystem(t, name)
		n, _, _ := sys.Dims()
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse} {
				d, g, err := sys.C2DMap(dt, C2DOptions{Method: meth})
				if err != nil {
					continue
				}
				nd, m, _ := d.Dims()
				var cw mat.Dense
				cw.Mul(d.C, g.Slice(0, nd, 0, n))
				cw.Sub(&cw, sys.C)
				if e := mat.Norm(&cw, 2) / mat.Norm(sys.C, 2); !(e <= 1e-9) {
					t.Errorf("%s@%g/%s: ‖C_d·G_x − C‖/‖C‖ = %.3g", name, dt, meth, e)
				}
				if (meth == C2DMethodZOH || meth == C2DMethodImpulse) && m > 0 && mat.Norm(g.Slice(0, nd, n, n+m), 1) != 0 {
					t.Errorf("%s@%g/%s: G has nonzero u0 columns", name, dt, meth)
				}
			}
		}
	}
}

// Riders take [E A] through the split without entering its cut or
// certificate: the parts are bitwise those of the caller's model. Appended
// to B as inputs instead, they change ζ″ and with it the parts on cond/567
// (EJDS3U ruling R8).
func TestStiffSplitRidersLeaveSplit(t *testing.T) {
	for _, name := range []string{"cond/13", "stfr/399", "strad1m/17", "cond/567"} {
		sys := c2dMapRowSystem(t, name)
		n, m, p := sys.Dims()
		riders := (&c2dCoords{}).riders(sys)
		plain, ok := stiffExplicitSplit(sys)
		rode, rok := stiffExplicitSplitRiders(sys, riders)
		if !ok || !rok || len(plain.parts) != len(rode.parts) || plain.zeta != rode.zeta || (plain.err == nil) != (rode.err == nil) {
			t.Fatalf("%s: split with riders differs", name)
		}
		for k := range plain.parts {
			if !bitsEqualSystem(plain.parts[k], rode.parts[k]) {
				t.Errorf("%s: part %d differs with riders", name, k)
			}
			pk, _, _ := plain.parts[k].Dims()
			if r, c := rode.riders[k].Dims(); r != pk || c != 2*n {
				t.Errorf("%s: part %d riders %d×%d, want %d×%d", name, k, r, c, pk, 2*n)
			}
		}
		if name != "cond/567" {
			continue
		}
		wide := sys.Copy()
		wide.B = newDense(n, m+2*n)
		wide.B.Slice(0, n, 0, m).(*mat.Dense).Copy(sys.B)
		wide.B.Slice(0, n, m, m+2*n).(*mat.Dense).Copy(riders)
		wide.D = newDense(p, m+2*n)
		wide.D.Slice(0, p, 0, m).(*mat.Dense).Copy(sys.D)
		wide.InputName = nil
		ws, _ := stiffExplicitSplit(wide)
		if len(ws.parts) == len(plain.parts) && ws.zeta == plain.zeta && (ws.err == nil) == (plain.err == nil) {
			t.Errorf("%s: appending [E A] to B no longer changes the split; the row no longer tests the riders", name)
		}
	}
}
