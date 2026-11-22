package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// descForm is a model stored in generalized Schur form, Q = Z = I, with
// its split data at n1 computed as splitGeneralizedSchur computes it.
type descForm struct {
	sys         *System
	s, t, q, z  []float64
	n, n1, m, p int
	r, l        []float64
	bq0, cz0    []float64
	bq, cz      []float64
	a1, b1f     *mat.Dense
}

func newDescForm(t *testing.T, n, n1 int, s, tt, b, c []float64) descForm {
	t.Helper()
	m, p := len(b)/n, len(c)/n
	sys, err := NewDescriptor(mat.NewDense(n, n, slices.Clone(s)), mat.NewDense(n, m, slices.Clone(b)),
		mat.NewDense(p, n, slices.Clone(c)), mat.NewDense(p, m, nil), mat.NewDense(n, n, slices.Clone(tt)), 0)
	if err != nil {
		t.Fatal(err)
	}
	f := descForm{sys: sys, s: slices.Clone(s), t: slices.Clone(tt), n: n, n1: n1, m: m, p: p}
	f.q, f.z = make([]float64, n*n), make([]float64, n*n)
	for i := range n {
		f.q[i*n+i], f.z[i*n+i] = 1, 1
	}
	n2 := n - n1
	f.bq0, f.cz0 = slices.Clone(b), slices.Clone(c)
	f.bq, f.cz = slices.Clone(b), slices.Clone(c)
	f.r, f.l = make([]float64, n1*n2), make([]float64, n1*n2)
	for i := range n1 {
		for j := range n2 {
			f.r[i*n2+j] = -f.s[i*n+n1+j]
			f.l[i*n2+j] = -f.t[i*n+n1+j]
		}
	}
	scale, ok := qzSylvester(n, n1, false, f.s, f.t, f.r, f.l)
	if !ok || scale == 0 {
		t.Fatal("qzSylvester failed")
	}
	for i := range f.r {
		f.r[i] /= scale
		f.l[i] /= scale
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: f.l},
		blas64.General{Rows: n2, Cols: m, Stride: m, Data: f.bq[n1*m:]}, 1, blas64.General{Rows: n1, Cols: m, Stride: m, Data: f.bq})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: p, Cols: n1, Stride: n, Data: f.cz},
		blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: f.r}, 1, blas64.General{Rows: p, Cols: n2, Stride: n, Data: f.cz[n1:]})
	f.a1 = extractModalBlock(f.s, n, 0, n1, 0, n1)
	f.b1f = extractModalBlock(f.bq, m, 0, n1, 0, m)
	t11 := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: n1, Stride: n, Data: f.t}
	blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, f.a1.RawMatrix())
	blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, f.b1f.RawMatrix())
	return f
}

// budget runs addGeneralizedSplit on the form (fields may be altered by
// the caller first) and returns the plain totals.
func (f descForm) budget() (d descriptorBudget, zS, zT, z2 float64) {
	d.addGeneralizedSplit(newDescriptorBar(f.sys), f.sys, f.s, f.t, f.q, f.z, f.n1, f.r, f.l, f.bq0, f.cz0, f.bq, f.cz, f.a1, f.b1f)
	zS, zT, z2 = d.zS, d.zT, d.z2
	for _, sp := range d.splits {
		zS, zT, z2 = zS+sp.zS, zT+sp.zT, z2+sp.z2
	}
	return d, zS, zT, z2
}

// tiers is ζ″ of the plain terms and of the ridge-sharpened ones.
func (d *descriptorBudget) tiers() (plain, sharp float64) {
	zS, zT, z2 := d.zS, d.zT, d.z2
	sS, sT, s2 := d.zS, d.zT, d.z2
	for _, sp := range d.splits {
		zS, zT, z2 = zS+sp.zS, zT+sp.zT, z2+sp.z2
		a, b, c := sp.zS, sp.zT, sp.z2
		if sp.sharpen != nil {
			x, y, w := sp.sharpen()
			a, b, c = math.Min(a, x), math.Min(b, y), math.Min(c, w)
		}
		sS, sT, s2 = sS+a, sT+b, s2+c
	}
	th := d.theta
	if !(th > 0) {
		th = 1
	}
	return secondTierZeta(zS, zT, z2, th), secondTierZeta(sS, sT, s2, th)
}

// descTriangularForm: S = [−1 γ 1/2; 0 1 1/4; 0 0 1], T = diag(1, 1, 0),
// one stable, one unstable and one infinite mode, split at n1 = 1 with
// coupling R, L ≈ γ/2.
func descTriangularForm(t *testing.T, gamma float64) descForm {
	return newDescForm(t, 3, 1, []float64{-1, gamma, 0.5, 0, 1, 0.25, 0, 0, 1}, []float64{1, 0, 0, 0, 1, 0, 0, 0, 0},
		[]float64{1, 0.5, 1}, []float64{0.5, 1, 1})
}

// TestDescriptorDecouplingCertificateTerms: each test of the descriptor
// certificate fires on its own (EUNKOP ruling 1): with every other term
// at most 0.1, a defect in the Sylvester residual of S only (P1S) or T only
// (P1T), a genuine coupling (P2), the fold of A or B, and the measured form
// of A or E each exceed their budget, and the first-term ones reject; a
// non-finite block and a cut inside a 2×2 block are rejected. (P2 and fold
// B weigh against ‖C‖‖M‖‖B‖, which θ may bound below the first bar term.)
func TestDescriptorDecouplingCertificateTerms(t *testing.T) {
	type which int
	const (
		termS which = iota
		termT
		term2
	)
	check := func(name string, f descForm, want which) {
		t.Helper()
		d, zS, zT, z2 := f.budget()
		got := []float64{zS, zT, z2}
		for i, v := range got {
			if which(i) == want {
				if !(v > 1) {
					t.Errorf("%s: term %d = %.3g, want > 1 (all %.3g)", name, i, v, got)
				}
			} else if !(v <= 0.1) {
				t.Errorf("%s: term %d = %.3g, want ≤ 0.1 (all %.3g)", name, i, v, got)
			}
		}
		if want != term2 && d.accepted() {
			t.Errorf("%s: accepted", name)
		}
	}
	base := descTriangularForm(t, 4)
	if _, zS, zT, z2 := base.budget(); !(zS <= 0.1 && zT <= 0.1 && z2 <= 0.1) {
		t.Fatalf("base terms %.3g %.3g %.3g", zS, zT, z2)
	}

	f := descTriangularForm(t, 4)
	f.l[1] += 1e-9
	check("P1S residual (L column of the infinite state, T22 column 0)", f, termS)

	f = newDescForm(t, 3, 1, []float64{0, 4, 0.5, 0, 1, 0.25, 0, 0, 1}, []float64{1, 0, 0, 0, 1, 0, 0, 0, 0},
		[]float64{1, 0.5, 1}, []float64{0.5, 1, 1})
	f.r[0] += 1e-9
	check("P1T residual (S11 = 0)", f, termT)

	f = newDescForm(t, 3, 1, []float64{-1, 4, 0.5, 0, 1, 0.25, 0, 0, 1}, []float64{1, 0, 0, 0, 1, 0, 0, 0, 0},
		[]float64{1, 0.25, 1}, []float64{0.5, 1, 1})
	f.a1.Set(0, 0, f.a1.At(0, 0)+1e-9)
	check("fold A", f, termS)

	f = descTriangularForm(t, 4)
	f.b1f.Set(0, 0, f.b1f.At(0, 0)+1e-9)
	check("fold B", f, term2)

	f = descTriangularForm(t, 4)
	f.sys.A.Set(2, 2, 1+1e-9)
	check("form A", f, termS)

	f = descTriangularForm(t, 4)
	f.sys.E.Set(1, 1, 1+1e-9)
	check("form E", f, termT)

	f = newDescForm(t, 3, 1, []float64{-1, 0x1p20, 0, 0, 1, 0, 0, 0, 1}, []float64{1, 0, 0, 0, 0, 0, 0, 0, 0},
		[]float64{1, 0.5, 1}, []float64{0.5, 1, 1})
	check("P2 (L ≈ 2²⁰ against infinite modes, R = 0)", f, term2)

	f = descTriangularForm(t, 4)
	f.r[0] = math.NaN()
	if d, _, _, _ := f.budget(); d.accepted() || !d.failed {
		t.Error("non-finite R accepted")
	}
	f = descTriangularForm(t, 4)
	f.bq[0] = math.Inf(1)
	if d, _, _, _ := f.budget(); d.accepted() || !d.failed {
		t.Error("non-finite B̂1 accepted")
	}

	g := descTriangularForm(t, 4)
	g.s[1*3+0] = 0.25
	_, _, err := splitGeneralizedSchur(newRealizationTransformPolicy(g.sys), g.sys, g.s, g.t, g.q, g.z, 1, false, nil)
	if !errors.Is(err, ErrSchurFailed) {
		t.Errorf("cut inside a 2×2 block of S: err = %v, want ErrSchurFailed", err)
	}
}

// TestDescriptorDecouplingBandBoundary: on the triangular form the plain
// terms grow like γ; the first tier accepts γ = 2⁸ (ζ″ ≈ 0.32) and
// rejects 2¹⁰ (≈ 1.27). The split stays accurate for every γ, so these are
// the certificate's band, not wrong splits.
func TestDescriptorDecouplingBandBoundary(t *testing.T) {
	for _, tc := range []struct {
		k      int
		accept bool
	}{{8, true}, {10, false}} {
		f := descTriangularForm(t, math.Ldexp(1, tc.k))
		d, _, _, _ := f.budget()
		plain, _ := d.tiers()
		if (plain <= 1) != tc.accept {
			t.Errorf("γ = 2^%d: first-tier ζ″ = %.3g, accept want %v", tc.k, plain, tc.accept)
		}
	}
}

// TestDescriptorDecouplingSecondTierBoundary: the ridge bounds certify
// γ = 2¹⁰ (sharpened ζ″ ≈ 0.28, first tier 1.27) and not 2¹² (≈ 1.12), and
// the parts of the certified split are within the bar; the θ floor 10·U·ε
// keeps a near-singular pencil's ‖C‖‖M‖‖B‖ term from being discounted.
func TestDescriptorDecouplingSecondTierBoundary(t *testing.T) {
	for _, tc := range []struct {
		k      int
		accept bool
	}{{10, true}, {12, false}} {
		f := descTriangularForm(t, math.Ldexp(1, tc.k))
		var d descriptorBudget
		g1, g2, err := splitGeneralizedSchur(newRealizationTransformPolicy(f.sys), f.sys, slices.Clone(f.s), slices.Clone(f.t), f.q, f.z, 1, false, &d)
		if err != nil {
			t.Fatal(err)
		}
		plain, sharp := d.tiers()
		if d.accepted() != tc.accept || (sharp <= 1) != tc.accept {
			t.Errorf("γ = 2^%d: ζ″ plain %.3g sharp %.3g accepted %v, want %v", tc.k, plain, sharp, d.accepted(), tc.accept)
		}
		if tc.accept {
			if !(plain > 1) {
				t.Errorf("γ = 2^%d: first tier ζ″ = %.3g, want > 1", tc.k, plain)
			}
			descRequireNilWithinBar(t, "second tier", f.sys, g1, g2, nil, sepPointsC)
		}
	}

	ns := []float64{0x1p-100, 1, 0.5, 0, 1, 0.25, 0, 0, 1}
	nt := []float64{0x1p-100, 0, 0, 0, 1, 0, 0, 0, 0}
	near, err := NewDescriptor(mat.NewDense(3, 3, slices.Clone(ns)), mat.NewDense(3, 1, []float64{1, 0.5, 1}), mat.NewDense(1, 3, []float64{0.5, 1, 1}),
		mat.NewDense(1, 1, nil), mat.NewDense(3, 3, slices.Clone(nt)), 0)
	if err != nil {
		t.Fatal(err)
	}
	bar := newDescriptorBar(near)
	theta := thetaBound(bar, ns, nt, 3, 1, 0, 0)
	floor := 10 * bar.u() * eps()
	if theta != floor {
		t.Fatalf("near-singular pencil: θ = %.3g, want the floor %.3g", theta, floor)
	}
	if z := secondTierZeta(0.5, 0, 1e13, theta); !(z > 1) {
		t.Errorf("floored θ: ζ″ = %.3g with z2 = 1e13, want > 1", z)
	}
	if z := secondTierZeta(0.5, 0, 1e13, 1e-15); !(z <= 1) {
		t.Errorf("unfloored θ: ζ″ = %.3g, want ≤ 1 (the floor is what rejects)", z)
	}
}

// TestDescriptorBarTermRatioBound: θ bounds the second term of the bar by
// the first, 1 ≤ θ·(‖A‖ + |s|‖E‖)·‖(sE − A)⁻¹‖, on a sweep of s (320-bit
// inverse), for graded non-symmetric forms.
func TestDescriptorBarTermRatioBound(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for trial := range 6 {
		n, n1 := 5, 2
		s := make([]float64, n*n)
		tt := make([]float64, n*n)
		for i := range n {
			g := math.Pow(10, -float64(trial%3)*float64(i))
			for j := i; j < n; j++ {
				s[i*n+j] = g * rng.NormFloat64()
				if i < n-1 {
					tt[i*n+j] = g * rng.NormFloat64()
				}
			}
		}
		f := newDescForm(t, n, n1, s, tt, []float64{1, 1, 1, 1, 1}, []float64{1, 1, 1, 1, 1})
		bar := newDescriptorBar(f.sys)
		theta := thetaBound(bar, f.s, f.t, n, n1, 0, 0)
		scored := 0
		for k := -8; k <= 16; k++ {
			for _, th := range []float64{math.Pi / 2, 1.1, 0.15} {
				x := cmplx.Rect(math.Pow(10, float64(k)/2), th)
				mf, ok := desc320InverseFro(f.sys, x)
				if !ok {
					continue
				}
				scored++
				if v := theta * (bar.a + cmplx.Abs(x)*bar.e) * mf; !(v >= 1-1e-12) {
					t.Errorf("trial %d s=%v: θ(‖A‖+|s|‖E‖)‖M‖ = %.3g < 1", trial, x, v)
				}
			}
		}
		if scored == 0 {
			t.Fatalf("trial %d: no point scored", trial)
		}
	}
}

// desc320InverseFro is ‖(sE − A)⁻¹‖_F in 320 bits.
func desc320InverseFro(sys *System, s complex128) (float64, bool) {
	n, _, _ := sys.Dims()
	K := newHbMat(2*n, 2*n)
	for i := range n {
		for j := range n {
			e := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.E.At(i, j))
			re := new(big.Float).SetPrec(hbPrec).SetFloat64(real(s))
			re.Mul(re, e)
			re.Sub(re, new(big.Float).SetPrec(hbPrec).SetFloat64(sys.A.At(i, j)))
			im := new(big.Float).SetPrec(hbPrec).SetFloat64(imag(s))
			im.Mul(im, e)
			K.at(i, j).Set(re)
			K.at(n+i, n+j).Set(re)
			K.at(i, n+j).Neg(im)
			K.at(n+i, j).Set(im)
		}
	}
	id := newHbMat(2*n, 2*n)
	for i := range 2 * n {
		id.at(i, i).SetFloat64(1)
	}
	inv := hbSolve(K, id)
	var acc l2
	for i := range n {
		for j := range n {
			re, _ := inv.at(i, j).Float64()
			im, _ := inv.at(n+i, j).Float64()
			acc.addC(complex(re, im))
		}
	}
	v := acc.norm()
	return v, !math.IsInf(v, 0) && !math.IsNaN(v)
}

// TestDescriptorDecouplingRidgeBoundsExact: for the triangular form with
// coupling γ = 2¹⁰ and a perturbation along R, ridgeBounds.right and left
// bound ‖Δ·M22·B̃2‖/‖M‖ and
// ‖C̃1·M11·Δ‖/‖M‖ (M = (sT − S)⁻¹ in 320 bits) for |Δ| at the given bound,
// on a sweep of s, and are sharper than the plain ‖Δ‖‖B̃2‖, ‖C̃1‖‖Δ‖.
func TestDescriptorDecouplingRidgeBoundsExact(t *testing.T) {
	n, n1 := 3, 1
	n2 := n - n1
	f := descTriangularForm(t, 0x1p10)
	rb := newRidgeBounds(n1, n2, f.m, f.p, f.r, f.l, f.bq0, f.cz0)
	if !rb.okR || !rb.okL {
		t.Fatal("ridge factor failed")
	}
	dR := []float64{1, 1e-3}
	dL := []float64{1}
	right, left := rb.right(dR, 1), rb.left(dL, 1)
	if !(right < froAll(dR)*rb.b2) || !(left < rb.c1*froAll(dL)) {
		t.Fatalf("ridge bounds %.3g %.3g not sharper than plain %.3g %.3g", right, left, froAll(dR)*rb.b2, rb.c1*froAll(dL))
	}
	scored := 0
	for k := -6; k <= 12; k++ {
		for _, th := range []float64{math.Pi / 2, 1.1, 0.15} {
			x := cmplx.Rect(math.Pow(10, float64(k)/2), th)
			M, mf, ok := desc320Inverse(f.sys, x)
			if !ok {
				continue
			}
			scored++
			for _, sg := range []float64{1, -1} {
				var vr, vl l2
				for j := range f.m {
					var acc complex128
					for a := range n2 {
						var row complex128
						for b := range n2 {
							row += M[(n1+a)*n+n1+b] * complex(f.bq0[(n1+b)*f.m+j], 0)
						}
						sign := 1.0
						if a%2 == 1 {
							sign = sg
						}
						acc += complex(sign*dR[a], 0) * row
					}
					vr.addC(acc)
				}
				for i := range f.p {
					var acc complex128
					for b := range n1 {
						var col complex128
						for a := range n1 {
							col += complex(f.cz0[i*n+a], 0) * M[a*n+b]
						}
						sign := 1.0
						if b%2 == 1 {
							sign = sg
						}
						acc += col * complex(sign*dL[b], 0)
					}
					vl.addC(acc)
				}
				if r := vr.norm() / mf; !(r <= right*(1+1e-6)) {
					t.Errorf("s=%v: ‖Δ·M22·B̃2‖/‖M‖ = %.6g > right %.6g", x, r, right)
				}
				if r := vl.norm() / mf; !(r <= left*(1+1e-6)) {
					t.Errorf("s=%v: ‖C̃1·M11·Δ‖/‖M‖ = %.6g > left %.6g", x, r, left)
				}
			}
		}
	}
	if scored == 0 {
		t.Fatal("no point scored")
	}
}

// desc320Inverse is (sE − A)⁻¹ and its Frobenius norm in 320 bits.
func desc320Inverse(sys *System, s complex128) ([]complex128, float64, bool) {
	n, _, _ := sys.Dims()
	K := newHbMat(2*n, 2*n)
	for i := range n {
		for j := range n {
			e := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.E.At(i, j))
			re := new(big.Float).SetPrec(hbPrec).SetFloat64(real(s))
			re.Mul(re, e)
			re.Sub(re, new(big.Float).SetPrec(hbPrec).SetFloat64(sys.A.At(i, j)))
			im := new(big.Float).SetPrec(hbPrec).SetFloat64(imag(s))
			im.Mul(im, e)
			K.at(i, j).Set(re)
			K.at(n+i, n+j).Set(re)
			K.at(i, n+j).Neg(im)
			K.at(n+i, j).Set(im)
		}
	}
	id := newHbMat(2*n, 2*n)
	for i := range 2 * n {
		id.at(i, i).SetFloat64(1)
	}
	inv := hbSolve(K, id)
	out := make([]complex128, n*n)
	var acc l2
	for i := range n {
		for j := range n {
			re, _ := inv.at(i, j).Float64()
			im, _ := inv.at(n+i, j).Float64()
			out[i*n+j] = complex(re, im)
			acc.addC(out[i*n+j])
		}
	}
	v := acc.norm()
	return out, v, !math.IsInf(v, 0) && !math.IsNaN(v) && v > 0
}

// TestDescriptorRidgeOverflowStaysPlain: a decoupling ‖R‖ ≈ 1e200 makes
// I + RRᵀ overflow; that side's ridge bound falls back to the plain one.
func TestDescriptorRidgeOverflowStaysPlain(t *testing.T) {
	x := []float64{1e200, -3e199}
	if _, _, ok := ridgeFactor(x, 1, 2, true); ok {
		t.Fatal("ridgeFactor accepted an overflowing decoupling")
	}
	rb := newRidgeBounds(1, 2, 1, 1, x, x, []float64{1, 0.5, 0.25}, []float64{0.5, 1, 1})
	d := []float64{1e-16, 2e-16}
	if got, want := rb.right(d, 1), froAll(d)*rb.b2; got != want {
		t.Errorf("right = %v, want plain %v", got, want)
	}
	if got, want := rb.left([]float64{1e-16}, 1), rb.c1*1e-16; got != want {
		t.Errorf("left = %v, want plain %v", got, want)
	}
}

// TestDescriptorDecouplingCertificateBoundsExactRounding: on generalized
// Schur forms of random non-symmetric singular-E pencils the bounds the
// certificate charges for rounding hold against 320-bit values: the
// Sylvester residuals (componentwise, as the ridge tier uses them, so also
// normwise), the update roundings δb of B̃1 − L·B̃2 and δc of C̃2 + C̃1·R,
// the fold residuals F_A, F_B, and the form residual.
func TestDescriptorDecouplingCertificateBoundsExactRounding(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 23))
	checked := 0
	for trial := range 8 {
		n, m, p := 6, 2, 2
		A := mat.NewDense(n, n, nil)
		E := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				A.Set(i, j, rng.NormFloat64())
				if j < n-1 {
					E.Set(i, j, rng.NormFloat64())
				}
			}
		}
		B := mat.NewDense(n, m, nil)
		C := mat.NewDense(p, n, nil)
		for i := range n {
			for j := range m {
				B.Set(i, j, rng.NormFloat64())
			}
			for j := range p {
				C.Set(j, i, rng.NormFloat64())
			}
		}
		s, tt, q, z := make([]float64, n*n), make([]float64, n*n), make([]float64, n*n), make([]float64, n*n)
		ar, ai, be := make([]float64, n), make([]float64, n), make([]float64, n)
		selct := func(a, i, b float64) bool { return b != 0 && math.Abs(a/b) < 0.8 }
		n1, recip, ok := orderedQZ(n, A.RawMatrix(), E.RawMatrix(), selct, s, tt, q, z, ar, ai, be)
		if !ok || recip || n1 == 0 || n1 == n || s[n1*n+n1-1] != 0 {
			continue
		}
		n2 := n - n1
		ep := eps()
		bq := make([]float64, n*m)
		cz := make([]float64, p*n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: n, Stride: n, Data: q}, B.RawMatrix(), 0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bq})
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, C.RawMatrix(), blas64.General{Rows: n, Cols: n, Stride: n, Data: z}, 0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cz})
		r, l := make([]float64, n1*n2), make([]float64, n1*n2)
		for i := range n1 {
			for j := range n2 {
				r[i*n2+j], l[i*n2+j] = -s[i*n+n1+j], -tt[i*n+n1+j]
			}
		}
		scale, ok := qzSylvester(n, n1, false, s, tt, r, l)
		if !ok || scale == 0 {
			continue
		}
		for i := range r {
			r[i] /= scale
			l[i] /= scale
		}
		checked++
		half := 0.5 * float64(n+1) * ep
		for name, f := range map[string][]float64{"S": s, "T": tt} {
			res := sylvesterResidual(f, n, n1, r, l)
			x1 := absMulTight(stridedBlock(f, n, 0, 0, n1, n1), r, n1, n1, n2)
			x2 := absMulTight(l, stridedBlock(f, n, n1, n1, n2, n2), n1, n2, n2)
			ex := hbAdd(hbAdd(hbMul(hbSlice(f, n, 0, 0, n1, n1), hbVec(r, n1, n2)), hbMul(hbVec(l, n1, n2), hbSlice(f, n, n1, n1, n2, n2)), -1),
				hbSlice(f, n, 0, n1, n1, n2), 1)
			for i := range n1 {
				for j := range n2 {
					k := i*n2 + j
					e, _ := ex.at(i, j).Float64()
					if bound := math.Abs(res[k]) + half*(x1[k]+x2[k]+math.Abs(f[i*n+n1+j])); !(math.Abs(e) <= bound) {
						t.Errorf("trial %d %s residual [%d,%d]: exact %.3g > bound %.3g", trial, name, i, j, e, bound)
					}
				}
			}
		}
		bq0, cz0 := slices.Clone(bq), slices.Clone(cz)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: l},
			blas64.General{Rows: n2, Cols: m, Stride: m, Data: bq[n1*m:]}, 1, blas64.General{Rows: n1, Cols: m, Stride: m, Data: bq})
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: p, Cols: n1, Stride: n, Data: cz},
			blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: r}, 1, blas64.General{Rows: p, Cols: n2, Stride: n, Data: cz[n1:]})
		exB := hbAdd(hbSlice(bq0, m, 0, 0, n1, m), hbMul(hbVec(l, n1, n2), hbSlice(bq0, m, n1, 0, n2, m)), -1)
		lb := absMulTight(l, stridedBlock(bq0, m, n1, 0, n2, m), n1, n2, m)
		for i := range n1 {
			for j := range m {
				e, _ := exB.at(i, j).Float64()
				if d, bound := math.Abs(bq[i*m+j]-e), 0.5*float64(n2+1)*ep*(math.Abs(bq0[i*m+j])+lb[i*m+j]); !(d <= bound) {
					t.Errorf("trial %d δb[%d,%d] = %.3g > %.3g", trial, i, j, d, bound)
				}
			}
		}
		exC := hbAdd(hbSlice(cz0, n, 0, n1, p, n2), hbMul(hbSlice(cz0, n, 0, 0, p, n1), hbVec(r, n1, n2)), 1)
		cr := absMulTight(stridedBlock(cz0, n, 0, 0, p, n1), r, p, n1, n2)
		for i := range p {
			for j := range n2 {
				e, _ := exC.at(i, j).Float64()
				if d, bound := math.Abs(cz[i*n+n1+j]-e), 0.5*float64(n1+1)*ep*(math.Abs(cz0[i*n+n1+j])+cr[i*n2+j]); !(d <= bound) {
					t.Errorf("trial %d δc[%d,%d] = %.3g > %.3g", trial, i, j, d, bound)
				}
			}
		}
		a1 := extractModalBlock(s, n, 0, n1, 0, n1)
		b1f := extractModalBlock(bq, m, 0, n1, 0, m)
		t11 := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: n1, Stride: n, Data: tt}
		blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, a1.RawMatrix())
		blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, b1f.RawMatrix())
		for _, fc := range []struct {
			name string
			y    *mat.Dense
			w    []float64
			ldw  int
			c    int
		}{{"F_A", a1, s, n, n1}, {"F_B", b1f, bq, m, m}} {
			yr := fc.y.RawMatrix()
			comp := make([]float64, n1*fc.c)
			copyStrided(comp, fc.c, fc.w, fc.ldw, n1, fc.c)
			gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, tt), gen(n1, fc.c, yr.Stride, yr.Data), -1, gen(n1, fc.c, fc.c, comp))
			bound := froStrided(comp, fc.c, n1, fc.c) + 0.5*float64(n1+1)*ep*absProductFro(tt, n, yr.Data, yr.Stride, fc.w, fc.ldw, n1, n1, fc.c)
			ex := hbAdd(hbMul(hbSlice(tt, n, 0, 0, n1, n1), hbSlice(yr.Data, yr.Stride, 0, 0, n1, fc.c)), hbSlice(fc.w, fc.ldw, 0, 0, n1, fc.c), -1)
			if e := hbFro(ex); !(e <= bound) {
				t.Errorf("trial %d %s: exact %.3g > bound %.3g", trial, fc.name, e, bound)
			}
		}
		bar := newDescriptorBar(&System{A: A, E: E, B: B, C: C})
		for name, pair := range map[string]struct {
			x *mat.Dense
			f []float64
			n float64
		}{"A": {A, s, bar.a}, "E": {E, tt, bar.e}} {
			got := formResidual(pair.x.RawMatrix(), pair.f, q, z, n, n1, pair.n, bar.u()) * bar.u() * ep * pair.n
			ex := hbAdd(hbMul(hbMul(hbVec(q, n, n).t(), hbFrom(pair.x)), hbVec(z, n, n)), hbVec(pair.f, n, n), -1)
			for i := n1; i < n; i++ {
				for j := range n1 {
					ex.at(i, j).Set(new(big.Float).SetPrec(hbPrec).SetFloat64(0))
					v := new(big.Float).SetPrec(hbPrec)
					for a := range n {
						for b := range n {
							pr := new(big.Float).SetPrec(hbPrec).SetFloat64(q[a*n+i])
							pr.Mul(pr, new(big.Float).SetPrec(hbPrec).SetFloat64(pair.x.At(a, b)))
							pr.Mul(pr, new(big.Float).SetPrec(hbPrec).SetFloat64(z[b*n+j]))
							v.Add(v, pr)
						}
					}
					ex.at(i, j).Set(v)
				}
			}
			e := hbFro(ex)
			slack := 2 * float64(n+1) * ep * pair.n * float64(n)
			if !(math.Abs(e-got) <= slack) {
				t.Errorf("trial %d form %s: exact %.3g, measured %.3g", trial, name, e, got)
			}
		}
	}
	if checked < 3 {
		t.Fatalf("only %d of 8 trials split; the check is vacuous", checked)
	}
}

func hbSlice(d []float64, ld, r0, c0, r, c int) hbMat {
	m := newHbMat(r, c)
	for i := range r {
		for j := range c {
			m.at(i, j).SetFloat64(d[(r0+i)*ld+c0+j])
		}
	}
	return m
}

func hbVec(d []float64, r, c int) hbMat { return hbSlice(d, c, 0, 0, r, c) }

func hbFro(m hbMat) float64 {
	s := new(big.Float).SetPrec(hbPrec)
	p := new(big.Float).SetPrec(hbPrec)
	for _, v := range m.d {
		s.Add(s, p.Mul(v, v))
	}
	f, _ := s.Float64()
	return math.Sqrt(f)
}
