package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzCheckSimilar checks xc = T·x: T·A ≈ Ac·T, Bc ≈ T·B, Cc·T ≈ C within
// a backward bound scaled by cond(T), and D unchanged; it returns cond(T).
func fzCheckSimilar(t *testing.T, s *fzSpec, op string, T *mat.Dense, got *System) float64 {
	t.Helper()
	const eps = 0x1p-52
	n, m, p := s.N, s.M, s.P
	A, B, C := mat.NewDense(n, n, s.A), mat.NewDense(n, m, s.B), mat.NewDense(p, n, s.C)
	condT := mat.Cond(T, 2)
	if math.IsInf(condT, 0) || math.IsNaN(condT) {
		t.Fatalf("%s: singular T\n%s", op, s)
	}
	nT := mat.Norm(T, 2)
	tol := 1e3 * float64(n) * eps * condT
	resid := func(name string, x, y mat.Matrix, scale float64) {
		var d mat.Dense
		d.Sub(x, y)
		if r := mat.Norm(&d, 2); r > tol*scale {
			t.Fatalf("%s: ‖%s‖ = %.3g > %.3g (cond(T) %.3g)\n%s", op, name, r, tol*scale, condT, s)
		}
	}
	var ta, at, tb, ct mat.Dense
	ta.Mul(T, A)
	at.Mul(got.A, T)
	resid("T·A − Ac·T", &ta, &at, nT*(mat.Norm(A, 2)+mat.Norm(got.A, 2)))
	tb.Mul(T, B)
	resid("Bc − T·B", got.B, &tb, nT*mat.Norm(B, 2))
	ct.Mul(got.C, T)
	resid("Cc·T − C", &ct, C, mat.Norm(got.C, 2)*nT+mat.Norm(C, 2))
	if !slices.Equal(fzDenseData(got.D), s.D) {
		t.Fatalf("%s: D changed\n%s", op, s)
	}
	return condT
}

// fzQuasiBlocks splits a quasi-triangular Ac into 1×1/2×2 diagonal blocks,
// failing when entries below the block subdiagonal exceed tol or 2×2
// blocks overlap, and returns the block eigenvalues in order.
func fzQuasiBlocks(ac []float64, n int, tol float64) ([]complex128, string) {
	var ev []complex128
	for i := 0; i < n; {
		for j := 0; j+1 < i; j++ {
			for r := i; r < n; r++ {
				if math.Abs(ac[r*n+j]) > tol {
					return nil, fmt.Sprintf("entry [%d,%d] = %v below the quasi-triangle", r, j, ac[r*n+j])
				}
			}
		}
		if i+1 < n && math.Abs(ac[(i+1)*n+i]) > tol {
			if i+2 < n && math.Abs(ac[(i+2)*n+i+1]) > tol {
				return nil, fmt.Sprintf("overlapping 2×2 blocks at %d", i)
			}
			for r := i + 2; r < n; r++ {
				if math.Abs(ac[r*n+i]) > tol || math.Abs(ac[r*n+i+1]) > tol {
					return nil, fmt.Sprintf("entry below 2×2 block at column %d", i)
				}
			}
			a, b, c, d := ac[i*n+i], ac[i*n+i+1], ac[(i+1)*n+i], ac[(i+1)*n+i+1]
			h := (a - d) / 2
			disc := cmplx.Sqrt(complex(h*h+b*c, 0))
			mid := complex((a+d)/2, 0)
			ev = append(ev, mid+disc, mid-disc)
			i += 2
			continue
		}
		for r := i + 1; r < n; r++ {
			if math.Abs(ac[r*n+i]) > tol {
				return nil, fmt.Sprintf("entry [%d,%d] = %v below a 1×1 block", r, i, ac[r*n+i])
			}
		}
		ev = append(ev, complex(ac[i*n+i], 0))
		i++
	}
	return ev, ""
}

// FuzzCanon: modal form is a similarity (fzCheckSimilar) into a
// quasi-triangular Ac whose diagonal blocks carry the intended eigenvalues
// (Bauer–Fike radius) in ascending magnitude; companion form (when the
// Krylov matrix of the first input is well conditioned) has ones on the
// subdiagonal, zeros elsewhere outside the last column, −(exact char poly
// coefficients) in the last column and Bc[:,0] = e₁.
func FuzzCanon(f *testing.F) {
	for _, s := range fzBalredSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		const eps = 0x1p-52
		sys := s.System(t, fzSysOpts{})
		n := s.N
		res, err := Canon(sys, CanonModal)
		fzContract(t, s, "Canon modal", res != nil, err)
		if n == 0 {
			fzErr(t, s, "Canon", err, ErrDimensionMismatch)
		}
		fzErr(t, s, "Canon modal", err)
		condT := fzCheckSimilar(t, s, "Canon modal", res.T, res.Sys)
		ac := fzDenseData(res.Sys.A)
		normA := fzFrob(s.A) + fzFrob(ac)
		tol := 1e4 * float64(n) * eps * condT * normA
		ev, msg := fzQuasiBlocks(ac, n, tol)
		if msg != "" {
			t.Fatalf("Canon modal: %s (tol %.3g)\nAc=%v\n%s", msg, tol, ac, s)
		}
		r := s.EigErr() + 10*tol*condT
		if _, _, msg := fzMatchEig(ev, s.Eig, r); msg != "" {
			t.Fatalf("Canon modal block eigenvalues %v: %s\n%s", ev, msg, s)
		}
		for i := 1; i < len(ev); i++ {
			if cmplx.Abs(ev[i]) < cmplx.Abs(ev[i-1])-2*r {
				t.Fatalf("Canon modal: block eigenvalues %v not in ascending magnitude\n%s", ev, s)
			}
		}

		K := mat.NewDense(n, n, nil)
		col := make([]float64, n)
		for i := range n {
			col[i] = s.B[i*s.M]
		}
		for j := range n {
			K.SetCol(j, col)
			nx := make([]float64, n)
			for i := range n {
				for l := range n {
					nx[i] += s.A[i*n+l] * col[l]
				}
			}
			col = nx
		}
		kc := mat.Cond(K, 2)
		cr, err := Canon(sys, CanonCompanion)
		fzContract(t, s, "Canon companion", cr != nil, err)
		if errors.Is(err, ErrSingularTransform) && kc > 1e4 {
			fzSkip(t, "Canon companion: Krylov matrix of input 1 ill-conditioned")
		}
		fzErr(t, s, "Canon companion", err)
		if kc > 1e10 {
			fzSkip(t, "companion coefficients ill-conditioned")
		}
		fzCheckSimilar(t, s, "Canon companion", cr.T, cr.Sys)
		cp := fzCharPolyExact(s.A, n)
		coef := make([]float64, n)
		for k := range n {
			coef[k], _ = cp[k+1].Float64()
		}
		cn := math.Max(fzFrob(coef), 1)
		ctol := 1e3 * float64(n) * eps * kc * cn
		a := fzDenseData(cr.Sys.A)
		for i := range n {
			for j := range n - 1 {
				want := 0.0
				if i == j+1 {
					want = 1
				}
				if math.Abs(a[i*n+j]-want) > ctol {
					t.Fatalf("Canon companion A[%d,%d] = %v, want %v (tol %.3g)\n%s", i, j, a[i*n+j], want, ctol, s)
				}
			}
			if want := -coef[n-1-i]; math.Abs(a[i*n+n-1]-want) > ctol {
				t.Fatalf("Canon companion A[%d,%d] = %v, want −a_%d = %v (tol %.3g)\n%s", i, n-1, a[i*n+n-1], n-i, want, ctol, s)
			}
			want := 0.0
			if i == 0 {
				want = 1
			}
			if g := cr.Sys.B.At(i, 0); math.Abs(g-want) > 1e3*float64(n)*eps*kc {
				t.Fatalf("Canon companion B[%d,0] = %v, want %v\n%s", i, g, want, s)
			}
		}
	})
}
