package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzStabilizable: (A, B) built in Kalman form from the decoded spectrum,
// A0 = [J_c A12; 0 J_u], B0 = [B1; 0], then A = P·Q·A0·Qᵀ·P⁻¹, B = P·Q·B0
// with Q a Givens sweep and P an exact power-of-two diagonal. J_u takes the
// last u blocks of the spectrum (v[12]), A12 is a texture scaled by
// 10^clamp(v[13],-3,3)·‖J‖, P = diag(2^(k_i)) with |k_i| ≤ clamp(|v[15]|,0,10).
// Oracle: stabilizable iff every J_u eigenvalue is stable and every unstable
// J_c eigenvalue λ has PBH margin σ_min[λI−J_c, B1] clear of rounding.
// IsDetectable(Aᵀ, Bᵀ) must agree.
func FuzzStabilizable(f *testing.F) {
	for _, s := range fzStabSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, fzCheckStabilizable)
}

func fzCheckStabilizable(t *testing.T, s *fzSpec) {
	n, m := s.N, s.M
	if n == 0 {
		fzSkip(t, "no states")
	}
	if !s.Decidable() {
		fzSkip(t, "eigenvalue near the stability boundary")
	}
	var starts []int
	for k := 0; k < n; {
		starts = append(starts, k)
		if imag(s.S[k]) != 0 || k+1 < n {
			k += 2
		} else {
			k++
		}
	}
	u := int(math.Mod(math.Abs(s.Vals[12])*2, float64(len(starts)+1)))
	nc := n
	if u > 0 {
		nc = starts[len(starts)-u]
	}
	nu := n - nc
	J := make([]float64, n*n)
	for k := 0; k < n; {
		e := s.Eig[k]
		if imag(s.S[k]) != 0 {
			J[k*n+k], J[k*n+k+1] = real(e), math.Abs(imag(e))
			J[(k+1)*n+k], J[(k+1)*n+k+1] = -math.Abs(imag(e)), real(e)
			k += 2
			continue
		}
		J[k*n+k] = real(e)
		k++
	}
	cpl := math.Pow(10, fzClamp(s.Vals[13], -3, 3)) * math.Max(1, s.NormJ)
	A0 := slicesCopy(J)
	for i := range nc {
		for j := nc; j < n; j++ {
			A0[i*n+j] = cpl * math.Sin(0.3+1.7*float64(i)+0.9*float64(j))
		}
	}
	B0 := make([]float64, n*m)
	copy(B0, s.B[:nc*m])

	stable := func(e complex128) bool {
		if s.Discrete {
			return cmplx.Abs(e) < 1
		}
		return real(e) < 0
	}
	want := true
	for _, e := range s.Eig[nc:] {
		if !stable(e) {
			want = false
		}
		for _, f := range s.Eig[:nc] {
			if cmplx.Abs(e-f) <= 1e-6*(1+cmplx.Abs(e)) {
				fzSkip(t, "controllable and uncontrollable modes coincide")
			}
		}
	}
	Q := fzMixer(n, s.Vals[14], 1, 0)
	QT := make([]float64, n*n)
	for i := range n {
		for j := range n {
			QT[j*n+i] = Q[i*n+j]
		}
	}
	A := fzMatMul(fzMatMul(Q, A0, n, n, n), QT, n, n, n)
	B := fzMatMul(Q, B0, n, n, m)
	kmax := int(fzClamp(math.Abs(s.Vals[15]), 0, 10))
	for i := range n {
		k := 0
		if kmax > 0 {
			k = (i*7+3)%(2*kmax+1) - kmax
		}
		for j := range n {
			A[i*n+j] = math.Ldexp(A[i*n+j], k)
			A[j*n+i] = math.Ldexp(A[j*n+i], -k)
		}
		for j := range m {
			B[i*m+j] = math.Ldexp(B[i*m+j], k)
		}
	}
	for _, x := range append(slicesCopy(A), B...) {
		if fzBadFloat(x) {
			fzSkip(t, "scaled entries out of range")
		}
	}
	band := float64(n) * math.Sqrt(0x1p-52) * fzFrob(A)
	for _, e := range s.Eig {
		if s.Margin(e) <= 4*band {
			fzSkip(t, "eigenvalue within the normwise boundary band of the scaled A")
		}
	}
	if want && nc > 0 {
		nrm := math.Max(fzFrob(A), fzFrob(B))
		for _, e := range s.Eig[:nc] {
			if stable(e) {
				continue
			}
			g := make([]complex128, n*(n+m))
			for i := range n {
				for j := range n {
					g[i*(n+m)+j] = -complex(A[i*n+j], 0)
				}
				g[i*(n+m)+i] += e
				for j := range m {
					g[i*(n+m)+n+j] = complex(B[i*m+j], 0)
				}
			}
			sv := fzSingularValues(g, n, n+m)
			if sv[len(sv)-1] <= 1e-6*math.Max(nrm, 1e-300) {
				fzSkip(t, "PBH margin of an unstable controllable mode near rounding")
			}
		}
	}
	got, err := IsStabilizable(mat.NewDense(n, n, A), mat.NewDense(n, m, B), !s.Discrete)
	fzErr(t, s, "IsStabilizable", err)
	if got != want {
		t.Fatalf("IsStabilizable = %v, want %v (nc=%d nu=%d coupling %g, scale 2^±%d)\nA=%v\nB=%v\n%s", got, want, nc, nu, cpl, kmax, A, B, s)
	}
	AT := make([]float64, n*n)
	for i := range n {
		for j := range n {
			AT[j*n+i] = A[i*n+j]
		}
	}
	BT := make([]float64, m*n)
	for i := range n {
		for j := range m {
			BT[j*n+i] = B[i*m+j]
		}
	}
	dg, err := IsDetectable(mat.NewDense(n, n, AT), mat.NewDense(m, n, BT), !s.Discrete)
	fzErr(t, s, "IsDetectable", err)
	if dg != want {
		t.Fatalf("IsDetectable(Aᵀ, Bᵀ) = %v, want %v (nc=%d)\n%s", dg, want, nc, s)
	}
}

func fzStabSeeds() []fzSeed {
	off := fzSeed{In: 0.6, Out: 0, IO: 0.3, Tau: 0, Shear: -10, Rot: 0.37, DScal: 1, Gain: 6, Phase: 0.8}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, 2.5 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp = 4, [3]float64{0, 0.5}, [3]float64{1.5, -1.5} }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.In = 4, [3]float64{0, 0.5}, [3]float64{-1.5, 1.5}, 0.6 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.Mag, s.Damp, s.Tau = 5, 2, [3]float64{0, 0.5, 1}, [3]float64{-2, 3, -1}, 6
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.Mag, s.Damp, s.In, s.Out = 3, [3]float64{0.3, -0.2}, [3]float64{1.2, -1}, 1, 1.5
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.Mag, s.Damp, s.In, s.Out, s.Tau = 6, 1, [3]float64{-1, 0, 1}, [3]float64{2, -2.5, 1.8}, 1.1, 2, 3
		}),
	}
}
