package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzHinfNorm: on decidable delay-free models HinfNorm is +Inf exactly when
// an intended pole is unstable; otherwise the norm is attained at the
// returned frequency, no frequency of a pole-clustered sweep
// (fzFreqGrid + fzSweepMax) exceeds it, both confirmed by the exact
// rational response, and the exact γ-crossing test certifies it from above
// (fzCheckPeak).
func FuzzHinfNorm(f *testing.F) {
	for _, s := range fzHinfNormSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		norm, omega, err := HinfNorm(sys)
		fzErr(t, s, "HinfNorm", err)
		fzDescriptorNorm(t, s, "HinfNorm", func(d *System) (float64, error) {
			v, _, err := HinfNorm(d)
			return v, err
		}, norm, err)
		if math.IsNaN(norm) || math.IsNaN(omega) || norm < 0 {
			t.Fatalf("HinfNorm = %v at %v\n%s", norm, omega, s)
		}
		if !s.Decidable() {
			if s.N > 0 && fzSurelyNotStable(s) {
				if !math.IsInf(norm, 1) {
					t.Fatalf("HinfNorm = %v, but the exact float64 A has a root on or beyond the boundary band\n%s", norm, s)
				}
				return
			}
			fzSkip(t, "verdict unresolvable: pole within error band of stability boundary")
		}
		if s.Unstable() > 0 {
			if !math.IsInf(norm, 1) {
				t.Fatalf("HinfNorm = %v for %d unstable poles, want +Inf\n%s", norm, s.Unstable(), s)
			}
			return
		}
		if math.IsInf(norm, 0) {
			t.Fatalf("HinfNorm = %v for a stable model\n%s", norm, s)
		}
		fzCheckPeak(t, s, "HinfNorm", norm, omega)
	})
}

// fzFreqPoint maps ω to s = jω, or z = e^{jω·Dt} for discrete models.
func fzFreqPoint(s *fzSpec, w float64) complex128 {
	if s.Discrete {
		return cmplx.Exp(complex(0, w*s.Dt))
	}
	return complex(0, w)
}

func fzComplexD(s *fzSpec) []complex128 {
	d := make([]complex128, len(s.D))
	for i, x := range s.D {
		d[i] = complex(x, 0)
	}
	return d
}

// fzSigmaMax is the largest singular value of the p×m complex g, via the
// real embedding [Re −Im; Im Re].
func fzSigmaMax(g []complex128, p, m int) float64 {
	e := mat.NewDense(2*p, 2*m, nil)
	for i := range p {
		for j := range m {
			re, im := real(g[i*m+j]), imag(g[i*m+j])
			e.Set(i, j, re)
			e.Set(i, m+j, -im)
			e.Set(p+i, j, im)
			e.Set(p+i, m+j, re)
		}
	}
	var svd mat.SVD
	if !svd.Factorize(e, mat.SVDNone) {
		return math.NaN()
	}
	return svd.Values(nil)[0]
}

// fzResponseFloat returns C(sI − A)⁻¹B + D by complex Gaussian elimination
// with partial pivoting; only used to locate peaks, which are then
// confirmed with fzExactResponse.
func fzResponseFloat(s *fzSpec, x complex128) []complex128 {
	n, m, p := s.N, s.M, s.P
	cols := n + m
	M := make([]complex128, n*cols)
	for i := range n {
		for j := range n {
			M[i*cols+j] = complex(-s.A[i*n+j], 0)
		}
		M[i*cols+i] += x
		for j := range m {
			M[i*cols+n+j] = complex(s.B[i*m+j], 0)
		}
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(M[i*cols+k]) > cmplx.Abs(M[piv*cols+k]) {
				piv = i
			}
		}
		for j := range cols {
			M[k*cols+j], M[piv*cols+j] = M[piv*cols+j], M[k*cols+j]
		}
		if M[k*cols+k] == 0 {
			return fzComplexD(s)
		}
		for i := k + 1; i < n; i++ {
			f := M[i*cols+k] / M[k*cols+k]
			for j := k; j < cols; j++ {
				M[i*cols+j] -= f * M[k*cols+j]
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		for j := n; j < cols; j++ {
			v := M[k*cols+j]
			for l := k + 1; l < n; l++ {
				v -= M[k*cols+l] * M[l*cols+j]
			}
			M[k*cols+j] = v / M[k*cols+k]
		}
	}
	g := fzComplexD(s)
	for i := range p {
		for j := range m {
			for k := range n {
				g[i*m+j] += complex(s.C[i*n+k], 0) * M[k*cols+n+j]
			}
		}
	}
	return g
}

func fzHinfNormSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzIsStableSeeds(),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 2, 0, 4 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.DScal = 2, 1, 5.5, 0.3 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 6, 3, 3
			s.Mag, s.Damp = [3]float64{-1, 0.3, 2}, [3]float64{3, 5, 0.5}
			s.Shear, s.Rot, s.Phase, s.Gain, s.DScal = -1, 0.8, 0.4, 7, 0.2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.1)
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0.5, 1}, [3]float64{4.5, 2}
			s.Rot, s.Phase = -1.3, 0.9
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, 0
			s.N, s.M, s.P = 3, 1, 2
			s.Mag, s.Damp = [3]float64{0, -0.3}, [3]float64{3, 1}
			s.DScal = -2
		}),
	)
}
