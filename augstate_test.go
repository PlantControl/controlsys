package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"
)

func TestAugstateDelayedMatchesStateOracle(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		for _, descriptor := range []bool{false, true} {
			for _, delays := range []string{"none", "io", "residual", "internal"} {
				label := fmt.Sprintf("dt=%g/desc=%v/%s", dt, descriptor, delays)
				sys := absorbScopePlant(t, dt, delays == "internal", descriptor)
				scale := 1.0
				if dt == 0 {
					scale = 0.125
				}
				switch delays {
				case "io", "internal":
					absorbScopeCases[0].apply(t, sys, scale)
				case "residual":
					absorbScopeCases[1].apply(t, sys, scale)
				}
				aug, err := Augstate(sys)
				if err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				if err := aug.Validate(); err != nil {
					t.Fatalf("%s: Augstate result invalid: %v", label, err)
				}
				assertAugstateResponse(t, label, sys, aug)
			}
		}
	}
}

func assertAugstateResponse(t *testing.T, label string, sys, aug *System) {
	t.Helper()
	n, m, p := sys.Dims()
	if _, ma, pa := aug.Dims(); ma != m || pa != p+n {
		t.Fatalf("%s: dims m=%d p=%d, want m=%d p=%d", label, ma, pa, m, p+n)
	}
	want, err := sys.FreqResponsePointwise(absorbScopeOmega)
	if err != nil {
		t.Fatalf("%s: reference FreqResponse: %v", label, err)
	}
	got, err := aug.FreqResponsePointwise(absorbScopeOmega)
	if err != nil {
		t.Fatalf("%s: Augstate FreqResponse: %v", label, err)
	}
	for k, w := range absorbScopeOmega {
		for j := range m {
			for i := range p {
				if d := cmplx.Abs(got.At(k, i, j) - want.At(k, i, j)); d > absorbScopeTol {
					t.Fatalf("%s: ω=%g y(%d,%d) differs by %g", label, w, i, j, d)
				}
			}
		}
		x, _ := lftSignalOracle(sys, w)
		for r := range n {
			for j := range m {
				if d := cmplx.Abs(got.At(k, p+r, j) - x[r][j]); d > absorbScopeTol {
					t.Fatalf("%s: ω=%g x(%d,%d) = %v, want %v", label, w, r, j, got.At(k, p+r, j), x[r][j])
				}
			}
		}
	}
}

// lftSignalOracle solves (λE−A)x = Bũ + B2·w, w = Θz, z = C2x + D21ũ + D22w,
// ũ = diag(delay(InputDelay))u, by dense complex elimination; returns x/u, z/u.
func lftSignalOracle(sys *System, w float64) (x, z [][]complex128) {
	n, m, _ := sys.Dims()
	lambda := complex(0, w)
	delayFactor := func(tau float64) complex128 { return cmplx.Exp(complex(0, -w*tau)) }
	if sys.IsDiscrete() {
		lambda = cmplx.Exp(complex(0, w*sys.Dt))
		delayFactor = func(tau float64) complex128 { return cmplx.Exp(complex(0, -w*sys.Dt*tau)) }
	}
	N := sys.internalDelayCount()
	sz := n + N
	M := make([][]complex128, sz)
	rhs := make([][]complex128, sz)
	for i := range sz {
		M[i] = make([]complex128, sz)
		rhs[i] = make([]complex128, m)
	}
	for i := range n {
		for c := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, c)
			} else if i == c {
				e = 1
			}
			M[i][c] = lambda*complex(e, 0) - complex(sys.A.At(i, c), 0)
		}
		for j := range m {
			rhs[i][j] = complex(sys.B.At(i, j), 0)
		}
		for k := range N {
			M[i][n+k] = complex(-sys.LFT.B2.At(i, k), 0)
		}
	}
	for k := range N {
		th := delayFactor(sys.LFT.Tau[k])
		for c := range n {
			M[n+k][c] = -th * complex(sys.LFT.C2.At(k, c), 0)
		}
		for l := range N {
			M[n+k][n+l] = -th * complex(sys.LFT.D22.At(k, l), 0)
		}
		M[n+k][n+k] += 1
		for j := range m {
			rhs[n+k][j] = th * complex(sys.LFT.D21.At(k, j), 0)
		}
	}
	for col := range sz {
		piv := col
		for r := col + 1; r < sz; r++ {
			if cmplx.Abs(M[r][col]) > cmplx.Abs(M[piv][col]) {
				piv = r
			}
		}
		M[col], M[piv] = M[piv], M[col]
		rhs[col], rhs[piv] = rhs[piv], rhs[col]
		for r := col + 1; r < sz; r++ {
			f := M[r][col] / M[col][col]
			for c := col; c < sz; c++ {
				M[r][c] -= f * M[col][c]
			}
			for j := range m {
				rhs[r][j] -= f * rhs[col][j]
			}
		}
	}
	for r := sz - 1; r >= 0; r-- {
		for j := range m {
			s := rhs[r][j]
			for c := r + 1; c < sz; c++ {
				s -= M[r][c] * rhs[c][j]
			}
			rhs[r][j] = s / M[r][r]
		}
	}
	for j := range m {
		if sys.InputDelay == nil || sys.InputDelay[j] == 0 {
			continue
		}
		f := delayFactor(sys.InputDelay[j])
		for r := range sz {
			rhs[r][j] *= f
		}
	}
	for k := range N {
		th := delayFactor(sys.LFT.Tau[k])
		for j := range m {
			rhs[n+k][j] /= th
		}
	}
	return rhs[:n], rhs[n:]
}

func TestAugstatePadsOutputSideDelays(t *testing.T) {
	sys := absorbScopePlant(t, 0, true, false)
	absorbScopeCases[1].apply(t, sys, 0.125)
	aug, err := Augstate(sys)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	for i := p; i < p+n; i++ {
		if aug.OutputDelay[i] != 0 {
			t.Fatalf("state output %d OutputDelay = %g, want 0", i-p, aug.OutputDelay[i])
		}
		for j := range m {
			if aug.Delay.At(i, j) != 0 {
				t.Fatalf("state output %d IODelay[%d] = %g, want 0", i-p, j, aug.Delay.At(i, j))
			}
		}
		for k := range sys.internalDelayCount() {
			if aug.LFT.D12.At(i, k) != 0 {
				t.Fatalf("state output %d D12[%d] = %g, want 0", i-p, k, aug.LFT.D12.At(i, k))
			}
		}
	}
	for j := range m {
		if aug.InputDelay[j] != sys.InputDelay[j] {
			t.Fatalf("InputDelay[%d] = %g, want %g", j, aug.InputDelay[j], sys.InputDelay[j])
		}
	}
}

func TestAugmentInternalDelayOutputsMatchesOracle(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		for _, delays := range []string{"none", "io", "residual"} {
			label := fmt.Sprintf("dt=%g/%s", dt, delays)
			sys := absorbScopePlant(t, dt, true, false)
			scale := 1.0
			if dt == 0 {
				scale = 0.125
			}
			switch delays {
			case "io":
				absorbScopeCases[0].apply(t, sys, scale)
			case "residual":
				absorbScopeCases[1].apply(t, sys, scale)
			}
			aug, err := sys.AugmentInternalDelayOutputs("z")
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			if err := aug.Validate(); err != nil {
				t.Fatalf("%s: result invalid: %v", label, err)
			}
			_, m, p := sys.Dims()
			want, err := sys.FreqResponsePointwise(absorbScopeOmega)
			if err != nil {
				t.Fatalf("%s: reference FreqResponse: %v", label, err)
			}
			got, err := aug.FreqResponsePointwise(absorbScopeOmega)
			if err != nil {
				t.Fatalf("%s: FreqResponse: %v", label, err)
			}
			for k, w := range absorbScopeOmega {
				_, z := lftSignalOracle(sys, w)
				for j := range m {
					for i := range p {
						if d := cmplx.Abs(got.At(k, i, j) - want.At(k, i, j)); d > absorbScopeTol {
							t.Fatalf("%s: ω=%g y(%d,%d) differs by %g", label, w, i, j, d)
						}
					}
					for r := range z {
						if d := cmplx.Abs(got.At(k, p+r, j) - z[r][j]); d > absorbScopeTol {
							t.Fatalf("%s: ω=%g z(%d,%d) = %v, want %v", label, w, r, j, got.At(k, p+r, j), z[r][j])
						}
					}
				}
			}
		}
	}
}
