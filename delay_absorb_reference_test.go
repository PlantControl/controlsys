package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// delayClass names which delay a factor is evaluated for.
type delayClass int

const (
	delayIn delayClass = iota
	delayOut
	delayIO
	delayInternal
)

// delayFactorFunc returns the frequency factor of delay tau of class c at
// point s (Laplace variable, or z for discrete systems).
type delayFactorFunc func(c delayClass, tau float64, s complex128) complex128

func exactDelayFactor(dt float64) delayFactorFunc {
	return func(_ delayClass, tau float64, s complex128) complex128 {
		if dt > 0 {
			return cmplx.Pow(s, complex(-tau, 0))
		}
		return cmplx.Exp(-s * complex(tau, 0))
	}
}

// padeFactor evaluates the [order/order] Padé approximant of exp(-s*tau)
// from the closed-form coefficients q_k = (2n-k)! n! / ((2n)! k! (n-k)!).
func padeFactor(tau float64, order int, s complex128) complex128 {
	if tau == 0 {
		return 1
	}
	fact := func(k int) float64 { return math.Gamma(float64(k) + 1) }
	x := s * complex(tau, 0)
	var num, den complex128
	for k := 0; k <= order; k++ {
		q := complex(fact(2*order-k)*fact(order)/(fact(2*order)*fact(k)*fact(order-k)), 0)
		xk := cmplx.Pow(x, complex(float64(k), 0))
		den += q * xk
		if k%2 == 1 {
			num -= q * xk
		} else {
			num += q * xk
		}
	}
	return num / den
}

// csolve solves a*x = b by Gaussian elimination with partial pivoting.
func csolve(t *testing.T, a, b [][]complex128) [][]complex128 {
	t.Helper()
	n := len(a)
	A := make([][]complex128, n)
	X := make([][]complex128, n)
	for i := range n {
		A[i] = slices.Clone(a[i])
		X[i] = slices.Clone(b[i])
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(A[i][k]) > cmplx.Abs(A[piv][k]) {
				piv = i
			}
		}
		if A[piv][k] == 0 {
			t.Fatal("csolve: singular")
		}
		A[k], A[piv] = A[piv], A[k]
		X[k], X[piv] = X[piv], X[k]
		for i := k + 1; i < n; i++ {
			f := A[i][k] / A[k][k]
			for j := k; j < n; j++ {
				A[i][j] -= f * A[k][j]
			}
			for j := range X[i] {
				X[i][j] -= f * X[k][j]
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		for j := range X[k] {
			for i := k + 1; i < n; i++ {
				X[k][j] -= A[k][i] * X[i][j]
			}
			X[k][j] /= A[k][k]
		}
	}
	return X
}

func cmatOf(m *mat.Dense, r, c int) [][]complex128 {
	out := make([][]complex128, r)
	for i := range r {
		out[i] = make([]complex128, c)
		if m == nil || m.IsEmpty() {
			continue
		}
		for j := range c {
			out[i][j] = complex(m.At(i, j), 0)
		}
	}
	return out
}

func cmul(a, b [][]complex128) [][]complex128 {
	out := make([][]complex128, len(a))
	for i := range a {
		out[i] = make([]complex128, len(b[0]))
		for k := range b {
			for j := range b[0] {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

// evalDelaySystem evaluates sys at s directly from its matrices: the
// descriptor resolvent of [B B2] and [C; C2], the lower LFT over internal
// delays, then output, IO and input delay factors.
func evalDelaySystem(t *testing.T, sys *System, s complex128, f delayFactorFunc) [][]complex128 {
	t.Helper()
	n, m, p := sys.Dims()
	var tau []float64
	var B2, C2, D12, D21, D22 *mat.Dense
	if sys.LFT != nil {
		tau = sys.LFT.Tau
		B2, C2, D12, D21, D22 = sys.LFT.B2, sys.LFT.C2, sys.LFT.D12, sys.LFT.D21, sys.LFT.D22
	}
	N := len(tau)
	H := cmatOf(nil, p+N, m+N)
	if n > 0 {
		pencil := cmatOf(sys.A, n, n)
		for i := range n {
			for j := range n {
				e := 0.0
				if sys.E != nil {
					e = sys.E.At(i, j)
				} else if i == j {
					e = 1
				}
				pencil[i][j] = s*complex(e, 0) - pencil[i][j]
			}
		}
		rhs := cmatOf(nil, n, m+N)
		for i := range n {
			for j := range m {
				rhs[i][j] = complex(sys.B.At(i, j), 0)
			}
			for j := range N {
				rhs[i][m+j] = complex(B2.At(i, j), 0)
			}
		}
		lhs := cmatOf(nil, p+N, n)
		for j := range n {
			for i := range p {
				lhs[i][j] = complex(sys.C.At(i, j), 0)
			}
			for i := range N {
				lhs[p+i][j] = complex(C2.At(i, j), 0)
			}
		}
		H = cmul(lhs, csolve(t, pencil, rhs))
	}
	for i := range p + N {
		for j := range m + N {
			var d float64
			switch {
			case i < p && j < m:
				d = sys.D.At(i, j)
			case i < p:
				d = D12.At(i, j-m)
			case j < m:
				d = D21.At(i-p, j)
			default:
				d = D22.At(i-p, j-m)
			}
			H[i][j] += complex(d, 0)
		}
	}
	G := make([][]complex128, p)
	for i := range p {
		G[i] = slices.Clone(H[i][:m])
	}
	if N > 0 {
		delta := make([]complex128, N)
		for k := range N {
			delta[k] = f(delayInternal, tau[k], s)
		}
		loop := cmatOf(nil, N, N)
		h21 := cmatOf(nil, N, m)
		for i := range N {
			for j := range N {
				loop[i][j] = -H[p+i][m+j] * delta[j]
			}
			loop[i][i] += 1
			copy(h21[i], H[p+i][:m])
		}
		w := csolve(t, loop, h21)
		for i := range p {
			for k := range N {
				for j := range m {
					G[i][j] += H[i][m+k] * delta[k] * w[k][j]
				}
			}
		}
	}
	for i := range p {
		for j := range m {
			if sys.Delay != nil {
				G[i][j] *= f(delayIO, sys.Delay.At(i, j), s)
			}
			if sys.OutputDelay != nil {
				G[i][j] *= f(delayOut, sys.OutputDelay[i], s)
			}
			if sys.InputDelay != nil {
				G[i][j] *= f(delayIn, sys.InputDelay[j], s)
			}
		}
	}
	return G
}

func scopeAbsorbs(scope AbsorbScope, c delayClass) bool {
	switch scope {
	case AbsorbAll:
		return true
	case AbsorbInternal:
		return c == delayInternal
	case AbsorbIO:
		return c != delayInternal
	case AbsorbInput:
		return c == delayIn
	case AbsorbOutput:
		return c == delayOut
	}
	return false
}

// combinedPadeSystem moves a decomposable IO delay matrix onto the input and
// output delays, matching continuous IO absorption, which approximates each
// channel's total delay with one Padé bank.
func combinedPadeSystem(sys *System, in, out []float64) *System {
	cp := sys.Copy()
	cp.Delay = nil
	for j := range cp.InputDelay {
		cp.InputDelay[j] += in[j]
	}
	for i := range cp.OutputDelay {
		cp.OutputDelay[i] += out[i]
	}
	return cp
}

func assertScopeMatchesReference(t *testing.T, label string, scope AbsorbScope, ref, got *System) {
	t.Helper()
	exact := exactDelayFactor(ref.Dt)
	approx := func(c delayClass, tau float64, s complex128) complex128 {
		if ref.Dt == 0 && scopeAbsorbs(scope, c) {
			return padeFactor(tau, DefaultPadeOrder, s)
		}
		return exact(c, tau, s)
	}
	for _, w := range absorbScopeOmega {
		s := complex(0, w)
		if ref.Dt > 0 {
			s = cmplx.Exp(complex(0, w*ref.Dt))
		}
		want := evalDelaySystem(t, ref, s, approx)
		have := evalDelaySystem(t, got, s, exact)
		for i := range want {
			for j := range want[i] {
				if d := cmplx.Abs(want[i][j] - have[i][j]); d > absorbScopeTol {
					t.Fatalf("%s: w=%g (%d,%d) differs by %g", label, w, i, j, d)
				}
			}
		}
	}
}

func assertScopeStructure(t *testing.T, label string, scope AbsorbScope, src, got *System) {
	t.Helper()
	keeps := func(c delayClass) bool { return !scopeAbsorbs(scope, c) }
	if keeps(delayIn) && !slices.Equal(got.InputDelay, src.InputDelay) {
		t.Fatalf("%s: InputDelay %v, want %v", label, got.InputDelay, src.InputDelay)
	}
	if keeps(delayOut) && !slices.Equal(got.OutputDelay, src.OutputDelay) {
		t.Fatalf("%s: OutputDelay %v, want %v", label, got.OutputDelay, src.OutputDelay)
	}
	if keeps(delayIn) && keeps(delayOut) && keeps(delayIO) && !mat.Equal(ioDelayOrZero(got.Delay, 2, 2), ioDelayOrZero(src.Delay, 2, 2)) {
		t.Fatalf("%s: Delay changed", label)
	}
	if !keeps(delayIn) && delaySliceHasNonzero(got.InputDelay) {
		t.Fatalf("%s: InputDelay %v left", label, got.InputDelay)
	}
	if !keeps(delayOut) && delaySliceHasNonzero(got.OutputDelay) {
		t.Fatalf("%s: OutputDelay %v left", label, got.OutputDelay)
	}
	if !keeps(delayIO) && delayMatrixHasNonzero(got.Delay) {
		t.Fatalf("%s: Delay left", label)
	}
	if keeps(delayInternal) != got.HasInternalDelay() {
		t.Fatalf("%s: internal delay kept=%v", label, got.HasInternalDelay())
	}
	if src.IsDescriptor() != got.IsDescriptor() {
		t.Fatalf("%s: descriptor=%v", label, got.IsDescriptor())
	}
	if scope == AbsorbInternal {
		n, _, _ := src.Dims()
		ng, _, _ := got.Dims()
		want := n + DefaultPadeOrder*len(src.LFT.Tau)
		if src.IsDiscrete() {
			want = n
			for _, v := range src.LFT.Tau {
				want += int(v)
			}
		}
		if ng != want {
			t.Fatalf("%s: %d states, want %d", label, ng, want)
		}
	}
}

func TestAbsorbDelayEveryScopeMatchesReference(t *testing.T) {
	scopes := []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbInternal, AbsorbAll}
	for _, dt := range []float64{0, 1} {
		for _, descriptor := range []bool{false, true} {
			for _, c := range absorbScopeCases {
				for _, scope := range scopes {
					if dt == 0 && c.resid && (scope == AbsorbIO || scope == AbsorbAll) {
						continue
					}
					label := c.name + "/" + string(scope)
					if dt == 0 {
						label = "continuous/" + label
					}
					if descriptor {
						label = "descriptor/" + label
					}
					scale := 1.0
					if dt == 0 {
						scale = 0.1
					}
					sys := absorbScopePlant(t, dt, true, descriptor)
					c.apply(t, sys, scale)
					got, err := sys.AbsorbDelay(scope)
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					assertScopeStructure(t, label, scope, sys, got)
					ref := sys
					if dt == 0 && (scope == AbsorbIO || scope == AbsorbAll) {
						ref = combinedPadeSystem(sys, []float64{0.1, 0.2}, []float64{0, 0.2})
					}
					assertScopeMatchesReference(t, label, scope, ref, got)
				}
			}
		}
	}
}

func TestAbsorbInternalContinuousKeepsIODelaysOnce(t *testing.T) {
	sys := absorbScopePlant(t, 0, true, false)
	if err := sys.SetInputDelay([]float64{0.25, 0}); err != nil {
		t.Fatal(err)
	}
	got, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := got.Dims(); n != 3+2*DefaultPadeOrder {
		t.Fatalf("%d states, want %d", n, 3+2*DefaultPadeOrder)
	}
	assertScopeStructure(t, "input", AbsorbInternal, sys, got)
	assertScopeMatchesReference(t, "input", AbsorbInternal, sys, got)
}

func TestDecomposeIODelaySnapsDecimalRoundoff(t *testing.T) {
	vals := []float64{0, 0.1, 0.3, 0.7}
	for _, i0 := range vals {
		for _, i1 := range vals {
			for _, o0 := range vals {
				for _, o1 := range vals {
					io := mat.NewDense(2, 2, []float64{o0 + i0, o0 + i1, o1 + i0, o1 + i1})
					in, out, residual := decomposeIODelay(io)
					if delayMatrixHasNonzero(residual) {
						t.Fatalf("in=[%g %g] out=[%g %g]: residual %v", i0, i1, o0, o1, mat.Formatted(residual))
					}
					for i := range 2 {
						for j := range 2 {
							if d := math.Abs(out[i] + in[j] - io.At(i, j)); d > 4*0x1p-52 {
								t.Fatalf("in=[%g %g] out=[%g %g]: (%d,%d) off by %g", i0, i1, o0, o1, i, j, d)
							}
						}
					}
				}
			}
		}
	}
	_, out, residual := decomposeIODelay(mat.NewDense(2, 2, []float64{0.1, 0.2, 0.1 * 3, 0.4}))
	if delayMatrixHasNonzero(residual) || out[0] != 0 {
		t.Fatalf("0.1*[1 2;3 4]: out=%v residual=%v", out, mat.Formatted(residual))
	}
	_, _, residual = decomposeIODelay(mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.3}))
	if !delayMatrixHasNonzero(residual) {
		t.Fatal("genuine residual snapped")
	}
}

func TestPadeDecimalDelaysHaveNoRoundoffStates(t *testing.T) {
	const order = 3
	for _, descriptor := range []bool{false, true} {
		P := absorbScopePlant(t, 0, false, descriptor)
		absorbScopeCases[0].apply(t, P, 0.1)
		lft, err := P.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(lft.LFT.Tau, []float64{0.2, 0.2, 0.4}) {
			t.Fatalf("descriptor=%v: taus %v, want [0.2 0.2 0.4]", descriptor, lft.LFT.Tau)
		}
		got, err := P.Pade(order)
		if err != nil {
			t.Fatal(err)
		}
		if n, _, _ := got.Dims(); n != 3+3*order {
			t.Fatalf("descriptor=%v: %d states, want %d", descriptor, n, 3+3*order)
		}
		if mx := mat.Norm(got.A, math.Inf(1)); mx > 1e6 {
			t.Fatalf("descriptor=%v: |A|inf = %g", descriptor, mx)
		}
		ref := combinedPadeSystem(P, []float64{0.1, 0.2}, []float64{0, 0.2})
		pade := func(_ delayClass, tau float64, s complex128) complex128 { return padeFactor(tau, order, s) }
		exact := exactDelayFactor(0)
		for _, w := range absorbScopeOmega {
			s := complex(0, w)
			want := evalDelaySystem(t, ref, s, pade)
			have := evalDelaySystem(t, got, s, exact)
			for i := range want {
				for j := range want[i] {
					if d := cmplx.Abs(want[i][j] - have[i][j]); d > absorbScopeTol {
						t.Fatalf("descriptor=%v w=%g (%d,%d) differs by %g", descriptor, w, i, j, d)
					}
				}
			}
		}
	}
}
