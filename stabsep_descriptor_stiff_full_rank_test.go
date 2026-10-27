package controlsys

import (
	"cmp"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func sortedPolesByModulus(p []complex128) []complex128 {
	p = slices.Clone(p)
	slices.SortFunc(p, func(a, b complex128) int {
		return cmp.Or(cmp.Compare(cmplx.Abs(a), cmplx.Abs(b)), cmp.Compare(imag(a), imag(b)))
	})
	return p
}

// pencilBackwardError is σ_min(pE − A)/(|p|·‖E‖₂ + ‖A‖₂), from the real
// 2n×2n embedding of the complex matrix.
func pencilBackwardError(sys *System, p complex128) float64 {
	n, _, _ := sys.Dims()
	m := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			v := p*complex(sys.E.At(i, j), 0) - complex(sys.A.At(i, j), 0)
			m.Set(i, j, real(v))
			m.Set(i+n, j+n, real(v))
			m.Set(i, j+n, -imag(v))
			m.Set(i+n, j, imag(v))
		}
	}
	var svd mat.SVD
	if !svd.Factorize(m, mat.SVDNone) {
		return math.Inf(1)
	}
	sv := svd.Values(nil)
	return sv[2*n-1] / (cmplx.Abs(p)*mat.Norm(sys.E, 2) + mat.Norm(sys.A, 2))
}

// requirePartsKeepPoles checks that the poles of g1 and g2 together are
// those of Poles(sys): the stiff one that of ToExplicit, the others
// eigenvalues of the pencil to backward error 1e-12, and g1 + g2 = sys.
func requirePartsKeepPoles(t *testing.T, name string, sys, g1, g2 *System) {
	t.Helper()
	want, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	ex, err := sys.ToExplicit()
	if err != nil {
		t.Fatalf("%s: ToExplicit: %v", name, err)
	}
	exPoles, err := ex.Poles()
	if err != nil {
		t.Fatal(err)
	}
	stiff := largestPole(exPoles)
	p1, err1 := g1.Poles()
	p2, err2 := g2.Poles()
	if err1 != nil || err2 != nil {
		t.Fatalf("%s: part poles: %v, %v", name, err1, err2)
	}
	got := sortedPolesByModulus(append(slices.Clone(p1), p2...))
	want = sortedPolesByModulus(want)
	if len(got) != len(want) {
		t.Errorf("%s: parts have poles %v ∪ %v, want %v", name, p1, p2, want)
		return
	}
	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-9*max(1, cmplx.Abs(want[i])) {
			t.Errorf("%s: parts have poles %v ∪ %v, want %v", name, p1, p2, want)
			return
		}
	}
	if g := got[len(got)-1]; cmplx.Abs(g-stiff) > 1e-9*cmplx.Abs(stiff) {
		t.Errorf("%s: stiff pole %v, ToExplicit %v", name, g, stiff)
	}
	for _, p := range got[:len(got)-1] {
		if be := pencilBackwardError(sys, p); !(be < 1e-12) {
			t.Errorf("%s: pole %v has pencil backward error %.3g", name, p, be)
		}
	}
	if rel := maxPencilRel(t, sys, []complex128{0.05i, 0.37i, complex(0.5, 3), 10i, 1e3i}, func(s complex128) [][]complex128 {
		g, q := pencilResponse(t, g1, s), pencilResponse(t, g2, s)
		for i := range g {
			for j := range g[i] {
				g[i][j] += q[i][j]
			}
		}
		return g
	}); rel > 1e-9 || math.IsNaN(rel) {
		t.Errorf("%s: parts off the pencil by %.3g", name, rel)
	}
}

// YW5B4U: for an E of full MATLAB rank whose stiff mode QZ gives β = 0,
// Poles(sys) has the stiff pole of E⁻¹A (RL6IT7) while Stabsep kept the
// mode in Unstable with β = 0, where Poles dropped it as infinite (window
// seeds 7878, 70276, 274104; 121514 for the unstable one). Each pole of
// sys is now a pole of exactly one part, on its side of the boundary.
func TestStabsepFullRankEStiffModeKeepsPolesOfSys(t *testing.T) {
	full := 0
	for i, c := range stiffFullRankECases {
		sys := stiffFullRankEModel(t, i)
		if !matlabRankFull(t, sys.E) {
			continue
		}
		full++
		sep, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		requirePartsKeepPoles(t, c.name, sys, sep.Stable, sep.Unstable)
		if sep.Stable.IsDescriptor() {
			t.Errorf("%s: Stable is a descriptor", c.name)
		}
		if ok, err := sep.Stable.IsStable(); err != nil || !ok {
			t.Errorf("%s: Stable.IsStable = %v, %v", c.name, ok, err)
		}
		pu, _ := sep.Unstable.Poles()
		for _, p := range pu {
			if real(p) < 0 {
				t.Errorf("%s: stable pole %v in Unstable", c.name, p)
			}
		}
	}
	if full == 0 {
		t.Skip("no model has full MATLAB rank E on this platform")
	}
}

func TestModsepFullRankEStiffModeKeepsPolesOfSys(t *testing.T) {
	full := 0
	for i, c := range stiffFullRankECases {
		sys := stiffFullRankEModel(t, i)
		if !matlabRankFull(t, sys.E) {
			continue
		}
		full++
		for _, cutoff := range []float64{1, 1e16} {
			sep, err := Modsep(sys, cutoff)
			if err != nil {
				t.Fatalf("%s cutoff %g: %v", c.name, cutoff, err)
			}
			requirePartsKeepPoles(t, c.name, sys, sep.Slow, sep.Fast)
		}
	}
	if full == 0 {
		t.Skip("no model has full MATLAB rank E on this platform")
	}
}
