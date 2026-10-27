package controlsys

import (
	"cmp"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// polesMatchRel reports whether got and want pair up one to one, each pair
// within 1e-9 relative, matched greedily from the largest modulus down.
func polesMatchRel(got, want []complex128) bool {
	if len(got) != len(want) {
		return false
	}
	want = slices.Clone(want)
	slices.SortFunc(want, func(a, b complex128) int { return cmp.Compare(cmplx.Abs(b), cmplx.Abs(a)) })
	used := make([]bool, len(got))
	for _, w := range want {
		best := -1
		for i, g := range got {
			if !used[i] && (best < 0 || cmplx.Abs(g-w) < cmplx.Abs(got[best]-w)) {
				best = i
			}
		}
		if cmplx.Abs(got[best]-w) > 1e-9*max(1, cmplx.Abs(w)) {
			return false
		}
		used[best] = true
	}
	return true
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
	got := append(slices.Clone(p1), p2...)
	if !polesMatchRel(got, want) {
		t.Errorf("%s: parts have poles %v ∪ %v, want %v", name, p1, p2, want)
		return
	}
	slices.SortFunc(got, func(a, b complex128) int { return cmp.Compare(cmplx.Abs(a), cmplx.Abs(b)) })
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

// YW5B4U review: the stiff mode of a full-rank E goes into the same part
// as another finite mode (an unstable stiff pole, continuous and discrete;
// Modsep cutoff 1). That part's E then had t_ii = s_ii/λ ~ 2e-16 next to
// an O(1) entry, which Poles ranked deficient, dropping λ as infinite;
// and where ordering swapped the stiff block, its β left the β = 0 test
// and the mode stayed in the wrong part with a QZ value off by 10–1000×.
// The last three are from a sweep of U·diag(σ₁, σ₂, ~3·eps(σ₁))·Vᵀ
// models (PCG seeds 6399, 65907, 543647).
var stiffFullRankEWithFiniteCases = []struct {
	name    string
	dt      float64
	a, b, c []float64
	d       float64
	e       []float64
}{
	{"continuous", 0, []float64{0.24665062817826747, -0.5168762889537303, -0.4437368615387086, 0.36669838919294756, -0.2826256918821503, -0.7054277024900218, -0.3334131685262898, 0.25091000372278466, -0.21674060061310185},
		[]float64{0.4356246855672622, -0.3587507059176413, -0.20561710053869475},
		[]float64{-0.4816235549440774, -0.31312773883159195, 0.6815830382484291}, 0.10242424745601617,
		[]float64{0.17810535573272404, 0.33765643954843605, 0.3870045996052451, 0.3715325149135733, 0.2520175401541143, 0.27727542077278367, -0.5132921467882982, 0.311030811053703, 0.38934542028207547}},
	{"discrete", 0.1, []float64{1.0803219316796577, -0.6172928420860001, -1.341923302032613, -0.07768319635811793, -0.5049410441351468, -0.7195690931113559, -1.0475401319837139, 0.6588016123645511, -0.10699389762808095},
		[]float64{0.5485357884955011, 0.3879774310356333, -0.7097768715393529},
		[]float64{-0.45755468578851705, -0.13856606134800634, 1.4592072148821345}, 0.32726740793221776,
		[]float64{-0.32673461810996524, -0.24473543419525945, -0.03030785458439381, 0.32458012687273347, 0.09387718839421573, 0.1229980454673914, 0.25298841985975745, -0.6476364229141479, 0.5445006418438522}},
	{"stable stiff with stable mode", 0, []float64{0.0558736733776502, -0.2953174338928413, -0.4437753114282508, -0.17524280202603368, 0.8564527184231843, -0.041311131897976416, -0.019825563296573326, 0.40017070271775257, 0.05024600650950428},
		[]float64{-0.49054166380366837, -0.08502680326213508, -1.0870673480050064},
		[]float64{0.926814842557221, 0.7936685684201816, 0.471014496469557}, -1.3841037780352146,
		[]float64{0.14916507871523393, -0.371648582708343, -0.1766230855265791, 0.45715258342646803, -0.40300966065052346, -0.0016843406058446862, -0.4394817537619985, -0.21612784606615765, -0.4408995530658936}},
	{"unstable stiff, modes below 1", 0, []float64{-0.28097908876458877, -0.002231873191225975, 0.32049857398064086, 0.18786997134144717, 0.5450272993727336, 0.11560868905846874, -0.47830274636368136, 0.14577882037389, -0.49769936690320793},
		[]float64{-1.2683587935427072, 0.844666301239474, -0.5610859523435483},
		[]float64{1.4036921023228364, 1.1311207388348687, 0.4773709529991661}, 0.044113444306354,
		[]float64{-0.05878050812528363, 0.20690617481519474, 0.28431208389180207, -0.43904771402926895, 0.3016529107221213, 0.3143170298151606, -0.3863076138139306, -0.34932443998461726, -0.6176810487421636}},
	{"stable stiff, reordered", 0, []float64{-0.5647722732372393, 0.15850063362118655, 0.008218557792939826, -0.19267596584672436, -0.20238192340576866, 0.004938715122535603, -0.694795634591211, -0.2330960445582697, 0.5061680126409953},
		[]float64{1.0035861569997486, 0.29511504445647974, 0.038913351942788596},
		[]float64{-1.7116284497699275, -1.03847145732074, -0.7434224630873866}, 0.690412284956057,
		[]float64{-0.31029842765077836, 0.0768793541262584, 0.22242779761792938, 0.274653356326923, -0.2534691924135258, 0.027988032114484966, -0.15023725837894025, 0.6125010848517517, -0.5899609295388961}},
}

func TestStabsepFullRankEStiffModeSharingPartKeepsPolesOfSys(t *testing.T) {
	full := 0
	for _, c := range stiffFullRankEWithFiniteCases {
		sys, err := NewDescriptor(mat.NewDense(3, 3, c.a), mat.NewDense(3, 1, c.b), mat.NewDense(1, 3, c.c),
			mat.NewDense(1, 1, []float64{c.d}), mat.NewDense(3, 3, c.e), c.dt)
		if err != nil {
			t.Fatal(err)
		}
		if !matlabRankFull(t, sys.E) {
			continue
		}
		full++
		sep, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		requirePartsKeepPoles(t, c.name, sys, sep.Stable, sep.Unstable)
		if ok, err := sep.Stable.IsStable(); err != nil || !ok {
			t.Errorf("%s: Stable.IsStable = %v, %v", c.name, ok, err)
		}
		if sep.Stable.IsDescriptor() {
			t.Errorf("%s: Stable is a descriptor", c.name)
		}
		pu, _ := sep.Unstable.Poles()
		for _, p := range pu {
			if c.dt == 0 && real(p) < 0 || c.dt > 0 && cmplx.Abs(p) < 1 {
				t.Errorf("%s: stable pole %v in Unstable", c.name, p)
			}
		}
		if c.dt > 0 {
			continue
		}
		for _, cutoff := range []float64{1, 1e16} {
			ms, err := Modsep(sys, cutoff)
			if err != nil {
				t.Fatalf("%s cutoff %g: %v", c.name, cutoff, err)
			}
			requirePartsKeepPoles(t, c.name+" modsep", sys, ms.Slow, ms.Fast)
		}
	}
	if full == 0 {
		t.Skip("no model has full MATLAB rank E on this platform")
	}
}
