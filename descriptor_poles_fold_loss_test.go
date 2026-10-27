package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pencilModelCase is a descriptor model (A, B, C, D, E), row-major.
type pencilModelCase struct {
	name          string
	n, m, p       int
	a, b, c, d, e []float64
}

func (c pencilModelCase) system(t *testing.T) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b), mat.NewDense(c.p, c.n, c.c),
		mat.NewDense(c.p, c.m, c.d), mat.NewDense(c.n, c.n, c.e), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

const pencilOraclePrec = 320

func pencilOracleFloat(x float64) *big.Float {
	return new(big.Float).SetPrec(pencilOraclePrec).SetFloat64(x)
}

func pencilOracleMatrix(m *mat.Dense) [][]*big.Float {
	r, c := m.Dims()
	out := make([][]*big.Float, r)
	for i := range r {
		out[i] = make([]*big.Float, c)
		for j := range c {
			out[i][j] = pencilOracleFloat(m.At(i, j))
		}
	}
	return out
}

// pencilOracleSolve returns M⁻¹X by Gauss-Jordan elimination with partial
// pivoting at pencilOraclePrec bits, or nil for an exactly singular M.
func pencilOracleSolve(M, X [][]*big.Float) [][]*big.Float {
	n, k := len(M), len(X[0])
	W := make([][]*big.Float, n)
	for i := range n {
		W[i] = make([]*big.Float, n+k)
		for j := range n {
			W[i][j] = new(big.Float).SetPrec(pencilOraclePrec).Set(M[i][j])
		}
		for j := range k {
			W[i][n+j] = new(big.Float).SetPrec(pencilOraclePrec).Set(X[i][j])
		}
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if new(big.Float).Abs(W[r][c]).Cmp(new(big.Float).Abs(W[piv][c])) > 0 {
				piv = r
			}
		}
		W[c], W[piv] = W[piv], W[c]
		if W[c][c].Sign() == 0 {
			return nil
		}
		for r := range n {
			if r == c || W[r][c].Sign() == 0 {
				continue
			}
			f := new(big.Float).SetPrec(pencilOraclePrec).Quo(W[r][c], W[c][c])
			for j := c; j < n+k; j++ {
				W[r][j].Sub(W[r][j], new(big.Float).SetPrec(pencilOraclePrec).Mul(f, W[c][j]))
			}
		}
	}
	out := make([][]*big.Float, n)
	for i := range n {
		out[i] = make([]*big.Float, k)
		for j := range k {
			out[i][j] = new(big.Float).SetPrec(pencilOraclePrec).Quo(W[i][n+j], W[i][i])
		}
	}
	return out
}

// pencilOracleRealRoot refines a real root of det(A − λE) from x0 by Newton,
// f/f' = −1/tr((A − λE)⁻¹E), at pencilOraclePrec bits.
func pencilOracleRealRoot(A, E [][]*big.Float, x0 float64) (float64, bool) {
	n := len(A)
	lam := pencilOracleFloat(x0)
	for range 80 {
		M := make([][]*big.Float, n)
		for i := range n {
			M[i] = make([]*big.Float, n)
			for j := range n {
				M[i][j] = new(big.Float).SetPrec(pencilOraclePrec).Sub(A[i][j], new(big.Float).SetPrec(pencilOraclePrec).Mul(lam, E[i][j]))
			}
		}
		X := pencilOracleSolve(M, E)
		if X == nil {
			l, _ := lam.Float64()
			return l, true
		}
		tr := pencilOracleFloat(0)
		for i := range n {
			tr.Add(tr, X[i][i])
		}
		if tr.Sign() == 0 {
			return 0, false
		}
		step := new(big.Float).SetPrec(pencilOraclePrec).Quo(pencilOracleFloat(1), tr)
		lam.Add(lam, step)
		s, _ := step.Float64()
		l, _ := lam.Float64()
		if math.Abs(s) <= 1e-60*math.Max(1, math.Abs(l)) {
			return l, true
		}
	}
	return 0, false
}

// pencilOracleRoots refines each real start to a root of det(A − λE) at
// pencilOraclePrec bits; ok is false unless every start converges to its
// own root (distinct to 1e-8), so the roots are all n eigenvalues.
func pencilOracleRoots(sys *System, starts []complex128) (roots []float64, ok bool) {
	A, E := pencilOracleMatrix(sys.A), pencilOracleMatrix(sys.E)
	for _, s := range starts {
		if imag(s) != 0 {
			return nil, false
		}
		r, conv := pencilOracleRealRoot(A, E, real(s))
		if !conv {
			return nil, false
		}
		for _, q := range roots {
			if math.Abs(q-r) <= 1e-8*math.Max(1, math.Abs(r)) {
				return nil, false
			}
		}
		roots = append(roots, r)
	}
	return roots, true
}

// poleFoldLossCases are nearSingularEContinuousK draws (seeds 355, 46, 2210,
// 4312; HK6D3N) with E of full MATLAB rank and cond(E) ~ 1e16, whose LU fold
// E⁻¹A (ToExplicit) misplaces a slow pole by O(0.1) and flips its
// stability: 355 to exactly 0, 2210 to +0.099, 46 and 4312 into the left
// half-plane. The stiff ~1e15 pole has the same sign in QZ and the fold.
var poleFoldLossCases = []pencilModelCase{
	{name: "seed 355", n: 2, m: 1, p: 1,
		a: []float64{0.22399326286783883, 0.8402255269963316, -0.3783616459247681, -0.9974968703206462},
		b: []float64{0.9097247634578738, -0.7662664682401428},
		c: []float64{-1.5350587018630855, -1.7388160724801482},
		d: []float64{-0.28270713794423386},
		e: []float64{0.5245065011343613, -0.19913998015010564, -0.001982441398173575, 0.000752675781572609}},
	{name: "seed 46", n: 4, m: 1, p: 1,
		a: []float64{0.5414084488417618, 1.5679446698902575, -1.1476050053714222, 0.48714585101009084, 0.29842073369612226, 0.4784530477013085, 0.3534586715827879, 0.03826668165634744, -0.3468957827943742, -0.4597765039600787, -0.600492027779978, -0.38066805901593326, 0.08850027223164926, 0.7730903828348706, 0.3437969696434461, -0.16787335839168466},
		b: []float64{-1.0866238501149552, 1.8868658258980304, -0.6186020487654416, -0.5625192378754207},
		c: []float64{2.4110016145960063, 0.6405766884785891, -0.175372503188046, 1.3249776484032758},
		d: []float64{0.4486562990926004},
		e: []float64{0.5569856441747267, -0.16455967036395622, 0.1667268459574106, -0.15144396102195812, -0.004920043352418276, 0.22960000410890716, -0.32854613002690847, -0.598615526025465, 0.37687524678697715, 0.06877283632692356, 0.30197676082100755, -0.3218519098316367, 0.28244638106340114, -0.26209416073907515, -0.5040335106125737, -0.08710489309758754}},
	{name: "seed 2210", n: 4, m: 2, p: 1,
		a: []float64{-1.0136393817160623, 0.41020766831170286, -0.09408133555158105, -0.644179004526586, 1.07418114205301, -0.9218150379285601, -1.231723246386637, -0.4687636393685979, -0.4784512557501716, 0.10403339381831038, -0.20208321866594542, -0.33632021307486626, -0.6580025519940232, 0.32980882689020774, 0.3231255322725699, 0.21958078122713193},
		b: []float64{-0.3626171625118464, 0.1944985799807775, -0.7708366418667391, 1.251931747212658, -0.40474469480612923, -1.0336305172761757, -0.9947060093932467, 0.0016913529662841962},
		c: []float64{1.7945515896385373, 0.9289844921303549, -0.7535614308719014, 0.3635245322319877},
		d: []float64{-1.1233747942280599, -1.527784706145031},
		e: []float64{0.06431405765055008, -0.09793366984003522, 0.5845997476492618, 0.07814003323238274, 0.243435295920452, 0.37896528924668094, 0.044798667658080306, -0.2597269810728716, 0.05389243080007255, 0.20790324116625455, 0.16412401124056547, 0.40771103662944136, -0.04850114202678998, 0.25795304454007617, -0.5177231761406489, 0.2995265591040745}},
	{name: "seed 4312", n: 3, m: 1, p: 1,
		a: []float64{-1.508799949444929, -0.9908733309271828, -0.23306747705955372, -2.6807188376475897, -0.574824929849027, -0.6439455090966051, -0.4911676708095334, 0.4326453223586859, -0.3770803441025254},
		b: []float64{-1.8026187197654808, -0.5174592290459684, -0.11128888859008108},
		c: []float64{-0.6301857100380259, 1.9627097714501562, 0.1977142507505732},
		d: []float64{0.8647726842582911},
		e: []float64{0.20418495938098255, 0.12201940186391004, -0.6947173956684332, 0.048638532495039424, 0.2703977125745278, -0.18385760644761337, 0.11292817202362494, -0.7530086338224561, -0.3217698127466444}},
}

// On a full-rank E whose fold E⁻¹A loses the slow poles to cond(E)·ε, Poles
// and IsStable follow the pencil, as its 320-bit generalized eigenvalues
// confirm, and must not take the fold's slow poles.
func TestIsStableFullRankEFollowsPencilWhereFoldLosesSlowPoles(t *testing.T) {
	checked := 0
	for _, c := range poleFoldLossCases {
		sys := c.system(t)
		if !matlabRankFull(t, sys.E) {
			continue
		}
		pol, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", c.name, err)
		}
		if len(pol) != c.n {
			t.Errorf("%s: Poles = %v, want %d", c.name, pol, c.n)
			continue
		}
		roots, ok := pencilOracleRoots(sys, pol)
		if !ok {
			t.Errorf("%s: oracle did not resolve %d distinct roots from %v", c.name, c.n, pol)
			continue
		}
		checked++
		want := true
		for i, r := range roots {
			if r >= 0 {
				want = false
			}
			if math.Abs(r) < 1e6 && cmplx.Abs(pol[i]-complex(r, 0)) > 1e-9*math.Max(1, math.Abs(r)) {
				t.Errorf("%s: slow pole %v, oracle %v", c.name, pol[i], r)
			}
		}
		got, err := sys.IsStable()
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: IsStable = %v, oracle poles %v", c.name, got, roots)
		}
	}
	if checked == 0 {
		t.Skip("no model has full MATLAB rank E on this platform")
	}
}
