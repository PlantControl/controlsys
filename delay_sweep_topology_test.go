package controlsys_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

// Area B of the delay sweep: MIMO topology preservation and system algebra.
// References are closed forms of e^{-sτ}·rational evaluated in-test.
// Tolerances: exact algebra 1e-12; frequency response
// ‖got−ref‖_F ≤ 1e-9·(1+‖ref‖_F) per frequency.

const (
	sweepTopologyExactTol = 1e-12
	sweepTopologyFreqTol  = 1e-9
)

type sweepTopologyMat = [][]complex128

type sweepTopologyFO struct {
	gain, pole, delay [2][2]float64
}

var (
	sweepTopologyG = sweepTopologyFO{
		gain:  [2][2]float64{{1, 2}, {3, 1}},
		pole:  [2][2]float64{{1, 2}, {3, 4}},
		delay: [2][2]float64{{2, 5}, {7, 1}},
	}
	sweepTopologyG2 = sweepTopologyFO{
		gain:  [2][2]float64{{1, -1}, {2, 1}},
		pole:  [2][2]float64{{5, 6}, {7, 8}},
		delay: [2][2]float64{{1, 4}, {0.5, 3}},
	}
)

func (p sweepTopologyFO) eval(s complex128) sweepTopologyMat {
	h := sweepTopologyZeros(2, 2)
	for i := range 2 {
		for j := range 2 {
			h[i][j] = complex(p.gain[i][j], 0) * cmplx.Exp(-s*complex(p.delay[i][j], 0)) / (s + complex(p.pole[i][j], 0))
		}
	}
	return h
}

func (p sweepTopologyFO) delayDense() *mat.Dense {
	return mat.NewDense(2, 2, []float64{p.delay[0][0], p.delay[0][1], p.delay[1][0], p.delay[1][1]})
}

// G3(s) = [[e^{-0.3s}/(s+0.5), (0.5s+0.25)/(s+0.5)], [(s+2)/(s+2), e^{-1.5s}(s+1)/(s+2)]]:
// biproper entries with delayed feedthrough.
func sweepTopologyG3Eval(s complex128) sweepTopologyMat {
	return sweepTopologyMat{
		{cmplx.Exp(-0.3*s) / (s + 0.5), 0.5},
		{1, cmplx.Exp(-1.5*s) * (s + 1) / (s + 2)},
	}
}

func sweepTopologyG3(t *testing.T) *controlsys.System {
	t.Helper()
	tf := &controlsys.TransferFunc{
		Num: [][][]float64{
			{{0, 1}, {0.5, 0.25}},
			{{1, 2}, {1, 1}},
		},
		Den:   [][]float64{{1, 0.5}, {1, 2}},
		Delay: [][]float64{{0.3, 0}, {0, 1.5}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	return res.Sys
}

func sweepTopologyZeros(p, m int) sweepTopologyMat {
	h := make(sweepTopologyMat, p)
	for i := range h {
		h[i] = make([]complex128, m)
	}
	return h
}

func sweepTopologyMul(a, b sweepTopologyMat) sweepTopologyMat {
	h := sweepTopologyZeros(len(a), len(b[0]))
	for i := range a {
		for j := range b[0] {
			for k := range b {
				h[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return h
}

func sweepTopologyAdd(a, b sweepTopologyMat, sb complex128) sweepTopologyMat {
	h := sweepTopologyZeros(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			h[i][j] = a[i][j] + sb*b[i][j]
		}
	}
	return h
}

func sweepTopologyInv2(a sweepTopologyMat) sweepTopologyMat {
	det := a[0][0]*a[1][1] - a[0][1]*a[1][0]
	return sweepTopologyMat{{a[1][1] / det, -a[0][1] / det}, {-a[1][0] / det, a[0][0] / det}}
}

func sweepTopologyEye(n int) sweepTopologyMat {
	h := sweepTopologyZeros(n, n)
	for i := range n {
		h[i][i] = 1
	}
	return h
}

func sweepTopologyFrob(a sweepTopologyMat) float64 {
	var s float64
	for i := range a {
		for j := range a[i] {
			s += real(a[i][j])*real(a[i][j]) + imag(a[i][j])*imag(a[i][j])
		}
	}
	return math.Sqrt(s)
}

// sweepTopologyOmega spans log 1e-3..8·2π/τmin plus a dense linear segment to
// 6·2π/τmin, so phase wraps of every path delay are resolved.
func sweepTopologyOmega(tauMin float64) []float64 {
	wMax := 8 * 2 * math.Pi / tauMin
	var w []float64
	const nLog = 80
	for k := range nLog {
		w = append(w, 1e-3*math.Pow(wMax/1e-3, float64(k)/float64(nLog-1)))
	}
	const nLin = 600
	wLin := 6 * 2 * math.Pi / tauMin
	for k := 1; k <= nLin; k++ {
		w = append(w, wLin*float64(k)/nLin)
	}
	return w
}

func sweepTopologyAssertFreq(t *testing.T, label string, sys *controlsys.System, omega []float64, ref func(complex128) sweepTopologyMat, tol float64) {
	t.Helper()
	fr, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	worst, worstW := 0.0, 0.0
	for k, w := range omega {
		r := ref(complex(0, w))
		if fr.P != len(r) || fr.M != len(r[0]) {
			t.Fatalf("%s: dims %dx%d, want %dx%d", label, fr.P, fr.M, len(r), len(r[0]))
		}
		got := sweepTopologyZeros(fr.P, fr.M)
		for i := range fr.P {
			for j := range fr.M {
				got[i][j] = fr.At(k, i, j)
			}
		}
		e := sweepTopologyFrob(sweepTopologyAdd(got, r, -1)) / (1 + sweepTopologyFrob(r))
		if e > worst {
			worst, worstW = e, w
		}
	}
	if worst > tol {
		t.Errorf("%s: worst mixed freq error %.3e at ω=%g (tol %.0e)", label, worst, worstW, tol)
	}
}

func sweepTopologyFOSS(t *testing.T, p sweepTopologyFO) *controlsys.System {
	t.Helper()
	A := mat.NewDense(4, 4, nil)
	B := mat.NewDense(4, 2, nil)
	C := mat.NewDense(2, 4, nil)
	for i := range 2 {
		for j := range 2 {
			k := 2*i + j
			A.Set(k, k, -p.pole[i][j])
			B.Set(k, j, 1)
			C.Set(i, k, p.gain[i][j])
		}
	}
	sys, err := controlsys.NewWithDelay(A, B, C, mat.NewDense(2, 2, nil), p.delayDense(), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sweepTopologyFOTF(p sweepTopologyFO) *controlsys.TransferFunc {
	tf := &controlsys.TransferFunc{
		Num:   make([][][]float64, 2),
		Den:   make([][]float64, 2),
		Delay: [][]float64{p.delay[0][:], p.delay[1][:]},
	}
	for i := range 2 {
		a, b := p.pole[i][0], p.pole[i][1]
		tf.Den[i] = []float64{1, a + b, a * b}
		tf.Num[i] = [][]float64{
			{0, p.gain[i][0], p.gain[i][0] * b},
			{0, p.gain[i][1], p.gain[i][1] * a},
		}
	}
	return tf
}

func sweepTopologyFOTFSS(t *testing.T, p sweepTopologyFO) *controlsys.System {
	t.Helper()
	res, err := sweepTopologyFOTF(p).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	return res.Sys
}

// sweepTopologyFOInternal realizes p with one internal delay per path, the
// delay sitting between input j and the path state.
func sweepTopologyFOInternal(t *testing.T, p sweepTopologyFO) *controlsys.System {
	t.Helper()
	A := mat.NewDense(4, 4, nil)
	C := mat.NewDense(2, 4, nil)
	D21 := mat.NewDense(4, 2, nil)
	B2 := mat.NewDense(4, 4, nil)
	tau := make([]float64, 4)
	for i := range 2 {
		for j := range 2 {
			k := 2*i + j
			A.Set(k, k, -p.pole[i][j])
			C.Set(i, k, p.gain[i][j])
			D21.Set(k, j, 1)
			B2.Set(k, k, 1)
			tau[k] = p.delay[i][j]
		}
	}
	sys, err := controlsys.New(A, mat.NewDense(4, 2, nil), C, mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay(tau, B2, mat.NewDense(4, 4, nil), mat.NewDense(2, 4, nil), D21, mat.NewDense(4, 4, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

// sweepTopologyGFactored is G as InputDelay [1 0] + OutputDelay [1 1] +
// residual Delay [[0 4] [5 0]]; G admits no residual-free split.
func sweepTopologyGFactored(t *testing.T) *controlsys.System {
	t.Helper()
	sys := sweepTopologyFOSS(t, sweepTopologyG)
	sys.Delay = nil
	if err := sys.SetInputDelay([]float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, []float64{0, 4, 5, 0})); err != nil {
		t.Fatal(err)
	}
	return sys
}

func sweepTopologyGain(t *testing.T, r, c int, d []float64) *controlsys.System {
	t.Helper()
	g, err := controlsys.NewGain(mat.NewDense(r, c, d), 0)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func sweepTopologyMust(t *testing.T, label string) func(*controlsys.System, error) *controlsys.System {
	return func(sys *controlsys.System, err error) *controlsys.System {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		return sys
	}
}

func sweepTopologyAssertDense(t *testing.T, label string, got, want *mat.Dense, tol float64) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("%s: got %v, want %v", label, got, want)
		}
		return
	}
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("%s: dims %dx%d, want %dx%d", label, gr, gc, wr, wc)
	}
	scale := 1 + mat.Norm(want, 2)
	var d mat.Dense
	d.Sub(got, want)
	if e := mat.Norm(&d, 2) / scale; e > tol {
		t.Errorf("%s: rel error %.3e > %.0e\ngot  %v\nwant %v", label, e, tol, mat.Formatted(got), mat.Formatted(want))
	}
}

// TestSweepTopologyBaseFormsMatchClosedForm (B-01): every construction of G —
// IODelay on diagonal SS, TF with Delay, TF→SS companion, Input+Output+residual
// factorization, hand internal-delay LFT, PullDelaysToLFT — matches G(jω).
func TestSweepTopologyBaseFormsMatchClosedForm(t *testing.T) {
	omega := sweepTopologyOmega(1)
	io := sweepTopologyFOSS(t, sweepTopologyG)
	pulled, err := io.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	pulledFactored, err := sweepTopologyGFactored(t).PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	forms := map[string]*controlsys.System{
		"iodelay":         io,
		"tf-ss":           sweepTopologyFOTFSS(t, sweepTopologyG),
		"factored":        sweepTopologyGFactored(t),
		"internal":        sweepTopologyFOInternal(t, sweepTopologyG),
		"pulled":          pulled,
		"pulled-factored": pulledFactored,
	}
	for name, sys := range forms {
		sweepTopologyAssertFreq(t, name, sys, omega, sweepTopologyG.eval, sweepTopologyFreqTol)
	}
	if !pulled.HasInternalDelay() || pulled.Delay != nil || pulled.InputDelay != nil || pulled.OutputDelay != nil {
		t.Errorf("PullDelaysToLFT left external delays: io=%v in=%v out=%v", pulled.Delay, pulled.InputDelay, pulled.OutputDelay)
	}

	tf := sweepTopologyFOTF(sweepTopologyG)
	worst := 0.0
	for _, w := range omega {
		h, err := tf.Eval(complex(0, w))
		if err != nil {
			t.Fatal(err)
		}
		r := sweepTopologyG.eval(complex(0, w))
		worst = max(worst, sweepTopologyFrob(sweepTopologyAdd(h, r, -1))/(1+sweepTopologyFrob(r)))
	}
	if worst > sweepTopologyFreqTol {
		t.Errorf("TransferFunc.Eval worst mixed error %.3e", worst)
	}

	total, err := sweepTopologyGFactored(t).TotalDelay()
	if err != nil {
		t.Fatal(err)
	}
	sweepTopologyAssertDense(t, "factored TotalDelay", total, sweepTopologyG.delayDense(), sweepTopologyExactTol)
}

// TestSweepTopologyConversionRoundTrips (B-02): SS→TF keeps per-path delays
// exactly; TF→SS, GetDelayModel→SetDelayModel and TF of the factored form all
// preserve G(jω); internal-delay forms are rejected by TransferFunction.
func TestSweepTopologyConversionRoundTrips(t *testing.T) {
	omega := sweepTopologyOmega(1)
	for name, sys := range map[string]*controlsys.System{
		"iodelay":  sweepTopologyFOSS(t, sweepTopologyG),
		"factored": sweepTopologyGFactored(t),
	} {
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", name, err)
		}
		for i := range 2 {
			for j := range 2 {
				if d := res.TF.Delay[i][j]; math.Abs(d-sweepTopologyG.delay[i][j]) > sweepTopologyExactTol {
					t.Errorf("%s: TF.Delay[%d][%d] = %v, want %v", name, i, j, d, sweepTopologyG.delay[i][j])
				}
			}
		}
		worst := 0.0
		for _, w := range omega {
			h, err := res.TF.Eval(complex(0, w))
			if err != nil {
				t.Fatal(err)
			}
			r := sweepTopologyG.eval(complex(0, w))
			worst = max(worst, sweepTopologyFrob(sweepTopologyAdd(h, r, -1))/(1+sweepTopologyFrob(r)))
		}
		if worst > sweepTopologyFreqTol {
			t.Errorf("%s: SS→TF Eval worst mixed error %.3e", name, worst)
		}
		back, err := res.TF.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		sweepTopologyAssertFreq(t, name+" SS→TF→SS", back.Sys, omega, sweepTopologyG.eval, sweepTopologyFreqTol)
	}

	for name, sys := range map[string]*controlsys.System{
		"iodelay":  sweepTopologyFOSS(t, sweepTopologyG),
		"factored": sweepTopologyGFactored(t),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
	} {
		H, tau, err := sys.GetDelayModel()
		if err != nil {
			t.Fatalf("%s: GetDelayModel: %v", name, err)
		}
		if H.HasDelay() {
			t.Errorf("%s: GetDelayModel H has delays", name)
		}
		rebuilt, err := controlsys.SetDelayModel(H, tau)
		if err != nil {
			t.Fatalf("%s: SetDelayModel: %v", name, err)
		}
		sweepTopologyAssertFreq(t, name+" Get/SetDelayModel", rebuilt, omega, sweepTopologyG.eval, sweepTopologyFreqTol)
	}

	if _, err := sweepTopologyFOInternal(t, sweepTopologyG).TransferFunction(nil); err == nil {
		t.Error("TransferFunction of internal-delay model: want documented rejection, got nil error")
	}
}

// TestSweepTopologySeriesParallelPreservePathDelays (B-03): Series(G,G2) =
// G2·G, Series(G2,G) = G·G2, Parallel both orders = G+G2, across IO, TF→SS
// and internal-delay operand forms.
func TestSweepTopologySeriesParallelPreservePathDelays(t *testing.T) {
	omega := sweepTopologyOmega(0.5)
	g := map[string]*controlsys.System{
		"io":       sweepTopologyFOSS(t, sweepTopologyG),
		"factored": sweepTopologyGFactored(t),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
	}
	g2 := map[string]*controlsys.System{
		"io":    sweepTopologyFOSS(t, sweepTopologyG2),
		"tf-ss": sweepTopologyFOTFSS(t, sweepTopologyG2),
	}
	for n1, s1 := range g {
		for n2, s2 := range g2 {
			label := n1 + "/" + n2
			ser := sweepTopologyMust(t, label+" Series")(controlsys.Series(s1, s2))
			sweepTopologyAssertFreq(t, label+" Series(G,G2)", ser, omega, func(s complex128) sweepTopologyMat {
				return sweepTopologyMul(sweepTopologyG2.eval(s), sweepTopologyG.eval(s))
			}, sweepTopologyFreqTol)
			ser = sweepTopologyMust(t, label+" Series rev")(controlsys.Series(s2, s1))
			sweepTopologyAssertFreq(t, label+" Series(G2,G)", ser, omega, func(s complex128) sweepTopologyMat {
				return sweepTopologyMul(sweepTopologyG.eval(s), sweepTopologyG2.eval(s))
			}, sweepTopologyFreqTol)
			sum := func(s complex128) sweepTopologyMat {
				return sweepTopologyAdd(sweepTopologyG.eval(s), sweepTopologyG2.eval(s), 1)
			}
			par := sweepTopologyMust(t, label+" Parallel")(controlsys.Parallel(s1, s2))
			sweepTopologyAssertFreq(t, label+" Parallel(G,G2)", par, omega, sum, sweepTopologyFreqTol)
			par = sweepTopologyMust(t, label+" Parallel rev")(controlsys.Parallel(s2, s1))
			sweepTopologyAssertFreq(t, label+" Parallel(G2,G)", par, omega, sum, sweepTopologyFreqTol)
		}
	}
}

// TestSweepTopologyAssociativity (B-04): (G3·G2)·G and G3·(G2·G), built as
// Series(Series(G,G2),G3) and Series(G,Series(G2,G3)), match the closed-form
// triple product; G3 carries delayed feedthrough.
func TestSweepTopologyAssociativity(t *testing.T) {
	omega := sweepTopologyOmega(0.3)
	g, g2, g3 := sweepTopologyFOSS(t, sweepTopologyG), sweepTopologyFOTFSS(t, sweepTopologyG2), sweepTopologyG3(t)
	ref := func(s complex128) sweepTopologyMat {
		return sweepTopologyMul(sweepTopologyG3Eval(s), sweepTopologyMul(sweepTopologyG2.eval(s), sweepTopologyG.eval(s)))
	}
	sweepTopologyAssertFreq(t, "G3 alone", g3, omega, sweepTopologyG3Eval, sweepTopologyFreqTol)
	left := sweepTopologyMust(t, "left inner")(controlsys.Series(g, g2))
	left = sweepTopologyMust(t, "left outer")(controlsys.Series(left, g3))
	right := sweepTopologyMust(t, "right inner")(controlsys.Series(g2, g3))
	right = sweepTopologyMust(t, "right outer")(controlsys.Series(g, right))
	sweepTopologyAssertFreq(t, "(G3·G2)·G", left, omega, ref, sweepTopologyFreqTol)
	sweepTopologyAssertFreq(t, "G3·(G2·G)", right, omega, ref, sweepTopologyFreqTol)

	pl := sweepTopologyMust(t, "par left")(controlsys.Parallel(sweepTopologyMust(t, "p")(controlsys.Parallel(g, g2)), g3))
	pr := sweepTopologyMust(t, "par right")(controlsys.Parallel(g, sweepTopologyMust(t, "p")(controlsys.Parallel(g3, g2))))
	psum := func(s complex128) sweepTopologyMat {
		return sweepTopologyAdd(sweepTopologyAdd(sweepTopologyG.eval(s), sweepTopologyG2.eval(s), 1), sweepTopologyG3Eval(s), 1)
	}
	sweepTopologyAssertFreq(t, "(G+G2)+G3", pl, omega, psum, sweepTopologyFreqTol)
	sweepTopologyAssertFreq(t, "G+(G3+G2)", pr, omega, psum, sweepTopologyFreqTol)
}

// TestSweepTopologyAppendBlkDiag (B-05): Append and BlkDiag of delayed MIMO
// models put each G(jω) on the diagonal and exact zeros off it.
func TestSweepTopologyAppendBlkDiag(t *testing.T) {
	omega := sweepTopologyOmega(0.3)
	g, g2, g3 := sweepTopologyFOSS(t, sweepTopologyG), sweepTopologyFOInternal(t, sweepTopologyG2), sweepTopologyG3(t)
	blocks := []func(complex128) sweepTopologyMat{sweepTopologyG.eval, sweepTopologyG2.eval, sweepTopologyG3Eval}
	ref := func(nb int) func(complex128) sweepTopologyMat {
		return func(s complex128) sweepTopologyMat {
			h := sweepTopologyZeros(2*nb, 2*nb)
			for b := range nb {
				r := blocks[b](s)
				for i := range 2 {
					for j := range 2 {
						h[2*b+i][2*b+j] = r[i][j]
					}
				}
			}
			return h
		}
	}
	app := sweepTopologyMust(t, "Append")(controlsys.Append(g, g2))
	sweepTopologyAssertFreq(t, "Append(G,G2)", app, omega, ref(2), sweepTopologyFreqTol)
	bd := sweepTopologyMust(t, "BlkDiag")(controlsys.BlkDiag(g, g2, g3))
	sweepTopologyAssertFreq(t, "BlkDiag(G,G2,G3)", bd, omega, ref(3), sweepTopologyFreqTol)
	fr, err := bd.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		for i := range 6 {
			for j := range 6 {
				if i/2 != j/2 && fr.At(k, i, j) != 0 {
					t.Fatalf("BlkDiag off-diagonal (%d,%d) at ω=%g = %v, want exact 0", i, j, omega[k], fr.At(k, i, j))
				}
			}
		}
	}
}

// TestSweepTopologyPermutation (B-06): SelectByIndex reorders rows/columns of
// G(jω) and keeps the IODelay permuted exactly; Series with permutation gains
// gives G·P and P·G.
func TestSweepTopologyPermutation(t *testing.T) {
	omega := sweepTopologyOmega(1)
	swap := func(h sweepTopologyMat, rows, cols bool) sweepTopologyMat {
		r := sweepTopologyZeros(2, 2)
		for i := range 2 {
			for j := range 2 {
				si, sj := i, j
				if rows {
					si = 1 - i
				}
				if cols {
					sj = 1 - j
				}
				r[i][j] = h[si][sj]
			}
		}
		return r
	}
	for name, g := range map[string]*controlsys.System{
		"io":       sweepTopologyFOSS(t, sweepTopologyG),
		"factored": sweepTopologyGFactored(t),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
	} {
		sel := sweepTopologyMust(t, name+" select")(g.SelectByIndex([]int{1, 0}, []int{1, 0}))
		sweepTopologyAssertFreq(t, name+" select swap both", sel, omega, func(s complex128) sweepTopologyMat {
			return swap(sweepTopologyG.eval(s), true, true)
		}, sweepTopologyFreqTol)
		if !g.HasInternalDelay() {
			total, err := sel.TotalDelay()
			if err != nil {
				t.Fatal(err)
			}
			sweepTopologyAssertDense(t, name+" select TotalDelay", total, mat.NewDense(2, 2, []float64{1, 7, 5, 2}), sweepTopologyExactTol)
		}
		one := sweepTopologyMust(t, name+" select one")(g.SelectByIndex([]int{1}, []int{0}))
		sweepTopologyAssertFreq(t, name+" select (0,1)", one, omega, func(s complex128) sweepTopologyMat {
			return sweepTopologyMat{{sweepTopologyG.eval(s)[0][1]}}
		}, sweepTopologyFreqTol)

		P := sweepTopologyGain(t, 2, 2, []float64{0, 1, 1, 0})
		colSwap := sweepTopologyMust(t, name+" Series(P,G)")(controlsys.Series(P, g))
		sweepTopologyAssertFreq(t, name+" G·P", colSwap, omega, func(s complex128) sweepTopologyMat {
			return swap(sweepTopologyG.eval(s), false, true)
		}, sweepTopologyFreqTol)
		rowSwap := sweepTopologyMust(t, name+" Series(G,P)")(controlsys.Series(g, P))
		sweepTopologyAssertFreq(t, name+" P·G", rowSwap, omega, func(s complex128) sweepTopologyMat {
			return swap(sweepTopologyG.eval(s), true, false)
		}, sweepTopologyFreqTol)
	}
}

func sweepTopologyGrid(dt, tFinal float64) []float64 {
	n := int(math.Round(tFinal / dt))
	tg := make([]float64, n+1)
	for k := range tg {
		tg[k] = float64(k) * dt
	}
	return tg
}

// sweepTopologyLsimStep returns y[i][k] for a unit step on input in.
func sweepTopologyLsimStep(t *testing.T, label string, sys *controlsys.System, in int, tg []float64) [][]float64 {
	t.Helper()
	_, m, p := sys.Dims()
	u := mat.NewDense(len(tg), m, nil)
	for k := range tg {
		u.Set(k, in, 1)
	}
	resp, err := controlsys.Lsim(sys, u, tg, nil)
	if err != nil {
		t.Fatalf("%s: Lsim: %v", label, err)
	}
	r, c := resp.Y.Dims()
	if r != p || c != len(tg) {
		t.Fatalf("%s: Lsim Y %dx%d, want %dx%d", label, r, c, p, len(tg))
	}
	y := make([][]float64, p)
	for i := range p {
		y[i] = make([]float64, len(tg))
		for k := range tg {
			y[i][k] = resp.Y.At(i, k)
		}
	}
	return y
}

func sweepTopologyFOStep(p sweepTopologyFO, i, j int, tt float64) float64 {
	if tt < p.delay[i][j] {
		return 0
	}
	a := p.pole[i][j]
	return p.gain[i][j] / a * (1 - math.Exp(-a*(tt-p.delay[i][j])))
}

// TestSweepTopologyEquivalentFactorizationsStep (B-07): Lsim unit steps of
// the IODelay, Input+Output+residual, TF→SS, hand internal-delay and
// PullDelaysToLFT forms of G equal the closed-form delayed first-order steps
// on a grid dividing every delay, and so does Step on its own grid. A constant
// step enters every delay line, so sampling is exact; tolerance 1e-9 absolute.
func TestSweepTopologyEquivalentFactorizationsStep(t *testing.T) {
	io := sweepTopologyFOSS(t, sweepTopologyG)
	pulled, err := io.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	forms := map[string]*controlsys.System{
		"iodelay":  io,
		"factored": sweepTopologyGFactored(t),
		"tf-ss":    sweepTopologyFOTFSS(t, sweepTopologyG),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
		"pulled":   pulled,
	}
	tg := sweepTopologyGrid(0.01, 15)
	for name, sys := range forms {
		for j := range 2 {
			y := sweepTopologyLsimStep(t, name, sys, j, tg)
			worst, at := 0.0, 0.0
			for i := range 2 {
				for k, tt := range tg {
					if e := math.Abs(y[i][k] - sweepTopologyFOStep(sweepTopologyG, i, j, tt)); e > worst {
						worst, at = e, tt
					}
				}
			}
			if worst > 1e-9 {
				t.Errorf("%s input %d: worst step error %.3e at t=%g", name, j, worst, at)
			}
		}
	}

	for name, sys := range forms {
		resp, err := controlsys.Step(sys, 15)
		if err != nil {
			t.Fatalf("%s: Step: %v", name, err)
		}
		worst := 0.0
		for j := range 2 {
			for i := range 2 {
				for k, tt := range resp.T {
					worst = max(worst, math.Abs(resp.Y.At(j*2+i, k)-sweepTopologyFOStep(sweepTopologyG, i, j, tt)))
				}
			}
		}
		if worst > 1e-9 {
			t.Errorf("Step(%s) worst error %.3e (dt %g)", name, worst, resp.T[1]-resp.T[0])
		}
	}
}

// TestSweepTopologyRepeatedDelaysD09 (D09): with repeated delays
// [[3 2] [3 2]] the column-shared InputDelay form, the IODelay form, the
// per-path internal form and its MinimalLFT agree with the closed form;
// MinimalLFT merges the identical per-input delay lines 4 → 2.
func TestSweepTopologyRepeatedDelaysD09(t *testing.T) {
	rep := sweepTopologyG
	rep.delay = [2][2]float64{{3, 2}, {3, 2}}
	omega := sweepTopologyOmega(2)

	io := sweepTopologyFOSS(t, rep)
	factored := sweepTopologyFOSS(t, rep)
	factored.Delay = nil
	if err := factored.SetInputDelay([]float64{3, 2}); err != nil {
		t.Fatal(err)
	}
	internal := sweepTopologyFOInternal(t, rep)
	minimal, err := internal.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}
	pulled, err := io.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	pulledMin, err := pulled.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*controlsys.System{
		"iodelay": io, "inputdelay": factored, "internal": internal, "minimal": minimal,
		"pulled": pulled, "pulled-minimal": pulledMin,
	} {
		sweepTopologyAssertFreq(t, name, sys, omega, rep.eval, sweepTopologyFreqTol)
	}
	if !minimal.HasInternalDelay() || len(minimal.LFT.Tau) != 2 {
		t.Errorf("MinimalLFT tau = %v, want two merged channels", minimal.LFT)
	}
	ser := sweepTopologyMust(t, "Series")(controlsys.Series(factored, sweepTopologyGain(t, 2, 2, []float64{1, 0.5, -0.25, 1})))
	sweepTopologyAssertFreq(t, "Series(inputdelay,K)", ser, omega, func(s complex128) sweepTopologyMat {
		return sweepTopologyMul(sweepTopologyMat{{1, 0.5}, {-0.25, 1}}, rep.eval(s))
	}, sweepTopologyFreqTol)
	if ser.HasInternalDelay() {
		t.Errorf("Series with a column-uniform delay should keep it external, got internal tau %v", ser.LFT.Tau)
	}
}

func sweepTopologySISO(t *testing.T, pole, tau float64) *controlsys.System {
	t.Helper()
	sys, err := controlsys.New(mat.NewDense(1, 1, []float64{-pole}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	return sys
}

// sweepTopologyDDEStep integrates x_g' = −x_g + e(t−2), x_h' = −2x_h + x_g(t−3),
// e = 1 + sign·x_h with zero history by Heun's method on step h dividing both
// delays; the r(t−2) jump is aligned with the grid, so the error is O(h²).
func sweepTopologyDDEStep(sign, h, tFinal float64) []float64 {
	n := int(math.Round(tFinal / h))
	d2, d3 := int(math.Round(2/h)), int(math.Round(3/h))
	xg := make([]float64, n+1)
	xh := make([]float64, n+1)
	at := func(x []float64, k int) float64 {
		if k < 0 {
			return 0
		}
		return x[k]
	}
	for k := range n {
		r := 0.0
		if k >= d2 {
			r = 1
		}
		e0 := r + sign*at(xh, k-d2)
		e1 := r + sign*at(xh, k+1-d2)
		f0g := -xg[k] + e0
		f0h := -2*xh[k] + at(xg, k-d3)
		pg, ph := xg[k]+h*f0g, xh[k]+h*f0h
		f1g := -pg + e1
		f1h := -2*ph + at(xg, k+1-d3)
		xg[k+1] = xg[k] + h/2*(f0g+f1g)
		xh[k+1] = xh[k] + h/2*(f0h+f1h)
	}
	return xg
}

// TestSweepTopologyFeedbackDifferentDeadTimes (B-08, critical):
// G = e^{-2s}/(s+1), H = e^{-3s}/(s+2). Feedback(G,H,∓1) = G/(1±GH) at every ω,
// the result carries internal delays and no external ones, TransferFunction
// rejects it, and the Lsim step is exactly 0 on [0,2), equals 1−e^{−(t−2)} on
// [2,7) (loop delay 5 not yet closed), and matches a Heun DDE reference
// (h = 1e-4) on [7,15] within 0.25·dt (observed ≈0.16·dt) with first-order convergence in dt (the
// documented ZOH approximation of non-common internal delays, as MATLAB).
func TestSweepTopologyFeedbackDifferentDeadTimes(t *testing.T) {
	omega := sweepTopologyOmega(2)
	g := func(s complex128) complex128 { return cmplx.Exp(-2*s) / (s + 1) }
	hh := func(s complex128) complex128 { return cmplx.Exp(-3*s) / (s + 2) }
	for _, sign := range []float64{-1, 1} {
		label := fmt.Sprintf("sign=%v", sign)
		cl, err := controlsys.Feedback(sweepTopologySISO(t, 1, 2), sweepTopologySISO(t, 2, 3), sign)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		sweepTopologyAssertFreq(t, label, cl, omega, func(s complex128) sweepTopologyMat {
			return sweepTopologyMat{{g(s) / (1 - complex(sign, 0)*g(s)*hh(s))}}
		}, sweepTopologyFreqTol)
		if !cl.HasInternalDelay() || cl.Delay != nil || cl.InputDelay != nil || cl.OutputDelay != nil {
			t.Errorf("%s: want internal delays only, got io=%v in=%v out=%v lft=%v", label, cl.Delay, cl.InputDelay, cl.OutputDelay, cl.LFT)
		}
		if _, err := cl.TransferFunction(nil); err == nil {
			t.Errorf("%s: TransferFunction of closed loop with internal delays: want error", label)
		}

		const tFinal = 15.0
		const href = 1e-4
		ref := sweepTopologyDDEStep(sign, href, tFinal)
		var errs []float64
		for _, dt := range []float64{0.01, 0.005} {
			tg := sweepTopologyGrid(dt, tFinal)
			y := sweepTopologyLsimStep(t, label, cl, 0, tg)[0]
			stride := int(math.Round(dt / href))
			late := 0.0
			for k, tt := range tg {
				switch {
				case tt < 2-1e-9:
					if y[k] != 0 {
						t.Fatalf("%s dt=%g: y(%g) = %g, want exact 0 before forward delay 2", label, dt, tt, y[k])
					}
				case tt < 7-1e-9:
					if e := math.Abs(y[k] - (1 - math.Exp(-(tt - 2)))); e > 1e-9 {
						t.Errorf("%s dt=%g: y(%g) = %.12g, want 1−e^{−(t−2)} (loop not closed before 7), err %.3e", label, dt, tt, y[k], e)
					}
				default:
					late = max(late, math.Abs(y[k]-ref[k*stride]))
				}
			}
			if late > 0.25*dt {
				t.Errorf("%s dt=%g: worst error on [7,15] %.3e > %.3e", label, dt, late, 0.25*dt)
			}
			errs = append(errs, late)
		}
		t.Logf("%s: worst Lsim error on [7,15] vs DDE reference, dt=0.01/0.005: %.3e/%.3e", label, errs[0], errs[1])
		if ratio := errs[0] / errs[1]; errs[0] > 1e-6 && (ratio < 1.6 || ratio > 2.6) {
			t.Errorf("%s: error ratio dt 0.01/0.005 = %.3f (errs %v), want first-order ≈2", label, ratio, errs)
		}

		resp, err := controlsys.Step(cl, tFinal)
		if err != nil {
			t.Fatal(err)
		}
		for k, tt := range resp.T {
			if tt < 2-1e-9 && resp.Y.At(0, k) != 0 {
				t.Fatalf("%s: Step y(%g) = %g, want exact 0", label, tt, resp.Y.At(0, k))
			}
		}
	}
}

// TestSweepTopologyMIMOFeedback (B-09): Feedback of the delayed 2×2 G with a
// static K and a delayed dynamic K gives (I∓GK)^{-1}G for sign ∓1; Connect on
// Append(G,K) and ConnectByName with a SumBlk reproduce the same closed loops.
func TestSweepTopologyMIMOFeedback(t *testing.T) {
	omega := sweepTopologyOmega(0.5)
	kStatic := sweepTopologyMat{{0.5, 0.1}, {0.2, 0.4}}
	kDyn := func(s complex128) sweepTopologyMat {
		return sweepTopologyMat{
			{0.8 * cmplx.Exp(-0.5*s) / (s + 1), 0},
			{0.1, 0.3 * (s + 2) / (s + 5)},
		}
	}
	kDynSys := func() *controlsys.System {
		tf := &controlsys.TransferFunc{
			Num:   [][][]float64{{{0, 0.8}, {0, 0}}, {{0.1, 0.5}, {0.3, 0.6}}},
			Den:   [][]float64{{1, 1}, {1, 5}},
			Delay: [][]float64{{0.5, 0}, {0, 0}},
		}
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}()
	controllers := map[string]struct {
		sys *controlsys.System
		ref func(complex128) sweepTopologyMat
	}{
		"static":  {sweepTopologyGain(t, 2, 2, []float64{0.5, 0.1, 0.2, 0.4}), func(complex128) sweepTopologyMat { return kStatic }},
		"dynamic": {kDynSys, kDyn},
	}
	closed := func(k func(complex128) sweepTopologyMat, sign float64) func(complex128) sweepTopologyMat {
		return func(s complex128) sweepTopologyMat {
			gs := sweepTopologyG.eval(s)
			return sweepTopologyMul(sweepTopologyInv2(sweepTopologyAdd(sweepTopologyEye(2), sweepTopologyMul(gs, k(s)), complex(-sign, 0))), gs)
		}
	}
	for gName, g := range map[string]*controlsys.System{
		"io":       sweepTopologyFOSS(t, sweepTopologyG),
		"factored": sweepTopologyGFactored(t),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
	} {
		for kName, k := range controllers {
			sweepTopologyAssertFreq(t, kName+" K alone", k.sys, omega, k.ref, sweepTopologyFreqTol)
			for _, sign := range []float64{-1, 1} {
				label := fmt.Sprintf("%s/%s sign=%v", gName, kName, sign)
				cl := sweepTopologyMust(t, label)(controlsys.Feedback(g, k.sys, sign))
				sweepTopologyAssertFreq(t, label+" Feedback", cl, omega, closed(k.ref, sign), sweepTopologyFreqTol)
			}
			blk := sweepTopologyMust(t, "Append")(controlsys.Append(g, k.sys))
			conn := sweepTopologyMust(t, gName+"/"+kName+" Connect")(controlsys.Connect(blk, []controlsys.Junction{
				{Input: 0, Minus: []int{2}},
				{Input: 1, Minus: []int{3}},
				{Input: 2, Plus: []int{0}},
				{Input: 3, Plus: []int{1}},
			}, []int{0, 1}, []int{0, 1}))
			sweepTopologyAssertFreq(t, gName+"/"+kName+" Connect", conn, omega, closed(k.ref, -1), sweepTopologyFreqTol)
		}

		sum, err := controlsys.SumBlk("e = r - y", 2)
		if err != nil {
			t.Fatal(err)
		}
		gn := g.Copy()
		gn.InputName = []string{"e(1)", "e(2)"}
		gn.OutputName = []string{"y(1)", "y(2)"}
		cbn, err := controlsys.ConnectByName([]*controlsys.System{sum, gn}, []controlsys.Connection{
			{From: "e(1)", To: "e(1)"}, {From: "e(2)", To: "e(2)"},
			{From: "y(1)", To: "y(1)"}, {From: "y(2)", To: "y(2)"},
		}, []string{"r(1)", "r(2)"}, []string{"y(1)", "y(2)"})
		if err != nil {
			t.Fatalf("%s ConnectByName: %v", gName, err)
		}
		sweepTopologyAssertFreq(t, gName+" SumBlk unity feedback", cbn, omega,
			closed(func(complex128) sweepTopologyMat { return sweepTopologyEye(2) }, -1), sweepTopologyFreqTol)
	}
}

// TestSweepTopologySimilarityTransform (B-10): SS2SS with a dense
// non-orthogonal circulant-pattern T maps (A,B,C) → (TAT⁻¹,TB,CT⁻¹), internal channels
// (B2,C2) → (TB2,C2T⁻¹), keeps every delay exactly, and preserves G(jω).
func TestSweepTopologySimilarityTransform(t *testing.T) {
	omega := sweepTopologyOmega(1)
	prod := func(ms ...mat.Matrix) *mat.Dense {
		var r mat.Dense
		r.CloneFrom(ms[0])
		for _, m := range ms[1:] {
			var n mat.Dense
			n.Mul(&r, m)
			r = n
		}
		return &r
	}
	io := sweepTopologyFOSS(t, sweepTopologyG)
	pulled, err := sweepTopologyGFactored(t).PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*controlsys.System{
		"iodelay":  io,
		"factored": sweepTopologyGFactored(t),
		"tf-ss":    sweepTopologyFOTFSS(t, sweepTopologyG),
		"internal": sweepTopologyFOInternal(t, sweepTopologyG),
		"pulled":   pulled,
	} {
		n, _, _ := sys.Dims()
		T := mat.NewDense(n, n, nil)
		for i := range n {
			T.Set(i, i, 2+0.1*float64(i))
			T.Set(i, (i+1)%n, 0.7)
			T.Set(i, (i+3)%n, -0.4)
		}
		var Ti mat.Dense
		if err := Ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		tr, err := controlsys.SS2SS(sys, T)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		sweepTopologyAssertFreq(t, name+" SS2SS", tr, omega, sweepTopologyG.eval, sweepTopologyFreqTol)
		sweepTopologyAssertDense(t, name+" A", tr.A, prod(T, sys.A, &Ti), sweepTopologyExactTol)
		sweepTopologyAssertDense(t, name+" B", tr.B, prod(T, sys.B), sweepTopologyExactTol)
		sweepTopologyAssertDense(t, name+" C", tr.C, prod(sys.C, &Ti), sweepTopologyExactTol)
		sweepTopologyAssertDense(t, name+" D", tr.D, sys.D, 0)
		sweepTopologyAssertDense(t, name+" Delay", tr.Delay, sys.Delay, 0)
		if fmt.Sprint(tr.InputDelay, tr.OutputDelay) != fmt.Sprint(sys.InputDelay, sys.OutputDelay) {
			t.Errorf("%s: in/out delays %v %v, want %v %v", name, tr.InputDelay, tr.OutputDelay, sys.InputDelay, sys.OutputDelay)
		}
		if sys.HasInternalDelay() {
			if !tr.HasInternalDelay() || fmt.Sprint(tr.LFT.Tau) != fmt.Sprint(sys.LFT.Tau) {
				t.Fatalf("%s: internal tau changed: %v", name, tr.LFT)
			}
			sweepTopologyAssertDense(t, name+" B2", tr.LFT.B2, prod(T, sys.LFT.B2), sweepTopologyExactTol)
			sweepTopologyAssertDense(t, name+" C2", tr.LFT.C2, prod(sys.LFT.C2, &Ti), sweepTopologyExactTol)
			sweepTopologyAssertDense(t, name+" D12", tr.LFT.D12, sys.LFT.D12, 0)
			sweepTopologyAssertDense(t, name+" D21", tr.LFT.D21, sys.LFT.D21, 0)
			sweepTopologyAssertDense(t, name+" D22", tr.LFT.D22, sys.LFT.D22, 0)
		}
	}
}
