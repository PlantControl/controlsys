package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorDCBars is the worst entry of |ĥ − g0|/(bar₀ + allowance) for
// the discrete result out of the continuous descriptor sys: ĥ and g0 exact
// for the stored entries (fraction-free solves), bar₀ the descriptor bar
// u·(‖C_i‖‖M‖‖A‖‖M‖‖B_j‖ + ‖C_i‖‖M‖‖B_j‖ + |D_ij|), M = A⁻¹, and the
// allowance ε·S of out, S = |D| + Σ|C||X| + |C(I−Φ)⁻¹|(|Φ||X| + |Γ|),
// where it is within √ε·max(|g0|, T0) (the C2D DC contract), both formed
// here in float64 from mat's inverse. ok is false where an exact gain or an
// inverse is singular.
func descriptorDCBars(t *testing.T, sys, out *System) (float64, bool) {
	t.Helper()
	free := ioDelayFree(sys)
	g0, _, ok0 := exactDCGainAtNonsingular(nil, free.A, free.B, free.C, free.D, 0)
	h, _, ok := exactDCGainAtNonsingular(nil, out.A, out.B, out.C, out.D, 1)
	if !ok0 || !ok {
		return 0, false
	}
	n, m, p := free.Dims()
	nd, _, _ := out.Dims()
	var M, X0, W, X, Y mat.Dense
	if M.Inverse(free.A) != nil {
		return 0, false
	}
	X0.Mul(&M, free.B)
	ip := mat.NewDense(nd, nd, nil)
	for i := range nd {
		for j := range nd {
			ip.Set(i, j, -out.A.At(i, j))
		}
		ip.Set(i, i, ip.At(i, i)+1)
	}
	if W.Inverse(ip) != nil {
		return 0, false
	}
	X.Mul(&W, out.B)
	Y.Mul(out.C, &W)
	u := 4 * float64(n+5) * 0x1p-52
	normM, normA := mat.Norm(&M, 2), mat.Norm(free.A, 2)
	worst := 0.0
	for i := range p {
		ci := mat.Norm(free.C.Slice(i, i+1, 0, n), 2)
		for j := range m {
			bj := mat.Norm(free.B.Slice(0, n, j, j+1), 2)
			d := math.Abs(free.D.At(i, j))
			bar := u * (ci*normM*normA*normM*bj + ci*normM*bj + d)
			t0 := d
			for l := range n {
				t0 += math.Abs(free.C.At(i, l) * X0.At(l, j))
			}
			s := math.Abs(out.D.At(i, j))
			for l := range nd {
				w := math.Abs(out.B.At(l, j))
				for k := range nd {
					w += math.Abs(out.A.At(l, k) * X.At(k, j))
				}
				s += math.Abs(out.C.At(i, l)*X.At(l, j)) + math.Abs(Y.At(i, l))*w
			}
			a := 0x1p-52 * s
			if !(a <= math.Sqrt(0x1p-52)*math.Max(math.Abs(g0.At(i, j)), t0)) {
				a = 0
			}
			worst = math.Max(worst, math.Abs(h.At(i, j)-g0.At(i, j))/(bar+a))
		}
	}
	return worst, true
}

// checkDescriptorDCContract asserts the C2D DC contract of a ZOH, FOH or
// Tustin result: nil within the bar, unverified with a model, any other
// error without one. It returns "nil", "unv" or "err".
func checkDescriptorDCContract(t *testing.T, name string, sys, out *System, err error) string {
	t.Helper()
	switch {
	case err == nil:
		if bars, ok := descriptorDCBars(t, sys, out); ok && !(bars <= 1) {
			t.Errorf("%s: nil with DC %.3g bars off DCGain", name, bars)
		}
		return "nil"
	case errors.Is(err, ErrAccuracyUnverified):
		if out == nil {
			t.Errorf("%s: unverified without a model", name)
		}
		return "unv"
	default:
		if out != nil {
			t.Errorf("%s: model with %v", name, err)
		}
		return "err"
	}
}

// Every ZOH, FOH and Tustin C2D of the frozen EJDS3U census rows meets
// the DC contract, and the rows the PR-1 routes fix keep their verdicts:
//   - stfr/45, /91 (rank-deficient E, triangular standard form with the
//     stiff mode leading): nil within the bar once the hold keeps that
//     order (R1); base was 5e10 and 4e6 bars;
//   - cond/13 (LU cond 4.6e5): nil through the pencil parts (R4); the LU
//     fold was 65 bars at dt = 0.01;
//   - cond/644 (LU cond 2.4e3, ‖A·dt‖ large): ZOH at dt = 1 nil in the real
//     Schur coordinates of E⁻¹A, without state names (R2); the fold was
//     2e12 bars;
//   - cond/111 at dt = 0.01: 13.9 bars in fold and 41 in Schur
//     coordinates, unverified (R5);
//   - strad1m/47, /283 (band, split refused): the fold, always unverified
//     (U-2);
//   - stfr/3408, /4980 as generated on amd64 (‖B‖ ≫ ‖A‖ reductions whose
//     augmented exponential loses Γ, O8): nil only inside the bar.
func TestC2DDescriptorDCContractFrozenRows(t *testing.T) {
	want := map[string]string{
		"stfr/45@0.01/zoh": "nil", "stfr/45@0.01/foh": "nil", "stfr/45@1/zoh": "nil",
		"stfr/91@0.01/zoh": "nil", "stfr/91@1/zoh": "nil",
		"cond/13@0.01/zoh": "nil", "cond/13@0.01/foh": "nil", "cond/13@1/zoh": "nil",
		"cond/644@1/zoh": "nil", "cond/644@1/tustin": "nil",
		"cond/111@0.01/zoh": "unv", "cond/111@0.01/foh": "unv",
		"strad1m/47@0.0001/zoh": "unv", "strad1m/47@0.01/foh": "unv", "strad1m/47@0.01/tustin": "unv",
		"strad1m/283@1/zoh": "unv",
	}
	for _, r := range descriptorDCGateRows {
		sys, err := r.system()
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				key := fmt.Sprintf("%s@%g/%s", r.name, dt, m)
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				got := checkDescriptorDCContract(t, key, sys, out, err)
				if w, ok := want[key]; ok && got != w {
					t.Errorf("%s: %s (%v), want %s", key, got, err, w)
				}
				if key == "cond/644@1/zoh" && out != nil && out.StateName != nil {
					t.Errorf("%s: state names %v, want none (real Schur coordinates)", key, out.StateName)
				}
			}
		}
	}
}

// C2DMap returns C2D's model and verdict (T3IVBX): cond/13 at dt = 0.01,
// 65 bars off on the LU fold, takes the band parts and is nil; within the
// bar it stays nil. A model outside the bar would come with its G.
func TestC2DMapDescriptorDCVerdict(t *testing.T) {
	for _, tc := range []struct {
		row  string
		dt   float64
		want string
	}{{"cond/13", 0.01, "nil"}, {"cond/644", 0.01, "nil"}} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == tc.row {
				var err error
				if sys, err = r.system(); err != nil {
					t.Fatal(err)
				}
			}
		}
		out, g, err := sys.C2DMap(tc.dt, C2DOptions{})
		got := checkDescriptorDCContract(t, tc.row, sys, out, err)
		if got != tc.want {
			t.Errorf("%s: %s (%v), want %s", tc.row, got, err, tc.want)
		}
		if got == "unv" && g == nil {
			t.Errorf("%s: unverified without G", tc.row)
		}
	}
}

// mimoDescriptor is a 2×2 dense descriptor with E = U·diag(1, …, 10^-x)·Wᵀ.
func mimoDescriptor(t *testing.T, x float64) *System {
	t.Helper()
	A := mat.NewDense(4, 4, []float64{
		-0.9, 0.3, -0.2, 0.4,
		0.5, -1.1, 0.25, -0.3,
		-0.1, 0.6, -0.7, 0.2,
		0.3, -0.2, 0.45, -1.3})
	c, s := math.Cos(0.7), math.Sin(0.7)
	U := mat.NewDense(4, 4, []float64{c, -s, 0, 0, s, c, 0, 0, 0, 0, c, s, 0, 0, -s, c})
	W := mat.NewDense(4, 4, []float64{c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1})
	var T, E mat.Dense
	T.Mul(U, mat.NewDiagDense(4, []float64{1, 0.6, 0.35, math.Pow(10, -x)}))
	E.Mul(&T, W.T())
	B := mat.NewDense(4, 2, []float64{1, -0.4, 0.3, 0.8, -0.7, 0.2, 0.5, 1.1})
	C := mat.NewDense(2, 4, []float64{0.6, -0.9, 0.4, 0.2, -0.3, 0.5, 1.2, -0.8})
	D := mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3})
	sys, err := NewDescriptor(A, B, C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// The DC check of C2D is entrywise for MIMO descriptors: every entry of a
// nil result lies within its own bar (row C_i, column B_j, |D_ij|), from
// a well-conditioned E through the band to cond 1e12.
func TestC2DDescriptorDCContractMIMO(t *testing.T) {
	nils := 0
	for _, x := range []float64{1, 4.5, 5.5, 8, 12} {
		sys := mimoDescriptor(t, x)
		for _, dt := range []float64{1e-4, 1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				if checkDescriptorDCContract(t, fmt.Sprintf("x=%g dt=%g %s", x, dt, m), sys, out, err) == "nil" {
					nils++
				}
			}
		}
	}
	if nils == 0 {
		t.Error("no nil result; the check is not exercised")
	}
}

// integratorDescriptor has an exact pole at s = 0 (first column of A
// zero) and an ill-conditioned triangular E; with decoupled it has a
// second output reached only by a state no path joins to the integrator.
func integratorDescriptor(t *testing.T, decoupled bool) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{0, 0.4, -0.3, 0, -1.2, 0.5, 0, 0.2, -0.8})
	E := mat.NewDense(3, 3, []float64{1, 0.3, -0.2, 0, 1, 0.4, 0, 0, 1e-7})
	B := mat.NewDense(3, 1, []float64{0.7, -1.1, 0.9})
	C := mat.NewDense(1, 3, []float64{1, 0.5, -0.4})
	D := mat.NewDense(1, 1, []float64{0.2})
	if decoupled {
		A.Set(2, 1, 0)
		A.Set(0, 2, 0)
		A.Set(1, 2, 0)
		E.Set(0, 2, 0)
		E.Set(1, 2, 0)
		C = mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 0, 0.6})
		D = mat.NewDense(2, 1, []float64{0.2, -0.1})
	}
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// An entry of DCGain that is infinite (a pole at DC) is not checked: a
// SISO descriptor with an integrator keeps its nil ZOH, FOH and Tustin.
// Where another entry is finite no bar exists for it, and the result is
// unverified.
func TestC2DDescriptorDCContractIntegrator(t *testing.T) {
	siso := integratorDescriptor(t, false)
	g, err := siso.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) {
		t.Fatalf("setup: DCGain %v, %v; want ±Inf", g, err)
	}
	mixed := integratorDescriptor(t, true)
	g, err = mixed.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) || math.IsInf(g.At(1, 0), 0) {
		t.Fatalf("setup: DCGain %v, %v; want ±Inf and a finite entry", g, err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		if out, err := siso.C2D(0.1, C2DOptions{Method: m}); err != nil || out == nil {
			t.Errorf("SISO %s: %v, want nil", m, err)
		}
		if out, err := mixed.C2D(0.1, C2DOptions{Method: m}); !errors.Is(err, ErrAccuracyUnverified) || out == nil {
			t.Errorf("mixed %s: %v, want a model with ErrAccuracyUnverified", m, err)
		}
	}
}

// The DC check reserves the rounding bounds of both gains: for a result
// whose DC exceeds g0 by δ, nil requires δ + err(ĥ) + err(g0) ≤ bar +
// allowance, so the largest δ it accepts leaves that room. Caller ẋ = −x + u
// (E = I as a descriptor, g0 = 1, T0 = 1); result Φ = 1/2, Γ = (1+δ)/2,
// whose DC 1 + δ is exact.
func TestC2DDescriptorDCGateErrorBounds(t *testing.T) {
	one := func(v float64) *mat.Dense { return mat.NewDense(1, 1, []float64{v}) }
	sys, err := NewDescriptor(one(-1), one(1), one(1), one(0), one(1), 0)
	if err != nil {
		t.Fatal(err)
	}
	gate := newHoldGate(sys)
	bar := gate.bar.fullBar(0)
	accepted := -1.0
	for k := range 200 {
		delta := float64(k) * 0x1p-52
		out := &System{A: one(0.5), B: one((1 + delta) / 2), C: one(1), D: one(0), Dt: 0.1}
		if gate.check(out) != nil {
			break
		}
		accepted = delta
	}
	if accepted < 0 {
		t.Fatal("the exact gain is rejected")
	}
	edge := &System{A: one(0.5), B: one((1 + accepted) / 2), C: one(1), D: one(0), Dt: 0.1}
	_, herr, _, _ := holdDC(edge)
	limit := bar + holdAllowance(edge)[0] - herr[0] - 0x1p-52*1 - halfUlp(1)
	if !(accepted <= limit+0x1p-53) || !(accepted+0x1p-52 > limit-0x1p-53) {
		t.Errorf("largest δ accepted %.4g, want the bar less both rounding bounds, %.4g", accepted, limit)
	}
}

// A pole at DC beside a finite entry leaves that entry without a bar, so
// the result is unverified. A's 2×2 block is the singular [3 6; 1 2]: with
// a fused multiply-add the LU pivot 2 − (1/3)·6 is 2⁻⁵² and not 0, so the
// descriptor bar of that LU would pass anything; without one it is 0.
func TestC2DDescriptorDCContractPoleBesideFiniteEntry(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-3, -6, 0,
		-1, -2, 0,
		0, 0, -1.5})
	E := mat.NewDense(3, 3, []float64{1, 0.25, 0, 0, 1, 0, 0, 0, 0x1p-20})
	B := mat.NewDense(3, 1, []float64{1, -0.5, 0.75})
	C := mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 0, 1})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(2, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) || math.IsInf(g.At(1, 0), 0) {
		t.Fatalf("setup: DCGain %v, %v; want ±Inf and a finite entry", g, err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		if out, err := sys.C2D(0.1, C2DOptions{Method: m}); !errors.Is(err, ErrAccuracyUnverified) || out == nil {
			t.Errorf("%s: %v, want a model with ErrAccuracyUnverified", m, err)
		}
	}
}

// Where DCGain of the model is unverified (a pole within rounding of DC
// whose exact limit is finite), C2D cannot check the result: a model with
// ErrAccuracyUnverified, never nil.
func TestC2DDescriptorDCContractDCGainUnverified(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.3, 0.2, 0.4, -2, 0.1, 0, 0, 0})
	for j := range 3 {
		A.Set(2, j, 0.3*A.At(0, j)+0.7*A.At(1, j))
	}
	E := mat.NewDense(3, 3, []float64{1, 0.2, 0, 0, 1, 0, 0, 0, 1e-6})
	B := mat.NewDense(3, 2, []float64{1, 0.5, -0.3, 0.8, 0.2, -0.4})
	C := mat.NewDense(2, 3, []float64{1, 0, 0.5, 0, 1, -0.2})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(2, 2, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sys.DCGain(); !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("setup: DCGain error %v, want ErrAccuracyUnverified", err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		if _, err := sys.C2D(0.1, C2DOptions{Method: m}); err == nil {
			t.Errorf("%s: nil with DCGain unverified", m)
		}
	}
}

// A descriptor without inputs or outputs has no DC gain to check (DCGain
// returns ErrDimensionMismatch): C2D stays nil.
func TestC2DDescriptorDCContractNoChannels(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.3, 0.2, 0.4, -2, 0.1, 0, 0, -0.5})
	E := mat.NewDense(3, 3, []float64{1, 0.2, 0, 0, 1, 0, 0, 0, 1e-6})
	for _, d := range [][2]int{{0, 1}, {1, 0}} {
		m, p := d[0], d[1]
		sys, err := NewDescriptor(A, newDense(3, m), newDense(p, 3), newDense(p, m), E, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
			if out, err := sys.C2D(0.1, C2DOptions{Method: meth}); err != nil || out == nil {
				t.Errorf("m=%d p=%d %s: %v", m, p, meth, err)
			}
		}
	}
}

// With fractional I/O delays the model returned, its internal delays
// closed, is checked: cond/111 at dt = 0.01 is 13.9 bars in fold and 41 in
// Schur coordinates, so with a 0.25-sample input delay ZOH and FOH (which
// holds the delay channel by channel) stay unverified, while cond/644,
// 1.2 bars in fold coordinates, stays nil.
func TestC2DDescriptorDCContractFractionalInputDelay(t *testing.T) {
	for _, tc := range []struct {
		row  string
		want string
	}{{"cond/111", "unv"}, {"cond/644", "nil"}} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == tc.row {
				var err error
				if sys, err = r.system(); err != nil {
					t.Fatal(err)
				}
			}
		}
		sys.InputDelay = []float64{0.0025}
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
			out, err := sys.C2D(0.01, C2DOptions{Method: m})
			if out != nil {
				checkDescriptorDCContract(t, tc.row+" "+string(m), sys, delayClosedAtDC(t, out), err)
			}
			got := "err"
			switch {
			case err == nil:
				got = "nil"
			case errors.Is(err, ErrAccuracyUnverified) && out != nil:
				got = "unv"
			}
			if got != tc.want {
				t.Errorf("%s %s: %s (%v), want %s", tc.row, m, got, err, tc.want)
			}
		}
	}
}

// delayClosedAtDC returns the discrete out with its internal delays closed
// as identity (z = 1) without rounding: the delayed signals w become states
// of x⁺ = A·x + B2·w + B·u, w⁺ = C2·x + D22·w + D21·u, whose DC gain is that
// of out. I/O delay fields are dropped (identity at z = 1).
func delayClosedAtDC(t *testing.T, out *System) *System {
	t.Helper()
	if out.LFT == nil {
		c := *out
		c.InputDelay, c.OutputDelay, c.Delay = nil, nil, nil
		return &c
	}
	n, m, p := out.Dims()
	L := out.LFT
	N := len(L.Tau)
	A := mat.NewDense(n+N, n+N, nil)
	B := mat.NewDense(n+N, m, nil)
	C := mat.NewDense(p, n+N, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(out.A)
	A.Slice(0, n, n, n+N).(*mat.Dense).Copy(L.B2)
	A.Slice(n, n+N, 0, n).(*mat.Dense).Copy(L.C2)
	A.Slice(n, n+N, n, n+N).(*mat.Dense).Copy(L.D22)
	B.Slice(0, n, 0, m).(*mat.Dense).Copy(out.B)
	B.Slice(n, n+N, 0, m).(*mat.Dense).Copy(L.D21)
	C.Slice(0, p, 0, n).(*mat.Dense).Copy(out.C)
	C.Slice(0, p, n, n+N).(*mat.Dense).Copy(L.D12)
	return &System{A: A, B: B, C: C, D: out.D, Dt: out.Dt}
}

// A fractional I/O delay held after the real Schur retry is applied in
// the Schur coordinates the model was discretized in: strad1m/17 at
// dt = 0.01 with a 1.2345-sample delay returns ZOH nil, its delay-closed DC
// within the bar (applied in the fold's coordinates instead, DC was 0.064
// against −1.898, 4e11 bars).
func TestC2DDescriptorDCContractDelayAfterSchurRetry(t *testing.T) {
	var sys *System
	for _, r := range descriptorDCGateRows {
		if r.name == "strad1m/17" {
			var err error
			if sys, err = r.system(); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, d := range []struct{ in, out []float64 }{{[]float64{0.012345}, nil}, {nil, []float64{0.012345}}} {
		s := sys.Copy()
		s.InputDelay, s.OutputDelay = d.in, d.out
		out, err := s.C2D(0.01, C2DOptions{})
		if err != nil {
			t.Fatalf("in %v out %v: %v", d.in, d.out, err)
		}
		checkDescriptorDCContract(t, "strad1m/17 delayed", sys, delayClosedAtDC(t, out), nil)
	}
}

// The delays the hold adds can spoil DC that the delay-free core holds:
// stfr/399 with a 2.34-sample output delay at dt = 1 is 8.6e11 bars off
// in the ZOH returned, its core inside the bar. The returned model is what
// is checked, so it is unverified.
func TestC2DDescriptorDCContractDelaySpoilsCore(t *testing.T) {
	var sys *System
	for _, r := range descriptorDCGateRows {
		if r.name == "stfr/399" {
			var err error
			if sys, err = r.system(); err != nil {
				t.Fatal(err)
			}
		}
	}
	core, err := sys.C2D(1, C2DOptions{})
	if err != nil {
		t.Fatalf("setup: delay-free core %v", err)
	}
	checkDescriptorDCContract(t, "stfr/399 core", sys, core, err)
	s := sys.Copy()
	s.OutputDelay = []float64{0.0234}
	out, err := s.C2D(1, C2DOptions{})
	if got := checkDescriptorDCContract(t, "stfr/399 delayed", sys, delayClosedAtDC(t, out), err); got != "unv" {
		t.Errorf("stfr/399 delayed: %s (%v), want unverified", got, err)
	}
}
