package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

func negatedPlant(t *testing.T, num, den []float64, dt float64) (pos, neg *System) {
	t.Helper()
	build := func(n []float64) *System {
		r, err := (&TransferFunc{Num: [][][]float64{{n}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return r.Sys
	}
	n := make([]float64, len(num))
	for i, v := range num {
		n[i] = -v
	}
	return build(num), build(n)
}

// XYTCJ7: MATLAB pidtune returns negative gains for a stable negative-DC plant.
func TestPidtuneNegativeDCDelayedPI(t *testing.T) {
	p := makePlant(t, []float64{-1}, []float64{1, 1})
	if err := p.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	pid, err := Pidtune(p, PidtunePI, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Kp >= 0 || pid.Ki >= 0 {
		t.Fatalf("gains %+v, want both negative", pid)
	}
	stable, _ := pidDelayLoopCheckPole(t, &PID2{Kp: -pid.Kp, Ki: -pid.Ki}, 1, 1)
	if !stable {
		t.Fatalf("closed loop of %+v unstable by winding check", pid)
	}
}

// Pidtune(−G) mirrors Pidtune(G) for stable plants with nonzero DC gain.
func TestPidtuneNegativeDCMirrorsPositive(t *testing.T) {
	plants := []struct {
		name     string
		num, den []float64
		dt       float64
	}{
		{"fo", []float64{2}, []float64{1, 10}, 0},
		{"so", []float64{1}, []float64{1, 1.4, 1}, 0},
		{"third", []float64{1}, []float64{1, 3, 3, 1}, 0},
		{"lead", []float64{1, 1}, []float64{1, 5, 6}, 0},
		{"dso", []float64{0.01, 0.009}, []float64{1, -1.7, 0.72}, 0.1},
	}
	families := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	checked := 0
	for _, pl := range plants {
		pos, neg := negatedPlant(t, pl.num, pl.den, pl.dt)
		for _, f := range families {
			for _, wc := range []float64{0, 0.3} {
				cp, errP := Pidtune(pos, f, wc, nil)
				cn, errN := Pidtune(neg, f, wc, nil)
				if (errP == nil) != (errN == nil) {
					t.Errorf("%s %s wc=%g: pos err %v, neg err %v", pl.name, f, wc, errP, errN)
					continue
				}
				if errP != nil {
					continue
				}
				checked++
				for _, g := range [][2]float64{{cp.Kp, cn.Kp}, {cp.Ki, cn.Ki}, {cp.Kd, cn.Kd}} {
					if math.Abs(g[0]+g[1]) > 1e-12*math.Max(1, math.Abs(g[0])) {
						t.Errorf("%s %s wc=%g: pos %+v, neg %+v not mirrored", pl.name, f, wc, cp, cn)
						break
					}
				}
				if cn.Tf != cp.Tf || cn.DFormula != cp.DFormula {
					t.Errorf("%s %s wc=%g: Tf/DFormula %v/%v vs %v/%v", pl.name, f, wc, cn.Tf, cn.DFormula, cp.Tf, cp.DFormula)
				}
				if cn.Kp > 0 || cn.Ki > 0 || cn.Kd > 0 {
					t.Errorf("%s %s wc=%g: positive gain on negative-DC plant %+v", pl.name, f, wc, cn)
				}
			}
		}
	}
	if checked < 40 {
		t.Fatalf("only %d designs compared", checked)
	}
}

// Routh: 1/(s+1)^3 under P is stable iff Kp < 8; the wc=2 design needs
// Kp = |G(2j)|⁻¹ = 5√5 and must be rejected, not returned.
func TestPidtuneRejectsUnstableDesign(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	pid, err := Pidtune(p, PidtuneP, 2, nil)
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("got %+v, %v; want unattainable (Kp=%g exceeds Routh bound 8)", pid, err, 5*math.Sqrt(5))
	}
	pid, err = Pidtune(p, PidtuneP, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * math.Sqrt(2); math.Abs(pid.Kp-want) > 1e-9 {
		t.Fatalf("Kp=%g, want %g", pid.Kp, want)
	}
}

// Every design Pidtune returns on a delay-free continuous plant has a
// Hurwitz characteristic polynomial den_G·den_C + num_G·num_C.
func TestPidtuneReturnsOnlyStableDesigns(t *testing.T) {
	plants := []struct{ num, den []float64 }{
		{[]float64{1}, []float64{1, .4, 1}},
		{[]float64{1}, []float64{1, 3, 3, 1}},
		{[]float64{1}, []float64{1, 4, 6, 4, 1}},
		{[]float64{-1, 1}, []float64{1, 3, 2}},
		{[]float64{1}, []float64{1, -1}},
		{[]float64{1}, []float64{1, 1, -2}},
		{[]float64{1}, []float64{1, 1, 0}},
	}
	families := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	rejected := 0
	for _, pl := range plants {
		for _, sgn := range []float64{1, -1} {
			num := make([]float64, len(pl.num))
			for i, v := range pl.num {
				num[i] = sgn * v
			}
			p := makePlant(t, num, pl.den)
			for _, f := range families {
				for _, wc := range []float64{0, 0.3, 2} {
					pid, err := Pidtune(p, f, wc, nil)
					if err != nil {
						if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
							t.Errorf("%v/%v %s wc=%g: %v", num, pl.den, f, wc, err)
						}
						rejected++
						continue
					}
					tf := pid.Tf
					numC := Poly{pid.Kd + pid.Kp*tf, pid.Kp + pid.Ki*tf, pid.Ki}
					denC := Poly{tf, 1, 0}
					if pid.Ki == 0 {
						numC, denC = numC[:2], denC[:2]
					}
					char := Poly(pl.den).Mul(denC).Add(Poly(num).Mul(numC))
					for len(char) > 1 && math.Abs(char[0]) < 1e-14 {
						char = char[1:]
					}
					roots, err := char.Roots()
					if err != nil {
						t.Fatal(err)
					}
					for _, r := range roots {
						if real(r) >= -1e-9 || cmplx.IsNaN(r) {
							t.Errorf("%v/%v %s wc=%g: %+v closed-loop root %v", num, pl.den, f, wc, pid, r)
							break
						}
					}
				}
			}
		}
	}
	if rejected == 0 {
		t.Fatal("expected some designs to be rejected")
	}
}

// e^{-s}/(s+1) with an ideal PID has limsup |L(jω)| = |Kd|: Kd < 1 is a
// stable neutral chain, Kd > 1 is unstable. Pidtune must not apply TunePID's
// 0.75 robustness cap.
func TestPidtuneDelayedIdealPIDTail(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	if err := p.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	pid, err := Pidtune(p, PidtunePID, 2, &PidtuneOptions{PhaseMargin: 45})
	if err != nil {
		t.Fatal(err)
	}
	if pid.Kd <= pidTuningMaxTail || pid.Kd >= 1 {
		t.Fatalf("Kd=%g, want in (%g, 1) to exercise the tail", pid.Kd, pidTuningMaxTail)
	}
	if stable, _ := pidDelayLoopCheckPole(t, &PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd}, 1, 1); !stable {
		t.Fatalf("%+v unstable by winding check", pid)
	}
	pid, err = Pidtune(p, PidtunePID, 2, nil)
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("got %+v, %v; want unattainable (Kd > 1)", pid, err)
	}
}

// YVAIMU: Pidtune(−G) mirrors Pidtune(G) for integrating and unstable
// plants too; −1/(s(s+1)), −1/(s−1) and −0.1/(z−1) need negative gains.
func TestPidtuneUnstableMirrorsPositive(t *testing.T) {
	plants := []struct {
		name     string
		num, den []float64
		dt       float64
		mustTune []PidtuneType
	}{
		{"integ", []float64{1}, []float64{1, 1, 0}, 0, []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePID, PidtunePIDF}},
		{"unst", []float64{1}, []float64{1, -1}, 0, []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePID}},
		{"dinteg", []float64{.1}, []float64{1, -1}, .1, []PidtuneType{PidtuneP, PidtunePI}},
		{"nmp", []float64{1, -3}, []float64{1, 1, -2}, 0, nil},
		{"dbl", []float64{1, 2}, []float64{1, 1, 0, 0}, 0, nil},
		{"cplx", []float64{1, 1}, []float64{1, 4.5, 1.5, 20}, 0, nil},
	}
	families := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	checked := 0
	for _, pl := range plants {
		pos, neg := negatedPlant(t, pl.num, pl.den, pl.dt)
		for _, f := range families {
			cp, errP := Pidtune(pos, f, 0, nil)
			cn, errN := Pidtune(neg, f, 0, nil)
			if (errP == nil) != (errN == nil) {
				t.Errorf("%s %s: pos err %v, neg err %v", pl.name, f, errP, errN)
				continue
			}
			if errP != nil {
				if slices.Contains(pl.mustTune, f) {
					t.Errorf("%s %s: %v", pl.name, f, errP)
				}
				continue
			}
			checked++
			for _, g := range [][2]float64{{cp.Kp, cn.Kp}, {cp.Ki, cn.Ki}, {cp.Kd, cn.Kd}} {
				if math.Abs(g[0]+g[1]) > 1e-12*math.Max(1, math.Abs(g[0])) {
					t.Errorf("%s %s: pos %+v, neg %+v not mirrored", pl.name, f, cp, cn)
					break
				}
			}
			if cn.Tf != cp.Tf {
				t.Errorf("%s %s: Tf %v vs %v", pl.name, f, cn.Tf, cp.Tf)
			}
		}
	}
	if checked < 20 {
		t.Fatalf("only %d designs compared", checked)
	}
}

// Brute force: every stabilizing P gain (Ki of every stabilizing PI pair)
// has the sign pidtuneGainSign picks, judged by the roots of
// den·denC + num·numC.
func TestPidtuneGainSignMatchesStabilizingGains(t *testing.T) {
	plants := []struct {
		num, den []float64
		dt       float64
	}{
		{[]float64{1}, []float64{1, -1}, 0},
		{[]float64{1, -3}, []float64{1, 1, -2}, 0},
		{[]float64{1}, []float64{1, 1, 0}, 0},
		{[]float64{2, 1}, []float64{1, -3, 2, 0}, 0},
		{[]float64{1, 4}, []float64{1, 2, -5, -6}, 0},
		{[]float64{.1}, []float64{1, -1}, .1},
		{[]float64{.2, -.1}, []float64{1, -2.3, 1.2}, .1},
		{[]float64{1, .5}, []float64{1, -1.5, .5}, .5},
		{[]float64{2.7, 4.239, -1.6929}, []float64{1, 9.88, 27.4856, 404.15328}, 0},
	}
	stableRoots := func(char Poly, dt float64) bool {
		for len(char) > 1 && char[0] == 0 {
			char = char[1:]
		}
		roots, err := char.Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range roots {
			if dt == 0 && real(r) >= -1e-9 || dt > 0 && cmplx.Abs(r) >= 1-1e-9 {
				return false
			}
		}
		return true
	}
	var gains []float64
	for e := -2.0; e <= 2; e += .05 {
		gains = append(gains, math.Pow(10, e), -math.Pow(10, e))
	}
	found := map[bool]int{}
	for _, pl := range plants {
		for _, sgn := range []float64{1, -1} {
			num := make(Poly, len(pl.num))
			for i, v := range pl.num {
				num[i] = sgn * v
			}
			p, _ := negatedPlant(t, num, pl.den, pl.dt)
			eval, err := newSISOEval(p)
			if err != nil {
				t.Fatal(err)
			}
			for _, integral := range []bool{false, true} {
				want := pidtuneGainSign(p, eval.at, 1, integral)
				kis := []float64{0}
				if integral {
					kis = gains
				}
				for _, kp := range gains {
					for _, ki := range kis {
						numC, denC := Poly{kp}, Poly{1}
						if integral && pl.dt == 0 {
							numC, denC = Poly{kp, ki}, Poly{1, 0}
						} else if integral {
							numC, denC = Poly{kp, ki*pl.dt - kp}, Poly{1, -1}
						}
						if !stableRoots(Poly(pl.den).Mul(denC).Add(num.Mul(numC)), pl.dt) {
							continue
						}
						found[integral]++
						g := kp
						if integral {
							g = ki
						}
						if math.Copysign(1, g) != want {
							t.Errorf("%v/%v dt=%g: stabilizing Kp=%g Ki=%g against sign %g", num, pl.den, pl.dt, kp, ki, want)
						}
					}
				}
			}
		}
	}
	if found[false] < 100 || found[true] < 100 {
		t.Fatalf("too few stabilizing gains: %v", found)
	}
}
