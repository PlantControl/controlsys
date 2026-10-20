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
// P and PI on 1/(s−1) keep at most 45° at wc ≈ 1 (PM = atan ω + ∠C) and
// design above √3, mirrored.
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
		{[]float64{.3, -.15}, []float64{1, -1.93062, 1.21}, .1},
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
				want, _ := pidtuneGainSign(p, eval.try, 1, integral)
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

// Brute force over ideal PD and PID gains: the forced sign holds only while
// the closed-loop polynomial's leading coefficient has the sign of a; a
// relative-degree-one plant flips it with Kd·n₁/a < −1. Every stabilizing
// triple against the forced sign must flip it. Oracle: roots of
// den·denC + num·numC.
func TestPidtuneGainSignWithDerivative(t *testing.T) {
	plants := []struct{ num, den Poly }{
		{Poly{-0.38686, 0.064537, -0.71409}, Poly{1, -0.49821, -1.41706, 0.93640}},
		{Poly{1, -3}, Poly{1, 1, -2}},
		{Poly{-1, 2}, Poly{1, -1, 0}},
		{Poly{2, 1}, Poly{1, -3, 2, 0}},
		{Poly{1}, Poly{1, 1, -2}},
	}
	var gains []float64
	for e := -2.0; e <= 2; e += .2 {
		gains = append(gains, math.Pow(10, e), -math.Pow(10, e))
	}
	against, flipped := 0, 0
	for _, pl := range plants {
		p := makePlant(t, pl.num, pl.den)
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		for _, integral := range []bool{false, true} {
			want, rule := pidtuneGainSign(p, eval.try, 1, integral)
			kis := []float64{0}
			if integral {
				kis = gains
			}
			for _, kp := range gains {
				for _, ki := range kis {
					for _, kd := range gains {
						pid := &PID{Kp: kp, Ki: ki, Kd: kd}
						stable, _ := pidtuneOracleStable(t, pid, pl.num, pl.den, 0, 0)
						if !stable {
							continue
						}
						g := kp
						if integral {
							g = ki
						}
						if rule != pidtuneSignForced || math.Copysign(1, g) == want {
							continue
						}
						against++
						numC, denC := pidtuneOracleController(pid, 0)
						lead := pl.den.Mul(denC).Add(pl.num.Mul(numC))[0]
						if lead*pl.den[0] >= 0 {
							t.Errorf("%v/%v: stable %+v against forced sign %g with leading coefficient %g", pl.num, pl.den, pid, want, lead)
						} else {
							flipped++
						}
					}
				}
			}
		}
	}
	if against == 0 || flipped != against {
		t.Fatalf("%d stabilizing designs against the forced sign, %d with a flipped leading coefficient", against, flipped)
	}
}

// YVAIMU with a flipped leading coefficient: on this relative-degree-one
// unstable plant the forced-sign ideal PID is unstable at the automatic
// crossover and at wc = 0.5, while the positive one has a Hurwitz
// characteristic only because 1 + Kd·n₁ < 0. M5IIG3: that design is now
// rejected, since every derivative filter Tf adds a root near
// −(1 + Kd·n₁)/Tf > 0. Oracle: characteristic roots with and without the
// filter, and |L| = 1 with a 60° margin at the crossover.
func TestPidtuneMirroredSignWhenLeadFlips(t *testing.T) {
	num, den := Poly{-0.38686, 0.064537, -0.71409}, Poly{1, -0.49821, -1.41706, 0.93640}
	p := makePlant(t, num, den)
	auto, err := findCrossoverFreq(p, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		wc     float64
		former PID
	}{
		{0, PID{Kp: 1.6036731692862303, Ki: 0.13045843094084314, Kd: 3.0674122167888167}},
		{0.5, PID{Kp: 1.9805683643314023, Ki: 0.19805683643314023, Kd: 2.6256569062850224}},
	} {
		pid := c.former
		if stable, _ := pidtuneOracleStable(t, &pid, num, den, 0, 0); !stable {
			t.Fatalf("wc=%g: %+v unstable", c.wc, pid)
		}
		if lead := den[0] + pid.Kd*num[0]; lead >= 0 || pid.Ki*num[2] >= 0 {
			t.Fatalf("wc=%g: %+v keeps the forced sign (lead %g)", c.wc, pid, lead)
		}
		w := c.wc
		if w == 0 {
			w = auto
		}
		l := pidtuneOracleLoop(&pid, num, den, 0, 0)(w)
		if pm := 180 - math.Abs(cmplx.Phase(l))*180/math.Pi; math.Abs(cmplx.Abs(l)-1) > 1e-9 || math.Abs(pm-60) > 1e-6 {
			t.Fatalf("wc=%g: |L|=%g PM=%g at the crossover", c.wc, cmplx.Abs(l), pm)
		}
		mirror := &PID{Kp: -pid.Kp, Ki: -pid.Ki, Kd: -pid.Kd}
		if stable, _ := pidtuneOracleStable(t, mirror, num, den, 0, 0); stable {
			t.Fatalf("wc=%g: forced-sign %+v is stable, the fallback is untested", c.wc, mirror)
		}
		for _, tf := range []float64{1e-3, 1e-6} {
			filtered := pid
			filtered.Tf = tf / w
			if stable, _ := pidtuneOracleStable(t, &filtered, num, den, 0, 0); stable {
				t.Fatalf("wc=%g: %+v stable", c.wc, filtered)
			}
		}
		if got, err := Pidtune(p, PidtunePID, c.wc, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Fatalf("wc=%g: got %+v, %v; want unattainable", c.wc, got, err)
		}
	}
}

// YVAIMU: rounding moves the integrator of (0.1z+0.05)/((z−1)(z−0.999)²)
// off z = 1 by ~3e-10, inside or outside depending on the platform. The sign
// rule must still treat it as integrating: Pidtune(−G) mirrors Pidtune(G).
// Oracle: roots of den·denC + num·numC with the exact integrator.
func TestPidtuneNearIntegratorMirrors(t *testing.T) {
	num, den := Poly{0.1, 0.05}, Poly{1, -1}.Mul(Poly{1, -0.999}).Mul(Poly{1, -0.999})
	pos, neg := negatedPlant(t, num, den, .1)
	for _, p := range []*System{pos, neg} {
		poles, err := p.Poles()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.ContainsFunc(poles, func(z complex128) bool { return cmplx.Abs(z-1) < 1e-8 }) {
			t.Fatalf("precondition: no pole near z=1 in %v", poles)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		if s := pidTuningDCSign(p, eval.try, 1); s != 0 {
			t.Fatalf("precondition: DC sign %g", s)
		}
	}
	negNum := Poly{-num[0], -num[1]}
	for _, c := range []struct {
		f  PidtuneType
		wc float64
	}{{PidtunePID, 0}, {PidtuneP, 1e-3}, {PidtunePI, 1e-3}, {PidtunePID, 1e-3}} {
		cp, err := Pidtune(pos, c.f, c.wc, nil)
		if err != nil {
			t.Fatalf("%s wc=%g: G: %v", c.f, c.wc, err)
		}
		cn, err := Pidtune(neg, c.f, c.wc, nil)
		if err != nil {
			t.Fatalf("%s wc=%g: −G: %v", c.f, c.wc, err)
		}
		for _, g := range [][2]float64{{cp.Kp, cn.Kp}, {cp.Ki, cn.Ki}, {cp.Kd, cn.Kd}} {
			if math.Abs(g[0]+g[1]) > 1e-12*math.Abs(g[0]) {
				t.Errorf("%s wc=%g: %+v, %+v not mirrored", c.f, c.wc, cp, cn)
			}
		}
		if stable, ok := pidtuneOracleStable(t, cn, negNum, den, .1, 0); !ok || !stable {
			t.Errorf("%s wc=%g: −G design %+v unstable", c.f, c.wc, cn)
		}
		if stable, ok := pidtuneOracleStable(t, cp, num, den, .1, 0); !ok || !stable {
			t.Errorf("%s wc=%g: G design %+v unstable", c.f, c.wc, cp)
		}
	}
}

// With a continuous internal delay no stability test applies, so the design
// against the unforced sign must not be tried. G = e^{−0.2s}/(s(s+1)) +
// 0.01e^{−0.5s}/(s+5) has no poles in Re s > 0, and 1+C(σ)G(σ) → 1 as
// σ → +∞. Oracle: 1+C(σ)G(σ) > 0 on a real grid in σ > 0, else a real
// closed-loop root lies in Re s > 0 (Ki < 0 drives it to −∞ as σ → 0+).
func TestPidtuneNoMirrorWithoutScreen(t *testing.T) {
	a := makePlant(t, []float64{1}, []float64{1, 1, 0})
	b := makePlant(t, []float64{0.01}, []float64{1, 5})
	if err := a.SetInputDelay([]float64{0.2}); err != nil {
		t.Fatal(err)
	}
	if err := b.SetInputDelay([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	g, err := Parallel(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !g.HasInternalDelay() {
		t.Fatal("precondition: internal delay")
	}
	plantAt := func(s float64) float64 {
		return math.Exp(-0.2*s)/(s*(s+1)) + 0.01*math.Exp(-0.5*s)/(s+5)
	}
	for _, c := range [][2]float64{{3, 80}, {4, 60}, {4, 80}, {5, 60}, {5, 80}, {6, 45}, {6, 60}, {6, 80}, {8, 30}, {8, 45}, {8, 60}, {8, 80}} {
		pid, err := Pidtune(g, PidtunePID, c[0], &PidtuneOptions{PhaseMargin: c[1]})
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("wc=%g pm=%g: %v", c[0], c[1], err)
			}
			continue
		}
		for k := -6.0; k <= 4; k += 0.05 {
			s := math.Pow(10, k)
			cs := pid.Kp + pid.Ki/s + pid.Kd*s/(pid.Tf*s+1)
			if 1+cs*plantAt(s) <= 0 {
				t.Errorf("wc=%g pm=%g: %+v has a real closed-loop root in (0, %g]", c[0], c[1], pid, s)
				break
			}
		}
	}
}

// A zero at s = 0 cancels the near integrator of −s e^{−0.5s}/((s+ε)(s+1)),
// so the sign rule stays fixed and −G gets no mirrored design. Oracle: the
// characteristic function D·Dc + N·Nc·e^{−0.5s} vanishes at s = 0 for every
// integral design (N(0) = Dc(0) = 0), a closed-loop pole at the origin.
// Pidtune(+G) still returns such a design (G4YFOH).
func TestPidtuneNearIntegratorCancelledByZero(t *testing.T) {
	for _, eps := range []float64{1e-10, 1e-8, 1e-7} {
		g := makePlant(t, []float64{-1, 0}, []float64{1, 1 + eps, eps})
		if err := g.SetInputDelay([]float64{0.5}); err != nil {
			t.Fatal(err)
		}
		for _, c := range []struct {
			f  PidtuneType
			wc float64
		}{{PidtunePI, 1}, {PidtunePID, 3}} {
			if pid, err := Pidtune(g, c.f, c.wc, nil); err == nil {
				t.Errorf("eps=%g %s wc=%g: %+v has a closed-loop pole at s=0", eps, c.f, c.wc, pid)
			}
		}
	}
}

// QIQEGI: a zero-DC stable plant guesses the sign of its numerator term
// largest at wc, not of its leading nonnegligible term. Oracle: of the P
// gains ±1/|G(jwc)| crossing over at wc, only the guessed sign is stable by
// the closed-loop roots; above √(0.3311/0.0563) the s² term leads.
func TestPidtuneGuessedSignDominantTerm(t *testing.T) {
	num, den := Poly{0.0563, -0.3311, 0}, Poly{1, 7.063, 15.54, 12.54}
	pos, neg := negatedPlant(t, num, den, 0)
	for i, p := range []*System{pos, neg} {
		n, flip := num, 1.0
		if i == 1 {
			n, flip = Poly{-num[0], -num[1], 0}, -1
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		const wc = 0.895762703582968
		sign, rule := pidtuneGainSign(p, eval.try, wc, false)
		if rule != pidtuneSignStable {
			t.Fatalf("rule %d, want pidtuneSignStable", rule)
		}
		if sign != -flip {
			t.Errorf("flip=%g: sign %g at wc=%g, want %g", flip, sign, wc, -flip)
		}
		k := 1 / cmplx.Abs(eval.at(wc))
		if st, _ := pidtuneOracleStable(t, &PID{Kp: sign * k}, n, den, 0, 0); !st {
			t.Errorf("flip=%g: Kp=%g unstable", flip, sign*k)
		}
		if st, _ := pidtuneOracleStable(t, &PID{Kp: -sign * k}, n, den, 0, 0); st {
			t.Errorf("flip=%g: Kp=%g stable", flip, -sign*k)
		}
		if s, _ := pidtuneGainSign(p, eval.try, 20, false); s != flip {
			t.Errorf("flip=%g: sign %g at wc=20, want %g", flip, s, flip)
		}
	}
}
