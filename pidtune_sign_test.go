package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
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
