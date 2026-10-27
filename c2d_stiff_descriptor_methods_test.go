package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// gInf is G(∞) = D of the pencil modes: G(0) + Σ Rᵢ/λᵢ.
func (m stepOracleModes) gInf(i, j int) complex128 {
	g := complex(m.dc.At(i, j), 0)
	for k, l := range m.poles {
		g += m.residues[k][i][j] / l
	}
	return g
}

// foh is the exact triangle-hold discretization (z−1)²/(z·T)·Z{G(s)/s²}
// of the modes: G(∞) + Σ Rᵢ·[(z−1)(eᵢ−1)/(T·λᵢ²·(z−eᵢ)) − 1/λᵢ], eᵢ =
// e^{λᵢT}.
func (m stepOracleModes) foh(i, j int, dt float64, z complex128) complex128 {
	g := m.gInf(i, j)
	for k, l := range m.poles {
		e := cmplx.Exp(l * complex(dt, 0))
		g += m.residues[k][i][j] * ((z-1)*(e-1)/(complex(dt, 0)*l*l*(z-e)) - 1/l)
	}
	return g
}

// impulseTail is the strictly proper part T·Σ Rᵢ·eᵢ/(z − eᵢ) of the
// impulse-invariant T·Σ Rᵢ·z/(z − eᵢ), eᵢ = e^{λᵢT}. Its feedthrough T·Σ Rᵢ
// holds the residue of the stiff pole (~1e14), which an ε change of E
// moves by O(1) relative, so only the tail is compared.
func (m stepOracleModes) impulseTail(i, j int, dt float64, z complex128) complex128 {
	var g complex128
	for k, l := range m.poles {
		e := cmplx.Exp(l * complex(dt, 0))
		g += m.residues[k][i][j] * e / (z - e)
	}
	return complex(dt, 0) * g
}

// tustin is G(s) at s = (2/T)(z−1)/(z+1).
func (m stepOracleModes) tustin(i, j int, dt float64, z complex128) complex128 {
	s := complex(2/dt, 0) * (z - 1) / (z + 1)
	g := m.gInf(i, j)
	for k, l := range m.poles {
		g += m.residues[k][i][j] / (s - l)
	}
	return g
}

// FOH, impulse-invariant and Tustin C2D of a full-rank E whose LU fold
// E⁻¹A misplaces the slow poles (WOIVWF) follow the pencil as ZOH does:
// the frequency response on the unit circle matches the exact method
// applied to the 320-bit partial fractions, so the slow discrete pole of
// seed 355 is e^{−0.15833·T} (fold: 0.96961 for FOH and impulse) and its
// Tustin image (1 + λT/2)/(1 − λT/2) (fold: 0.98151).
func TestC2DMethodsFullRankIllConditionedEFollowsPencil(t *testing.T) {
	const dt = 0.1
	methods := []struct {
		method C2DMethod
		want   func(m stepOracleModes, i, j int, z complex128) complex128
	}{
		{C2DMethodFOH, func(m stepOracleModes, i, j int, z complex128) complex128 { return m.foh(i, j, dt, z) }},
		{C2DMethodImpulse, func(m stepOracleModes, i, j int, z complex128) complex128 { return m.impulseTail(i, j, dt, z) }},
		{C2DMethodTustin, func(m stepOracleModes, i, j int, z complex128) complex128 { return m.tustin(i, j, dt, z) }},
	}
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		_, m, p := sys.Dims()
		for _, mt := range methods {
			d, err := sys.C2D(dt, C2DOptions{Method: mt.method})
			if err != nil {
				t.Errorf("%s %s: C2D: %v", names[k], mt.method, err)
				continue
			}
			resp := d
			if mt.method == C2DMethodImpulse {
				resp = d.Copy()
				resp.D.Zero()
			}
			worst := 0.0
			for _, th := range []float64{0.05, 0.5, 2, 3} {
				z := cmplx.Exp(complex(0, th))
				g := pencilResponse(t, resp, z)
				for i := range p {
					for j := range m {
						want := mt.want(modes[k], i, j, z)
						worst = math.Max(worst, cmplx.Abs(g[i][j]-want)/math.Max(1, cmplx.Abs(want)))
					}
				}
			}
			if !(worst <= 1e-9) {
				t.Errorf("%s %s: frequency response off the pencil by %.3g", names[k], mt.method, worst)
			}
			if mt.method == C2DMethodTustin {
				continue
			}
			got, err := d.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if got != modes[k].stable() {
				t.Errorf("%s %s: IsStable = %v, pencil poles %v", names[k], mt.method, got, modes[k].poles)
			}
		}
	}
}

// Matched C2D of the same SISO models maps the pencil poles: every discrete
// pole is e^{λT} of a 320-bit pencil pole and the DC gain is the pencil's
// G(0). A pencil with a real zero beyond 709/T (seeds 355, 46: ~1e16 next
// to the stiff pole) has e^{zT} = +Inf, so ErrOverflow is the answer
// there; the sign of G(709/T) against G(∞) certifies that zero.
func TestC2DMatchedFullRankIllConditionedEFollowsPencil(t *testing.T) {
	const dt = 0.1
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		if _, m, p := sys.Dims(); m != 1 || p != 1 {
			continue
		}
		d, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
		if errors.Is(err, ErrOverflow) {
			s := complex(709/dt, 0)
			g := modes[k].gInf(0, 0)
			for q, l := range modes[k].poles {
				g += modes[k].residues[q][0][0] / (s - l)
			}
			if real(g)*real(modes[k].gInf(0, 0)) >= 0 {
				t.Errorf("%s: matched C2D: %v, but the pencil has no real zero beyond 709/T", names[k], err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: matched C2D: %v", names[k], err)
			continue
		}
		poles, err := d.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, q := range poles {
			best := math.Inf(1)
			for _, l := range modes[k].poles {
				best = math.Min(best, cmplx.Abs(q-cmplx.Exp(l*complex(dt, 0))))
			}
			if !(best <= 1e-9) {
				t.Errorf("%s: matched pole %v is %.3g from every e^{λT} of %v", names[k], q, best, modes[k].poles)
			}
		}
		g := pencilResponse(t, d, 1)[0][0]
		want := complex(modes[k].dc.At(0, 0), 0)
		if rel := cmplx.Abs(g-want) / math.Max(1, cmplx.Abs(want)); !(rel <= 1e-8) {
			t.Errorf("%s: matched DC gain %v, pencil G(0) %v", names[k], g, want)
		}
	}
}
