package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zohDelayed returns the exact ZOH discretization of e^{−sτ}·G_ij(s) at z,
// τ = k·dt + θ, 0 ≤ θ < dt: z^{−k}·zoh for θ = 0, else
// z^{−(k+1)}·[D + Σ Rᵢ/λᵢ·(c(1 − z⁻¹)/(1 − a·z⁻¹) − 1)], a = e^{λᵢdt},
// c = e^{λᵢ(dt−θ)}, from the sampled delayed step response.
func (m stepOracleModes) zohDelayed(i, j int, dt, tau float64, z complex128) complex128 {
	k := math.Floor(tau/dt + 1e-12)
	theta := tau - k*dt
	if theta <= 1e-12*dt {
		return m.zoh(i, j, dt, z) * cmplx.Pow(z, complex(-k, 0))
	}
	g := complex(m.dc.At(i, j), 0)
	for q, l := range m.poles {
		g += m.residues[q][i][j] / l
	}
	zi := 1 / z
	for q, l := range m.poles {
		a, c := cmplx.Exp(l*complex(dt, 0)), cmplx.Exp(l*complex(dt-theta, 0))
		g += m.residues[q][i][j] / l * (c*(1-zi)/(1-a*zi) - 1)
	}
	return g * cmplx.Pow(z, complex(-(k+1), 0))
}

// C2D of the full-rank ill-conditioned-E models with input, output or I/O
// delays of whole samples follows the pencil (AZUMSC): the delay path
// used to fold E⁻¹A, which moves the slow poles by cond(E)·ε. ZOH matches
// the exact delayed ZOH of the 320-bit modes, Tustin G((2/dt)(z−1)/(z+1))
// of the pencil behind the delay, and FOH, impulse and matched the
// delay-free C2D (oracle-checked by TestC2DZOHFullRankIllConditionedEFollowsPencil
// and the C2D method tests) times z⁻ᵏ.
func TestC2DDelayedFullRankIllConditionedEFollowsPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	const dt = 0.1
	for k, base := range systems {
		_, m, p := base.Dims()
		for _, dc := range []struct {
			name string
			set  func(*System) func(i, j int) float64
		}{
			{"input 2 samples", func(s *System) func(i, j int) float64 {
				s.InputDelay = make([]float64, m)
				s.InputDelay[0] = 2 * dt
				return func(_, j int) float64 { return s.InputDelay[j] }
			}},
			{"output 1 sample", func(s *System) func(i, j int) float64 {
				s.OutputDelay = make([]float64, p)
				s.OutputDelay[0] = dt
				return func(i, _ int) float64 { return s.OutputDelay[i] }
			}},
			{"io 3 samples", func(s *System) func(i, j int) float64 {
				s.Delay = mat.NewDense(p, m, nil)
				s.Delay.Set(0, 0, 3*dt)
				return func(i, j int) float64 { return s.Delay.At(i, j) }
			}},
		} {
			for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH, C2DMethodImpulse, C2DMethodMatched} {
				if meth == C2DMethodMatched && (m != 1 || p != 1) {
					continue
				}
				opts := C2DOptions{Method: meth}
				free, err := base.C2D(dt, opts)
				if err != nil {
					continue
				}
				sys := base.Copy()
				tau := dc.set(sys)
				d, err := sys.C2D(dt, opts)
				if err != nil {
					t.Errorf("%s, %s, %s: C2D: %v", names[k], dc.name, meth, err)
					continue
				}
				worst := 0.0
				for _, th := range []float64{0.05, 0.5, 2, 3} {
					z := cmplx.Exp(complex(0, th))
					g, err := d.EvalFr(z)
					if err != nil {
						t.Fatal(err)
					}
					gf, err := free.EvalFr(z)
					if err != nil {
						t.Fatal(err)
					}
					var gt [][]complex128
					if meth == C2DMethodTustin {
						gt = pencilResponse(t, base, complex(2/dt, 0)*(z-1)/(z+1))
					}
					for i := range p {
						for j := range m {
							shift := cmplx.Pow(z, complex(-math.Round(tau(i, j)/dt), 0))
							var want complex128
							switch meth {
							case C2DMethodZOH:
								want = modes[k].zohDelayed(i, j, dt, tau(i, j), z)
							case C2DMethodTustin:
								want = gt[i][j] * shift
							default:
								want = gf[i][j] * shift
							}
							worst = math.Max(worst, cmplx.Abs(g[i][j]-want)/math.Max(1, cmplx.Abs(want)))
						}
					}
				}
				if !(worst <= 1e-9) {
					t.Errorf("%s, %s, %s: C2D frequency response off the pencil by %.3g", names[k], dc.name, meth, worst)
				}
			}
		}
	}
}
