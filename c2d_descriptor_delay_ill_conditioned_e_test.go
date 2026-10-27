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
	if sys := twoStiffDelayCase.system(t); matlabRankFull(t, sys.E) {
		pol, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		if m, ok := newStepOracleModes(sys, pol); ok {
			names, systems, modes = append(names, twoStiffDelayCase.name), append(systems, sys), append(modes, m)
		}
	}
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

// twoStiffDelayCase has two stiff singular values of E (poles near -4e11
// and -5e8, too close for splitFastPencil), so its fast part is a dense
// 2×2: the delayed C2D takes it to real Schur form, without which one
// exponential of the block-diagonal parts loses the slow modes (1e-2).
var twoStiffDelayCase = stepDescriptorCase{name: "two stiff singular values", n: 6, m: 1, p: 2,
	a: []float64{0.042420132812795726, -0.2919049357477474, -0.15479350397654157, -0.36496961655294907, -0.23912291192769108, -0.5465938655810121, 0.7464611660032728, 1.9599954056323963, -1.2795376355493338, 2.2775773369374464, 0.551226280056841, -0.38625888489301186, 0.06121597482523611, -0.17703887501863852, 1.1516929454408633, 0.020607505348234703, -0.012828036736695147, -0.029950194096911128, -1.2870031484303517, -1.1287140369647495, 0.47419761534680455, 0.47240439883799906, 1.3893640782949692, 0.7868244194688921, 0.2881430912428596, 0.15426833864122683, 0.020828091790205467, 0.1014996831927455, 0.05095392286020483, -0.7275361867754782, 0.693941800653298, 0.08358548565881922, -0.44876825785350133, -0.8419148898849496, -1.0061957052324744, -1.363215878946758},
	b: []float64{0.4667681077422996, 2.0683899509843298, 0.5154007759495757, 1.0828388871904064, 0.4150919146294636, -0.8072091449746803},
	c: []float64{0.5565255454261786, 0.7961660476662538, -0.6747754353636533, -0.03620192331379848, 1.085048672478534, 0.1940613679058396, 0.709264391536892, -0.064264631817831, -0.5566732992698904, -0.6529520642175414, 0.8246911084278216, 1.9168873816961471},
	d: []float64{1.3901585580331661, 3.0012670721047847},
	e: []float64{0.22501534337282603, -0.3187422713301665, -0.20241916721369674, 0.06134876827190296, 0.5710126084058056, -0.3764654036302967, -0.40678920178167677, 0.26076033327907966, 0.22044707428947358, 0.31767106763437336, 0.30723803534643424, 0.14321917455349756, 0.24176618119531432, -0.37907622231523763, -0.09446671299982655, -0.4030448230588998, -0.5223762441659099, -0.831858055148577, -0.2502635682827489, -0.056029387484123234, 0.09132372537472816, 0.13031826849688258, 0.3645550197099552, -0.5665355446853706, -0.3067934287341147, -0.1185649205870722, 0.4755845724460442, 0.44864066950360754, -0.4167088086634805, -0.26574090496145225, 0.13489025956832765, -0.3364879031093305, 0.10054959707389696, 0.33295283628559413, 0.055622880711041614, -0.15073956027075439}}
