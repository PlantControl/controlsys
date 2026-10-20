package controlsys

import (
	"context"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// V2AYQB: stable discrete plants with rounded poles near z = 1, where the
// point solve is singular over decades that the band, the phase walks (from
// wc·1e-6) or the one-gain crossover sample reach, while FreqResponse
// resolves them. TunePID failed every family with "loop response at ω=…:
// singular complex matrix" or "plant gain (+Inf+0i)", where Pidtune designs.
// Every family now returns a design; on the pidtuneIdealNearUnit plants,
// PD and PID do (their I and PI stay unattainable, as no Pidtune I or PI
// there meets the robustness bar). Oracle: exact Schur–Cohn test; the loop
// with the plant in 256-bit arithmetic and the controller in z−1 form
// (expanded polynomials in z lose it near z = 1) meets the reported margin
// at the reported crossover, every crossover keeps at least it, and a free
// P or I, whose search bounds |S| on an undelayed plant, keeps |S| ≤ 4 over
// the reported band.
func TestTunePIDNearUnitPolesFallBackToFreqResponse(t *testing.T) {
	skipSlowUnderRace(t)
	const dt = 0.01
	ideal := []PidtuneType{PidtunePD, PidtunePID}
	ip3 := Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}
	for _, c := range []struct {
		name     string
		num, den Poly
		fams     []PidtuneType
	}{
		{"IP3XIC k=100", Poly{2.761969998249568e-14, 2.6332895798497864e-14, 6.196985260193668e-15}, ip3, nil},
		{"IP3XIC k=1", Poly{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}, ip3, nil},
		{"IP3XIC k=10", Poly{2.761969998249568e-15, 2.6332895798497864e-15, 6.196985260193668e-16}, ip3, nil},
		{"GK5SUC A", Poly{1.5379663637764417e-12, -3.387450861436736e-13}, Poly{1, -3.9146780141541093, 5.7453934609919255, -3.7467528452104566, 0.9160373983726545}, nil},
		{"GK5SUC B", Poly{1.7295311989236164e-12}, Poly{1, -2.999890083275418, 2.999780168026009, -0.999890084750586}, nil},
		{"review1", Poly{1.1201261000572445e-12, 7.985425046796576e-13}, Poly{1, -4.956471969268471, 9.826418031565801, -9.740421021787181, 4.827475826360807, -0.95700086687092}, nil},
		{"review2 a", Poly{1.6478992314667082e-12}, Poly{1, -3.9988962961964294, 5.996689296298131, -3.9966897039442033, 0.998896703842506}, nil},
		{"review2 b", Poly{8.58763892349696e-12, -2.307232265714587e-12}, Poly{1, -3.997506422481467, 5.9925206599047645, -3.992522052100717, 0.9975078146774358}, nil},
		{"ideal a", pidtuneIdealNearUnit[0].num, pidtuneIdealNearUnit[0].den, ideal},
		{"ideal b", pidtuneIdealNearUnit[1].num, pidtuneIdealNearUnit[1].den, ideal},
		{"ideal c", pidtuneIdealNearUnit[2].num, pidtuneIdealNearUnit[2].den, ideal},
		{"ideal d", pidtuneIdealNearUnit[3].num, pidtuneIdealNearUnit[3].den, ideal},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, dt, 0)
		fams := c.fams
		if fams == nil {
			fams = pidtuneAllTypes
		}
		for _, f := range fams {
			name := fmt.Sprintf("%s %s", c.name, f)
			r, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			k := r.Controller
			if !pidtuneExactSchurStable(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}, c.num, c.den, dt) {
				t.Errorf("%s: %+v unstable", name, *k)
			}
			L := func(w float64) complex128 {
				h := math.Sin(w * dt / 2)
				zm1 := complex(-2*h*h, math.Sin(w*dt))
				cc := complex(k.Kp, 0) + complex(k.Ki*dt, 0)/zm1
				if k.Kd != 0 {
					cc += complex(k.Kd, 0) / (complex(k.Tf, 0) + complex(dt, 0)/zm1)
				}
				z := 1 + zm1
				return cc * pidtuneBigEval(c.num, z) / pidtuneBigEval(c.den, z)
			}
			ev := r.Evidence
			l := L(ev.AchievedCrossover)
			pm := 180 + cmplx.Phase(l)*180/math.Pi
			if math.Abs(cmplx.Abs(l)-1) > 1e-3 || math.Abs(pm-ev.AchievedPhaseMargin) > .5 {
				t.Errorf("%s: |L(%.4g)| %.6g, margin %.4g, reported %.4g", name, ev.AchievedCrossover, cmplx.Abs(l), pm, ev.AchievedPhaseMargin)
			}
			pms, _ := pidtuneSignedMargins(L, ev.AchievedCrossover*1e-4, math.Pi/dt)
			for _, m := range pms {
				if math.Abs(m) < ev.AchievedPhaseMargin-.5 {
					t.Errorf("%s: crossover margin %.4g below reported %.4g", name, m, ev.AchievedPhaseMargin)
				}
			}
			if f != PidtuneP && f != PidtuneI {
				continue
			}
			if _, ms := pidtuneSignedMargins(L, ev.FrequencyBand[0], ev.FrequencyBand[1]); ms > pidTuningMaxSensitivity*1.01 {
				t.Errorf("%s: |S| %.4g over band %v", name, ms, ev.FrequencyBand)
			}
		}
	}
}

// pidtuneIdealNearUnit: stable discrete plants with four or five poles near
// z = 1, and the forward-Euler ideal-derivative designs (Tf = 0) TunePID
// returned for them, whose loops are unstable by the exact Schur–Cohn test
// (largest root modulus 1+2e-6 to 1+3e-5, above rounding of the
// coefficients): float roots of the characteristic err by more.
var pidtuneIdealNearUnit = []struct {
	num, den Poly
	c        PID2
}{
	{Poly{5.9946902928981455e-09, 7.470390124394129e-10}, Poly{1, -3.0564255156743267, 3.169470396078771, -1.1696642443685965, 0.05661936396415252},
		PID2{Kp: 9.063129246083328e-06, Ki: 2.849366096412798e-06, Kd: 0.05918551923253021}},
	{Poly{1.3648010760220169e-08, 4.9450826577340375e-09}, Poly{1, -3.8812605154764728, 5.6438477883801, -3.643914028667036, 0.8813267557634102},
		PID2{Kp: 1.5612265480604035e-05, Ki: 7.234308557632492e-06, Kd: 0.019923657514040235}},
	{Poly{1.938971367075846e-13}, Poly{1, -3.714512469265432, 5.1435531512915125, -3.143568894717929, 0.7145282126918486},
		PID2{Kp: -0.010404296947905413, Kd: 40.34301142978002}},
	{Poly{8.43892612359764e-14}, Poly{1, -4.580963310835395, 8.325983494925048, -7.4921701917364585, 3.3302431420460463, -0.5830931343992402},
		PID2{Kp: -0.027140250933708015, Kd: 10.039757768093317}},
}

// The stability screen of an ideal derivative retests the characteristic
// exactly when float roots cannot settle it. Oracle: exact Schur–Cohn on
// the given coefficients.
func TestPIDStabilityScreenIdealNearUnitExact(t *testing.T) {
	const dt = 0.01
	for i, tc := range pidtuneIdealNearUnit {
		p := pidtuneBackoffPlant(t, tc.num, tc.den, dt, 0)
		screen, err := newPIDStabilityScreen(p, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		for _, scale := range []float64{1, 0.1} {
			c := tc.c
			c.Kp, c.Ki, c.Kd = math.Abs(c.Kp)*scale, c.Ki*scale, c.Kd*scale
			if scale == 1 {
				c.Kp = tc.c.Kp
			}
			c.B, c.C, c.Dt = 1, 1, dt
			want := pidtuneExactSchurStable(t, &PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Dt: dt}, tc.num, tc.den, dt)
			if scale == 1 && want {
				t.Fatalf("%d: oracle stable", i)
			}
			if got, err := screen.closedLoopPoles(&c); err != nil || got != want {
				t.Errorf("%d: %+v screen stable %v %v, oracle %v", i, c, got, err, want)
			}
		}
	}
}
