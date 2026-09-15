package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"strings"
)

type PidtuneOptions struct {
	CrossoverFrequency float64
	PhaseMargin        float64
}

type PidtuneType string

const (
	PidtuneP    PidtuneType = "P"
	PidtuneI    PidtuneType = "I"
	PidtunePI   PidtuneType = "PI"
	PidtunePD   PidtuneType = "PD"
	PidtunePDF  PidtuneType = "PDF"
	PidtunePID  PidtuneType = "PID"
	PidtunePIDF PidtuneType = "PIDF"
)

func Pidtune(plant *System, pidType PidtuneType, opts ...PidtuneOptions) (*PID, error) {
	if _, err := newSISOLoopModel(plant, "pidtune"); err != nil {
		return nil, err
	}

	var opt PidtuneOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	if opt.PhaseMargin == 0 {
		opt.PhaseMargin = 60
	}

	pidType = PidtuneType(strings.ToUpper(string(pidType)))
	switch pidType {
	case PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF:
	default:
		return nil, fmt.Errorf("pidtune: unsupported type %q", pidType)
	}

	wc, err := findCrossoverFreq(plant, opt.CrossoverFrequency)
	if err != nil {
		return nil, err
	}

	pid, err := computePIDGains(plant, pidType, wc, opt.PhaseMargin)
	if err != nil {
		return nil, err
	}
	pid.Dt = plant.Dt

	return pid, nil
}

func findCrossoverFreq(plant *System, wcTarget float64) (float64, error) {
	if wcTarget > 0 {
		return wcTarget, nil
	}

	omega, err := marginFreqs(plant, 500)
	if err != nil {
		return 0, err
	}
	if len(omega) == 0 {
		return 1.0, nil
	}

	eval, err := newSISOEval(plant)
	if err != nil {
		return 0, err
	}

	nw := len(omega)
	magDB := make([]float64, nw)
	for k, w := range omega {
		magDB[k] = 20 * math.Log10(cmplx.Abs(eval.at(w)))
	}

	crossings := findCrossings(omega, magDB, 0.0)
	if len(crossings) > 0 {
		c := crossings[0]
		return refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			return 20 * math.Log10(cmplx.Abs(eval.at(w)))
		}), nil
	}

	bw, err := Bandwidth(plant, -3)
	if err == nil && bw > 0 && !math.IsInf(bw, 0) {
		return bw, nil
	}

	poles, _ := plant.Poles()
	if len(poles) > 0 {
		minW := math.Inf(1)
		for _, pole := range poles {
			w := cmplx.Abs(pole)
			if w > 0 && w < minW {
				minW = w
			}
		}
		if !math.IsInf(minW, 1) {
			return minW * 0.5, nil
		}
	}

	return 1.0, nil
}

func evalPlantAt(plant *System, w float64) complex128 {
	resp, err := plant.FreqResponse([]float64{w})
	if err != nil {
		return 0
	}
	return resp.At(0, 0, 0)
}

func computePIDGains(plant *System, pidType PidtuneType, wc, pmDeg float64) (*PID, error) {
	h := evalPlantAt(plant, wc)
	magP := cmplx.Abs(h)
	phaseP := cmplx.Phase(h) * 180 / math.Pi

	if magP == 0 || math.IsNaN(magP) || math.IsInf(magP, 0) {
		return nil, fmt.Errorf("pidtune: plant has zero or infinite gain at wc=%g", wc)
	}

	phiC := -180 + pmDeg - phaseP

	for phiC > 180 {
		phiC -= 360
	}
	for phiC < -180 {
		phiC += 360
	}

	pid := &PID{}

	switch pidType {
	case PidtuneP:
		pid.Kp = 1.0 / magP

	case PidtuneI:
		pid.Ki = wc / magP

	case PidtunePI:
		computePI(pid, wc, magP, phiC)

	case PidtunePD:
		computePD(pid, wc, magP, phiC)

	case PidtunePDF:
		pid.Tf = .1 / wc
		target := cmplx.Rect(1/magP, phiC*math.Pi/180)
		pid.Kd = imag(target) * (1 + wc*wc*pid.Tf*pid.Tf) / wc
		pid.Kp = real(target) - imag(target)*wc*pid.Tf
		if pid.Kp < 0 || pid.Kd < 0 {
			return nil, fmt.Errorf("pidtune: requested phase is unattainable by a positive-gain PDF at wc=%g", wc)
		}
	case PidtunePID:
		computePID(pid, wc, magP, phiC)

	case PidtunePIDF:
		computePIDF(pid, wc, magP, phiC)
	}

	return pid, nil
}

func computePI(pid *PID, wc, magP, phiC float64) {
	// C(jw) = Kp*(1 + 1/(Ti*jw)), angle = -atan(1/(Ti*wc))
	// Ti = -1/(wc*tan(phiC_rad))
	phiCRad := phiC * math.Pi / 180

	// PI phase range: (-90, 0)
	if phiCRad >= 0 {
		phiCRad = -0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	Ti := -1.0 / (wc * math.Tan(phiCRad))
	if Ti <= 0 {
		Ti = 10.0 / wc
	}

	magC := math.Sqrt(1 + 1.0/(Ti*Ti*wc*wc))
	pid.Kp = 1.0 / (magP * magC)
	pid.Ki = pid.Kp / Ti
}

// computePD preserves the legacy PD tuning seed, including its Td/10 filter.
// Callers requesting an ideal PD must explicitly set Tf=0.
func computePD(pid *PID, wc, magP, phiC float64) {
	// C(jw) = Kp*(1 + Td*jw), angle = atan(Td*wc)
	phiCRad := phiC * math.Pi / 180

	// PD phase range: (0, 90)
	if phiCRad <= 0 {
		phiCRad = 0.05
	}
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}

	Td := math.Tan(phiCRad) / wc
	if Td <= 0 {
		Td = 0.1 / wc
	}

	magC := math.Sqrt(1 + Td*Td*wc*wc)
	pid.Kp = 1.0 / (magP * magC)
	pid.Kd = pid.Kp * Td
	pid.Tf = Td / 10
	if pid.Tf <= 0 {
		pid.Tf = 1.0 / (10 * wc)
	}
}

func pidFromPhase(wc, magP, phiCRad, b float64) (Kp, Ki, Kd float64) {
	Kp = math.Cos(phiCRad) / magP
	imPart := math.Sin(phiCRad) / magP

	Ki = Kp * wc / b
	Kd = (imPart + Ki/wc) / wc

	if Kd < 0 {
		Ki = Kp * wc / (b * 2)
		Kd = (imPart + Ki/wc) / wc
	}
	if Kd < 0 {
		Kd = 0
		Ki = -imPart * wc
		if Ki < 0 {
			Ki = Kp * wc / (b * 4)
		}
	}
	return
}

func computePID(pid *PID, wc, magP, phiC float64) {
	phiCRad := phiC * math.Pi / 180
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	pid.Kp, pid.Ki, pid.Kd = pidFromPhase(wc, magP, phiCRad, 5.0)
}

func computePIDF(pid *PID, wc, magP, phiC float64) {
	// Iteratively design PID then compensate for filter (Tf=Td/10) phase loss.
	phiCRad := phiC * math.Pi / 180
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	Kp, Ki, Kd := pidFromPhase(wc, magP, phiCRad, 5.0)

	for range 20 {
		Td := 0.0
		if Kp > 0 {
			Td = Kd / Kp
		}
		Tf := Td / 10
		if Tf <= 0 {
			Tf = 1.0 / (10 * wc)
		}

		jw := complex(0, wc)
		C := complex(Kp, 0) + complex(Ki, 0)/jw + complex(Kd, 0)*jw/(complex(Tf, 0)*jw+1)
		cMag := cmplx.Abs(C)
		cPhase := cmplx.Phase(C)

		if cMag == 0 {
			break
		}
		scale := 1.0 / (cMag * magP)
		Kp *= scale
		Ki *= scale
		Kd *= scale

		pErr := phiCRad - cPhase
		if math.Abs(pErr) < 0.001 {
			break
		}

		phiAdj := phiCRad + pErr
		if phiAdj >= math.Pi/2 {
			phiAdj = math.Pi/2 - 0.05
		}
		if phiAdj <= -math.Pi/2 {
			phiAdj = -math.Pi/2 + 0.05
		}
		Kp, Ki, Kd = pidFromPhase(wc, magP, phiAdj, 5.0)
	}

	pid.Kp = Kp
	pid.Ki = Ki
	pid.Kd = Kd
	Td := 0.0
	if Kp > 0 {
		Td = Kd / Kp
	}
	pid.Tf = Td / 10
	if pid.Tf <= 0 {
		pid.Tf = 1.0 / (10 * wc)
	}
}
