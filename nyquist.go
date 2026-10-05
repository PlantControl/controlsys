package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
)

type NyquistResult struct {
	Omega         []float64
	Contour       []complex128
	ContourN      []complex128
	Encirclements int
	RHPPoles      int
	RHPZerosCL    int
}

func (sys *System) Nyquist(omega []float64, nPoints int) (*NyquistResult, error) {
	if _, err := newSISOLoopModel(sys, "Nyquist"); err != nil {
		return nil, err
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}

	maxAbs := 0.0
	for _, pole := range poles {
		if a := cmplx.Abs(pole); a > maxAbs {
			maxAbs = a
		}
	}
	tol := eps() * math.Max(maxAbs, 1)

	rhpPoles := countRHPPoles(poles, sys.IsContinuous(), tol)

	imagPoles := findImagAxisPoles(poles, sys.IsContinuous(), sys.Dt, tol)

	if omega == nil {
		omega = autoNyquistFreqs(sys, poles, imagPoles, nPoints)
	}

	contour, contourOmega, err := buildNyquistContour(sys, omega, imagPoles, tol)
	if err != nil {
		return nil, err
	}

	contourN := make([]complex128, len(contour))
	for k := range contour {
		contourN[k] = cmplx.Conj(contour[len(contour)-1-k])
	}

	enc := nyquistEncirclements(contour, contourN, sys)

	return &NyquistResult{
		Omega:         contourOmega,
		Contour:       contour,
		ContourN:      contourN,
		Encirclements: enc,
		RHPPoles:      rhpPoles,
		RHPZerosCL:    enc + rhpPoles,
	}, nil
}

func countRHPPoles(poles []complex128, continuous bool, tol float64) int {
	count := 0
	for _, pole := range poles {
		if poleOutsideStabilityBoundary(pole, continuous, tol) {
			count++
		}
	}
	return count
}

type imagAxisPole struct {
	w0           float64
	multiplicity int
}

func findImagAxisPoles(poles []complex128, continuous bool, dt, tol float64) []imagAxisPole {
	type freqCount struct {
		w0    float64
		count int
	}
	var found []freqCount

	used := make([]bool, len(poles))
	for i, pole := range poles {
		if used[i] {
			continue
		}

		var onAxis bool
		var w0 float64
		if continuous {
			onAxis = poleOnStabilityBoundary(pole, continuous, tol)
			w0 = math.Abs(imag(pole))
		} else {
			onAxis = poleOnStabilityBoundary(pole, continuous, tol)
			if onAxis {
				angle := cmplx.Phase(pole)
				if dt > 0 {
					w0 = math.Abs(angle) / dt
				} else {
					w0 = math.Abs(angle)
				}
			}
		}

		if !onAxis {
			continue
		}

		used[i] = true
		mult := 1

		for j := i + 1; j < len(poles); j++ {
			if used[j] {
				continue
			}
			if cmplx.Abs(poles[j]-cmplx.Conj(pole)) < tol*10 {
				used[j] = true
				mult++
			} else if cmplx.Abs(poles[j]-pole) < tol*10 {
				used[j] = true
				mult++
			}
		}

		if w0 < tol {
			w0 = 0
		}
		merged := false
		for k := range found {
			if math.Abs(found[k].w0-w0) < tol*10 {
				found[k].count += mult
				merged = true
				break
			}
		}
		if !merged {
			found = append(found, freqCount{w0, mult})
		}
	}

	result := make([]imagAxisPole, len(found))
	for i, f := range found {
		result[i] = imagAxisPole{w0: f.w0, multiplicity: f.count}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].w0 < result[j].w0 })
	return result
}

func autoNyquistFreqs(sys *System, poles []complex128, imagPoles []imagAxisPole, nPoints int) []float64 {
	if nPoints <= 0 {
		nPoints = 500
	}

	td := newTimeDomain(sys.Dt)
	natFreqs := make([]float64, 0, len(poles))
	for _, p := range poles {
		wn := td.naturalFrequency(p)
		if wn > 0 && !math.IsInf(wn, 0) {
			natFreqs = append(natFreqs, wn)
		}
	}

	wMin, wMax := 0.01, 100.0
	if len(natFreqs) > 0 {
		lo, hi := natFreqs[0], natFreqs[0]
		for _, w := range natFreqs[1:] {
			if w < lo {
				lo = w
			}
			if w > hi {
				hi = w
			}
		}
		wMin = lo / 100
		wMax = hi * 100
		if wMin < 1e-6 {
			wMin = 1e-6
		}
		if wMax > 1e6 {
			wMax = 1e6
		}
	}

	if sys.IsDiscrete() && sys.Dt > 0 {
		nyquistFreq := math.Pi / sys.Dt
		if wMax > nyquistFreq {
			wMax = nyquistFreq
		}
		wMin = min(wMin, wMax/100)
	}

	baseN := max(nPoints*7/10, 50)
	omega := logspace(math.Log10(wMin), math.Log10(wMax), baseN)

	for _, ip := range imagPoles {
		if ip.w0 == 0 {
			continue
		}
		lo := ip.w0 * 0.99
		hi := ip.w0 * 1.01
		if lo < wMin {
			lo = wMin
		}
		if hi > wMax {
			hi = wMax
		}
		extra := logspace(math.Log10(lo), math.Log10(hi), 30)
		omega = append(omega, extra...)
	}

	sort.Float64s(omega)
	deduped := omega[:1]
	for i := 1; i < len(omega); i++ {
		if omega[i]-deduped[len(deduped)-1] > 1e-12*omega[i] {
			deduped = append(deduped, omega[i])
		}
	}
	omega = deduped

	epsIndent := 1e-4
	filtered := omega[:0]
	for _, w := range omega {
		tooClose := false
		for _, ip := range imagPoles {
			epsW := epsIndent * math.Max(1, ip.w0)
			if math.Abs(w-ip.w0) < epsW {
				tooClose = true
				break
			}
		}
		if !tooClose {
			filtered = append(filtered, w)
		}
	}

	return filtered
}

func buildNyquistContour(sys *System, omega []float64, imagPoles []imagAxisPole, _ float64) ([]complex128, []float64, error) {
	if len(omega) == 0 && len(imagPoles) == 0 {
		return nil, nil, nil
	}

	type segment struct {
		freqs []float64
		isArc bool
		arcW0 float64
	}

	sortedPoles := make([]imagAxisPole, len(imagPoles))
	copy(sortedPoles, imagPoles)
	sort.Slice(sortedPoles, func(i, j int) bool { return sortedPoles[i].w0 < sortedPoles[j].w0 })

	segments := make([]segment, 0, len(imagPoles)*2+1)
	curFreqs := make([]float64, 0, len(omega))

	for _, w := range omega {
		insertArc := false
		var arcPole imagAxisPole
		for _, ip := range sortedPoles {
			if ip.w0 == 0 {
				continue
			}
			epsW := 1e-4 * math.Max(1, ip.w0)
			if len(curFreqs) > 0 && curFreqs[len(curFreqs)-1] < ip.w0-epsW && w > ip.w0+epsW {
				insertArc = true
				arcPole = ip
				break
			}
		}

		if insertArc {
			if len(curFreqs) > 0 {
				segments = append(segments, segment{freqs: curFreqs})
				curFreqs = nil
			}
			segments = append(segments, segment{isArc: true, arcW0: arcPole.w0})
		}

		curFreqs = append(curFreqs, w)
	}
	if len(curFreqs) > 0 {
		segments = append(segments, segment{freqs: curFreqs})
	}

	hasOriginPole := false
	var originMult int
	for _, ip := range sortedPoles {
		if ip.w0 == 0 {
			hasOriginPole = true
			originMult = ip.multiplicity
			break
		}
	}
	if hasOriginPole {
		originArc := segment{isArc: true, arcW0: 0}
		newSegments := make([]segment, 0, len(segments)+1)
		newSegments = append(newSegments, originArc)
		newSegments = append(newSegments, segments...)
		segments = newSegments
		_ = originMult
	}

	estSize := len(omega) + len(imagPoles)*50
	contour := make([]complex128, 0, estSize)
	contourOmega := make([]float64, 0, estSize)

	for _, seg := range segments {
		if seg.isArc {
			arc, err := indentContour(sys, seg.arcW0, 50)
			if err != nil {
				return nil, nil, err
			}
			contour = append(contour, arc...)
			for range arc {
				contourOmega = append(contourOmega, seg.arcW0)
			}
		} else {
			resp, err := sys.FreqResponse(seg.freqs)
			if err != nil {
				return nil, nil, err
			}
			for k := range seg.freqs {
				contour = append(contour, resp.At(k, 0, 0))
				contourOmega = append(contourOmega, seg.freqs[k])
			}
		}
	}

	return contour, contourOmega, nil
}

func indentContour(sys *System, w0 float64, nArc int) ([]complex128, error) {
	epsR := 1e-4 * math.Max(1, w0)
	arc := make([]complex128, nArc)

	for i := range nArc {
		theta := math.Pi/2 - float64(i)*math.Pi/float64(nArc-1)

		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w0) + complex(epsR, 0)*cmplx.Exp(complex(0, theta))
		} else {
			angle := w0 * sys.Dt
			z := complex((1+epsR)*math.Cos(angle+theta*epsR), (1+epsR)*math.Sin(angle+theta*epsR))
			s = z
		}

		vals, err := sys.EvalFr(s)
		if err != nil {
			return nil, err
		}
		arc[i] = vals[0][0]
	}
	return arc, nil
}

func nyquistEncirclements(contour, contourN []complex128, sys *System) int {
	// Full Nyquist contour order: neg freqs (high→low w) → pos freqs (low→high w) → closure at ∞
	// contourN[k] = conj(contour[len-1-k]), so contourN goes high-freq→low-freq for negative freqs
	var full []complex128
	full = append(full, contourN...)
	full = append(full, contour...)

	var closure complex128
	if allZeroDense(sys.D) {
		closure = 0
	} else {
		closure = complex(sys.D.At(0, 0), 0)
	}
	full = append(full, closure)

	// windingNumber returns positive for CCW; Nyquist convention: CW positive
	return -windingNumber(full, complex(-1, 0))
}

func windingNumber(contour []complex128, point complex128) int {
	if len(contour) < 2 {
		return 0
	}

	totalAngle := 0.0
	n := len(contour)
	for k := range n {
		z1 := contour[k] - point
		z2 := contour[(k+1)%n] - point

		if cmplx.Abs(z1) < 1e-15 || cmplx.Abs(z2) < 1e-15 {
			continue
		}

		dtheta := cmplx.Phase(z2) - cmplx.Phase(z1)
		for dtheta > math.Pi {
			dtheta -= 2 * math.Pi
		}
		for dtheta < -math.Pi {
			dtheta += 2 * math.Pi
		}
		totalAngle += dtheta
	}

	return int(math.Round(totalAngle / (2 * math.Pi)))
}
