package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func negateSys(sys *System) *System {
	cp := sys.Copy()
	n, m, p := sys.Dims()
	if n > 0 {
		cRaw := cp.C.RawMatrix()
		for i := range cRaw.Data {
			cRaw.Data[i] = -cRaw.Data[i]
		}
	}
	if p > 0 && m > 0 {
		dRaw := cp.D.RawMatrix()
		for i := range dRaw.Data {
			dRaw.Data[i] = -dRaw.Data[i]
		}
	}
	return cp
}

// SmithPredictor returns the continuous Smith-predictor controller for a
// plant G·e^{-s·delay} with delay-free model G and primary controller C:
//
//	K = C / (1 + C·(G - G·P))
//
// where P is the padeOrder Padé approximation of e^{-s·delay}; K is formed
// by feedback of C around G - G·P. MATLAB has no counterpart function. model
// must be delay-free (its dead time is passed as delay); a delayed model
// returns ErrDelayUnsupported. A nil controller or model, or padeOrder < 1,
// returns ErrInvalidArgument; a non-positive or non-finite delay returns
// ErrNegativeDelay.
func SmithPredictor(controller, model *System, delay float64, padeOrder int) (*System, error) {
	if controller == nil {
		return nil, fmt.Errorf("SmithPredictor: controller is nil: %w", ErrInvalidArgument)
	}
	if model == nil {
		return nil, fmt.Errorf("SmithPredictor: model is nil: %w", ErrInvalidArgument)
	}
	if err := requireSystem("SmithPredictor", controller); err != nil {
		return nil, err
	}
	if err := requireSystem("SmithPredictor", model); err != nil {
		return nil, err
	}
	if model.HasDelay() {
		return nil, fmt.Errorf("SmithPredictor: model must be delay-free; pass its dead time as delay: %w", ErrDelayUnsupported)
	}
	if !controller.IsContinuous() {
		return nil, fmt.Errorf("SmithPredictor: controller must be continuous: %w", ErrWrongDomain)
	}
	if !model.IsContinuous() {
		return nil, fmt.Errorf("SmithPredictor: model must be continuous: %w", ErrWrongDomain)
	}
	if !(delay > 0) || math.IsInf(delay, 0) {
		return nil, fmt.Errorf("SmithPredictor: delay must be positive and finite, got %v: %w", delay, ErrNegativeDelay)
	}
	if padeOrder < 1 {
		return nil, fmt.Errorf("SmithPredictor: padeOrder must be positive, got %d: %w", padeOrder, ErrInvalidArgument)
	}

	_, mCtrl, pCtrl := controller.Dims()
	_, mMod, pMod := model.Dims()

	if pMod != mCtrl {
		return nil, fmt.Errorf("SmithPredictor: model outputs %d != controller inputs %d: %w", pMod, mCtrl, ErrDimensionMismatch)
	}
	if mMod != pCtrl {
		return nil, fmt.Errorf("SmithPredictor: model inputs %d != controller outputs %d: %w", mMod, pCtrl, ErrDimensionMismatch)
	}

	modelClean := model.Copy()

	modelDelayed := model.Copy()
	delayMat := mat.NewDense(pMod, mMod, nil)
	raw := delayMat.RawMatrix()
	for i := range raw.Data {
		raw.Data[i] = delay
	}
	if err := modelDelayed.SetDelay(delayMat); err != nil {
		return nil, fmt.Errorf("SmithPredictor: SetDelay: %w", err)
	}

	modelPade, err := modelDelayed.Pade(padeOrder)
	if err != nil {
		return nil, fmt.Errorf("SmithPredictor: Pade: %w", err)
	}

	negPade := negateSys(modelPade)

	diff, err := Parallel(modelClean, negPade)
	if err != nil {
		return nil, fmt.Errorf("SmithPredictor: Parallel: %w", err)
	}

	result, err := Feedback(controller, diff, -1.0)
	if err != nil {
		return nil, fmt.Errorf("SmithPredictor: Feedback: %w", err)
	}

	return result, nil
}
