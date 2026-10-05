package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// FeedbackOption configures how Feedback treats delays when closing the loop.
type FeedbackOption func(*feedbackConfig)

type feedbackConfig struct {
	approximateDelays bool
	padeOrder         int
	thiranOrder       int
}

func newFeedbackConfig(opts []FeedbackOption) feedbackConfig {
	cfg := feedbackConfig{padeOrder: 5}
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// WithApproximatedDelays makes Feedback return a delay-free rational model
// instead of an exact model with internal delays. Continuous-time delays are
// replaced by Pade approximations (default order 5); exact integer discrete
// delays are absorbed into states; fractional discrete delays require
// WithThiranOrder.
func WithApproximatedDelays() FeedbackOption {
	return func(c *feedbackConfig) {
		c.approximateDelays = true
	}
}

// WithPadeOrder implies WithApproximatedDelays and sets the Pade order used
// for continuous-time delays.
func WithPadeOrder(n int) FeedbackOption {
	return func(c *feedbackConfig) {
		c.approximateDelays = true
		c.padeOrder = n
	}
}

// WithThiranOrder implies WithApproximatedDelays and enables Thiran allpass
// modeling for fractional discrete delays. Exact integer discrete delays
// remain state-space delays; continuous-time delays use Pade approximation.
func WithThiranOrder(n int) FeedbackOption {
	return func(c *feedbackConfig) {
		c.approximateDelays = true
		c.thiranOrder = n
	}
}

func feedbackWithApproximatedDelays(plant, controller *System, sign float64, cfg feedbackConfig) (*System, error) {
	strategy := feedbackDelayStrategy{cfg: cfg}
	p, c, err := strategy.prepare(plant, controller, sign)
	if err != nil {
		return nil, err
	}

	result, err := Feedback(p, c, sign)
	if err != nil {
		return nil, err
	}
	result.InputName = copyStringSlice(plant.InputName)
	result.OutputName = copyStringSlice(plant.OutputName)
	return result, nil
}

type feedbackDelayStrategy struct {
	cfg feedbackConfig
}

func (s feedbackDelayStrategy) prepare(plant, controller *System, sign float64) (*System, *System, error) {
	if plant.IsDiscrete() {
		p, err := s.replaceDiscreteDelays(plant, "plant")
		if err != nil {
			return nil, nil, err
		}
		c, err := s.replaceDiscreteDelays(controller, "controller")
		if err != nil {
			return nil, nil, err
		}
		return p, c, nil
	}

	p, err := replaceContinuousDelays(plant, s.cfg.padeOrder)
	if err != nil {
		return nil, nil, fmt.Errorf("Feedback: pade plant: %w", err)
	}
	c, err := replaceContinuousDelays(controller, s.cfg.padeOrder)
	if err != nil {
		return nil, nil, fmt.Errorf("Feedback: pade controller: %w", err)
	}
	if err := s.requireWellPosedPade(p, c, sign); err != nil {
		return nil, nil, err
	}
	return p, c, nil
}

func (s feedbackDelayStrategy) replaceDiscreteDelays(sys *System, role string) (*System, error) {
	if s.cfg.thiranOrder == 0 {
		if err := sys.validate(); err != nil {
			return nil, fmt.Errorf("Feedback: validate %s: %w", role, err)
		}
		out, err := sys.AbsorbDelay()
		if err != nil {
			return nil, fmt.Errorf("Feedback: absorb %s: %w", role, err)
		}
		return out, nil
	}
	out, err := replaceDiscreteExternalDelaysWithThiran(sys, s.cfg.thiranOrder)
	if err != nil {
		return nil, fmt.Errorf("Feedback: thiran %s: %w", role, err)
	}
	return out, nil
}

func (s feedbackDelayStrategy) requireWellPosedPade(plant, controller *System, sign float64) error {
	_, mPlant, pPlant := plant.Dims()
	_, mCtrl, pCtrl := controller.Dims()
	if pPlant != mCtrl || pCtrl != mPlant {
		return nil
	}
	if _, err := solveFeedbackFeedthrough(plant.D, controller.D, sign, pPlant, "Feedback", ErrAlgebraicLoop); err != nil {
		return fmt.Errorf("Feedback: Pade approximation creates singular algebraic loop; try a different padeOrder (even vs odd) to flip feedthrough sign: %w", err)
	}
	return nil
}

func replaceDiscreteExternalDelaysWithThiran(sys *System, thiranOrder int) (*System, error) {
	return newDelayConversionPolicy(sys.Dt, thiranOrder, 0).replaceDiscreteExternal(sys, "Feedback")
}

func replaceContinuousDelays(sys *System, padeOrder int) (*System, error) {
	return newDelayConversionPolicy(sys.Dt, 0, padeOrder).replaceContinuousExternal(sys, "Feedback")
}

func buildDiagWithPade(channel, size int, pade *System) (*System, error) {
	if size == 1 {
		return pade, nil
	}

	np, _, _ := pade.Dims()

	n := np
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, size, nil)
	C := mat.NewDense(size, n, nil)
	dData := make([]float64, size*size)
	for i := range size {
		dData[i*size+i] = 1
	}
	D := mat.NewDense(size, size, dData)

	if np > 0 {
		setBlock(A, 0, 0, pade.A)

		padeB := pade.B.RawMatrix()
		for i := range np {
			B.Set(i, channel, padeB.Data[i*padeB.Stride])
		}

		padeC := pade.C.RawMatrix()
		for j := range np {
			C.Set(channel, j, padeC.Data[j])
		}
	}

	padeD := pade.D.At(0, 0)
	D.Set(channel, channel, padeD)

	return newNoCopy(A, B, C, D, pade.Dt)
}
