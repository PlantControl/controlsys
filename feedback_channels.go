package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// FeedbackChannels selects the loop channels of MATLAB
// feedback(sys1,sys2,feedin,feedout). Indices are 0-based (MATLAB's minus
// one), non-empty, distinct and in range.
type FeedbackChannels struct {
	// FeedIn lists the plant inputs driven by the controller outputs.
	FeedIn []int
	// FeedOut lists the plant outputs fed to the controller inputs.
	FeedOut []int
}

// feedbackController returns the controller that closes the loop over every
// plant channel: controller (identity when nil) placed on the selected
// channels, zero elsewhere.
func feedbackController(plant, controller *System, ch []FeedbackChannels) (*System, error) {
	_, m, p := plant.Dims()
	if len(ch) > 1 {
		return nil, fmt.Errorf("Feedback: %d channel specs, want at most 1: %w", len(ch), ErrInvalidArgument)
	}
	if len(ch) == 0 {
		if controller != nil {
			return controller, nil
		}
		if p != m {
			return nil, fmt.Errorf("Feedback: plant must be square when controller is nil, got %d outputs × %d inputs: %w", p, m, ErrDimensionMismatch)
		}
		return NewGain(eyeOrEmptyDense(m), plant.Dt)
	}
	in, out := ch[0].FeedIn, ch[0].FeedOut
	if len(in) == 0 || len(out) == 0 {
		return nil, fmt.Errorf("Feedback: feedin and feedout must be non-empty: %w", ErrInvalidArgument)
	}
	if err := validateChannelIndices("Feedback", "feedin", in, m); err != nil {
		return nil, err
	}
	if err := validateChannelIndices("Feedback", "feedout", out, p); err != nil {
		return nil, err
	}
	if controller == nil {
		if len(in) != len(out) {
			return nil, fmt.Errorf("Feedback: nil controller needs len(feedin) %d == len(feedout) %d: %w", len(in), len(out), ErrDimensionMismatch)
		}
		var err error
		if controller, err = NewGain(eyeOrEmptyDense(len(in)), plant.Dt); err != nil {
			return nil, err
		}
	}
	if err := domainMatch(plant, controller); err != nil {
		return nil, err
	}
	_, mc, pc := controller.Dims()
	if mc != len(out) || pc != len(in) {
		return nil, fmt.Errorf("Feedback: controller is %d×%d, want len(feedin)×len(feedout) = %d×%d: %w", pc, mc, len(in), len(out), ErrDimensionMismatch)
	}
	gather := mat.NewDense(len(out), p, nil)
	for k, j := range out {
		gather.Set(k, j, 1)
	}
	scatter := mat.NewDense(m, len(in), nil)
	for k, i := range in {
		scatter.Set(i, k, 1)
	}
	g, err := NewGain(gather, plant.Dt)
	if err != nil {
		return nil, err
	}
	s, err := NewGain(scatter, plant.Dt)
	if err != nil {
		return nil, err
	}
	full, err := Series(g, controller)
	if err != nil {
		return nil, fmt.Errorf("Feedback: %w", err)
	}
	if full, err = Series(full, s); err != nil {
		return nil, fmt.Errorf("Feedback: %w", err)
	}
	return full, nil
}
