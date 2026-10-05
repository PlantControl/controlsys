package controlsys

import (
	"fmt"
)

// LoopsensResult holds the sensitivity functions of the negative-feedback
// loop of plant P and controller C, a subset of MATLAB loopsens (Li, Lo,
// PSi, CSo and Stable are not returned; form them with Series/Feedback).
type LoopsensResult struct {
	So *System // output sensitivity: (I + P*C)^{-1}
	To *System // output complementary sensitivity: P*C*(I + P*C)^{-1}
	Si *System // input sensitivity: (I + C*P)^{-1}
	Ti *System // input complementary sensitivity: C*P*(I + C*P)^{-1}
}

// Loopsens computes the output and input sensitivities and complementary
// sensitivities of the loop u = -C·y, y = P·u, like MATLAB loopsens(P,C).
// P and C must be compatible (C has P's outputs as inputs and P's inputs as
// outputs): ErrDimensionMismatch otherwise; nil models return
// ErrInvalidArgument.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.loopsens.html.
func Loopsens(P, C *System) (*LoopsensResult, error) {
	if err := requireSystem("Loopsens", P); err != nil {
		return nil, err
	}
	if err := requireSystem("Loopsens", C); err != nil {
		return nil, err
	}

	_, pm, pp := P.Dims()
	_, cm, cp := C.Dims()

	if pp != cm {
		return nil, fmt.Errorf("Loopsens: P outputs %d != C inputs %d: %w", pp, cm, ErrDimensionMismatch)
	}
	if cp != pm {
		return nil, fmt.Errorf("Loopsens: C outputs %d != P inputs %d: %w", cp, pm, ErrDimensionMismatch)
	}

	Lo, err := Series(C, P)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: cannot form output loop P*C: %w", err)
	}

	Li, err := Series(P, C)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: cannot form input loop C*P: %w", err)
	}

	eyeO, err := makeIdentityGain(pp, Lo.Dt)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: %w", err)
	}
	eyeI, err := makeIdentityGain(pm, Li.Dt)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: %w", err)
	}

	So, err := Feedback(eyeO, Lo, -1)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: So: %w", err)
	}

	To, err := Feedback(Lo, eyeO, -1)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: To: %w", err)
	}

	Si, err := Feedback(eyeI, Li, -1)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: Si: %w", err)
	}

	Ti, err := Feedback(Li, eyeI, -1)
	if err != nil {
		return nil, fmt.Errorf("Loopsens: Ti: %w", err)
	}

	return &LoopsensResult{So: So, To: To, Si: Si, Ti: Ti}, nil
}

func makeIdentityGain(n int, dt float64) (*System, error) {
	return NewGain(eyeOrEmptyDense(n), dt)
}
