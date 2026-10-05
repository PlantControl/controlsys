package controlsys

import (
	"fmt"
)

type LoopsensResult struct {
	So *System // output sensitivity: (I + P*C)^{-1}
	To *System // output complementary sensitivity: P*C*(I + P*C)^{-1}
	Si *System // input sensitivity: (I + C*P)^{-1}
	Ti *System // input complementary sensitivity: C*P*(I + C*P)^{-1}
}

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
		return nil, fmt.Errorf("controlsys: loopsens: P outputs %d != C inputs %d: %w", pp, cm, ErrDimensionMismatch)
	}
	if cp != pm {
		return nil, fmt.Errorf("controlsys: loopsens: C outputs %d != P inputs %d: %w", cp, pm, ErrDimensionMismatch)
	}

	Lo, err := Series(C, P)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens: cannot form output loop P*C: %w", err)
	}

	Li, err := Series(P, C)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens: cannot form input loop C*P: %w", err)
	}

	eyeO := makeIdentityGain(pp, Lo.Dt)
	eyeI := makeIdentityGain(pm, Li.Dt)

	So, err := Feedback(eyeO, Lo, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens So: %w", err)
	}

	To, err := Feedback(Lo, eyeO, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens To: %w", err)
	}

	Si, err := Feedback(eyeI, Li, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens Si: %w", err)
	}

	Ti, err := Feedback(Li, eyeI, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens Ti: %w", err)
	}

	return &LoopsensResult{So: So, To: To, Si: Si, Ti: Ti}, nil
}

func makeIdentityGain(n int, dt float64) *System {
	sys, _ := NewGain(eyeOrEmptyDense(n), dt)
	return sys
}
