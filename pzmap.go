package controlsys

import (
	"errors"
	"fmt"
)

// PzmapResult holds the poles and transmission zeros of a model.
type PzmapResult struct {
	Poles []complex128
	Zeros []complex128
}

// Pzmap returns the poles and transmission zeros of sys, as the
// [p,z] = pzmap(sys) outputs of MATLAB pzmap
// (https://www.mathworks.com/help/control/ref/lti.pzmap.html). Zeros that
// Zeros does not certify within the data's rounding terms are returned
// with the result and an error wrapping ErrAccuracyUnverified.
func Pzmap(sys *System) (*PzmapResult, error) {
	if err := requireFiniteSystem("Pzmap", sys); err != nil {
		return nil, err
	}
	poles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("Pzmap: %w", err)
	}
	zeros, err := sys.Zeros()
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		return nil, fmt.Errorf("Pzmap: %w", err)
	}
	res := &PzmapResult{Poles: poles, Zeros: zeros}
	if err != nil {
		return res, fmt.Errorf("Pzmap: %w", err)
	}
	return res, nil
}
