package controlsys

import "fmt"

// PzmapResult holds the poles and transmission zeros of a model.
type PzmapResult struct {
	Poles []complex128
	Zeros []complex128
}

// Pzmap returns the poles and transmission zeros of sys, as the
// [p,z] = pzmap(sys) outputs of MATLAB pzmap
// (https://www.mathworks.com/help/control/ref/lti.pzmap.html).
func Pzmap(sys *System) (*PzmapResult, error) {
	if err := requireFiniteSystem("Pzmap", sys); err != nil {
		return nil, err
	}
	poles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("Pzmap: %w", err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		return nil, fmt.Errorf("Pzmap: %w", err)
	}
	return &PzmapResult{Poles: poles, Zeros: zeros}, nil
}
