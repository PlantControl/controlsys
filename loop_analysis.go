package controlsys

import "fmt"

type sisoLoopModel struct {
	sys *System
}

func newSISOLoopModel(sys *System, context string) (*sisoLoopModel, error) {
	if err := requireFiniteSystem(context, sys); err != nil {
		return nil, err
	}
	_, m, p := sys.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("%s: SISO model required: %w", context, ErrNotSISO)
	}
	return &sisoLoopModel{sys: sys}, nil
}
