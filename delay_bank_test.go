package controlsys

import (
	"errors"
	"testing"
)

func TestBuildDelayBankAlwaysReturnsModelOrError(t *testing.T) {
	if _, err := buildDiscreteSampleDelayBank(nil, 0, 0.1, 3); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("no channels: err = %v, want ErrDimensionMismatch", err)
	}
	bank, err := buildDiscreteSampleDelayBank([]float64{2, 0}, 2, 0.1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, m, p := bank.Dims(); m != 2 || p != 2 {
		t.Errorf("integer bank dims %dx%d, want 2x2", p, m)
	}
}
