package controlsys

import (
	"errors"
	"strings"
	"testing"
)

func TestSeriesDomainMismatchNamesSampleTimes(t *testing.T) {
	d := makeSISO(0.5, 1, 1, 0)
	d.Dt = 0.1
	_, err := Series(d, makeSISO(-1, 1, 1, 0))
	if !errors.Is(err, ErrDomainMismatch) || !strings.Contains(err.Error(), "sample times 0.1 and 0") {
		t.Errorf("err = %v, want sample-time detail wrapping ErrDomainMismatch", err)
	}
}
