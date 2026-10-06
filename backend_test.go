package controlsys

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
)

// TestNativeBackend reports the architecture and Gonum backend of this test
// binary. CI and scripts/benchcmp set the CONTROLSYS_EXPECT_* variables to
// assert them; unset variables are only reported.
func TestNativeBackend(t *testing.T) {
	backend := fmt.Sprintf("%T/%s", blas64.Implementation(), gonumKernels)
	fmt.Printf("controlsys-backend goos=%s goarch=%s backend=%s\n", runtime.GOOS, runtime.GOARCH, backend)
	if want := os.Getenv("CONTROLSYS_EXPECT_GOARCH"); want != "" && want != runtime.GOARCH {
		t.Errorf("GOARCH = %s, want %s", runtime.GOARCH, want)
	}
	if want := os.Getenv("CONTROLSYS_EXPECT_BACKEND"); want != "" && want != backend {
		t.Errorf("backend = %s, want %s", backend, want)
	}
}
