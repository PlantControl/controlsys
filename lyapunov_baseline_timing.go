//go:build lyaptiming

package controlsys

// lyapunovUncertifiedBaseline, in the timing harness's build only, skips
// the Lyapunov certificate (see lyapunov_baseline.go).
var lyapunovUncertifiedBaseline = false
