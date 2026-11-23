//go:build !lyaptiming

package controlsys

// lyapunovUncertifiedBaseline is false in every build but the timing
// harness's (build tag lyaptiming), where it lets one process time the
// uncertified baseline against the certified path.
const lyapunovUncertifiedBaseline = false
