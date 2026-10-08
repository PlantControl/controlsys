//go:build !amd64 || noasm || gccgo || safe

package controlsys

const gonumKernels = "pure-go"

// Hessenberg sweep cost in twentieths of a dense point, fitted on Apple M1 Pro
// and the macos-latest runner: setup 6 points, then 0.9·c/n per point.
const (
	sweepSetup20      = 120
	sweepSetupFixed20 = 0
	sweepPoint20      = 18
	sweepPointFloor20 = 0
)
