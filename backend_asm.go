//go:build amd64 && !noasm && !gccgo && !safe

package controlsys

// gonumKernels mirrors the build constraint of Gonum's internal/asm/f64
// assembly kernels.
const gonumKernels = "amd64-asm"

// Hessenberg sweep cost in twentieths of a dense point, fitted on AMD EPYC:
// setup 1 + 16/n points, then 0.5·c/n + 0.15 per point.
const (
	sweepSetup20      = 20
	sweepSetupFixed20 = 320
	sweepPoint20      = 10
	sweepPointFloor20 = 3
)
