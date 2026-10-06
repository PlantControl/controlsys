//go:build amd64 && !noasm && !gccgo && !safe

package controlsys

// gonumKernels mirrors the build constraint of Gonum's internal/asm/f64
// assembly kernels.
const gonumKernels = "amd64-asm"

// Hessenberg sweep cost in twentieths of a dense point, fitted on AMD EPYC
// (docs/benchmarks/frequency-dispatch): setup 1.4 points, then 0.75·c/n per
// point.
const (
	sweepSetup20 = 28
	sweepPoint20 = 15
)
