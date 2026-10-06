//go:build amd64 && !noasm && !gccgo && !safe

package controlsys

// gonumKernels mirrors the build constraint of Gonum's internal/asm/f64
// assembly kernels.
const gonumKernels = "amd64-asm"
