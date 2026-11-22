//go:build !amd64

package controlsys

// hasFMA: arm64 and the other supported targets fuse in hardware.
const hasFMA = true
