//go:build !arm64

package controlsys

// hasFMA is false off arm64: math.FMA may be emulated in software (amd64
// without FMA3, as under Rosetta), so twoProductLo splits (Dekker).
const hasFMA = false
