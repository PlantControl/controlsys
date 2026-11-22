package controlsys

import "golang.org/x/sys/cpu"

// hasFMA reports a fused multiply-add in hardware; without it (Rosetta),
// math.FMA is emulated in software and twoProductLo splits instead.
var hasFMA = cpu.X86.HasFMA
