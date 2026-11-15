package controlsys

import "testing"

// benchDescriptorChainRows times TransferFunction, IsProper and Step on
// singular-E chain descriptors where the float reduction does not verify
// (the exact tier's fallback rows) and on one where it does. Each
// iteration scales D by a distinct factor 1 + kε so no verdict is reused
// from a memo.
func benchDescriptorChainRows(b *testing.B, mk func(t *testing.T) *System) {
	sys := mk(&testing.T{})
	d := sys.D.At(0, 0)
	b.ResetTimer()
	for i := range b.N {
		c := sys.Copy()
		c.D.Set(0, 0, d*(1+float64(i+1)*eps())+float64(i+1)*0x1p-60)
		_, _ = c.TransferFunction(nil)
		_, _ = c.IsProper()
		_, _ = Step(c, 1)
	}
}

func BenchmarkDescriptorChainFallback_Stiff2(b *testing.B) {
	benchDescriptorChainRows(b, func(t *testing.T) *System { s, _, _ := stiffPoleChainDescriptor(t, 2); return s })
}

func BenchmarkDescriptorChainFallback_Refuted0(b *testing.B) {
	benchDescriptorChainRows(b, func(t *testing.T) *System { s, _ := refutedChainDescriptor(t, 0); return s })
}

func BenchmarkDescriptorChainFallback_Small1e8_7(b *testing.B) {
	benchDescriptorChainRows(b, func(t *testing.T) *System { s, _ := smallSTermChainDescriptor(t, 7, 1e-8); return s })
}

func BenchmarkDescriptorChainFallback_LargePole0_4(b *testing.B) {
	benchDescriptorChainRows(b, func(t *testing.T) *System { s, _ := largePoleChainDescriptor(t, 0, []int{4}, 1e-4); return s })
}

func BenchmarkDescriptorChainFloatVerified_Stiff159(b *testing.B) {
	benchDescriptorChainRows(b, func(t *testing.T) *System { s, _, _ := stiffPoleChainDescriptor(t, 159); return s })
}
