package controlsys

import (
	"fmt"
	"hash/fnv"
	"math"
	"testing"
)

// fzDecodeGoldenInputs spans every fzDecode branch: discrete and
// continuous, n = 0..6, all m/p, ζ < 1 and ζ ≥ 1, mirrored blocks, origin
// poles, shear and Givens on/off, texture phases and delays. Literals only:
// seed builders using math.Log10 would themselves depend on GOARCH.
var fzDecodeGoldenInputs = []struct {
	kind uint8
	v    [fzNVals]float64
}{
	{0x00, [fzNVals]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 0.37, 0, -10, -10, -10, -10}},
	{0x20, [fzNVals]float64{-12, 1, 0, 0, 0, 0, 0, 0, 0, -10, 0.37, 0, -10, -10, -10, -10}},
	{0x40, [fzNVals]float64{0, 2.5, 0, 0, 0, 0, 0, 0, 0, -10, 0.37, 0, -10, -10, -10, -10}},
	{0x40, [fzNVals]float64{0, -2.5, 0, 0, 0, 0, 0, 0, 0, -10, 0.37, 0, -10, -10, -10, -10}},
	{0x40, [fzNVals]float64{0, 13, 0, 0, 0, 0, 0, 0, 0, -10, 0.3, 0, -10, -10, -10, -10}},
	{0x40, [fzNVals]float64{0, -10, 0, 0, 0, 0, 0, 0, 0, -10, 0.3, 0, -10, -10, -10, -10}},
	{0x61, [fzNVals]float64{-0.30103, 13, 0, 1, 0, 0, 0, 0, -1, -0.5228787452803376, 0.4, 0, -10, -10, -10, -10}},
	{0x60, [fzNVals]float64{-40, 1, 0, 1, 0, 0, 0, 0, 0, -10, 0.37, 0, -10, -10, -10, -10}},
	{0xba, [fzNVals]float64{-0.16, 6, -0.16, 6.3, 3.91, 1, 0, 0, 0, -1, 0.7, 0.2, -10, -10, -10, -10}},
	{0xd6, [fzNVals]float64{-1, 3.5, 0.5, -4, 2, 0, 8, 0.5, 0, -0.3, 1.1, 0, 0, -0.5, 0.2, -10}},
	{0xcb, [fzNVals]float64{0, 4, 1, 3.2, -0.5, -0.5, 0, 0, -1.3, -0.5, -2, 0, 0.5, 0, 0.7, 0.4}},
	{0xab, [fzNVals]float64{-40, 10, 0.4, 1.5, 42, 0, 0, 2, -75, 2, 0, -204, -0.15625, -1, 20, -17.5}},
	{0xd4, [fzNVals]float64{84.6, 1, -44.5, 28, -78, 34, -87, 65.5, 0, -90, 0.37, 0, -1.1111111111111112, -10, -0.4, 23}},
	{0xa2, [fzNVals]float64{-92.33333333333333, 36, 33.5, 3.4, 0, 0, -38, 0.5, 130, 81, 0, 32.5, -100, -10, -5, 6}},
	{0x7f, [fzNVals]float64{0.1, 0.3, 2.4000000000000004, 0.2, -10.625, 75, 86, 0, -0.23299000144533957, 41, 0.015416666666666667, 550, -10, -1, 9, -10}},
	{0x72, [fzNVals]float64{-0.6786402423930551, 13, -0.8522801975522336, 11.64528661442717, -40, 9.730159100767608, -5.7124287797345685, 0.8515788906023601, -1.8699414019192915, -0.9009541575928393, 0.6904845230869261, -1.8615858804687453, 0.37193006986806204, -10, -10, -0.9291681166465303}},
	{0xd7, [fzNVals]float64{-0.8603088727097932, 4.928966109060965, -0.5305909486010396, 13, -1.7603256926868887, 13, 5.5794598346733535, -0.5028651286615842, -1.6756909102435067, -9.522762649906602, -1.2934801749704727, -2.8222193382401946, 0.1393351192955018, -10, -0.4596958443663803, 0.02128779065627474}},
	{0xa1, [fzNVals]float64{-4.840958682425959, 5.75742924209089, -4.840958682425959, 5.75742924209089, -4.974373469618411, 0.8119640876303578, -5, 1e-06, -0.6935062785976278, -9.308799076568224, 0.871442593513744, 9.362982810868214, -10, -10, -10, -10}},
	{0xc9, [fzNVals]float64{0.6413769500533629, 3.225485305446771, -6.920211238942823, 5.194656803421655, 2.5649118927522707, 5.9960354875740425, 0.7321123327177131, 0, 0.8225408446783429, -8.084719954920434, -0.12421061853394999, -14.98391252021944, -10, -10, -10, -10}},
	{0xc0, [fzNVals]float64{0, 0, 2.6999999999999997, 0, -0.2, 0, 0, 0, 21, -7.5, 2.5, -56, -10, -10, -10, -10}},
	{0xa7, [fzNVals]float64{-0.3010299956639812, 45.6, -68, 1, 0, -6, -30, 156, 1.2, 23.7824, 135.4, 48, -10, -2, 47, -1.1111}},
	{0xe9, [fzNVals]float64{1.7, 0.4, -3.3, 12, 0.9, -1.2, 3.3, -7, -3.9, 0.5, -3.7, 999, 1.3, 1.2, -2.9, 1.25}},
}

// fzDecodeDigests returns one FNV-1a digest per fzSpec field (0 for a
// skipped decode), so a GOARCH mismatch names the drifting field.
func fzDecodeDigests(t *testing.T) [][]uint64 {
	fields := []string{"Dt", "S", "Eig", "A", "B", "C", "D", "CondT", "NormJ", "delays", "EigErr", "verdict", "internal"}
	out := make([][]uint64, len(fzDecodeGoldenInputs))
	for i, in := range fzDecodeGoldenInputs {
		out[i] = make([]uint64, len(fields))
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			s := fzDecode(t, in.kind, in.v)
			cx := func(z []complex128) []float64 {
				var r []float64
				for _, e := range z {
					r = append(r, real(e), imag(e))
				}
				return r
			}
			dec, uns := 0.0, float64(s.Unstable())
			if s.Decidable() {
				dec = 1
			}
			vals := [][]float64{
				{s.Dt}, cx(s.S), cx(s.Eig), s.A, s.B, s.C, s.D, {s.CondT}, {s.NormJ},
				append(append(append([]float64{s.Tau}, s.In...), s.Out...), s.IO...),
				{s.EigErr()}, {dec, uns}, nil,
			}
			b2, c2 := fzFRInternalLoop(s)
			vals[len(vals)-1] = append(b2, c2...)
			for f, xs := range vals {
				h := fnv.New64a()
				for _, x := range xs {
					var b [8]byte
					u := math.Float64bits(x)
					for k := range b {
						b[k] = byte(u >> (8 * k))
					}
					h.Write(b[:])
				}
				out[i][f] = h.Sum64()
			}
		})
	}
	for i, d := range out {
		t.Logf("%2d %v", i, fmtDigests(fields, d))
	}
	return out
}

func fmtDigests(fields []string, d []uint64) string {
	s := ""
	for f, x := range d {
		s += fmt.Sprintf(" %s=%016x", fields[f], x)
	}
	return s
}

// TestFzBigMath: the math/big elementary functions agree with package math
// to 1 ulp and reproduce its constants exactly.
func TestFzBigMath(t *testing.T) {
	c := fzBigConsts()
	if fzF64(c.ln2) != math.Ln2 || fzF64(c.ln10) != math.Ln10 || fzF64(c.pi) != math.Pi {
		t.Fatalf("constants ln2=%v ln10=%v pi=%v", fzF64(c.ln2), fzF64(c.ln10), fzF64(c.pi))
	}
	ulps := func(got, want float64) float64 {
		if got == want {
			return 0
		}
		return math.Abs(got-want) / (math.Nextafter(math.Abs(want), math.Inf(1)) - math.Abs(want))
	}
	// Correctly rounded references (Python decimal, 80 digits); math.Pow
	// itself is off by up to 3 ulp at the first three.
	for _, c := range []struct{ x, want float64 }{
		{-5.728571428571429, 1.8682223847710366e-06},
		{7.456428571428571, 28604118.68956633},
		{9.491428571428571, 3100477409.9477873},
		{0.5, 3.1622776601683795},
		{-12.25, 5.62341325190349e-13},
	} {
		if got := fzPow10(c.x); got != c.want {
			t.Errorf("fzPow10(%v) = %v, want %v", c.x, got, c.want)
		}
	}
	for k := -22; k <= 22; k++ {
		if got, want := fzPow10(float64(k)), math.Pow10(k); got != want {
			t.Errorf("fzPow10(%d) = %v, want %v", k, got, want)
		}
	}
	for i := range 4001 {
		x := -1000 + 0.5*float64(i) + 1.0/7
		if u := ulps(fzCos(x), math.Cos(x)); u > 1 {
			t.Errorf("fzCos(%v) off by %v ulp", x, u)
		}
		if u := ulps(fzSin(x), math.Sin(x)); u > 1 {
			t.Errorf("fzSin(%v) off by %v ulp", x, u)
		}
		y := x / 100
		if u := ulps(fzPow10(y), math.Pow(10, y)); u > 4 {
			t.Errorf("fzPow10(%v) off by %v ulp", y, u)
		}
		if e := fzBigExp(fzBig(y * 30)); ulps(fzF64(e), math.Exp(y*30)) > 1 {
			t.Errorf("fzBigExp(%v) off", y*30)
		}
	}
	if got := fzCAbs(complex(3, -4)); got != 5 {
		t.Errorf("fzCAbs(3-4i) = %v", got)
	}
	if got := fzSin(0); got != 0 || fzCos(0) != 1 {
		t.Errorf("sin/cos(0) = %v/%v", got, fzCos(0))
	}
}

// TestFzDecodeArchIndependent pins a digest of fzDecode outputs, so any
// CPU- or GOARCH-dependent arithmetic in decoding fails loudly (QS7HIC).
func TestFzDecodeArchIndependent(t *testing.T) {
	h := fnv.New64a()
	for _, d := range fzDecodeDigests(t) {
		for _, x := range d {
			fmt.Fprintf(h, "%016x", x)
		}
	}
	const want = 0xa882a581dc165ce0
	if got := h.Sum64(); got != want {
		t.Errorf("fzDecode digest %#016x, want %#016x", got, uint64(want))
	}
}
