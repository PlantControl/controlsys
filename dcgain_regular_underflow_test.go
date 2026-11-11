package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A DC solve X = (−A)⁻¹B that underflows to 0 used to pass the regular-path
// certificate: the residual B is nonzero but η = ν·ρ rounded to 0, so DCGain
// returned 0 with a nil error for gains near 1e-236..1e-300 (OCRNXB; DZSEVY
// MIMO census rows mhd#52, mosc#608, mhd#70, mhd#130). B holds subnormal
// entries, frozen as hex literals.
func TestDCGainRegularPathUnderflowedSolve(t *testing.T) {
	cases := []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
	}{
		{
			name: "siso order 2",
			n:    2, m: 1, p: 1,
			a: []float64{-0x1.cce55995f1d73p+447, -0x1.a32c6ce193042p+447, 0x1.c0ec862b5e332p+386, -0x1.0b726bec3496cp+449},
			b: []float64{-0x0.00253f10a069bp-1022, -0x1.7985baa3cf616p-874},
			c: []float64{-0x1.2ef4ea4e8ef6dp+535, 0x1.abd222365fcd7p+541},
			d: []float64{0},
		},
		{
			name: "oscillatory order 3",
			n:    3, m: 1, p: 1,
			a: []float64{
				-0x1.43b838ff20970p+852, 0x1.35523409cc660p+852, 0,
				0, -0x1.f604932d71759p+787, 0x1.097a7c4eeddc0p+838,
				0, -0x1.097a7c4eeddc0p+838, -0x1.f604932d71759p+787,
			},
			b: []float64{0x0.00182226e461fp-1022, -0x1.5440eb12a356bp-873, 0x1.21b8f1158e31bp-887},
			c: []float64{0x1.15be9cf456dfdp+859, -0x1.8ed629f9f9e1ep+858, -0x1.5f73511fb8779p+858},
			d: []float64{0},
		},
		{
			name: "one underflowed column of two",
			n:    4, m: 2, p: 2,
			a: []float64{
				-0x1.7ee901bdaf54cp+426, 0x1.f36fc22b0e8cbp+426, 0, 0,
				0, -0x1.50a935030f1dfp+427, 0x1.90fe72200bfbdp+426, 0,
				0, 0, -0x1.e102c00d7d9ddp+428, -0x1.f018d0e4427d8p+426,
				0, 0, 0x1.edbff58bb129bp+374, -0x1.dfff31c9c498bp+427,
			},
			b: []float64{
				0x1.2160892d17ea6p-82, -0x1.93c0789720665p-906,
				-0x1.4b1f84268e686p-85, 0x1.c185c41ccc1eep-872,
				-0x1.eaed0c227c7b7p-83, -0x1.71d8373378bddp-867,
				0x1.32cddaf035fadp-84, 0x0.0000058ce45e8p-1022,
			},
			c: []float64{
				-0x1.0dff6b4f25b19p+501, 0x1.31c5d974dde85p+503, 0x1.157e02d0dec5dp+501, 0x1.0948af4239ec2p+503,
				0x1.deca02d8ed9e9p+501, -0x1.a29f93d8bd8a5p+501, -0x1.6ce8a02b9ea60p+501, -0x1.26b9dd6b292d8p+502,
			},
			d: make([]float64, 4),
		},
		{
			name: "mimo every column underflowed",
			n:    2, m: 3, p: 3,
			a: []float64{-0x1.5dcb3a2646a28p+994, -0x1.31d3bf76d875dp+994, 0x1.65a244b4505d2p+942, -0x1.0ac875496cf5ep+995},
			b: []float64{
				0x1.2810cde37b188p-103, 0x1.5ab0d2636e504p-108, -0x1.3bcd3455c9ba6p-105,
				-0x1.e4cb4a4c77635p-108, 0x1.6590fcd300460p-104, 0x1.0caca944a2618p-104,
			},
			c: []float64{
				0x1.b8516c23ec67fp+103, -0x1.302047895a728p+103,
				0x1.0b039f425e730p+102, -0x1.2caeebc586a14p+104,
				0x1.596f2a14cf89ep+101, -0x1.5d9974f3f39dbp+103,
			},
			d: make([]float64, 9),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := dcGainCertTestSystem(t, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, 0)
			if !dcGainRegularPath(sys) {
				t.Fatal("model no longer reaches the regular DC path")
			}
			g, err := sys.DCGain()
			if err != nil {
				t.Fatalf("DCGain error %v, want nil", err)
			}
			ex, ok := dcGainOwnRoundingExcess(sys, g)
			if !ok {
				t.Fatal("−A exactly singular")
			}
			if ex > 0 {
				t.Fatalf("DCGain = %v exceeds the exact gain's own rounding by %g", mat.Formatted(g), ex)
			}
		})
	}
}

// One B column spanning 1e5..1e-317 with subnormal entries (DZSEVY MIMO
// census mgraded#840, filed as OCRNXB with DCGain 2.2e-32 against an exact
// -2.04e-31 and a nil error): a nil error must be the exact gain, otherwise
// ErrAccuracyUnverified.
func TestDCGainWideSpanInputColumn(t *testing.T) {
	sys := dcGainCertTestSystem(t, 3, 1, 2,
		[]float64{
			-0x1.207cb06cbb859p+160, -0x1.e1902ff277620p-14, -0x1.09293d13d478bp+1,
			-0x1.747b114da1ac4p+324, -0x1.e1182f378a35bp+158, 0x1.c195635dee639p+168,
			0x1.df9e7004b1bd6p+311, 0x1.0190a71ea3d5ap+148, -0x1.445a271439cb7p+160,
		},
		[]float64{-0x0.0005913730fccp-1022, 0x1.3d9d74e838270p+20, -0x0.00000003c5ca0p-1022},
		[]float64{
			-0x1.d37bf400b7cf5p+203, -0x1.0fcd088fe65cap+35, -0x1.09df4f9cf455cp+48,
			0x1.84bfd0cf4ac63p+202, -0x1.51b36aa45166bp+38, 0x1.41faa9681d580p+43,
		},
		[]float64{0, 0}, 0)
	g, err := sys.DCGain()
	if err != nil {
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("DCGain error %v, want nil or ErrAccuracyUnverified", err)
		}
		return
	}
	ex, ok := dcGainOwnRoundingExcess(sys, g)
	if !ok {
		t.Fatal("−A exactly singular")
	}
	if ex > 0 {
		t.Fatalf("DCGain = %v exceeds the exact gain's own rounding by %g", mat.Formatted(g), ex)
	}
}
