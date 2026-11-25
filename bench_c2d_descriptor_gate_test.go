package controlsys

import "testing"

// BenchmarkC2DDescriptorRoutes times C2D of frozen EJDS3U rows on each
// descriptor route: band parts (cond/13), dense fold passing the DC check
// (cond/644 at 0.01), fold failing it with the real Schur retry (cond/644
// at 1, cond/111 at 0.01), the reduced standard form (stfr/45) and the
// refused band split (strad1m/47).
func BenchmarkC2DDescriptorRoutes(b *testing.B) {
	for _, tc := range []struct {
		row    string
		dt     float64
		method C2DMethod
	}{
		{"cond/13", 0.01, C2DMethodZOH},
		{"cond/13", 0.01, C2DMethodTustin},
		{"cond/644", 0.01, C2DMethodZOH},
		{"cond/644", 1, C2DMethodZOH},
		{"cond/111", 0.01, C2DMethodZOH},
		{"stfr/45", 0.01, C2DMethodZOH},
		{"strad1m/47", 0.01, C2DMethodZOH},
	} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == tc.row {
				var err error
				if sys, err = r.system(); err != nil {
					b.Fatal(err)
				}
			}
		}
		b.Run(tc.row+"/"+string(tc.method)+"/"+formatBenchDt(tc.dt), func(b *testing.B) {
			for b.Loop() {
				_, _ = sys.C2D(tc.dt, C2DOptions{Method: tc.method})
			}
		})
	}
}

func formatBenchDt(dt float64) string {
	switch dt {
	case 0.01:
		return "dt0.01"
	case 1:
		return "dt1"
	}
	return "dt"
}
