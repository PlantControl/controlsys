package controlsys

import (
	"fmt"
	"testing"
)

func BenchmarkSSToZPK_MIMO(b *testing.B) {
	for _, d := range [][3]int{{8, 3, 3}, {12, 2, 4}, {20, 3, 3}} {
		b.Run(fmt.Sprintf("n%d_m%d_p%d", d[0], d[1], d[2]), func(b *testing.B) {
			sys := benchSys(b, d[0], d[1], d[2])
			b.ReportAllocs()
			for b.Loop() {
				if _, err := sys.ZPKModel(nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
