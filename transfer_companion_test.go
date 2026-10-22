package controlsys

import (
	"math/cmplx"
	"testing"
)

// An ill-scaled companion chain whose Dgebal realization loses every state
// (Num=[0]) while the Osborne one is right once its Markov noise is trimmed:
// the realization choice must compare trimmed results (WGMRIE). Smaller
// scales already converted and guard the base choice.
func TestTransferFunctionIllScaledCompanion(t *testing.T) {
	for _, sc := range []float64{64, 128, 256, 512, 1024} {
		r := sc / 1024
		poles := []float64{0, -1536 * r, -2048 * r, -2048 * r, -3072 * r}
		den := []float64{1}
		for _, p := range poles {
			nd := make([]float64, len(den)+1)
			for i, d := range den {
				nd[i] += d
				nd[i+1] -= d * p
			}
			den = nd
		}
		const n = 5
		a := make([]float64, n*n)
		for i := range n - 1 {
			a[i*n+i+1] = 1
		}
		for j := range n {
			a[(n-1)*n+j] = -den[n-j]
		}
		k := 2.251799813685248e15 * r * r * r * r
		z := 1024 * r
		sys, err := NewFromSlices(n, 1, 1, a, []float64{0, 0, 0, 0, 1}, []float64{k, k / z, 0, 0, 0}, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("sc=%v: %v", sc, err)
		}
		tf := res.TF
		if res.RowDegrees[0] != n || len(tf.Den[0]) != n+1 || len(tf.Num[0][0]) != 2 {
			t.Fatalf("sc=%v: degrees %v, num %v, den %v", sc, res.RowDegrees, tf.Num[0][0], tf.Den[0])
		}
		for _, w := range []float64{10, 300, 1000, 2048, 5000, 3e4} {
			s := complex(0, w*r)
			want := complex(k/z, 0) * (s + complex(z, 0))
			for _, p := range poles {
				want /= s - complex(p, 0)
			}
			g, err := tf.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			if e := cmplx.Abs(g[0][0]-want) / cmplx.Abs(want); e > 1e-8 {
				t.Errorf("sc=%v w=%v: G=%v want %v (rel %.2g)", sc, w*r, g[0][0], want, e)
			}
		}
	}
}
