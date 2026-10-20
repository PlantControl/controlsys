package controlsys

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func staticDelayOracle(d *mat.Dense, in, out []float64, io *mat.Dense, s complex128) [][]complex128 {
	p, m := d.Dims()
	h := make([][]complex128, p)
	for i := range p {
		h[i] = make([]complex128, m)
		for j := range m {
			tau := 0.0
			if in != nil {
				tau += in[j]
			}
			if out != nil {
				tau += out[i]
			}
			if io != nil {
				tau += io.At(i, j)
			}
			h[i][j] = complex(d.At(i, j), 0) * cmplx.Exp(-s*complex(tau, 0))
		}
	}
	return h
}

func checkStaticPull(t *testing.T, d *mat.Dense, in, out []float64, io *mat.Dense) {
	t.Helper()
	g, err := NewGain(d, 0)
	if err != nil {
		t.Fatal(err)
	}
	if in != nil {
		if err := g.SetInputDelay(in); err != nil {
			t.Fatal(err)
		}
	}
	if out != nil {
		if err := g.SetOutputDelay(out); err != nil {
			t.Fatal(err)
		}
	}
	if io != nil {
		if err := g.SetDelay(io); err != nil {
			t.Fatal(err)
		}
	}
	pl, err := g.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	if pl.Delay != nil || pl.InputDelay != nil || pl.OutputDelay != nil {
		t.Fatal("external delays left")
	}
	for _, w := range []float64{0.3, 1, 2.7, 12} {
		s := complex(0, w)
		want := staticDelayOracle(d, in, out, io, s)
		got, err := pl.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			for j := range want[i] {
				if e := cmplx.Abs(got[i][j] - want[i][j]); e > 1e-12 {
					t.Fatalf("in %v out %v io %v w=%v (%d,%d): got %v want %v", in, out, mat.Formatted(io), w, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

func TestPullDelaysToLFTStaticMixedDelays(t *testing.T) {
	d := mat.NewDense(2, 2, []float64{0.03, -0.24, -0.47, -0.19})
	checkStaticPull(t, d, []float64{0, 1}, []float64{1, 0}, mat.NewDense(2, 2, []float64{1, 0, 0, 0}))
	checkStaticPull(t, d, []float64{1, 0}, []float64{1, 0}, mat.NewDense(2, 2, []float64{0, 1, 0, 0}))
	checkStaticPull(t, mat.NewDense(2, 3, []float64{0.7, -1.1, 0.4, 0.2, 0.9, -0.6}),
		[]float64{1, 2, 3}, []float64{2, 1}, mat.NewDense(2, 3, []float64{0, 1, 2, 0, 1, 2}))

	rng := rand.New(rand.NewPCG(13, 5))
	for range 300 {
		p, m := 1+rng.IntN(3), 1+rng.IntN(3)
		dd := mat.NewDense(p, m, nil)
		io := mat.NewDense(p, m, nil)
		for i := range p {
			for j := range m {
				if rng.IntN(5) > 0 {
					dd.Set(i, j, rng.NormFloat64())
				}
				io.Set(i, j, float64(rng.IntN(3))*0.5)
			}
		}
		in := make([]float64, m)
		for j := range in {
			in[j] = float64(rng.IntN(3)) * 0.7
		}
		out := make([]float64, p)
		for i := range out {
			out[i] = float64(rng.IntN(3)) * 0.3
		}
		checkStaticPull(t, dd, in, out, io)
	}
}

func TestLFTStaticBlocksWithMixedDelays(t *testing.T) {
	m, err := NewGain(mat.NewDense(2, 3, []float64{0.8, -0.35, 0.6, 0.45, 1.2, -0.3}), 0)
	if err != nil {
		t.Fatal(err)
	}
	const tau = 19.95
	if err := m.SetInputDelay([]float64{tau, 2 * tau, 3 * tau}); err != nil {
		t.Fatal(err)
	}
	if err := m.SetOutputDelay([]float64{2 * tau, tau}); err != nil {
		t.Fatal(err)
	}
	if err := m.SetDelay(mat.NewDense(2, 3, []float64{0, tau, 2 * tau, 0, tau, 2 * tau})); err != nil {
		t.Fatal(err)
	}
	k, err := NewGain(mat.NewDense(1, 1, []float64{-1.6787}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := k.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	if err := k.SetOutputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	cl, err := LFT(m, k)
	if err != nil {
		t.Fatal(err)
	}
	if _, mc, pc := cl.Dims(); mc != 2 || pc != 1 {
		t.Fatalf("dims %dx%d, want 1x2", pc, mc)
	}
	upper, err := LFT(k, m)
	if err != nil {
		t.Fatal(err)
	}
	if _, mc, pc := upper.Dims(); mc != 2 || pc != 1 {
		t.Fatalf("upper dims %dx%d, want 1x2", pc, mc)
	}
	for _, w := range []float64{0.1, 1, 12} {
		s := complex(0, w)
		hm, err := m.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		hk, err := k.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		got, err := cl.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		up, err := upper.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		kk := hk[0][0]
		loop := kk / (1 - hm[1][2]*kk)
		uloop := kk / (1 - hm[0][0]*kk)
		for j := range 2 {
			want := hm[0][j] + hm[0][2]*loop*hm[1][j]
			if e := cmplx.Abs(got[0][j] - want); e > 1e-10 {
				t.Fatalf("lower w=%v z<-w%d: got %v want %v", w, j, got[0][j], want)
			}
			want = hm[1][j+1] + hm[1][0]*uloop*hm[0][j+1]
			if e := cmplx.Abs(up[0][j] - want); e > 1e-10 {
				t.Fatalf("upper w=%v z<-w%d: got %v want %v", w, j, up[0][j], want)
			}
		}
	}
}
