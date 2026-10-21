package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestZPKConjugatePairNegativeImagFirst(t *testing.T) {
	cases := []struct {
		poles []complex128
		want  []float64
	}{
		{[]complex128{-1i, 1i}, []float64{1, 0, 1}},
		{[]complex128{-1 - 2i, -1 + 2i}, []float64{1, 2, 5}},
		{[]complex128{-1 - 2i, -3 + 1i, -1 + 2i, -3 - 1i}, []float64{1, 8, 27, 50, 50}},
	}
	for _, c := range cases {
		z, err := NewZPK(nil, c.poles, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		tf, err := z.transferFunction()
		if err != nil {
			t.Fatal(err)
		}
		got := tf.Den[0]
		if len(got) != len(c.want) {
			t.Fatalf("poles %v: den %v, want %v", c.poles, got, c.want)
		}
		for i := range got {
			if math.Abs(got[i]-c.want[i]) > 1e-12*(1+math.Abs(c.want[i])) {
				t.Fatalf("poles %v: den %v, want %v", c.poles, got, c.want)
			}
		}
	}

	z, err := NewZPK([]complex128{-3, -4}, []complex128{-1 - 2i, -1 + 2i}, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := z.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := res.Sys.Dims(); n != 2 {
		t.Fatalf("states = %d, want 2", n)
	}
	if d := res.Sys.D.At(0, 0); math.Abs(d-2) > 1e-12 {
		t.Fatalf("D = %v, want 2", d)
	}
}

func TestMatchedAliasedComplexPoles(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1, []float64{0, 1, -16.01, -0.2}, []float64{0, 1}, []float64{16.01, 0}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, ts := range []float64{0.5, 1, 1.2} {
		sd, err := sys.C2D(ts, C2DOptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatalf("Ts=%v: %v", ts, err)
		}
		if n, _, _ := sd.Dims(); n != 2 {
			t.Fatalf("Ts=%v: states = %d, want 2", ts, n)
		}
		p1 := cmplx.Exp(complex(-0.1, 4) * complex(ts, 0))
		p2 := cmplx.Conj(p1)
		h := func(z complex128) complex128 { return (z + 1) / ((z - p1) * (z - p2)) }
		k := 1 / h(1)
		for _, z := range []complex128{1, cmplx.Exp(0.3i), cmplx.Exp(2i), -0.5} {
			got, err := sd.EvalFr(z)
			if err != nil {
				t.Fatal(err)
			}
			want := k * h(z)
			if d := cmplx.Abs(got[0][0] - want); d > 1e-10*(1+cmplx.Abs(want)) {
				t.Fatalf("Ts=%v z=%v: H = %v, want %v", ts, z, got[0][0], want)
			}
		}
	}
}

func TestZPKTransferFunctionNearbyRealAndComplexPoles(t *testing.T) {
	cases := []struct {
		name   string
		poles  [][]complex128
		denDeg int
		pts    []complex128
	}{
		{"tiny", [][]complex128{{1e-14 + 1e-14i, 1e-14 - 1e-14i}, {1e-15}}, 3, []complex128{10, 2i, -3 + 1i}},
		{"unit", [][]complex128{{1 + 1e-15i, 1 - 1e-15i}, {1}}, 3, []complex128{3, 2i, -3 + 1i}},
		{"realFirst", [][]complex128{{1}, {1 - 1e-15i, 1 + 1e-15i}}, 3, []complex128{3, 2i}},
		{"pairAndReal", [][]complex128{{1 + 1e-15i, 1 - 1e-15i, 1}, {1}}, 3, []complex128{3, 2i}},
		{"shared",[][]complex128{{-1 + 2i, -1 - 2i}, {-3, -1 - 2i, -1 + 2i}}, 3, []complex128{1, 2i}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			z, err := NewZPKMIMO([][][]complex128{{nil, nil}}, [][][]complex128{c.poles}, [][]float64{{1, 2}}, 0)
			if err != nil {
				t.Fatal(err)
			}
			tf, err := z.TransferFunction()
			if err != nil {
				t.Fatal(err)
			}
			if got := len(tf.Den[0]) - 1; got != c.denDeg {
				t.Fatalf("den %v degree %d, want %d", tf.Den[0], got, c.denDeg)
			}
			for j, num := range tf.Num[0] {
				if len(num) > len(tf.Den[0]) {
					t.Fatalf("num[%d] %v improper over den %v", j, num, tf.Den[0])
				}
			}
			for _, s := range c.pts {
				got, err := tf.Eval(s)
				if err != nil {
					t.Fatal(err)
				}
				for j, ps := range c.poles {
					want := complex(float64(j+1), 0)
					for _, p := range ps {
						want /= s - p
					}
					if d := cmplx.Abs(got[0][j] - want); d > 1e-12*cmplx.Abs(want) {
						t.Errorf("tf(%v)[%d] = %v, want %v", s, j, got[0][j], want)
					}
				}
			}
		})
	}
}
