package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
	"time"
)

func TestPolyDegree(t *testing.T) {
	tests := []struct {
		p    Poly
		want int
	}{
		{Poly{1, -3, 2}, 2},
		{Poly{5}, 0},
		{Poly{}, -1},
	}
	for _, tt := range tests {
		if got := tt.p.Degree(); got != tt.want {
			t.Errorf("Poly%v.Degree() = %d, want %d", tt.p, got, tt.want)
		}
	}
}

func TestPolyEval(t *testing.T) {
	p := Poly{1, -3, 2} // s²-3s+2 = (s-1)(s-2)
	if got := p.Eval(1); cmplx.Abs(got) > 1e-14 {
		t.Errorf("p(1) = %v, want 0", got)
	}
	if got := p.Eval(2); cmplx.Abs(got) > 1e-14 {
		t.Errorf("p(2) = %v, want 0", got)
	}
	if got := p.Eval(0); cmplx.Abs(got-2) > 1e-14 {
		t.Errorf("p(0) = %v, want 2", got)
	}
}

func TestPolyEvalComplex(t *testing.T) {
	p := Poly{1, 0, 1} // s²+1, roots at ±i
	if got := p.Eval(1i); cmplx.Abs(got) > 1e-14 {
		t.Errorf("p(i) = %v, want 0", got)
	}
}

func TestPolyMul(t *testing.T) {
	p := Poly{1, -1} // s-1
	q := Poly{1, -2} // s-2
	got := p.Mul(q)
	want := Poly{1, -3, 2} // s²-3s+2
	if !got.Equal(want, 1e-14) {
		t.Errorf("(%v)*(%v) = %v, want %v", p, q, got, want)
	}
}

func TestPolyMulEmpty(t *testing.T) {
	got := Poly{}.Mul(Poly{1, 2})
	if len(got) != 0 {
		t.Errorf("empty*p = %v, want empty", got)
	}
}

func TestPolyAdd(t *testing.T) {
	p := Poly{1, 0, 0} // s²
	q := Poly{3, 2}     // 3s+2
	got := p.Add(q)
	want := Poly{1, 3, 2}
	if !got.Equal(want, 1e-14) {
		t.Errorf("(%v)+(%v) = %v, want %v", p, q, got, want)
	}
}

func TestPolyAddEmpty(t *testing.T) {
	p := Poly{1, 2}
	got := Poly{}.Add(p)
	if !got.Equal(p, 1e-14) {
		t.Errorf("empty+p = %v, want %v", got, p)
	}
}

func TestPolyMonic(t *testing.T) {
	p := Poly{2, -6, 4}
	got, err := p.Monic()
	if err != nil {
		t.Fatal(err)
	}
	want := Poly{1, -3, 2}
	if !got.Equal(want, 1e-14) {
		t.Errorf("Monic(%v) = %v, want %v", p, got, want)
	}
}

func TestPolyMonicError(t *testing.T) {
	_, err := Poly{0, 1, 2}.Monic()
	if err == nil {
		t.Error("expected error for zero leading coefficient")
	}
	_, err = Poly{}.Monic()
	if err == nil {
		t.Error("expected error for empty polynomial")
	}
}

func TestPolyIsMonic(t *testing.T) {
	p1 := Poly{1, -3, 2}
	if !p1.IsMonic() {
		t.Error("expected monic")
	}
	p2 := Poly{2, -3, 2}
	if p2.IsMonic() {
		t.Error("expected not monic")
	}
}

func TestPolyScale(t *testing.T) {
	p := Poly{1, -3, 2}
	got := p.Scale(2)
	want := Poly{2, -6, 4}
	if !got.Equal(want, 1e-14) {
		t.Errorf("Scale(%v, 2) = %v, want %v", p, got, want)
	}
}

func TestPolyEqual(t *testing.T) {
	a := Poly{1, 2, 3}
	b := Poly{1, 2, 3}
	if !a.Equal(b, 1e-14) {
		t.Error("equal polys should be equal")
	}
	c := Poly{1, 2}
	if c.Equal(a, 1e-14) {
		t.Error("different length polys should not be equal")
	}
	d := Poly{1, 2, 4}
	if a.Equal(d, 0) {
		t.Error("different polys should not be equal")
	}
}

func TestPolyEvalEmpty(t *testing.T) {
	empty := Poly{}
	got := empty.Eval(5)
	if got != 0 {
		t.Errorf("empty.Eval(5) = %v, want 0", got)
	}
}

func TestPolyHornerAccuracy(t *testing.T) {
	// Wilkinson-style: (s-1)(s-2)...(s-5)
	p := Poly{1, -1}
	for i := 2; i <= 5; i++ {
		p = p.Mul(Poly{1, float64(-i)})
	}
	for i := 1; i <= 5; i++ {
		got := p.Eval(complex(float64(i), 0))
		if cmplx.Abs(got) > 1e-10 {
			t.Errorf("p(%d) = %v, want ≈0", i, got)
		}
	}
	got := real(p.Eval(0))
	want := -120.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("p(0) = %v, want %v", got, want)
	}
}

func TestPolyRootsRejectsNonFiniteWithoutHanging(t *testing.T) {
	cases := []Poly{
		{1, math.NaN(), 2},
		{1, 2, 3, math.Inf(1)},
		{math.NaN(), 1, 2},
		{0, math.Inf(-1), 1, 2},
		{1, math.NaN()},
	}
	for _, p := range cases {
		var err error
		finishesWithin(t, 5*time.Second, func() { _, err = p.Roots() })
		if !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "Poly.Roots: ") {
			t.Fatalf("%v.Roots(): %v", p, err)
		}
	}
}
