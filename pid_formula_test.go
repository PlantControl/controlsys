package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func TestPIDIndependentDiscreteFormulas(t *testing.T) {
	for _, fi := range []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal} {
		for _, fd := range []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal} {
			p := mustPID2(t, 2, 3, .4, .7, .6, .2, .1, WithPIDFormulas(fi, fd))
			sys, err := p.System()
			if err != nil {
				t.Fatal(err)
			}
			weights := map[PIDFormula]float64{ForwardEuler: 0, BackwardEuler: 1, Trapezoidal: .5}
			qi, qd := weights[fi], weights[fd]
			omega := []float64{.1, 1, 12, 25}
			response, err := sys.FreqResponse(omega)
			if err != nil {
				t.Fatal(err)
			}
			for k, w := range omega {
				z := cmplx.Exp(complex(0, w*p.Dt))
				integral := complex(p.Dt, 0) * (complex(qi, 0)*z + complex(1-qi, 0)) / (z - 1)
				derivative := (z - 1) / (complex(p.Tf, 0)*(z-1) + complex(p.Dt, 0)*(complex(qd, 0)*z+complex(1-qd, 0)))
				want := []complex128{complex(p.Kp*p.B, 0) + complex(p.Ki, 0)*integral + complex(p.Kd*p.C, 0)*derivative, -complex(p.Kp, 0) - complex(p.Ki, 0)*integral - complex(p.Kd, 0)*derivative}
				for j := range 2 {
					if cmplx.Abs(response.At(k, 0, j)-want[j]) > 1e-10 {
						t.Fatalf("formulas %v,%v frequency %g channel %d: got %v want %v", fi, fd, w, j, response.At(k, 0, j), want[j])
					}
				}
			}
			// The oracle updates the explicit I and D difference equations, independently
			// of the realized state-space matrices and without using a conversion helper.
			state := make([]float64, sys.A.RawMatrix().Rows)
			previousE, previousV, integralValue, derivativeValue := 0., 0., 0., 0.
			for k := range 30 {
				r, y := math.Sin(float64(k)*.4), math.Cos(float64(k)*.23)
				e, v := r-y, p.C*r-y
				integralValue += p.Ki * p.Dt * (qi*e + (1-qi)*previousE)
				derivativeValue = ((p.Tf-(1-qd)*p.Dt)*derivativeValue + p.Kd*(v-previousV)) / (p.Tf + qd*p.Dt)
				want := p.Kp*(p.B*r-y) + integralValue + derivativeValue
				got := sys.D.At(0, 0)*r + sys.D.At(0, 1)*y
				for j, x := range state {
					got += sys.C.At(0, j) * x
				}
				if math.Abs(got-want) > 1e-10 {
					t.Fatalf("formulas %v,%v step %d: got %g want %g", fi, fd, k, got, want)
				}
				next := make([]float64, len(state))
				for i := range next {
					next[i] = sys.B.At(i, 0)*r + sys.B.At(i, 1)*y
					for j, x := range state {
						next[i] += sys.A.At(i, j) * x
					}
				}
				state = next
				previousE, previousV = e, v
			}
		}
	}
}

func TestPIDIdealDerivativeAndValidation(t *testing.T) {
	for _, f := range []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal} {
		p := mustPID(t, 1, 0, 2, 0, .1, WithPIDFormulas(f, f))
		sys, err := p.System()
		if f == ForwardEuler {
			if err == nil {
				t.Fatal("ideal forward derivative must reject noncausal model")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		h, err := sys.FreqResponse([]float64{2})
		if err != nil {
			t.Fatal(err)
		}
		z := cmplx.Exp(.2i)
		q := 1.
		if f == Trapezoidal {
			q = .5
		}
		want := 1 + 2*(z-1)/(.1*(complex(q, 0)*z+complex(1-q, 0)))
		if cmplx.Abs(h.At(0, 0, 0)-want) > 1e-10 {
			t.Fatalf("ideal derivative %v != %v", h.At(0, 0, 0), want)
		}
	}
	if _, err := mustPID(t, 1, 0, 1, 0, 0).System(); err == nil {
		t.Fatal("continuous ideal derivative must be explicit improper error")
	}
	for _, p := range []*PID{{Kp: math.NaN()}, {Kp: 1, Dt: -1}, {Kp: 1, Tf: -1}, {Kp: 1, IFormula: 7}} {
		if _, err := p.System(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("invalid parameters accepted: %+v: %v", p, err)
		}
	}
	for _, args := range [][5]float64{{math.NaN(), 0, 0, 0, 0}, {1, math.Inf(1), 0, 0, 0}, {1, 0, 0, 0, -1}, {1, 0, 0, -1, 0}, {1, 0, 0, math.NaN(), 0}} {
		if _, err := NewPID(args[0], args[1], args[2], args[3], args[4]); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("NewPID%v err = %v, want ErrInvalidArgument", args, err)
		}
	}
	if _, err := NewPID(1, 0, 0, 0, 0, WithPIDFormulas(7, 0)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("unknown formula err = %v", err)
	}
}

func TestPIDStandardZeroTermsPreserveFormulas(t *testing.T) {
	for _, g := range [][3]float64{{0, 0, 0}, {2, 0, 0}, {2, 0, 3}, {2, 4, 0}} {
		p := mustPID(t, g[0], g[1], g[2], .2, .1, WithPIDFormulas(BackwardEuler, Trapezoidal))
		standard, err := p.Standard()
		if err != nil {
			t.Fatal(err)
		}
		ti, okI := standard.Ti()
		td, okD := standard.Td()
		if !okI || !okD {
			t.Fatalf("%v: standard form undefined", g)
		}
		N := math.Inf(1)
		if td != 0 {
			N = td / standard.Tf
		}
		roundtrip, err := NewPIDStd(standard.Kp, ti, td, N, standard.Dt, WithPIDFormulas(standard.IFormula, standard.DFormula))
		if err != nil {
			t.Fatal(err)
		}
		want := *p
		if want.Kd == 0 {
			want.Tf = 0
		}
		if want != *roundtrip.Parallel() {
			t.Fatalf("lost form conversion: %+v -> %+v", want, roundtrip)
		}
	}
	for _, p := range []*PID{mustPID(t, 0, 1, 0, 0, 0), mustPID(t, 0, 0, 1, 0, 0)} {
		if _, err := p.Standard(); err == nil {
			t.Fatal("singular conversion accepted")
		}
	}
}

func TestPidtunePDFIndependentTarget(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	p, err := Pidtune(plant, PidtunePDF, 1, &PidtuneOptions{PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if p.Ki != 0 || p.Tf <= 0 {
		t.Fatalf("PDF gains %+v", p)
	}
	jw := complex(0, 1)
	loop := (complex(p.Kp, 0) + complex(p.Kd, 0)*jw/(1+complex(p.Tf, 0)*jw)) / ((1 + jw) * (1 + jw) * (1 + jw))
	if math.Abs(cmplx.Abs(loop)-1) > 1e-12 || math.Abs(cmplx.Phase(loop)*180/math.Pi+120) > 1e-10 {
		t.Fatalf("PDF target not met: %v", loop)
	}
}
