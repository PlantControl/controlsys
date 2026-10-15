package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type neutralBlock struct{ a, b, c float64 }

// neutralPair returns two neutral equations ẋi = −ai·xi + ui + wi with
// wi(t) = zi(t−τi), zi = ci·ẋi − bi·xi + k·w2·[i=1], so that with k = 0
// χi(s) = s + ai + (bi − ci·s)·e^{−sτi} and Xi/Ui = (1 − ci·ei)/χi, and with
// k ≠ 0 H22 is upper triangular and χ = χ1·χ2. For |ci| < 1, χi is stable for
// every τ when ai > |bi| (|jω + a| > |b − c·jω| on the axis, root
// −(a+b)/(1−c) at τ = 0), and has a real root s > 0 when bi < −ai
// (χi(0) < 0 < χi(+∞)).
func neutralPair(t *testing.T, p [2]neutralBlock, k float64, tau []float64) *System {
	t.Helper()
	A := mat.NewDense(2, 2, []float64{-p[0].a, 0, 0, -p[1].a})
	B := mat.NewDense(2, 4, []float64{1, 0, 1, 0, 0, 1, 0, 1})
	C := mat.NewDense(4, 2, []float64{1, 0, 0, 1, -(p[0].c*p[0].a + p[0].b), 0, 0, -(p[1].c*p[1].a + p[1].b)})
	D := mat.NewDense(4, 4, nil)
	D.Set(2, 2, p[0].c)
	D.Set(2, 3, k)
	D.Set(3, 3, p[1].c)
	H, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func neutralBlockGain(p neutralBlock, tau, w float64) float64 {
	s := complex(0, w)
	e := cmplx.Exp(-s * complex(tau, 0))
	return cmplx.Abs((1 - complex(p.c, 0)*e) / (s + complex(p.a, 0) + (complex(p.b, 0)-complex(p.c, 0)*s)*e))
}

func TestNeutralStronglyStableUnequalDelays(t *testing.T) {
	tau := []float64{0.5, 0.5 * math.Sqrt2}
	stable := [2]neutralBlock{{2, 1, 0.6}, {1.5, -1, -0.7}}
	unstable := [2]neutralBlock{{2, 1, 0.6}, {1.5, -2, -0.7}}
	for _, k := range []float64{0, 3} {
		t.Run(fmt.Sprintf("k=%g", k), func(t *testing.T) {
			sys := neutralPair(t, stable, k, tau)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if l := e.tail(math.Inf(1)); l < 1 {
				t.Fatalf("tail(∞) = %g; the case must defeat the norm and permanent bounds", l)
			}
			if got, err := sys.IsStable(); err != nil || !got {
				t.Fatalf("IsStable = %v, %v; want true", got, err)
			}
			n, _, err := HinfNorm(sys)
			if err != nil || math.IsInf(n, 1) {
				t.Fatalf("HinfNorm = %g, %v; want finite", n, err)
			}
			if k == 0 {
				var want float64
				for i := range 2 {
					p, _ := oraclePeak(func(w float64) float64 { return neutralBlockGain(stable[i], tau[i], w) }, 60, 1e-3)
					want = max(want, p)
				}
				if math.Abs(n-want) > 1e-8*want {
					t.Fatalf("HinfNorm = %.12g, want %.12g", n, want)
				}
			}
			bad := neutralPair(t, unstable, k, tau)
			if got, err := bad.IsStable(); err != nil || got {
				t.Fatalf("unstable: IsStable = %v, %v; want false", got, err)
			}
			if n, _, err := HinfNorm(bad); err != nil || !math.IsInf(n, 1) {
				t.Fatalf("unstable: HinfNorm = %g, %v; want +Inf", n, err)
			}
		})
	}
}

// constantDifferenceLoop returns G = D11 + C1(sI−A)⁻¹(B1 + B2·Δ(I − D22Δ)⁻¹·D21)
// with C2 = 0 and D12 = 0, so H22 = D22 and χ = det(sI−A)·det(I − D22Δ):
// stable iff A is Hurwitz and the difference operator has no root in
// Re s >= 0. gain evaluates |G(jω)| in closed form.
func constantDifferenceLoop(t *testing.T, d22 []float64, tau []float64) (*System, func(float64) float64) {
	t.Helper()
	a := []float64{-1, 2, -0.7, -3}
	b1 := []float64{1, 0.5}
	b2 := []float64{0.3, 0, 0, 1}
	c1 := []float64{1, -0.4}
	d11 := 0.1
	d21 := []float64{1, 0.5}
	H, err := NewFromSlices(2, 3, 3,
		a,
		[]float64{b1[0], b2[0], b2[1], b1[1], b2[2], b2[3]},
		[]float64{c1[0], c1[1], 0, 0, 0, 0},
		[]float64{d11, 0, 0, d21[0], d22[0], d22[1], d21[1], d22[2], d22[3]}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	gain := func(w float64) float64 {
		s := complex(0, w)
		e1, e2 := cmplx.Exp(-s*complex(tau[0], 0)), cmplx.Exp(-s*complex(tau[1], 0))
		m := [4]complex128{1 - complex(d22[0], 0)*e1, -complex(d22[1], 0) * e2, -complex(d22[2], 0) * e1, 1 - complex(d22[3], 0)*e2}
		det := m[0]*m[3] - m[1]*m[2]
		v0 := (m[3]*complex(d21[0], 0) - m[1]*complex(d21[1], 0)) / det
		v1 := (m[0]*complex(d21[1], 0) - m[2]*complex(d21[0], 0)) / det
		v0, v1 = e1*v0, e2*v1
		u0 := complex(b1[0], 0) + complex(b2[0], 0)*v0 + complex(b2[1], 0)*v1
		u1 := complex(b1[1], 0) + complex(b2[2], 0)*v0 + complex(b2[3], 0)*v1
		r := [4]complex128{s - complex(a[0], 0), -complex(a[1], 0), -complex(a[2], 0), s - complex(a[3], 0)}
		rd := r[0]*r[3] - r[1]*r[2]
		x0 := (r[3]*u0 - r[1]*u1) / rd
		x1 := (r[0]*u1 - r[2]*u0) / rd
		return cmplx.Abs(complex(d11, 0) + complex(c1[0], 0)*x0 + complex(c1[1], 0)*x1)
	}
	return sys, gain
}

func TestNeutralDifferenceOperatorCertificates(t *testing.T) {
	cases := []struct {
		name string
		d22  []float64
		tau  []float64
		// want: 1 stable, -1 unstable
		want int
		cert string
	}{
		// S = diag(1, 0.1) gives S⁻¹D22S = [[0.6, 0.5], [−0.5, 0.6]] of norm
		// 0.781, so sup_θ ρ(D22·diag(e^{jθ})) < 1, while ‖D22‖ ≈ 5 and
		// ρ(|D22|) = 1.1.
		{"diagonal-scaling", []float64{0.6, 5, -0.05, 0.6}, []float64{0.4, 0.9}, 1, "polydisk"},
		// det(I − D22) = 0.16 − 0.25 < 0 < det(I − D22·e^{−στ}) as σ → ∞.
		{"real-root", []float64{0.6, 5, 0.05, 0.6}, []float64{0.4, 0.9}, -1, ""},
		{"real-root-coupled", []float64{0.5, 1, 1, 0.5}, []float64{1, math.Sqrt2}, -1, ""},
		// Equal delays: det(I − z·D22) = 1 + 0.19z² has |z| = 1/√0.19 > 1,
		// yet Θ = diag(1, −1) gives ρ(D22Θ) = 1.9: stable, not strongly.
		{"equal-stein", []float64{0.9, 1, -1, -0.9}, []float64{0.7, 0.7}, 1, "rhp"},
		// (1 + 1.5·e^{−0.6s})(1 − 0.5·e^{−0.6√3s}) vanishes at
		// s = (ln 1.5 + jπ(2m+1))/0.6 and has no real root.
		{"complex-root", []float64{-1.5, 0, 0, 0.5}, []float64{0.6, 0.6 * math.Sqrt(3)}, -1, ""},
		{"complex-root-coupled", []float64{-1.5, 0.2, 0.1, 0.5}, []float64{0.6, 0.6 * math.Sqrt(3)}, -1, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, gain := constantDifferenceLoop(t, tc.d22, tc.tau)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if l := e.tail(math.Inf(1)); l < 1 {
				t.Fatalf("tail(∞) = %g; the case must defeat the norm and permanent bounds", l)
			}
			d := e.difference()
			switch tc.cert {
			case "polydisk":
				if math.IsInf(d.polydisk, 1) {
					t.Fatal("strong stability not certified")
				}
			case "rhp":
				if !math.IsInf(d.polydisk, 1) || math.IsInf(d.rhp, 1) {
					t.Fatalf("bounds %+v, want rhp only", *d)
				}
			}
			got, err := sys.IsStable()
			if err != nil || got != (tc.want == 1) {
				t.Fatalf("IsStable = %v, %v; want %v", got, err, tc.want == 1)
			}
			n, _, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if tc.want == -1 {
				if !math.IsInf(n, 1) {
					t.Fatalf("HinfNorm = %g, want +Inf", n)
				}
				return
			}
			want, _ := oraclePeak(gain, 80, 1e-3)
			if math.Abs(n-want) > 1e-8*want {
				t.Fatalf("HinfNorm = %.12g, want %.12g", n, want)
			}
		})
	}
}

func TestNeutralDifferenceOperatorBoundary(t *testing.T) {
	// det(I − D22Δ) = (1 − e^{−0.5s})(1 − 0.5e^{−0.8s}) vanishes on the axis:
	// sup_θ ρ = 1 is left undecided.
	sys, _ := constantDifferenceLoop(t, []float64{1, 0, 0, 0.5}, []float64{0.5, 0.8})
	if got, err := sys.IsStable(); err == nil && got || err != nil && !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("IsStable = %v, %v; want false or ErrDelayUnsupported", got, err)
	}
}

func TestNeutralLinfAntiStableDifferenceOperator(t *testing.T) {
	// D22 = diag(1.5, −2): |1 − 1.5z| >= 0.5 and |1 + 2z| >= 1 on |z| = 1, so
	// the gain is bounded on the axis although the model is unstable.
	sys, gain := constantDifferenceLoop(t, []float64{1.5, 0, 0, -2}, []float64{0.4, 0.9})
	if got, err := sys.IsStable(); err != nil || got {
		t.Fatalf("IsStable = %v, %v; want false", got, err)
	}
	if n, _, err := HinfNorm(sys); err != nil || !math.IsInf(n, 1) {
		t.Fatalf("HinfNorm = %g, %v; want +Inf", n, err)
	}
	want, _ := oraclePeak(gain, 80, 1e-3)
	got, err := Norm(sys, math.Inf(1))
	if err != nil || math.Abs(got-want) > 1e-8*want {
		t.Fatalf("Norm(Inf) = %.12g, %v; want %.12g", got, err, want)
	}
}
