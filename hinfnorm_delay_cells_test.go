package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// cellD22 has sup_θ ρ(D22·diag(e^{jθ})) ≈ 0.97, near θ = cellPeak, yet
// defeats diagonalScalingBound, the D-scaled μ upper bound, which is not
// exact for four complex scalar blocks.
var (
	cellD22 = []float64{
		-0.3192, -0.0487, 0.2922, -0.1732,
		-0.2976, 0.1894, -0.7305, 0.3788,
		-0.2002, 0.3138, -0.3409, -0.1732,
		-0.6764, -0.6006, 0.487, 0.5627,
	}
	cellPeak = []float64{0, 3.4629568515112874, 2.8170313158523106, 4.804676546374199}
)

// phasedD22 returns the complex D22·diag(e^{jθ}) and I − D22·diag(e^{jθ}).
func phasedD22(d22 []float64, theta []float64) (m, b *mat.CDense) {
	n := len(theta)
	m, b = mat.NewCDense(n, n, nil), mat.NewCDense(n, n, nil)
	for i := range n {
		for j := range n {
			v := complex(d22[i*n+j], 0) * cmplx.Exp(complex(0, theta[j]))
			m.Set(i, j, v)
			b.Set(i, j, -v)
		}
		b.Set(i, i, 1+b.At(i, i))
	}
	return m, b
}

// spectralRadiusC returns ρ(m) from the eigenvalues of its real embedding
// [[Re, −Im], [Im, Re]], whose spectrum is that of m and its conjugate.
func spectralRadiusC(t *testing.T, m *mat.CDense) float64 {
	n, _ := m.Dims()
	e := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			v := m.At(i, j)
			e.Set(i, j, real(v))
			e.Set(i+n, j+n, real(v))
			e.Set(i, j+n, -imag(v))
			e.Set(i+n, j, imag(v))
		}
	}
	var eig mat.Eigen
	if !eig.Factorize(e, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	r := 0.0
	for _, v := range eig.Values(nil) {
		r = max(r, cmplx.Abs(v))
	}
	return r
}

func TestTorusCellBound(t *testing.T) {
	d22 := mat.NewDense(4, 4, cellD22)
	if p := diagonalScalingBound(d22, 0, false); !math.IsInf(p, 1) {
		t.Fatalf("diagonal scaling certifies %g; the case must defeat it", p)
	}
	m, _ := phasedD22(cellD22, cellPeak)
	if r := spectralRadiusC(t, m); !(r > 0.969 && r < 0.971) {
		t.Fatalf("ρ at cellPeak = %g, want ≈ 0.97", r)
	}
	bound := torusCellBound(d22)
	if math.IsInf(bound, 1) {
		t.Fatal("strong stability not certified")
	}
	rng := rand.New(rand.NewPCG(3, 4))
	theta := make([]float64, 4)
	for trial := range 2000 {
		for k := range theta {
			theta[k] = rng.Float64() * 2 * math.Pi
			if trial == 0 {
				theta[k] = cellPeak[k]
			}
		}
		z := complex(rng.Float64(), 0)
		if trial%2 == 0 {
			z = 1
		}
		_, b := phasedD22(cellD22, theta)
		if z != 1 {
			for i := range 4 {
				for j := range 4 {
					v := b.At(i, j)
					if i == j {
						v--
					}
					b.Set(i, j, z*v)
				}
				b.Set(i, i, 1+b.At(i, i))
			}
		}
		for col := range 4 {
			e := make([]complex128, 4)
			e[col] = 1
			x := complexSolve(b.RawCMatrix().Data, e, 4)
			var nrm float64
			for _, v := range x {
				nrm += real(v)*real(v) + imag(v)*imag(v)
			}
			if math.Sqrt(nrm) > bound {
				t.Fatalf("‖(I − zD22Θ)⁻¹e%d‖ = %g at θ = %v, z = %g exceeds bound %g", col, math.Sqrt(nrm), theta, z, bound)
			}
		}
	}

	var over mat.Dense
	over.Scale(1.01/0.97, d22)
	m, _ = phasedD22(over.RawMatrix().Data, cellPeak)
	if r := spectralRadiusC(t, m); !(r > 1) {
		t.Fatalf("scaled ρ at cellPeak = %g, want > 1", r)
	}
	if c := torusCellBound(&over); !math.IsInf(c, 1) {
		t.Fatalf("certified %g with ρ(D22Θ) > 1", c)
	}
}

// TestNeutralFourDelayCellCertificate checks a four-delay neutral loop
// G = D11 + C1(sI−A)⁻¹(B1 + B2·Δ(I − D22Δ)⁻¹·D21), C2 = 0, D12 = 0, with
// incommensurate delays, strongly stable only through torusCellBound.
func TestNeutralFourDelayCellCertificate(t *testing.T) {
	skipSlowUnderRace(t)
	a := []float64{-1, 2, -0.7, -3}
	b1 := []float64{1, 0.5}
	b2 := []float64{0.3, 0, 0.1, 0, 0, 0.2, 0, 0.1}
	c1 := []float64{1, -0.4}
	d11 := 0.1
	d21 := []float64{1, 0.5, -0.3, 0.2}
	tau := []float64{0.4, 0.4 * math.Sqrt2, 0.4 * math.Sqrt(3), 0.4 * math.Sqrt(5)}
	bm := make([]float64, 0, 10)
	for i := range 2 {
		bm = append(bm, b1[i])
		bm = append(bm, b2[i*4:i*4+4]...)
	}
	cm := make([]float64, 10)
	copy(cm, c1)
	dm := make([]float64, 25)
	dm[0] = d11
	for i := range 4 {
		dm[(i+1)*5] = d21[i]
		copy(dm[(i+1)*5+1:(i+1)*5+5], cellD22[i*4:i*4+4])
	}
	H, err := NewFromSlices(2, 5, 5, a, bm, cm, dm, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	if e.multiples() != nil {
		t.Fatal("delays must be incommensurate")
	}
	if l := e.tail(math.Inf(1)); l < 1 {
		t.Fatalf("tail(∞) = %g; the case must defeat the norm and permanent bounds", l)
	}
	if d := e.difference(); math.IsInf(d.polydisk, 1) {
		t.Fatal("strong stability not certified")
	}
	if got, err := sys.IsStable(); err != nil || !got {
		t.Fatalf("IsStable = %v, %v; want true", got, err)
	}
	gain := func(w float64) float64 {
		s := complex(0, w)
		theta := make([]float64, 4)
		for k := range theta {
			theta[k] = -w * tau[k]
		}
		_, b := phasedD22(cellD22, theta)
		r := make([]complex128, 4)
		for i := range r {
			r[i] = complex(d21[i], 0)
		}
		v := complexSolve(b.RawCMatrix().Data, r, 4)
		u := [2]complex128{complex(b1[0], 0), complex(b1[1], 0)}
		for k := range 4 {
			v[k] *= cmplx.Exp(complex(0, theta[k]))
			u[0] += complex(b2[k], 0) * v[k]
			u[1] += complex(b2[4+k], 0) * v[k]
		}
		m := [4]complex128{s - complex(a[0], 0), -complex(a[1], 0), -complex(a[2], 0), s - complex(a[3], 0)}
		det := m[0]*m[3] - m[1]*m[2]
		x0 := (m[3]*u[0] - m[1]*u[1]) / det
		x1 := (m[0]*u[1] - m[2]*u[0]) / det
		return cmplx.Abs(complex(d11, 0) + complex(c1[0], 0)*x0 + complex(c1[1], 0)*x1)
	}
	n, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	want, _ := oraclePeak(gain, 200, 1e-3)
	if math.Abs(n-want) > 1e-8*want {
		t.Fatalf("HinfNorm = %.12g, want %.12g", n, want)
	}
}
