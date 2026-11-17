package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zpkDescriptorOracleRatio returns the largest |Z_ij − G_ij|/(√ε·max(scale,
// T_ij)) over the entries of the descriptor sys at s, G and its terms T =
// Σ_l |C_il||U_lj| + |D_ij|, U = (sE − A)⁻¹B, from the 320-bit pencil
// oracle and scale the row's largest |G|, |Z|.
func zpkDescriptorOracleRatio(t *testing.T, sys *System, z *ZPK, s complex128) float64 {
	t.Helper()
	o := newStepOraclePencil(sys)
	xr, xi := o.solveShifted(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))}, o.b)
	if xr == nil {
		t.Fatalf("oracle: sE − A singular at %v", s)
	}
	zv, err := z.Eval(s)
	if err != nil {
		t.Fatal(err)
	}
	tol := math.Sqrt(eps())
	worst := 0.0
	for i := range o.p {
		g, terms := make([]complex128, o.m), make([]float64, o.m)
		scale := 0.0
		for j := range o.m {
			r, q, tt := stepOracleNew(), stepOracleNew(), stepOracleNew()
			for k := range o.n {
				r.Add(r, stepOracleNew().Mul(o.c[i][k], xr[k][j]))
				q.Add(q, stepOracleNew().Mul(o.c[i][k], xi[k][j]))
				ax := stepOracleNew().Sqrt(stepOracleNew().Add(stepOracleNew().Mul(xr[k][j], xr[k][j]), stepOracleNew().Mul(xi[k][j], xi[k][j])))
				tt.Add(tt, stepOracleNew().Mul(stepOracleNew().Abs(o.c[i][k]), ax))
			}
			rf, _ := r.Float64()
			qf, _ := q.Float64()
			tf, _ := tt.Float64()
			d := sys.D.At(i, j)
			g[j], terms[j] = complex(rf+d, qf), tf+math.Abs(d)
			scale = math.Max(scale, math.Max(cmplx.Abs(g[j]), cmplx.Abs(zv[i][j])))
		}
		for j := range o.m {
			worst = math.Max(worst, cmplx.Abs(zv[i][j]-g[j])/(tol*math.Max(scale, terms[j])))
		}
	}
	return worst
}

func zpkDescriptorCase(t *testing.T, n, m, p int, a, e, b, c, d []float64, dt float64) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), mat.NewDense(n, n, e), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// zpkDescriptorCases are descriptor models with non-symmetric A and
// non-identity E: E nonsingular, E singular with an algebraic state
// (index 1), E singular with a nilpotent chain that adds an s-term
// (improper), and a discrete two-by-two model.
func zpkDescriptorCases(t *testing.T) map[string]*System {
	return map[string]*System{
		"nonsingular E": zpkDescriptorCase(t, 3, 1, 1,
			[]float64{-1.5, 0.7, 0.2, -0.4, -2.1, 1.3, 0.6, -0.9, -0.8},
			[]float64{1.2, 0.3, 0, -0.2, 0.9, 0.4, 0.1, 0, 1.5},
			[]float64{1, -0.5, 0.8}, []float64{0.6, 1.1, -0.7}, []float64{0.25}, 0),
		"singular E index 1": zpkDescriptorCase(t, 3, 1, 1,
			[]float64{-1.1, 0.4, 0.3, 0.2, -0.7, -0.6, 0.5, 0.9, 1.4},
			[]float64{1, 0.3, 0, 0, 2, 0, 0, 0, 0},
			[]float64{0.8, -1.2, 0.5}, []float64{1, 0.4, -0.9}, []float64{0}, 0),
		"singular E improper chain": zpkDescriptorCase(t, 3, 1, 1,
			[]float64{-2, 1, 0, 0, 1, 0, 0.3, 0, 1},
			[]float64{1, 0, 0, 0, 0, 1, 0, 0, 0},
			[]float64{1, 0.5, 1}, []float64{1, 1, 0.7}, []float64{0.1}, 0),
		"discrete two by two": zpkDescriptorCase(t, 3, 2, 2,
			[]float64{0.5, 0.2, -0.1, -0.3, 0.4, 0.25, 0.1, -0.2, -0.6},
			[]float64{1.1, 0.2, 0, 0, 0.8, -0.1, 0.3, 0, 1.3},
			[]float64{1, 0, 0.5, -1, -0.4, 0.7}, []float64{0.9, -0.3, 0.2, 0.1, 0.6, -1}, []float64{0, 0.1, 0.2, 0}, 0.1),
	}
}

// TestZPKModelDescriptorVerified checks that ZPKModel certifies the
// zero-pole-gain form of descriptor models against the pencil sE − A
// (descriptorZPKVerified), and that the form is within terms of the stored
// model's 320-bit response at axis points.
func TestZPKModelDescriptorVerified(t *testing.T) {
	for name, sys := range zpkDescriptorCases(t) {
		t.Run(name, func(t *testing.T) {
			res, err := sys.ZPKModel(nil)
			if err != nil {
				t.Fatal(err)
			}
			pts := []complex128{0, 0.01i, 0.3i, 1i, 3i, 30i, 1e3i}
			if sys.Dt != 0 {
				pts = []complex128{1, cmplx.Rect(1, 0.01), cmplx.Rect(1, 0.3), cmplx.Rect(1, 1), cmplx.Rect(1, 2.5), -1}
			}
			for _, s := range pts {
				if r := zpkDescriptorOracleRatio(t, sys, res.ZPK, s); !(r <= 1) {
					t.Errorf("s = %v: |Z − G| = %.3g·√ε·max(scale, T)", s, r)
				}
			}
		})
	}
}

// TestZPKModelDescriptorRefutesPerturbedForm checks that the descriptor
// certificate rejects a zero-pole-gain form whose gain or one zero is moved
// by 10⁻⁶ relative, far beyond √ε of the response.
func TestZPKModelDescriptorRefutesPerturbedForm(t *testing.T) {
	for name, sys := range zpkDescriptorCases(t) {
		t.Run(name, func(t *testing.T) {
			tfr, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			z, err := tfr.TF.ZPK()
			if err != nil {
				t.Fatal(err)
			}
			if !descriptorZPKVerified(sys, tfr.TF, z) {
				t.Fatal("unperturbed form not verified")
			}
			gain := z.Copy()
			gain.Gain[0][0] *= 1 + 1e-6
			if descriptorZPKVerified(sys, tfr.TF, gain) {
				t.Error("gain moved by 1e-6 verified")
			}
			if len(z.Zeros[0][0]) > 0 {
				zero := z.Copy()
				zero.Zeros[0][0][0] *= 1 + 1e-6
				if descriptorZPKVerified(sys, tfr.TF, zero) {
					t.Error("zero moved by 1e-6 verified")
				}
			}
		})
	}
}

// TestZPKModelDescriptorRootLossUnverified checks a discrete descriptor with
// nearly singular E (census row nse-dt/614, R4JG5M) whose transfer function
// verifies but whose roots, with a pole and a zero near −1.4e15, lose the
// response: at z = −0.99 + 0.1411j the returned form is 1.17·√ε of
// max(scale, terms) from the 320-bit response, beyond the data bound too,
// so ZPKModel must return it with ErrAccuracyUnverified, not nil.
func TestZPKModelDescriptorRootLossUnverified(t *testing.T) {
	sys := zpkDescriptorCase(t, 5, 1, 1,
		[]float64{-1.867235453097552, 0.26869448282682806, 0.34767806491133224, -0.6513009728904159, -0.20964928641572542, -0.5035160174784982, 1.3198309872649163, -0.028499691681299323, -0.1724267794991691, -1.8923731079617157, -0.236200648115944, 0.042725880803948305, 0.9729534362192397, 0.16988616385952138, -0.45419257954694414, -0.181769934854848, -0.4523947296568792, -0.3250213398014731, -0.19913805355784317, 0.3574191656123325, 0.11192359587163864, 1.3610212058640727, -0.30122849520914113, -0.24290493274262595, -0.8155070495478653},
		[]float64{0.08699747792698119, 0.043271393245455536, -0.5802446047239337, -0.543382420786657, 0.24808802144538963, 0.6357638317296826, -0.15618412614041136, 0.46285136669220145, -0.024452479410982074, 0.2693981607690124, -0.032762095047655516, -0.17356120224676, 0.334974760560241, -0.3593748397245398, -0.6195865742138292, 0.13753153347586658, 0.13101116437898458, 0.08647662260387971, 0.2719771560906276, 0.3968140621898865, 0.2690223681493535, -0.6289740274220219, -0.32043743561201643, 0.2801286948487717, -0.1214775501471306},
		[]float64{-1.2680798063187106, -0.17871289717466862, -0.4141160630497934, 0.37557669743544675, 0.6817532846301649},
		[]float64{-0.9417381975503051, 0.43987846977260897, 0.36814570045960376, 1.582866841586228, 0.8190080972418367},
		[]float64{0.49380971384603356}, 0.1)
	if _, err := sys.TransferFunction(nil); err != nil {
		t.Fatalf("TransferFunction: %v", err)
	}
	res, err := sys.ZPKModel(nil)
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("err = %v, want ErrAccuracyUnverified", err)
	}
	if res == nil || res.ZPK == nil {
		t.Fatal("no zero-pole-gain form returned with ErrAccuracyUnverified")
	}
	if r := zpkDescriptorOracleRatio(t, sys, res.ZPK, complex(-0.99, 0.1411)); !(r > 1) {
		t.Errorf("|Z − G| = %.3g·√ε·max(scale, T) at −0.99 + 0.1411j, want > 1", r)
	}
}
