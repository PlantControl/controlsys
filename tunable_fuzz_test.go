package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// FuzzTunableBlocks (task O5FYRC): tunable blocks built from the spec.
// TunableSS (full) reproduces the realization entry for entry and Sample
// writes each "<name>.X(i,j)" to row i, column j; tridiag/companion forms fix
// exactly the entries outside their structure at zero. TunablePID and
// TunableTF realize, at the current and at sampled values, the MATLAB
// formulas Kp + Ki·IF(z) + Kd/(Tf + DF(z)) and num/den evaluated directly.

func FuzzTunableBlocks(f *testing.F) {
	for _, s := range fzTunSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		fzTunSS(t, s, sys)
		fzTunPID(t, s)
		fzTunTF(t, s)
	})
}

func fzTunSS(t *testing.T, s *fzSpec, sys *System) {
	t.Helper()
	b, err := NewTunableSSFrom("K", sys, TunableSSFull)
	fzContract(t, s, "NewTunableSSFrom", b != nil, err)
	fzErr(t, s, "NewTunableSSFrom", err)
	cur, err := b.CurrentSystem()
	fzErr(t, s, "TunableSS.CurrentSystem", err)
	fzTunSameMatrices(t, s, "CurrentSystem", cur, s.A, s.B, s.C, s.D)
	vals := map[string]float64{}
	shift := func(x float64, i, j, k int) float64 { return 1.5*x + float64(100*k+10*i+j) }
	want := [4][]float64{make([]float64, len(s.A)), make([]float64, len(s.B)), make([]float64, len(s.C)), make([]float64, len(s.D))}
	dims := [4][2]int{{s.N, s.N}, {s.N, s.M}, {s.P, s.N}, {s.P, s.M}}
	src := [4][]float64{s.A, s.B, s.C, s.D}
	for k, nm := range []string{"A", "B", "C", "D"} {
		r, c := dims[k][0], dims[k][1]
		for i := range r {
			for j := range c {
				v := shift(src[k][i*c+j], i, j, k)
				vals[fmt.Sprintf("K.%s(%d,%d)", nm, i+1, j+1)] = v
				want[k][i*c+j] = v
			}
		}
	}
	if got := len(b.FreeParameters()); got != len(vals) {
		t.Fatalf("TunableSS full: %d free parameters, want %d\n%s", got, len(vals), s)
	}
	smp, err := b.Sample(vals)
	fzErr(t, s, "TunableSS.Sample", err)
	scur, err := smp.CurrentSystem()
	fzErr(t, s, "TunableSS.Sample.CurrentSystem", err)
	fzTunSameMatrices(t, s, "Sample", scur, want[0], want[1], want[2], want[3])
	if cur, err = b.CurrentSystem(); err != nil {
		t.Fatalf("CurrentSystem after Sample: %v\n%s", err, s)
	}
	fzTunSameMatrices(t, s, "CurrentSystem after Sample", cur, s.A, s.B, s.C, s.D)

	for _, form := range []TunableSSStructure{TunableSSTridiag, TunableSSCompanion} {
		b, err := NewTunableSSFrom("K", sys, form)
		fzContract(t, s, "NewTunableSSFrom("+string(form)+")", b != nil, err)
		if errors.Is(err, ErrSingularTransform) {
			continue
		}
		fzErr(t, s, "NewTunableSSFrom("+string(form)+")", err)
		free, _ := tunableSSFreeA("", form, s.N)
		for i := range s.N {
			for j := range s.N {
				p := b.A[i][j]
				if p.Fixed() == free(i, j) || (!free(i, j) && p.Value() != 0) {
					t.Fatalf("%s: A(%d,%d) fixed=%v value=%g, structure free=%v\n%s", form, i+1, j+1, p.Fixed(), p.Value(), free(i, j), s)
				}
			}
		}
	}
}

func fzTunSameMatrices(t *testing.T, s *fzSpec, op string, sys *System, a, b, c, d []float64) {
	t.Helper()
	n, m, p := sys.Dims()
	if n != s.N || m != s.M || p != s.P {
		t.Fatalf("%s: dims %d,%d,%d, want %d,%d,%d\n%s", op, n, m, p, s.N, s.M, s.P, s)
	}
	for _, x := range []struct {
		name string
		got  func(i, j int) float64
		want []float64
		c    int
	}{{"A", sys.A.At, a, n}, {"B", sys.B.At, b, m}, {"C", sys.C.At, c, n}, {"D", sys.D.At, d, m}} {
		for idx, w := range x.want {
			i, j := idx/x.c, idx%x.c
			if g := x.got(i, j); g != w {
				t.Fatalf("%s: %s(%d,%d) = %v, want %v\n%s", op, x.name, i+1, j+1, g, w, s)
			}
		}
	}
}

// fzTunPIDResp is the MATLAB parallel PID formula at x (s, or z with Ts).
func fzTunPIDResp(kp, ki, kd, tf, dt float64, fi, fd PIDFormula, x complex128) complex128 {
	integ := func(f PIDFormula) complex128 {
		if dt == 0 {
			return 1 / x
		}
		ts := complex(dt, 0)
		switch f {
		case BackwardEuler:
			return ts * x / (x - 1)
		case Trapezoidal:
			return ts / 2 * (x + 1) / (x - 1)
		default:
			return ts / (x - 1)
		}
	}
	h := complex(kp, 0)
	if ki != 0 {
		h += complex(ki, 0) * integ(fi)
	}
	if kd != 0 {
		if dt == 0 {
			h += complex(kd, 0) * x / (complex(tf, 0)*x + 1)
		} else {
			h += complex(kd, 0) / (complex(tf, 0) + integ(fd))
		}
	}
	return h
}

func fzTunPID(t *testing.T, s *fzSpec) {
	t.Helper()
	kp := s.D[0]
	ki := 0.0
	if s.Vals[12] > -5 {
		ki = math.Pow(10, fzClamp(s.Vals[12], -3, 3)) * math.Copysign(1, s.Vals[6])
	}
	kd := 0.0
	if s.Vals[13] > -5 {
		kd = math.Pow(10, fzClamp(s.Vals[13], -3, 3))
	}
	tf := math.Pow(10, fzClamp(s.Vals[14], -3, 1))
	fi, fd := PIDFormula(int(s.Kind>>1)%3), PIDFormula(int(s.Kind>>3)%3)
	pid := &PID{Kp: kp, Ki: ki, Kd: kd, Tf: tf, Dt: s.Dt, IFormula: fi, DFormula: fd}
	b, err := NewTunablePIDFrom("C", pid)
	fzContract(t, s, "NewTunablePIDFrom", b != nil, err)
	fzErr(t, s, "NewTunablePIDFrom", err)
	if b.Ki.Fixed() != (ki == 0) || b.Kd.Fixed() != (kd == 0) || b.Tf.Fixed() != (kd == 0) || b.Kp.Fixed() {
		t.Fatalf("TunablePID free set Kp=%v Ki=%v Kd=%v Tf=%v for ki=%g kd=%g\n%s", !b.Kp.Fixed(), !b.Ki.Fixed(), !b.Kd.Fixed(), !b.Tf.Fixed(), ki, kd, s)
	}
	check := func(op string, blk *TunablePID, kp, ki, kd, tf float64) {
		sys, err := blk.CurrentSystem()
		fzErr(t, s, op, err)
		n, _, _ := sys.Dims()
		for _, w := range []float64{0.37, 1.9, 13} {
			x := complex(0, w)
			if s.Dt > 0 {
				x = cmplx.Exp(complex(0, w*s.Dt))
			}
			want := fzTunPIDResp(kp, ki, kd, tf, s.Dt, fi, fd, x)
			got, ok := fzExactResponse(n, 1, 1, sys.A.RawMatrix().Data, sys.B.RawMatrix().Data, sys.C.RawMatrix().Data, sys.D.RawMatrix().Data, x)
			if !ok {
				fzSkip(t, "tunable: PID evaluated at a pole")
			}
			scale := math.Abs(kp) + math.Abs(ki)*cmplx.Abs(fzTunPIDResp(0, 1, 0, 0, s.Dt, fi, fd, x)) + math.Abs(kd)*cmplx.Abs(fzTunPIDResp(0, 0, 1, tf, s.Dt, fi, fd, x))
			if s.Dt > 0 {
				scale *= 1 + 1/cmplx.Abs(x-1)
			}
			if cmplx.Abs(got[0]-want) > 1e-9*scale {
				t.Fatalf("%s at ω=%g: %v, PID formula %v (Kp=%g Ki=%g Kd=%g Tf=%g IF=%d DF=%d)\n%s", op, w, got[0], want, kp, ki, kd, tf, fi, fd, s)
			}
		}
	}
	check("TunablePID.CurrentSystem", b, kp, ki, kd, tf)
	k2, i2, d2, t2 := 2*kp-1, ki, kd, tf
	vals := map[string]float64{"C.Kp": k2, "C.Ki": 3, "C.Kd": 5, "C.Tf": 2 * tf}
	if ki != 0 {
		i2 = 3
	}
	if kd != 0 {
		d2, t2 = 5, 2*tf
	}
	smp, err := b.Sample(vals)
	fzErr(t, s, "TunablePID.Sample", err)
	check("TunablePID.Sample", smp, k2, i2, d2, t2)
}

func fzTunTF(t *testing.T, s *fzSpec) {
	t.Helper()
	den := []complex128{1}
	for _, e := range s.Eig {
		next := make([]complex128, len(den)+1)
		for i, c := range den {
			next[i] += c
			next[i+1] -= c * e
		}
		den = next
	}
	dr := make([]float64, len(den))
	for i, c := range den {
		if dr[i] = real(c); math.IsInf(dr[i], 0) || math.IsNaN(dr[i]) {
			fzSkip(t, "tunable: denominator overflow")
		}
	}
	nz := s.N
	if s.Vals[7] == 0 && s.N > 0 {
		nz = s.N - 1
	}
	num := make([]float64, nz+1)
	for i := range num {
		num[i] = float64(i+1) / 3
		if i < len(s.B) {
			num[i] = s.B[i]
		}
	}
	if nz == s.N {
		num[0] = s.D[0]
	}
	tfm := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{dr}, Dt: s.Dt}
	b, err := NewTunableTFFrom("G", tfm)
	fzContract(t, s, "NewTunableTFFrom", b != nil, err)
	if len(trimLeadingZeros(num)) > len(dr) {
		return
	}
	fzErr(t, s, "NewTunableTFFrom", err)
	check := func(op string, blk *TunableTF, num, den []float64) {
		sys, err := blk.CurrentSystem()
		fzErr(t, s, op, err)
		n, _, _ := sys.Dims()
		for _, w := range []float64{0.41, 2.3, 17} {
			x := complex(0, w)
			if s.Dt > 0 {
				x = cmplx.Exp(complex(0, w*s.Dt))
			}
			dv := fzBigPolyEval(den, x)
			want := fzBigPolyEval(num, x) / dv
			mag := func(c []float64) float64 {
				a, xa := 0.0, cmplx.Abs(x)
				for _, v := range c {
					a = a*xa + math.Abs(v)
				}
				return a
			}
			if cmplx.Abs(dv) < 1e-6*mag(den) {
				fzSkip(t, "tunable: TF evaluated near a pole")
			}
			got, ok := fzExactResponse(n, 1, 1, sys.A.RawMatrix().Data, sys.B.RawMatrix().Data, sys.C.RawMatrix().Data, sys.D.RawMatrix().Data, x)
			if !ok {
				fzSkip(t, "tunable: TF evaluated at a pole")
			}
			// A biproper realization stores num − D·den, rounded at |D|·|den|.
			lead := 0.0
			if len(num) == len(den) {
				lead = math.Abs(num[0])
			}
			tol := 1e-9 * (mag(num) + (lead+cmplx.Abs(want))*mag(den)) / cmplx.Abs(dv)
			if cmplx.Abs(got[0]-want) > tol {
				t.Fatalf("%s at ω=%g: %v, num/den %v (tol %g) num=%v den=%v\n%s", op, w, got[0], want, tol, num, den, s)
			}
		}
	}
	nt := trimLeadingZeros(num)
	check("TunableTF.CurrentSystem", b, nt, dr)
	vals := map[string]float64{}
	n2, d2 := make([]float64, len(nt)), make([]float64, len(dr))
	d2[0] = 1
	for i, v := range nt {
		n2[i] = v + float64(i+1)/8
		vals[fmt.Sprintf("G.Numerator(%d)", i+1)] = n2[i]
	}
	for i := 1; i < len(dr); i++ {
		d2[i] = dr[i] * 1.25
		vals[fmt.Sprintf("G.Denominator(%d)", i+1)] = d2[i]
	}
	vals["G.Denominator(1)"] = 7
	smp, err := b.Sample(vals)
	fzErr(t, s, "TunableTF.Sample", err)
	check("TunableTF.Sample", smp, n2, d2)
}

func fzTunSeeds() []fzSeed {
	none := fzSeed{In: 0, Out: 0, IO: -1, Tau: fzNoDelay, Shear: -10}
	mk := func(f func(*fzSeed)) fzSeed {
		x := none
		f(&x)
		return x
	}
	return []fzSeed{
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.DScal = 2, [3]float64{0}, [3]float64{2.5}, 6, 1 }),
		mk(func(x *fzSeed) {
			x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.DScal = 3, 2, 3, [3]float64{0, 0.5}, [3]float64{2, 1}, 6, 0.4
		}),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.In = 1, [3]float64{0}, [3]float64{1}, 6, -8 }),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.DScal = true, -1, 2, 2, 2, [3]float64{0.5}, [3]float64{2.2}, 6, 0.3
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.Out = true, -1.5, 3, 3, 2, [3]float64{0, 1}, [3]float64{1, 2.5}, 6, -8
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.M, x.Mag, x.Damp, x.Gain = true, -0.5, 4, 1, [3]float64{0, 0.2}, [3]float64{2.5, 3}, 6
		}),
		mk(func(x *fzSeed) { x.N, x.DScal = 0, 2 }),
	}
}
