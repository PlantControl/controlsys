package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzZPKToTF: a MIMO ZPK built from the intended spectrum (per-channel
// conjugate-closed pole subsets, scaled zeros, optional exact pole-zero
// cancellation and zero gains) keeps its response through Eval,
// FreqResponse, TransferFunction and StateSpace, checked against the
// product g·Π(s−z)/Π(s−p) with coefficient-error bounds from Π(|s|+|r|);
// the TF realization is checked against exact rational evaluation of its
// float64 matrices. Row denominators are monic of the common-pole degree,
// and TransferFunc.ZPK recovers the gains exactly and the roots within a
// Rouché radius from a backward-error bound on the polynomial roots.
// Rows with distinct poles inside the library's pole-merge window are
// skipped (merging breaks conjugate pairs; followup).
func FuzzZPKToTF(f *testing.F) {
	for _, s := range fzConvSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		p, m := s.P, s.M
		ch := fzBuildZPK(s)
		zs, ps, gs := make([][][]complex128, p), make([][][]complex128, p), make([][]float64, p)
		for i := range p {
			zs[i], ps[i], gs[i] = make([][]complex128, m), make([][]complex128, m), make([]float64, m)
			for j := range m {
				c := ch[i][j]
				zs[i][j], ps[i][j], gs[i][j] = c.z, c.p, c.g
			}
		}
		for i := range p {
			if fzNearMerge(fzRowLCM(ch[i], false)) {
				fzSkip(t, "distinct row poles within the ZPK pole-merge tolerance (merge breaks conjugate pairs)")
			}
		}
		z, err := NewZPKMIMO(zs, ps, gs, s.Dt)
		fzContract(t, s, "NewZPKMIMO", z != nil, err)
		fzErr(t, s, "NewZPKMIMO", err)
		tf, err := z.TransferFunction()
		fzContract(t, s, "ZPK.TransferFunction", tf != nil, err)
		fzErr(t, s, "ZPK.TransferFunction", err)
		if tf.Dt != s.Dt || len(tf.Den) != p {
			t.Fatalf("TF dt=%g rows=%d, want %g, %d\n%s", tf.Dt, len(tf.Den), s.Dt, p, s)
		}
		lcm := make([][]complex128, p)
		for i := range p {
			lcmAll, lcmLive := fzRowLCM(ch[i], false), fzRowLCM(ch[i], true)
			lcm[i] = lcmAll
			deg := len(tf.Den[i]) - 1
			if tf.Den[i][0] != 1 {
				t.Fatalf("Den[%d] = %v not monic\n%s", i, tf.Den[i], s)
			}
			if fzConvDistinct(lcmAll) && (deg < len(lcmLive) || deg > len(lcmAll)) {
				t.Fatalf("Den[%d] degree %d, want %d..%d (common poles)\n%s", i, deg, len(lcmLive), len(lcmAll), s)
			}
			if len(tf.Num[i]) != m {
				t.Fatalf("Num[%d] has %d channels, want %d\n%s", i, len(tf.Num[i]), m, s)
			}
			for j := range m {
				if len(tf.Num[i][j]) > len(tf.Den[i]) {
					t.Fatalf("Num[%d][%d] = %v improper over Den %v\n%s", i, j, tf.Num[i][j], tf.Den[i], s)
				}
			}
		}
		ss, err := z.StateSpace()
		fzContract(t, s, "ZPK.StateSpace", ss != nil, err)
		fzErr(t, s, "ZPK.StateSpace", err)
		fzCheckRealization(t, s, "ZPK.StateSpace", tf, ss)

		pts := fzConvPoints(s)
		const eps = 0x1p-52
		for _, x := range pts {
			ev, err := z.Eval(x)
			fzErr(t, s, "ZPK.Eval", err)
			tev, err := tf.Eval(x)
			fzErr(t, s, "TransferFunc.Eval", err)
			for i := range p {
				for j := range m {
					c := ch[i][j]
					h, sens := fzProdEval(c, x)
					if math.IsNaN(sens) {
						continue
					}
					if got := ev[i][j]; cmplx.Abs(got-h) > 8*eps*(sens+1)*cmplx.Abs(h) {
						t.Fatalf("ZPK.Eval(%v)[%d,%d] = %v, product %v\n%s", x, i, j, got, h, s)
					}
					errTF := fzZPKCoefErr(c, lcm[i], x, h)
					num, den := tf.Num[i][j], tf.Den[i]
					hp := fzBigPolyEval(num, x) / fzBigPolyEval(den, x)
					if math.IsInf(errTF, 1) {
						continue
					}
					tol := errTF + 8*eps*(sens+1)*cmplx.Abs(h)
					if cmplx.Abs(hp-h) > tol {
						t.Fatalf("TF(%v)[%d,%d] = %v, product %v (tol %g)\nnum=%v den=%v\n%s", x, i, j, hp, h, tol, num, den, s)
					}
					if he := tol + fzHornerErr(num, den, x); he < cmplx.Abs(h) && cmplx.Abs(tev[i][j]-h) > he {
						t.Fatalf("TransferFunc.Eval(%v)[%d,%d] = %v, product %v\n%s", x, i, j, tev[i][j], h, s)
					}
				}
			}
		}
		if ws := fzConvFreqs(s); len(ws) > 0 {
			fr, err := z.FreqResponse(ws)
			fzContract(t, s, "ZPK.FreqResponse", fr != nil, err)
			fzErr(t, s, "ZPK.FreqResponse", err)
			for k, w := range ws {
				x := fzFreqPoint(s, w)
				for i := range p {
					for j := range m {
						h, sens := fzProdEval(ch[i][j], x)
						if math.IsNaN(sens) {
							continue
						}
						if got := fr.At(k, i, j); cmplx.Abs(got-h) > 16*eps*(sens+1)*cmplx.Abs(h) {
							t.Fatalf("ZPK.FreqResponse(%g)[%d,%d] = %v, product %v\n%s", w, i, j, got, h, s)
						}
					}
				}
			}
		}

		back, err := tf.ZPK()
		fzContract(t, s, "TransferFunc.ZPK", back != nil, err)
		fzErr(t, s, "TransferFunc.ZPK", err)
		for i := range p {
			for j := range m {
				c := ch[i][j]
				if g := back.Gain[i][j]; g != c.g {
					t.Fatalf("round-trip gain[%d,%d] = %v, want %v\n%s", i, j, g, c.g, s)
				}
				if c.g == 0 {
					if len(back.Zeros[i][j])+len(back.Poles[i][j]) != 0 {
						t.Fatalf("zero channel (%d,%d) has zeros %v poles %v\n%s", i, j, back.Zeros[i][j], back.Poles[i][j], s)
					}
					continue
				}
				if !fzConvDistinct(lcm[i]) {
					continue
				}
				if msg := fzRouche(back.Poles[i][j], lcm[i], tf.Den[i], nil); msg != "" {
					t.Fatalf("round-trip poles (%d,%d): %s\ngot %v want %v\n%s", i, j, msg, back.Poles[i][j], lcm[i], s)
				}
				wantZ := slices.Concat(c.z, fzMultisetMinus(lcm[i], c.p))
				nn := tf.Num[i][j]
				mon := make([]float64, len(nn))
				for k, v := range nn {
					mon[k] = v / nn[0]
				}
				if msg := fzRouche(back.Zeros[i][j], wantZ, mon, nil); msg != "" {
					t.Fatalf("round-trip zeros (%d,%d): %s\ngot %v want %v\n%s", i, j, msg, back.Zeros[i][j], wantZ, s)
				}
			}
		}
	})
}

type fzZPKChan struct {
	z, p []complex128
	g    float64
}

// fzConvGroups splits Eig into conjugate-closed groups.
func fzConvGroups(s *fzSpec) [][]complex128 {
	var g [][]complex128
	for k := 0; k < s.N; {
		e := s.Eig[k]
		if imag(s.S[k]) != 0 && imag(e) != 0 {
			e = complex(real(e), math.Abs(imag(e)))
			g = append(g, []complex128{e, cmplx.Conj(e)})
			k += 2
			continue
		}
		if imag(s.S[k]) != 0 {
			g = append(g, []complex128{complex(real(e), 0), complex(real(e), 0)})
			k += 2
			continue
		}
		g = append(g, []complex128{complex(real(e), 0)})
		k++
	}
	return g
}

func fzBuildZPK(s *fzSpec) [][]fzZPKChan {
	groups := fzConvGroups(s)
	ng := len(groups)
	phase := fzClamp(s.Vals[11], -1e3, 1e3)
	g0 := math.Pow(10, fzClamp(math.Abs(s.Vals[6]), 0, 12)-6)
	if s.Vals[6] < 0 {
		g0 = -g0
	}
	out := make([][]fzZPKChan, s.P)
	for i := range s.P {
		out[i] = make([]fzZPKChan, s.M)
		for j := range s.M {
			drop := (i + 2*j + int(s.Kind)) % (ng + 1)
			var live [][]complex128
			var c fzZPKChan
			for k, gr := range groups {
				if k != drop {
					live = append(live, gr)
					c.p = append(c.p, gr...)
				}
			}
			nzg := (3*i + 5*j + int(s.Kind>>5)) % (len(live) + 1)
			rho := math.Pow(10, 0.8*math.Sin(phase+1.7*float64(i)+2.3*float64(j)))
			if i == 0 && j == 0 && s.Vals[10] < 0 {
				rho = 1
			}
			for _, gr := range live[:nzg] {
				for _, v := range gr {
					c.z = append(c.z, v*complex(rho, 0))
				}
			}
			c.g = g0 * math.Cos(phase+0.9*float64(i)+1.1*float64(j))
			if s.Vals[7] < 0 && i == s.P-1 && j == s.M-1 {
				c.g = 0
			}
			out[i][j] = c
		}
	}
	return out
}

// fzRowLCM is the multiset union (max multiplicity) of the row's poles,
// over every channel or only those with nonzero gain.
func fzRowLCM(row []fzZPKChan, live bool) []complex128 {
	var out []complex128
	for _, c := range row {
		if live && c.g == 0 {
			continue
		}
		out = append(out, fzMultisetMinus(c.p, out)...)
	}
	return out
}

// fzMultisetMinus returns a minus b as exact-value multisets.
func fzMultisetMinus(a, b []complex128) []complex128 {
	rest := slices.Clone(b)
	var out []complex128
	for _, v := range a {
		if k := slices.Index(rest, v); k >= 0 {
			rest = slices.Delete(rest, k, k+1)
			continue
		}
		out = append(out, v)
	}
	return out
}

// fzConvDistinct reports that values in v are equal or clearly apart, so
// the library's tolerance-based pole matching agrees with exact equality.
func fzConvDistinct(v []complex128) bool {
	for i := range v {
		for j := i + 1; j < len(v); j++ {
			d := cmplx.Abs(v[i] - v[j])
			if d != 0 && d < 1e-11*(1+cmplx.Abs(v[i])) {
				return false
			}
		}
	}
	return true
}

// fzNearMerge reports unequal values within the 100·eps·(1+|p|) window
// that commonDenomPoles and poleSetDifference treat as one pole.
func fzNearMerge(v []complex128) bool {
	for i := range v {
		for j := i + 1; j < len(v); j++ {
			if d := cmplx.Abs(v[i] - v[j]); d != 0 && d < 400*0x1p-52*(1+math.Max(cmplx.Abs(v[i]), cmplx.Abs(v[j]))) {
				return true
			}
		}
	}
	return false
}

// fzProdEval returns g·Π(x−z)/Π(x−p) and Σ (|x|+|r|)/|x−r|, the relative
// sensitivity of each factor to rounding; sens is NaN when x hits a root.
func fzProdEval(c fzZPKChan, x complex128) (h complex128, sens float64) {
	h = complex(c.g, 0)
	if c.g == 0 {
		return 0, 0
	}
	for _, r := range c.z {
		d := x - r
		if d == 0 {
			return 0, math.NaN()
		}
		h *= d
		sens += (cmplx.Abs(x) + cmplx.Abs(r)) / cmplx.Abs(d)
	}
	for _, r := range c.p {
		d := x - r
		if d == 0 {
			return 0, math.NaN()
		}
		h /= d
		sens += (cmplx.Abs(x) + cmplx.Abs(r)) / cmplx.Abs(d)
	}
	return h, sens
}

func fzAbsProd(r []complex128, x complex128) float64 {
	v := 1.0
	for _, q := range r {
		v *= cmplx.Abs(x) + cmplx.Abs(q)
	}
	return v
}

// fzZPKCoefErr bounds |num/den(x) − h| when num and den are formed from
// roots in floating point: each coefficient errs by at most γ times the
// coefficient of Π(s+|r|). It is +Inf when the denominator error may
// reach half of |den(x)|, where no first-order bound holds.
func fzZPKCoefErr(c fzZPKChan, lcm []complex128, x, h complex128) float64 {
	const eps = 0x1p-52
	if c.g == 0 {
		return 0
	}
	nr := slices.Concat(c.z, fzMultisetMinus(lcm, c.p))
	dp := complex(1, 0)
	for _, r := range lcm {
		dp *= x - r
	}
	k := 16 * float64(len(lcm)+len(nr)+2)
	rd := k * eps * fzAbsProd(lcm, x) / cmplx.Abs(dp)
	if rd > 0.5 {
		return math.Inf(1)
	}
	return (k*eps*math.Abs(c.g)*fzAbsProd(nr, x)/cmplx.Abs(dp) + cmplx.Abs(h)*rd) / (1 - rd)
}

// fzHornerErr bounds the Horner rounding error of num(x)/den(x).
func fzHornerErr(num, den []float64, x complex128) float64 {
	const eps = 0x1p-52
	ax := cmplx.Abs(x)
	abs := func(c []float64) float64 {
		v := 0.0
		for _, a := range c {
			v = v*ax + math.Abs(a)
		}
		return v
	}
	d := cmplx.Abs(fzBigPolyEval(den, x))
	h := cmplx.Abs(fzBigPolyEval(num, x)) / d
	k := 8 * float64(len(den)+len(num))
	return k * eps * (abs(num) + h*abs(den)) / d
}

// fzCheckRealization checks that res realizes tf: dimensions, block sizes,
// delays and names carry over, and the exact response of the float64
// realization equals num/den within the companion rounding bound.
func fzCheckRealization(t *testing.T, s *fzSpec, op string, tf *TransferFunc, res *StateSpaceResult) {
	t.Helper()
	const eps = 0x1p-52
	p, m := len(tf.Den), len(tf.Num[0])
	sys := res.Sys
	n, mm, pp := sys.Dims()
	if mm != m || pp != p || sys.Dt != tf.Dt {
		t.Fatalf("%s: %d×%d dt=%g, want %d×%d dt=%g\n%s", op, pp, mm, sys.Dt, p, m, tf.Dt, s)
	}
	total := 0
	for i := range p {
		if res.BlockSizes[i] != len(tf.Den[i])-1 {
			t.Fatalf("%s: BlockSizes = %v, Den = %v\n%s", op, res.BlockSizes, tf.Den, s)
		}
		total += res.BlockSizes[i]
	}
	if n != total || res.MinimalOrder != total {
		t.Fatalf("%s: order %d MinimalOrder %d, want Σ row degrees %d\n%s", op, n, res.MinimalOrder, total, s)
	}
	if tf.Delay != nil {
		for i := range p {
			for j := range m {
				if got := sys.Delay.At(i, j); got != tf.Delay[i][j] {
					t.Fatalf("%s: delay[%d,%d] = %g, TF %g\n%s", op, i, j, got, tf.Delay[i][j], s)
				}
			}
		}
	}
	a, b, c, d := fzRaw(sys)
	for _, x := range fzConvPoints(s) {
		g, ok := fzExactResponse(n, m, p, a, b, c, d, x)
		if !ok {
			continue
		}
		ax := cmplx.Abs(x)
		for i := range p {
			den := tf.Den[i]
			dx := cmplx.Abs(fzBigPolyEval(den, x))
			if dx == 0 {
				continue
			}
			for j := range m {
				num := tf.Num[i][j]
				pad := make([]float64, len(den))
				copy(pad[len(den)-len(num):], num)
				h := fzBigPolyEval(num, x) / fzBigPolyEval(den, x)
				var en, ed float64
				for k := range den {
					en = en*ax + math.Abs(pad[k]) + math.Abs(pad[0]/den[0]*den[k])
					ed = ed*ax + math.Abs(den[k])
				}
				kk := 8 * float64(len(den)+1)
				rd := kk * eps * ed / dx
				if rd > 0.5 {
					continue
				}
				tol := (kk*eps*en/dx + cmplx.Abs(h)*rd) / (1 - rd)
				if diff := cmplx.Abs(g[i*m+j] - h); diff > tol {
					t.Fatalf("%s(%v)[%d,%d] = %v, TF %v (tol %g)\nnum=%v den=%v\n%s", op, x, i, j, g[i*m+j], h, tol, num, den, s)
				}
			}
		}
	}
}

func fzRaw(sys *System) (a, b, c, d []float64) {
	n, m, p := sys.Dims()
	get := func(x interface {
		At(int, int) float64
	}, r, cc int) []float64 {
		out := make([]float64, r*cc)
		for i := range r {
			for j := range cc {
				out[i*cc+j] = x.At(i, j)
			}
		}
		return out
	}
	if n > 0 {
		a, b, c = get(sys.A, n, n), get(sys.B, n, m), get(sys.C, p, n)
	}
	return a, b, c, get(sys.D, p, m)
}

// fzConvPoints returns deterministic evaluation points off the stability
// boundary at the spectral scales of the spec.
func fzConvPoints(s *fzSpec) []complex128 {
	lo, hi := math.Inf(1), 0.0
	for _, e := range s.S {
		if a := cmplx.Abs(e); a > 0 {
			lo, hi = math.Min(lo, a), math.Max(hi, a)
		}
	}
	if hi == 0 {
		lo, hi = 1, 1
	}
	th := 0.37 + 0.1*math.Sin(s.Vals[11])
	var out []complex128
	for k, r := range []float64{lo, math.Sqrt(lo * hi), hi * 3} {
		x := cmplx.Rect(r, th+0.9*float64(k))
		if s.Discrete {
			xd := x * complex(s.Dt, 0)
			if a := cmplx.Abs(xd); a > 3 {
				xd *= complex(3/a, 0)
			}
			x = cmplx.Exp(xd)
		}
		out = append(out, x)
	}
	return out
}

// fzConvFreqs returns real frequencies for FreqResponse (below Nyquist).
func fzConvFreqs(s *fzSpec) []float64 {
	pts := fzConvPoints(s)
	var ws []float64
	for _, x := range pts {
		w := cmplx.Abs(x)
		if s.Discrete {
			w = math.Min(math.Abs(imag(cmplx.Log(x)))/s.Dt, 0.9*math.Pi/s.Dt)
		}
		if w > 0 && !math.IsInf(w, 0) {
			ws = append(ws, w)
		}
	}
	return ws
}

// fzRouche checks returned roots of the float64 polynomial c (monic up to
// rounding) against the intended roots want. Each coefficient of
// Π(s−want) is perturbed by at most E(|x|) = K·eps·(coef of Π(s+|w|) +
// ‖ĉ‖·max(1,‖ĉ‖)) |x|^k (coefficient rounding plus a normwise companion
// eigenvalue backward error; extra adds further absolute coefficient
// error bounds, aligned at the constant term), so by Rouché a disc about want[j] whose
// boundary has |Π(x−want)| > E holds as many returned as intended roots.
func fzRouche(got, want []complex128, c, extra []float64) string {
	const eps = 0x1p-52
	if len(got) != len(want) {
		return "root count differs"
	}
	cn := 0.0
	for _, v := range c[1:] {
		cn = math.Max(cn, math.Abs(v/c[0]))
	}
	abs := make([]float64, 1, len(want)+1)
	abs[0] = 1
	for _, w := range want {
		r := cmplx.Abs(w)
		next := make([]float64, len(abs)+1)
		for k, v := range abs {
			next[k] += v
			next[k+1] += v * r
		}
		abs = next
	}
	for k, v := range extra {
		abs[len(abs)-len(extra)+k] += v
	}
	kk := 64 * float64(len(want)+1)
	E := func(ax float64) float64 {
		v := 0.0
		for _, a := range abs {
			v = v*ax + kk*eps*(a+cn*math.Max(1, cn))
		}
		return v
	}
	for _, w := range want {
		scale := 1 + cmplx.Abs(w)
		for e := 28; e >= -4; e-- {
			rho := scale * math.Pow(10, -float64(e)/2)
			L := 1.0
			for _, q := range want {
				L *= math.Abs(cmplx.Abs(w-q) - rho)
			}
			if !(E(cmplx.Abs(w)+rho) < L) {
				continue
			}
			nw, ng := 0, 0
			for _, q := range want {
				if cmplx.Abs(q-w) < rho {
					nw++
				}
			}
			for _, q := range got {
				if cmplx.Abs(q-w) < rho {
					ng++
				}
			}
			if nw != ng {
				return fmt.Sprintf("Rouché disc |s−%v| < %.3g holds %d returned, %d intended roots", w, rho, ng, nw)
			}
			break
		}
	}
	return ""
}

// fzConvSeeds: SISO and MIMO, continuous and discrete, integrators and
// z = 1 poles, repeated and clustered poles, exact pole-zero cancellation
// (Rot < 0), a zero-gain channel (DScal < 0), static gains, wide dynamic
// range, a strongly sheared (badly scaled, BLYSOP) A and poles clustered
// at z = 1 (small Dt). Output gain is 1 (Gain 6) so dynamics dominate D,
// except one seed with gain 1e-6.
func fzConvSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 1, Gain: 6}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0] = 2, 1, 1, 2.5 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.Gain = 2, 1, 1, 2.5, 0 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Rot = 3, 2, 2, [3]float64{0, 0.5, 0}, [3]float64{1.5, 3, 0}, -0.4
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.DScal = 4, 2, 3, [3]float64{-40, 0.3, 0}, -0.5 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Shear = 5, 3, 1, [3]float64{-1, 1, 2}, [3]float64{2.2, -13, 1}, 0.2
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Gain = 6, 1, 2, [3]float64{-2, 0, 2}, [3]float64{1.8, 2.4, 3.5}, 5
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.Damp = 4, 1, 1, [3]float64{0.5, 0.5, 0}, [3]float64{2, 2, 0} }),
		with(func(s *fzSeed) { s.N, s.M, s.P = 0, 2, 2 }),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.LogDt, s.Mag, s.Damp = true, 3, 2, 2, -1, [3]float64{0, -40, 0}, [3]float64{1.5, 1, 0}
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.LogDt, s.Mag, s.Damp, s.Rot = true, 5, 3, 3, -3, [3]float64{-1, 0.5, 1}, [3]float64{2, -1.5, 13}, -1
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.LogDt, s.Mag, s.Damp, s.Shear = true, 4, 1, 1, 0, [3]float64{0.2, 0.4, 0}, [3]float64{1.2, -2.5, 0}, 0.4
		}),
	}
}
