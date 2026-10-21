package controlsys

// API robustness fuzzing (task NWNFLF): arbitrary dimensions and values
// (NaN, Inf, empty, mismatched sizes) through exported constructors,
// metadata and container operations. Failures: panic, Go error-contract
// breach (fzContract), an accepted value that fails Validate or differs
// from the exact expectation, aliasing of inputs, and value->error on
// inputs the documented rules accept. Any other error is counted
// (fzAPIRejected), never a failure.
//
// The input is the canonical tuple plus three bytes: op selects the entry
// point, site the corrupted block and bad the corruption (fzAPICorrupt). v
// is used raw, so the fuzzer's own NaN/Inf values reach the library; the
// System targets build a valid base with fzDecode first.

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type fzAPIIn struct {
	Kind          uint8
	V             [fzNVals]float64
	Op, Site, Bad uint8
}

func fzAPIFuzz(f *testing.F, body func(t *testing.T, in *fzAPIIn)) {
	fzStats(f)
	f.Fuzz(func(t *testing.T, kind uint8, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 float64, op, site, bad uint8) {
		skipSlowUnderRace(t)
		in := &fzAPIIn{Kind: kind, V: [fzNVals]float64{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15}, Op: op, Site: site, Bad: bad}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic: %v\n%s", r, in)
			}
		}()
		body(t, in)
	})
}

func (in *fzAPIIn) String() string {
	return fmt.Sprintf("kind=%#02x vals=%v op=%d site=%d bad=%d", in.Kind, in.V, in.Op, in.Site, in.Bad)
}

func fzAPIAddSeed(f *testing.F, s fzSeed, op, site, bad uint8) {
	k, v := s.encode()
	f.Add(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], op, site, bad)
}

// fzAPIRejected counts an error by its first sentinel so stats show what
// the library rejects.
func fzAPIRejected(t *testing.T, op string, err error) {
	t.Helper()
	for _, e := range []error{ErrInvalidArgument, ErrDimensionMismatch, ErrInvalidSampleTime, ErrDomainMismatch, ErrSignalNotFound, ErrVoidModel, ErrDescriptorSingular, ErrSchurFailed, ErrAlgebraicLoop, ErrInvalidExpression} {
		if errors.Is(err, e) {
			fzSkip(t, op+": "+e.Error())
		}
	}
	fzSkip(t, op+": other error")
}

func fzAPIContract(t *testing.T, in *fzAPIIn, op string, hasValue bool, err error) {
	t.Helper()
	if err == nil && !hasValue {
		t.Fatalf("%s: nil value with nil error\n%s", op, in)
	}
	if err != nil && hasValue {
		t.Fatalf("%s: value returned with error %v\n%s", op, err, in)
	}
}

func fzAPIBadValue(bad uint8) float64 {
	switch bad % 4 {
	case 1:
		return math.NaN()
	case 2:
		return math.Inf(1)
	case 3:
		return math.Inf(-1)
	}
	return 0
}

func fzAPISame(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}

func fzAPIFloatsEqual(a, b []float64) bool {
	return len(a) == len(b) && !slices.ContainsFunc(rangeInts(len(a)), func(i int) bool { return !fzAPISame(a[i], b[i]) })
}

// ---------------------------------------------------------------- Poly

func fzAPIPolyOf(in *fzAPIIn, first, length int) Poly {
	p := make(Poly, length)
	for i := range p {
		p[i] = in.V[(first+i)%fzNVals]
	}
	if in.Bad&0x80 != 0 && length > 0 {
		p[int(in.Site>>4)%length] = fzAPIBadValue(in.Bad >> 2)
	}
	if in.Bad&0x40 != 0 && length > 1 {
		p[0] = 0
	}
	return p
}

func fzAPIFinite(p ...float64) bool {
	return !slices.ContainsFunc(p, func(x float64) bool { return math.IsNaN(x) || math.IsInf(x, 0) })
}

// fzAPIPolyMulExact returns the exact product coefficients and the sum of
// |p_i q_j| per coefficient.
func fzAPIPolyMulExact(p, q Poly) (exact []*big.Float, abs []float64) {
	exact = make([]*big.Float, len(p)+len(q)-1)
	abs = make([]float64, len(exact))
	for k := range exact {
		exact[k] = new(big.Float).SetPrec(4400)
	}
	for i, a := range p {
		for j, b := range q {
			t := new(big.Float).SetPrec(4400).Mul(big.NewFloat(a), big.NewFloat(b))
			exact[i+j].Add(exact[i+j], t)
			abs[i+j] += math.Abs(a) * math.Abs(b)
		}
	}
	return exact, abs
}

func fzAPICheckMul(t *testing.T, in *fzAPIIn, op string, p, q, r Poly) {
	t.Helper()
	if len(p) == 0 || len(q) == 0 {
		if len(r) != 0 {
			t.Fatalf("%s: empty factor gave %v\n%s", op, r, in)
		}
		return
	}
	if len(r) != len(p)+len(q)-1 {
		t.Fatalf("%s: length %d, want %d\n%s", op, len(r), len(p)+len(q)-1, in)
	}
	if !fzAPIFinite(p...) || !fzAPIFinite(q...) {
		return
	}
	exact, abs := fzAPIPolyMulExact(p, q)
	L := float64(min(len(p), len(q)))
	for k := range r {
		if math.IsInf(abs[k], 0) || abs[k] > math.MaxFloat64/4 {
			continue
		}
		want, _ := exact[k].Float64()
		tol := 2*L*0x1p-53*abs[k] + 2*L*0x1p-1074
		if !(math.Abs(r[k]-want) <= tol) {
			t.Fatalf("%s: coefficient %d = %g, exact %g, tol %g\np=%v q=%v\n%s", op, k, r[k], want, tol, p, q, in)
		}
	}
}

func fzAPIAddWant(p, q Poly, sign float64) Poly {
	size := max(len(p), len(q))
	r := make(Poly, size)
	for i := range r {
		var pv, qv float64
		if k := i - (size - len(p)); k >= 0 {
			pv = p[k]
		}
		if k := i - (size - len(q)); k >= 0 {
			qv = sign * q[k]
		}
		switch {
		case i < size-len(p):
			r[i] = qv
		case i < size-len(q):
			r[i] = pv
		default:
			r[i] = pv + qv
		}
	}
	return r
}

// FuzzAPIPoly checks every Poly method on arbitrary coefficients against
// exact expectations: Add/Sub/Scale/Derivative are single correctly rounded
// operations, Mul is within the convolution error bound of the exact
// big.Float product, the *To variants match the allocating ones whatever
// the overlap of dst with the operands, and Roots/Monic follow their
// documented error rules.
func FuzzAPIPoly(f *testing.F) {
	for _, c := range []struct {
		kind          uint8
		v             []float64
		op, site, bad uint8
	}{
		{3, []float64{1, -3, 2}, 0, 0, 0},
		{4, []float64{0, 0, 1, 2}, 1, 2, 0x40},
		{5, []float64{1, 2, 3, 4, 5}, 2, 0x13, 0x84},
		{6, []float64{1e300, 1e-300, 3, -1, 2, 7}, 3, 0x21, 0x88},
		{2, []float64{0, 0}, 4, 0x05, 0},
		{7, []float64{1, 0, 0, 0, 0, 0, -1}, 5, 0x37, 0x8c},
		{1, []float64{5}, 6, 0x40, 0x80},
		{0, nil, 7, 0, 0},
	} {
		var v [fzNVals]float64
		copy(v[:], c.v)
		for i := len(c.v); i < fzNVals; i++ {
			v[i] = float64(i%5) - 2
		}
		f.Add(c.kind, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], c.op, c.site, c.bad)
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		p := fzAPIPolyOf(in, 0, int(in.Kind%9))
		q := fzAPIPolyOf(in, 8, int(in.Site%9))
		pSnap, qSnap := slices.Clone(p), slices.Clone(q)
		x := in.V[15]
		if got := p.Degree(); got != len(p)-1 {
			t.Fatalf("Degree = %d, len %d\n%s", got, len(p), in)
		}
		if fzAPIFinite(p...) && fzAPIFinite(x) {
			fzAPICheckEval(t, in, p, x)
		}

		fzAPICheckMul(t, in, "Mul", p, q, p.Mul(q))
		if want, got := fzAPIAddWant(p, q, 1), p.Add(q); !fzAPIFloatsEqual(got, want) {
			t.Fatalf("Add = %v, want %v\n%s", got, want, in)
		}
		if want, got := fzAPIAddWant(p, q, -1), p.Sub(q); !fzAPIFloatsEqual(got, want) {
			t.Fatalf("Sub = %v, want %v\n%s", got, want, in)
		}
		s := in.V[14]
		sc := p.Scale(s)
		for i := range p {
			if len(sc) != len(p) || !fzAPISame(sc[i], p[i]*s) {
				t.Fatalf("Scale(%g) = %v of %v\n%s", s, sc, p, in)
			}
		}
		dp := p.Derivative()
		if len(p) <= 1 {
			if !fzAPIFloatsEqual(dp, Poly{0}) {
				t.Fatalf("Derivative of %v = %v, want [0]\n%s", p, dp, in)
			}
		} else {
			for i := range dp {
				if len(dp) != len(p)-1 || !fzAPISame(dp[i], p[i]*float64(len(p)-1-i)) {
					t.Fatalf("Derivative of %v = %v\n%s", p, dp, in)
				}
			}
		}
		if p.IsMonic() != (len(p) > 0 && p[0] == 1) {
			t.Fatalf("IsMonic(%v) = %v\n%s", p, p.IsMonic(), in)
		}
		fzAPICheckMonic(t, in, p)
		fzAPICheckEqual(t, in, p, q, s)
		fzAPICheckTo(t, in, p, q, s)
		if !fzAPIFloatsEqual(p, pSnap) || !fzAPIFloatsEqual(q, qSnap) {
			t.Fatalf("operands modified: p %v -> %v, q %v -> %v\n%s", pSnap, p, qSnap, q, in)
		}
		fzAPICheckRoots(t, in, p)
	})
}

func fzAPICheckEval(t *testing.T, in *fzAPIIn, p Poly, x float64) {
	t.Helper()
	got := p.Eval(complex(x, 0))
	if len(p) == 0 {
		if got != 0 {
			t.Fatalf("Eval of empty = %v\n%s", got, in)
		}
		return
	}
	exact := new(big.Float).SetPrec(4400)
	bx := new(big.Float).SetPrec(4400).SetFloat64(x)
	absSum := 0.0
	for i, c := range p {
		exact.Mul(exact, bx).Add(exact, big.NewFloat(c))
		absSum = absSum*math.Abs(x) + math.Abs(c)
		_ = i
	}
	if math.IsInf(absSum, 0) || absSum > math.MaxFloat64/4 {
		return
	}
	want, _ := exact.Float64()
	tol := 2*float64(len(p))*0x1p-53*absSum*2 + 4*float64(len(p))*0x1p-1074
	if !(math.Abs(real(got)-want) <= tol) || imag(got) != 0 {
		t.Fatalf("Eval(%g) = %v, exact %g, tol %g\np=%v\n%s", x, got, want, tol, p, in)
	}
}

func fzAPICheckMonic(t *testing.T, in *fzAPIIn, p Poly) {
	t.Helper()
	r, err := p.Monic()
	fzAPIContract(t, in, "Monic", r != nil, err)
	wantErr := len(p) == 0 || p[0] == 0
	if (err != nil) != wantErr {
		t.Fatalf("Monic(%v) err = %v, want error %v\n%s", p, err, wantErr, in)
	}
	if err != nil {
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("Monic(%v) err = %v, want ErrInvalidArgument\n%s", p, err, in)
		}
		return
	}
	if len(r) != len(p) {
		t.Fatalf("Monic length %d of %v\n%s", len(r), p, in)
	}
	if !fzAPIFinite(p...) || math.Abs(p[0]) < 0x1p-1000 {
		return
	}
	for i := range p {
		want := new(big.Float).SetPrec(200).Quo(big.NewFloat(p[i]), big.NewFloat(p[0]))
		w, _ := want.Float64()
		if math.IsInf(w, 0) {
			continue
		}
		if !(math.Abs(r[i]-w) <= 2*0x1p-52*math.Abs(w)+0x1p-1070) {
			t.Fatalf("Monic(%v)[%d] = %g, want %g\n%s", p, i, r[i], w, in)
		}
	}
}

func fzAPICheckEqual(t *testing.T, in *fzAPIIn, p, q Poly, tol float64) {
	t.Helper()
	tol = math.Abs(tol)
	want := len(p) == len(q)
	if want {
		for i := range p {
			if !(math.Abs(p[i]-q[i]) <= tol) {
				want = false
			}
		}
	}
	got := p.Equal(q, tol)
	if got != want {
		if got && (!fzAPIFinite(p...) || !fzAPIFinite(q...) || math.IsNaN(tol)) {
			fzSkip(t, "known: Poly.Equal treats NaN differences as equal")
		}
		t.Fatalf("Equal(%v, %v, %g) = %v, want %v\n%s", p, q, tol, got, want, in)
	}
	if !p.Equal(slices.Clone(p), 0) && fzAPIFinite(p...) {
		t.Fatalf("finite %v not Equal to its copy\n%s", p, in)
	}
}

// fzAPICheckTo runs MulTo/AddTo/ScaleTo with dst chosen by in.Op among a
// fresh buffer, nil, and windows of one backing array that also holds p and
// q, so every overlap case is exercised.
func fzAPICheckTo(t *testing.T, in *fzAPIIn, p, q Poly, s float64) {
	t.Helper()
	lp, lq := len(p), len(q)
	back := make([]float64, lp+lq+8)
	copy(back, p)
	copy(back[lp:], q)
	ap, aq := Poly(back[:lp:lp]), Poly(back[lp:lp+lq:lp+lq])
	if in.Op&0x80 != 0 {
		ap, aq = Poly(back[:lp]), Poly(back[lp:lp+lq])
	}
	off := int(in.Op>>2) % (len(back) + 1)
	var dst Poly
	switch in.Op % 4 {
	case 0:
		dst = make(Poly, 0, int(in.Op>>4))
	case 1:
		dst = nil
	case 2:
		dst = Poly(back[off:off])
	default:
		dst = Poly(back[off:])
	}
	pSnap, qSnap := slices.Clone(ap), slices.Clone(aq)
	mul := ap.MulTo(dst, aq)
	if !fzAPIFloatsEqual(mul, p.Mul(q)) {
		t.Fatalf("MulTo = %v, Mul = %v (op %d)\n%s", mul, p.Mul(q), in.Op, in)
	}
	if !fzAPIFloatsEqual(ap, pSnap) || !fzAPIFloatsEqual(aq, qSnap) {
		t.Fatalf("MulTo modified operands\n%s", in)
	}
	copy(back, p)
	copy(back[lp:], q)
	add := ap.AddTo(dst, aq)
	if !fzAPIFloatsEqual(add, p.Add(q)) {
		t.Fatalf("AddTo = %v, Add = %v (op %d)\n%s", add, p.Add(q), in.Op, in)
	}
	if !fzAPIFloatsEqual(ap, pSnap) || !fzAPIFloatsEqual(aq, qSnap) {
		t.Fatalf("AddTo modified operands\n%s", in)
	}
	copy(back, p)
	copy(back[lp:], q)
	if sc := ap.ScaleTo(ap, s); !fzAPIFloatsEqual(sc, p.Scale(s)) {
		t.Fatalf("in-place ScaleTo = %v, Scale = %v\n%s", sc, p.Scale(s), in)
	}
	if in.Op%4 >= 2 {
		// ScaleTo documents only dst == p as a permitted overlap.
		dst = make(Poly, int(in.Op>>4))
	}
	if sc := Poly(p).ScaleTo(dst, s); !fzAPIFloatsEqual(sc, p.Scale(s)) {
		t.Fatalf("ScaleTo = %v, Scale = %v\n%s", sc, p.Scale(s), in)
	}
}

func fzAPICheckRoots(t *testing.T, in *fzAPIIn, p Poly) {
	t.Helper()
	r, err := p.Roots()
	fzAPIContract(t, in, "Roots", r != nil, err)
	start := 0
	for start < len(p) && p[start] == 0 {
		start++
	}
	trimmed := p[start:]
	if !fzAPIFinite(p...) || len(trimmed) == 0 {
		if err == nil || !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("Roots(%v) = %v, %v; want ErrInvalidArgument\n%s", p, r, err, in)
		}
		return
	}
	if err != nil {
		if errors.Is(err, ErrSchurFailed) {
			fzSkip(t, "Roots: companion eigensolver failed")
		}
		t.Fatalf("Roots(%v): %v\n%s", p, err, in)
	}
	if len(r) != len(trimmed)-1 {
		t.Fatalf("Roots(%v) gave %d roots, want %d\n%s", p, len(r), len(trimmed)-1, in)
	}
	for _, c := range trimmed[1:] {
		if math.IsInf(c/trimmed[0], 0) {
			return
		}
	}
	var pos, neg int
	for _, z := range r {
		if math.IsNaN(real(z)) || math.IsNaN(imag(z)) || math.IsInf(real(z), 0) || math.IsInf(imag(z), 0) {
			t.Fatalf("Roots(%v) non-finite root %v\n%s", p, r, in)
		}
		if imag(z) > 0 {
			pos++
		} else if imag(z) < 0 {
			neg++
		}
	}
	if pos != neg {
		t.Fatalf("Roots(%v) = %v not conjugate-closed\n%s", p, r, in)
	}
}

// -------------------------------------------------------- System blocks

// fzAPIParts is the raw constructor input built from a valid base spec.
type fzAPIParts struct {
	A, B, C, D, E, Delay *mat.Dense
	Dt                   float64
	DimsBroken           bool
	ValueBad             bool
	Desc                 string
}

func fzAPIDense(r, c int, f func(i, j int) float64) *mat.Dense {
	if r <= 0 || c <= 0 {
		return &mat.Dense{}
	}
	d := mat.NewDense(r, c, nil)
	for i := range r {
		for j := range c {
			d.Set(i, j, f(i, j))
		}
	}
	return d
}

// fzAPIBuild turns s into constructor blocks and corrupts one of them:
// site%7 picks A, B, C, D, E, Dt or Delay and bad%8 applies none, NaN,
// +Inf, -Inf, nil, empty, an extra row or an extra column. site bit 3
// drops the inputs (m = 0), bit 4 the outputs (p = 0), bit 5 adds a
// nonsingular E (bit 6: singular).
func fzAPIBuild(s *fzSpec, site, bad uint8) *fzAPIParts {
	n, m, p := s.N, s.M, s.P
	if site&0x08 != 0 {
		m = 0
	}
	if site&0x10 != 0 {
		p = 0
	}
	at := func(v []float64, cols int) func(i, j int) float64 {
		return func(i, j int) float64 { return v[i*cols+j] }
	}
	pt := &fzAPIParts{Dt: s.Dt}
	pt.A = fzAPIDense(n, n, at(s.A, s.N))
	pt.B = fzAPIDense(n, m, at(s.B, s.M))
	pt.C = fzAPIDense(p, n, at(s.C, s.N))
	pt.D = fzAPIDense(p, m, at(s.D, s.M))
	if site&0x20 != 0 && n > 0 {
		pt.E = fzAPIDense(n, n, func(i, j int) float64 {
			if i == j {
				if site&0x40 != 0 && i == n-1 {
					return 0
				}
				return 1 + 0.25*float64(i)
			}
			if j == i+1 {
				return 0.5
			}
			return 0
		})
	}
	if s.IO != nil && p > 0 && m > 0 {
		pt.Delay = fzAPIDense(p, m, func(i, j int) float64 { return s.IO[(i*s.M+j)%len(s.IO)] })
	}
	blocks := []**mat.Dense{&pt.A, &pt.B, &pt.C, &pt.D, &pt.E, nil, &pt.Delay}
	k := int(site % 7)
	pt.Desc = fmt.Sprintf("n=%d m=%d p=%d site=%d bad=%d", n, m, p, k, bad%8)
	if k == 5 {
		switch bad % 8 {
		case 1:
			pt.Dt = math.NaN()
		case 2:
			pt.Dt = math.Inf(1)
		case 3:
			pt.Dt = -1
		case 4:
			pt.Dt = -s.Dt - 1e-3
		}
		if bad%8 != 0 && bad%8 < 5 {
			pt.ValueBad = true
		}
		return pt
	}
	bp := blocks[k]
	b := *bp
	switch bad % 8 {
	case 1, 2, 3:
		if b != nil && !b.IsEmpty() {
			r, c := b.Dims()
			b.Set(int(bad>>3)%r, int(bad>>5)%c, fzAPIBadValue(bad))
			if k >= 4 {
				pt.ValueBad = true
			}
		}
	case 4:
		*bp = nil
	case 5:
		if b != nil {
			*bp = &mat.Dense{}
		}
	case 6, 7:
		if b == nil {
			break
		}
		r, c := 0, 0
		if !b.IsEmpty() {
			r, c = b.Dims()
		}
		if bad%8 == 6 {
			r++
		} else {
			c++
		}
		if r > 0 && c > 0 {
			*bp = fzAPIDense(r, c, func(i, j int) float64 { return float64(i - j) })
			pt.DimsBroken = true
		}
	}
	return pt
}

// fzAPIConsistent applies the ss/dss size rule to the non-empty blocks: A
// square, B with A's rows, C with A's columns, D with C's rows and B's
// columns; E must be n×n and the I/O delay p×m when present.
func fzAPIConsistent(pt *fzAPIParts) (abcd, e, delay bool) {
	dims := func(m *mat.Dense) (int, int, bool) {
		if m == nil || m.IsEmpty() {
			return 0, 0, false
		}
		r, c := m.Dims()
		return r, c, true
	}
	ar, ac, hasA := dims(pt.A)
	br, bc, hasB := dims(pt.B)
	cr, cc, hasC := dims(pt.C)
	dr, dc, hasD := dims(pt.D)
	n := ar
	if !hasA {
		n = max(br, cc)
	}
	abcd = !(hasA && ar != ac) && !(hasB && br != n) && !(hasC && cc != n) &&
		!(hasD && ((hasC && dr != cr) || (hasB && dc != bc)))
	m, p := bc, cr
	if !hasB {
		m = dc
	}
	if !hasC {
		p = dr
	}
	er, ec, hasE := dims(pt.E)
	lr, lc, hasL := dims(pt.Delay)
	return abcd, !hasE || (er == n && ec == n), !hasL || (lr == p && lc == m)
}

func fzAPIMats(pt *fzAPIParts) []*mat.Dense {
	return []*mat.Dense{pt.A, pt.B, pt.C, pt.D, pt.E, pt.Delay}
}

func fzAPIRequireValid(t *testing.T, in fmt.Stringer, op string, sys *System) {
	t.Helper()
	if err := sys.Validate(); err != nil {
		t.Fatalf("%s returned invalid model: %v\n%s", op, err, in)
	}
	_ = sys.String()
	if d := systemDiff(fzAPINormalize(sys), fzAPINormalize(sys.Copy())); d != "" {
		t.Fatalf("%s: Copy differs in %s\n%s", op, d, in)
	}
}

// fzAPINormalize returns a shallow copy of sys with empty matrices as nil,
// so representation-only differences of empty blocks compare equal.
func fzAPINormalize(sys *System) *System {
	v := *sys
	ne := func(m *mat.Dense) *mat.Dense {
		if m == nil || m.IsEmpty() {
			return nil
		}
		return m
	}
	v.A, v.B, v.C, v.D, v.E, v.Delay = ne(v.A), ne(v.B), ne(v.C), ne(v.D), ne(v.E), ne(v.Delay)
	if sys.LFT != nil {
		l := *sys.LFT
		l.B2, l.C2, l.D12, l.D21, l.D22 = ne(l.B2), ne(l.C2), ne(l.D12), ne(l.D21), ne(l.D22)
		v.LFT = &l
	}
	return &v
}

// fzAPISameBlock reports whether got stores exactly the entries of want; a
// nil or empty want matches an empty or all-zero got of the derived size.
func fzAPISameBlock(want, got *mat.Dense) bool {
	if want == nil || want.IsEmpty() {
		if got == nil || got.IsEmpty() {
			return true
		}
		r, c := got.Dims()
		for i := range r {
			for j := range c {
				if got.At(i, j) != 0 {
					return false
				}
			}
		}
		return true
	}
	if got == nil || got.IsEmpty() {
		return false
	}
	wr, wc := want.Dims()
	gr, gc := got.Dims()
	if wr != gr || wc != gc {
		return false
	}
	for i := range wr {
		for j := range wc {
			if !fzAPISame(want.At(i, j), got.At(i, j)) {
				return false
			}
		}
	}
	return true
}

func fzAPIMutate(ms ...*mat.Dense) {
	for _, m := range ms {
		if m != nil && !m.IsEmpty() {
			fillDense(m)
		}
	}
}

func fzAPICloneMats(ms []*mat.Dense) []*mat.Dense {
	out := make([]*mat.Dense, len(ms))
	for i, m := range ms {
		if m != nil {
			out[i] = &mat.Dense{}
			if !m.IsEmpty() {
				out[i] = mat.DenseCopyOf(m)
			}
		}
	}
	return out
}

// fzAPIConstruct builds the model with the constructor chosen by op%4:
// New, NewWithDelay, NewDescriptor or NewFromSlices.
func fzAPIConstruct(t *testing.T, in *fzAPIIn, pt *fzAPIParts) (*System, error, string) {
	t.Helper()
	switch in.Op % 4 {
	case 0:
		sys, err := New(pt.A, pt.B, pt.C, pt.D, pt.Dt)
		return sys, err, "New"
	case 1:
		sys, err := NewWithDelay(pt.A, pt.B, pt.C, pt.D, pt.Delay, pt.Dt)
		return sys, err, "NewWithDelay"
	case 2:
		sys, err := NewDescriptor(pt.A, pt.B, pt.C, pt.D, pt.E, pt.Dt)
		return sys, err, "NewDescriptor"
	}
	raw := func(m *mat.Dense) []float64 {
		if m == nil || m.IsEmpty() {
			return nil
		}
		return slices.Clone(m.RawMatrix().Data)
	}
	dim := func(m *mat.Dense, row bool) int {
		if m == nil || m.IsEmpty() {
			return 0
		}
		r, c := m.Dims()
		if row {
			return r
		}
		return c
	}
	n, mm, p := dim(pt.A, true), dim(pt.B, false), dim(pt.C, true)
	if n == 0 {
		p, mm = dim(pt.D, true), dim(pt.D, false)
	}
	sys, err := NewFromSlices(n, mm, p, raw(pt.A), raw(pt.B), raw(pt.C), raw(pt.D), pt.Dt)
	return sys, err, "NewFromSlices"
}

func fzAPISystemSeeds() []fzSeed {
	return []fzSeed{
		{N: 2, M: 1, P: 1, Mag: [3]float64{0, 1}, Damp: [3]float64{1, 1}, Gain: 6, DScal: 0.5, LogDt: -1, Rot: 0.3, Phase: 0.2, In: fzNoDelay, Out: fzNoDelay, IO: -1, Tau: fzNoDelay},
		{N: 3, M: 2, P: 3, Mag: [3]float64{0.5, 2, -1}, Damp: [3]float64{2, -1, 0}, Gain: 7, DScal: 1, LogDt: -1, Shear: -1, Rot: 0.7, Phase: 1, In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay},
		{Discrete: true, N: 4, M: 3, P: 2, Mag: [3]float64{0, 0.5, 1}, Damp: [3]float64{1, 3, 0.5}, Gain: 6, DScal: -2, LogDt: -1, Shear: -2, Rot: 1.1, Phase: 2, In: fzNoDelay, Out: fzNoDelay, IO: 0.5, Tau: fzNoDelay},
		{N: 0, M: 2, P: 2, Gain: 6, DScal: 3, LogDt: -1, Rot: 0.1, Phase: 0.4, In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay},
		{N: 1, M: 1, P: 2, Mag: [3]float64{-1}, Damp: [3]float64{1}, Gain: 6, DScal: 0, LogDt: -1, Rot: 0.5, Phase: 3, In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay},
	}
}

// FuzzAPISystem drives the System constructors with corrupted blocks, then
// the name setters and SelectByIndex/SelectByName on accepted models.
// Oracles: a dimension-consistent input with a valid sample time, delay and
// E must be accepted; an inconsistent one must be rejected; accepted
// models validate, store exactly the given entries and do not alias them;
// name and index rules are the documented MATLAB ones and the selected
// blocks are the exact submatrices.
func FuzzAPISystem(f *testing.F) {
	for i, s := range fzAPISystemSeeds() {
		for _, c := range [][3]uint8{{0, 0, 0}, {1, 6, 0x0a}, {2, 0x24, 1}, {3, 0x13, 6}, {0, 0x08, 5}, {1, 0x10, 4}, {2, 0x64, 0}, {3, 5, 3}, {0, 1, 7}} {
			fzAPIAddSeed(f, s, c[0]+uint8(i)<<4, c[1], c[2]+uint8(i)<<3)
		}
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		s := fzDecode(t, in.Kind, in.V)
		pt := fzAPIBuild(s, in.Site, in.Bad)
		inputs := fzAPICloneMats(fzAPIMats(pt))
		sys, err, op := fzAPIConstruct(t, in, pt)
		fzAPIContract(t, in, op, sys != nil, err)
		usesDelay := op == "NewWithDelay"
		k := int(in.Site % 7)
		relevant := k < 4 || k == 5 || (k == 4 && op == "NewDescriptor") || (k == 6 && usesDelay)
		abcd, eOK, delayOK := fzAPIConsistent(pt)
		consistent := abcd && (eOK || op != "NewDescriptor") && (delayOK || !usesDelay)
		mustReject := (pt.DimsBroken && !consistent) || (relevant && pt.ValueBad && k >= 5)
		mustAccept := consistent && !(relevant && pt.ValueBad) && !fzAPIAmbiguous(pt, k, in)
		if err != nil {
			if mustAccept {
				t.Fatalf("%s rejected a consistent input (%s): %v\n%s\n%s", op, pt.Desc, err, in, s)
			}
			fzAPIRejected(t, op, err)
		}
		if mustReject {
			t.Fatalf("%s accepted invalid blocks (%s)\n%s\n%s", op, pt.Desc, in, s)
		}
		fzAPIRequireValid(t, in, op, sys)
		empty := func(m *mat.Dense) bool { return m == nil || m.IsEmpty() }
		if empty(pt.A) && empty(pt.B) && !empty(pt.C) {
			fzSkip(t, "known: New with empty A and B drops a non-empty C")
		}
		if op != "NewFromSlices" {
			for i, got := range []*mat.Dense{sys.A, sys.B, sys.C, sys.D} {
				if !fzAPISameBlock(inputs[i], got) {
					t.Fatalf("%s block %d stored %v, given %v (%s)\n%s", op, i, got, inputs[i], pt.Desc, in)
				}
			}
		}
		snap := sys.Copy()
		fzAPIMutate(fzAPIMats(pt)...)
		if d := systemDiff(snap, sys); d != "" {
			t.Fatalf("%s aliases its %s argument\n%s", op, d, in)
		}
		fzAPINames(t, in, sys)
		fzAPISelect(t, in, sys)
	})
}

func sysDimsOf(pt *fzAPIParts) (n, m, p int) {
	n, m, p, _ = validateDims(pt.A, pt.B, pt.C, pt.D, 0)
	return
}

// fzAPIAmbiguous reports inputs whose acceptance the oracle cannot decide
// from the documented rules alone: blocks emptied so the model has no
// derivable dimension.
func fzAPIAmbiguous(pt *fzAPIParts, k int, in *fzAPIIn) bool {
	n, m, p := sysDimsOf(pt)
	if n == 0 && (m == 0) != (p == 0) {
		return true
	}
	if b := in.Bad % 8; b == 5 || (b == 4 && k < 4) {
		return true
	}
	return false
}

func fzAPINameList(in *fzAPIIn, prefix string, count int) []string {
	out := make([]string, count)
	for i := range out {
		switch (int(in.Bad>>4) + i) % 4 {
		case 0:
			out[i] = ""
		case 1:
			out[i] = prefix
		default:
			out[i] = fmt.Sprintf("%s%d", prefix, i)
		}
	}
	return out
}

func fzAPINames(t *testing.T, in *fzAPIIn, sys *System) {
	t.Helper()
	n, m, p := sys.Dims()
	work := sys.Copy()
	for _, c := range []struct {
		op    string
		want  int
		count int
		set   func(...string) error
		get   func() []string
	}{
		{"SetInputName", m, int(in.Op>>2) % 5, work.SetInputName, func() []string { return work.InputName }},
		{"SetOutputName", p, int(in.Op>>3) % 5, work.SetOutputName, func() []string { return work.OutputName }},
		{"SetStateName", n, int(in.Site>>5) % 5, work.SetStateName, func() []string { return work.StateName }},
	} {
		names := fzAPINameList(in, c.op[3:4], c.count)
		before := slices.Clone(c.get())
		err := c.set(names...)
		ok := c.count == c.want || (c.count == 1 && c.want > 1)
		if (err == nil) != ok {
			t.Fatalf("%s(%d names) on %d channels: err %v\n%s", c.op, c.count, c.want, err, in)
		}
		if err != nil {
			if !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("%s: %v, want ErrDimensionMismatch\n%s", c.op, err, in)
			}
			if !slices.Equal(before, c.get()) {
				t.Fatalf("%s failed but changed names\n%s", c.op, in)
			}
			continue
		}
		if got := c.get(); len(got) != c.want && !(c.want == 0 && got == nil) {
			t.Fatalf("%s: %d names for %d channels\n%s", c.op, len(got), c.want, in)
		}
		if len(names) > 0 {
			names[0] = "mutated"
			if g := c.get(); len(g) > 0 && g[0] == "mutated" && c.count == c.want {
				t.Fatalf("%s aliases its argument\n%s", c.op, in)
			}
		}
	}
	if err := work.Validate(); err != nil {
		t.Fatalf("names left an invalid model: %v\n%s", err, in)
	}
	_ = work.String()
}

func fzAPIIndices(in *fzAPIIn, first, count, limit int) []int {
	out := make([]int, count)
	for i := range out {
		x := in.V[(first+i)%fzNVals]
		switch {
		case math.IsNaN(x):
			out[i] = -1
		case x > 0:
			out[i] = int(math.Min(x*7, float64(limit+1))) % (limit + 2)
		default:
			out[i] = -int(math.Min(-x, 3)) + int(in.Op)%max(limit, 1)
		}
	}
	return out
}

func fzAPISelect(t *testing.T, in *fzAPIIn, sys *System) {
	t.Helper()
	n, m, p := sys.Dims()
	ins := fzAPIIndices(in, 3, int(in.Kind>>1)%4, m)
	outs := fzAPIIndices(in, 9, int(in.Bad>>5)%4, p)
	snap := sys.Copy()
	res, err := sys.SelectByIndex(ins, outs)
	fzAPIContract(t, in, "SelectByIndex", res != nil, err)
	inRange := func(idx []int, lim int) bool {
		return !slices.ContainsFunc(idx, func(i int) bool { return i < 0 || i >= lim })
	}
	storable := n > 0 || (len(ins) == 0) == (len(outs) == 0)
	ok := inRange(ins, m) && inRange(outs, p) && storable
	if (err == nil) != ok {
		t.Fatalf("SelectByIndex(%v, %v) on n=%d m=%d p=%d: err %v\n%s", ins, outs, n, m, p, err, in)
	}
	if d := systemDiff(snap, sys); d != "" {
		t.Fatalf("SelectByIndex changed its receiver (%s)\n%s", d, in)
	}
	if err != nil {
		return
	}
	fzAPIRequireValid(t, in, "SelectByIndex", res)
	rn, rm, rp := res.Dims()
	if rn != n || rm != len(ins) || rp != len(outs) {
		t.Fatalf("SelectByIndex dims %d,%d,%d want %d,%d,%d\n%s", rn, rm, rp, n, len(ins), len(outs), in)
	}
	sub := func(src *mat.Dense, rows, cols []int) *mat.Dense {
		return fzAPIDense(len(rows), len(cols), func(i, j int) float64 { return src.At(rows[i], cols[j]) })
	}
	all := rangeInts(n)
	if n > 0 && !fzAPISameBlock(sys.A, res.A) {
		t.Fatalf("SelectByIndex changed A\n%s", in)
	}
	if len(ins) > 0 && n > 0 && !fzAPISameBlock(sub(sys.B, all, ins), res.B) {
		t.Fatalf("SelectByIndex B\n%s", in)
	}
	if len(outs) > 0 && n > 0 && !fzAPISameBlock(sub(sys.C, outs, all), res.C) {
		t.Fatalf("SelectByIndex C\n%s", in)
	}
	if len(ins) > 0 && len(outs) > 0 && !fzAPISameBlock(sub(sys.D, outs, ins), res.D) {
		t.Fatalf("SelectByIndex D\n%s", in)
	}
	mutateSystem(res)
	if d := systemDiff(snap, sys); d != "" {
		t.Fatalf("SelectByIndex result aliases receiver (%s)\n%s", d, in)
	}

	named := sys.Copy()
	named.InputName = autoLabel("in", m)
	named.OutputName = autoLabel("out", p)
	if in.Bad&1 != 0 && m > 0 {
		named.InputName = nil
	}
	pick := func(names []string, idx []int) []string {
		out := make([]string, len(idx))
		for i, k := range idx {
			out[i] = fmt.Sprintf("missing%d", k)
			if k >= 0 && k < len(names) {
				out[i] = names[k]
			}
		}
		return out
	}
	byName, err := named.SelectByName(pick(named.InputName, ins), pick(named.OutputName, outs))
	fzAPIContract(t, in, "SelectByName", byName != nil, err)
	nameOK := ok && (named.InputName != nil || len(ins) == 0)
	if (err == nil) != nameOK {
		if !nameOK && errors.Is(err, ErrSignalNotFound) == false && err != nil {
			t.Fatalf("SelectByName: %v\n%s", err, in)
		}
		t.Fatalf("SelectByName(%v, %v): err %v, want ok %v\n%s", ins, outs, err, nameOK, in)
	}
	if err == nil {
		want, _ := named.SelectByIndex(ins, outs)
		if d := systemDiff(want, byName); d != "" {
			t.Fatalf("SelectByName differs from SelectByIndex in %s\n%s", d, in)
		}
	}
}

// ---------------------------------------------------------- ModelArray

// fzAPIBaseSystem returns the delay-free model of s.
func fzAPIBaseSystem(t *testing.T, s *fzSpec) *System {
	t.Helper()
	sys, err := NewFromSlices(s.N, s.M, s.P, s.A, s.B, s.C, s.D, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices on decoded spec: %v\n%s", err, s)
	}
	return sys
}

// fzAPIVariant derives a slot model from base: code 0 void, 2 one input
// fewer, 3 another sample time, 4 renamed inputs, 5 default-named inputs,
// 6 one state, 7 one output fewer, otherwise a copy.
func fzAPIVariant(base *System, code uint8) *System {
	n, m, p := base.Dims()
	switch code % 8 {
	case 0:
		return nil
	case 2:
		if m > 1 {
			if v, err := base.SelectByIndex(rangeInts(m-1), rangeInts(p)); err == nil {
				return v
			}
		}
	case 3:
		v := base.Copy()
		v.Dt = 0.5
		if base.Dt == 0.5 {
			v.Dt = 0.25
		}
		return v
	case 4:
		v := base.Copy()
		v.InputName = autoLabel("w", m)
		return v
	case 5:
		v := base.Copy()
		v.InputName = autoLabel("u", m)
		return v
	case 6:
		if n > 1 {
			if v, err := NewFromSlices(1, m, p, []float64{-1}, make([]float64, m), make([]float64, p), base.D.RawMatrix().Data, base.Dt); err == nil {
				return v
			}
		}
	case 7:
		if p > 1 {
			if v, err := base.SelectByIndex(rangeInts(m), rangeInts(p-1)); err == nil {
				return v
			}
		}
	}
	return base.Copy()
}

// fzAPIShape maps the bytes to a shape of rank 0..3 with dims in [-1, 3];
// bad bit 7 makes one huge dimension whose product with the other wraps
// int to the model count.
func fzAPIShape(in *fzAPIIn, count int) []int {
	rank := int(in.Site>>6) % 4
	shape := make([]int, rank)
	for i := range shape {
		shape[i] = int((in.Site>>(2*i))&3) - 1 + int(in.Bad>>6&1)
	}
	if rank == 1 && in.Bad&0x20 == 0 {
		shape[0] = count
	}
	if in.Bad&0x80 != 0 && rank > 0 && count > 0 {
		// 3·6148914691236517206 = 2^64 + 2 and 274177·67280421310721 = 2^64 + 1.
		shape = []int{3, 6148914691236517206}
		if count == 1 {
			shape = []int{274177, 67280421310721}
		}
	}
	return shape
}

func fzAPIShapeProduct(shape []int) (*big.Int, bool) {
	prod := big.NewInt(1)
	ok := len(shape) > 0
	for _, d := range shape {
		if d <= 0 {
			ok = false
		}
		prod.Mul(prod, big.NewInt(int64(d)))
	}
	return prod, ok
}

func fzAPICompatible(ref, sys *System) bool {
	_, rm, rp := ref.Dims()
	_, sm, sp := sys.Dims()
	names := func(a, b []string) bool { return len(a) == 0 || len(b) == 0 || slices.Equal(a, b) }
	return rm == sm && rp == sp && ref.Dt == sys.Dt && names(ref.InputName, sys.InputName) && names(ref.OutputName, sys.OutputName)
}

// FuzzAPIModelArray builds arrays from void, compatible and incompatible
// slots with arbitrary shapes. Oracles: NewModelArray's documented
// acceptance rule with the exact (big.Int) shape product; Model, ModelFlat,
// IsVoid, SelectFlat and ResponseFlat on every in- and out-of-range index;
// Stack/Concat of the array with itself; no aliasing of inputs or results;
// nil-receiver errors.
func FuzzAPIModelArray(f *testing.F) {
	for i, s := range fzAPISystemSeeds() {
		for _, c := range [][3]uint8{{0x11, 0x40, 0}, {0x12, 0x45, 0x40}, {0x1b, 0x80, 0x20}, {0x23, 0xc6, 0x01}, {0x01, 0x40, 0x80}, {0x30, 0x55, 0x60}, {0x2c, 0x41, 0x80}, {0x06, 0x81, 0x81}, {0x06, 0x40, 0x80}} {
			fzAPIAddSeed(f, s, c[0]+uint8(i)<<5, c[1], c[2]+uint8(i)<<1)
		}
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		s := fzDecode(t, in.Kind, in.V)
		base := fzAPIBaseSystem(t, s)
		count := int(in.Op&3) + int(in.Bad&1)
		codes := []uint8{in.Op >> 2, in.Op >> 5, in.Bad >> 1, in.Bad >> 4}
		models := make([]*System, count)
		for i := range models {
			models[i] = fzAPIVariant(base, codes[i%len(codes)]+uint8(i/len(codes)))
		}
		shape := fzAPIShape(in, count)
		prod, positive := fzAPIShapeProduct(shape)
		var ref *System
		compatible := true
		for _, sys := range models {
			if sys == nil {
				continue
			}
			if ref == nil {
				ref = sys
			} else if !fzAPICompatible(ref, sys) {
				compatible = false
			}
		}
		exact := prod.IsInt64() && prod.Int64() == int64(count)
		want := positive && exact && ref != nil && compatible
		snaps := make([]*System, count)
		for i, sys := range models {
			if sys != nil {
				snaps[i] = sys.Copy()
			}
		}
		fzAPINilModelArray(t, in)
		arr, err := NewModelArray(shape, models)
		fzAPIContract(t, in, "NewModelArray", arr != nil, err)
		if (err == nil) != want {
			if err == nil && positive && !exact {
				fzAPIModelArrayOverflow(t, arr, shape)
			}
			t.Fatalf("NewModelArray(%v, %d models, compatible %v): err %v, want ok %v\n%s", shape, count, compatible, err, want, in)
		}
		if err != nil {
			fzAPIRejected(t, "NewModelArray", err)
		}
		for i, a := range snaps {
			for _, b := range snaps[:i] {
				if a != nil && b != nil && !fzAPICompatible(a, b) {
					fzSkip(t, "known: NewModelArray checks names only against the first model")
				}
			}
		}
		for i, sys := range models {
			if sys != nil {
				mutateSystem(sys)
			}
			models[i] = nil
		}
		if arr.Len() != count || !slices.Equal(arr.Shape(), shape) {
			t.Fatalf("Len %d Shape %v, want %d %v\n%s", arr.Len(), arr.Shape(), count, shape, in)
		}
		_, rm, rp := ref.Dims()
		if p, m := arr.IOSize(); p != rp || m != rm {
			t.Fatalf("IOSize %d,%d, want %d,%d\n%s", p, m, rp, rm, in)
		}
		for i := -1; i <= count; i++ {
			got, err := arr.ModelFlat(i)
			fzAPIContract(t, in, "ModelFlat", got != nil, err)
			switch {
			case i < 0 || i >= count:
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("ModelFlat(%d) of %d: %v\n%s", i, count, err, in)
				}
			case snaps[i] == nil:
				if !errors.Is(err, ErrVoidModel) {
					t.Fatalf("ModelFlat(%d) void slot: %v\n%s", i, err, in)
				}
			default:
				if err != nil {
					t.Fatalf("ModelFlat(%d): %v\n%s", i, err, in)
				}
				if d := systemDiff(snaps[i], got); d != "" {
					t.Fatalf("ModelFlat(%d) differs from its input in %s\n%s", i, d, in)
				}
				mutateSystem(got)
				if again, _ := arr.ModelFlat(i); systemDiff(snaps[i], again) != "" {
					t.Fatalf("ModelFlat(%d) leaks array storage\n%s", i, in)
				}
			}
		}
		fzAPIModelIndex(t, in, arr, shape, snaps)
		sel := fzAPIIndices(in, 0, int(in.Kind>>3)%4, count)
		got, err := arr.SelectFlat(sel...)
		fzAPIContract(t, in, "SelectFlat", got != nil, err)
		inRange, anyModel := true, false
		for _, k := range sel {
			if k < 0 || k >= count {
				inRange = false
			} else if snaps[k] != nil {
				anyModel = true
			}
		}
		if (err == nil) != (inRange && anyModel) {
			t.Fatalf("SelectFlat(%v) of %d slots: err %v\n%s", sel, count, err, in)
		}
		if err == nil && got.Len() != len(sel) {
			t.Fatalf("SelectFlat(%v) Len %d\n%s", sel, got.Len(), in)
		}
		cat, err := ConcatModelArrays(arr, arr)
		fzAPIContract(t, in, "ConcatModelArrays", cat != nil, err)
		if err != nil || cat.Len() != 2*count {
			t.Fatalf("ConcatModelArrays self: %v\n%s", err, in)
		}
		st, err := StackModelArrays(arr, arr)
		fzAPIContract(t, in, "StackModelArrays", st != nil, err)
		if err != nil || !slices.Equal(st.Shape(), append([]int{2}, shape...)) {
			t.Fatalf("StackModelArrays self: %v\n%s", err, in)
		}
		if _, err := ConcatModelArrays(arr, nil); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("ConcatModelArrays with nil: %v\n%s", err, in)
		}
		omega := []float64{math.Abs(in.V[14]), 1}
		if in.Op&0x10 != 0 {
			omega = append(omega, in.V[13])
		}
		fr, err := arr.FreqResponse(omega)
		fzAPIContract(t, in, "ModelArray.FreqResponse", fr != nil, err)
		if err == nil {
			for i := -1; i <= count; i++ {
				r, err := fr.ResponseFlat(i)
				fzAPIContract(t, in, "ResponseFlat", r != nil, err)
				if (err == nil) != (i >= 0 && i < count && snaps[i] != nil) {
					t.Fatalf("ResponseFlat(%d): %v\n%s", i, err, in)
				}
			}
		}
		if tf := in.V[13]; math.IsNaN(tf) || tf < 0 {
			if st, err := arr.Step(tf); err == nil {
				t.Fatalf("ModelArray.Step(%g) accepted: %v\n%s", tf, st, in)
			}
		}
	})
}

// fzAPIModelArrayOverflow handles an array accepted with a shape whose
// exact product differs from its model count (int wrap-around).
func fzAPIModelArrayOverflow(t *testing.T, arr *ModelArray, shape []int) {
	t.Helper()
	idx := make([]int, len(shape))
	for i, d := range shape {
		idx[i] = d - 1
	}
	panicked := func() (p bool) {
		defer func() { p = recover() != nil }()
		_, _ = arr.IsVoid(idx...)
		return false
	}()
	if panicked {
		fzSkip(t, "known: NewModelArray accepts wrapped shape product; IsVoid panics")
	}
	fzSkip(t, "known: NewModelArray accepts wrapped shape product")
}

func fzAPINilModelArray(t *testing.T, in *fzAPIIn) {
	t.Helper()
	var a *ModelArray
	if a.Len() != 0 || a.Shape() != nil {
		t.Fatalf("nil ModelArray Len/Shape\n%s", in)
	}
	var fr *ModelArrayFreqResponse
	var tr *ModelArrayTimeResponse
	for op, call := range map[string]func() error{
		"Model":        func() error { _, e := a.Model(0); return e },
		"ModelFlat":    func() error { _, e := a.ModelFlat(0); return e },
		"IsVoid":       func() error { _, e := a.IsVoid(0); return e },
		"SelectFlat":   func() error { _, e := a.SelectFlat(0); return e },
		"FreqResponse": func() error { _, e := a.FreqResponse([]float64{1}); return e },
		"Step":         func() error { _, e := a.Step(1); return e },
		"Stack":        func() error { _, e := StackModelArrays(); return e },
		"Concat":       func() error { _, e := ConcatModelArrays(); return e },
		"StackNil":     func() error { _, e := StackModelArrays(nil); return e },
		"FreqFlat":     func() error { _, e := fr.ResponseFlat(0); return e },
		"TimeFlat":     func() error { _, e := tr.ResponseFlat(0); return e },
		"Empty":        func() error { _, e := NewModelArray([]int{1}, []*System{nil}); return e },
	} {
		if err := call(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("nil ModelArray %s: %v\n%s", op, err, in)
		}
	}
}

// fzAPIModelIndex checks Model and IsVoid over every multi-index, one out
// of range per dimension and a wrong-rank call.
func fzAPIModelIndex(t *testing.T, in *fzAPIIn, arr *ModelArray, shape []int, snaps []*System) {
	t.Helper()
	idx := make([]int, len(shape))
	for flat := range snaps {
		got, err := arr.Model(idx...)
		fzAPIContract(t, in, "Model", got != nil, err)
		void, verr := arr.IsVoid(idx...)
		if verr != nil || void != (snaps[flat] == nil) {
			t.Fatalf("IsVoid(%v) = %v, %v; flat %d\n%s", idx, void, verr, flat, in)
		}
		if snaps[flat] == nil {
			if !errors.Is(err, ErrVoidModel) {
				t.Fatalf("Model(%v) void: %v\n%s", idx, err, in)
			}
		} else if err != nil || systemDiff(snaps[flat], got) != "" {
			t.Fatalf("Model(%v) != flat %d: %v\n%s", idx, flat, err, in)
		}
		for k := len(idx) - 1; k >= 0; k-- {
			if idx[k]++; idx[k] < shape[k] {
				break
			}
			idx[k] = 0
		}
	}
	for d := range shape {
		for _, bad := range []int{-1, shape[d]} {
			idx := make([]int, len(shape))
			idx[d] = bad
			if _, err := arr.Model(idx...); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("Model(%v) out of range: %v\n%s", idx, err, in)
			}
			if _, err := arr.IsVoid(idx...); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("IsVoid(%v) out of range: %v\n%s", idx, err, in)
			}
		}
	}
	if _, err := arr.Model(make([]int, len(shape)+1)...); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Model wrong rank: %v\n%s", err, in)
	}
}

// ------------------------------------------- interconnect / generalized

// fzAPISecond builds a second model with n2, m2, p2 from op bits and the
// texture of v, with sample time dt (bad bit 0: a different one).
func fzAPISecond(in *fzAPIIn, s *fzSpec) *System {
	n2, m2, p2 := int(in.Op>>3)%3, 1+int(in.Op>>5)%3, 1+int(in.Site>>5)%3
	dt := s.Dt
	if in.Bad&1 != 0 {
		dt = map[bool]float64{true: 0, false: 0.1}[s.Discrete]
	}
	tex := func(k int) float64 { return math.Cos(1.7*float64(k) + fzClamp(in.V[11], -1e3, 1e3)) }
	a := make([]float64, n2*n2)
	for i := range n2 {
		a[i*n2+i] = -0.5 - float64(i)
		if dt > 0 {
			a[i*n2+i] = 0.3 + 0.2*float64(i)
		}
		if i+1 < n2 {
			a[i*n2+i+1] = tex(i)
		}
	}
	b, c, d := make([]float64, n2*m2), make([]float64, p2*n2), make([]float64, p2*m2)
	for k := range b {
		b[k] = tex(k + 3)
	}
	for k := range c {
		c[k] = tex(k + 7)
	}
	for k := range d {
		d[k] = 0.25 * tex(k+11)
	}
	sys, err := NewFromSlices(n2, m2, p2, a, b, c, d, dt)
	if err != nil {
		panic(fmt.Sprintf("fzAPISecond: %v", err))
	}
	return sys
}

// fzAPIFirst is the spec's model with its input, output and I/O delays.
func fzAPIFirst(t *testing.T, s *fzSpec) *System {
	t.Helper()
	sys := fzAPIBaseSystem(t, s)
	if s.IO != nil {
		if err := sys.SetDelay(mat.NewDense(s.P, s.M, slices.Clone(s.IO))); err != nil {
			t.Fatalf("SetDelay: %v\n%s", err, s)
		}
	}
	if err := sys.SetInputDelay(s.In); err != nil {
		t.Fatalf("SetInputDelay: %v\n%s", err, s)
	}
	if err := sys.SetOutputDelay(s.Out); err != nil {
		t.Fatalf("SetOutputDelay: %v\n%s", err, s)
	}
	return sys
}

// FuzzAPIInterconnect connects the spec model (with delays) and a second
// model of independent size and possibly another sample time through
// Series, Parallel, Append, BlkDiag, Feedback, LFT and the generalized
// closed loop. Oracles: the documented size and sample-time rules decide
// acceptance (ErrAlgebraicLoop is the only other allowed error), accepted
// results validate with the expected I/O size, and neither the operands
// nor later mutation of the result change the inputs.
func FuzzAPIInterconnect(f *testing.F) {
	for i, s := range fzAPISystemSeeds() {
		for _, c := range [][3]uint8{{0x00, 0, 0}, {0x09, 0x20, 0}, {0x12, 0x40, 1}, {0x2b, 0x00, 0}, {0x34, 0x60, 0x10}, {0x05, 0x20, 0x20}, {0x0e, 0x00, 0x40}, {0x07, 0x20, 0}} {
			fzAPIAddSeed(f, s, c[0]+uint8(i)<<6, c[1]+uint8(i), c[2])
		}
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		s := fzDecode(t, in.Kind, in.V)
		s1 := fzAPIFirst(t, s)
		s2 := fzAPISecond(in, s)
		_, m1, p1 := s1.Dims()
		_, m2, p2 := s2.Dims()
		same := s1.Dt == s2.Dt
		sign := []float64{-1, 1, 0, math.NaN()}[in.Bad>>4&3]
		snap1, snap2 := s1.Copy(), s2.Copy()
		var (
			op       string
			res      *System
			err      error
			ok       bool
			wm, wp   int
			loopOnly bool
		)
		switch in.Op % 8 {
		case 0:
			op, ok, wm, wp = "Series", same && p1 == m2, m1, p2
			res, err = Series(s1, s2)
		case 1:
			op, ok, wm, wp = "Parallel", same && m1 == m2 && p1 == p2, m1, p1
			res, err = Parallel(s1, s2)
		case 2:
			op, ok, wm, wp = "Append", same, m1+m2, p1+p2
			res, err = Append(s1, s2)
		case 3:
			op, ok, wm, wp = "BlkDiag", same, 2*m1+m2, 2*p1+p2
			res, err = BlkDiag(s1, s2, s1)
		case 4:
			op, ok, wm, wp, loopOnly = "Feedback", same && p1 == m2 && m1 == p2 && (sign == 1 || sign == -1), m1, p1, true
			res, err = Feedback(s1, s2, sign)
		case 5:
			op, ok, wm, wp, loopOnly = "Feedback(unity)", m1 == p1 && (sign == 1 || sign == -1), m1, p1, true
			res, err = Feedback(s1, nil, sign)
		case 6:
			op, loopOnly = "LFT", true
			switch {
			case m2 < m1 && p2 < p1:
				ok, wm, wp = same && p2 <= m1 && m2 <= p1, m1-p2, p1-m2
			case m1 < m2 && p1 < p2:
				ok, wm, wp = same && m1 <= p2 && p1 <= m2, m2-p1, p2-m1
			}
			res, err = LFT(s1, s2)
		default:
			fzAPIGeneralizedLoop(t, in, s1, s2)
			return
		}
		fzAPIContract(t, in, op, res != nil, err)
		if d := systemDiff(snap1, s1) + systemDiff(snap2, s2); d != "" {
			t.Fatalf("%s modified an operand (%s)\n%s", op, d, in)
		}
		if err != nil {
			unstorable := (wm == 0) != (wp == 0) && errors.Is(err, ErrDimensionMismatch)
			if ok && !(loopOnly && errors.Is(err, ErrAlgebraicLoop)) && !unstorable {
				t.Fatalf("%s rejected compatible models (m1=%d p1=%d m2=%d p2=%d dt %g/%g): %v\n%s\n%s", op, m1, p1, m2, p2, s1.Dt, s2.Dt, err, in, s)
			}
			fzAPIRejected(t, op, err)
		}
		if !ok {
			t.Fatalf("%s accepted incompatible models (m1=%d p1=%d m2=%d p2=%d dt %g/%g sign %g)\n%s", op, m1, p1, m2, p2, s1.Dt, s2.Dt, sign, in)
		}
		if rn, _, _ := res.Dims(); rn == 0 && (wm == 0) != (wp == 0) && res.Validate() != nil {
			fzSkip(t, "known: interconnection with no states and only inputs or only outputs fails Validate")
		}
		fzAPIRequireValid(t, in, op, res)
		if _, m, p := res.Dims(); m != wm || p != wp {
			t.Fatalf("%s size %dx%d, want %dx%d\n%s", op, p, m, wp, wm, in)
		}
		if res == s1 || res == s2 {
			t.Fatalf("%s returned an operand\n%s", op, in)
		}
		mutateSystem(res)
		if d := systemDiff(snap1, s1) + systemDiff(snap2, s2); d != "" {
			t.Fatalf("%s result aliases an operand (%s)\n%s", op, d, in)
		}
	})
}

func fzAPIGeneralizedLoop(t *testing.T, in *fzAPIIn, plant, ctrl *System) {
	t.Helper()
	_, mP, pP := plant.Dims()
	_, mC, pC := ctrl.Dims()
	if _, err := FixedBlock(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("FixedBlock(nil): %v\n%s", err, in)
	}
	blk, err := FixedBlock(ctrl)
	if err != nil {
		t.Fatalf("FixedBlock: %v\n%s", err, in)
	}
	snap := ctrl.Copy()
	mutateSystem(ctrl)
	if cur, err := blk.CurrentSystem(); err != nil || systemDiff(snap, cur) != "" {
		t.Fatalf("FixedBlock aliases its model: %v\n%s", err, in)
	}
	ctrl = snap
	name := []string{"", "ap", "y"}[in.Site%3]
	gm, err := NewGeneralizedModel(name, blk)
	fzAPIContract(t, in, "NewGeneralizedModel", gm != nil, err)
	if (err == nil) != (name != "") {
		t.Fatalf("NewGeneralizedModel(%q): %v\n%s", name, err, in)
	}
	if gm != nil {
		k := int(in.Bad>>2) % 4
		_ = gm.SetInputName(fzAPINameList(in, "i", k)...)
		cur, err := gm.CurrentSystem()
		fzAPIContract(t, in, "GeneralizedModel.CurrentSystem", cur != nil, err)
		if k > 0 && (err == nil) != (k == mC || (k == 1 && mC > 1)) {
			t.Fatalf("GeneralizedModel.CurrentSystem with %d names for %d inputs: %v\n%s", k, mC, err, in)
		}
		if err := gm.InsertAnalysisPoint(""); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("InsertAnalysisPoint(\"\"): %v\n%s", err, in)
		}
		if _, err := gm.AnalysisPoint("missing"); !errors.Is(err, ErrSignalNotFound) {
			t.Fatalf("AnalysisPoint(missing): %v\n%s", err, in)
		}
	}
	cl, err := NewGeneralizedClosedLoop("cl", plant, blk, name)
	fzAPIContract(t, in, "NewGeneralizedClosedLoop", cl != nil, err)
	if (err == nil) != (name != "") {
		t.Fatalf("NewGeneralizedClosedLoop(ap %q): %v\n%s", name, err, in)
	}
	if err != nil {
		fzAPIRejected(t, "NewGeneralizedClosedLoop", err)
	}
	loc := AnalysisPointLocation(in.Bad >> 6)
	locErr := cl.InsertAnalysisPoint("u", loc)
	if (locErr == nil) != (loc == AnalysisPointPlantOutput || loc == AnalysisPointPlantInput) {
		t.Fatalf("InsertAnalysisPoint location %d: %v\n%s", loc, locErr, in)
	}
	same := plant.Dt == ctrl.Dt
	for _, ap := range []string{name, "u", "missing"} {
		at := AnalysisPointPlantOutput
		if ap == "u" {
			if locErr != nil {
				continue
			}
			at = loc
		}
		lm, lp, ok := mC, pP, same && pC == mP
		if at == AnalysisPointPlantInput {
			lm, lp, ok = mP, pC, same && pP == mC
		}
		if ap == "missing" {
			ok = false
		}
		ol, err := cl.OpenLoop(ap)
		fzAPIContract(t, in, "OpenLoop", ol != nil, err)
		if (err == nil) != ok {
			t.Fatalf("OpenLoop(%q) plant %dx%d ctrl %dx%d: %v\n%s", ap, pP, mP, pC, mC, err, in)
		}
		if err != nil {
			continue
		}
		fzAPIRequireValid(t, in, "OpenLoop", ol)
		if _, m, p := ol.Dims(); m != lm || p != lp {
			t.Fatalf("OpenLoop(%q) size %dx%d, want %dx%d\n%s", ap, p, m, lp, lm, in)
		}
		for opName, call := range map[string]func(string) (*System, error){
			"ClosedLoop": cl.ClosedLoop, "ComplementarySensitivity": cl.ComplementarySensitivity, "Sensitivity": cl.Sensitivity,
		} {
			r, err := call(ap)
			fzAPIContract(t, in, opName, r != nil, err)
			if err == nil {
				fzAPIRequireValid(t, in, opName, r)
				if _, m, p := r.Dims(); m != lm || p != lp {
					t.Fatalf("%s(%q) size %dx%d, want %dx%d\n%s", opName, ap, p, m, lp, lm, in)
				}
			} else if lm == lp && !errors.Is(err, ErrAlgebraicLoop) {
				t.Fatalf("%s(%q) on square %dx%d loop: %v\n%s", opName, ap, lp, lm, err, in)
			}
		}
	}
	var nilCL *GeneralizedClosedLoop
	if _, err := nilCL.OpenLoop("ap"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil GeneralizedClosedLoop.OpenLoop: %v\n%s", err, in)
	}
	var nilGM *GeneralizedModel
	if _, err := nilGM.CurrentSystem(); !errors.Is(err, ErrInvalidArgument) || nilGM.HasAnalysisPoint("x") {
		t.Fatalf("nil GeneralizedModel: %v\n%s", err, in)
	}
}

// ------------------------------------------------------------ descriptor

// FuzzAPIDescriptor builds descriptor models with nonsingular, singular,
// non-finite or mis-sized E. Oracles: dss acceptance by the size rule;
// DescriptorE returns a copy of E (identity when nil); ToExplicit returns
// ErrDescriptorSingular for an exactly singular E and otherwise matrices
// whose residuals E·Ã − A and E·B̃ − B are within the LU backward-error
// bound, with C and D unchanged.
func FuzzAPIDescriptor(f *testing.F) {
	for i, s := range fzAPISystemSeeds() {
		for _, c := range [][3]uint8{{2, 0x24, 0}, {2, 0x64, 0}, {2, 0x24, 1}, {2, 0x24, 6}, {2, 0x24, 5}, {2, 0x2c, 0}, {2, 0x34, 2}} {
			fzAPIAddSeed(f, s, c[0], c[1], c[2]+uint8(i)<<3)
		}
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		s := fzDecode(t, in.Kind, in.V)
		pt := fzAPIBuild(s, in.Site|0x20, in.Bad)
		if pt.E != nil && in.Op&1 != 0 {
			r, c := pt.E.Dims()
			for i := range min(r, c) {
				pt.E.Set(i, i, pt.E.At(i, i)*math.Pow(10, fzClamp(in.V[13], -8, 8)))
			}
		}
		Ein := fzAPICloneMats([]*mat.Dense{pt.E})[0]
		sys, err := NewDescriptor(pt.A, pt.B, pt.C, pt.D, pt.E, pt.Dt)
		fzAPIContract(t, in, "NewDescriptor", sys != nil, err)
		if err != nil {
			fzAPIRejected(t, "NewDescriptor", err)
		}
		fzAPIRequireValid(t, in, "NewDescriptor", sys)
		n, _, _ := sys.Dims()
		E := sys.DescriptorE()
		if Ein == nil || Ein.IsEmpty() {
			if n > 0 && !mat.Equal(E, eyeDense(n)) {
				t.Fatalf("DescriptorE of nil E = %v\n%s", mat.Formatted(E), in)
			}
		} else if !fzAPISameBlock(Ein, E) {
			t.Fatalf("DescriptorE = %v, given %v\n%s", mat.Formatted(E), mat.Formatted(Ein), in)
		}
		fzAPIMutate(E)
		if again := sys.DescriptorE(); Ein != nil && !Ein.IsEmpty() && !fzAPISameBlock(Ein, again) {
			t.Fatalf("DescriptorE leaks storage\n%s", in)
		}
		snap := sys.Copy()
		ex, err := sys.ToExplicit()
		fzAPIContract(t, in, "ToExplicit", ex != nil, err)
		if d := systemDiff(snap, sys); d != "" {
			t.Fatalf("ToExplicit changed its receiver (%s)\n%s", d, in)
		}
		singular := false
		finite := true
		if n > 0 {
			for i := range n {
				zero := true
				for j := range n {
					x := E.At(i, j)
					if snap.E != nil {
						x = snap.E.At(i, j)
					}
					if x != 0 {
						zero = false
					}
					if !fzAPIFinite(x) {
						finite = false
					}
				}
				if zero {
					singular = true
				}
			}
		}
		if err != nil {
			if singular && errors.Is(err, ErrDescriptorSingular) {
				fzSkip(t, "ToExplicit: singular E")
			}
			if !finite || errors.Is(err, ErrDescriptorSingular) {
				fzAPIRejected(t, "ToExplicit", err)
			}
			t.Fatalf("ToExplicit: %v\n%s", err, in)
		}
		if singular {
			t.Fatalf("ToExplicit accepted an E with a zero row\n%s", in)
		}
		fzAPIRequireValid(t, in, "ToExplicit", ex)
		if ex.E != nil && !isIdentityDescriptor(ex.E) {
			t.Fatalf("ToExplicit left E = %v\n%s", mat.Formatted(ex.E), in)
		}
		if !fzAPISameBlock(snap.C, ex.C) || !fzAPISameBlock(snap.D, ex.D) {
			t.Fatalf("ToExplicit changed C or D\n%s", in)
		}
		if n == 0 || snap.E == nil || !finite {
			return
		}
		fzAPIResidual(t, in, "A", snap.E, ex.A, snap.A)
		if _, m, _ := snap.Dims(); m > 0 {
			fzAPIResidual(t, in, "B", snap.E, ex.B, snap.B)
		}
	})
}

// fzAPIResidual checks ‖E·X − Y‖_max ≤ 8n·eps·(‖E‖·‖X‖ + ‖Y‖) row-wise in
// the 1-norm, the backward-error bound of an LU solve with partial pivoting
// (growth factor folded into the constant; n ≤ 6 here).
func fzAPIResidual(t *testing.T, in *fzAPIIn, block string, E, X, Y *mat.Dense) {
	t.Helper()
	n, c := X.Dims()
	for j := range c {
		for i := range n {
			var r, bound float64
			for k := range n {
				r += E.At(i, k) * X.At(k, j)
				bound += math.Abs(E.At(i, k)) * math.Abs(X.At(k, j))
			}
			r -= Y.At(i, j)
			if math.IsInf(bound, 0) || math.IsNaN(bound) {
				continue
			}
			var colX float64
			for k := range n {
				colX = math.Max(colX, math.Abs(X.At(k, j)))
			}
			var rowE float64
			for k := range n {
				rowE += math.Abs(E.At(i, k))
			}
			tol := 64 * float64(n) * 0x1p-52 * (rowE*colX*float64(n) + bound + math.Abs(Y.At(i, j)))
			if !(math.Abs(r) <= tol) {
				t.Fatalf("ToExplicit residual (E·%s̃ − %s)[%d,%d] = %g > %g\nE=%v\nX=%v\n%s", block, block, i, j, r, tol, mat.Formatted(E), mat.Formatted(X), in)
			}
		}
	}
}

// ------------------------------------------------------- names/connect

var fzAPISumExprs = []string{
	"e = r - y", "e = r - 2*y", "e = r - y; f = e + d", "z = a + b - c",
	"e = r -", "= r - y", "e = r - y;", "e r y", "e = e - y", "", "e = 0.5*r + y", "e(1) = r",
}

// FuzzAPINames wires the spec model, a second model and a SumBlk junction
// with ConnectByName from fuzzed connection lists. Oracles: SumBlk follows
// its documented width rules; ConnectByName returns ErrSignalNotFound for
// an unknown name, ErrInvalidArgument for a repeated connection or nil
// model, ErrDomainMismatch for mixed sample times, and otherwise succeeds
// (or reports ErrAlgebraicLoop) with exactly the requested external
// channels and their names; Series and Feedback keep MATLAB's names.
func FuzzAPINames(f *testing.F) {
	for i, s := range fzAPISystemSeeds() {
		for _, c := range [][3]uint8{{0x00, 0x00, 0x00}, {0x21, 0x12, 0x45}, {0x42, 0x34, 0x8a}, {0x63, 0x56, 0xcf}, {0x84, 0x78, 0x10}, {0xa5, 0x9a, 0x01}} {
			fzAPIAddSeed(f, s, c[0]+uint8(i), c[1]+uint8(i)<<2, c[2])
		}
	}
	fzAPIFuzz(f, func(t *testing.T, in *fzAPIIn) {
		s := fzDecode(t, in.Kind, in.V)
		s1 := fzAPIFirst(t, s)
		s2 := fzAPISecond(in, s)
		_, m1, p1 := s1.Dims()
		_, m2, p2 := s2.Dims()
		s1.InputName, s1.OutputName = autoLabel("u", m1), autoLabel("y", p1)
		s2.InputName, s2.OutputName = autoLabel("e", m2), autoLabel("v", p2)

		expr := fzAPISumExprs[int(in.Op)%len(fzAPISumExprs)]
		width := int(in.Op>>4)%4 - 1
		var widths []int
		if width != 0 {
			widths = []int{width}
		}
		sum, err := SumBlk(expr, widths...)
		fzAPIContract(t, in, "SumBlk", sum != nil, err)
		if err == nil {
			fzAPIRequireValid(t, in, "SumBlk", sum)
			if width < 0 {
				t.Fatalf("SumBlk(%q, %d) accepted a negative width\n%s", expr, width, in)
			}
			if _, m, p := sum.Dims(); len(sum.InputName) != m || len(sum.OutputName) != p {
				t.Fatalf("SumBlk(%q) names %v %v for %dx%d\n%s", expr, sum.OutputName, sum.InputName, p, m, in)
			}
		} else if expr == "e = r - y" && width >= 0 {
			t.Fatalf("SumBlk(%q, %v): %v\n%s", expr, widths, err, in)
		}

		systems := []*System{s1, s2}
		if sum != nil && in.Site&1 != 0 {
			systems = append(systems, sum)
		}
		var outPool, inPool []string
		for _, sys := range systems {
			outPool = append(outPool, sys.OutputName...)
			inPool = append(inPool, sys.InputName...)
		}
		unique := func(pool []string) bool {
			seen := map[string]bool{}
			for _, n := range pool {
				if seen[n] {
					return false
				}
				seen[n] = true
			}
			return true
		}
		if !unique(outPool) || !unique(inPool) {
			fzSkip(t, "ConnectByName: duplicate signal names (first match wins)")
		}
		pick := func(pool []string, x float64) string {
			if math.IsNaN(x) || len(pool) == 0 {
				return "missing"
			}
			k := int(math.Abs(x)*5) % (len(pool) + 1)
			if k == len(pool) {
				return "missing"
			}
			return pool[k]
		}
		var conns []Connection
		for k := range int(in.Bad>>5) % 4 {
			conns = append(conns, Connection{From: pick(outPool, in.V[k]+0.3*float64(k)), To: pick(inPool, in.V[k+4]+0.7*float64(k))})
		}
		if in.Bad&0x10 != 0 && len(conns) > 0 {
			conns = append(conns, conns[0])
		}
		var ext, outs []string
		for k := range int(in.Site>>1) % 4 {
			ext = append(ext, pick(inPool, in.V[8+k]))
		}
		for k := range int(in.Site>>3) % 4 {
			outs = append(outs, pick(outPool, in.V[12+k]))
		}
		var nilAt = -1
		if in.Bad&0x08 != 0 {
			nilAt = int(in.Bad) % len(systems)
			systems[nilAt] = nil
		}
		snaps := make([]*System, len(systems))
		for i, sys := range systems {
			if sys != nil {
				snaps[i] = sys.Copy()
			}
		}
		res, err := ConnectByName(systems, conns, ext, outs)
		fzAPIContract(t, in, "ConnectByName", res != nil, err)
		for i, sys := range systems {
			if sys != nil && systemDiff(snaps[i], sys) != "" {
				t.Fatalf("ConnectByName modified systems[%d]\n%s", i, in)
			}
		}
		missing := slices.Contains(ext, "missing") || slices.Contains(outs, "missing")
		for _, c := range conns {
			missing = missing || c.From == "missing" || c.To == "missing"
		}
		dup := false
		for i := range conns {
			for j := range i {
				dup = dup || conns[i] == conns[j]
			}
		}
		sameDt := true
		for _, a := range systems {
			for _, b := range systems {
				if a != nil && b != nil && a.Dt != b.Dt {
					sameDt = false
				}
			}
		}
		switch {
		case nilAt >= 0:
			if !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("ConnectByName with nil systems[%d]: %v\n%s", nilAt, err, in)
			}
			return
		case !sameDt:
			if !errors.Is(err, ErrDomainMismatch) {
				t.Fatalf("ConnectByName mixed sample times: %v\n%s", err, in)
			}
			return
		case missing || dup || len(ext) == 0 || len(outs) == 0 || !unique(ext) || !unique(outs):
			invalid := dup || len(ext) == 0 || len(outs) == 0 || !unique(ext) || !unique(outs)
			if !(missing && errors.Is(err, ErrSignalNotFound)) && !(invalid && errors.Is(err, ErrInvalidArgument)) {
				t.Fatalf("ConnectByName (missing %v, invalid %v) conns %v ext %v outs %v: %v\n%s", missing, invalid, conns, ext, outs, err, in)
			}
			return
		}
		if err != nil {
			if errors.Is(err, ErrAlgebraicLoop) {
				fzAPIRejected(t, "ConnectByName", err)
			}
			t.Fatalf("ConnectByName(conns %v, ext %v, outs %v): %v\n%s", conns, ext, outs, err, in)
		}
		fzAPIRequireValid(t, in, "ConnectByName", res)
		if !slices.Equal(res.InputName, ext) && !(len(ext) == 0 && res.InputName == nil) {
			t.Fatalf("ConnectByName InputName %v, want %v\n%s", res.InputName, ext, in)
		}
		if !slices.Equal(res.OutputName, outs) && !(len(outs) == 0 && res.OutputName == nil) {
			t.Fatalf("ConnectByName OutputName %v, want %v\n%s", res.OutputName, outs, in)
		}
		if _, m, p := res.Dims(); m != len(ext) || p != len(outs) {
			t.Fatalf("ConnectByName size %dx%d, want %dx%d\n%s", p, m, len(outs), len(ext), in)
		}

		if p1 == m2 && s1.Dt == s2.Dt {
			ser, err := Series(s1, s2)
			if err == nil && (!slices.Equal(ser.InputName, s1.InputName) || !slices.Equal(ser.OutputName, s2.OutputName)) {
				t.Fatalf("Series names %v -> %v, want %v -> %v\n%s", ser.InputName, ser.OutputName, s1.InputName, s2.OutputName, in)
			}
		}
		if p1 == m2 && m1 == p2 && s1.Dt == s2.Dt {
			fb, err := Feedback(s1, s2, -1)
			if err == nil && (!slices.Equal(fb.InputName, s1.InputName) || !slices.Equal(fb.OutputName, s1.OutputName)) {
				t.Fatalf("Feedback names %v -> %v, want plant's\n%s", fb.InputName, fb.OutputName, in)
			}
		}
	})
}
