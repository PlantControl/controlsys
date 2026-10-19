package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzExt is the decoded model with exact structural defects appended and the
// states permuted (exact): Ku states with zero B rows that the core never
// drives (uncontrollable, structurally unreachable) and Ko states with zero
// C columns that never drive the core or the Ku states (unobservable), so
// its transfer function is exactly the core's. The delay fields are spare in
// delay-free targets:
//
//	v[12] Ku = ⌊clamp(|v|,0,2.5)⌋;  v[14] Ko likewise
//	v[13] defect pole scale ±10^clamp(|v|-3,-3,3), sign of v (unstable if v < 0)
//	v[15] permutation keys cos(3.7·v·(i+1) + i)
type fzExt struct {
	N, M, P    int
	Ku, Ko     int
	A, B, C, D []float64
	Kind       []int
}

const (
	fzCore = iota
	fzUnctrb
	fzUnobsv
)

func fzExtend(s *fzSpec) *fzExt {
	ku := int(fzClamp(math.Abs(s.Vals[12]), 0, 2.5))
	ko := int(fzClamp(math.Abs(s.Vals[14]), 0, 2.5))
	n0, m, p := s.N, s.M, s.P
	N := n0 + ku + ko
	sc := math.Pow(10, fzClamp(math.Abs(s.Vals[13])-3, -3, 3))
	if s.Vals[13] < 0 {
		sc = -sc
	}
	ph := fzClamp(s.Vals[11], -1e3, 1e3)
	tex := func(i, j int) float64 { return 0.5 * math.Cos(ph+1.7*float64(i+1)+0.9*float64(j+1)) }
	pole := func(k int) float64 {
		if s.Discrete {
			return math.Tanh(sc) * float64(k+1) / 2
		}
		return -sc * float64(k+1)
	}
	kind := make([]int, N)
	A := make([]float64, N*N)
	B := make([]float64, N*m)
	C := make([]float64, p*N)
	cpl := math.Max(1, s.NormJ)
	for i := range N {
		switch {
		case i < n0:
			kind[i] = fzCore
		case i < n0+ku:
			kind[i] = fzUnctrb
		default:
			kind[i] = fzUnobsv
		}
	}
	for i := range N {
		for j := range N {
			switch {
			case i < n0 && j < n0:
				A[i*N+j] = s.A[i*n0+j]
			case i == j:
				A[i*N+j] = pole(i - n0)
			case kind[i] == fzCore && kind[j] == fzUnctrb,
				kind[i] == fzUnobsv && kind[j] != fzUnobsv,
				kind[i] == kind[j] && j == i+1:
				A[i*N+j] = cpl * tex(i, j)
			}
		}
		for k := range m {
			switch kind[i] {
			case fzCore:
				B[i*m+k] = s.B[i*m+k]
			case fzUnobsv:
				B[i*m+k] = tex(k, i)
			}
		}
		for k := range p {
			switch kind[i] {
			case fzCore:
				C[k*N+i] = s.C[k*n0+i]
			case fzUnctrb:
				C[k*N+i] = tex(i, k+3)
			}
		}
	}
	perm := make([]int, N)
	for i := range perm {
		perm[i] = i
	}
	key := func(i int) float64 { return math.Cos(3.7*fzClamp(s.Vals[15], -1e3, 1e3)*float64(i+1) + float64(i)) }
	sort.SliceStable(perm, func(a, b int) bool { return key(perm[a]) < key(perm[b]) })
	e := &fzExt{N: N, M: m, P: p, Ku: ku, Ko: ko, D: slices.Clone(s.D),
		A: make([]float64, N*N), B: make([]float64, N*m), C: make([]float64, p*N), Kind: make([]int, N)}
	for i, pi := range perm {
		e.Kind[i] = kind[pi]
		for j, pj := range perm {
			e.A[i*N+j] = A[pi*N+pj]
		}
		copy(e.B[i*m:(i+1)*m], B[pi*m:(pi+1)*m])
		for k := range p {
			e.C[k*N+i] = C[k*N+pi]
		}
	}
	return e
}

func (e *fzExt) System(t *testing.T, s *fzSpec) *System {
	t.Helper()
	sys, err := NewFromSlices(e.N, e.M, e.P, e.A, e.B, e.C, e.D, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v\n%s", err, s)
	}
	return sys
}

// fzTestPoints returns evaluation points off the stability boundary at
// radii 1 and the smallest, geometric-mean and largest intended pole
// magnitude, alternating r·e^{1.2i} and 1.05·r·i (continuous) or radii
// 1.25 and 0.8 at angle r·dt + 0.3 (discrete).
func fzTestPoints(s *fzSpec) []complex128 {
	lo, hi := math.Inf(1), 0.0
	for _, p := range s.S {
		if a := cmplx.Abs(p); a > 0 {
			lo, hi = math.Min(lo, a), math.Max(hi, a)
		}
	}
	rs := []float64{1}
	if hi > 0 {
		rs = append(rs, lo, math.Sqrt(lo*hi), hi)
	}
	var out []complex128
	for i, r := range rs {
		switch {
		case s.Discrete:
			w := math.Mod(r*s.Dt, math.Pi)
			out = append(out, cmplx.Rect(1.25-0.45*float64(i%2), w+0.3))
		case i%2 == 0:
			out = append(out, cmplx.Rect(r, 1.2))
		default:
			out = append(out, complex(0, 1.05*r))
		}
	}
	return out
}

func fzNormsFrob(a, b, c, d []float64) (na, nb, nc, nd float64) {
	return fzFrob(a), fzFrob(b), fzFrob(c), fzFrob(d)
}

// fzRespBound bounds the response change at x caused by a backward
// perturbation of relative size c·eps of (A, B, C, D), first order in
// ‖(xI−A)⁻¹‖. ok is false when xI−A is too ill-conditioned for the bound to
// be meaningful.
func fzRespBound(n int, a, b, c, d []float64, x complex128, factor float64) (bound float64, ok bool) {
	const eps = 0x1p-52
	na, nb, nc, nd := fzNormsFrob(a, b, c, d)
	if n == 0 {
		return factor * eps * nd, true
	}
	M := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			M[i*n+j] = complex(-a[i*n+j], 0)
		}
		M[i*n+i] += x
	}
	sv := fzSingularValues(M, n, n)
	if sv == nil || sv[n-1] == 0 {
		return 0, false
	}
	R := 1 / sv[n-1]
	if (na+cmplx.Abs(x))*R > 1e8 {
		return 0, false
	}
	return factor * eps * (nc*R*nb*(1+(na+cmplx.Abs(x))*R) + nc*R*nb + nd), true
}

// fzRespCheck compares the exact responses of two float64 models at the
// test points; refN etc. describe the reference.
func fzRespCheck(t *testing.T, s *fzSpec, op string, ref, got [4][]float64, nRef, nGot int, factor float64) {
	t.Helper()
	m, p := s.M, s.P
	for _, x := range fzTestPoints(s) {
		bound, ok := fzRespBound(nRef, ref[0], ref[1], ref[2], ref[3], x, factor)
		if !ok {
			continue
		}
		if nGot > 0 {
			b2, ok := fzRespBound(nGot, got[0], got[1], got[2], got[3], x, factor)
			if !ok {
				continue
			}
			bound += b2
		}
		g0, ok0 := fzExactResponse(nRef, m, p, ref[0], ref[1], ref[2], ref[3], x)
		g1, ok1 := fzExactResponse(nGot, m, p, got[0], got[1], got[2], got[3], x)
		if !ok0 || !ok1 {
			continue
		}
		for k := range g0 {
			if d := cmplx.Abs(g0[k] - g1[k]); d > bound {
				t.Fatalf("%s: response at %v entry %d is %v, want %v (|Δ| %.3g > bound %.3g)\n%s", op, x, k, g1[k], g0[k], d, bound, s)
			}
		}
	}
}

func fzSysMats(sys *System) [4][]float64 {
	n, m, p := sys.Dims()
	raw := func(x *mat.Dense, r, c int) []float64 {
		if r == 0 || c == 0 {
			return nil
		}
		return slices.Clone(mat.DenseCopyOf(x).RawMatrix().Data)
	}
	return [4][]float64{raw(sys.A, n, n), raw(sys.B, n, m), raw(sys.C, p, n), raw(sys.D, p, m)}
}

// FuzzSminreal: on the extended model (fzExtend), Sminreal removes exactly
// the injected disconnected states and keeps the core states, in order and
// bit for bit (the core's B and C have no zero entries).
func FuzzSminreal(f *testing.F) {
	for _, s := range fzStructSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		e := fzExtend(s)
		got, err := Sminreal(e.System(t, s))
		fzContract(t, s, "Sminreal", got != nil, err)
		fzErr(t, s, "Sminreal", err)
		var keep []int
		for i, k := range e.Kind {
			if k == fzCore {
				keep = append(keep, i)
			}
		}
		n, m, p := got.Dims()
		if n != len(keep) || m != e.M || p != e.P {
			t.Fatalf("Sminreal dims %d,%d,%d, want %d,%d,%d (ku=%d ko=%d)\n%s", n, m, p, len(keep), e.M, e.P, e.Ku, e.Ko, s)
		}
		g := fzSysMats(got)
		for ii, i := range keep {
			for jj, j := range keep {
				if g[0][ii*n+jj] != e.A[i*e.N+j] {
					t.Fatalf("Sminreal A[%d,%d] = %v, want %v\n%s", ii, jj, g[0][ii*n+jj], e.A[i*e.N+j], s)
				}
			}
			for k := range m {
				if g[1][ii*m+k] != e.B[i*m+k] {
					t.Fatalf("Sminreal B[%d,%d] changed\n%s", ii, k, s)
				}
			}
			for k := range p {
				if g[2][k*n+ii] != e.C[k*e.N+i] {
					t.Fatalf("Sminreal C[%d,%d] changed\n%s", k, ii, s)
				}
			}
		}
		if !slices.Equal(g[3], e.D) || got.Dt != s.Dt {
			t.Fatalf("Sminreal D/Dt changed: %v dt=%v\n%s", g[3], got.Dt, s)
		}
	})
}

// FuzzReduce: on the extended model Reduce (MATLAB minreal) removes every
// injected defect state its mode targets when fzStairClear says the count
// is decidable (for ReduceAll also on the library's own controllable stage,
// whose rounding feeds the observability stage; never with Equalize),
// keeps D and Dt, and its exact
// response matches the core's within a backward-error bound at points
// where the resolvent is well conditioned.
//
//	v[12] < 0: Equalize (response not checked: the bound needs the scaling)
//	v[14] < 0: ReduceUncontrollable if frac(|v|·4) < 0.5 else ReduceUnobservable
func FuzzReduce(f *testing.F) {
	for _, s := range fzStructSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		e := fzExtend(s)
		opts := &ReduceOpts{Equalize: s.Vals[12] < 0}
		want := s.N
		if s.Vals[14] < 0 {
			if _, fr := math.Modf(math.Abs(s.Vals[14]) * 4); fr < 0.5 {
				opts.Mode, want = ReduceUncontrollable, e.N-e.Ku
			} else {
				opts.Mode, want = ReduceUnobservable, e.N-e.Ko
			}
		}
		res, err := e.System(t, s).Reduce(opts)
		fzContract(t, s, "Reduce", res != nil, err)
		fzErr(t, s, "Reduce", err)
		n, m, p := res.Sys.Dims()
		if n != res.Order || m != e.M || p != e.P {
			t.Fatalf("Reduce dims %d,%d,%d order %d\n%s", n, m, p, res.Order, s)
		}
		clear := !opts.Equalize
		switch opts.Mode {
		case ReduceUncontrollable:
			clear = clear && fzStairClear(e.A, e.B, e.N, e.M, e.N-e.Ku)
		case ReduceUnobservable:
			clear = clear && fzStairClear(fzTranspose(e.A, e.N, e.N), fzTranspose(e.C, e.P, e.N), e.N, e.P, e.N-e.Ko)
		default:
			clear = clear && fzStairClear(e.A, e.B, e.N, e.M, e.N-e.Ku)
			if clear {
				st, err := e.System(t, s).Reduce(&ReduceOpts{Mode: ReduceUncontrollable})
				fzErr(t, s, "Reduce", err)
				k := st.Order
				g1 := fzSysMats(st.Sys)
				clear = k == e.N-e.Ku && fzStairClear(fzTranspose(g1[0], k, k), fzTranspose(g1[2], e.P, k), k, e.P, s.N)
			}
		}
		if clear && n > want {
			t.Fatalf("Reduce(mode %d) kept %d states, at most %d expected (n=%d ku=%d ko=%d)\n%s", opts.Mode, n, want, s.N, e.Ku, e.Ko, s)
		}
		g := fzSysMats(res.Sys)
		if !slices.Equal(g[3], e.D) || res.Sys.Dt != s.Dt {
			t.Fatalf("Reduce D/Dt changed\n%s", s)
		}
		for _, x := range slices.Concat(g[0], g[1], g[2]) {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("Reduce returned non-finite entries\n%s", s)
			}
		}
		if opts.Equalize {
			return
		}
		ext := [4][]float64{e.A, e.B, e.C, e.D}
		fzRespCheck(t, s, "Reduce", ext, g, e.N, n, 1e4*float64(e.N+m+p))
	})
}

// fzStairClear reports whether a reference SVD staircase on (A, B), with
// the library's default tolerance n²·eps applied per singular value against
// ‖A‖_F, finds exactly want controllable states with every rank decision
// clear by 10³ either way. Exact zero blocks reappear as rounding noise
// amplified by the staircase's conditioning, so a defect count is only
// decidable when this gap exists.
func fzStairClear(a, b []float64, n, m, want int) bool {
	const eps = 0x1p-52
	if n == 0 {
		return want == 0
	}
	A := mat.NewDense(n, n, slices.Clone(a))
	floor := float64(n*n) * eps * mat.Norm(A, 2)
	block := mat.NewDense(n, m, slices.Clone(b))
	nc := 0
	for nc < n {
		var svd mat.SVD
		if !svd.Factorize(block, mat.SVDFull) {
			return false
		}
		rank := 0
		for _, v := range svd.Values(nil) {
			switch {
			case v > 1e3*floor:
				rank++
			case v > 1e-3*floor:
				return false
			}
		}
		if rank == 0 {
			break
		}
		var U mat.Dense
		svd.UTo(&U)
		Q := mat.NewDense(n, n, nil)
		for i := range nc {
			Q.Set(i, i, 1)
		}
		Q.Slice(nc, n, nc, n).(*mat.Dense).Copy(&U)
		var t1, t2 mat.Dense
		t1.Mul(Q.T(), A)
		t2.Mul(&t1, Q)
		A = &t2
		nc += rank
		if nc < n {
			block = mat.DenseCopyOf(A.Slice(nc, n, nc-rank, nc))
		}
	}
	return nc == want
}

func fzTranspose(a []float64, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}

// FuzzCtrbF: CtrbF/ObsvF on the extended model return an orthogonal T with
// Abar = T·A·Tᵀ, Bbar = T·B, Cbar = C·Tᵀ (residuals at rounding level), the
// documented zero blocks, a non-increasing staircase K with K[0] ≤ m (p),
// and, when fzStairClear decides it, at least the injected Ku (Ko) states in
// the uncontrollable (unobservable) part.
func FuzzCtrbF(f *testing.F) {
	for _, s := range fzStructSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		e := fzExtend(s)
		if e.N == 0 {
			fzSkip(t, "no states")
		}
		N, m, p := e.N, e.M, e.P
		A, B, C := mat.NewDense(N, N, e.A), mat.NewDense(N, m, e.B), mat.NewDense(p, N, e.C)
		cf, err := CtrbF(A, B, C, 0)
		fzContract(t, s, "CtrbF", cf != nil, err)
		fzErr(t, s, "CtrbF", err)
		maxRank := -1
		if fzStairClear(e.A, e.B, N, m, N-e.Ku) {
			maxRank = N - e.Ku
		}
		fzCheckStaircase(t, s, "CtrbF", cf, A, B, C, m, maxRank, false)
		of, err := ObsvF(A, B, C, 0)
		fzContract(t, s, "ObsvF", of != nil, err)
		fzErr(t, s, "ObsvF", err)
		maxRank = -1
		if fzStairClear(fzTranspose(e.A, N, N), fzTranspose(e.C, p, N), N, p, N-e.Ko) {
			maxRank = N - e.Ko
		}
		fzCheckStaircase(t, s, "ObsvF", of, A, B, C, p, maxRank, true)
	})
}

func fzCheckStaircase(t *testing.T, s *fzSpec, op string, sf *StaircaseForm, A, B, C *mat.Dense, width, maxRank int, obs bool) {
	t.Helper()
	const eps = 0x1p-52
	n, _ := A.Dims()
	tol := 1e3 * float64(n) * eps
	nk := 0
	for i, k := range sf.K {
		if k <= 0 || i == 0 && k > width || i > 0 && k > sf.K[i-1] {
			t.Fatalf("%s: staircase K = %v not a non-increasing positive sequence with K[0] ≤ %d\n%s", op, sf.K, width, s)
		}
		nk += k
	}
	if maxRank >= 0 && nk > maxRank {
		t.Fatalf("%s: sum(K) = %d (K=%v), but at most %d states are structurally reachable\n%s", op, nk, sf.K, maxRank, s)
	}
	var tt mat.Dense
	tt.Mul(sf.T, sf.T.T())
	for i := range n {
		for j := range n {
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(tt.At(i, j)-want) > tol {
				t.Fatalf("%s: T·Tᵀ[%d,%d] = %v, T not orthogonal\n%s", op, i, j, tt.At(i, j), s)
			}
		}
	}
	var ta, tat, tb, ct mat.Dense
	ta.Mul(sf.T, A)
	tat.Mul(&ta, sf.T.T())
	tb.Mul(sf.T, B)
	ct.Mul(C, sf.T.T())
	resid := func(name string, got, want *mat.Dense, scale float64) {
		var d mat.Dense
		d.Sub(got, want)
		if r := mat.Norm(&d, 2); r > tol*scale {
			t.Fatalf("%s: ‖%s − transformed‖ = %.3g > %.3g\n%s", op, name, r, tol*scale, s)
		}
	}
	nA, nB, nC := mat.Norm(A, 2), mat.Norm(B, 2), mat.Norm(C, 2)
	resid("Abar", sf.A, &tat, nA)
	resid("Bbar", sf.B, &tb, nB)
	resid("Cbar", sf.C, &ct, nC)
	nu := n - nk
	for i := range nu {
		for j := nu; j < n; j++ {
			x := sf.A.At(i, j)
			if obs {
				x = sf.A.At(j, i)
			}
			if math.Abs(x) > tol*nA {
				t.Fatalf("%s: coupling entry %v in the zero block (nu=%d)\n%s", op, x, nu, s)
			}
		}
		if obs {
			for k := range sf.C.RawMatrix().Rows {
				if x := sf.C.At(k, i); math.Abs(x) > tol*(nC+nA) {
					t.Fatalf("%s: Cbar[%d,%d] = %v in the unobservable block\n%s", op, k, i, x, s)
				}
			}
			continue
		}
		for k := range sf.B.RawMatrix().Cols {
			if x := sf.B.At(i, k); math.Abs(x) > tol*(nB+nA) {
				t.Fatalf("%s: Bbar[%d,%d] = %v in the uncontrollable block\n%s", op, i, k, x, s)
			}
		}
	}
}

// FuzzCtrbObsv: Ctrb and Obsv match the exact rational Krylov blocks AᵏB
// and CAᵏ within the standard |A|ᵏ|B| rounding bound.
func FuzzCtrbObsv(f *testing.F) {
	for _, s := range fzStructSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m, p := s.N, s.M, s.P
		if n == 0 {
			fzSkip(t, "no states")
		}
		A := mat.NewDense(n, n, s.A)
		co, err := Ctrb(A, mat.NewDense(n, m, s.B))
		fzContract(t, s, "Ctrb", co != nil, err)
		fzErr(t, s, "Ctrb", err)
		fzCheckKrylov(t, s, "Ctrb", co, s.A, s.B, n, m)
		AT := mat.DenseCopyOf(A.T()).RawMatrix().Data
		CT := mat.DenseCopyOf(mat.NewDense(p, n, s.C).T()).RawMatrix().Data
		ob, err := Obsv(A, mat.NewDense(p, n, s.C))
		fzContract(t, s, "Obsv", ob != nil, err)
		fzErr(t, s, "Obsv", err)
		var obT mat.Dense
		obT.CloneFrom(ob.T())
		fzCheckKrylov(t, s, "Obsv", &obT, AT, CT, n, p)
	})
}

// fzCheckKrylov checks got (n×n·m, block k = AᵏB; Obsv is passed
// transposed with Aᵀ, Cᵀ) against exact rational products.
func fzCheckKrylov(t *testing.T, s *fzSpec, op string, got *mat.Dense, a, b []float64, n, m int) {
	t.Helper()
	const eps = 0x1p-52
	r, c := got.Dims()
	if r != n || c != n*m {
		t.Fatalf("%s: dims %d×%d, want %d×%d\n%s", op, r, c, n, n*m, s)
	}
	ar := fzRatPoly(a)
	cur := fzRatPoly(b)
	absA := make([]float64, n*n)
	for i, x := range a {
		absA[i] = math.Abs(x)
	}
	bnd := make([]float64, n*m)
	for i, x := range b {
		bnd[i] = math.Abs(x)
	}
	for k := range n {
		gam := 4 * float64((k+1)*n) * eps
		for i := range n {
			for j := range m {
				col := k*m + j
				want, _ := cur[i*m+j].Float64()
				g := got.At(i, col)
				if fzBadFloat(want) || math.IsInf(bnd[i*m+j], 0) {
					fzSkip(t, "Krylov entry out of float64 range")
				}
				if d := math.Abs(g - want); d > gam*bnd[i*m+j]+0x1p-1000 {
					t.Fatalf("%s: block %d entry [%d,%d] = %v, exact %v (|Δ| %.3g > %.3g)\n%s", op, k, i, j, g, want, d, gam*bnd[i*m+j], s)
				}
			}
		}
		next := make([]*big.Rat, n*m)
		nb := make([]float64, n*m)
		for i := range n {
			for j := range m {
				v := new(big.Rat)
				var tmp big.Rat
				for l := range n {
					v.Add(v, tmp.Mul(ar[i*n+l], cur[l*m+j]))
					nb[i*m+j] += absA[i*n+l] * bnd[l*m+j]
				}
				next[i*m+j] = v
				nb[i*m+j] *= 1 + 4*float64(n)*eps
			}
		}
		cur, bnd = next, nb
	}
}

// fzStructSeeds: delay fields encode the fzExtend defects.
func fzStructSeeds() []fzSeed {
	off := fzSeed{In: 0, Out: 0, IO: 0, Tau: 0.3, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0], s.In, s.IO = 2, 2.5, 1, 1 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0, 1}, [3]float64{1.5, -0.7}
			s.Shear, s.Rot, s.In, s.Out, s.IO, s.Tau = -1, 0.8, 2, 2.2, 1, -1.3
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P = 3, 1, 2
			s.Mag, s.Damp = [3]float64{0.3, -0.2}, [3]float64{1.2, 1}
			s.In, s.Out, s.IO, s.Tau = 1, 3.5, 2, 2.4
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 6, 3, 1
			s.Mag, s.Damp = [3]float64{-1, 0.5, 1.5}, [3]float64{2, 3, 0.8}
			s.Shear, s.Rot, s.Phase, s.In, s.IO = -0.5, 1.4, 0.6, -2, -1.1
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 1, 1
			s.Mag, s.Damp = [3]float64{0, 0, 0.7}, [3]float64{1.5, 1.5, 1}
			s.In, s.IO = 1, -2.3
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal, s.In, s.IO = 0, 2, 2, 3, 2, 1 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.2)
			s.N, s.M, s.P = 6, 2, 3
			s.Mag, s.Damp = [3]float64{0.5, -0.3, 0.2}, [3]float64{-2, 3, 1}
			s.Shear, s.Rot, s.In, s.Out, s.IO, s.Tau = -0.8, 1.7, -1, -4.5, -1.7, 5
		}),
	}
}
