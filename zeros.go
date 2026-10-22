package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// ZerosResult holds the invariant zeros and the normal rank of the transfer
// matrix.
type ZerosResult struct {
	// Zeros are the finite invariant zeros; empty when there are none.
	Zeros []complex128
	// Rank is the normal rank of the system pencil [A-sE B; C D] minus n,
	// which equals the normal rank of the transfer matrix (rank(D) for a
	// static gain).
	Rank int
}

// Zeros returns the finite invariant zeros: the points s where the system
// pencil [A-sE B; C D] loses rank below its normal rank (MATLAB tzero). This
// holds for SISO models too, so zeros cancelled by uncontrollable or
// unobservable poles are kept, matching MATLAB zero on ss models; they coincide
// with the transfer function numerator roots only for minimal realizations. As
// in MATLAB zero, internal delays are set to zero (zero-order Padé) and an
// ill-posed zero-delay loop returns ErrAlgebraicLoop; input/output delays
// contribute no finite zeros and are ignored. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.zero.html and
// https://www.mathworks.com/help/control/ref/dynamicsystem.tzero.html.
func (sys *System) Zeros() ([]complex128, error) {
	res, err := sys.zerosDetail("Zeros")
	if err != nil {
		return nil, err
	}
	return res.Zeros, nil
}

// ZerosDetail returns the invariant zeros together with the normal rank of
// the transfer matrix, as MATLAB [z, nrank] = tzero(sys). A static gain has no
// zeros and rank rank(D); a model with no inputs or outputs has rank 0.
// Non-finite model data returns ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.tzero.html.
func (sys *System) ZerosDetail() (*ZerosResult, error) {
	return sys.zerosDetail("ZerosDetail")
}

func (sys *System) zerosDetail(op string) (*ZerosResult, error) {
	if err := requireFiniteSystem(op, sys); err != nil {
		return nil, err
	}
	sys, err := zeroInternalDelays(sys)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	n, m, p := sys.Dims()
	var res *ZerosResult
	switch {
	case m == 0 || p == 0:
		return &ZerosResult{}, nil
	case n == 0:
		e := &mat.Dense{}
		_, _, _, rank, _ := zerosStaircase(e, e, e, sys.D, 0, m, p)
		return &ZerosResult{Rank: rank}, nil
	case sys.IsDescriptor():
		res, err = descriptorZeros(sys)
	default:
		res, err = mimoZeros(sys)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return res, nil
}

// zeroInternalDelays closes the internal-delay loop of sys with every delay
// set to zero, w = z, and drops external delays.
func zeroInternalDelays(sys *System) (*System, error) {
	if !sys.HasDelay() && sys.LFT == nil {
		return sys, nil
	}
	out, err := sys.ZeroDelayApprox()
	if err != nil {
		return nil, err
	}
	out.Delay, out.InputDelay, out.OutputDelay = nil, nil, nil
	return out, nil
}

// descriptorZeros compresses E = U diag(Σ_r, 0) Vᵀ and treats the n-r
// algebraic states as extra inputs and the n-r algebraic equations as extra
// outputs. The resulting order-r explicit system has the same system pencil up
// to row/column permutation, so its invariant zeros are those of sys, and its
// normal rank exceeds that of sys by n-r.
func descriptorZeros(sys *System) (*ZerosResult, error) {
	n, m, p := sys.Dims()
	var svd mat.SVD
	if !svd.Factorize(sys.E, mat.SVDFull) {
		return nil, fmt.Errorf("SVD of E failed: %w", ErrSingularTransform)
	}
	sv := svd.Values(nil)
	tol := float64(n) * eps() * sv[0]
	r := 0
	for r < n && sv[r] > tol {
		r++
	}
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)

	var At, tmp, Bt, Ct mat.Dense
	tmp.Mul(U.T(), sys.A)
	At.Mul(&tmp, &V)
	Bt.Mul(U.T(), sys.B)
	Ct.Mul(sys.C, &V)
	for i := range r {
		s := 1 / sv[i]
		for j := range n {
			At.Set(i, j, At.At(i, j)*s)
		}
		for j := range m {
			Bt.Set(i, j, Bt.At(i, j)*s)
		}
	}

	q := n - r
	mh, ph := m+q, p+q
	at, bt, ct := At.RawMatrix(), Bt.RawMatrix(), Ct.RawMatrix()
	dh := make([]float64, ph*mh)
	copyBlock(dh, mh, 0, 0, at.Data, at.Stride, r, r, q, q)
	copyBlock(dh, mh, 0, q, bt.Data, bt.Stride, r, 0, q, m)
	copyBlock(dh, mh, q, 0, ct.Data, ct.Stride, 0, r, p, q)
	dRaw := sys.D.RawMatrix()
	copyBlock(dh, mh, q, q, dRaw.Data, dRaw.Stride, 0, 0, p, m)
	Dh := mat.NewDense(ph, mh, dh)

	if r == 0 {
		_, _, _, rank, _ := zerosStaircase(&mat.Dense{}, &mat.Dense{}, &mat.Dense{}, Dh, 0, mh, ph)
		return &ZerosResult{Rank: rank - q}, nil
	}
	ah := make([]float64, r*r)
	bh := make([]float64, r*mh)
	ch := make([]float64, ph*r)
	copyBlock(ah, r, 0, 0, at.Data, at.Stride, 0, 0, r, r)
	copyBlock(bh, mh, 0, 0, at.Data, at.Stride, 0, r, r, q)
	copyBlock(bh, mh, 0, q, bt.Data, bt.Stride, 0, 0, r, m)
	copyBlock(ch, r, 0, 0, at.Data, at.Stride, r, 0, q, r)
	copyBlock(ch, r, q, 0, ct.Data, ct.Stride, 0, 0, p, r)
	aug := &System{A: mat.NewDense(r, r, ah), B: mat.NewDense(r, mh, bh), C: mat.NewDense(ph, r, ch), D: Dh, Dt: sys.Dt}
	res, err := mimoZeros(aug)
	if err != nil {
		return nil, err
	}
	res.Rank -= q
	return res, nil
}

// mimoZeros returns eig(A − B·D⁻¹·C) for square models with invertible D,
// else the finite eigenvalues of the staircase-reduced pencil of the
// balanced model (prescaledStaircaseZeros). For square models the zeros
// those may have computed inaccurately are then refined against the exact
// pencil (refineZeros): those below ε·max(‖zs‖, ‖sys‖)/zeroRefineTol, and
// all of them when the scaling shrank the pencil norm by more than
// zeroRefineAllRatio, since that leaves zeros of any modulus with errors
// well above ε·‖sys‖; for non-square ones with outputs (inputs) dependent
// to rounding, the zeros of the square model without them replace the
// staircase ones when at least as many pass zeroBackwardOK
// (squareDownZeros). eig(A − B·D⁻¹·C) is backward stable for
// M = A − B·D⁻¹·C, so its zeros carry an absolute error of about
// ε·(‖A‖ + ‖B‖‖D⁻¹C‖), which swamps zeros small against ‖B·D⁻¹·C‖; when such
// a zero cannot be refined, the staircase zeros of that size, accurate to
// ε·‖[A B; C D]‖, seed the refinement instead (hybridZeros).
func mimoZeros(sys *System) (*ZerosResult, error) {
	_, m, p := sys.Dims()
	if m == p {
		var luD mat.LU
		luD.Factorize(sys.D)
		if !nearZero(luD.Det()) || luD.Cond() <= dInvCondMax {
			var DinvC mat.Dense
			if err := luD.SolveTo(&DinvC, false, sys.C); err == nil {
				zeros, err := eigZeros(sys, &DinvC)
				if err != nil {
					return nil, err
				}
				return &ZerosResult{Zeros: zeros, Rank: m}, nil
			}
		}
	}

	zs := zerosPrescale(sys)
	zeros, rank, err := prescaledStaircaseZeros(sys, zs)
	if err != nil {
		return nil, err
	}
	if m == p && rank == m && len(zeros) > 0 {
		scale, raw := pencilNorm(zs), pencilNorm(sys)
		tol := eps() * math.Max(scale, raw) / zeroRefineTol
		huge := math.Min(scale, raw)
		all := raw > zeroRefineAllRatio*scale
		refineZeros(zs, zeros, scale, func(az float64) bool { return all || az < tol || az > huge })
	}
	if m != p && rank == min(m, p) {
		if sq, err := squareDownZeros(sys, zs, rank); err == nil && len(sq) > 0 && len(sq) >= countBackwardOK(zs, zeros, rank) {
			zeros = sq
		}
	}
	sortZeros(zeros)
	return &ZerosResult{Zeros: zeros, Rank: rank}, nil
}

// eigZeros returns the refined eigenvalues of A − B·(D⁻¹C), sorted.
func eigZeros(sys *System, DinvC *mat.Dense) ([]complex128, error) {
	var M mat.Dense
	M.Mul(sys.B, DinvC)
	M.Sub(sys.A, &M)
	var eig mat.Eigen
	if !eig.Factorize(&M, mat.EigenNone) {
		return nil, fmt.Errorf("eigenvalues of A-BD⁻¹C did not converge: %w", ErrSchurFailed)
	}
	zeros := eig.Values(nil)
	scale := mat.Norm(sys.A, 1) + mat.Norm(sys.B, 1)*mat.Norm(DinvC, 1)
	tol := eps() * scale / zeroRefineTol
	if failed := refineZeros(sys, zeros, scale, func(az float64) bool { return az < tol }); failed > 0 {
		zeros = hybridZeros(sys, zeros, scale, tol, failed)
	}
	sortZeros(zeros)
	return zeros, nil
}

// hybridZeros replaces the eig zeros below tol by the staircase zeros below
// tol when both sets have the same size and either the staircase seeds
// refine with fewer failures or their a priori error ε·‖[A B; C D]‖ is below
// the eig one ε·scale (a near-multiple zero fails refinement either way).
func hybridZeros(sys *System, zeros []complex128, scale, tol float64, failed int) []complex128 {
	st, _, err := prescaledStaircaseZeros(sys, zerosPrescale(sys))
	if err != nil {
		return zeros
	}
	var low []complex128
	for _, z := range st {
		if cmplx.Abs(z) < tol {
			low = append(low, z)
		}
	}
	cand := make([]complex128, 0, len(zeros))
	for _, z := range zeros {
		if cmplx.Abs(z) >= tol {
			cand = append(cand, z)
		}
	}
	if len(cand)+len(low) != len(zeros) {
		return zeros
	}
	cand = append(cand, low...)
	if f := refineZeros(sys, cand, scale, func(az float64) bool { return az < tol }); f < failed || pencilNorm(sys) < scale {
		return cand
	}
	return zeros
}

// prescaledStaircaseZeros returns the staircase zeros and normal rank of the
// balanced model zs. Balancing evens out the entries the rank decisions
// compare with ε·‖[A B; C D]‖, which often recovers zeros the unbalanced
// reduction loses, but it can also lift the roundoff in a block that is
// exactly zero above that threshold, or amplify it through a nearly
// singular D and then clear the threshold by a wide margin. So when a
// decision on zs came within rankMarginMin of its threshold, or within
// rankMarginTrust and the reductions disagree on the pencil order or
// normal rank (unless zs is square of full normal rank, whose zeros
// mimoZeros refines on the exact pencil), the unscaled reduction of sys
// competes: the set with more zeros passing zeroBackwardOK on zs, and
// distinguishable there from points of the same modulus, wins, then
// the one with fewer clearly not zeros (classifyBackward), then the
// unscaled one unless its smallest margin is below that of zs. The test
// uses the larger of the two normal ranks unless the pencil at a generic
// point already passes it.
func prescaledStaircaseZeros(sys, zs *System) ([]complex128, int, error) {
	pre := reduceStaircase(zs)
	if zs == sys {
		z, err := pre.zeros()
		return z, pre.rank, err
	}
	_, m, p := sys.Dims()
	low := pre.margin < rankMarginMin
	if !low && (pre.margin >= rankMarginTrust || m == p && pre.rank == m) {
		z, err := pre.zeros()
		return z, pre.rank, err
	}
	raw := reduceStaircase(sys)
	preferRaw := !low || raw.margin >= pre.margin
	if !low && raw.nu == pre.nu && raw.rank == pre.rank {
		z, err := pre.zeros()
		return z, pre.rank, err
	}
	zr, errRaw := raw.zeros()
	zp, errPre := pre.zeros()
	switch {
	case errRaw != nil && errPre != nil:
		return nil, 0, errRaw
	case errRaw != nil:
		return zp, pre.rank, nil
	case errPre != nil:
		return zr, raw.rank, nil
	}
	rank := max(raw.rank, pre.rank)
	if zeroBackwardOK(zs, complex(0.31, 0.83)*complex(pencilNorm(zs), 0), rank) {
		rank = min(raw.rank, pre.rank)
	}
	okP, badP := classifyBackward(zs, zp, rank, true)
	okR, badR := classifyBackward(zs, zr, rank, true)
	if okP > okR || okP == okR && (badP < badR || badP == badR && !preferRaw) {
		return zp, pre.rank, nil
	}
	return zr, raw.rank, nil
}

// squareDownZeros returns, when all but r rows of the I/O-scaled [C D] of
// zs (columns of [B; D] when m > p) are combinations of the other r to
// within rounding, the zeros of the square model keeping those r that pass
// zeroBackwardOK on zs. The tall pencil then has the zeros of the square
// one, and selecting rows rounds nothing, so an exactly dependent output
// (input) costs no accuracy, unlike the staircase's elimination of it
// through a nearly singular D.
func squareDownZeros(sys, zs *System, r int) ([]complex128, error) {
	n, m, p := zs.Dims()
	a, b, c, d := zs.A, zs.B, zs.C, zs.D
	if m > p {
		a, b, c, d = mat.DenseCopyOf(a.T()), mat.DenseCopyOf(c.T()), mat.DenseCopyOf(b.T()), mat.DenseCopyOf(d.T())
		m, p = p, m
	}
	rows := dependentRowBasis(c, d, n, m, p, r)
	if rows == nil {
		return nil, nil
	}
	cs, ds := mat.NewDense(r, n, nil), mat.NewDense(r, m, nil)
	for k, i := range rows {
		cs.SetRow(k, c.RawRowView(i))
		ds.SetRow(k, d.RawRowView(i))
	}
	res, err := mimoZeros(&System{A: a, B: b, C: cs, D: ds, Dt: sys.Dt})
	if err != nil || res.Rank != r {
		return nil, err
	}
	var out []complex128
	for _, z := range res.Zeros {
		if zeroBackwardOK(zs, z, r) {
			out = append(out, z)
		}
	}
	return out, nil
}

// rowDependentTol is the residual, relative to the largest row norm and in
// units of (n+m)·ε, below which dependentRowBasis counts a row of [C D] as
// a combination of the chosen ones. Rows dependent up to the rounding of
// their entries leave a residual below about 1.5 in these units; a looser
// bound squares down models whose extra row is only nearly dependent and
// returns zeros of the square model the tall pencil lacks.
const rowDependentTol = 2

// dependentRowBasis picks r rows of the p×(n+m) matrix [C D] by
// Gram–Schmidt with largest-residual pivoting and returns them when every
// other row's residual is within rowDependentTol, else nil.
func dependentRowBasis(c, d *mat.Dense, n, m, p, r int) []int {
	w := n + m
	q := make([]float64, p*w)
	var nmax float64
	bl := blas64.Implementation()
	for i := range p {
		copy(q[i*w:i*w+n], c.RawRowView(i))
		copy(q[i*w+n:(i+1)*w], d.RawRowView(i))
		nmax = math.Max(nmax, bl.Dnrm2(w, q[i*w:(i+1)*w], 1))
	}
	tol := rowDependentTol * float64(w) * eps() * nmax
	used := make([]bool, p)
	rows := make([]int, 0, r)
	for k := 0; k <= r; k++ {
		best, bn := -1, 0.0
		for i := range p {
			if !used[i] {
				if v := bl.Dnrm2(w, q[i*w:(i+1)*w], 1); v > bn {
					best, bn = i, v
				}
			}
		}
		if k == r {
			if bn > tol {
				return nil
			}
			return rows
		}
		if !(bn > tol) {
			return nil
		}
		used[best] = true
		rows = append(rows, best)
		qb := q[best*w : (best+1)*w]
		for i := range p {
			if !used[i] {
				qi := q[i*w : (i+1)*w]
				bl.Daxpy(w, -bl.Ddot(w, qi, 1, qb, 1)/(bn*bn), qb, 1, qi, 1)
			}
		}
	}
	return rows
}

// staircase is the reduced pencil s·bf − af of order nu from zerosStaircase.
type staircase struct {
	af, bf   []float64
	nu, rank int
	margin   float64
}

func reduceStaircase(sys *System) staircase {
	n, m, p := sys.Dims()
	af, bf, nu, rank, margin := zerosStaircase(sys.A, sys.B, sys.C, sys.D, n, m, p)
	return staircase{af: af, bf: bf, nu: nu, rank: rank, margin: margin}
}

func countBackwardOK(sys *System, zeros []complex128, rank int) int {
	ok, _ := classifyBackward(sys, zeros, rank, false)
	return ok
}

// classifyBackward counts the zeros passing zeroBackwardOK and those
// clearly not zeros (ratio above zeroNotZeroRatio). With distinct, a zero
// passes only if P fails the test at a generic point of modulus
// max(|z|, ‖sys‖): the ratio of a strictly proper model tends to 0 as |s|
// grows, so far out every point passes, including the rounding-noise zeros
// a rank decision on a vanishing Markov parameter leaves where the model
// has infinite zeros (MATLAB tzero returns those as infinite).
func classifyBackward(sys *System, zeros []complex128, rank int, distinct bool) (ok, bad int) {
	n, m, p := sys.Dims()
	tol := zeroBackwardTol * math.Sqrt(float64((n+p)*(n+m))) * eps()
	norm := pencilNorm(sys)
	for _, z := range zeros {
		switch r := zeroBackwardRatio(sys, z, rank); {
		case r <= tol:
			if distinct && zeroBackwardRatio(sys, cmplx.Rect(math.Max(cmplx.Abs(z), norm), 1+cmplx.Phase(z)), rank) <= tol {
				continue
			}
			ok++
		case r > zeroNotZeroRatio:
			bad++
		}
	}
	return ok, bad
}

// zeroBackwardTol bounds σ_{n+rank}(P(z))/σ₁(P(z)) in zeroBackwardOK, in
// units of the staircase rank tolerance ε·√((n+p)(n+m)).
const zeroBackwardTol = 1e3

// zeroBackwardOK reports whether P(z) = [A−zI B; C D] lies within
// zeroBackwardTol staircase tolerances of dropping below normal rank
// n+rank. P(z) is embedded as the real [Re −Im; Im Re], whose singular
// values are those of P(z), each twice.
func zeroBackwardOK(sys *System, z complex128, rank int) bool {
	n, m, p := sys.Dims()
	return zeroBackwardRatio(sys, z, rank) <= zeroBackwardTol*math.Sqrt(float64((n+p)*(n+m)))*eps()
}

// zeroNotZeroRatio is the σ_{n+rank}(P(z))/σ₁(P(z)) above which z is
// clearly not a zero, whatever its conditioning.
const zeroNotZeroRatio = 1e-6

// zeroBackwardRatio returns σ_{n+rank}(P(z))/σ₁(P(z)), +Inf when the rank
// is out of range or the SVD fails.
func zeroBackwardRatio(sys *System, z complex128, rank int) float64 {
	n, m, p := sys.Dims()
	r, c := n+p, n+m
	k := n + rank
	if k <= 0 || k > min(r, c) {
		return math.Inf(1)
	}
	R := mat.NewDense(2*r, 2*c, nil)
	raw := R.RawMatrix()
	set := func(i, j int, re, im float64) {
		raw.Data[i*raw.Stride+j] = re
		raw.Data[i*raw.Stride+j+c] = -im
		raw.Data[(i+r)*raw.Stride+j] = im
		raw.Data[(i+r)*raw.Stride+j+c] = re
	}
	a, b, cm, d := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix(), sys.D.RawMatrix()
	for i := range n {
		for j := range n {
			set(i, j, a.Data[i*a.Stride+j], 0)
		}
		set(i, i, a.Data[i*a.Stride+i]-real(z), -imag(z))
		for j := range m {
			set(i, n+j, b.Data[i*b.Stride+j], 0)
		}
	}
	for i := range p {
		for j := range n {
			set(n+i, j, cm.Data[i*cm.Stride+j], 0)
		}
		for j := range m {
			set(n+i, n+j, d.Data[i*d.Stride+j], 0)
		}
	}
	var svd mat.SVD
	if !svd.Factorize(R, mat.SVDNone) {
		return math.Inf(1)
	}
	sv := svd.Values(nil)
	if !(sv[0] > 0) {
		return math.Inf(1)
	}
	return sv[2*k-1] / sv[0]
}

// rankMarginTrust is the margin above which the balanced reduction is used
// without comparing it with the unscaled one; the disagreements seen on
// sweeps of nearly singular D came at margins up to about 1e4.
const rankMarginTrust = 1e8

// rankMarginMin is the factor by which every staircase rank decision on the
// balanced model must clear its threshold for its result to be used.
const rankMarginMin = 100

// dInvCondMax is the condition number of D up to which mimoZeros uses
// eig(A − B·D⁻¹·C) even when det D is tiny.
const dInvCondMax = 1e12

// staircaseZeros returns the finite eigenvalues of the staircase-reduced
// pencil, the normal rank and the smallest rank-decision margin.
func staircaseZeros(sys *System) ([]complex128, int, float64, error) {
	st := reduceStaircase(sys)
	z, err := st.zeros()
	if err != nil {
		return nil, 0, 0, err
	}
	return z, st.rank, st.margin, nil
}

// zeros returns the finite eigenvalues of the reduced pencil.
func (st staircase) zeros() ([]complex128, error) {
	afData, bfData, nu := st.af, st.bf, st.nu
	if nu == 0 {
		return nil, nil
	}

	lwork := dggevEigenvaluesWorkLen(nu)
	buf := make([]float64, 3*nu+lwork)
	alphar, alphai, beta, work := buf[:nu], buf[nu:2*nu], buf[2*nu:3*nu], buf[3*nu:]

	ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nu,
		afData, nu, bfData, nu,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, lwork)
	if !ok {
		return nil, fmt.Errorf("QZ of the reduced pencil did not converge: %w", ErrSchurFailed)
	}

	betaTol := float64(nu) * eps()
	var zeros []complex128
	for j := range nu {
		if math.Abs(beta[j]) <= betaTol {
			continue
		}
		re := alphar[j] / beta[j]
		im := alphai[j] / beta[j]
		if math.Abs(im) < math.Abs(re)*eps()*100 {
			im = 0
		}
		zeros = append(zeros, complex(re, im))
	}
	return zeros, nil
}

// zerosPrescale returns sys after the exact power-of-two state scaling of
// the frequency solvers (balance, SLICOT TB01ID as AB08ND with EQUIL = 'S'
// applies) alternated with power-of-two input and output scaling
// (scaleIO), all of which leave the zeros unchanged; sys itself when the
// scaling is the identity. Without the I/O scaling a model with ‖B‖ small
// and ‖C‖ large (or a D swamped by them) keeps blocks below the staircase
// rank tolerance ε·‖[A B; C D]‖ however its states are scaled.
func zerosPrescale(sys *System) *System {
	n, m, p := sys.Dims()
	br := newBalancedRealization(sys, n, m, p)
	D := sys.D
	if n > 0 && m > 0 && p > 0 {
		d := make([]float64, p*m)
		raw := sys.D.RawMatrix()
		copyStrided(d, m, raw.Data, raw.Stride, p, m)
		for range 4 {
			if !scaleIO(&br, d) {
				break
			}
			br.balance(nil)
		}
		D = mat.NewDense(p, m, d)
	}
	zs := &System{A: mat.NewDense(n, n, br.a), B: mat.NewDense(n, m, br.b), C: mat.NewDense(p, n, br.c), D: D, Dt: sys.Dt}
	if mat.Equal(zs.A, sys.A) && mat.Equal(zs.B, sys.B) && mat.Equal(zs.C, sys.C) && mat.Equal(zs.D, sys.D) {
		return sys
	}
	return zs
}

// scaleIO scales each input column of [B; D] (output row of [C D]) by the
// power of two bringing its largest entry nearest the geometric mean of
// those of the state columns of [A; C] (rows of [A B]), and reports
// whether any factor differed from 1.
func scaleIO(br *balancedRealization, d []float64) bool {
	n, m, p := br.n, br.m, br.p
	var colSum, rowSum, nc, nr int
	for k := range n {
		var cs, rs float64
		for i := range n {
			cs = math.Max(cs, math.Abs(br.a[i*n+k]))
			rs = math.Max(rs, math.Abs(br.a[k*n+i]))
		}
		for i := range p {
			cs = math.Max(cs, math.Abs(br.c[i*n+k]))
		}
		for j := range m {
			rs = math.Max(rs, math.Abs(br.b[k*m+j]))
		}
		if e, ok := binaryExp(cs); ok {
			colSum += e
			nc++
		}
		if e, ok := binaryExp(rs); ok {
			rowSum += e
			nr++
		}
	}
	if nc == 0 || nr == 0 {
		return false
	}
	changed := false
	colT, rowT := colSum/nc, rowSum/nr
	for j := range m {
		var s float64
		for i := range n {
			s = math.Max(s, math.Abs(br.b[i*m+j]))
		}
		for i := range p {
			s = math.Max(s, math.Abs(d[i*m+j]))
		}
		e, ok := binaryExp(s)
		if !ok || e == colT {
			continue
		}
		changed = true
		f := math.Ldexp(1, max(-500, min(500, colT-e)))
		for i := range n {
			br.b[i*m+j] *= f
		}
		for i := range p {
			d[i*m+j] *= f
		}
	}
	for i := range p {
		var s float64
		for j := range n {
			s = math.Max(s, math.Abs(br.c[i*n+j]))
		}
		for j := range m {
			s = math.Max(s, math.Abs(d[i*m+j]))
		}
		e, ok := binaryExp(s)
		if !ok || e == rowT {
			continue
		}
		changed = true
		f := math.Ldexp(1, max(-500, min(500, rowT-e)))
		for j := range n {
			br.c[i*n+j] *= f
		}
		for j := range m {
			d[i*m+j] *= f
		}
	}
	return changed
}

// binaryExp returns the exponent e with x in [2^(e-1), 2^e) for finite
// positive x.
func binaryExp(x float64) (int, bool) {
	if !(x > 0) || math.IsInf(x, 0) {
		return 0, false
	}
	_, e := math.Frexp(x)
	return e, true
}

// pencilNorm returns ‖[A B; C D]‖_F.
func pencilNorm(sys *System) float64 {
	var s float64
	for _, a := range []*mat.Dense{sys.A, sys.B, sys.C, sys.D} {
		s = math.Hypot(s, mat.Norm(a, 2))
	}
	return s
}

// refineZeros polishes in place the zeros selected by sel(|z|), which seed
// Newton's method on the exact pencil (zeroRefiner); scale is the norm the
// zeros' absolute error is proportional to. A polished value replaces the
// original only when the iteration converged without moving more than half
// way to another zero or by more than half its modulus (plus 1e3·ε·scale),
// and no two polished values merged, so a failed refinement leaves the
// original value. It returns the number of selected zeros (conjugate pairs
// counted once) left unpolished.
func refineZeros(sys *System, zeros []complex128, scale float64, sel func(az float64) bool) (failed int) {
	const slack = 1e3
	if !(scale > 0) || math.IsInf(scale, 0) {
		return 0
	}
	var zr *zeroRefiner
	ref := make([]complex128, len(zeros))
	copy(ref, zeros)
	tried := make([]bool, len(zeros))
	for i, z := range zeros {
		if imag(z) < 0 {
			continue
		}
		az := cmplx.Abs(z)
		if !sel(az) {
			continue
		}
		tried[i] = true
		failed++
		conj := -1
		if imag(z) > 0 {
			best := 1e-8 * az
			for j, v := range zeros {
				if d := cmplx.Abs(v - cmplx.Conj(z)); imag(v) < 0 && d <= best {
					conj, best = j, d
				}
			}
			if conj < 0 {
				continue
			}
		}
		gap := math.Inf(1)
		for j, v := range zeros {
			if j != i && j != conj {
				gap = math.Min(gap, cmplx.Abs(v-z))
			}
		}
		if zr == nil {
			zr = newZeroRefiner(sys)
		}
		w, ok := zr.refine(z)
		if !ok {
			continue
		}
		if d := cmplx.Abs(w - z); !(d <= gap/2 && d <= az/2+slack*eps()*scale) {
			continue
		}
		if imag(z) == 0 {
			w = complex(real(w), 0)
		}
		ref[i] = w
		if conj >= 0 {
			ref[conj] = cmplx.Conj(w)
		}
		failed--
	}
	merged := make([]bool, len(zeros))
	for i := range ref {
		for j := i + 1; j < len(ref); j++ {
			if zeros[i] != ref[i] || zeros[j] != ref[j] {
				if cmplx.Abs(ref[i]-ref[j]) <= 1e-8*cmplx.Abs(ref[i]) && zeros[i] != zeros[j] {
					merged[i], merged[j] = true, true
				}
			}
		}
	}
	for i, v := range ref {
		if merged[i] {
			if tried[i] && imag(zeros[i]) >= 0 {
				failed++
			}
			continue
		}
		zeros[i] = v
	}
	return failed
}

// zeroRefineTol is the estimated relative error above which
// refineZeros polishes a zero.
const zeroRefineTol = 1e-12

// zeroRefineAllRatio is the ‖sys‖/‖zs‖ above which mimoZeros refines every
// staircase zero of a square model. In sweeps of strongly scaled square
// models, unrefined zeros at ratios down to 256 came out 100× less accurate
// than without I/O scaling, none below 64; well-scaled models skip the
// extra cost.
const zeroRefineAllRatio = 16

// zeroRefiner runs Newton's method on P(s)w = 0, P(s) = [sI−A −B; C D],
// with w_k = 1 for the largest entry of the initial inverse-iteration
// vector. The residual P(z)w is accumulated in compensated arithmetic from
// the stored float64 entries, so the iteration converges to a zero of the
// stored model to about working precision unless the zero is multiple or
// ill-conditioned to the reciprocal of ε.
type zeroRefiner struct {
	n, m       int
	a, b, c, d blas64.General
	lu, w, r   []complex128
	piv        []int
	scale      float64 // max |A_ij|; zeros below ε²·scale count as converged
}

func newZeroRefiner(sys *System) *zeroRefiner {
	n, m, _ := sys.Dims()
	N := n + m
	buf := make([]complex128, N*N+2*N)
	return &zeroRefiner{
		n: n, m: m,
		a: sys.A.RawMatrix(), b: sys.B.RawMatrix(), c: sys.C.RawMatrix(), d: sys.D.RawMatrix(),
		lu: buf[:N*N], w: buf[N*N : N*N+N], r: buf[N*N+N:],
		piv: make([]int, N), scale: maxAbsDense(sys.A),
	}
}

// fill sets lu to P(z), with column k replaced by [w₁..wₙ; 0] when k ≥ 0.
func (zr *zeroRefiner) fill(z complex128, k int) {
	n, m := zr.n, zr.m
	N := n + m
	for i := range n {
		row := zr.lu[i*N : (i+1)*N]
		for j, v := range zr.a.Data[i*zr.a.Stride : i*zr.a.Stride+n] {
			row[j] = complex(-v, 0)
		}
		row[i] += z
		for j, v := range zr.b.Data[i*zr.b.Stride : i*zr.b.Stride+m] {
			row[n+j] = complex(-v, 0)
		}
	}
	for i := range m {
		row := zr.lu[(n+i)*N : (n+i+1)*N]
		for j, v := range zr.c.Data[i*zr.c.Stride : i*zr.c.Stride+n] {
			row[j] = complex(v, 0)
		}
		for j, v := range zr.d.Data[i*zr.d.Stride : i*zr.d.Stride+m] {
			row[n+j] = complex(v, 0)
		}
	}
	if k >= 0 {
		for i := range N {
			if i < n {
				zr.lu[i*N+k] = zr.w[i]
			} else {
				zr.lu[i*N+k] = 0
			}
		}
	}
}

// residual sets r = −P(z)w in compensated arithmetic.
func (zr *zeroRefiner) residual(z complex128) {
	n, m, w := zr.n, zr.m, zr.w
	zre, zim := real(z), imag(z)
	for i := range n {
		var re, im compensatedSum
		re.addProd(zre, real(w[i]))
		re.addProd(-zim, imag(w[i]))
		im.addProd(zre, imag(w[i]))
		im.addProd(zim, real(w[i]))
		for j, v := range zr.a.Data[i*zr.a.Stride : i*zr.a.Stride+n] {
			re.addProd(-v, real(w[j]))
			im.addProd(-v, imag(w[j]))
		}
		for j, v := range zr.b.Data[i*zr.b.Stride : i*zr.b.Stride+m] {
			re.addProd(-v, real(w[n+j]))
			im.addProd(-v, imag(w[n+j]))
		}
		zr.r[i] = complex(-re.value(), -im.value())
	}
	for i := range m {
		var re, im compensatedSum
		for j, v := range zr.c.Data[i*zr.c.Stride : i*zr.c.Stride+n] {
			re.addProd(v, real(w[j]))
			im.addProd(v, imag(w[j]))
		}
		for j, v := range zr.d.Data[i*zr.d.Stride : i*zr.d.Stride+m] {
			re.addProd(v, real(w[n+j]))
			im.addProd(v, imag(w[n+j]))
		}
		zr.r[n+i] = complex(-re.value(), -im.value())
	}
}

// refine returns the zero Newton's method reaches from z0 and whether it
// converged.
func (zr *zeroRefiner) refine(z0 complex128) (complex128, bool) {
	const maxIt = 10
	N := zr.n + zr.m
	k, ok := -1, false
	for _, h := range [...]float64{0, 0x1p-40, 0x1p-26} {
		if k, ok = zr.nullVector(z0 + complex(h, h)*complex(math.Max(cmplx.Abs(z0), 1), 0)); ok {
			break
		}
	}
	if !ok {
		return z0, false
	}
	z := z0
	for range maxIt {
		zr.residual(z)
		zr.fill(z, k)
		if !cLUFactor(zr.lu, zr.piv, N) {
			return z0, false
		}
		cLUSolve(zr.lu, zr.piv, zr.r, N, 1)
		dz := zr.r[k]
		for i, v := range zr.r {
			if i != k {
				zr.w[i] += v
			}
		}
		z += dz
		if cmplx.IsNaN(z) || cmplx.IsInf(z) {
			return z0, false
		}
		if cmplx.Abs(dz) <= 4*eps()*(cmplx.Abs(z)+eps()*zr.scale) {
			return z, true
		}
	}
	return z0, false
}

// nullVector sets w to the inverse-iteration approximation of the right
// null vector of P(z), scaled so its largest entry w_k is 1, and returns k.
func (zr *zeroRefiner) nullVector(z complex128) (int, bool) {
	N := zr.n + zr.m
	zr.fill(z, -1)
	if !cLUFactor(zr.lu, zr.piv, N) {
		return 0, false
	}
	for i := range zr.w {
		zr.w[i] = complex(1+0.25*float64(i%3), 0)
	}
	cLUSolve(zr.lu, zr.piv, zr.w, N, 1)
	k, wMax := 0, 0.0
	for i, v := range zr.w {
		a := cabs1(v)
		if math.IsNaN(a) || math.IsInf(a, 0) {
			return 0, false
		}
		if a > wMax {
			k, wMax = i, a
		}
	}
	if !(wMax > 0) {
		return 0, false
	}
	inv := crecip(zr.w[k])
	for i := range zr.w {
		zr.w[i] *= inv
	}
	zr.w[k] = 1
	return k, true
}

// dggevEigenvaluesWorkLen returns the Dggev workspace length for an
// eigenvalues-only n×n pencil. The optimal size includes Dormqr's 64×64
// block-reflector buffer, but while k ≤ n stays within the Dgeqrf crossover
// and the Dormqr block size both run unblocked whatever lwork is, so the
// minimum 8n executes the same arithmetic.
func dggevEigenvaluesWorkLen(n int) int {
	nx := impl.Ilaenv(3, "DGEQRF", " ", n, n, -1, -1)
	nb := min(64, impl.Ilaenv(1, "DORMQR", "LT", n, n, n, -1))
	if n <= nx && n <= nb {
		return max(1, 8*n)
	}
	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1)
	return int(work[0])
}

func zerosStaircase(A, B, C, D *mat.Dense, n, m, p int) (afOut, bfOut []float64, nu, rank int, margin float64) {
	margin = math.Inf(1)
	if n == 0 && min(m, p) == 0 {
		return nil, nil, 0, 0, margin
	}

	np := n + p
	mn := m + n
	stride := mn
	if stride == 0 {
		stride = 1
	}

	// Build compound matrix [B A; D C], (n+p) × (m+n), row-major
	bf := make([]float64, np*stride)
	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()
	dRaw := D.RawMatrix()

	if m > 0 {
		copyStrided(bf, stride, bRaw.Data, bRaw.Stride, n, m)
	}
	if n > 0 {
		copyBlock(bf, stride, 0, m, aRaw.Data, aRaw.Stride, 0, 0, n, n)
	}
	if m > 0 {
		copyBlock(bf, stride, n, 0, dRaw.Data, dRaw.Stride, 0, 0, p, m)
	}
	if n > 0 {
		copyBlock(bf, stride, n, m, cRaw.Data, cRaw.Stride, 0, 0, p, n)
	}

	// Tolerance
	thresh := math.Sqrt(float64(np*mn)) * eps()
	tol := thresh

	// svlmax = Frobenius norm of compound matrix
	svlmax := 0.0
	for _, v := range bf {
		svlmax += v * v
	}
	svlmax = math.Sqrt(svlmax)

	// Pass 1: reduce D to full row rank
	infz := make([]int, max(n, 1))
	kronl := make([]int, n+1)
	kronr := make([]int, n+1)

	nu, mu, _, ninfz := zerosStaircasePass(n, m, p, p, 0, svlmax, bf, stride, 0, infz, kronl, tol, &margin)
	rank = mu

	numu := nu + mu
	if numu == 0 {
		return nil, nil, 0, rank, margin
	}
	mnu := m + nu

	// Pertranspose (anti-transpose): AF[r][c] = BF[NUMU-1-c][MNU-1-r].
	// Fortran DCOPY copies each row of BF into a reversed column of AF.
	afStride := numu
	if afStride == 0 {
		afStride = 1
	}
	af := make([]float64, mnu*afStride)
	for r := range mnu {
		for c := range numu {
			af[r*afStride+c] = bf[(numu-1-c)*stride+(mnu-1-r)]
		}
	}

	nn := nu
	pp := m
	mm := mu

	if mu != m {
		// Pass 2: reduce D to square invertible
		ro2 := pp - mm
		sigma2 := mm
		nu, mu, _, _ = zerosStaircasePass(nn, mm, pp, ro2, sigma2, svlmax, af, afStride, ninfz, infz, kronr, tol, &margin)
	}

	if nu == 0 {
		return nil, nil, 0, rank, margin
	}

	// Pencil extraction
	i1 := nu + mu
	bfPencil := make([]float64, nu*i1)
	for i := 0; i < nu; i++ {
		bfPencil[i*i1+mu+i] = 1
	}

	if rank != 0 && mu > 0 {
		// D' block: MU rows × I1 cols, extract for DTZRZF
		dp := make([]float64, mu*i1)
		copyBlock(dp, i1, 0, 0, af, afStride, nu, 0, mu, i1)

		tauRZ := make([]float64, mu)
		work := make([]float64, 1)
		impl.Dtzrzf(mu, i1, dp, i1, tauRZ, work, -1)
		work = make([]float64, max(int(work[0]), 1))
		impl.Dtzrzf(mu, i1, dp, i1, tauRZ, work, len(work))

		l := i1 - mu
		work2 := make([]float64, 1)
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, af, afStride, work2, -1)
		work2 = make([]float64, max(int(work2[0]), 1))
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, af, afStride, work2, len(work2))
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, bfPencil, i1, work2, len(work2))
	}

	// Extract Af = af[0:nu, mu:mu+nu] and Bf = bf[0:nu, mu:mu+nu]
	afOut = make([]float64, nu*nu)
	bfOut = make([]float64, nu*nu)
	copyBlock(afOut, nu, 0, 0, af, afStride, 0, mu, nu, nu)
	copyBlock(bfOut, nu, 0, 0, bfPencil, i1, 0, mu, nu, nu)

	return afOut, bfOut, nu, rank, margin
}

func zerosStaircasePass(n, m, p, ro, sigma int, svlmax float64, abcd []float64, stride int,
	ninfz int, infz, kronl []int, tol float64, margin *float64) (nu, mu, nkrol, ninfzOut int) {

	mu = p
	nu = n
	nkrol = 0
	ninfzOut = ninfz

	iz := 0
	ik := 0
	mm1 := m

	work := make([]float64, max(max(m+n, p+n), 1))

	for mu > 0 {
		ro1 := ro
		mnu := m + nu

		if m > 0 {
			// Step a: compress D rows, merge SIGMA triangular cols with RO new rows
			if sigma != 0 {
				irow := nu
				for i1 := 0; i1 < sigma; i1++ {
					colLen := ro + 1
					beta, t := impl.Dlarfg(colLen, abcd[irow*stride+i1], abcd[(irow+1)*stride+i1:], stride)
					abcd[irow*stride+i1] = beta

					if t != 0 && i1+1 < mnu {
						saved := abcd[irow*stride+i1]
						abcd[irow*stride+i1] = 1
						impl.Dlarf(blas.Left, colLen, mnu-i1-1, abcd[irow*stride+i1:], stride, t, abcd[irow*stride+i1+1:], stride, work)
						abcd[irow*stride+i1] = saved
					}
					irow++
				}
				// Zero lower triangular part
				for r := nu + 1; r < nu+ro+sigma; r++ {
					for c := 0; c < sigma && c < r-nu; c++ {
						abcd[r*stride+c] = 0
					}
				}
			}

			// Step b: rank-revealing QR on remaining D block
			if sigma < m {
				i1 := sigma
				irow := nu + sigma
				rankQR, svQR, jpvtSub, tauSub := colPivotQR(ro1, m-sigma, abcd[irow*stride+i1:], stride, tol, svlmax)
				*margin = math.Min(*margin, rankMargin(svQR, rankQR, min(ro1, m-sigma), tol, svlmax))

				// Apply column permutation to rows 0:nu+sigma, cols i1:i1+m-sigma
				impl.Dlapmt(true, nu+sigma, m-sigma, abcd[i1:], stride, jpvtSub)

				if rankQR > 0 {
					// Apply Q^T to C submatrix
					impl.Dormqr(blas.Left, blas.Trans, ro1, nu, rankQR,
						abcd[irow*stride+i1:], stride, tauSub[:rankQR],
						abcd[irow*stride+mm1:], stride, work, len(work))

					// Zero lower triangle of QR result
					if ro1 > 1 {
						for r := 1; r < ro1; r++ {
							for c := 0; c < min(r, rankQR); c++ {
								abcd[(irow+r)*stride+i1+c] = 0
							}
						}
					}
					ro1 -= rankQR
				}
			}
		}

		tau := ro1
		sigma = mu - tau

		// Infinite zero determination
		if iz > 0 {
			infz[iz-1] += ro - tau
			ninfzOut += iz * (ro - tau)
		}
		if ro1 == 0 {
			break
		}
		iz++

		if nu <= 0 {
			mu = sigma
			nu = 0
			ro = 0
		} else {
			// Step c: rank-revealing RQ on C2 block
			c2row := nu + sigma
			mntau := min(tau, nu)
			rank2, sv2, _, tau2 := rowPivotRQ(tau, nu, abcd[c2row*stride+mm1:], stride, tol, svlmax)
			*margin = math.Min(*margin, rankMargin(sv2, rank2, min(tau, nu), tol, svlmax))

			if rank2 > 0 {
				irow2 := c2row + tau - rank2

				// Apply Q^T from RQ to [A;C1] from the right
				impl.Dormr2(blas.Right, blas.Trans, c2row, nu, rank2,
					abcd[irow2*stride+mm1:], stride, tau2[mntau-rank2:],
					abcd[mm1:], stride, work)

				// Apply Q to [B A] from the left
				impl.Dormr2(blas.Left, blas.NoTrans, nu, mnu, rank2,
					abcd[irow2*stride+mm1:], stride, tau2[mntau-rank2:],
					abcd[0:], stride, work)

				// Zero out
				for r := range rank2 {
					for c := 0; c < nu-rank2; c++ {
						abcd[(irow2+r)*stride+mm1+c] = 0
					}
				}
				if rank2 > 1 {
					for r := 1; r < rank2; r++ {
						for c := 0; c < r; c++ {
							abcd[(irow2+r)*stride+mm1+nu-rank2+c] = 0
						}
					}
				}
			}

			ro = rank2
		}

		// Kronecker indices
		kronl[ik] += tau - ro
		nkrol += kronl[ik]
		ik++

		nu -= ro
		mu = sigma + ro
		if ro == 0 {
			break
		}
	}

	return nu, mu, nkrol, ninfzOut
}

// rankMargin returns how far a pivoted QR/RQ rank decision with estimates
// sv = (σmax, σmin of the kept part, σmin had one more been kept) cleared
// its threshold, as a ratio ≥ 1 (+Inf when nothing was rejected or kept).
func rankMargin(sv [3]float64, rank, k int, rcond, svlmax float64) float64 {
	thr := rcond * math.Max(svlmax, sv[0])
	margin := math.Inf(1)
	if rank > 0 && thr > 0 {
		margin = sv[1] / thr
	}
	if rank < k && sv[2] > 0 {
		margin = math.Min(margin, thr/sv[2])
	}
	return margin
}

func nearZero(x float64) bool {
	return math.Abs(x) < 1e-14
}

func sortZeros(z []complex128) {
	sort.Slice(z, func(i, j int) bool {
		return cmplx.Abs(z[i]) < cmplx.Abs(z[j])
	})
}
