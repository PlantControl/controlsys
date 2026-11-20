package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// ModalTruncateOptions selects the retained modes. Set at most one of Order
// and MaxRealPart; with neither, ModalTruncate keeps about half the modes.
type ModalTruncateOptions struct {
	// Order is the number of retained states, rounded up to the end of a
	// mode cluster it would split; 0 means unset.
	Order int
	// MaxRealPart, when non-nil, keeps every mode with real part at or
	// above *MaxRealPart (any value, including 0), plus all unstable modes.
	MaxRealPart *float64
}

// ModalReductionResult is a modal truncation: the reduced model, its
// order, the retained poles in Schur order, and the n×Order Basis and
// Order×n Projection with Projection·Basis = I and Sys.A = Projection·A·Basis.
type ModalReductionResult struct {
	Sys        *System
	Order      int
	KeptPoles  []complex128
	Basis      *mat.Dense
	Projection *mat.Dense
}

// ModalTruncate keeps the dominant modes of a delay-free state-space model
// (slowest first; unstable modes, including those on the stability boundary
// within rounding, always kept) by projecting onto an ordered
// real Schur basis, never splitting a complex-conjugate pair; the feedthrough
// D is kept unchanged. It mirrors MATLAB reducespec(sys,"modal") with getrom.
// Like MATLAB's mode clusters, modes that cannot be separated reliably
// (repeated eigenvalues, or a cut whose Sylvester separation is
// ill-conditioned) are kept or dropped together: the kept order is rounded
// up to the end of such a cluster, so Result.Order may exceed Order, up to
// the full model.
// Order out of range, an order that would split a lone pair or drop an
// unstable mode, and a MaxRealPart that selects nothing return ErrInvalidOrder; both
// Order and MaxRealPart set return ErrInvalidArgument; a model without states
// returns ErrDimensionMismatch.
// See https://www.mathworks.com/help/control/ref/lti.reducespec.html.
func ModalTruncate(sys *System, opts *ModalTruncateOptions) (*ModalReductionResult, error) {
	if err := requireFiniteSystem("ModalTruncate", sys); err != nil {
		return nil, err
	}
	if opts == nil {
		opts = &ModalTruncateOptions{}
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("ModalTruncate"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("ModalTruncate"); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	if opts.Order != 0 && opts.MaxRealPart != nil {
		return nil, fmt.Errorf("ModalTruncate: Order and MaxRealPart are both set: %w", ErrInvalidArgument)
	}
	if opts.MaxRealPart != nil && math.IsNaN(*opts.MaxRealPart) {
		return nil, fmt.Errorf("ModalTruncate: MaxRealPart is NaN: %w", ErrInvalidArgument)
	}
	if opts.Order < 0 || opts.Order > n {
		return nil, fmt.Errorf("ModalTruncate: order %d outside [0,%d]: %w", opts.Order, n, ErrInvalidOrder)
	}
	if n == 0 {
		return nil, fmt.Errorf("ModalTruncate: system has no states: %w", ErrDimensionMismatch)
	}

	t, z, err := modalSchur(sys)
	if err != nil {
		return nil, fmt.Errorf("ModalTruncate: %w", err)
	}
	var maxRealPart float64
	threshold := opts.MaxRealPart != nil
	if threshold {
		maxRealPart = *opts.MaxRealPart
	}
	clusters := modalClusters(t, n, sys.Dt, maxRealPart, threshold)
	order, err := modalReductionOrder(clusters, n, opts)
	if err != nil {
		return nil, err
	}
	if err := reorderModalClusters(t, z, n, clusters); err != nil {
		return nil, fmt.Errorf("ModalTruncate: %w", err)
	}
	var x []float64
	for order < n {
		if splitsSchurBlock(t, n, order) {
			return nil, fmt.Errorf("ModalTruncate: reordered Schur form splits a block at %d: %w", order, ErrSchurFailed)
		}
		if x, err = separateSchurBlocks(t, n, order, "ModalTruncate"); err == nil && modalSeparable(x) {
			break
		}
		order = nextModalClusterEnd(clusters, order)
	}
	keptPoles := schurEigenvaluesRaw(t, n)[:order:order]
	if order == n {
		return &ModalReductionResult{
			Sys:        sys.Copy(),
			Order:      n,
			KeptPoles:  keptPoles,
			Basis:      eyeDense(n),
			Projection: eyeDense(n),
		}, nil
	}

	bModal, cModal := modalInputOutput(sys, z, n, m, p)
	aReduced := extractModalBlock(t, n, 0, order, 0, order)
	bReduced := newDense(order, m)
	for i := range order {
		for j := range m {
			value := bModal[i*m+j]
			for k := order; k < n; k++ {
				value -= x[i*(n-order)+k-order] * bModal[k*m+j]
			}
			bReduced.Set(i, j, value)
		}
	}
	cReduced := newDense(p, order)
	for i := range p {
		copy(cReduced.RawMatrix().Data[i*order:(i+1)*order], cModal[i*n:i*n+order])
	}
	reduced, err := policy.resultWithOriginalFeedthrough(aReduced, bReduced, cReduced)
	if err != nil {
		return nil, err
	}
	basis, projection := modalProjectionBases(z, x, n, order)
	return &ModalReductionResult{
		Sys:        reduced,
		Order:      order,
		KeptPoles:  keptPoles,
		Basis:      basis,
		Projection: projection,
	}, nil
}

func modalSchur(sys *System) (t, z []float64, err error) {
	n, _, _ := sys.Dims()
	t = make([]float64, n*n)
	aRaw := sys.A.RawMatrix()
	copyStrided(t, n, aRaw.Data, aRaw.Stride, n, n)
	z = make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	bwork := make([]bool, n)
	query := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, query, -1, bwork)
	work := make([]float64, int(query[0]))
	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, work, len(work), bwork)
	if !ok {
		return nil, nil, fmt.Errorf("real Schur form of A did not converge: %w", ErrSchurFailed)
	}
	return t, z, nil
}

// modalClusterTol is the distance, relative to ‖T‖_F, within which Schur
// eigenvalues always count as copies of one repeated eigenvalue: computed
// copies of an exactly repeated semisimple eigenvalue lie within ~1e-15.
const modalClusterTol = 1e-10

// modalConditionMerge scales the first-order rounding move eps·‖T‖_F·κ of
// an eigenvalue with condition number κ: two eigenvalues closer than
// modalConditionMerge times the sum of their moves are one cluster. A
// defective eigenvalue of multiplicity m spreads into copies ~m moves
// apart, so copies are merged for m up to about 8, while well-conditioned
// close eigenvalues stay apart.
const modalConditionMerge = 16

// modalMoveCap bounds the rounding move, relative to ‖T‖_F, of an
// eigenvalue whose condition number is huge or infinite. A defective
// eigenvalue of multiplicity m spreads by ~eps^(1/m) and the first-order
// move of each copy is ~1/m of that, so the cap with modalConditionMerge
// reaches spreads of 3.2e-4·‖T‖_F (m ≤ 4); eigenvectors are computed only
// for blocks within that reach of another.
// Members of a modalClusterTol cluster take no move: their eigenvectors
// are those of an exactly repeated eigenvalue, whose condition number
// says nothing about distant eigenvalues.
const modalMoveCap = 1e-5

// modalScoreTieTol is the score difference, relative to ‖T‖_F, below which
// two clusters count as equally dominant (rounding noise of equal exact
// scores) and are ordered by frequency instead.
const modalScoreTieTol = 1e-12

// modalBoundaryBand is the width, in units of eps·‖T‖_F, of the band
// below the stability boundary (Re = 0, or |z| = 1) within which a
// cluster mean counts as on it, and so unstable: rounding moves the mean
// of a cluster, even of defective copies split across the boundary, by a
// few such units, so exactly marginal modes are kept whatever the
// rounding, while modes farther inside stay stable.
const modalBoundaryBand = 64

// modalSeparationBound caps ‖X‖_F of the Sylvester separation of the kept
// modes from the discarded ones; a larger X means the cut cannot be made
// reliably and the kept set grows by the next cluster.
const modalSeparationBound = 1e5

// modalCluster is a set of Schur blocks kept or dropped together. blocks
// and sizes are the start rows and orders of its blocks in the unordered
// Schur form; mean, score, imag and freq are the mean real part, real part
// (modulus when discrete), |imaginary part| and frequency (|imaginary
// part|, or |arg| when discrete) of its eigenvalues, which rounding spreads
// about the exact ones far less than each copy; unstable holds when the
// mean score is within modalBoundaryBand rounding units of the stability
// boundary or beyond it; selected (MaxRealPart) holds for unstable
// clusters and by the mean real part; move is the largest rounding
// move of a member; pair marks a lone complex-conjugate block whose
// conjugates are farther apart than their moves (a rounded real double
// is not a pair).
type modalCluster struct {
	blocks, sizes []int
	states        int
	mean, score   float64
	imag, freq    float64
	move          float64
	selected      bool
	unstable      bool
	pair          bool
}

// modalClusters groups the blocks of the real Schur form t into clusters,
// transitively merging eigenvalues within modalClusterTol·‖T‖_F or within
// modalConditionMerge times their rounding moves, and returns them in
// retention order: unstable clusters first, then selected (MaxRealPart)
// ones, then by descending score, where scores within modalScoreTieTol
// count as tied and are ordered by ascending frequency. All decisions use
// the eigenvalues of the unordered form, so ulp differences between copies
// never decide which copy is kept.
func modalClusters(t []float64, n int, dt, maxRealPart float64, threshold bool) []modalCluster {
	evals := schurEigenvaluesRaw(t, n)
	starts := make([]int, 0, n)
	sizes := make([]int, 0, n)
	reps := make([]complex128, 0, n)
	for i := 0; i < n; i += schurBlockSize(t, n, i) {
		l := evals[i]
		if imag(l) < 0 {
			l = cmplx.Conj(l)
		}
		starts = append(starts, i)
		sizes = append(sizes, schurBlockSize(t, n, i))
		reps = append(reps, l)
	}
	var norm l2
	for _, v := range t {
		norm.add(v)
	}
	tol := modalClusterTol * norm.norm()
	tieTol := modalScoreTieTol * norm.norm()
	band := modalBoundaryBand * eps() * norm.norm()
	parent := rangeInts(len(reps))
	find := func(i int) int {
		for parent[i] != i {
			parent[i] = parent[parent[i]]
			i = parent[i]
		}
		return i
	}
	union := func(i, j int) {
		a, b := find(i), find(j)
		parent[max(a, b)] = min(a, b)
	}
	tight := make([]bool, len(reps))
	for i := range reps {
		for j := i + 1; j < len(reps); j++ {
			if cmplx.Abs(reps[i]-reps[j]) <= tol {
				union(i, j)
				tight[i], tight[j] = true, true
			}
		}
	}
	move := make([]float64, len(reps))
	reach := 2 * modalConditionMerge * modalMoveCap * norm.norm()
	need := make([]bool, len(reps))
	near := false
	for i := range reps {
		if !tight[i] && sizes[i] == 2 && 2*imag(reps[i]) <= reach {
			need[i], near = true, true
		}
		for j := i + 1; j < len(reps); j++ {
			if !(tight[i] && tight[j]) && cmplx.Abs(reps[i]-reps[j]) <= reach {
				need[i] = need[i] || !tight[i]
				need[j] = need[j] || !tight[j]
				near = true
			}
		}
	}
	if near {
		kappa := schurEigenvalueConditions(t, n, starts, sizes, need)
		for i := range move {
			if !tight[i] {
				move[i] = min(kappa[i]*eps()*norm.norm(), modalMoveCap*norm.norm())
			}
		}
		for i := range reps {
			for j := i + 1; j < len(reps); j++ {
				if cmplx.Abs(reps[i]-reps[j]) <= modalConditionMerge*(move[i]+move[j]) {
					union(i, j)
				}
			}
		}
	}
	index := make([]int, len(reps))
	count := make([]int, len(reps))
	clusters := make([]modalCluster, 0, len(reps))
	for i := range reps {
		root := find(i)
		if root == i {
			index[i] = len(clusters)
			clusters = append(clusters, modalCluster{})
		}
		count[index[root]]++
	}
	backing := make([]int, 2*len(reps))
	for k := range clusters {
		clusters[k].blocks = backing[:0:count[k]]
		clusters[k].sizes = backing[len(reps) : len(reps) : len(reps)+count[k]]
		backing = backing[count[k]:]
	}
	for i, l := range reps {
		c := &clusters[index[find(i)]]
		c.blocks = append(c.blocks, starts[i])
		c.sizes = append(c.sizes, sizes[i])
		c.states += sizes[i]
		c.move = max(c.move, move[i])
		w := float64(sizes[i])
		c.mean += w * real(l)
		c.score += w * cmplx.Abs(l)
		c.imag += w * imag(l)
		if dt > 0 {
			c.freq += w * math.Abs(cmplx.Phase(l))
		} else {
			c.freq += w * imag(l)
		}
	}
	for i := range clusters {
		c := &clusters[i]
		w := float64(c.states)
		c.mean /= w
		c.score /= w
		c.imag /= w
		c.freq /= w
		boundary := 1.0
		if dt == 0 {
			c.score = c.mean
			boundary = 0
		}
		c.unstable = c.score >= boundary-band
		c.selected = c.unstable || threshold && c.mean >= maxRealPart
		c.pair = len(c.blocks) == 1 && c.sizes[0] == 2 && 2*c.imag > max(tol, 2*modalConditionMerge*c.move)
	}
	class := func(c modalCluster) int {
		switch {
		case c.unstable:
			return 0
		case c.selected:
			return 1
		}
		return 2
	}
	slices.SortStableFunc(clusters, func(a, b modalCluster) int {
		if c := cmp.Compare(class(a), class(b)); c != 0 {
			return c
		}
		return cmp.Compare(b.score, a.score)
	})
	for lo := 0; lo < len(clusters); {
		hi := lo + 1
		for hi < len(clusters) && class(clusters[hi]) == class(clusters[lo]) && clusters[hi-1].score-clusters[hi].score <= tieTol {
			hi++
		}
		slices.SortStableFunc(clusters[lo:hi], func(a, b modalCluster) int {
			if c := cmp.Compare(a.freq, b.freq); c != 0 {
				return c
			}
			return cmp.Compare(b.score, a.score)
		})
		lo = hi
	}
	return clusters
}

// modalReductionOrder is the number of kept states before the separation
// check: Order rounded up to the end of the cluster it falls in (an Order
// inside a lone complex pair or below the unstable states is
// ErrInvalidOrder), the selected clusters for MaxRealPart, or about half
// the states, rounded up and covering the unstable clusters.
func modalReductionOrder(clusters []modalCluster, n int, opts *ModalTruncateOptions) (int, error) {
	unstable := 0
	for _, c := range clusters {
		if c.unstable {
			unstable += c.states
		}
	}
	switch {
	case opts.Order > 0:
		order := opts.Order
		end := 0
		for _, c := range clusters {
			start := end
			end += c.states
			if start < order && order < end {
				if c.pair {
					return 0, fmt.Errorf("ModalTruncate: order %d splits a complex-conjugate mode pair: %w", order, ErrInvalidOrder)
				}
				order = end
				break
			}
		}
		if order < unstable {
			return 0, fmt.Errorf("ModalTruncate: order %d would discard an unstable mode: %w", opts.Order, ErrInvalidOrder)
		}
		return order, nil
	case opts.MaxRealPart != nil:
		order := 0
		for _, c := range clusters {
			if c.selected {
				order += c.states
			}
		}
		if order == 0 {
			return 0, fmt.Errorf("ModalTruncate: no modes satisfy MaxRealPart %g: %w", *opts.MaxRealPart, ErrInvalidOrder)
		}
		return order, nil
	}
	order := max(n/2, 1)
	end := 0
	for _, c := range clusters {
		end += c.states
		if end >= order {
			break
		}
	}
	return max(end, unstable), nil
}

// schurEigenvalueConditions returns the condition number ‖x‖‖y‖/|yᴴx| of
// the eigenvalue of each block of the quasi-triangular t (starts, sizes)
// marked in need, from its right and left eigenvectors (+Inf where yᴴx
// vanishes), and 0 for the others.
func schurEigenvalueConditions(t []float64, n int, starts, sizes []int, need []bool) []float64 {
	selected := make([]bool, n)
	cols := 0
	for b, j := range starts {
		if need[b] {
			selected[j] = true
			cols += sizes[b]
		}
	}
	vl := make([]float64, n*cols)
	vr := make([]float64, n*cols)
	work := make([]float64, max(1, 3*n))
	impl.Dtrevc3(lapack.EVBoth, lapack.EVSelected, selected, n, t, n, vl, cols, vr, cols, cols, work, len(work))
	kappa := make([]float64, len(starts))
	col := 0
	for b := range starts {
		if !need[b] {
			continue
		}
		var nx, ny l2
		var dot complex128
		for k := range n {
			x := complex(vr[k*cols+col], 0)
			y := complex(vl[k*cols+col], 0)
			if sizes[b] == 2 {
				x = complex(vr[k*cols+col], vr[k*cols+col+1])
				y = complex(vl[k*cols+col], vl[k*cols+col+1])
			}
			nx.addC(x)
			ny.addC(y)
			dot += cmplx.Conj(y) * x
		}
		col += sizes[b]
		kappa[b] = nx.norm() * ny.norm() / cmplx.Abs(dot)
		if math.IsNaN(kappa[b]) {
			kappa[b] = math.Inf(1)
		}
	}
	return kappa
}

func nextModalClusterEnd(clusters []modalCluster, order int) int {
	end := 0
	for _, c := range clusters {
		end += c.states
		if end > order {
			return end
		}
	}
	return end
}

func modalSeparable(x []float64) bool {
	var norm l2
	for _, v := range x {
		norm.add(v)
	}
	return norm.norm() <= modalSeparationBound
}

// reorderModalClusters moves the Schur blocks of t into the order of
// clusters, updating z. Blocks are tracked by their state counts, so a
// 2×2 block that Dtrexc splits into two 1×1 blocks still moves whole.
func reorderModalClusters(t, z []float64, n int, clusters []modalCluster) error {
	type block struct{ start, size int }
	var pending []block
	for i := 0; i < n; i += schurBlockSize(t, n, i) {
		pending = append(pending, block{i, schurBlockSize(t, n, i)})
	}
	work := make([]float64, n)
	placed := 0
	for _, c := range clusters {
		for _, start := range c.blocks {
			j := slices.IndexFunc(pending, func(b block) bool { return b.start == start })
			pos := placed
			for _, b := range pending[:j] {
				pos += b.size
			}
			remaining := pending[j].size
			pending = slices.Delete(pending, j, j+1)
			for remaining > 0 {
				got := schurBlockSize(t, n, pos)
				if got > remaining {
					return fmt.Errorf("reordering the real Schur form merged blocks: %w", ErrSchurFailed)
				}
				if pos != placed {
					if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, pos, placed, work); !ok {
						return fmt.Errorf("reordering the real Schur form failed: %w", ErrSchurFailed)
					}
				}
				placed += got
				pos += got
				remaining -= got
			}
		}
	}
	return nil
}

func separateSchurBlocks(t []float64, n, order int, context string) ([]float64, error) {
	discarded := n - order
	x := make([]float64, order*discarded)
	for i := range order {
		for j := range discarded {
			x[i*discarded+j] = -t[i*n+order+j]
		}
	}
	scale, ok := impl.Dtrsyl(blas.NoTrans, blas.NoTrans, -1, order, discarded, t, n, t[order*n+order:], n, x, discarded)
	if !ok || scale == 0 || scale < 1e-12 {
		return nil, fmt.Errorf("%s: retained and discarded modes cannot be separated reliably: %w", context, ErrSchurFailed)
	}
	if scale != 1 {
		for i := range x {
			x[i] /= scale
		}
	}
	return x, nil
}

func modalInputOutput(sys *System, z []float64, n, m, p int) (bModal, cModal []float64) {
	zGeneral := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}
	bModal = make([]float64, n*m)
	if m > 0 {
		bRaw := sys.B.RawMatrix()
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGeneral,
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bModal})
	}
	cModal = make([]float64, p*n)
	if p == 0 {
		return bModal, cModal
	}
	cRaw := sys.C.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data}, zGeneral,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cModal})
	return bModal, cModal
}

func modalProjectionBases(z, x []float64, n, order int) (*mat.Dense, *mat.Dense) {
	basis := mat.NewDense(n, order, nil)
	for i := range n {
		copy(basis.RawMatrix().Data[i*order:(i+1)*order], z[i*n:i*n+order])
	}
	schurProjection := mat.NewDense(order, n, nil)
	for i := range order {
		schurProjection.Set(i, i, 1)
		for j := order; j < n; j++ {
			schurProjection.Set(i, j, -x[i*(n-order)+j-order])
		}
	}
	projection := mat.NewDense(order, n, nil)
	projection.Mul(schurProjection, mat.NewDense(n, n, z).T())
	return basis, projection
}

func extractModalBlock(data []float64, stride, rowStart, rowEnd, colStart, colEnd int) *mat.Dense {
	rows := rowEnd - rowStart
	cols := colEnd - colStart
	if rows == 0 || cols == 0 {
		return &mat.Dense{}
	}
	out := mat.NewDense(rows, cols, nil)
	for i := range rows {
		copy(out.RawMatrix().Data[i*cols:(i+1)*cols], data[(rowStart+i)*stride+colStart:(rowStart+i)*stride+colEnd])
	}
	return out
}

func schurBlockSize(t []float64, n, start int) int {
	if start+1 < n && t[(start+1)*n+start] != 0 {
		return 2
	}
	return 1
}

func splitsSchurBlock(t []float64, n, order int) bool {
	return order > 0 && order < n && t[order*n+order-1] != 0
}

func unstableModalCount(poles []complex128, dt float64) int {
	count := 0
	for _, pole := range poles {
		if modalPoleUnstable(pole, dt) {
			count++
		}
	}
	return count
}

func modalPoleUnstable(pole complex128, dt float64) bool {
	if dt > 0 {
		return cmplx.Abs(pole) >= 1
	}
	return real(pole) >= 0
}

func rangeInts(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
