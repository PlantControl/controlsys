package controlsys

import "slices"

import "plantcontrol.org/v1/gonum/mat"

// integratorStates marks the rows and the columns of the square a that are
// exactly pole·e_i, the integrators of a continuous (pole 0) or discrete
// (pole 1) state matrix; rows is nil when there is none. A row is an
// integrator fed only by inputs; a column is one that feeds no other state.
func integratorStates(a *mat.Dense, pole float64) (rows, cols []bool) {
	ar := a.RawMatrix()
	n := ar.Rows
	unit := func(i, j int) bool {
		v := ar.Data[i*ar.Stride+j]
		return v == 0 && i != j || v == pole && i == j
	}
	kind := func(i int) (row, col bool) {
		row, col = true, true
		for j := 0; j < n && (row || col); j++ {
			row = row && unit(i, j)
			col = col && unit(j, i)
		}
		return row, col
	}
	first := 0
	for ; first < n; first++ {
		if row, col := kind(first); row || col {
			break
		}
	}
	if first == n {
		return nil, nil
	}
	rows, cols = make([]bool, n), make([]bool, n)
	for i := first; i < n; i++ {
		rows[i], cols[i] = kind(i)
	}
	return rows, cols
}

// integratorOrder is a reordering of the states of a square a around its
// integrators (see integratorStates): integrator columns first, integrator
// rows last, repeatedly on the states left, so integrator chains go too.
// In this order a is block upper triangular, with a 1×1 block pole for each
// such state, and so are the exponential and the bilinear map of a: a solve
// or squaring in this order keeps the zeros below the blocks exactly and
// computes the block of the other states from that block alone. In the
// original order the LU pivoting mixes the integrator rows in; their
// rounding then moves the integrator off its pole and enters the other
// block, ε·‖a‖ absolute, large against its small eigenvalues, which a later
// logarithm (D2C) amplifies.
type integratorOrder struct {
	perm, pos  []int
	peeled     []bool
	rows, cols []bool
}

func newIntegratorOrder(a *mat.Dense, pole float64) (integratorOrder, bool) {
	rows, cols := integratorStates(a, pole)
	if rows == nil {
		return integratorOrder{}, false
	}
	ar := a.RawMatrix()
	n := ar.Rows
	peeled := make([]bool, n)
	unit := func(i, j int) bool {
		v := ar.Data[i*ar.Stride+j]
		return v == 0 && i != j || v == pole && i == j
	}
	var front, back []int
	for changed := true; changed; {
		changed = false
		for i := range n {
			if peeled[i] {
				continue
			}
			row, col := true, true
			for j := 0; j < n && (row || col); j++ {
				if !peeled[j] {
					row = row && unit(i, j)
					col = col && unit(j, i)
				}
			}
			switch {
			case row:
				back = append(back, i)
			case col:
				front = append(front, i)
			default:
				continue
			}
			peeled[i], changed = true, true
		}
	}
	perm := append(make([]int, 0, n), front...)
	for i := range n {
		if !peeled[i] {
			perm = append(perm, i)
		}
	}
	for _, b := range slices.Backward(back) {
		perm = append(perm, b)
	}
	pos := make([]int, n)
	for p, i := range perm {
		pos[i] = p
	}
	return integratorOrder{perm: perm, pos: pos, peeled: peeled, rows: rows, cols: cols}, true
}

// square returns a in the order, or back from it with inverse.
func (o integratorOrder) square(a *mat.Dense, inverse bool) *mat.Dense {
	n := len(o.perm)
	ar := a.RawMatrix()
	out := mat.NewDense(n, n, nil)
	or := out.RawMatrix()
	for p, i := range o.perm {
		for q, j := range o.perm {
			if inverse {
				or.Data[i*or.Stride+j] = ar.Data[p*ar.Stride+q]
			} else {
				or.Data[p*or.Stride+q] = ar.Data[i*ar.Stride+j]
			}
		}
	}
	return out
}

// rowsOf returns the rows of b in the order, or back from it with inverse.
func (o integratorOrder) rowsOf(b *mat.Dense, inverse bool) *mat.Dense {
	if b == nil || b.IsEmpty() {
		return b
	}
	r, c := b.Dims()
	br := b.RawMatrix()
	out := mat.NewDense(r, c, nil)
	or := out.RawMatrix()
	for p, i := range o.perm {
		src, dst := i, p
		if inverse {
			src, dst = p, i
		}
		copy(or.Data[dst*or.Stride:dst*or.Stride+c], br.Data[src*br.Stride:src*br.Stride+c])
	}
	return out
}

// colsOf returns the columns of c in the order, or back from it with
// inverse.
func (o integratorOrder) colsOf(c *mat.Dense, inverse bool) *mat.Dense {
	if c.IsEmpty() {
		return c
	}
	r, k := c.Dims()
	cr := c.RawMatrix()
	out := mat.NewDense(r, k, nil)
	or := out.RawMatrix()
	for q := range r {
		for p, j := range o.perm {
			src, dst := j, p
			if inverse {
				src, dst = p, j
			}
			or.Data[q*or.Stride+dst] = cr.Data[q*cr.Stride+src]
		}
	}
	return out
}

// exact sets in ad, a function of a that keeps its order's block upper
// triangular form, the entries that form fixes: zero below the blocks, value
// on the 1×1 blocks, and value·e_i on the integrator rows and columns of a.
func (o integratorOrder) exact(ad *mat.Dense, value float64) {
	dr := ad.RawMatrix()
	for i := range o.perm {
		row := dr.Data[i*dr.Stride : i*dr.Stride+len(o.perm)]
		for j := range row {
			switch {
			case i == j && o.peeled[i]:
				row[j] = value
			case o.rows[i] || o.cols[j] || o.pos[i] > o.pos[j] && (o.peeled[i] || o.peeled[j]):
				row[j] = 0
			}
		}
	}
}

// integratorHold runs hold, a discretization of (a, b) over dt returning
// e^{a·dt} and the hold integrals Γ_k (Γ₀ = ∫₀^dt e^{a·s} ds·b, Γ₁ of FOH),
// in the integratorOrder of a and sets what the integrators fix (exactHold).
// b may be nil.
func integratorHold(a, b *mat.Dense, dt float64, hold func(a, b *mat.Dense) (*mat.Dense, []*mat.Dense)) (*mat.Dense, []*mat.Dense) {
	o, ok := newIntegratorOrder(a, 0)
	if !ok {
		return hold(a, b)
	}
	ad, gammas := hold(o.square(a, false), o.rowsOf(b, false))
	ad = o.square(ad, true)
	for k := range gammas {
		gammas[k] = o.rowsOf(gammas[k], true)
	}
	o.exactHold(b, dt, ad, gammas)
	return ad, gammas
}

// exactHold sets the structure of e^{a·dt} (exact) and dt·b_i/(k+1) for row
// i of Γ_k at an integrator row i of a, x_i' = b_i·u. A non-finite e^{a·dt}
// is left for the caller's overflow check.
func (o integratorOrder) exactHold(b *mat.Dense, dt float64, ad *mat.Dense, gammas []*mat.Dense) {
	for _, v := range ad.RawMatrix().Data {
		if !isFinite(v) {
			return
		}
	}
	o.exact(ad, 1)
	if len(gammas) == 0 || b.IsEmpty() {
		return
	}
	br := b.RawMatrix()
	for i, row := range o.rows {
		if !row {
			continue
		}
		for k, g := range gammas {
			gr := g.RawMatrix()
			for j := range br.Cols {
				gr.Data[i*gr.Stride+j] = dt * br.Data[i*br.Stride+j] / float64(k+1)
			}
		}
	}
}

// exactLogIntegrators sets in the D2C of (ad, bd) over dt, ac the logarithm
// of ad over dt and bc its input matrix, what the integrators of ad fix: a
// row i of ad that is exactly e_i makes row i of ac zero and row i of bc
// bd_i/dt, the augmented matrix [ad bd; 0 I] being block triangular around
// [1 bd_i; 0 I], whose logarithm is [0 bd_i; 0 0]; a column j of ad that is
// exactly e_j makes column j of ac zero. bc may be nil.
func exactLogIntegrators(ad, bd *mat.Dense, dt float64, ac, bc *mat.Dense) {
	rows, cols := integratorStates(ad, 1)
	if rows == nil {
		return
	}
	cr := ac.RawMatrix()
	for i := range rows {
		for j := range rows {
			if rows[i] || cols[j] {
				cr.Data[i*cr.Stride+j] = 0
			}
		}
	}
	if bc == nil {
		return
	}
	br, bcr := bd.RawMatrix(), bc.RawMatrix()
	for i, row := range rows {
		if row {
			for j := range br.Cols {
				bcr.Data[i*bcr.Stride+j] = br.Data[i*br.Stride+j] / dt
			}
		}
	}
}

// integratorBilinear runs bilinear, the map of (a, b, c) to ad = pbeta·I −
// 2αβ·(a + palpha·I)⁻¹, bd = scaleB·(a + palpha·I)⁻¹·b and cd =
// scaleC·c·(a + palpha·I)⁻¹, in the integratorOrder of a (pole 0 for C2D, 1
// for D2C) and sets what the integrators fix. An integrator row i of a is
// k·e_i of a + palpha·I, k = pole + palpha, so row i of the inverse is e_i/k:
// row i of ad is image·e_i, image = pbeta − 2αβ/k the other domain's
// integrator, and row i of bd is scaleB·b_i/k; an integrator column j
// likewise makes column j of cd scaleC·c_j/k.
func integratorBilinear(sys *System, pole, palpha, image, scaleB, scaleC float64, bilinear func(*System) (*System, error)) (*System, error) {
	o, ok := newIntegratorOrder(sys.A, pole)
	if !ok {
		return bilinear(sys)
	}
	out, err := bilinear(&System{A: o.square(sys.A, false), B: o.rowsOf(sys.B, false), C: o.colsOf(sys.C, false), D: sys.D, Dt: sys.Dt})
	if err != nil {
		return nil, err
	}
	out.A, out.B, out.C = o.square(out.A, true), o.rowsOf(out.B, true), o.colsOf(out.C, true)
	o.exact(out.A, image)
	k := pole + palpha
	br, bdr := sys.B.RawMatrix(), out.B.RawMatrix()
	cr, cdr := sys.C.RawMatrix(), out.C.RawMatrix()
	for i := range o.perm {
		if o.rows[i] && !sys.B.IsEmpty() {
			for j := range br.Cols {
				bdr.Data[i*bdr.Stride+j] = scaleB * (br.Data[i*br.Stride+j] / k)
			}
		}
		if o.cols[i] && !sys.C.IsEmpty() {
			for q := range cr.Rows {
				cdr.Data[q*cdr.Stride+i] = scaleC * (cr.Data[q*cr.Stride+i] / k)
			}
		}
	}
	propagateNames(out, sys)
	return out, nil
}
