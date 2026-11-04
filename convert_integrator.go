package controlsys

import "plantcontrol.org/v1/gonum/mat"

// integratorStates marks the rows and the columns of the square a that are
// exactly pole·e_i, the integrators of a continuous (pole 0) or discrete
// (pole 1) state matrix, and lists the other states in rest; rows is nil
// when there is none. A row is an integrator fed only by inputs; a column is
// one that feeds no other state.
func integratorStates(a *mat.Dense, pole float64) (rows, cols []bool, rest []int) {
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
		return nil, nil, nil
	}
	rows, cols = make([]bool, n), make([]bool, n)
	rest = make([]int, first, n)
	for i := range first {
		rest[i] = i
	}
	for i := first; i < n; i++ {
		rows[i], cols[i] = kind(i)
		if !rows[i] && !cols[i] {
			rest = append(rest, i)
		}
	}
	return rows, cols, rest
}

func submatrix(a *mat.Dense, idx []int) *mat.Dense {
	ar := a.RawMatrix()
	out := mat.NewDense(len(idx), len(idx), nil)
	or := out.RawMatrix()
	for p, i := range idx {
		for q, j := range idx {
			or.Data[p*or.Stride+q] = ar.Data[i*ar.Stride+j]
		}
	}
	return out
}

// setSubmatrix copies block into dst at rows and columns idx unless block
// has a non-finite entry.
func setSubmatrix(dst *mat.Dense, idx []int, block *mat.Dense) {
	br, dr := block.RawMatrix(), dst.RawMatrix()
	for p := range idx {
		for _, v := range br.Data[p*br.Stride : p*br.Stride+len(idx)] {
			if !isFinite(v) {
				return
			}
		}
	}
	for p, i := range idx {
		for q, j := range idx {
			dr.Data[i*dr.Stride+j] = br.Data[p*br.Stride+q]
		}
	}
}

// setIntegratorStates sets the rows and the columns of ad marked in rows and
// cols to value·e_i.
func setIntegratorStates(ad *mat.Dense, rows, cols []bool, value float64) {
	dr := ad.RawMatrix()
	for i := range rows {
		row := dr.Data[i*dr.Stride : i*dr.Stride+len(rows)]
		for j := range row {
			if rows[i] || cols[j] {
				row[j] = 0
				if i == j {
					row[j] = value
				}
			}
		}
	}
}

// exactIntegrators sets the parts of the hold discretization of (a, b) over
// dt that the integrators of a fix and the exponential's Padé solve only
// rounds to. A zero row i of a is x_i' = b_i·u: row i of ad = e^{a·dt} is
// e_i and row i of the k-th hold integral gammas[k] (Γ₀ = ∫₀^dt e^{a·s}
// ds·b, Γ₁ of FOH) is dt·b_i/(k+1). A zero column j feeds no state: column j
// of ad is e_j. With deflate, the block of ad on the other states is the
// exponential of that block of a alone, around which a is block triangular:
// the exponential of all of a rounds it by ε·‖a·dt‖ through the integrator
// coupling, which is large relative to its small eigenvalues and moves them
// in a later logarithm (D2C). b may be nil when gammas is empty. A
// non-finite ad is left for the caller's overflow check.
func exactIntegrators(a, b *mat.Dense, dt float64, deflate bool, ad *mat.Dense, gammas ...*mat.Dense) {
	n, _ := a.Dims()
	rows, cols, rest := integratorStates(a, 0)
	if rows == nil {
		return
	}
	for _, v := range ad.RawMatrix().Data {
		if !isFinite(v) {
			return
		}
	}
	if deflate && len(rest) > 0 {
		setSubmatrix(ad, rest, expmScaled(submatrix(a, rest), dt))
	}
	setIntegratorStates(ad, rows, cols, 1)
	if len(gammas) == 0 {
		return
	}
	br := b.RawMatrix()
	for i := range n {
		if !rows[i] {
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

// exactLogIntegrators is exactIntegrators for D2C of (ad, bd) over dt, ac
// the logarithm of ad over dt and bc its input matrix: a row i of ad that is
// exactly e_i makes row i of ac zero and row i of bc bd_i/dt, the augmented
// matrix [ad bd; 0 I] being block triangular around [1 bd_i; 0 I], whose
// logarithm is [0 bd_i; 0 0]; a column j of ad that is exactly e_j makes
// column j of ac zero. bc may be nil.
func exactLogIntegrators(ad, bd *mat.Dense, dt float64, ac, bc *mat.Dense) {
	n, _ := ad.Dims()
	rows, cols, _ := integratorStates(ad, 1)
	if rows == nil {
		return
	}
	setIntegratorStates(ac, rows, cols, 0)
	if bc == nil {
		return
	}
	br, cr := bd.RawMatrix(), bc.RawMatrix()
	for i := range n {
		if rows[i] {
			for j := range br.Cols {
				cr.Data[i*cr.Stride+j] = br.Data[i*br.Stride+j] / dt
			}
		}
	}
}

// exactBilinearIntegrators sets the parts of the bilinear map of (a, b, c),
// ad = pbeta·I − 2αβ·(a + palpha·I)⁻¹, bd = scaleB·(a + palpha·I)⁻¹·b and
// cd = scaleC·c·(a + palpha·I)⁻¹, that the integrators of a fix (pole·e_i
// rows and columns; pole 0 for C2D, 1 for D2C) and the LU solve only rounds
// to. Such a row i is k·e_i of a + palpha·I, k = pole + palpha, so row i of
// the inverse is e_i/k: row i of ad is image·e_i, image = pbeta − 2αβ/k the
// other domain's integrator, and row i of bd is scaleB·b_i/k. A column j
// likewise makes column j of ad image·e_j and column j of cd scaleC·c_j/k.
// The block of ad on the other states is block(that block of a), as in
// exactIntegrators.
func exactBilinearIntegrators(a, b, c *mat.Dense, pole, palpha, image, scaleB, scaleC float64, block func(*mat.Dense) (*mat.Dense, bool), ad, bd, cd *mat.Dense) {
	n, _ := a.Dims()
	rows, cols, rest := integratorStates(a, pole)
	if rows == nil {
		return
	}
	if len(rest) > 0 {
		if e, ok := block(submatrix(a, rest)); ok {
			setSubmatrix(ad, rest, e)
		}
	}
	setIntegratorStates(ad, rows, cols, image)
	k := pole + palpha
	br, bdr := b.RawMatrix(), bd.RawMatrix()
	cr, cdr := c.RawMatrix(), cd.RawMatrix()
	for i := range n {
		if rows[i] {
			for j := range br.Cols {
				bdr.Data[i*bdr.Stride+j] = scaleB * (br.Data[i*br.Stride+j] / k)
			}
		}
		if cols[i] {
			for q := range cr.Rows {
				cdr.Data[q*cdr.Stride+i] = scaleC * (cr.Data[q*cr.Stride+i] / k)
			}
		}
	}
}
