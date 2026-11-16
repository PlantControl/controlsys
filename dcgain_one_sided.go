package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// dcGainOneSidedCertified proves the DC gain of an explicit model whose
// poles at x0 are integrators coupled one way only (ONYZO7): a set K of
// states with A_kk = x0 exactly and every row (or every column) of K
// exactly zero off the diagonal, the block of K then x0·I exactly. With
// the remaining states r, a zero-row K gives each entry the 1/σ coefficient
//
//	ρ_ij = Σ_k B_kj·β_ik,  β = C_K + C_r·(x0·I − A_rr)⁻¹·A_rK,
//
// and a zero-column K the transpose, ρ_ij = Σ_k C_ik·β_kj with
// β = B_K + A_Kr·(x0·I − A_rr)⁻¹·B_r. β is the DC gain of a regular model
// of stored entries, so solveDCRegular certifies it; its certificate
// bounds ‖I − P·Rinv‖ < 1 for the exact P = x0·I − A_rr, which proves
// x0·I − A_rr nonsingular rather than assuming it, so K holds every pole
// at x0. An entry whose terms all have a B_kj (C_ik) of exactly 0 has no
// pole there and is the gain of (A_rr, B_r, C_r, D), certified with
// dcZeroRule. Otherwise every term with a nonzero B_kj (C_ik) needs a β
// proven nonzero (|β| beyond its bound) and all terms one sign, the sign
// of ρ as a product of signs; the entry is that infinity, the one-sided
// limit σ → 0⁺ of exactDCGainLimit. A β whose bound reaches 0 leaves the
// coefficient and any derivative term undecided, and ok is false.
func (sys *System) dcGainOneSidedCertified(x0 float64) (*mat.Dense, bool) {
	n, _, _ := sys.Dims()
	aRaw := sys.A.RawMatrix()
	inK := make([]bool, n)
	for _, rows := range []bool{true, false} {
		nk := 0
		for k := range n {
			inK[k] = aRaw.Data[k*aRaw.Stride+k] == x0 && zeroOffDiagonalLine(aRaw, k, rows)
			if inK[k] {
				nk++
			}
		}
		if nk == 0 {
			continue
		}
		if g, ok := sys.dcGainOneSidedIntegrators(x0, inK, nk, rows); ok {
			return g, true
		}
	}
	return nil, false
}

func zeroOffDiagonalLine(a blas64.General, k int, row bool) bool {
	for l := range a.Rows {
		if l == k {
			continue
		}
		v := a.Data[l*a.Stride+k]
		if row {
			v = a.Data[k*a.Stride+l]
		}
		if v != 0 {
			return false
		}
	}
	return true
}

func (sys *System) dcGainOneSidedIntegrators(x0 float64, inK []bool, nk int, rows bool) (*mat.Dense, bool) {
	n, m, p := sys.Dims()
	nr := n - nk
	ks, rs := make([]int, 0, nk), make([]int, 0, nr)
	for k := range n {
		if inK[k] {
			ks = append(ks, k)
		} else {
			rs = append(rs, k)
		}
	}
	a, b, c := sys.A, sys.B, sys.C
	pickAs := func(tr bool, src *mat.Dense, ri, ci []int) *mat.Dense {
		if len(ri) == 0 || len(ci) == 0 {
			return nil
		}
		raw := src.RawMatrix()
		if tr {
			d := mat.NewDense(len(ci), len(ri), nil)
			for ii, i := range ri {
				for jj, j := range ci {
					d.Set(jj, ii, raw.Data[i*raw.Stride+j])
				}
			}
			return d
		}
		d := mat.NewDense(len(ri), len(ci), nil)
		for ii, i := range ri {
			for jj, j := range ci {
				d.Set(ii, jj, raw.Data[i*raw.Stride+j])
			}
		}
		return d
	}
	pick := func(src *mat.Dense, ri, ci []int) *mat.Dense { return pickAs(false, src, ri, ci) }
	all := func(k int) []int {
		v := make([]int, k)
		for i := range v {
			v[i] = i
		}
		return v
	}
	outs, ins := all(p), all(m)
	var beta *mat.Dense
	var btau []float64
	if rows {
		beta, btau = dcCertifiedGainOf(pick(a, rs, rs), pick(a, rs, ks), pick(c, outs, rs), pick(c, outs, ks), p, nk, x0)
	} else {
		// β = B_K + A_Kr·P⁻¹·B_r solved transposed, as Σ_q y_q·B_qj with
		// y = A_Kr·P⁻¹: the rounded limit's terms are y_q·B_qj, which τ then
		// sums, not A_kl·(P⁻¹B)_l.
		beta, btau = dcCertifiedGainOf(pickAs(true, a, rs, rs), pickAs(true, a, ks, rs), pickAs(true, b, rs, ins), pickAs(true, b, ks, ins), m, nk, x0)
	}
	if beta == nil {
		return nil, false
	}
	var fin *mat.Dense
	var ftau []float64
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			sign := 0
			for kk, k := range ks {
				w, bi, bj := b.At(k, j), i, kk
				if !rows {
					w, bi, bj = c.At(i, k), j, kk
				}
				if w == 0 {
					continue
				}
				v, t := beta.At(bi, bj), btau[bi*beta.RawMatrix().Cols+bj]
				if v == 0 || !dcZeroRule(v, t) {
					return nil, false
				}
				s := signInt(w) * signInt(v)
				if sign == -s {
					return nil, false
				}
				sign = s
			}
			if sign != 0 {
				gain.Set(i, j, math.Inf(sign))
				continue
			}
			if fin == nil {
				if fin, ftau = dcCertifiedGainOf(pick(a, rs, rs), pick(b, rs, ins), pick(c, outs, rs), denseCopySafe(sys.D, p, m), p, m, x0); fin == nil {
					return nil, false
				}
			}
			v := fin.At(i, j)
			if !dcZeroRule(v, ftau[i*m+j]) {
				return nil, false
			}
			gain.Set(i, j, v)
		}
	}
	return gain, true
}

// dcCertifiedGainOf is D + C·(x0·I − A)⁻¹·B certified by solveDCRegular,
// with its per-entry τ; D alone, exact with τ = 0, when A is empty. It is
// nil where the solve fails, overflows or does not certify.
func dcCertifiedGainOf(a, b, c, d *mat.Dense, p, m int, x0 float64) (*mat.Dense, []float64) {
	if d == nil {
		d = mat.NewDense(p, m, nil)
	}
	if a == nil {
		return mat.DenseCopyOf(d), make([]float64, p*m)
	}
	r := solveDCRegular(nil, a, b, c, d, x0, true)
	if r.solveErr != nil || r.overflow || r.g == nil || !r.certified {
		return nil, nil
	}
	return r.g, r.tau
}
