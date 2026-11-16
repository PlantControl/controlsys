package controlsys

import (
	"math"
	"math/big"

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
// dcZeroRule. Otherwise ρ̂ = Σ_k B_kj·β̂_ik, summed exactly, must exceed
// Σ_k |B_kj|·(2τ_ik + ½ulp(β̂_ik)), an isolated integrator's β exact with
// τ = 0: one τ covers the certificate's error, the other keeps |ρ| beyond
// ε of its own terms, where exactDCGainLimitRounded reads it as no
// cancellation. The entry is the infinity of ρ̂'s sign, the one-sided
// limit σ → 0⁺ of exactDCGainLimit. A ρ̂ within that bound leaves the
// coefficient and any derivative term undecided, and ok is false; so does
// a zero-row integrator beside a zero-column one (a second-order pole).
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
		mixed := false
		for k := range n {
			mixed = mixed || !inK[k] && aRaw.Data[k*aRaw.Stride+k] == x0 && zeroOffDiagonalLine(aRaw, k, !rows)
		}
		if mixed {
			// A zero-row and a zero-column integrator chained through the
			// regular states make a second-order pole.
			return nil, false
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
	isolated := make([]bool, nk)
	aRaw := a.RawMatrix()
	for kk, k := range ks {
		isolated[kk] = zeroOffDiagonalLine(aRaw, k, !rows)
	}
	bcols := beta.RawMatrix().Cols
	var fin *mat.Dense
	var ftau []float64
	var rho, bound, tr big.Rat
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			terms, single := 0, 0
			for kk, k := range ks {
				w, bi := b.At(k, j), i
				if !rows {
					w, bi = c.At(i, k), j
				}
				if w == 0 {
					continue
				}
				terms++
				single = kk
				if isolated[kk] {
					btau[bi*bcols+kk] = 0
				}
			}
			if terms == 0 {
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
				continue
			}
			bi := i
			if !rows {
				bi = j
			}
			if terms == 1 {
				v, t := beta.At(bi, single), btau[bi*bcols+single]
				w := b.At(ks[single], j)
				if !rows {
					w = c.At(i, ks[single])
				}
				if v == 0 || !(math.Abs(v)*(1-0x1p-52) > 2*t) {
					return nil, false
				}
				gain.Set(i, j, math.Inf(signInt(w)*signInt(v)))
				continue
			}
			if sg, ok := oneSidedSumSign(ks, func(k int) float64 {
				if rows {
					return b.At(k, j)
				}
				return c.At(i, k)
			}, beta.RawMatrix().Data[bi*bcols:(bi+1)*bcols], btau[bi*bcols:(bi+1)*bcols]); ok {
				gain.Set(i, j, math.Inf(sg))
				continue
			}
			rho.SetInt64(0)
			bound.SetInt64(0)
			for kk, k := range ks {
				w := b.At(k, j)
				if !rows {
					w = c.At(i, k)
				}
				if w == 0 {
					continue
				}
				v, t := beta.At(bi, kk), btau[bi*bcols+kk]
				tr.SetFloat64(w)
				tr.Mul(&tr, new(big.Rat).SetFloat64(v))
				rho.Add(&rho, &tr)
				e := new(big.Rat).SetFloat64(2 * t)
				e.Add(e, new(big.Rat).SetFloat64(math.Abs(v)*0x1p-52))
				tr.SetFloat64(math.Abs(w))
				tr.Mul(&tr, e)
				bound.Add(&bound, &tr)
			}
			if new(big.Rat).Abs(&rho).Cmp(&bound) <= 0 {
				return nil, false
			}
			gain.Set(i, j, math.Inf(rho.Sign()))
		}
	}
	return gain, true
}

// oneSidedSumSign decides sign(ρ), ρ = Σ_k w_k·β_k, against the bound
// E = Σ_k |w_k|·(2τ_k + |β_k|·2⁻⁵²) in floating point where its rounding
// is bounded: the sum of K products errs by at most γ_{K+1}·Σ|w_k·β_k| and
// E by a factor 1 ± γ_{K+4}, so |S| − g·Σ|t| > E·(1 + g), g = (2K + 8)·u,
// proves |ρ| > E. ok is false where that does not decide it or a product
// leaves the normal range; the caller then sums exactly.
func oneSidedSumSign(ks []int, weight func(k int) float64, beta, tau []float64) (sign int, ok bool) {
	var sum, abs, bound float64
	terms := 0
	for kk, k := range ks {
		w := weight(k)
		if w == 0 {
			continue
		}
		terms++
		t := w * beta[kk]
		e := math.Abs(w) * (2*tau[kk] + math.Abs(beta[kk])*0x1p-52)
		if !(math.Abs(t) >= 0x1p-900 && math.Abs(t) <= 0x1p900) || e != 0 && !(e >= 0x1p-900 && e <= 0x1p900) {
			return 0, false
		}
		sum += t
		abs += math.Abs(t)
		bound += e
	}
	g := float64(2*terms+8) * 0x1p-53
	if math.Abs(sum)-g*abs > bound*(1+g) {
		return signInt(sum), true
	}
	return 0, false
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
