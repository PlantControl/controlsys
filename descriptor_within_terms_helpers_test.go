package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// storedResponse320 is the response of the stored descriptor sys at s from
// a 320-bit solve of (sE − A) on its float data, nil when it fails.
func storedResponse320(sys *System, s complex128) [][]complex128 {
	re, im := newStepOraclePencil(sys).transfer(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))})
	if re == nil {
		return nil
	}
	out := make([][]complex128, len(re))
	for i := range re {
		out[i] = make([]complex128, len(re[i]))
		for j := range re[i] {
			a, _ := re[i][j].Float64()
			b, _ := im[i][j].Float64()
			out[i][j] = complex(a, b)
		}
	}
	return out
}

// chainCutoff is ≈ε^(1/(2k))·|λ_c|, past which the stored response is the
// rounded chain's: k the longest chain of the structured oracle (an
// E-superdiagonal run over its zero-diagonal states) and |λ_c| the least
// QZ eigenvalue of the stored pencil not within 1e-3 relative of an oracle
// finite pole; +Inf when there is none.
func chainCutoff(sys, oracle *System) float64 {
	n, _, _ := oracle.Dims()
	k, run := 0, 0
	for i := range n {
		if oracle.E.At(i, i) != 0 {
			run = 0
			continue
		}
		if i > 0 && oracle.E.At(i-1, i) != 0 && oracle.E.At(i-1, i-1) == 0 {
			run++
		} else {
			run = 1
		}
		k = max(k, run)
	}
	finite := func(s *System) []complex128 {
		m, _, _ := s.Dims()
		ar, ai, b, ok := qzEigenvalues(m, denseData(s.A), m, denseData(s.E), m)
		if !ok {
			return nil
		}
		var out []complex128
		for i := range m {
			if b[i] != 0 {
				if z := complex(ar[i]/b[i], ai[i]/b[i]); !cmplx.IsInf(z) && !cmplx.IsNaN(z) {
					out = append(out, z)
				}
			}
		}
		return out
	}
	op := finite(oracle)
	lc := math.Inf(1)
	for _, z := range finite(sys) {
		near := false
		for _, o := range op {
			near = near || cmplx.Abs(z-o) <= 1e-3*cmplx.Abs(o)
		}
		if !near {
			lc = math.Min(lc, cmplx.Abs(z))
		}
	}
	if k == 0 || math.IsInf(lc, 0) {
		return math.Inf(1)
	}
	return math.Pow(eps(), 1/float64(2*k)) * lc
}

// withinTermsWorst judges tf, divided at 320 bits (bigDen.quo), against the
// 320-bit stored response of sys by the certificate's refute rule at its
// own points (descriptorTFPoints) with η < 1, |s| ≤ cutoff and outside tf's windows: the largest (|tf − G| − δ)/
// (√ε·max(scale, T̂ + δ_T) + δ_data), δ = ε|G| + 4ε|tf|, and where. Above 1
// tf is outside terms of the stored data. Points the GEPP bounds already
// place within are not solved at 320 bits.
func withinTermsWorst(t *testing.T, sys *System, tf *TransferFunc, cutoff float64) (float64, complex128) {
	t.Helper()
	_, m, p := sys.Dims()
	roots, poles, _, _, ok := tfShape(tf)
	if !ok {
		t.Fatal("tfShape failed")
	}
	tol := math.Sqrt(eps())
	pts := descriptorTFPoints(sys, roots)
	full := newPencilBoundedResponse(sys, pts)
	win := tfPoleWindows(tf, poles, tol)
	pm := p * m
	r := make([]complex128, pm)
	worst, at := 0.0, complex128(0)
	for k, s := range pts {
		if !full.ok[k] || !(full.eta[k] < 1) || cmplx.Abs(s) > cutoff || win.pole.contain(s) {
			continue
		}
		need := false
		for i := range p {
			d := newBigDen(tf.Den[i], s)
			scale := 0.0
			for j := range m {
				v, ok := d.quo(tf.Num[i][j], s)
				if !ok {
					v = cmplx.Inf()
				}
				r[i*m+j] = v
				g := full.g[k*pm+i*m+j] + complex(sys.D.At(i, j), 0)
				scale = math.Max(scale, math.Max(cmplx.Abs(r[i*m+j]), cmplx.Abs(g)-full.comp[k*pm+i*m+j]))
			}
			for j := range m {
				e := k*pm + i*m + j
				g := full.g[e] + complex(sys.D.At(i, j), 0)
				if !win.guards(s, i, j) && cmplx.Abs(g-r[i*m+j])+full.comp[e]+eps()*(cmplx.Abs(g)+full.comp[e])+4*eps()*cmplx.Abs(r[i*m+j]) > tol*math.Max(scale, full.terms[e]+full.dTer[e]) {
					need = true
				}
			}
		}
		if !need {
			continue
		}
		w := storedResponse320(sys, s)
		if w == nil {
			continue
		}
		for i := range p {
			scale := 0.0
			for j := range m {
				scale = math.Max(scale, math.Max(cmplx.Abs(r[i*m+j]), cmplx.Abs(w[i][j])))
			}
			for j := range m {
				if win.guards(s, i, j) {
					continue
				}
				e := k*pm + i*m + j
				d := cmplx.Abs(w[i][j]-r[i*m+j]) - eps()*cmplx.Abs(w[i][j]) - 4*eps()*cmplx.Abs(r[i*m+j])
				if q := d / (tol*math.Max(scale, full.terms[e]+full.dTer[e]) + full.data[e]); q > worst {
					worst, at = q, s
				}
			}
		}
	}
	return worst, at
}

// oracleSTermRatio is the largest |Σ_{k≥1} s^k·c·N^k·b| of the structured
// oracle's chains (N its E on the zero-diagonal states, A = I there) over
// the stored model's refute threshold √ε·max(|G|, T̂ + δ_T) + δ_data, at
// tf's certificate points (descriptorTFPoints) with η < 1 and |s| ≤ cutoff:
// at or above 1 the oracle's s-terms are visible to the stored data, so a
// proper reading of them is wrong.
func oracleSTermRatio(sys, oracle *System, tf *TransferFunc, cutoff float64) float64 {
	n, m, p := oracle.Dims()
	roots, _, _, _, ok := tfShape(tf)
	if !ok {
		return math.Inf(1)
	}
	var ch []int
	for i := range n {
		if oracle.E.At(i, i) == 0 {
			ch = append(ch, i)
		}
	}
	pts := descriptorTFPoints(sys, roots)
	full := newPencilBoundedResponse(sys, pts)
	tol := math.Sqrt(eps())
	worst := 0.0
	for k, s := range pts {
		if !full.ok[k] || !(full.eta[k] < 1) || cmplx.Abs(s) > cutoff {
			continue
		}
		for i := range p {
			for j := range m {
				v := make([]complex128, len(ch))
				for q, r := range ch {
					v[q] = complex(oracle.B.At(r, j), 0)
				}
				var eff complex128
				sk := complex(1, 0)
				for range len(ch) {
					nv := make([]complex128, len(ch))
					for q, r := range ch {
						for q2, r2 := range ch {
							nv[q] += complex(oracle.E.At(r, r2), 0) * v[q2]
						}
					}
					v = nv
					sk *= s
					var cv complex128
					for q, r := range ch {
						cv += complex(oracle.C.At(i, r), 0) * v[q]
					}
					eff += sk * cv
				}
				e := k*p*m + i*m + j
				g := full.g[e] + complex(sys.D.At(i, j), 0)
				thr := tol*math.Max(cmplx.Abs(g), full.terms[e]+full.dTer[e]) + full.data[e]
				worst = math.Max(worst, cmplx.Abs(eff)/thr)
			}
		}
	}
	return worst
}

// oracleTFRatio is the largest |G_oracle − tf| over the stored model's
// refute threshold √ε·max(|G_oracle|, T̂ + δ_T) + δ_data at tf's
// certificate points (descriptorTFPoints) with η < 1 and |s| ≤ cutoff,
// G_oracle the structured oracle's 320-bit response: below 1 tf reads the
// oracle within terms of the stored data wherever they determine it.
func oracleTFRatio(sys, oracle *System, tf *TransferFunc, cutoff float64) float64 {
	_, m, p := sys.Dims()
	roots, _, _, _, ok := tfShape(tf)
	if !ok {
		return math.Inf(1)
	}
	pts := descriptorTFPoints(sys, roots)
	full := newPencilBoundedResponse(sys, pts)
	tol := math.Sqrt(eps())
	worst := 0.0
	for k, s := range pts {
		if !full.ok[k] || !(full.eta[k] < 1) || cmplx.Abs(s) > cutoff {
			continue
		}
		o := storedResponse320(oracle, s)
		g, err := tf.Eval(s)
		if o == nil || err != nil {
			continue
		}
		for i := range p {
			for j := range m {
				e := k*p*m + i*m + j
				thr := tol*math.Max(cmplx.Abs(o[i][j]), full.terms[e]+full.dTer[e]) + full.data[e]
				worst = math.Max(worst, cmplx.Abs(o[i][j]-g[i][j])/thr)
			}
		}
	}
	return worst
}

// regimeDiag describes the descriptor regime of sys (chainRegime) for test
// logs: which path stands, and for an uncertified exact model why its last
// candidate is not verified.
func regimeDiag(sys *System) string {
	rg := sys.chainRegime()
	names := map[int]string{regimeFloat: "float", regimeExact: "exact", regimeNone: "none"}
	d := fmt.Sprintf("regime %s, float err %v", names[rg.kind], rg.mainErr)
	if rg.hasV {
		d += fmt.Sprintf(", exact: certified %v, climbs %d, ceiling %v, %s", rg.v.certified, rg.v.climbs, rg.v.ceiling, rg.v.why)
	}
	return d
}

// requireTFWithinTermsOrUnverified fails unless TransferFunction's res,
// err keep its within-terms contract (SL352H): a value with nil only when
// withinTermsWorst ≤ 1 (an explicit sys judged as (A, B, C, D, I)), and a
// value with ErrAccuracyUnverified only when withinTermsWorst > 1, so that
// no TF within terms at a determined point is refused. It reports whether
// err is nil.
func requireTFWithinTermsOrUnverified(t *testing.T, sys *System, res *TransferFuncResult, err error) bool {
	t.Helper()
	w, at := tfWithinTermsWorstOf(t, sys, res, err)
	switch {
	case err == nil && w > 1:
		t.Fatalf("nil TF outside terms at %v: %.3g× the threshold", at, w)
	case err != nil && !(w > 1):
		t.Fatalf("TF within terms (%.3g× the threshold) returned with %v", w, err)
	}
	return err == nil
}

// requireTFNilWithinTerms is requireTFWithinTermsOrUnverified allowing an
// unverified TF the stored data determine at no point, as where the
// pencil's determinacy fails everywhere (short of verified points).
func requireTFNilWithinTerms(t *testing.T, sys *System, res *TransferFuncResult, err error) bool {
	t.Helper()
	if w, at := tfWithinTermsWorstOf(t, sys, res, err); err == nil && w > 1 {
		t.Fatalf("nil TF outside terms at %v: %.3g× the threshold", at, w)
	}
	return err == nil
}

func tfWithinTermsWorstOf(t *testing.T, sys *System, res *TransferFuncResult, err error) (float64, complex128) {
	t.Helper()
	if res == nil || err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("TransferFunction: %v", err)
	}
	pencil := sys
	if !sys.IsDescriptor() {
		n, _, _ := sys.Dims()
		pencil = &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D, E: eyeDense(n), Dt: sys.Dt}
	}
	return withinTermsWorst(t, pencil, res.TF, math.Inf(1))
}
