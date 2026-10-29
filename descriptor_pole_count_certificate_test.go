package controlsys

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strings"
)

// poleCountCert is an exact-arithmetic finite-pole count for the stored
// pencil (A, E): the Van Dooren staircase run in big.Float on the float64
// data, with every tolerance input computed from that data in big.
//
// Hi is the staircase that zeroes only singular values ≤ τ_E = n·eps(σ₁(E))
// at every level (MATLAB rank at every level, in exact arithmetic). Lo is
// the staircase that zeroes every value ≤ ρ_j, where ρ_j bounds how far a
// τ-perturbation of the data (τ_E on E, τ_A = n·eps(σ₁(A)) on A) can move a
// level-j singular value through the rotations of levels < j. NF = Hi is
// the deliverable; Decided reports Lo == Hi. Hi is not an upper bound when
// two stiff poles share a finite block whose σ_min is below τ_E: a level ≥ 1
// then zeroes a well-conditioned pole (ELT76F step 2, rvb two/cpx).
type poleCountCert struct {
	NF, Lo, Hi int
	Decided    bool
	Reason     string
	Levels     []poleCountLevel
	LoLevels   []poleCountLevel
}

// poleCountLevel records one staircase level: the k×k E block's singular
// values SV, the zone radius Rho (= τ_E at level 0), the counts in the
// zones s ≤ τ_E, τ_E < s ≤ ρ, s > ρ, the rank R kept, and for a deflating
// level the smallest singular value SigmaA of the A row block, the E gap
// σ_R − σ_{R+1} and the rotation bound SinZ.
type poleCountLevel struct {
	K, R                     int
	Zero, Undecided, Nonzero int
	Rho                      *big.Float
	SV                       []*big.Float
	SigmaA, Gap, SinZ        *big.Float
}

const poleCountPrec = pencilOraclePrec

func pcNew() *big.Float { return new(big.Float).SetPrec(poleCountPrec) }

func pcSet(x *big.Float) *big.Float { return pcNew().Set(x) }

func pcFloat(x float64) *big.Float { return pcNew().SetFloat64(x) }

// pcMat copies a row-major n×n float64 matrix exactly.
func pcMat(n int, data []float64) [][]*big.Float {
	m := make([][]*big.Float, n)
	for i := range n {
		m[i] = make([]*big.Float, n)
		for j := range n {
			m[i][j] = pcFloat(data[i*n+j])
		}
	}
	return m
}

func pcDot(x, y []*big.Float) *big.Float {
	s, t := pcNew(), pcNew()
	for i := range x {
		s.Add(s, t.Mul(x[i], y[i]))
	}
	return s
}

// pcJacobiStop is 2⁻⁶⁰⁰: rows p, q count as orthogonal once
// γ² ≤ 2⁻⁶⁰⁰·α·β, a cosine of at most 2⁻³⁰⁰.
var pcJacobiStop = new(big.Float).SetMantExp(big.NewFloat(1), -600)

const pcJacobiSweeps = 100

// pcJacobiRows orthogonalizes the first k columns of rows 0..len(x)-1 of x
// by one-sided Jacobi in fixed cyclic order, applying every rotation to the
// same rows of each matrix in also (first k columns), then sorts the rows
// by norm, descending (stable). It returns the row norms (the singular
// values of x[:, :k]) and whether the sweep converged.
func pcJacobiRows(x [][]*big.Float, k int, also ...[][]*big.Float) ([]*big.Float, bool) {
	m := len(x)
	rot := func(rows [][]*big.Float, p, q int, c, s *big.Float) {
		for j := range k {
			a, b := rows[p][j], rows[q][j]
			na := pcNew().Mul(c, a)
			na.Sub(na, pcNew().Mul(s, b))
			nb := pcNew().Mul(s, a)
			nb.Add(nb, pcNew().Mul(c, b))
			rows[p][j], rows[q][j] = na, nb
		}
	}
	one := pcFloat(1)
	converged := false
	for sweep := 0; sweep < pcJacobiSweeps && !converged; sweep++ {
		converged = true
		for p := range m {
			for q := p + 1; q < m; q++ {
				alpha, beta, gamma := pcDot(x[p][:k], x[p][:k]), pcDot(x[q][:k], x[q][:k]), pcDot(x[p][:k], x[q][:k])
				if gamma.Sign() == 0 {
					continue
				}
				g2 := pcNew().Mul(gamma, gamma)
				if g2.Cmp(pcNew().Mul(pcNew().Mul(alpha, beta), pcJacobiStop)) <= 0 {
					continue
				}
				converged = false
				zeta := pcNew().Sub(beta, alpha)
				zeta.Quo(zeta, pcNew().Mul(pcFloat(2), gamma))
				den := pcNew().Sqrt(pcNew().Add(one, pcNew().Mul(zeta, zeta)))
				den.Add(den, pcNew().Abs(zeta))
				t := pcNew().Quo(one, den)
				if zeta.Sign() < 0 {
					t.Neg(t)
				}
				c := pcNew().Quo(one, pcNew().Sqrt(pcNew().Add(one, pcNew().Mul(t, t))))
				s := pcNew().Mul(c, t)
				rot(x, p, q, c, s)
				for _, y := range also {
					rot(y, p, q, c, s)
				}
			}
		}
	}
	norms := make([]*big.Float, m)
	for i := range m {
		norms[i] = pcNew().Sqrt(pcDot(x[i][:k], x[i][:k]))
	}
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(i, j int) int { return norms[j].Cmp(norms[i]) })
	permute := func(rows [][]*big.Float) {
		cp := slices.Clone(rows)
		for i, j := range idx {
			rows[i] = cp[j]
		}
	}
	permute(x)
	for _, y := range also {
		permute(y)
	}
	sorted := make([]*big.Float, m)
	for i, j := range idx {
		sorted[i] = norms[j]
	}
	return sorted, converged
}

// pcSigma1 returns σ₁ of the n×n matrix m.
func pcSigma1(m [][]*big.Float) (*big.Float, bool) {
	cp := make([][]*big.Float, len(m))
	for i := range m {
		cp[i] = make([]*big.Float, len(m[i]))
		for j := range m[i] {
			cp[i][j] = pcSet(m[i][j])
		}
	}
	sv, ok := pcJacobiRows(cp, len(m))
	if len(sv) == 0 {
		return pcNew(), ok
	}
	return sv[0], ok
}

// pcTau returns n·eps(σ) with eps the float64 spacing at the correctly
// rounded σ, the floor descriptorRank applies (MATLAB rank).
func pcTau(n int, sigma *big.Float) *big.Float {
	s, _ := sigma.Float64()
	return pcFloat(float64(n) * (math.Nextafter(s, math.Inf(1)) - s))
}

func pcMinOne(x *big.Float) *big.Float {
	if x.Cmp(pcFloat(1)) > 0 {
		return pcFloat(1)
	}
	return x
}

// pcRowComplement returns an orthonormal basis of the orthogonal complement
// of the orthonormal rows q in R^k: greedy, each step taking the unit vector
// e_j with the largest residual (lowest j on ties), projected out twice.
func pcRowComplement(q [][]*big.Float, k int) [][]*big.Float {
	basis := slices.Clone(q)
	var out [][]*big.Float
	project := func(v []*big.Float) {
		for range 2 {
			for _, b := range basis {
				d := pcDot(b, v)
				for i := range k {
					v[i].Sub(v[i], pcNew().Mul(d, b[i]))
				}
			}
		}
	}
	for len(basis) < k {
		var best []*big.Float
		var bestN *big.Float
		for j := range k {
			v := make([]*big.Float, k)
			for i := range k {
				v[i] = pcNew()
			}
			v[j].SetInt64(1)
			project(v)
			if nn := pcDot(v, v); bestN == nil || nn.Cmp(bestN) > 0 {
				best, bestN = v, nn
			}
		}
		project(best)
		nrm := pcNew().Sqrt(pcDot(best, best))
		for i := range k {
			best[i].Quo(best[i], nrm)
		}
		basis = append(basis, best)
		out = append(out, best)
	}
	return out
}

// pcStaircase runs the exact staircase on copies of (A, E). With lo false a
// level zeroes the singular values ≤ τ_E; with lo true those ≤ ρ_j. It
// returns the finite count (a bound in the direction of lo when a level
// cannot be certified) and the level records.
func pcStaircase(n int, a0, e0 [][]*big.Float, normA, normE, tauA, tauE *big.Float, lo bool) (nf int, levels []poleCountLevel, reason string) {
	cp := func(m [][]*big.Float) [][]*big.Float {
		out := make([][]*big.Float, n)
		for i := range n {
			out[i] = make([]*big.Float, n)
			for j := range n {
				out[i][j] = pcSet(m[i][j])
			}
		}
		return out
	}
	a, e := cp(a0), cp(e0)
	sumSin := pcNew()
	two := pcFloat(2)
	prevD := n + 1
	k := n
	for level := 0; ; level++ {
		if k == 0 {
			return 0, levels, reason
		}
		sv, ok := pcJacobiRows(e[:k], k, a[:k])
		if !ok {
			reason = fmt.Sprintf("Jacobi unconverged at level %d", level)
		}
		rho := pcNew().Mul(two, pcNew().Mul(normE, sumSin))
		rho.Add(rho, tauE)
		thr := tauE
		if lo {
			thr = rho
		}
		lv := poleCountLevel{K: k, Rho: rho, SV: sv}
		for _, s := range sv {
			switch {
			case s.Cmp(tauE) <= 0:
				lv.Zero++
			case s.Cmp(rho) <= 0:
				lv.Undecided++
			default:
				lv.Nonzero++
			}
			if s.Cmp(thr) > 0 {
				lv.R++
			}
		}
		r := lv.R
		d := k - r
		if d == 0 {
			levels = append(levels, lv)
			return k, levels, reason
		}
		if d > prevD && reason == "" {
			reason = fmt.Sprintf("nullity rises at level %d (%d > %d)", level, d, prevD)
		}
		prevD = d
		for i := r; i < k; i++ {
			for j := range k {
				e[i][j] = pcNew()
			}
		}
		q := make([][]*big.Float, d)
		for i := range d {
			q[i] = make([]*big.Float, k)
			for j := range k {
				q[i][j] = pcSet(a[r+i][j])
			}
		}
		sa, _ := pcJacobiRows(q, k)
		lv.SigmaA = sa[d-1]
		if r > 0 {
			lv.Gap = pcNew().Sub(sv[r-1], sv[r])
		}
		levels = append(levels, lv)
		if lv.SigmaA.Cmp(tauA) <= 0 {
			if reason == "" {
				reason = fmt.Sprintf("A row block σ_min ≤ τ_A at level %d", level)
			}
			if lo {
				return 0, levels, reason
			}
			return k, levels, reason
		}
		if r == 0 {
			return 0, levels, reason
		}
		for i := range d {
			for j := range k {
				q[i][j].Quo(q[i][j], sa[i])
			}
		}
		zc := append(pcRowComplement(q, k), q...)
		mulZ := func(m [][]*big.Float) {
			for i := range k {
				row := make([]*big.Float, k)
				for j := range k {
					row[j] = pcDot(m[i][:k], zc[j])
				}
				copy(m[i][:k], row)
			}
		}
		mulZ(a)
		mulZ(e)
		for i := r; i < k; i++ {
			for j := range r {
				a[i][j] = pcNew()
			}
		}
		thQ := pcFloat(1)
		if lv.Gap.Cmp(tauE) > 0 {
			thQ = pcMinOne(pcNew().Quo(tauE, lv.Gap))
		}
		num := pcNew().Add(thQ, sumSin)
		num.Mul(num, normA)
		num.Add(num, tauA)
		sinZ := pcMinOne(num.Quo(num, pcNew().Sub(lv.SigmaA, tauA)))
		levels[len(levels)-1].SinZ = sinZ
		sumSin.Add(sumSin, sinZ)
		k = r
	}
}

// certifyFinitePoleCount returns the exact-arithmetic finite-pole count
// certificate of the stored n×n pencil (A, E), row-major.
func certifyFinitePoleCount(n int, a, e []float64) poleCountCert {
	A, E := pcMat(n, a), pcMat(n, e)
	normA, okA := pcSigma1(A)
	normE, okE := pcSigma1(E)
	tauA, tauE := pcTau(n, normA), pcTau(n, normE)
	var c poleCountCert
	var rh, rl string
	c.Hi, c.Levels, rh = pcStaircase(n, A, E, normA, normE, tauA, tauE, false)
	c.Lo, c.LoLevels, rl = pcStaircase(n, A, E, normA, normE, tauA, tauE, true)
	c.NF = c.Hi
	c.Decided = c.Lo == c.Hi
	switch {
	case !okA || !okE:
		c.Reason = "Jacobi unconverged on σ₁"
	case rh != "":
		c.Reason = "hi: " + rh
	case rl != "":
		c.Reason = "lo: " + rl
	}
	if c.Reason != "" {
		c.Decided = false
	}
	return c
}

// digest hashes the per-level records of both staircases, with every big
// value in exact binary ('p') form.
func (c poleCountCert) digest() string {
	var b strings.Builder
	txt := func(x *big.Float) string {
		if x == nil {
			return "-"
		}
		return x.Text('p', 0)
	}
	for _, run := range [][]poleCountLevel{c.Levels, c.LoLevels} {
		for _, lv := range run {
			fmt.Fprintf(&b, "%d %d %d %d %d %s %s %s %s", lv.K, lv.R, lv.Zero, lv.Undecided, lv.Nonzero, txt(lv.Rho), txt(lv.SigmaA), txt(lv.Gap), txt(lv.SinZ))
			for _, s := range lv.SV {
				b.WriteString(" " + txt(s))
			}
			b.WriteString("\n")
		}
		b.WriteString("|\n")
	}
	fmt.Fprintf(&b, "%d %d %d %v %q", c.NF, c.Lo, c.Hi, c.Decided, c.Reason)
	return fmt.Sprintf("%x", sha256.Sum256([]byte(b.String())))
}

// String summarises the certificate levels in float64 for logs.
func (c poleCountCert) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "NF=%d Lo=%d Hi=%d decided=%v %s", c.NF, c.Lo, c.Hi, c.Decided, c.Reason)
	for _, p := range []struct {
		name string
		run  []poleCountLevel
	}{{"hi", c.Levels}, {"lo", c.LoLevels}} {
		for j, lv := range p.run {
			sv := make([]float64, len(lv.SV))
			for i, s := range lv.SV {
				sv[i], _ = s.Float64()
			}
			f := func(x *big.Float) float64 {
				if x == nil {
					return math.NaN()
				}
				v, _ := x.Float64()
				return v
			}
			fmt.Fprintf(&b, "\n  %s L%d k=%d r=%d zones=%d/%d/%d rho=%.3g sigmaA=%.3g gap=%.3g sinZ=%.3g sv=%.4g",
				p.name, j, lv.K, lv.R, lv.Zero, lv.Undecided, lv.Nonzero, f(lv.Rho), f(lv.SigmaA), f(lv.Gap), f(lv.SinZ), sv)
		}
	}
	return b.String()
}
