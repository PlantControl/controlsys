package controlsys

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// poleCountCases returns the literal table plus the in-repo stiff-chain
// cases, whose structural count is the number of nonzero diagonal entries
// of the structured W.
func poleCountCases() []poleCountLiteral {
	out := slices.Clone(poleCountLiterals)
	for _, set := range [][]stiffChainCase{stiffChainResplitCases, stiffChainClosedCases, largePoleAbsorbCases} {
		for _, tc := range set {
			want := 0
			for i := range tc.n {
				if tc.w[i*tc.n+i] != 0 {
					want++
				}
			}
			out = append(out, poleCountLiteral{name: tc.name, n: tc.n, want: want, a: tc.a, e: tc.e})
		}
	}
	return out
}

// poleCountGate reports the certificate's violations of Lo ≤ want ≤ Hi and
// of its own invariants.
func poleCountGate(c poleCountCert, want int) []string {
	var bad []string
	if c.Hi < want {
		bad = append(bad, fmt.Sprintf("Hi %d < structural %d", c.Hi, want))
	}
	if c.Lo > want {
		bad = append(bad, fmt.Sprintf("Lo %d > structural %d", c.Lo, want))
	}
	if c.Lo > c.Hi {
		bad = append(bad, fmt.Sprintf("Lo %d > Hi %d", c.Lo, c.Hi))
	}
	if c.NF != c.Hi {
		bad = append(bad, fmt.Sprintf("NF %d != Hi %d", c.NF, c.Hi))
	}
	if c.Decided && (c.Lo != c.Hi || c.NF != want) {
		bad = append(bad, fmt.Sprintf("decided NF %d (Lo %d, Hi %d) != structural %d", c.NF, c.Lo, c.Hi, want))
	}
	undecided := 0
	for _, lv := range c.Levels {
		undecided += lv.Undecided
	}
	if undecided == 0 && c.Lo != c.Hi {
		bad = append(bad, fmt.Sprintf("no undecided value on the Hi path but Lo %d != Hi %d", c.Lo, c.Hi))
	}
	return bad
}

// TestFinitePoleCountCertificate_BracketsStructuralCount: on the parked
// stiff-chain literals (ELT76F, IP4PC3, F4OPPD, in-repo resplit cases) the
// exact staircase brackets the structural pole count, Lo ≤ want ≤ Hi, and
// never decides a wrong count. The ELT76F ruling gates only the literals
// the 2⁻¹⁰⁰ sensitivity classifier (S·2⁻⁵³ < 1) confirms; this gates every
// literal, and checks the level records hash as on arm64, amd64 and -race.
// Logged beside: the classifier verdicts and the float64 deflateStiffChain
// count, one short of Hi = structural on two 5093/5552/12/357 (ELT76F).
func TestFinitePoleCountCertificate_BracketsStructuralCount(t *testing.T) {
	skipSlowUnderRace(t)
	cases := poleCountCases()
	ruledAmbiguous, bandAmbiguous, decided := 0, 0, 0
	var digests strings.Builder
	for _, l := range cases {
		c := certifyFinitePoleCount(l.n, l.a, l.e)
		digests.WriteString(c.digest())
		for _, b := range poleCountGate(c, l.want) {
			t.Errorf("%s: %s\n%v", l.name, b, c)
		}
		if c.Decided {
			decided++
		}
		cl := classifyPencilRoots(l.n, l.a, l.e)
		if !cl.OK {
			t.Errorf("%s: Aberth iteration did not converge", l.name)
		}
		fin, gray, smear := cl.bandCounts()
		if cl.trueCount() != l.want {
			ruledAmbiguous++
		}
		if gray > 0 || fin != l.want {
			bandAmbiguous++
		}
		_, _, _, _, fnf, _, _, _, fok := deflateStiffChain(mat.NewDense(l.n, l.n, l.a), mat.NewDense(l.n, l.n, l.e))
		t.Logf("%s n=%d structural=%d Lo=%d Hi=%d decided=%v | S·u<1 count=%d | band finite/gray/smear=%d/%d/%d | float deflateStiffChain nf=%d ok=%v | S=%.2g",
			l.name, l.n, l.want, c.Lo, c.Hi, c.Decided, cl.trueCount(), fin, gray, smear, fnf, fok, cl.S)
	}
	t.Logf("%d literals: decided %d; oracle-ambiguous %d under S·u<1, %d under the memo band", len(cases), decided, ruledAmbiguous, bandAmbiguous)
	const wantDigest = "3a023662f51be79a581aaea427fb130ceaa016bc235889c1e78e377dc0632ce2"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(digests.String()))); got != wantDigest {
		t.Errorf("combined level-record digest %s, want %s (arm64, amd64 and -race agree on it)", got, wantDigest)
	}
}

// TestFinitePoleCountCertificate_DecidesExactStructure: on exactly
// structured pencils P·(blkdiag(F, I), blkdiag(W, N))·Q with permutations
// P, Q, dense F, diagonal W and nilpotent chains N, every chain value is an
// exact zero and the certificate decides NF = nf.
func TestFinitePoleCountCertificate_DecidesExactStructure(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 54))
	shapes := [][]int{{1}, {2}, {3}, {2, 1}, {3, 2}, {4}}
	for run := range 12 {
		nf := rng.IntN(4)
		chains := shapes[rng.IntN(len(shapes))]
		n := nf
		for _, k := range chains {
			n += k
		}
		f, w := make([]float64, n*n), make([]float64, n*n)
		for i := range nf {
			for j := range nf {
				f[i*n+j] = rng.NormFloat64()
			}
			w[i*n+i] = 0.5 + rng.Float64()
		}
		o := nf
		for _, k := range chains {
			for i := range k {
				f[(o+i)*n+o+i] = 1
				if i+1 < k {
					w[(o+i)*n+o+i+1] = 1
				}
			}
			o += k
		}
		p, q := rng.Perm(n), rng.Perm(n)
		a, e := make([]float64, n*n), make([]float64, n*n)
		for i := range n {
			for j := range n {
				a[i*n+j], e[i*n+j] = f[p[i]*n+q[j]], w[p[i]*n+q[j]]
			}
		}
		c := certifyFinitePoleCount(n, a, e)
		if !c.Decided || c.NF != nf {
			t.Errorf("run %d nf=%d chains=%v: %v", run, nf, chains, c)
		}
	}
}

// TestFinitePoleCountCertificate_DigestStable: the per-level records are
// exact big.Float values and must hash identically on arm64, amd64 and
// under -race (ELT76F determinism constraint).
func TestFinitePoleCountCertificate_DigestStable(t *testing.T) {
	want := map[string]string{
		"two:5093:0": "ef1679cd6c05d87b41a9ab05bb7270edc3500eb3e282a0420567624a6853ab90",
		"cpx:1297:2": "f5968f8506356ca5f65977b86222a3cff596ee916e634c0a8dd4f1fc6b74fe38",
	}
	for _, l := range poleCountLiterals {
		w, ok := want[l.name]
		if !ok {
			continue
		}
		if got := certifyFinitePoleCount(l.n, l.a, l.e).digest(); got != w {
			t.Errorf("%s: digest %s, want %s", l.name, got, w)
		}
	}
}

// poleCountSweepDraws is the per-generator draw count of the chain sweep,
// POLECOUNT_SWEEP when set.
func poleCountSweepDraws() int {
	if v, err := strconv.Atoi(os.Getenv("POLECOUNT_SWEEP")); err == nil && v > 0 {
		return v
	}
	return 100
}

// TestFinitePoleCountCertificate_ChainSweep: chainDescriptor and
// condChainDescriptor pencils (stiff finite pole 1e-2..1e-9 relative beside
// index-2..4 chains). Where Lo ≤ nf ≤ Hi fails, the sensitivity classifier
// decides: a case it confirms (S·2⁻⁵³ < 1 count = nf) fails the test, any
// other is listed as oracle-ambiguous. Logs per generator the decided rate
// and the log10(s/ρ_j) distribution of the level ≥ 1 values above τ_E.
// The gate holds on these two generators only: on the IP4PC3 rvb two/cpx
// family (two stiff poles sharing a block with σ_min < τ_E) Hi drops a
// well-conditioned pole at a level ≥ 1 (ELT76F step 2).
func TestFinitePoleCountCertificate_ChainSweep(t *testing.T) {
	skipSlowUnderRace(t)
	draws := poleCountSweepDraws()
	shapes := [][]int{{2}, {3}, {4}, {2, 1}, {3, 1}, {2, 2}}
	for _, gen := range []string{"chain", "condChain"} {
		decided, hiExact, failures, ambiguous := 0, 0, 0, 0
		hist := map[int]int{}
		for seed := range uint64(draws) {
			rng := rand.New(rand.NewPCG(seed, 5454))
			nf := 1 + rng.IntN(3)
			chains := shapes[rng.IntN(len(shapes))]
			sr := math.Pow(10, -2-7*rng.Float64())
			proper := rng.IntN(2) == 0
			var sys *System
			if gen == "chain" {
				sys, _, _ = chainDescriptor(t, rng, nf, chains, sr, proper)
			} else {
				sys, _, _ = condChainDescriptor(t, rng, nf, chains, sr, proper, 2*rng.Float64())
			}
			n, _ := sys.A.Dims()
			a, e := mat.DenseCopyOf(sys.A).RawMatrix().Data, mat.DenseCopyOf(sys.E).RawMatrix().Data
			c := certifyFinitePoleCount(n, a, e)
			if c.Decided {
				decided++
			}
			if c.Hi == nf {
				hiExact++
			}
			for j, lv := range c.Levels {
				if j == 0 {
					continue
				}
				for _, s := range lv.SV {
					if s.Cmp(c.Levels[0].Rho) > 0 {
						r, _ := pcNew().Quo(s, lv.Rho).Float64()
						hist[int(math.Floor(math.Log10(r)))]++
					}
				}
			}
			bad := poleCountGate(c, nf)
			if len(bad) == 0 {
				continue
			}
			cl := classifyPencilRoots(n, a, e)
			if cl.OK && cl.trueCount() == nf {
				failures++
				t.Errorf("%s seed %d nf=%d chains=%v stiffRel=%.3g: %s\n%v", gen, seed, nf, chains, sr, strings.Join(bad, "; "), c)
				continue
			}
			ambiguous++
			t.Logf("%s seed %d nf=%d: oracle-ambiguous (S·u<1 count %d), %s", gen, seed, nf, cl.trueCount(), strings.Join(bad, "; "))
		}
		keys := make([]int, 0, len(hist))
		for k := range hist {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		var b strings.Builder
		for _, k := range keys {
			fmt.Fprintf(&b, " [1e%d,1e%d):%d", k, k+1, hist[k])
		}
		t.Logf("%s: %d draws, decided %d (%.0f%%), Hi = nf %d, confirmed failures %d, oracle-ambiguous %d; level>=1 s/rho:%s",
			gen, draws, decided, 100*float64(decided)/float64(draws), hiExact, failures, ambiguous, b.String())
	}
}
