package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// kdChainDescriptor is chainDescriptor with p outputs, m inputs, D = 0.3
// and Wf diagonal with Wf[0] = stiffRel: the chain entries of b past the
// kernel state are scaled by kd (kd = 0: proper), so the oracle's s-terms
// are of order kd.
func kdChainDescriptor(t *testing.T, rng *rand.Rand, nf int, chains []int, stiffRel, kd float64, p, m int) (sys, oracle *System) {
	t.Helper()
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil)
	for i := range nf {
		for r := range p {
			c.Set(r, i, rng.NormFloat64())
		}
		for r := range m {
			b.Set(i, r, rng.NormFloat64())
		}
		f, w := rng.NormFloat64(), 0.5+rng.Float64()
		if i == 0 {
			w = stiffRel
		}
		F.Set(i, i, f)
		W.Set(i, i, w)
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			for r := range p {
				c.Set(r, o+i, rng.NormFloat64())
			}
			for r := range m {
				v := rng.NormFloat64()
				if i > 0 {
					v *= kd
				}
				b.Set(o+i, r, v)
			}
		}
		o += k
	}
	U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			U.Set(i, j, rng.NormFloat64())
			V.Set(i, j, rng.NormFloat64())
		}
	}
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	D := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			D.Set(i, j, 0.3)
		}
	}
	sys, err := NewDescriptor(&A, &B, &C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

var kdChains = [][]int{{2}, {3}, {4}, {5}, {2, 1}, {3, 2}, {3, 1}}

// kdSeedDescriptor draws the kd family of the XSDPXI review: chains, nf,
// stiffRel 10^(−1..−5), kd 10^(−2..−10) for about half the seeds (else 1).
func kdSeedDescriptor(t *testing.T, seed uint64, mimo bool) (sys, oracle *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 4242))
	chains := kdChains[rng.IntN(len(kdChains))]
	nf := 1 + rng.IntN(3)
	sr := math.Pow(10, -1-4*rng.Float64())
	_ = rng.IntN(2)
	kd := 1.0
	if rng.Float64() >= 0.5 {
		kd = math.Pow(10, -2-8*rng.Float64())
	}
	_ = rng.Float64()
	p, m := 1, 1
	if mimo {
		p, m = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	return kdChainDescriptor(t, rng, nf, chains, sr, kd, p, m)
}

func jitterDescriptor(sys *System, rng *rand.Rand, ulps float64) *System {
	c := sys.Copy()
	for _, m := range []*mat.Dense{c.A, c.E, c.B, c.C} {
		raw := m.RawMatrix()
		for i := range raw.Rows {
			for j := range raw.Cols {
				raw.Data[i*raw.Stride+j] *= 1 + ulps*eps()*(2*rng.Float64()-1)
			}
		}
	}
	return c
}

// TestDescriptorStiffLeakKeepsGenuineSTerm: an improper index-3 chain
// (chains [3 1], b tail 9.6e-9, s² coefficient 2.2e-10) beside a finite
// pole 3.49e4 of E singular value 3.4e-5 relative. The s-term c·N·b is
// dominated by the stiff pole's leak (it moves 3× under 4ε jitter) but
// c·N²·b is the chain's own (it moves 7%). XSDPXI 059fbf8 absorbed both
// into the poles 3.49e4 and 1.83 (an exact fit, two terms, two unknowns),
// so IsProper said true and TF(1e4i) was 4.6% off. Oracle: a 400-bit solve
// of (sE − A) from the float data, −0.455617 − 0.048224i at 1e4i (the
// structured model gives −0.455633 − 0.047521i). Base eb8d9e1 read 0/30
// 8-ulp jittered copies proper.
func TestDescriptorStiffLeakKeepsGenuineSTerm(t *testing.T) {
	A := mat.NewDense(7, 7, []float64{1.4550415957907963, -0.6074275418734002, -1.5194925413997091, 2.6143577362222548, 0.41800818576196963, 1.9058975884310243, 0.8019975946207689, 1.0971115890031153, 1.6714606804806265, 3.54708390464768, -2.7515640958832375, -1.1647489026539128, -1.8047313927297357, 2.6537845128906583, 0.3703424784997346, -0.700297030596136, -0.4121180173368246, 2.652769913141961, 0.26027225723819214, -0.866567136514819, 0.19978633374090962, 0.05022193839607921, 1.2421982445296698, -3.3697580254690633, -1.013090741800605, 0.8555454458509284, 4.983342628066672, -0.19824228548189746, 1.5355798092250659, 0.6089871930714438, -4.508429583295725, 2.85460772063167, -0.7562482994689589, 0.37522815557573913, -0.6580703655894513, 1.2949976107472039, 2.3741712610436583, -1.694625616120018, -1.2297807139841193, -4.191859704607417, -2.6428219608377614, 0.9645951391661647, 0.11137961178380765, 0.0482235457346981, 3.5775467105229346, -2.5668891350912477, -1.9280797344926028, -3.1158129618206645, 1.0445755785886268})
	E := mat.NewDense(7, 7, []float64{2.327729875087586, -0.94036092273467, 1.4687732981631887, 1.3615710620479264, 1.6785132516231358, -2.348649328855017, -2.604460274924044, -1.4936485533631327, -0.3179675636469131, 0.6004718793159686, -0.5188200678501622, 1.0283370828977376, 1.5710723169215561, 0.42874446345793704, 2.9117534442573296, -0.5044705905396474, 0.8483630573312408, 1.6516953016817624, 1.000532834383117, -2.461175565660915, -2.0423162083113278, -3.474483872302602, -0.3136574257759236, -0.45247895964201207, -1.1230582050047941, -1.5950083120835228, 1.7560044661682688, 2.6195917383864926, -2.2267538614717615, 0.6597967254961264, -0.5949567518054691, -2.860390559488662, -0.4673799713713706, 0.9624139072766914, -1.0760808384218292, -6.1251768666028275, 0.6269180901098201, -1.6911734466033248, -5.45890355495126, -3.518075083933895, 1.2696489468011365, 0.08892506772495438, -1.674146159728552, 0.7029468010995102, -0.6485292116293299, -1.7302201536902975, -0.2706681521333612, 1.6203784883189989, 0.31263670739535104})
	B := mat.NewDense(7, 1, []float64{-2.822931408217144, -2.0769670026705858, -2.179861090537163, 2.580375311698919, 1.5311285681076074, 0.21257479973814358, -0.07743715072038702})
	C := mat.NewDense(1, 7, []float64{-0.211401528091813, -1.3723949681550884, 0.4232803828021012, -0.3611187058220649, 2.1029537970642167, 1.0505667661986493, -2.7039489492129274})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, []float64{0.3}), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if pr, err := sys.IsProper(); err != nil || pr {
		t.Fatalf("IsProper = %v, %v; want false", pr, err)
	}
	if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("Step: %v, want ErrImproperModel", err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !fzTFImproper(res.TF) {
		t.Fatal("TF proper, want improper")
	}
	g, err := res.TF.Eval(1e4i)
	if err != nil {
		t.Fatal(err)
	}
	if w := complex(-0.455617, -0.048224); !(cmplx.Abs(g[0][0]-w) <= 2e-3) {
		t.Fatalf("G(1e4i) = %v, want %v", g[0][0], w)
	}
	rng := rand.New(rand.NewPCG(371, 31))
	proper := 0
	for range 30 {
		if pr, err := jitterDescriptor(sys, rng, 8).IsProper(); err == nil && pr {
			proper++
		}
	}
	if proper > 1 {
		t.Fatalf("%d/30 8-ulp copies IsProper true, want ≤ 1", proper)
	}
}

// TestDescriptorStiffLeakImproperSweep: the kd family (improper chains, b
// tail 1e-2..1e-10, beside a pole of E singular value 1e-1..1e-5
// relative). XSDPXI 059fbf8 read 11/600 SISO and 7/600 MIMO (p, m ≤ 2)
// improper ones proper, base eb8d9e1 5 and 3. Fits into a group beyond the
// largest |λ| (seeds 371, 432, 542) and terms that move less than the leak
// (373; MIMO 149, 456, 542) are rejected now. Left: exact one-term fits
// (247, 315; base reads 2/30 and 7/30 jittered copies of those proper)
// and base's own (40, 48, 355, 472, 483; MIMO 40, 48, 127, 483), whose
// s-terms are 1e-10 relative. Counts allow one platform rounding flip.
func TestDescriptorStiffLeakImproperSweep(t *testing.T) {
	for _, tc := range []struct {
		mimo    bool
		pinned  []uint64
		maxRead int
	}{{false, []uint64{371, 432, 542, 373}, 8}, {true, []uint64{149, 456, 542}, 5}} {
		read := 0
		for seed := range uint64(600) {
			sys, _ := kdSeedDescriptor(t, seed, tc.mimo)
			pr, err := sys.IsProper()
			if err != nil || !pr {
				continue
			}
			read++
			if slices.Contains(tc.pinned, seed) {
				t.Errorf("mimo %v seed %d: improper IsProper true", tc.mimo, seed)
			}
		}
		if read > tc.maxRead {
			t.Errorf("mimo %v: %d/600 improper IsProper true, want ≤ %d", tc.mimo, read, tc.maxRead)
		}
	}
}

// TestDescriptorStiffLeakMIMO: XSDPXI for p or m = 2: proper chains beside
// a pole of E singular value 1e-2..1e-4 relative leak the stiff pole's
// residue ΔR into the nilpotent block on both sides, ΔR = C·X·β + γ·Yᵀ·B;
// with only the B side fitted, 059fbf8 read 46/220 improper (base eb8d9e1
// 78/220). A returned TF must match the structured oracle to 1e-5 up to
// 1e6i.
func TestDescriptorStiffLeakMIMO(t *testing.T) {
	falseRead, total := 0, 0
	for seed := range uint64(300) {
		rng := rand.New(rand.NewPCG(seed, 6161))
		chains := kdChains[rng.IntN(len(kdChains))]
		nf := 1 + rng.IntN(3)
		sr := math.Pow(10, -2-2*rng.Float64())
		p, m := 1+rng.IntN(2), 1+rng.IntN(2)
		if p == 1 && m == 1 {
			continue
		}
		total++
		sys, oracle := kdChainDescriptor(t, rng, nf, chains, sr, 0, p, m)
		pr, err := sys.IsProper()
		if err != nil || !pr {
			falseRead++
			continue
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("seed %d: TransferFunction: %v", seed, err)
		}
		for _, s := range []complex128{0.37i, complex(0.5, 3), 1e3i, 1e6i} {
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			w := pencilResponse(t, oracle, s)
			for i := range p {
				for j := range m {
					if d := cmplx.Abs(g[i][j] - w[i][j]); !(d <= 1e-5*math.Max(1, cmplx.Abs(w[i][j]))) {
						t.Fatalf("seed %d: G(%v)[%d][%d] = %v, want %v", seed, s, i, j, g[i][j], w[i][j])
					}
				}
			}
		}
	}
	if falseRead > 5 {
		t.Errorf("%d/%d proper MIMO chains IsProper false, want ≤ 5", falseRead, total)
	}
}
