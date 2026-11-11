package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// oracleStrictlyProper is the 320-bit C(sI−A)⁻¹B of sys at s, row-major
// p×m; nil at a pole.
func oracleStrictlyProper(sys *System, s complex128) []stepOracleComplex {
	o := newStepOraclePencil(sys)
	out := make([]stepOracleComplex, o.p*o.m)
	if o.n == 0 {
		for i := range out {
			out[i] = stepOracleComplex{stepOracleFloat(0), stepOracleFloat(0)}
		}
		return out
	}
	xr, xi := o.solveShifted(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))}, o.b)
	if xr == nil {
		return nil
	}
	for i := range o.p {
		for j := range o.m {
			r, q := stepOracleFloat(0), stepOracleFloat(0)
			for k := range o.n {
				r.Add(r, stepOracleNew().Mul(o.c[i][k], xr[k][j]))
				q.Add(q, stepOracleNew().Mul(o.c[i][k], xi[k][j]))
			}
			out[i*o.m+j] = stepOracleComplex{r, q}
		}
	}
	return out
}

// oracleDist is |a − b| for 320-bit values, b as stored.
func oracleDist(a, b stepOracleComplex) float64 {
	re, im := stepOracleNew().Sub(a.re, b.re), stepOracleNew().Sub(a.im, b.im)
	v := stepOracleNew().Add(stepOracleNew().Mul(re, re), stepOracleNew().Mul(im, im))
	f, _ := v.Float64()
	return math.Sqrt(f)
}

func oracleOf(z complex128) stepOracleComplex {
	return stepOracleComplex{stepOracleFloat(real(z)), stepOracleFloat(imag(z))}
}

// oracleTerms is, per entry of oracleStrictlyProper at s, the own-terms
// scale Σ_l |C_il||X_lj|, X = (sI−A)⁻¹B, at 320 bits; nil at a pole.
func oracleTerms(sys *System, s complex128) []float64 {
	o := newStepOraclePencil(sys)
	out := make([]float64, o.p*o.m)
	if o.n == 0 {
		return out
	}
	xr, xi := o.solveShifted(oracleOf(s), o.b)
	if xr == nil {
		return nil
	}
	for i := range o.p {
		for j := range o.m {
			t := stepOracleFloat(0)
			for k := range o.n {
				x := stepOracleNew().Add(stepOracleNew().Mul(xr[k][j], xr[k][j]), stepOracleNew().Mul(xi[k][j], xi[k][j]))
				t.Add(t, stepOracleNew().Mul(stepOracleNew().Abs(o.c[i][k]), stepOracleNew().Sqrt(x)))
			}
			out[i*o.m+j], _ = t.Float64()
		}
	}
	return out
}

// contractSweep is the largest exact |G_ij − R_ij|/(tol·max(scale,
// terms_ij)) of Reduce's contract over a dense axis sweep (8 points a
// decade, θ from 1e-16 to π on the unit circle, |s| from 1e-20 to 1e3
// times the largest |λ| on the imaginary axis, and the band's probes of
// sys) outside the shared-pole windows of red, with the point it is at;
// scale is the larger exact |G|, |R| of the point.
func contractSweep(sys, red *System, tol float64) (float64, complex128) {
	pts, poles, ok := axisProbes(sys, nil)
	if !ok {
		return math.Inf(1), 0
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	if sys.Dt != 0 {
		for e := -16.0; e < math.Log10(math.Pi); e += 0.125 {
			pts = append(pts, cmplx.Rect(1, math.Pow(10, e)))
		}
		pts = append(pts, 1, -1)
	} else {
		hi := 1.0
		for _, ev := range poles.evals {
			hi = math.Max(hi, cmplx.Abs(ev))
		}
		for e := -20.0; e <= 3; e += 0.125 {
			pts = append(pts, complex(0, hi*math.Pow(10, e)))
		}
		pts = append(pts, 0)
	}
	worst, at := 0.0, complex128(0)
	for _, s := range pts {
		if win.contain(s) {
			continue
		}
		g, r, tg := oracleStrictlyProper(sys, s), oracleStrictlyProper(red, s), oracleTerms(sys, s)
		if g == nil || r == nil || tg == nil {
			continue
		}
		scale := 0.0
		for i := range g {
			scale = math.Max(scale, math.Max(oracleDist(g[i], oracleOf(0)), oracleDist(r[i], oracleOf(0))))
		}
		for i := range g {
			if d := oracleDist(g[i], r[i]) / (tol * math.Max(scale, tg[i])); d > worst {
				worst, at = d, s
			}
		}
	}
	return worst, at
}

// bandStaircaseCut is the staircase cut of sys under opts with the
// structural pre-pass skipped, and its removed stage blocks.
func bandStaircaseCut(t *testing.T, sys *System, opts *ReduceOpts) (*System, []*mat.Dense) {
	t.Helper()
	cut, removed, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), opts, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	return cut.Sys, removed
}

// probeNear is the index of the point of pts nearest s.
func probeNear(pts []complex128, s complex128) int {
	k := -1
	for i, z := range pts {
		if k < 0 || cmplx.Abs(z-s) < cmplx.Abs(pts[k]-s) {
			k = i
		}
	}
	return k
}

// bandLiteral is a model of the AYT3ND census corpora (set/id) frozen as
// IEEE-754 bit patterns, row-major.
type bandLiteral struct {
	n, m, p    int
	dt         float64
	a, b, c, d []uint64
}

func (l bandLiteral) system(t *testing.T) *System {
	t.Helper()
	f := func(bits []uint64) []float64 {
		out := make([]float64, len(bits))
		for i, b := range bits {
			out[i] = math.Float64frombits(b)
		}
		return out
	}
	sys, err := NewFromSlices(l.n, l.m, l.p, f(l.a), f(l.b), f(l.c), f(l.d), l.dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// gwnbqeLiteral is the FuzzReduce model of
// TestReduceRemovesStructurallyDisconnectedStatesOfZeroTransfer.
var gwnbqeLiteral = bandLiteral{
	n: 4, m: 1, p: 1, dt: 100,
	a: []uint64{
		0x4000000000000000, 0xbfdfb537fc3ebd42, 0xbfd716d2e56f7b22, 0,
		0, 0x3ff0000000000000, 0, 0,
		0, 0xbfbff6c21bee9174, 0x3fe0000000000000, 0,
		0xbfdf1e69ec1e7775, 0xbf68a6ad13652450, 0x3fd8f240493ace37, 0x3ff8000000000000,
	},
	b: []uint64{0x3fdffdcc5ab614ff, 0, 0, 0x3fd34e17d6f2439b},
	c: []uint64{0, 0xbfab1702d86abaf9, 0x3fdffdcc5ab614ff, 0},
	d: []uint64{0x8000000000000000},
}

// bandLiterals are census models of NGG5TW (ruling literals), drawn on
// amd64 unless the key ends in @arm64: the corpora round differently there.
var bandLiterals = map[string]bandLiteral{
	"rotfast/13": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{0xc2f0e32ddd610a14, 0xc2f7984b8f95435c, 0x428c115d1a18a232, 0xc2f7984b8f954363, 0xc3007bb6f8fd1b8c, 0x42939ba72348d095, 0x428c115d1a189c5a, 0x42939ba72348d193, 0xc227533c91820143},
		b: []uint64{0x3fe50aad4ce04564, 0xbfde58bb6743a23d, 0xbfd8602902b03322},
		c: []uint64{0x3fd634955a925115, 0x3fd830a96657e2ec, 0x400698811b7b9b68},
		d: []uint64{0},
	},
	"matched-hid/144-unobs-true": {
		n: 6, m: 1, p: 1, dt: 0.001,
		a: []uint64{0xbfec37436f7109ce, 0x4055b60454ead832, 0xc0434c54b2678ce1, 0x4049bf660c11f9eb, 0xc03669649c978d17, 0xc043f1536e5bfc5f, 0xc0292e1c36e7ffe3, 0x40848118a9dceef6, 0xc0723fae89cc666e, 0x407857bdf0b4bab7, 0xc0651c477349b069, 0xc072d50bd8648283, 0xc02c7853758d8778, 0x40874de722ec0fb4, 0xc074b59410592d0d, 0x407bb433178fc8cc, 0xc0680b2754a0fcc2, 0xc075788113846607, 0x4010b3c7c495b439, 0xc069de5bb7e3d896, 0x405709d24f399306, 0xc05e7d891504323e, 0x404aa9c3b4d18828, 0x4057bb80c0a9d744, 0x4031e74bf3e8e880, 0xc08c1d47f422684c, 0x407905f32839202a, 0xc080ae966ee8066b, 0x406d30cdd11fc262, 0x4079ff32700ef80f, 0xc0321de78c16fefe, 0x408d2e096abf9b5c, 0xc079ff15bf92b114, 0x408155c3faf143b6, 0xc06e1f1bd0e7e3ec, 0xc07ad7a3ebecdf9b},
		b: []uint64{0xbfe1829edf3344a9, 0x3fd33e2be391d10a, 0x3fc5ece73f5763a5, 0xbfe78a0758125206, 0x3f8e49090f334360, 0xbfdc77f085f43031},
		c: []uint64{0x3e4d9f3956498f85, 0xbea7c3dcfaebe6fc, 0x3e952b7cdd2b4901, 0xbe9c3b8ef189ce4c, 0x3e88891d9783aac1, 0x3e95eadb53ad79cd},
		d: []uint64{0},
	},
	"nse-rot/171": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{
			0x42e10df0de1cf245, 0x42e9a3d83c5fe4a2, 0x42f6bf26c19c9355, 0x42e9a3d83c5fe456,
			0x42f3461bea6a972f, 0x430119528547d6e9, 0x42f6bf26c19c9358, 0x430119528547d6f1,
			0x430e56bb5a1b70ce,
		},
		b: []uint64{
			0x42ef6ff026937b55, 0x42f7a1b79b947134, 0x4304f70a54b8cec4,
		},
		c: []uint64{
			0xbfdb9db7d6ee84bf, 0xbfe29223aca38a1c, 0x3fed7a01844d932a,
		},
		d: []uint64{
			0xbfb0dff51e9a5250,
		},
	},
	"nse-rot/1517": {
		n: 2, m: 1, p: 1, dt: 0,
		a: []uint64{
			0xc30ca8e59742e496, 0xc31130ec8c9cdeea, 0xc31130ec8c9cdeec, 0xc3149fa6fee5293b,
		},
		b: []uint64{
			0x4322bb846b23bc1a, 0x432679089b78188f,
		},
		c: []uint64{
			0x3ff1ea69d04ea3e4, 0x3fccf82d8565f5ff,
		},
		d: []uint64{
			0xbfdc8cdd2b26ba40,
		},
	},
	"fz/7727": {
		n: 6, m: 1, p: 3, dt: 7.145549447652503,
		a: []uint64{
			0xbfa9ed078b87ae9a, 0x3fe15df6c751bc8a, 0x40072a40581d71f1, 0xc024c95d79a0cc62,
			0xc03d401b9d8f3d00, 0x404425faff46aafe, 0xbfb2d7f4cd8b5ac2, 0x3fe93edf377c183c,
			0x4010d640a6152608, 0xc02c864e560654b8, 0xc0443fa459b9a978, 0x404a0f288d229f86,
			0xbfac14c331e72845, 0x3fe2cf8e066d1ea5, 0x4009173af0846ada, 0xc01cd4dec4782a28,
			0xc035ffcba619d6da, 0x40297417d6e5879b, 0xbf912e89255bf950, 0x3fc704eab6851258,
			0x3feeb4452a215421, 0xc019ef9e71f4f35c, 0xc035489fc88bf3c3, 0x4055b83519cc2986,
			0x3fb94fa5a8ff641c, 0xbff0f489b09eb24a, 0xc0169da0ef32c1b2, 0x4035a8c0e03d52ff,
			0x404d7171d29a50ca, 0x4052b9a7ba3c3b96, 0xbfcfbe6aef6b36be, 0x400543aa0ffb6d54,
			0x402c5d08d1187372, 0xc0492d8e1c4359bd, 0xc060c6a1750eaab7, 0x4058d0a4eab2d7ac,
		},
		b: []uint64{
			0xbfeff30e4e5d45cf, 0xbfd48ee55fe6cd7d, 0x3fea733cc43fafc8, 0x3fe86e0b4be29d6e,
			0xbfdac2ad44860f7a, 0xbfef96979758fd4e,
		},
		c: []uint64{
			0xbf24c4b80df51842, 0xbf1fe17dc76410a0, 0x3f2f133e7407aaff, 0xbf009bcbdcf57a82,
			0xbf2c63f6982a2f8b, 0x3f2682433172bbbd, 0xbf2f4eaa4cdea3c9, 0x3f05d1f136792183,
			0x3f2bc7ba105d57f1, 0xbf276ac79bc9f2b9, 0xbf19474131d7461f, 0x3f2f96de6c3e98b4,
			0xbf2b1f3070663ca4, 0x3f2848ed846331f1, 0x3f16d6e8270a6061, 0xbf2fab2bf9f7e3e8,
			0x3f121cf309fbfb64, 0x3f29d0182c834fc3,
		},
		d: []uint64{
			0xc00b0fb4a25a90cb, 0x400941741d595f59, 0xc005fbaa0dd4db38,
		},
	},
	"kalman-d0/1582/4.1.1": {
		n: 6, m: 2, p: 2, dt: 0,
		a: []uint64{
			0xbff4ed8dbe5f2d76, 0xbfe936e0386f043a, 0x3fdf974643975815, 0x3f81843656fd11bc,
			0x3ffa72e26fe91513, 0xbffa67b421dffde6, 0xbfbb7a4499769485, 0x3ff44715230f12d8,
			0xbfd8d07a15418ec4, 0x3ff2620cc15b082b, 0x3ffb72a95d6113c0, 0xbfbbf785caedba29,
			0xbfed254df26edb6b, 0xbff5f3a45dea3694, 0xbfea8b1b4c9f2c94, 0x40000ab2a1bff81e,
			0x3ff1683291fa1811, 0x3fdb1c6b2b04ee4a, 0x3fe082f84804942a, 0x3fec5a1435285fb6,
			0x3ff6988d0cf6bc11, 0xc008e6d6d14cedfb, 0x3ffe77aea54b4632, 0x3fd60a5029b5a227,
			0x400113ba20a9492d, 0x3fc11e64e44c295c, 0xbfe8aa731342180b, 0x3feec6b622b6ac0b,
			0xbff591934c5555fb, 0x3fd771c0486500e7, 0xc00233d80cbaad5e, 0xc00848a0f9c5525c,
			0xbffdc125ec9323f2, 0x3fe52114c973a5bd, 0x3fafa8032ca8f3db, 0xc008b86d22503f48,
		},
		b: []uint64{
			0xc16e83c9f179cf59, 0x4156e0d7d9f464aa, 0xc17375384337cd74, 0xc18f9d71d4ef79aa,
			0xc190868193b28f09, 0xc182e63b6164454c, 0x41b4f82bd725d042, 0x419ea80326ac778e,
			0xc1925195624b77da, 0xc17f8efa725a33fc, 0x4187dac23cfd303e, 0xc1955d8ceea07a10,
		},
		c: []uint64{
			0xc0b7801380885e23, 0xc0b15a2e57ee5ff3, 0xc0901b0709ffa72d, 0xc0d4346e204d2926,
			0x40c5f1df5e2a8ad4, 0xc0a078e7db753551, 0xc07f22461ad6c6a7, 0xc0b528c8a404ea2e,
			0xc0c12df76a129247, 0x40c3391c5407d341, 0xc09d2b056f3e0d2a, 0xc0826ca4348198e4,
		},
		d: []uint64{
			0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
		},
	},
	"matched-hid/221-unc-true": {
		n: 6, m: 1, p: 1, dt: 0.01,
		a: []uint64{
			0xc01ac64eca4110e2, 0x401766bc251a335b, 0xc023d857bc5175de, 0xc01622e49eef9a35,
			0xc02e1da156e8ed97, 0x402f892de31c830b, 0xc03267fd1ad4a2f0, 0x402a06513e4a3948,
			0xc038d5c5c3cd8023, 0xc029723a654e1a9a, 0xc0426f95413ecc2c, 0x4043ea090e214606,
			0x404610acf55af642, 0xc03d6e3d9611b497, 0x404e1a0aafbf9dad, 0x403eac3dcaef0bc6,
			0x405608d082eb300a, 0xc057b61cbcb24ab6, 0xc03dcc6b13ac979f, 0x40345fabefca7c9b,
			0xc043da7236451536, 0xc033d8b9e9badd34, 0xc04da71ae152d4e4, 0x404fc32e0cc74cbf,
			0xc02b6fd65af09dc2, 0x40209bec3e958303, 0xc032dab2bee347a2, 0xc022994f6b21e535,
			0xc03ac3de1110eadd, 0x403e5546f21c18c3, 0x401e41ec8d36eb9b, 0xc01913eaf2df7f7b,
			0x4022f69561cc6e7f, 0x401653fcfa646fb5, 0x402d2fae603d823d, 0xc02bf6df9bff3c5e,
		},
		b: []uint64{
			0xbfc6901c83e0ca71, 0xbfbd26818cb78ec0, 0x3ff51e8e0a562972, 0x3fe1947158116f06,
			0xbfee88c4f5f2654a, 0x3fa2f5b5a26ce6f8,
		},
		c: []uint64{
			0xbf996a8097ec2b96, 0x3fbe5c37dffc25e5, 0x3f9704c3a96a882b, 0xbfa6b4a0b567b42f,
			0xbf78e88923264013, 0xbfab669332470be3,
		},
		d: []uint64{
			0x0000000000000000,
		},
	},
	"matched-hid/253-unc-true": {
		n: 6, m: 1, p: 1, dt: 0.01,
		a: []uint64{
			0x401c70d02978fb53, 0x4036104816c84459, 0xc02b0af8e34d23ad, 0x40468b71da9faf52,
			0xc056423fcb121cc0, 0xc012b23eb9bd633c, 0x40427a14b0915d51, 0x40609738cc8c28ff,
			0xc054b92f21bd5203, 0x407050da09b235d7, 0xc080506382a45a18, 0xc03b58b124fc17c9,
			0x4036978b90f88c46, 0x405424e1eea80763, 0xc048d35e6534ea44, 0x4063fcd7cec1f439,
			0xc073f948e34172c9, 0xc030bdd8afdd321f, 0x400c9a3cdfe63dc1, 0x402992f11ec2aaa0,
			0xc01ff80d7a5d42b1, 0x403a88eb180009b2, 0xc0497487ee1b84d5, 0xc0055bb2697cb088,
			0x402078316d80f5ec, 0x403d3fa616f1dae6, 0xc03289fce8a590e2, 0x404cb91ba6cc0eaa,
			0xc05c94c0db94d8ea, 0xc0181f43fa610106, 0xc009b21a962dd51b, 0xc0276e3e6c80456a,
			0x401bde784b0fa16e, 0xc038bcfa28d57569, 0x40481b34de913b4f, 0x400c33049da8f047,
		},
		b: []uint64{
			0x3fd105099dfc7ebf, 0xbff975c4aefc65ce, 0xbfe03d77b91792be, 0xbfdef12610d8f343,
			0xbfe254b7fb0bdcea, 0x3fe852b782bce520,
		},
		c: []uint64{
			0xbf7a04e69f590418, 0xbf8d8a0ad5632e27, 0x3f962bde2703aead, 0x3f931bd9becf65d6,
			0x3f5e5bb31e4cb86e, 0xbf3061908d1e6e73,
		},
		d: []uint64{
			0x0000000000000000,
		},
	},
	"matched-hid/462-unc-true": {
		n: 7, m: 1, p: 1, dt: 0.01,
		a: []uint64{
			0xc01ab0105a88b0ec, 0x3fd37a8472b98bd5, 0xc0115aadba70e623, 0xc01dbada64bdb93b,
			0xc01812b928136162, 0x4017300ca3da3d8f, 0xc014354dd3253914, 0xc03f7db484f003ee,
			0x40042812dd89b187, 0xc03194f430ea34cf, 0xc03e7efe1968bd59, 0xc0386eecad12d9f3,
			0x40377abfccf17df9, 0xc03458c5fcb92c20, 0x3ff95b85bd6979b0, 0x3fd8b5b3610d672e,
			0x40034ddd24482995, 0x3ffb0a258090fbe0, 0x3ffd3b1b992e0357, 0xbffc62950e6b9ae4,
			0x3ffc87f23c634676, 0x403556a67c87da19, 0xbff1c37aa1559f40, 0x4027ac645736ef70,
			0x4035a12315ddc25f, 0x40307ab3a0a98025, 0xc02fae2bf3913ffe, 0x402b5d3567ca1730,
			0x40302accde2db015, 0xbff8811ae6e831fb, 0x40205f0c30ef442c, 0x402ed72f28688ca6,
			0x402930e760e397f7, 0xc0263c14a89a5949, 0x40227e1094ae57eb, 0x405e0266660526f9,
			0xc017f18deb487c8d, 0x4050b53a35149cce, 0x405d0c6451b114f5, 0x40573b152cfc107c,
			0xc056141be5eba7cf, 0x405353c953375bb2, 0x4057f4131519c3e6, 0xc013097d51356380,
			0x404ab0a089da3ffa, 0x40573052929e85f7, 0x40528da21dd38fb2, 0xc051d4371150f815,
			0x404f5e34c365245e,
		},
		b: []uint64{
			0xbfdd028e88356d0a, 0xbfef02c3183c1a57, 0x3fdc4718b14638c5, 0x3fbb662479550502,
			0x3fe084c2b56ee308, 0x3fef53ffa7b1caa1, 0x3fe4d510431c0b60,
		},
		c: []uint64{
			0xbf75625d4df4802b, 0x3f760c6be62bcd58, 0x3f685c1d6bc91df6, 0xbf6bc63a75e41f48,
			0x3f64115a86e52174, 0xbf65d68790acf489, 0x3f7469a292f428b0,
		},
		d: []uint64{
			0x0000000000000000,
		},
	},
	"rotfast/13@arm64": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{0xc2f0e32ddd610a14, 0xc2f7984b8f95435c, 0x428c115d1a18a0d1, 0xc2f7984b8f954362, 0xc3007bb6f8fd1b8a, 0x42939ba72348cf9d, 0x428c115d1a189af8, 0x42939ba72348d09b, 0xc227533c9181fef6},
		b: []uint64{0x3fe50aad4ce04564, 0xbfde58bb6743a23c, 0xbfd8602902b03322},
		c: []uint64{0x3fd634955a92510e, 0x3fd830a96657e2e3, 0x400698811b7b9b67},
		d: []uint64{0},
	},
	"matched-hid/20-unobs-true@arm64": {
		n: 6, m: 1, p: 1, dt: 0.01,
		a: []uint64{0x3fe56e781e94e9e3, 0x3fe6184f461fdb2b, 0x40022020aa1059e8, 0x3fe670fb40f40ceb, 0xbf6e16e5dbbe6314, 0xbfe0362bbbacaecd, 0xbfeb717c7d525a97, 0x3ffe43d20c65b78c, 0x3ff11a953278cbcf, 0x3fff8bd72942e291, 0xbfee1ec4d0f85a51, 0xbfeeb672730e0108, 0xbfa390641067f6e7, 0xbfd7077d813de7b9, 0x3ffb0c61f9e85f35, 0xbfe0481b6e630653, 0xbfd39ddef6eae715, 0xbfd26849e0ad0c7a, 0xbfb629a8a95d5d26, 0x3f7b8f280d203beb, 0x40036489fb170a8e, 0x3fe3664420a7f86c, 0xbfcd9161f181206b, 0xbfe3c8d197e51808, 0x3fd310dc4ae91281, 0x3ff87a1e7e490468, 0x3fe20b74c4ce2975, 0x3ff119cf1f16b1ca, 0x3f9a9e3460a3bbac, 0xbfc10a94eb328977, 0xbfc5b1f70f256369, 0xbffa9edd47f28026, 0xbff2c1d2296bb990, 0xbff753e376551d3a, 0xbfd3928471486c08, 0x3ff0c438ef509d7d},
		b: []uint64{0x3fd27821a322fced, 0xbfdff49dd4005845, 0xbfe50e6fa8919313, 0x3feefe65915612c2, 0xbfb7720f67251c6f, 0x3fdef0030c71c0af},
		c: []uint64{0xbdaae5329c0586c1, 0xbdb9e8b8dad2a155, 0xbdb8a110346f383c, 0xbdb9cf8386999bab, 0xbd9a51cf0f30466a, 0x3db8d2a19973930d},
		d: []uint64{0},
	},
	"fz/13634@arm64": {
		n: 5, m: 3, p: 1, dt: 0.4961308755831542,
		a: []uint64{0x40c04b845cd6f463, 0x40f45fbcd4276ef5, 0x411957fe27253dd7, 0x4134f9795e0f925b, 0x414a0eea41cf7760, 0x40c600426d021c3b, 0x40fb826a2441d5ef, 0x41211c2e81e3b4cf, 0x413c52162bf9dfc9, 0x415197ac44f32528, 0x40d6ca2526147f87, 0x410c7e2ec9e7129f, 0x4131b8c464587da2, 0x414d5544d51d6c04, 0x416238ac969db4e2, 0x40ec1cf9d5062fb7, 0x41219312ce78a6fc, 0x4145dc9545078ae4, 0x416217bd4a638450, 0x41767a5eae1c6386, 0x40ffb1656bce41dc, 0x4133cff575431824, 0x4158a5380439e701, 0x41746584860a77a7, 0x41895718e2712e9d},
		b: []uint64{0x3fe6e97bcab09a03, 0xbfeed9b5db538ba4, 0x3fd0798bd0d18dcb, 0xbfdecad29152564e, 0xbfe0715809f7255a, 0x3fefff8d91fab869, 0xbfef26211d617605, 0x3fe60db8a3f4563e, 0x3fd1c3608f0c0d30, 0xbfa44bab0ac7b4a9, 0x3fec3dcaa2b53c1b, 0xbfeb3f228b5d2cd5, 0x3fee786663eada97, 0xbfcbc732e7df7c8c, 0xbfe7755e93cc9d83},
		c: []uint64{0x3eb65d8148138f63, 0xbe9a9d071cc99ac4, 0xbeb21036f063a6ee, 0x3eb2552d787a01c9, 0x3e98d6d0c7743070},
		d: []uint64{0x3ff22e1bd4cdb4a2, 0x3fe6357c9265e405, 0x3fc23491e1cdefc0},
	},
	"nse-rot/355@arm64": {
		n: 2, m: 1, p: 1, dt: 0,
		a: []uint64{0xc2f6ea644f8a9bc9, 0xc30e2d9f61eba9a7, 0xc30e2d9f61eba9b1, 0xc323df0c851abb1b},
		b: []uint64{0xc30726fae9d91d52, 0xc31e7d69bf73d831},
		c: []uint64{0xbff88f99b69f131d, 0xbffbd230cd50fb81},
		d: []uint64{0xbfd217dfadf43d5c},
	},
	"matched-hid/197-unobs-true": {
		n: 6, m: 1, p: 1, dt: 0.001,
		a: []uint64{0xc03b596e36bc5183, 0xc050b29c8574faef, 0xc0803470b0b82ce0, 0x407bc6c43be4e68b, 0xc052131fa5bb0b98, 0xc05fb4d233e165f5, 0xc068629ef499690c, 0xc07c33d1cc2dd4c6, 0xc0aa9ab0c221f069, 0x40a6ea96bcbd87b1, 0xc07e1db6e3878a91, 0xc089fc6d72d7eaf4, 0xc04a66e7081b566e, 0xc05ec664a11ef649, 0xc08cfbb22995d4cc, 0x4088f5370e5643f9, 0xc0606a19fda60e08, 0xc06c4ee133134ed2, 0xc056472b181edf34, 0xc069eeed351b2384, 0xc09870c357e9791e, 0x40950bc0c29a73dc, 0xc06ba873c58a0a89, 0xc077da60de3e7100, 0xc060ae3b73455987, 0xc0735fc6b47ae406, 0xc0a23cd579065447, 0x409f6782b95d216e, 0xc0748c444fb4bdfb, 0xc081c702a269e063, 0x4056ff6d6023ecc2, 0x406ab23eec2fccc9, 0x40992d00f342f3c0, 0xc095ad428d14d639, 0x406c6d0daf447928, 0x40789de94006cead},
		b: []uint64{0xbfdba92308d2d67a, 0xbfe536f448aa05e4, 0x3fbc607d019a523c, 0xbfd249c8caf74f40, 0xbff1a0c97fe545a8, 0xbfd6fc1cbb403d62},
		c: []uint64{0xbd4a5da3955a0706, 0xbd535db15410b60e, 0x3d47cd202d65b921, 0x3d4aa967218ef04e, 0xbd296feff9650b26, 0x3d4b770ffe5ef6fc},
		d: []uint64{0},
	},
	"nse-rot/355": {
		n: 2, m: 1, p: 1, dt: 0,
		a: []uint64{0xc2f6eaebfcdbb53a, 0xc30e2e520f3d2a5f, 0xc30e2e520f3d2a6a, 0xc323df822c249145},
		b: []uint64{0xc3072783fde44a73, 0xc31e7e1e453052a5},
		c: []uint64{0xbff88f99b69f131d, 0xbffbd230cd50fb82},
		d: []uint64{0xbfd217dfadf43d5c},
	},
	"nse-rot/1518": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{0xc2eb4b66432f0066, 0xc2f1bef3ed2c4a55, 0x42ea749d0f8e0973, 0xc2f1bef3ed2c4a1a, 0xc2f7135340b6e9eb, 0x42f1334e8ca60258, 0x42ea749d0f8e09a0, 0x42f1334e8ca6024d, 0xc2e9a46e0d24a874},
		b: []uint64{0xc2ffd90c54a827a1, 0xc304b4d0b12f17bb, 0x42fede6e69cc06eb},
		c: []uint64{0x3fdde000802a8135, 0xbf9a0863685c0360, 0xbfdcc947ef74e9e8},
		d: []uint64{0xbfd2d9e67cb13092},
	},
	"nse-rot/1197@arm64": {
		n: 3, m: 1, p: 1, dt: 0,
		a: []uint64{0x42e8c41959fa56a2, 0xc2e2da5996e51f3a, 0xc2dfea9d35ecf746, 0xc2e2da5996e51f19, 0x42dcb40a816e6451, 0x42d84bd0a3d8e9c9, 0xc2dfea9d35ecf709, 0x42d84bd0a3d8e9cb, 0x42d490d167b4cec6},
		b: []uint64{0xc310fdff2248cdcd, 0x4309deccf788951c, 0x4305e5edc5819d46},
		c: []uint64{0xbff09cddda757bf3, 0xbfe02a80e3d03f3b, 0xbfc04ec607cd7eef},
		d: []uint64{0x3fb0c0400da9cbc0},
	},
}

// TestBoundedResponseGEPPErrorWithinBound checks comp ≤ ver against the
// 320-bit error of the computed strictly proper response at the band's
// probes, for the model and its staircase cut, on models where GEPP's
// error exceeds the data resolution data = ε(|L||A||U| + |L||B| + |C||U|):
// the structurally sparse GWNBQE model (on arm64 its computed response
// near z = 1 is fill rounding, ≈1.6e-18 against a data bound of ≈1.8e-31;
// amd64 rounds it to zero), fz/7727 (A with zeros) and the unrotated
// staircase cut of kalman-d0/1582.
func TestBoundedResponseGEPPErrorWithinBound(t *testing.T) {
	for _, tc := range []struct {
		name      string
		lit       bandLiteral
		opts      *ReduceOpts
		dataModel bool
		dataCut   bool
	}{
		{"gwnbqe", gwnbqeLiteral, &ReduceOpts{Mode: ReduceUncontrollable}, false, false},
		{"fz/7727", bandLiterals["fz/7727"], &ReduceOpts{}, true, false},
		{"kalman-d0/1582", bandLiterals["kalman-d0/1582/4.1.1"], &ReduceOpts{}, false, true},
	} {
		sys := tc.lit.system(t)
		red, removed := bandStaircaseCut(t, sys, tc.opts)
		pts, _, ok := axisProbes(sys, removed)
		if !ok {
			t.Fatalf("%s: no probes", tc.name)
		}
		for side, s := range []*System{sys, red} {
			br := newBoundedResponse(s, pts)
			pm := br.p * br.m
			beyondData := 0
			for k, z := range pts {
				br.sharpen(k)
				exact := oracleStrictlyProper(s, z)
				if !br.ok[k] || exact == nil {
					continue
				}
				for i := range pm {
					x := k*pm + i
					e := oracleDist(exact[i], oracleOf(br.g[x]))
					if !(e <= br.comp[x]) || !(br.comp[x] <= br.ver[x]) {
						t.Errorf("%s side %d s=%v: error %.3g, comp %.3g, ver %.3g", tc.name, side, z, e, br.comp[x], br.ver[x])
					}
					if e > br.data[x] {
						beyondData++
					}
				}
			}
			if want := []bool{tc.dataModel, tc.dataCut}[side]; want && beyondData == 0 {
				t.Errorf("%s side %d: GEPP error within data at every probe", tc.name, side)
			}
		}
	}
}

// TestBoundedResponseResidualExact checks the Dot2 residual of sharpen
// against the 320-bit residual of the same Û on graded random models,
// |r̂ − r| ≤ u|r| + 2γ²_{k+3}·ρ, and that comp, ver, data and Ĝ are
// unchanged by a power-of-two diagonal similarity of the realization at
// points far enough off the spectrum that the pivot order is fixed.
func TestBoundedResponseResidualExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	e := eps()
	for c := range 200 {
		n, m, p := 1+rng.IntN(7), 1+rng.IntN(2), 1+rng.IntN(2)
		A, B, C := mat.NewDense(n, n, nil), mat.NewDense(n, m, nil), mat.NewDense(p, n, nil)
		for _, x := range []*mat.Dense{A, B, C} {
			raw := x.RawMatrix()
			for i := range raw.Data {
				if rng.IntN(4) > 0 {
					raw.Data[i] = rng.NormFloat64() * math.Pow(10, 8*rng.Float64()-4)
				}
			}
		}
		sys, err := New(A, B, C, mat.NewDense(p, m, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		norm := mat.Norm(A, 1) + 1
		pts := []complex128{complex(0, 0.3*norm), complex(0.1*norm, norm), complex(0, 1e3*norm)}
		br := newBoundedResponse(sys, pts)
		ev := br.ev
		ev.prepareSharpen()
		nm := n * m
		f := make([]float64, 7*nm)
		absU, au, rb, uh, ul := f[:nm], f[nm:2*nm], f[2*nm:3*nm], f[3*nm:5*nm], f[5*nm:]
		r := make([]complex128, nm)
		for k, s := range pts {
			if !br.ok[k] {
				continue
			}
			u := ev.u[k*nm : (k+1)*nm]
			ev.residual(s, u, uh, ul, absU, au, r, rb, m)
			for i := range n {
				kl := ev.rows[i+1] - ev.rows[i]
				g := gammaN(kl + 3)
				for j := range m {
					y := i*m + j
					re, im := stepOracleFloat(ev.br.b[y]), stepOracleFloat(0)
					rho := math.Abs(ev.br.b[y]) + au[y] + cmplx.Abs(s)*absU[y]
					for l := range n {
						a := stepOracleFloat(ev.br.a[i*n+l])
						re.Add(re, stepOracleNew().Mul(a, stepOracleFloat(real(u[l*m+j]))))
						im.Add(im, stepOracleNew().Mul(a, stepOracleFloat(imag(u[l*m+j]))))
					}
					us := stepOracleComplex{stepOracleFloat(real(u[y])), stepOracleFloat(imag(u[y]))}
					ss := oracleOf(s)
					re.Sub(re, stepOracleNew().Sub(stepOracleNew().Mul(ss.re, us.re), stepOracleNew().Mul(ss.im, us.im)))
					im.Sub(im, stepOracleNew().Add(stepOracleNew().Mul(ss.re, us.im), stepOracleNew().Mul(ss.im, us.re)))
					exact := stepOracleComplex{re, im}
					absR := oracleDist(exact, oracleOf(0))
					if d := oracleDist(exact, oracleOf(r[y])); !(d <= e*absR+2*g*g*rho) {
						t.Errorf("case %d s=%v (%d,%d): |r̂−r| %.3g above %.3g", c, s, i, j, d, e*absR+2*g*g*rho)
					}
				}
			}
		}
		rz := newRealizationCopy(sys, n, m, p)
		sc := newRealizationCopy(sys, n, m, p)
		exp := make([]int, n)
		for i := range exp {
			exp[i] = rng.IntN(7) - 3
		}
		for i := range n {
			for l := range n {
				sc.a[i*n+l] = math.Ldexp(rz.a[i*n+l], exp[l]-exp[i])
			}
			for j := range m {
				sc.b[i*m+j] = math.Ldexp(rz.b[i*m+j], -exp[i])
			}
			for j := range p {
				sc.c[j*n+i] = math.Ldexp(rz.c[j*n+i], exp[i])
			}
		}
		far := []complex128{complex(0, 1e3*norm), complex(1e3*norm, 1e3*norm)}
		b1, b2 := boundedResponseOf(rz, far), boundedResponseOf(sc, far)
		for k := range far {
			b1.sharpen(k)
			b2.sharpen(k)
			if !b1.ok[k] || !b2.ok[k] || !slices.Equal(b1.ev.piv, b2.ev.piv) {
				t.Fatalf("case %d s=%v: pivot order changed", c, far[k])
			}
			for x := k * p * m; x < (k+1)*p*m; x++ {
				if !ulpClose(b1.comp[x], b2.comp[x]) || !ulpClose(b1.ver[x], b2.ver[x]) || !ulpClose(b1.data[x], b2.data[x]) || b1.g[x] != b2.g[x] {
					t.Errorf("case %d s=%v: scaled comp/ver/data/g %v %v %v %v, want %v %v %v %v", c, far[k], b2.comp[x], b2.ver[x], b2.data[x], b2.g[x], b1.comp[x], b1.ver[x], b1.data[x], b1.g[x])
				}
			}
		}
	}
}

// singlePointResponse is a sharpened 1×1 boundedResponse at one point.
func singlePointResponse(g complex128, comp, ver, data float64) *boundedResponse {
	return &boundedResponse{
		p: 1, m: 1,
		g: []complex128{g}, comp: []float64{comp}, ver: []float64{ver}, data: []float64{data}, terms: []float64{0},
		ok: []bool{true}, sharp: []bool{true},
	}
}

// TestClassifyProbeChargesModelBound checks that verification charges the
// model's own bound: with δ_G = 1.2·tol·scale and |Ĝ−R̂| = 0.2·tol·scale
// the exact deviation can reach 1.4·tol, so the point is uninformative,
// where charging only δ_R verified it. The data resolution is slack on
// refutation inside a shared-pole window only: there a deviation of
// 3·tol·scale below δ_G,data = 5·tol·scale (a rounding-level move of a
// shared pole) is not refuted, and is with δ_G,data = 0; outside a window
// the stored model is the reference and it is refuted either way.
func TestClassifyProbeChargesModelBound(t *testing.T) {
	tol := math.Sqrt(eps())
	full, cut := singlePointResponse(1, 1.2*tol, 1.2*tol, 0.2*tol), singlePointResponse(complex(1+0.2*tol, 0), 0, 0, 0)
	if v := classifyProbe(full, cut, 0, tol, false); v != probeUninformative {
		t.Errorf("charged model bound: verdict %d, want uninformative", v)
	}
	full, cut = singlePointResponse(1, 0.1*tol, 0.1*tol, 5*tol), singlePointResponse(complex(1+3*tol, 0), 0.1*tol, 0.1*tol, 0)
	if v := classifyProbe(full, cut, 0, tol, true); v != probeUninformative {
		t.Errorf("deviation within data resolution in a window: verdict %d, want uninformative", v)
	}
	if v := classifyProbe(full, cut, 0, tol, false); v != probeReject {
		t.Errorf("deviation within data resolution outside a window: verdict %d, want reject", v)
	}
	full.data[0] = 0
	if v := classifyProbe(full, cut, 0, tol, true); v != probeReject {
		t.Errorf("deviation beyond data resolution in a window: verdict %d, want reject", v)
	}
}

// TestClassifyProbeRefutesBeyondTerms checks the refutation scale outside a
// shared-pole window: an entry off by comp_G + comp_R + 2·tol·scale is
// refuted outside a window although δ_G,data = 10·tol·scale, and is
// uninformative inside one; an entry whose deviation is within tol of its
// own terms Σ|C||Û| but far above tol·scale, as at a transmission zero
// where the response cancels, is uninformative.
func TestClassifyProbeRefutesBeyondTerms(t *testing.T) {
	tol := math.Sqrt(eps())
	gc, rc := 0.1*tol, 0.2*tol
	full := singlePointResponse(1, gc, 1.5*tol, 10*tol)
	cut := singlePointResponse(complex(1-(gc+rc+2*tol), 0), rc, rc, 0)
	if v := classifyProbe(full, cut, 0, tol, false); v != probeReject {
		t.Errorf("outside a window: verdict %d, want reject", v)
	}
	if v := classifyProbe(full, cut, 0, tol, true); v != probeUninformative {
		t.Errorf("inside a window: verdict %d, want uninformative", v)
	}
	g := complex(1e-6, 0)
	full = singlePointResponse(g, 1e-3*tol, 1e-3*tol, 0)
	full.terms[0] = 1
	cut = singlePointResponse(g+complex(0.5*tol, 0), 1e-3*tol, 1e-3*tol, 0)
	if v := classifyProbe(full, cut, 0, tol, false); v != probeUninformative {
		t.Errorf("deviation 0.5·tol·terms at a zero: verdict %d, want uninformative", v)
	}
	full.terms[0] = 0.1
	if v := classifyProbe(full, cut, 0, tol, false); v != probeReject {
		t.Errorf("deviation 5·tol·terms: verdict %d, want reject", v)
	}
}

// TestClassifyProbeExactZeroCutIsDetermined checks the band on the GWNBQE
// model with the structural pre-pass skipped. The ReduceUncontrollable
// staircase cut has C = 0 exactly: at every probe its response and bound
// are exactly zero, which is an exact interpolation condition, not noise.
// The model's response is exact zero too except at the two probes beside
// the pole z = 1 (e^{0.01i}, e^{0.1i}), where it is fill rounding within
// comp: those are uninformative, the other seven verify, and the band
// accepts. The
// ReduceUnobservable cut carries the staircase's rounding in B (R ≈ 1e-16
// against G_sp = 0) and is refuted.
func TestClassifyProbeExactZeroCutIsDetermined(t *testing.T) {
	sys := gwnbqeLiteral.system(t)
	tol := math.Sqrt(eps())
	red, removed := bandStaircaseCut(t, sys, &ReduceOpts{Mode: ReduceUncontrollable})
	pts, poles, ok := axisProbes(sys, removed)
	if !ok {
		t.Fatal("no probes")
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	for k, s := range pts {
		want := probeVerified
		if cmplx.Abs(s-1) < 0.2 {
			want = probeUninformative
		}
		if v := classifyProbe(&full, &cut, k, tol, win.contain(s)); v != want {
			t.Errorf("z=%v: verdict %d, want %d (Ĝ %v comp %.3g, R̂ %v comp %.3g)", s, v, want, full.g[k], full.comp[k], cut.g[k], cut.comp[k])
		}
	}
	if !frequencyAxisCutInBand(sys, red, removed, tol) {
		t.Error("band rejects the exact ReduceUncontrollable cut")
	}
	red, removed = bandStaircaseCut(t, sys, &ReduceOpts{Mode: ReduceUnobservable})
	if frequencyAxisCutInBand(sys, red, removed, tol) {
		t.Error("band accepts the ReduceUnobservable cut")
	}
}

// strictBandRecheck installs bandAcceptHook for the test and reports every
// verified entry of an accepted cut whose 320-bit deviation |G_sp − R_sp|
// exceeds tol·scale.
func strictBandRecheck(t *testing.T, name string) {
	t.Helper()
	bandAcceptHook = func(sys, red *System, tol, cutoff float64, pts []complex128, verified []bool, slow, fast int, full, cut *boundedResponse) {
		pm := full.p * full.m
		for k, s := range pts {
			if !verified[k] {
				continue
			}
			g, r := oracleStrictlyProper(sys, s), oracleStrictlyProper(red, s)
			if g == nil || r == nil {
				t.Errorf("%s: verified probe %v at a pole", name, s)
				continue
			}
			scale := 0.0
			for i := range pm {
				scale = math.Max(scale, math.Max(cmplx.Abs(full.g[k*pm+i]), cmplx.Abs(cut.g[k*pm+i])))
			}
			for i := range pm {
				if d := oracleDist(g[i], r[i]); d > tol*scale {
					t.Errorf("%s: verified probe %v entry %d: exact |G−R| %.3g above tol·scale %.3g", name, s, i, d, tol*scale)
				}
			}
		}
	}
	t.Cleanup(func() { bandAcceptHook = nil })
}

// TestReduceBandStrictClaimAtVerifiedProbes checks, at every probe the
// band verifies for an accepted cut, that the exact responses of the model
// and the cut agree to tol·scale, on census models where the band verified
// probes beyond it by not charging the model's GEPP error (matched-hid
// 221 by 10.6×, 462 by 1.3×, nse-rot/1517 by 1.18×; 253 is a cut the
// ruling's sample lost under the strict claim). 1517's cut is lost: its
// exact deviation exceeds tol·scale at s ≈ 1.5e7i.
func TestReduceBandStrictClaimAtVerifiedProbes(t *testing.T) {
	modes := []*ReduceOpts{nil, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}, {Equalize: true}}
	for _, name := range []string{"matched-hid/221-unc-true", "matched-hid/253-unc-true", "matched-hid/462-unc-true", "nse-rot/1517"} {
		sys := bandLiterals[name].system(t)
		strictBandRecheck(t, name)
		for _, opts := range modes {
			if _, err := sys.Reduce(opts); err != nil {
				t.Fatalf("%s %+v: %v", name, opts, err)
			}
		}
	}
	bandAcceptHook = nil
	sys := bandLiterals["nse-rot/1517"].system(t)
	if res, _ := sys.Reduce(nil); res.Order != 2 {
		t.Errorf("1517: order %d, want 2", res.Order)
	}
	red, removed := bandStaircaseCut(t, sys, &ReduceOpts{})
	pts, _, _ := axisProbes(sys, removed)
	k := probeNear(pts, 1.5e7i)
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	g, r := oracleStrictlyProper(sys, pts[k]), oracleStrictlyProper(red, pts[k])
	scale := math.Max(cmplx.Abs(full.g[k]), cmplx.Abs(cut.g[k]))
	if d := oracleDist(g[0], r[0]); !(d > math.Sqrt(eps())*scale) {
		t.Errorf("1517 cut at %v: exact |G−R| %.3g within tol·scale %.3g", pts[k], d, math.Sqrt(eps())*scale)
	}
}

// TestBoundedResponseRefinementStep checks the refinement step at the
// matched-hid/221 probe beside z = 1, where the unrefined model response
// is off by ≈3.9e-8 against tol·scale ≈ 3.8e-9 (the error main's band left
// uncharged): unrefined, both the actual error and ver exceed tol·|Ĝ|;
// one step of û += δ brings the actual error below 1e-11·|Ĝ|, within the
// refined ver. ver itself stays near ε·|L̂|ρ there, so the probe does not
// verify.
func TestBoundedResponseRefinementStep(t *testing.T) {
	sys := bandLiterals["matched-hid/221-unc-true"].system(t)
	_, removed := bandStaircaseCut(t, sys, &ReduceOpts{})
	pts, _, _ := axisProbes(sys, removed)
	k := probeNear(pts, complex(0.99999999756, 6.98e-5))
	before, after := newBoundedResponse(sys, pts), newBoundedResponse(sys, pts)
	before.sharpenAt(k, false)
	after.sharpenAt(k, true)
	exact := oracleStrictlyProper(sys, pts[k])[0]
	tol := math.Sqrt(eps())
	g := cmplx.Abs(after.g[k])
	if e := oracleDist(exact, oracleOf(before.g[k])); !(e > tol*g) || !(before.ver[k] > tol*g) {
		t.Errorf("unrefined: error %.3g, ver %.3g, want both above tol·|Ĝ| %.3g", e, before.ver[k], tol*g)
	}
	if e := oracleDist(exact, oracleOf(after.g[k])); !(e < 1e-11*g) || !(e <= after.ver[k]) {
		t.Errorf("refined: error %.3g, want below 1e-11·|Ĝ| %.3g and ver %.3g", e, 1e-11*g, after.ver[k])
	}
}

// TestReduceBandRefutesBeyondDataResolution checks that the band rejects
// the nse-rot/171 cut, which misfits the model by 0.10 beyond its data
// resolution: at s ≈ 3.456i the deviation (0.129) exceeds tol·scale. The
// stored model is the reference, so its data bound (0.120) is no slack and
// the sharpened bounds refute it without the refinement step.
func TestReduceBandRefutesBeyondDataResolution(t *testing.T) {
	sys := bandLiterals["nse-rot/171"].system(t)
	if res, err := sys.Reduce(nil); err != nil || res.Order != 3 {
		t.Fatalf("Reduce: order %v err %v, want the unreduced model", res, err)
	}
	red, removed := bandStaircaseCut(t, sys, &ReduceOpts{})
	tol := math.Sqrt(eps())
	pts, poles, _ := axisProbes(sys, removed)
	win, _ := sharedPoleWindows(poles, red, tol)
	k := probeNear(pts, 3.456i)
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	if v := classifyProbe(&full, &cut, k, tol, win.contain(pts[k])); v != probeReject {
		t.Errorf("s=%v: verdict %d, want reject", pts[k], v)
	}
	full, cut = newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	full.sharpenAt(k, false)
	cut.sharpenAt(k, false)
	if v := classifyProbe(&full, &cut, k, tol, win.contain(pts[k])); v != probeReject {
		t.Errorf("s=%v: unrefined bounds: verdict %d, want reject", pts[k], v)
	}
	if frequencyAxisCutInBand(sys, red, removed, tol) {
		t.Error("band accepts the nse-rot/171 cut")
	}
}

// ulpClose reports whether a and b agree to an ulp.
func ulpClose(a, b float64) bool {
	return a == b || math.Abs(a-b) <= eps()*math.Abs(a)
}

// TestClassifyProbeWindowRefutationCarriesCutResolution checks the
// refutation inside a shared-pole window, where the cut's pole is known
// only to its rounding: an entry whose deviation exceeds both responses'
// comp, the model's data bound and tol·scale by 2·tol·scale is
// uninformative when the cut's data bound is 3·tol·scale and refuted when
// it is 0.5·tol·scale. Outside a window the cut's data bound is no slack.
func TestClassifyProbeWindowRefutationCarriesCutResolution(t *testing.T) {
	tol := math.Sqrt(eps())
	gc, rc, gd := 0.1*tol, 0.2*tol, 0.3*tol
	diff := gc + rc + gd + 2*tol
	for _, tc := range []struct {
		rd       float64
		inShared bool
		want     probeVerdict
	}{
		{3 * tol, true, probeUninformative},
		{0.5 * tol, true, probeReject},
		{3 * tol, false, probeReject},
	} {
		full := singlePointResponse(1, gc, gc, gd)
		cut := singlePointResponse(complex(1-diff, 0), rc, rc, tc.rd)
		if v := classifyProbe(full, cut, 0, tol, tc.inShared); v != tc.want {
			t.Errorf("data_R %.2g·tol, inShared %v: verdict %d, want %d", tc.rd/tol, tc.inShared, v, tc.want)
		}
	}
}

// TestClassifyProbeCutNoiseByDataBound checks the cut-noise rule on its
// data half at s = 0 for the ReduceUnobservable staircase cut of the
// rotated near-singular-E fold 1518, which drops a slow state carrying
// O(1) of the response: the cut's sharpened comp is far below |R̂|, but
// its data bound is above it, so its slow dynamics are not determined by
// its own data and the point rejects. Where the gap is resolved (arm64)
// the probes omit s = 0 and the sided count rejects the cut instead
// (TestReduceResolvedGapPerSideCount). An exactly zero response with zero
// bounds is not noise.
func TestClassifyProbeCutNoiseByDataBound(t *testing.T) {
	sys := nearSingularEFoldOf(t, 1518, true)
	tol := math.Sqrt(eps())
	red, removed := bandStaircaseCut(t, sys, &ReduceOpts{Mode: ReduceUnobservable})
	pts, poles, ok := axisProbes(sys, removed)
	if !ok {
		t.Fatal("no probes")
	}
	win, _ := sharedPoleWindows(poles, red, tol)
	k := probeNear(pts, 0)
	if pts[k] != 0 {
		t.Skipf("no probe at s = 0 (nearest %v): the gap is resolved here and the sided count rejects", pts[k])
	}
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	full.sharpen(k)
	cut.sharpen(k)
	rs := cmplx.Abs(cut.g[k])
	if !(cut.comp[k] < 0.1*rs) || !(cut.data[k] >= rs) {
		t.Errorf("cut at s = 0: comp %.3g, data %.3g, want comp < 0.1·|R̂| and data ≥ |R̂| = %.3g", cut.comp[k], cut.data[k], rs)
	}
	if v := classifyProbe(&full, &cut, k, tol, win.contain(0)); v != probeReject {
		t.Errorf("s = 0: verdict %d, want reject", v)
	}
	if cutNoise(0, 0, 0) {
		t.Error("exact zero read as noise")
	}
}

// TestReduceBandSureUninformativeSkipKeepsVerdicts checks the skip of
// frequencyAxisCutAcrossGap on the staircase cuts of the band literals in
// every mode: no probe outside a shared-pole window is skipped, since the
// stored model is the reference there and a sharpened bound can refute,
// and at every window probe probeSurelyUninformative skips, the sharpened
// classifyProbe finds the probe uninformative too. The literals put few
// probes in windows, so a synthetic window point within both data bounds
// is checked as well.
func TestReduceBandSureUninformativeSkipKeepsVerdicts(t *testing.T) {
	models := map[string]*System{"gwnbqe": gwnbqeLiteral.system(t)}
	for name, l := range bandLiterals {
		models[name] = l.system(t)
	}
	for name, sys := range reduceVerifyLiterals(t) {
		models[name] = sys
	}
	tol := math.Sqrt(eps())
	skipped := 0
	for name, sys := range models {
		for _, opts := range []*ReduceOpts{{}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}, {Equalize: true}} {
			cut, removed, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), opts, true, -1)
			if err != nil {
				continue
			}
			red := cut.Sys
			pts, poles, ok := axisProbes(sys, removed)
			if !ok {
				continue
			}
			win, ok := sharedPoleWindows(poles, red, tol)
			if !ok {
				continue
			}
			full, rc := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
			for k, s := range pts {
				in := win.contain(s)
				if !in && probeSurelyUninformative(&full, &rc, k, tol, false) {
					t.Errorf("%s %+v s=%v: skipped outside a shared-pole window", name, opts, s)
				}
				if !full.ok[k] || !rc.ok[k] || probeVerifies(&full, &rc, k, tol) || !probeSurelyUninformative(&full, &rc, k, tol, in) {
					continue
				}
				skipped++
				if v := classifyProbe(&full, &rc, k, tol, win.contain(s)); v != probeUninformative {
					t.Errorf("%s %+v s=%v: skipped probe sharpens to verdict %d", name, opts, s, v)
				}
			}
		}
	}
	tol2 := 2 * tol
	full := singlePointResponse(1, 0.5*tol2, 0.5*tol2, 2*tol2)
	cut := singlePointResponse(complex(1+3*tol2, 0), 0.5*tol2, 0.5*tol2, 2*tol2)
	if !probeSurelyUninformative(full, cut, 0, tol2, true) {
		t.Error("window point within both data bounds not skipped")
	}
	if v := classifyProbe(full, cut, 0, tol2, true); v != probeUninformative {
		t.Errorf("skipped window point: verdict %d, want uninformative", v)
	}
	t.Logf("%d corpus probes skipped", skipped)
}

// TestReduceRefusesCutWrongBelowVerifiedProbes freezes census cuts that
// verify at n + nr probes yet are wrong against the stored model where no
// probe verifies: rotfast 13 (ReduceUnobservable,
// gap cut, 0.1–0.35% of the response at s = 0 against verified probes at
// |s| ≥ 1e6, both corpora), matched-hid 20- and 197-unobs-true (the
// matched-C2D zero z = −1 missed by 149 and 2.2e4·tol·terms), fz 13634
// (16·tol·scale at two probes the cut's own bound leaves undecided) and
// nse-rot 355 (200% of the response at s = 0). Reduce must keep the order
// on every arch, and the exact oracle confirms each staircase cut off by
// more than tol·max(scale, terms) somewhere on the axis outside the
// shared-pole windows. matched-hid 144-unobs-true (ReduceOpts{} and
// ReduceUnobservable) is off by a few percent of the response at θ = 4e-7
// inside the z ≈ 1 cluster, below the band's lowest probe, beyond the data
// bound but within the contract (0.55·tol·terms: the cluster's terms are
// 1e8 times the response); the band keeps it, its cluster probes being
// undecided.
func TestReduceRefusesCutWrongBelowVerifiedProbes(t *testing.T) {
	tol := math.Sqrt(eps())
	unobs := &ReduceOpts{Mode: ReduceUnobservable}
	for _, tc := range []struct {
		name     string
		opts     *ReduceOpts
		contract bool
	}{
		{"matched-hid/144-unobs-true", &ReduceOpts{}, true},
		{"matched-hid/144-unobs-true", unobs, true},
		{"rotfast/13", unobs, false},
		{"rotfast/13@arm64", unobs, false},
		{"matched-hid/20-unobs-true@arm64", &ReduceOpts{}, false},
		{"matched-hid/20-unobs-true@arm64", unobs, false},
		{"matched-hid/197-unobs-true", &ReduceOpts{}, false},
		{"matched-hid/197-unobs-true", unobs, false},
		{"fz/13634@arm64", &ReduceOpts{}, false},
		{"fz/13634@arm64", unobs, false},
		{"nse-rot/355", &ReduceOpts{}, false},
		{"nse-rot/355@arm64", &ReduceOpts{}, false},
	} {
		sys := bandLiterals[tc.name].system(t)
		n, _, _ := sys.Dims()
		res, err := sys.Reduce(tc.opts)
		if err != nil {
			t.Fatalf("%s %+v: %v", tc.name, tc.opts, err)
		}
		if res.Order != n {
			c, at := contractSweep(sys, res.Sys, tol)
			t.Errorf("%s %+v: order %d, want %d (off by %.3g·tol·max(scale, terms) at %v)", tc.name, tc.opts, res.Order, n, c, at)
		}
		red, _ := bandStaircaseCut(t, sys, tc.opts)
		if c, at := contractSweep(sys, red, tol); !(c > 1) != tc.contract {
			t.Errorf("%s %+v: staircase cut off by %.3g·tol·max(scale, terms) at %v, within the contract %v", tc.name, tc.opts, c, at, tc.contract)
		}
	}
}
