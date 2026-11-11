package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type leastSquaresFrozenCase struct {
	name    string
	n       int
	a, b, c []float64
	d       float64
	ceil    float64
}

func (tc leastSquaresFrozenCase) system(t *testing.T) *System {
	t.Helper()
	sys, err := New(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, 1, tc.b), mat.NewDense(1, tc.n, tc.c), mat.NewDense(1, 1, []float64{tc.d}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func checkLeastSquaresFrozenFit(t *testing.T, cases []leastSquaresFrozenCase) {
	t.Helper()
	for _, tc := range cases {
		sys := tc.system(t)
		if k, limit, err := leastSquaresIntegratorLimit(sys, true, 0.1); err != nil || k != 0 {
			t.Errorf("%s: integrators %d limit %g err %v, want 0", tc.name, k, limit, err)
			continue
		}
		d, fit, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares})
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		poles, err := d.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range poles {
			if p == 1 {
				t.Errorf("%s: fit holds a pole at z = 1", tc.name)
			}
		}
		if !(fit.MaxRelativeError <= tc.ceil) {
			t.Errorf("%s: MaxRelativeError %g, want ≤ %g", tc.name, fit.MaxRelativeError, tc.ceil)
		}
	}
}

// Stored poles resolved from s = 0 by their rounding bounds, |λ| 1e-7…1e-4
// beside poles of size 1e2…1e4 (fz census rows, arm64 models), or the slow
// pole of a rotated stiff fold (nse-rot 1076): the integrator count is 0
// under contract (C) (NDFJC6), where main held them at z = 1 with a better
// fit. The fit is free and its MaxRelativeError is capped at the measured
// head value (+0.1%) as a regression ceiling; a fit parameterization that
// pins such poles is AIFFAD.
func TestC2DFitLeastSquaresFreeFitOfPolesResolvedFromZero(t *testing.T) {
	checkLeastSquaresFrozenFit(t, []leastSquaresFrozenCase{
		// base k=3: MaxRelativeError 0.362 (arm64 model) / 0.0037 (amd64 model); head k=0: 0.0042 (arm64 model; 0.0157 on the amd64 model, where it crosses 1e-2).
		{
			name: "nse-rot-7704/1076",
			n:    4,
			a:    []float64{-0x1.139da87f9b955p+51, 0x1.138644381c773p+50, -0x1.6bc23117900cdp+47, 0x1.1145ba0559d14p+49, 0x1.138644381c76cp+50, -0x1.136ee1ecd81fp+49, 0x1.6ba351be9a0c8p+46, -0x1.112e88a88281ep+48, -0x1.6bc23117900aap+47, 0x1.6ba351be9a0bfp+46, -0x1.e016ece4a88ccp+43, 0x1.68aa6683cc12ap+45, 0x1.1145ba0559d1dp+49, -0x1.112e88a882811p+48, 0x1.68aa6683cc09ep+45, -0x1.0ef2e56940916p+47},
			b:    []float64{0x1.f4f6786507b6bp+49, -0x1.f4cbf40639f6ap+48, 0x1.4a9605c1bdc35p+46, -0x1.f0b406fcf1ab3p+47},
			c:    []float64{0x1.3f9798332f6fcp-01, 0x1.ac07b7648784dp-01, 0x1.62d613af89935p-05, -0x1.17f9d141e3bdfp+00},
			d:    -0x1.170277ef4738bp-03,
			ceil: 0.00421,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 5.68e-07, exact 5.68e-07; head k=0: 0.00253.
		{
			name: "fz/62",
			n:    6,
			a:    []float64{0x1.128ad3ba0aaeap+09, -0x1.a0d9913af8d2cp+06, -0x1.62d488a320ac8p+10, -0x1.dfc1cf6bc3d42p+09, -0x1.a3ef43f56209p+10, -0x1.a5ec1235a6c3dp-02, -0x1.9f9debd6b6f3fp+06, 0x1.3b865defc0e2cp+04, 0x1.0c949d13aeb9bp+08, 0x1.6b228418ce144p+07, 0x1.3ddbb699e32dep+08, 0x1.3f56a20fa10a3p-04, -0x1.62964ac32f36dp+10, 0x1.0d3156b11dfabp+08, 0x1.ca4894702b36cp+11, 0x1.35d0cd2a99f8fp+11, 0x1.0f2f3af567478p+12, 0x1.107694f2c1a3bp+00, -0x1.dfa4a43bd9c5dp+09, 0x1.6c21bfecf04c8p+07, 0x1.35f4a5cb6eb6p+11, 0x1.a701a935a700ep+10, 0x1.6fcd4aa6c7796p+11, 0x1.72b5f900e2437p-01, -0x1.a3b3998ad2e81p+10, 0x1.3ea0379f31d9bp+08, 0x1.0f385a71297d7p+12, 0x1.6faf344f6f963p+11, 0x1.4139c5b36ba9ep+12, 0x1.5662ea545f58p+00, -0x1.7845c3cc28fd9p-07, 0x1.1da7dd1884a0dp-09, 0x1.e64f8a7814d2fp-06, 0x1.49a34bfec1fa6p-06, -0x1.4b7b6a958bad3p-05, -0x1.781ed18b97766p-09},
			b:    []float64{-0x1.8f8e42f748ce5p-01, 0x1.9336f077fde12p-02, 0x1.fb6a46e5ccd0ep-01, 0x1.176f377b08004p-03, -0x1.d60a7bcd86a05p-01, -0x1.415454bcb61b6p-01},
			c:    []float64{0x1.7b2453cf56c63p-18, -0x1.36b76a76d9bf1p-20, -0x1.48ea8c9514a9ep-18, 0x1.2259771b026b9p-18, 0x1.1a5d55f63c3fcp-19, -0x1.7da28f6638c2cp-18},
			d:    0x1.cfd2a1ce8b84p-01,
			ceil: 0.00253,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.000777, exact 0.000406; head k=0: 0.0629.
		{
			name: "fz/1306",
			n:    6,
			a:    []float64{0x1.a754a0f942156p+15, -0x1.8467b5807d8dfp+14, 0x1.239ea3873dffdp+16, 0x1.6b3db3a68d26bp+16, -0x1.6ca11dfcedf16p+13, 0x1.0b9bbd2d284a8p+14, 0x1.1e379fedec88ep+15, -0x1.12da4e12b2ddep+14, 0x1.e8a6f772fe77p+14, 0x1.dd84ef4a98fbdp+15, -0x1.4e12111e72667p+12, 0x1.35c68cf86c667p+13, -0x1.44650ecca99e5p+13, 0x1.cb58469755362p+12, -0x1.26c9b666d0603p+13, -0x1.ca2d6c72e19b6p+13, 0x1.b1e1d192777afp+10, -0x1.47cdeaa6898fdp+11, -0x1.2b61475b68a6p+15, 0x1.28b30742507e8p+13, -0x1.241d401d44a9cp+16, -0x1.275e373314a81p+16, 0x1.5410e487b7329p+13, -0x1.cb01fe3229926p+13, 0x1.8769965a2d9bdp+14, -0x1.83e86067c32efp+12, 0x1.7de9c474c8312p+15, 0x1.81686e7fcc351p+15, -0x1.bfb5a2a6ceeacp+12, 0x1.2ba0f08514023p+13, -0x1.236aaa487d4e3p-13, 0x1.20ceab2c82903p-15, -0x1.1c581cd01d95p-12, -0x1.1ef236901f7afp-12, -0x1.26663d0709597p-12, 0x1.5e8df7819be94p-10},
			b:    []float64{-0x1.0bf45240fc50dp-01, -0x1.ec10345bf707ep-01, 0x1.2ce820a9b91fdp-07, 0x1.ee942463f2098p-01, 0x1.03e5928b2a26ep-01, -0x1.6388d63673cd6p-01},
			c:    []float64{-0x1.2bfc78cfa4e8dp-04, 0x1.2f15fb28dcbc3p-02, -0x1.e3e3c774a6c66p-04, -0x1.c1bc3717c9da6p-03, 0x1.0a5dfd1fd8047p-02, 0x1.a51f11e4f0acap-05},
			d:    -0x1.a723675184184p-01,
			ceil: 0.063,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.00506, exact 0.00251; head k=0: 0.136.
		{
			name: "fz/2981",
			n:    6,
			a:    []float64{0x1.025a31129ac1ep+13, 0x1.4f5f794d43727p+13, 0x1.bdf55383af59cp+11, -0x1.70abcf40076f9p+12, -0x1.61b974f6db774p+17, -0x1.0bed9b000edaep+19, -0x1.98b58072ffd68p+13, -0x1.093d7a30079f1p+14, -0x1.60ec707960128p+12, 0x1.23ac7c74cf66ep+13, 0x1.17c90628e4b98p+18, 0x1.a7d8a696aa7bap+19, -0x1.00b3fb599b86ap+14, -0x1.4d28e4f4866bcp+14, -0x1.bb7b972a01de1p+12, 0x1.6e9c100f27021p+13, 0x1.5f6d951a193fep+18, 0x1.0a30d685d6cebp+20, -0x1.71fb140984197p+13, -0x1.e0295eb5bb9bep+13, -0x1.3f9d02e92e48bp+12, 0x1.0826b85015e16p+13, 0x1.fa83ef226b422p+17, 0x1.7fa952c45cc67p+19, -0x1.1b303e9461972p+11, -0x1.6f85bbf8c4a1cp+11, -0x1.e945e4141155ap+09, 0x1.945021c2015c5p+10, 0x1.83b41b5a69aa7p+15, 0x1.25aa9dd14b53bp+17, -0x1.987fef768dd89p-21, -0x1.0913670c6af33p-20, -0x1.60e37526cc34ap-22, 0x1.239c6db0de2e9p-21, 0x1.4b5cb6ebd71e3p-19, -0x1.629a7a3e925ap-16},
			b:    []float64{0x1.b2b2f18e4dd82p-01, 0x1.78f155eed0b34p-01, -0x1.d21230abc79b7p-02, -0x1.f59dbde4b3f56p-01, -0x1.1a9fa5c6c112ap-04, 0x1.e2b73f1c7360fp-01},
			c:    []float64{-0x1.758206afc8ae6p+00, -0x1.7824c503399adp+02, 0x1.509566ee05ef2p+02, 0x1.3d08097fccb3bp+01, -0x1.b713986660e85p+02, 0x1.f58733f2fb2cp+00},
			d:    0x1.4106ecc97a971p+00,
			ceil: 0.136,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.000397, exact 0.00037; head k=0: 0.0177.
		{
			name: "fz/7716",
			n:    6,
			a:    []float64{-0x1.1272bb7d1df17p+02, 0x1.166970b25301fp-02, -0x1.0dfe9b063e0e3p+05, -0x1.a2a2c5283332ap+02, -0x1.0c571f616eb0dp+07, -0x1.3e9ad8f2f56bdp+06, -0x1.00270f1b9507ap+02, 0x1.03da1f891ce47p-02, -0x1.f7fdce2e6770ep+04, -0x1.86bb18756598cp+02, -0x1.fb0960397fd37p+06, -0x1.9f4035220c6b7p+06, 0x1.1332f2b0c2f74p+01, -0x1.172c6e91fb034p-03, 0x1.0ebbac273b1p+04, 0x1.a3d40bfc6d423p+01, 0x1.4095fa7ddc75p+06, 0x1.1f76566ca5cf6p+08, 0x1.29fefef7c3a94p+02, -0x1.2e4cc4449e4ffp-02, 0x1.252900a0fb60ep+05, 0x1.c67abfb8ac1a9p+02, 0x1.2b64f6b08bb9dp+07, 0x1.476246c466d01p+07, -0x1.829a10e38ea4ap+02, 0x1.882f6f871ce75p-02, -0x1.7c53f81c2bc8p+05, -0x1.26ddda50a4c1ep+03, -0x1.8016e5c4ca784p+07, 0x1.6f601312aa682p+09, -0x1.cecc7816cbb0cp+04, 0x1.d57b8f15b86aep+00, -0x1.c749d20016ba4p+07, -0x1.60fba8bcda9cfp+05, -0x1.c36d1d3f6ddap+09, -0x1.55b96207f6ba9p+08},
			b:    []float64{0x1.4e69f1709601cp-02, 0x1.ff051d61573adp-01, 0x1.a8c012cc1c2afp-03, -0x1.c635ba8790f19p-01, -0x1.5d3054acfd8a2p-01, 0x1.0b650a8543c6dp-01},
			c:    []float64{-0x1.419ca96a8d45cp-07, 0x1.618db545d2ac8p-06, -0x1.0f2d6e59addfdp-08, -0x1.35b8188d6f187p-06, 0x1.0c0d62d97cf0dp-06, 0x1.10cde1073be0ap-07},
			d:    0x1.7d7657441ef41p-01,
			ceil: 0.0177,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.00018, exact 5.99e-05; head k=0: 0.0576.
		{
			name: "fz/9355",
			n:    6,
			a:    []float64{0x1.1ced2c714ce13p+11, 0x1.ec2ea35924b44p+10, 0x1.bb00cf0ad46d1p+08, 0x1.2b3f3145a7aecp+12, 0x1.26e047c6d18e7p+14, -0x1.dc56fe3615ec4p+13, 0x1.c34fa2d97f8e4p+06, 0x1.85afbd0f1dd47p+06, 0x1.5da0afee25ef8p+04, 0x1.d94a86c69183ep+07, 0x1.d32a36fb5a69p+09, -0x1.954500f76a776p+09, 0x1.a50322c845689p+10, 0x1.6ba090ff4ac12p+10, 0x1.47473bccb5676p+08, 0x1.ba2994270be75p+11, 0x1.b3b73a75af475p+13, -0x1.5ebe0f91f5fb5p+13, 0x1.1b777be40c4fap+09, 0x1.e9a7c9b8ab8abp+08, 0x1.b8acf703c37f6p+06, 0x1.29b270b2e3062p+10, 0x1.255ded135180ep+12, -0x1.e9edb55a5bf4p+11, 0x1.4730d310aff7cp+08, 0x1.1a977bb22727bp+08, 0x1.fca6103727c0ep+05, 0x1.579db2e93a7e8p+09, 0x1.529e3f07b9abbp+11, -0x1.57f65ece46eb8p+11, 0x1.c8cfab27928f6p+09, 0x1.8a8b364d2d87bp+09, 0x1.6313f45a110ap+07, 0x1.dfbe5a9511096p+10, 0x1.d8c41ecfb9721p+12, -0x1.9cc355f42283p+12},
			b:    []float64{0x1.33dde38b08032p-01, 0x1.dc8b75f9978b3p-01, -0x1.a754d323f377ep-04, -0x1.f8dadc9c48f69p-01, -0x1.b25c0ce1ff065p-02, 0x1.84aa0c6dce9a1p-01},
			c:    []float64{-0x1.cca5715884f4ep-06, 0x1.6fa1fd5541debp-03, -0x1.683eec4b2bfc8p-04, -0x1.f6568fbd9c7ebp-04, 0x1.568608e599881p-03, 0x1.db35202a20b3fp-07},
			d:    0x1.13204760f086ap+00,
			ceil: 0.0577,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.000825, exact 0.000772; head k=0: 0.0247.
		{
			name: "fz/13152",
			n:    6,
			a:    []float64{0x1.9a716374af6e2p+11, -0x1.627ef4dcc4388p+11, 0x1.3f093847a11d6p+09, -0x1.af5fa7b7af3dep+12, 0x1.96f355ece248ap+14, 0x1.590332da1dbe2p+13, 0x1.424f1cfb0873p+10, -0x1.16601e84ef5e3p+10, 0x1.f50f426de1cf1p+07, -0x1.52d201a72cd24p+11, 0x1.47da4872be414p+13, 0x1.26617c2e1594cp+12, -0x1.6c40acb14bb6p+11, 0x1.3a9a0e35c3129p+11, -0x1.1b21e8588e976p+09, 0x1.7edc330a36301p+12, -0x1.6cb8813a0777ep+14, -0x1.3c485d1a4325dp+13, 0x1.ecf4f07b032fcp+11, -0x1.a9c32e256eb48p+11, 0x1.7f2c7e54e5559p+09, -0x1.02e16c91dd363p+13, 0x1.f2c5cf86ba917p+14, 0x1.bbabee4d8c1bfp+13, 0x1.20d3ffc8f134ap+10, -0x1.f2eaa57efd545p+09, 0x1.c102ab2aaeec9p+07, -0x1.2f53dbcf196afp+11, 0x1.284430171a849p+13, 0x1.0f8511a77dba3p+12, -0x1.6c3822a1ec8e2p+11, 0x1.3a92ae19fb11fp+11, -0x1.1b1b4556db17cp+09, 0x1.7e80e1582db89p+12, -0x1.78f0b3aa05de4p+14, -0x1.5fe3e2d500994p+13},
			b:    []float64{-0x1.82345a6532533p-01, 0x1.b9298efd24427p-02, 0x1.f8370489b9a3fp-01, 0x1.89618adf42476p-04, -0x1.dde85d335b59cp-01, -0x1.30da241605315p-01},
			c:    []float64{-0x1.bba2148816302p-11, 0x1.b0a34e608d1a3p-13, 0x1.75b3100ea16fbp-11, -0x1.5dc9273061a41p-11, -0x1.271212079312fp-12, 0x1.bd2dd2ba1d72p-11},
			d:    0x1.9ad9006dcbb33p+00,
			ceil: 0.02474,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 0.000643, exact 0.000607; head k=0: 0.377.
		{
			name: "fz/15841",
			n:    4,
			a:    []float64{-0x1.3591ac601e089p+13, 0x1.08a3a2c3f8d0fp+13, 0x1.2cdeecaa79237p+13, 0x1.4f1190c382eaep+16, -0x1.467eb95b22b33p+13, 0x1.16d3a4013cb24p+13, 0x1.3cff52dff3676p+13, 0x1.61437c4fd3687p+16, 0x1.1bd422ec5eff8p+12, -0x1.ea47d58efaacbp+11, -0x1.16bde291ed343p+12, -0x1.3453d497e249dp+15, -0x1.489146267474cp-08, 0x1.1bc7e407f8e79p-08, -0x1.fec9fab00fe19p-11, 0x1.590298d5e0558p-07},
			b:    []float64{0x1.d1c269ba995ap-05, 0x1.f454a5c06ce14p-01, 0x1.dd21abc8388f6p-02, -0x1.74b2d2d3f97dbp-01},
			c:    []float64{0x1.b3ef31b551834p-06, -0x1.26f42254d0d91p-05, -0x1.b419cf72a1b53p-09, 0x1.3893b69255a46p-05},
			d:    -0x1.b2b9abc81fb4bp-04,
			ceil: 0.377,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 5.79e-05, exact 5.11e-05; head k=0: 0.0132.
		{
			name: "fz/18425",
			n:    6,
			a:    []float64{0x1.a9a9390728998p+12, 0x1.ba76ff9f7be95p+13, -0x1.71a490103ac9cp+13, -0x1.4ec839a246ba3p+11, -0x1.f650b14623482p+08, -0x1.258977aa88897p+12, -0x1.b8f8132862b4cp+10, -0x1.ca606ae77d52cp+11, 0x1.7eef049d4a5a1p+11, 0x1.5acf77792bdb6p+09, 0x1.042e21358430ep+07, 0x1.301794f48835dp+10, 0x1.aa650affd4989p+10, 0x1.bb3a6369eacecp+11, -0x1.7248b8eac693ap+11, -0x1.4f6030ea730c3p+09, -0x1.f733017c9af9p+06, -0x1.260b81749c232p+10, -0x1.3dfbd1cea7c6fp-02, -0x1.4a898317d87aep-01, 0x1.250db3eeaec22p-01, 0x1.24b1996b01b4fp-03, 0x1.eee0e00e84477p-07, 0x1.c3167ac2b9d41p-03, 0x1.0b1af6df0251cp-04, 0x1.15a674d82a90dp-03, -0x1.ec5402ca5cea4p-04, -0x1.b5fdfd7035d4bp-06, -0x1.12b5b15fd77f6p-14, -0x1.7d3d1bf62dd8p-05, -0x1.b201fd26234a2p-12, -0x1.c3244686f89d4p-11, 0x1.8ffb338214aap-11, 0x1.265cef654dee3p-12, 0x1.2997c70bd397ep-12, 0x1.10ff8070cf727p-10},
			b:    []float64{-0x1.f59a93d5cb085p-01, -0x1.d22e4db05bc01p-02, 0x1.78e6a6ae475cp-01, 0x1.b2bb48b61b5a9p-01, -0x1.20a42655401f2p-02, -0x1.fff158f79d122p-01},
			c:    []float64{-0x1.8f5b893fabe9p-13, -0x1.cddf5e879c3a2p-13, 0x1.5cff5291cb117p-12, 0x1.520a04b810c6bp-18, -0x1.6069994708ca3p-12, 0x1.bd29521324f6fp-13},
			d:    -0x1.4127865c60d37p+01,
			ceil: 0.0132,
		},
	})
}

// The count resolves every pole from s = 0 (fz 12571: exact stored pole
// −8.83e-8, 120× its rounding bound) or takes one whose 1/s coefficient is
// within its bound (fz 13418: pole 1.65e-10, neighbour 1.6e-8 in A22), so
// k = 0, but FreqResponse(0) takes the pole limit, +Inf, although G(0) of
// the stored model is finite (−284.9069183, 2.384283283e6; NSYMIV). The
// free fit samples the open grid and returns nil. Main held 3 and 2
// integrators.
func TestC2DFitLeastSquaresOpenGridWhereResponseIsInfiniteAtDC(t *testing.T) {
	checkLeastSquaresFrozenFit(t, []leastSquaresFrozenCase{
		// base (main, arm64 model) k>0: MaxRelativeError 1.42e-05, exact 1.35e-05; head k=0: 1.16e-05.
		{
			name: "fz/12571",
			n:    5,
			a:    []float64{-0x1.053651b347954p+17, -0x1.3d5f27abe429dp+18, 0x1.301aade8911cap+18, 0x1.611d3ee69673ep+16, -0x1.b403be4598927p+17, 0x1.574b59aa941bdp+14, 0x1.9f52a5baef0fdp+15, -0x1.8f1bbb0ecadedp+15, -0x1.cee0b7e01adb9p+13, 0x1.1dc60d9d4ba08p+15, -0x1.d57eebd26163ap+14, -0x1.1e078e6465b8cp+16, 0x1.118c5e8ccffb4p+16, 0x1.3de2bcf21c574p+14, -0x1.8883f830d225fp+15, -0x1.5386b62f56bf9p-10, -0x1.9db27461eef66p-09, 0x1.c51c6510e77d2p-09, 0x1.65d85e90fd3f6p-10, -0x1.2e1a90b02e5b4p-09, 0x1.987ed3c9dc4eap-12, 0x1.f1bb9eb196717p-11, -0x1.1093954fd310fp-10, -0x1.8fdae67b14b47p-12, 0x1.779f36abbb3f3p-11},
			b:    []float64{-0x1.c70385196a53cp-05, -0x1.f42ffb31e6e4ep-01, -0x1.de524cca020a6p-02, 0x1.743cab5f8f06dp-01, 0x1.b64e813f7e6a6p-01},
			c:    []float64{0x1.19cad07eba302p-16, -0x1.7c8ec13ae1efbp-16, -0x1.1dda5a2ab48ecp-19, 0x1.93a93449e44d3p-16, -0x1.c288c59f3f4ebp-17},
			d:    -0x1.55a6e837610d9p-05,
			ceil: 1.164e-05,
		},
		// base (main, arm64 model) k>0: MaxRelativeError 6.26e-06, exact 3.39e-06; head k=0: 2.5e-06.
		{
			name: "fz/13418",
			n:    6,
			a:    []float64{0x1.453f7c319b18fp+16, 0x1.18e9ef834be13p+16, 0x1.f9a0e714742a7p+13, 0x1.5b647040326cbp+17, 0x1.4c87569af3c3fp+19, -0x1.419d42820d1e9p+18, 0x1.886d9725e3ea5p+11, 0x1.52efc979737eep+11, 0x1.31087c5a8e5ecp+09, 0x1.532451117226cp+12, 0x1.e65bc804b65e8p+14, -0x1.f28d06a664effp+12, 0x1.e5c3081b509c6p+15, 0x1.a38c5742bbfb3p+15, 0x1.7994c8c648706p+13, 0x1.0516e8258db84p+17, 0x1.ed1391ce4f7dap+18, -0x1.e622b1c515df2p+17, 0x1.4909dbca0d707p+14, 0x1.1c3009dd86763p+14, 0x1.ff85707064f21p+11, 0x1.546e64667077cp+15, 0x1.582e3c957d639p+17, -0x1.36cf2ada3005p+16, 0x1.2248db526c849p+14, 0x1.f56eb6cae0994p+13, 0x1.c3464f8a3f88ep+11, 0x1.2fbbf5c79b059p+15, 0x1.2ce001865be0ep+17, -0x1.1779bf56fdaf4p+16, 0x1.2cf8eb9f6ec9fp+15, 0x1.03f2789429066p+15, 0x1.d3e3c74b1f4b2p+12, 0x1.3c14d9c9052dp+16, 0x1.377bfb5bcdfa5p+18, -0x1.22a980a00a4fp+17},
			b:    []float64{0x1.de66f5b235fc9p-01, 0x1.2fbd634d66a85p-01, -0x1.3be6f07be269ep-01, -0x1.d8bf26a8cc686p-01, 0x1.f7deb26ec98a4p-04, 0x1.fa715db8cd433p-01},
			c:    []float64{0x1.f1a2ae2e09c89p-18, 0x1.d281dc9064a4bp-17, -0x1.1339ce1eedbfep-16, -0x1.ba6312c503d4dp-19, 0x1.36fb071be5f0bp-16, -0x1.238ce3f1f7ce6p-17},
			d:    0x1.9909f1fcb57b5p+01,
			ceil: 2.5e-06,
		},
	})
}
