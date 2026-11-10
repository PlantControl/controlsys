package controlsys

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// acceptUnverifiedLqg is acceptUnverified for Lqg: an ErrAccuracyUnverified
// partial comes back with a nil error, its accuracy 1 readable. It panics
// on a result that breaks the carrier contract.
func acceptUnverifiedLqg(res *LqgResult, err error) (*LqgResult, error) {
	if msg := synthesisContractBreach(res != nil, accuracyOfLqg(res), err); msg != "" {
		panic("Lqg: " + msg)
	}
	if errors.Is(err, ErrAccuracyUnverified) {
		return res, nil
	}
	return res, err
}

// acceptUnverifiedH2Syn is acceptUnverifiedLqg for H2Syn.
func acceptUnverifiedH2Syn(res *H2SynResult, err error) (*H2SynResult, error) {
	if msg := synthesisContractBreach(res != nil, accuracyOfH2Syn(res), err); msg != "" {
		panic("H2Syn: " + msg)
	}
	if errors.Is(err, ErrAccuracyUnverified) {
		return res, nil
	}
	return res, err
}

func accuracyOfLqg(res *LqgResult) int {
	if res == nil {
		return 0
	}
	return res.accuracy
}

func accuracyOfHinfSyn(res *HinfSynResult) int {
	if res == nil {
		return 0
	}
	return res.accuracy
}

func accuracyOfH2Syn(res *H2SynResult) int {
	if res == nil {
		return 0
	}
	return res.accuracy
}

// synthesisContractBreach describes how a synthesis result (present or
// not, of accuracy acc) and its error break the carrier contract, or is
// empty: a result iff the error is nil (accuracy 0) or
// ErrAccuracyUnverified (accuracy 1); ErrNoStabilizing never with
// ErrNotConverged or ErrAccuracyUnverified; no unexported sentinel.
func synthesisContractBreach(has bool, acc int, err error) string {
	unverified := errors.Is(err, ErrAccuracyUnverified)
	switch {
	case err == nil && !has:
		return "nil result with nil error"
	case err == nil && acc != 0:
		return fmt.Sprintf("nil error with accuracy %d", acc)
	case unverified && (!has || acc == 0):
		return fmt.Sprintf("ErrAccuracyUnverified without an uncertified result: %v", err)
	case err != nil && !unverified && has:
		return fmt.Sprintf("result with error %v", err)
	case errors.Is(err, ErrNoStabilizing) && (errors.Is(err, ErrNotConverged) || unverified):
		return fmt.Sprintf("ErrNoStabilizing with ErrNotConverged or ErrAccuracyUnverified: %v", err)
	case errors.Is(err, errNoCandidate) || errors.Is(err, errSingularBasis):
		return fmt.Sprintf("unexported sentinel escapes: %v", err)
	}
	return ""
}

// synthesisRegulatorPlant is the generalized plant of the LQG problem of
// (A, B, C): w = [process noise (n); sensor noise (p)], z = [x; u], y = Cx
// + σv, with D12 = ρI and D21 = [0 σI].
func synthesisRegulatorPlant(t *testing.T, A, B, C *mat.Dense, rho, sig, dt float64) *System {
	t.Helper()
	n, m := B.Dims()
	p, _ := C.Dims()
	Bp := mat.NewDense(n, n+p+m, nil)
	setBlock(Bp, 0, 0, eyeDense(n))
	setBlock(Bp, 0, n+p, B)
	Cp := mat.NewDense(n+m+p, n, nil)
	setBlock(Cp, 0, 0, eyeDense(n))
	setBlock(Cp, n+m, 0, C)
	Dp := mat.NewDense(n+m+p, n+p+m, nil)
	for i := range m {
		Dp.Set(n+i, n+p+i, rho)
	}
	for i := range p {
		Dp.Set(n+m+i, n+i, sig)
	}
	P, err := New(A, Bp, Cp, Dp, dt)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// synthesisFixture is a plant of synthesisRegulatorPlant found by a seeded
// search over random 3- and 4-state plants with cheap control (ρ) or
// noise-free sensing (σ); its class held on arm64, amd64 and arm64 -race.
type synthesisFixture struct {
	a, b, c  []float64
	rho, sig float64
}

func (f synthesisFixture) plant(t *testing.T) *System {
	n := len(f.b)
	return synthesisRegulatorPlant(t, mat.NewDense(n, n, f.a), mat.NewDense(n, 1, f.b), mat.NewDense(1, n, f.c), f.rho, f.sig, 0)
}

var (
	// synthesisCheapControl: X not certified, closed loop stable.
	synthesisCheapControl = synthesisFixture{
		a:   []float64{-0.872870081022652, 0.7186764128060497, 1.6198895468912102, -1.1591409714965253, 1.9451069196603277, 0.7644069638268441, -0.14479984043343452, -0.89948429613954, 0.8252417009014956, 0.4865449657168542, -0.829980999499651, 0.20559569977223913, -0.6980936100300938, -0.2791755001723427, 1.8907173647710693, 0.2523429641833512},
		b:   []float64{-1.1824995006080652, -0.10686911328794949, 0.5177924785161483, 0.8654825243432116},
		c:   []float64{0.1359807317413111, -0.9303133293290122, 0.9336497081719497, 0.48256480094314647},
		rho: 9.801674418925994e-07, sig: 7.495075384191857e-08,
	}
	// synthesisControlNotConverged: no state-feedback X passes
	// verification. HinfSyn, which found no γ below its give-up bound by
	// the Hamiltonian alone, gets a controller from the extended-pencil
	// repeat, returned with ErrAccuracyUnverified (BMNGOS).
	synthesisControlNotConverged = synthesisFixture{
		a:   []float64{0.6387616393876916, 0.4587988411199844, 0.528907006112241, 0.12157279348037137, 0.4342701769944363, 1.5828768967155433, -0.014001459255651827, -0.2030728131081514, 0.31309763284412817, -0.25266771115404013, -1.9007061455699903, -1.2059618903768583, -0.30241700661402304, -0.5891295338839894, 1.4718530704475516, 0.6057568301514619},
		b:   []float64{-0.45122737797268053, 0.5921509980895266, -0.44063398438152035, 0.5301782392019727},
		c:   []float64{-0.7041421969970909, 0.6282796401468824, -0.2875344743526116, -2.090565833890424},
		rho: 1.1859994693143048e-08, sig: 0.003916790560348642,
	}
	// synthesisBoundaryNotConverged (census row bnd 674, BMNGOS): a weakly
	// controllable mode 3.5e-15 off the imaginary axis puts an X-Hamiltonian
	// pair within rounding of the axis at every γ, for the pencil too, so
	// HinfSyn finds no γ below its give-up bound; the certified H2
	// controller proves γ finite, so that is ErrNotConverged.
	synthesisBoundaryNotConverged = synthesisFixture{
		a:   []float64{-0.7237063632621797, 4.985017928162791, -3.3857174500105076, -6.7705326494338856, -6.572878620402842, -0.7119680305799931, -2.8597634157896747, 2.278636715820158, 3.1566246028357945, 3.3730235162541815, -0.3614962805114312, 3.0842547727398455, -2.2679969795214827, -3.591309605365047, -3.4151533180466713, 0.12162630019987758, -2.1428046687485893, 0.6170010327290477, 1.1736835087018616, 2.194532441673459, -0.4736889559752902, -0.9227047654508497, 1.7056898513788366, 2.0854524879187126, 1.0955068971788282},
		b:   []float64{-3.122549315482755, 0.3567123996251636, -3.143046326280349, -0.9044921028519853, 2.543648011802267},
		c:   []float64{-0.07585885440170685, 3.5602123269066883, -3.0339938598453187, -5.943132558771198, -5.703127607118304},
		rho: 1, sig: 1,
	}
	// synthesisBoundaryInfeasible (census row bnd 224, BMNGOS): no γ passes
	// below the give-up bound and the H2 filter Riccati solution is not
	// certified either, so nothing proves γ finite.
	synthesisBoundaryInfeasible = synthesisFixture{
		a:   []float64{0.3551887068056787, 0.43135054207754314, -0.6240507168391118, -1.7860586410743449, -0.8633291557886179, 0.8330267482366999, -1.161034993878373, -0.16033554383441181, -0.615146249022356},
		b:   []float64{0.19761118564758431, -0.3237695982512298, -1.2466237692836402},
		c:   []float64{-0.11226294500214501, 0.05034198731287463, -0.22039968272361465},
		rho: 1, sig: 0.00015829728631378115,
	}
	// synthesisFilterNotConverged: no filter Y passes verification.
	synthesisFilterNotConverged = synthesisFixture{
		a:   []float64{0.3219644310129748, -0.019669945290228252, 0.27063314941777217, -0.3056982879897324, 0.7357374024660253, 0.6319897026486008, -1.5443787160616465, -0.6335094362853653, 0.5133448662890427, 0.045095166773168105, -0.7639299780140478, -0.4059741817182856, 0.3228282223941571, 1.047526651323474, 2.2079007160633926, 1.8439689948006481},
		b:   []float64{-2.0942776181871694, -0.1724562315622008, -0.39099450028300775, -1.9667896535903502},
		c:   []float64{0.6155184942698835, -0.1063322113767462, 0.3985745956947928, 0.3673343407680497},
		rho: 0.00028000132194381437, sig: 1.1829475175887219e-08,
	}
	// synthesisUnstableLoop: X and Y not certified, and their gains leave
	// the assembled closed loop unstable in float64 (weakBStall does so on
	// arm64 only).
	synthesisUnstableLoop = synthesisFixture{
		a:   []float64{-0.021989573585728783, 0.4843413465470531, -0.0026274405098550557, 1.604536652436811, -0.0036586979290615385, -0.23715622583214785, -1.6371225127305533, 0.41088519495677267, -0.39001350622271924},
		b:   []float64{0.5903108398592843, -1.2836748607725734, -0.3863958356053746},
		c:   []float64{0.8417210102407582, -1.5486771209478385, 0.301144403219971},
		rho: 2.07339239930966e-08, sig: 0.0073488015694472344,
	}
)

// weakBStallPlant returns weakBStall's (A, B, Q, R) as an Lqg plant with
// C = I and as the H2Syn plant of TestRiccatiAccuracy_LqgH2Syn.
func weakBStallPlant(t *testing.T) (sys *System, qxu *mat.Dense, P *System) {
	t.Helper()
	A, B, Q, R, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	n, _ := A.Dims()
	sys, err := New(A, B, eyeDense(n), mat.NewDense(n, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	var chol mat.Cholesky
	if !chol.Factorize(mat.NewSymDense(n, mat.DenseCopyOf(Q).RawMatrix().Data)) {
		t.Fatal("Q not positive definite")
	}
	var U mat.TriDense
	chol.UTo(&U)
	Bp := mat.NewDense(n, 2*n+1, nil)
	Cp := mat.NewDense(2*n+1, n, nil)
	Dp := mat.NewDense(2*n+1, 2*n+1, nil)
	for i := range n {
		Bp.Set(i, i, 1)
		Bp.Set(i, 2*n, B.At(i, 0))
		for j := range n {
			Cp.Set(i, j, U.At(i, j))
		}
		Cp.Set(n+1+i, i, 1)
		Dp.Set(n+1+i, n+i, 1)
	}
	Dp.Set(n, 2*n, math.Sqrt(R.At(0, 0)))
	if P, err = New(A, Bp, Cp, Dp, 0); err != nil {
		t.Fatal(err)
	}
	return sys, blkdiagTest(Q, R), P
}

// synthesisClass is the outcome class of an exported synthesis operation
// under the result contract (ZOJBZX).
type synthesisClass int

const (
	synthesisCertified synthesisClass = iota
	synthesisUnverified
	synthesisNotConverged
	synthesisClosedLoopUnstable
	synthesisProvedControl
	synthesisProvedFilter
	synthesisStructural
	synthesisInfeasible
)

func (c synthesisClass) String() string {
	return [...]string{"certified", "unverified", "not converged", "closed loop unstable", "proved (control)", "proved (filter)", "structural", "infeasible"}[c]
}

// TestSynthesisContractByOperation checks Lqg, H2Syn and HinfSyn in each
// outcome class they have a fixture for (ZOJBZX step 3):
//   - certified: (res, nil);
//   - uncertified: (res ≠ nil, ErrAccuracyUnverified): Lqg and H2Syn with
//     an uncertified Riccati solution, HinfSyn with the controller of its
//     extended-pencil repeat;
//   - not converged: (nil, ErrNotConverged), never ErrNoStabilizing: a
//     Riccati solve with no verified solution (Lqg, H2Syn), or HinfSyn with
//     no backed-off controller meeting its γ (Q7, with the last try's
//     sentinel: here ErrGammaNotAchievable of an unstable loop; no plant
//     found keeps a last Hamiltonian stable-count failure on arm64, amd64
//     and -race alike, so TestHinfSynVerifiedNotConverged checks that and
//     the other last errors);
//   - closed loop unstable: H2Syn whose uncertified gains leave CL unstable
//     in float64 (Q6): (nil, ErrNotConverged) with ErrUnstable;
//   - proved: (nil, ErrNoStabilizing) with ErrNotStabilizable from the
//     regulator and ErrNotDetectable from the filter, never the other,
//     never ErrNotConverged;
//   - structural: H2Syn and HinfSyn reject an exactly unstabilizable or
//     undetectable plant with ErrNotStabilizable or ErrNotDetectable alone
//     before any Riccati solve (IsStabilizable, IsDetectable); a plant
//     that passes it can still reach the proved Riccati verdict (H2Syn
//     discrete rows);
//   - infeasible: HinfSyn with no γ passing the Riccati test below its
//     give-up bound and no certified H2 controller: ErrGammaNotAchievable
//     alone (with a certified one, ErrNotConverged too).
func TestSynthesisContractByOperation(t *testing.T) {
	cA := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	cB := mat.NewDense(2, 1, []float64{0, 1})
	cC := mat.NewDense(1, 2, []float64{1, 0})
	dA := mat.NewDense(2, 2, []float64{1.1, 0.2, 0, 0.5})
	cSys, err := New(cA, cB, cC, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	dSys, err := New(dA, cB, cC, mat.NewDense(1, 1, nil), 1)
	if err != nil {
		t.Fatal(err)
	}
	uA := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	uB := mat.NewDense(2, 1, []float64{0, 1})
	uC := mat.NewDense(1, 2, []float64{0, 1})
	ctlSys, err := New(uA, uB, mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	obsSys, err := New(uA, eyeDense(2), uC, mat.NewDense(1, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	wSys, wQXU, _ := weakBStallPlant(t)
	wn, _ := wSys.A.Dims()

	xa, xb, xq, xr, xs := hamiltonianAxisPair()
	xQXU := mat.NewDense(7, 7, nil)
	setBlock(xQXU, 0, 0, mat.NewDense(4, 4, xq))
	setBlock(xQXU, 4, 4, mat.NewDense(3, 3, xr))
	setBlock(xQXU, 0, 4, mat.NewDense(4, 3, xs))
	setBlock(xQXU, 4, 0, mat.DenseCopyOf(mat.NewDense(4, 3, xs).T()))
	xSys, err := New(mat.NewDense(4, 4, xa), mat.NewDense(4, 3, xb), eyeDense(4), mat.NewDense(4, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	pA, pB, pC, pE := boundaryUnreachablePlant(0)
	pAt, pCt, pQ, pEt := dualDare(pA, pB, pC, pE)
	pSys, err := NewDescriptor(pAt, pCt, eyeDense(4), mat.NewDense(4, 1, nil), pEt, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Discrete plants of a seeded boundary-mode sweep (a mode within 1e-15
	// to 1e-3 of the unit circle, weakly reached or seen) on which the
	// structural check passes and the Riccati PBH verdict finds the
	// state-feedback (bnd 4147) or filter (bnd 2305) Dare deficient within
	// rounding plus 1× its first-order bound, σmin at most 0.54× that
	// threshold in every pass; their class held native, amd64 and -race.
	// bnd 415 and 785, deficient only within the 10× band, now end
	// ErrNotConverged (MZTFJJ).
	provedCtl, err := New(mat.NewDense(2, 2, []float64{2.0631430204584746, 1.253953758355174, -2.1270172657669817, -1.5087699803029766}),
		mat.NewDense(2, 5, []float64{1, 0, 0, 0, -0.23356869811929923, 0, 1, 0, 0, 0.4672980437083926}),
		mat.NewDense(5, 2, []float64{1, 0, 0, 1, 0, 0, 0.028435633524425054, -0.7915248809145411, -1.426937377449151, -1.5640149320046575}),
		mat.NewDense(5, 5, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0020590024951322944, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}), 1)
	if err != nil {
		t.Fatal(err)
	}
	provedObs, err := New(mat.NewDense(2, 2, []float64{-1.7175478007041722, -0.9298352341847946, 1.016150079248334, 0.3167793782910796}),
		mat.NewDense(2, 6, []float64{1, 0, 0, 0, -0.7394194976397351, 0.6034707892924778, 0, 1, 0, 0, -0.35636888345186574, -0.31437625715817197}),
		mat.NewDense(6, 2, []float64{1, 0, 0, 1, 0, 0, 0, 0, 2.322589862241148, 3.009731040016948, -0.7863551111939422, -1.0189992753833155}),
		mat.NewDense(6, 6, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0026237940536626746, 0, 0, 0, 0, 0, 0, 0.0026237940536626746, 0, 0}), 1)
	if err != nil {
		t.Fatal(err)
	}

	regP := func(A, B, C *mat.Dense, dt float64) *System { return synthesisRegulatorPlant(t, A, B, C, 1, 1, dt) }
	weakB := mat.NewDense(2, 1, []float64{1e-8, 1})

	type out struct {
		has bool
		acc int
		err error
	}
	lqg := func(sys *System, qxu, qwv *mat.Dense) func() out {
		return func() out { r, e := Lqg(sys, qxu, qwv, nil); return out{r != nil, accuracyOfLqg(r), e} }
	}
	h2 := func(P *System, ny, nu int) func() out {
		return func() out { r, e := H2Syn(P, ny, nu); return out{r != nil, accuracyOfH2Syn(r), e} }
	}
	hinf := func(P *System, ny, nu int) func() out {
		return func() out { r, e := HinfSyn(P, ny, nu); return out{r != nil, accuracyOfHinfSyn(r), e} }
	}
	cases := []struct {
		name  string
		class synthesisClass
		run   func() out
		also  error
	}{
		{"Lqg", synthesisCertified, lqg(cSys, eyeDense(3), eyeDense(3)), nil},
		{"Lqg discrete", synthesisCertified, lqg(dSys, eyeDense(3), eyeDense(3)), nil},
		{"Lqg", synthesisUnverified, lqg(wSys, wQXU, eyeDense(2*wn)), nil},
		{"Lqg", synthesisNotConverged, lqg(xSys, xQXU, eyeDense(8)), nil},
		{"Lqg discrete", synthesisNotConverged, lqg(pSys, blkdiagTest(pQ, eyeDense(1)), eyeDense(8)), nil},
		{"Lqg", synthesisProvedControl, lqg(ctlSys, eyeDense(3), eyeDense(3)), nil},
		{"Lqg", synthesisProvedFilter, lqg(obsSys, eyeDense(4), eyeDense(3)), nil},
		{"H2Syn", synthesisCertified, h2(regP(cA, cB, cC, 0), 1, 1), nil},
		{"H2Syn discrete", synthesisCertified, h2(regP(dA, cB, cC, 1), 1, 1), nil},
		{"H2Syn", synthesisUnverified, h2(synthesisCheapControl.plant(t), 1, 1), nil},
		{"H2Syn control", synthesisNotConverged, h2(synthesisControlNotConverged.plant(t), 1, 1), nil},
		{"H2Syn filter", synthesisNotConverged, h2(synthesisFilterNotConverged.plant(t), 1, 1), nil},
		{"H2Syn discrete", synthesisProvedControl, h2(provedCtl, 2, 1), nil},
		{"H2Syn discrete", synthesisProvedFilter, h2(provedObs, 2, 2), nil},
		{"H2Syn", synthesisClosedLoopUnstable, h2(synthesisUnstableLoop.plant(t), 1, 1), nil},
		{"H2Syn control", synthesisStructural, h2(regP(uA, uB, mat.NewDense(1, 2, []float64{1, 1}), 0), 1, 1), ErrNotStabilizable},
		{"H2Syn filter", synthesisStructural, h2(regP(uA, eyeDense(2), uC, 0), 1, 2), ErrNotDetectable},
		{"HinfSyn", synthesisCertified, hinf(regP(cA, cB, cC, 0), 1, 1), nil},
		{"HinfSyn discrete", synthesisCertified, hinf(regP(dA, cB, cC, 1), 1, 1), nil},
		{"HinfSyn unstable loop", synthesisNotConverged, hinf(regP(uA, weakB, mat.NewDense(1, 2, []float64{1, 1}), 0), 1, 1), ErrGammaNotAchievable},
		{"HinfSyn control", synthesisStructural, hinf(regP(uA, uB, mat.NewDense(1, 2, []float64{1, 1}), 0), 1, 1), ErrNotStabilizable},
		{"HinfSyn filter", synthesisStructural, hinf(regP(uA, eyeDense(2), uC, 0), 1, 2), ErrNotDetectable},
		{"HinfSyn cheap control", synthesisUnverified, hinf(synthesisControlNotConverged.plant(t), 1, 1), nil},
		{"HinfSyn boundary", synthesisNotConverged, hinf(synthesisBoundaryNotConverged.plant(t), 1, 1), ErrGammaNotAchievable},
		{"HinfSyn", synthesisInfeasible, hinf(synthesisBoundaryInfeasible.plant(t), 1, 1), nil},
	}
	for _, c := range cases {
		o := c.run()
		what := fmt.Sprintf("%s %v", c.name, c.class)
		if msg := synthesisContractBreach(o.has, o.acc, o.err); msg != "" {
			t.Errorf("%s: %s", what, msg)
		}
		err := o.err
		is := func(e error) bool { return errors.Is(err, e) }
		nilWith := func(ok bool, want string) {
			t.Helper()
			if !ok || o.has {
				t.Errorf("%s: result %v, err %v; want nil with %s", what, o.has, err, want)
			}
		}
		if c.also != nil && !is(c.also) {
			t.Errorf("%s: err %v is not %v", what, err, c.also)
		}
		switch c.class {
		case synthesisCertified:
			if err != nil || !o.has {
				t.Errorf("%s: result %v, err %v; want a result with nil", what, o.has, err)
			}
		case synthesisUnverified:
			if !is(ErrAccuracyUnverified) || !o.has || o.acc != 1 {
				t.Errorf("%s: result %v, err %v; want the uncertified result with ErrAccuracyUnverified", what, o.has, err)
			}
		case synthesisNotConverged:
			nilWith(is(ErrNotConverged) && !is(ErrNoStabilizing) && !is(ErrNotStabilizable) && !is(ErrNotDetectable), "ErrNotConverged")
		case synthesisClosedLoopUnstable:
			nilWith(is(ErrNotConverged) && is(ErrUnstable) && !is(ErrNoStabilizing), "ErrNotConverged and ErrUnstable")
		case synthesisProvedControl:
			nilWith(is(ErrNoStabilizing) && is(ErrNotStabilizable) && !is(ErrNotDetectable) && !is(ErrNotConverged), "ErrNoStabilizing and ErrNotStabilizable")
		case synthesisProvedFilter:
			nilWith(is(ErrNoStabilizing) && is(ErrNotDetectable) && !is(ErrNotStabilizable) && !is(ErrNotConverged), "ErrNoStabilizing and ErrNotDetectable")
		case synthesisStructural:
			nilWith(!is(ErrNoStabilizing) && !is(ErrNotConverged), "the structural sentinel alone")
		case synthesisInfeasible:
			nilWith(is(ErrGammaNotAchievable) && !is(ErrNotConverged) && !is(ErrNoStabilizing), "ErrGammaNotAchievable alone")
		}
	}
}

// TestSynthesisUnverifiedPartialResult: the ErrAccuracyUnverified result
// of H2Syn and Lqg is the full one the uncertified Riccati solutions give,
// bit for bit: H2Syn's K, X and Y those of h2Syn, CL and Gamma those of
// its closed loop; Lqg's Xc and Xf those of its cores, with the controller.
func TestSynthesisUnverifiedPartialResult(t *testing.T) {
	P := synthesisCheapControl.plant(t)
	core, err := h2Syn(P, 1, 1)
	if err != nil || core.accuracy != 1 {
		t.Fatalf("h2Syn: err %v, want an uncertified result", err)
	}
	res, err := H2Syn(P, 1, 1)
	if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
		t.Fatalf("H2Syn: result %v, err %v; want ErrAccuracyUnverified with the result", res != nil, err)
	}
	if !bitEqualDense(res.X, core.X) || !bitEqualDense(res.Y, core.Y) || !bitEqualDense(res.K.A, core.K.A) || !bitEqualDense(res.K.B, core.K.B) || !bitEqualDense(res.K.C, core.K.C) || !slices.Equal(res.CLPoles, core.CLPoles) {
		t.Error("H2Syn: partial result differs from h2Syn's")
	}
	if res.CL == nil {
		t.Fatal("H2Syn: partial result without CL")
	}
	if g, err := H2Norm(res.CL); err != nil || math.Float64bits(g) != math.Float64bits(res.Gamma) {
		t.Errorf("H2Syn: Gamma %v, H2Norm of CL %v (%v)", res.Gamma, g, err)
	}

	sys, qxu, _ := weakBStallPlant(t)
	n, _ := sys.A.Dims()
	lres, err := Lqg(sys, qxu, eyeDense(2*n), nil)
	if !errors.Is(err, ErrAccuracyUnverified) || lres == nil || lres.Controller == nil {
		t.Fatalf("Lqg: result %v, err %v; want ErrAccuracyUnverified with the controller", lres != nil, err)
	}
	kc, err := careCore(sys.A, sys.B, subDense(qxu, 0, 0, n, n), subDense(qxu, n, n, 1, 1), &RiccatiOpts{})
	if err != nil || kc.accuracy != 1 {
		t.Fatalf("regulator core: err %v, want an uncertified X", err)
	}
	if !bitEqualDense(lres.Xc, kc.X) || !bitEqualDense(lres.K, kc.K) {
		t.Error("Lqg: Xc or K differs from the regulator core's")
	}
}

// TestHinfSynVerifiedNotConverged: when no backed-off build gives a
// controller, verified returns ErrNotConverged with the last build's
// error mapped to its exported sentinel, never an unexported one (Q7).
func TestHinfSynVerifiedNotConverged(t *testing.T) {
	for _, c := range []struct {
		name string
		last error
		want error
	}{
		{"stable count", fmt.Errorf("HinfSyn: Hamiltonian: %w", errNoCandidate), ErrNotConverged},
		{"singular basis", fmt.Errorf("HinfSyn: Hamiltonian %w", errSingularBasis), ErrSingularEquation},
		{"unstable loop", fmt.Errorf("HinfSyn: %w", ErrGammaNotAchievable), ErrGammaNotAchievable},
		{"Schur", fmt.Errorf("HinfSyn: %w", ErrSchurFailed), ErrSchurFailed},
	} {
		calls := 0
		d := hinfDesign{gammaEdge: 1, build: func(float64, bool) (*HinfSynResult, error) { calls++; return nil, c.last }}
		res, err := d.verified(&generalizedPlantPartition{op: "HinfSyn"}, nil, 1, 1)
		if res != nil || !errors.Is(err, ErrNotConverged) || !errors.Is(err, c.want) || errors.Is(err, ErrNoStabilizing) {
			t.Errorf("%s: result %v, err %v; want nil with ErrNotConverged and %v", c.name, res != nil, err, c.want)
		}
		if msg := synthesisContractBreach(res != nil, 0, err); msg != "" {
			t.Errorf("%s: %s", c.name, msg)
		}
		if calls != hinfBackoffTries {
			t.Errorf("%s: %d builds, want %d", c.name, calls, hinfBackoffTries)
		}
	}
}

// TestHamiltonianRiccatiErrors: each rejection of hamiltonianRiccatiWork
// has its sentinel (ZOJBZX Q7): a float64 stable count other than n is
// errNoCandidate, an eigenvalue on the imaginary axis and an X not
// semidefinite are ErrGammaNotAchievable, and a singular U11
// errSingularBasis; none is ErrNoStabilizing.
func TestHamiltonianRiccatiErrors(t *testing.T) {
	const d, w = 1e-17, 1.0
	for _, c := range []struct {
		name    string
		F, G, Q *mat.Dense
		want    error
	}{
		{"stable count", mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}), errNoCandidate},
		{"axis", mat.NewDense(2, 2, []float64{-d, w, -w, -d}), mat.NewDense(2, 2, nil), mat.NewDense(2, 2, nil), ErrGammaNotAchievable},
		{"singular U11", mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}), errSingularBasis},
		{"not semidefinite", mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{-2}), ErrGammaNotAchievable},
	} {
		n, _ := c.F.Dims()
		hw := newHamiltonianRiccatiWork(n)
		hw.setHamiltonian(c.F, c.G, c.Q)
		X, err := hw.solve()
		if X != nil || !errors.Is(err, c.want) || errors.Is(err, ErrNoStabilizing) {
			t.Errorf("%s: X %v, err %v; want nil with %v", c.name, X != nil, err, c.want)
		}
	}
	hw := newHamiltonianRiccatiWork(1)
	hw.setHamiltonian(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{2}))
	if X, err := hw.solve(); err != nil || math.Abs(X.At(0, 0)-1) > 1e-15 {
		t.Errorf("X = %v, err %v; want 1", X, err)
	}
}
