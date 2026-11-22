package controlsys

import (
	"errors"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Frozen census literals (M3HDEP). rotZ rows with seed%3 = 0 are dyadic and
// identical on every arch; the others are arm64 data.
var (
	// rotZ/864: lightly damped modes; Truncate order 1 keeps a pole at
	// −4.75e-12, 21× inside IsStable's band.
	stabRotZ864 = balanceCertCase{
		name: "boundary band truncation", n: 4, m: 1, p: 1,
		a: []float64{-2.3283064365386963e-10, 1, -1, -1, -1, -2.3283064365386963e-10, 0.5, 0, 0, 0, -2.9802322387695312e-08, 2, 0, 0, -2, -2.9802322387695312e-08},
		b: []float64{0.5, 1, 0.5, 0},
		c: []float64{1, 2, -1, 2},
		d: []float64{0},
	}
	// rotZ/39: Truncate order 1 keeps a pole at −1.01e-11.
	stabRotZ39 = balanceCertCase{
		name: "boundary band truncation 2", n: 4, m: 1, p: 1,
		a: []float64{-2, 0.5, 0, 0, -0.5, -2, -0.5, -0.5, 0, 0, -4.656612873077393e-10, 0.5, 0, 0, -0.5, -4.656612873077393e-10},
		b: []float64{0, 0.5, 1, 0},
		c: []float64{2, 0.5, 2, 0.5},
		d: []float64{0},
	}
	// rotZ/183: MatchDC order 3 has exact poles −2.33e-10 ± 0.5i next to
	// −3e8, below the eigensolver's rounding.
	stabRotZ183 = balanceCertCase{
		name: "unresolvable slow pole", n: 6, m: 1, p: 1,
		a: []float64{-0.5, 0.5, 0.5, 0, -1, -1, -0.5, -0.5, 1, -0.5, -1, 0, 0, 0, -2.3283064365386963e-10, 0.5, 0, 1, 0, 0, -0.5, -2.3283064365386963e-10, -1, -1, 0, 0, 0, 0, -1.862645149230957e-09, 1, 0, 0, 0, 0, -1, -1.862645149230957e-09},
		b: []float64{0, 2, -1, 0, -1, 0.5},
		c: []float64{0.5, 0.5, 2, 0.5, 2, 0},
		d: []float64{0},
	}
	// rotZ/242 arm64: MatchDC order 2 has the exact pole −9.31e-10.
	stabRotZ242 = balanceCertCase{
		name: "unresolvable slow pole 2", n: 7, m: 1, p: 1,
		a: []float64{-0.4835820339315944, 1.0487110226257583, 0.5715336690213004, -0.6755995504675517, -0.7158621379500779, 0.22859720720056115, -0.3358343187354233, 0.7776826791822232, -0.7752538746818954, -0.891775305522787, -0.5198736377408358, -0.0457224031314647, -0.7878691635818258, -0.8086598096875851, -0.7577772433464662, 0.9481930785793281, -1.8814626445398148, 1.231260612209049, -0.057554857255247074, 0.39406114977096846, -0.005991153875746411, 0.20820471134040583, -0.9488588481601617, -1.0255408811120337, -1.1943455982138764, -0.06361575610884514, -0.46769631973607007, 0.9133151361746735, 0.06764047906829701, 0.5899813457581667, 0.28823639161582953, 0.6303388974431884, -0.5318192241997152, 0.425742532065858, 1.8557974726871471, 0.021201276681264827, 0.9393571005580851, -1.2455045106578917, 0.07513119828084812, -0.8045307145827116, -1.1307579155646879, -0.7873048352669227, 0.821671689913014, 0.18419315065457262, -0.38075669175421506, -1.1210386217465735, -1.181674248843972, -0.08518536572575737, -0.002779186636896842},
		b: []float64{1, 0.5, 0.5, -1, 2, 1, 2},
		c: []float64{2, 0.5, 1, 0.5, -1, 1, 1},
		d: []float64{0},
	}
	// rotH/189 arm64: Hadamard similarity of a double mode; both Hankel
	// singular values are exactly 0.25.
	stabRotH189 = balanceCertCase{
		name: "equal pair", n: 2, m: 1, p: 1,
		a: []float64{-0.9999999999999999, -1.9999999999999998, 1.9999999999999996, -0.9999999999999999},
		b: []float64{1, 0.5},
		c: []float64{1, 0},
		d: []float64{0},
	}
	// rotZ/776 arm64: MatchDC order 1 residualizes a block with σmin at
	// 19.5·n·ε·‖A‖_F, above the singular floor.
	stabRotZ776 = balanceCertCase{
		name: "near-singular block above floor", n: 8, m: 1, p: 1,
		a: []float64{-0.41743210559354665, -1.1299376021213732, -0.2465756477794638, -0.44724441471001386, 0.024439958972939147, 0.016044323286366412, 0.27681495901694547, -0.41415923131338844, -0.5180933241377401, -0.5307601613331101, 0.23721552390503833, -1.2820253825626922, -0.5381008072756954, 0.438168136260888, -1.361548321648888, -0.8574828881801189, 0.10398627763755662, -0.5092928928157688, -1.501198373171069, 1.3899021443102393, 0.2867890749132563, 1.3081123286612135, -0.4474008163980232, 0.014871201028566787, -0.7731815118926311, 1.2119732673159662, -0.3823832283002585, -1.0772204445157003, 0.4289245062010444, 0.5172234882846294, -0.0992669094872093, -0.6016244181472695, -0.09383400122208617, -0.5863122145074025, -1.3464703136596674, -0.1572887108274703, -0.8951634737225341, 0.150951347334927, -0.9564128218954175, 0.8206495042959007, 0.8959327566549791, -0.16950974211367822, -1.1753949537797408, 0.0063320593708103175, -0.5166635631751352, -1.1804923105216851, 1.0140088644150629, -1.6482565939050973, -1.4548842502302899, -0.001425148172172685, 0.43982059624221154, -0.0967796452581299, 0.029377590996976406, -1.0414691495847352, 0.3565669528720165, 1.1025341457912001, -0.47233970545092674, 0.1888358488439466, 0.7438424615172623, 0.1764782570006049, 0.4348527456626048, 0.8790965715510152, -0.8541715650069897, -0.7543000858770161},
		b: []float64{0, -1, 0, -1, -1, -1, 2, 0},
		c: []float64{1, -1, 1, 2, 1, 0.5, 2, 1},
		d: []float64{0},
	}
	// rotQ/270 arm64: Hankel singular values equal to rounding; Truncate
	// order 1 keeps a pole at −2.2e-16.
	stabRotQ270 = balanceCertCase{
		name: "equal pair truncation", n: 2, m: 1, p: 1,
		a: []float64{-0.9999999999999998, -0.9999999999999999, 1, -0.9999999999999999},
		b: []float64{0, 2},
		c: []float64{-1, -1},
		d: []float64{0},
	}
)

func balredModredOf(t *testing.T, br *BalrealResult, k int, proj StateProjection) (*System, error) {
	t.Helper()
	n := len(br.HSV)
	var elim []int
	for i := k; i < n; i++ {
		elim = append(elim, i)
	}
	return Modred(br.Sys, elim, proj)
}

func sameModel(a, b *System) bool {
	if !mat.Equal(a.D, b.D) {
		return false
	}
	n, _, _ := a.Dims()
	return n == 0 || mat.Equal(a.A, b.A) && mat.Equal(a.B, b.B) && mat.Equal(a.C, b.C)
}

// balredStabilityContract checks Balred(order k) on a certified model:
// above nmin it is ErrSingularTransform with no model; otherwise a nil
// result passes IsStable and equals Modred of Balreal's realization,
// ErrAccuracyUnverified carries that same model, which fails IsStable, and
// the only other outcome is ErrSingularA22 (MatchDC) with no model.
func balredStabilityContract(t *testing.T, sys *System, br *BalrealResult, k int, proj StateProjection) error {
	t.Helper()
	red, hsv, err := Balred(sys, k, BalredOptions{StateProjection: proj})
	nmin := 0
	for _, v := range br.HSV {
		if v > float64(len(br.HSV))*eps()*br.HSV[0] {
			nmin++
		}
	}
	if k > nmin {
		if red != nil || !errors.Is(err, ErrSingularTransform) {
			t.Fatalf("Balred(%d, %v) above nmin %d: value %v, err %v", k, proj, nmin, red != nil, err)
		}
		return err
	}
	unv := errors.Is(err, ErrAccuracyUnverified)
	if err != nil && !unv {
		if red != nil || !errors.Is(err, ErrSingularA22) || proj != MatchDC {
			t.Fatalf("Balred(%d, %v): value %v, err %v", k, proj, red != nil, err)
		}
		return err
	}
	if red == nil || !slices.Equal(hsv, br.HSV) {
		t.Fatalf("Balred(%d, %v): value %v, HSV %v, err %v", k, proj, red != nil, hsv, err)
	}
	if errors.Is(err, ErrUnstable) || errors.Is(err, ErrSingularTransform) {
		t.Fatalf("Balred(%d, %v): err %v", k, proj, err)
	}
	want, werr := balredModredOf(t, br, k, proj)
	if werr != nil || !sameModel(red, want) {
		t.Fatalf("Balred(%d, %v) not bit-identical to Modred of Balreal (%v)", k, proj, werr)
	}
	st, serr := red.IsStable()
	if serr != nil || st == unv {
		t.Fatalf("Balred(%d, %v): IsStable %v (%v) with err %v", k, proj, st, serr, err)
	}
	return err
}

// M3HDEP: an order splitting a near-equal pair keeps a pole inside IsStable's
// band; Balred returns that model with ErrAccuracyUnverified (rotZ/864).
func TestBalred_BoundaryBandReductionUnverified(t *testing.T) {
	sys := stabRotZ864.sys(t)
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	red, hsv, err := Balred(sys, 1, BalredOptions{StateProjection: Truncate})
	if !errors.Is(err, ErrAccuracyUnverified) || errors.Is(err, ErrUnstable) || errors.Is(err, ErrSingularTransform) {
		t.Fatalf("Balred(1) err = %v; want ErrAccuracyUnverified only", err)
	}
	if red == nil || !slices.Equal(hsv, br.HSV) {
		t.Fatalf("Balred(1): value %v, HSV %v", red != nil, hsv)
	}
	want, err := balredModredOf(t, br, 1, Truncate)
	if err != nil || !sameModel(red, want) {
		t.Fatalf("Balred(1) not bit-identical to Modred of Balreal (%v)", err)
	}
	if st, err := red.IsStable(); err != nil || st {
		t.Fatalf("Balred(1) IsStable = %v, %v; want false", st, err)
	}
	if a := red.A.At(0, 0); !(a < 0) || a < -1e-11 {
		t.Fatalf("kept pole %v; want in (−1e-11, 0)", a)
	}
}

// M3HDEP: every order and projection obeys the stability contract,
// including MatchDC orders whose slow pole the eigensolver cannot resolve
// (rotZ/183, rotZ/242), whose outcome is arch-dependent.
func TestBalred_NilReductionIsStable(t *testing.T) {
	for _, tc := range []balanceCertCase{stabRotZ864, stabRotZ39, stabRotZ183, stabRotZ242} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			br, err := Balreal(sys)
			if err != nil {
				t.Fatal(err)
			}
			for _, proj := range []StateProjection{MatchDC, Truncate} {
				for k := range tc.n {
					balredStabilityContract(t, sys, br, k, proj)
				}
			}
		})
	}
}

// M3HDEP: with Hankel singular values exactly equal, the residualized state
// of the balanced realization is rounding (rotH/189): MatchDC returns
// ErrSingularA22; Truncate keeps the pole −2.
func TestBalred_EqualPairMatchDCSingularA22(t *testing.T) {
	sys := stabRotH189.sys(t)
	red, hsv, err := Balred(sys, 1, BalredOptions{StateProjection: MatchDC})
	if !errors.Is(err, ErrSingularA22) || red != nil || hsv != nil {
		t.Fatalf("MatchDC: value %v, err %v; want ErrSingularA22 and no value", red != nil, err)
	}
	red, _, err = Balred(sys, 1, BalredOptions{StateProjection: Truncate})
	if err != nil {
		t.Fatal(err)
	}
	if a := red.A.At(0, 0); math.Abs(a+2) > 1e-12 {
		t.Fatalf("Truncate pole %v; want −2", a)
	}
	requireStable(t, "Truncate", red)
}

// M3HDEP: a residualized block near, but above, the singular floor reduces
// unchanged (rotZ/776 at 19.5·n·ε).
func TestBalred_NearSingularA22AboveFloorUnchanged(t *testing.T) {
	sys := stabRotZ776.sys(t)
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	red, _, err := Balred(sys, 1, BalredOptions{StateProjection: MatchDC})
	if err != nil {
		t.Fatal(err)
	}
	want, err := balredModredOf(t, br, 1, MatchDC)
	if err != nil || !sameModel(red, want) {
		t.Fatalf("Balred(1) not bit-identical to Modred of Balreal (%v)", err)
	}
	if residualizedBlockSingular(br.Sys, 1, stabRotZ776.n) {
		t.Fatal("block reported singular")
	}
	requireStable(t, "Balred(1)", red)
}

// M3HDEP: Truncate of a pair equal to rounding (rotQ/270) obeys the
// stability contract, whichever side of the band the kept pole falls.
func TestBalred_EqualPairTruncateOutcome(t *testing.T) {
	sys := stabRotQ270.sys(t)
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	for _, proj := range []StateProjection{MatchDC, Truncate} {
		for k := range 2 {
			balredStabilityContract(t, sys, br, k, proj)
		}
	}
}
