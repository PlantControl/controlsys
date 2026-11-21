package controlsys

import (
	"math"
	"testing"
)

// syntheticPoint is a one-point boundedResponse of shape p×m with the given
// entries and bounds, marked sharp so classifyProbe judges it as is.
func syntheticPoint(p, m int, g []complex128, comp, ver, data, terms []float64) *boundedResponse {
	return &boundedResponse{p: p, m: m, g: g, comp: comp, ver: ver, data: data, terms: terms,
		ok: []bool{true}, sharp: []bool{true}}
}

func smallBounds(k int) []float64 {
	out := make([]float64, k)
	for i := range out {
		out[i] = 1e-20
	}
	return out
}

// A probe whose model response overflows in one entry has no finite scale:
// it must neither verify (Inf ≤ Inf) nor refute (QMVNHR).
func TestBandProbeOverflowEntryNotVerified(t *testing.T) {
	tol := math.Sqrt(eps())
	full := syntheticPoint(1, 2, []complex128{1, complex(math.Inf(1), 0)}, smallBounds(2), smallBounds(2), smallBounds(2), []float64{1, 1})
	cut := syntheticPoint(1, 2, []complex128{2, 1}, smallBounds(2), smallBounds(2), smallBounds(2), []float64{1, 1})
	if ok, _ := probeMargin(full, cut, 0, tol); ok {
		t.Fatal("probeMargin verifies a point with an overflowed entry")
	}
	if probeRefuted(full, cut, 0, tol, false) {
		t.Fatal("probeRefuted refutes on a point with no finite scale")
	}
	if v := classifyProbe(full, cut, 0, tol, false); v != probeUninformative {
		t.Fatalf("classifyProbe = %v, want uninformative", v)
	}
}

// An entry whose bounds or terms overflow while every response is finite
// is undetermined alone: a finite wrong entry still refutes, before or
// after it, and a finite right one does not verify the point (QMVNHR,
// 5CFETF ruling 2b).
func TestBandProbeNonFiniteBoundEntrySkipped(t *testing.T) {
	tol := math.Sqrt(eps())
	inf := math.Inf(1)
	for _, bad := range []int{0, 1} {
		good := 1 - bad
		terms := []float64{1, 1}
		terms[bad] = inf
		full := syntheticPoint(1, 2, []complex128{1, 1}, smallBounds(2), smallBounds(2), smallBounds(2), terms)
		rg := []complex128{1, 1}
		rg[good] = 2
		cut := syntheticPoint(1, 2, rg, smallBounds(2), smallBounds(2), smallBounds(2), []float64{1, 1})
		if !probeRefuted(full, cut, 0, tol, false) {
			t.Errorf("bad=%d: finite wrong entry not refuted", bad)
		}
		if v := classifyProbe(full, cut, 0, tol, false); v != probeReject {
			t.Errorf("bad=%d: classifyProbe = %v, want reject", bad, v)
		}
		cut.g[good] = 1
		if ok, _ := probeMargin(full, cut, 0, tol); ok {
			t.Errorf("bad=%d: point with a non-finite entry verified", bad)
		}
		if v := classifyProbe(full, cut, 0, tol, false); v != probeUninformative {
			t.Errorf("bad=%d: classifyProbe = %v, want uninformative", bad, v)
		}
	}
}

// sharpen marks a point it cannot bound with comp = ver = +Inf; on the
// cut that is undetermined, not cut noise that rejects (QMVNHR).
func TestBandProbeUnboundedCutNotRejected(t *testing.T) {
	tol := math.Sqrt(eps())
	inf := math.Inf(1)
	full := syntheticPoint(1, 1, []complex128{1}, smallBounds(1), smallBounds(1), smallBounds(1), []float64{1})
	cut := syntheticPoint(1, 1, []complex128{1}, []float64{inf}, []float64{inf}, smallBounds(1), []float64{1})
	if v := classifyProbe(full, cut, 0, tol, false); v != probeUninformative {
		t.Fatalf("classifyProbe = %v, want uninformative", v)
	}
}

// The double-double verdict and cut verification of judgeNearPoles and
// resolveOpenProbes neither verify by an overflowed response nor by an
// overflowed terms bound lo (QMVNHR).
func TestBandDDJudgeNonFiniteNotVerified(t *testing.T) {
	tol := math.Sqrt(eps())
	inf := math.Inf(1)
	d := &ddJudge{tol: tol}
	d.g = []zpkEntryBounds{{g: 1, ver: 1e-20, comp: 1e-20, lo: 1, hi: 1, determined: true}, {g: complex(inf, 0), lo: 1, hi: 1, determined: true}}
	d.r = []zpkEntryBounds{{g: 2, lo: 1, hi: 1, determined: true}, {g: 1, lo: 1, hi: 1, determined: true}}
	if v := d.verdict(); v != probeUninformative {
		t.Errorf("overflowed response: verdict = %v, want uninformative", v)
	}
	d.g = []zpkEntryBounds{{g: 1, ver: 1e-20, comp: 1e-20, lo: inf, hi: inf, determined: true}}
	d.r = []zpkEntryBounds{{g: 1.5}}
	if v := d.verdict(); v != probeUninformative {
		t.Errorf("overflowed lo: verdict = %v, want uninformative", v)
	}
	cb := syntheticPoint(1, 1, []complex128{1.5}, smallBounds(1), smallBounds(1), smallBounds(1), []float64{1})
	if d.cutVerifies(cb, 0, false) {
		t.Error("cutVerifies verifies by an overflowed lo")
	}
	d.g = []zpkEntryBounds{{g: complex(inf, 0), lo: 1, hi: 1, determined: true}}
	if d.cutVerifies(cb, 0, false) {
		t.Error("cutVerifies verifies by an overflowed response")
	}
}
