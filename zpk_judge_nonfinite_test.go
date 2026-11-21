package controlsys

import (
	"math"
	"testing"
)

// zpkJudgeEntries takes an entry with a non-finite quantity as
// undetermined: an overflowed response (Inf ≤ Inf), an overflowed terms
// bound lo and an overflowed deviation neither verify nor refute, while a
// finite wrong entry still refutes beside one (QMVNHR).
func TestZPKJudgeEntriesNonFinite(t *testing.T) {
	tol := math.Sqrt(eps())
	inf := math.Inf(1)
	c := &zpkRowCheck{tol: tol}
	cand := &zpkCandidate{win: &poleWindows{}}
	judge := func(e []zpkEntryBounds, zv []complex128) zpkPointVerdict {
		return c.zpkJudgeEntries(cand, 1i, e, zv, make([]float64, len(e)))
	}
	if v := judge([]zpkEntryBounds{{g: 2, lo: 1, hi: 1, determined: true}, {g: complex(inf, 0), lo: 1, hi: 1, determined: true}}, []complex128{1, 1}); v != zpkPointUndetermined {
		t.Errorf("overflowed response: %v, want undetermined", v)
	}
	if v := judge([]zpkEntryBounds{{g: 1, lo: inf, hi: inf, determined: true}}, []complex128{1.5}); v != zpkPointUndetermined {
		t.Errorf("overflowed lo: %v, want undetermined", v)
	}
	big := complex(1e308, 1e308)
	if v := judge([]zpkEntryBounds{{g: big, lo: 1, hi: 1, determined: true}}, []complex128{-big}); v != zpkPointUndetermined {
		t.Errorf("overflowed deviation: %v, want undetermined", v)
	}
	for _, bad := range []int{0, 1} {
		e := []zpkEntryBounds{{g: 1, lo: 1, hi: 1, determined: true}, {g: 1, lo: 1, hi: 1, determined: true}}
		zv := []complex128{1, 1}
		e[bad].lo, e[bad].hi = inf, inf
		zv[1-bad] = 2
		if v := judge(e, zv); v != zpkPointRefuted {
			t.Errorf("bad=%d: finite wrong entry: %v, want refuted", bad, v)
		}
	}
}
