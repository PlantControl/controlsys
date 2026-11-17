package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// TestReduceNearPoleJudgeDoubleCutAgreesWithDoubleDouble checks that
// judgeNearPoles' shortcuts by the cut's double response keep ddJudge's
// verdict on the staircase cuts of axisPoleLiterals in every mode: at
// every probe and at every pole's foot and side points, refutes is judge's
// rejection, and verifies holds where judge verifies and never where it
// rejects.
func TestReduceNearPoleJudgeDoubleCutAgreesWithDoubleDouble(t *testing.T) {
	tol := math.Sqrt(eps())
	modes := []*ReduceOpts{{}, {Equalize: true}, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}}
	names := make([]string, 0, len(axisPoleLiterals))
	for k := range axisPoleLiterals {
		names = append(names, k)
	}
	slices.Sort(names)
	checked, skipped := 0, 0
	for _, name := range names {
		sys := axisPoleLiterals[name].system(t)
		n, _, _ := sys.Dims()
		for _, opts := range modes {
			cut, _, err := reduceByStaircase(sys, newRealizationTransformPolicy(sys), opts, true, -1)
			if err != nil || cut.Order == n || cut.Order == 0 {
				continue
			}
			pts, mp, ok := axisProbes(sys, nil)
			if !ok {
				t.Fatalf("%s: eigenvalues did not converge", name)
			}
			full, red := newBoundedResponse(sys, pts), newBoundedResponse(cut.Sys, pts)
			for k := range pts {
				full.sharpen(k)
				red.sharpen(k)
			}
			ref, dd := newDDJudge(&full, tol), newDDJudge(&full, tol)
			for k, s := range pts {
				want := ref.judge(&full, &red, s) == probeReject
				if got := dd.refutes(&full, &red, &red, k, s); got != want {
					t.Errorf("%s %+v: refutes %v at probe %v, judge rejects %v", name, *opts, got, s, want)
				}
				if dd.cutCannotRefute(&red, k, false) {
					skipped++
				}
				checked++
			}
			var near []complex128
			for j, ev := range mp.evals {
				foot, d := mp.foot[j], mp.reach[j]
				near = append(near, foot)
				if sys.Dt != 0 {
					th := cmplx.Phase(foot)
					near = append(near, cmplx.Rect(1, th+d), cmplx.Rect(1, th-d))
				} else {
					near = append(near, complex(0, imag(ev)+d), complex(0, imag(ev)-d))
				}
			}
			cb := boundedResponseOf(red.ev.br, near)
			for k, s := range near {
				v := ref.judge(&full, &red, s)
				if got := dd.refutes(&full, &red, &cb, k, s); got != (v == probeReject) {
					t.Errorf("%s %+v: refutes %v at %v, judge %v", name, *opts, got, s, v)
				}
				got := dd.verifies(&full, &red, &cb, k, s)
				if v == probeVerified && !got || v == probeReject && got {
					t.Errorf("%s %+v: verifies %v at %v, judge %v", name, *opts, got, s, v)
				}
				checked++
			}
			cb.release()
			full.release()
			red.release()
		}
	}
	if checked == 0 || skipped == 0 {
		t.Fatalf("checked %d points, %d by the cut's double bounds alone", checked, skipped)
	}
}
