package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
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

// TestDDEntryStagesMatchPerRowResponse checks the staged double-double
// response (HO6UP6): ddEntryBounds, one factorization for all output rows,
// equals zpkDDResponseWork row by row in every field, bitwise, and
// ddEntrySolve's response is bitwise ddEntryBounds' with terms T̂ ≤ hi,
// on random non-symmetric MIMO models at axis points and beside poles.
func TestDDEntryStagesMatchPerRowResponse(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 11))
	bits := func(z complex128) [2]uint64 { return [2]uint64{math.Float64bits(real(z)), math.Float64bits(imag(z))} }
	checked := 0
	for trial := range 60 {
		n, m, p := 1+rng.IntN(6), 1+rng.IntN(3), 1+rng.IntN(3)
		a, b, c := make([]float64, n*n), make([]float64, n*m), make([]float64, p*n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		for i := range c {
			c[i] = rng.NormFloat64()
		}
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		sys, err := NewFromSlices(n, m, p, a, b, c, make([]float64, p*m), dt)
		if err != nil {
			t.Fatal(err)
		}
		br := newBalancedRealization(sys, n, m, p)
		pts := []complex128{0.3i, 2.5i, 1e-3i}
		if dt != 0 {
			pts = []complex128{cmplx.Rect(1, 0.3), cmplx.Rect(1, 2.5), -1}
		}
		evals, _, ok := eigenRoundingBounds(sys.A)
		if ok {
			for _, ev := range evals {
				pts = append(pts, ev+complex(0, 1e-9))
			}
		}
		var w zpkDDWork
		full, solve := make([]zpkEntryBounds, p*m), make([]zpkEntryBounds, p*m)
		for _, s := range pts {
			okFull := ddEntryBounds(&br, s, &w, full)
			okSolve := ddEntrySolve(&br, s, &w, solve)
			rowsOK := true
			for i := range p {
				row := br
				row.p, row.c = 1, br.c[i*n:(i+1)*n]
				got, _, ok := zpkDDResponseWork(&row, s, &zpkDDWork{})
				if !ok {
					rowsOK = false
					break
				}
				if okFull && !slices.Equal(got, full[i*m:(i+1)*m]) {
					t.Fatalf("trial %d s=%v row %d: staged %v, per-row %v", trial, s, i, full[i*m:(i+1)*m], got)
				}
			}
			if okFull != rowsOK {
				t.Fatalf("trial %d s=%v: staged ok %v, per-row ok %v", trial, s, okFull, rowsOK)
			}
			if !okFull {
				continue
			}
			if !okSolve {
				t.Fatalf("trial %d s=%v: solve stage failed where the bounds formed", trial, s)
			}
			for i := range full {
				if bits(solve[i].g) != bits(full[i].g) || !(solve[i].hi <= full[i].hi) || solve[i].comp != 0 {
					t.Fatalf("trial %d s=%v entry %d: solve %+v, full %+v", trial, s, i, solve[i], full[i])
				}
			}
			checked++
		}
	}
	if checked < 200 {
		t.Errorf("only %d points checked", checked)
	}
}
