package controlsys

import (
	"context"
	"testing"
)

// Regression for TAO5TM: the last grid point rounded above Upper and
// TunableReal.Sample rejected it, aborting the search.
var gridBoundsCases = []struct {
	lower, upper float64
	points       int
}{
	{0.1, 0.4, 8},
	{0.1, 0.5, 4},
	{0.1, 0.9, 4},
}

func gridBoundsGain(t *testing.T, lower, upper float64) *TunableGain {
	t.Helper()
	g, err := NewTunableGain("C", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Gain[0][0].SetValue(lower); err != nil {
		t.Fatal(err)
	}
	if err := g.Gain[0][0].SetBounds(lower, upper); err != nil {
		t.Fatal(err)
	}
	return g
}

func checkGridBoundsResult(t *testing.T, api string, res *SystuneResult, err error, upper float64) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", api, err)
	}
	if k := res.Parameters["C.Gain(1,1)"]; k != upper {
		t.Fatalf("%s: C.Gain(1,1) = %v, want Upper %v", api, k, upper)
	}
}

func TestParameterGridEndsAtBounds(t *testing.T) {
	for _, c := range gridBoundsCases {
		for points := 2; points <= 64; points++ {
			p := gridBoundsGain(t, c.lower, c.upper).Gain[0][0]
			grid, err := parameterGrid(p, points)
			if err != nil {
				t.Fatal(err)
			}
			if grid[0] != c.lower || grid[points-1] != c.upper {
				t.Fatalf("[%v,%v] points %d: ends %v %v", c.lower, c.upper, points, grid[0], grid[points-1])
			}
			for i, v := range grid {
				if v < c.lower || v > c.upper || (i > 0 && v < grid[i-1]) {
					t.Fatalf("[%v,%v] points %d: grid[%d] = %v", c.lower, c.upper, points, i, v)
				}
			}
		}
	}
}

func TestGridTuneReachesUpperBound(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	for _, c := range gridBoundsCases {
		loop, err := NewGeneralizedClosedLoop("loop", plant, gridBoundsGain(t, c.lower, c.upper), "u")
		if err != nil {
			t.Fatal(err)
		}
		goal := mustOK(NewTrackingGoal("u", "u", 1, 0.01, 1.5))
		res, err := GridTune(context.Background(), loop, []TuningGoal{goal}, &SystuneOptions{GridPoints: c.points})
		checkGridBoundsResult(t, "GridTune", res, err, c.upper)
	}
}

func TestSystuneReachesUpperBound(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	for _, c := range gridBoundsCases {
		loop, err := NewGeneralizedClosedLoop("loop", plant, gridBoundsGain(t, c.lower, c.upper), "u")
		if err != nil {
			t.Fatal(err)
		}
		goal := mustOK(NewTrackingGoal("u", "u", 1, 0.01, 1.5))
		res, err := Systune(context.Background(), loop, []TuningGoal{goal}, nil, &SystuneOptions{GridPoints: c.points})
		checkGridBoundsResult(t, "Systune", res, err, c.upper)
	}
}

func TestLooptuneReachesUpperBound(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	for _, c := range gridBoundsCases {
		res, err := Looptune(context.Background(), plant, gridBoundsGain(t, c.lower, c.upper), []float64{1}, nil, &LooptuneOptions{SystuneOptions: SystuneOptions{GridPoints: c.points}})
		checkGridBoundsResult(t, "Looptune", res, err, c.upper)
	}
}
