package controlsys

import (
	"context"
	"fmt"
	"maps"
)

// TuneMethod names the search a tuning result came from.
type TuneMethod string

// TuneMethodCartesianGrid is the exhaustive search over a Cartesian grid of
// the free parameters' bounds.
const TuneMethodCartesianGrid TuneMethod = "cartesian-grid"

// SystuneOptions configures GridTune and Systune. The zero value selects
// 5 grid points per parameter and at most 100000 evaluations; negative
// values return ErrInvalidArgument.
type SystuneOptions struct {
	// GridPoints is the number of grid points per free parameter, spanning
	// its finite bounds; 1 holds every parameter at its current value.
	GridPoints int
	// MaxEvaluations bounds the grid size; a larger grid is rejected.
	MaxEvaluations int
}

// SystuneResult is the best grid candidate of GridTune, Systune or Looptune.
type SystuneResult struct {
	Method TuneMethod
	// Pass reports whether every hard goal (for GridTune, every goal) is met.
	Pass bool
	// Score is the summed normalized violation of the soft goals (for
	// GridTune, of all goals), MATLAB systune's fSoft analogue.
	Score float64
	// HardScore is the summed normalized violation of the hard goals; 0 when
	// all are met.
	HardScore  float64
	Iterations int
	Parameters map[string]float64
	Controller *System
	ClosedLoop *System
	// Goals holds the soft goal results followed by the hard goal results.
	Goals []TuningGoalResult
}

// LooptuneOptions configures Looptune. Zero margins select MATLAB's
// looptuneOptions defaults of 7.6 dB and 45°.
type LooptuneOptions struct {
	SystuneOptions
	GainMarginDB   float64
	PhaseMarginDeg float64
}

// GridTune searches a Cartesian grid of the controller's free parameters for
// the candidate with the smallest summed goal violation. Every free parameter
// needs finite bounds (ErrInvalidArgument otherwise). A candidate whose
// evaluation fails (for example an unstable closed loop under an overshoot
// goal) or scores NaN/Inf is skipped as infeasible; when no candidate scores,
// the first candidate error is returned, or ErrNoTuningCandidate. ctx is
// checked before each evaluation.
func GridTune(ctx context.Context, model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	res, err := gridSearch(ctx, model, goals, nil, opts)
	if err != nil {
		return nil, fmt.Errorf("GridTune: %w", err)
	}
	res.Pass = allGoalsPass(res.Goals)
	return res, nil
}

// Systune tunes the controller of CL0 to minimize the soft goals subject to
// the hard goals, as MATLAB systune(CL0, SoftReqs, HardReqs, opts), by
// Cartesian grid search (see GridTune). Among candidates meeting every hard
// goal the smallest soft score wins; when none does, the candidate with the
// smallest hard violation is returned with Pass false, as MATLAB returns
// gHard > 1. See https://www.mathworks.com/help/control/ref/inputoutputmodel.systune.html.
func Systune(ctx context.Context, CL0 *GeneralizedClosedLoop, soft, hard []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	res, err := gridSearch(ctx, CL0, soft, hard, opts)
	if err != nil {
		return nil, fmt.Errorf("Systune: %w", err)
	}
	return res, nil
}

// Looptune tunes C0 in the negative feedback loop with plant G0, as MATLAB
// looptune(G0, C0, wc, Req1, ..., opts). wc is a target crossover frequency
// or a band [wcmin, wcmax] in rad/s. Looptune adds
// NewLoopShapeGoalWc("looptune", wc) and NewMarginsGoal("looptune", gm, pm)
// (opts margins, default 7.6 dB and 45°) to reqs and treats them all as hard
// goals in Systune. The loop's analysis point is "looptune": its closed loop
// has C0's InputName and G0's OutputName, so reqs can name those channels or
// "looptune". See
// https://www.mathworks.com/help/control/ref/dynamicsystem.looptune.html.
func Looptune(ctx context.Context, G0 *System, C0 TunableBlock, wc []float64, reqs []TuningGoal, opts *LooptuneOptions) (*SystuneResult, error) {
	if C0 == nil {
		return nil, fmt.Errorf("Looptune: C0 is nil: %w", ErrInvalidArgument)
	}
	gm, pm := 7.6, 45.0
	var sopts *SystuneOptions
	if opts != nil {
		if opts.GainMarginDB < 0 || opts.PhaseMarginDeg < 0 {
			return nil, fmt.Errorf("Looptune: negative margin: %w", ErrInvalidArgument)
		}
		if opts.GainMarginDB > 0 {
			gm = opts.GainMarginDB
		}
		if opts.PhaseMarginDeg > 0 {
			pm = opts.PhaseMarginDeg
		}
		sopts = &opts.SystuneOptions
	}
	crossover, err := NewLoopShapeGoalWc("looptune", wc)
	if err != nil {
		return nil, fmt.Errorf("Looptune: %w", err)
	}
	margins, err := NewMarginsGoal("looptune", gm, pm)
	if err != nil {
		return nil, fmt.Errorf("Looptune: %w", err)
	}
	loop, err := NewGeneralizedClosedLoop("looptune", G0, C0, "looptune")
	if err != nil {
		return nil, fmt.Errorf("Looptune: %w", err)
	}
	hard := append([]TuningGoal{crossover, margins}, reqs...)
	res, err := gridSearch(ctx, loop, nil, hard, sopts)
	if err != nil {
		return nil, fmt.Errorf("Looptune: %w", err)
	}
	return res, nil
}

type tuneCandidate struct {
	hard, soft float64
}

func (c tuneCandidate) better(o tuneCandidate) bool {
	if c.hard != o.hard {
		return c.hard < o.hard
	}
	return c.soft < o.soft
}

func gridSearch(ctx context.Context, model *GeneralizedClosedLoop, soft, hard []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("ctx is nil: %w", ErrInvalidArgument)
	}
	if model == nil {
		return nil, fmt.Errorf("model is nil: %w", ErrInvalidArgument)
	}
	controller := model.tunableController
	if controller == nil {
		return nil, fmt.Errorf("controller is not tunable: %w", ErrInvalidArgument)
	}
	if len(soft)+len(hard) == 0 {
		return nil, fmt.Errorf("no goals: %w", ErrInvalidArgument)
	}
	gridPoints := 5
	maxEvaluations := 100_000
	if opts != nil {
		if opts.GridPoints < 0 || opts.MaxEvaluations < 0 {
			return nil, fmt.Errorf("GridPoints %d and MaxEvaluations %d must be non-negative: %w", opts.GridPoints, opts.MaxEvaluations, ErrInvalidArgument)
		}
		if opts.GridPoints > 0 {
			gridPoints = opts.GridPoints
		}
		if opts.MaxEvaluations > 0 {
			maxEvaluations = opts.MaxEvaluations
		}
	}
	params := controller.FreeParameters()
	if len(params) == 0 {
		return nil, fmt.Errorf("no free tunable parameters: %w", ErrInvalidArgument)
	}
	grids := make([][]float64, len(params))
	evaluationCount := 1
	for i, param := range params {
		grid, err := parameterGrid(param, gridPoints)
		if err != nil {
			return nil, err
		}
		grids[i] = grid
		if evaluationCount > maxEvaluations/len(grid) {
			return nil, fmt.Errorf("Cartesian grid exceeds %d evaluations: %w", maxEvaluations, ErrInvalidArgument)
		}
		evaluationCount *= len(grid)
	}
	goals := append(append([]TuningGoal(nil), soft...), hard...)

	var best *SystuneResult
	var bestKey tuneCandidate
	var firstErr error
	iterations := 0
	values := make(map[string]float64, len(params))
	evaluate := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		iterations++
		sampled, err := controller.SampleBlock(values)
		if err != nil {
			return err
		}
		candidate := model.withSampledController(sampled)
		closed, err := candidate.ClosedLoop(candidate.primaryAnalysisPointName())
		if err == nil {
			var results []TuningGoalResult
			results, err = evaluateTuningGoals(candidate, closed, goals)
			if err == nil {
				key := tuneCandidate{soft: sumViolation(results[:len(soft)]), hard: sumViolation(results[len(soft):])}
				if !isFinite(key.soft) || !isFinite(key.hard) || (best != nil && !key.better(bestKey)) {
					return nil
				}
				ctrl, err := sampled.CurrentSystem()
				if err != nil {
					return err
				}
				bestKey = key
				best = &SystuneResult{
					Method:     TuneMethodCartesianGrid,
					Pass:       allGoalsPass(results[len(soft):]),
					Score:      key.soft,
					HardScore:  key.hard,
					Parameters: copyStringFloatMap(values),
					Controller: ctrl,
					ClosedLoop: closed,
					Goals:      results,
				}
				return nil
			}
		}
		if firstErr == nil {
			firstErr = fmt.Errorf("candidate %v: %w", copyStringFloatMap(values), err)
		}
		return nil
	}
	var search func(int) error
	search = func(idx int) error {
		if idx == len(params) {
			return evaluate()
		}
		for _, value := range grids[idx] {
			values[params[idx].Name()] = value
			if err := search(idx + 1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := search(0); err != nil {
		return nil, err
	}
	if best == nil {
		if firstErr != nil {
			return nil, firstErr
		}
		return nil, fmt.Errorf("%d candidates all scored NaN or Inf: %w", iterations, ErrNoTuningCandidate)
	}
	best.Iterations = iterations
	return best, nil
}

func sumViolation(results []TuningGoalResult) float64 {
	total := 0.0
	for _, r := range results {
		total += r.Violation
	}
	return total
}

func allGoalsPass(results []TuningGoalResult) bool {
	for _, r := range results {
		if !r.Pass {
			return false
		}
	}
	return true
}

func evaluateTuningGoals(model *GeneralizedClosedLoop, primaryClosedLoop *System, goals []TuningGoal) ([]TuningGoalResult, error) {
	results := make([]TuningGoalResult, len(goals))
	cache := map[tuningGoalResponseKey]*System{
		{point: model.primaryAnalysisPointName(), response: tuningGoalClosedLoopResponse}: primaryClosedLoop,
	}
	for i, goal := range goals {
		if err := goal.valid(); err != nil {
			return nil, fmt.Errorf("goal %d: %w", i, err)
		}
		key, err := goal.responseKey(model)
		if err != nil {
			return nil, fmt.Errorf("goal %q: %w", goal.Name(), err)
		}
		sys := cache[key]
		if sys == nil {
			if sys, err = model.goalResponse(key); err != nil {
				return nil, fmt.Errorf("goal %q: %w", goal.Name(), err)
			}
			cache[key] = sys
		}
		if sys, err = goal.selectResponse(model, sys); err != nil {
			return nil, fmt.Errorf("goal %q: %w", goal.Name(), err)
		}
		result, err := goal.evaluateSystem(sys)
		if err != nil {
			return nil, fmt.Errorf("goal %q: %w", goal.Name(), err)
		}
		results[i] = result
	}
	return results, nil
}

func parameterGrid(param *TunableReal, points int) ([]float64, error) {
	bounds := param.Bounds()
	if points < 2 {
		return []float64{param.Value()}, nil
	}
	if !bounds.finite() {
		return nil, fmt.Errorf("free parameter %q has bounds [%g, %g]; grid search needs finite bounds: %w", param.Name(), bounds.Lower, bounds.Upper, ErrInvalidArgument)
	}
	out := make([]float64, points)
	for i := range out {
		out[i] = bounds.Lower + float64(i)*(bounds.Upper-bounds.Lower)/float64(points-1)
	}
	return out, nil
}

func copyStringFloatMap(src map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(src))
	maps.Copy(out, src)
	return out
}
