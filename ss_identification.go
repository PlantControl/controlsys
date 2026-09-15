package controlsys

import (
	"context"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// IOStateSpaceOptions identifies SISO models from arbitrary sampled input/output.
// Order excludes explicit input-delay states. Validation never fits dynamics.
type IOStateSpaceOptions struct {
	Order                      int    `json:"order,omitempty"`
	MinOrder                   int    `json:"minOrder,omitempty"`
	MaxOrder                   int    `json:"maxOrder,omitempty"`
	DirectFeedthrough          bool   `json:"directFeedthrough"`
	InputDelay                 int    `json:"inputDelay"`
	InitialCondition           string `json:"initialCondition"`
	ValidationInitialCondition string `json:"validationInitialCondition"`
	InitializationSamples      int    `json:"initializationSamples,omitempty"`
	MaxEvaluations             int    `json:"maxEvaluations,omitempty"`
}

type IOStateSpaceCandidate struct {
	TrainingPredicted               []float64 `json:"trainingPredicted,omitempty"`
	ValidationPredicted             []float64 `json:"validationPredicted,omitempty"`
	Order                           int       `json:"order"`
	Numerator                       []float64 `json:"numerator,omitempty"`
	Denominator                     []float64 `json:"denominator,omitempty"`
	SingularValues                  []float64 `json:"singularValues,omitempty"`
	Rank                            int       `json:"rank"`
	Condition                       float64   `json:"condition,omitempty"`
	TrainingNRMSE                   float64   `json:"trainingNrmse"`
	ValidationNRMSE                 float64   `json:"validationNrmse"`
	ValidationBaselineNRMSE         float64   `json:"validationBaselineNrmse"`
	ResidualAutocorrelation         float64   `json:"residualAutocorrelation"`
	ResidualInputCorrelation        float64   `json:"residualInputCorrelation"`
	InitialHistory                  []float64 `json:"initialHistory,omitempty"`
	ValidationInitializationSamples int       `json:"validationInitializationSamples"`
	Stable                          bool      `json:"stable"`
	Convergence                     string    `json:"convergence"`
	Failure                         string    `json:"failure,omitempty"`
	System                          *System   `json:"-"`
}

type IOStateSpaceResult struct {
	Options         IOStateSpaceOptions     `json:"options"`
	Candidates      []IOStateSpaceCandidate `json:"candidates"`
	Selected        int                     `json:"selected"`
	SelectionReason string                  `json:"selectionReason"`
	Evaluations     int                     `json:"evaluations"`
}

func IdentifyIOStateSpace(ctx context.Context, trainU, trainY, validationU, validationY []float64, dt float64, options IOStateSpaceOptions) (*IOStateSpaceResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(trainU) != len(trainY) || len(validationU) != len(validationY) || len(trainU) < 20 || len(validationU) < 2 || len(trainU)+len(validationU) > 10000 {
		return nil, fmt.Errorf("identify state space: matching training/validation records require 20/2 minimum and 10000 total samples maximum")
	}
	if !finitePID(dt) || dt <= 0 {
		return nil, fmt.Errorf("identify state space: sample time must be positive and finite")
	}
	for _, values := range [][]float64{trainU, trainY, validationU, validationY} {
		for _, v := range values {
			if !finitePID(v) {
				return nil, fmt.Errorf("identify state space: samples must be finite")
			}
		}
	}
	if options.Order != 0 {
		options.MinOrder, options.MaxOrder = options.Order, options.Order
	}
	if options.MinOrder == 0 {
		options.MinOrder = 1
	}
	if options.MaxOrder == 0 {
		options.MaxOrder = 6
	}
	if options.MinOrder < 1 || options.MaxOrder > 12 || options.MaxOrder < options.MinOrder {
		return nil, fmt.Errorf("identify state space: orders must lie between 1 and 12")
	}
	if options.InputDelay < 0 || options.InputDelay > 100 {
		return nil, fmt.Errorf("identify state space: input delay must be 0 to 100 samples")
	}
	if options.MaxEvaluations == 0 {
		options.MaxEvaluations = 2000
	}
	if options.MaxEvaluations < 1 || options.MaxEvaluations > 2000 {
		return nil, fmt.Errorf("identify state space: evaluation budget must be 1 to 2000")
	}
	if options.InitialCondition == "" {
		options.InitialCondition = "estimate"
	}
	if options.InitialCondition != "zero" && options.InitialCondition != "estimate" {
		return nil, fmt.Errorf("identify state space: initial condition must be zero or estimate")
	}
	if options.ValidationInitialCondition == "" {
		options.ValidationInitialCondition = "continuation"
	}
	switch options.ValidationInitialCondition {
	case "continuation", "zero":
		if options.InitializationSamples != 0 {
			return nil, fmt.Errorf("identify state space: initialization samples require estimate validation mode")
		}
	case "estimate":
		if options.InitializationSamples < options.MaxOrder || options.InitializationSamples >= len(validationU)-1 {
			return nil, fmt.Errorf("identify state space: declare at least maxOrder initialization samples, leaving at least two held-out samples")
		}
	default:
		return nil, fmt.Errorf("identify state space: invalid validation initialization mode")
	}
	variance := ioVariance(trainU)
	if variance <= 1e-24 {
		return nil, fmt.Errorf("identify state space: insufficient changing input excitation")
	}
	result := &IOStateSpaceResult{Options: options, Selected: -1}
	for order := options.MinOrder; order <= options.MaxOrder; order++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := ioFitOrder(ctx, trainU, trainY, validationU, validationY, dt, order, options, &result.Evaluations)
		result.Candidates = append(result.Candidates, candidate)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	best := math.Inf(1)
	for _, candidate := range result.Candidates {
		if candidate.Failure == "" && candidate.ValidationNRMSE < best {
			best = candidate.ValidationNRMSE
		}
	}
	if math.IsInf(best, 1) {
		return nil, fmt.Errorf("identify state space: no identifiable candidate: %s", result.Candidates[0].Failure)
	}
	// The numerical floor prevents roundoff deciding between equivalent noiseless orders.
	threshold := math.Max(best*1.01, 1e-8)
	for i, candidate := range result.Candidates {
		if candidate.Failure == "" && candidate.ValidationNRMSE <= threshold {
			result.Selected = i
			break
		}
	}
	result.SelectionReason = "Lowest order within 1% of the minimum held-out normalized RMS error (numerical floor 1e-8). Dynamics and initial conditions use training data only; a declared validation initialization prefix is excluded from scoring."
	return result, nil
}

func ioFitOrder(ctx context.Context, u, y, vu, vy []float64, dt float64, order int, options IOStateSpaceOptions, evaluations *int) IOStateSpaceCandidate {
	candidate := IOStateSpaceCandidate{Order: order}
	fail := func(reason string) IOStateSpaceCandidate {
		candidate.Failure = reason
		candidate.Convergence = "failed"
		return candidate
	}
	nb := order
	if options.DirectFeedthrough {
		nb++
	}
	nk := options.InputDelay
	if !options.DirectFeedthrough {
		nk++
	}
	skip := max(order, nk+nb-1)
	rows := len(u) - skip
	columns := order + nb
	if rows < 10*columns {
		return fail("insufficient training rows: require ten rows per fitted coefficient")
	}
	x := mat.NewDense(rows, columns, nil)
	target := mat.NewDense(rows, 1, nil)
	for k := skip; k < len(u); k++ {
		for j := range order {
			x.Set(k-skip, j, y[k-j-1])
		}
		for j := range nb {
			x.Set(k-skip, order+j, u[k-nk-j])
		}
		target.Set(k-skip, 0, y[k])
	}
	scales := make([]float64, columns)
	for j := range columns {
		sum := 0.
		for k := range rows {
			sum += x.At(k, j) * x.At(k, j)
		}
		scales[j] = math.Sqrt(sum)
		if scales[j] == 0 {
			return fail("rank-deficient regression: insufficient excitation")
		}
		for k := range rows {
			x.Set(k, j, x.At(k, j)/scales[j])
		}
	}
	var svd mat.SVD
	if !svd.Factorize(x, mat.SVDThin) {
		return fail("regression singular-value decomposition failed")
	}
	candidate.SingularValues = svd.Values(nil)
	candidate.Rank = svd.Rank(1e-10)
	if candidate.Rank < columns {
		return fail("rank-deficient regression: insufficient excitation or redundant model order")
	}
	candidate.Condition = candidate.SingularValues[0] / candidate.SingularValues[len(candidate.SingularValues)-1]
	var qr mat.QR
	qr.Factorize(x)
	var fitted mat.Dense
	if err := qr.SolveTo(&fitted, false, target); err != nil {
		return fail("QR regression failed: " + err.Error())
	}
	theta := make([]float64, columns)
	for j := range theta {
		theta[j] = fitted.At(j, 0) / scales[j]
	}
	if options.InitialCondition == "estimate" {
		history, err := ioEstimateHistory(theta, u, y, order, nb, nk)
		if err != nil {
			return fail(err.Error())
		}
		theta = append(theta, history...)
	}
	theta, training, convergence, err := ioRefine(ctx, theta, u, y, order, nb, nk, options.MaxEvaluations, evaluations)
	if err != nil {
		return fail(err.Error())
	}
	candidate.Convergence = convergence
	candidate.TrainingPredicted = append([]float64(nil), training...)
	candidate.TrainingNRMSE = ioNormalizedError(training, y)
	if len(theta) > columns {
		candidate.InitialHistory = append([]float64(nil), theta[columns:]...)
	}
	var prediction []float64
	start := 0
	switch options.ValidationInitialCondition {
	case "continuation":
		all := append(append([]float64(nil), u...), vu...)
		combined, _, valid := ioPredict(theta, all, order, nb, nk, false)
		if !valid {
			return fail("validation free run diverged")
		}
		prediction = combined[len(u):]
	case "zero":
		var valid bool
		prediction, _, valid = ioPredict(theta[:columns], vu, order, nb, nk, false)
		if !valid {
			return fail("validation free run diverged")
		}
	case "estimate":
		history, e := ioEstimateHistory(theta[:columns], vu[:options.InitializationSamples], vy[:options.InitializationSamples], order, nb, nk)
		if e != nil {
			return fail("validation initialization: " + e.Error())
		}
		validationTheta := append(append([]float64(nil), theta[:columns]...), history...)
		var valid bool
		prediction, _, valid = ioPredict(validationTheta, vu, order, nb, nk, false)
		if !valid {
			return fail("validation free run diverged")
		}
		start = options.InitializationSamples
	}
	candidate.ValidationPredicted = append([]float64(nil), prediction...)
	candidate.ValidationInitializationSamples = start
	candidate.ValidationNRMSE = ioNormalizedError(prediction[start:], vy[start:])
	baseline := make([]float64, len(vy)-start)
	mean := ioMean(y)
	for k := range baseline {
		baseline[k] = mean
	}
	candidate.ValidationBaselineNRMSE = ioNormalizedError(baseline, vy[start:])
	residual := make([]float64, len(vy)-start)
	for k := range residual {
		residual[k] = vy[k+start] - prediction[k+start]
	}
	candidate.ResidualAutocorrelation, candidate.ResidualInputCorrelation = ioResidualCorrelation(residual, vu[start:])
	den := make([]float64, order+options.InputDelay+1)
	den[0] = 1
	for j := range order {
		den[j+1] = -theta[j]
	}
	num := append([]float64(nil), theta[order:columns]...)
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	realized, e := tf.StateSpace(nil)
	if e != nil {
		return fail("state-space realization: " + e.Error())
	}
	candidate.Numerator, candidate.Denominator, candidate.System = num, den, realized.Sys
	candidate.Stable, e = candidate.System.IsStable()
	if e != nil {
		return fail("stability analysis: " + e.Error())
	}
	return candidate
}

func ioPredict(theta, u []float64, n, nb, nk int, jacobian bool) ([]float64, *mat.Dense, bool) {
	columns := len(theta)
	dynamics := n + nb
	y := make([]float64, len(u))
	var j *mat.Dense
	if jacobian {
		j = mat.NewDense(len(u), columns, nil)
	}
	for k := range u {
		value := 0.
		for lag := range n {
			index := k - lag - 1
			previous := 0.
			if index >= 0 {
				previous = y[index]
			} else if dynamics-index-1 < len(theta) {
				previous = theta[dynamics-index-1]
			}
			value += theta[lag] * previous
			if jacobian {
				j.Set(k, lag, j.At(k, lag)+previous)
				if index >= 0 {
					for c := range columns {
						j.Set(k, c, j.At(k, c)+theta[lag]*j.At(index, c))
					}
				} else if dynamics-index-1 < len(theta) {
					c := dynamics - index - 1
					j.Set(k, c, j.At(k, c)+theta[lag])
				}
			}
		}
		for lag := range nb {
			index := k - nk - lag
			if index >= 0 {
				value += theta[n+lag] * u[index]
				if jacobian {
					j.Set(k, n+lag, j.At(k, n+lag)+u[index])
				}
			}
		}
		if !finitePID(value) || math.Abs(value) > 1e100 {
			return nil, nil, false
		}
		y[k] = value
	}
	return y, j, true
}

func ioEstimateHistory(theta, u, y []float64, n, nb, nk int) ([]float64, error) {
	extended := append(append([]float64(nil), theta[:n+nb]...), make([]float64, n)...)
	forced, j, valid := ioPredict(extended, u, n, nb, nk, true)
	if !valid {
		return nil, fmt.Errorf("initial-state forced response diverged")
	}
	h := mat.NewDense(len(u), n, nil)
	target := mat.NewDense(len(u), 1, nil)
	for k := range u {
		target.Set(k, 0, y[k]-forced[k])
		for c := range n {
			h.Set(k, c, j.At(k, n+nb+c))
		}
	}
	var svd mat.SVD
	if !svd.Factorize(h, mat.SVDThin) {
		return nil, fmt.Errorf("initial-state least squares failed")
	}
	rank := svd.Rank(1e-10)
	history := make([]float64, n)
	if rank == 0 {
		return history, nil
	}
	var solved mat.Dense
	svd.SolveTo(&solved, target, rank)
	for j := range history {
		history[j] = solved.At(j, 0)
	}
	return history, nil
}

func ioRefine(ctx context.Context, theta, u, y []float64, n, nb, nk, maxEvaluations int, evaluations *int) ([]float64, []float64, string, error) {
	if *evaluations >= maxEvaluations {
		return nil, nil, "", fmt.Errorf("training evaluation budget exhausted")
	}
	prediction, j, valid := ioPredict(theta, u, n, nb, nk, true)
	*evaluations++
	if !valid {
		return nil, nil, "", fmt.Errorf("ARX seed free run diverged")
	}
	cost := ioSSE(prediction, y)
	lambda := 1e-3
	for iteration := 0; iteration < 100 && *evaluations < maxEvaluations; iteration++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, "", err
		}
		if cost <= 1e-22*math.Max(1, float64(len(y))*ioVariance(y)) {
			return theta, prediction, "converged", nil
		}
		columns := len(theta)
		rows := len(y)
		augmented := mat.NewDense(rows+columns, columns, nil)
		rhs := mat.NewDense(rows+columns, 1, nil)
		scales := make([]float64, columns)
		for c := range columns {
			sum := 0.
			for k := range rows {
				v := j.At(k, c)
				sum += v * v
			}
			scales[c] = math.Sqrt(sum)
			if !finitePID(scales[c]) {
				return nil, nil, "", fmt.Errorf("output-error sensitivity overflow")
			}
			if scales[c] < 1e-14 {
				scales[c] = 1
			}
			for k := range rows {
				augmented.Set(k, c, j.At(k, c)/scales[c])
			}
			augmented.Set(rows+c, c, math.Sqrt(lambda))
		}
		for k := range rows {
			rhs.Set(k, 0, y[k]-prediction[k])
		}
		var qr mat.QR
		qr.Factorize(augmented)
		var delta mat.Dense
		if err := qr.SolveTo(&delta, false, rhs); err != nil {
			return nil, nil, "", fmt.Errorf("output-error least squares: %w", err)
		}
		trial := append([]float64(nil), theta...)
		norm := 0.
		for c := range trial {
			d := delta.At(c, 0) / scales[c]
			trial[c] += d
			norm += d * d
		}
		next, nextJ, valid := ioPredict(trial, u, n, nb, nk, true)
		*evaluations++
		nextCost := math.Inf(1)
		if valid {
			nextCost = ioSSE(next, y)
		}
		if nextCost < cost {
			relative := (cost - nextCost) / math.Max(cost, 1e-30)
			theta, prediction, j, cost = trial, next, nextJ, nextCost
			lambda = math.Max(1e-12, lambda*.3)
			if relative < 1e-10 || norm < 1e-20 {
				return theta, prediction, "converged", nil
			}
		} else {
			lambda *= 10
			if lambda > 1e12 {
				return theta, prediction, "stationary", nil
			}
		}
	}
	return theta, prediction, "evaluation_or_iteration_limit", nil
}

func ioMean(values []float64) float64 {
	sum := 0.
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
func ioVariance(values []float64) float64 {
	mean := ioMean(values)
	sum := 0.
	for _, v := range values {
		d := v - mean
		sum += d * d
	}
	return sum / float64(len(values))
}
func ioSSE(prediction, truth []float64) float64 {
	sum := 0.
	for k, v := range truth {
		d := v - prediction[k]
		sum += d * d
	}
	return sum
}
func ioNormalizedError(prediction, truth []float64) float64 {
	scale := math.Sqrt(ioVariance(truth))
	if scale < 1e-12 {
		scale = math.Max(1, math.Abs(ioMean(truth)))
	}
	return math.Sqrt(ioSSE(prediction, truth)/float64(len(truth))) / scale
}
func ioResidualCorrelation(residual, input []float64) (auto, cross float64) {
	correlation := func(a, b []float64) float64 {
		ma, mb := ioMean(a), ioMean(b)
		xy, xx, yy := 0., 0., 0.
		for k, v := range a {
			x, y := v-ma, b[k]-mb
			xy += x * y
			xx += x * x
			yy += y * y
		}
		if xx*yy <= 1e-30 {
			return 0
		}
		return xy / math.Sqrt(xx*yy)
	}
	if len(residual) > 2 {
		auto = correlation(residual[1:], residual[:len(residual)-1])
	}
	for lag := 0; lag < min(20, len(residual)-1); lag++ {
		cross = math.Max(cross, math.Abs(correlation(residual[lag:], input[:len(input)-lag])))
	}
	return
}
