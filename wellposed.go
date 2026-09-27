package controlsys

import (
	"errors"
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

func solveIdentityMinusProduct(left, right *mat.Dense, size int, context string, singular error) (*mat.Dense, error) {
	return solveIdentityMinusScaledProduct(left, right, 1, size, context, singular)
}

func solveIdentityMinusScaledProduct(left, right *mat.Dense, scale float64, size int, context string, singular error) (*mat.Dense, error) {
	loop := mat.NewDense(size, size, nil)
	loop.Mul(left, right)
	if scale != 1 {
		loop.Scale(scale, loop)
	}
	eye := eyeDense(size)
	loop.Sub(eye, loop)

	var lu mat.LU
	lu.Factorize(loop)
	condition := lu.Cond()
	if nearSingularCondition(condition) {
		return nil, directFeedthroughSolveError(
			context, "is singular", singular, loop, right, condition,
		)
	}

	result := mat.NewDense(size, size, nil)
	if err := lu.SolveTo(result, false, eye); err != nil {
		return nil, directFeedthroughSolveError(
			context, "solve failed", singular, loop, right, condition,
		)
	}
	return result, nil
}

func directFeedthroughSolveError(context, failure string, singular error, loop, feedthrough *mat.Dense, condition float64) error {
	cause := singular
	if errors.Is(singular, ErrAlgebraicLoop) {
		cause = newAlgebraicLoopError(loop, feedthrough, condition)
	}
	return fmt.Errorf("%s: direct feedthrough loop %s: %w", context, failure, cause)
}

func solveFeedbackFeedthrough(plantD, controllerD *mat.Dense, sign float64, size int, context string, singular error) (*mat.Dense, error) {
	return solveIdentityMinusScaledProduct(plantD, controllerD, sign, size, context, singular)
}
