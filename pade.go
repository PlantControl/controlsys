package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// Pade replaces every time delay of a continuous model with its order-th
// order Padé approximant, as MATLAB pade(sys, N)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.pade.html). A
// delay-free model is returned unchanged. order must be in 1..10 (else
// ErrInvalidArgument); discrete models return ErrWrongDomain.
func (sys *System) Pade(order int) (*System, error) {
	if err := requireSystem("Pade", sys); err != nil {
		return nil, err
	}
	if err := validatePadeOrder(order); err != nil {
		return nil, fmt.Errorf("Pade: %w", err)
	}
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Pade: continuous only, use AbsorbDelay for discrete: %w", ErrWrongDomain)
	}
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		return nil, fmt.Errorf("Pade: %w", err)
	}
	if lft.internalDelayCount() == 0 {
		return sys.Copy(), nil
	}
	result, err := padeCloseInternalDelay(lft, order)
	if err != nil {
		return nil, fmt.Errorf("Pade: %w", err)
	}
	propagateIONames(result, sys)
	return result, nil
}

// padeCloseInternalDelay closes every internal delay of lft with an
// order-order Padé approximant. External delays of lft are ignored; callers
// own them.
func padeCloseInternalDelay(lft *System, order int) (*System, error) {
	N := lft.internalDelayCount()
	n, m, p := lft.Dims()

	var delayBank *System
	for j := range N {
		pd, err := PadeDelay(lft.LFT.Tau[j], order)
		if err != nil {
			return nil, fmt.Errorf("delay %d (tau=%v): %w", j, lft.LFT.Tau[j], err)
		}
		if delayBank == nil {
			delayBank = pd
		} else {
			delayBank, err = Append(delayBank, pd)
			if err != nil {
				return nil, err
			}
		}
	}

	nd, _, _ := delayBank.Dims()
	nTotal := n + nd

	B2 := lft.LFT.B2
	C2 := lft.LFT.C2
	D12 := lft.LFT.D12
	D21 := lft.LFT.D21
	D22 := lft.LFT.D22

	Dd := delayBank.D
	Einv, err := solveIdentityMinusProduct(D22, Dd, N, "delay loop", ErrSingularTransform)
	if err != nil {
		return nil, err
	}

	DdE := mulDims(N, N, Dd, Einv)
	DdEC2 := mulDims(N, n, DdE, C2)
	DdED21 := mulDims(N, m, DdE, D21)
	D22Cd := mulDims(N, nd, D22, delayBank.C)
	DdED22CdPlusCd := addMulDims(N, nd, delayBank.C, DdE, D22Cd)
	BdE := mulDims(nd, N, delayBank.B, Einv)

	Acl := newDense(nTotal, nTotal)
	Bcl := newDense(nTotal, m)
	Ccl := newDense(p, nTotal)

	setBlock(Acl, 0, 0, lft.A)
	addBlock(Acl, 0, 0, mulDims(n, n, B2, DdEC2))
	setBlock(Acl, 0, n, mulDims(n, nd, B2, DdED22CdPlusCd))
	setBlock(Acl, n, 0, mulDims(nd, n, BdE, C2))
	setBlock(Acl, n, n, delayBank.A)
	addBlock(Acl, n, n, mulDims(nd, nd, BdE, D22Cd))

	setBlock(Bcl, 0, 0, lft.B)
	addBlock(Bcl, 0, 0, mulDims(n, m, B2, DdED21))
	setBlock(Bcl, n, 0, mulDims(nd, m, BdE, D21))

	setBlock(Ccl, 0, 0, lft.C)
	addBlock(Ccl, 0, 0, mulDims(p, n, D12, DdEC2))
	setBlock(Ccl, 0, n, mulDims(p, nd, D12, DdED22CdPlusCd))

	Dcl := addMulDims(p, m, lft.D, D12, DdED21)

	result, err := newNoCopy(Acl, Bcl, Ccl, Dcl, lft.Dt)
	if err != nil {
		return nil, err
	}
	result.E = augmentDescriptorE(lft.E, n, nTotal)
	return result, nil
}

func validatePadeOrder(order int) error {
	if order < 1 || order > 10 {
		return fmt.Errorf("order %d must be in 1..10: %w", order, ErrInvalidArgument)
	}
	return nil
}

// PadeDelay returns the order-th order Padé approximant of e^{-tau·s} as a
// SISO continuous model, as MATLAB [num,den] = pade(tau, N)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.pade.html).
// tau must be finite and non-negative; order must be in 1..10.
func PadeDelay(tau float64, order int) (*System, error) {
	if err := validateDelayValue(tau, 0); err != nil {
		return nil, fmt.Errorf("PadeDelay: %w", err)
	}
	if err := validatePadeOrder(order); err != nil {
		return nil, fmt.Errorf("PadeDelay: %w", err)
	}
	if tau == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	}

	n := order

	// q_k = (2n-k)! * n! / ((2n)! * k! * (n-k)!)
	// Recurrence: q_k = q_{k-1} * (n-k+1) / (k * (2n-k+1))
	q := make([]float64, n+1)
	q[0] = 1.0
	for k := 1; k <= n; k++ {
		q[k] = q[k-1] * float64(n-k+1) / float64(k*(2*n-k+1))
	}

	// den(s) = sum_{k=0}^n q_k * (tau*s)^k, descending order
	// num(s) = sum_{k=0}^n (-1)^k * q_k * (tau*s)^k, descending order
	den := make([]float64, n+1)
	num := make([]float64, n+1)
	for k := 0; k <= n; k++ {
		tauPow := math.Pow(tau, float64(k))
		den[n-k] = q[k] * tauPow
		sign := 1.0
		if k%2 != 0 {
			sign = -1.0
		}
		num[n-k] = sign * q[k] * tauPow
	}

	tf := &TransferFunc{
		Num: [][][]float64{{num}},
		Den: [][]float64{den},
		Dt:  0,
	}

	result, err := tf.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("PadeDelay: %w", err)
	}
	return result.Sys, nil
}
