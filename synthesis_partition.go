package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type generalizedPlantPartition struct {
	op                             string
	A                              *mat.Dense
	B1, B2                         *mat.Dense
	C1, C2                         *mat.Dense
	D11, D12, D21, D22             *mat.Dense
	n, m, p, m1, m2, p1, p2        int
	measurementNames, controlNames []string
}

// partitionGeneralizedPlant validates the generalized plant P for the
// synthesis operation op and splits it into the standard blocks. Delayed
// plants are rejected, as MATLAB h2syn/hinfsyn do.
func partitionGeneralizedPlant(op string, P *System, nmeas, ncont int) (*generalizedPlantPartition, error) {
	if err := requireFiniteSystem(op, P); err != nil {
		return nil, err
	}
	if !P.IsContinuous() {
		return nil, fmt.Errorf("%s: plant must be continuous: %w", op, ErrWrongDomain)
	}
	if P.HasDelay() {
		return nil, fmt.Errorf("%s: plant has time delays: %w", op, ErrDelayUnsupported)
	}
	if err := newDescriptorPolicy(P).requireRiccatiStandard(op); err != nil {
		return nil, err
	}

	n, m, p := P.Dims()
	if n == 0 {
		return nil, fmt.Errorf("%s: plant has no states: %w", op, ErrInvalidPartition)
	}
	if ncont <= 0 || nmeas <= 0 || ncont > m || nmeas > p {
		return nil, fmt.Errorf("%s: nmeas=%d, ncont=%d for plant with %d outputs and %d inputs: %w", op, nmeas, ncont, p, m, ErrInvalidPartition)
	}

	m1 := m - ncont
	m2 := ncont
	p1 := p - nmeas
	p2 := nmeas
	if m1 <= 0 || p1 <= 0 {
		return nil, fmt.Errorf("%s: no disturbance inputs or performance outputs left (m1=%d, p1=%d): %w", op, m1, p1, ErrInvalidPartition)
	}

	return &generalizedPlantPartition{
		op:               op,
		A:                P.A,
		B1:               extractBlock(P.B, 0, 0, n, m1),
		B2:               extractBlock(P.B, 0, m1, n, m2),
		C1:               extractBlock(P.C, 0, 0, p1, n),
		C2:               extractBlock(P.C, p1, 0, p2, n),
		D11:              extractBlock(P.D, 0, 0, p1, m1),
		D12:              extractBlock(P.D, 0, m1, p1, m2),
		D21:              extractBlock(P.D, p1, 0, p2, m1),
		D22:              extractBlock(P.D, p1, m1, p2, m2),
		n:                n,
		m:                m,
		p:                p,
		m1:               m1,
		m2:               m2,
		p1:               p1,
		p2:               p2,
		measurementNames: selectTrailingNames(P.OutputName, p1, p2),
		controlNames:     selectTrailingNames(P.InputName, m1, m2),
	}, nil
}

func (gp *generalizedPlantPartition) applyControllerNames(K *System) {
	controllerMetadata(gp.measurementNames, gp.controlNames).applyIO(K)
}

func (gp *generalizedPlantPartition) validateControllerChannels() error {
	stab, err := IsStabilizable(gp.A, gp.B2, true)
	if err != nil {
		return fmt.Errorf("%s: %w", gp.op, err)
	}
	if !stab {
		return fmt.Errorf("%s: (A, B2) not stabilizable: %w", gp.op, ErrNotStabilizable)
	}

	det, err := IsDetectable(gp.A, gp.C2, true)
	if err != nil {
		return fmt.Errorf("%s: %w", gp.op, err)
	}
	if !det {
		return fmt.Errorf("%s: (A, C2) not detectable: %w", gp.op, ErrNotDetectable)
	}
	return nil
}

// newController builds K from the controller K0 = (Ak, Bk, Ck, Dk) designed
// for the plant with D22 = 0; a nil Dk means K0 is strictly proper. A nonzero
// D22 is removed by the loop shift K = K0 (I + D22 K0)^-1, which is
// (Ak - Bk M D22 Ck, Bk M, Ck - Dk M D22 Ck, Dk M) with M = (I + D22 Dk)^-1,
// so the closed loop equals the D22 = 0 design.
func (gp *generalizedPlantPartition) newController(Ak, Bk, Ck, Dk *mat.Dense) (*System, error) {
	if !allZeroDense(gp.D22) {
		if Dk == nil {
			shifted := mulDense(mulDense(Bk, gp.D22), Ck)
			shifted.Sub(Ak, shifted)
			Ak = shifted
		} else {
			IDD := mulDense(gp.D22, Dk)
			for i := range gp.p2 {
				IDD.Set(i, i, IDD.At(i, i)+1)
			}
			M, err := invertSmall(IDD, gp.p2)
			if err != nil {
				return nil, fmt.Errorf("%s: I+D22·Dk singular (%v): %w", gp.op, err, ErrAlgebraicLoop)
			}
			MD22Ck := mulDense(mulDense(M, gp.D22), Ck)
			shifted := mulDense(Bk, MD22Ck)
			shifted.Sub(Ak, shifted)
			Ak = shifted
			shiftedC := mulDense(Dk, MD22Ck)
			shiftedC.Sub(Ck, shiftedC)
			Ck = shiftedC
			Bk = mulDense(Bk, M)
			Dk = mulDense(Dk, M)
		}
	}
	if Dk == nil {
		Dk = mat.NewDense(gp.m2, gp.p2, nil)
	}
	K, err := New(Ak, Bk, Ck, Dk, 0)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}
	gp.applyControllerNames(K)
	return K, nil
}

func (gp *generalizedPlantPartition) closedLoopPoles(Ak, Bk, Ck, Dk *mat.Dense) ([]complex128, error) {
	clN := 2 * gp.n
	clA := mat.NewDense(clN, clN, nil)
	if Dk == nil {
		setBlock(clA, 0, 0, gp.A)
	} else {
		ABDC := mulDense(mulDense(gp.B2, Dk), gp.C2)
		ABDC.Add(gp.A, ABDC)
		setBlock(clA, 0, 0, ABDC)
	}
	setBlock(clA, 0, gp.n, mulDense(gp.B2, Ck))
	setBlock(clA, gp.n, 0, mulDense(Bk, gp.C2))
	setBlock(clA, gp.n, gp.n, Ak)

	var eig mat.Eigen
	if ok := eig.Factorize(clA, mat.EigenNone); !ok {
		return nil, fmt.Errorf("%s: closed-loop eigenvalues: %w", gp.op, ErrSchurFailed)
	}
	return eig.Values(nil), nil
}

func selectTrailingNames(names []string, start, count int) []string {
	if names == nil {
		return nil
	}
	return selectStringSlice(names, contiguousIndices(start, count))
}

func contiguousIndices(start, count int) []int {
	idx := make([]int, count)
	for i := range idx {
		idx[i] = start + i
	}
	return idx
}
