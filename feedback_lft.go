package controlsys

import "plantcontrol.org/v1/gonum/mat"

// feedbackWithLFT closes u = r + sign*K*y around the delay models of plant
// and controller. Every plant and controller delay lies inside the loop, so
// all of them become internal delays of the result.
func feedbackWithLFT(plant, controller *System, sign float64) (*System, error) {
	plantLFT, err := plant.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	ctrlLFT, err := controller.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	pH, plantTau := plantLFT.GetDelayModel()
	cH, ctrlTau := ctrlLFT.GetDelayModel()

	_, m1, p1 := plant.Dims()
	n1, _, _ := pH.Dims()
	n2, _, _ := cH.Dims()
	Np, Nc := len(plantTau), len(ctrlTau)
	nT := n1 + n2
	rows := nT + p1 + Np + Nc
	cols := nT + m1 + Np + Nc
	rOff, w1Off, w2Off := nT, nT+m1, nT+m1+Np
	yOff, z1Off, z2Off := nT, nT+p1, nT+p1+Np

	D1ee := extractBlock(pH.D, 0, 0, p1, m1)
	D2ee := extractBlock(cH.D, 0, 0, m1, p1)
	E12, err := solveFeedbackFeedthrough(D2ee, D1ee, sign, m1, "feedback", ErrSingularTransform)
	if err != nil {
		return nil, err
	}

	put := func(dst *mat.Dense, r0, c0 int, src *mat.Dense, sr0, sc0, nr, nc int) {
		if nr == 0 || nc == 0 {
			return
		}
		d, s := dst.RawMatrix(), src.RawMatrix()
		copyBlock(d.Data, d.Stride, r0, c0, s.Data, s.Stride, sr0, sc0, nr, nc)
	}
	addMul := func(dst *mat.Dense, r0 int, src *mat.Dense, sr0, sc0, nr, nc int, x *mat.Dense) {
		if nr == 0 || nc == 0 {
			return
		}
		t := mat.NewDense(nr, cols, nil)
		t.Mul(extractBlock(src, sr0, sc0, nr, nc), x)
		addBlock(dst, r0, 0, t)
	}

	// Rows of M map [x1; x2; r; w1; w2] to [x1'; x2'; y; z1; z2].
	M := mat.NewDense(rows, cols, nil)
	put(M, 0, 0, pH.A, 0, 0, n1, n1)
	put(M, 0, w1Off, pH.B, 0, m1, n1, Np)
	put(M, z1Off, 0, pH.C, p1, 0, Np, n1)
	put(M, z1Off, w1Off, pH.D, p1, m1, Np, Np)
	put(M, n1, n1, cH.A, 0, 0, n2, n2)
	put(M, n1, w2Off, cH.B, 0, p1, n2, Nc)
	put(M, z2Off, n1, cH.C, m1, 0, Nc, n2)
	put(M, z2Off, w2Off, cH.D, m1, p1, Nc, Nc)

	Y := mat.NewDense(p1, cols, nil)
	put(Y, 0, 0, pH.C, 0, 0, p1, n1)
	put(Y, 0, w1Off, pH.D, 0, m1, p1, Np)

	G := mat.NewDense(m1, cols, nil)
	put(G, 0, n1, cH.C, 0, 0, m1, n2)
	put(G, 0, w2Off, cH.D, 0, p1, m1, Nc)
	addMul(G, 0, D2ee, 0, 0, m1, p1, Y)
	G.Scale(sign, G)
	for i := range m1 {
		G.Set(i, rOff+i, G.At(i, rOff+i)+1)
	}

	U := mat.NewDense(m1, cols, nil)
	U.Mul(E12, G)
	addMul(Y, 0, D1ee, 0, 0, p1, m1, U)

	addMul(M, 0, pH.B, 0, 0, n1, m1, U)
	addMul(M, z1Off, pH.D, p1, 0, Np, m1, U)
	addMul(M, n1, cH.B, 0, 0, n2, p1, Y)
	addMul(M, z2Off, cH.D, m1, 0, Nc, p1, Y)
	addBlock(M, yOff, 0, Y)

	sub := func(r0, c0, nr, nc int) *mat.Dense {
		out := newDense(nr, nc)
		put(out, 0, 0, M, r0, c0, nr, nc)
		return out
	}
	H := &System{
		A:  sub(0, 0, nT, nT),
		B:  sub(0, nT, nT, cols-nT),
		C:  sub(nT, 0, rows-nT, nT),
		D:  sub(nT, nT, rows-nT, cols-nT),
		E:  blkDiagDescriptorE(pH, cH),
		Dt: plant.Dt,
	}

	taus := make([]float64, 0, Np+Nc)
	taus = append(taus, plantTau...)
	taus = append(taus, ctrlTau...)

	result, err := SetDelayModel(H, taus)
	if err != nil {
		return nil, err
	}
	result.InputName = copyStringSlice(plant.InputName)
	result.OutputName = copyStringSlice(plant.OutputName)
	return result, nil
}

func eyeDense(n int) *mat.Dense {
	data := make([]float64, n*n)
	for i := range n {
		data[i*(n+1)] = 1
	}
	return mat.NewDense(n, n, data)
}
