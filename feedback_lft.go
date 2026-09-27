package controlsys

import "plantcontrol.org/v1/gonum/mat"

func feedbackWithLFT(plant, controller *System, sign float64) (*System, error) {
	savedPlantInput := copySliceOrNil(plant.InputDelay)

	plantStripped := plant.Copy()
	plantStripped.InputDelay = nil

	plantLFT, err := plantStripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	ctrlLFT, err := controller.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	var plantTau, ctrlTau []float64
	if plantLFT.LFT != nil {
		plantTau = plantLFT.LFT.Tau
	}
	if ctrlLFT.LFT != nil {
		ctrlTau = ctrlLFT.LFT.Tau
	}

	pH, _ := plantLFT.GetDelayModel()
	cH, _ := ctrlLFT.GetDelayModel()

	Np := len(plantTau)
	Nc := len(ctrlTau)
	Nd := Np + Nc

	_, m1, p1 := plant.Dims()
	np, _, _ := pH.Dims()
	nc, _, _ := cH.Dims()

	isign := -sign
	nTotal := np + nc
	mTotal := m1 + Nd
	pTotal := p1 + Nd

	A := mat.NewDense(max(nTotal, 1), max(nTotal, 1), nil)
	B := mat.NewDense(max(nTotal, 1), max(mTotal, 1), nil)
	C := mat.NewDense(max(pTotal, 1), max(nTotal, 1), nil)
	D := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)

	type lftTempWS struct {
		buf []float64
		pos int
	}
	// Pre-allocate a reasonable buffer for small-to-mid systems to avoid constant resizing.
	ws := &lftTempWS{buf: make([]float64, 512)}

	extract := func(m *mat.Dense, r0, c0, rows, cols int) *mat.Dense {
		if rows == 0 || cols == 0 {
			return mat.NewDense(max(rows, 1), max(cols, 1), nil)
		}
		raw := m.RawMatrix()
		req := rows * cols
		if ws.pos+req > len(ws.buf) {
			newBuf := make([]float64, max(ws.pos+req, len(ws.buf)*2))
			copy(newBuf, ws.buf)
			ws.buf = newBuf
		}
		data := ws.buf[ws.pos : ws.pos+req]
		ws.pos += req
		copyBlock(data, cols, 0, 0, raw.Data, raw.Stride, r0, c0, rows, cols)
		return mat.NewDense(rows, cols, data)
	}

	mul := func(a, b *mat.Dense) *mat.Dense {
		r, _ := a.Dims()
		_, c := b.Dims()
		if r == 0 || c == 0 {
			return mat.NewDense(max(r, 1), max(c, 1), nil)
		}
		req := r * c
		if ws.pos+req > len(ws.buf) {
			newBuf := make([]float64, max(ws.pos+req, len(ws.buf)*2))
			copy(newBuf, ws.buf)
			ws.buf = newBuf
		}
		data := ws.buf[ws.pos : ws.pos+req]
		ws.pos += req
		m := mat.NewDense(r, c, data)
		m.Mul(a, b)
		return m
	}

	D1ee := extract(pH.D, 0, 0, p1, m1)
	D2ee := extract(cH.D, 0, 0, m1, p1)

	E21, err := solveFeedbackFeedthrough(D1ee, D2ee, sign, p1, "feedback", ErrSingularTransform)
	if err != nil {
		return nil, err
	}

	E12 := eyeDense(m1)
	{
		t := mul(mul(D2ee, E21), D1ee)
		t.Scale(isign, t)
		E12.Sub(E12, t)
	}

	B1e := extract(pH.B, 0, 0, np, m1)
	C1e := extract(pH.C, 0, 0, p1, np)
	B2e := extract(cH.B, 0, 0, nc, p1)
	C2e := extract(cH.C, 0, 0, m1, nc)

	E21D1ee := mul(E21, D1ee)

	var B1eE12, B1eE12D2ee *mat.Dense
	if np > 0 {
		B1eE12 = mul(B1e, E12)
		B1eE12D2ee = mul(B1eE12, D2ee)
	}

	var B2eE21, B2eE21D1ee *mat.Dense
	if nc > 0 {
		B2eE21 = mul(B2e, E21)
		B2eE21D1ee = mul(B2eE21, D1ee)
	}

	var E21C1e *mat.Dense
	if np > 0 {
		E21C1e = mul(E21, C1e)
	}

	var D1de, D1ed, E21D1ed *mat.Dense
	if Np > 0 {
		D1de = extract(pH.D, p1, 0, Np, m1)
		D1ed = extract(pH.D, 0, m1, p1, Np)
		E21D1ed = mul(E21, D1ed)
	}

	var D2de, D2ed *mat.Dense
	if Nc > 0 {
		D2de = extract(cH.D, m1, 0, Nc, p1)
		D2ed = extract(cH.D, 0, p1, m1, Nc)
	}

	var D1deD2ee *mat.Dense
	if Np > 0 {
		D1deD2ee = mul(D1de, D2ee)
	}

	var D2deE21, D2deE21D1ee *mat.Dense
	if Nc > 0 {
		D2deE21 = mul(D2de, E21)
		D2deE21D1ee = mul(D2deE21, D1ee)
	}

	var E12D2ed *mat.Dense
	if Nc > 0 {
		E12D2ed = mul(E12, D2ed)
	}

	if np > 0 {
		setBlock(A, 0, 0, pH.A)
		t := mul(B1eE12D2ee, C1e)
		t.Scale(isign, t)
		subBlock(A, 0, 0, t)
		if nc > 0 {
			t2 := mul(B1eE12, C2e)
			t2.Scale(-isign, t2)
			setBlock(A, 0, np, t2)
		}
	}
	if nc > 0 {
		if np > 0 {
			setBlock(A, np, 0, mul(B2eE21, C1e))
		}
		setBlock(A, np, np, cH.A)
		t := mul(B2eE21D1ee, C2e)
		t.Scale(isign, t)
		subBlock(A, np, np, t)
	}

	if np > 0 {
		setBlock(B, 0, 0, B1eE12)
	}
	if nc > 0 {
		setBlock(B, np, 0, B2eE21D1ee)
	}

	if Np > 0 {
		B1d := extract(pH.B, 0, m1, np, Np)
		if np > 0 {
			t := mul(B1eE12D2ee, E21D1ed)
			t.Scale(-isign, t)
			t.Add(B1d, t)
			setBlock(B, 0, m1, t)
		}
		if nc > 0 {
			setBlock(B, np, m1, mul(B2eE21, D1ed))
		}
	}

	if Nc > 0 {
		B2d := extract(cH.B, 0, p1, nc, Nc)
		if np > 0 {
			t := mul(B1eE12, D2ed)
			t.Scale(-isign, t)
			setBlock(B, 0, m1+Np, t)
		}
		if nc > 0 {
			t := mul(mul(B2eE21D1ee, E12), D2ed)
			t.Scale(isign, t)
			t.Add(B2d, t)
			setBlock(B, np, m1+Np, t)
		}
	}

	if np > 0 {
		setBlock(C, 0, 0, E21C1e)
	}
	if nc > 0 {
		t := mul(E21D1ee, C2e)
		t.Scale(-isign, t)
		setBlock(C, 0, np, t)
	}

	if Np > 0 {
		C1d := extract(pH.C, p1, 0, Np, np)
		if np > 0 {
			t := mul(D1deD2ee, E21C1e)
			t.Scale(isign, t)
			r := mat.NewDense(Np, np, nil)
			r.Sub(C1d, t)
			setBlock(C, p1, 0, r)
		}
		if nc > 0 {
			t := mul(mul(D1deD2ee, E21D1ee), C2e)
			t.Scale(-isign, t)
			setBlock(C, p1, np, t)
		}
	}

	if Nc > 0 {
		C2d := extract(cH.C, m1, 0, Nc, nc)
		if np > 0 {
			setBlock(C, p1+Np, 0, mul(D2deE21, C1e))
		}
		if nc > 0 {
			t := mul(D2deE21D1ee, C2e)
			t.Scale(-isign, t)
			t.Add(C2d, t)
			setBlock(C, p1+Np, np, t)
		}
	}

	setBlock(D, 0, 0, E21D1ee)

	if Np > 0 {
		setBlock(D, 0, m1, E21D1ed)
	}
	if Nc > 0 {
		t := mul(E21D1ee, E12D2ed)
		t.Scale(-isign, t)
		setBlock(D, 0, m1+Np, t)
	}

	if Np > 0 {
		D1dd := extract(pH.D, p1, m1, Np, Np)

		setBlock(D, p1, 0, mul(D1de, E12))

		t := mul(D1deD2ee, E21D1ed)
		t.Scale(isign, t)
		r := mat.NewDense(Np, Np, nil)
		r.Sub(D1dd, t)
		setBlock(D, p1, m1, r)

		if Nc > 0 {
			t := mul(D1de, E12D2ed)
			t.Scale(-isign, t)
			setBlock(D, p1, m1+Np, t)
		}
	}

	if Nc > 0 {
		D2dd := extract(cH.D, m1, p1, Nc, Nc)

		setBlock(D, p1+Np, 0, D2deE21D1ee)

		if Np > 0 {
			setBlock(D, p1+Np, m1, mul(D2deE21, D1ed))
		}

		t := mul(D2deE21D1ee, E12D2ed)
		t.Scale(-isign, t)
		t.Add(D2dd, t)
		setBlock(D, p1+Np, m1+Np, t)
	}

	if nTotal == 0 {
		A = &mat.Dense{}
	} else {
		A = resizeDense(A, nTotal, nTotal)
	}
	B = resizeDense(B, nTotal, mTotal)
	C = resizeDense(C, pTotal, nTotal)
	D = resizeDense(D, pTotal, mTotal)

	H := &System{A: A, B: B, C: C, D: D, Dt: plant.Dt}

	taus := make([]float64, Nd)
	copy(taus, plantTau)
	copy(taus[Np:], ctrlTau)

	result, err := SetDelayModel(H, taus)
	if err != nil {
		return nil, err
	}
	result.InputDelay = savedPlantInput
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
