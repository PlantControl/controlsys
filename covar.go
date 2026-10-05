package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// Covar computes the steady-state output covariance P = E[y yᵀ] of sys driven
// by white noise with intensity W, as MATLAB covar(sys,W). For continuous
// models, entries fed by white noise through D are infinite.
//
// Delays are honoured exactly. Discrete delays (input, output, I/O and
// internal) are absorbed as poles at z=0 before the Lyapunov solve. For
// continuous models, input, output and I/O delays shift each path's impulse
// response, so P_ij = Σ_kl W_kl ∫ g_ik(t) g_jl(t) dt with
// g_ik(t) = D_ik δ(t-τ_ik) + c_i e^{A(t-τ_ik)} b_k for t > τ_ik; a D-path entry
// is infinite only when two feedthrough paths share the same total delay.
// Continuous internal delays have no finite-order form and return
// ErrContinuousInternalDelay. A model with no outputs has no covariance and
// returns ErrDimensionMismatch; with no inputs P is the p×p zero matrix.
func Covar(sys *System, W *mat.Dense) (*mat.Dense, error) {
	if err := requireFiniteSystem("Covar", sys); err != nil {
		return nil, err
	}
	if err := requireStandardCovarianceSystem(sys, "Covar"); err != nil {
		return nil, err
	}
	_, m, _ := sys.Dims()
	if err := validateCovarianceRole("Covar", covarianceInputNoise, W, m); err != nil {
		return nil, err
	}
	var err error
	if sys.IsDiscrete() && sys.HasDelay() {
		sys, err = sys.AbsorbDelay()
	} else {
		sys, err = finiteDimensionalModel(sys, "Covar")
	}
	if err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, fmt.Errorf("Covar: system is unstable: %w", ErrUnstable)
	}
	if p == 0 {
		return nil, fmt.Errorf("Covar: model has no outputs: %w", ErrDimensionMismatch)
	}
	if m == 0 {
		return mat.NewDense(p, p, nil), nil
	}
	if sys.IsContinuous() && sys.HasDelay() {
		return covarContinuousDelayed(sys, W)
	}

	P := mat.NewDense(p, p, nil)
	if n > 0 {
		Q := inputNoiseIntensity(sys.B, W, n, m)

		// symmetrize Q for Lyapunov solver
		qRaw := Q.RawMatrix()
		for i := range n {
			for j := i + 1; j < n; j++ {
				avg := (qRaw.Data[i*qRaw.Stride+j] + qRaw.Data[j*qRaw.Stride+i]) / 2
				qRaw.Data[i*qRaw.Stride+j] = avg
				qRaw.Data[j*qRaw.Stride+i] = avg
			}
		}

		var X *mat.Dense
		if sys.IsContinuous() {
			X, err = Lyap(sys.A, Q, nil)
		} else {
			X, err = DLyap(sys.A, Q, nil)
		}
		if err != nil {
			return nil, err
		}

		var CX mat.Dense
		CX.Mul(sys.C, X)
		P.Mul(&CX, sys.C.T())
	}

	var DW, DWDt mat.Dense
	DW.Mul(sys.D, W)
	DWDt.Mul(&DW, sys.D.T())
	if sys.IsDiscrete() {
		P.Add(P, &DWDt)
		return P, nil
	}
	for i := range p {
		for j := range p {
			if v := DWDt.At(i, j); v != 0 {
				P.Set(i, j, math.Inf(int(math.Copysign(1, v))))
			}
		}
	}
	return P, nil
}

// covarContinuousDelayed evaluates P_ij = Σ_kl W_kl ∫ g_ik g_jl for a
// continuous model with external delays τ_ik. Inputs sharing a delay column
// share one copy of the state, so the state cross-covariances
// Y_GH = Σ_{k∈G,l∈H} W_kl ∫ e^{At} b_k b_lᵀ e^{Aᵀt} dt come from a single
// Lyapunov solve on blkdiag(A,…,A).
func covarContinuousDelayed(sys *System, W *mat.Dense) (*mat.Dense, error) {
	n, m, p := sys.Dims()
	tau := effectiveIODelayData(sys, p, m, true)

	group := make([]int, m)
	var reps []int
	for k := range m {
		group[k] = -1
		for g, r := range reps {
			same := true
			for i := range p {
				if !sameDelay(tau[i*m+k], tau[i*m+r]) {
					same = false
					break
				}
			}
			if same {
				group[k] = g
				break
			}
		}
		if group[k] < 0 {
			group[k] = len(reps)
			reps = append(reps, k)
		}
	}
	ng := len(reps)

	expCache := map[float64]*mat.Dense{}
	expA := func(t float64) *mat.Dense {
		if e, ok := expCache[t]; ok {
			return e
		}
		var at, e mat.Dense
		at.Scale(t, sys.A)
		e.Exp(&at)
		expCache[t] = &e
		return &e
	}

	P := mat.NewDense(p, p, nil)
	if n > 0 {
		N := n * ng
		aAug := mat.NewDense(N, N, nil)
		bAug := mat.NewDense(N, m, nil)
		for g := range ng {
			aAug.Slice(g*n, (g+1)*n, g*n, (g+1)*n).(*mat.Dense).Copy(sys.A)
		}
		for k := range m {
			for r := range n {
				bAug.Set(group[k]*n+r, k, sys.B.At(r, k))
			}
		}
		Q := inputNoiseIntensity(bAug, W, N, m)
		symmetrize(Q.RawMatrix().Data, N, N)
		Y, err := Lyap(aAug, Q, nil)
		if err != nil {
			return nil, err
		}

		ci := mat.NewDense(1, n, nil)
		cj := mat.NewDense(1, n, nil)
		var tmp, row, v mat.Dense
		for i := range p {
			ci.Copy(sys.C.Slice(i, i+1, 0, n))
			for j := range p {
				cj.Copy(sys.C.Slice(j, j+1, 0, n))
				sum := 0.0
				for g, kg := range reps {
					for h, lh := range reps {
						ygh := Y.Slice(g*n, (g+1)*n, h*n, (h+1)*n)
						delta := tau[i*m+kg] - tau[j*m+lh]
						if sameDelay(tau[i*m+kg], tau[j*m+lh]) {
							delta = 0
						}
						if delta >= 0 {
							tmp.Mul(ci, ygh)
							row.Mul(&tmp, expA(delta).T())
						} else {
							tmp.Mul(ci, expA(-delta))
							row.Mul(&tmp, ygh)
						}
						v.Mul(&row, cj.T())
						sum += v.At(0, 0)
					}
				}
				for k := range m {
					for l := range m {
						wkl := W.At(k, l)
						if wkl == 0 {
							continue
						}
						delta := tau[i*m+k] - tau[j*m+l]
						switch {
						case sameDelay(tau[i*m+k], tau[j*m+l]):
						case delta > 0 && sys.D.At(i, k) != 0:
							row.Mul(cj, expA(delta))
							sum += sys.D.At(i, k) * wkl * dotCol(&row, sys.B, l)
						case delta < 0 && sys.D.At(j, l) != 0:
							row.Mul(ci, expA(-delta))
							sum += sys.D.At(j, l) * wkl * dotCol(&row, sys.B, k)
						}
					}
				}
				P.Set(i, j, sum)
			}
		}
	}

	for i := range p {
		for j := range p {
			impulse := 0.0
			for k := range m {
				for l := range m {
					if sameDelay(tau[i*m+k], tau[j*m+l]) {
						impulse += sys.D.At(i, k) * W.At(k, l) * sys.D.At(j, l)
					}
				}
			}
			if impulse != 0 {
				P.Set(i, j, math.Inf(int(math.Copysign(1, impulse))))
			}
		}
	}
	return P, nil
}

func dotCol(row, B *mat.Dense, col int) float64 {
	_, n := row.Dims()
	s := 0.0
	for r := range n {
		s += row.At(0, r) * B.At(r, col)
	}
	return s
}

// sameDelay treats delays assembled from different input/output/IO sums as
// equal when they differ only by rounding.
func sameDelay(a, b float64) bool {
	return math.Abs(a-b) <= 1e-12*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
