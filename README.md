<p align="center">
  <img src="assets/controlsys-gopher.png"
       alt="controlsys"
       width="280">
</p>

<h1 align="center">controlsys</h1>

<p align="center">
  A control-systems toolbox for Go
</p>

Go control-system toolbox for modeling, analyzing, transforming, and designing continuous-time and discrete-time linear time-invariant models.

The state-space model is the fundamental representation. Transfer-function, zero-pole-gain, frequency-response data, model-array, generalized-model, and tunable-block workflows convert to or build on that core representation where needed. See [docs/codebase-interface-diagram.md](docs/codebase-interface-diagram.md) for the current interface map.

## Install

```bash
go get plantcontrol.org/v1/controlsys
```

Formerly `github.com/jamestjsp/controlsys` (releases up to v1.8.0). From v1.9.0 the module path is `plantcontrol.org/v1/controlsys`.

Linear algebra uses [`plantcontrol.org/v1/gonum`](https://github.com/PlantControl/gonum), PlantControl's Gonum fork with additional LAPACK routines. It is an ordinary module dependency; no `replace` directive is needed.

## Production Readiness

This package is intended to be usable in production control and estimation code, with the usual caveat that numerical software still needs application-specific validation.

- Pin `controlsys` and `plantcontrol.org/v1/gonum` to explicit versions.
- Validate mission-critical models against an external reference, especially for ill-conditioned realizations and delay-heavy systems.
- `System` values are mutable. Use `Copy` before sharing a model across goroutines that may mutate names, delays, notes, or other receiver state. Use `Validate` after direct field edits.
- The repository CI runs `go fix ./...`, `go vet ./...`, `go test -v -count=1 -race ./...`, and a downstream consumer import check; those are the recommended baseline checks for downstream integrations.
- Public API and mutation semantics are tracked in [docs/api-mutation-audit.md](docs/api-mutation-audit.md).

## Features

- **Model interfaces:** state-space, transfer function, zero-pole-gain (ZPK), frequency-response data (FRD), model arrays, generalized models, and tunable blocks
- **Frequency response:** Bode, Nyquist, Nichols, singular values
- **Stability and response analysis:** gain/phase margins, disk margins, bandwidth, damping, root locus, passivity, step-response metrics
- **Control design:** LQR, LQE (Kalman), LQI, LQG, H2 synthesis, H-infinity synthesis, pole placement, Ackermann placement, Riccati solvers (CARE/DARE)
- **PID and fixed-structure tuning:** PID/PID2 controller models, standard/parallel forms, `Pidtune`, tunable blocks, generalized closed-loop models, tuning goals, `Systune`, and `Looptune`
- **State estimation:** Extended Kalman Filter (EKF) for nonlinear systems
- **System identification:** Eigensystem Realization Algorithm (ERA) and frequency-response estimation from I/O data
- **Nonlinear systems:** Jacobian linearization around operating points; Smith predictor for time-delay plants
- **Model arrays and physical assembly:** compatible model grids for parameter sweeps and descriptor assembly of connected physical ports
- **Model reduction & decomposition:** controllability/observability staircase, balanced realization, balanced truncation, stable/unstable and modal separation, modal truncation
- **System norms & covariance:** H2/H-infinity norms, Hankel singular values, state covariance
- **Interconnection:** series, parallel, feedback, safe feedback, append, block diagonal, named/indexed connect, FRD interconnections, sum blocks, and LFT
- **Time-domain:** step, impulse, initial condition, arbitrary input (lsim), discrete simulation
- **Discretization:** ZOH, FOH, Tustin (bilinear), matched pole-zero, discrete-to-discrete resampling
- **Transport delays:** input/output/internal delays, Pade and Thiran approximations, LFT representation
- **Transmission zeros & poles** via staircase decomposition
- **Gramians:** controllability and observability

## Quick Start

```go
package main

import (
	"fmt"

	"plantcontrol.org/v1/controlsys"
	"plantcontrol.org/v1/gonum/mat"
)

func main() {
	// Double integrator: x'' = u
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, _ := controlsys.New(A, B, C, D, 0)

	poles, _ := sys.Poles()
	fmt.Println("Poles:", poles)
	stable, _ := sys.IsStable()
	fmt.Println("Stable:", stable)

	tf, _ := sys.TransferFunction(nil)
	fmt.Println("Transfer function:", tf.TF)
}
```

## API Overview

### System Construction

| Function | Description |
|----------|-------------|
| `New` | Create from A, B, C, D matrices |
| `NewWithDelay` | Create with transport delays |
| `NewGain` | Pure feedthrough (D only) |
| `NewFromSlices` | Create from row-major flat arrays |
| `NewZPK` | SISO zero-pole-gain model |
| `NewZPKMIMO` | MIMO zero-pole-gain model |
| `NewFRD` | Frequency-response data model from sampled complex responses |
| `NewModelArray` | Compatible array of state-space models for sweeps or model grids |
| `StackModelArrays` | Stack equal-shaped compatible arrays along a new leading axis |
| `ConcatModelArrays` | Flatten and concatenate compatible model arrays |
| `NewGeneralizedModel` | Wrap a fixed or tunable block and attach analysis points |
| `NewGeneralizedClosedLoop` | Build a plant/controller closed-loop model with a plant-output analysis point |
| `NewPhysicalComponent` | Wrap a model with named physical ports |
| `AssemblePhysical` | Assemble connected across/through port equations into a descriptor model |
| `NewDescriptor` | Descriptor state-space model with explicit E matrix |
| `Rss` | Random stable continuous-time state-space model |
| `Drss` | Random stable discrete-time state-space model |

### PID & Classical Loop Design

| Function/Type | Description |
|---------------|-------------|
| `NewPID` | PID in parallel form (`Kp`, `Ki`, `Kd`) |
| `NewPIDStd` | PID in standard/ISA form (`Kp`, `Ti`, `Td`) |
| `NewPID2` | 2-DOF PID controller with setpoint weighting |
| `Pidtune` | Autotune `P`, `PI`, `PD`, `PID`, or `PIDF` for a SISO plant |
| `WithFilter` | PID option for derivative filter time constant |
| `WithTs` | PID option for discrete sample time |
| `(*PID).System` / `(*PID2).System` | Convert controller model to state-space |
| `Loopsens` | Sensitivity and complementary-sensitivity functions |
| `Pzmap` | Pole-zero map |

### Generalized & Tunable Models

| Function/Type | Description |
|---------------|-------------|
| `NewGeneralizedModel` | Wrap a fixed or tunable block and attach analysis points |
| `NewGeneralizedClosedLoop` | Build a plant/controller closed-loop model with a plant-output analysis point |
| `(*GeneralizedClosedLoop).InsertAnalysisPoint` | Bind another named break to the plant input or output |
| `TunableReal` | Bounded scalar parameter used by tunable blocks |
| `TunableGain` | Tunable static-gain block |
| `TunablePID` | Tunable PID controller block |
| `TunableTF` | Tunable transfer-function block |
| `TunableSS` | Tunable state-space block |
| `NewTrackingGoal` / `NewRejectionGoal` | Tuning-goal constructors for tracking and disturbance rejection |
| `NewSensitivityGoal` / `NewWeightedGainGoal` | Tuning-goal constructors for gain and sensitivity limits |
| `NewLoopShapeGoal` / `NewMarginGoal` | Tuning-goal constructors for loop-shape and robustness constraints |
| `NewPoleGoal` / `NewOvershootGoal` | Tuning-goal constructors for pole-location and step-response constraints |
| `GridTune` | Bounded Cartesian-grid tuning over free parameters and point-specific goals |
| `Systune` / `Looptune` | Compatibility wrappers around `GridTune` |

### Frequency Response & Plotting

| Method | Description |
|--------|-------------|
| `FreqResponse` | H(jw) at given frequencies |
| `FreqResponsePointwise` | H(jw) with each frequency bit-identical to a single-point `FreqResponse` call, regardless of sweep length |
| `Bode` | Magnitude (dB) and phase (deg) vs frequency |
| `Nyquist` | Nyquist plot with encirclement counting |
| `Nichols` | Nichols chart (magnitude vs phase) |
| `Sigma` | Singular value frequency response |
| `EvalFr` | Evaluate at arbitrary complex s |

### FRD Workflows

| Function/Method | Description |
|-----------------|-------------|
| `(*System).FRD` | Sample a system on a frequency grid and build an FRD model |
| `(*FRD).Bode` | Bode data from FRD samples |
| `(*FRD).Nyquist` | Nyquist contour from FRD samples |
| `(*FRD).Sigma` | Singular values from FRD samples |
| `(*FRD).Abs` | Magnitude-only FRD response |
| `(*FRD).SelectFrequencies` | Select samples by frequency index |
| `(*FRD).SelectFrequencyRange` | Select samples within a frequency band |
| `(*FRD).MapResponse` | Transform each sampled complex response matrix |
| `(*FRD).PeakGain` | Peak gain over sampled frequencies |
| `FRDConcat` | Concatenate compatible FRD models along the frequency grid |
| `FRDMargin` | Gain/phase margins from SISO FRD data |
| `FRDSeries` | Cascade composition of FRD models |
| `FRDParallel` | Parallel composition of FRD models |
| `FRDFeedback` | Closed-loop feedback composition of FRD models |

### Stability & Margins

| Function/Method | Description |
|-----------------|-------------|
| `Poles` | Eigenvalues of A |
| `Zeros` | Transmission zeros |
| `IsStable` | Stability check |
| `IsStabilizable` | Stabilizability test (unstable modes reachable from input) |
| `IsDetectable` | Detectability test (unstable modes observable from output) |
| `DCGain` | Steady-state (DC) gain |
| `Damp` | Natural frequency, damping ratio, time constant |
| `Margin` | Gain and phase margins (SISO) |
| `AllMargin` | All gain/phase crossover points |
| `DiskMargin` | Disk-based stability margin |
| `Bandwidth` | -3 dB bandwidth |
| `RootLocus` | Root locus as a function of loop gain |
| `Pzmap` | Poles and transmission zeros for plotting/inspection |
| `SampledPassive` / `FRDPassive` | Passivity evidence on an explicit frequency grid; passing samples are not an analytic certificate. The grid must be strictly increasing, finite, non-negative, and at or below the Nyquist frequency for sampled models |
| `Passive` | Compatibility alias for `SampledPassive` |
| `SpectralFactor` | Spectral factor for supported static-gain models |

### Control Design

| Function | Description |
|----------|-------------|
| `Lqr` | Continuous-time LQR regulator |
| `Dlqr` | Discrete-time LQR regulator |
| `Lqrd` | Discrete LQR obtained from continuous data and sample time |
| `Lqe` | Kalman filter (observer) gain |
| `Kalman` | Kalman estimator from a `System` model |
| `Kalmd` | Discrete-time Kalman estimator from sampled model data |
| `Estim` | Observer model assembled from a plant and observer gain |
| `Reg` | Observer-based regulator assembled from plant, state-feedback gain, and observer gain |
| `Lqi` | LQR with integral action |
| `Lqg` | LQG controller (combined LQR + Kalman filter) |
| `H2Syn` | H2 optimal controller synthesis from generalized plant |
| `HinfSyn` | H-infinity controller synthesis from generalized plant |
| `Place` | Pole placement |
| `Acker` | Ackermann pole placement |
| `Care` | Continuous algebraic Riccati equation |
| `Dare` | Discrete algebraic Riccati equation |
| `SmithPredictor` | Smith predictor for time-delay plants |

### State Estimation

| Function/Type | Description |
|---------------|-------------|
| `NewEKF(model, x0, P0)` | Create an Extended Kalman Filter |
| `(*EKF).Predict(u)` | Propagate state and covariance one step |
| `(*EKF).Update(y)` | Correct state with a measurement |
| `(*EKF).Step(u, z)` | Run a predict-then-update cycle |
| `type EKFModel` | Nonlinear model: F, H, Jacobians FJac/HJac, noise Q/R |

### System Identification

| Function/Type | Description |
|---------------|-------------|
| `ERA(markov, order, dt)` | Eigensystem Realization Algorithm — recover state-space model from Markov parameters |
| `FreqRespEst(input, output, dt, opts)` | Estimate a frequency response from sampled I/O data |
| `type ERAResult` | Result: identified `System`, singular value ratios |
| `type FreqRespEstResult` | Result: estimated response data and metadata |

### Nonlinear Systems

| Function/Type | Description |
|---------------|-------------|
| `Linearize(model, x0, u0)` | Jacobian linearization of a nonlinear model around an operating point |
| `type NonlinearModel` | Nonlinear state and measurement model definition (`F`, `H`, dimensions `N`, `M`, `P`) |

### Model Reduction

| Function/Method | Description |
|-----------------|-------------|
| `Balreal` | Balanced realization |
| `Balred` | Balanced truncation / singular perturbation |
| `Reduce` / `MinimalRealization` | Controllable/observable state reduction workflows |
| `Modred` | Model reduction by eliminating selected states |
| `Ssbal` | State-space balancing / scaling |
| `Sminreal` | Minimal realization via staircase reduction |
| `Stabsep` | Stable/unstable decomposition |
| `Modsep` | Modal decomposition around a cutoff |
| `ModalTruncate` | Basis-invariant real-Schur modal truncation with retained poles and projection bases |
| `Canon` | Modal or companion canonical form |
| `SS2SS` | Similarity transform with a user-supplied state basis |
| `StateTransform` | Alias-style state-basis transform helper |
| `Xperm` | State permutation transform |
| `Prescale` | Pre-scale states/inputs/outputs for numerical conditioning |
| `ToExplicit` | Convert supported descriptor models to explicit state-space form |
| `DescriptorE` | Return a copy of the descriptor E matrix |
| `SelectByName` / `SelectByIndex` | Select input/output channels by signal names or indices |
| `EliminateStates` | Remove selected states using the model-reduction methods |
| `FixedInputReduction` | Reduce a model by fixing selected input channels |
| `AugmentInternalDelayOutputs` | Expose internal delay-bank outputs with prefixed names |
| `Ctrb` | Controllability matrix |
| `Obsv` | Observability matrix |
| `CtrbF` | Controllability staircase decomposition |
| `ObsvF` | Observability staircase decomposition |
| `Gram` | Controllability/observability gramian |
| `Covar` | State covariance from process-noise covariance |

### System Norms

| Function | Description |
|----------|-------------|
| `Norm` | Generic norm entry point (`NormH2` or `math.Inf(1)`) |
| `H2Norm` | H2 norm (RMS gain) |
| `HinfNorm` | H-infinity norm (peak gain) |
| `HSV` | Hankel singular values |

### Lyapunov Equations

| Function | Description |
|----------|-------------|
| `Lyap(A, Q, opts)` | Solve continuous Lyapunov equation AX + XAᵀ + Q = 0 |
| `DLyap(A, Q, opts)` | Solve discrete Lyapunov equation AXAᵀ − X + Q = 0 |
| `NewLyapunovWorkspace(n)` | Pre-allocate workspace for repeated solves |

### Representation Conversion

| Function/Method | Description |
|-----------------|-------------|
| `(*System).TransferFunction` | State-space → transfer function |
| `(*TransferFunc).StateSpace` | Transfer function → state-space |
| `(*TransferFunc).ZPK` | Transfer function → zero-pole-gain |
| `(*ZPK).TransferFunction` | ZPK → transfer function |
| `(*ZPK).StateSpace` | ZPK → state-space |
| `(*System).FRD` | State-space → frequency-response data over a frequency grid |

### Discretization

| Method | Description |
|--------|-------------|
| `Discretize` | Bilinear (Tustin) c2d |
| `DiscretizeWithOpts` | Option-driven c2d with method and delay-modeling controls |
| `DiscretizeZOH` | Zero-order hold c2d |
| `DiscretizeFOH` | First-order hold c2d |
| `DiscretizeImpulse` | Impulse-invariant c2d |
| `DiscretizeMatched` | Matched pole-zero c2d |
| `D2D` | Discrete-to-discrete resampling |
| `Undiscretize` | Bilinear d2c |
| `D2C` | Discrete-to-continuous conversion by Tustin or ZOH assumptions |

### Interconnection

| Function | Description |
|----------|-------------|
| `Series` | Cascade connection |
| `Parallel` | Sum connection |
| `Feedback` | Closed-loop with feedback |
| `WithApproximatedDelays` / `WithPadeOrder` / `WithThiranOrder` | Feedback options for a delay-free rational closed loop |
| `Append` | Block diagonal concatenation |
| `SumBlk` | Sum block from string expression |
| `Connect` / `ConnectByName` | General interconnection by indices or signal names |
| `BlkDiag` | Block-diagonal composition of multiple systems |
| `Inv` | System inversion when the model is invertible |
| `LFT` | Linear fractional transformation |

### Time-Domain Simulation

| Function/Method | Description |
|-----------------|-------------|
| `Step` | Unit step response |
| `Impulse` | Unit impulse response |
| `Initial` | Free response to initial state |
| `Lsim` | Response to arbitrary input on a uniform time grid |
| `Simulate` | Discrete-time simulation |
| `GenSig` | Generate test signals (step, sine, square, pulse) |
| `StepInfo` | Step-response rise time, settling time, overshoot, peak, and steady-state metrics |
| `StepInfoForSystem` | Simulate a stable model's step response and compute step metrics |

### Model Arrays

| Function/Method | Description |
|-----------------|-------------|
| `NewModelArray` | Create a shaped array of compatible state-space models |
| `StackModelArrays` | Stack equal-shaped compatible arrays along a new leading axis |
| `ConcatModelArrays` | Flatten and concatenate compatible model arrays |
| `(*ModelArray).Model` / `ModelFlat` | Retrieve a model by multidimensional or flat index |
| `(*ModelArray).SelectFlat` | Select a flat-index subset of models |
| `(*ModelArray).FreqResponse` | Frequency response for every model in the array |
| `(*ModelArray).Step` | Step response for every model in the array |

### Physical Assembly

| Function/Type | Description |
|---------------|-------------|
| `NewPhysicalComponent` | Wrap a model with named physical ports |
| `AssemblePhysical` | Assemble connected across/through port equations into a descriptor model |
| `PhysicalPort` / `PhysicalConnection` | Bind component input/output channels to physical ports and node connections |

### Transport Delays

| Function/Method | Description |
|-----------------|-------------|
| `SetDelay` | Set MIMO delay matrix |
| `SetInputDelay` | Set per-input delays |
| `SetOutputDelay` | Set per-output delays |
| `SetDelayModel` | Attach a custom internal delay model |
| `GetDelayModel` | Read the internal delay model and delay times |
| `DecomposeIODelay` | Split a full I/O delay matrix into input/output/residual pieces |
| `PullDelaysToLFT` | Move external delays into the internal LFT delay representation |
| `MinimalLFT` | Reduce redundant internal delay blocks |
| `ZeroDelayApprox` | Replace internal delay blocks with zero-delay behavior |
| `PadeDelay` | Pade rational approximation |
| `ThiranDelay` | Thiran allpass (fractional discrete delays) |
| `Pade` | Replace all delays with Pade approximations |
| `AbsorbDelay` | Augment state for discrete delays |

## Core Algorithms

| Algorithm | Purpose |
|-----------|---------|
| Staircase decomposition | Transmission zeros via rank-revealing factorization |
| Column-pivoting QR | Rank determination with incremental condition estimation |
| Row-pivoting RQ | Dual rank-revealing factorization |
| Controllability staircase | Subspace decomposition for reduction and transfer functions |
| Balanced realization | Gramian-based state transformation for model reduction |
| Schur decomposition | Riccati equation solvers (CARE/DARE) |

## License

MIT
