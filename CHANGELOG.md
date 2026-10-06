# Changelog

## v2.4.0

- Behaviour change: `Place` returns the new `ErrPoleAccuracy` instead of a gain when the achieved closed-loop poles are more than 10% from the requested ones (where MATLAB `place` warns). It previously returned such gains with a nil error, including destabilizing ones (e.g. max Re eig(A−BK) = +702 for a 50-state tridiagonal plant; scipy `place_poles` also misses these by >10%).
- Behaviour change: `Acker` returns `ErrPoleAccuracy` under the same 10% criterion as `Place` (where MATLAB `acker` warns), instead of a gain with a nil error. E.g. a 14-state chain with poles −0.1…−1.4: the gain is exact to 1e-14 but its closed-loop poles are 30% off (34% in float64). As in MATLAB, the check uses float64 eig(A−BK), so it also rejects gains whose closed-loop eigenproblem alone is ill-conditioned.
- `Place` keeps the robust (KNV) gain whenever its eigenvector matrix is numerically nonsingular and its poles are accurate, instead of switching to the Schur gain when κ₁(X) > 1/√ε. On a random 30-state, 5-input plant with clustered poles the error drops from 7% to 1e-5; the Schur gain is still tried when the robust one fails and the more accurate one is kept.
- Performance: `FreqResponse` (and Bode, Sigma, Nichols) chooses per-point LU or the Hessenberg sweep from a cost model in state count, input count and grid length, with constants per Gonum kernel backend (amd64 assembly vs pure Go). Short grids above the old size threshold stay dense (up to 2.4x faster at 3 points on arm64); on amd64 mid-size models and long grids use the sweep (up to 43% faster). Results agree with the single-point path to componentwise rounding as before.
- Fix: `System.Bode`, `System.Nichols` and `FRD.Bode` unwrap phase across multiple 360° turns (previously one turn per step, e.g. a 1 s delay at ω≈9.74 rad/s gave −198° instead of −558°). Jumps of an odd multiple of 180° resolve as MATLAB `unwrap`; NaN samples are skipped.
- Benchmarks fail on unexpected errors; `benchSysNonSym` and the Place N50/N100 fixtures were invalid (benchmarks timed error returns) and are replaced, so their baselines shift.
- CI: native macOS arm64 test/bench jobs; label- or manually-triggered performance comparison workflow (`perf.yml`, `scripts/benchcmp`).

## v2.3.0

- Breaking: the module path is now `plantcontrol.org/v2/controlsys/v2` (site namespace v2; Go still requires the trailing `/v2` for v2.x tags). The API is unchanged from v2.2.0. Migrate with:

```diff
- import "plantcontrol.org/v1/controlsys/v2"
+ import "plantcontrol.org/v2/controlsys/v2"
```

```sh
go get plantcontrol.org/v2/controlsys/v2@v2.3.0
```

  v2.0.0–v2.2.0 still resolve under `plantcontrol.org/v1/controlsys/v2` and receive no further releases there.

## v2.2.0

- Dependency: `plantcontrol.org/v1/gonum` v0.20.4. SVD (Dgesvd/Dbdsqr/Dlasq1) no longer panics or loops forever on NaN/Inf input (Go `math.Max`/`Min` propagate NaN where gfortran's MAX/MIN ignore it); `mat.SVD.Factorize` returns false for non-finite input (PlantControl/gonum#28).
- CI fails on unformatted code (`gofmt -l`).

### Added

- `HinfSynResult.CL`, `Gamma` and `PeakFrequency`, MATLAB `[K,CL,gamma] = hinfsyn(...)`: the closed loop `LFT(P, K)` from w to z, its H∞ norm `Gamma ≤ GammaOpt` and the frequency where it peaks, taken from the verification `HinfSyn` already runs (no extra computation). `Mixsyn` now reuses them instead of rebuilding the closed loop (`MixsynResult.CL` is `Info.CL`).
- `H2SynResult.CL` and `Gamma`, MATLAB `[K,CL,gamma] = h2syn(...)`: the closed loop and its H2 norm. This adds one LFT and an H2Norm Lyapunov solve of twice P's order per call (14 → 23 µs on a 2-state plant).

## v2.1.0

- Dependency: `plantcontrol.org/v1/gonum` v0.20.3. Dhseqr's Dlaqr04 fallback (reached by Dgees/Dgeev when Dlahqr fails, always for NaN input) passed wrong bounds, eigenvalue slices and an undersized Z; fixed, and non-finite input no longer panics (PlantControl/gonum#24).
- `HinfNorm` and `Norm(sys, Inf)` accept continuous internal-delay models instead of returning `ErrContinuousInternalDelay`. Stability is decided exactly by the Nyquist count of det(I − H22·Δ) (unstable returns `(+Inf, +Inf, nil)`); the peak uses the exact delay factors e^{−jωτ}. Neutral-type or uncertifiable cases return `ErrDelayUnsupported`.
- `DiskMargin`/`DiskMarginSkew` on continuous delayed loops stop the dense π/(8τ) frequency grid where a rigorous tail bound certifies |L| ≤ 0.3 and the sensitivity disk bound cannot exceed the grid peak (≈33k → ≈1k evaluations for a PID + e^{−0.5s} loop). Sensitivity and disk-margin peaks bracket each grid maximum by neighbours of distinct frequency and refine every grid local maximum within 10% of the top, fixing a missed resonance top (2e-5 relative) when two scale points coincide up to rounding (seen on amd64). Loops whose delay grid previously exceeded the point budget may now return a margin instead of `ErrDelayUnsupported`.
- Behaviour change: `(*System).IsStable` no longer returns `ErrContinuousInternalDelay` for continuous internal-delay models; it decides stability exactly with the same Nyquist count of χ(s) = det(sI−A)·det(I − H22·Δ) as `HinfNorm` (roots on the imaginary axis are not stable, as MATLAB `isstable`). Neutral-type or undecidable loops return `ErrDelayUnsupported`, descriptor ones `ErrDescriptorUnsupported`. `StepInfoForSystem` now returns `ErrUnstable` for such models when they are unstable, and skips the gate only on `ErrDelayUnsupported`.
- `HinfNorm` and `IsStable` no longer panic on a continuous internal-delay model with no inputs or no outputs.
- Behaviour change: `Norm(sys, Inf)` of an unstable continuous internal-delay model returns its L∞ peak gain instead of `ErrDelayUnsupported`, as MATLAB `norm(sys, Inf)` does regardless of stability. It is +Inf, at that frequency, when the characteristic function χ has a root on the imaginary axis (hidden modes included, as for rational models).
- `HinfNorm`/`Norm(sys, Inf)` of continuous internal-delay models and `DiskMargin`/`DiskMarginSkew` of continuous delayed loops certify their peak between grid samples to 1e-9 relative: a Taylor bound (second and third order) from the descriptor form of the delay LFT, with Neumann-series resolvent bounds valid for non-normal A, is bisected until no interval can exceed the reported peak. Resonances narrower than the grid spacing (ζ down to 1e-4 in tests) are no longer missed. A peak that cannot be certified within the point budget returns `ErrDelayUnsupported`. Costs ≈1.2–1.4× on a PID + e^{−0.5s} loop.
- `HinfSyn` and `H2Syn` check that D12 has full column rank and D21 full row rank before synthesis and return `ErrInvalidPartition`. Behaviour change: with D11 = 0, `HinfSyn` previously bisected gamma up to 1e12 and returned a misleading `ErrGammaNotAchievable`; `H2Syn` returned `ErrSingularR`.
- Behaviour change: `HinfSyn`, `H2Syn` and `Mixsyn` accept discrete-time plants (Ts > 0), as MATLAB `hinfsyn`/`h2syn`/`mixsyn` do, instead of returning `ErrWrongDomain`. `HinfSyn` designs on the Tustin-equivalent continuous plant (the H∞ norm and stability are invariant) and maps K back, so the rank condition applies to P12/P21 at z = −1; a plant mode at z = −1 is handled by designing for P(−z), and modes at both z = ±1 by a static output-feedback shift (below). `H2Syn` uses the discrete-Riccati current-estimator controller (with feedthrough) and allows D11 ≠ 0.
- `HinfSyn` on discrete plants with modes at both z = 1 and z = −1 returns a controller instead of `ErrOptionUnsupported`, as MATLAB `hinfsyn` does: a static output feedback u = D0·y + v first moves the unit-circle modes (an exact loop transformation, so the closed loops and γ are unchanged) and K is the Tustin design on the shifted plant plus D0. The same shift now applies when the modes are only near ±1, where the Tustin-only design lost accuracy (γ = 1.5 instead of 0.44 with a mode at −1 + 1e-5, 4e4 at −1 + 1e-7, and controllers exceeding γ by 0.2% at −1 + 1e-4).
- `HinfSyn`/`Mixsyn` no longer fail with `ErrAlgebraicLoop` when the central controller's D22 loop shift is ill-posed (the square one-block problem, e.g. `Mixsyn(G, W1, nil, nil)` with biproper G, where I + D22·Dk = 0). HinfSyn then returns the non-central controller of a constant Youla parameter ‖Q‖ = γ/2, which meets the same γ. For MIMO G this case previously returned a destabilizing controller with a nil error, since I + D22·Dk cancelled to rounding noise.
- `HinfSyn`/`Mixsyn` verify `‖T_zw‖∞ ≤ GammaOpt` on the returned K and, when the build fails or K misses its γ, back γ off upward (doubling margin) until a controller meets it. This fixes problems whose infimum is 0 and unattained (e.g. `Mixsyn(G, W1, nil, nil)` with minimum-phase biproper G), which failed with `ErrNoStabilizing` for MIMO G and, on amd64 or for discrete G, returned controllers exceeding the reported γ by 25% to 200×. Bisection now also stops at an absolute γ tolerance of 1e-9 (relative 1e-6 still governs γ > 1e-3). Behaviour change: in that regime `GammaOpt` is the smallest backed-off γ whose controller meets it (≈ 1e-4), not within 1e-4 of the infimum.

### Added

- `Makeweight(dcgain, freqMag, hfgain, Ts, N)`, MATLAB `makeweight`: monotonic loop-shaping weight through `dcgain`, `mag` at `freq` (or `[wc]` for `|W| = 1`) and `hfgain`; order `N` with Butterworth-pattern poles and zeros; discrete (`Ts > 0`) by Tustin prewarped at `freq`.
- `Augw(G, W1, W2, W3)`, MATLAB `augw`: mixed-sensitivity generalized plant `[W1 −W1·G; 0 W2; 0 W3·G; I −G]` with nil weights omitted, SISO weights expanded as `W·I`, and improper `W3` accepted when `W3·G` is proper; channels named `w`, `u`, `z1`, `z2`, `z3`, `e`.
- `Mixsyn(G, W1, W2, W3)`, MATLAB `mixsyn`: `HinfSyn` on `Augw`, returning `MixsynResult{K, CL, Gamma, Info}` with `Gamma = ‖CL‖∞`; rank-deficient D12 (e.g. `W2` nil with strictly proper continuous `G`) returns `ErrInvalidPartition`; discrete `G` supported.

### Performance

- Delayed-loop `DiskMargin`/`DiskMarginSkew`/`Margin`: one reusable point evaluator per loop instead of fresh LFT/balancing/realization workspaces per frequency point, plus the tail-bound grid cut above: the PID + e^{−0.5s} case goes from 54 ms / 81 MiB (v2.0.0) to about 0.5 ms / 45 KB (#305, #308, #313).
- `DefaultFrequencyGrid`/`Zeros`: minimum Dggev workspace for pencils below the blocking threshold; grid unchanged bit-for-bit, about 90% less memory (#306).
- MIMO internal-delay `HinfNorm`: flat cached torus grid and pooled certification samples, 8514 → 172 allocations per call, bit-identical results (#315).

## v2.0.0

v2 is a breaking release. It applies one set of rules to the whole API:

- **Go error pattern.** Every call returns a complete value with a nil error, or a non-nil error. No more nil, empty, NaN, 0 or -1 placeholders returned with a nil error, and no panics on well-typed input, nil pointers included.
- **Comma-ok accessors.** A quantity that can be undefined is read through a method returning `(value, ok bool)`, for example `w, ok := r.GainCrossover()`. `+Inf` appears only where it is mathematically true.
- **MATLAB argument shapes and defaults.** Functions take the arguments, defaults and option structs of their Control System Toolbox counterparts (`c2d`, `lqg`, `estim`, `lft`, `connect`, `feedback`, `pid`, `pidtune`, `systune`, `TuningGoal.*`, ...).
- **0-based indexing everywhere.** Every index argument, accessor and error message is 0-based, with no exceptions. MATLAB's 1-based indices map to `index-1`.
- **Correctness fixes.** An oracle audit of the whole library found results that were silently wrong (see [Fixes](#fixes)). Several of these fixes change numbers on unchanged signatures.

Migrate the import path:

```diff
- import "plantcontrol.org/v1/controlsys"
+ import "plantcontrol.org/v1/controlsys/v2"
```

```sh
go get plantcontrol.org/v1/controlsys/v2@v2.0.0
```

Only `errors.Is` / `errors.As` on the exported sentinels is a stable way to match errors. Error message text changed throughout. Every message now has the form `OpName: detail: <sentinel>`.

---

## Silent behaviour changes

These calls still compile, or migrate in an obvious way, but **return different results**. Review them first.

### Defaults and conventions

| Area | v1.13.1 | v2 |
|---|---|---|
| Discretization default | `Discretize(dt)` / `Undiscretize()` used **Tustin** | `C2D(dt, C2DOptions{})` / `D2C(D2COptions{})` use **ZOH**, the MATLAB default. Moving `Discretize(dt)` to `C2D(dt, C2DOptions{})` changes the method. Pass `Method: C2DMethodTustin` to keep the old result. (An empty method in `DiscretizeWithOpts` was already ZOH.) [#264] |
| `Modred` / `EliminateStates` / `Balred` method | `BalredMethod` zero value (and literal `0`) = `Truncate` | `StateProjection` zero value = `MatchDC`. `Truncate` is now `1`. A literal `0`, a zero `BalredOptions{}` or a persisted integer now selects MatchDC. [#255] |
| `Balred(sys, order, ...)` | order 0 auto-selected an order from the HSV; order n returned `ErrInvalidOrder` | Order 0 returns the static-gain reduction (choose the order yourself from the returned HSV); order n returns the balanced realization; order outside [0, n] returns `ErrInvalidOrder`. [#255, #197] |
| `Modred`, every state eliminated, MatchDC | returned `D` | returns the DC gain of the model (`D − C·M⁻¹·B`). Discrete singular perturbation uses `(I−A22)⁻¹`, so the DC gain is now kept. [#255, #197] |
| `CanonResult.T` | `x = T·xc` | MATLAB convention: `xc = T·x`, so `Sys.A = T·A·T⁻¹`. Invert old uses. [#255] |
| `Canon(sys, CanonCompanion)` | observable companion form, SISO only | MATLAB companion form: ones on the subdiagonal and the characteristic polynomial in the last column. Requires controllability from input 0. MIMO is supported. An empty form selects modal. [#255] |
| `Estim` default (`known` nil) | every input known, so estimator inputs were `[u; y]` | MATLAB `estim(sys,L)`: every input is stochastic and the estimator input is `y` only. Pass all input indices as `known` to keep `[u; y]`. [#284] |
| `Feedback(plant, k, sign)` with plant `InputDelay` | input delay hoisted outside the loop | Every plant and controller delay is inside the loop as an internal delay (MATLAB). The closed loop changes whenever the plant has an input delay. [#192] |
| `Connect` with delayed blocks | Input/OutputDelay of selected channels hoisted even when looped | External delays remain only on channels outside every connection. [#192] |
| `DiskMargin` | skew σ = 1 (α = 1/‖S‖∞) | Balanced skew σ = 0, the MATLAB default: α = 1/‖S − ½‖∞, GM `[(2−α)/(2+α), (2+α)/(2−α)]`, PM `2·atan(α/2)`. `DiskMarginSkew(sys, 1)` reproduces v1. An unstable closed loop gives Alpha 0, GM [1 1], PM 0. [#217] |
| `HinfNorm` of an unstable model | `ErrUnstable` | `(+Inf, +Inf, nil)`, as in MATLAB `hinfnorm`. A peak reached only as ω → ∞ returns `omega = +Inf` (`π/T` for discrete models). The discrete peak frequency is no longer Tustin-warped. [#217, #240, #208] |
| `Norm(sys, Inf)` / `Norm(sys, 2)`, unstable model | `ErrUnstable` | `Inf` returns the L∞ peak regardless of stability (`+Inf` with a pole on the boundary); `2` returns `+Inf`. [#208] |
| `Margin` selection | smallest positive margin | MATLAB rule: the GM closest to 0 dB and the PM closest to 0°, with the lower frequency on ties. PM is wrapped to (−180°, 180°]. Phase crossings are taken at −180° mod 360°, so the 0° wrap no longer counts. Crossovers are exact pencil roots, no longer grid-bounded. The same rules apply to `FRDMargin`. [#193, #199, #227, #245] |
| `TransferFunction()` → `TransferFunc.Delay` | IODelay matrix only | Total external delay (IODelay + InputDelay + OutputDelay). Internal delays return `ErrDelayNotRepresentable`. [#195] |
| `Damp` | origin pole ζ = 0; unstable pole τ = +Inf; z = 0 gave ζ = NaN | MATLAB `damp`: ζ = −cos∠s, so ζ = −1 at s = 0 or z = 1. τ = −1/Re s is negative for unstable poles and +Inf only on the boundary. At z = 0: Wn = +Inf, ζ = 1, τ = 0. [#258, #196] |
| `Poles` / `Damp` / `Pzmap` with internal delays | eig(A), ignoring the delay loop | MATLAB `pole`: internal delays set to zero, loop closed. Can return `ErrAlgebraicLoop`. [#206] |
| `Zeros` with internal delays; SISO `Zeros` | delay-free block; SISO used minimal-TF numerator roots | Internal delays are set to zero, as in MATLAB `zero`. All models use pencil invariant zeros, so cancelled zeros of non-minimal SISO realizations are kept. `ZerosDetail().Rank` is the normal rank, and rank(D) for static gains. [#203, #206, #253] |
| `Lqrd(A, B, Q, R, dt, opts)` | Dlqr on raw Q, R; `opts.S` a raw discrete cross term | Discretizes the continuous cost as MATLAB `lqrd` (Van Loan); `opts.S` is the continuous N. **K and X change for every input.** [#202] |
| `Kalman` with D ≠ 0 | D ignored | MATLAB `kalman`: every input is noise (G = B, H = D). [#202] |
| `Simulate` → `Response.XFinal` | always set | `nil` unless `SimulateOpts{FinalState: true}`. When requested but not representable, returns an error. Never aliases `Workspace`. XFinal now reflects delayed inputs. [#241, #267] |
| `Simulate` with x0 and an IODelay matrix | free response shifted by OutputDelay only | Shifted by the output share of `Delay`, the same split as `PullDelaysToLFT`. A nondecomposable `Delay` with nonzero x0 returns `ErrDelayUnsupported`. [#244] |

### Analysis and frequency response

- `Stabsep`: the feedthrough D moves to `Stable` (MATLAB). `Modsep`: D stays in `Fast`, now also for static gains. Descriptor input returns explicit parts. [#203, #197]
- `IsStable`, `Stabsep` and everything gated on `IsStable` (Gram, HSV, Balreal, H2Norm, Covar): a pole within 1e-10·max(1,|p|) of the boundary counts as unstable. Discrete internal delays are absorbed exactly. Continuous internal delays return `ErrContinuousInternalDelay`. [#197, #201]
- `FreqResponse` / `EvalFr` exactly at a pole: the entries the pole reaches are `Inf` and the rest are finite, for every realization. This used to be `ErrSingularTransform` (descriptor, internal delay, explicit `EvalFr`) or NaN. MIMO `Sigma` at a pole now returns `ErrInvalidArgument`; SISO `Sigma` there is `+Inf`. [#228, #254]
- Discrete frequency responses are evaluated exactly on the unit circle, and near-pole solves are refined. Low-order digits change, toward the exact value. Discrete `EvalFr(e^{jωT})` is no longer bit-identical to `FreqResponse(ω)`. [#270, #299, #302]
- Default grid (`Bode`/`Sigma`/`Nichols` with nil omega, now `DefaultFrequencyGrid`): it spans poles, zeros and delay corners, has no [1e-4, 1e4] clamp, and discrete grids end at π/Dt. `Bandwidth` now finds drops past high-frequency zeros. [#223, #230]
- `Nyquist`: `Omega` is strictly increasing and `Contour` excludes indentation points (MATLAB). Encirclements are counted on an internal indented D-contour, so boundary poles count as stable. [#236] `(*FRD).Nyquist` winds the full closed contour, and `ContourN` is now `conj(Contour)` reversed. [#237]
- `DiskMargin` on continuous delayed loops now returns an exact Nyquist-based margin. In v1 it ignored the delay. [#201, #213]
- `Covar` honours input, output and IO delays. Continuous `D` gives `Inf` only where two feedthrough paths share a delay. [#197, #200]
- `HSV` of discrete internal-delay models: one value per state plus one per delay sample. [#201]
- `Prescale` no longer permutes states; `StateScale` is a true scale. `Ssbal` uses the TB01ID loop: the `T` values differ, and the diagonal of A is excluded from the norms. [#207, #212]
- `DescriptorE()` returns `eye(n)` for explicit models (was nil). Test `IsDescriptor()` instead. [#257]
- `AbsorbDelay` with several scopes absorbs their union (v1 used only the first). `SetInternalDelay` with an empty non-nil tau clears the internal delays. [#261]
- `Place` with rank(B) ≥ 2 uses KNV robust assignment (MATLAB `place`), so K differs and is better conditioned. Repeated real poles (multiplicity ≤ rank B) give a non-defective closed loop. [#234, #229, #191]

### Time responses

- Grids now include `tFinal`. Continuous auto-dt follows the requested `tFinal`. The auto horizon accounts for damping and delays, so it is longer. The minimum is max(10, n+1) samples and the cap is 1e5. [#196]
- Continuous `Impulse` is exact (`C·e^{At}·B`) and drops the Dirac `D·δ(t)`, as MATLAB does. In v1 it was a 1/dt pulse. [#196, #205]
- `Initial` / `Lsim` with x0: input and IO delays no longer delay the free response; only the output share does. [#196, #244]
- Continuous responses of models whose internal delays share one length τ are computed exactly by the method of steps. [#205]
- `StepInfoForSystem` uses `DCGain` as the steady state, and `PeakTime` is the first sample of a flat peak (MATLAB). [#196]
- `GenSig` (new signature): `"square"` levels are 0/1, `"pulse"` is a unit sample at each multiple of tau, every signal is 0 at t = 0, and `"step"` is removed. [#269]

### Identification and tuning

- `ERA`: Hankel size is `(len−1)/2`, so even-length sequences no longer leave a zero block. [#194]
- `FitProcess` / `EvaluateProcess` with `EstimateInitialState` assume the input was held at `Input[0]` before the record. [#194]
- `IdentifyIOStateSpace`: `InitialHistory` has length `Order+InputDelay`, and the `estimate` validation needs `InitializationSamples ≥ MaxOrder+InputDelay`. [#194]
- `FreqRespEst`: bins where the estimator is undefined are omitted, so `Omega`/`H` can be shorter than NFFT/2+1. [#279]
- `Pidtune` on discrete plants solves against the discrete controller, and PID (Tf = 0) returns `DFormula = BackwardEuler`. The PIDF phase loop is fixed. [#194]
- `NewTunableReal` is unbounded (±Inf) by default. Margin goals are disk-based. `TuningGoalResult.Value`/`Limit` are normalized (f, 1). `SystuneResult.Parameters` keys are generated names such as `"C.Kp"`. [#283, #297, #300]

### Persisted integers

- `Truncate` changed from 0 to 1.
- `TuningGoalType` values are renumbered: Rejection 1→3, Sensitivity 2→4, WeightedGain 3→2, LoopShape 4→5, Overshoot 7→8.

Typed string enums keep their string values, so JSON is unchanged.

### Same signature, now returns an error

| Call | New error |
|---|---|
| nil `*System`, nil matrix, NaN/Inf model or argument, bad index (all ops) | `ErrInvalidArgument`. These used to panic, return garbage or return `ErrDimensionMismatch`. |
| `Validate` (and every op that validates) with `InputName`/`OutputName`/`StateName` length ≠ m/p/n | `ErrDimensionMismatch` |
| `Lqg`, `Kalman`, `Kalmd`, `Estim` (output/IO/internal), `Reg`, `H2Syn`, `HinfSyn`, continuous `RootLocus` on delayed plants | `ErrDelayUnsupported` |
| `Kalmd` with D ≠ 0 | `ErrNoiseFeedthrough` |
| `Kalmd` / `Kalman` / `Lqe` with `opts.S` | `ErrOptionUnsupported` (pass Nn) |
| `Place` with a pole repeated more than rank(B) times, **including SISO** | `ErrPoleMultiplicity` (use `Acker`) |
| `Feedback` / `FRDFeedback` with sign ∉ {−1, +1} | `ErrInvalidArgument` |
| Interconnects (`Series`/`Parallel`/`Append`/`BlkDiag`/`Feedback`/`Connect`/`Augstate`) with fractional discrete delays or invalid models | `ErrFractionalDelay` / the `Validate` error |
| `Step`/`Impulse`/`Initial` with negative or non-finite tFinal (0 = auto) | `ErrInvalidArgument` |
| `DCGain`/`Step`/`Impulse` with m = 0 or p = 0; `Initial`/`Lsim` with p = 0 | `ErrDimensionMismatch` |
| `Simulate` with zero samples; with p = 0 and no `FinalState` | `ErrInsufficientData`; `ErrDimensionMismatch` |
| n = 0 in `Gram`, `Lyap`, `DLyap`, `Care`, `Dare`, `Lqr`, `Dlqr`, `Lqe`, `Lqrd`, `Balreal`, `Canon`, `Place`, `Acker`, `Ctrb`, `Obsv`, `Ssbal`, `ModalTruncate`, `RootLocus`, `CtrbF`/`ObsvF` | `ErrDimensionMismatch` |
| Ops that would leave a p×0 / 0×m static gain (`Reduce`, `MinimalRealization`, `Stabsep`, `Modsep`, `Modred`, `Sminreal`, ...) | `ErrDimensionMismatch` |
| `Bandwidth` with zero or infinite DC gain; on MIMO models; dbDrop not finite negative | `ErrInvalidArgument`; `ErrNotSISO`; `ErrInvalidArgument` |
| `Nyquist` with closed-loop poles on the contour; `FRD.Nyquist` contour through −1 | `ErrSingularTransform` |
| `FreqResponse`/`Bode`/`Nichols`/`Sigma`/`System.FRD` with a non-nil empty omega (nil = default grid) | `ErrInvalidArgument` |
| `Covar` with no outputs; `Sigma` with m = 0 or p = 0 | `ErrDimensionMismatch` |
| `NewFRD` with empty, duplicate or unsorted omega, or NaN data; empty `SelectFrequencies` | `ErrInvalidArgument` |
| `Poly{}.Roots()` (zero polynomial) | `ErrInvalidArgument`. Eigen failure is `ErrSchurFailed`. |
| `Pade(order)` with order outside 1..10, even with no delay | `ErrInvalidArgument` |
| `Reduce` unknown Mode / bad Tol; `AbsorbDelay` unknown scope; `Canon` unknown form | `ErrInvalidArgument` |
| `ERA` order above the Hankel numerical rank | `ErrInvalidOrder` |
| `SumBlk` with unequal or non-positive widths | `ErrDimensionMismatch` / `ErrInvalidArgument` |
| `SmithPredictor` with a delayed model | `ErrDelayUnsupported` |
| `Pidtune` with an explicit phase margin that is unattainable; no inferable crossover | `ErrPIDTuningTargetUnattainable`; `ErrInvalidArgument` |
| `TunePID` with `UnstablePoles` on the System path | `ErrOptionUnsupported` |
| `Gram` on internal-delay models | `ErrInternalDelayUnsupported` |
| continuous `Impulse` with internal delay and D21 ≠ 0 | `ErrInternalDelayImpulse` |
| time responses of improper descriptors; singular E with x0 ≠ 0 | `ErrImproperModel`; `ErrDescriptorInitialState` |

The reverse also happened: several calls that used to error now succeed. These include proper singular-E descriptors in time responses and conversions, `D2C`/`D2D` with internal delays, `Ssbal`/`Prescale`/`SS2SS`/`Xperm` on descriptor and delayed models, Riccati designs with nonsingular E (`RiccatiOpts.E`), `RootLocus` with D ≠ 0, `FixedInputReduction` on descriptors, and m = 0 / p = 0 models across ~95 ops (these used to panic). [#211, #214, #216, #222, #215, #243, #253, #198, #226]

---

## API changes

### Errors, sentinels and accessors

- **New sentinels**: `ErrInvalidArgument`, `ErrOptionUnsupported`, `ErrDelayUnsupported`, `ErrDelayNotRepresentable`, `ErrContinuousInternalDelay`, `ErrInternalDelayUnsupported`, `ErrInternalDelayImpulse`, `ErrDescriptorInitialState`, `ErrImproperModel`, `ErrFixedInputDelayMismatch`, `ErrNoiseFeedthrough`, `ErrPoleMultiplicity`, `ErrVoidModel`, `ErrNoTuningCandidate`.
- **Remapped sentinels**: nil arguments, NaN/Inf values, out-of-range or duplicate indices, and unknown enums or methods are `ErrInvalidArgument`. Many of these were `ErrDimensionMismatch`, `ErrInsufficientData` or bare errors. Update `errors.Is(err, ErrDimensionMismatch)` checks on those paths. `ErrDimensionMismatch` now means size disagreement or an unstorable result only. Eigen/Schur/QZ non-convergence is `ErrSchurFailed`.
- **Typed errors** for failures that carry data:
  - `TunePID`/`TunePIDFRD` return `(nil, *PIDTuningError)`, with `.Evidence`, and the error unwraps to `ErrPIDTuningTargetUnattainable`.
  - `FitProcess` returns `(nil, *ProcessFitCanceledError)`, with `.Best`, and the error unwraps to the context error.

**Fields → comma-ok methods**

| v1.13.1 | v2 |
|---|---|
| `MarginResult.WgFreq` / `.WpFreq` (NaN if none) | `GainCrossover()` / `PhaseCrossover()` `(w, ok)` |
| `DiskMarginResult.PeakFreq` (field) | `PeakFreq()` / `Frequency()` `(w, ok)`; ok is false for an unstable loop |
| `StepMetric.RiseTime` / `.SettlingTime` / `.Settled` | `RiseTime()` / `SettlingTime()` `(t, ok)` |
| `RootLocusResult.AsymptoteCentroid` | `AsymptoteCentroid()` `(c, ok)` |
| `FreqRespEstResult.Coherence`; `CoherenceAt(...) float64` | `Coherence()` `([]float64, ok)`; `CoherenceAt(...)` `(float64, ok)` |
| `ProcessFitResult.ResidualLagOne` | `ResidualLagOne()` `(v, ok)`. The JSON key is kept and omitted when undefined. |
| `(*PID).Ti()` / `Td()` `float64` | `(float64, bool)` |
| `LqgResult.Ki/Kw/Mx/Mw` (nil if not requested) | `Ki()`, `Kw()`, `Mx()`, `Mw()` `(gain, ok)` |
| `PassivityResult.Passive` (field); `PassivityCertified` | `Passive()`; removed |
| `EKF.X` / `EKF.P` | `State()` / `StateCovariance()`, `SetState` / `SetStateCovariance` |

**Now return an error** (add `err`): `(*System).TotalDelay()` (returns zeros when delay-free), `GetDelayModel()` → `(H, tau, err)`, `(*TransferFunc).Eval` / `EvalMulti`, `(*ZPK).Eval`, `IsProper()` (renamed from `Isproper`, on System and TransferFunc), `NewPhysicalComponent`, `NewPID`/`NewPID2`/`NewTunable*`/`NewXxxGoal`, `GeneralizedModel.InsertAnalysisPoint`. `GeneralizedModel.SetInputName`/`SetOutputName` also return an error now, but statement calls still compile.

**Typed string enums** (untyped literals still compile, and JSON is unchanged):
- `PIDTuningEvidence.Stability` / `.Termination` → `PIDStabilityEvidence` / `PIDTuningTermination`
- `ProcessFitOptions.ValidationInitialCondition` → `ProcessValidationInit`; `ProcessFitResult.Termination` → `ProcessFitTermination`
- `IOStateSpaceOptions.InitialCondition` / `ValidationInitialCondition` → `IOInitialCondition`; `IOStateSpaceCandidate.Convergence` → `IOConvergence`
- `SystuneResult.Method` → `TuneMethod`

`PIDTuningEvidence.Feasible` is removed. `IOStateSpaceCandidate` gains `Err`; failed candidates no longer carry partial metrics. `PrescaleResult.Info` is the named type `PrescaleInfo`.

### Conversion and discretization

| v1.13.1 | v2 |
|---|---|
| `Discretize(dt)` (Tustin) | `C2D(dt, C2DOptions{Method: C2DMethodTustin})` |
| `DiscretizeZOH/FOH/Impulse/Matched(dt)`, `DiscretizeWithOpts(dt, o)` | `C2D(dt, C2DOptions{Method: ...})` (zero Method = ZOH) |
| `DiscretizeLeastSquares(dt, order)`, `LeastSquaresResult` | `C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: order})` → `(*System, LeastSquaresFit, error)` |
| `Undiscretize()` (Tustin) | `D2C(D2COptions{Method: C2DMethodTustin})` |
| `D2C(method)`, `D2CWithOpts(o)` | `D2C(D2COptions{Method: method})` |
| `D2D(dt, C2DOptions)` | `D2D(dt, D2DOptions{Method, PrewarpFrequency})`. ThiranOrder, DelayModeling and FitOrder are not accepted (MATLAB `d2dOptions`). |
| `DiscretizeWithResult`, `D2CWithResult`, `D2DWithResult`, `ConversionResult`, `MapInitialState` | `C2DMap` / `D2CMap` return `(sys, G, err)`, and G is never nil. For d2d, compose `D2CMap` then `C2DMap`. `Warnings` and `Approximate` are dropped. |
| `C2DDelayModelingDelay` | `C2DDelayModelingInternal` |
| `ThiranDelay(tau, order, dt)` | `ThiranDelay(tau, dt)` with the MATLAB automatic order `ceil(tau/dt)`. A fixed order is no longer available (MATLAB has none); `C2DOptions.ThiranOrder` is a maximum order, as in MATLAB `c2dOptions`. |
| `(*TransferFunc).StateSpace(opts)` / `(*ZPK).StateSpace(opts)`, `StateSpaceOpts` | `StateSpace()` (MATLAB `ss(tf)`, non-minimal; follow with `MinimalRealization`) |
| `DecomposeIODelay` | unexported. Test `D[i][j] − D[i][0] − D[0][j] + D[0][0] == 0` locally. `AbsorbDelay`/`PullDelaysToLFT`/`Pade` still split. |

- Error prefixes `Discretize*:` / `Undiscretize:` are now `C2D:` / `D2C:`. A negative `FitOrder` returns `ErrInvalidConversionOptions`.
- New conversion features: `C2D`/`D2C`/`D2D` of proper singular-E descriptors (reduced as in `dss2ss`), and `D2C` zoh/foh and `D2D` with internal delays. [#214, #216]

### Interconnection

| v1.13.1 | v2 |
|---|---|
| `Connect(sys, Q *mat.Dense, inputs, outputs)`, where `Q[i][j]` is the gain from output j to input i | `Connect(blksys, []Junction, inputs, outputs)`. `Q[i][j] = +1` → `Junction{Input: i, Plus: []int{j}}`; `−1` → `Minus: []int{j}`. Group the terms per input. `inputs`/`outputs` stay 0-based and unchanged. Put non-unit gains in a gain block in `blksys`, or use `ConnectByName`. MATLAB row `{7, 2, -15, 6}` = `Junction{Input: 6, Plus: []int{1, 5}, Minus: []int{14}}`. |
| `Connection.Gain` (`ConnectByName`) | Removed. Connections are unity; put signs and gains in a model (`SumBlk`, `NewGain`). A duplicate (From, To) returns `ErrInvalidArgument`. |
| `LFT(M, Delta, nu, ny int)`, counting **external** channels | `LFT(sys1, sys2, LFTFeedback{Nu, Ny})`, the MATLAB `lft` star product where Nu/Ny count **feedback** channels. `LFT(M, D, nw, nz)` → `LFT(M, D, LFTFeedback{Nu: mM−nw, Ny: pM−nz})`, or `LFT(M, D)` when D is strictly smaller. Old call sites fail to compile. `LFT(M, nil, ...)` → `M.SelectByIndex(...)`. States are ordered sys1 then sys2. |
| `Feedback(p, c, sign, opts ...FeedbackOption)` with `WithApproximatedDelays`, `WithPadeOrder(n)`, `WithThiranOrder(n)` | `Feedback(p, c, sign, ...FeedbackChannels{FeedIn, FeedOut})` (MATLAB feedin/feedout, 0-based). For the old options: `cl.AbsorbDelay()` or `cl.Pade(n)` on the result. Calls without options compile unchanged. |
| `NewGeneralizedModel(name, any)`, `NewGeneralizedClosedLoop(name, plant, any, ap)` | Take a `NumericBlock`. Wrap a `*System` with `FixedBlock(sys)`. |

- `SumBlk` checks widths. Delayed `Feedback` results keep `StateName`. `Inv` of a static gain swaps the I/O names. `GeneralizedModel` copies its `*System` blocks at construction.

### Analysis, margins, norms, reduction

| v1.13.1 | v2 |
|---|---|
| `Gram(sys, t) (*GramResult, error)` | `Gram(sys, t) (*mat.Dense, error)`. Pass `GramControllabilityFactor` / `GramObservabilityFactor` for the factor R (W = Rᵀ·R). |
| `BalrealResult.T` / `.Tinv` | `TR` / `TL` (MATLAB `[sysb,g,TL,TR] = balreal`) |
| `Balred(sys, order, BalredMethod)`, `BalredMethod`, `SingularPerturbation` | `Balred(sys, order, BalredOptions{StateProjection: p})`, `StateProjection`, `MatchDC` (zero value) / `Truncate`. `Modred` and `EliminateStates` take `StateProjection`. |
| `CtrbF(A,B,C)` / `ObsvF(A,B,C)` → `*StaircaseResult`; `ControllabilityStaircase` | `CtrbF(A,B,C,tol)` / `ObsvF(A,B,C,tol)` → `*StaircaseForm{A,B,C,T,K}` (MATLAB layout: controllable/observable states last). NCont = `sum(K)`, BlockSizes = `K`. `tol` 0 selects the default. |
| `ReduceResult.BlockSizes` | Removed. Use `CtrbF(...).K`. |
| `ModalTruncateOptions.MaxRealPart float64` (0 = unset) | `*float64` (e.g. `new(-0.5)`); `ModalReductionResult.Method` / `.Kept` removed |
| `Ssbal(sys)` | `Ssbal(sys, ...SsbalOption)`, e.g. `WithCondT(c)`. Existing calls compile. |
| `(*FRD).Nyquist() *NyquistResult` | `*FRDNyquistResult` (no `RHPPoles` / `RHPZerosCL`; add your own open-loop unstable pole count to `Encirclements`) |
| `Isproper()` | `IsProper() (bool, error)` |

- New: `DiskMarginSkew(sys, σ)`, `DiskMarginResult.Skew`, `(*FRD).Validate`, `(*System).DefaultFrequencyGrid(n)`.
- `HinfNorm` / `Norm` / `HSV` / `IsStable` account for internal delays (see above).

### Synthesis: Lqg, Lqi, Kalman, Lqe, Estim, Place

| v1.13.1 | v2 |
|---|---|
| `Lqg(sys, Q, R, Qn, Rn, *RiccatiOpts)`, noise at the plant inputs (G = B) | `Lqg(sys, QXU, QWV, *LqgOpts{QI, OneDOF, Current})`, the MATLAB `lqg`: `QXU = [Q N; N' R]`, `QWV = [Qn Nn; Nn' Rn]` with G = I, H = 0, both symmetric. v1 noise model: `QWV = [B;D]·Qw·[B;D]' + blkdiag(0, Rn)`. `RiccatiOpts.S`/`Workspace` are not accepted. |
| `Lqi(A, B, C, Q, R, opts)` | `Lqi(sys, Q, R, opts)`, the MATLAB `lqi`: `Ba = [B; −D]`; discrete uses forward Euler with Ts. N ((n+p)×m) goes in `opts.S`. |
| `Kalman(sys, Qn, Rn, opts)` with `opts.S` | `Kalman(sys, Qn, Rn, Nn, opts)` (nil Nn = 0). `opts.S` returns `ErrOptionUnsupported`. |
| `Lqe(A, G, C, Qn, Rn, opts)` | `Lqe(A, G, C, Qn, Rn, Nn, opts)` |
| `Estim(sys, L)` | `Estim(sys, L, sensors, known []int)` (0-based; nil sensors = all, nil known = none) |

- New: `RiccatiOpts.E` for generalized `Care`/`Dare` (MATLAB `icare`/`idare`). `Lqr`/`Lqi`/`Lqg`/`Kalman` accept descriptor plants with nonsingular E. A singular E returns `ErrDescriptorSingular`. [#243]
- An undersized `RiccatiWorkspace` returns `ErrDimensionMismatch` instead of panicking.

### Time response and Simulate

| v1.13.1 | v2 |
|---|---|
| `Response.XFinal` always set | Request it with `SimulateOpts{FinalState: true}` |
| — | `SimulateOpts.Steps` simulates models with no inputs (MATLAB `lsim` with a `length(t)×0` u) |
| `GenSig(type, period, dt) (t, u, err)` | `GenSig(type, tau, tf, ts) (u, t, err)`. The outputs are swapped. 0 selects the defaults Tf = 5·tau and Ts = tau/64. |

### Identification

| v1.13.1 | v2 |
|---|---|
| `FreqRespEstOpts.NOverlap int` (0 = default) | `*int`: nil means NFFT/2 and `&0` means no overlap. An invalid NFFT/NOverlap/Window errors instead of being clamped. |
| `FitProcess` cancel returned `(best, ctx.Err())` | `(nil, *ProcessFitCanceledError)` |
| `IOStateSpace*` string fields | typed enums (above) |

`ProcessFitResult` needs ≥ 3 held-out samples for `ResidualLagOne`. Zero gain with `EstimateInitialState` returns `ErrProcessFit`.

### PID and tuning

| v1.13.1 | v2 |
|---|---|
| `NewPID(Kp, Ki, Kd, opts...) *PID` | `NewPID(Kp, Ki, Kd, Tf, Ts, opts...) (*PID, error)` |
| `NewPIDStd(Kp, Ti, Td, opts...)` | `NewPIDStd(Kp, Ti, Td, N, Ts, opts...)` with N = Td/Tf (`+Inf` = no filter) |
| `NewPID2(Kp, Ki, Kd, Tf, b, c, opts...) *PID2` | `NewPID2(Kp, Ki, Kd, Tf, b, c, Ts, opts...) (*PID2, error)` |
| `WithFilter`, `WithTs` | Removed: Tf and Ts are positional. `WithPIDFormulas` is the only `PIDOption`. |
| `Pidtune(plant, type, ...PidtuneOptions)` with `CrossoverFrequency` | `Pidtune(plant, type, wc, *PidtuneOptions)` (wc 0 = auto) |
| `NewTunableReal(name, v, TunableBounds{})` | `NewTunableReal(name, v)` then `SetBounds(lo, hi)`. The default is unbounded (±Inf). |
| `NewTunableGain(name, [][]*TunableReal, dt)` | `NewTunableGain(name, ny, nu)` / `NewTunableGainFrom(name, G)`. Field `D` → `Gain`, plus `Dt`. |
| `NewTunablePID(name, kp, ki, kd, tf, dt)` | `NewTunablePID(name, type, Ts)` / `NewTunablePIDFrom(name, pid)`. `Tf` is a `*TunableReal`; `blk.Tf.SetFixed(true)` keeps it fixed. New `TunablePID2`. |
| `NewTunableTF(name, num, den, dt)`, `Num`/`Den` | `NewTunableTF(name, nz, np, Ts)` / `NewTunableTFFrom(name, tf)`, SISO only, fields `Numerator`/`Denominator`. The denominator is tunable. |
| `NewTunableSS(name, A, B, C, D, dt)` | `NewTunableSS(name, nx, ny, nu, Ts, TunableSSStructure)` / `NewTunableSSFrom(name, sys, s)` |
| `RandomSample(*math/rand.Rand)` | `RandomSample(*math/rand/v2.Rand)` (nil is rejected) |
| `Systune(model, goals, opts)` / `Looptune(...)` / `GridTune(...)` | `Systune(ctx, CL0, soft, hard, opts)`; `Looptune(ctx, G0, C0, wc, reqs, *LooptuneOptions)`; `GridTune(ctx, ...)`. `SystuneResult.HardScore` is added. A free parameter with infinite bounds is rejected by the grid search. |
| `NewTuningGoal`, `TuningGoalSpec` (`.Omega`), name argument | Per-class constructors, `goal.WithName(name)`, `goal.WithFocus(wmin, wmax)` (returns an error) |
| `NewMarginGoal(name, gm, pm)` | `NewMarginsGoal(location, gm, pm)` (disk-based) |
| `NewPoleGoal(name, maxRe)` | `NewPolesGoal("", -maxRe, 0, math.Inf(1))` |
| `NewTrackingGoal(name, maxDCErr)` | `NewTrackingGoal(in, out, responsetime, dcerror, peakerror)` (frequency-domain profile) |
| `NewRejectionGoal` / `NewSensitivityGoal(name, max)` | `(location, profile *System)`. Use `NewGain` for a constant. Rejection takes attfact = 1/max. |
| `NewWeightedGainGoal(name, max)` | `NewGainGoal(in, out, max)` or `NewWeightedGainGoal(in, out, WL, WR)` |
| `NewLoopShapeGoal(name, min, max)` | `NewLoopShapeGoal(location, loopgain, crosstol)` / `NewLoopShapeGoalWc(location, []float64{wc})` or `{wcmin, wcmax}` |
| `NewOvershootGoal(name, pct)` | `NewOvershootGoal(in, out, pct)` |
| `TuningGoalMargin` / `TuningGoalPole` | `TuningGoalMargins` / `TuningGoalPoles`; `TuningGoalGain` is added. `String()` returns MATLAB class names. |
| `TuningGoal.Evaluate(any)` | `Evaluate(TuningGoalModel)` |

`TunablePID` is no longer comparable (no `==`, not a map key). I/O goals evaluated on a bare `*System` need its `InputName`/`OutputName` set. A single Looptune `wc` gets a 0.1-decade tolerance; pass `[wcmin, wcmax]` for a wider band.

### FRD and model arrays

- `NewFRD` validates its input (see the error table). `FRDFeedback` wraps `ErrAlgebraicLoop`. `(*FRD).Nyquist` returns `*FRDNyquistResult`.
- `ModelArray.Model` / `ModelFlat` return `(*System, error)`, with `ErrVoidModel` for void slots; `IsVoid(...)` probes a slot. `Dims` → `IOSize() (p, m)`. `ModelArray{Freq,Time}Response.Responses` / `.Void` → `ResponseFlat(i)`. An array whose slots are all void returns `ErrInvalidArgument`.

---

## Fixes

Correctness fixes found by the oracle audit. Most also appear above as behaviour changes.

- Feedback/Connect/BlkDiag delay semantics; Feedback LFT formula with feedthrough on both sides; Loopsens through Feedback [#192](https://github.com/PlantControl/controlsys/pull/192)
- Place: wrong poles with a complex open-loop pair, and a silent non-converged K [#191](https://github.com/PlantControl/controlsys/pull/191); 2×2 accuracy at scale, near-scalar blocks [#229](https://github.com/PlantControl/controlsys/pull/229); KNV robust MIMO assignment [#234](https://github.com/PlantControl/controlsys/pull/234)
- Lqg/Kalmd option leaks; Estim/Reg descriptor and delays [#190](https://github.com/PlantControl/controlsys/pull/190); Kalman feedthrough, Lqrd cost, estimator delays [#202](https://github.com/PlantControl/controlsys/pull/202); Lqg MATLAB semantics [#232](https://github.com/PlantControl/controlsys/pull/232); Lqi D and discrete Ts [#238](https://github.com/PlantControl/controlsys/pull/238); nonsingular-E Riccati designs [#243](https://github.com/PlantControl/controlsys/pull/243)
- Margin crossing detection, PM wrap, delay-aware grid [#193](https://github.com/PlantControl/controlsys/pull/193); FRDMargin −180° mod 360 [#199](https://github.com/PlantControl/controlsys/pull/199); exact crossovers and MATLAB selection [#227](https://github.com/PlantControl/controlsys/pull/227); state-space evaluator for SISO loops [#233](https://github.com/PlantControl/controlsys/pull/233); pencil-based crossings at high order [#245](https://github.com/PlantControl/controlsys/pull/245)
- ERA even length, delayed-excerpt identification, process-fit delay history, discrete and PIDF Pidtune [#194](https://github.com/PlantControl/controlsys/pull/194); least-squares c2d on high-order sources [#242](https://github.com/PlantControl/controlsys/pull/242)
- Structural ops dropped delays, LFT and descriptor data (Select, Reduce, SS2SS, Xperm, TF/ZPK) [#195](https://github.com/PlantControl/controlsys/pull/195); TF conversion balances A [#210](https://github.com/PlantControl/controlsys/pull/210); descriptor/delay parity for SS2SS, Xperm, Prescale, Ssbal [#222](https://github.com/PlantControl/controlsys/pull/222)
- Time responses: final sample dropped, auto dt/horizon, exact continuous Impulse/Initial/Lsim, DCGain at DC poles, Damp at z = 0, StepInfo [#196](https://github.com/PlantControl/controlsys/pull/196); exact responses with a common internal delay [#205](https://github.com/PlantControl/controlsys/pull/205); descriptor time responses [#211](https://github.com/PlantControl/controlsys/pull/211)
- Descriptor/stability in Zeros, Stabsep, Modsep, IsStable, H2Norm, Covar, singular perturbation [#197](https://github.com/PlantControl/controlsys/pull/197); singular-E Zeros/Stabsep/Modsep [#203](https://github.com/PlantControl/controlsys/pull/203); MATLAB pole/zero semantics for internal delays [#206](https://github.com/PlantControl/controlsys/pull/206)
- FixedInputReduction on LFT, descriptor and delayed models [#198](https://github.com/PlantControl/controlsys/pull/198)
- Covar with input, output and IO delays [#200](https://github.com/PlantControl/controlsys/pull/200)
- Delay-aware IsStable/HinfNorm/HSV/Gram [#201](https://github.com/PlantControl/controlsys/pull/201); Norm(sys, Inf) as L∞ [#208](https://github.com/PlantControl/controlsys/pull/208); HinfNorm Inf for unstable models, DiskMargin σ = 0 [#217](https://github.com/PlantControl/controlsys/pull/217); exact DiskMargin/TunePID for continuous delayed loops [#213](https://github.com/PlantControl/controlsys/pull/213); accurate H∞ peak [#240](https://github.com/PlantControl/controlsys/pull/240)
- Prescale permuted states [#207](https://github.com/PlantControl/controlsys/pull/207); Ssbal divergence [#212](https://github.com/PlantControl/controlsys/pull/212); Ssbal descriptor and condT [#215](https://github.com/PlantControl/controlsys/pull/215)
- c2d/d2c/d2d of singular-E descriptors [#214](https://github.com/PlantControl/controlsys/pull/214); d2c/d2d with internal delays [#216](https://github.com/PlantControl/controlsys/pull/216)
- 2×2 Schur eigenvalues by Dlanv2 [#221](https://github.com/PlantControl/controlsys/pull/221)
- Auto frequency grid [#223](https://github.com/PlantControl/controlsys/pull/223), exported as DefaultFrequencyGrid [#230](https://github.com/PlantControl/controlsys/pull/230)
- Panics on no-input/no-output models; non-finite dt and delays accepted [#226](https://github.com/PlantControl/controlsys/pull/226); zero-width partitions, Simulate Steps [#235](https://github.com/PlantControl/controlsys/pull/235); unstorable p×0/0×m gains [#239](https://github.com/PlantControl/controlsys/pull/239); phantom states in Append [#231](https://github.com/PlantControl/controlsys/pull/231)
- Inf, not an error, at poles [#228](https://github.com/PlantControl/controlsys/pull/228)
- Nyquist grid and indented-contour encirclements [#236](https://github.com/PlantControl/controlsys/pull/236); FRD Nyquist full contour [#237](https://github.com/PlantControl/controlsys/pull/237); poles on the contour [#263](https://github.com/PlantControl/controlsys/pull/263), [#294](https://github.com/PlantControl/controlsys/pull/294)
- Simulate XFinal with delayed inputs [#241](https://github.com/PlantControl/controlsys/pull/241); x0 with an IODelay output share [#244](https://github.com/PlantControl/controlsys/pull/244)
- Frequency-response accuracy and speed: balanced solves, iterative refinement, fused descriptor pencils, exact unit-circle points, near-pole refinement [#204](https://github.com/PlantControl/controlsys/pull/204), [#218](https://github.com/PlantControl/controlsys/pull/218), [#219](https://github.com/PlantControl/controlsys/pull/219), [#220](https://github.com/PlantControl/controlsys/pull/220), [#224](https://github.com/PlantControl/controlsys/pull/224), [#225](https://github.com/PlantControl/controlsys/pull/225), [#270](https://github.com/PlantControl/controlsys/pull/270), [#299](https://github.com/PlantControl/controlsys/pull/299), [#302](https://github.com/PlantControl/controlsys/pull/302)
- Go error pattern sweep (panics, swallowed errors, placeholders, sentinels, NaN guards, MATLAB argument shapes): [#246](https://github.com/PlantControl/controlsys/pull/246)–[#248](https://github.com/PlantControl/controlsys/pull/248), [#250](https://github.com/PlantControl/controlsys/pull/250)–[#259](https://github.com/PlantControl/controlsys/pull/259), [#261](https://github.com/PlantControl/controlsys/pull/261)–[#269](https://github.com/PlantControl/controlsys/pull/269), [#271](https://github.com/PlantControl/controlsys/pull/271)–[#284](https://github.com/PlantControl/controlsys/pull/284), [#286](https://github.com/PlantControl/controlsys/pull/286)–[#289](https://github.com/PlantControl/controlsys/pull/289), [#291](https://github.com/PlantControl/controlsys/pull/291)–[#293](https://github.com/PlantControl/controlsys/pull/293), [#295](https://github.com/PlantControl/controlsys/pull/295)–[#298](https://github.com/PlantControl/controlsys/pull/298), [#300](https://github.com/PlantControl/controlsys/pull/300), [#301](https://github.com/PlantControl/controlsys/pull/301). Notable silent-wrong results fixed there:
  - Care `Rcnd` was computed from LU factors [#252](https://github.com/PlantControl/controlsys/pull/252)
  - `Poly.MulTo`/`AddTo` aliasing [#247](https://github.com/PlantControl/controlsys/pull/247)
  - Lyap/DLyap near-singular garbage [#254](https://github.com/PlantControl/controlsys/pull/254)
  - RootLocus ignored E and delays [#253](https://github.com/PlantControl/controlsys/pull/253)
  - H2Syn/HinfSyn dropped plant delays [#253](https://github.com/PlantControl/controlsys/pull/253)
  - EKF aliased F's output [#265](https://github.com/PlantControl/controlsys/pull/265)
  - `NewPID2(..., WithFilter)` dropped the filter [#278](https://github.com/PlantControl/controlsys/pull/278)
  - Pidtune clamped the PM silently [#278](https://github.com/PlantControl/controlsys/pull/278)
  - `RandomSample` reseeded per parameter [#283](https://github.com/PlantControl/controlsys/pull/283)
  - SumBlk truncated unequal widths [#274](https://github.com/PlantControl/controlsys/pull/274)
  - ERA returned NaN models [#271](https://github.com/PlantControl/controlsys/pull/271)
  - FreqRespEst replaced user windows [#279](https://github.com/PlantControl/controlsys/pull/279)
- Dependency: `plantcontrol.org/v1/gonum` v0.20.2 [#249](https://github.com/PlantControl/controlsys/pull/249)
