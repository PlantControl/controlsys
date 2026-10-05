# API and Mutation Audit

This audit records the public API ownership and mutation contract for
`plantcontrol.org/v2/controlsys/v2` after the release-finding fixes. It is intentionally
API-by-API: every exported function or method discovered from the current source
is named below. Public data types are grouped separately because most result and
model types expose mutable fields directly.

The package is a toolbox-style control-system package, so a broad public API and
a central state-space model are expected. The audit target is consistency,
ownership clarity, and release readiness, not shrinking the toolbox shape.

## Legend

| Mark | Meaning |
| --- | --- |
| `copy-in` | Copies caller-owned matrices, slices, or models before storing them. |
| `copy-out` | Returns a copy of internally stored data. |
| `view-in` | Reads caller-owned inputs without retaining them. |
| `alias-risk` | May retain caller-owned data or return mutable backing data. |
| `mutates` | Mutates its receiver or an explicitly supplied workspace/buffer. |
| `pure` | Does not intentionally mutate receiver or arguments. |
| `panic-risk` | Can panic for invalid public inputs instead of returning an error. |
| `returns-mutable` | Returns public structs, slices, or matrices that callers can mutate. |
| `workspace` | Uses caller-supplied reusable storage; returned data may share it. |

## Resolved Release Findings

| Priority | API | Finding | Action |
| --- | --- | --- | --- |
| P0 | `NewFromSlices` | Dynamic models aliased caller-owned `a`, `b`, `c`, and `d` slices through `mat.NewDense` and `newNoCopy`. | Fixed by copying dynamic input slices before storage and adding `TestNewFromSlicesDynamicCopiesInputSlices`. |
| P1 | `System`, `TransferFunc`, `ZPK`, `FRD`, result structs | Public fields make mutation possible by design, but ownership rules were not stated uniformly. | Fixed with package-level ownership docs for public mutable models, results, constructors, workspace-backed outputs, and nil receivers. |
| P1 | `Lyap`, `DLyap`, `Care`, `Dare`, `Lqr`, `Dlqr`, `Lqi`, `Lqrd` | Workspace-backed result ownership was not consistently visible across Riccati/controller APIs. | Fixed by documenting workspace-backed result lifetime in package docs, `RiccatiOpts`, `RiccatiResult`, and `LyapunovOpts`. |
| P1 | `PID`, `PID2`, `TransferFunc`, `ZPK`, `FRD`, `EKF` | Public model structs had no uniform copy helper outside `System`/array workflows. | Fixed with `Copy` methods and independence tests for each model type. |
| P2 | `D2C`, `Pidtune`, `FreqRespEst`, `C2DOptions` | Several APIs used free-form strings for method/type selectors. | Fixed with typed selector fields and parameters for C2D/D2C methods, C2D delay modeling, PID tuning types, and frequency-response estimator methods. |
| P2 | `NewTrackingGoal`, `NewRejectionGoal`, `NewSensitivityGoal`, `NewWeightedGainGoal`, `NewLoopShapeGoal`, `NewMarginsGoal`, `NewPolesGoal`, `NewOvershootGoal` | Convenience constructors called `mustTuningGoal` and could panic on invalid names or bounds. | Fixed: every goal constructor takes MATLAB arguments and returns an error. |
| P2 | Nil receiver behavior | Some methods are nil-safe (`ModelArray`, `TunableReal` accessors), while most model methods are not. | Fixed with package-level docs: nil receivers are unsupported unless a method explicitly documents nil-safe behavior. |

## Core State-Space APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `New` | `copy-in`, `returns-mutable` | Copies matrix inputs before storing. |
| `NewGain` | `copy-in`, `returns-mutable` | Copies feedthrough matrix. |
| `NewFromSlices` | `copy-in`, `returns-mutable` | Dynamic and gain-only paths copy caller slices. |
| `NewWithDelay` | `copy-in`, `returns-mutable` | Builds through `New` and delay setter. |
| `NewDescriptor` | `copy-in`, `returns-mutable` | Builds standard model and copies descriptor matrix. |
| `Rss` | `returns-mutable` | Generated random stable continuous-time model. |
| `Drss` | `returns-mutable` | Generated random stable discrete-time model. |
| `(*System).Copy` | `copy-in`, `returns-mutable` | Deep-copies matrices, delays, LFT, names, and notes. |
| `(*System).Validate` | `pure` | Validates dimensions, sample time, descriptor, and delays. |
| `(*System).Dims` | `pure` | Reads dimensions; nil receiver unsupported. |
| `(*System).IsDescriptor` | `pure` | Reads descriptor policy. |
| `(*System).IsContinuous` | `pure` | Reads sample time. |
| `(*System).IsDiscrete` | `pure` | Reads sample time. |
| `(*System).Poles` | `returns-mutable` | Returns caller-owned slice. |
| `(*System).IsStable` | `pure` | Reads poles and sample-time stability boundary. |
| `(*System).String` | `pure` | Renders labels and matrices. |
| `(*System).Isproper` | `pure` | Always true for state-space model. |
| `(*System).SetInputName` | `mutates`, `copy-in` | Mutates receiver metadata; copies names. |
| `(*System).SetOutputName` | `mutates`, `copy-in` | Mutates receiver metadata; copies names. |
| `(*System).SetStateName` | `mutates`, `copy-in` | Mutates receiver metadata; copies names. |
| `(*System).SetDelay` | `mutates`, `copy-in` | Mutates receiver delay matrix. |
| `(*System).SetInputDelay` | `mutates`, `copy-in` | Mutates receiver input delays. |
| `(*System).SetOutputDelay` | `mutates`, `copy-in` | Mutates receiver output delays. |
| `(*System).SetInternalDelay` | `mutates`, `copy-in` | Mutates receiver LFT delay model. |
| `(*System).DescriptorE` | `returns-mutable` | Returns descriptor matrix copy or identity-like dense matrix. |
| `(*System).TotalDelay` | `returns-mutable` | Returns newly allocated total delay matrix. |
| `(*System).HasDelay` | `pure` | Reads delay fields. |
| `(*System).HasInternalDelay` | `pure` | Reads LFT delay field. |

## State-Space Transformations and Interconnections

| API | Classification | Audit note |
| --- | --- | --- |
| `Augstate` | `pure`, `returns-mutable` | Returns augmented copy. |
| `Series` | `pure`, `returns-mutable` | Composes copies/new matrices; does not mutate inputs. |
| `Parallel` | `pure`, `returns-mutable` | Composes copies/new matrices; does not mutate inputs. |
| `Feedback` | `view-in`, `returns-mutable` | Builds closed-loop model; reads `FeedbackChannels` indices by value; does not mutate inputs. |
| `Append` | `pure`, `returns-mutable` | Appends models into a new model. |
| `BlkDiag` | `pure`, `returns-mutable` | Builds block-diagonal model. |
| `Connect` | `view-in`, `returns-mutable` | Reads connection rows by value during construction. |
| `ConnectByName` | `view-in`, `returns-mutable` | Uses named signals and builds through `BlkDiag`/`Connect`. |
| `Inv` | `pure`, `returns-mutable` | Returns inverse model. |
| `LFT` | `pure`, `returns-mutable` | Builds LFT model or visible extraction. |
| `Loopsens` | `pure`, `returns-mutable` | Returns four new loop-sensitivity models. |
| `SS2SS` | `view-in`, `returns-mutable` | Uses transform matrix and returns transformed copy. |
| `Xperm` | `view-in`, `returns-mutable` | Returns permuted copy. |
| `(*System).StateTransform` | `view-in`, `returns-mutable` | Wrapper around state transformation. |
| `(*System).EliminateStates` | `pure`, `returns-mutable` | Wrapper around model reduction. |
| `(*System).FixedInputReduction` | `view-in`, `returns-mutable` | Builds reduced-input model from fixed input map. |
| `(*System).AugmentInternalDelayOutputs` | `pure`, `returns-mutable` | Returns model with augmented internal-delay outputs. |
| `(*System).SelectByIndex` | `view-in`, `returns-mutable` | Returns selected I/O copy. |
| `(*System).SelectByName` | `view-in`, `returns-mutable` | Resolves names and calls `SelectByIndex`. |
| `(*System).ToExplicit` | `pure`, `returns-mutable` | Converts descriptor model to explicit model. |

## Delay and Discretization APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `SetDelayModel` | `view-in`, `returns-mutable` | Builds model from delay model and delay vector. |
| `PadeDelay` | `returns-mutable` | Constructs delay approximation model. |
| `ThiranDelay` | `returns-mutable` | Constructs discrete allpass delay model. |
| `SmithPredictor` | `view-in`, `returns-mutable` | Builds Smith-predictor interconnection from controller, model, and delay. |
| `(*System).AbsorbDelay` | `pure`, `returns-mutable` | Returns copy with selected delays absorbed. |
| `(*System).PullDelaysToLFT` | `pure`, `returns-mutable` | Returns copy with delays pulled into LFT representation. |
| `(*System).GetDelayModel` | `returns-mutable` | Returns delay model and delay slice; caller can mutate both. |
| `(*System).MinimalLFT` | `pure`, `returns-mutable` | Returns reduced LFT delay representation. |
| `(*System).ZeroDelayApprox` | `pure`, `returns-mutable` | Returns zero-delay approximation. |
| `(*System).Pade` | `pure`, `returns-mutable` | Returns delay approximation of receiver. |
| `(*System).C2D` | `pure`, `returns-mutable` | Default discretization wrapper. |
| `(*System).C2D` | `pure`, `returns-mutable` | Option strings remain compatibility risk. |
| `(*System).C2D` | `pure`, `returns-mutable` | Zero-order hold conversion. |
| `(*System).C2D` | `pure`, `returns-mutable` | Impulse-invariant conversion. |
| `(*System).C2D` | `pure`, `returns-mutable` | First-order hold conversion. |
| `(*System).C2D` | `pure`, `returns-mutable` | Matched pole-zero conversion. |
| `(*System).D2C` | `pure`, `returns-mutable` | Inverse bilinear conversion. |
| `(*System).D2C` | `pure`, `returns-mutable` | Uses typed `C2DMethod` selector. |
| `(*System).D2D` | `pure`, `returns-mutable` | Resamples through conversion policy. |

## Analysis, Synthesis, and Numerical Solvers

| API | Classification | Audit note |
| --- | --- | --- |
| `Ctrb` | `view-in`, `returns-mutable` | Returns controllability matrix. |
| `Obsv` | `view-in`, `returns-mutable` | Returns observability matrix. |
| `CtrbF` | `view-in`, `returns-mutable` | Returns staircase result with mutable matrices. |
| `ObsvF` | `view-in`, `returns-mutable` | Returns staircase result with mutable matrices. |
| `ControllabilityStaircase` | `view-in`, `returns-mutable` | Returns mutable staircase result. |
| `IsStabilizable` | `view-in`, `pure` | Matrix predicate. |
| `IsDetectable` | `view-in`, `pure` | Matrix predicate. |
| `Lqr` | `view-in`, `workspace`, `returns-mutable` | Matrix result may be workspace-backed through options. |
| `Dlqr` | `view-in`, `workspace`, `returns-mutable` | Matrix result may be workspace-backed through options. |
| `Lqi` | `view-in`, `workspace`, `returns-mutable` | Matrix result may be workspace-backed through options. |
| `Lqrd` | `view-in`, `workspace`, `returns-mutable` | Matrix result may be workspace-backed through options. |
| `Acker` | `view-in`, `returns-mutable` | Returns gain matrix. |
| `Place` | `view-in`, `returns-mutable` | Returns gain matrix. |
| `Lqe` | `view-in`, `workspace`, `returns-mutable` | Returns estimator Riccati result. |
| `Kalman` | `view-in`, `workspace`, `returns-mutable` | Reads model and returns estimator result. |
| `Kalmd` | `view-in`, `workspace`, `returns-mutable` | Reads model and returns sampled estimator result. |
| `Estim` | `view-in`, `returns-mutable` | Builds observer model. |
| `Reg` | `view-in`, `returns-mutable` | Builds observer-regulator model. |
| `Lqg` | `view-in`, `returns-mutable` | Returns LQG result with mutable model fields. |
| `H2Syn` | `view-in`, `returns-mutable` | Returns synthesis result with mutable controller fields. |
| `HinfSyn` | `view-in`, `returns-mutable` | Returns synthesis result with mutable controller fields. |
| `Care` | `view-in`, `workspace`, `returns-mutable` | Riccati result ownership should be documented with workspace use. |
| `Dare` | `view-in`, `workspace`, `returns-mutable` | Riccati result ownership should be documented with workspace use. |
| `NewRiccatiWorkspace` | `returns-mutable` | Allocates reusable mutable workspace. |
| `Lyap` | `view-in`, `workspace`, `returns-mutable` | Documented to return workspace-backed dense matrix. |
| `DLyap` | `view-in`, `workspace`, `returns-mutable` | Same ownership rule as `Lyap`. |
| `NewLyapunovWorkspace` | `returns-mutable` | Allocates reusable mutable workspace. |
| `Gram` | `view-in`, `returns-mutable` | Returns mutable Gramian/cholesky matrices. |
| `Covar` | `view-in`, `returns-mutable` | Returns mutable covariance matrix. |
| `Norm` | `pure` | Dispatches to norm routines. |
| `H2Norm` | `pure` | Scalar result. |
| `HinfNorm` | `pure` | Scalar/frequency result. |
| `HSV` | `returns-mutable` | Returns caller-owned slice. |
| `Balreal` | `pure`, `returns-mutable` | Returns mutable model and transform matrices. |
| `Balred` | `pure`, `returns-mutable` | Returns reduced model and HSV slice. |
| `Modred` | `view-in`, `returns-mutable` | Returns reduced model. |
| `(*System).Reduce` | `view-in`, `returns-mutable` | Returns reduction result with mutable reduced model. |
| `(*System).MinimalRealization` | `pure`, `returns-mutable` | Wrapper around `Reduce` for full minimal realization. |
| `ModalTruncate` | `view-in`, `returns-mutable` | Returns mutable reduction result. |
| `Modsep` | `pure`, `returns-mutable` | Returns slow/fast models. |
| `Sminreal` | `pure`, `returns-mutable` | Returns minimal model. |
| `Ssbal` | `pure`, `returns-mutable` | Returns balanced model and transform. |
| `Stabsep` | `pure`, `returns-mutable` | Returns stable/unstable models. |
| `Prescale` | `pure`, `returns-mutable` | Returns scaled model and scale slices. |
| `Canon` | `pure`, `returns-mutable` | Returns transformed model and transform matrix. |
| `Pzmap` | `pure`, `returns-mutable` | Returns pole/zero slices. |
| `(*System).Zeros` | `pure`, `returns-mutable` | Returns transmission-zero slice. |
| `(*System).ZerosDetail` | `pure`, `returns-mutable` | Returns transmission zeros plus rank metadata. |
| `Damp` | `pure`, `returns-mutable` | Returns damping info slice. |
| `RootLocus` | `view-in`, `returns-mutable` | Returns mutable locus slices. |
| `Margin` | `pure` | Scalar result struct. |
| `AllMargin` | `returns-mutable` | Returns crossover slices. |
| `Bandwidth` | `pure` | Scalar result. |
| `DiskMargin` | `pure` | Scalar result struct. |
| `DiskMarginSkew` | `pure` | Scalar result struct. |
| `Passive` | `view-in`, `returns-mutable` | Returns passivity result. |
| `FRDPassive` | `view-in`, `returns-mutable` | Reads FRD, returns passivity result. |
| `SpectralFactor` | `pure`, `returns-mutable` | Currently static-gain limited. |

## Time, Frequency, and Identification APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `(*System).FreqResponse` | `view-in`, `returns-mutable` | Returns mutable `FreqResponseMatrix`. |
| `(*System).FRD` | `view-in`, `returns-mutable` | Returns copied FRD data. |
| `(*System).Bode` | `view-in`, `returns-mutable` | Result has unexported data plus accessor methods. |
| `(*System).EvalFr` | `pure`, `returns-mutable` | Returns newly allocated nested complex matrix. |
| `(*System).Nichols` | `view-in`, `returns-mutable` | Result has unexported data plus accessor methods. |
| `(*System).Sigma` | `view-in`, `returns-mutable` | Result has unexported singular-value data. |
| `(*System).Nyquist` | `view-in`, `returns-mutable` | Returns mutable Nyquist slices. |
| `(*System).DCGain` | `pure`, `returns-mutable` | Returns mutable dense matrix. |
| `Step` | `pure`, `returns-mutable` | Returns time response with mutable `T` and `Y`. |
| `Impulse` | `pure`, `returns-mutable` | Returns time response with mutable `T` and `Y`. |
| `Initial` | `view-in`, `returns-mutable` | Reads initial state and returns time response. |
| `Lsim` | `view-in`, `returns-mutable` | Reads input/time arrays and returns time response. |
| `(*System).Simulate` | `view-in`, `workspace`, `returns-mutable` | Mutates supplied buffers in `SimulateOpts`. |
| `StepInfo` | `view-in`, `returns-mutable` | Reads response and returns metrics slice. |
| `StepInfoForSystem` | `pure`, `returns-mutable` | Runs `Step` then `StepInfo`. |
| `FreqRespEst` | `view-in`, `returns-mutable` | Uses typed `FreqRespEstMethod` selector through options. |
| `(*FreqRespEstResult).CoherenceAt` | `pure` | Accessor over exported coherence slice. |
| `(*FreqRespEstResult).FRD` | `returns-mutable` | Converts estimate to FRD. |
| `ERA` | `view-in`, `returns-mutable` | Reads Markov matrices and returns identified model. |
| `Linearize` | `view-in`, `returns-mutable` | Calls user callbacks and returns local linear model. |
| `GenSig` | `returns-mutable` | Returns generated signal slices. |

## Representation Model APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `NewFRD` | `copy-in`, `returns-mutable` | Deep-copies response grid and frequency vector. |
| `(*FRD).Copy` | `copy-out`, `returns-mutable` | Deep-copies response grid, frequency vector, names, and sample time. |
| `(*FRD).Dims` | `pure` | Nil receiver unsupported. |
| `(*FRD).NumFrequencies` | `pure` | Reads exported frequency slice length. |
| `(*FRD).IsContinuous` | `pure` | Reads sample time. |
| `(*FRD).IsDiscrete` | `pure` | Reads sample time. |
| `(*FRD).At` | `pure` | Direct accessor; indexes are unchecked. |
| `(*FRD).Abs` | `pure`, `returns-mutable` | Returns transformed FRD. |
| `(*FRD).SelectFrequencies` | `view-in`, `returns-mutable` | Returns copied subset. |
| `(*FRD).SelectFrequencyRange` | `returns-mutable` | Returns copied subset. |
| `(*FRD).MapResponse` | `alias-risk`, `returns-mutable` | Mapper output is copied into new storage; callback sees mutable nested matrix. |
| `(*FRD).PeakGain` | `pure`, `returns-mutable` | Returns scalar result struct. |
| `FRDConcat` | `view-in`, `returns-mutable` | Concatenates copied FRD data. |
| `(*FRD).EvalFr` | `alias-risk` | Returns nested matrix for one stored frequency; caller can mutate backing response. |
| `(*FRD).FreqResponse` | `returns-mutable` | Returns compatibility matrix from FRD data. |
| `(*FRD).Bode` | `returns-mutable` | Returns Bode result. |
| `(*FRD).Nyquist` | `returns-mutable` | Returns Nyquist result. |
| `(*FRD).Sigma` | `returns-mutable` | Returns singular-value result. |
| `FRDMargin` | `view-in`, `returns-mutable` | Returns margin result. |
| `FRDSeries` | `view-in`, `returns-mutable` | Returns composed FRD. |
| `FRDParallel` | `view-in`, `returns-mutable` | Returns composed FRD. |
| `FRDFeedback` | `view-in`, `returns-mutable` | Returns composed FRD. |
| `(*FreqResponseMatrix).At` | `pure` | Accessor over exported `Data` slice. |
| `(*BodeResult).MagDBAt` | `pure` | Accessor over unexported magnitude data. |
| `(*BodeResult).PhaseAt` | `pure` | Accessor over unexported phase data. |
| `(*NicholsResult).MagDBAt` | `pure` | Accessor over unexported magnitude data. |
| `(*NicholsResult).PhaseAt` | `pure` | Accessor over unexported phase data. |
| `(*SigmaResult).At` | `pure` | Accessor over unexported singular-value data. |
| `(*SigmaResult).NSV` | `pure` | Accessor over singular-value count. |
| `(*TransferFunc).Dims` | `pure` | Reads exported polynomial fields. |
| `(*TransferFunc).Copy` | `copy-out`, `returns-mutable` | Deep-copies polynomial, delay, and name storage. |
| `(*TransferFunc).Eval` | `pure`, `returns-mutable` | Returns newly allocated nested complex matrix. |
| `(*TransferFunc).EvalMulti` | `pure`, `returns-mutable` | Returns newly allocated nested complex grid. |
| `(*TransferFunc).StateSpace` | `view-in`, `returns-mutable` | Converts exported polynomial fields to new model. |
| `(*TransferFunc).Isproper` | `pure` | Reads polynomial degrees. |
| `(*TransferFunc).HasDelay` | `pure` | Reads exported delay field. |
| `(*TransferFunc).ZPK` | `view-in`, `returns-mutable` | Converts to new ZPK model. |
| `NewZPK` | `copy-in`, `returns-mutable` | Copies zeros/poles. |
| `NewZPKMIMO` | `copy-in`, `returns-mutable` | Deep-copies zeros, poles, and gain. |
| `(*ZPK).Dims` | `pure` | Reads exported gain field. |
| `(*ZPK).Copy` | `copy-out`, `returns-mutable` | Deep-copies zero, pole, gain, and name storage. |
| `(*ZPK).IsContinuous` | `pure` | Reads sample time. |
| `(*ZPK).IsDiscrete` | `pure` | Reads sample time. |
| `(*ZPK).Eval` | `pure`, `returns-mutable` | Returns newly allocated nested complex matrix. |
| `(*ZPK).FreqResponse` | `view-in`, `returns-mutable` | Returns new frequency-response matrix. |
| `(*ZPK).TransferFunction` | `pure`, `returns-mutable` | Converts to new transfer-function model. |
| `(*ZPK).StateSpace` | `pure`, `returns-mutable` | Converts via transfer function. |
| `(*System).TransferFunction` | `pure`, `returns-mutable` | Returns mutable transfer-function result. |
| `(*System).ZPKModel` | `pure`, `returns-mutable` | Returns mutable ZPK result. |

## Arrays, Generalized Models, and Tunables

| API | Classification | Audit note |
| --- | --- | --- |
| `NewModelArray` | `copy-in`, `returns-mutable` | Copies shape and each non-nil model. |
| `StackModelArrays` | `copy-in`, `returns-mutable` | Copies constituent models into a new array. |
| `(*ModelArray).Shape` | `copy-out` | Returns shape copy. |
| `(*ModelArray).Len` | `pure` | Nil-safe. |
| `(*ModelArray).Dims` | `pure` | Nil-safe. |
| `(*ModelArray).InputName` | `copy-out` | Returns name copy. |
| `(*ModelArray).OutputName` | `copy-out` | Returns name copy. |
| `(*ModelArray).Model` | `copy-out`, `returns-mutable` | Returns copied model and void flag. |
| `(*ModelArray).ModelFlat` | `copy-out`, `returns-mutable` | Returns copied model and void flag. |
| `(*ModelArray).SelectFlat` | `copy-in`, `returns-mutable` | Returns array of copied models. |
| `(*ModelArray).FreqResponse` | `returns-mutable` | Returns mutable response container. |
| `(*ModelArray).Step` | `returns-mutable` | Returns mutable response container. |
| `NewGeneralizedModel` | `view-in`, `returns-mutable` | Stores `NumericBlock`; `*System` blocks are copied on `CurrentSystem`. |
| `(*GeneralizedModel).SetInputName` | `mutates`, `copy-in` | Mutates generalized-model metadata. |
| `(*GeneralizedModel).SetOutputName` | `mutates`, `copy-in` | Mutates generalized-model metadata. |
| `(*GeneralizedModel).InsertAnalysisPoint` | `mutates` | Mutates analysis-point map. |
| `(*GeneralizedModel).HasAnalysisPoint` | `pure` | Nil-safe. |
| `(*GeneralizedModel).AnalysisPoint` | `pure` | Returns value object. |
| `(*GeneralizedModel).CurrentSystem` | `returns-mutable` | Returns current numeric model. |
| `NewGeneralizedClosedLoop` | `copy-in`, `returns-mutable` | Copies plant; stores controller block. |
| `(*GeneralizedClosedLoop).InsertAnalysisPoint` | `mutates` | Binds a named plant-input or plant-output loop break. |
| `(*GeneralizedClosedLoop).AnalysisPoint` | `pure` | Returns value object. |
| `(*GeneralizedClosedLoop).OpenLoop` | `returns-mutable` | Builds current open-loop model. |
| `(*GeneralizedClosedLoop).ClosedLoop` | `returns-mutable` | Alias for complementary sensitivity. |
| `(*GeneralizedClosedLoop).ComplementarySensitivity` | `returns-mutable` | Builds current closed-loop model. |
| `(*GeneralizedClosedLoop).Sensitivity` | `returns-mutable` | Builds current sensitivity model. |
| `NewTunableReal` | `returns-mutable` | Creates mutable scalar parameter. |
| `(*TunableReal).Name` | `pure` | Nil-safe. |
| `(*TunableReal).Value` | `pure` | Nil-safe. |
| `(*TunableReal).Bounds` | `pure` | Nil-safe. |
| `(*TunableReal).Fixed` | `pure` | Nil-safe. |
| `(*TunableReal).SetFixed` | `mutates` | Mutates receiver. |
| `(*TunableReal).SetValue` | `mutates` | Mutates receiver after bounds check. |
| `(*TunableReal).Sample` | `copy-out`, `returns-mutable` | Returns sampled copy. |
| `(*TunableReal).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled copy. |
| `NewTunableGain` | `returns-mutable` | MATLAB `tunableGain(name,Ny,Nu)`; creates zero, free entries. |
| `NewTunableGainFrom` | `copy-in`, `returns-mutable` | MATLAB `tunableGain(name,G)`; copies values from `G`. |
| `(*TunableGain).CurrentSystem` | `returns-mutable` | Builds gain model from current values. |
| `(*TunableGain).Sample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableGain).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableGain).FreeParameters` | `alias-risk` | Returns pointers to internal free parameters. |
| `(*TunableGain).SampleBlock` | `copy-out`, `returns-mutable` | Interface wrapper around `Sample`. |
| `NewTunablePID` | `returns-mutable` | MATLAB `tunablePID(name,type,Ts)`; creates the family's parameters. |
| `NewTunablePIDFrom` | `copy-in`, `returns-mutable` | MATLAB `tunablePID(name,sys)`; copies values from the PID. |
| `(*TunablePID).CurrentSystem` | `returns-mutable` | Builds controller model. |
| `(*TunablePID).Sample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunablePID).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunablePID).FreeParameters` | `alias-risk` | Returns pointers to internal free parameters. |
| `(*TunablePID).SampleBlock` | `copy-out`, `returns-mutable` | Interface wrapper around `Sample`. |
| `NewTunablePID2` | `returns-mutable` | MATLAB `tunablePID2(name,type,Ts)`. |
| `NewTunablePID2From` | `copy-in`, `returns-mutable` | MATLAB `tunablePID2(name,sys)`; copies values from the PID2. |
| `(*TunablePID2).CurrentSystem` | `returns-mutable` | Builds 2-DOF controller model. |
| `(*TunablePID2).Sample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunablePID2).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunablePID2).FreeParameters` | `alias-risk` | Returns pointers to internal free parameters. |
| `(*TunablePID2).SampleBlock` | `copy-out`, `returns-mutable` | Interface wrapper around `Sample`. |
| `NewTunableTF` | `returns-mutable` | MATLAB `tunableTF(name,Nz,Np,Ts)`; SISO. |
| `NewTunableTFFrom` | `copy-in`, `returns-mutable` | MATLAB `tunableTF(name,sys)`; copies normalized SISO coefficients. |
| `(*TunableTF).CurrentSystem` | `returns-mutable` | Builds transfer-function current model. |
| `(*TunableTF).Sample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableTF).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableTF).FreeParameters` | `alias-risk` | Returns pointers to internal free parameters. |
| `(*TunableTF).SampleBlock` | `copy-out`, `returns-mutable` | Interface wrapper around `Sample`. |
| `NewTunableSS` | `returns-mutable` | MATLAB `tunableSS(name,Nx,Ny,Nu,Ts,Astruct)`. |
| `NewTunableSSFrom` | `copy-in`, `returns-mutable` | MATLAB `tunableSS(name,sys,Astruct)`; copies a structured realization. |
| `(*TunableSS).CurrentSystem` | `returns-mutable` | Builds current state-space model. |
| `(*TunableSS).Sample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableSS).RandomSample` | `copy-out`, `returns-mutable` | Returns sampled block copy. |
| `(*TunableSS).FreeParameters` | `alias-risk` | Returns pointers to internal free parameters. |
| `(*TunableSS).SampleBlock` | `copy-out`, `returns-mutable` | Interface wrapper around `Sample`. |
| `GridTune` | `view-in`, `returns-mutable` | Returns a mutable result from a bounded Cartesian search. |
| `Systune` | `view-in`, `returns-mutable` | MATLAB `systune(CL0,SoftReqs,HardReqs,opts)` by grid search. |
| `Looptune` | `view-in`, `returns-mutable` | MATLAB `looptune(G0,C0,wc,Req...,opts)` by grid search. |

## Controller, PID, EKF, Physical, and Goal APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `NewPID` | `returns-mutable` | Returns public mutable `PID`. |
| `NewPIDStd` | `returns-mutable` | Returns public mutable `PID`. |
| `NewPID2` | `returns-mutable` | Returns public mutable `PID2`. |
| `(*PID).Copy` | `copy-out`, `returns-mutable` | Copies controller scalar fields. |
| `(*PID).Parallel` | `copy-out`, `returns-mutable` | Returns copy in parallel form. |
| `(*PID).Standard` | `copy-out`, `returns-mutable` | Returns copy in standard form. |
| `(*PID).Ti` | `pure` | Derived scalar. |
| `(*PID).Td` | `pure` | Derived scalar. |
| `(*PID).System` | `returns-mutable` | Builds controller model. |
| `(*PID2).Copy` | `copy-out`, `returns-mutable` | Copies 2-DOF controller scalar fields. |
| `(*PID2).System` | `returns-mutable` | Builds 2-DOF controller model. |
| `Pidtune` | `view-in`, `returns-mutable` | Uses typed `PidtuneType` selector. |
| `NewEKF` | `copy-in`, `returns-mutable` | Copies `x0`, `P0`, `Q`, and `R`; stores model callback functions by reference. |
| `(*EKF).Copy` | `copy-out`, `returns-mutable` | Deep-copies filter state, covariance, noise matrices, and work buffers. |
| `(*EKF).Predict` | `mutates`, `alias-risk` | Mutates filter state/covariance; stores callback-returned state vector. |
| `(*EKF).Update` | `mutates` | Mutates filter state/covariance and work buffers. |
| `(*EKF).Step` | `mutates` | Predict-then-update. |
| `NewPhysicalComponent` | `copy-in`, `returns-mutable` | Copies system and ports. |
| `AssemblePhysical` | `copy-in`, `returns-mutable` | Copies component systems before assembling. |
| `NewTrackingGoal` | `returns-mutable` | MATLAB `TuningGoal.Tracking(inputname,outputname,responsetime,dcerror,peakerror)`; value object. |
| `NewGainGoal` | `returns-mutable` | MATLAB `TuningGoal.Gain(inputname,outputname,gainvalue)`. |
| `NewWeightedGainGoal` | `copy-in`, `returns-mutable` | MATLAB `TuningGoal.WeightedGain(inputname,outputname,WL,WR)`; copies weights. |
| `NewRejectionGoal` | `copy-in`, `returns-mutable` | MATLAB `TuningGoal.Rejection(distloc,attfact)`; copies the profile. |
| `NewSensitivityGoal` | `copy-in`, `returns-mutable` | MATLAB `TuningGoal.Sensitivity(location,maxsens)`; copies the profile. |
| `NewLoopShapeGoal` | `copy-in`, `returns-mutable` | MATLAB `TuningGoal.LoopShape(location,loopgain,crosstol)`; copies the profile. |
| `NewLoopShapeGoalWc` | `returns-mutable` | MATLAB `TuningGoal.LoopShape(location,wc|wcrange)`. |
| `NewMarginsGoal` | `returns-mutable` | MATLAB `TuningGoal.Margins(location,gainmargin,phasemargin)`; disk-based. |
| `NewPolesGoal` | `returns-mutable` | MATLAB `TuningGoal.Poles(location,mindecay,mindamping,maxfreq)`. |
| `NewOvershootGoal` | `returns-mutable` | MATLAB `TuningGoal.Overshoot(inputname,outputname,maxpercent)`. |
| `TuningGoal.Name` / `TuningGoal.Type` | `pure` | Name defaults to the MATLAB class name. |
| `TuningGoal.WithName` / `TuningGoal.WithFocus` | `copy-out` | Return modified copies (MATLAB Name and Focus properties). |
| `TuningGoal.Evaluate` | `view-in`, `returns-mutable` | MATLAB evalGoal; normalized value f with Limit 1. |

## Polynomial and Utility APIs

| API | Classification | Audit note |
| --- | --- | --- |
| `Poly.Degree` | `pure` | Reads slice length. |
| `Poly.Eval` | `pure` | Evaluates polynomial. |
| `Poly.Mul` | `returns-mutable` | Returns new polynomial slice. |
| `Poly.Add` | `returns-mutable` | Returns new polynomial slice. |
| `Poly.IsMonic` | `pure` | Reads leading coefficient. |
| `Poly.Monic` | `returns-mutable` | Returns scaled polynomial copy. |
| `Poly.Scale` | `returns-mutable` | Returns scaled polynomial copy. |
| `Poly.MulTo` | `mutates` | Appends into caller-provided destination slice. |
| `Poly.AddTo` | `mutates` | Appends into caller-provided destination slice. |
| `Poly.ScaleTo` | `mutates` | Appends into caller-provided destination slice. |
| `Poly.Roots` | `returns-mutable` | Returns roots slice. |
| `Poly.Sub` | `returns-mutable` | Returns new polynomial slice. |
| `Poly.Derivative` | `returns-mutable` | Returns derivative slice. |
| `Poly.Equal` | `pure` | Approximate comparison. |
| `SumBlk` | `returns-mutable` | Builds summing-junction model. |

## Public Type Ownership Summary

| Type group | Types | Audit note |
| --- | --- | --- |
| Core mutable models | `System`, `LFTDelay`, `TransferFunc`, `ZPK`, `FRD`, `PID`, `PID2`, `EKF`, `PhysicalComponent` | Public fields or pointer fields allow mutation; only `System` has a general `Copy`. |
| Model containers | `ModelArray`, `GeneralizedModel`, `GeneralizedClosedLoop`, `TunableReal`, `TunableGain`, `TunablePID`, `TunableTF`, `TunableSS` | Mostly private fields with mutating methods; `FreeParameters` exposes parameter pointers. |
| Options/workspaces | `C2DOptions`, `TransferFuncOpts`, `StateSpaceOpts`, `FreqRespEstOpts`, `StepInfoOptions`, `SimulateOpts`, `RiccatiOpts`, `RiccatiWorkspace`, `LyapunovOpts`, `LyapunovWorkspace`, `PidtuneOptions`, `SystuneOptions`, `PassivityOptions`, `ReduceOpts`, `ModalTruncateOptions` | Options are caller-owned; workspaces and simulation buffers are mutable and should not be shared concurrently. |
| Result structs | `BalrealResult`, `CanonResult`, `GramResult`, `H2SynResult`, `HinfSynResult`, `LqgResult`, `LoopsensResult`, `MarginResult`, `AllMarginResult`, `DiskMarginResult`, `ReduceResult`, `ModalReductionResult`, `ModsepResult`, `PrescaleResult`, `PzmapResult`, `TimeResponse`, `StepInfoResult`, `RiccatiResult`, `RootLocusResult`, `Response`, `SsbalResult`, `StabsepResult`, `StaircaseForm`, `StateSpaceResult`, `TransferFuncResult`, `SystuneResult`, `TuningGoalResult`, `ZerosResult`, `ZPKResult`, `ERAResult`, `FRDPeakGainResult`, `FreqRespEstResult`, `ModelArrayFreqResponse`, `ModelArrayTimeResponse` | Results are mutable data containers; callers should treat them as owned outputs unless workspace-backed docs say otherwise. |
| Value and enum types | `StateProjection`, `BalredOptions`, `CanonForm`, `AbsorbScope`, `GramType`, `PhysicalPortKind`, `PIDForm`, `PidtuneType`, `C2DMethod`, `C2DDelayModeling`, `FreqRespEstMethod`, `ReduceMode`, `TuningGoalType`, `TuningGoal`, `TunableSSStructure`, `TunableBounds`, `AnalysisPointLocation`, `AnalysisPoint`, `PhysicalPort`, `PhysicalConnection`, `Connection`, `DampInfo`, `StepMetric`, `NonlinearModel`, `EKFModel`, `NumericBlock`, `TunableBlock`, `FRDResponseMapper`, `PIDOption`, `FeedbackChannels` | Mostly value types; callback and option function types may retain references through user code. |

## Release Gates

Before tagging a production release:

- CI must run `go fix ./...`, `go vet ./...`, and `go test -v -count=1 -race ./...`.
- CI must build a downstream module that imports `plantcontrol.org/v2/controlsys/v2`
  and resolves `plantcontrol.org/v1/gonum` without `replace` directives.
- Public API changes should update `README.md`, `doc.go`, examples, or this audit
  when behavior or ownership expectations change.
- Numerically sensitive changes should include external-reference tests using
  MATLAB or python-control fixtures when possible.

## Current Follow-Up Items

- Add required LAPACK routines to `plantcontrol.org/v1/gonum`; upstream Gonum
  does not accept AI-generated code.
