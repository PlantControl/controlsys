# Controlsys Codebase Interface Diagram

This diagram shows the current package interfaces. It is a codebase-level view, not a complete call graph: the public model interfaces are centered, and the internal seams show where recurring rules are localized.

## Public Interface Map

Rendered SVG: [codebase-public-interface-map.svg](codebase-public-interface-map.svg)

```mermaid
flowchart LR
    caller["External callers"]

    subgraph models["Model interfaces"]
        System["System<br/>fundamental state-space model<br/>A, B, C, D, optional E, delays, names"]
        transferFunc["TransferFunc<br/>polynomial-ratio model"]
        ZPK["ZPK<br/>zero-pole-gain model"]
        FRD["FRD<br/>frequency-response data model"]
        ModelArray["ModelArray<br/>compatible model grid"]
        generalizedModels["GeneralizedModel / GeneralizedClosedLoop<br/>analysis-point model interface"]
        tunableBlocks["TunableBlock<br/>tunable gain, PID, TF, or SS block"]
        freqResponseMatrix["FreqResponseMatrix<br/>sampled complex response"]
        timeResponse["TimeResponse<br/>sampled time-domain output"]
    end

    subgraph construction["Construction and identification"]
        constructStateSpace["New / NewGain / NewFromSlices<br/>NewWithDelay / Rss / Drss"]
        constructDescriptor["NewDescriptor / ToExplicit"]
        realizeTransferFunc["TransferFunc.StateSpace"]
        constructZPK["NewZPK / NewZPKMIMO"]
        constructFRD["NewFRD"]
        constructModelArray["NewModelArray / StackModelArrays<br/>ConcatModelArrays"]
        constructGeneralized["NewGeneralizedModel<br/>NewGeneralizedClosedLoop"]
        identifyERA["ERA<br/>Markov parameters to state-space model"]
        estimateFreqResponse["FreqRespEst<br/>sampled input/output to response estimate"]
        linearizeNonlinear["Linearize / NewEKF<br/>local nonlinear approximation"]
        assemblePhysical["NewPhysicalComponent / AssemblePhysical<br/>constraint-based descriptor assembly"]
    end

    subgraph interconnection["Interconnection interfaces"]
        seriesOp["Series"]
        parallelOp["Parallel"]
        feedbackOp["Feedback"]
        connectOps["Append / BlkDiag / Connect<br/>ConnectByName / LFT / SumBlk"]
        frdConnectOps["FRDSeries / FRDParallel<br/>FRDFeedback / FRDConcat"]
        delayOps["PadeDelay / ThiranDelay<br/>PullDelaysToLFT / AbsorbDelay<br/>SetDelayModel / GetDelayModel"]
    end

    subgraph conversion["Representation and domain conversion"]
        convertToTF["System.TransferFunction"]
        convertToZPK["System.ZPKModel"]
        convertToFRD["System.FRD"]
        convertToDiscrete["System.C2D / C2D<br/>C2D / FOH / Impulse / Matched / D2D"]
        convertToContinuous["System.D2C / System.D2C"]
        stateSpaceUtilities["StateTransform / EliminateStates<br/>FixedInputReduction / SelectByName / SelectByIndex"]
    end

    subgraph analysis["Analysis interfaces"]
        timeAnalysis["Step / Impulse / Initial / Lsim<br/>Simulate / StepInfo / StepInfoForSystem"]
        signalGeneration["GenSig<br/>test-signal generation"]
        freqAnalysis["FreqResponse / Bode / Nyquist / Nichols<br/>Margin / AllMargin / DiskMargin / Bandwidth / Sigma"]
        modelAnalysis["Poles / Zeros / Damp / IsStable<br/>DCGain / Pzmap"]
        energyAnalysis["Gram / HSV / Norm<br/>H2Norm / HinfNorm / Covar"]
        structureAnalysis["Ctrb / Obsv / IsStabilizable / IsDetectable"]
        loopAnalysis["Loopsens / RootLocus / FRDMargin"]
        passivityAnalysis["SampledPassive / FRDPassive<br/>Passive compatibility alias / SpectralFactor"]
    end

    subgraph transforms["Transformation and reduction"]
        realizationTransforms["SS2SS / Xperm / Canon"]
        balancingTransforms["Balreal / Balred / Modred / Sminreal<br/>Reduce / MinimalRealization / ModalTruncate"]
        decompositionTransforms["Stabsep / Modsep / Prescale / Ssbal"]
        algebraTransforms["Inv / Augstate"]
    end

    subgraph synthesis["Design and synthesis"]
        riccatiSolvers["Care / Dare / Lyap / DLyap"]
        controllerDesign["Lqr / Dlqr / Lqi / Lqrd<br/>Place / Acker"]
        observerDesign["Kalman / Kalmd / Lqe / Lqg<br/>Estim / Reg"]
        robustSynthesis["H2Syn / HinfSyn"]
        pidDesign["NewPID / NewPIDStd / Pidtune<br/>PID / PID2 / SmithPredictor"]
        fixedStructureTuning["GridTune / Systune / Looptune<br/>point-specific tuning goals"]
    end

    caller --> constructStateSpace
    caller --> constructDescriptor
    caller --> realizeTransferFunc
    caller --> constructZPK
    caller --> constructFRD
    caller --> constructModelArray
    caller --> constructGeneralized
    caller --> identifyERA
    caller --> estimateFreqResponse
    caller --> linearizeNonlinear
    caller --> assemblePhysical
    caller --> tunableBlocks

    constructStateSpace --> System
    constructDescriptor --> System
    realizeTransferFunc --> System
    constructZPK --> ZPK
    constructFRD --> FRD
    constructModelArray --> ModelArray
    constructGeneralized --> generalizedModels
    identifyERA --> System
    estimateFreqResponse --> freqResponseMatrix
    estimateFreqResponse --> FRD
    linearizeNonlinear --> System
    assemblePhysical --> System
    tunableBlocks --> generalizedModels

    transferFunc <--> ZPK
    transferFunc --> System
    ZPK --> transferFunc
    ZPK --> System
    System --> convertToTF --> transferFunc
    System --> convertToZPK --> ZPK
    System --> convertToFRD --> FRD
    FRD --> freqResponseMatrix
    ModelArray --> System
    ModelArray --> freqResponseMatrix
    ModelArray --> timeResponse
    generalizedModels --> System

    System --> seriesOp --> System
    System --> parallelOp --> System
    System --> feedbackOp --> System
    System --> connectOps --> System
    FRD --> frdConnectOps --> FRD
    System --> delayOps --> System

    System --> convertToDiscrete --> System
    System --> convertToContinuous --> System
    System --> stateSpaceUtilities --> System

    System --> timeAnalysis --> timeResponse
    caller --> signalGeneration
    System --> freqAnalysis --> freqResponseMatrix
    FRD --> freqAnalysis
    System --> modelAnalysis
    System --> energyAnalysis
    System --> structureAnalysis
    System --> loopAnalysis
    System --> passivityAnalysis
    FRD --> passivityAnalysis

    System --> realizationTransforms --> System
    System --> balancingTransforms --> System
    System --> decompositionTransforms --> System
    System --> algebraTransforms --> System

    System --> controllerDesign
    System --> observerDesign
    System --> robustSynthesis
    System --> pidDesign
    System --> generalizedModels
    generalizedModels --> fixedStructureTuning
    tunableBlocks --> fixedStructureTuning
    riccatiSolvers --> controllerDesign
    riccatiSolvers --> observerDesign
    riccatiSolvers --> robustSynthesis
    controllerDesign --> System
    observerDesign --> System
    robustSynthesis --> System
    pidDesign --> System
    fixedStructureTuning --> System
```

## Internal Seam Map

Rendered SVG: [codebase-internal-seam-map.svg](codebase-internal-seam-map.svg)

```mermaid
classDiagram
    class System {
        +Dims()
        +Validate()
        +Poles()
        +IsStable()
        +DCGain()
        +IsContinuous()
        +IsDiscrete()
        +Simulate()
        +FreqResponse()
        +TransferFunction()
    }

    class TransferFunc {
        +Dims()
        +Eval()
        +EvalMulti()
        +StateSpace()
        +ZPK()
    }

    class ZPK {
        +Dims()
        +Eval()
        +FreqResponse()
        +TransferFunction()
        +StateSpace()
    }

    class FRD {
        +Dims()
        +NumFrequencies()
        +At()
        +Abs()
        +SelectFrequencies()
        +MapResponse()
        +PeakGain()
        +FreqResponse()
        +Bode()
    }

    class ModelArray {
        +Shape()
        +Model()
        +SelectFlat()
        +ModelFlat()
        +FreqResponse()
        +Step()
    }

    class GeneralizedModel {
        +InsertAnalysisPoint()
        +AnalysisPoint()
        +HasAnalysisPoint()
        +CurrentSystem()
    }

    class GeneralizedClosedLoop {
        +InsertAnalysisPoint()
        +AnalysisPoint()
        +OpenLoop()
        +ClosedLoop()
        +Sensitivity()
        +ComplementarySensitivity()
    }

    class TunableBlock {
        +CurrentSystem()
        +FreeParameters()
        +SampleBlock()
    }

    class descriptorPolicy {
        +validate()
        +poles()
        +requireStandard()
        +requireRiccatiStandard()
    }

    class timeDomain {
        +validateSampleTime()
        +frequencyVariable()
        +ensureCompatible()
    }

    class delayTopology {
        +totalExternal()
        +decomposedExternal()
        +decomposableExternal()
    }

    class delayConversionPolicy {
        +applyDiscreteDelayFields()
        +applyContinuousDelayFields()
        +replaceDiscreteExternal()
        +replaceContinuousExternal()
    }

    class interconnectionTopology {
        +seriesDelayPlan()
        +parallelDelayPlan()
        +seriesRequiresLFT()
        +parallelRequiresLFT()
    }

    class realizationTransformPolicy {
        +requireStandard()
        +requireDelayFree()
        +result()
        +resultWithOriginalFeedthrough()
        +resultWithZeroFeedthrough()
    }

    class stateSpaceUtilitySeam {
        +NewDescriptor()
        +ToExplicit()
        +EliminateStates()
        +FixedInputReduction()
        +SelectByName()
        +SelectByIndex()
        +AugmentInternalDelayOutputs()
    }

    class frequencyEvaluator {
        +response()
        +eval()
    }

    class timeResponsePlanner {
        +auto()
        +lsim()
    }

    class simulationDispatcher {
        +run()
    }

    class sampledResponseLayout {
        +offset()
        +blockOffset()
    }

    class generalizedPlantPartition {
        +validateControllerChannels()
        +newController()
        +closedLoopPoles()
    }

    class generalizedTuningSeam {
        +numericBlockFromAny()
        +primaryAnalysisPointName()
        +GridTune()
        +evaluateTuningGoals()
    }

    class tuningGoalEvaluator {
        +evaluateSystem()
        +tuningGoalResponse()
        +frequencyGainRange()
        +maximumComplexSingularValue()
    }

    class passivitySeam {
        +SampledPassive()
        +Passive()
        +FRDPassive()
        +SpectralFactor()
    }

    class physicalAssemblyPlan {
        +bindPorts()
        +bindConnections()
        +assemble()
        +aggregateMatrices()
    }

    class modelArraySeam {
        +validateModelArrayCompatible()
        +validateModelArrayHeadersCompatible()
        +StackModelArrays()
        +ConcatModelArrays()
        +flatIndex()
    }

    class controllerObserverPolicy {
        +validateNoise()
        +regulator()
        +estimator()
    }

    class matrixEquationProblem {
        +riccatiProblem
        +lyapunovProblem
    }

    System --> descriptorPolicy : descriptor gate
    System --> timeDomain : time-domain rules
    System --> delayTopology : external delay decomposition
    System --> delayConversionPolicy : delay conversion
    System --> interconnectionTopology : interconnection planning
    System --> realizationTransformPolicy : realization assembly
    System --> stateSpaceUtilitySeam : state-space utilities
    System --> frequencyEvaluator : frequency response
    System --> timeResponsePlanner : time response
    System --> simulationDispatcher : sampled simulation
    System --> passivitySeam : passivity and spectral factor
    System --> physicalAssemblyPlan : physical constraint assembly
    System --> ModelArray : compatible model arrays

    TransferFunc --> System : realization
    ZPK --> TransferFunc : rational-channel conversion
    FRD --> sampledResponseLayout : sampled response access
    frequencyEvaluator --> sampledResponseLayout : flat response layout
    ModelArray --> modelArraySeam : array validation and analysis
    GeneralizedModel --> generalizedTuningSeam : analysis-point model wrapper
    GeneralizedClosedLoop --> generalizedTuningSeam : loop extraction
    TunableBlock --> generalizedTuningSeam : sampled controller blocks

    generalizedPlantPartition --> System : H2/Hinf controller synthesis
    generalizedTuningSeam --> tuningGoalEvaluator : fixed-structure tuning
    tuningGoalEvaluator --> System : evaluates point-specific loop responses
    controllerObserverPolicy --> System : regulator and estimator assembly
    matrixEquationProblem --> controllerObserverPolicy : Riccati and Lyapunov validation
```

## Interface Reading Guide

- `System` is the fundamental representation. Most public workflows either consume it, return it, or convert another model interface into it.
- `TransferFunc`, `ZPK`, `FRD`, `ModelArray`, and generalized/tunable model wrappers are alternate caller-facing interfaces. They preserve input/output names, sample time, and analysis-point metadata where the representation supports them.
- Interconnection routines concentrate compatibility checks, direct feedthrough handling, delay movement, and metadata propagation behind a small caller-facing interface.
- Delay behavior is intentionally split between topology and conversion seams: topology answers what delay structure exists; conversion decides whether it remains explicit, becomes a delay bank, or moves into LFT form.
- Analysis routines share sampled-response layouts so frequency-response data, Bode results, singular-value analysis, and frequency-response estimates use the same output/input/frequency indexing.
- Model-array, physical-assembly, and state-space utility seams support model-grid semantics, across/through descriptor constraints, and signal-selection workflows while keeping compatibility checks and metadata rules localized.
- Synthesis routines route generalized-plant, point-specific tuning, and controller/observer rules through policy modules before returning controller or closed-loop state-space models.
