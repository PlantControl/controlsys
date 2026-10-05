# Controlsys

Controlsys is a Go control-system toolbox for modeling, analyzing, transforming, and designing linear time-invariant models. Its language keeps precise distinctions between control systems, their mathematical models, model representations, time domains, delays, and analysis results.

## Language

**Control system**:
A practical implementation of control theory for shaping dynamic behavior.
_Avoid_: system before the control-system context is clear, plant when referring to an arbitrary model, controller when referring to the combined closed-loop system

**Model**:
A mathematical representation of a system.
_Avoid_: physical system, representation when the mathematical model itself is meant

**Linear time-invariant model**:
A model whose dynamics are linear and do not change over time.
_Avoid_: linear system when the modeling assumption and time invariance both matter

**Nonlinear model**:
A model whose dynamics or measurements are nonlinear and are handled here through local approximation techniques.
_Avoid_: linear time-invariant model

**State-space model**:
A representation of a system in terms of state, input, output, and feedthrough relationships.
_Avoid_: state model, SS object

**State**:
The internal variables that, with the input signal, determine future model behavior.
_Avoid_: output, mode

**State signal**:
The signal corresponding to one state of a state-space model.
_Avoid_: state trajectory when referring to a single state signal

**State trajectory**:
The time-domain values of a model's states over a simulation.
_Avoid_: state when referring to values over time

**Feedthrough**:
Direct input-to-output dependence that does not pass through dynamic states.
_Avoid_: gain when the direct path inside a dynamic model is meant

**Mode**:
A dynamic behavior component of a model, commonly associated with a pole when that behavior is visible in the input-output representation.
_Avoid_: state, pole when the distinction matters

**Natural frequency**:
The angular frequency associated with a pole or mode.
_Avoid_: damped frequency unless the oscillation frequency is specifically meant

**Damping ratio**:
A dimensionless measure of how quickly a mode decays relative to its natural frequency.
_Avoid_: damping coefficient

**Time constant**:
The characteristic decay time associated with a stable mode.
_Avoid_: sample time

**Controllability**:
The property that a model's states can be influenced through its input channels.
_Avoid_: reachability unless the discrete-time distinction is specifically intended

**Observability**:
The property that a model's states can be inferred from its output channels.
_Avoid_: visibility unless speaking informally

**Stabilizability**:
The property that every unstable mode can be influenced through input channels.
_Avoid_: controllability when stable uncontrollable modes are allowed

**Detectability**:
The property that every unstable mode can be inferred from output channels.
_Avoid_: observability when stable unobservable modes are allowed

**Gramian**:
An energy matrix that summarizes controllability or observability for a stable model.
_Avoid_: covariance unless the stochastic interpretation is specifically intended

**Controllability Gramian**:
A Gramian that measures how input channels transfer energy into states.
_Avoid_: input Gramian

**Observability Gramian**:
A Gramian that measures how states transfer energy to output channels.
_Avoid_: output Gramian

**Lyapunov equation**:
A matrix equation used to compute energy and stability-related quantities for linear models.
_Avoid_: algebraic Riccati equation

**Continuous Lyapunov equation**:
A Lyapunov equation for continuous-time models.
_Avoid_: discrete Lyapunov equation

**Discrete Lyapunov equation**:
A Lyapunov equation for discrete-time models.
_Avoid_: continuous Lyapunov equation

**Fundamental representation**:
The state-space representation used as the core form for computation and interconnection.
_Avoid_: canonical representation when no canonical-form transformation is meant

**Descriptor system**:
A state-space model whose state equation includes a descriptor matrix that may make the dynamics implicit.
_Avoid_: generalized system unless discussing generalized eigenvalues

**Random stable model**:
A generated stable state-space model used for tests, benchmarks, or examples.
_Avoid_: identified model

**Transfer function**:
A human-friendly polynomial-ratio representation of input-to-output dynamics.
_Avoid_: transfer, TF unless matching established API names

**Transfer-function model**:
A model specified through transfer functions rather than directly through state-space matrices.
_Avoid_: transfer function when the model container or representation is meant

**Proper model**:
A model whose transfer-function numerator degree does not exceed its denominator degree.
_Avoid_: stable model; properness is not stability

**Strictly proper model**:
A proper model whose transfer-function numerator degree is less than its denominator degree.
_Avoid_: proper model when the strict degree distinction matters

**Improper model**:
A model whose transfer-function numerator degree exceeds its denominator degree.
_Avoid_: unstable model; improperness is not stability

**Biproper model**:
A proper model whose transfer-function numerator and denominator degrees are equal.
_Avoid_: improper model

**Zero-pole-gain model**:
A representation of input-to-output dynamics by zeros, poles, and gain.
_Avoid_: root-gain model

**FOPDT model**:
A first-order-plus-dead-time transfer-function model.
_Avoid_: first-order model when the delay is part of the model

**SOPDT model**:
A second-order-plus-dead-time transfer-function model.
_Avoid_: SODT, second-order model when the delay is part of the model

**Frequency-response data**:
A sampled representation of a system response over frequency points.
_Avoid_: frequency plot, Bode data when the representation is not specifically a Bode result

**FRD model**:
A frequency-response data model represented by sampled complex response matrices over a frequency grid.
_Avoid_: state-space model

**Sampled complex response**:
A complex-valued input-output response matrix sample at one frequency.
_Avoid_: magnitude or phase alone

**FRD interconnection**:
An interconnection computed directly on compatible frequency-response data models.
_Avoid_: state-space interconnection

**Frequency-response estimate**:
A frequency-response matrix estimated from sampled input and output signals.
_Avoid_: frequency-response data when the source from sampled signals matters

**Welch estimate**:
A frequency-response estimate computed by averaging windowed, overlapping FFT segments.
_Avoid_: direct FFT estimate

**Direct FFT estimate**:
A frequency-response estimate computed from one input FFT and one output FFT without segment averaging.
_Avoid_: Welch estimate

**H1 estimator**:
A frequency-response estimator based on output-input cross spectrum divided by input auto spectrum.
_Avoid_: H2 estimator

**H2 estimator**:
A frequency-response estimator based on output auto spectrum divided by input-output cross spectrum.
_Avoid_: H1 estimator, H2 norm

**Coherence**:
A frequency-domain quality metric for how strongly an input-output pair is linearly related at a frequency.
_Avoid_: correlation unless the frequency-domain estimate is meant

**FFT length**:
The number of time samples used in each FFT segment for frequency-response estimation.
_Avoid_: number of frequency points when referring to the time-domain segment length

**Window function**:
A weighting function applied to each time segment before FFT-based estimation.
_Avoid_: frequency window unless the frequency-domain operation is meant

**Overlap**:
The number of shared samples between adjacent FFT segments in a Welch estimate.
_Avoid_: sample time

**Frequency-response matrix**:
The MIMO input-output response matrix evaluated at one frequency.
_Avoid_: frequency-response data when referring to one frequency value

**DC gain**:
The zero-frequency transfer-matrix limit for each input-output channel of a model.
_Avoid_: steady-state value when model stability or step-response convergence is meant

**Static-gain model**:
A model with direct input-to-output gain and no dynamic state.
_Avoid_: gain system unless matching established API names, zero-state system

**Autonomous model**:
A state-space model with states but no inputs, driven only by its initial state; a state-space model may likewise have no outputs. Its empty input or output blocks are stored as empty matrices and its dimensions come from the state matrices. A static-gain model with no inputs or no outputs but not both cannot be stored, since its empty gain carries no dimensions. An operation whose result would be such a gain, such as removing every state of a model with no inputs, returns ErrDimensionMismatch instead.
_Avoid_: unforced system, zero-input model

**Continuous-time model**:
A model whose dynamics evolve over continuous time.
_Avoid_: analog system, continuous-time system before the modeling context is clear

**Discrete-time model**:
A model whose dynamics evolve at a fixed sample time.
_Avoid_: digital system, discrete-time system before the modeling context is clear

**Sample time**:
The time interval between updates in a discrete-time model.
_Avoid_: timestep when referring to the system property

**Nyquist frequency**:
The highest frequency represented without aliasing for a discrete-time model.
_Avoid_: Nyquist plot

**Pole**:
A value that characterizes a mode of the system dynamics.
_Avoid_: eigenvalue unless specifically discussing the state matrix or a matrix pencil

**Right-half-plane pole**:
A continuous-time pole with positive real part.
_Avoid_: unstable pole when the continuous-time location matters

**Stable model**:
A model whose poles all lie strictly inside the stability region for its time domain.
_Avoid_: stable system before the modeling context is clear

**Stability boundary**:
The pole-location boundary that separates stable and unstable model behavior for a time domain.
_Avoid_: stability margin when referring only to pole location

**Unstable model**:
A model with at least one pole outside the stability region or on its stability boundary.
_Avoid_: unstable system before the modeling context is clear

**Marginally stable model**:
A model with at least one pole on the stability boundary and no poles outside it.
_Avoid_: stable model; marginal models are not treated as stable by this toolbox

**Zero**:
A value where an input-output channel loses transmission.
_Avoid_: root when the input-output meaning matters

**Right-half-plane zero**:
A continuous-time zero with positive real part.
_Avoid_: nonminimum-phase zero unless the behavioral implication is being discussed

**Nonminimum-phase zero**:
A zero outside the stability region for the model's time domain.
_Avoid_: right-half-plane zero unless specifically discussing continuous-time location

**Minimum-phase model**:
A stable model whose zeros also lie inside the stability region for its time domain.
_Avoid_: stable model when zero locations matter

**Channel gain**:
The scalar multiplier for an input-output channel in a transfer-function or zero-pole-gain representation.
_Avoid_: static-gain model when referring only to a channel multiplier

**DC gain**:
The steady-state gain of a model at zero frequency.
_Avoid_: channel gain when the steady-state input-output gain is meant

**Bandwidth**:
The frequency where model response magnitude first drops by a specified amount from DC gain.
_Avoid_: frequency range unless referring to this response metric

**Input channel**:
One independent signal entering a system.
_Avoid_: input when channel cardinality matters

**Output channel**:
One independent signal leaving a system.
_Avoid_: output when channel cardinality matters

**Input signal**:
The time-domain values applied through an input channel.
_Avoid_: input channel when referring to signal values

**Output signal**:
The time-domain values observed from an output channel.
_Avoid_: output channel when referring to signal values

**Test signal**:
A generated input signal used for simulation, identification, or validation.
_Avoid_: measurement signal

**Step signal**:
A test signal that jumps to a constant value and remains there.
_Avoid_: step response

**Sine signal**:
A sinusoidal test signal over a specified period.
_Avoid_: frequency response

**Square signal**:
A periodic test signal that switches between positive and negative levels.
_Avoid_: pulse signal

**Pulse signal**:
A test signal with a single nonzero sample used as a discrete impulse-like input.
_Avoid_: impulse response

**Signal name**:
A human-readable identifier for an input, output, or state signal.
_Avoid_: variable name when referring to model channel naming

**Reference signal**:
The desired target signal for a controlled output.
_Avoid_: setpoint unless the surrounding context uses process-control language

**Measurement signal**:
The observed plant output used by a controller or observer.
_Avoid_: output signal when the measurement role matters

**Control signal**:
The controller output applied to a plant input.
_Avoid_: manipulated variable unless the surrounding context uses process-control language

**Error signal**:
The difference between a reference signal and measurement signal in a feedback loop.
_Avoid_: residual unless discussing estimation or identification

**MIMO model**:
A model with one or more input channels and one or more output channels.
_Avoid_: matrix system, MIMO system before the modeling context is clear

**SISO model**:
A model with exactly one input channel and one output channel.
_Avoid_: scalar system when discussing channel count, SISO system before the modeling context is clear

**SIMO model**:
A model with one input channel and multiple output channels.
_Avoid_: SIMO system before the modeling context is clear

**MISO model**:
A model with multiple input channels and one output channel.
_Avoid_: MISO system before the modeling context is clear

**Transport delay**:
A delay that shifts an input-output response in time without changing the undelayed dynamics.
_Avoid_: dead time except in conventional FOPDT and SOPDT model names

**Exact delay**:
A transport delay represented directly as delay metadata rather than approximated by finite-dimensional dynamics.
_Avoid_: delay approximation

**External delay**:
A transport delay attached to external input or output channels of a model.
_Avoid_: internal delay

**Input delay**:
A transport delay attached to an input channel.
_Avoid_: actuator delay unless the physical source is known

**Output delay**:
A transport delay attached to an output channel.
_Avoid_: sensor delay unless the physical source is known

**Internal delay**:
A delay represented inside an interconnection rather than only at external input or output channels.
_Avoid_: I/O delay

**Total delay**:
The combined input, output, and input-output delay for each input-output path.
_Avoid_: input delay when all path delays are included

**Delay model**:
An internal-delay representation consisting of an augmented model and delay times.
_Avoid_: delay approximation

**Zero-delay approximation**:
A transformation that replaces delay blocks with zero-delay behavior.
_Avoid_: exact delay

**Linear fractional transformation**:
An interconnection form where a main model is closed around a lower block through selected internal inputs and outputs.
_Avoid_: feedback interconnection when the partitioned block structure matters

**Uncertainty block**:
The lower block in a linear fractional transformation, representing dynamics or variation connected to the main model.
_Avoid_: plant or controller unless the block has that design role

**Delay block**:
An uncertainty block whose behavior is a transport delay.
_Avoid_: delay approximation when the delay is exact

**Nominal model**:
The baseline model used for analysis or design before accounting for uncertainty.
_Avoid_: true model

**Model uncertainty**:
The difference between the nominal model and the real behavior being controlled or estimated.
_Avoid_: modeling error when the uncertainty structure matters

**Uncertain model**:
A model family described by a nominal model together with model uncertainty.
_Avoid_: nominal model

**Additive uncertainty**:
Model uncertainty represented as an added perturbation to a nominal model.
_Avoid_: multiplicative uncertainty

**Multiplicative uncertainty**:
Model uncertainty represented as a perturbation that scales or factors around a nominal model.
_Avoid_: additive uncertainty

**Parametric uncertainty**:
Model uncertainty represented through uncertain physical or fitted parameter values.
_Avoid_: unstructured uncertainty when named parameters are the source

**Absorbed delay**:
A delay converted from metadata into additional state-space dynamics.
_Avoid_: exact delay when the delay has become states

**Fractional delay**:
A discrete-time delay with a non-integer number of sample intervals.
_Avoid_: integer delay

**Integer delay**:
A discrete-time delay with a whole number of sample intervals.
_Avoid_: fractional delay

**Augmented model**:
A model formed by adding states or channels to represent additional dynamics or constraints.
_Avoid_: original model when the state or channel dimension has changed

**Delay bank**:
A block-diagonal collection of delay models applied across multiple channels.
_Avoid_: single delay model

**Delay approximation**:
A finite-dimensional model used to approximate transport-delay behavior when an exact delay cannot be kept in that representation or operation.
_Avoid_: transport delay when the approximation itself is meant

**Pade approximation**:
A continuous-time rational delay approximation used to replace transport delays with finite-dimensional dynamics.
_Avoid_: exact delay

**Thiran allpass delay**:
A discrete-time allpass delay approximation used for fractional sample delays.
_Avoid_: Pade approximation

**Interconnection**:
A composition of systems by series, parallel, feedback, block-diagonal, or signal-routing relationships.
_Avoid_: wiring when referring to the mathematical result

**Series interconnection**:
An interconnection where the output of one model feeds the input of another model.
_Avoid_: cascade unless that term is already used in the surrounding discussion

**Parallel interconnection**:
An interconnection where models share compatible inputs and their outputs are added.
_Avoid_: parallel PID form

**Block-diagonal interconnection**:
An interconnection that combines models without coupling their input or output channels.
_Avoid_: append unless matching an established API name

**Signal routing**:
An interconnection that explicitly maps or sums named or indexed signals between model channels.
_Avoid_: wiring when referring to the mathematical interconnection

**Named interconnection**:
A signal routing workflow that connects models by signal names.
_Avoid_: indexed interconnection

**Indexed interconnection**:
A signal routing workflow that connects models by numeric channel indices.
_Avoid_: named interconnection

**Summing junction**:
A signal-combination point where input signals are added or subtracted.
_Avoid_: addition when the block-diagram role matters

**Sum block**:
A static-gain model generated from a summing-junction expression.
_Avoid_: summing junction when referring to the generated model

**Feedback interconnection**:
An interconnection that closes a feedback loop between compatible model channels.
_Avoid_: feedback loop when referring specifically to the operation that creates the model

**Safe feedback**:
A feedback interconnection workflow that handles supported delay cases before closing the loop.
_Avoid_: feedback interconnection when the delay-handling workflow is specifically meant

**Algebraic loop**:
An instantaneous closed-loop dependence caused by direct feedthrough around a feedback path.
_Avoid_: feedback loop when the issue is specifically instantaneous dependence

**Well-posed feedback**:
A feedback interconnection whose instantaneous equations have a unique solution.
_Avoid_: no algebraic loop when a direct loop exists but is still solvable

**Feedback loop**:
An interconnection where output information is routed back into an input path.
_Avoid_: closed loop when the loop has not yet been formed

**Negative feedback**:
A feedback loop where the returned signal is subtracted at the summing junction.
_Avoid_: feedback when the sign matters

**Positive feedback**:
A feedback loop where the returned signal is added at the summing junction.
_Avoid_: feedback when the sign matters

**Open-loop model**:
A model evaluated without closing a feedback loop around it.
_Avoid_: loop transfer when the feedback-loop path is specifically meant

**Closed-loop model**:
A model produced by closing a feedback loop.
_Avoid_: feedback system when the result, not the operation, is meant

**Loop transfer**:
The open-loop transfer around a feedback loop.
_Avoid_: closed-loop model

**Controller**:
A system designed to shape the behavior of another system through an interconnection.
_Avoid_: compensator unless the design context uses that term

**Smith predictor**:
A controller structure for time-delay plants that uses a delay-free model path and a delayed model path.
_Avoid_: delay approximation

**Regulator**:
A controller intended to hold a target behavior or reject disturbances.
_Avoid_: PID controller, linear-quadratic regulator, or observer-based controller when the specific structure matters

**Compensator**:
A controller designed to alter model dynamics or frequency-response behavior.
_Avoid_: controller unless the compensating role matters

**Lead compensator**:
A compensator that adds phase lead over a frequency range.
_Avoid_: special model type; represent it as a model

**Lag compensator**:
A compensator that adds low-frequency gain or phase lag over a frequency range.
_Avoid_: special model type; represent it as a model

**State feedback**:
A control structure where the control signal is computed from model states.
_Avoid_: output feedback when only measured outputs are available

**Pole placement**:
A controller-design method that chooses feedback gains to assign closed-loop pole locations.
_Avoid_: root locus when the poles are directly assigned rather than swept over gain

**Ackermann pole placement**:
A SISO pole-placement method based on Ackermann's formula.
_Avoid_: pole placement when the method does not matter

**Output feedback**:
A control structure where the control signal is computed from measured outputs.
_Avoid_: state feedback when states are not directly available

**Observer-based controller**:
A controller that uses an observer state estimate together with state-feedback gains.
_Avoid_: state feedback when estimator dynamics are included

**Plant**:
A system described in the context of controller design.
_Avoid_: process unless the physical domain is specifically process control

**Plant model**:
A model of a plant used for controller design or analysis.
_Avoid_: plant when referring specifically to the mathematical representation

**Controller model**:
A model of a controller used for analysis, synthesis, or interconnection.
_Avoid_: controller when referring specifically to the mathematical representation

**Observer**:
A system that estimates states from measured signals.
_Avoid_: estimator when the distinction from general statistical estimation matters

**Estimator**:
A design that infers states, parameters, disturbances, or other quantities from available signals.
_Avoid_: observer when specifically estimating states of a dynamic model

**Noise covariance**:
A matrix describing assumed noise intensity and correlation for estimation or filtering.
_Avoid_: cost matrix

**Output covariance**:
The steady-state covariance of output signals for a stable model driven by input noise.
_Avoid_: noise covariance when referring to the resulting output statistic

**Process noise**:
Unmodeled disturbance entering model dynamics in an estimation problem.
_Avoid_: measurement noise

**Measurement noise**:
Noise corrupting measured output signals in an estimation problem.
_Avoid_: process noise

**Linear-quadratic estimator**:
An observer gain design based on model and noise covariance assumptions.
_Avoid_: linear-quadratic regulator

**Kalman filter**:
An observer designed from a model and noise covariance assumptions.
_Avoid_: filter when the observer role matters

**Extended Kalman filter**:
A nonlinear estimator that linearizes state-transition and measurement functions at each step.
_Avoid_: Kalman filter when nonlinear model functions and Jacobians are required

**Jacobian**:
A linearization matrix of a nonlinear function around an operating point.
_Avoid_: state-space matrix unless referring to the resulting linear model

**Linearization**:
A local linear time-invariant model derived from a nonlinear model around an operating point.
_Avoid_: discretization

**Operating point**:
The state and input values around which a nonlinear model is linearized.
_Avoid_: initial condition unless it is only used to start a simulation

**Linear-quadratic-Gaussian controller**:
An observer-based controller combining linear-quadratic regulation with Kalman filtering.
_Avoid_: linear-quadratic regulator when estimator dynamics are included

**H2 controller**:
A synthesized controller designed to optimize an H2 performance criterion.
_Avoid_: H2 norm when referring to the controller rather than the performance metric

**H2 synthesis**:
A controller-synthesis method that computes a controller from a continuous-time generalized plant and an H2 performance criterion.
_Avoid_: H2 norm when referring to controller synthesis

**Synthesis Riccati solution**:
A Riccati-equation solution matrix returned by a controller-synthesis method.
_Avoid_: cost matrix

**Controller order**:
The number of dynamic states in a controller model.
_Avoid_: plant order

**H-infinity controller**:
A synthesized controller designed to satisfy or optimize an H-infinity performance criterion.
_Avoid_: H-infinity norm when referring to the controller rather than the performance metric

**H-infinity synthesis**:
A controller-synthesis method that computes a controller from a continuous-time generalized plant and an H-infinity performance bound.
_Avoid_: H-infinity norm when referring to controller synthesis

**Gamma**:
The H-infinity performance bound used or achieved during H-infinity synthesis.
_Avoid_: gain margin

**Closed-loop poles**:
The poles of the model formed after closing a controller around a plant or generalized plant.
_Avoid_: open-loop poles

**Generalized plant**:
A partitioned plant model used for synthesis, with disturbance, control, performance, and measurement channels.
_Avoid_: plant when the synthesis partition matters

**Performance channel**:
A synthesis output channel whose response is being minimized or bounded.
_Avoid_: output channel when the synthesis role matters

**Measurement channel**:
A synthesis output channel available to the controller.
_Avoid_: measurement signal when referring to model channel partitioning

**Disturbance channel**:
A generalized-plant input channel representing exogenous signals entering the synthesis problem.
_Avoid_: input channel when the synthesis role matters

**Control channel**:
A generalized-plant input channel driven by the synthesized controller.
_Avoid_: input channel when the controller-output role matters

**Performance output**:
A generalized-plant output whose response defines the synthesis objective.
_Avoid_: output signal when the synthesis role matters

**Measurement output**:
A generalized-plant output measured by the synthesized controller.
_Avoid_: output signal when the controller-input role matters

**Linear-quadratic regulator**:
A state-feedback controller designed from quadratic state and input costs.
_Avoid_: optimal controller unless the specific design method is important

**Discrete linear-quadratic regulator**:
A linear-quadratic regulator for a discrete-time model.
_Avoid_: linear-quadratic regulator when the time domain matters

**Linear-quadratic integral regulator**:
A linear-quadratic regulator with integral action added through augmented states.
_Avoid_: linear-quadratic regulator when integral action matters

**Sampled-data linear-quadratic regulator**:
A discrete linear-quadratic regulator designed from a continuous-time model after discretization.
_Avoid_: API-specific abbreviations in glossary prose

**Cost matrix**:
A weighting matrix that defines state, output, or control-signal penalties in an optimal-control problem.
_Avoid_: noise covariance

**Algebraic Riccati equation**:
A matrix equation whose solution is used to compute optimal feedback or estimator gains.
_Avoid_: Lyapunov equation

**Continuous algebraic Riccati equation**:
An algebraic Riccati equation for continuous-time models.
_Avoid_: discrete algebraic Riccati equation

**Discrete algebraic Riccati equation**:
An algebraic Riccati equation for discrete-time models.
_Avoid_: continuous algebraic Riccati equation

**Cross term**:
A state-input weighting term in a generalized Riccati equation.
_Avoid_: feedthrough

**PID controller**:
A controller formed from proportional, integral, and derivative actions.
_Avoid_: regulator when the PID structure matters

**Parallel PID form**:
A PID parameterization using independent proportional, integral, and derivative gains.
_Avoid_: parallel interconnection

**Standard PID form**:
A PID parameterization using proportional gain, integral time, and derivative time.
_Avoid_: ISA form unless the surrounding discussion uses that convention

**Setpoint weight**:
A PID coefficient that controls how strongly the reference signal enters proportional or derivative action.
_Avoid_: reference gain when discussing two-degree-of-freedom PID structure

**Derivative filter**:
A first-order filter applied to PID derivative action.
_Avoid_: filter when the derivative-action role matters

**Two-degree-of-freedom PID controller**:
A PID controller that weights setpoint and measurement paths separately.
_Avoid_: PID when the setpoint weighting is relevant

**Simulation response**:
A time-domain output produced by applying an input signal to a system.
_Avoid_: experiment result when referring to a model output

**Simulation time vector**:
The ordered time values associated with a time-domain response.
_Avoid_: sample time when referring to all response times

**Initial condition**:
The state vector used to start a simulation or initial-response calculation.
_Avoid_: operating point unless used for linearization

**Final state**:
The state vector returned after the last simulation step.
_Avoid_: state trajectory when only the terminal state is meant

**Simulation workspace**:
A reusable buffer supplied to simulation to reduce repeated allocation.
_Avoid_: model state

**Step response**:
A simulation response to a step input.
_Avoid_: step test unless referring to a physical experiment

**Impulse response**:
A simulation response to an impulse input.
_Avoid_: shock response

**Initial response**:
A time-domain response produced from an initial condition with zero input.
_Avoid_: step response

**Free response**:
The part of a time-domain response caused by the initial condition rather than forced input.
_Avoid_: forced response

**Forced response**:
A time-domain response produced by an applied input signal.
_Avoid_: free response

**Frequency response**:
The input-output response of a system evaluated over frequency.
_Avoid_: spectrum

**Frequency grid**:
The ordered frequency points where a frequency response is evaluated.
_Avoid_: frequency-response data when referring only to the evaluation points

**Angular frequency**:
Frequency measured in radians per unit time.
_Avoid_: hertz unless values are explicitly cycles per unit time

**Realization**:
A state-space model that implements the same input-output behavior as another representation.
_Avoid_: representation when the state-space implementation is specifically meant

**Bode result**:
A frequency-response result expressed as magnitude and phase over frequency.
_Avoid_: frequency-response data when magnitude and phase are specifically meant

**Bode plot**:
A visualization of Bode magnitude and phase over frequency.
_Avoid_: Bode result when referring to computed data rather than a visualization

**Nyquist result**:
A SISO frequency-domain result containing the Nyquist contour and encirclement counts.
_Avoid_: Nyquist plot when referring to computed data rather than a visualization

**Nyquist plot**:
A visualization of a Nyquist contour in the complex plane.
_Avoid_: Nyquist result when referring to computed data rather than a visualization

**Nichols result**:
A frequency-domain result expressed as open-loop phase versus magnitude.
_Avoid_: Nichols plot when referring to computed data rather than a visualization

**Nichols plot**:
A visualization of open-loop phase versus magnitude.
_Avoid_: Nichols result when referring to computed data rather than a visualization

**Sigma result**:
A frequency-domain result containing singular values of the frequency-response matrix over frequency.
_Avoid_: sigma plot when referring to computed data rather than a visualization

**Sigma plot**:
A visualization of singular values over frequency.
_Avoid_: sigma result when referring to computed data rather than a visualization

**Root locus result**:
A SISO result showing how closed-loop poles move as loop gain changes.
_Avoid_: root locus plot when referring to computed data rather than a visualization

**Root locus plot**:
A visualization of root locus branches over loop gain.
_Avoid_: root locus result when referring to computed data rather than a visualization

**Pole-zero map result**:
A result containing the poles and zeros of a model.
_Avoid_: pole-zero map when referring to computed data rather than a visualization

**Pole-zero map**:
A visualization of model poles and zeros.
_Avoid_: pole-zero map result when referring to computed data rather than a visualization

**Root locus branch**:
The path followed by one closed-loop pole as loop gain changes.
_Avoid_: pole trajectory unless the root-locus context is unclear

**Breakaway point**:
A root-locus point where branches leave or enter the real axis.
_Avoid_: branch point unless a different mathematical branch point is meant

**Root-locus asymptote**:
An asymptotic direction followed by root-locus branches that go to infinity.
_Avoid_: asymptote when the root-locus context is unclear

**Departure angle**:
The root-locus branch angle leaving an open-loop pole.
_Avoid_: phase angle when the root-locus role matters

**Arrival angle**:
The root-locus branch angle approaching an open-loop zero.
_Avoid_: phase angle when the root-locus role matters

**Stability margin**:
A frequency-domain measure of how close a feedback loop is to instability.
_Avoid_: safety margin unless discussing non-control safety

**Gain margin**:
A classical SISO stability margin measuring allowable loop-gain change at a phase crossover.
_Avoid_: disk gain margin when referring to the classical phase-crossover margin

**Phase margin**:
A classical SISO stability margin measuring allowable phase lag at a gain crossover.
_Avoid_: disk phase margin when referring to the classical gain-crossover margin

**Gain crossover**:
A frequency where the loop response magnitude is unity.
_Avoid_: phase crossover

**Phase crossover**:
A frequency where the loop response phase is -180 degrees.
_Avoid_: gain crossover

**Disk margin**:
A SISO advanced robustness margin αmax = 1/‖S + (σ−1)/2‖∞ for skew σ (default 0, balanced, as MATLAB diskmargin) that gives simultaneous gain and phase variation bounds; σ = 1 is the peak-sensitivity margin 1/‖S‖∞.
_Avoid_: classical gain margin, classical phase margin

**Sensitivity function**:
The closed-loop response S = 1/(1+L) for a loop transfer L.
_Avoid_: sensitivity when not referring to the closed-loop transfer function

**Output sensitivity**:
The output-side sensitivity model for a plant-controller loop.
_Avoid_: input sensitivity when multiplication order matters

**Input sensitivity**:
The input-side sensitivity model for a plant-controller loop.
_Avoid_: output sensitivity when multiplication order matters

**Complementary sensitivity**:
The closed-loop transfer from loop input to controlled output complementing the sensitivity function.
_Avoid_: sensitivity function

**Loop sensitivity result**:
A result containing output sensitivity, output complementary sensitivity, input sensitivity, and input complementary sensitivity models.
_Avoid_: stability margin

**Peak sensitivity**:
The largest frequency-domain magnitude of the sensitivity function.
_Avoid_: sensitivity when the peak norm is specifically meant

**System norm**:
A scalar measure of model size or amplification behavior.
_Avoid_: matrix norm unless measuring a matrix directly

**H2 norm**:
A system norm measuring energy amplification for a stable model.
_Avoid_: H2 controller when referring to the performance metric

**H-infinity norm**:
A system norm measuring peak frequency-domain gain for a stable model.
_Avoid_: H-infinity controller when referring to the performance metric

**Minimal realization**:
A controllable and observable state-space model with the same input-output behavior as a transfer function and the minimum number of states.
_Avoid_: reduced model when exact minimality, not approximation, is meant

**Model reduction**:
A transformation that lowers model order while preserving selected behavior.
_Avoid_: simplification when preserving behavior is the point

**Balanced realization**:
A realization whose stable part has equal diagonal controllability and observability Gramians while unstable modes are preserved.
_Avoid_: normalized realization, reduced model

**Balanced truncation**:
A model-reduction method that removes states with small Hankel singular values from a balanced realization.
_Avoid_: minimal realization

**Singular perturbation reduction**:
A model-reduction method that removes selected states while matching low-frequency gain behavior.
_Avoid_: balanced truncation when the reduction method matters

**Stable-unstable decomposition**:
A decomposition that separates stable modes from unstable modes.
_Avoid_: balanced realization

**Modal decomposition**:
A decomposition that separates modes by a cutoff.
_Avoid_: canonical form

**State-space balancing**:
A numerical scaling transformation that balances state magnitudes for conditioning.
_Avoid_: balanced realization when Gramian balancing is meant

**Prescaling**:
A response-preserving diagonal state-coordinate scaling (MATLAB `prescale`) applied before computation; input and output scales are only suggested, not applied.
_Avoid_: model reduction

**Similarity transform**:
A state-coordinate transformation that preserves input-output behavior.
_Avoid_: model reduction

**State permutation**:
A similarity transform that reorders state coordinates.
_Avoid_: signal routing

**Controllability matrix**:
A matrix whose rank indicates state controllability from the input matrix.
_Avoid_: controllability Gramian

**Observability matrix**:
A matrix whose rank indicates state observability from the output matrix.
_Avoid_: observability Gramian

**Staircase decomposition**:
A rank-revealing decomposition used for controllability, observability, zeros, or reduction workflows.
_Avoid_: canonical form

**Hankel singular values**:
The diagonal energy values of a balanced realization that indicate how strongly states contribute to input-output behavior, with infinite values marking unstable modes.
_Avoid_: singular values when the balanced-realization energy meaning matters

**Eigensystem realization algorithm**:
A system-identification method that computes a discrete-time state-space model from Markov parameters.
_Avoid_: frequency-response estimation

**Markov parameter**:
An impulse-response matrix sample used by the eigensystem realization algorithm.
_Avoid_: frequency-response data

**Identified model**:
A model estimated from measured, simulated, or sampled response data.
_Avoid_: nominal model unless the model is being used as the baseline for uncertainty analysis

**Canonical form**:
A state-space realization with a chosen structured coordinate form.
_Avoid_: normal form unless the specific mathematical convention is meant

**Discretization**:
A transformation from a continuous-time model to a discrete-time model.
_Avoid_: sampling when referring to the full model transformation

**Discretization method**:
The numerical assumption or transform used to convert a continuous-time model into a discrete-time model.
_Avoid_: sample time when referring to the conversion algorithm

**Zero-order hold**:
A discretization method that assumes each input signal is held constant over one sample interval.
_Avoid_: hold when the specific method matters

**First-order hold**:
A discretization method that assumes each input signal varies linearly over one sample interval.
_Avoid_: linear interpolation when referring to model conversion

**Tustin method**:
A bilinear discretization method based on a trapezoidal integration transform.
_Avoid_: zero-order hold

**Impulse-invariant method**:
A discretization method that preserves samples of the continuous-time impulse response.
_Avoid_: impulse response when referring to the conversion method

**Matched pole-zero method**:
A discretization method that maps continuous-time poles and zeros into the discrete-time domain.
_Avoid_: pole placement

**Discrete-to-continuous conversion**:
A transformation from a discrete-time model to a continuous-time model.
_Avoid_: continuous reconstruction, undiscretization unless matching an established API name

**Discrete-to-discrete conversion**:
A transformation from one discrete-time sample time to another discrete-time sample time.
_Avoid_: discretization when the source model is already discrete-time

**System inverse**:
A model whose input-output behavior inverts another square model when the direct feedthrough structure permits it.
_Avoid_: reciprocal when referring to MIMO models

## Relationships

- A **model** represents a **system**.
- A **model** can be represented as a **state-space model**, **descriptor system**, **transfer function**, **zero-pole-gain model**, or **frequency-response data**.
- A **linear time-invariant model** can be represented as a **state-space model**, **transfer function**, **zero-pole-gain model**, **descriptor system**, or **frequency-response data**.
- A **nonlinear model** is handled through **linearization** or **extended Kalman filter** workflows in this toolbox.
- A **state-space model** includes **states**, input channels, output channels, and feedthrough.
- A **state signal** corresponds to one **state**.
- A **state trajectory** records **state** values over time.
- **Feedthrough** is the direct input-to-output part of a **state-space model**.
- A **mode** is a dynamic behavior component of a **model**.
- **Natural frequency**, **damping ratio**, and **time constant** describe modal behavior.
- **Controllability** describes whether states can be influenced from **input channels**.
- **Observability** describes whether states can be inferred from **output channels**.
- **Stabilizability** is weaker than **controllability**.
- **Detectability** is weaker than **observability**.
- A **Gramian** is either a **controllability Gramian** or an **observability Gramian**.
- A **controllability Gramian** describes input-to-state energy transfer.
- An **observability Gramian** describes state-to-output energy transfer.
- **Gramians** are computed from **Lyapunov equations**.
- A **continuous Lyapunov equation** applies to **continuous-time models**.
- A **discrete Lyapunov equation** applies to **discrete-time models**.
- A **state-space model** is the **fundamental representation**.
- A **transfer-function model** is a human-friendly specification of a **model**.
- A **state-space model** is always a **proper model** in this toolbox.
- A **descriptor system** with a singular descriptor matrix may be an **improper model**.
- A **zero-pole-gain model** carries no delays; converting a delayed **model** to one is an error, not a silent drop.
- A **strictly proper model** is a **proper model**.
- A **biproper model** is a **proper model**.
- An **improper model** cannot be realized as a **state-space model** by this toolbox.
- A **FOPDT model** is a **transfer-function model**.
- A **SOPDT model** is a **transfer-function model**.
- A **state-space model**, **transfer function**, and **zero-pole-gain model** each belong to either the **continuous-time model** domain or the **discrete-time model** domain.
- A **discrete-time model** has exactly one **sample time**.
- A **sample time** is finite and positive; zero marks a **continuous-time model**.
- A **MIMO model** has one or more **input channels** and one or more **output channels**.
- A **SISO model** is the most minimal **MIMO model**.
- A **SISO model** has exactly one **input channel** and exactly one **output channel**.
- A **SIMO model** has exactly one **input channel** and multiple **output channels**.
- A **MISO model** has multiple **input channels** and exactly one **output channel**.
- An **input signal** is applied through an **input channel**.
- An **output signal** is observed from an **output channel**.
- A **test signal** can be a **step signal**, **sine signal**, **square signal**, or **pulse signal** in this toolbox.
- A **pulse signal** is an input signal; an **impulse response** is an output behavior.
- A **signal name** can identify an **input signal**, **output signal**, or **state signal**.
- A **reference signal** defines the desired target for a **closed-loop model**.
- A **measurement signal** is used by a **controller** or **observer**.
- A **control signal** is produced by a **controller** and applied to a **plant**.
- An **error signal** is formed from a **reference signal** and **measurement signal** in a **feedback loop**.
- A **static-gain model** is a **state-space model** with only feedthrough and no dynamic state.
- A **pole** describes system dynamics; a **zero** describes input-output transmission.
- A **right-half-plane pole** is outside the continuous-time stability region.
- A **right-half-plane zero** affects input-output behavior without directly making a model unstable.
- A **nonminimum-phase zero** is outside the stability region but is a **zero**, not a **pole**.
- A **minimum-phase model** has no **nonminimum-phase zeros**.
- **Channel gain** scales an input-output channel.
- **DC gain** is evaluated at s = 0 for a **continuous-time model** and z = 1 for a **discrete-time model**.
- **Bandwidth** is measured relative to **DC gain**, using a default -3 dB drop in this toolbox.
- The **Nyquist frequency** of a **discrete-time model** is pi divided by its **sample time**.
- A **pole** characterizes a **mode** when that mode appears in the input-output representation.
- A **continuous-time model** is stable only when every pole has negative real part.
- A **discrete-time model** is stable only when every pole has magnitude less than one.
- The **stability boundary** is the imaginary axis for a **continuous-time model** and the unit circle for a **discrete-time model**.
- A **marginally stable model** is treated as an **unstable model** by stability checks in this toolbox.
- A **transport delay** can appear as an **input delay**, **output delay**, or **internal delay**.
- An **exact delay** is preserved as delay metadata.
- An **external delay** is either an **input delay** or an **output delay**.
- An **internal delay** is delay metadata inside an interconnection.
- A **linear fractional transformation** connects a main model with an **uncertainty block**.
- A **delay block** is an **uncertainty block** used to represent an **internal delay**.
- A **nominal model** is the baseline for analysis or controller design.
- **Model uncertainty** describes variation around a **nominal model**.
- An **uncertain model** combines a **nominal model** with **model uncertainty**.
- **Additive uncertainty**, **multiplicative uncertainty**, and **parametric uncertainty** are forms of **model uncertainty**.
- An **absorbed delay** is represented by added **states** instead of delay metadata.
- A **fractional delay** cannot be represented as an integer sample delay without approximation or internal-delay handling.
- An **integer delay** can be represented directly as discrete-time delay metadata.
- An **augmented model** can add states for **absorbed delay**, integral action, or interconnection structure.
- A **delay bank** applies **delay approximations** or exact discrete delay models across multiple channels.
- A **delay approximation** replaces exact **transport delay** behavior with finite-dimensional dynamics.
- A **Pade approximation** is a **delay approximation** for continuous-time transport delays.
- A **Thiran allpass delay** is a **delay approximation** for discrete-time fractional sample delays.
- **Safe feedback** can use **Pade approximation** for continuous-time delayed models and exact delay absorption for discrete-time delayed models.
- An **interconnection** combines one or more systems into another system.
- A **series interconnection**, **parallel interconnection**, **block-diagonal interconnection**, **signal routing**, and **feedback interconnection** are kinds of **interconnection**.
- A **summing junction** combines signals inside **signal routing** or a **feedback loop**.
- A **feedback loop** produces a **closed-loop model**.
- A **feedback interconnection** closes a **feedback loop**.
- An **algebraic loop** can arise when **feedthrough** is present around a **feedback loop**.
- **Well-posed feedback** can be valid even when direct feedthrough creates an **algebraic loop**.
- **Negative feedback** and **positive feedback** are feedback-loop sign conventions.
- An **open-loop model** is evaluated before feedback closure.
- A **closed-loop model** is produced by closing a **feedback loop**.
- A **loop transfer** describes the open-loop dynamics around a **feedback loop**.
- A **plant** is a **control system** named from the viewpoint of controller design.
- A **plant model** is a **model** of a **plant**.
- A **controller model** is a **model** of a **controller**.
- A **controller** is connected to a **plant** to form a **closed-loop system**.
- A **regulator** is a kind of **controller**.
- A **compensator** is a kind of **controller**.
- A **lead compensator** and **lag compensator** can be represented as ordinary **models**.
- **State feedback** uses states to compute the **control signal**.
- **Pole placement** designs **state feedback** gains from desired closed-loop **poles**.
- **Ackermann pole placement** is a SISO form of **pole placement**.
- **Output feedback** uses measured outputs to compute the **control signal**.
- An **observer-based controller** is a kind of **output feedback** controller.
- An **observer** is a state **estimator**.
- An **observer** estimates states for a **plant** or **control system**.
- A **linear-quadratic estimator** is a kind of **observer**.
- A **Kalman filter** is an **observer**.
- An **extended Kalman filter** is an **estimator** for nonlinear models.
- An **extended Kalman filter** uses **Jacobians** during prediction and update.
- **Linearization** produces a **linear time-invariant model** from a nonlinear model at an **operating point**.
- A **Kalman filter** is defined by **noise covariance** assumptions.
- **Process noise** and **measurement noise** are represented by **noise covariance** matrices.
- **Output covariance** is computed from a stable model and input **noise covariance**.
- A **linear-quadratic regulator**, **linear-quadratic-Gaussian controller**, **H2 controller**, **H-infinity controller**, **PID controller**, and **two-degree-of-freedom PID controller** are kinds of **controller**.
- A **linear-quadratic-Gaussian controller** combines a **linear-quadratic regulator** with a **Kalman filter**.
- A **linear-quadratic-Gaussian controller** is an **observer-based controller**.
- **H2 synthesis** computes an **H2 controller** from a continuous-time **generalized plant**.
- An **H2 controller** and **H-infinity controller** are synthesized from a **generalized plant**.
- **Synthesis Riccati solutions** are returned by **H2 synthesis** and **H-infinity synthesis**.
- **Controller order** is the state dimension of a **controller model**.
- **H-infinity synthesis** computes an **H-infinity controller** from a continuous-time **generalized plant**.
- **Gamma** is the performance bound reported by **H-infinity synthesis**.
- **Closed-loop poles** describe the dynamics after controller synthesis or feedback closure.
- A **generalized plant** separates **performance channels** from **measurement channels**.
- A **generalized plant** separates **disturbance channels** from **control channels**.
- A **performance output** belongs to a **performance channel**.
- A **measurement output** belongs to a **measurement channel**.
- **H2 synthesis** and **H-infinity synthesis** partition a **generalized plant** by the number of **measurement outputs** and **control channels**.
- A **discrete linear-quadratic regulator**, **linear-quadratic integral regulator**, and **sampled-data linear-quadratic regulator** are variants of **linear-quadratic regulator**.
- A **linear-quadratic regulator** is defined by **cost matrices**.
- A **linear-quadratic regulator** is computed from an **algebraic Riccati equation**.
- A **continuous algebraic Riccati equation** applies to **continuous-time models**.
- A **discrete algebraic Riccati equation** applies to **discrete-time models**.
- **Parallel PID form** and **standard PID form** are parameterizations of a **PID controller**.
- A **two-degree-of-freedom PID controller** uses **setpoint weights**.
- A **derivative filter** makes PID derivative action proper.
- A **step response** and **impulse response** are kinds of **simulation response**.
- A **simulation time vector** indexes a time-domain **simulation response**.
- An **initial condition** sets the starting **state** for simulation.
- A **final state** can be reused as the **initial condition** for a later simulation.
- A **simulation workspace** is a reusable computation buffer, not a model state.
- An **initial response** is a **free response** from an **initial condition**.
- A **forced response** is produced by an applied **input signal**.
- A **frequency response** evaluates a model over frequency.
- A **frequency grid** provides the evaluation points for a **frequency response**.
- A **frequency grid** uses **angular frequency** values in this toolbox.
- A **frequency-response matrix** is one value of a **frequency response**.
- **Frequency-response data** samples **frequency-response matrices** over frequency points.
- An **FRD model** stores **sampled complex responses** over a **frequency grid**.
- An **FRD interconnection** combines compatible **FRD models** without converting them to **state-space models**.
- A **frequency-response estimate** is computed from sampled **input signals** and **output signals**.
- A **Welch estimate** uses windowed overlapping FFT segments.
- A **direct FFT estimate** uses one FFT ratio without segment averaging.
- **H1 estimator** is the default frequency-response estimation method in this toolbox.
- **H1 estimator** and **H2 estimator** are frequency-response estimation methods.
- **Coherence** is reported for each estimated input-output channel and frequency.
- **FFT length**, **window function**, and **overlap** configure FFT-based frequency-response estimation.
- **Overlap** applies to a **Welch estimate**.
- A **Bode result** is derived from a **frequency response**.
- A **Bode plot** visualizes a **Bode result**.
- A **Nyquist plot** visualizes a **Nyquist result**.
- A **Nichols plot** visualizes a **Nichols result**.
- A **Sigma plot** visualizes a **sigma result**.
- A **root locus plot** visualizes a **root locus result**.
- A **pole-zero map** visualizes a **pole-zero map result**.
- A **root locus result** contains **root locus branches** over loop gain.
- A **departure angle** is associated with an open-loop **pole**.
- An **arrival angle** is associated with an open-loop **zero**.
- **Gain margin**, **phase margin**, and **disk margin** are kinds of **stability margin**.
- **Stability margin** is evaluated from a **loop transfer**.
- A **gain margin** is evaluated at a **phase crossover**.
- A **phase margin** is evaluated at a **gain crossover**.
- **Peak sensitivity** is computed from the **sensitivity function**.
- **Output sensitivity** and **input sensitivity** differ for MIMO loops when plant-controller multiplication does not commute.
- **Complementary sensitivity** is paired with a **sensitivity function** in loop analysis.
- A **loop sensitivity result** contains both input-side and output-side sensitivity models.
- A **disk margin** is derived from the peak of the shifted **sensitivity function** S + (σ−1)/2; with skew σ = 1 that is **peak sensitivity**.
- **H2 norm** and **H-infinity norm** are kinds of **system norm**.
- An **H2 norm** is finite only for stable models without continuous-time direct feedthrough in this toolbox.
- An **H-infinity norm** is the peak gain over frequency for a stable model and +Inf for an unstable one.
- A **realization** is a **state-space model**.
- A **minimal realization** is both controllable and observable.
- A **minimal realization** preserves the input-output behavior of a **transfer function** with the minimum number of states.
- A **minimal realization**, **balanced realization**, and **canonical form** are specialized **realizations**.
- A **balanced realization** balances input-to-state and state-to-output energy transfers for the stable part of a model while preserving unstable modes.
- **Hankel singular values** identify states that are candidates for removal during **model reduction**.
- Infinite **Hankel singular values** mark unstable modes that must be preserved.
- The **eigensystem realization algorithm** identifies a discrete-time **state-space model** from **Markov parameters**.
- A **Markov parameter** is an impulse-response matrix sample.
- An **identified model** is estimated from data rather than specified directly.
- **Discretization** changes a **continuous-time model** into a **discrete-time model**.
- A **discretization method** defines how **discretization** approximates or transforms continuous-time behavior.
- **Zero-order hold** is the default **discretization method** for C2D, D2C and D2D, as in MATLAB c2d, d2c and d2d.
- **First-order hold**, **Tustin method**, **impulse-invariant method**, and **matched pole-zero method** are supported **discretization methods**.
- The **Tustin method** is the inverse of Tustin **discrete-to-continuous conversion** for valid models in this toolbox.
- The **matched pole-zero method** is limited to SISO models in this toolbox.
- **Discrete-to-continuous conversion** changes a **discrete-time model** into a **continuous-time model**.
- **Discrete-to-continuous conversion** supports zero-order-hold and **Tustin method** assumptions in this toolbox.
- **Discrete-to-discrete conversion** changes the **sample time** of a **discrete-time model**.
- A **random stable model** is generated for tests, benchmarks, or examples.
- A **total delay** combines **input delay**, **output delay**, and input-output path delay.
- A **delay model** represents **internal delay** through an augmented model and delay times.
- A **zero-delay approximation** replaces **delay blocks** with zero-delay behavior.
- **Named interconnection** and **indexed interconnection** are forms of **signal routing**.
- A **sum block** is a **static-gain model** generated from a summing expression.
- A **Smith predictor** is a controller structure for **time-delay plants**.
- **Balanced truncation** and **singular perturbation reduction** are **model reduction** methods.
- **Stable-unstable decomposition** separates stable and unstable modes.
- **Modal decomposition** separates modes around a cutoff.
- **State-space balancing** and **prescaling** are numerical conditioning transformations.
- A **similarity transform** changes state coordinates without changing input-output behavior.
- A **state permutation** is a **similarity transform** that reorders states.
- A **controllability matrix** supports **controllability** analysis.
- An **observability matrix** supports **observability** analysis.
- A **staircase decomposition** supports controllability, observability, zero, and reduction workflows.
- A **system inverse** inverts input-output behavior for supported square models.

## Error conventions

Every exported operation follows Go's error pattern strictly and keeps MATLAB's argument shapes. Sentinels live in errors.go with a doc comment each; shared guards live in require.go.

**Result or error**:
An exported operation returns a fully formed value with a nil error, or a non-nil error. It never returns a nil, empty or placeholder value (nil `*System`, 0×0 `&mat.Dense{}`, NaN, 0 or -1 standing for "none") with a nil error, even where MATLAB returns an empty model; it returns a sentinel error instead. It never panics on well-typed input, including nil pointers.

**Message format**:
`fmt.Errorf("OpName: detail: %w", ..., ErrX)`. OpName is the exported Go identifier the caller invoked: the function name (`Lyap:`), the bare method name for `*System` methods (`Poles:`), and `Type.Method:` for methods of other types (`FRD.Nyquist:`, `EKF.Update:`); detail is lowercase, names the offending argument and value (`Lyap: Q is 3×3, want 2×2: ...`), and never repeats the `controlsys:` prefix the sentinel already carries. Wrap exactly one sentinel with `%w`; never return a bare sentinel or an error without one from an exported operation. Unexported helpers return unprefixed detail (the exported caller adds `OpName: %w`) or take an `op string` and prefix it themselves, never both.

**Choosing the sentinel**:
- ErrInvalidArgument: nil model, function or required matrix; NaN/Inf scalar, slice or matrix entry; value out of its documented range (negative time, sign not ±1, order < 0); unknown enum, method or option string; out-of-range or non-increasing indices.
- ErrDimensionMismatch: sizes that disagree, and results that cannot be stored (see zero-dimension models).
- ErrOptionUnsupported: a valid option this operation does not implement for the given model.
- A specific sentinel (ErrWrongDomain, ErrDelayUnsupported, ErrDescriptorUnsupported, ErrSchurFailed, ...) whenever one names the condition more precisely. Add a new sentinel only when callers need to branch on it and several operations share it.
- ErrSchurFailed: any eigenvalue, Schur or QZ iteration that did not converge.

**Guards** (require.go):
- `requireSystem(op, sys)`: nil model → ErrInvalidArgument; otherwise Validate, wrapped `op: %w`. Does not check finiteness, since plain model arithmetic propagates NaN/Inf as MATLAB does.
- `requireFiniteSystem(op, sys)`: requireSystem plus every A, B, C, D, E and internal-delay block finite. Use before eigen, Schur, QZ, balancing or other LAPACK work: gonum Dgebal never terminates on NaN, so non-finite input must be rejected with ErrInvalidArgument before it reaches LAPACK.
- `requireFiniteDense(op, name, m)`: nil or non-finite matrix → ErrInvalidArgument; check nil yourself first when the matrix is optional.
- `requireFinite(op, name, v...)`: non-finite scalar or slice element → ErrInvalidArgument.

**Undefined quantities**:
A result quantity that may not exist is exposed through a comma-ok accessor method `(value, ok bool)` named for the quantity, with no Get/Has prefix; the backing field is unexported. For example a margin result with no gain crossover reports `wcg, ok := r.GainCrossover()` with ok false, rather than a NaN WgFreq field. Inf is returned only where it is mathematically true (the gain margin of a loop that never crosses -180°), never as a "none" marker.

**MATLAB-shaped arguments**:
Arguments, their order and their defaults follow the MATLAB Control System Toolbox function the operation mirrors; cite its doc page in the doc comment. A MATLAB optional output (lsim's x, balreal's T) becomes an opt-in request; when requested but unavailable the operation returns an error, never a nil field with a nil error.

**Zero-dimension models**:
- A state-space model with states but no inputs or no outputs (m = 0 or p = 0, n > 0) is valid; its empty blocks may be nil or empty `*mat.Dense` and both validate (see **Autonomous model**).
- A static gain with exactly one of m, p zero cannot be stored, so an operation whose result would be one returns ErrDimensionMismatch. A model with no states, inputs or outputs is a valid 0×0 gain.
- An operation whose matrix result cannot be formed for n = 0 (gramian, Lyapunov solution, transformation, gain) returns `OpName: system has no states: ErrDimensionMismatch`; gonum cannot use a 0×0 `*mat.Dense`.
- A zero-length slice is a valid value when the count is legitimately zero (the poles or Hankel singular values of a model with no states).

## Example dialogue

> **Dev:** "When a **controller** is connected to a **plant** with feedback, should the result still be called a controller?"
> **Domain expert:** "No. The **controller** and **plant** are separate systems; the result of the **feedback loop** is the **closed-loop system**."

## Flagged ambiguities

- "toolbox" names the project shape, not a domain object; use **control system** or a specific representation when discussing modeled behavior.
- "process" can mean a physical process, stochastic process, or operating-system process; use **plant** only in controller-design contexts and **control system** otherwise.
- "system" is accepted shorthand for **control system** only after the control-system context is established.
- "system model" and **model** are the same fundamental representation; use **model** unless the longer phrase improves readability.
- "linear time-invariant system" is accepted shorthand for **linear time-invariant model** only after the modeling assumption is clear.
- "continuous-time system" and "discrete-time system" are accepted shorthand for **continuous-time model** and **discrete-time model** only after the modeling context is clear.
- "open-loop system" and **open-loop model** are equivalent in this toolbox; use **open-loop model** when referring to the library result.
- "closed-loop system" and **closed-loop model** are equivalent in this toolbox; use **closed-loop model** when referring to the library result.
- "gain system" is accepted shorthand for **static-gain model** when matching established API names.
- "SISO system", "SIMO system", "MISO system", and "MIMO system" are accepted shorthand for their model terms only after the modeling context is clear.
- "dead time" is accepted in **FOPDT model** and **SOPDT model** names; otherwise use **transport delay**.
- "setpoint" is a process-control synonym for **reference signal**; use **reference signal** in general toolbox docs.
- "manipulated variable" is a process-control synonym for **control signal**; use **control signal** in general toolbox docs.
- "filter" is accepted for estimator designs that filter noisy measurements, such as a **Kalman filter**; use **observer** when state-estimation structure is the point.
