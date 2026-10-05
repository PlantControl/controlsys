# Matched and least-squares conversion

`C2D` and `D2C(C2DMethodMatched)` support ordinary proper SISO
models. Forward conversion maps finite poles/zeros through `exp(s*Dt)` and adds
`max(relativeDegree-1,0)` zeros at -1. Both directions match the leading
low-frequency response coefficient, including integrators and zeros at DC.
Reverse conversion uses the principal logarithm, removes -1 numerator factors
as artificial zeros, and rejects other nonpositive real roots and non-real
reconstruction. This convention cannot distinguish a genuine Nyquist zero from
an artificial matched zero. The source model remains unchanged. Input/output
names and external delays are preserved; state names are cleared because the
new realization has different coordinates. Nonsingular descriptor models use
their explicit form; singular descriptor and internal-delay models are rejected.

These conventions follow the [MathWorks conversion methods](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html).
The [MathWorks PID reference](https://www.mathworks.com/help/control/ref/pid2.html)
specifies forward-Euler integration for matched conversion. A
[MATLAB Central executed example](https://www.mathworks.com/matlabcentral/answers/2175248-one-zero-is-missing-in-the-result-of-c2d-function-uses-matche)
reports `11/(s*(s+1))`, sample time .1, as approximately
`.052339*(z+1)/((z-1)*(z-.9048))`. This is a user-published MATLAB result;
our tests also derive the exact coefficients independently from root mapping
and the low-frequency limit. No local MATLAB execution is claimed.

`C2DFit(Dt, fitOrder)` returns the fitted model, selected order,
RMS relative response error, maximum normalized response error, and fitted
stability. `C2D` selects it with `C2DMethodLeastSquares` and
`C2DOptions.FitOrder`. Zero order selects the source state order; positive values
choose a different fit order. Fits use equally weighted frequency samples from
DC through Nyquist, up to twelve denominator-reweighted least-squares iterations,
followed by up to eight response-error Gauss-Newton refinements, and a
singular-value decomposition with column scaling and rank truncation.
The smallest actual response-error iterate is retained. Stability is reported
without pole reflection.

Integrators are retained at z=1 with their multiplicity and leading
low-frequency residue. These models use an open midpoint grid to avoid
sampling the singular DC response, and the fit order must accommodate the
integrators. Singular responses at other sampled frequencies are rejected.
Exact integer external delays are preserved; the direct fitting API rejects
fractional external delays, internal delays, MIMO, and singular descriptor
models.
Option-based conversion applies its common external-delay policy separately.

Fit quality uses a distinct midpoint validation grid. RMS relative error is
RMS absolute error divided by RMS source magnitude. Maximum normalized error
is maximum absolute error divided by maximum source magnitude. These finite-grid
measurements cannot certify unsampled narrow resonances. Fitting an unstable
model does not preserve its unstable dynamics automatically.

The objective and order options follow [MathWorks c2dOptions](https://www.mathworks.com/help/control/ref/c2doptions.html).
MathWorks documents vector-fitting optimization; this implementation uses
polynomial denominator reweighting. Its proprietary optimizer and coefficients
have not been reproduced or verified. Persistent tests compare independent
analytic frequency responses, exact rational data, and an offline NumPy
variable-projection reference, whose script is
`testdata/conversion/least_squares_numpy.py`. That reference solves numerator
least squares at each scalar denominator candidate and minimizes over the
remaining denominator coefficient. It is an independent fitting oracle,
not a MATLAB fixture.

MATLAB verification gate: run
`generate_matlab_least_squares` from `testdata/conversion` in MATLAB with Control
System Toolbox, then run
`CONTROLSYS_MATLAB_LS_FIXTURES=/absolute/path/matlab-least-squares.json go test -v -run TestMATLABLeastSquaresReference`.
The generated file records MATLAB/toolbox versions, normalized transfer
coefficients and an independent validation-frequency grid for default/reduced/
increased order, fast dynamics and an integrator. The gate checks coefficients
and responses at relative tolerance1e-6. It skips explicitly without supplied
MATLAB output; a skipped gate establishes no MATLAB parity. The generator has
not been executed locally because MATLAB is unavailable.
