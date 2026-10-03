# Conversion delay methods

`DiscretizeWithOpts` defaults to ZOH and `DelayModeling: "delay"`.
`C2DDelayModelingInternal` and `C2DDelayModelingDelay` name that format;
`C2DDelayModelingState` turns approximation memory (fractional-hold histories,
Thiran filters) into states; integer sample delays stay delay metadata, as in MATLAB.
`Discretize` remains the explicit plain-Tustin convenience operation;
`DiscretizeZOH` and `DiscretizeFOH` select their named hold methods.

| Method | External input/output/path delays | Internal feedback delays |
| --- | --- | --- |
| ZOH | Exact sampled response to held inputs, including fractional delays | Integer delays exact; fractional `tau = k*Ts + rho` absorbs `rho` into the augmented rational model and keeps `k` as internal delay (MATLAB policy; approximate for feedback) |
| FOH | Exact sampled response to linearly interpolated inputs, including fractional delays | Modified FOH on the full augmented rational model with the same `k*Ts + rho` split; approximate for feedback |
| Tustin | Nearest integer delay by default; optional Thiran approximation | Full augmented bilinear transformation; nearest integer delay or optional Thiran filters; zero-delay channels eliminated algebraically |
| Matched | Nearest integer delay by default; optional Thiran approximation; SISO only | Unsupported |
| Impulse | Exact delayed impulse-response samples; source direct feedthrough omitted | Unsupported |
| Least squares | Consult the fitting contract and returned error metrics | Unsupported |

`ThiranOrder` is valid only for Tustin and matched. A positive order models
external fractional delays with filters; `"delay"` represents their recurrence
using internal unit delays, and `"state"` uses ordinary state variables.
`ThiranOrder` is a maximum: order `N = min(ceil(D), ThiranOrder)` with integer
remainder `ceil(D) - N` kept exact. Internal feedback delays receive the same
filters; singular instantaneous loops are rejected. Nondecomposable fractional
MIMO path delays with Thiran are still rejected (tracked: L3TY7D).

Fractional external ZOH conversion preserves the original rational state order
when delays decompose into input and output delays. It splits input kernels and
uses one- or two-sample histories. Fractional FOH and residual MIMO path delays
use independent channel realizations; their rational order can reach the
original order times the input count times the output count. These fallbacks
trade memory and conversion work for exact sampled response. Delay-free
conversion retains its direct numerical fast path.

Discrete delay metadata uses sample counts. Reverse conversion multiplies
retained delay metadata by the original sample time. It cannot recover an exact
continuous delay from an ordinary realization containing absorbed history or
Thiran states. Tustin inverse conversion transforms every internal input,
output, and feedthrough matrix; retaining only the old internal matrices would
change the model's response.

The optional conversion result reports state-map and delay limitations, and warns
when Tustin/matched round fractional delays or approximate them with Thiran filters.
Fractional delayed realizations require input/output history and may change
state coordinates; unsupported initial-history maps are identified rather
than treated as exact zero-history reconstruction.

`D2D` accepts only ZOH and Tustin. The default uses inverse and forward ZOH.
Tustin uses the same prewarp frequency at the old and new sample times, requiring
that frequency below both Nyquist limits. All options are validated even when
the sample time is unchanged.

Independent fixtures cover nonsymmetric MIMO sampled responses, fractional
input/output/path delays, both delay formats, delayed gains, impulse samples,
all-port Tustin frequency identities, internal series products, and state/history
absorption. They are analytic or numerical-integration oracles, not a claim of
comparison against an installed MATLAB runtime.

References: [MathWorks conversion methods](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html),
[MathWorks c2d options](https://www.mathworks.com/help/control/ref/c2doptions.html),
[MathWorks d2d](https://www.mathworks.com/help/control/ref/dynamicsystem.d2d.html).
