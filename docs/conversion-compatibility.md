# Linear model conversion

Canonical methods follow MathWorks ordinary proper LTI semantics. Existing method
signatures remain; method behavior is corrected directly. No migration layer.

| Operation | Methods | Options |
| --- | --- | --- |
| C2D | ZOH, triangle FOH, Tustin, matched, impulse, least-squares | Sample time; Tustin prewarp; Tustin/matched Thiran order; least-squares fit order; delay or state realization |
| D2C | ZOH, triangle FOH, Tustin, matched | Tustin prewarp |
| D2D | ZOH, Tustin | Target sample time; Tustin prewarp; consistent inverse/forward method |

Zero-valued method selects ZOH. Prewarp frequency is radians/second, zero disables
it, and it must lie below Nyquist in every affected discrete domain. D2D validates
options even at unchanged sample time. DelayModeling defaults to `delay`;
`state` turns approximation memory into states and keeps integer delays. Tustin/matched without
Thiran round external delays to the nearest sample, matching documented MATLAB
policy. Thiran is an explicit approximation, available only with those methods.

FOH uses linear interpolation between consecutive input samples. Its rational
state is `xd = xc - Gamma1*u`; it retains the original state count for ordinary
models. Tustin uses MATLAB coordinates `xd = (I - A/beta)*xc - (B/beta)*u`,
where `beta = 2/Ts` or the prewarped frequency factor. Its inverse and returned
state maps use the same coordinates. Impulse conversion includes the initial dynamic sample `Ts*C*B`;
the continuous direct impulse term is omitted. It rejects internal feedback
delays. ZOH reverse conversion handles integrators and defective repeated poles
through an augmented matrix logarithm. A guarded spectral fast path handles
well-conditioned common cases. A pole at zero has no finite continuous logarithm.
Negative real poles add one real alias state per negative mode; positive and
existing conjugate modes retain their state count. Original states stay first,
with appended alias states initially zero. Sampled input/output and nonzero
initial-state responses are preserved. A verified invariant-subspace compression
rejects numerically unresolved branch cuts and ill-conditioned projections
explicitly; it adds no work to the ordinary positive-pole path.

Matched conversion and least-squares fitting are SISO. Their conventions,
independent fixtures, fitting diagnostics, and limitations are described in
[matched-least-squares-conversion.md](matched-least-squares-conversion.md).
Least-squares implements the documented frequency-fitting objective and order
selection; exact proprietary MATLAB optimizer/coefficient parity is unverified.

External fractional ZOH input/output delays use a shared rational realization
where delays decompose by channel. Nondecomposable path delays and fractional
FOH use per-path realizations; their state count and cost can grow with the
number of input/output paths. Discrete delay histories always use integer sample
counts. Internal Tustin transforms every LFT port and feedthrough matrix.
Fractional internal feedback delays under ZOH/FOH absorb the fractional part
into the augmented model (approximate). Tustin/matched with Thiran realize
nondecomposable fractional MIMO path delays by copying the model per fractional
row or column (see [conversion-delay-methods.md](conversion-delay-methods.md)).

`DiscretizeWithResult`, `D2CWithResult`, and `D2DWithResult` return the converted
model, method, approximation flag, added-state warnings, and initial-state map.
`MapInitialState` validates dimensions and finite values and accepts source state,
initial input, and internal delay output. Ordinary ZOH/FOH/Tustin and retained
Tustin internal ports have mappings. Integer ZOH histories append zero states.
Fractional delay histories and matched/fitted coordinates can lack a mapping;
nonzero initial conditions are refused where unsupported. Delay history still
requires separate initialization. Reverse conversion cannot recover aliased
frequencies.

Descriptor models with nonsingular E are converted in explicit form (E\A,
E\B and internal-delay E\B2; state coordinates unchanged); singular E is
rejected with ErrDescriptorSingular and ErrDescriptorUnsupported. Sparse/TRBDF2,
identified/stochastic model families, and arbitrary absorbed-delay continuous
inverses are outside this ordinary proper LTI API.
Exact full MATLAB parity remains gated by the documented unverified cases.

Validation uses analytic scalar and nonsymmetric MIMO sampled responses,
independent RK4 integration, SciPy triangle-FOH fixtures, matched root/low-frequency
limits, NumPy fitting oracles, all-port LFT frequency products, and persistent
initial-condition/failure tests. Roundtrip tests supplement these independent
checks. No local MATLAB execution is claimed.

## Evidence matrix

Every supported cell below has persistent independent evidence (analytic,
numerical-integration, SciPy/NumPy oracle, or published MathWorks numbers).
"Gated" means only a MATLAB-run fixture can close exact parity.

| Op | Method | SISO | MIMO | Delays | Initial state | Evidence (tests) |
| --- | --- | --- | --- | --- | --- | --- |
| C2D | ZOH | yes | yes | ext exact incl. fractional; internal integer exact, fractional approx | map | `TestZOH_ScalarAnalytical`, `TestCrossval_ZOH_*`, `TestDelayedZOHIndependentSampledMIMO`, `TestC2D_MATLAB_ZOH_FractionalDelay`, `TestFractionalFeedbackHold*`, `TestConversionInitialStateMIMOIndependentResponse` |
| C2D | FOH (triangle) | yes | yes | ext exact; internal approx | map | `TestModifiedFOHSciPyReference`, `TestModifiedFOHPiecewiseLinearMIMO`, `TestDelayedFOHAndPathZOHIndependentSamples`, `TestFOHInternalDelayInitialMapAndApproximation`, `TestConversionInitialStateMIMOIndependentResponse` |
| C2D | Tustin | yes | yes | rounded (warned) or Thiran; internal all-port | map (none for Thiran memory) | `TestCrossval_Tustin_Reference*`, `TestInternalTustinAllPortsIndependentResponse`, `TestC2D_MATLAB_Tustin_Thiran`, `TestInternalThiranIndependentMIMOResponseAndRepresentations`, `TestTustinInternalDelayInitialMap` |
| C2D | Tustin prewarp | yes | yes | as Tustin | map | `TestPrewarpMIMOIndependentFrequency`, `TestPrewarpZeroLimitAndDelayUnits`, `TestPrewarpInvalidOptions` |
| C2D | Matched | yes | no | rounded (warned) or Thiran; internal unsupported | none | `TestMatchedForwardIndependentCoefficients`, `TestMatchedDeflateRepeatedArtificialZeros`, `TestConversionThiranPublishedCoefficientsAndMaximumOrder` |
| C2D | Impulse | yes | yes | ext path exact; internal unsupported | n/a | `TestImpulseInvariantSamplesIncludingZero`, `TestImpulseFractionalPathDelayIndependentSamples`, `TestImpulse_ImpulseResponseMatch` |
| C2D | Least squares | yes | no | see fitting contract | none | `TestLeastSquaresIndependentFrequencyResponse`, `TestLeastSquaresOfflineVariableProjectionReference`, `TestLeastSquaresIntegratorConstraints`; MATLAB coefficients **gated** (`TestMATLABLeastSquaresReference`) |
| D2C | ZOH | yes | yes | integer metadata retained | map | `TestD2C_ZOH_MIMO_Roundtrip`, `TestD2CZOHDefectiveAndNearUnit`, `TestD2CZOHNegativeMixedDefectiveResponses`, `TestNegativeZOH*`, `TestMATLABC2DD2CDelayRoundtrip` |
| D2C | FOH | yes | yes | integer metadata retained | map | `TestFOHIntegratorFeedthroughAndReverseLimitations`, `TestModifiedFOHSciPyReference` |
| D2C | Tustin (+prewarp) | yes | yes | integer metadata retained | map | `TestD2C_Tustin_Roundtrip`, `TestPrewarpSISOIndependentInverse` |
| D2C | Matched | yes | no | integer metadata retained | none | `TestMatchedReverseIndependentResponse`, `TestMatchedReverseComplexPolesAndMetadata` |
| D2D | ZOH | yes | yes | integer metadata retained | composed map | `TestD2DZOHAnalyticRecurrence`, `TestD2D_MIMO`, `TestD2DInitialStateOutput` |
| D2D | Tustin (+prewarp) | yes | yes | integer metadata retained | composed map | `TestD2DTustinIndependentFrequency`, `TestD2DValidatesSameRateOptions` |

Nondecomposable Thiran path delays: `TestTustinThiranNondecomposablePathDelays`,
`TestTustinThiranPathDelaysWithInternalFeedback`. Explicit-order `ThiranDelay`
accepts any stable delay `D > N-1`: `TestThiranDelayShortStableDelays`,
`TestThiranDelayRejectsBelowStabilityBound`.

Known gaps: least-squares exact MATLAB coefficients (gated). Performance:
[conversion-performance.md](conversion-performance.md).

References: [c2d](https://www.mathworks.com/help/control/ref/dynamicsystem.c2d.html),
[d2c](https://www.mathworks.com/help/control/ref/dynamicsystem.d2c.html),
[d2d](https://www.mathworks.com/help/control/ref/dynamicsystem.d2d.html),
[conversion methods](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html),
[c2dOptions](https://www.mathworks.com/help/control/ref/c2doptions.html).
