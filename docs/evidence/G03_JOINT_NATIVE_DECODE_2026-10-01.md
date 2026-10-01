# Joint Native Decode, 2026-10-01

The joint source-fit artifact now has a loader for the existing compositional
decoder. `GroundedNativeChartDecoder.from_fit` independently verifies the fit
and completion, checks the exact source parent and native model/arithmetic
basis, then attaches the selected native suffix and relational pointer. The
caller owns the exact loaded model and exclusive lane. The loader neither
loads another backbone nor grants serving authority.

Only public source tokens, frozen parent features and public input values
enter `decode`. Source-only prefix capture and the selected suffix produce
actual distinct layer states. The ordinary global operation/argument chart
passes all admitted alternatives to the pointer-conditioned solver. No target
graph, answer or construction label is passed to the decoder.

The output receipt matches the selected graph and argument score against the
recorded chart resolutions. Operation spans identify result definitions even
after dependency reordering. The last chart examined is not assumed to be the
winner. Refusals remain refusals; a prior receipt is cleared on a new request.

The default loader requires a positive-step selected checkpoint. A declared
initial-checkpoint diagnostic mode can inspect the zero-update state, but its
receipt records that no learned checkpoint was selected. It cannot satisfy
the positive-step qualification gate.

Source calibration can use the existing construction/depth stratified,
source-identity quota. The selected identities are bound into the native
plan. This changes source-side measurement cost, not the held evaluation
population or acceptance standard.

## Checks And Boundary

The seven affected suites passed 105 tests in 28.98 seconds. After output
receipt linkage was added, the native and chart suites passed 17 tests in
16.61 seconds. These checks cover an actual small Qwen2 joint fit, artifact
reload, selected native layer capture, an ordinary public-input decode call,
changed parent rejection and receipt linkage to the returned graph.

The small native fixture does not establish semantic gain: its four-update
fit can select the initial checkpoint and the integrated chart can refuse.
The separate learned-pointer chart fixture demonstrates correction of its
known erroneous baseline, not unseen-language transfer. Neither fixture is
the resident 27B. G03, later G items, live serving and general promotion remain
open pending their own measurements.
