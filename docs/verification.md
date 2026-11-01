# Verification and implementation decisions

2026-11-01. User requested a GitHub project with a Hugging Face-compatible demo.

## Decisions

- Implemented a tiny independent transport/refinement predictor, not a Wan adapter.
  This makes the demo affordable on CPU but does not test a production diffusion model.
- Used explicitly synthetic constants; no biological dataset was supplied.
- Replaced an early direct-pixel base after visual inspection showed rapid blur.
  Retrained every final condition; the demonstration is exploratory, not preregistered.
- Training is CLI-only. The Space starts from bundled actual weights and never
  trains on visitors' uploads. User recordings are reduced to short grayscale clips.
- Built in a fresh checkout/feature branch; original MIT license preserved.

## Checks

- Full suite: **20 passed** using Python 3.12, CPU PyTorch 2.15.0+cpu,
  Gradio 5.49.1. Two third-party Starlette/Gradio deprecation warnings.
- Live local HTTP/API test: queued toy-world and uploaded-video predictions,
  video input preprocessing, rendered MP4 and JSON download serialization all pass.
- Video export decoded and inspected visually. Browser screenshot automation was
  unavailable because the browser download was truncated; UI layout was not checked
  in an automated browser. The live Gradio API and component construction were tested.
- Frozen shared weights, matched adapter initialization, no future-frame inputs,
  stable unforced E/I decay, one memory update per frame, explicit budget call counts,
  checkpoint roundtrip/checksum, invalid uploads, and generated-file cleanup tested.
- Actual base + three adapters trained. Main benchmark: 168 condition/sequence rows;
  three intervention runs each contain 72 rows. Raw CSV/JSON and provenance included.

## Independent review

A separate reviewer inspected source, scientific controls and tests and independently
ran the then-current 17-test suite. No critical or important source findings.
Two reported minor findings were treated as correctness fixes: malformed checkpoint
errors are now actionable, and shuffled-constant exports now record effective channel
assignments. Both fixes were reproduced with failing tests and verified in the final
20-test suite. No reported finding was left unfixed.

CUDA execution and a hosted Hugging Face deployment were not available for validation.
The free-CPU Space configuration is provided; no claim that a Space has been published.
Video decoding is bounded by file size, resolution, frame count and output duration;
adversarial decoder hang/timeouts were not tested. Model quality limitations and
negative comparisons are documented in `results/DEMO.md`.
