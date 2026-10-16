"""Suite-wide test configuration.

The app picks its classifier from whether a model file exists (main.py,
_gesture_stack). The repository ships a trained model, so without this the
suite's behaviour would depend on what happens to be in ml/artifacts/. Tests
run on the stub by default -- its classes are defined by geometry, so the
protocol tests can predict them -- and the ones about the real model opt in
by pointing MODEL_PATH at an artifact of their own.

Set here, before anything imports app.config: Settings reads the environment
once, at import, and environment variables outrank the .env file.
"""

import os

os.environ["MODEL_PATH"] = "ml/artifacts/no-model-in-tests.joblib"
