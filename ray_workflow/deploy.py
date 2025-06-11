import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray import serve
from remote import full_pipeline
from serve_inference import InferenceModel

ray.init(ignore_reinit_error=True)

# 1. Chạy pipeline bằng Ray Tasks
result_ref = full_pipeline.remote()
checkpoint_path = ray.get(result_ref)

# 2. Bắt đầu Serve
serve.start(detached=True)

# 3. Khởi chạy mô hình inference
serve.run(InferenceModel.bind(checkpoint_path=checkpoint_path), route_prefix="/predict")

print("🚀 Inference endpoint ready at: http://localhost:8000/predict")
