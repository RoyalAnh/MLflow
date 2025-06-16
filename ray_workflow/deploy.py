import os
import sys
import ray
from ray import serve
import mlflow
from serve_inference import InferenceModel
import tempfile
import zipfile
import shutil

def get_checkpoint_from_mlflow(run_id, artifact_path="checkpoints"):
    client = mlflow.tracking.MlflowClient()
    # Tải artifact checkpoint (có thể là folder hoặc file zip)
    artifacts = client.list_artifacts(run_id, path=artifact_path)
    if len(artifacts) == 1 and artifacts[0].path.endswith(".zip"):
        # Nếu checkpoint là file zip
        zip_artifact_path = artifacts[0].path
        tmpdir = tempfile.mkdtemp()
        zip_local = client.download_artifacts(run_id, zip_artifact_path, tmpdir)
        unzip_dir = os.path.join(tmpdir, "checkpoint_unzipped")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_local, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        # Trả về đúng thư mục chứa checkpoint (có thể phải điều chỉnh nếu bên trong zip có 1 thư mục)
        for root, dirs, files in os.walk(unzip_dir):
            if "params.json" in files or "rllib_checkpoint.json" in files:
                return root
        # Nếu không tìm thấy file params.json, trả về luôn thư mục unzip
        return unzip_dir
    else:
        # Nếu checkpoint là thư mục artifact
        local_path = client.download_artifacts(run_id, artifact_path)
        return local_path

if __name__ == "__main__":
    # Nhận run_id (từ dòng lệnh hoặc biến môi trường)
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = os.environ.get("RUN_ID")
    if not run_id:
        raise RuntimeError("Bạn cần truyền run_id qua dòng lệnh hoặc biến môi trường RUN_ID")
    checkpoint_path = get_checkpoint_from_mlflow(run_id)
    print(f"Checkpoint path dùng để deploy: {checkpoint_path}")
    ray.init(ignore_reinit_error=True)
    serve.start(detached=True)
    serve.run(InferenceModel.bind(checkpoint_path=checkpoint_path), route_prefix="/predict")
    print("🚀 Inference endpoint ready at: http://localhost:8000/predict")
    import time
    print("🔄 Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
