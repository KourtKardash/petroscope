import os
import subprocess
import multiprocessing
from datetime import datetime
import time
import signal
import sys
import argparse

MODELS = [
    "petroscope.segmentation.models.resunet.train",
    "petroscope.segmentation.models.resunet.train_polirized_sift",
    "petroscope.segmentation.models.resunet.train_polirized_loftr_1",
    "petroscope.segmentation.models.resunet.train_polirized_loftr_3",
    "petroscope.segmentation.models.resunet.train_polirized_loftr_6",
    "petroscope.segmentation.models.resunet.train_polirized_loftr_12"
]

class ModelRunner:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.num_gpus = self._get_available_gpus()
        self.processes = []
        self._prepare_logs()

    def _get_available_gpus(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
            )
            return len(output.decode().strip().split('\n'))
        except:
            print("Couldn't identify GPU, using CPU")
            return 0

    def _prepare_logs(self):
        os.makedirs("logs", exist_ok=True)

    def _run_model(self, model: str, gpu_id: int):
        log_file = f"logs/{model.replace('.', '_')}.log"
        
        env = os.environ.copy()
        if self.num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [sys.executable, "-m", model]
        if self.test_mode:
            cmd.append("--test")
        print(cmd)
        with open(log_file, 'a') as f:
            f.write(f"=== Started at {datetime.now()} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Using gpu number is: {gpu_id}\n\n")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            return proc

    def _monitor_processes(self):
        while any(p.poll() is None for p in self.processes):
            self._print_status()
            time.sleep(5)
        self._print_status()

    def _print_status(self):
        os.system('clear')
        print("=== Model Training Monitor ===")
        print(f"Test mode: {'ON' if self.test_mode else 'OFF'}")
        print(f"Available GPUs: {self.num_gpus or 'CPU only'}")
        
        print("\n[ Process Status ]")
        for i, (model, proc) in enumerate(zip(MODELS, self.processes)):
            status = "RUNNING" if proc.poll() is None else f"EXITED ({proc.returncode})"
            gpu = i % self.num_gpus if self.num_gpus else "N/A"
            print(f"{model[:50]:<50} | GPU: {gpu:<3} | {status}")
        
        if self.num_gpus > 0:
            print("\n[ GPU Utilization ]")
            os.system('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv')

    def run_all(self):
        try:
            for i, model in enumerate(MODELS):
                gpu_id = i % self.num_gpus if self.num_gpus else None
                proc = self._run_model(model, gpu_id)
                self.processes.append(proc)
                time.sleep(1)            
            self._monitor_processes()
            
        except KeyboardInterrupt:
            print("\Interrupt (Ctrl+C). Stopping all processes...")
            for proc in self.processes:
                proc.terminate()
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    runner = ModelRunner(test_mode=args.test)
    runner.run_all()
    print("All models have completed their training")