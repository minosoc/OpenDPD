__author__ = "Yizhuo Wu, Chang Gao, Ang Li"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl, a.li-2@tudelft.nl"

import os
import sys

# Prevent duplicate OpenMP runtime initialization on macOS (libomp vs. Apple's libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from steps import train_pa, run_pa, train_dpd, run_dpd
from project import Project


class Tee:
    """stdout/stderr를 파일과 터미널에 동시에 출력하는 클래스"""
    def __init__(self, file_path, mode="a"):
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        # 원본 스트림에도 출력
        if hasattr(self, '_is_stdout'):
            self.stdout.write(data)
            self.stdout.flush()
        else:
            self.stderr.write(data)
            self.stderr.flush()
    
    def flush(self):
        self.file.flush()
        if hasattr(self, '_is_stdout'):
            self.stdout.flush()
        else:
            self.stderr.flush()
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    proj = Project()
    
    # Project 생성 후 모든 출력을 로그 파일에 기록
    log_file_path = os.path.join(proj.path_dir_terminal_log, f"{proj.args.step}.log")
    tee_stdout = Tee(log_file_path, mode='w')
    tee_stdout._is_stdout = True
    tee_stderr = Tee(log_file_path, mode='a')
    
    # stdout과 stderr를 모두 리다이렉트
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    try:
        # PA Modeling
        if proj.step == 'train_pa':
            print("####################################################################################################")
            print("# Step: Train PA                                                                                   #")
            print("####################################################################################################")
            train_pa.main(proj)

        # PA Running
        elif proj.step == 'run_pa':
            print("####################################################################################################")
            print("# Step: Run PA                                                                                     #")
            print("####################################################################################################")
            run_pa.main(proj)

        # DPD Learning
        elif proj.step == 'train_dpd':
            print("####################################################################################################")
            print("# Step: Train DPD                                                                                  #")
            print("####################################################################################################")
            train_dpd.main(proj)

        # Run DPD to Generate Predistorted PA Outputs
        elif proj.step == 'run_dpd':
            print("####################################################################################################")
            print("# Step: Run DPD                                                                                    #")
            print("####################################################################################################")
            run_dpd.main(proj)
        else:
            raise ValueError(f"The step '{proj.step}' is not supported.")
    finally:
        # stdout과 stderr를 원본으로 복원
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()
