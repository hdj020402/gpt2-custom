import sys, os, shutil, hashlib

class LogManager:
    def __init__(self, param: dict):
        self.param = param
        self.log_file = None
        self.original_stdout = None
    
    def start_logging(self):
        if self.param['mode'] == 'training':
            self.param['output_dir'] = f"Training_Recording/{self.param['jobtype']}/{self.param['time']}"
            log_file_path = f"{self.param['output_dir']}/training_{self.param['time']}.log"
        elif self.param['mode'] == 'generation':
            self.param['output_dir'] = f"Generation_Recording/{self.param['jobtype']}/{self.param['time']}"
            log_file_path = f"{self.param['output_dir']}/generation_{self.param['time']}.log"
        elif self.param['mode'] == 'hpo':
            self.param['output_dir'] = f"HPTuning_Recording/{self.param['jobtype']}/{self.param['time']}"
            log_file_path = f"{self.param['output_dir']}/hptuning_{self.param['time']}.log"
            self.optuna_db = f'sqlite:///{self.param['output_dir']}/hptuning_{self.param['time']}.db'

        os.makedirs(self.param['output_dir'], exist_ok=True)
        shutil.copy('model_parameters.yml', self.param['output_dir'])

        self.log_file = open(log_file_path, 'w', buffering=1)
        self.original_stdout = sys.stdout
        sys.stdout = self.log_file

    def end_logging(self):
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.log_file:
            self.log_file.close()

def hash_files(*file_paths, chunk_size=65536):
    if not file_paths:
        raise ValueError("At least one file path must be provided.")
    
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
    return hasher.hexdigest()
