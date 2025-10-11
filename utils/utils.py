import sys, os, shutil, hashlib, logging, json

def setup_logger(logger_name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

class LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buf = ''

    def write(self, s: str):
        if not s:
            return
        self.buf += s
        lines = self.buf.splitlines()
        if self.buf.endswith('\n') or self.buf.endswith('\r'):
            for line in lines:
                self.logger.log(self.level, line)
            self.buf = ''
        else:
            for line in lines[:-1]:
                self.logger.log(self.level, line)
            self.buf = lines[-1] if lines else self.buf

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)
            self.buf = ''

class LogManager:
    def __init__(self, param: dict):
        self.param = param
        self.logger: logging.Logger | None = None
        self.original_stdout = None
    
    def start_logging(self):
        if self.param['mode'] == 'training':
            self.param['output_dir'] = f"Training_Recording/{self.param['jobtype']}/{self.param['time']}"
            logger_name = f"training_{self.param['time']}"
        elif self.param['mode'] == 'generation':
            self.param['output_dir'] = f"Generation_Recording/{self.param['jobtype']}/{self.param['time']}"
            logger_name = f"generation_{self.param['time']}"
            if self.param['device'] == 'cuda':
                with open('generation/logging_config.json') as f:
                    logging_config = json.load(f)
                logging_config['handlers']['vllm']['filename'] = os.path.abspath(
                    f"{self.param['output_dir']}/{logger_name}.log"
                    )
                with open('generation/logging_config.json', 'w') as f:
                    json.dump(logging_config, f, indent=2)
        elif self.param['mode'] == 'hpo':
            self.param['output_dir'] = f"HPO_Recording/{self.param['jobtype']}/{self.param['time']}"
            logger_name = f"hpo_{self.param['time']}"
            self.optuna_db = f'sqlite:///{self.param['output_dir']}/hpo_{self.param['time']}.db'

        os.makedirs(self.param['output_dir'], exist_ok=True)
        log_file_path = f"{self.param['output_dir']}/{logger_name}.log"
        shutil.copy('model_parameters.yml', self.param['output_dir'])

        self.logger = setup_logger(logger_name, log_file_path)
        self.original_stdout = sys.stdout
        sys.stdout = LoggerWriter(self.logger, logging.INFO)

    def end_logging(self):
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            

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
