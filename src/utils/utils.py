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

    def fileno(self):
        return sys.__stdout__.fileno()

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)
            self.buf = ''


class LogManager:
    """Context manager that redirects stdout to a log file.

    Usage::

        with LogManager(param) as lm:
            # stdout is captured to log file
            ...
        # stdout is always restored, even on exception
    """

    def __init__(self, param: dict):
        self.param = param
        self.logger: logging.Logger | None = None
        self._original_stdout = None

    def __enter__(self):
        self._setup_output_dir()
        os.makedirs(self.param['output_dir'], exist_ok=True)
        log_file_path = f"{self.param['output_dir']}/{self._logger_name}.log"
        shutil.copy('configs/model_parameters.yml', self.param['output_dir'])
        if self.param['mode'] == 'hpo':
            shutil.copy('configs/hpo.yml', self.param['output_dir'])

        self.logger = setup_logger(self._logger_name, log_file_path)

        self._original_stdout = sys.stdout
        sys.stdout = LoggerWriter(self.logger, logging.INFO)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always restore stdout, even if an exception occurred
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self.logger is not None:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        # Don't suppress exceptions
        return False

    def _setup_output_dir(self):
        mode = self.param['mode']
        jobtype = self.param['jobtype']
        time_str = self.param['time']

        if mode == 'training':
            self.param['output_dir'] = f"outputs/training/{jobtype}/{time_str}"
            self._logger_name = f"training_{time_str}"
        elif mode == 'generation':
            self.param['output_dir'] = f"outputs/generation/{jobtype}/{time_str}"
            self._logger_name = f"generation_{time_str}"
            if self.param['device'] == 'cuda':
                self._setup_vllm_log_config()
        elif mode == 'hpo':
            self.param['output_dir'] = f"outputs/hpo/{jobtype}/{time_str}"
            self._logger_name = f"hpo_{time_str}"
            self.optuna_db = f'sqlite:///{self.param["output_dir"]}/hpo_{time_str}.db'
        else:
            self.param['output_dir'] = f"outputs/{mode}/{jobtype}/{time_str}"
            self._logger_name = f"{mode}_{time_str}"

    def _setup_vllm_log_config(self):
        with open('configs/logging_config.example.json') as f:
            logging_config = json.load(f)
        logging_config['handlers']['vllm']['filename'] = os.path.abspath(
            f"{self.param['output_dir']}/{self._logger_name}.log"
        )
        with open('configs/logging_config.json', 'w') as f:
            json.dump(logging_config, f, indent=2)


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
