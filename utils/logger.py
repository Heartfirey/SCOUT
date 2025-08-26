import sys
import logging
from torch.distributed import get_rank
from colorama import Back, Fore, Style, init

class ColoredFormatter(logging.Formatter):
    def __init__(self, gpu_rank="RANK0"):
        # self.gpu_rank = gpu_rank
        super().__init__('%(asctime)s %(levelname)s %(message)s'.format(gpu_rank), datefmt='%m-%d %H:%M:%S')

    def format(self, record):
        if record.levelno == logging.INFO:
            asctime = f"{Fore.RED}{record.asctime}{Style.RESET_ALL}"
            levelname = f"{Back.GREEN}{record.levelname}{Style.RESET_ALL}"
            message = f"{Fore.WHITE}{record.message}{Style.RESET_ALL}"
            record.asctime = asctime
            record.levelname = levelname
            record.msg = message
        return super().format(record)

class Logger():
    def __init__(self, name="TalNet", path="log.txt", multi_gpu=False):
        self.logger = logging.getLogger(name)
        # self.process_rank = "RANK0" if not multi_gpu else "RANK{}".format(get_rank())
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setLevel(logging.INFO)
        self.stdout_handler.setFormatter(ColoredFormatter())
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%m-%d %H:%M:%S'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        

    def info(self, txt):
        self.logger.info(txt)
    
    def key_info(self, txt):
        self.logger.info(f"{Fore.YELLOW}{txt}{Style.RESET_ALL}")
    
    def success_info(self, txt):
        self.logger.info(f"{Fore.GREEN}{txt}{Style.RESET_ALL}")
    
    def freeze_info(self, txt):
        self.logger.info(f"{Fore.CYAN}{txt}{Style.RESET_ALL}")
        
    def warn_info(self, txt):
        self.logger.info(f"{Fore.RED}{txt}{Style.RESET_ALL}")
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()

