import os

COMMON_CONFIG_DIR='config/base/common.py'
MODEL_CONFIG_DIR='config/base/model.py'
REFCOD_CONFIG_DIR='config/base/refcod.py'
ACTIVE_LEARNING_DIR='config/base/active_learning.py'

from utils import Logger
logger = Logger(name='Config', path='/tmp/talnet_preload.log')

class Config:
    def __init__(self, run_cfg: str):
        logger.key_info("Initialize config...")
        self.merge_from_file(COMMON_CONFIG_DIR)
        self.merge_from_file(MODEL_CONFIG_DIR)
        self.merge_from_file(REFCOD_CONFIG_DIR)
        self.merge_from_file(ACTIVE_LEARNING_DIR)
        self.merge_from_file(run_cfg)
        logger.success_info("Config merged from {}.".format(run_cfg))
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
    
    def merge_from_file(self, config_dir: str) -> None:
        assert os.path.isfile(config_dir), f'[-] {config_dir} not found!'
        with open(config_dir, 'r') as f:
            local_vars = {}
            exec(f.read(), {}, local_vars)
            for key, value in local_vars.items():
                setattr(self, key, value)
