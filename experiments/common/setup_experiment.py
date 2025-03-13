import os, sys, json
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import dataclasses
from stable_baselines3.common.logger import configure
import time
from datetime import datetime


def setup_experiment(exp_name, options: dataclass):
    """Parse params, setup a directory, and map std outputs to files"""
    parser = ArgumentParser()
    timestr = datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    parser.add_argument('--log_dir', default=f"logs/{exp_name}", type=str, help="Output Folder path")
    parser.add_arguments(options, 'params')
    args = parser.parse_args()
    args = parser.parse_args()
    args.log_dir = f"{args.log_dir}_{timestr}"
    
    log_dir = args.log_dir
    print("Logging to", log_dir)
    sys.stdout.flush()

    os.makedirs(log_dir, exist_ok=True)
    # save cmd args
    with open(f'{log_dir}/commandline_args.txt', 'w') as f:
        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                return super().default(o)
        data = {**args.__dict__, "experiment_name": exp_name}
        json.dump(data, f, indent=2, cls=EnhancedJSONEncoder)

    sys.stdout = open(log_dir+"/log.txt", "w")
    # sys.stderr = open(log_dir+"/error_log.txt", "w")
    return args

def get_value_logger(logger_folder):
    return configure(logger_folder, ["stdout", "csv", "tensorboard"])
    
def flush_logs():
    sys.stdout.flush()