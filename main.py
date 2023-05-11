import argparse
import json

from src import run_env

parser = argparse.ArgumentParser()

parser.add_argument("--config_file_path_for_environment", '-ec', type=str, required=False,
                    help="A json file containing enviroment configuration: ")

parser.add_argument("--config_file_path_for_agent", '-ac', type=str, required=False,
                    help="A json file containing enviroment configuration: ")

args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config_file_path_for_environment) as f:
        env_config = json.load(f)

    with open(args.config_file_path_for_agent) as f:
        agent_config = json.load(f)

    run_env(env_config, agent_config)