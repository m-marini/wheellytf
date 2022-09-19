"""Creates the initial model of agent for the mock robot environment """

import argparse
import logging
import os
import shutil

from tensorforce import Environment
from tensorforce.agents import Agent
from tensorforce.core.optimizers import TFOptimizer, NaturalGradient

from wheelly.envs import EncodedRobotEnv, MockRobotEnv
from wheelly.optimizers import TDOptimizer

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        #usage="%(prog)s [OPTION] [FILE]...",
        description="Create agent model."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 0.1.0"
    )
    parser.add_argument(
        "-a", "--agent", default='agent.json',
        dest='agent',
        help='the path of agent definition (default=agent.json)'
    )
    parser.add_argument(
        "-f", "--frequency", default=100,
        type=int,
        dest='frequency',
        help='the model saving frequency (default=100)'
    )
    parser.add_argument(
        "-m", "--model", default='models/default',
        dest='model',
        help='the path of agent model (default=models/default)'
    )
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    args = init_argparse().parse_args()

    env1:MockRobotEnv = MockRobotEnv()

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )

    shutil.rmtree(args.model, ignore_errors=True)
    os.makedirs(args.model)
    agent:Agent = Agent.create(
        agent=args.agent,
        environment=environment,
        saver=dict(
            directory=args.model,
            frequency=args.frequency
        )
    )
    agent.close()
    environment.close()
    logging.info(f'Created {args.model} from {args.agent}')

if __name__ == '__main__':
    main()
