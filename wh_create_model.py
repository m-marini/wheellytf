import argparse
import logging

from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, MockRobotEnv

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
        help='the path of agent model (default=agent.json)'
    )
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    args = init_argparse().parse_args()

    env1:MockRobotEnv = Environment.create(environment=MockRobotEnv)

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )

    agent:Agent = Agent.create(
        agent=args.agent,
        environment=environment
    )
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()
