"""Run the training of actor critic agent in a sequence mock environment"""
import logging
from tensorforce.agents import Agent
from tensorforce.environments import Environment


def main():
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
    logging.info("Started")
    #env = SequenceMockEnv(num_states=4)
    env = Environment.create(
        environment='tests.mocks1.SequenceMockEnv',
        num_states=4,
        max_episode_timesteps=100
    )
    agent: Agent = Agent.create(
        agent='ac',
        environment=env,
        batch_size=10,
        exploration=3e-2,
        summarizer=dict(
            directory='data/summaries',
            summaries='all'
        ))
    for i in range(1000):
        states = env.reset()
        returns = 0
        counts = 0
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(reward=reward, terminal=terminal)
            returns += reward
            counts += 1
        logging.debug(f"avg ret {i} {returns / counts}")

    logging.info("Completed")


if __name__ == '__main__':
    main()
