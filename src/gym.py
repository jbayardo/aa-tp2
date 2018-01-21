import logging
import os
from collections import defaultdict
from typing import List

import pickle

import settings
from rl.agent import Agent


class EpochFilter(logging.Filter):
    def __init__(self, epoch):
        super().__init__()

        self._epoch = str(epoch)

    def filter(self, record):
        record.epoch = self._epoch
        return True


class Gym(object):
    def __init__(self, environment_class, step_model_class, *args):
        assert len(args) > 0

        self._environment_class = environment_class
        self._step_model_class = step_model_class
        self._agents = list(args)
        self._agents_evaluation = [agent.deploy() for agent in self._agents]

    def train(self,
              experiment_id: str, results_path: str,
              epochs: int, training_episodes: int, validation_episodes: int):
        assert os.path.exists(results_path)
        assert epochs > 0
        assert training_episodes >= 0
        assert validation_episodes > 0

        log = logging.getLogger(experiment_id)

        checkpoints_path = os.path.join(results_path, 'checkpoints')
        os.makedirs(checkpoints_path, exist_ok=True)

        log.info('Model checkpoints stored at %s', checkpoints_path)
        log.info('Training starting')

        samples = {
            'training': defaultdict(list),
            'validation': defaultdict(list)
        }

        for epoch in range(1, epochs + 1):
            epoch_filter = EpochFilter(epoch)
            log.addFilter(epoch_filter)

            log.info('Training starting')
            for episode in range(1, training_episodes + 1):
                if (episode - 1) % settings.EPISODE_LOG_EVERY_N == 0:
                    log.info('Training episode %d/%d', episode - 1, training_episodes)

                metadata = self._run(epoch, episode, self._agents)
                samples['training'][epoch].append(metadata)
            log.info('Training finished. Checkpointing starting')

            for agent in self._agents:
                checkpoint_file_name = 'agent{0}_epoch{1}.chkp'.format(agent.identifier, epoch)
                checkpoint_file_path = os.path.join(checkpoints_path, checkpoint_file_name)

                with open(checkpoint_file_path, 'wb') as checkpoint_handle:
                    pickle.dump(agent, checkpoint_handle)

                log.info('Agent %s has been saved to %s', agent.identifier, checkpoint_file_path)

            log.info('Checkpoint finished. Validation starting')
            for episode in range(1, validation_episodes + 1):
                if (episode - 1) % settings.EPISODE_LOG_EVERY_N == 0:
                    log.info('Validation episode %d/%d', episode - 1, validation_episodes)

                metadata = self._run(epoch, episode, self._agents_evaluation)
                samples['validation'][epoch].append(metadata)
            log.info('Validation finished')

            log.removeFilter(epoch_filter)

        log.info('Training finished')

        return samples

    def _run(self, epoch: int, episode: int, agents: List[Agent]):
        start_environment = self._environment_class.generate()
        match = self._step_model_class(*agents)
        end_environment, metadata = match.play(start_environment, **{
            'epoch': epoch,
            'episode': episode
        })

        q_averages = {}
        for agent in self._agents:
            if hasattr(agent, '_q_average'):
                q_averages[agent.identifier] = agent._q_average()

        metadata.update({
            'q_averages': q_averages
        })

        return metadata
