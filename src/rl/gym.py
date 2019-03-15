import logging
import os
from collections import defaultdict
from typing import List

import pickle

import settings
from rl.agent.baseagent import BaseAgent


class EpochFilter(logging.Filter):
    def __init__(self, epoch):
        super().__init__()

        self._epoch = str(epoch)

    def filter(self, record):
        record.epoch = self._epoch
        return True


class Gym(object):
    def __init__(self, environment_class, environment_generator, step_model_class, *args):
        assert len(args) > 0

        self._environment_class = environment_class
        self._environment_generator = environment_generator
        self._step_model_class = step_model_class
        self._agents = list(args)
        self._agents_evaluation = [agent.deploy() for agent in self._agents]

    def train(self,
              experiment_id: str, results_path: str,
              epochs: int, training_episodes: int, validation_episodes: int,
              gather_statistics: bool):
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
            'episodic': {
                'training': defaultdict(list),
                'validation': defaultdict(list)
            },
            'epochal': {}
        }

        for epoch in range(1, epochs + 1):
            epoch_filter = EpochFilter(epoch)
            log.addFilter(epoch_filter)

            log.info('Training for epoch %d starting', epoch)
            for episode in range(1, training_episodes + 1):
                if (episode - 1) % settings.EPISODE_LOG_EVERY_N == 0:
                    log.info('Epoch %d. Training episode %d/%d', epoch, episode - 1, training_episodes)

                generated_metadata = self._run(epoch, episode, self._agents, gather_statistics)
                samples['episodic']['training'][epoch].append(generated_metadata)
            log.info('Training for epoch %d finished. Checkpointing starting', epoch)

            for agent in self._agents:
                checkpoint_file_name = 'agent{0}_epoch{1}.chkp'.format(agent.identifier, epoch)
                checkpoint_file_path = os.path.join(checkpoints_path, checkpoint_file_name)

                with open(checkpoint_file_path, 'wb') as checkpoint_handle:
                    pickle.dump(agent, checkpoint_handle)

                log.info('Agent %s has been saved to %s', agent.identifier, checkpoint_file_path)

            log.info('Checkpoint finished. Validation for epoch %d starting', epoch)
            for episode in range(1, validation_episodes + 1):
                if (episode - 1) % settings.EPISODE_LOG_EVERY_N == 0:
                    log.info('Epoch %d. Validation episode %d/%d', epoch, episode - 1, validation_episodes)

                generated_metadata = self._run(epoch, episode, self._agents_evaluation, gather_statistics)
                samples['episodic']['validation'][epoch].append(generated_metadata)
            log.info('Validation for epoch %d finished', epoch)

            if gather_statistics:
                statistics = {}
                for agent in self._agents:
                    stats = agent.epoch_statistics()
                    if stats is not None:
                        statistics[agent.identifier] = stats

                samples['epochal'][epoch] = statistics

            log.removeFilter(epoch_filter)

        log.info('Training finished')

        return samples

    def _run(self, epoch: int, episode: int, agents: List[BaseAgent], gather_statistics: bool):
        start_environment = self._environment_generator()
        match = self._step_model_class(*agents)
        end_environment, metadata = match.play(start_environment, **{
            'epoch': epoch,
            'episode': episode
        })

        assert 'epoch' not in metadata or metadata['epoch'] == epoch
        assert 'episode' not in metadata or metadata['episode'] == episode

        if gather_statistics:
            statistics = {}
            for agent in self._agents:
                stats = agent.episode_statistics()
                if stats is not None:
                    statistics[agent.identifier] = stats

            metadata.update({
                'statistics': statistics
            })

        return metadata
