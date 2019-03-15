import random

from rl.agent.baseagent import BaseAgent
from rl.environment.goalenvironment import GoalEnvironment


class TurnBasedMatch(object):
    def __init__(self, left: BaseAgent, right: BaseAgent):
        assert left is not None
        assert right is not None
        assert left.identifier != right.identifier

        self._left = left
        self._right = right

    def play(self, environment: GoalEnvironment, **metadata):
        # Generate what the order of playing will be. This ensures agents have both knowledge in how to play when the
        # game just started, and when the game already has at least one move in
        players = [self._left, self._right]
        random.shuffle(players)

        previous_action, previous_agent, previous_environment = None, None, None

        # Keep track of numbers
        turn = 0
        agent = players[turn]

        metadata.update({
            'start': players[0].identifier,
            'turns': 0
        })

        while not environment.is_terminal:
            action = agent.policy(environment, **metadata)
            new_environment = environment.execute(agent, action)

            # Give the previous agent feedback on its actions
            if previous_agent is not None and previous_environment is not None and previous_action is not None:
                previous_agent.feedback(previous_environment, previous_action, new_environment, **metadata)

            previous_action, previous_agent, previous_environment = action, agent, environment
            environment = new_environment

            turn = (turn + 1) % 2
            metadata['turns'] += 1
            agent = players[turn]

        # This simulates one last turn in which the loosing agent realises that it lost and adjusts accordingly
        metadata['turns'] += 1

        if previous_agent is not None and previous_environment is not None and previous_action is not None:
            previous_agent.feedback(previous_environment, previous_action, environment, **metadata)

        metadata['winner'] = environment.winner

        return environment, metadata
