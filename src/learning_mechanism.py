from abstract.agent import Agent
from impl.four_row_agent import FourRowAgent
from impl.four_row_state import FourRowState
import random


class LearningMatch(object):
    def __init__(self, teaching_agent: Agent, student_player: FourRowAgent):
        self.teaching_player = teaching_agent
        self.learning_player = student_player
        assert self.teaching_player.identifier != self.learning_player.identifier

    def train_single_match(self, learning_rate: float, discount_factor: float) -> FourRowState:
        if not self.learning_player.learning:
            self.learning_player.toggle_learning()

        state = FourRowState.generate()

        # Generate what the order of playing will be. This is to avoid having the learning agent only know how to play
        # from the beginning set of states or similar.
        players = [self.learning_player, self.teaching_player]
        random.shuffle(players)

        previous_action = None
        previous_agent = None
        previous_state = None

        # Keep track of numbers
        turn = 0
        turns = 0

        while not state.is_terminal:
            agent = players[turn]
            action = agent.policy(state)
            new_state = state.execute(agent, action)

            # Give the previous agent feedback on its actions
            if previous_agent is not None and previous_state is not None and previous_action is not None and hasattr(previous_agent, 'feedback'):
                previous_agent.feedback(previous_state, previous_action, new_state, learning_rate, discount_factor)

            previous_action = action
            previous_agent = agent
            previous_state = state
            state = new_state

            turn = (turn + 1) % 2
            turns += 1

        return state.winner, turns

    def train_many_matches(self, episodes: int, learning_rate: float, discount_factor: float) -> FourRowAgent:
        samples = []

        for episode in range(1, episodes + 1):
            if episode % 10 == 0:
                print('Playing episode', episode)
            winner, turns = self.train_single_match(learning_rate, discount_factor)

            # Compute average value for the Q
            avgq = 0.0
            totq = 0.0
            for key in self.learning_player._q_definition:
                avgq += self.learning_player._q_definition[key]
                totq += 1.0
            avgq /= totq

            samples.append({
                'episode_number': episode,
                'winner': winner,
                'avg_q': avgq
            })

        return self.learning_player, samples
