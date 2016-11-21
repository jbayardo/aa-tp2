from abstract.agent import Agent
from impl.four_row_agent import FourRowAgent
from impl.four_row_state import FourRowState
import random
import numpy as np


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
        agent = players[turn]

        while not state.is_terminal:
            action = agent.policy(state)
            new_state = state.execute(agent, action)

            #print("TURN", turns,"BY PLAYER", agent.identifier, "PLAYING ACTION", action)
            #new_state.print_board()

            # Give the previous agent feedback on its actions
            if previous_agent is not None and previous_state is not None and previous_action is not None and hasattr(
                    previous_agent, 'feedback'):
                previous_agent.feedback(previous_state, previous_action, new_state, learning_rate, discount_factor)

            previous_action = action
            previous_agent = agent
            previous_state = state
            state = new_state

            turn = (turn + 1) % 2
            turns += 1
            agent = players[turn]

        return state.winner, turns

    def train_many_matches(self, episodes: int, learning_rate: float, discount_factor: float) -> FourRowAgent:
        samples = []

        for episode in range(1, episodes + 1):
            if episode % 100 == 0:
                print('Playing episode', episode)
            winner, turns = self.train_single_match(learning_rate, discount_factor)

            # Compute average value for the Q
            vals = np.array(list(self.learning_player._q_definition.values()), dtype=np.float64)
            avgq = np.mean(vals)
            devq = np.std(vals)

            samples.append({
                'episode_number': episode,
                'winner': winner,
                'turns': turns,
                'avg_q': avgq,
                'std_q': devq
            })

        return self.learning_player, samples
