from abstract.agent import Agent
from impl.four_row_agent import FourRowAgent
from impl.four_row_state import FourRowState
import random


class LearningMatch(object):
    def __init__(self, teaching_agent: Agent, student_player: FourRowAgent):
        self.teacher = teaching_agent
        self.student = student_player
        assert self.teacher.identifier != self.student.identifier

    def train_single_match(self, episode: int) -> FourRowState:
        assert self.student.learning
        state = FourRowState.generate()

        # Generate what the order of playing will be. This is to avoid having the learning agent only know how to play
        # from the beginning set of states or similar.
        players = [self.student, self.teacher]
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

            # Give the previous agent feedback on its actions
            if previous_agent is not None and previous_state is not None and previous_action is not None and hasattr(
                    previous_agent, 'feedback'):
                previous_agent.feedback(previous_state, previous_action, new_state, episode, turns)

            previous_action = action
            previous_agent = agent
            previous_state = state
            state = new_state

            turn = (turn + 1) % 2
            turns += 1
            agent = players[turn]

        # This simulates one last turn in which the loosing agent realises that it lost and adjusts accordingly.
        turns += 1

        if previous_agent is not None and previous_state is not None and previous_action is not None and hasattr(
                previous_agent, 'feedback'):
            previous_agent.feedback(previous_state, previous_action, state, episode, turns)

        return state.winner, turns

    def train_many_matches(self, episodes: int) -> FourRowAgent:
        samples = []

        for episode in range(1, episodes + 1):
            if episode % 1000 == 0:
                print('Playing episode', episode)
                self.student.perturb()

            winner, turns = self.train_single_match(episode)

            samples.append({
                'episode_number': episode,
                'winner': winner,
                'turns': turns
            })

        return self.student, samples
