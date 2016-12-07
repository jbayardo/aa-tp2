from abstract.state import State
from impl.four_row_agent import FourRowAgent
from impl.four_row_state import FourRowState
import random


class LearningMatch(object):
    def __init__(self, teacher: FourRowAgent, student: FourRowAgent):
        self.teacher = teacher
        self.student = student
        assert self.teacher.identifier != self.student.identifier

    def train_single_match(self, episode: int, state: State):
        self.student.enable_learning()
        self.teacher.enable_learning()

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
            action = agent.policy(state, **{'episode': episode, 'turns': turns})
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

        return state, turns

    def play_single_match(self, state: State):
        self.student.disable_learning()
        self.teacher.disable_learning()

        # Generate what the order of playing will be. This is to avoid having the learning agent only know how to play
        # from the beginning set of states or similar.
        players = [self.student, self.teacher]
        random.shuffle(players)

        # Keep track of numbers
        turn = 0
        turns = 0
        agent = players[turn]

        while not state.is_terminal:
            action = agent.policy(state)
            state = state.execute(agent, action)

            turn = (turn + 1) % 2
            turns += 1
            agent = players[turn]

        # This simulates one last turn in which the loosing agent realises that it lost and adjusts accordingly.
        turns += 1

        return state, turns

    def train_many_matches(self, run_id: str, episodes: int):
        training_samples = []
        running_samples = []

        episodes += 1
        for episode in range(1, episodes):
            if episode % 1000 == 0:
                print('{} Playing episode {} out of {}'.format(run_id, episode, episodes - 1))

            start_state = FourRowState.generate()

            output, turns = self.train_single_match(episode, start_state)

            training_samples.append({
                'episode_number': episode,
                'winner': output.winner,
                'turns': turns
            })

            output, turns = self.play_single_match(start_state)

            running_samples.append({
                'episode_number': episode,
                'winner': output.winner,
                'turns': turns
            })

        return training_samples, running_samples
