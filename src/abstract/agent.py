from abstract.state import State
import uuid


class Agent(object):
    def __init__(self, *args, **kwargs):
        assert isinstance(kwargs['identifier'], int)
        assert kwargs['identifier'] >= 0
        self._identifier = kwargs['identifier']

        if 'name' in kwargs:
            self._name = kwargs['name']
        else:
            self._name = uuid.uuid4()
        self._name = str(self._name)

        self._learning = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> int:
        return self._identifier

    def policy(self, state: State, **kwargs):
        raise NotImplementedError()

    @property
    def learning(self) -> bool:
        return self._learning

    def disable_learning(self) -> None:
        self._learning = False

    def enable_learning(self) -> None:
        self._learning = True

    def toggle_learning(self) -> None:
        self._learning = not self._learning
