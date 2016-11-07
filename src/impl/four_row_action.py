from abstract.action import Action


class FourRowAction(Action):
    def __init__(self, column: int):
        self._column = column

    @property
    def column(self) -> int:
        return self._column
