from abstract.action import Action


class FourRowAction(Action):
    def __init__(self, column: int):
        self._column = column

    @property
    def column(self) -> int:
        return self._column

    # The following are required for dictionary usage
    def __hash__(self):
        return hash(self._column)

    def __eq__(self, other):
        return self._column == other._column

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)