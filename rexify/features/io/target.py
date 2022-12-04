_SUPPORTED_TARGETS = ["user", "item"]


class HasTargetInput:
    def __init__(self, target: str):
        self._target = target

    @property
    def target(self):
        return self._target

    @staticmethod
    def _validate_target(target: str):
        if target not in _SUPPORTED_TARGETS:
            raise ValueError(f"Target {target} not sypported")
