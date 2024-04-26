# License: BSD 3 clause


class PathEvent():
    """
    Class for an event in a finite path of a point processes.

    """

    def __init__(self, time=0., dim=0, mark=None, state=None):
        self.time = time
        self.dim = dim
        self.mark = mark
        self.state = state
