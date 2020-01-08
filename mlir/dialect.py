""" MLIR Dialect representation. """


class Dialect(object):
    def __init__(self, path: str, name: str):
        # Load lark file
        with open(path, 'r') as fp:
            self.contents = fp.read()

        # TODO: Validate contents?

        self.name = name
        self.path = path
