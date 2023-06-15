class DebugWrapper:
    """
    Wrap a device and print every function call and arguments
    """

    def __init__(self, dev):
        self.dev = dev

    def __getattr__(self, name):

        def wrapper(*args, **kwargs):

            def toHex(arg):
                if isinstance(arg, int):
                    return hex(arg)
                elif isinstance(arg, bytearray):
                    return 'bytearray'
                else:
                    return arg

            # print args in hex
            print(
                name,
                *map(toHex, args),
                **kwargs,
            )
            return getattr(self.dev, name)(*args, **kwargs)

        return wrapper