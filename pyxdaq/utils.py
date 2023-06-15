from subprocess import getstatusoutput


def git_rev_parse(rev):
    status, h = getstatusoutput(f'git rev-parse {rev}')
    if status != 0:
        raise RuntimeError(f'Failed to get git rev-parse {rev}')
    return h


def git_version_diff():
    status, diff = getstatusoutput('git diff HEAD')
    if status != 0:
        raise RuntimeError('Failed to get git diff HEAD')
    h = git_rev_parse('HEAD')
    return h + ('-dirty' if len(diff) > 0 else ''), diff


class DebugWrapper:
    """
    Wrap a device and print every function call and arguments
    """

    def __init__(self, dev, outfn=print):
        self.dev = dev
        self.outfn = outfn

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
            self.outfn(
                name,
                *map(toHex, args),
                **kwargs,
            )
            return getattr(self.dev, name)(*args, **kwargs)

        return wrapper