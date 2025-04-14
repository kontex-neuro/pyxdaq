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

