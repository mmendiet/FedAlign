'''
Custom non-daemonic Pool class for all python version
Code adapted from https://github.com/LoLab-VU/PyDREAM/pull/17/commits
'''
import multiprocessing
import multiprocessing.pool

class NonDaemonMixin(object):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


from multiprocessing import context


# Exists on all platforms
class NonDaemonSpawnProcess(NonDaemonMixin, context.SpawnProcess):
    pass


class NonDaemonSpawnContext(context.SpawnContext):
    Process = NonDaemonSpawnProcess


_nondaemon_context_mapper = {
    'spawn': NonDaemonSpawnContext()
}


class DreamPool(multiprocessing.pool.Pool):
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        if context is None:
            context = multiprocessing.get_context()
        context = _nondaemon_context_mapper[context._name]
        super(DreamPool, self).__init__(processes=processes,
                                        initializer=initializer,
                                        initargs=initargs,
                                        maxtasksperchild=maxtasksperchild,
                                        context=context)