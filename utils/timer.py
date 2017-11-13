import time

__all__ = ['Timer']

class Timer(object):
    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self,*args):
        dt = time.time()-self.t0
        print('Time Elapse = %.4fs'%dt)
