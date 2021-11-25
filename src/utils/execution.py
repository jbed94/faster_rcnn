import time


def timeit(f):
    def timed(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()

        duration = end - start

        print(f'Function: {f.__name__} | took: {duration:2.4f}s')

        return result

    return timed
