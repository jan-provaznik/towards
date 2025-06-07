# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#

import sys
import tqdm

import time
import pickle
import zstandard as zstd
import numpy as np

# Helpers: timing execution
#

class Stopwatch:
    def __enter__ (self, * args):
        self._tic = time.time()
    def __exit__ (self, * args):
        self._toc = time.time()
    def __call__ (self):
        return float(self)
    def __float__ (self):
        return float(self._toc - self._tic)
    def __str__ (self):
        return f'Execution took {float(self)} seconds.'

# Helpers: transformation decorator
#

def transform (transformer):
    def decorator (function):
        def wrapper (* pargs, ** kargs):
            return transformer(function(* pargs, ** kargs))
        return wrapper
    return decorator

# Helpers: compressed pickles
#

def zstd_pickle_load (path):
    with zstd.open(path, 'rb') as file:
        return pickle.load(file)

def zstd_pickle_dump (path, what):
    with zstd.open(path, 'wb') as file:
        pickle.dump(what, file)

def zstd_pickle_reader (path : str):
    with zstd.open(path, 'rb') as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                return

# Helpers: array do splits
#

def array_split_blocks (array, block_size, axis = 0):
    array = np.swapaxes(array, axis, 0)
    for offset in range(0, array.shape[0], block_size):
        block = array[offset:(offset + block_size)]
        yield np.swapaxes(block, axis, 0)

# Helpers: callable wrappers
#

class starcall:
    def __init__ (self, callable):
        self._callable = callable

    def __call__ (self, task):
        return self._callable(* task)

class taskwrap:
    def __init__ (self, callable, * head_args):
        self._callable = callable
        self._head_args = head_args
    def __call__ (self, task_spec):
        task_head, task_args = task_spec
        with (task_time := Stopwatch()):
            task_data = self._callable(* self._head_args, * task_args)
        return task_time(), task_spec, task_data

# Helpers: dispatchers
#

def make_tuple_like (what):
    if hasattr(what, '__iter__'):
        return what
    return (what, )

def make_task_spec (zspace, i1, i2, * tail_args):
    return (i1, i2), (zspace[i1], zspace[i2], * tail_args)

def master_target_dispatcher (zspace, pool, 
    worker, 
    target_name, 
    target_tail_list,
    target_name_tail):

    # Wrap me like a burrito.
    zshape = zspace.size, zspace.size
    result = np.zeros(shape = (* zshape, 6), dtype = np.float64)

    for task_tail in target_tail_list:
        task_tail = make_tuple_like(task_tail)
        task_list = [ make_task_spec(zspace, * ix, * task_tail) for ix in np.ndindex(zshape) ]
        task_size = len(task_list)

        file_tail = target_name_tail.format(* task_tail)
        file_name = f'{target_name}_{file_tail}'

        print(f'Processing {file_name}')
        with make_tqdm_progress(task_size) as progress:
            for task_pack in pool.map(worker, task_list, unordered = 1):
                task_exec, task_spec, task_data = task_pack
                task_head, task_args = task_spec
                result[task_head] = task_data
                progress.update(1)

        # with (runtime := Stopwatch()):
        #     for task_pack in pool.map(worker, task_list, unordered = 1):
        #         task_exec, task_spec, task_data = task_pack
        #         task_head, task_args = task_spec
        #         result[task_head, ...] = task_data
        # print(f'=> {target_name} with {task_tail} took {runtime():.4f} total')

        file_path = f'result/{file_name}.pickle.zstd'
        zstd_pickle_dump(file_path, result)

def make_tqdm_progress (total):
    return tqdm.tqdm(
        bar_format = '[{bar}] ({n_fmt:4} of {total_fmt:4}) took {elapsed_s:8.3f}',
        total = total,
        ascii = 0, 
        ncols = 80)

# Helpers: unit conversion
#

def sq2db (v):
    return v * (20.0 * np.log10(np.e))
def db2sq (v):
    return v / (20.0 * np.log10(np.e))

