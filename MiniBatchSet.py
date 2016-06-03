import importlib
import csv
import linecache
import os
import json
import sys
from collections import Iterator
from random import shuffle as shuffle_list
from math import ceil
from multiprocessing.pool import ThreadPool
from threading import current_thread
from datetime import datetime

if not importlib.find_loader('numpy') is None:
    import numpy as np
else:
    print("WARNING! Numpy was not installed on the vm. Numerical functionalities will not work.")

class MiniBatchSet(Iterator):
    """ A class to generate mini-batches of data from a set of csv files. 
    
        parameters:
            file_paths_or_folder: A directory path (str) or list of file paths (list<str>)
            batch_size: The size of min-batches
            encoding: The encoding of csv file data
            delimiter: The delimiter in the csv format
            surrounding: In case values in csv are surrounded by a special character
            max_line_cache_size: Clears the line cache when the cache reaches the max size (TODO:This needs to be replaced with an implementation to dump the old entries when the cache reaches the max)
            random_distribution: If set to true, it will use randomization to pick the indices, otherwise it will keep a list of indices and shuffles it in each reset (This method ensures true unified distribution, however could take up large amount of memory relative to the ).
            pool_async: If set to true, will pool the next instance of batch in memory on each fetch.

        usage:
            mb = MiniBatchSet(path)

            mb[12]

            len(mb)

            next(mb)

            for mini_batch in mb:
                mb[0:2,1:5]
    """

    def __init__(self, batch_size=None, file_paths_or_folder=None, encoding='utf8', delimiter=",", surrounding="\"", max_line_cache_size = 0, random_distribution=True, pool_async=True, hold_in_memory=False, np_dtype=np.float64):
        super().__init__()
        #TODO: Handle using, saving and restoring the dtype
        self._encoding = encoding
        self._delimiter = delimiter
        self._batch_size = batch_size
        self._surrounding = surrounding
        self._max_line_cache_size = max_line_cache_size
        self._random_distribution = random_distribution
        self._pool_async = pool_async
        self._thread_pool = ThreadPool(processes=1)
        self._hold_in_memory = hold_in_memory

        self._files = []
        self._data = None
        self._indices = []
        self._total_columns= None
        self._total_lines = 0
        self._next_counter = 0
        self._line_cache_counter = 0
        self._fetch_thread = None

        if not file_paths_or_folder is None:
            self._load(file_paths_or_folder)

    _files = []
    _indices = []
    _encoding = "utf8"
    _delimiter = ","
    _surrounding="\""
    _total_columns= None
    _total_lines = 0
    _batch_size = 0
    _next_counter = 0
    _max_line_cache_size = 0
    _line_cache_counter = 0
    _random_distribution = True
    _pool_async = True
    _hold_in_memory = False

    _thread_pool = None
    _fetch_thread = None

    def _load(self, file_paths_or_folder):
        if not isinstance(file_paths_or_folder, list) and not isinstance(file_paths_or_folder, str):
            raise TypeError("Parameter file_paths_or_folder must be a folder path or a collection of file paths.")

        if isinstance(file_paths_or_folder, str) and not os.path.exists(file_paths_or_folder):
            raise FileNotFoundError("Given folder path did not exist.")

        if isinstance(file_paths_or_folder, str) and not os.path.isdir(file_paths_or_folder):
            raise NotADirectoryError("Given path was not a directory. Either provide a directory path or a collection of file paths.")

        if isinstance(file_paths_or_folder, str) and \
            os.path.isdir(file_paths_or_folder):
            file_paths = [  os.path.join(file_paths_or_folder, f) \
                            for f in os.listdir(file_paths_or_folder) \
                            if os.path.isfile(os.path.join(file_paths_or_folder, f))]
        else:
            file_paths = file_paths_or_folder

        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError("The requested file not found on the given path: " + file_path)
            
            shape = self._get_shape(file_path, self._delimiter, self._encoding)
            if self._total_columns and self._total_columns != shape[1]:
                raise IndexError("The dimensions of the files do not match. Breaking on the first occurance: \"" + file_path + "\"")

            self._total_columns = shape[1]
            self._files.append( {\
                                'file_path' : file_path,\
                                'shape': shape,\
                                'index_range': (self._total_lines, self._total_lines + shape[0] - 1)\
                                })
            
            self._total_lines += shape[0]
            print("File %s added to source list."%os.path.basename(file_path))

        if (not self._random_distribution):
            self._indices = list(range(self._total_lines))
        elif not np:
            raise ImportError("Numpy was not detected. Random distribution requires numpy.random package.")

        if self._batch_size is None:
            self._batch_size = self._total_lines

        if self._hold_in_memory:
            self._load_on_memory()
    
    def _load_on_memory(self):
        print("Manager was set to fetch the data in the memory, loading all the data")
        self._data = {}
        total_lines_read = 0
        start_time = datetime.now()
        try:
            for file in self._files:
                file_path = file['file_path']
                lines_in_file = np.loadtxt(file_path, delimiter=self._delimiter, dtype=np.float64)
                self._data[file_path] = lines_in_file
                total_lines_read += lines_in_file.shape[0]

                if total_lines_read == max(20, ceil(self._total_lines/10)):
                    minutes_passed = ceil((datetime.now() - start_time).total_seconds()/60)
                    eta = ceil(((self._total_lines - total_lines_read) * minutes_passed)/total_lines_read)
                    print("Load completion ETA: %d minute(s)"%eta)
                    print("Read %d lines of total %d"%(total_lines_read, self._total_lines))

                print("Total lines: " + str(file['shape']))
                print("Np size: " + str(self._data[file_path].shape))

            print("Data fully loaded in the memory. Allocated size is %gKB."%(sys.getsizeof(self._data)/1000))
        except MemoryError as m_err:
            print("Failed to load the data on memory due to a memory error: " + str(m_err))
    
    def _get_shape(self, file_path, delimiter, encoding):
        file_row_no = 0
        file_col_no = 0
        with open(file_path, "rU", encoding=encoding) as f:
            file_row_no = sum(1 for x in f)

        if file_row_no <= 0:
            file_col_no = 0
        else:
            row = linecache.getline(file_path, 1)
            linecache.clearcache()
            file_col_no = len(self._array_from_csv_row(row))

        return (file_row_no, file_col_no)

    def _get_line(self, line_no, file_path):
        line = linecache.getline(file_path, line_no)
        self._line_cache_counter += 1

        if (self._line_cache_counter > self._max_line_cache_size):
            linecache.clearcache()
            self._line_cache_counter = 0

        return line

    def _get_row(self, row_no, file_path):
        if self._hold_in_memory:
            if not self._data:
                self._load_on_memory()

            return self._data[file_path][row_no]
        else:
            row = self._get_line(row_no, file_path)
            return self._array_from_csv_row(row)

    def _array_from_csv_row(self, row):
        return row.replace("\n", "").replace(self._surrounding, "").split(self._delimiter)

    def _get_batch(self, index=None):
        if not self._random_distribution:
            start = index * self._batch_size
            if start > self._total_lines:
                raise IndexError("Given batch index was out of the acceptable range. Acceptable values are 0-indexed numbers less than the number of batches.")

            end = start + self._batch_size
            sequence = self._indices[start: min(end, self._total_lines)]
        else:
            sequence = np.random.random_integers(0, self._total_lines - 1, self._batch_size).tolist()

        lines = {indx: None for indx in sequence}
        indices = sequence[:]
        indices.sort()
        file_gen = (file for file in self._files)
        #print("[BATCH] %d Started getting the code"%current_thread().ident)
        file = next(file_gen, None)
        while indices:
            if file is None:
                raise OverflowError("Indices in the MiniBatchSet do not match the file set.")

            indx = indices.pop(0)
            range = file["index_range"]

            if indx < range[0]:
                raise ValueError("Index to fetch from file was less than file index range.")

            while indx > range[1]:
                file = next(file_gen, None)
                range = file["index_range"]

            lines[indx] = self._get_row(indx + 1 - range[0], file["file_path"])
            #print("[BATCH] %d Got a row"%current_thread().ident)

        file_gen.close()
        return [lines[indx] for indx in sequence]
    
    def _retrieve_next(self):
        start = self._next_counter * self._batch_size
        if start > self._total_lines:
            #If end of batch sequence is reached, reset 
            self.reset()

        if not self._pool_async:
            return self._get_batch(self._next_counter)
            self._next_counter += 1

        if self._fetch_thread is None:
            self._fetch_thread = self._thread_pool.apply_async(self._get_batch, (self._next_counter,))
            self._next_counter += 1
            batch = self._get_batch(self._next_counter)
            self._next_counter += 1
            return batch
        else:
            minibatch = self._fetch_thread.get()
            self._fetch_thread = self._thread_pool.apply_async(self._get_batch, (self._next_counter,))
            self._next_counter += 1
            return minibatch
        

    def get_files(self):
        return self._files

    def get_total_lines(self):
        return self._total_lines

    def get_total_columns(self):
        return self._total_columns

    def reset(self):
        self._next_counter = 0
        if not self._random_distribution:
            shuffle_list(self._indices)

    def transform_data(self, action, output_path, *running_vars):
        if not output_path is None and not os.path.exists(output_path):
                os.makedirs(output_path)

        for file in self._files:
            file_path = file["file_path"]
            file_name = os.path.basename(file_path)
            #print("[START] %s on file %s."%(str(action), file_name))

            with open(file_path, 'r', encoding=self._encoding) as f_read:
                reader = csv.reader(f_read, delimiter=self._delimiter)
                f_write = None
                try:
                    if not output_path is None:
                        write_path = os.path.join(output_path, file_name)
                        f_write = open(write_path, 'w', encoding=self._encoding)
                        writer = csv.writer(f_write, quoting=csv.QUOTE_NONE)

                    for index, row in enumerate(reader):                        
                        if not output_path is None:
                            data, running_vars = action(row, *running_vars)
                            if not data is None:
                                writer.writerow(data)
                        else:
                            running_vars = action(row, *running_vars)
                #except Exception as ex:
                    #raise Exception("Failed to process {0} due to: {1}.".format(file_name, ex))
                finally:
                    if f_write:
                        f_write.close()

            print("[DONE] %s on file %s."%(str(action), file_name))
        return running_vars

    def save_snapshot(self, snapshot_path, force_overwrite=True):
        if os.path.exists(snapshot_path):
            if force_overwrite:
                os.remove(snapshot_path)
            else:
                raise FileExistsError("The snapshot file existed on the given path. To force overwrite the file set the parameter force_overwrite to True.")

        with open(snapshot_path, 'w') as f:
            content = { "files":self._files, \
                        "indices":self._indices,\
                        "total_lines":self._total_lines,\
                        "random_distribution":self._random_distribution,\
                        "max_line_cache_size":self._max_line_cache_size,\
                        "batch_size":self._batch_size,\
                        "next_counter":self._next_counter,\
                        "encoding":self._encoding,\
                        "delimiter":self._delimiter,\
                        "pool_async":self._pool_async\
                        }
            json.dump(content, f)

    def is_random(self):
        return self._random_distribution

    def normalize(self, output_path, indices=None):
        if self._total_lines <= 1:
            print("Skipping the normalization since the set includes 1 or less lines.")
            return

        if not np:
            raise ImportError("Numpy was not detected. This operation requires numpy for calculations.")

        indices = indices or [i for i in range(self._total_columns)]
        sums = np.array([0.0] * len(indices),np.float64)
        deviations = np.array([0.0] * len(indices),np.float64)
        mu = np.float64(0.0)
        sigma = np.float64(0.0)

        def get_sum_action(data, sums):
            data = np.array([data[i] for i in indices], dtype=np.float64)
            sums = np.add(sums, data)

            #if any(val for val in sums if val == sys.float_info.max):
            #    raise OverflowError("The sum exceeds the maximum float size for sums {0} on row {1}. Try deducting the values to reduce the sum below the max float size of {2}".format(sums.tolist(), data, sys.float_info.max))
            return [sums]


        def get_deviations_action(data, mu, deviations):
            data = np.array([data[i] for i in indices], dtype=np.float64)
            deviations = np.add(deviations, np.power(np.subtract(data, mu), 2))

            #if any(val for val in deviations if val == sys.float_info.max):
            #    raise OverflowError("The devations sum exceeds the maximum float size for deviations {0} on row {1}. Try deducting the values to reduce the sum below the max float size of {2}".format(deviations.tolist(), data, sys.float_info.max))

            return [mu, deviations]

        def produce_normalized_files_action(data, mu, sigma):
            p_data = np.array([data[i] for i in indices], dtype=np.float64)
            mu = np.array([0.0 if np.float64(sigma[i]) == 0.0 else mu[i] for i in range(len(sigma))])             #Ignore single value lists
            sigma = np.array([np.float64(1.0) if sigma[i] == np.float64(0.0) else sigma[i] for i in range(len(sigma))])        #Ignore single value lists

            #print("mu:" + str(mu))
            #print("sigma:" + str(sigma))
            #print("Before:" + str(data))
            p_data = np.divide(np.subtract(p_data, mu), sigma)
            for i in range(len(indices)):
                index = indices[i]
                p = p_data[i]
                data[index] = p 
            #print("After:" + str(data))
            return data, [mu, sigma]

        running1 = self.transform_data(get_sum_action, None, sums)
        sums = running1[0]
        mu = np.divide(sums, self._total_lines)

        mu, deviations = self.transform_data(get_deviations_action, None, mu, deviations)
        sigma = np.sqrt(np.divide(deviations, self._total_lines - 1))
        
        #print("mean=" + str(mu))
        #print("std=" + str(sigma))
        self.transform_data(produce_normalized_files_action, output_path, mu, sigma)

    def map_features(self, output_path, mapping):
        pass

    @staticmethod
    def from_snapshot(snapshot_path):
        content = None
        with open(snapshot_path, 'r') as f:
            content = json.load(f)

        bs = MiniBatchSet()
        bs._files = content["files"]
        bs._indices = content["indices"]
        bs._total_lines = content["total_lines"]
        bs._random_distribution = content["random_distribution"]
        bs._max_line_cache_size = content["max_line_cache_size"]
        bs._batch_size = content["batch_size"]
        bs._next_counter = content["next_counter"]
        bs._encoding = content["encoding"]
        bs._delimiter = content["delimiter"]
        bs._pool_async = content["pool_async"] 
        return bs


    def __len__(self):
        return ceil(self._total_lines / self._batch_size)

    def __getitem__(self, the_slices):
        if  the_slices is None or\
            (not isinstance(the_slices, int) and\
            not isinstance(the_slices, slice)):
            raise ValueError("The indices must either be a single index or a slice.")

        if isinstance(the_slices, int):
            return self._get_batch(the_slices)

        result = []
        for batch_index in range(the_slices.start, the_slices.stop):
            result.append(self._get_batch(batch_index))

        return result

    def __eq__(self, other):
        other_files = other.get_files()
        for file_key, file_shape in self._files.items():
            if other_files.get(file_key, None) is None:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return self

    def __next__(self):
        return self._retrieve_next()

