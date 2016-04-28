import importlib
import csv
import linecache
import os
from collections import Iterator
from random import shuffle as shuffle_list
from math import ceil

if not importlib.find_loader('numpy') is None:
    import numpy as np

class MiniBatchSet(Iterator):
    """ A class to generate mini-batches of data from a set of csv files. 
    
        parameters:
            file_paths_or_folder: A directory path (str) or list of file paths (list<str>)
            batch_size: The size of min-batches
            encoding: The encoding of csv file data
            delimiter: The delimiter in the csv format
            next_line: Line separator character in the csv format
            surrounding: In case values in csv are surrounded by a special character
            max_line_cache_size: Clears the line cache when the cache reaches the max size (TODO:This needs to be replaced with an implementation to dump the old entries when the cache reaches the max)
            random_distribution:If set to true, it will use randomization to pick the indices, otherwise it will keep a list of indices and shuffles it in each reset (This method ensures true unified distribution, however could take up large amount of memory relative to the ).
    """

    def __init__(self, file_paths_or_folder, batch_size, encoding='utf8', delimiter=",", next_line="\n", surrounding="\"", max_line_cache_size = 0, random_distribution=True):
        super().__init__()
        if  file_paths_or_folder is None or\
            (isinstance(file_paths_or_folder, str) and not os.path.isdir(file_paths_or_folder)) or\
            (not isinstance(file_paths_or_folder, list) and not isinstance(file_paths_or_folder, str)):
            raise FileNotFoundError("Parameter file_paths_or_folder must be a folder path or a collection of file paths.")

        if isinstance(file_paths_or_folder, str) and \
            os.path.isdir(file_paths_or_folder):
            file_paths = [  os.path.join(file_paths_or_folder, f) \
                            for f in os.listdir(file_paths_or_folder) \
                            if os.path.isfile(os.path.join(file_paths_or_folder, f))]
        else:
            file_paths = file_paths_or_folder

        self._encoding = encoding
        self._delimiter = delimiter
        self._next_line = next_line
        self._batch_size = batch_size
        self._surrounding = surrounding
        self._max_line_cache_size = max_line_cache_size
        self._random_distribution = random_distribution

        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError("The requested file not found on the given path.")
            
            shape = self._get_shape(file_path, self._delimiter, self._encoding)
            self._files.append( type('obj', (object,), {\
                                            'file_path' : file_path,\
                                            'shape': shape,\
                                            'index_range': (self._total_lines, self._total_lines + shape[0] - 1)\
                                        }))
            
            self._total_lines += shape[0]

        if (not random_distribution):
            self._indices = list(range(self._total_lines))
        elif not np:
            raise NotImplementedError("Numpy was not detected. Random distribution requires numpy.random package.")

    _files = []
    _indices = []
    _encoding = "utf8"
    _delimiter = ","
    _next_line = "\n"
    _surrounding="\""
    _total_lines = 0
    _batch_size = 0
    _next_counter = 0
    _max_line_cache_size = 0
    _line_cache_counter = 0
    _random_distribution = True
    
    def _get_shape(self, file_path, delimiter, encoding):
        file_row_no = 0
        file_col_no = 0
        with open(file_path, "rU", encoding=encoding) as f:
            file_row_no = sum(1 for x in f)

        if file_row_no <= 0:
            file_col_no = 0
        else:
            row = linecache.getline(file_path, 1)
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
        row = self._get_line(row_no, file_path)
        return self._array_from_csv_row(row)

    def _array_from_csv_row(self, row):
        return row.replace(self._next_line, "").replace(self._surrounding, "").split(self._delimiter)

    def _get_batch(self, index):

        if not self._random_distribution:
            start = index * self._batch_size
            if start > self._total_lines:
                return None

            end = start + self._batch_size
            sequence = self._indices[start: min(end, self._total_lines)]
        else:
            sequence = np.random.random_integers(0, self._total_lines - 1, self._batch_size).tolist()

        lines = {indx: None for indx in sequence}
        indices = sequence[:]
        indices.sort()
        file_gen = (file for file in self._files)

        file = next(file_gen, None)
        while indices:
            if file is None:
                raise OverflowError("Indices in the MiniBatchSet do not match the file set.")

            indx = indices.pop(0)
            range = file.index_range

            if indx < range[0]:
                raise ValueError("Index to fetch from file was less than file index range.")

            while indx > range[1]:
                file = next(file_gen, None)
                range = file.index_range

            lines[indx] = self._get_row(indx + 1 - range[0], file.file_path)

        file_gen.close()
        return [lines[indx] for indx in sequence]
    
    def get_files(self):
        return self._files

    def get_total_lines(self):
        return self._total_lines

    def reset(self):
        self._next_counter = 0
        if not self._random_distribution:
            shuffle_list(self._indices)


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
        if self._random_distribution:
            batch = self._get_batch()
        else:
            batch = self._get_batch(self._next_counter)
            self._next_counter += 1
        
        if not batch:
            self._next_counter = 0
            raise StopIteration
        return batch