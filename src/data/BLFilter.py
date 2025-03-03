#! /usr/bin/env python
__author__ = "Ben Raanan"
__copyright__ = "Copyright 2021, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL-3.0"
__maintainer__ = "Ben Raanan"
__email__ = "byraanan at mbari.org"
__doc__ = """

Defines the Filter class for filtering UBAT data using rolling window filters.

@author: __author__
@license: __license__
"""  # noqa: A001
import logging

import numpy as np
from rolling import Mean

logger = logging.getLogger("pybl")


class Filter:
    def __init__(self, window_size, target_record_size=60):
        self.window_size = window_size  # rolling filter window size
        self.target_record_size = target_record_size  # target size of filtered record

    def apply_filter(self, data: tuple, filt_func):
        """
        Apply rolling window filtering function (Min | Median) followed by a rolling mean filter.

        :param data: data tuple containing (un-filtered time-series data, filtered time-series data before mean)
        :param filt_func: a pointer to the rolling filter function (e.g., rolling.Min, rolling.Median)
        :return: tuple (filtered time-series, mean filtered time-series data)
        """  # noqa: E501
        # unpack data tuple
        data_, data_filt_ = data

        # apply filter and extract filtered data relevant to the current record
        background_ = self.filter(data_, filt_func)

        # shift the filtered data to the left by half the window size
        background_ = np.roll(background_, int(-self.window_size / 2))

        # append the filtered data form the current record
        data_filt_ = np.concatenate([data_filt_, background_])

        # shift the filtered data to the left by half the window size
        data_filt_ = np.roll(data_filt_, int(-self.window_size / 2))

        # apply mean filter and extract filtered data relevant to the current record
        background_smooth_ = self.filter(data_filt_, Mean)

        return background_, background_smooth_

    def filter(self, data, filt_func):
        """
        Apply rolling window filtering function (e.g., Min, Median)

        :param data: data to be filtered
        :param filt_func: a pointer to the rolling filter function (e.g., rolling.Min, rolling.Median)
        :return: filtered data of target size length
        """  # noqa: E501
        N = len(data)

        # determine appropriate window type: default to a fixed size window, but use a variable
        # window if sample size is insufficient
        window_type = (
            "fixed" if (N - self.target_record_size >= self.window_size - 1) else "variable"
        )
        # logger.info('Applying a {} {} dp {} filter to data with {} dp.'.format(window_type, self.window_size,  # noqa: E501
        #                                                                        filt_func.__name__, len(data)))  # noqa: E501
        # apply the filter!
        filt_ = list(
            filt_func(data, window_size=self.window_size, window_type=window_type),
        )
        # extract the filtered data that's relevant to the current record and return
        return self.extract_filtered_data(filt_, window_type)

    def extract_filtered_data(self, filt_data, window_type):
        """
        Extract filtered data relevant to the current record

        :param filt_data: list containing filtered data
        :param window_type: window type specification (str: 'fixed' or 'variable')
        :return: numpy array containing the relevant filtered data
        """

        n = self.target_record_size  # target size of filtered record
        N = len(filt_data)  # actual size of filtered record
        win_size = self.window_size

        if window_type == "fixed" or (window_type == "variable" and win_size > N):
            ret_val = filt_data[-n:]
        else:
            ret_val = filt_data[1 - n - win_size : 1 - win_size]

        return np.array(ret_val)
