#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Thu Aug  2 23:23:58 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsaplots import plot_acf
from tsaplots import plot_pacf

sample_size = 1000
lag_size = 100
white_gaussian_noise = np.random.normal(loc = 0, scale = 1, size = sample_size)
non_random_numbers = np.array(range(1, sample_size + 1))

rand_samples = pd.Series(white_gaussian_noise)
related_sequence = pd.Series(non_random_numbers)

#plt.plot(
#        rand_samples,
#        color = 'blue',
#        lineWidth = 0.7,
#        label = 'Random Samples'
#        )

#plt.plot(
#        related_sequence,
#        color = 'orange',
#        lineWidth = 0.7,
#        label='Related Sequence'
#        )

figure, axes = plt.subplots(2, 2)
figure.tight_layout()

plot_acf(
        x = rand_samples,
        ax = axes[0][0],
        lags = lag_size,
        title = 'ACF Correlogram (Random Samples)'
        )

plot_pacf(
        x = rand_samples,
        ax = axes[0][1],
        lags = lag_size,
        title = 'PACF Correlogram (Random Samples)'
        )

plot_acf(
        x = related_sequence,
        ax = axes[1][0],
        lags = lag_size,
        title = 'ACF Correlogram (Related Sequence)'
        ) 

plot_pacf(
        x = related_sequence,
        ax = axes[1][1],
        lags = lag_size,
        title = 'PACF Correlogram (Related Sequence)'
        )

plt.show()
