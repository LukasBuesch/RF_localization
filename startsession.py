import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path
import time as t

t.time()

def waypoint_file_generating():

    x0 = [1293, 1347, 0]
    xn = [1543, 1347, 0]
    dxdyda = [250, 0, 0]


    rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)


def analyze_measdata():
    wp_filename_rel_path = path.relpath('Aktuell/wp_list_2018_08_31_grid_meas_d50d50d50.txt')

    measfile_rel_path = path.relpath('Measurements/first_try.txt')

    rf_tools.analyze_measdata_from_file(analyze_tx=[1, 2], measfile_path=measfile_rel_path)

def check_antennas(show_power_spectrum=False):
    sdr_type = 'NooElec'  # 'AirSpy' / 'NooElec'
    Rf = rf.RfEar(sdr_type, 434.0e6, 1e5)  # NooElec


    plt.ion()

    freq6tx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]  # NooElec

    tx_6pos = [[830, 430, 600],
               [1854, 435, 600],
               [2874, 445, 600],
               [2884, 1230, 600],
               [1849, 1235, 600],
               [834, 1225, 600]]

    Rf.set_txparams(freq6tx, tx_6pos)

    Rf.set_samplesize(32)

    if show_power_spectrum:
        Rf.plot_power_spectrum_density()
    else:
        Rf.plot_txrss_live()



if __name__ == '__main__':

    #waypoint_file_generating()

    #analyze_measdata()

    #check_antennas(True)