import rf  # comment this out when using a pc without SDR libaries
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path
import time as t
import hippocampus_toolbox as hc_tools
import gantry_control  # comment this out when using pc without serial control libaries
import estimator as est

t.time()


def waypoint_file_generating(filename=None):
    x0 = [1020, 1147, 0]
    xn = [1693, 1147, 0]
    dxdyda = [100, 0, 0]

    if filename is not None:
        wp_filename_rel_path = path.relpath('Waypoints/' + filename + '.txt')
    else:
        wp_filename_rel_path = hc_tools.save_as_dialog('Save way point list as...(waypoint_file_generating)')

    rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)


def start_field_measurement():
    gc = gantry_control.GantryControl([0, 3100, 0, 1600, 0, 600], use_gui=True,
                                      sdr_type='NooElec')  # TODO: check differrences to use_gui=False
    gc.set_new_max_speed_x(1000)
    gc.set_new_max_speed_y(2000)
    gc.set_new_max_speed_z(1000)
    gc.start_field_measurement_file_select()


def analyze_measdata(filename=None):
    if filename is not None:
        measfile_rel_path = path.relpath('Measurements/' + filename + '.txt')
    else:
        measfile_rel_path = hc_tools.select_file(functionname='analyze_measdata')

    lambda_t, gamma_t = rf_tools.analyze_measdata_from_file(analyze_tx=[1, 2], measfile_path=measfile_rel_path)
    return lambda_t, gamma_t


def write_cal_param_file(lambda_, gamma_, cal_param_file=None):
    if cal_param_file is not None:
        param_filename = path.relpath('Cal_files/' + cal_param_file + '.txt')
    else:
        param_filename = hc_tools.save_as_dialog('Save Cal_param_file as...(write_cal_param_file)')

    rf_tools.write_cal_param_to_file(lambda_, gamma_, param_filename)

def check_antennas(show_power_spectrum=False):
    sdr_type = 'NooElec'
    Rf = rf.RfEar(sdr_type, 434.0e6, 1e5)  # for NooElec stick

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


def position_estimation(filename=None, cal_param_file=None):
    if filename is not None:
        measfile_rel_path = path.relpath('Measurements/' + filename + '.txt')
    else:
        measfile_rel_path = hc_tools.select_file(functionname='position estimation')

    est.main(measfile_rel_path, cal_param_file, True)


if __name__ == '__main__':
    '''
    start all functions from here
    '''
    # waypoint_file_generating('Waypointlist')  # if no input is selected file function active

    # start_field_measurement()

    # lambda_t, gamma_t = analyze_measdata('second_try')  # if no input is selected file function active

    # write_cal_param_file(lambda_t, gamma_t, cal_param_file='Test_file')  # if no input is selected file function active

    position_estimation(filename='second_try', cal_param_file='Test_file')  # if no input is selected file function active

    # check_antennas(False)
