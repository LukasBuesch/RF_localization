import rf  # comment this out when using a pc without SDR libraries
import rf_tools
import estimator_tools as est_to
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path
import time as t
import hippocampus_toolbox as hc_tools
import gantry_control  # comment this out when using pc without serial control libraries
import estimator as est

t.time()


class TxData(object):
    def __init__(self, num_tx):
        """
        change positions and frequencies of tx antennas here
        :param num_tx: number of used tx antennas
        """
        if num_tx is 6:
            self.__freqtx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]  # NooElec

            self.__tx_pos = [[770, 432, 0],
                             [1794, 437, 0],
                             [2814, 447, 0],
                             [2824, 1232, 0],
                             [1789, 1237, 0],
                             [774, 1227, 0]]
        elif num_tx is 2:
            self.__freqtx = [434.325e6, 434.62e6]

            self.__tx_pos = [[1120, 1374, 0],
                             [1370, 1374, 0]]
        else:
            print('Check number of TX in TxData object!')
            exit(1)

    def get_freq_tx(self):
        return self.__freqtx

    def get_tx_pos(self):
        return self.__tx_pos


def waypoint_file_generating(filename=None):  # TODO: write function to create a rectangular waypoint path in inclined plane (alpha)
    x0 = [1020, 1147, 0]
    xn = [1693, 1147, 1000]

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


def simulate_field_measurement(tx_num=2, way_filename=None, meas_filename=None):
    TX = TxData(num_tx=tx_num)
    tx_pos = TX.get_tx_pos()
    freq_tx = TX.get_freq_tx()
    est_to.measurement_simulation(tx_pos, freq_tx, way_filename, meas_filename)
    return True


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
    Rf = rf.RfEar(434.0e6, sdr_type, 1e5)  # for NooElec stick

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


def position_estimation(filename=None, cal_param_file=None, sym_meas=None):
    if filename is not None:
        if sym_meas is True:
            measfile_rel_path = path.relpath('Simulated_measurements/' + filename + '.txt')
        else:
            measfile_rel_path = path.relpath('Measurements/' + filename + '.txt')

    else:
        measfile_rel_path = hc_tools.select_file(functionname='position estimation')

    est.main(measfile_rel_path, cal_param_file, False, simulate_meas=sym_meas)


if __name__ == '__main__':
    '''
    start all functions from here
    '''
    waypoint_file_generating()  # if no input is selected file function active

    # start_field_measurement()  # initialize start_RFEar with correct values

    # simulate_field_measurement(tx_num=2, way_filename='Waypointlist_for_simulation', meas_filename='second_try_sim')

    # lambda_t, gamma_t = analyze_measdata('second_try')  # if no input is selected file function active

    # write_cal_param_file(lambda_t, gamma_t, cal_param_file='Test_file')  # if no input is selected file function active

    # position_estimation(filename='second_try',
    #                     cal_param_file='Test_file', sym_meas=False)  # if no input is selected file function active

    # check_antennas(False)
