import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path
import time as t
import hippocampus_toolbox as hc_tools
import gantry_control
import estimator as esti




t.time()

def waypoint_file_generating(filename=None):

    x0 = [1020, 1147, 0]
    xn = [1693, 1147, 0]
    dxdyda = [100, 0, 0]

    if filename is not None:
        wp_filename_rel_path = path.relpath('Waypoints/'+ filename + '.txt')
    else:
        wp_filename_rel_path = hc_tools.save_as_dialog('Save way point list as...')

    rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)

def start_field_measurement():
    gc = gantry_control.GantryControl([0, 3100, 0, 1600, 0, 600], use_gui=True, sdr_type='NooElec') # TODO: chack differrences to use_gui=False
    gc.set_new_max_speed_x(1000)
    gc.set_new_max_speed_y(2000)
    gc.set_new_max_speed_z(1000)
    gc.start_field_measurement_file_select()

def analyze_measdata(filename=None):
    if filename is not None:
        measfile_rel_path = path.relpath('Measurements/'+ filename + '.txt')
    else:
        measfile_rel_path =  hc_tools.select_file()

    rf_tools.analyze_measdata_from_file(analyze_tx=[1, 2], measfile_path=measfile_rel_path)

def check_antennas(show_power_spectrum=False):
    sdr_type = 'NooElec'
    Rf = rf.RfEar(sdr_type, 434.0e6, 1e5)  # for NooElec


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

def position_estimation():

    EKF = esti.ExtendedKalmanFilter()
    EKF_plotter = esti.EKF_Plot(EKF.get_tx_pos(), EKF.get_tx_num())


    # set EKF init position

    x_log = np.array([[500], [500]])
    EKF.init_x_0(x_log)

    ### Start EKF-loop ###
    tracking = True
    while tracking:
        try:
            # x_est[:, 0] = x_log[:, -1]
            EKF.ekf_prediction()
            EKF.ekf_update()

            x_log = np.append(x_log, EKF.get_x_est(), axis=1)

            # add new x_est to plot
            EKF_plotter.add_data_to_plot([EKF.get_x_est()[0, -1], EKF.get_x_est()[1, -1]])
            EKF_plotter.update_plot()

        except KeyboardInterrupt:
            print ('Localization interrupted by user')
            tracking = False




if __name__ == '__main__':

    waypoint_file_generating('Waypointlist')  # if no input choose file function activ

    start_field_measurement()

    analyze_measdata('second_try')  # if no input choose file function activ

    # check_antennas(False)