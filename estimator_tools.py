import numpy as np
import rf_tools
import hippocampus_toolbox as hc_tools
import time as t
import os
from os import path

t.time()


class MeasurementSimulation(object):

    def __init__(self, tx_pos, freqtx, way_filename, meas_filename, alpha):
        self.__tx_pos = tx_pos
        self.__freqtx = freqtx
        self.__numtx = None
        self.__way_filename = way_filename
        self.__meas_filename = meas_filename
        self.__wp = []
        self.__z_UR = []
        self.__r_dist = None
        self.__psi_low = None
        self.__phi_cap = None
        self.__theta_low = None
        self.__theta_cap = None
        self.__mean = None
        self.__var = None
        self.__y_rss = None
        self.__tx_lambda = None
        self.__tx_gamma = None
        self.__n_tx = 2  # coefficient for cos in rss model
        self.__n_rec = 2  # coefficient for cos in rss model
        self.__alpha = alpha
        self.__z_UR = [[0], [np.sin(self.__alpha)], [np.cos(self.__alpha)]]  # orientation of z axis of UR in TA coord.

    def set_init_values(self):
        self.__r_dist = np.zeros(self.__numtx)
        self.__psi_low = range(self.__numtx)
        self.__phi_cap = range(self.__numtx)
        self.__theta_low = range(self.__numtx)
        self.__theta_cap = range(self.__numtx)
        self.get_angles_sym()
        self.__mean = range(self.__numtx)
        self.__var = np.ones(self.__numtx)  # = tx_sigma must be dimension of tx_num

    def get_distance(self, itx, i):
        """
        build distance vector and makes amount
        :param itx:
        :param i:
        :return:
        """
        r = np.asarray(self.__wp[i, :]).reshape(3, 1) - np.asarray(self.__tx_pos[itx]).reshape(3, 1)
        self.__r_dist = np.linalg.norm(r)
        return self.__r_dist

    def set_cal_params(self, cal_param_file=None):
        if cal_param_file is not None:
            (self.__tx_lambda, self.__tx_gamma) = rf_tools.get_cal_param_from_file(
                param_filename=cal_param_file)
            print('Take lambda/gamma from cal-file')
            print('lambda = ' + str(self.__tx_lambda))
            print('gamma = ' + str(self.__tx_gamma))

        else:  # TODO: set new params as default
            self.__tx_lambda = [-0.0199179, -0.0185479]
            self.__tx_gamma = [-5.9438, -8.1549]
            print('Used default values for lambda and gamma (set cal_param_file if needed)\n')

    def rss_value_generator(self, itx, i):
        """
        generates RSS values for simulation purposes
        :param i:
        :param itx:
        :param add_noise: set bool to simulate measurement noise
        :return: rss: simulated RSS value
        """
        self.get_distance(itx, i)

        self.__mean[itx] = -20 * np.log10(self.__r_dist) + self.__r_dist * self.__tx_lambda[itx] \
                + self.__tx_gamma[itx] + np.log10(np.cos(self.__psi_low[itx])**2) \
                + self.__n_tx * np.log10(np.cos(self.__theta_cap[itx])) \
                + self.__n_rec * np.log10(np.cos(self.__theta_cap[itx] + self.__theta_low[itx]))  # see log rules

        return True

    def get_angles_sym(self):
        """
        estimates angles depending on current position -> set new angles in EKF-class
        :return: True
        """
        phi_cap = []
        theta_cap = []
        psi_low = []
        theta_low = []
        for i in range(self.__numtx):
            r = np.asarray(self.__wp[i, :]).reshape(3, 1) - np.asarray(self.__tx_pos[i]).reshape(3, 1)
            r_abs = np.linalg.norm(r)
            r_xy = np.sqrt(r[0] ** 2 + r[1] ** 2)
            '''Phi -> twisting angle'''
            phi_cap.append(np.arccos(r[0][0] / r_xy))
            if r[1] <= 0.0:
                phi_cap[i] = 2 * np.pi - phi_cap[i]

            '''Theta -> height angle'''
            theta_cap.append(np.arctan(r[2] / r_xy))

            '''rotation matrix G -> R'''
            S_GR = np.array([[np.cos(phi_cap[i]), -np.sin(phi_cap[i]), 0.0],
                             [np.sin(phi_cap[i]), np.cos(phi_cap[i]), 0.0],
                             [0.0, 0.0, 1.0]]).T

            '''rotation matrix G -> R_prime'''
            S_GR_prime = np.array(
                [[np.cos(phi_cap[i]) * np.cos(theta_cap[i]), -np.sin(phi_cap[i]),
                  -np.cos(phi_cap[i]) * np.sin(theta_cap[i])],
                 [np.sin(phi_cap[i]) * np.cos(theta_cap[i]), np.cos(phi_cap[i]),
                  -np.sin(phi_cap[i]) * np.sin(theta_cap[i])],
                 [np.sin(theta_cap[i]), 0.0, np.cos(theta_cap[i])]]).T

            '''psi -> polarisation angle'''
            psi_low.append(get_angle_v_on_plane(np.asarray(self.__z_UR), np.array(S_GR_prime[2])[np.newaxis].T,
                                                np.array(S_GR_prime[1])[np.newaxis].T))

            '''theta -> inclination angle'''
            theta_low.append(get_angle_v_on_plane(np.asarray(self.__z_UR), np.array(S_GR[2])[np.newaxis].T,
                                                  np.array(S_GR[0])[np.newaxis].T))

            '''returning values'''
            self.__phi_cap[i] = phi_cap[i]
            self.__theta_cap[i] = theta_cap[i]
            self.__psi_low[i] = psi_low[i]
            self.__theta_low[i] = theta_low[i]

        return True

    def measurement_covariance_model(self, itx, i, covariance_of):
        """
        estimate measurement noise based on the received signal strength

        :param: itx:
        :return: r_mat -- measurement covariance matrix
        """
        if covariance_of:
            r_mat = 0

        else:
            rss_noise_model = self.__mean[itx]

            self.get_distance(itx, i)
            r_dist = self.__r_dist

            if -35 < rss_noise_model or r_dist >= 1900:
                r_sig = 100
                if -35 < rss_noise_model:
                    print('RSS too high -> rx close to tx ')
                else:
                    print('Distance too large or angles to high.')

                print('Meas_Cov_Model: for tx antenna = ' + str(itx) + ' r_sig set to ' + str(r_sig))

            else:
                # parameter for alpha have to be tuned in experiments
                alpha_1 = 30.0
                alpha_2 = 60.0
                alpha_3 = 0.5
                r_sig = np.exp(-(1.0 / alpha_1) * (rss_noise_model + alpha_2)) + alpha_3

            '''uncertainty caused by angles'''
            # parameter have to be tuned in experiments
            beta = 15
            gamma_1 = 8
            gamma_2 = 8
            if self.__theta_cap != 0.0:  # != statement means is not equal -> returns bool
                r_sig += abs(self.__theta_cap[itx]) ** 2 * beta
            if self.__theta_low != 0.0:
                r_sig += abs(self.__theta_low[itx]) ** 2 * gamma_1
            if self.__psi_low != 0.0:
                r_sig += abs(self.__psi_low[itx]) ** 2 * gamma_2

            r_mat = r_sig ** 2

        self.__var[itx] = r_mat
        return True

    def measurement_simulation(self, cal_param_file=None, covariance_of=False):
        """
        simulates a measurement -> writes header like real measurements and in measdata rss values with variance

        :return:
        """

        '''get values from waypoint file'''
        if self.__way_filename is not None:
            wplist_filename = path.relpath('Waypoints/' + self.__way_filename + '.txt')
        else:
            wplist_filename = hc_tools.select_file(functionname='simulate_field_measurement')

        if self.__meas_filename is not None:
            measdata_filename = path.relpath('Simulated_measurements/' + self.__meas_filename + '.txt')
        else:
            measdata_filename = hc_tools.save_as_dialog()
        print measdata_filename

        meas_description = hc_tools.write_descrition()
        print meas_description

        self.__numtx = len(self.__freqtx)
        print('freqtx ' + str(self.__freqtx))
        print('numtx ' + str(self.__numtx))
        print('tx_pos ' + str(self.__tx_pos))

        '''get values from waypoint file'''
        with open(wplist_filename, 'r') as wpfile:
            load_description = True
            load_grid_settings = False
            load_wplist = False
            wp_append_list = []
            for i, line in enumerate(wpfile):

                if line == '### begin grid settings\n':
                    print('griddata found')
                    load_description = False
                    load_grid_settings = True
                    load_wplist = False
                    continue
                elif line == '### begin wp_list\n':
                    load_description = False
                    load_grid_settings = False
                    load_wplist = True
                    print('### found')
                    continue
                if load_description:
                    print('file description')
                    print(line)

                if load_grid_settings and not load_wplist:
                    grid_settings = map(float, line.split(' '))
                    x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                    xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                    grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                    timemeas = grid_settings[9]



                    data_shape = []
                    for i in range(3):  # range(num_dof)
                        try:
                            shapei = int((xn[i] - x0[i]) / grid_dxdyda[i] + i)
                        except ZeroDivisionError:
                            shapei = 1
                        data_shape.append(shapei)

                if load_wplist and not load_grid_settings:
                    # print('read wplist')
                    wp_append_list.append(map(float, line[:-2].split(' ')))

            num_wp = len(wp_append_list)
            wp_data_mat = np.asarray(wp_append_list)
            # print('wp_data_mat: ' + str(wp_data_mat))

            # wp = range(num_wp)
            # for itx in range(num_wp):
            #     wp[itx] = wp_data_mat[:, 1:4]
            self.__wp = np.asarray(wp_data_mat[:, 1:4])

        '''write values in measurement file'''
        with open(measdata_filename, 'w') as measfile:

            # write header to measurement file
            file_description = ('Measurement simulation file\n' + 'Simulation was performed on ' + t.ctime() + '\n'
                                + 'Description: ' + meas_description + '\n')

            txdata = str(self.__numtx) + ' '
            for itx in range(self.__numtx):
                txpos = self.__tx_pos[itx]
                txdata += str(txpos[0]) + ' ' + str(txpos[1]) + ' ' + str(txpos[2]) + ' '
            for itx in range(self.__numtx):
                txdata += str(self.__freqtx[itx]) + ' '

            print('txdata = ' + txdata)

            measfile.write(file_description)
            measfile.write('### begin grid settings\n')
            measfile.write(str(x0[0]) + ' ' + str(x0[1]) + ' ' + str(x0[2]) + ' ' +
                           str(xn[0]) + ' ' + str(xn[1]) + ' ' + str(xn[2]) + ' ' +
                           str(grid_dxdyda[0]) + ' ' + str(grid_dxdyda[1]) + ' ' + str(grid_dxdyda[2]) + ' ' +
                           str(timemeas) + ' ' + txdata +
                           '\n')
            measfile.write('### begin measurement data\n')
            print wp_data_mat.shape[0]

            self.set_cal_params(cal_param_file)
            self.set_init_values()

            for i in range(wp_data_mat.shape[0]):

                wp_pos = wp_data_mat[i][1:4]  # needed for error plots

                for itx in range(self.__numtx):
                    self.rss_value_generator(itx, i)  # generates a rss value for certain distance
                    self.measurement_covariance_model(itx, i, covariance_of)

                measfile.write(str(wp_pos[0]) + ' ' + str(wp_pos[1]) + ' ' + str(wp_pos[2]) + ' ')
                for itx in range(self.__numtx):
                    measfile.write(str(self.__mean[itx][0]) + ' ')
                for itx in range(self.__numtx):
                    measfile.write(str(self.__var[itx]) + ' ')
                for itx in range(self.__numtx):
                    measfile.write(str(self.get_distance(itx, i)) + ' ')
                measfile.write('\n')  # -> x,y,z,meantx1,...,meantxn,vartx1,...vartxn

        print('The simulated values are saved in :\n' + str(measdata_filename))

        return True


def get_meas_values(simulate_meas, measdata_filename=None):
    """

    :param simulate_meas:
    :param measdata_filename:
    :return:
    """

    '''get values from object'''

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 0
        measured_wp_list = []

        if simulate_meas:
            print('\n The used measuring data are simulated.\n')
        else:
            print('The used data are from measurements.\n')

        for i, line in enumerate(measfile):

            if line == '### begin grid settings\n':
                # print('griddata found')
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                # print('Measurement data found')
                continue
            if load_description:
                found_dis = True
                print(line)

            if load_grid_settings and not load_measdata:
                found_grid = True

            if load_measdata and not load_grid_settings:
                found_meas = True

                if simulate_meas:
                    plotdata_line = map(float, line[:-2].split(' '))  # [:-2] to avoid converting '\n' into float
                    plotdata_mat_lis.append(plotdata_line)  # -> x,y,z,meantx1,...,meantxn,vartx1,...vartxn

                else:
                    totnumwp += 1
                    meas_data_line = map(float, line[:-2].split(' '))
                    meas_data_append_list.append(meas_data_line)

                    meas_data_mat_line = np.asarray(meas_data_line)

                    measured_wp_list.append(int(meas_data_mat_line[3]))
                    num_tx = int(meas_data_mat_line[4])
                    num_meas = int(meas_data_mat_line[5])

                    first_rss = 6 + num_tx

                    meas_data_mat_rss = meas_data_mat_line[first_rss:]

                    rss_mat_raw = meas_data_mat_rss.reshape([num_tx, num_meas])  # mat_dim: num_tx x num_meas

                    def reject_outliers(data, m=5.):
                        d = np.abs(data - np.median(data))
                        mdev = np.median(d)
                        s = d / mdev if mdev else 0.
                        # print('kicked out samples' + str([s < m]))
                        return data[s < m]

                    mean = np.zeros([num_tx])
                    var = np.zeros([num_tx])
                    for itx in range(num_tx):
                        rss_mat_row = reject_outliers(rss_mat_raw[itx, :])
                        mean[itx] = np.mean(rss_mat_row)
                        var[itx] = np.var(rss_mat_row)

                    wp_pos = np.array([meas_data_mat_line[0], meas_data_mat_line[1], meas_data_mat_line[2]])

                    plotdata_line = np.concatenate((wp_pos, mean, var),
                                                   axis=0)  # -> x,y,z,meantx1,...,meantxn,vartx1,...vartxn

                    plotdata_mat_lis.append(plotdata_line)
                    # print plotdata_mat_lis

    if found_dis and found_grid and found_meas is not True:
        print('Not all data found! -> check file')
        exit(1)

    plotdata_mat = np.asarray(plotdata_mat_lis)
    # print('plotdata_mat_list = ')
    # print plotdata_mat_lis
    # print('plotdata_mat = \n')
    # print plotdata_mat

    return plotdata_mat


def read_measfile_header(object, analyze_tx=[1, 2, 3, 4, 5, 6], measfile_path=None):
    """
    function writes data from the measurement header to the EKF object
    :param: object: existing object of EKF to write values in
    :param analyze_tx: Number of used tx
    :param measfile_path: relative path to measfile
    :return: True
    """

    print('analyze_tx = ')
    print analyze_tx
    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0

    if measfile_path is not None:
        measdata_filename = measfile_path
    else:
        measdata_filename = hc_tools.select_file(functionname='read_measfile_header')

    if os.stat(str(measdata_filename)).st_size == 0:
        print('Chosen file is empty.')
        exit(1)

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False

        meas_data_append_list = []
        totnumwp = 0
        measured_wp_list = []
        freqtx = []
        txpos_list = []

        for i, line in enumerate(measfile):

            if line == '### begin grid settings\n':
                # print('griddata found')
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue

            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                # print('Measurement data found')
                continue

            if load_description:
                # print('file description')
                print(line)
                found_dis = True

            if load_grid_settings and not load_measdata:
                found_grid = True

                grid_settings = map(float, line[:-2].split(' '))
                x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                timemeas = grid_settings[9]

                data_shape_file = []
                for i in range(3):  # range(num_dof)
                    try:
                        shapei = int((xn[i] - x0[i]) / grid_dxdyda[i] + 1)
                    except ZeroDivisionError:
                        shapei = 1
                    data_shape_file.append(shapei)
                print('data shape  = ' + str(data_shape_file))

                numtx = int(grid_settings[10])
                txdata = grid_settings[11:(11 + 4 * numtx)]  # urspruenglich [(2+numtx):(2+numtx+3*numtx)]

                # read tx positions
                txpos_list = []
                for itx in range(numtx):
                    itxpos = txdata[3 * itx:3 * itx + 3]  # urspruenglich [2*itx:2*itx+2]
                    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                freqtx_list = []
                for itx in range(numtx):
                    freqtx_list.append(txdata[3 * numtx + itx])  # urspruenglich (txdata[2*numtx+itx])
                freqtx = np.asarray(freqtx_list)

                # print out
                print('filename = ' + measdata_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0] * data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdyda = ' + str(grid_dxdyda))
                print('tx_pos = ' + str(txpos_list))
                print('freqtx = ' + str(freqtx))

                startx = x0[0]
                endx = xn[0]
                stepx = data_shape_file[0]

                starty = x0[1]
                endy = xn[1]
                stepy = data_shape_file[1]

                startz = x0[2]
                endz = xn[2]
                stepz = data_shape_file[2]

                xpos = np.linspace(startx, endx, stepx)
                ypos = np.linspace(starty, endy, stepy)
                zpos = np.linspace(startz, endz, stepz)

                wp_maty, wp_matz, wp_matx = np.meshgrid(ypos, zpos, xpos)

            if load_measdata and not load_grid_settings:
                found_meas = True
                totnumwp += 1

    if found_dis and found_grid and found_meas is not True:
        print('Not all data found! -> check file')
        exit(1)
    '''
    write data into object
    '''
    object.set_tx_freq(freqtx)
    object.set_tx_pos(txpos_list)
    object.set_tx_num(len(freqtx))
    object.set_num_meas(totnumwp)

    return True


def get_angle_v_on_plane(v_x, v_1main, v_2):
    """
    Vektor wird auf Ebene projeziert und Winkel mit main-Vektor gebildet

    :param v_x:
    :param v_1main:
    :param v_2:
    :return:
    """
    v_x_proj = np.dot(v_x.T, v_2)[0][0] * v_2 + np.dot(v_x.T, v_1main)[0][0] * v_1main  # np.dot -> dot product

    if np.linalg.norm(v_x_proj) == 0:
        angle_x = np.pi * 0.5  # if dot product = 0 -> angle_x is pi/2
    else:
        cos_angle = np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main))

        if cos_angle > 1:
            angle_x = np.arccos(
                cos_angle - 1e-10)  # subtract 1e-10 if PC calculates accidental few more than 1 - arccos would not work
        else:
            angle_x = np.arccos(cos_angle)

    return angle_x

def get_distance_3d(x_a, h_a, x_b, h_b):
    x_ab = x_a - x_b
    h_ab = h_a - h_b
    dist = ((x_ab[0][0])**2 + (x_ab[1][0])**2 + h_ab**2)**0.5
    return dist


def get_distance_2d(x_a, x_b):
    x_ab = x_a - x_b
    dist = ((x_ab[0])**2 + (x_ab[1])**2)**0.5
    return dist


def get_distance_1d(x_a, x_b):
    dist = abs(x_a - x_b)
    return dist
