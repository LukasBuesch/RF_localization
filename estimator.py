import numpy as np
import matplotlib.pyplot as plt
import estimator_err_comp_plot_tools as ept
import rf_tools
import rf

"""
map/track the position of the mobile node using an EKF

    Keyword arguments:
    :param 
    :param 
    :param
    :param 
    :param 
"""


def read_measfile_header(object, analyze_tx=[1, 2, 3, 4, 5, 6], measfile_path=None):
    """
    :param: object: existing object of EKF to write values in
    :param analyze_tx: Number of used tx
    :param measfile_path: path to measfile
    :return:
    """
# TODO: give values direct to class EKF -> give created object to this function if possible
    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0
    print analyze_tx

    if measfile_path is not None:
        measdata_filename = measfile_path
    else:
        measdata_filename = hc_tools.select_file(functionname='read_measfile_header')

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 0
        measured_wp_list = []

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

            if load_grid_settings and not load_measdata:
                # print(line)

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
                pass
        '''
        write data into object
        '''
        object.set_tx_freq(freqtx)
        object.set_tx_pos(txpos_list)


        data_shape = [data_shape_file[1], data_shape_file[0], data_shape_file[2]]  # data_shape: n_x, n_y, n_z
        plotdata_mat = np.asarray(plotdata_mat_lis)

        return data_shape, plotdata_mat


class Extended_Kalman_Filter(object):
    def __init__(self, set_model_type='log', cal_param_from_file=None):
        """
        Initialize EKF object

        :param set_model_type: lin or log
        :param cal_param_from_file: get lambda and gamma from file or use default
        :param
        :param
        """
        self.__model_type = set_model_type
        self.__cal_param_from_file = cal_param_from_file
        self.__tx_freq = []
        self.__tx_pos = []
        self.__tx_alpha = []
        self.__tx_gamma = []




    '''
    parameter access
    '''
    def set_cal_params(self):
        if self.__model_type == 'log':
            """ parameter for log model """

            if self.__cal_param_from_file is not None: # TODO: is there a write cal_param_file function ?
                (self.__tx_alpha, self.__tx_gamma) = rf_tools.get_cal_param_from_file(param_filename=cal_param_from_file) # TODO: check function -> give in cal_param_from_file the path
                print('Take alpha/gamma from cal-file')
                print('alpha = ' + str(self.__tx_alpha))
                print('gamma = ' + str(self.__tx_gamma))

            else:
                self.__tx_alpha = [0.011100059337162281, 0.014013732682386724, 0.011873535003719441,
                                   0.013228415946149144,  # TODO: set new params as default
                                   0.010212580857184312, 0.010286057191882235]
                self.__tx_gamma = [-0.49471304043015696, -1.2482393190627841, -0.17291318936462172,
                                   -0.61587988305564456,
                                   0.99831151034040444, 0.85711994311461936]

        elif self.__model_type == 'lin':
            """ parameter for linear model """
            # self.__tx_alpha = [-0.020559673796613238, -0.027175147727451984, -0.023111068055053488, -0.024454023111282586,
            #             -0.024854213762496295, -0.021694731996127509]
            # self.__tx_gamma = [-42.189573853301056, -37.651222316888159, -40.60648965954146, -41.523883513559113,
            #             -42.342411649995938, -42.349468350676268]

    def set_tx_freq(self, tx_freq):
        self.__tx_freq = tx_freq
        return True

    def set_tx_pos(self,tx_pos):
        self.__tx_pos = tx_pos
        return True




    # Old Block
    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0

    def get_x_est(self):
        return self.__x_est

    def get_p_mat(self):
        return self.__p_mat

    def get_z_meas(self):
        return self.__z_meas

    def get_y_est(self):
        return self.__y_est

    def get_tx_num(self):
        return self.__tx_num

    def get_tx_pos(self):
        return self.__tx_pos

    def get_tx_alpha(self):
        return self.__tx_alpha

    def get_tx_gamma(self):
        return self.__tx_gamma


class measurement_simulator(object):
    pass


def main(measfile_rel_path=None):
    EKF = Extended_Kalman_Filter(cal_param_from_file=None)  # enter cal param filename here
    read_measfile_header(object=EKF, analyze_tx=[1, 2], measfile_path=measfile_rel_path)
