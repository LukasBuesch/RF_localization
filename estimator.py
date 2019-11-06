import numpy as np
import rf_tools

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
    :param measfile_path: relative path to measfile
    :return: True
    """

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
        object.set_tx_num(len(freqtx))

        data_shape = [data_shape_file[1], data_shape_file[0], data_shape_file[2]]  # data_shape: n_x, n_y, n_z
        plotdata_mat = np.asarray(plotdata_mat_lis)  # TODO: check if needed

        return True


class Extended_Kalman_Filter(object):
    # FIXME: at first I try to implement the functions of Viktor (great syntax) -> later implement extensions from Jonas

    def __init__(self, set_model_type='log', x_start=[1000, 1000]):
        """
        Initialize EKF object

        :param set_model_type: lin or log
        :param x_start: assumed starting point  # TODO: check if it's true
        :param
        """
        self.__model_type = set_model_type
        self.__tx_freq = []
        self.__tx_pos = []
        self.__tx_lambda = []
        self.__tx_gamma = []
        self.__tx_num = None
        self.__tx_param = []

        """ initialize EKF """  # TODO: copied from Viktor change values and syntax
        self.__x_est_0 = np.array([[x0[0]], [x0[1]]]).reshape((2, 1))
        self.__x_est = self.__x_est_0
        # standard deviations
        self.__sig_x1 = 500
        self.__sig_x2 = 500
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2, self.__sig_x2 ** 2]))
        self.__p_mat = self.__p_mat_0

        # process noise  # TODO: use better model
        self.__sig_w1 = 100
        self.__sig_w2 = 100
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2, self.__sig_w2 ** 2]))

        # initial values and system dynamic (=eye)
        self.__i_mat = np.eye(2)

        self.__z_meas = np.zeros(self.__tx_num)
        self.__y_est = np.zeros(self.__tx_num)
        self.__r_dist = np.zeros(self.__tx_num)

    '''
    parameter access
    '''

    '''set params'''

    def set_tx_param(self):
        lambda_ = self.__tx_lambda
        gamma_ = self.__tx_gamma
        tx_pos = self.__tx_pos
        print('set tx_param with following parameters:')
        print('lambda=' + str(lambda_) + '\n' + 'gamma=' + str(gamma_) + '\n' + 'tx_pos=' + str(tx_pos))

        if len(lambda_) and len(gamma_) and len(tx_pos) is not 0:
            for itx in range(self.__tx_num):
                self.__tx_param.append([self.__tx_pos[itx], self.__tx_lambda[itx], self.__tx_gamma[itx]])
        else:
            print('define params for tx_param first')
            exit(1)

    def set_cal_params(self, cal_param_file=None):
        self.cal_param_file = cal_param_file
        if self.__model_type == 'log':
            """ parameter for log model """

            if self.cal_param_file is not None:
                (self.__tx_lambda, self.__tx_gamma) = rf_tools.get_cal_param_from_file(
                    param_filename=self.cal_param_file)
                print('Take lambda/gamma from cal-file')
                print('lambda = ' + str(self.__tx_lambda))
                print('gamma = ' + str(self.__tx_gamma))

            else:  # TODO: set new params as default

                self.__tx_lambda = [0.011100059337162281, 0.014013732682386724, 0.011873535003719441,
                                    0.013228415946149144,
                                    0.010212580857184312, 0.010286057191882235]
                self.__tx_gamma = [-0.49471304043015696, -1.2482393190627841, -0.17291318936462172,
                                   -0.61587988305564456,
                                   0.99831151034040444, 0.85711994311461936]
                print('Used default values for lambda and gamma (change in class if needed)\n')

        elif self.__model_type == 'lin':  # currently no support for linear model type
            """ parameter for linear model """
            print('Currently no support for linear model type!\n'
                  'Choose model_type == \'log\'')
            exit()

    def set_tx_freq(self, tx_freq):
        self.__tx_freq = tx_freq
        return True

    def set_tx_pos(self, tx_pos):
        self.__tx_pos = tx_pos
        return True

    def set_tx_num(self, tx_num):
        self.__tx_num = tx_num
        return True

    '''get params'''

    def get_tx_freq(self):
        return self.__tx_freq

    def get_tx_pos(self):
        return self.__tx_pos

    def get_tx_lambda(self):
        return self.__tx_lambda

    def get_tx_gamma(self):
        return self.__tx_gamma

    def get_tx_num(self):
        return self.__tx_num

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

    '''
    EKF functions
    '''

    '''distance model (measurement function)'''  # TODO: old version from Viktor -> edit to Jonas model

    def h_rss(self, x, tx_param, model_type):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        gamma = tx_param[2]

        # r = sqrt((x - x_tx) ^ 2 + (y - y_tx) ^ 2)S
        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)
        if model_type == 'log':
            y_rss = -20 * np.log10(r_dist) - alpha * r_dist - gamma
        elif model_type == 'lin':
            y_rss = alpha * r_dist + gamma  # rss in db

        return y_rss, r_dist

    '''jacobian of the measurement function'''  # TODO: old version from Viktor -> edit to Jonas model

    def h_rss_jacobian(self, x, tx_param, model_type):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        # gamma = tx_param[2]  # not used here

        R_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)

        if model_type == 'log':
            # dh / dx1
            h_rss_jac_x1 = -20 * (x[0] - tx_pos[0]) / (np.log(10) * R_dist ** 2) - alpha * (x[0] - tx_pos[0]) / R_dist
            # dh / dx2
            h_rss_jac_x2 = -20 * (x[1] - tx_pos[1]) / (np.log(10) * R_dist ** 2) - alpha * (x[1] - tx_pos[1]) / R_dist
        elif model_type == 'lin':
            # dh /dx1
            h_rss_jac_x1 = alpha * (x[0] - tx_pos[0]) / R_dist
            # dh /dx2
            h_rss_jac_x2 = alpha * (x[1] - tx_pos[1]) / R_dist

        h_rss_jac = np.array([[h_rss_jac_x1], [h_rss_jac_x2]])

        return h_rss_jac.reshape((2, 1))

    def measurement_covariance_model(self, rss_noise_model, r_dist, itx):
        # TODO: old version from Viktor -> edit to Jonas model (continous model)
        """
        estimate measurement noise based on the received signal strength
        :param rss_noise_model: measured signal strength
        :param r_dist:
        :return: r_mat -- measurement covariance matrix
        """

        ekf_param = [6.5411, 7.5723, 9.5922, 11.8720, 21.6396, 53.6692, 52.0241]
        # if r_dist <= 120 or r_dist >= 1900:
        #        r_sig = 100
        #
        #         print('r_dist = ' + str(r_dist))
        # rss_max_lim = [-42, -42, -42, -42, -42, -42]*0
        if -35 < rss_noise_model or r_dist >= 1900:
            r_sig = 100
            # print('Meas_Cov_Model: itx = ' + str(itx) + ' r_sig set to ' + str(r_sig))

        else:
            if rss_noise_model >= -55:
                r_sig = ekf_param[0]
            elif rss_noise_model < -55:
                r_sig = ekf_param[1]
            elif rss_noise_model < -65:
                r_sig = ekf_param[2]
            elif rss_noise_model < -75:
                r_sig = ekf_param[3]
            elif rss_noise_model < -80:
                r_sig = ekf_param[4]

        r_mat = r_sig ** 2
        return r_mat


class measurement_simulator(object):
    """
    simulates a RSS measurement
    -> writes values in file or set values directly??
    """

    def __init__(self):
        pass

    pass


def main(measfile_rel_path=None, cal_param_file=None, make_plot=False):
    """
    executive program

    :param measfile_rel_path:
    :param cal_param_file: just filename (without ending .txt)
    :param make_plot: decide to use plot by setting boolean
    :return: True
    """

    '''initialize values for EKF'''
    EKF = Extended_Kalman_Filter()  # initialize object
    read_measfile_header(object=EKF, analyze_tx=[1, 2],
                         measfile_path=measfile_rel_path)  # write params from header in object
    EKF.set_cal_params(cal_param_file=cal_param_file)
    EKF.set_tx_param()

    '''EKF loop'''
    tracking = True
    while tracking:
        try:
            tracking = False
            # TODO: use make_plot here

        except KeyboardInterrupt:
            print ('Localization interrupted by user')
            tracking = False
    print('* * * * * *\n'
          'estimator.py stopped!\n'
          '* * * * * *\n')
    return True
