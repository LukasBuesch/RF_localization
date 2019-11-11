import numpy as np
import rf_tools
import estimator_tools as est_to
import hippocampus_toolbox as hc_tools
import estimator_plot_tools as ept




class Extended_Kalman_Filter(object):
    # FIXME: at first I try to implement the functions of Viktor (great syntax) -> later implement extensions from Jonas

    def __init__(self, set_model_type='log', x_start=[1000, 1000], sig_x1=500, sig_x2=500, sig_w1=100, sig_w2=100):
        # TODO: check default values
        """
        initialize EKF class

        :param set_model_type: lin or log -> currently only log is supported
        :param x_start: first position entry for EKF -> kidnapped-robot-problem (0,0)
        :param sig_x1: initial value for P matrix (p_mat) -> uncertainty in x direction
        :param sig_x2: initial value for P matrix (p_mat) -> uncertainty in y direction
        :param sig_w1: initial value for Q matrix (q_mat) -> process noise
        :param sig_w2: initial value for Q matrix (q_mat) -> process noise
        """
        self.__model_type = set_model_type
        self.__tx_freq = []
        self.__tx_pos = []
        self.__tx_lambda = []
        self.__tx_gamma = []
        self.__tx_num = None
        self.__tx_param = []

        """ initialize EKF """  # TODO: copied from Viktor change values and syntax
        self.__x_est_0 = np.array([[x_start[0]], [x_start[1]]]).reshape((2, 1))
        self.__x_est = self.__x_est_0
        # standard deviations of position P matrix
        self.__sig_x1 = sig_x1
        self.__sig_x2 = sig_x2
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2, self.__sig_x2 ** 2]))
        self.__p_mat = self.__p_mat_0

        # process noise Q matrix
        self.__sig_w1 = sig_w1
        self.__sig_w2 = sig_w2
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2, self.__sig_w2 ** 2]))

        # initial values and system dynamic (=eye)
        self.__i_mat = None  # gradient of f(x) for F matrix

        self.__z_meas = None  # measurement matrix
        self.__y_est = None  # y matrix for expected measurement
        self.__r_dist = None  # to write estimated distance in

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

                self.__tx_lambda = [-0.0199179, -0.0185479]
                self.__tx_gamma = [-5.9438, -8.1549]
                print('Used default values for lambda and gamma (set cal_param_file if needed)\n')

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

    def set_initial_values(self):
        self.__i_mat = np.eye(2)

        self.__z_meas = np.zeros(self.__tx_num)
        self.__y_est = np.zeros(self.__tx_num)
        self.__r_dist = np.zeros(self.__tx_num)
        return True

    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
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

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0

    def ekf_prediction(self):
        pass

    def ekf_update(self):
        pass


class measurement_simulator(object):
    """
    simulates a RSS measurement
    -> writes values in file or set values directly??
    """

    def __init__(self):
        pass

    pass


def main(measfile_rel_path=None, cal_param_file=None, make_plot=False, simulate_meas=False):
    """
    executive program

    :param measfile_rel_path:
    :param cal_param_file: just filename (without ending .txt)
    :param make_plot: decide to use plot by setting boolean
    :param simulate_meas: decide to simulate meas data by setting boolean
    :return: True
    """

    '''initialize values for EKF'''
    EKF = Extended_Kalman_Filter()  # initialize object ->check initial values in __init__ function
    est_to.read_measfile_header(object=EKF, analyze_tx=[1, 2],
                         measfile_path=measfile_rel_path)  # write params from header in object
    EKF.set_cal_params(cal_param_file=cal_param_file)
    EKF.set_tx_param()
    EKF.set_initial_values()

    '''load measurement data'''
    if not simulate_meas:  # TODO: implement a load measurement function
        meas_data = est_to.get_meas_values(EKF, 'second_try')
    else:
        pass

    '''EKF loop'''
    tracking = True
    while tracking:
        try:
            tracking = False
            # TODO: use make_plot here

        except KeyboardInterrupt:
            print ('Localization interrupted by user')
            tracking = False
    print('\n* * * * * *\n'
          'estimator.py stopped!\n'
          '* * * * * *\n')
    return True
