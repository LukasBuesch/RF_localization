import numpy as np
import rf_tools
import estimator_tools as est_to
import hippocampus_toolbox as hc_tools
import estimator_plot_tools as ept


class Extended_Kalman_Filter(object):

    def __init__(self, x_start=[1000, 1000, 0], sig_x1=500, sig_x2=500, sig_w1=100, sig_w2=100, alpha=0):
        # TODO: check default values
        """
        initialize EKF class

        :param set_model_type: lin or log -> currently only log is supported
        :param x_start: first position entry for EKF -> kidnapped-robot-problem (0,0,0)
        :param sig_x1: initial value for P matrix (p_mat) -> uncertainty in x direction
        :param sig_x2: initial value for P matrix (p_mat) -> uncertainty in y direction
        -> kidnapped robot problem, first unknown position (somewhere in the tank, possibly in the middle) and very high
        variance
        :param sig_w1: initial value for Q matrix (q_mat) -> process noise
        :param sig_w2: initial value for Q matrix (q_mat) -> process noise
        :param alpha: inclination angle of RF-plane
        """
        self.__model_type = 'log'
        self.__tx_freq = []
        self.__tx_pos = []
        self.__tx_lambda = []
        self.__tx_gamma = []
        self.__tx_num = None
        self.__tx_param = []
        self.__num_meas = None
        self.__theta_low = []  # theta -> inclination angle
        self.__theta_cap = []  # Theta -> height angle -> always 0 in this application
        self.__psi_low = []  # psi -> polarisation angle
        self.__phi_cap = []  # Phi -> twisting angle
        self.__n_tx = 2  # coefficient of rss model for cos
        self.__n_rec = 2  # coefficient of rss model for cos

        """ initialize EKF """
        self.__x_est_0 = np.array([[x_start[0]], [x_start[1]]])
        self.__z_TA_0 = x_start[2]  # z position in TA coordinates (from depth sensor -> here simulatet by z-Gantry)
        self.__x_est = self.__x_est_0
        self.__z_TA = self.__z_TA_0
        self.__alpha = alpha  # inclination angle of RF-plane
        self.__z_UR = [[0], [np.sin(self.__alpha)], [np.cos(self.__alpha)]]  # orientation of z axis of UR in TA coord.
        self.get_angles()  # initialize values for angles

        # standard deviations of position P matrix
        self.__sig_x1 = sig_x1
        self.__sig_x2 = sig_x2
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2, self.__sig_x2 ** 2]))
        self.__p_mat = self.__p_mat_0

        # Jacobi matrix of measurement (H)
        self.__h_jac = []

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

    def set_alpha(self, alpha):
        self.__alpha = alpha
        return True

    def set_z_UR(self, z_UR):
        self.__z_UR = z_UR
        return True

    def set_n_tx(self, n_tx):
        self.__n_tx = n_tx
        return True

    def set_n_rec(self, n_rec):
        self.__n_rec = n_rec
        return True

    def set_theta_low(self, theta_low):
        self.__theta_low = theta_low
        return True

    def set_theta_cap(self, theta_cap):
        self.__theta_cap = theta_cap
        return True

    def set_phi_cap(self, phi_cap):
        self.__phi_cap = phi_cap
        return True

    def set_psi(self, psi_low):
        self.__psi_low = psi_low
        return True

    def set_num_meas(self, num_meas):
        self.__num_meas = num_meas
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

    def get_num_meas(self):
        return self.__num_meas

    def get_alpha(self):
        return self.__alpha

    def get_z_UR(self):
        return self.__z_UR

    def get_n_tx(self):
        return self.__n_tx

    def get_n_rec(self):
        return self.__n_rec

    def get_theta_low(self):
        return self.__theta_low

    def get_theta_cap(self):
        return self.__theta_cap

    def get_phi_cap(self):
        return self.__phi_cap

    def get_psi(self):
        return self.__psi_low

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

    '''distance model (measurement function)'''

    def h_rss(self, itx):
        x = self.__x_est
        tx_pos = self.__tx_param[itx][0]  # position of the transceiver

        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)

        y_rss = -20 * np.log10(r_dist) + r_dist * self.__tx_lambda[itx] \
                + self.__tx_gamma[itx] + np.log10(np.cos(self.__psi_low)) \
                + self.__n_tx * np.log10(np.cos(self.__theta_cap)) \
                + self.__n_rec * np.log10(np.cos(self.__theta_cap + self.__theta_low))  # see log rules

        self.__r_dist[itx] = r_dist
        self.__y_est[itx] = y_rss

        return True

    '''jacobian of the measurement function'''

    def h_rss_jacobian(self, itx):  # FIXME: not working proper
        x = self.__x_est

        h_rss_jac = np.zeros((self.__tx_num, 2))
        jac_rss_analytic = False  # set bool to solve Jacobi matrix analytically
        if jac_rss_analytic:

            # dh / dx1
            h_rss_jac_x1 = 0

            # dh / dx2
            h_rss_jac_x2 = 0

            # dh / dx3
            h_rss_jac_x3 = 0

            h_rss_jac = np.array([[h_rss_jac_x1], [h_rss_jac_x2]])

        else:

            d_xy = 1  # stepsize for approximated calculation -> change if needed
            print range(self.__tx_num)
            for i in range(self.__tx_num):
                y_est_p, r_dist_p = self.h_rss(i, x + np.array([[d_xy], [0]]))
                y_est_n, r_dist_n = self.h_rss(i, x - np.array([[d_xy], [0]]))
                h_rss_jac[0, i] = (y_est_p - y_est_n) / (2 * d_xy)

                y_est_p, r_dist_p = self.h_rss(i, x + np.array([[0], [d_xy]]))
                y_est_n, r_dist_n = self.h_rss(i, x - np.array([[0], [d_xy]]))
                h_rss_jac[0, i] = (y_est_p - y_est_n) / (2 * d_xy)

        self.__h_jac = h_rss_jac
        return True

    def get_angles(self):
        """
        estimates angles depending on current position -> set new angles in EKF-class
        :return: True
        """
        phi_cap = []
        theta_cap = []
        for i in self.__tx_num:
            r = self.__x_est - self.__tx_pos[i]
            r_abs = np.linalg.norm(r)
            '''Phi -> twisting angle'''
            phi_cap[i] = np.arccos(r[0][0] / r_abs)
            if r[1][0] <= 0.0:
                phi_cap[i] = 2 * np.pi - phi_cap[i]

            '''Theta -> height angle'''
            theta_cap[i] = 0

            '''rotation matrix G -> R'''
            S_GR = np.array([[np.cos(phi_cap[i]), -np.sin(phi_cap[i]), 0.0],
                              [np.sin(phi_cap[i]), np.cos(phi_cap[i]), 0.0],
                              [0.0, 0.0, 1.0]]).T  # TODO: why transposed ? -> see later at theta and psi

            '''rotation matrix G -> R_prime'''
            S_GR_prime = np.array(
                [[np.cos(phi_cap[i]) * np.cos(theta_cap[i]), -np.sin(phi_cap[i]),
                  -np.cos(phi_cap[i]) * np.sin(theta_cap[i])],
                 [np.sin(phi_cap[i]) * np.cos(theta_cap[i]), np.cos(phi_cap[i]),
                  -np.sin(phi_cap[i]) * np.sin(theta_cap[i])],
                 [np.sin(theta_cap[i]), 0.0, np.cos(theta_cap[i])]]).T  # TODO: Why transposed?

            '''psi -> polarisation angle'''
            psi_low = est_to.get_angle_v_on_plane(self.__z_UR, np.array(S_GR_prime[2])[np.newaxis].T,
                                                  np.array(S_GR_prime[1])[np.newaxis].T)

            '''theta -> inclination angle'''
            theta_low = est_to.get_angle_v_on_plane(self.__z_UR, np.array(S_GR[2])[np.newaxis].T,
                                                    np.array(S_GR[0])[np.newaxis].T)

            '''returning values'''
            self.__phi_cap[i] = phi_cap[i]
            self.__theta_cap[i] = theta_cap[i]
            self.__psi_low[i] = psi_low[i]
            self.__theta_low[i] = theta_low[i]

        return True

    def measurement_covariance_model(self, itx):
        """
        estimate measurement noise based on the received signal strength

        :param: itx:
        :return: r_mat -- measurement covariance matrix
        """
        rss_noise_model = self.__z_meas[itx]
        r_dist = self.__r_dist[itx]

        if -35 < rss_noise_model or r_dist <= 100:  # TODO: check values
            r_sig = 100
            print('~Antenna ' + str(itx) + ' is to close!~')

        else:
            r_sig = np.exp(-(1.0 / 30.0) * (rss_noise_model + 60.0)) + 0.5  # TODO: check values

        '''uncertainty caused by angles'''  # TODO: check values
        if self.__theta_cap != 0.0:  # != statement means is not equal -> returns bool
            r_sig += abs(self.__theta_cap[itx]) ** 2 * 15
        if self.__theta_low != 0.0:
            r_sig += abs(self.__theta_low[itx]) ** 2 * 8
        if self.__psi_low != 0.0:
            r_sig += abs(self.__psi_low[itx]) ** 2 * 8

        r_mat = r_sig ** 2
        return r_mat

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__z_TA = self.__z_TA_0
        self.__p_mat = self.__p_mat_0
        self.get_angles()

    def ekf_prediction(self):  # FIXME: if necessary use other prediction model
        """ prediction """
        self.__x_est = self.__x_est  # prediction assumes that vehicle holds the position
        self.__p_mat = self.__i_mat.dot(self.__p_mat.dot(self.__i_mat)) + self.__q_mat  # .dot -> dot product
        return True

    def ekf_update(self, meas_data, rss_low_lim=-120):
        self.__z_meas = meas_data[3:3 + self.__tx_num]  # corresponds to rss
        print('rss = \n')
        print self.__z_meas

        ''' if no valid measurement signal is received, reset ekf '''
        if np.max(self.__z_meas) < rss_low_lim:
            self.reset_ekf()
            print('reset_ekf' + str(np.max(self.__z_meas)))
            print('rss' + str(self.__z_meas))
            return True

        '''iterate through all tx-rss-values'''
        for itx in range(self.__tx_num):
            # estimate measurement from x_est
            self.__y_est[itx], self.__r_dist[itx] = self.h_rss(itx)
            y_tild = self.__z_meas[itx] - self.__y_est[itx]

            # estimate measurement noise based on
            r_mat = self.measurement_covariance_model(itx)

            # calc K-gain  # TODO: not worked on this part yet
            self.h_rss_jacobian(itx)  # set Jacobi matrix
            s_mat = np.dot(self.__h_jac.transpose(), np.dot(self.__p_mat, self.__h_jac)) + r_mat
            # = H^t * P * H + R
            k_mat = np.dot(self.__p_mat, self.__h_jac / s_mat)  # 1/s_scal since s_mat is dim = 1x1

            self.__x_est = self.__x_est + k_mat * y_tild  # = x_est + k * y_tild
            self.__p_mat = (self.__i_mat - np.dot(k_mat, self.__h_jac.transpose())) * self.__p_mat  # = (I-KH)*P

        return True


def main(measfile_rel_path=None, cal_param_file=None, make_plot=False, simulate_meas=True):
    """
    executive program

    :param measfile_rel_path:
    :param cal_param_file: just filename (without ending .txt)
    :param make_plot: decide to use plot by setting boolean
    :param simulate_meas: set boolean True if simulated data is used
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
    meas_data = est_to.get_meas_values(EKF, simulate_meas, measfile_rel_path)  # possibly write this data into class
    print('meas_data:\n' + str(meas_data))

    '''EKF loop'''
    num_meas = EKF.get_num_meas()
    for i in range(num_meas):
        print('\n \n \nPassage number:' + str(i + 1))
        print meas_data[i][:]

        EKF.ekf_prediction()

        EKF.ekf_update(meas_data[i][:])

        print('x_est = \n')
        print EKF.get_x_est()

    print('\n* * * * * *\n'
          'estimator.py stopped!\n'
          '* * * * * *\n')
    return True
