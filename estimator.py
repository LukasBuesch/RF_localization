import numpy as np
import rf_tools
import estimator_tools as est_to
import estimator_plot_tools as ept
import sys


class Extended_Kalman_Filter(object):

    def __init__(self, x_start=[1000, 1000, 0], sig_x1=500, sig_x2=500, sig_z=500, sig_w1=100, sig_w2=100, sig_wz=100,
                 z_depth_sigma=1, alpha=0, inclined_plane=False):
        # TODO: check default values
        """
        initialize EKF class

        :param x_start: first position entry for EKF -> kidnapped-robot-problem (0,0,0)
        :param sig_x1: initial value for P matrix (p_mat) -> uncertainty in x direction
        :param sig_x2: initial value for P matrix (p_mat) -> uncertainty in y direction
        :param sig_z: initial value for P matrix (p_mat) -> uncertainty from depth measurement
        -> kidnapped robot problem, first unknown position (somewhere in the tank, possibly in the middle) and very high
        variance
        :param sig_w1: initial value for Q matrix (q_mat) -> process noise
        :param sig_w2: initial value for Q matrix (q_mat) -> process noise
        :param sig_wz: initial value for Q matrix (q_mat) -> process noise
        :param z_depth_sigma: value for r_mat for measurement of depth z
        :param alpha: inclination angle of RF-plane
        :param inclined_plane: drive real inclined plane with gantry ? -> True
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
        self.__z_depth_sigma = z_depth_sigma
        self.__inclined_plane_real = inclined_plane  # drive real inclined plane with gantry ? -> True

        """ initialize EKF """
        self.__x_est_0 = np.array([[x_start[0]], [x_start[1]], [x_start[2]]])
        self.__x_est = self.__x_est_0
        self.__z_depth = 0  # z position in SR coordinates (from depth sensor -> here simulated by z-Gantry)
        self.__alpha = alpha  # inclination angle of RF-plane
        self.__z_UR = [[0], [np.sin(self.__alpha)], [np.cos(self.__alpha)]]  # orientation of z axis of UR in TA coord.
        # TODO: check vector
        # standard deviations of position P matrix
        self.__sig_x1 = sig_x1
        self.__sig_x2 = sig_x2
        self.__sig_z = sig_z
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2, self.__sig_x2 ** 2, self.__sig_z ** 2]))
        self.__p_mat = self.__p_mat_0

        # Jacobi matrix of measurement (H)
        self.__h_rss_jac = np.array([])
        self.__h_z_jac = np.array([])

        # process noise Q matrix
        self.__sig_w1 = sig_w1
        self.__sig_w2 = sig_w2
        self.__sig_wz = sig_wz
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2, self.__sig_w2 ** 2, self.__sig_wz ** 2]))

        # initial values and system dynamic (=eye)
        self.__i_mat = None  # gradient of f(x) for F matrix -> Jacobi of prediction

        self.__rss_meas = None  # measurement matrix
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
            sys.exit(2)

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
        self.__i_mat = np.eye(3)

        self.__rss_meas = np.zeros(self.__tx_num)
        self.__y_est = np.zeros(self.__tx_num)
        self.__r_dist = np.zeros(self.__tx_num)

        self.__psi_low = range(self.__tx_num)
        self.__phi_cap = range(self.__tx_num)
        self.__theta_low = range(self.__tx_num)
        self.__theta_cap = range(self.__tx_num)
        self.get_angles()  # initialize values for angles
        return True

    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def set_z_depth(self, z_depth):
        self.__z_depth = z_depth
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

    def get_rss_meas(self):
        return self.__rss_meas

    def get_y_est(self):
        return self.__y_est

    '''
    EKF functions
    '''

    '''distance model (measurement function)'''

    def h_rss(self, itx, x=None):
        if x is None:
            x = self.__x_est[:2]
        tx_pos = self.__tx_param[itx][0]  # position of the transceiver

        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)

        y_rss = -20 * np.log10(r_dist) + r_dist * self.__tx_lambda[itx] \
                + self.__tx_gamma[itx] + np.log10(np.cos(self.__psi_low[itx])) \
                + self.__n_tx * np.log10(np.cos(self.__theta_cap[itx])) \
                + self.__n_rec * np.log10(np.cos(self.__theta_cap[itx] + self.__theta_low[itx]))  # see log rules

        return y_rss, r_dist

    '''jacobian of the measurement function'''

    def h_rss_jacobian(self, itx):
        x = self.__x_est[:2]

        h_rss_jac = np.zeros((self.__tx_num, 2))
        jac_rss_analytic = False  # set bool to solve Jacobi matrix analytically
        if jac_rss_analytic:  # not supported yet

            # dh / dx1
            h_rss_jac_x1 = 0

            # dh / dx2
            h_rss_jac_x2 = 0

            # dh / dx3
            h_rss_jac_x3 = 0

            h_rss_jac = np.array([[h_rss_jac_x1], [h_rss_jac_x2]])

            print('analytic solution of h_rss_jac is not supportet yet!')
            sys.exit(2)

        else:

            d_xy = 1  # stepsize for approximated calculation -> change if needed

            y_est_p, r_dist_p = self.h_rss(itx, x + np.array([[d_xy], [0]]))
            y_est_n, r_dist_n = self.h_rss(itx, x - np.array([[d_xy], [0]]))
            h_rss_jac[0, itx] = (y_est_p - y_est_n) / (2 * d_xy)

            y_est_p, r_dist_p = self.h_rss(itx, x + np.array([[0], [d_xy]]))
            y_est_n, r_dist_n = self.h_rss(itx, x - np.array([[0], [d_xy]]))
            h_rss_jac[1, itx] = (y_est_p - y_est_n) / (2 * d_xy)

        self.__h_rss_jac = np.asarray(h_rss_jac)
        return True

    def h_z_depth(self):
        """
        estimates z position in TA coordinates by calculating intersection of z-vector standing on xy plane starting in
         x_est[:2] with z depth in SR coordinates
        :return:
        """
        z_offset = np.asarray(self.__tx_pos)[0, 2]  # offset from SR to TA coordinate system

        if self.__inclined_plane_real:
            z_depth_prime = self.__z_depth - z_offset
            # calculates intersection of gradient from x_est with the depth in TA coordinates

        else:
            z_depth_prime = self.__z_depth / np.cos(self.__alpha) - z_offset

        z_est = np.cos(self.__alpha) * z_depth_prime + np.tan(self.__alpha) * (self.__x_est[0]
                                                                               + np.sin(self.__alpha) * z_depth_prime)

        return z_est

    def h_z_jacobian(self):

        h_z_jac = np.zeros(3)

        # jacobi should change only for z pos
        h_z_jac[0] = np.tan(self.__alpha)
        h_z_jac[1] = 0

        # jacobi for z position
        h_z_jac[2] = np.cos(self.__alpha) + np.tan(self.__alpha) * np.sin(self.__alpha)

        self.__h_z_jac = np.asarray(h_z_jac)
        return True

    def get_angles(self):
        """
        estimates angles depending on current position -> set new angles in EKF-class
        :return: True
        """
        phi_cap = []
        theta_cap = []
        psi_low = []
        theta_low = []
        for i in range(self.__tx_num):
            r = self.__x_est[:] - np.asarray(self.__tx_pos[i]).reshape(3, 1)
            r_abs = np.linalg.norm(r)
            '''Phi -> twisting angle'''
            phi_cap.append(np.arccos(r[0][0] / r_abs))
            if r[1] <= 0.0:
                phi_cap[i] = 2 * np.pi - phi_cap[i]

            '''Theta -> height angle'''
            theta_cap.append(0)

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
            psi_low.append(est_to.get_angle_v_on_plane(np.asarray(self.__z_UR), np.array(S_GR_prime[2])[np.newaxis].T,
                                                       np.array(S_GR_prime[1])[np.newaxis].T))

            '''theta -> inclination angle'''
            theta_low.append(est_to.get_angle_v_on_plane(np.asarray(self.__z_UR), np.array(S_GR[2])[np.newaxis].T,
                                                         np.array(S_GR[0])[np.newaxis].T))

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
        rss_noise_model = self.__rss_meas[itx]
        r_dist = self.__r_dist[itx]

        if -35 < rss_noise_model or r_dist >= 1900:
            r_sig = 100
            print('Meas_Cov_Model: itx = ' + str(itx) + ' r_sig set to ' + str(r_sig))

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
        self.__p_mat = self.__p_mat_0
        self.get_angles()

    def ekf_prediction(self):  # if necessary use other prediction model
        """ prediction """
        self.__x_est = self.__x_est  # prediction assumes that vehicle holds the position
        self.__p_mat = self.__i_mat.dot(self.__p_mat.dot(self.__i_mat)) + self.__q_mat  # .dot -> dot product

        # estimate angles
        self.get_angles()

        return True

    def ekf_update(self, meas_data, rss_low_lim=-120):
        self.__rss_meas = meas_data[3:3 + self.__tx_num]  # corresponds to rss
        print('rss = ')
        print(str(self.__rss_meas) + '\n')

        ''' if no valid measurement signal is received, reset ekf '''
        if np.max(self.__rss_meas) < rss_low_lim:
            self.reset_ekf()
            print('reset_ekf' + str(np.max(self.__rss_meas)))
            print('rss' + str(self.__rss_meas))
            return True

        '''iterate through all tx-rss-values and estimate xy position in TA coordinates'''
        for itx in range(self.__tx_num):
            # estimate measurement from x_est
            self.__y_est[itx], self.__r_dist[itx] = self.h_rss(itx)
            y_tild = self.__rss_meas[itx] - self.__y_est[itx]

            # estimate measurement noise based on
            r_mat = self.measurement_covariance_model(itx)

            # calc K-gain
            self.h_rss_jacobian(itx)  # set Jacobi matrix
            s_mat = np.dot(self.__h_rss_jac[:, itx].T, np.dot(self.__p_mat[:2, :2], self.__h_rss_jac[:, itx])) + r_mat
            # = H^t * P * H + R
            k_mat = np.dot(self.__p_mat[:2, :2], np.divide(self.__h_rss_jac[:, itx], s_mat).reshape(2, 1))
            # 1/s_scal since s_mat is dim = 1x1, need to reshape result of division by scalar

            self.__x_est[:2] = self.__x_est[:2] + k_mat * y_tild  # = x_est[:2] + k * y_tild
            self.__p_mat[:2, :2] = (self.__i_mat[:2, :2] - np.dot(k_mat.T, self.__h_rss_jac[:, itx])) * \
                                   self.__p_mat[:2, :2]
            # = (I-KH)*P

        '''determine z_TA position'''
        # estimate intersection from xy position
        z_est = self.h_z_depth()
        y_tild = z_est - self.__x_est[2]  # z_t - h(\mu_bar_t)

        # get measurement noise (for z_depth a constant value)
        r_mat = self.__z_depth_sigma ** 2

        # calculate K-gain
        self.h_z_jacobian()  # set jacobi matrix
        s_mat = np.dot(self.__h_z_jac.T, np.dot(self.__p_mat, self.__h_z_jac)) + r_mat
        # = H^t * P * H + R
        k_mat = np.dot(self.__p_mat, np.divide(self.__h_z_jac, s_mat).reshape(3, 1))
        # 1/s_scal since s_mat is dim = 1x1, need to reshape result of division by scalar

        a = (self.__i_mat - np.dot(k_mat.T, self.__h_z_jac)) * self.__p_mat

        self.__x_est[2] = self.__x_est[2] + k_mat[2] * y_tild  # = x_est[2] + k * y_tild
        self.__p_mat = (self.__i_mat - np.dot(k_mat.T, self.__h_z_jac)) * self.__p_mat
        # = (I-KH)*P

        return True


def main(measfile_rel_path=None, cal_param_file=None, make_plot=False, simulate_meas=True):
    """
    executive programm

    :param measfile_rel_path:
    :param cal_param_file:
    :param make_plot:
    :param simulate_meas:
    :return:
    """

    '''initialize values for EKF'''
    EKF = Extended_Kalman_Filter()  # initialize object ->check initial values in __init__ function
    est_to.read_measfile_header(object=EKF, analyze_tx=[1, 2],
                                measfile_path=measfile_rel_path)  # write params from header in object
    EKF.set_cal_params(cal_param_file=cal_param_file)
    EKF.set_tx_param()
    EKF.set_initial_values()

    '''load measurement data'''
    meas_data = est_to.get_meas_values(simulate_meas, measfile_rel_path)  # possibly write this data into class
    # print('meas_data:\n' + str(meas_data))

    '''EKF loop'''
    num_meas = EKF.get_num_meas()
    for i in range(num_meas):
        print('\n \n \nPassage number:' + str(i + 1) + '\n')
        print(str(meas_data[i][:]) + '\n')

        # setting depth value from wp position to simulate depth sensor
        EKF.set_z_depth(meas_data[i][2])

        EKF.ekf_prediction()

        EKF.ekf_update(meas_data[i][:])

        print('x_est = ')
        print(str(EKF.get_x_est()) + '\n')

    print('\n* * * * * *\n'
          'estimator.py stopped!\n'
          '* * * * * *\n')
    return True
