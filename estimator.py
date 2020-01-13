import numpy as np
import rf_tools
import estimator_tools as est_to
import estimator_plot_tools as ept
import sys
import matplotlib.pyplot as plt


class Extended_Kalman_Filter(object):

    def __init__(self, x_start=None, sig_x1=500, sig_x2=500, sig_z=500, sig_w1=500, sig_w2=500, sig_wz=500,
                 z_depth_sigma=1, alpha=0.0 * np.pi, inclined_plane=False):

        """
        initialize EKF class

        :param x_start: first position entry for EKF -> kidnapped-robot-problem somewhere in tank eg.(0,0,0)
        :param sig_x1: initial value for P matrix (p_mat) -> uncertainty in x direction -> greater than tank borders
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
        self.__z_depth_sigma = z_depth_sigma
        self.__inclined_plane_real = inclined_plane  # drive real inclined plane with gantry ? -> True

        """ initialize EKF """
        self.__alpha = alpha  # inclination angle of RF-plane

        self.__x_est_0 = np.array([[x_start[0]], [x_start[1]], [x_start[2]]])
        self.__x_est = self.__x_est_0

        self.__x_real = 0  # needed for z_depth calculation
        self.__z_depth_0 = np.sin(self.__alpha) * self.__x_real
        self.__z_depth = self.__z_depth_0  # z position in SR coordinates (from depth sensor -> simulated by z-Gantry)

        self.__z_UR = [[-np.sin(self.__alpha)], [0], [np.cos(self.__alpha)]]  # orientation of Rx in Tx-coordinates

        # values for RF model
        hpbw = 30.0  # 13.0  # half power beam width (check BA Jonas -> 32,47 +- 6,03 deg)
        hpbw_rad = hpbw * np.pi / 180
        antenna_D = -172.4 + 191 * np.sqrt(0.818 + (1.0 / hpbw))  # approx according to Pozar -> hpbw in deg
        antenna_n = np.log(0.5) / np.log(np.cos(hpbw_rad * 0.5))  # hpbw in cos -> needed in rad
        self.__n_tx = antenna_n * 6  # coefficient of rss model for cos
        self.__n_rx = antenna_n * 6  # coefficient of rss model for cos
        self.__D_0_tx = antenna_D  # coefficient before cos of transmitter (tx)
        self.__D_0_rx = antenna_D  # coefficient before cos of receiver (rx)

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

        # for plotting
        self.__y_tild = 0

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

            else:
                # default values from predecessor -> not checked
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

    def set_n_rx(self, n_rx):
        self.__n_rx = n_rx
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

    def set_x_real(self, x_real):
        self.__x_real = x_real
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

    def get_x_est_0(self):
        return self.__x_est_0

    def get_p_mat(self):
        return self.__p_mat

    def get_rss_meas(self):
        return self.__rss_meas

    def get_y_est(self):
        return self.__y_est

    def get_D_0_tx(self):
        return self.__D_0_tx

    def get_D_0_rx(self):
        return self.__D_0_rx

    def get_y_tild(self):
        return self.__y_tild

    '''
    EKF functions
    '''

    '''distance model (measurement function)'''

    def h_rss(self, itx, x=None):
        if x is None:
            x = self.__x_est
        tx_pos = self.__tx_param[itx][0]  # position of the transceiver

        a = x[2]

        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2 + + (x[2] - tx_pos[2]) ** 2)

        # RSS equation according to BA (Lukas Buesch)
        y_rss = -20 * np.log10(r_dist) + r_dist * self.__tx_lambda[itx] \
                + self.__tx_gamma[itx] + np.log10(np.cos(self.__psi_low[itx]) ** 2) \
                + self.__n_tx * np.log10(np.cos(self.__theta_cap[itx])) \
                + self.__n_rx * np.log10(np.cos(self.__theta_cap[itx] + self.__theta_low[itx]))

        return y_rss, r_dist

    '''jacobian of the measurement function'''

    def h_rss_jacobian(self, itx):
        x = self.__x_est

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

            print('analytic solution of h_rss_jac is not supported yet!')
            sys.exit(2)

        else:

            d_xy = 1  # stepsize for approximated calculation -> change if needed

            y_est_p, r_dist_p = self.h_rss(itx, x + np.array([[d_xy], [0], [0]]))
            y_est_n, r_dist_n = self.h_rss(itx, x - np.array([[d_xy], [0], [0]]))
            h_rss_jac[itx, 0] = (y_est_p - y_est_n) / (2 * d_xy)

            y_est_p, r_dist_p = self.h_rss(itx, x + np.array([[0], [d_xy], [0]]))
            y_est_n, r_dist_n = self.h_rss(itx, x - np.array([[0], [d_xy], [0]]))
            h_rss_jac[itx, 1] = (y_est_p - y_est_n) / (2 * d_xy)

        self.__h_rss_jac = np.asarray(h_rss_jac)
        return True

    def h_z_depth(self):
        """
        estimates z position in TA coordinates by calculating intersection of z-vector standing on xy plane starting in
         x_est[:2] with z depth in SR coordinates
        :return:
        """
        self.__z_depth = np.sin(self.__alpha) * self.__x_real
        # z_offset = np.asarray(self.__tx_pos)[0, 2]  # offset from SR to TA coordinate system
        z_offset = 0

        # if self.__inclined_plane_real:
        #     z_depth_prime = self.__z_depth - z_offset
        #     # calculates intersection of gradient from x_est with the depth in TA coordinates
        #
        # else:
        #     z_depth_prime = self.__z_depth / np.cos(self.__alpha) - z_offset

        z_depth_prime = self.__z_depth

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

            '''rotation matrix G -> R_prime'''  # FIXME: changes in sign to Jonas because of incorrect matrix mult
            S_GR_prime = np.array(
                [[np.cos(phi_cap[i]) * np.cos(theta_cap[i]), -np.sin(phi_cap[i]),
                  np.cos(phi_cap[i]) * np.sin(theta_cap[i])],
                 [np.sin(phi_cap[i]) * np.cos(theta_cap[i]), np.cos(phi_cap[i]),
                  np.sin(phi_cap[i]) * np.sin(theta_cap[i])],
                 [-np.sin(theta_cap[i]), 0.0, np.cos(theta_cap[i])]]).T

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

            # r_sig = 0.3  # FIXME: testing!!

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
        return r_mat

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0
        self.get_angles()

    def ekf_prediction(self):  # if necessary use other prediction model
        """ prediction """
        self.__x_est = self.__x_est  # prediction assumes that vehicle holds the position
        print('prediction ' + str(self.__x_est))
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

            self.__y_tild = y_tild  # for plotting purposes

            print('y_tild_RSS_' + str(itx) + ' = ' + str(y_tild) + '\n')

            print('RSS_est_' + str(itx) + ' = ' + str(self.__y_est[itx]))
            print('r_dist_' + str(itx) + ' = ' + str(self.__r_dist[itx]) + '\n')

            # estimate measurement noise based on
            r_mat = self.measurement_covariance_model(itx)

            # calc K-gain
            self.h_rss_jacobian(itx)  # set Jacobi matrix

            s_mat = np.dot(self.__h_rss_jac[itx, :].T, np.dot(self.__p_mat[:2, :2], self.__h_rss_jac[itx, :])) + r_mat
            # = H^t * P * H + R
            k_mat = np.dot(self.__p_mat[:2, :2], np.divide(self.__h_rss_jac[itx, :], s_mat).reshape(2, 1))
            # 1/s_scal since s_mat is dim = 1x1, need to reshape result of division by scalar

            print('update_' + str(itx) + ' = ' + str(self.__x_est))
            self.__x_est[:2] = self.__x_est[:2] + k_mat * y_tild  # = x_est[:2] + k * y_tild
            self.__p_mat[:2, :2] = (self.__i_mat[:2, :2] - np.dot(k_mat.T, self.__h_rss_jac[itx, :])) \
                                   * self.__p_mat[:2, :2]
            # = (I-KH)*P

        '''determine z_Tx position'''
        # estimate intersection from xy position
        z_est = self.h_z_depth()
        y_tild = z_est - self.__x_est[2]  # z_t - h(\mu_bar_t) --> not z -prediction through h but z through h - predict
        print('y_tild_Z_Tx' + ' = ' + str(y_tild))

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


def main(simulate_meas, x_start=None, measfile_rel_path=None, cal_param_file=None, make_plot=False):
    """
    executive programm

    :param x_start:
    :param measfile_rel_path:
    :param cal_param_file:
    :param make_plot:
    :param simulate_meas:
    :return:
    """

    '''initialize values for EKF'''
    EKF = Extended_Kalman_Filter(x_start=x_start)  # initialize object ->check initial values in __init__ function
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
    x_est_list = []
    x_est_list.append([1000, 1200, 0])
    y_est_list = []
    rss_meas = []

    for i in range(num_meas):
        print('\n \n \nPassage number:' + str(i + 1) + '\n')
        print('meas_data num :' + str(i) + str(meas_data[i][:]) + '\n')

        # setting depth value from wp position to simulate depth sensor
        EKF.set_x_real(meas_data[i][0])

        EKF.ekf_prediction()

        EKF.ekf_update(meas_data[i][:])

        print('x_est = ')
        print(str(EKF.get_x_est()) + '\n')

        '''make list for plotting'''
        x = EKF.get_x_est()[0][0]
        y = EKF.get_x_est()[1][0]
        z = EKF.get_x_est()[2][0]

        x_est_list.append([x, y, z])  # for plotting purposes
        # x_est_list.append(EKF.get_x_est())

        y_tild_list = EKF.get_y_tild()

        y_est_list.append(EKF.get_y_est().tolist())

        meas = meas_data[i][:]
        rss_meas.append(meas[3:3 + EKF.get_tx_num()].tolist())

    print('\n* * * * * *\n'
          'estimator.py stopped!\n'
          '* * * * * *\n')

    '''
    Plott functions from Jonas
    '''
    print('Start plotting results.\n')

    '''Erstellung der X und Y Koordinatenlisten zum einfachen und effizienteren Plotten'''
    x_n_x = [None] * num_meas
    x_n_y = [None] * num_meas
    x_n_z = [None] * num_meas

    x_est_x = [None] * len(x_est_list)
    x_est_y = [None] * len(x_est_list)
    x_est_z = [None] * len(x_est_list)

    tx_pos_x = [None] * len(EKF.get_tx_pos())
    tx_pos_y = [None] * len(EKF.get_tx_pos())
    tx_pos_z = [None] * len(EKF.get_tx_pos())

    for i in range(0, num_meas):
        x_n_x[i] = meas_data[i, 0]  # position of waypoints
        x_n_y[i] = meas_data[i, 1]  # position of waypoints
        x_n_z[i] = meas_data[i, 2]  # position of waypoints

    for i in range(0, len(x_est_list)):
        x_est_x[i] = x_est_list[i][0]
        x_est_y[i] = x_est_list[i][1]
        x_est_z[i] = x_est_list[i][2]

    for i in range(0, len(EKF.get_tx_pos())):
        tx_pos_x[i] = EKF.get_tx_pos()[i][0]
        tx_pos_y[i] = EKF.get_tx_pos()[i][1]
        tx_pos_z[i] = EKF.get_tx_pos()[i][2]

    fig = plt.figure(2, figsize=(10, 2.5))
    # fig = plt.figure(1, figsize=(25, 12))

    '''Fehlerplot ueber Koordinaten'''

    plot_fehlerplotueberkoordinaten = False

    if plot_fehlerplotueberkoordinaten:
        plt.subplot(144)

    x_est_fehler = [None] * len(x_est_x)

    for i in range(3, len(x_n_x)):
        x_est_fehler[i] = est_to.get_distance_1d(x_est_x[i], x_n_x[i - 1])
    # plt.plot(x_est_fehler)

    xmax = max(x_est_fehler)

    for i in range(3, len(x_n_y)):
        x_est_fehler[i] = est_to.get_distance_1d(x_est_y[i], x_n_y[i - 1])

    # plt.plot(x_est_fehler)

    ymax = max([max(x_est_fehler), xmax])

    a = np.asarray(x_est_list[3])
    b = meas_data[3 - 1, 0:3]

    for i in range(3, len(x_est_list)):
        x_est_fehler[i] = est_to.get_distance_2d(np.asarray(x_est_list[i]), meas_data[i - 1, 0:3])

    # plt.plot(x_est_fehler)

    ymax = max([max(x_est_fehler), ymax])  # maximum error

    x_est_fehler_ges_mean = [np.mean(x_est_fehler[3:])] * len(x_est_x)

    x_est_fehler_ges_sdt = np.std(x_est_fehler[3:])  # just for 2D error??

    if plot_fehlerplotueberkoordinaten:
        plt.plot(x_est_fehler_ges_mean, '--')
        plt.xlabel('Messungsnummer')
        plt.ylabel('Fehler')
        plt.legend(['Fehler X-Koordinate', 'Fehler Y-Koordinate', '(Gesamt-) Abstandsfehler',
                    ('Mittlerer Gesamtfehler = ' + str(np.round(x_est_fehler_ges_mean[0], 1)))], loc=0)
        plt.ylim(0, ymax + 300)

    '''Analyse der Einzelmessungen fuer einfacheres Tuning'''

    analyze_individual_meas = False

    if analyze_individual_meas:
        ekf_plotter = ept.EKF_Plot(EKF.get_tx_pos(), bplot_circles=True)
        # Einzelanalyse der Punkte mit Kreisen
        direct_terms = [None] * EKF.get_tx_num()

        for i in range(EKF.get_tx_num()):
            direct_terms[i] = np.log10(EKF.get_D_0_rx() * EKF.get_D_0_tx())

        for itx in range(len(x_n_x)):
            msg_x_est_temp = x_est_list[itx]
            # print('x= ' + str(msg_x_est))
            msg_yaw_rad = 0
            msg_z_meas = meas_data[itx, 3:(3 + EKF.get_tx_num())]
            msg_y_est = meas_data[itx, 3:(3 + EKF.get_tx_num())]
            msg_next_wp = meas_data[itx, 0:3]
            # print('wp=' + str(msg_next_wp))

            ekf_plotter.add_x_est_to_plot(msg_x_est_temp, msg_yaw_rad)
            ekf_plotter.update_next_wp(msg_next_wp)
            ekf_plotter.update_meas_circles(msg_z_meas, EKF.get_tx_lambda(), EKF.get_tx_gamma(), direct_terms,
                                            msg_y_est, b_plot_yest=True)
            ekf_plotter.plot_ekf_pos_live(b_yaw=False, b_next_wp=True)
            plt.show()  # Hier Breakpoint hinsetzen fuer Analyse der punkte

    '''Strecke im Scatterplot'''

    plot_3d = False

    if plot_3d:
        # ax = fig.add_subplot(121, projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tx_pos_x, tx_pos_y, tx_pos_z, marker="*", c='k', s=100, depthshade=True, zorder=0)
        ax.scatter(x_n_x, x_n_y, x_est_z, marker="^", c='c', s=25, depthshade=True, zorder=1)
        ax.scatter(x_est_x, x_est_y, x_est_z, marker="o", c='r', s=100, depthshade=True, zorder=2)
        xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_n_y), min(tx_pos_y)])) - 100.0
        xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_n_y), max(tx_pos_y)])) + 100.0
        ymin = float(min([min(x_n_y), min(tx_pos_y)])) - 100.0
        ymax = float(max([max(x_n_y), max(tx_pos_y)])) + 100.0

        if ymax < xmax:
            ymax += 0.5 * xmax
            ymin = ymax - xmax

        zmin = float(min([min(tx_pos_z), x_est_z]))
        zmax = float(max([max(tx_pos_z), x_est_z]))

        if zmax < xmax:
            zmax += 0.5 * xmax
            zmin = zmax - xmax

        elif zmax < ymax:
            zmax += 0.5 * ymax
            zmin = zmax - ymax

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X - Achse', fontsize=20)
        ax.set_ylabel('Y - Achse', fontsize=20)
        ax.set_zlabel('Z - Achse', fontsize=20)
        ax.grid()
        ax.legend(['Transmitter Antennen', 'Wahre Position', 'Geschaetzte Position'],
                  loc=3)  # best:0, or=1, ol=2, ul=3, ur=4
        ax.set_title('Plot der wahren und geschaetzten Punkte', fontsize=25)
        # plt.show()

    else:
        '''
        here begins the tracking plot
        '''
        plt.subplot(221)
        # plt.subplot(111)

        plt.plot(x_n_x, x_n_y, marker=".", c='c')
        plt.plot(x_est_x, x_est_y, marker=".", c='r')
        plt.scatter(tx_pos_x, tx_pos_y, marker="*", c='k', s=100)

        xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_est_x)])) - 100.0
        xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_est_x)])) + 100.0
        ymin = float(min([min(x_n_y), min(tx_pos_y), min(x_est_y)])) - 150.0  # - 350
        ymax = float(max([max(x_n_y), max(tx_pos_y), max(x_est_y)])) + 100.0
        ymax2 = ymin + ((ymax - ymin) / 2) + ((xmax - xmin) / 2)
        ymin2 = ymin + ((ymax - ymin) / 2) - ((xmax - xmin) / 2)

        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('X - Achse [mm]')
        plt.ylabel('Y - Achse [mm]')
        plt.grid()

        # add annotations

        # plt.legend(['$x_text{wahr}', '$x_text{R,est}', '$x_text{Ti}'], loc=3, ncol=3)  # best:0, or=1, ol=2, ul=3, ur=4
        # plt.title('Plot der wahren und geschaetzten Punkte', fontsize=25)
        # plt.annotate('Z-Koordinate Sendeantennen: ' + str(tx_h[0]) + 'mm', xy=(xmin+50, ymax-50), xytext=(xmin+50, ymax-50))
        # plt.annotate('Z-Koordinate Empfaengerantenne: ' + str(h_mauv) + 'mm', xy=(xmin+50, ymax-160), xytext=(xmin+50, ymax-160))
        plt.annotate('Durchschnittlicher Fehler: ' + str(np.round(x_est_fehler_ges_mean[0], 0)) + ' +- '
                     + str(np.round(x_est_fehler_ges_sdt))
                     + 'mm', xy=(xmin + 50, ymax - 350), xytext=(xmin + 50, ymax - 350))
        plt.annotate('$x_text{0,est,wahr}', xy=(x_est_x[0], x_est_y[0]), xytext=(x_est_x[0] - 50, x_est_y[0] - 100))
        # plt.annotate('$x_text{0,wahr}', xy=(x_n_x[0], x_n_y[0]), xytext=(x_n_x[0]-50, x_n_y[0]-100))

        # plt.show()

    '''Strecke im Linienplot'''

    plot_streckeimlinienplot = True

    if plot_streckeimlinienplot:
        # x_est_fehler = [None] * len((x_est_x))

        for i in range(len(x_est_x)):
            x_est_fehler[i] = est_to.get_distance_1d(x_est_x[i], x_est_y[i])

        # plt.figure(figsize=(12, 12))
        plt.subplot(222)

        plt.plot(range(1, (num_meas + 1)), x_n_x, c='c')
        plt.plot(x_est_x, c='r')

        plt.plot(range(1, (num_meas + 1)), x_n_y, c='m')
        plt.plot(x_est_y, c='y')

        plt.xlabel('Messungsnummer')
        plt.ylabel('Koordinate [mm]')
        # plt.legend(['$x_text{wahr}', '$x_text{est}', '$y_text{wahr}', '$y_text{est}'], loc=1)

        ymin = min([min(x_n_x), min(x_n_y), min(x_est_x), min(x_est_y)])
        ymax = max([max(x_n_x), max(x_n_y), max(x_est_x), max(x_est_y)])

        plt.ylim(ymin - 100, ymax + 100)
        locs2 = []
        labels2 = []
        plt.xticks(np.arange(0, len(x_n_x), step=20))

    plot_z = True
    if plot_z:
        # plt.figure(figsize=(12, 12))
        plt.subplot(223)

        plt.plot(range(1, (num_meas + 1)), x_n_z, c='c')
        plt.plot(x_est_z, c='r')

        plt.xlabel('Messungsnummer')
        plt.ylabel('Koordinate [mm]')
        # plt.legend(['$x_text{wahr}', '$x_text{est}', '$y_text{wahr}', '$y_text{est}'], loc=1)

        ymin = min([min(x_n_z), min(x_est_z)])
        ymax = max([max(x_n_z), max(x_est_z)])

        plt.ylim(ymin - 100, ymax + 100)
        locs2 = []
        labels2 = []
        plt.xticks(np.arange(0, len(x_n_x), step=20))

    plot_meas_sym = True
    if plot_meas_sym:
        # plt.figure(figsize=(12, 12))
        plt.subplot(224)

        # make list for every Tx
        y_est_0 = []
        y_est_1 = []
        rss_0 = []
        rss_1 = []
        for i in range(EKF.get_num_meas()):
            y_est_0.append(y_est_list[i][0])
            y_est_1.append(y_est_list[i][1])

            rss_0.append(rss_meas[i][0])
            rss_1.append(rss_meas[i][1])

        plt.plot(range(1, (num_meas + 1)), y_est_0,'.-', c='r')
        plt.plot(range(1, (num_meas + 1)), y_est_1,'.-', c='y')

        plt.plot(range(1, (num_meas + 1)), rss_0,'.-', c='c')
        plt.plot(range(1, (num_meas + 1)), rss_1,'.-', c='g')

        ymin = min([min(y_est_0), min(rss_0), min(y_est_1), min(rss_1)])
        ymax = max([max(y_est_0), max(rss_0), max(y_est_1), max(rss_1)])

        # plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('Messungsnummer')
        plt.ylabel('RSS')
        plt.legend(['RSS_est_T1', 'RSS_est_T2', 'RSS_true_T1', 'RSS_true_T2'], loc=1)

        plt.ylim(ymin - 2, ymax + 2)
        plt.xticks(np.arange(0, len(y_est_0) + 1, step=1))

    '''Fehlerhistogramm'''

    plot_fehlerhistogramm = False

    if plot_fehlerhistogramm:
        plt.subplot(248)
        plt.hist(x_est_fehler[3:], 30, (0, 800))  # 800
        plt.xlabel('Fehler')
        plt.ylabel('Anzahl der Vorkommnisse')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.4, hspace=None)
    plt.show()

    return True
