import matplotlib.pyplot as plt
# from matplotlib.patches import Arrow
from scipy.special import lambertw
import numpy as np
import time as t
import matplotlib

matplotlib.use('TkAgg')


class EKF_Plot(object):
    def __init__(self, tx_pos, model_type='log', bplot_circles=True, b_p_cov_plot=False):
        """ setup figure """
        self.__tx_pos = tx_pos
        self.__tx_num = len(tx_pos)
        self.__rsm_model_type = model_type
        # plt.ion()
        (self.__fig1, self.__ax1) = plt.subplots(1, 1)  # get figure/axis handles
        if b_p_cov_plot:
            (self.__fig2, self.__ax2) = plt.subplots()  # get figure/axis handles

        self.__act_time = 0.0  # time to calc plot update frequency

        self.__x1_list = []
        self.__x2_list = []
        self.__x_list = []
        self.__yaw_rad = 0
        self.__x1_gantry_list = []
        self.__x2_gantry_list = []
        self.__p11_list = []
        self.__p22_list = []

        self.__next_wp = [0, 0]

        self.__bplot_circles = bplot_circles
        self.init_plot(b_p_cov_plot)

    def init_plot(self, b_cov_plot):
        if b_cov_plot:
            # self.__ax2.set_xlabel
            # self.__ax2.set_ylabel
            self.__ax2.grid()
        # x_min = -500.0
        # x_max = 3800.0
        # y_min = -500.0
        # y_max = 2000.0

        x_min = -1000.0 * 2
        x_max = 4000.0 * 2
        y_min = -1000.0 * 2
        y_max = 3000.0 * 2

        self.__ax1.axis([x_min, x_max, y_min, y_max])
        self.__ax1.axis('equal')

        self.__ax1.grid()
        self.__ax1.set_xlabel('x-Axis [mm]')
        self.__ax1.set_ylabel('y-Axis [mm]')

        # tank_frame = np.array([[-250, -150], [3750, -150], [3750, 1850], [-250, 1850], [-250, -150]])
        # self.__ax1.plot(tank_frame[:, 0], tank_frame[:, 1], 'k-')

        self.plot_beacons()

        plt.show()
        plt.draw()
        self.__fig1background = self.__fig1.canvas.copy_from_bbox(self.__ax1.bbox)

        self.__plt_pos_tail = self.__ax1.plot(0, 0, 'b.-')[0]
        # self.__plt_pos = self.__ax1.plot(0, 0, 'ro')[0]
        self.__plt_pos_to_wp = self.__ax1.plot(0, 0, 'go-')[0]
        self.__plt_pos_yaw = self.__ax1.plot(0, 0, 'r-')[0]

        if self.__bplot_circles is True:
            # init measurement circles and add them to the plot
            self.__circle_meas = []
            self.__circle_meas_est = []
            for i in range(self.__tx_num):
                txpos_single = self.__tx_pos[i]
                self.__circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.1, color='r', fill=False))

                self.__circle_meas_est.append(
                    plt.Circle((txpos_single[0], txpos_single[1]), 0.1, color='g', fill=False))

                self.__ax1.add_artist(self.__circle_meas[i])
                self.__ax1.add_artist(self.__circle_meas_est[i])

    def plot_beacons(self):
        # plot beacons
        for i in range(self.__tx_num):
            txpos_single = self.__tx_pos[i]
            self.__ax1.plot(txpos_single[0], txpos_single[1], 'ko')

    """
    def plot_way_points(self, wp_list=np.array([0,0]), wp_rad=[0], b_plot_circles=False):
        x1_wp = wp_list[:, 0]
        x2_wp = wp_list[:, 1]
        self.__ax1.plot(x1_wp, x2_wp, 'g.-', label="Way - Point")
        num_wp = len(wp_rad)
        circle_wp = []
        if b_plot_circles:
            for i in range(num_wp):

                circle_wp.append(plt.Circle((x1_wp[i], x2_wp[i]), wp_rad[i], color='g', fill=False))
                self.__ax1.add_artist(circle_wp[i])
    """

    def update_meas_circles(self, z_meas, alpha, gamma, direct_term, y_est=[], b_plot_yest=False, rsm_model='log'):  # TODO: direct term probably from Jonas -> not in docstring
        """

        :param z_meas:
        :param b_plot_yest:
        :param y_est:
        :param alpha:
        :param gamma:
        :return:
        """

        for itx in range(self.__tx_num):
            z_dist = self.inverse_rsm(z_meas[itx], alpha[itx], gamma[itx], direct_term[itx])
            self.__circle_meas[itx].set_radius(z_dist)
            if b_plot_yest:
                z_est = self.inverse_rsm(y_est[itx], alpha[itx], gamma[itx], direct_term[itx])
                self.__circle_meas_est[itx].set_radius(z_est)

                # print('y_tild=' + str(z_meas-y_est))

    def inverse_rsm(self, rss, alpha, gamma, direct_term):
        """Inverse function of the RSM. Returns estimated range in [mm].

            Keyword arguments:
            :param rss -- received power values [dB]
            :param alpha
            :param gamma
            :param rsm_model_type
            """
        z_dist = -20 / (np.log(10) * alpha) * lambertw(
            -np.log(10) * alpha / 20 * np.exp(-np.log(10) / 20 * (rss - gamma - direct_term * 0)))
        control_rss = -20 * np.log10(z_dist.real) + z_dist.real * alpha + gamma + direct_term * 0
        # print("RSS-Fehler: " + str(rss-control_rss))
        return z_dist.real  # [mm]

    def add_x_est_to_plot(self, x_est, yaw_rad):
        # self.__x_list.append([x_est[0], x_est[1]])
        self.__x_list.append([x_est[0][0], x_est[1][0]])
        self.__yaw_rad = yaw_rad

    def update_next_wp(self, next_wp):
        self.__next_wp = next_wp

    def plot_gantry_pos(self, x_gantry):
        self.__x1_gantry_list.append(x_gantry[0])
        self.__x2_gantry_list.append(x_gantry[1])

    def add_p_cov_to_plot(self, p_mat):
        self.__p11_list.append(np.sqrt(p_mat[0, 0]))
        self.__p22_list.append(np.sqrt(p_mat[1, 1]))

    """
    def plot_p_cov(self,numofplottedsamples=300):
        firstdata = 0  # set max number of plotted points
        cnt = len(self.__x1_list)
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples
        if len(self.__p11_list) > 1:
            del self.__ax2.lines[-1]
            del self.__ax2.lines[-1]

        self.__ax2.plot(self.__p11_list[firstdata:-1], 'b.-', label='P11-std')
        self.__ax2.plot(self.__p22_list[firstdata:-1], 'r.-', label='P22-std')
        self.__ax2.legend(loc='upper right')

        plt.pause(0.001)
    """

    def plot_ekf_pos_live(self, b_yaw=True, b_next_wp=True, b_plot_gantry=False, numofplottedsamples=50):
        """
        This function must be the last plot function due to the ugly 'delete' workaround
        :param numofplottedsamples:
        :return:
        """
        new_time = float(t.time())
        # self.__ax1.set_title('Vehicle postition [update with ' + str(int(1 / (new_time - self.__act_time))) + ' Hz]')
        self.__act_time = new_time

        firstdata = 0  # set max number of plotted points
        cnt = len(self.__x_list)
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        # self.__ax1.plot(self.__x1_list[-1], self.__x2_list[-1], 'ro',
        #                label="x_k= " + str([int(self.__x1_list[-1]), int(self.__x2_list[-1])]))

        """
        if b_plot_gantry:
            self.__ax1.plot(self.__x1_gantry_list, self.__x2_gantry_list, 'go-',
                            label="x_k= " + str([int(self.__x1_gantry_list[-1]), int(self.__x2_gantry_list[-1])]))
        """

        x_temp = np.array(self.__x_list)
        self.__plt_pos_tail.set_data(x_temp[firstdata:-1, 0], x_temp[firstdata:-1, 1])
        # self.__plt_pos.set_data(x_temp[-1,0], x_temp[-1,1])

        if b_yaw:
            yaw_arrow = [400 * np.cos(self.__yaw_rad), 400 * np.sin(self.__yaw_rad)]
            self.__plt_pos_yaw.set_data([x_temp[-1, 0], x_temp[-1, 0] + yaw_arrow[0]],
                                        [x_temp[-1, 1], x_temp[-1, 1] + yaw_arrow[1]])

        if b_next_wp:
            self.__plt_pos_to_wp.set_data([x_temp[-1, 0], self.__next_wp[0]], [x_temp[-1, 1], self.__next_wp[1]])

        self.__fig1.canvas.restore_region(self.__fig1background)
        self.__ax1.draw_artist(self.__plt_pos_tail)
        # self.__ax1.draw_artist(self.__plt_pos)
        if b_yaw:
            self.__ax1.draw_artist(self.__plt_pos_yaw)
        if b_next_wp:
            self.__ax1.draw_artist(self.__plt_pos_to_wp)
        if self.__bplot_circles:
            for i in range(self.__tx_num):
                a = 1
                # self.__ax1.draw_artist(self.__circle_meas[i])
                # self.__ax1.draw_artist(self.__circle_meas_est[i])

        # self.__fig1.canvas.blit(self.__ax1.bbox)
        # self.__ax1.legend(loc='upper right')

        plt.pause(0.001)

    # '''
    # Plot functions from Jonas
    # '''
    #
    # '''Erstellung der X und Y Koordinatenlisten zum einfachen und effizienteren Plotten'''
    # x_n_x = [None] * len(x_n)
    # x_n_y = [None] * len(x_n)
    # x_est_x = [None] * len(x_est_list)
    # x_est_y = [None] * len(x_est_list)
    # tx_pos_x = [None] * len(tx_pos)
    # tx_pos_y = [None] * len(tx_pos)
    # tx_pos_z = [None] * len(tx_pos)
    #
    # for i in range(0, len(x_n)):
    #     x_n_x[i] = x_n[i][0]
    #     x_n_y[i] = x_n[i][1]
    # for i in range(0, len(x_est_list)):
    #     x_est_x[i] = x_est_list[i][0]
    #     x_est_y[i] = x_est_list[i][1]
    # for i in range(0, len(tx_pos)):
    #     tx_pos_x[i] = tx_pos[i][0]
    #     tx_pos_y[i] = tx_pos[i][1]
    #     tx_pos_z[i] = tx_h[i]
    #
    # fig = plt.figure(2, figsize=(10, 2.5))
    # # fig = plt.figure(1, figsize=(25, 12))
    #
    # '''Fehlerplot ueber Koordinaten'''
    # plot_fehlerplotueberkoordinaten = False
    # if plot_fehlerplotueberkoordinaten:
    #     plt.subplot(144)
    # x_est_fehler = [None] * len(x_est_x)
    # for i in range(3, len(x_n_x)):
    #     x_est_fehler[i] = get_distance_1D(x_est_x[i], x_n_x[i - 1])
    # # plt.plot(x_est_fehler)
    # ymax = max(x_est_fehler)
    # for i in range(3, len(x_n_y)):
    #     x_est_fehler[i] = get_distance_1D(x_est_y[i], x_n_y[i - 1])
    # # plt.plot(x_est_fehler)
    # ymax = max([max(x_est_fehler), ymax])
    # for i in range(3, len(x_est_list)):
    #     x_est_fehler[i] = get_distance_2D(x_est_list[i], x_n[i - 1])
    # # plt.plot(x_est_fehler)
    # ymax = max([max(x_est_fehler), ymax])
    # x_est_fehler_ges_mean = [np.mean(x_est_fehler[3:])] * len(x_est_x)
    # x_est_fehler_ges_sdt = np.std(x_est_fehler[3:])
    # if plot_fehlerplotueberkoordinaten:
    #     plt.plot(x_est_fehler_ges_mean, '--')
    #     plt.xlabel('Messungsnummer')
    #     plt.ylabel('Fehler')
    #     plt.legend(['Fehler X-Koordinate', 'Fehler Y-Koordinate', '(Gesamt-) Abstandsfehler',
    #                 ('Mittlerer Gesamtfehler = ' + str(np.round(x_est_fehler_ges_mean[0], 1)))], loc=0)
    #     plt.ylim(0, ymax + 300)
    #
    # '''
    # if True:  # Analyse der Einzelmessungen fuer einfacheres Tuning
    #     ekf_plotter = ept.EKF_Plot(tx_pos, bplot_circles=True)
    #     # Einzelanalyse der Punkte mit Kreisen
    #     direct_terms = [None] * tx_num
    #     for i in range(tx_num):
    #         direct_terms[i] = np.log10(directivity_r * directivities_t[i])
    #     for iterx in range(len(x_n_x)):
    #         msg_x_est_temp = x_est_list[iterx]
    #         # print('x= ' + str(msg_x_est))
    #         msg_yaw_rad = 0
    #         msg_z_meas = rss_list[iterx]
    #         msg_y_est = rss_list[iterx]
    #         msg_next_wp = x_n[iterx]
    #         # print('wp=' + str(msg_next_wp))
    #
    #         ekf_plotter.add_x_est_to_plot(msg_x_est_temp, msg_yaw_rad)
    #         ekf_plotter.update_next_wp(msg_next_wp)
    #         ekf_plotter.update_meas_circles(msg_z_meas, lambda_t, gamma_t, direct_terms,  msg_y_est, b_plot_yest=True)
    #         ekf_plotter.plot_ekf_pos_live(b_yaw=False, b_next_wp=True)
    #         plt.show()  # Hier Breakpoint hinsetzen fuer Analyse der punkte
    # '''
    #
    # '''Strecke im Scatterplot'''
    #
    # def plot_3d_scatter(self):
    #
    #     # ax = fig.add_subplot(121, projection='3d')
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(tx_pos_x, tx_pos_y, tx_pos_z, marker="*", c='k', s=100, depthshade=True, zorder=0)
    #     ax.scatter(x_n_x, x_n_y, h_mauv, marker="^", c='c', s=25, depthshade=True, zorder=1)
    #     ax.scatter(x_est_x, x_est_y, h_mauv, marker="o", c='r', s=100, depthshade=True, zorder=2)
    #     xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_n_y), min(tx_pos_y)])) - 100.0
    #     xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_n_y), max(tx_pos_y)])) + 100.0
    #     ymin = float(min([min(x_n_y), min(tx_pos_y)])) - 100.0
    #     ymax = float(max([max(x_n_y), max(tx_pos_y)])) + 100.0
    #     if ymax < xmax:
    #         ymax += 0.5 * xmax
    #         ymin = ymax - xmax
    #     zmin = float(min([min(tx_pos_z), h_mauv]))
    #     zmax = float(max([max(tx_pos_z), h_mauv]))
    #     if zmax < xmax:
    #         zmax += 0.5 * xmax
    #         zmin = zmax - xmax
    #     elif zmax < ymax:
    #         zmax += 0.5 * ymax
    #         zmin = zmax - ymax
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     ax.set_zlim(zmin, zmax)
    #     ax.set_xlabel('X - Achse', fontsize=20)
    #     ax.set_ylabel('Y - Achse', fontsize=20)
    #     ax.set_zlabel('Z - Achse', fontsize=20)
    #     ax.grid()
    #     ax.legend(['Transmitter Antennen', 'Wahre Position', 'Geschaetzte Position'],
    #               loc=3)  # best:0, or=1, ol=2, ul=3, ur=4
    #     ax.set_title('Plot der wahren und geschaetzten Punkte', fontsize=25)
    #     # plt.show()
    #
    # def plot_2d_scatter(self):
    #
    #     plt.subplot(121)
    #     # plt.subplot(111)
    #     plt.plot(x_n_x, x_n_y, marker=".", c='c')
    #     plt.plot(x_est_x, x_est_y, marker=".", c='r')
    #     plt.scatter(tx_pos_x, tx_pos_y, marker="*", c='k', s=100)
    #     xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_est_x)])) - 100.0
    #     xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_est_x)])) + 100.0
    #     ymin = float(min([min(x_n_y), min(tx_pos_y), min(x_est_y)])) - 150.0  # - 350
    #     ymax = float(max([max(x_n_y), max(tx_pos_y), max(x_est_y)])) + 100.0
    #     ymax2 = ymin + ((ymax - ymin) / 2) + ((xmax - xmin) / 2)
    #     ymin2 = ymin + ((ymax - ymin) / 2) - ((xmax - xmin) / 2)
    #     plt.axis([xmin, xmax, ymin, ymax])
    #     plt.xlabel('X - Achse [mm]')
    #     plt.ylabel('Y - Achse [mm]')
    #     plt.grid()
    #     # plt.legend(['$x_text{wahr}', '$x_text{R,est}', '$x_text{Ti}'], loc=3, ncol=3)  # best:0, or=1, ol=2, ul=3, ur=4
    #     # plt.title('Plot der wahren und geschaetzten Punkte', fontsize=25)
    #     # plt.annotate('Z-Koordinate Sendeantennen: ' + str(tx_h[0]) + 'mm', xy=(xmin+50, ymax-50), xytext=(xmin+50, ymax-50))
    #     # plt.annotate('Z-Koordinate Empfaengerantenne: ' + str(h_mauv) + 'mm', xy=(xmin+50, ymax-160), xytext=(xmin+50, ymax-160))
    #     plt.annotate('Durchschnittlicher Fehler: ' + str(np.round(x_est_fehler_ges_mean[0], 0)) + ' +- ' + str(
    #         np.round(x_est_fehler_ges_sdt)) + 'mm', xy=(xmin + 50, ymax - 350), xytext=(xmin + 50, ymax - 350))
    #     plt.annotate('$x_text{0,est,wahr}', xy=(x_est_x[0], x_est_y[0]), xytext=(x_est_x[0] - 50, x_est_y[0] - 100))
    #     # plt.annotate('$x_text{0,wahr}', xy=(x_n_x[0], x_n_y[0]), xytext=(x_n_x[0]-50, x_n_y[0]-100))
    #
    #     # plt.show()
    #
    # '''Strecke im Linienplot'''
    # def plot_line(self):
    #     x_est_fehler = [None] * len((x_est_x))
    #     for i in range(len(x_est_x)):
    #         x_est_fehler[i] = get_distance_1D(x_est_x[i], x_est_y[i])
    #     # plt.figure(figsize=(12, 12))
    #     plt.subplot(143)
    #     plt.plot(range(1, (len(x_n) + 1)), x_n_x)
    #     plt.plot(x_est_x)
    #     plt.plot(range(1, (len(x_n) + 1)), x_n_y)
    #     plt.plot(x_est_y)
    #     plt.xlabel('Messungsnummer')
    #     plt.ylabel('Koordinate [mm]')
    #     # plt.legend(['$x_text{wahr}', '$x_text{est}', '$y_text{wahr}', '$y_text{est}'], loc=1)
    #     ymin = min([min(x_n_x), min(x_n_y), min(x_est_x), min(x_est_y)])
    #     ymax = max([max(x_n_x), max(x_n_y), max(x_est_x), max(x_est_y)])
    #     plt.ylim(ymin - 100, ymax + 100)
    #     locs2 = []
    #     labels2 = []
    #     plt.xticks(np.arange(0, len(x_n_x), step=20))
    #
    #
    # def error_histogram(self):
    #
    #     plt.subplot(248)
    #     plt.hist(x_est_fehler[3:], 30, (0, 800))  # 800
    #     plt.xlabel('Fehler', fontsize=20)
    #     plt.ylabel('Anzahl der Vorkommnisse', fontsize=20)
    #
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    #                     wspace=0.4, hspace=None)
    # plt.show()
    #
