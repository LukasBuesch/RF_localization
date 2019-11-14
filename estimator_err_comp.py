# coding=utf-8
"""
Die Simulation einer sendenden Antenne und einer mobilen empfangenden Antenne,
welche sich zueinander verdrehen (und verschieben, welches aber durch
Verdehungen abgebildet werden kann).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import estimator_err_comp_plot_tools as ept

"""
Funktionsdeklarationen:
"""


def analyze_measdata_from_file(analyze_tx=[1, 2, 3, 4, 5, 6],  meantype='db_mean', measfilename='path'):
    """
    :param analyze_tx:
    :param txpos_tuning:
    :param meantype:
    :return:
    """

    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0
    measdata_filename = measfilename

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
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                continue
            if load_description:
                print(line)

            if load_grid_settings and not load_measdata:

                grid_settings = map(float, line[:-2].split(' '))
                x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]

                data_shape_file = []
                for i in range(3):  # range(num_dof)
                    try:
                        shapei = int((xn[i]-x0[i]) / grid_dxdyda[i] + 1)
                    except ZeroDivisionError:
                        shapei = 1
                    data_shape_file.append(shapei)

                numtx = int(grid_settings[10])
                txdata = grid_settings[11:(11+4*numtx)]  # urspruenglich [(2+numtx):(2+numtx+3*numtx)]

                # read tx positions
                txpos_list = []
                for itx in range(numtx):
                    itxpos = txdata[3*itx:3*itx+3]  # urspruenglich [2*itx:2*itx+2]
                    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                freqtx_list = []
                for itx in range(numtx):
                    freqtx_list.append(txdata[3*numtx+itx])  # urspruenglich (txdata[2*numtx+itx])
                freqtx = np.asarray(freqtx_list)

                # print out
                print('filename = ' + measdata_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0]*data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdyda = ' + str(grid_dxdyda))
                print('tx_pos = ' + str(txpos_list))
                print('freqtx = ' + str(freqtx))

            if load_measdata and not load_grid_settings:

                totnumwp += 1
                meas_data_line = map(float, line[:-2].split(' '))
                meas_data_append_list.append(meas_data_line)

                meas_data_mat_line = np.asarray(meas_data_line)

                measured_wp_list.append(int(meas_data_mat_line[3]))
                num_tx = int(meas_data_mat_line[4])
                num_meas = int(meas_data_mat_line[5])

                first_rss = 6 + num_tx

                meas_data_mat_rss = meas_data_mat_line[first_rss:]

                try:
                    rss_mat_raw = meas_data_mat_rss.reshape([num_tx, num_meas])  # mat_dim: num_tx x num_meas
                except:
                    print str(meas_data_mat_rss)
                    quit()

                def reject_outliers(data, m=5.):
                    d = np.abs(data - np.median(data))
                    mdev = np.median(d)
                    s = d / mdev if mdev else 0.
                    return data[s < m]

                if meantype is 'lin':
                    rss_mat_lin = 10**(rss_mat_raw/10)
                    mean_lin = np.mean(rss_mat_lin, axis=1)
                    var_lin = np.var(rss_mat_lin, axis=1)
                    mean = 10 * np.log10(mean_lin)
                    var = 10 * np.log10(var_lin)
                else:
                    mean = np.zeros([numtx])
                    var = np.zeros([numtx])
                    for itx in range(numtx):
                        rss_mat_row = reject_outliers(rss_mat_raw[itx, :])
                        mean[itx] = np.mean(rss_mat_row)
                        var[itx] = np.var(rss_mat_row)
                wp_pos = [meas_data_mat_line[0], meas_data_mat_line[1], meas_data_mat_line[2]]

                plotdata_line = np.concatenate((wp_pos, mean, var), axis=0)  # -> x,y,a,meantx1,...,meantxn,vartx1,...vartxn
                plotdata_mat_lis.append(plotdata_line)

        measfile.close()

        plotdata_matrix = np.asarray(plotdata_mat_lis)
        print('plotdata_matrix = \n')
        print plotdata_matrix


    return plotdata_matrix


def measurement_noise_model(r_abs, theta_cap_mnm, psi_low_mnm, theta_low_mnm):
    noise_cov = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.8]
    if r_abs < 100:
        ti_sig = noise_cov[0]
    elif r_abs < 300:
        ti_sig = noise_cov[1]
    elif r_abs < 850:
        ti_sig = noise_cov[2]
    elif r_abs < 1000:
        ti_sig = noise_cov[3]
    elif r_abs < 1300:
        ti_sig = noise_cov[4]
    elif r_abs < 1650:
        ti_sig = noise_cov[5]
    else:
        ti_sig = noise_cov[6]

    if theta_cap_mnm != 0.0:
        ti_sig += abs(theta_cap_mnm)**4 * 1
    if theta_low_mnm != 0.0:
        ti_sig += abs(theta_low_mnm)**2 * 1
    if psi_low_mnm != 0.0:
        ti_sig += abs(psi_low_mnm)**2 * 1

    return ti_sig


def get_distance_3D(x_a, h_a, x_b, h_b):
    x_ab = x_a - x_b
    h_ab = h_a - h_b
    dist = ((x_ab[0][0])**2 + (x_ab[1][0])**2 + h_ab**2)**0.5
    return dist


def get_distance_2D(x_a, x_b):
    x_ab = x_a - x_b
    dist = ((x_ab[0][0])**2 + (x_ab[1][0])**2)**0.5
    return dist


def get_distance_1D(x_a, x_b):
    dist = abs(x_a - x_b)
    return dist


# Vektor wird auf Ebene projeziert und Winkel mit main-Vektor gebildet
def get_angle_v_on_plane(v_x, v_1main, v_2):
    ''' old
    g_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    g_vec = np.array([[np.dot(v_x.T, v_2)[0][0]], [np.dot(v_x.T, v_1main)[0][0]]])
    gamma_x = np.linalg.solve(g_mat, g_vec)
    '''
    v_x_proj = np.dot(v_x.T, v_2)[0][0]*v_2 + np.dot(v_x.T, v_1main)[0][0]*v_1main
    if np.linalg.norm(v_x_proj) == 0:
        angle_x = np.pi*0.5
    elif (np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main))) > 1:
        angle_x = np.arccos((np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(
            v_1main))) - 1e-10)  # -1e-10, da PC gerne etwas mehr als 1 ausrechnet und daher arccos nicht funktioniert.
    else:
        angle_x = np.arccos(np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main)))
    return angle_x


def get_angles(x_current, tx_pos, h_tx, z_mauv, h_mauv):
    dh = h_mauv - h_tx
    r = x_current - tx_pos
    r_abs = np.linalg.norm(r)
    phi_cap = np.arccos(r[0][0]/r_abs)
    if r[1][0] <= 0.0:
        phi_cap = 2*np.pi - phi_cap
    theta_cap = np.arctan(dh / r_abs)
    S_G_R = np.array([[np.cos(phi_cap), -np.sin(phi_cap), 0.0],
                      [np.sin(phi_cap), np.cos(phi_cap), 0.0],
                      [0.0, 0.0, 1.0]]).T
    # Transformationsmatrix um z & phi --- [0]=x_R.T, [1]=y_R.T, [2]=z_R.T
    S_G_Rt = np.array([[np.cos(phi_cap) * np.cos(theta_cap), -np.sin(phi_cap), -np.cos(phi_cap) * np.sin(theta_cap)],
                       [np.sin(phi_cap) * np.cos(theta_cap), np.cos(phi_cap), -np.sin(phi_cap) * np.sin(theta_cap)],
                       [np.sin(theta_cap), 0.0, np.cos(theta_cap)]]).T
    # Transformationsmatrix um z & phi, dann y & theta --- [0]=x_Rt.T, [1]=y_Rt.T, [2]=z_Rt.T
    psi_low = get_angle_v_on_plane(z_mauv, np.array(S_G_Rt[2])[np.newaxis].T, np.array(S_G_Rt[1])[np.newaxis].T)
    theta_low = get_angle_v_on_plane(z_mauv, np.array(S_G_R[2])[np.newaxis].T, np.array(S_G_R[0])[np.newaxis].T)
    return phi_cap, theta_cap, psi_low, theta_low, dh


def h_rss_model(x_pos_mobil, h_mobil, x_pos_stat, h_stat, lambda_ti, gamma_ti, theta_cap, psi_low, theta_low, n_tx, n_rec, d_r, d_t):
    r = get_distance_3D(x_pos_mobil, h_mobil, x_pos_stat, h_stat)
    rss1 = -20*np.log10(r)+r*lambda_ti + gamma_ti + np.log10(abs(np.cos(psi_low))) + n_tx * np.log10(abs(np.cos(theta_cap))) + n_rec * np.log10(abs(np.cos(theta_cap + theta_low)))
    # rss1 = -20*np.log10(r)+r*lambda_ti + gamma_ti + 2 * n_tx * np.log10(abs(np.cos(theta_cap)))
    # rss1 = -20 * np.log10(r) + r * lambda_ti + gamma_ti + np.log10((np.cos(psi_low) ** 2) * d_r * (np.cos(theta_cap) ** n_tx) * d_t * (np.cos(theta_cap + theta_low) ** n_rec))
    # rss2 = -20*np.log10(r)+r*lambda_ti + gamma_ti
    # print'Kompensation ueber: ' + str(rss1 - rss2) + ' dB'
    return rss1, r


def h_rss_messungsemulator(x_pos_mobil, h_mobil, x_pos_stat, h_stat, lambda_ti, gamma_ti, theta_cap, psi_low, theta_low, n_tx, n_rec, d_r, d_t):
    r = get_distance_3D(x_pos_mobil, h_mobil, x_pos_stat, h_stat)
    # rss = -20*np.log10(r)+r*lambda_ti + gamma_ti + np.log10((np.cos(psi_low)**2) * d_r * (np.cos(theta_cap)**n_tx) * d_t * (np.cos(theta_cap + theta_low)**n_rec))
    rss = -20 * np.log10(r) + r * lambda_ti + gamma_ti + np.log10(np.cos(psi_low)) + n_tx * np.log10(np.cos(theta_cap)) + n_rec * np.log10(np.cos(theta_cap + theta_low))
    tx_sigma = measurement_noise_model(r, theta_cap, psi_low, theta_low)
    return rss  # + np.random.randn(1)*tx_sigma


def h_rss_jacobi(x_pos_mobil, h_mobil, x_pos_stat, h_stat, lambda_t, gamma_t, theta_cap, psi_low, theta_low, n_tx, n_rec, d_r, d_t):
    r = get_distance_2D(x_pos_mobil, x_pos_stat)
    h_rss_jacobimatrix = np.empty([tx_num, 2])
    jacmat_analytic = False
    for jacmat_i in range(tx_num):
        if jacmat_analytic:
            h_rss_jacobimatrix[jacmat_i, 0] = -20*(x_pos_mobil[0]-x_pos_stat[0])/(np.log(10)*r**2)-lambda_t*(x_pos_mobil[0]-x_pos_stat[0])/r
            h_rss_jacobimatrix[jacmat_i, 1] = -20*(x_pos_mobil[1]-x_pos_stat[1])/(np.log(10)*r**2)-lambda_t*(x_pos_mobil[1]-x_pos_stat[1])/r
        else:
            dxy = 1
            rss_pos, r_pos = h_rss_model(x_pos_mobil + np.array([[dxy], [0]]), h_mobil, x_pos_stat[jacmat_i], h_stat[jacmat_i], lambda_t[jacmat_i], gamma_t[jacmat_i], theta_cap[jacmat_i], psi_low[jacmat_i], theta_low[jacmat_i], n_tx[jacmat_i], n_rec[jacmat_i], d_r, d_t[jacmat_i])
            rss_neg, r_pos = h_rss_model(x_pos_mobil + np.array([[-dxy], [0]]), h_mobil, x_pos_stat[jacmat_i], h_stat[jacmat_i], lambda_t[jacmat_i], gamma_t[jacmat_i], theta_cap[jacmat_i], psi_low[jacmat_i], theta_low[jacmat_i], n_tx[jacmat_i], n_rec[jacmat_i], d_r, d_t[jacmat_i])
            h_rss_jacobimatrix[jacmat_i, 0] = (rss_pos - rss_neg)/(2 * dxy)
            rss_pos, r_pos = h_rss_model(x_pos_mobil + np.array([[0], [dxy]]), h_mobil, x_pos_stat[jacmat_i], h_stat[jacmat_i], lambda_t[jacmat_i], gamma_t[jacmat_i], theta_cap[jacmat_i], psi_low[jacmat_i], theta_low[jacmat_i], n_tx[jacmat_i], n_rec[jacmat_i], d_r, d_t[jacmat_i])
            rss_neg, r_pos = h_rss_model(x_pos_mobil + np.array([[0], [-dxy]]), h_mobil, x_pos_stat[jacmat_i], h_stat[jacmat_i], lambda_t[jacmat_i], gamma_t[jacmat_i], theta_cap[jacmat_i], psi_low[jacmat_i], theta_low[jacmat_i], n_tx[jacmat_i], n_rec[jacmat_i], d_r, d_t[jacmat_i])
            h_rss_jacobimatrix[jacmat_i, 1] = (rss_pos - rss_neg) / (2 * dxy)
    return h_rss_jacobimatrix


def measurement_covariance_model(rss_noise_model, r_dist_tot, phi_cap, theta_cap, psi_low, theta_low, dh):
    if -35 < rss_noise_model or r_dist_tot < 100:
        r_sig = 5  # ekf_param[-1]
        print('~~~~~ZU NAHE!~~~~~')
    else:
        r_sig = np.exp(-(1.0/30.0)*(rss_noise_model + 60.0)) + 0.5
        '''
        if rss_noise_model < -80:
            r_sig = 10
        elif rss_noise_model < -75:
            r_sig = 5
        elif rss_noise_model < -65:
            r_sig = 3
        elif rss_noise_model < -55:
            r_sig = 1.5
        elif rss_noise_model >= -55:
            r_sig =  0.1
        '''

    if theta_cap != 0.0:
        r_sig += abs(theta_cap)**2 * 15
    if theta_low != 0.0:
        r_sig += abs(theta_low)**2 * 8
    if psi_low != 0.0:
        r_sig += abs(psi_low)**2 * 8

    r_mat = r_sig ** 2
    return r_mat


def ekf_prediction(x_est, p_mat, q_mat, x_prev):
    x_est = x_est + (x_est - x_prev)*0.0
    p_mat = i_mat.dot(p_mat.dot(i_mat)) + q_mat  # Theoretisch mit .T transponierten zweiten I Matrix
    return x_est, p_mat


def ekf_update(z_meas, tx_pos, lambda_t, gamma_t, x_est, p_mat, txh, zmauv, hmauv, txn, rxn, d_r, d_t):
    r_mat = np.diag([0.0]*tx_num)
    y_tild = np.empty([tx_num, 1])
    phi_cap_itx = [0.0]*tx_num
    theta_cap_itx = [0.0] * tx_num
    psi_low_itx = [0.0] * tx_num
    theta_low_itx = [0.0] * tx_num
    height_diff = [0.0] * tx_num
    for itx in range(tx_num):
        phi_cap_itx[itx], theta_cap_itx[itx], psi_low_itx[itx], theta_low_itx[itx], height_diff[itx] = get_angles(x_est, tx_pos[itx], txh[itx], zmauv, hmauv)
        y_est, r_dist = h_rss_model(x_est, h_mauv, tx_pos[itx], tx_h[itx], lambda_t[itx], gamma_t[itx], theta_cap_itx[itx], psi_low_itx[itx], theta_low_itx[itx], txn[itx], rxn[itx], d_r, d_t[itx])
        y_tild[itx][0] = z_meas[itx] - y_est
        r_mat[itx, itx] = measurement_covariance_model(z_meas[itx], r_dist, phi_cap_itx[itx], theta_cap_itx[itx], psi_low_itx[itx], theta_low_itx[itx], height_diff[itx])
    h_jac_mat = h_rss_jacobi(x_est, h_mauv, tx_pos, txh, lambda_t, gamma_t, theta_cap_itx, psi_low_itx, theta_low_itx, txn, rxn, d_r, d_t)
    s_mat = np.dot(h_jac_mat, np.dot(p_mat, h_jac_mat.T)) + r_mat
    k_mat = np.dot(p_mat, np.dot(h_jac_mat.T, np.linalg.inv(s_mat)))
    x_est = x_est + np.dot(k_mat, y_tild)
    p_mat = (i_mat - np.dot(k_mat, h_jac_mat)) * p_mat
    return x_est, p_mat, y_tild, k_mat, h_jac_mat


"""
executive program
"""
def main(measfile_path, lambda_t=None, gamma_t=None):

    np.random.seed(12896)

    '''configuration of measurementpoints'''
    dist_messpunkte = 50.0
    start_messpunkte = np.array([[800.0], [700.0]])
    start_messpunkt = start_messpunkte
    mittel_messpunkt = np.array([[2600], [900]])

    x_n = [start_messpunkte]
    while x_n[-1][0] < mittel_messpunkt[0][0]:
        start_messpunkte = start_messpunkte + np.array([[dist_messpunkte], [0.0]])
        x_n.append(start_messpunkte)
    while x_n[-1][1] < mittel_messpunkt[1][0]:
        start_messpunkte = start_messpunkte + np.array([[0.0], [dist_messpunkte]])
        x_n.append(start_messpunkte)
    while x_n[-1][0] > start_messpunkt[0][0]:
        start_messpunkte = start_messpunkte + np.array([[-dist_messpunkte], [0.0]])
        x_n.append(start_messpunkte)
    while x_n[-1][1] > start_messpunkt[1][0]:
        start_messpunkte = start_messpunkte + np.array([[0.0], [-dist_messpunkte]])
        x_n.append(start_messpunkte)
    anz_messpunkte = len(x_n)

    '''Konfiguration der Hoehe und der Antennenverdrehung durch Beschreibung des mobilen Antennenvektors'''
    h_mauv = 500.0
    # z_mauv = np.array([[0], [0.34202014332], [0.93969262078]])
    z_mauv = np.array([[0.0], [0.0], [100.0]])
    # z_mauv = np.array([[0.0], [0.64278760968], [0.76604444311]])

    '''Bestimmung der Messfrequenzen'''
    tx_freq = [434.325e6, 434.62e6]  # FIXME: change if using not only 2 tx

    # tx_freq = [4.3400e+08, 4.341e+08, 4.3430e+08, 4.3445e+08, 4.3465e+08, 4.3390e+08]
    tx_num = len(tx_freq)

    '''Postion(en) der stationaeren Antenne(n)'''
    tx_pos = [[1120, 1374],  # FIXME: change this if changing tx position
               [1370, 1374]]
    print tx_pos
    tx_h = np.array([0, 0])

    # tx_pos = [np.array([[770], [432]]), np.array([[1794], [437]]), np.array([[2814], [447]]),
    #           np.array([[2824], [1232]]), np.array([[1789], [1237]]), np.array([[774], [1227]])]
    # print tx_pos
    # tx_h = np.array([600, 600, 600, 600, 600, 600])


    '''Berechnung von n und D'''  # TODO: verstehen, was er hier macht -> wofuer die Werte -> check paper
    hpbw = 30.0  # 13.0  # half_power_band_width -> paper der Koreaner (Ueber Kippwinkel)
    hpbwrad = hpbw * np.pi/180
    antenna_D = -172.4 + 191*np.sqrt(0.818+(1.0/hpbw))
    antenna_n = np.log(0.5)/np.log(np.cos(hpbwrad*0.5))

    '''Kennwerte der stationaeren Antenne(n)'''

    if lambda_t is None:
        lambda_t = np.array([-0.0108059, -0.0102862, -0.0095051, -0.0086124, -0.0086001, -0.0118771])
        gamma_t = np.array([-2.465, -4.7777, -7.358, -6.9102, -8.412, -3.2815])

    tx_n = [antenna_n]*6  # Normal

    directivities_t = [antenna_D] * 6  # Normal
    '''Kennwerte der mobilen Antenne'''
    rx_n = [antenna_n]*6  # Normal
    directivity_r = antenna_D

    '''Initialisierung der P-Matrix (Varianz der Position)'''
    p_mat = np.diag([140**2]*2)  # Abweichungen von x1 und x2 aufgrund der Messungen...

    '''Initialisierung der Q-Matrix (Varianz des Prozessrauschens / Modellunsicherheit)'''
    q_mat = np.diag([140**2]*2)  # Abweichungen von x1 und x2 aufgrund des Modelles

    '''Initialisierung der y-Matrix fuer die erwartete Messung'''
    y_est = 0  # np.zeros(tx_num)

    '''Initialisierung der F-Matrix -> Gradienten von f(x)'''
    i_mat = np.eye(2)

    '''Initialisierung der Distanzspeicherungsmatrix'''
    r_dist = 0  # np.zeros(tx_num)

    '''Initialisierung der Messmatrix'''
    z_meas = np.zeros(tx_num)
    directivities_t
    '''Initialisierung der geschaetzten Position'''
    # x_est = np.array([[0.0], [0.0]])
    x_est = start_messpunkte
    # x_est = np.array([[5000.0], [6000.0]])

    '''Initialisierung der Liste(n) fuer Plots'''
    # x_est_list = [x_est]
    # rss_list = []
    # x_est_kminus1 = x_est
    # phi_cap_t = [0.0]*tx_num
    # theta_cap_t = [0.0]*tx_num
    # psi_low_t = [0.0]*tx_num
    # theta_low_t = [0.0]*tx_num

    '''Einstellungen fuer Messwerterzeugung'''
    messung_benutzen = True
    if messung_benutzen:
        '''Laden der Messdatei'''
        plotdata_mat = analyze_measdata_from_file(analyze_tx=[1, 2], measfilename=measfile_path)

    extra_plotting = False
    direct_terms = [0.0] * tx_num
    # if extra_plotting:
    #     '''Setup fuer Kreisplots'''
    #     ekf_plotter = ept.EKF_Plot(tx_pos, bplot_circles=True)
    #
    #     for i in range(tx_num):
    #         direct_terms[i] = np.log10(directivity_r * directivities_t[i])
    #
    #     '''Setup fuer ytild Plot'''
    #     fig_ytild_p = plt.figure(42)
    #     sub1_ytild = fig_ytild_p.add_subplot(121)
    #     linspace_ploty_txnum = np.linspace(1, tx_num, tx_num)
    #     ydata_ploty = np.linspace(-20, 20, tx_num)
    #     line1sub1, = sub1_ytild.plot(linspace_ploty_txnum, ydata_ploty, 'r-')  # Returns a tuple of line objects, thus the comma
    #     plt.grid()
    #
    #     '''Setup fuer P Plot'''
    #     sub2_pmat = fig_ytild_p.add_subplot(122)
    #     linspace_plotp = [1, 2]
    #     ydata_plotp = np.linspace(0, 200, 2)
    #     line1sub2, = sub2_pmat.plot(linspace_plotp, ydata_plotp, 'r-')  # Returns a tuple of line objects, thus the comma
    #     plt.grid()

    '''
    Hier müsste der EKF loop starten 
    '''
    # -> TODO: check if statement is true
    for k in range(anz_messpunkte):
        print "\n \n \nDurchlauf Nummer", k

        rss = np.zeros(tx_num)
        if messung_benutzen:
            wp_index = np.where(np.logical_and(np.logical_and(plotdata_mat[:, 0] == x_n[k][0], plotdata_mat[:, 1] == x_n[k][1]), plotdata_mat[:, 2] == h_mauv))
        for i in range(tx_num):
            phi_cap_t[i], theta_cap_t[i], psi_low_t[i], theta_low_t[i], d_height = get_angles(x_n[k], tx_pos[i], tx_h[i], z_mauv, h_mauv)
            if not messung_benutzen:
                # Eigentlich hier Winkel
                rss[i] = h_rss_messungsemulator(x_n[k], h_mauv, tx_pos[i], tx_h[i], lambda_t[i], gamma_t[i], theta_cap_t[i], psi_low_t[i], theta_low_t[i], tx_n[i], rx_n[i], directivity_r, directivities_t[i])

            else:
                rss[i] = plotdata_mat[wp_index[0][0], 3+i]
                direct_terms[i] = np.log10(np.cos(psi_low_t[i])) + tx_n[i] * np.log10(np.cos(theta_cap_t[i])) + rx_n[i] * np.log10(np.cos(theta_cap_t[i] + theta_low_t[i]))  # TODO: what is direct terms?
        rss_list.append(rss)
        if k > 2:
            x_est, p_mat = ekf_prediction(x_est, p_mat, q_mat, x_est_list[-2])  # TODO: why no EKF_update ?
        else:
            x_est, p_mat = ekf_prediction(x_est, p_mat, q_mat, x_est)
        x_est, p_mat, y_tild, k_mat, h_jac = ekf_update(rss, tx_pos, lambda_t, gamma_t, x_est, p_mat, tx_h, z_mauv, h_mauv, tx_n, rx_n, directivity_r, directivities_t)

        #print(k_mat)

        #print "Die wirkliche Position ist: \n", x_n[k]
        #print "Die geschaetzte Position ist: \n", x_est
        print "( Die p-Matrix entspricht: \n", p_mat, ") \n"
        #print "( Die p-Matrix entspricht: \n", np.sqrt(np.diag(p_mat)), ") \n"
        #print "Die RSS sind: ", rss

        x_est_list.append(x_est)

        # if extra_plotting:
        #     line1sub1.set_ydata(y_tild)
        #     sub1_ytild.plot()
        #     if not k == 0:
        #         an00.remove()
        #         an10.remove()
        #         an20.remove()
        #         an30.remove()
        #         an40.remove()
        #         an50.remove()
        #     an00 = sub1_ytild.annotate(str(rss[0]), xy=(1, y_tild[0]), xytext=(1, y_tild[0]))
        #     an10 = sub1_ytild.annotate(str(rss[1]), xy=(2, y_tild[1]), xytext=(2, y_tild[1]))
        #     an20 = sub1_ytild.annotate(str(rss[2]), xy=(3, y_tild[2]), xytext=(3, y_tild[2]))
        #     an30 = sub1_ytild.annotate(str(rss[3]), xy=(4, y_tild[3]), xytext=(4, y_tild[3]))
        #     an40 = sub1_ytild.annotate(str(rss[4]), xy=(5, y_tild[4]), xytext=(5, y_tild[4]))
        #     an50 = sub1_ytild.annotate(str(rss[5]), xy=(6, y_tild[5]), xytext=(6, y_tild[5]))
        #
        #     p_data = np.diag(p_mat)
        #     line1sub2.set_ydata(np.diag(p_mat))
        #     sub2_pmat.plot()
        #     if not k == 0:
        #         an01.remove()
        #         an11.remove()
        #     an01 = sub2_pmat.annotate('X Unsicherheit', xy=(1, np.diag(p_mat)[0]), xytext=(1, np.diag(p_mat)[0]))
        #     an11 = sub2_pmat.annotate('Y Unsicherheit', xy=(2, np.diag(p_mat)[1]), xytext=(2, np.diag(p_mat)[1]))
        #     fig_ytild_p.canvas.draw()
        #     fig_ytild_p.canvas.flush_events()
        #
        # if extra_plotting:
        #     '''Einzelanalyse der Punkte mit Kreisen'''
        #     msg_x_est_temp = x_est
        #     # print('x= ' + str(msg_x_est))
        #     msg_yaw_rad = 0
        #     msg_z_meas = rss
        #     msg_y_est = rss
        #     msg_next_wp = x_n[k]
        #     # print('wp=' + str(msg_next_wp))
        #
        #     ekf_plotter.add_x_est_to_plot(msg_x_est_temp, msg_yaw_rad)
        #     ekf_plotter.update_next_wp(msg_next_wp)
        #     ekf_plotter.update_meas_circles(msg_z_meas, lambda_t, gamma_t, direct_terms, msg_y_est, b_plot_yest=False)
        #     ekf_plotter.plot_ekf_pos_live(b_yaw=False, b_next_wp=True)
        #     plt.show()  # Hier Breakpoint hinsetzen fuer Analyse der punkte
        #     # plt.pause(1)

    print('\nFertich!\n')

    '''
    Plotting
    '''

    '''Erstellung der X und Y Koordinatenlisten zum einfachen und effizienteren Plotten'''
    x_n_x = [None]*len(x_n)
    x_n_y = [None]*len(x_n)
    x_est_x = [None]*len(x_est_list)
    x_est_y = [None]*len(x_est_list)
    tx_pos_x = [None]*len(tx_pos)
    tx_pos_y = [None]*len(tx_pos)
    tx_pos_z = [None]*len(tx_pos)

    for i in range(0, len(x_n)):
        x_n_x[i] = x_n[i][0]
        x_n_y[i] = x_n[i][1]
    for i in range(0, len(x_est_list)):
        x_est_x[i] = x_est_list[i][0]
        x_est_y[i] = x_est_list[i][1]
    for i in range(0, len(tx_pos)):
        tx_pos_x[i] = tx_pos[i][0]
        tx_pos_y[i] = tx_pos[i][1]
        tx_pos_z[i] = tx_h[i]

    fig = plt.figure(2, figsize=(10, 2.5))
    # fig = plt.figure(1, figsize=(25, 12))
    '''Fehlerplot ueber Koordinaten'''
    plot_fehlerplotueberkoordinaten = False
    if plot_fehlerplotueberkoordinaten:
        plt.subplot(144)
    x_est_fehler = [None]*len(x_est_x)
    for i in range(3, len(x_n_x)):
        x_est_fehler[i] = get_distance_1D(x_est_x[i], x_n_x[i-1])
    #plt.plot(x_est_fehler)
    ymax = max(x_est_fehler)
    for i in range(3, len(x_n_y)):
        x_est_fehler[i] = get_distance_1D(x_est_y[i], x_n_y[i-1])
    #plt.plot(x_est_fehler)
    ymax = max([max(x_est_fehler), ymax])
    for i in range(3, len(x_est_list)):
        x_est_fehler[i] = get_distance_2D(x_est_list[i], x_n[i-1])
    #plt.plot(x_est_fehler)
    ymax = max([max(x_est_fehler), ymax])
    x_est_fehler_ges_mean = [np.mean(x_est_fehler[3:])]*len(x_est_x)
    x_est_fehler_ges_sdt = np.std(x_est_fehler[3:])
    if plot_fehlerplotueberkoordinaten:
        plt.plot(x_est_fehler_ges_mean, '--')
        plt.xlabel('Messungsnummer')
        plt.ylabel('Fehler')
        plt.legend(['Fehler X-Koordinate', 'Fehler Y-Koordinate', '(Gesamt-) Abstandsfehler',
                    ('Mittlerer Gesamtfehler = ' + str(np.round(x_est_fehler_ges_mean[0], 1)))], loc=0)
        plt.ylim(0, ymax + 300)

    '''
    if True:  # Analyse der Einzelmessungen fuer einfacheres Tuning
        ekf_plotter = ept.EKF_Plot(tx_pos, bplot_circles=True)
        # Einzelanalyse der Punkte mit Kreisen
        direct_terms = [None] * tx_num
        for i in range(tx_num):
            direct_terms[i] = np.log10(directivity_r * directivities_t[i])
        for iterx in range(len(x_n_x)):
            msg_x_est_temp = x_est_list[iterx]
            # print('x= ' + str(msg_x_est))
            msg_yaw_rad = 0
            msg_z_meas = rss_list[iterx]
            msg_y_est = rss_list[iterx]
            msg_next_wp = x_n[iterx]
            # print('wp=' + str(msg_next_wp))
    
            ekf_plotter.add_x_est_to_plot(msg_x_est_temp, msg_yaw_rad)
            ekf_plotter.update_next_wp(msg_next_wp)
            ekf_plotter.update_meas_circles(msg_z_meas, lambda_t, gamma_t, direct_terms,  msg_y_est, b_plot_yest=True)
            ekf_plotter.plot_ekf_pos_live(b_yaw=False, b_next_wp=True)
            plt.show()  # Hier Breakpoint hinsetzen fuer Analyse der punkte
    '''

    '''Strecke im Scatterplot'''
    plot_3d = False
    if plot_3d:
        # ax = fig.add_subplot(121, projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tx_pos_x, tx_pos_y, tx_pos_z, marker="*", c='k', s=100, depthshade=True, zorder=0)
        ax.scatter(x_n_x, x_n_y, h_mauv, marker="^", c='c', s=25, depthshade=True, zorder=1)
        ax.scatter(x_est_x, x_est_y, h_mauv, marker="o", c='r', s=100, depthshade=True, zorder=2)
        xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_n_y), min(tx_pos_y)])) - 100.0
        xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_n_y), max(tx_pos_y)])) + 100.0
        ymin = float(min([min(x_n_y), min(tx_pos_y)])) - 100.0
        ymax = float(max([max(x_n_y), max(tx_pos_y)])) + 100.0
        if ymax < xmax:
            ymax += 0.5*xmax
            ymin = ymax - xmax
        zmin = float(min([min(tx_pos_z), h_mauv]))
        zmax = float(max([max(tx_pos_z), h_mauv]))
        if zmax < xmax:
            zmax += 0.5*xmax
            zmin = zmax - xmax
        elif zmax < ymax:
            zmax += 0.5*ymax
            zmin = zmax - ymax
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X - Achse', fontsize=20)
        ax.set_ylabel('Y - Achse', fontsize=20)
        ax.set_zlabel('Z - Achse', fontsize=20)
        ax.grid()
        ax.legend(['Transmitter Antennen', 'Wahre Position', 'Geschaetzte Position'], loc=3)  # best:0, or=1, ol=2, ul=3, ur=4
        ax.set_title('Plot der wahren und geschaetzten Punkte', fontsize=25)
        # plt.show()
    else:
        plt.subplot(121)
        # plt.subplot(111)
        plt.plot(x_n_x, x_n_y, marker=".", c='c')
        plt.plot(x_est_x, x_est_y, marker=".", c='r')
        plt.scatter(tx_pos_x, tx_pos_y, marker="*", c='k', s=100)
        xmin = float(min([min(x_n_x), min(tx_pos_x), min(x_est_x)])) - 100.0
        xmax = float(max([max(x_n_x), max(tx_pos_x), max(x_est_x)])) + 100.0
        ymin = float(min([min(x_n_y), min(tx_pos_y), min(x_est_y)])) - 150.0  # - 350
        ymax = float(max([max(x_n_y), max(tx_pos_y), max(x_est_y)])) + 100.0
        ymax2 = ymin + ((ymax - ymin)/2) + ((xmax - xmin)/2)
        ymin2 = ymin + ((ymax - ymin)/2) - ((xmax - xmin)/2)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('X - Achse [mm]')
        plt.ylabel('Y - Achse [mm]')
        plt.grid()
        # plt.legend(['$x_text{wahr}', '$x_text{R,est}', '$x_text{Ti}'], loc=3, ncol=3)  # best:0, or=1, ol=2, ul=3, ur=4
        # plt.title('Plot der wahren und geschaetzten Punkte', fontsize=25)
        # plt.annotate('Z-Koordinate Sendeantennen: ' + str(tx_h[0]) + 'mm', xy=(xmin+50, ymax-50), xytext=(xmin+50, ymax-50))
        # plt.annotate('Z-Koordinate Empfaengerantenne: ' + str(h_mauv) + 'mm', xy=(xmin+50, ymax-160), xytext=(xmin+50, ymax-160))
        plt.annotate('Durchschnittlicher Fehler: ' + str(np.round(x_est_fehler_ges_mean[0], 0)) + ' +- ' + str(np.round(x_est_fehler_ges_sdt)) + 'mm', xy=(xmin+50, ymax-350), xytext=(xmin+50, ymax-350))
        plt.annotate('$x_text{0,est,wahr}', xy=(x_est_x[0], x_est_y[0]), xytext=(x_est_x[0]-50, x_est_y[0]-100))
        #plt.annotate('$x_text{0,wahr}', xy=(x_n_x[0], x_n_y[0]), xytext=(x_n_x[0]-50, x_n_y[0]-100))

        # plt.show()

    '''Strecke im Linienplot'''
    plot_streckeimlinienplot = True
    if plot_streckeimlinienplot:
        x_est_fehler = [None]*len((x_est_x))
        for i in range(len(x_est_x)):
            x_est_fehler[i] = get_distance_1D(x_est_x[i], x_est_y[i])
        # plt.figure(figsize=(12, 12))
        plt.subplot(143)
        plt.plot(range(1, (len(x_n)+1)), x_n_x)
        plt.plot(x_est_x)
        plt.plot(range(1, (len(x_n)+1)), x_n_y)
        plt.plot(x_est_y)
        plt.xlabel('Messungsnummer')
        plt.ylabel('Koordinate [mm]')
        # plt.legend(['$x_text{wahr}', '$x_text{est}', '$y_text{wahr}', '$y_text{est}'], loc=1)
        ymin = min([min(x_n_x), min(x_n_y), min(x_est_x), min(x_est_y)])
        ymax = max([max(x_n_x), max(x_n_y), max(x_est_x), max(x_est_y)])
        plt.ylim(ymin-100, ymax+100)
        locs2 = []
        labels2 = []
        plt.xticks(np.arange(0, len(x_n_x), step=20))

    '''Fehlerhistogramm'''
    plot_fehlerhistogramm = False
    if plot_fehlerhistogramm:
        plt.subplot(248)
        plt.hist(x_est_fehler[3:], 30, (0, 800))  # 800
        plt.xlabel('Fehler', fontsize=20)
        plt.ylabel('Anzahl der Vorkommnisse', fontsize=20)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4, hspace=None)
    plt.show()
