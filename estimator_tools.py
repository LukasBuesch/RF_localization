import numpy as np
import rf_tools
import hippocampus_toolbox as hc_tools
import time as t
import os
from os import path

t.time()


def get_meas_values(object, simulate_meas, measdata_filename=None):
    """

    :param object:
    :param measdata_filename: rel path to measfile
    :return:
    """

    '''get values from object'''
    num_tx = object.get_tx_num()
    txpos = object.get_tx_pos()

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 0
        measured_wp_list = []

        if simulate_meas:
            print('The used measuring data are simulated.\n')
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
                    plotdata_line = map(float, line.split(' '))
                    # currently wrong dimension and wrong values
                    plotdata_mat_lis.append(plotdata_line)

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

                    antenna_orientation = np.array([[0.0], [0.0], [1.0]])

                    wp_angles = [0.0] * num_tx * 4
                    for itx in range(num_tx):  # TODO: check this function (from Jonas)
                        pass
                        # wp_angles[itx * 4:itx * 4 + 4] = rf_tools.get_angles(np.transpose(wp_pos[0:2][np.newaxis]),
                        #                                                      np.transpose(txpos[itx, 0:2][np.newaxis]),
                        #                                                      txpos[itx, 2], antenna_orientation, wp_pos[2])
                    wp_angles = np.asarray(wp_angles)

                    plotdata_line = np.concatenate((wp_pos, mean, var, wp_angles),
                                                   axis=0)  # -> x,y,a,meantx1,...,meantxn,vartx1,...vartxn

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
        freqtx=[]
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


def get_distance(x_a, y_a, z_a, x_b, y_b, z_b):  # TODO: check if needed
    dist = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2) ** 0.5  # build distance vector and makes amount
    return dist


def rss_value_generator(tx_pos, wp_pos, add_noise=True):
    """
    generates RSS values for simulation purposes
    :param tx_pos: position of TX (vector)
    :param wp_pos: position of simulated receiver antenna
    :return: rss: simulated RSS value
    """
    pass
    # dist = get_distance()
    # rss = -20 * np.log10(dist) + dist * lambda_ti + gamma_ti + np.log10(np.cos(psi_low)) + n_tx * np.log10(
    #     np.cos(theta_cap)) + n_rec * np.log10(np.cos(theta_cap + theta_low))
    # if add_noise:
    #     tx_sigma = measurement_noise_model(r, theta_cap, psi_low, theta_low)
    #     rss += np.random.randn(1) * tx_sigma
    # return rss


def measurement_simulation(tx_pos, freqtx, way_filename, meas_filename):
    """
    simulates a measurement -> writes header like real measurements and in measdata rss values with variance
    :param tx_pos:
    :param freqtx:
    :param way_filename:
    :return:
    """
    if way_filename is not None:
        wplist_filename = path.relpath('Waypoints/' + way_filename + '.txt')
    else:
        wplist_filename = hc_tools.select_file(functionname='simulate_field_measurement')

    if meas_filename is not None:
        measdata_filename = path.relpath('Simulated_measurements/' + meas_filename + '.txt')
    else:
        measdata_filename = hc_tools.save_as_dialog()
    print(measdata_filename)

    meas_description = hc_tools.write_descrition()
    print meas_description

    numtx = len(freqtx)
    print('freqtx ' + str(freqtx))
    print('numtx ' + str(numtx))
    print('tx_pos ' + str(tx_pos))

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

        wp_data_mat = np.asarray(wp_append_list)
        print('wp_data_mat: ' + str(wp_data_mat))

    '''write values in measurement file'''
    with open(measdata_filename, 'w') as measfile:

        # write header to measurement file
        file_description = ('Measurement simulation file\n' + 'Simulation was performed on ' + t.ctime() + '\n'
                            + 'Description: ' + meas_description + '\n')

        txdata = str(numtx) + ' '
        for itx in range(numtx):
            txpos = tx_pos[itx]
            txdata += str(txpos[0]) + ' ' + str(txpos[1]) + ' ' + str(txpos[2]) + ' '
        for itx in range(numtx):
            txdata += str(freqtx[itx]) + ' '

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

        mean = [1, 1]
        var = [2, 2]
        for i in range(wp_data_mat.shape[0]):  # TODO: simulate values
            wp_pos = wp_data_mat[i][1:4]  # needed for error plots
            # for itx in range(numtx):
            #     mean[itx] = rss_value_generator()  # function that generates a rss mean value for certain distance
            # var = 1
            wp_angles = 1

            measfile.write(str(wp_pos[0]) + ' ' + str(wp_pos[1]) + ' ' + str(wp_pos[2]) + ' ')
            for itx in range(numtx):
                measfile.write(str(mean[itx]) + ' ')
            for itx in range(numtx):
                measfile.write(str(var[itx]) + ' ')
            measfile.write(str(wp_angles) + '\n')

    return True
