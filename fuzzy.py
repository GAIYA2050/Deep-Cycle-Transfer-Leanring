import random
import numpy as np
import tensorflow as tf
import data_read as dr
import copy
import math

m = 2


def initialize_U(num_data, num_cluster):
    U = []
    for data in range(num_data):
        u_data = []
        sum_random = 0.0
        for cluster in range(num_cluster):
            u_data.append(random.random())
            sum_random += u_data[-1]
        for cluster in range(num_cluster):
            u_data[cluster] /= sum_random
        U.append(u_data)

    return U


def initialize_C(data, U):
    """
    :param data: features of the data
    :param U:
    :param m:
    :return:
    """
    num_cluster = len(U[0])
    C = []
    for j in range(num_cluster):
        current_cluster_center = []
        for i in range(len(data[0])):
            dummy_sum_num = 0.0
            dummy_sum_dum = 0.0
            for k in range(len(data)):
                # 分子
                dummy_sum_num += (U[k][j] ** m) * data[k][i]
                # 分母
                dummy_sum_dum += (U[k][j] ** m)
            # 第i列的聚类中心
            current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
        C.append(current_cluster_center)
    return C


def calculate_J(data, U, C):
    """
    计算目标函数J
    :param data:
    :param C:
    :return:
    """
    J = 0.0
    for num_C in range(len(C)):
        for num_data in range(len(data)):
            J += (np.power(U[num_data][num_C], m) * np.square(np.subtract(C[num_C], data[num_data])))
    return J


def distance(point, center):
    """
    该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。        闵可夫斯基距离
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    """
    结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    """
    epsm = 0.00000001
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > epsm:
                return False
    return True


def updata_U(data, U, UC_name):

    cluster_number = len(U[0])

    for iter in range(10):
        print('第%d 次迭代更新U' % iter)
        U_old = copy.deepcopy(U)
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    dummy_sum_dum += (U[k][j] ** m)
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            C.append(current_cluster_center)

        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)
        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("结束聚类")
            break

    np.savetxt('./fuzzy_data/Uy' + UC_name + '.txt', U, fmt="%.20f", delimiter=",")
    np.savetxt('./fuzzy_data/Cy' + UC_name + '.txt', C, fmt="%.20f", delimiter=",")

    return U, C


def getSubU(U, idx):
    target = []
    for i in idx:
        target.append(U[i])

    return target


def initialize_UC(C_initial, UC_name, source_dir, target_dir):
    x_images, x_id_list, x_len, x_labels = dr.get_source_batch(0, 256, 256, ".jpg", source_dir=source_dir)
    y_images, y_id_list, y_len, y_labels = dr.get_target_batch(0, 256, 256, ".png", target_dir=target_dir)

    x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y = tf.placeholder(tf.float32, [None, 256, 256, 3])

    # C_initial = Classifier('C', True, reuse=True)
    fx = C_initial(x)
    fy = C_initial(y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        x_data = []
        y_data = []

        for img in x_images:
            data = sess.run(fx, feed_dict={x: [img]})
            x_data.append(data[0])
        for img in y_images:
            data = sess.run(fy, feed_dict={y: [img]})
            y_data.append(data[0])

    Ux = initialize_U(x_len, 1)
    Uy = initialize_U(y_len, 2)
    Cx = initialize_C(x_data, Ux)
    Cy = initialize_C(y_data, Uy)

    np.savetxt('./fuzzy_data/Ux' + UC_name + '.txt', Ux, fmt="%.20f", delimiter=",")
    np.savetxt('./fuzzy_data/Uy' + UC_name + '.txt', Uy, fmt="%.20f", delimiter=",")
    np.savetxt('./fuzzy_data/Cx' + UC_name + '.txt', Cx, fmt="%.20f", delimiter=",")
    np.savetxt('./fuzzy_data/Cy' + UC_name + '.txt', Cy, fmt="%.20f", delimiter=",")

    return Ux, Uy, Cx, Cy, x_data, y_data, y_labels


def get_nearst_accuracy(datas, C, label):
    distances = []
    for data in datas:
        distance = []
        distance_1 = np.mean(np.sqrt(np.square(np.subtract(data, C[0]))))
        distance_2 = np.mean(np.sqrt(np.square(np.subtract(data, C[1]))))
        distance.append(distance_1)
        distance.append(distance_2)
        distances.append(distance)

    accuracy = 0
    for index in range(len(distances)):
        arg = np.argmin(distances[index])
        if arg != label[index]:
            accuracy += 1

    return accuracy / len(datas)


if __name__ == '__main__':
    # initialize_UC()
    pass
