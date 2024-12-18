import numpy as np
import matplotlib.pyplot as plt
import scipy


m_0 = np.array([0, 1])
m_1 = np.array([-1, -1])
m_2 = np.array([2, 1])

b_0 = np.array([[0.35, 0.15], [0.1, 0.35]])

b_1 = np.array([[0.45, 0.15], [0.15, 0.45]])
b_2 = np.array([[0.15, 0.02], [0.02, 0.15]])
b_3 = np.array([[0.25, -0.17], [-0.17, 0.25]])


def get_errors_first(sample_1, sample_2, weights, w_n):
    counter_0 = 0
    counter_1 = 0

    w = weights[0:2]

    for x in sample_1:
        if w.T @ x + w_n > 0:
            counter_0 += 1

    for x in sample_2:
        if w.T @ x + w_n < 0:
            counter_1 += 1

    print(f"Ошибка первого рода: {counter_0 / sample_1.shape[0]}")
    print(f"Ошибка второго рода: {counter_1 / sample_2.shape[0]}")


def get_errors(sample_1, sample_2, w_popol):
    counter_0 = 0
    counter_1 = 0

    w = w_popol[0:2]
    w_n = w_popol[2]

    for x in sample_1:
        if w.T @ x + w_n > 0:
            counter_0 += 1

    for x in sample_2:
        if w.T @ x + w_n < 0:
            counter_1 += 1

    print(f"Ошибка первого рода: {counter_0 / sample_1.shape[0]}")
    print(f"Ошибка второго рода: {counter_1 / sample_2.shape[0]}")


def mahalanobis_distance(m0, m1, b):
    return (m1 - m0) @ np.linalg.inv(b) @ np.transpose(m1 - m0)


def boundary_of_bayes_classifier_for_n_with_same_b(x, m_l, m_j, b, threshold):
    m_dif = m_l - m_j
    m_dif_t = m_dif.reshape(1, 2)
    m_sum = (m_l + m_j)
    m_sum_t = m_sum.reshape(1, 2)
    b_inv = np.linalg.inv(b)

    # d_lj = d_l - d_j = 0
    # d_lj = ... = ax+b=0
    # a_0*x_0 + a_1*x_1 = const
    a = np.matmul(m_dif_t, b_inv)
    b = -0.5 * np.matmul(np.matmul(m_sum_t, b_inv), m_dif) + threshold  # page 31 случай 2

    return np.array((-b - a[0, 0] * x) / a[0, 1])


def get_count_fail(x, m_l, m_j, b, threshold):
    m_dif = m_l - m_j
    m_dif_t = m_dif.reshape(1, 2)
    m_sum = (m_l + m_j)
    m_sum_t = m_sum.reshape(1, 2)
    b_inv = np.linalg.inv(b)

    a = np.matmul(m_dif_t, b_inv)
    b = -0.5 * np.matmul(np.matmul(m_sum_t, b_inv), m_dif) + threshold  # page 31 случай 2

    # print(a)
    # print(x)

    counter = 0

    for xi in x:
        d_lj = np.matmul(a[0], xi.reshape(2, 1)) + b
        if d_lj < 0:
            counter += 1

    # print(counter)
    return counter / x.shape[0]


def boundary_of_bayes_classifier_for_n(x, m_l, m_j, b_l, b_j, threshold):
    m_l_t = m_l.reshape(1, 2)
    m_j_t = m_j.reshape(1, 2)

    b_l_inv = np.linalg.inv(b_l)
    b_j_inv = np.linalg.inv(b_j)

    det_b_l = np.linalg.det(b_l)
    det_b_j = np.linalg.det(b_j)

    # d_lj = d_l - d_j = 0
    # d_lj = ... = xTAx+bx+c=0
    a = 0.5 * (b_j_inv - b_l_inv)
    b = np.matmul(m_l_t, b_l_inv) - np.matmul(m_j_t, b_j_inv)
    c = (0.5 * np.matmul(np.matmul(m_j_t, b_j_inv), m_j) - 0.5 * np.matmul(np.matmul(m_l_t, b_l_inv), M_l)
         + 0.5 * np.log(det_b_j / det_b_l) + threshold)  # конец 4 лекции

    # print(f"A = {A}")
    # print(f"b = {b}")
    # print(f"c = {c}")

    boundary = []
    for x0 in x:
        coef_a = a[1, 1]
        coef_b = (a[1, 0] + a[0, 1]) * x0 + b[0, 1]
        coef_c = a[0, 0] * x0 ** 2 + b[0, 0] * x0 + c[0]
        d = coef_b ** 2 - 4 * coef_a * coef_c
        if d > 0:
            x1_0 = (-coef_b + np.sqrt(d)) / (2 * coef_a)
            x1_1 = (-coef_b - np.sqrt(d)) / (2 * coef_a)
            boundary.append([x0, x1_0])
            boundary.append([x0, x1_1])
        elif d == 0:
            x1_0 = -coef_b / (2 * coef_a)
            boundary.append([x0, x1_0])

    return np.array(boundary)


class mahalanobis_distance:
    pass


def get_erroneous_classification_probabilities(m_l, m_j, b):
    distance = mahalanobis_distance(m_l, m_j, b)
    print(f"Расстояние Махаланобиса: {distance}")
    p_0 = scipy.stats.norm.cdf(-0.5 * np.sqrt(distance))
    p_1 = 1 - scipy.stats.norm.cdf(0.5 * np.sqrt(distance))
    print(f"Ошибка первого рода: {p_0}")
    print(f"Ошибка второго рода: {p_1}")
    # R = 1/2(p_0+p_1) = 1 - Ф(0.5*sqrt(distance))
    print(f"Суммарный вероятность ошибочной классификации: {p_0 + p_1}")


def experimental_probability_error(x, m_l, m_j, b_l, b_j):
    count = 0

    calc_d = lambda vec, m, b, p: (np.log(p) - np.log(np.sqrt(np.linalg.det(b))) -
                                   0.5 * np.matmul(np.matmul((vec - m), np.linalg.inv(b)), (vec - m).reshape(2, 1))
                                   )  # page 29

    for xi in x:
        d_l = calc_d(xi, m_l, b_l, 0.5)
        d_j = calc_d(xi, m_j, b_j, 0.5)

        if d_j > d_l:
            count += 1

    return count / x.shape[0]


def get_eps(p, n):
    return np.sqrt((1 - p) / (n * p))


def get_n(p, err):
    return (1 - p) / (err ** 2 * p)


def task_1(boundary_of_bayes_classifier_for_n_with_same_b=None):
    # Data with same B
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    x = np.linspace(min_value, max_value, 100)
    threshold = np.log(0.5 / 0.5)
    y = boundary_of_bayes_classifier_for_n_with_same_b(x, m_0, m_1, b_0, threshold)

    # plt.suptitle("Байесовский классификатор")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y, color="red")
    # plt.show()

    get_erroneous_classification_probabilities(m_0, m_1, b_0)

    exper = experimental_probability_error(sample_1, m_0, m_1, b_0, b_0, 0.5, 0.5)
    print(f"Экспериментальная вероятность для БК {exper}")

    return x, y


def task_2():
    # Data with same B
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    x = np.linspace(min_value, max_value, 100)

    # MinMax classifier
    # p_0 = p_1 => Laplace functions equal => lambda = 0 => P(omega0) = P(omega1)
    p_0 = 0.5
    p_1 = 0.5
    threshold = np.log(p_0 / p_1)
    y_0 = boundary_of_bayes_classifier_for_n_with_same_b(x, m_0, m_1, b_0, threshold)

    # plt.suptitle("Минимаксный классификатор")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y_0, color="red")
    # plt.show()

    # Neyman-Pearson classifier
    p_0 = 0.05
    distance = mahalanobis_distance(m_0, m_1, b_0)
    # inv Laplace function for (1-p_0): p_0 = 0.05
    inv_laplace = 1.645
    lambda_tilda = -0.5 * distance + np.sqrt(distance) * inv_laplace
    y_1 = boundary_of_bayes_classifier_for_n_with_same_b(x, m_0, m_1, b_0, lambda_tilda)

    # plt.suptitle("Классификатор Неймана-Пирсона")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y_1, color="red")
    # plt.show()

    print(f"Экспериментальная верояность для Неймана-Пирсона: {get_count_fail(sample_1, m_0, m_1, b_0, lambda_tilda)}")

    return [(x, y_0), (x, y_1)]


def task_3():
    sample_1 = np.load("Files/arrayX3_1.npy")
    sample_2 = np.load("Files/arrayX3_2.npy")
    sample_3 = np.load("Files/arrayX3_3.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)
    sample_3 = np.transpose(sample_3)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]), np.min(sample_3[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]), np.max(sample_3[:, 0]))

    # x = np.linspace(-2, 2, 200)
    x = np.linspace(min_value, max_value, 200)

    boundary_01 = boundary_of_bayes_classifier_for_n(x, m_0, m_1, b_1, b_2, 0)
    boundary_02 = boundary_of_bayes_classifier_for_n(x, m_0, m_2, b_1, b_3, 0)
    boundary_12 = boundary_of_bayes_classifier_for_n(x, m_1, m_2, b_2, b_3, 0)

    plt.ylim(-2, 3)
    plt.xlim(min_value, max_value)

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='purple', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    plt.plot(sample_3[:, 0], sample_3[:, 1], color='blue', linestyle='none', marker='*')
    plt.scatter(boundary_01[:, 0], boundary_01[:, 1], color="red", s=[5])
    plt.scatter(boundary_02[:, 0], boundary_02[:, 1], color="red", s=[5])
    plt.scatter(boundary_12[:, 0], boundary_12[:, 1], color="red", s=[5])
    plt.show()

    p = experimental_probability_error(sample_1, m_0, m_1, b_1, b_2, 0.5, 0.5)
    print(f"Экспериментальные вероятности ошибочной классификации: {p}")

    eps = get_eps(p, sample_1.shape[0])
    print(f"Относительная погрешность: {eps}")

    n = get_n(p, 0.05)
    print(f"Объем обучающей выборки, обеспечивающий"
          f"получение оценок вероятностей ошибочной классификации"
          f"с погрешностью не более 5%: {n}")


def plot_task3(x_1, y_1_1, y_1_2, y_1_3):
    sample_1 = np.transpose(np.load("Files/arrayX2_1.npy"))
    sample_2 = np.transpose(np.load("Files/arrayX2_2.npy"))
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 7)
    ax[0].set_title("Байесовский классификатор")
    ax[0].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[0].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[0].plot(x_1, y_1_1, color="red")
    ax[1].set_title("Минимаксный классификатор")
    ax[1].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[1].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[1].plot(x_1, y_1_2, color="red")
    ax[2].set_title("Классификатор Неймана-Пирсона")
    ax[2].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[2].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[2].plot(x_1, y_1_3, color="red")
    plt.show()


def show_all_borders(borders):
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    plt.plot(borders[0][0], borders[0][1], color="red")
    plt.plot(borders[1][0], borders[1][1], color="purple", linestyle = '-.')
    plt.plot(borders[2][0], borders[2][1], color="pink")

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')

    plt.show()


def main():
    x_0, y_0 = task_1()
    value = task_2()
    x_1, y_1 = value[0]
    x_2, y_2 = value[1]
    # plot_task3(x_0, y_0, y_1, y_2)

    borders = [[x_0, y_0], [x_1, y_1], [x_2, y_2]]
    show_all_borders(borders)

    task_3()


if __name__ == "__main__":
    main()