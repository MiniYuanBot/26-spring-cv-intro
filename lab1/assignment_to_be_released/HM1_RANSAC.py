import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":

    np.random.seed(0)

    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here
    noise_points = np.loadtxt("HM1_ransac_points.txt")  # (130, 3)

    # RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0

    point_tot = noise_points.shape[0]
    confidence = 0.999
    outlier_ratio = 30 / 130
    min_sample = 3

    # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    sample_time = int(np.ceil(np.log(1 - confidence) /
                              np.log(1 - (1 - outlier_ratio) ** min_sample)))
    distance_threshold = 0.05

    # sample points group
    sample_idx_group = np.random.choice(np.arange(point_tot), size=(
        sample_time, min_sample), replace=False)  # (sample_time,)
    sample_group = noise_points[sample_idx_group]  # (sample_time, 3)

    # estimate the plane with sampled points group
    vec1_in_plane = sample_group[:, 0] - sample_group[:, 1]
    vec2_in_plane = sample_group[:, 0] - sample_group[:, 2]
    normal_group = np.cross(vec1_in_plane, vec2_in_plane)  # (sample_time, 3)
    normal_norm_group = np.sqrt(
        np.sum(normal_group * normal_group, axis=1))  # (sample_time,)
    D_group = - np.sum(normal_group * sample_group[:, 0], axis=1)  # (sample_time,)

    # evaluate inliers (with point-to-plance distance < distance_threshold)
    residual_group = (normal_group @ noise_points.T + D_group[:, None])
    residual_group = residual_group / normal_norm_group[:, None]  # (sample_time, 130)
    inliner_mask = (residual_group < distance_threshold) & (
        residual_group > -distance_threshold)
    inliner_tot_group = np.sum(inliner_mask, axis=1)  # (sample_time,)
    best_idx = np.argmax(inliner_tot_group)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    inliners = noise_points[inliner_mask[best_idx]]  # (inliner_tot, 3)
    inliner_tot = inliner_tot_group[best_idx]
    ones_column = np.ones((inliner_tot, 1))
    A = np.concatenate((inliners, ones_column), axis=1)  # (inliner_tot, 4)
    _, _, V = np.linalg.svd(A)
    pf = V[3, :]

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
