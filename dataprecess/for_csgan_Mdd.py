import os
import numpy as np
import scipy.io
from scipy.linalg import eigh

def compute_covariance_matrices(data, labels, num_classes):
    channels = data.shape[1]
    Cov = np.zeros((channels, channels, num_classes))
    for c in range(num_classes):
        class_data = data[labels == c]
        covs = []
        for trial in class_data:
            C = np.cov(trial)    # trial: (channels, time)
            tr = np.trace(C)
            if tr == 0 or np.isnan(tr) or np.isinf(tr):

                C_norm = np.eye(channels)
            else:
                C_norm = C / tr
            covs.append(C_norm)
        if len(covs) == 0:

            Cov[:, :, c] = np.eye(channels)
        else:
            Cov[:, :, c] = np.mean(covs, axis=0)
    return Cov

def compute_csp_matrices(Cov):
    cov0, cov1 = Cov[:, :, 0], Cov[:, :, 1]
    composite_cov = cov0 + cov1

    eigvals, eigvecs = eigh(composite_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    P = eigvecs @ D_inv_sqrt @ eigvecs.T
    S0 = P @ cov0 @ P.T
    eigvals_s0, eigvecs_s0 = eigh(S0)
    order_s0 = np.argsort(eigvals_s0)[::-1]
    Wb = eigvecs_s0[:, order_s0].T @ P
    PP = np.zeros_like(Cov)
    BB = np.zeros_like(Cov)
    PP[:, :, 0] = Wb @ cov0 @ Wb.T
    PP[:, :, 1] = Wb @ cov1 @ Wb.T
    BB[:, :, 0] = PP[:, :, 0]
    BB[:, :, 1] = PP[:, :, 1]

    return P, PP, BB, Wb

def compute_mean_std(data, labels, num_classes):
    channels = data.shape[1]
    Dis_mean = np.zeros((channels, num_classes))
    Dis_std  = np.zeros((channels, num_classes))
    for c in range(num_classes):
        class_data = data[labels == c]
        if len(class_data) == 0:

            Dis_mean[:, c] = 0
            Dis_std[:, c] = 0
            continue
        mu    = class_data.mean(axis=2).mean(axis=0)
        sigma = class_data.mean(axis=2).std(axis=0)
        Dis_mean[:, c] = mu
        Dis_std [:, c] = sigma
    return Dis_mean, Dis_std


def generate_mat(data, labels, save_path):
    unique_labels = np.unique(labels)
    num_classes   = len(unique_labels)
    channels      = data.shape[1]
    n_trials      = data.shape[0]


    Cov = compute_covariance_matrices(data, labels, num_classes)


    if np.isnan(Cov).any() or np.isinf(Cov).any():
        print("wrong")
        return

    if num_classes == 1:
        Cov2 = np.zeros((channels, channels, 2), dtype=Cov.dtype)
        Cov2[:, :, 0] = Cov[:, :, 0]
        Cov2[:, :, 1] = np.eye(channels, dtype=Cov.dtype)
        Cov = Cov2
        num_classes = 2
        labels = labels.copy()


    P, PP, BB, Wb = compute_csp_matrices(Cov)


    Dis_mean, Dis_std = compute_mean_std(data, labels, num_classes)


    csp_data = np.zeros_like(data)
    for i in range(n_trials):
        csp_data[i] = Wb @ data[i]


    scipy.io.savemat(save_path, {
        'data'     : data.astype(np.float32),
        'csp_data' : csp_data.astype(np.float32),
        'label'    : labels.astype(np.int64),
        'Cov'      : Cov.astype(np.float32),
        'Dis_mean' : Dis_mean.astype(np.float32),
        'Dis_std'  : Dis_std.astype(np.float32),
        'PP'       : PP.astype(np.float32),
        'BB'       : BB.astype(np.float32),
        'Wb'       : Wb.astype(np.float32),
    })
    print(f"[OK] save {save_path}")

def main():
    base_dir = 'EC/sub_cross/H_S28~30_EC'
    out_dir  = 'for_gan_mat'
    os.makedirs(out_dir, exist_ok=True)

    for i in range(25,31):
        npy_name = f'H_S{i}_EC.npy'
        full_path = os.path.join(base_dir, npy_name)
        if not os.path.isfile(full_path):
            print(f"[skip] no {full_path}")
            continue

        data = np.load(full_path)  # (n_trials, channels, time)
        labels = np.zeros((data.shape[0],), dtype=np.int64)

        save_name = f'H_S{i}_EC.mat'
        save_path = os.path.join(out_dir, save_name)
        print(f"-> process {npy_name}")
        generate_mat(data, labels, save_path)

if __name__ == '__main__':
    main()
