import numpy as np
import matplotlib.pyplot as plt

def plot_cov_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    radius = n_std * np.sqrt(vals)
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)

    ellipse = (vecs @ (radius[:, None] * circle)) + mean[:, None]

    ax.plot(ellipse[0], ellipse[1], **kwargs)