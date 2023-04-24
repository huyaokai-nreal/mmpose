import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa


def plot_3d_pose(ax,
                 pose_3d_1,
                 bones,
                 linewidth=5,
                 alpha=0.95,
                 colormap='gist_rainbow',
                 autoAxisRange=True,
                 flip_yz=True,
                 change_view=True,
                 isGT=False):
    cmap = plt.get_cmap(colormap)
    pose_3d = pose_3d_1.copy()
    pose_3d = np.reshape(pose_3d.transpose(), (3, -1))
    pose_3d[1, :] *= -1

    if flip_yz:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(
            np.array(pose_3d[2, :])), np.squeeze(np.array(pose_3d[1, :]))
    else:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(
            np.array(pose_3d[1, :])), np.squeeze(np.array(pose_3d[2, :]))
    XYZ = np.vstack([X, Y, Z])

    if change_view:
        ax.view_init(elev=0, azim=-90)
    cmap = plt.get_cmap(colormap)

    maximum = len(bones)

    if not isGT:
        for i, bone in enumerate(bones):
            colorIndex = cmap.N - cmap.N * i / float(
                maximum)  # cmap.N - to start from back (nicer color)
            color = cmap(int(colorIndex))
            depth = max(XYZ[1, bone])
            # otherwise bones with be ordered in the order
            # of drawing or something even more weird...
            zorder = -depth
            ax.scatter3D(XYZ[0, :], XYZ[1, :], XYZ[2, :], c=color, s=50)
            ax.plot(
                XYZ[0, bone],
                XYZ[1, bone],
                XYZ[2, bone],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                solid_capstyle='round')
    else:
        for i, bone in enumerate(bones):
            depth = max(XYZ[1, bone])
            # otherwise bones with be ordered in the order of drawing
            # or something even more weird...
            zorder = -depth
            ax.plot(
                XYZ[0, bone],
                XYZ[1, bone],
                XYZ[2, bone],
                color='black',
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                solid_capstyle='round')

    # maintain aspect ratio
    if autoAxisRange:
        max_range = np.array(
            [X.max() - X.min(),
             Y.max() - Y.min(),
             Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
