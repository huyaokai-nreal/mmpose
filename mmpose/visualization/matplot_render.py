import cv2
import io
import matplotlib.pyplot as plt
import numpy as np


def get_cv2mat_from_buf(fig, dpi=180):
    """Get numpy image from IO."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_3d_pose(ax,
                 keypints3d,
                 bones,
                 linewidth=5,
                 alpha=0.95,
                 colormap='gist_rainbow',
                 auto_axis_range=True,
                 flip_yz=True,
                 is_gt=False):
    cmap = plt.get_cmap(colormap)
    pose_3d = keypints3d.copy()
    pose_3d = np.reshape(pose_3d.transpose(), (3, -1))
    pose_3d[1, :] *= -1

    if flip_yz:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(
            np.array(pose_3d[2, :])), np.squeeze(np.array(pose_3d[1, :]))
    else:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(
            np.array(pose_3d[1, :])), np.squeeze(np.array(pose_3d[2, :]))
    XYZ = np.vstack([X, Y, Z])
    cmap = plt.get_cmap(colormap)
    maximum = len(bones)
    if not is_gt:
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
            ax.scatter3D(XYZ[0, :], XYZ[1, :], XYZ[2, :], c='red', s=50)
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
    if auto_axis_range:
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
