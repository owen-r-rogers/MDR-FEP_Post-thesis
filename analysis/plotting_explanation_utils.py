import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from matplotlib import colormaps
from matplotlib.colors import LightSource


"""
This is where I'll store all of the functions and classes for making figures for my thesis.
"""


def make_color_list(vals_list, color1=(245, 66, 129), color2=(92, 244, 255)):
    """
    Make a list of colors using a LinearSegmentedColormap.
    The vals_list has to be a list of Y-values
    """
    # divide values to be between 0 and 1
    colors = [color1, color2]
    cols = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    # make a colormap
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', cols, N=256)

    # make a range for the colormap
    cspace = np.linspace(0, 1, len(vals_list))

    # map them together or something
    colors_list = colormap(cspace)

    return colormap, colors_list


def draw_colorbar(vals_list, axis, orientation, color1=(245, 66, 129), color2=(92, 244, 255)):
    """
    Plots a colorbar on the given axis based on the list of values passed
    """

    colormap, colors1 = make_color_list(vals_list, color1, color2)

    norm = matplotlib.colors.Normalize(vmin=np.min(vals_list), vmax=np.max(vals_list))
    scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalarmap.set_array([])

    colorbar = plt.colorbar(scalarmap, orientation=orientation, ax=axis)

    return colorbar


def plot_surface(surface_obj, azimuth, altitude, save=False, dpi=100, colormap='RdPu'):
    """
    For plotting free energy surface using an EnergySurface object
    """

    # extract arrays from surface object
    x_array = surface_obj.X_array()
    y_array = surface_obj.Y_array()
    z_array = surface_obj.Z_array()

    # implement shading
    ls = LightSource(azimuth, altitude)
    rgb = ls.shade(z_array, cmap=colormaps[colormap], vert_exag=0.1, blend_mode='soft')

    # make figure
    figure, axis = plt.subplots(dpi=dpi, subplot_kw={'projection': '3d'})
    axis.plot_surface(x_array, y_array, z_array, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False)
    axis.set_axis_off()
    axis.view_init(elev=altitude, azim=azimuth)

    plt.tight_layout()
    plt.show()

    if save:
        os.makedirs('./surfaces', exist_ok=True)

        figure.savefig(f'surfaces/surface_{azimuth}_{altitude}.png', dpi=dpi)


class EnergySurface:
    def __init__(self, lb, ub, step):

        self.lb = int(lb)
        self.ub = int(ub)
        self.step = int(step)

        self.x = np.linspace(self.lb, self.ub, self.step)
        self.y = np.linspace(self.lb, self.ub, self.step)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.zeros_like(self.X)

        self.features = []

    def add_feature(self, feature):
        if isinstance(feature, (Peak, Trough)):
            self.features.append(feature)
            self.Z += feature.add_to_Z_array(self.X, self.Y)
        else:
            raise ValueError('Feature must be Peak or Trough')

    def reset(self):
        self.Z = np.zeros_like(self.X)
        self.features = []

    def X_array(self):
        return self.X

    def Y_array(self):
        return self.Y

    def Z_array(self):
        return self.Z


class Peak:
    def __init__(self, height=1, width=0.1, center_x=0, center_y=0):
        self.height = height
        self.width = width
        self.center_x = center_x
        self.center_y = center_y

    def add_to_Z_array(self, X, Y):
        return self.height * np.exp(-((X - self.center_x) ** 2 + (Y - self.center_y) ** 2) / (2 * self.width ** 2))


class Trough:
    def __init__(self, height=1, width=0.1, center_x=0, center_y=0):
        self.height = height
        self.width = width
        self.center_x = center_x
        self.center_y = center_y

    def add_to_Z_array(self, X, Y):
        return -self.height * np.exp(-((X - self.center_x) ** 2 + (Y - self.center_y) ** 2) / (2 * self.width ** 2))


