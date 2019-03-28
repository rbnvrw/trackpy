"""
Detect particles in brightfield mode using HoloPy.
"""
import io
from contextlib import redirect_stdout
import numpy as np
import xarray as xr
from holopy.core.metadata import update_metadata
import pandas as pd
import holopy as hp
from holopy.core.process import bg_correct, subimage, normalize
from holopy.core.metadata import get_spacing
from holopy.scattering import Sphere, calc_holo
from holopy.fitting import fit, Model
from holopy.fitting import Parameter as par

from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns)


def refine_holopy(image, radius, coords_df, pos_columns=None, **kwargs):
    """Find the center of mass of a brightfield feature starting from an
    estimate.

    Parameters
    ----------
    image : array (any dimension)
        processed image, used for locating center of mass
    radius : tuple(int, int)
        the estimated radii of the feature. Note: only the first value is used
        for now.
    coords_df : Series([x, y])
        estimated position of the feature
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.
    **kwargs:
        Passed to the min_edge function.

    Returns
    -------
    Series([x, y, r])
        where r means the radius of the fitted circle of dark pixels around
        the bright interior of the particle. Returns None on failure.
    """
    if not isinstance(coords_df, pd.core.series.Series) or len(coords_df) < 2 or len(coords_df) > 3:
        raise ValueError("Refine holopy only supports a Series" +
                         " of 1 particle with values x, y and optionally z")

    r = radius[0]
    result = _refine_holopy(image, r, coords_df, **kwargs)
    refined_r, refined_x, refined_y, refined_z = result

    if refined_r is None or refined_y is None or refined_x is None:
        return None

    coords_df['x'] = refined_x
    coords_df['y'] = refined_y
    coords_df['z'] = refined_z
    coords_df['r'] = refined_r

    return coords_df

def _refine_holopy(image, radius, coords_df, **kwargs):
    """Find the center of mass of a brightfield feature starting from an
    estimate.

    Parameters
    ----------
    image : array (any dimension)
        processed image, used for locating center of mass
    radius : int
        the estimated radius of the feature
    coords_df : DataFrame
        estimated positions

    Returns
    -------
    r : float
        the fitted radius of the feature
    x : float
        the fitted x coordinate of the feature
    y : float
        the fitted y coordinate of the feature
    """
    image = image.T
    coords_df = coords_df.astype(float)

    polarization = np.random.rand(2)
    polarization /= np.linalg.norm(polarization)
    default_args = {
            'spacing': 0.09, # micron / px
            'medium_index': 1.33, # water
            'illum_wavelen': 0.550, # micron
            'n_particle': 1.46, # silica
            'illum_polarization': polarization, # random
    }
    default_args.update(kwargs)
    spacing = default_args['spacing']

    umradius = radius * spacing

    frame_shape = image.shape
    dims = None
    if len(frame_shape) == 2:
        dims = ['z', 'x', 'y']
        coords = [[0], np.arange(0, frame_shape[0]) * spacing, np.arange(0, frame_shape[1]) * spacing]
    else:
        print('Unknown dimensions!')
        exit(1)

    image_data = xr.DataArray(data=[image], coords=coords, dims=dims)
    image_data = update_metadata(image_data, default_args['medium_index'], default_args['illum_wavelen'], default_args['illum_polarization'], None)

    # process the image
    search_range = 1.5*radius
    x = coords_df['y']
    y = coords_df['x']
    minx = int(np.round(x-search_range))
    if minx < 0:
        minx = 0
    maxx = int(np.round(x+search_range))
    if maxx >= frame_shape[0]:
        maxx = frame_shape[0]-1
    miny = int(np.round(y-search_range))
    if miny < 0:
        miny = 0
    maxy = int(np.round(y+search_range))
    if maxy >= frame_shape[1]:
        maxy = frame_shape[1]-1
    image_data = image_data[:, minx:maxx, miny:maxy]
    data_holo = normalize(image_data)

    x *= spacing
    y *= spacing

    dev = 0.5*umradius
    if 'z' in coords_df and coords_df['z'] is not None and ~np.isnan(coords_df['z']):
        z = coords_df['z'] * spacing
        zlim = [z-dev, z+dev]
    else:
        z = umradius
        zlim = [-5.0*umradius, 5.0*umradius]

    x_guess = par(guess=x, limit=[x-dev, x+dev])
    y_guess = par(guess=y, limit=[y-dev, y+dev])
    z_guess = par(guess=z, limit=zlim)
    r_guess = par(guess=umradius, limit=[umradius*(1.0-0.03), umradius*(1.0+0.03)])

    par_s = Sphere(center=(x_guess, y_guess, z_guess), r=r_guess, n=default_args['n_particle'])

    model = Model(par_s, calc_holo, alpha=par(1.0, [.1, 1.0]))

    f = io.StringIO()
    with redirect_stdout(f):
        result = fit(model, data_holo, random_subset=0.5)

    coords = result.scatterer.center

    return radius, coords[1]/spacing, coords[0]/spacing, coords[2]/spacing



