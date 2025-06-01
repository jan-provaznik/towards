# 2025 Jan Provaznik (provaznik@optics.upol.cz)
# 
# This module implements the certification procedure.
# Please refer to the documentation strings of the functions for details.

import numpy as np

@np.vectorize(excluded = { 'x_mul', 'y_mul', 'curve' }, otypes = [ bool ])
def threshold_curve_certify (curve, x_avg, x_std, x_mul, y_avg, y_std, y_mul):
    '''
    Certifies that the uncertainty rectangle does NOT intersect the particular
    threshold curve.

    Parameters
    ----------
    curve : scipy.interpolate.CubicSpline
      Threshold curve. 
      For example the hierarchical curve of the (Lachman, 2019) variety.
    x_avg : float
    x_std : float
    x_mul : float
      The mean X coordinate, its standard deviation, 
      and the certification factor.
    y_avg : float
    y_std : float
    y_mul : float
      The mean Y coordinate, its standard deviation,
      and the certification factor.

    Returns
    -------
    bool
      Indicates whether the bounding box around the mean (X, Y) lies 
      above the threshold curve.
    '''

    # Determine the threshold value to check against.
    # - Use the bottom of the bounding rectangle by default.
    #   If it's negative, there won't be any solution. Instead,
    # - use the top of the rectangle. If it's above 1, 
    #   there won't be any solution either. Bail out.

    threshold = y_avg - y_std * y_mul
    if threshold < 0.0:
        threshold = y_avg + y_std * y_mul
        if threshold > 1.0:
            return False

    # Find solutions...

    solutions = curve.solve(
        threshold,
        extrapolate = False
    )

    # The threshold value is above the Lachman curve,
    # there is no interection (no solutuion) and 
    # we are in the clear.
    
    if (solutions.size == 0):
        return True

    # The Lachman curve is a concave function, 
    # if there are any intersections, there should be exactly two.
    
    if (solutions.size != 2):
        return False

    solutionL, solutionR = np.sort(solutions)
    rectangleM = x_avg - x_std * x_mul
    rectangleP = x_avg + x_std * x_mul

    # The rectangle lies wihin...

    if (rectangleM <= solutionL) and (solutionR <= rectangleP):
        return False
    if (solutionL <= rectangleM) and (rectangleM <= solutionR):
        return False
    if (solutionL <= rectangleP) and (rectangleP <= solutionR):
        return False

    # The rectangle can only lie outside the curve, either on left or right.
    # The rectangle has been certified. Congratulations, it's a success.

    return True

