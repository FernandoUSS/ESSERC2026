from scipy import interpolate
from scipy import optimize
import numpy as np



def get_abscissa(x_data, y_data, ordinate, xmin=None, xmax=None, ymin=None, ymax=None, scale="lin-log", interpolator_function=None):

    """
    This function calculates the abscissa (x-coordinate) of a point, by interpolating on a given dataset!
    The dataset must represent an invertible function in the given range [xmin, xmax] and [ymin, ymax]!
    :param x_data: 1D array of x points
    :param y_data: 1D array of y points
    :param ordinate: ordinate (y-coordinate) of considered point
    :param xmin: minimum x-value of interpolation range
    :param xmax: maximum x-value of interpolation range
    :param ymin: minimum y-value of interpolation range
    :param ymax: maximum y-value of interpolation range
    :param scale: scaling of data, choose between ['lin-lin', 'log-lin', 'lin-log', 'log-log']
    :param interpolator_function: interpolator used to interpolate the data, if None then PchipInterpolator is used
    :return: abscissa, ordinate, interpolator
    """

    # test if data is numeric 
    if not (np.all(np.isfinite(x_data)) and np.all(np.isfinite(y_data))):
    #    raise ValueError("Data contains non-numeric entries (nan, inf)") Commented by Fernando
        print("Warning: Data contains non-numeric entries (nan, inf). They will be removed")
        y_data_new = y_data[~np.isnan(y_data) & (y_data > 0)]
        x_data = x_data[~np.isnan(y_data) & (y_data > 0)]
        y_data = y_data_new



    # select slice of curve
    if xmin is not None:
        cond = xmin < x_data
        x_data = x_data[cond]
        y_data = y_data[cond]
    # else:
    #     xmin = np.min(x_data) # Commented by Fernando

    if xmax is not None:
        cond = x_data < xmax
        x_data = x_data[cond]
        y_data = y_data[cond]
    # else:
    #     xmax = np.max(x_data) # Commented by Fernando

    if ymin is not None:
        cond = ymin < y_data
        x_data = x_data[cond]
        y_data = y_data[cond]
    # else:
    #     ymin = np.max(y_data) # Commented by Fernando

    if ymax is not None:
        cond = y_data < ymax
        x_data = x_data[cond]
        y_data = y_data[cond]
    # else:
    #    ymax = np.max(y_data) # Commented by Fernando

    if len(x_data) == 0 or len(y_data) == 0:
        print("No data points remain in the up sweep after applying ymin/ymax filters. Consider adjusting these thresholds.")
        abscissa = np.nan
        ordinate = np.nan
        interpolator = None
    else:
        # get actual min and max (added by Fernando)
        xmin = np.min(x_data)
        xmax = np.max(x_data)
        ymin = np.min(y_data)
        ymax = np.max(y_data)

        # check if function is invertible
        order = x_data.argsort()
        x_data, y_data = x_data[order], y_data[order]
        x_increasing = np.all(np.diff(x_data) > 0)
        y_increasing = np.all(np.diff(y_data) > 0)
        y_decreasing = np.all(np.diff(y_data) < 0)

        if not (x_increasing and (y_increasing or y_decreasing)):
            print("Warning: curve is not monotonically increasing/decreasing, inversion may be wrong")
        
        # scale data
        if scale == "lin-lin":
            scaled_xmin = xmin
            scaled_xmax = xmax
            scaled_x_data = x_data
            scaled_y_data = y_data
            scaled_ordinate = ordinate
        elif scale == "log-lin":
            scaled_xmin = np.log10(xmin)
            scaled_xmax = np.log10(xmax)
            scaled_x_data = np.log10(x_data)
            scaled_y_data = y_data
            scaled_ordinate = ordinate
        elif scale == "lin-log":
            scaled_xmin = xmin
            scaled_xmax = xmax
            scaled_x_data = x_data
            scaled_y_data = np.log10(y_data)
            scaled_ordinate = np.log10(ordinate)
        elif scale == "log-log":
            scaled_xmin = np.log10(xmin)
            scaled_xmax = np.log10(xmax)
            scaled_x_data = np.log10(x_data)
            scaled_y_data = np.log10(y_data)
            scaled_ordinate = np.log10(ordinate)
        else:
            raise ValueError("kind must be one of the following ['lin-lin', 'log-lin', 'lin-log', 'log-log]")

        # initialize interpolator
        if interpolator_function is None:
            try:
                scaled_interpolator = interpolate.PchipInterpolator(x=scaled_x_data, y=scaled_y_data)
            except ValueError:
                print("Warning: PchipInterpolator failed")
                scaled_interpolator = lambda x: np.nan*x
        else:
            scaled_interpolator = interpolator_function(x=scaled_x_data, y=scaled_y_data)

        # calculate abscissa added by Fernando
        if (scaled_interpolator(scaled_xmin) - scaled_ordinate) * (scaled_interpolator(scaled_xmax) - scaled_ordinate) > 0:
            scaled_abscissa = np.nan
        else:
            scaled_abscissa = optimize.root_scalar(lambda x: scaled_interpolator(x) - scaled_ordinate, x0=(scaled_xmin + scaled_xmax)/2, bracket=[scaled_xmin, scaled_xmax], method="bisect").root

        # Commented by Fernando
        # scaled_abscissa = optimize.root_scalar(lambda x: scaled_interpolator(x) - scaled_ordinate, x0=(scaled_xmin + scaled_xmax)/2, bracket=[scaled_xmin, scaled_xmax], method="bisect").root

        #np.savetxt("test.csv", np.column_stack((scaled_x_data, scaled_y_data)), delimiter="\t")
        # rescale data
        if scale == "lin-lin":
            abscissa = scaled_abscissa
            interpolator = lambda x: scaled_interpolator(x)
        elif scale == "log-lin":
            abscissa = np.power(10, scaled_abscissa)
            interpolator = lambda x: scaled_interpolator(np.log10(x))
        elif scale == "lin-log":
            abscissa = scaled_abscissa
            interpolator = lambda x: np.power(10, scaled_interpolator(x))
        elif scale == "log-log":
            abscissa = np.power(10, scaled_abscissa)
            interpolator = lambda x: np.power(10, scaled_interpolator(np.log10(x)))

    return abscissa, ordinate, interpolator

def get_ordinate(x_data, y_data, abscissa, xmin=None, xmax=None, ymin=None, ymax=None, scale="lin-log", interpolator_function=None):

    """
    This function calculates the ordinate (y-coordinate) of a point, by interpolating on a given dataset!
    The dataset must represent a function in the given range [xmin, xmax] and [ymin, ymax]!
    :param x_data: 1D array of x points
    :param y_data: 1D array of y points
    :param abscissa: abscissa (x-coordinate) of considered point
    :param xmin: minimum x-value of interpolation range
    :param xmax: maximum x-value of interpolation range
    :param ymin: minimum y-value of interpolation range
    :param ymax: maximum y-value of interpolation range
    :param scale: scaling of data, choose between ['lin-lin', 'log-lin', 'lin-log', 'log-log']
    :param interpolator_function: interpolator used to interpolate the data, if None then PchipInterpolator is used
    :return: abscissa, ordinate, interpolator
    """

    # test if data is numeric 
    if not (np.all(np.isfinite(x_data)) and np.all(np.isfinite(y_data))):
        print("Warning: Data contains non-numeric entries (nan, inf). They will be removed")
        y_data_new = y_data[~np.isnan(y_data) & (y_data > 0)]
        x_data = x_data[~np.isnan(y_data) & (y_data > 0)]
        y_data = y_data_new

    # select slice of curve
    if xmin is not None:
        cond = xmin < x_data
        x_data = x_data[cond]
        y_data = y_data[cond]

    if xmax is not None:
        cond = x_data < xmax
        x_data = x_data[cond]
        y_data = y_data[cond]

    if ymin is not None:
        cond = ymin < y_data
        x_data = x_data[cond]
        y_data = y_data[cond]

    if ymax is not None:
        cond = y_data < ymax
        x_data = x_data[cond]
        y_data = y_data[cond]

    if len(x_data) == 0 or len(y_data) == 0:
        print("No data points remain after applying filters. Consider adjusting these thresholds.")
        abscissa = np.nan
        ordinate = np.nan
        interpolator = None
    else:
        # get actual min and max
        xmin = np.min(x_data)
        xmax = np.max(x_data)
        ymin = np.min(y_data)
        ymax = np.max(y_data)

        # check if function is monotonic
        order = x_data.argsort()
        x_data, y_data = x_data[order], y_data[order]
        x_increasing = np.all(np.diff(x_data) > 0)
        y_increasing = np.all(np.diff(y_data) > 0)
        y_decreasing = np.all(np.diff(y_data) < 0)

        if not x_increasing:
            print("Warning: x data is not monotonically increasing")
        
        # scale data
        if scale == "lin-lin":
            scaled_xmin = xmin
            scaled_xmax = xmax
            scaled_x_data = x_data
            scaled_y_data = y_data
            scaled_abscissa = abscissa
        elif scale == "log-lin":
            scaled_xmin = np.log10(xmin)
            scaled_xmax = np.log10(xmax)
            scaled_x_data = np.log10(x_data)
            scaled_y_data = y_data
            scaled_abscissa = np.log10(abscissa)
        elif scale == "lin-log":
            scaled_xmin = xmin
            scaled_xmax = xmax
            scaled_x_data = x_data
            scaled_y_data = np.log10(y_data)
            scaled_abscissa = abscissa
        elif scale == "log-log":
            scaled_xmin = np.log10(xmin)
            scaled_xmax = np.log10(xmax)
            scaled_x_data = np.log10(x_data)
            scaled_y_data = np.log10(y_data)
            scaled_abscissa = np.log10(abscissa)
        else:
            raise ValueError("scale must be one of ['lin-lin', 'log-lin', 'lin-log', 'log-log']")

        # initialize interpolator
        if interpolator_function is None:
            scaled_interpolator = interpolate.PchipInterpolator(x=scaled_x_data, y=scaled_y_data)
        else:
            scaled_interpolator = interpolator_function(x=scaled_x_data, y=scaled_y_data)

        # calculate ordinate
        scaled_ordinate = scaled_interpolator(scaled_abscissa)

        # rescale data
        if scale == "lin-lin":
            ordinate = scaled_ordinate
            interpolator = lambda x: scaled_interpolator(x)
        elif scale == "log-lin":
            ordinate = scaled_ordinate
            interpolator = lambda x: scaled_interpolator(np.log10(x))
        elif scale == "lin-log":
            ordinate = np.power(10, scaled_ordinate)
            interpolator = lambda x: np.power(10, scaled_interpolator(x))
        elif scale == "log-log":
            ordinate = np.power(10, scaled_ordinate)
            interpolator = lambda x: np.power(10, scaled_interpolator(np.log10(x)))

    return abscissa, ordinate, interpolator


