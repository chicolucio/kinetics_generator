import numpy as np
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt
from scipy import stats, odr
import re


class KineticsGenerator:

    def __init__(self,
                 time=1000,
                 size=10,
                 conc0=1,
                 rate_constant=1,
                 order=0,
                 sigma_time=0,
                 sigma_conc=0,
                 sigma_rate=0):
        self.t0 = 0
        self.tf = time
        self.size = size
        self.conc0 = conc0
        self.k = rate_constant
        self.order = order
        self.sigma_time = sigma_time
        self.sigma_conc = sigma_conc
        self.sigma_k = sigma_rate

        def _rate_constant(self):
            """Joins nominal value and standard deviation for rate constant

            Returns
            -------
            numpy.ndarray
                Rate constant with nominal value and standard deviation
            """
            return unumpy.uarray(self.k, self.sigma_k)

        self.rate = _rate_constant(self)

        def _initial_conc(self):
            """Joins nominal value and standard deviation for initial
            concentration

            Returns
            -------
            numpy.ndarray
                Initial concentration with nominal value and standard deviation
            """
            return ufloat(self.conc0, self.sigma_conc)

        self.conc0_u = _initial_conc(self)

        def _time_array(self):
            """Creates a time array with values and the standard deviation

            Returns
            -------
            numpy.ndarray
                Time array with nominal values and standard deviation
            """
            array = np.linspace(self.t0, self.tf, self.size)
            time = unumpy.uarray(array, self.sigma_time)
            return time

        self.time_array_u = _time_array(self)

        if order == 0:
            array = -(self.rate * self.time_array_u) + self.conc0_u
        elif order == 1:
            array = self.conc0_u * \
                unumpy.exp(self.rate * self.time_array_u * (-1))
        elif order == 2:
            array = self.conc0_u / \
                (1 + self.rate * self.time_array_u * self.conc0_u)
        else:
            raise ValueError('Order not valid')
        self.conc_array_u = array

        def _half_life(self):
            """Calculates the half life for the data

            Returns
            -------
            uncertainties.core.AffineScalarFunc
                Half life with nominal value and standard deviation

            Raises
            ------
            ValueError
                Error when order is not 0, 1 or 2
            """
            if self.order == 0:
                half_life = self.conc0_u / (2 * self.rate)
            elif self.order == 1:
                half_life = np.log(2) / self.rate
            elif self.order == 2:
                half_life = 1 / (self.rate * self.conc0_u)
            else:
                raise ValueError('Order not valid')

            return half_life

        self.half_life = _half_life(self)

    def _plot_params(self, ax=None, plot_type='conc', time_unit='second',
                     formula='A', conc_unit='mol/L', size=12,
                     conc_or_p='conc'):
        """Internal function for plot parameters.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            axes for the plot, by default None
        plot_type : str, optional
            Plot type. 'conc' for concentration vs time; 'ln_conc' for
            ln(concentration) vs time; or 'inv_conc' for 1/concentration vs
            time.By default 'conc'
        time_unit : str, optional
            Time unit, by default 'second'
        formula : str, optional
            Chemical formula to be shown in vertical axis, by default 'A'
        conc_unit : str, optional
            Concentration unit, by default 'mol/L'
        size : int, optional
            Size reference for labels and text, by default 12
        conc_or_p : str, optional
            If data is for concentration ('conc') or pressure ('p'), by default
            'conc'

        Raises
        ------
        ValueError
            Error when plot_type is not 'conc', 'ln_conc' or 'inv_conc'
        """

        linewidth = 2

        # grid and ticks settings
        ax.minorticks_on()
        ax.grid(b=True, which='major', linestyle='--',
                linewidth=linewidth - 0.5)
        ax.grid(b=True, which='minor', axis='both',
                linestyle=':', linewidth=linewidth - 1)
        ax.tick_params(which='both', labelsize=size+2)
        ax.tick_params(which='major', length=6, axis='both')
        ax.tick_params(which='minor', length=3, axis='both')

        ax.set_xlabel('Time / {}'.format(time_unit), size=size+3)

        label_formula = re.sub("([0-9])", "_\\1", formula)

        if conc_or_p == 'conc':
            label_formula = '$\mathregular{['+label_formula+']}$'
        elif conc_or_p == 'p':
            label_formula = r'$\mathregular{P_{'+label_formula+r'}}$'

        if plot_type == 'conc':
            ax.set_ylabel('{0} / {1}'.format(label_formula, conc_unit),
                          size=size+3)
        elif plot_type == 'ln_conc':
            ax.set_ylabel('ln({0})'.format(label_formula),
                          size=size+3)
        elif plot_type == 'inv_conc':
            ax.set_ylabel('1/{0}'.format(label_formula) + r' / $\mathregular{{({0})^{{-1}}}}$'.format(conc_unit),  # NoQA
                          size=size+3)
        else:
            raise ValueError('Plot type not valid')

        return

    def _xy(self, plot_type='conc'):
        """Returns the x and y values, and their errors, for plots

        Parameters
        ----------
        plot_type : str, optional
            Plot type. 'conc' for concentration vs time; 'ln_conc' for
            ln(concentration) vs time; or 'inv_conc' for 1/concentration vs
            time.By default 'conc'

        Returns
        -------
        tuple
            x values, x errors, y values and y errors

        Raises
        ------
        ValueError
            Error when plot_type is not 'conc', 'ln_conc' or 'inv_conc'
        """
        if plot_type == 'conc':
            x = self.time_array_u
            y = self.conc_array_u
        elif plot_type == 'ln_conc':
            x = self.time_array_u
            y = unumpy.log(self.conc_array_u)
        elif plot_type == 'inv_conc':
            x = self.time_array_u
            y = 1/(self.conc_array_u)
        else:
            raise ValueError('Plot type not valid')

        x_values = unumpy.nominal_values(x)
        y_values = unumpy.nominal_values(y)

        x_err = unumpy.std_devs(x)
        y_err = unumpy.std_devs(y)

        x_err[x_err == 0] = 1e-30
        y_err[y_err == 0] = 1e-30

        return x_values, x_err, y_values, y_err

    def _linear_fit(self, x, y):
        """Ordinary least squares (OLS) regression for x and y

        Parameters
        ----------
        x : array
            x values
        y : array
            y values

        Returns
        -------
        tuple
            Regression results
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value**2, p_value, std_err

    def _linear_func(self, B, x):
        """Linear function to be used as model for orthogonal distance
        regression (ODR).

        Parameters
        ----------
        B : array
            Parameters [slope, intercept]
        x : array
            x values

        Returns
        -------
        array
            array of y values
        """
        return B[0]*x + B[1]

    def _odr_r_squared(self, y_pred, y):
        """R2 calculation for ODR

        Parameters
        ----------
        y_pred : array
            predicted values from ODR
        y : [type]
            values passed to ODR

        Returns
        -------
        float
            R squared
        """
        y_abs_error = y_pred - y
        r2 = 1 - (np.var(y_abs_error) / np.var(y))
        return r2

    def _odr(self, x, y, x_err, y_err):
        """Orthogonal distance regression (ODR).

        Parameters
        ----------
        x : array
            x values
        y : array
            y values
        x_err : array
            x standard deviations
        y_err : array
            y standard deviations

        Returns
        -------
        tuple
            Regression results
        """
        lin_reg = self._linear_fit(x, y)
        linear_model = odr.Model(self._linear_func)
        data = odr.RealData(x, y, sx=x_err, sy=y_err)
        odr_fit = odr.ODR(data, linear_model, beta0=lin_reg[0:2])
        out = odr_fit.run()

        slope = out.beta[0]
        intercept = out.beta[1]
        r2 = self._odr_r_squared(out.y, y)
        slope_std_err = out.sd_beta[0]
        intercept_std_err = out.sd_beta[0]

        return slope, intercept, r2, slope_std_err, intercept_std_err

    def plot(self, size=(8, 6), plot_type='conc', ax=None, time_unit='second',
             formula='A', conc_unit='mol/L', conc_or_p='conc',
             linear_fit=False):
        """Plot function

        Parameters
        ----------
        size : tuple, optional
            Plot size, by default (8, 6)
        plot_type : str, optional
            Plot type. 'conc' for concentration vs time; 'ln_conc' for
            ln(concentration) vs time; or 'inv_conc' for 1/concentration vs
            time.By default 'conc'
        ax : matplotlib.axes, optional
            axes for the plot, by default None
        time_unit : str, optional
            Time unit, by default 'second'
        formula : str, optional
            Chemical formula to be shown in vertical axis, by default 'A'
        conc_unit : str, optional
            Concentration unit, by default 'mol/L'
        If data is for concentration ('conc') or pressure ('p'), by default
        'conc'
        linear_fit : bool, optional
            If a linear regression line will be shown, by default False

        Returns
        -------
        matplotlib.axes
            Plot
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=size, facecolor=(1.0, 1.0, 1.0))

        self._plot_params(ax, plot_type=plot_type, time_unit=time_unit,
                          formula=formula, conc_unit=conc_unit,
                          conc_or_p=conc_or_p)

        x_values, x_err, y_values, y_err = self._xy(plot_type)

        ax.errorbar(x_values, y_values, fmt='ro', xerr=x_err,
                    yerr=y_err, ecolor='k', capsize=3)

        if linear_fit:
            slope, intercept, r2, slope_std_err, intercept_std_err = self._odr(
                x_values, y_values, x_err, y_err)
            ax.plot(x_values, slope * x_values + intercept,
                    label='y={:.2E}x{:+.2E}  $R^2= {:.2f}$'.format(slope,
                                                                   intercept,
                                                                   r2))

            ax.legend(loc='best', fontsize=14)

        return ax

    def export_csv(self, conc_label='[A]', time_label='time',
                   filename='kinetics_generator_data'):
        """Creates a csv file with concentration and time values

        Parameters
        ----------
        conc_label : str, optional
            Label for concentration/pressure, by default '[A]'
        time_label : str, optional
            Label for time, by default 'time'
        filename : str, optional
            File name (without extension), by default 'kinetics_generator_data'
        """
        conc = unumpy.nominal_values(self.conc_array_u)
        conc_unc = unumpy.std_devs(self.conc_array_u)

        time = unumpy.nominal_values(self.time_array_u)
        time_unc = unumpy.std_devs(self.time_array_u)

        header_string = conc_label + ',' + conc_label + \
            ' error' + ',' + time_label + ',' + time_label + ' error'

        file = filename + '.csv'

        np.savetxt(file, list(zip(conc, conc_unc, time, time_unc)),
                   header=header_string, delimiter=",")
        return
