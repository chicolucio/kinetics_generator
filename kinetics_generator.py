import numpy as np
import scipy.stats
from uncertainties import unumpy
import matplotlib.pyplot as plt
from scipy import stats
import re


class KineticsGenerator:

    def __init__(self,
                 t0=0,
                 tf=1000,
                 size=10,
                 conc0=1,
                 k=1,
                 order=0,
                 sigma_time=0,
                 sigma_conc=0,
                 sigma_k=0):
        self.t0 = t0
        self.tf = tf
        self.size = size
        self.conc0 = conc0
        self.k = k
        self.order = order
        self.sigma_time = sigma_time
        self.sigma_conc = sigma_conc
        self.sigma_k = sigma_k

        def _rate_constant(self):
            return unumpy.uarray(self.k, self.sigma_k)

        self.rate = _rate_constant(self)

        def _initial_conc(self):
            random_err = scipy.stats.norm.rvs(
                loc=0, scale=self.sigma_conc, size=1)
            conc = self.conc0 + random_err
            return unumpy.uarray(conc, abs(random_err))

        self.conc0_u = _initial_conc(self)

        def _time_array(self):
            array = np.linspace(self.t0, self.tf, self.size)
            random_err = scipy.stats.norm.rvs(
                loc=0, scale=self.sigma_time, size=self.size)
            time = array + random_err
            time = unumpy.uarray(time, abs(random_err))
            time[time < 0] = 0
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

        array[array < 0] = 0
        self.conc_array_u = array

        def _half_life(self):
            if self.order == 0:
                half_life = self.conc0_u / (2 * self.rate)
                half_life = half_life[0]
            elif self.order == 1:
                half_life = np.log(2) / self.rate
            elif self.order == 2:
                half_life = 1 / (self.rate * self.conc0_u)
                half_life = half_life[0]
            else:
                raise ValueError('Order not valid')

            return half_life

        self.half_life = _half_life(self)

    def _plot_params(self, ax=None, plot_type='conc', time_unit='second',
                     formula='A', conc_unit='mol/L', size=12,
                     conc_or_p='conc'):
        linewidth = 2

        # grid and ticks settings
        ax.minorticks_on()
        ax.grid(b=True, which='major', linestyle='--',
                linewidth=linewidth - 0.5)
        ax.grid(b=True, which='minor', axis='x',
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

        return x_values, x_err, y_values, y_err

    def _linear_fit(self, x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value, p_value, std_err

    def plot(self, size=(8, 6), plot_type='conc', ax=None, time_unit='second',
             formula='A', conc_unit='mol/L', conc_or_p='conc',
             linear_fit=False):

        if ax is None:
            fig, ax = plt.subplots(figsize=size, facecolor=(1.0, 1.0, 1.0))

        self._plot_params(ax, plot_type=plot_type, time_unit=time_unit,
                          formula=formula, conc_unit=conc_unit,
                          conc_or_p=conc_or_p)

        x_values, x_err, y_values, y_err = self._xy(plot_type)

        ax.errorbar(x_values, y_values, fmt='ro', xerr=x_err,
                    yerr=y_err, ecolor='k', capsize=3)

        if linear_fit:
            slope, intercept, r_value, p_value, std_err = self._linear_fit(
                x_values, y_values)
            ax.plot(x_values, slope * x_values + intercept,
                    label='y={:.2E}x{:+.2E}  $R^2= {:.2f}$'.format(slope,
                                                                   intercept,
                                                                   r_value**2))

            ax.legend(loc='best', fontsize=14)

        return ax
