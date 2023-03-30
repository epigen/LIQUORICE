import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import typing
import matplotlib

from liquorice.utils import FitGaussians


def polyfit(x: typing.List[float], y: typing.List[float], degree: int) -> dict:
    """
    fit a trendline to data, see https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy

    :param x: x values
    :param y: y values
    :param degree: 1 for linear, 2 for quadratic, etc.
    :return: A Dictionary, containing lists with the keys 'polynomial' (polynomial coefficients) and 'determination'
        (The coefficient of determination).
    """
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results


class Plotting:
    """
    An object used to generate plots of the aggragated coverage, differences between corrected and uncorrected
    coverage, as well as correlations between bias-factors and coverage.

    :param samplename: Name of the sample as it should appear in the plots
    :param out_dir: Path to the directory to which plots should be saved.
    """

    def __init__(self, samplename: str, out_dir: str) -> None:

        self.samplename=samplename
        self.out_dir=out_dir


    def plot_coverage_bias_correlation(self, df: pd.DataFrame, biasfactor_column: str, coverage_column: str,
                                       percent: bool = True, xmin: typing.Union[int,float] = 0,
                                       xmax: typing.Union[int,float] = 100,
                                       ymin: typing.Union[int,float] = 0,
                                       ymax: typing.Union[int,float] = 2,
                                       filename: typing.Union[str, None] = "auto",
                                       return_figure: bool = False) -> typing.Union[None,
                                       matplotlib.figure.Figure]:
        """
        Plot the correlation between coverage and a bias factor as a heatmap, and fit a trendline.

        :param df: `pandas.DataFrame` that contains **biasfactor_column** and  **coverage_column**.
        :param biasfactor_column: Name of the column that contains data for the bias-factor of interest
        :param coverage_column: Name of the column that contains the coverage data
        :param percent: Whether the output plot should multiply the x values by the factor 100
        :param xmin: min x coordinate to show in plot
        :param xmax: max y coordinate to show in plot
        :param ymin: min x coordinate to show in plot
        :param ymax: max y coordinate to show in plot
        :param filename: Filename to which the figure should be saved. "auto" sets this to
            " :attr:`.out_dir`/**biasfactor_column**__vs__**coverage_column**.pdf " (with spaces in column names
            replaced by '_'). Set to `None` to avoid saving as
            file. File extension determines in which format the plot is saved.
        :param return_figure: If True, return a `matplotlib figure` that can be altered, plotted, or saved.
        :return: A `matplotlib figure` if **return_figure** is True, nothing otherwise.
        """

        x=biasfactor_column
        y=coverage_column

        heatmap, xedges, yedges = np.histogram2d(df[x]*100 if percent else df[x],df[y], bins=(100,100),
                                                 range=[[xmin,xmax],[ymin,ymax]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')
        plt.ylabel(y)
        plt.xlabel(x)
        cbar=plt.colorbar()
        cbar.set_label('# of bins', rotation=270,labelpad=13)

        y_lim=plt.gca().get_ylim()

        plt.suptitle("Bias plot for %s"%(self.samplename))
        z = np.polyfit(df[x]*100, df[y], 2)
        f = np.poly1d(z)
        x_new=np.linspace(xmin,xmax,50)
        y_new = f(x_new)
        plt.plot(x_new, y_new, color="firebrick", linewidth=0.5, )
        plt.ylim(y_lim)

        r2=round(polyfit(df[x],df[y],2)["determination"],3)
        plt.title(r'$R^{2}$ = %s'%(r2,))
        plt.locator_params(nbins=2)
        if filename=="auto":
            plt.savefig(f"{self.out_dir}/{x.replace(' ','_')}__vs__{y.replace(' ','_')}.pdf")
        elif filename is not None:
            plt.savefig(filename)

        if return_figure:
            return plt.gcf()
        plt.close()

    none_or_list_of_float=typing.Union[None,typing.List[float]]
    none_or_list_of_int=typing.Union[None,typing.List[float]]
    none_or_list_of_str=typing.Union[None,typing.List[str]]


    def plot_correlations_biasfactors_coverage_bin_nr(self, df: pd.DataFrame,
                                                      x_varnames_list: none_or_list_of_str = None,
                                                      y_varnames_list: none_or_list_of_str = None,
                                                      bins: typing.Tuple[int,int]=(50,50),
                                                      return_figure: bool = False,
                                                      filename: str="bias_plots.pdf",
                                                      percentile_cutoffs: typing.Tuple[
                                                          int, int, int, int] = (0.1,0.99,0.1,0.99),
                                                      fit_regression: bool=True) -> typing.Union[None,
    matplotlib.figure.Figure]:
        """
        Plot the correlation between coverage and a bias factor as a heatmap, and fit a trendline.

        :param df: `pandas.DataFrame` that contains the variables in **x_varnames_list** and  **y_varnames_list**.
        :param x_varnames_list: List of the columns that should be plotted as independent variables. Default None uses
            all columns in the **df** except for ["chromosome","start","end","sequence","mappability"].
        :param y_varnames_list: List of the columns that should be plotted as dependent variables. Default None uses
            "coverage" and "bin nr.", plus "corrected coverage" if this column is present in the **df**.
        :param bins: Tuple specifying the number of bins to be used for the x and y axis
        :param return_figure: If True, return a `matplotlib figure` that can be altered, plotted, or saved.
        :param fit_regression: If True, fit a qadratic polynomial and plot the regression function and R^2.
        :param percentile_cutoffs: Tuple containing the min and max percentile of values that should be displayed on the
            x and y axes: (x_min,x_max,y_min,y_max). This setting is ignored for "bin nr.", for this column, all values
            are shown. Percentiles must be in the interval [0,1].
        :param filename: Filename to which the figure should be saved. File extension determines in which format the
            plot is saved.
        :return: A `matplotlib figure` if **return_figure** is True, nothing otherwise.
        """

        if y_varnames_list is None:
            y_varnames_list=(["corrected coverage"] if "corrected coverage" in df.columns else []) + \
                    ["coverage","bin nr."]
        if x_varnames_list is None:
            x_varnames_list=[col for col in df.columns if not col in
                                                              ["chromosome","start","end","sequence","mappability"]]

        for rownr,y in enumerate(y_varnames_list):
            for colnr,x in enumerate(x_varnames_list):
                plt.subplot(len(y_varnames_list), len(x_varnames_list), rownr * len(x_varnames_list) + colnr + 1)

                ymin=df[y].quantile(percentile_cutoffs[0]) if not y=="bin nr." else df[y].min()
                ymax=df[y].quantile(percentile_cutoffs[1]) if not y=="bin nr." else df[y].max()
                xmin=df[x].quantile(percentile_cutoffs[2]) if not x=="bin nr." else df[y].min()
                xmax=df[x].quantile(percentile_cutoffs[3]) if not x=="bin nr." else df[y].min()
                nr_of_bins=len(set(df["bin nr."].values))

                heatmap, xedges, yedges = np.histogram2d(df[x],df[y],
                                                         bins=(bins[0] if x!="bin nr." else list(range(0,nr_of_bins+1)),
                                                               bins[1] if y!="bin nr." else list(range(0,nr_of_bins+1))),
                                                         range=[[xmin,xmax],[ymin,ymax]])
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')

                plt.ylabel(y)
                plt.xlabel(x)
                cbar=plt.colorbar()
                cbar.set_label('# of bins', rotation=270,labelpad=13)
                plt.suptitle("Bias plot for %s" % self.samplename)
                y_lim=plt.gca().get_ylim()
                x_lim=plt.gca().get_xlim()

                if fit_regression:
                    df_tmp=df.copy().dropna()
                    z = np.polyfit(df_tmp[x], df_tmp[y], 2)
                    f = np.poly1d(z)
                    x_new=np.linspace(xmin,xmax,50)
                    y_new = f(x_new)
                    plt.plot(x_new, y_new, color="firebrick", linewidth=0.5, )
                    r2=round(polyfit(df_tmp[x],df_tmp[y],2)["determination"],3)
                    plt.title(r'$R^{2}$ = %s'%(r2,))

                # Show the mean as a seperate line
                if y=="bin nr." and not y==x:
                    plt.plot(df.groupby("bin nr.")[x].mean(),range(0,nr_of_bins),color="pink",linewidth=0.5)
                elif x=="bin nr." and not y==x:
                    plt.plot(range(0,nr_of_bins),df.groupby("bin nr.")[y].mean().values,color="pink",linewidth=0.5)
                else:
                    df_tmp=df.copy()
                    df_tmp["tmp bin"],borders=pd.cut(df_tmp[x],50,retbins=True) #create 50 bins based on the x variable
                    midpoints=[(borders[i]+borders[i+1])/2 for i in range(len(borders)-1)]
                    # plot the mean of the y variable for each bin
                    plt.plot(midpoints,df_tmp.groupby("tmp bin")[y].mean(),color="pink",linewidth=0.5)

                plt.ylim(y_lim)
                plt.xlim(x_lim)
                plt.locator_params(nbins=2)
                if colnr:
                    plt.ylabel("")

        plt.gcf().set_size_inches(len(x_varnames_list) * 3, len(y_varnames_list) * 3)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        if filename=="auto":
            plt.savefig(f"{self.out_dir}/{filename}")
        elif filename is not None:
            plt.savefig(filename)
        if return_figure:
            return plt.gcf()
        plt.close()

    def plot_fitted_model(self, fit_gaussians_obj: typing.Union[FitGaussians.FitGaussians,None] = None,
                          x_values: none_or_list_of_int = None, unfitted_y_values: none_or_list_of_float = None,
                          g1_y_values: none_or_list_of_float = None,
                          g2_y_values: none_or_list_of_float = None, g3_y_values: none_or_list_of_float = None,
                          intercept_y_values: none_or_list_of_float = None,
                          combined_model_y_values: none_or_list_of_float = None,
                          unfitted_y_values_label: none_or_list_of_float = "Corrected, aggregated coverage",
                          filename: typing.Union[str,None] = "auto",
                          return_figure: bool = False) -> typing.Union[None,
                          matplotlib.figure.Figure]:
        """
        Plots the fitted model, i.e. the raw data and lines for G1, G2, G3, the intercept, and the combined model.

        :param fit_gaussians_obj: object of class FitGaussians, on which fit_gaussian_models() has been called. This
            contains all nessesary information for plotting, so all other parameters except **filename** are ignored if
            this is set.
        :param x_values: List of x coordinates - the coordinates of the bins' centers relative to the center of the
            region of interest. Required if fit_gaussians_obj is not set, ignored otherwise.
        :param unfitted_y_values: List of y coordinates of the unfitted coverage data.
            Required if fit_gaussians_obj is not set, ignored otherwise.
        :param g1_y_values: List of y coordinates for the first, smallest gaussian.
            Required if fit_gaussians_obj is not set, ignored otherwise.
        :param g2_y_values: List of y coordinates for the second, middle gaussian.
            Required if fit_gaussians_obj is not set, ignored otherwise.
        :param g3_y_values: List of y coordinates for the first, largest gaussian.
            Required if fit_gaussians_obj is not set, ignored otherwise.
        :param intercept_y_values: List of y coordinates for the intercept.
          Required if fit_gaussians_obj is not set, ignored otherwise.
        :param combined_model_y_values: List of y coordinates for the fitted, combined model.
            Required if fit_gaussians_obj is not set, ignored otherwise.
        :param unfitted_y_values_label: Label for the unfitted_y_values. By default it is assumed that the corrected,
            aggregated, unfitted coverage is given.
        :param filename: Filename to which the figure should be saved. "auto" sets this to
            " :attr:`.out_dir`/fitted_gaussians.pdf ". Set to `None` to avoid saving as
            file. File extension determines in which format the plot is saved.
        :param return_figure: If True, return a `matplotlib figure` that can be altered, plotted, or saved.
        :return: A `matplotlib figure` if **return_figure** is True, nothing otherwise.
        """

        if fit_gaussians_obj is not None:
            x_values=fit_gaussians_obj.x_values
            unfitted_y_values=fit_gaussians_obj.unfitted_y_values
            g1_y_values=fit_gaussians_obj.g1_y_values
            g2_y_values=fit_gaussians_obj.g2_y_values
            g3_y_values=fit_gaussians_obj.g3_y_values
            intercept_y_values=fit_gaussians_obj.intercept_y_values
            combined_model_y_values=fit_gaussians_obj.combined_model_y_values

        if fit_gaussians_obj is None and any([param is None for param in [x_values,unfitted_y_values,g1_y_values,
            g2_y_values,g3_y_values,intercept_y_values,combined_model_y_values]]):
            raise TypeError("If fit_gaussians_obj is not specified (or set to None), than the parameters [x_values,"
                            "unfitted_y_values,g1_y_values, g2_y_values,g3_y_values,intercept_y_values,"
                            "combined_model_y_values] of this function must all be specified and be non-None values.")

        plt.plot(x_values,combined_model_y_values, label="Combined model", linewidth=1.3)
        plt.plot(x_values, g1_y_values,":",color="seagreen", label="First gaussian", alpha=0.5)
        plt.plot(x_values, g2_y_values,"-.",color="purple", label="Second gaussian",alpha=0.5)
        plt.plot(x_values, g3_y_values,"--",color="orange", label="Third gaussian",alpha=0.5)
        plt.plot(x_values, intercept_y_values ,"--",color="black", label="Intercept",linewidth=0.5)
        plt.plot(x_values,unfitted_y_values, "kx", alpha=0.35, label=unfitted_y_values_label)
        plt.title(f"Fitting of 3 Gaussian Dips and intercept to "
                  f"{unfitted_y_values_label[0].lower() + unfitted_y_values_label[1:]} for sample {self.samplename}")
        plt.xlabel("Distance from center of RoI [bp]")
        plt.ylabel(unfitted_y_values_label)
        plt.legend()
        fig=plt.gcf()
        fig.set_size_inches(10,5)
        if filename=="auto":
            fig.savefig(f"{self.out_dir}/fitted_gaussians.pdf")
        elif filename is not None:
            fig.savefig(filename)
        if return_figure:
            return fig
        plt.close()

    def plot_corrected_vs_uncorrected_coverage(self, x_values: typing.List[int],
                                               uncorrected_y_values: typing.List[float],
                                               corrected_y_values: typing.List[float],
                                               uncorrected_y_intercept: float, corrected_y_intercept: float,
                                               filename: typing.Union[str,None] ="auto",
                                               return_figure: bool = False) -> \
            typing.Union[None, matplotlib.figure.Figure]:
        """
        Plots the aggregated coverages, both corrected and uncorrected.

        :param x_values: List of x coordinates - the coordinates of the bins' centers relative to the center of the
            regions of interest.
        :param uncorrected_y_values: List of aggregated, uncorrected coverage values.
        :param corrected_y_values: List of aggregated, uncorrected coverage values.
        :param uncorrected_y_intercept: Intercept value that should be subtracted from every value in
            **uncorrected_y_values**
        :param corrected_y_intercept: Intercept value that should be subtracted from every value in
            **corrected_y_values**
        :param filename: Filename to which the figure should be saved. "auto" sets this to
            " :attr:`.out_dir`/corrected_vs_uncorrected_coverage.pdf ". Set to `None` to avoid saving as
            file. File extension determines in which format the plot is saved.
        :param return_figure: If True, return a `matplotlib figure` that can be altered, plotted, or saved.
        :return: A `matplotlib figure` if **return_figure** is True, nothing otherwise.

        Example:

        .. code-block:: python

            plotting.plot_corrected_vs_uncorrected_coverage(
                x_values=fit_gaussians_obj.x_values,
                uncorrected_y_values=fit_gaussians_obj_uncorrected.unfitted_y_values,
                corrected_y_values=fit_gaussians_obj.unfitted_y_values,
                uncorrected_y_intercept=fit_gaussians_obj_uncorrected.intercept_y_values[0],
                corrected_y_intercept=fit_gaussians_obj.intercept_y_values[0])
        """

        uncorrected_y_values=[y-uncorrected_y_intercept for y in uncorrected_y_values]
        corrected_y_values=[y-corrected_y_intercept for y in corrected_y_values]


        plt.plot(x_values,uncorrected_y_values, color="firebrick",label="Uncorrected, aggregated coverage",
                 linewidth=0.5,alpha=0.5,marker="x")
        plt.plot(x_values, corrected_y_values,color="navy", label="Corrected, aggregated coverage", alpha=0.5,
                 marker="x")

        plt.xlabel("Distance from center of RoI [bp]")
        plt.ylabel("Coverage, normalized by intercept")
        plt.title(f"Corrected vs uncorrected coverage for sample {self.samplename}")
        plt.legend(fontsize=8)

        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)

        if filename=="auto":
            plt.gcf().savefig(f"{self.out_dir}/corrected_vs_uncorrected_coverage.pdf")
        elif filename is not None:
            plt.gcf().savefig(f"{self.out_dir}/{filename}")
        if return_figure:
            return plt.gcf()
        plt.close()