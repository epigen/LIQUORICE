import numpy as np
from lmfit.models import GaussianModel, ConstantModel
import pandas as pd
import logging
import typing


class FitGaussians:
    """
    An object that can be used for quantification of coverage drops by fitting gaussian functions and an intercept
    to the coverage values.

    :param unfitted_y_values: A pandas `Series` of coverage values, one value per bin (and sorted by increasing bin
        nr.). A suitable input is the result of
        :func:`liquorice.utils.AggregateAcrossrRegions.aggregate_across_regions()`.
    :param extend_to: `extend_to` setting that was used when calculating the y values. Needed to calculate relative
        bin positions.
    :param binsize: `binsize` setting that was used when calculating the y values. Needed to calculate relative
        bin positions.
    :param samplename: If not `None`, add this in a column "sample" to the result `DataFrame` when calling
        :func:`.fit_gaussian_models`
    :param regionset_name: If not `None`, add this in a column "region-set" to the result `DataFrame` when calling
        :func:`.fit_gaussian_models`
    :param avg_centersize: Average size (i.e. width in bp) of the regions of interest (before extension by extend_to).
    """

    def __init__(self, unfitted_y_values: typing.List[float], extend_to: int, binsize: int,
                 avg_centersize: float, samplename: str = None,
                 regionset_name: str = None,) -> None:

        self.extend_to=extend_to
        self.binsize=binsize
        self.unfitted_y_values=unfitted_y_values

        self.avg_centersize=int(avg_centersize)

        self.samplename=samplename
        self.regionset_name=regionset_name

        self.x_values=None
        self.intercept_y_values=None
        self.g1_y_values=None
        self.g2_y_values=None
        self.g3_y_values=None
        self.combined_model_y_values=None

    def fit_gaussian_models(self, g1_min_sigma: typing.Union[int,float] = 20,
                            g1_max_sigma: typing.Union[int,float] = 200, g2_min_sigma: typing.Union[int,float] = 200,
                            g2_max_sigma: typing.Union[int,float] = 3000, g3_min_sigma: typing.Union[int,float] = 3000,
                            g3_max_sigma: typing.Union[int,float] = 40000, method: str = "leastsq") -> pd.DataFrame:
        """
        Fits three gaussian functions to :attr:`.unfitted_y_values` (usually aggregated coverage values), based on given
        limits. Sets :attr:`x_values`, :attr:`g1_y_values`, :attr:`g2_y_values`, :attr:`g3_y_values`,
        :attr:`intercept_y_values` and :attr:`combined_model_y_values` for the object.

        :param g1_min_sigma: Min σ value for the smallest gaussian
        :param g1_max_sigma: Max σ value for the smallest gaussian
        :param g2_min_sigma: Min σ value for the middle gaussian
        :param g2_max_sigma: Max σ value for the middle gaussian
        :param g3_min_sigma: Min σ value for the widest gaussian
        :param g3_max_sigma: Max σ value for the widest gaussian
        :param method: A method for fitting the combined model, used as parameter in lmfit's .fit() function.
        :return: A `pandas.DataFrame` containing several summary metrics based on the fitted model: The amplitude and σ
            values for each of the three gaussians, the total depth of the fitted model at the central bin, the Bayesian
            Information Criterion (measures model fit), and the area under the curve of the fitted model,
            calculated as the area between the fitted intercept and the fitted model.
            If :attr:`samplename` or :attr:`regionset_name` are not `None`, columns "sample" or "region-set" are
            added, respectively.
        """

        logging.info("Fitting gaussian functions and intercept to coverage values ...")

        y=self.unfitted_y_values

        avg_centersize=self.avg_centersize
        central_bin_x_values=[x*avg_centersize-avg_centersize/2 for x in [0.05,0.175,0.5,0.825,0.95]]
        # try if all bins have the same size:
        x=list(np.arange(-self.extend_to,self.extend_to+1, self.binsize))
        # if not, assume that the central bins are smaller and have the coordinates in central_bin_x_values
        if len(x)!=len(y):
            x=list(np.arange(-self.extend_to+(self.binsize/2)-avg_centersize/2,(self.binsize/2)-avg_centersize/2,self.binsize))+central_bin_x_values+list(np.arange(avg_centersize/2+(self.binsize/2),self.extend_to+avg_centersize/2+(self.binsize/2),self.binsize))
        assert len(x)==len(y)

        comb_dipmodel=GaussianModel(prefix='G1_')+GaussianModel(prefix='G2_')+GaussianModel(prefix='G3_')+\
            ConstantModel(prefix="Const_")
        comb_dipmodel.set_param_hint('G1_center',value=0,vary=False)
        comb_dipmodel.set_param_hint('G2_center',value=0,vary=False)
        comb_dipmodel.set_param_hint('G3_center',value=0,vary=False)
        comb_dipmodel.set_param_hint('G1_sigma',value=(g1_min_sigma+g1_max_sigma)/2,min=g1_min_sigma, max=g1_max_sigma)
        comb_dipmodel.set_param_hint('G2_sigma',value=(g2_min_sigma+g2_max_sigma)/2,min=g2_min_sigma, max=g2_max_sigma)
        comb_dipmodel.set_param_hint('G3_sigma',value=(g3_min_sigma+g3_max_sigma)/2,min=g3_min_sigma, max=g3_max_sigma)
        comb_dipmodel.set_param_hint('Const_c',value=0)
        comb_dipmodel_paramas=comb_dipmodel.make_params()
        comb_dipmodel=comb_dipmodel.fit(data=y,x=x,params=comb_dipmodel_paramas, method=method)

        comps = comb_dipmodel.eval_components()

        # Sort the gaussians such that the narrowest one is the first:
        sortedgaussians=[("G1_",comb_dipmodel.best_values["G1_sigma"]),
            ("G2_",comb_dipmodel.best_values["G2_sigma"]),("G3_",comb_dipmodel.best_values["G3_sigma"])]
        sortedgaussians=sorted(sortedgaussians,key=lambda x: x[1])


        self.x_values=x
        self.g1_y_values=comps[sortedgaussians[0][0]]
        self.g2_y_values=comps[sortedgaussians[1][0]]
        self.g3_y_values=comps[sortedgaussians[2][0]]
        self.intercept_y_values=[comps["Const_"] for _ in x]
        self.combined_model_y_values=comb_dipmodel.best_fit

        sum_g_heights=comb_dipmodel.params["G1_height"].value+comb_dipmodel.params["G2_height"].value+\
            comb_dipmodel.params["G3_height"].value

        sampledict={}
        for key, value in comb_dipmodel.params.valuesdict().items():
            if "center" in key:  # center is 0 anyway
                continue
            # this is required such that the narrow gaussian is always the G1:
            if "G" in key and "amplitude" in key or "sigma" in key:
                for index,(name,sigma) in enumerate(sortedgaussians):
                    # key is, e.g. G1_height. If G2 is the second-most narrow peak, rename G1_height to G2_height
                    if name.startswith(key[:2]):
                        sampledict["G"+str(index+1)+key[2:]]=value

        sampledict["Bayesian Information Criterion"]=comb_dipmodel.bic
        sampledict["Total dip depth"]=-sum_g_heights
        sampledict["Intercept"]=comb_dipmodel.best_values["Const_c"]

        # calculate area under the curve, taking the intercept as a reference
        y_vals_for_AOC=[y-comps["Const_"] for y in comb_dipmodel.best_fit]
        sampledict["Total dip area (AOC combined model)"]=np.trapz(x=x,y=y_vals_for_AOC)

        param_df=pd.Series(sampledict).to_frame().T

        if self.samplename is not None:
            param_df["sample"]=self.samplename
        if self.regionset_name is not None:
            param_df["region-set"]=self.regionset_name
        return param_df
