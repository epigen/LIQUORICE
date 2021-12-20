from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from joblib import dump,load,parallel_backend
import pandas as pd
import logging
import typing
from typing import List,Union
import typing_extensions
import numpy as np
import sys

class SklearnStyleRegressor(typing_extensions.Protocol):
    """
    Introduced here for typing purposes only.
    """
    def fit(self,X,y,sample_weight=None): ...
    def predict(self, X): ...
    def score(self,X, y): ...
    def set_params(self, X,y): ...


class BiasModel:
    """
    An object that can be used to train a machine learning model that predicts coverage based on bias
    factors. The performance of the trained model can be tested, and it can be used to correct coverage values
    by regressing out the influence of the bias factors.

    :param training_df: `pandas.DataFrame` used for training (and optionally for performance assessment) of the
        model. Must contain the column *coverage* and all columns specified under **features**. Can be None if
        :func:`train_biasmodel` won't be called. Ignored in :func`train_biasmodel_2fold_CV_and_predict`.
    :param df_to_correct: `pandas.DataFrame` for which coverage should be corrected by the trained model. Must
        contain the column *coverage* and all columns specified under **features**. Can be None if
        :func:`get_table_with_corrected_coverage_using_trained_biasmodel` won't be called.
    :param biasmodel_path: Path to which the trained biasmodel should be saved to and/or loaded from. Must be a
        .joblib file. Can be None if :func:`get_table_with_corrected_coverage_using_trained_biasmodel` won't be
        called, in that case, no biasmodel will be saved.
    :param features: A list of bias factors that should be used as features for the machine learning model. Default
        "all" sets all bias-factors as features: forward,reverse, and max mappability, di/trinucleotide factors and
        GC-content. Bin size is not included by default, it can be added as a feature via `use_binsize_as_feature`.
    :param nr_of_bins_for_training_and_testing: Subset the training_df to this many bins. Can speed up the
        computation time, but using too few bins will make the model less precise. Can be None, then all bins will be
        used.
    :param sklearn_model: A regressor that implements to functions .fit() and .predict() (e.g. from `sklearn`).
        Default of None means using sklearn.ensemble.HistGradientBoostingRegressor with default settings.
    :param n_jobs: How many jobs to run in parallel when training the model
    :param filename_performance_metrics: If **test_fraction** or **cross_validate_k** is set, save a .csv containing
        the performance metrics (r2, MSE) to this path.
    :param filename_feature_importances: If set, save a .csv file containing the feature importances inferred from
        the trained model to this path.
    :param use_binsize_as_feature: If True, include "bin size" as a feature for the model.
    """
    # :param test_fraction: Use this fraction of bins in training_df to test model performance. This fraction of
    #     bins will not be used to train the model and are only used for testing. Set to None to disable performance
    #     metrics for training_df (unless **cross_validate_k** is set).
    # :param cross_validate_k: Perform a k-fold cross-validation instead of a simple train/test split to evaluate
    #     model performance. The saved model will be trained on the full dataset.

    def __init__(self, training_df: Union[pd.DataFrame,None], df_to_correct: Union[pd.DataFrame,None],
                 biasmodel_path: str = "trained_biasmodel.joblib", features: typing.Union[str, List[str]] = "all",
                 nr_of_bins_for_training_and_testing: Union[int,None] = 10000,
                 sklearn_model: Union[None,SklearnStyleRegressor] = None,
                 #test_fraction: Union[None, float] = None, cross_validate_k: Union[None, int] = None,
                 n_jobs: int = 1,
                 filename_performance_metrics: typing.Union[None,str] = "biasmodel_performance_metrics.csv",
                 filename_feature_importances: typing.Union[None,str] = None, #"biasmodel_feature_importances.csv"
                 use_binsize_as_feature: bool = False) -> None:


        if training_df is None and df_to_correct is None:
            raise TypeError("training_df and df_to_correct cannot both be None.")
        self.training_df=training_df
        if training_df is not None:
            nr_rows_with_na=self.training_df.shape[0]
            self.training_df=self.training_df.dropna()
            if self.training_df.shape[0]<nr_rows_with_na:
                logging.warning(f"Training dataset contained {nr_rows_with_na-self.training_df.shape[0]} rows with"
                                f" empty values, these were discarded.")
        self.df_to_correct=df_to_correct
        if df_to_correct is not None:
            nr_rows_with_na=self.df_to_correct.shape[0]
            self.df_to_correct=self.df_to_correct.dropna()
            if self.df_to_correct.shape[0]<nr_rows_with_na:
                logging.warning(f"Dataset that should be corrected contained "
                                f"{nr_rows_with_na-self.df_to_correct.shape[0]}"
                                f" rows with empty values, these were discarded.")

        self.biasmodel_path=biasmodel_path
        if self.biasmodel_path is not None and not self.biasmodel_path.endswith(".joblib"):
            raise TypeError("biasmodel_path must have a .joblib extension.")

        self.features=features
        self.use_binsize_as_feature=use_binsize_as_feature
        if type(self.features)==str and self.features=="all":
            df_for_feature_extraction=training_df if training_df is not None else df_to_correct
            drop_from_features=["chromosome","start","end","bin nr.","sequence","mappability","coverage",
                                   "CNA-uncorrected coverage","corrected coverage"]+\
                               (["bin size"] if not self.use_binsize_as_feature else [])
            self.features=[x for x in df_for_feature_extraction.columns if not x in drop_from_features]
        elif not type(self.features)==list:
            raise TypeError("Please specify either the string 'all' or a list of features as parameter 'features'.")

        self.nr_of_bins_for_training_and_testing=nr_of_bins_for_training_and_testing
        # self.test_fraction=test_fraction
        # self.cross_validate_k=cross_validate_k
        self.n_jobs=n_jobs
        self.filename_performance_metrics=filename_performance_metrics
        self.filename_feature_importances=filename_feature_importances

        if sklearn_model is None:

            self.sklearn_model=HistGradientBoostingRegressor(
                random_state=42,learning_rate= 0.1,max_iter=54, min_samples_leaf=10,
                max_depth=10,early_stopping=False)

        else:
            self.sklearn_model=sklearn_model

        self.y = "coverage"


    def train_biasmodel(self) -> None:
        """
        Train a machine learning model that predicts the values of the 'coverage' column based on the given features. If
        :attr:`.test_fraction` is set, this fraction of the :attr:`.training_df` is set aside to evaluate R^2 and MSE
        of the model. Prior to training and a potential train/test split, the :attr:`.training_df` is subsetted to
        :attr:`nr_of_bins_for_training_and_testing` if it is not *None*. Writes writes a *.joblib*
        file containing the biasmodel to :attr:`.biasmodel_path` (unless it is `None`)
        """
        logging.info(f"Training a bias model ...")
        logging.info(f"Using features: {self.features}")

        df=self.training_df[[self.y]+self.features]

        if self.nr_of_bins_for_training_and_testing is None:
            self.nr_of_bins_for_training_and_testing=df.shape[0]

        df=df.sample(n=self.nr_of_bins_for_training_and_testing,random_state=42)

        # if self.test_fraction:
        #     test_df=df.iloc[:int(df.shape[0]*self.test_fraction)]
        #     training_df=df.iloc[int(df.shape[0]*self.test_fraction):]
        # else:
        #     training_df=df
        training_df=df

        with parallel_backend('threading', n_jobs=min([self.n_jobs,10])):
            self.sklearn_model.fit(training_df[self.features],training_df[self.y])

        if self.biasmodel_path is not None:
            dump(self.sklearn_model, self.biasmodel_path)

        # if self.cross_validate_k:
        #     scores = cross_validate(estimator=self.sklearn_model,
        #                             X=training_df[self.features],
        #                             y=training_df[self.y],
        #                             scoring=["r2","neg_mean_squared_error"],
        #                             cv=self.cross_validate_k,
        #                             n_jobs=self.n_jobs)
        #     if self.filename_performance_metrics:
        #         score_df=pd.DataFrame({"r2":scores["test_r2"],"MSE":-scores["test_neg_mean_squared_error"]})[[
        #             "r2","MSE"]].transpose()
        #         score_df["mean"]=score_df.mean(axis=1)
        #         score_df["median"]=score_df.median(axis=1)
        #         score_df["std"]=score_df.std(axis=1)
        #         score_df["n bins training"]=int(training_df.shape[0]*(1-1/self.cross_validate_k))
        #         score_df.columns=["fold "+str(k) for k in range(self.cross_validate_k)]+\
        #                          list(score_df.columns[self.cross_validate_k:])
        #         score_df.to_csv(self.filename_performance_metrics)
        #
        # if self.test_fraction:
        #     y_pred=self.sklearn_model.predict(test_df[self.features])
        #     y_true=test_df[self.y]
        #
        #     r2=r2_score(y_true,y_pred)
        #     logging.info(f"R^2 for test dataset: {r2}")
        #     mse=mean_squared_error(y_true,y_pred)
        #     logging.info(f"MSE for test dataset: {mse}")
        #     if self.filename_performance_metrics:
        #         with open(self.filename_performance_metrics, "w") as outfile:
        #             print(f"r2,MSE\n{r2},{mse}",end="",file=outfile)

        if self.filename_feature_importances:
            try:
                feature_importances=pd.Series({key:value for key,value in zip(
                    self.features,self.sklearn_model.feature_importances_)}).sort_values(ascending=False)
                feature_importances.name="Feature importance"
                feature_importances.to_csv(self.filename_feature_importances)
            except AttributeError:
                try: # input may be a pipeline - try this
                    feature_importances=pd.Series({key:value for key,value in zip(
                        self.features,self.sklearn_model[-1][1].feature_importances_)}).sort_values(ascending=False)
                    feature_importances.name="Feature importance"
                    feature_importances.to_csv(self.filename_feature_importances)
                except:
                    logging.warning("Feature importance file could not be written. Maybe the passed model has no "
                                    "attribute '.feature_importances_'?")

    def get_table_with_corrected_coverage_using_trained_biasmodel(self) -> pd.DataFrame:
        """
        Predicts coverage based on :attr:`.features` in the DataFrame :attr:`df_to_correct` and the biasmodel
        under :attr:`.biasmodel_path`. Subtracts this prediction from the observed coverage to regress out the effect
        of the bias-factors (i.e. :attr:`.features`) on the coverage.
        
        :return: Returns :attr:`df_to_correct` with an additional column "corrected coverage".
        """
        logging.info(f"Correcting using trained bias model ...")

        trained_model = load(self.biasmodel_path)

        if trained_model._n_features==len(self.features):
            df=self.df_to_correct[self.features]
        else:
            features_without_binsize=[feature for feature in self.features if not feature == "bin size"]
            if trained_model._n_features==len(features_without_binsize):
                df=self.df_to_correct[features_without_binsize]
                logging.info("Bin size was omitted as a feature for coverage prediction, as the provided bias-model has"
                             " been trained without this feature.")
            else:
                sys.exit("Features do not agree between the provided pre-trained bias model and the current run. "
                              "Please use a bias-model that has been trained with comparable settings. "
                              "Maybe the bias-model has been generated without the --all_bins_same_size setting, "
                              "and the current run of LIQUORICE was started with the --all_bins_same_size setting?")
        y_pred = trained_model.predict(df)
        y_corr = self.df_to_correct[self.y] - y_pred

        self.df_to_correct["corrected coverage"]=y_corr

        r2=r2_score(self.df_to_correct[self.y],y_pred)
        logging.info(f"R^2 for dataset to be corrected: {r2}")

        mse=mean_squared_error(self.df_to_correct[self.y],y_pred)
        logging.info(f"MSE for dataset to be corrected: {mse}")

        if self.filename_performance_metrics:
            with open(self.filename_performance_metrics, "w") as outf:
                print("R^2\tMSE",file=outf)
                print(f"{r2}\t{mse}",file=outf)

        return self.df_to_correct


    def train_biasmodel_2fold_CV_and_predict(self,exclude_these_bin_nrs:List[int] =[]) -> None:
        """
        Train a machine learning model that predicts the values of the 'coverage' column based on the given features.
        Will use each half of the :attr:`df_to_correct` to train the model for predictions of the other half.
        Ignores :attr:`nr_of_bins_for_training_and_testing`, :attr:`cross_validate_k` and writes returns the performance
         metrics and returns the dataframe with predictions.
         Important: does not use :attr`training_df`.

        """
        logging.info(f"Using CVs to train biasmodel and predict...")
        #logging.info(f"Excluding from training the following bins: {exclude_these_bin_nrs}")
        logging.info(f"Using features: {self.features}")

        df_first_half=self.df_to_correct.iloc[:int(self.df_to_correct.shape[0]/2)]
        train_df_first_half=df_first_half[~ df_first_half["bin nr."].isin(exclude_these_bin_nrs)]
        df_second_half=self.df_to_correct.iloc[int(self.df_to_correct.shape[0]/2):]
        train_df_second_half=df_second_half[~ df_second_half["bin nr."].isin(exclude_these_bin_nrs)]

        with parallel_backend('threading', n_jobs=min([10,self.n_jobs])):
            self.sklearn_model.fit(train_df_second_half[self.features], train_df_second_half[self.y])
        y_pred=self.sklearn_model.predict(df_first_half[self.features])
        logging.info(f"Training-set R^2 for the first cross validation fold: {r2_score(df_second_half[self.y],self.sklearn_model.predict(df_second_half[self.features]))}")
        logging.info(f"Out-of-sample R^2 for the first cross validation fold: {r2_score(df_first_half[self.y],y_pred)}")

        with parallel_backend('threading', n_jobs=min([10,self.n_jobs])):
            self.sklearn_model.fit(train_df_first_half[self.features], train_df_first_half[self.y])
        y_pred_second_half=self.sklearn_model.predict(df_second_half[self.features])
        logging.info(f"Training-set R^2 for the second cross validation fold: {r2_score(df_first_half[self.y],self.sklearn_model.predict(df_first_half[self.features]))}")
        logging.info(f"Out-of-sample R^2 for the second cross validation fold: {r2_score(df_second_half[self.y],y_pred_second_half)}")

        y_pred=np.append(y_pred,y_pred_second_half)

        y_corr = self.df_to_correct[self.y] - y_pred
        # use .assign to avoid settingWithCopyWarning
        self.df_to_correct=self.df_to_correct.assign(**{"corrected coverage":y_corr.values})

        r2=r2_score(self.df_to_correct[self.y],y_pred)
        logging.info(f"Cross-validated R^2 overall: {r2}")

        mse=mean_squared_error(self.df_to_correct[self.y],y_pred)
        logging.info(f"Cross-validated MSE overall: {mse}")

        if self.filename_performance_metrics:
            with open(self.filename_performance_metrics, "w") as outf:
                print("R^2\tMSE",file=outf)
                print(f"{r2}\t{mse}",file=outf)

        return self.df_to_correct

