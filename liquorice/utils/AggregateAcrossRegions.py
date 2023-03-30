import pandas as pd


def aggregate_across_regions(df: pd.DataFrame, column_of_interest: str) -> pd.Series:
    """
    :param df: A :class:`pandas.DataFrame`, containing the columns "bin nr." and **column_of_interest**
    :param column_of_interest: Column name for which the mean per bin nr. should be returned
    :return: A :class:`pandas.Series` with mean values of **column_of_interest** per bin, aggregated across all
        regions in the :class:`pandas.DataFrame`.

    Example:

    .. code-block:: python

        mean_corrected_coverage_leftmost_bin = aggregate_across_regions(df,"corrected coverage").loc[0]
    """
    return df.groupby("bin nr.")[column_of_interest].mean()