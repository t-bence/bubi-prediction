from datetime import datetime, timedelta


def get_train_cutoff(date_string: str) -> datetime:
    """Get train cutoff: if date_string is a valid date representation
        cast that into datetime. If it is not, return the date 2 weeks ago.

    Parameters
    ----------
    date_string : str
        A string representation of a datetime, or empty string

    Returns
    -------
    datetime
        The training cutoff, where training data stops and test data starts.
    """
    try:
        train_cutoff = datetime.fromisoformat(date_string)
    except ValueError:
        train_cutoff = datetime.now() - timedelta(days=14)

    return train_cutoff
