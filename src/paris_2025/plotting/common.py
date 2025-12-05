import datetime
import sys


def get_metadata(description=None):
    """Get metadata for the plots."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    caller_name = sys._getframe().f_back.f_code.co_name  # type: ignore
    _description = f"Created by function '{caller_name}' on {date_str}."
    if description is not None:
        _description += f"\n{description}"
    return {"Description": _description}
