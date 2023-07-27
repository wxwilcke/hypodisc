#! /usr/bin/env python

from datetime import datetime, timedelta

from rdf.terms import IRIRef, Literal
from rdf.namespaces import XSD


EPOCH_TIME = datetime(year = 1970, month = 1, day = 1, hour = 1)
SECONDS_PER_MIN = 60
SECONDS_PER_HOUR = SECONDS_PER_MIN * 60
DAYS_PER_YEAR = 365

def cast_datetime_rev(dtype:IRIRef, timestamp:float) -> str:
    """ Cast date and similar to iso format

    :param dtype:
    :type dtype: IRIRef
    :param timestamp:
    :type timestamp: float
    :rtype: str
    """
    date = datetime.fromtimestamp(timestamp)
    if dtype == XSD + 'gYear':
        date_str = "%Y"
    elif dtype == XSD + 'gYearMonth':
        date_str = "%Y-%m"
    elif dtype == XSD + 'gDate':
        date_str = "%Y-%m-%d"
    elif dtype == XSD + 'gDateTime':
        date_str = "%Y-%m-%dT%H:%M:%S"
    else:  # gDateTimeStamp
        date_str = "%Y-%m-%dT%H:%M:%SZ%Z"

    return date.strftime(date_str)

def cast_datetime_delta(timestamp:float) -> str:
    """ Cast datetime delta to duration

    :param timestamp:
    :type timestamp: float
    :rtype: str
    """
    delta = datetime.fromtimestamp(timestamp) - EPOCH_TIME

    y, d = divmod(delta.days, DAYS_PER_YEAR)
    h, rest = divmod(delta.seconds, SECONDS_PER_HOUR)
    m, s = divmod(rest, SECONDS_PER_MIN)

    date_str = 'P'
    if y > 0:
        date_str += f'{y}Y'
    if d > 0:
        date_str += f'{d}D'
    if h > 0 or m > 0 or s > 0:
        date_str += 'T'
    if h > 0:
        date_str += f'{h:02d}H'
    if m > 0:
        date_str += f'{m:02d}M'
    if s > 0:
        date_str += f'{s:02d}S'

    return date_str

def cast_datefrag_rev(dtype:IRIRef, days:float) -> str:
    """ Cast days to months and days

    :param dtype:
    :type dtype: IRIRef
    :param days:
    :type days: float
    :rtype: str
    """
    delta = timedelta(days = days)
    date = EPOCH_TIME + delta

    if dtype == XSD + 'gMonthDay':
        return f"{date.month:02d}-{date.day:02d}"
    elif dtype == XSD + 'gMonth':
        return f"{date.month:02d}"
    else:  # gDay
        return f"{date.day:02d}"

def cast_datefrag_delta(days:float) -> str:
    """ Cast datefrag delta to dayTimeDuration

    :param days:
    :type days: float
    :rtype: str
    """
    delta = timedelta(days = days)
    d = delta.days
    h, rest = divmod(delta.seconds, SECONDS_PER_HOUR)
    m, s = divmod(rest, SECONDS_PER_MIN)

    date_str = f'P{d}D'
    if h > 0 or m > 0 or s > 0:
        date_str += 'T'
    if h > 0:
        date_str += f'{h:02d}H'
    if m > 0:
        date_str += f'{m:02d}M'
    if s > 0:
        date_str += f'{s:02d}S'

    return date_str

def cast_datetime(dtype:IRIRef, v:Literal) -> float:
    """ Cast dates to UNIX timestamps

    :param dtype:
    :type dtype: IRIRef
    :param v:
    :type v: Literal
    :rtype: float
    """
    if dtype == XSD + 'gYear':
        y = int(v.value)
        v = datetime(year = y, month = 1, day = 1, hour = 1)
    elif dtype == XSD + 'gYearMonth':
        y, m = v.value.split('-')
        y, m = int(y), int(m)
        v = datetime(year = y, month = m, day = 1, hour = 1)
    else:  # date, dateTime, dateTimeStamp
        v = datetime.fromisoformat(v.value)

    return v.timestamp()

def cast_datefrag(dtype:IRIRef, v:Literal) -> float:
    """ Cast date fragments to days

    :param dtype:
    :type dtype: IRIRef
    :param v:
    :type v: Literal
    :rtype: float
    """
    # use EPOCH YEAR if non given
    if dtype == XSD + 'gMonth':
        m = int(v.value)
        v = datetime(year = 1970, month = m, day = 1, hour = 1)

        days = (v - EPOCH_TIME).days
    elif dtype == XSD + 'gMonthDay':
        m, d = v.value.split('-')
        m, d = int(m), int(d)
        v = datetime(year = 1970, month = m, day = d, hour = 1)

        days = (v - EPOCH_TIME).days
    else:  # gDay
        days = int(v.value)

    return float(days)


