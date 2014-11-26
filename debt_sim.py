#!/usr/bin/env python
# encoding: utf-8

import numpy
import pandas
import datetime

def load_monthly_SnP(start_date = '01/01/1930'):
    from pandas.io.data import DataReader
    prices = DataReader('^GSPC', data_source = 'yahoo', start = start_date)

    return prices['Adj Close'].resample('BM', kind = 'timestamp')

def conc_payout(return_series, debt_amount, int_rate, debt_payment, excess_inv):
    """
    Scenario where student debt is paid concomitantly to investments.  Once 
    the student debt is paid down, the remainder goes towards investment

    INPUTS:
    -------

    return_series: series of investment returns, where the index is dates
    debt_amount: total amount of debt to be paid off
    int_rate: the APR on the debt
    debt_payment: the monthly payment to towards the debt
    excess_inv: the amount going towards investment, stated as an excess to the
    debt payment
    """

    #determine how many months it will take to payoff the debt
    n_months = payoff_nmonths(debt_amount, debt_payment, int_rate)
    assert len(return_series) > n_months, "not enough investment periods"

    start_date = return_series.index[0]

    #create the debt payoff
    dbt_sched = debt_schedule(start_date, debt_amount, int_rate, debt_payment)
    dbt_pymnts = dbt_sched['payment']

    #create the investment payoff
    inv_a = pandas.Series([debt_payment + excess_inv], 
        index = return_series.index[: len(dbt_sched)])
    inv_a[-1] = debt_payment - dbt_pymnts[-1] + excess_inv + debt_payment
    inv_b = pandas.Series([debt_payment + excess_inv + debt_payment],
        index = return_series.index[len(dbt_pymnts):])
    inv_payment = inv_a.append(inv_b)
    inv_sched = inv_schedule(inv_payment, return_series)
    inv_sched.name = 'inv_value'

    agg = pandas.concat( [dbt_sched, inv_sched], axis = 1)
    agg = agg.fillna(0.)
    return agg
    #return agg['inv_value'] - agg['loan_schedule']

def loans_first_payout(return_series, debt_amount, int_rate, debt_payment, excess_inv):
    """
    Scenario where student debt is paid off first, then the entire amount goes 
    towards investment

    INPUTS:
    -------

    return_series: series of investment returns, where the index is dates
    debt_amount: total amount of debt to be paid off
    int_rate: the APR on the debt
    debt_payment: the monthly payment to towards the debt
    excess_inv: the amount going towards investment, stated as an excess to the
    debt payment
    """

    #determine how many months it will take to payoff the debt
    n_months = payoff_nmonths(debt_amount, debt_payment, int_rate)
    assert len(return_series) > n_months, "not enough investment periods"

    start_date = return_series.index[0]

    #create the debt payoff
    dbt_sched = debt_schedule(start_date, debt_amount, int_rate, 2. * debt_payment + excess_inv)
    dbt_pymnts = dbt_sched['payment']

    #create the investment payoff
    inv = pandas.Series([2. * debt_payment + excess_inv], 
        index = return_series.index[len(dbt_sched) - 1: ])
    inv[0] = debt_payment - dbt_pymnts[-1] + excess_inv

    inv_sched = inv_schedule(inv, return_series[inv.index])
    inv_sched.name = 'inv_value'

    agg = pandas.concat( [dbt_sched, inv_sched], axis = 1)
    agg = agg.fillna(0.)
    return agg
#    return agg['inv_value'] - agg['loan_schedule']


def gen_gbm_price_series(num_years, N, price_0, vol, drift):
    """
    Return a price series generated using GBM
    
    INPUTS:
    -------
    num_years: number of years (if 20 trading days, then 20/252)
    N: number of total periods
    price_0: starting price for the security
    vol: the volatility of the security
    return: the expected return of the security
    
    RETURNS:
    --------
    Pandas.Series of length n of the simulated price series
    
    """
    
    dt = num_years/float(N)
    e1 = (drift - 0.5*vol**2)*dt
    e2 = (vol*numpy.sqrt(dt))
    cum_shocks = numpy.cumsum(numpy.random.randn(N,))
    cum_drift = numpy.arange(1, N + 1)
    series = numpy.append(price_0, price_0 * numpy.exp(
        cum_drift*e1 + cum_shocks*e2)[:-1])
    return pandas.Series(series)


def inv_schedule(inv_payments,inv_path):
    """

    INPUTS:
    -------

    start_date: string of 'mm-dd-yyyy' of the start datetime
    inv_payment: the amount to invest monthly
    inv_path: the monthly linear return of the investment, first value is NaN
    """

    ret_d = {inv_path.index[0]: 0.}
    value = 0.
    for i, date in enumerate(inv_path.index[1:]):
        value = (inv_payments[i] + value ) * (1 + inv_path[date])
        ret_d[date] = value

    return pandas.Series(ret_d)

def debt_schedule(start_date, debt_amount, int_rate, debt_payment):
    """

    INPUTS:
    --------

    start_date: string of 'mm-dd-yyyy' of the start datetime
    debt_amount: amount of the loan
    int_rate: rate of the loan
    num_years: num_years to pay down the loan
    """
    from dateutil.relativedelta import relativedelta
    from pandas.tseries.offsets import BMonthEnd

    strptime = datetime.datetime.strptime
    d = start_date
    if isinstance(start_date, str):
        d = strptime(start_date, '%m-%d-%Y')
    debt_schedule = [debt_amount]
    debt = debt_amount
    month = 0
    paid_in = [0.]
    while debt > 0.:
        new_bal = debt * (1. + int_rate / 12.) - debt_payment
        debt_schedule.append( new_bal )
        debt = new_bal
        month += 1
        paid_in.append ( debt_payment )

    #the final payment of the remaining balance
    debt_schedule[-1] = 0.
    paid_in[-1] = debt_payment + new_bal

    index = map(lambda x: d + BMonthEnd(n = x), range(month + 1))

    return pandas.DataFrame(
        {'payment': paid_in, 'loan_schedule': debt_schedule }, 
        index = index )

def payoff_nmonths(debt_amount, debt_payment, int_rate):
    """

    INPUTS:
    -------

    debt_amount: amount of the loan
    debt_payment: the amount of payment to be made each month
    int_rate: rate of the loan

    RETURNS:
    --------

    the number of months needed to repay the loan
    """
    debt = debt_amount
    month = 0
    while debt > 0.:
        new_bal = debt * (1. + int_rate / 12.) - debt_payment
        debt = new_bal
        month += 1
    return month

