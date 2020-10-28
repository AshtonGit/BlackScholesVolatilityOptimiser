import numpy as np
import scipy.stats as si
import math
import csv
import re
import datetime
from pathlib import Path
from decimal import *

def black_and_scholes_no_dividend(min_date, max_date, option, volatility, interest):
    start = parse_date_y_m_d(option["date"])
    expiry = parse_date_d_m_y(option["expiry_date"])
    if start < min_date or start > max_date:
        return None
    S = float(option["heads_close"]) #spot price
    K = float(option["strike_price"]) #strike price
    T = days_to_maturity(start, expiry) #maturity
    r = interest
    sigma = volatility
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call

# Find the optimum volatility for the black and scholes formula by  iterating through all possible volatility values
# in a given range(min_volatility, max_volatility) inclusive. Increment the volatility by v_increment each iteration.
# For each volatility, calculate call prices for all stocks at all dates, storing results as a matrix
# then calculate the mean squared error for that volatility by comparing its calls to the real price.
# Store the mean squared error in a dict as final result.
# Find and return the volatility with the min error
# assume the indices of input_matrix and real_prices matchup and reference the same stock_ticker and date.

def optimise_volatility_for_date_range(data, min_date, max_date, min_volatility, max_volatility, v_increment, interest):
    inputs, controls = data[0], data[1]
    volatility = min_volatility
    results = {}
    while volatility <= max_volatility:
        #dict of calls, option ticker is key
        calls = {}
        for option in inputs:
            call = black_and_scholes_no_dividend(min_date, max_date, option, volatility, interest)
            if call is None:
                continue
            calls[option["option_ticker"]+option["date"]] = call

        mse = mean_squared_error(calls, controls)
        results[volatility] = mse
        volatility += v_increment

    best_fit_volatility = get_min(results)
    return (best_fit_volatility, results)

# data: (options, controls) tuple containing data used to produce the model
#       options: {ticker: days} dictionary of option data. key is the option_ticker, value is a dict
#       days: {ticker+date: stock_data} dictionary, key is option_ticker + date, value is the pricing data for that option on that date
#
# controls: {ticker+date: option_price} dictionary containing pricing data for options on a given day. Used as the control when evaluating accuracy of black and scholes formula.

def get_outlier_volatilies(data,std_dev, min_date, max_date, min_volatility, max_volatility, v_increment, interest):
    price_data, controls = data[0], data[1]
    option_tickers = price_data.keys()
    best_fit_volatilities = {}

    for ticker in option_tickers:
        errors = {}
        volatility = min_volatility
        while volatility <= max_volatility:
            calls = {}
            for option in price_data[ticker]:
                date = option["date"]
                call = black_and_scholes_no_dividend(min_date, max_date, option,volatility, interest)
                if call is None:
                    continue
                key = ticker + date
                calls[key] = call
            mse = mean_squared_error(calls, controls)
            errors[volatility] = mse
            volatility += v_increment
        best_fit_v = get_min(errors)
        best_fit_volatilities[ticker] = best_fit_v[0]

    #get the n outliers.
    return find_outliers(best_fit_volatilities, standard_deviations= std_dev)

def get_outlier_volatility_deltas(data, std_dev, date_range_A, date_range_B,  min_volatility, max_volatility, v_increment, interest):
    price_data, controls = data[0], data[1]
    option_tickers = price_data.keys()
    deltas = {}
    min_A, max_A = date_range_A[0], date_range_A[1]
    min_B, max_B = date_range_B[0], date_range_B[1]
    for ticker in option_tickers:
        # first go through all date_rangeA
        errors_A = {}
        errors_B = {}
        volatility = min_volatility
        while volatility <= max_volatility:
            calls_A = {}
            calls_B = {}
            # then go through all date_rangeB
            for option in price_data[ticker]:
                date = option["date"]
                start = parse_date_y_m_d(date)
                #if option data point is outside either date range, skip iteration
                if (start < min_A or start > max_A) and (start > max_B or start < min_B):
                    continue

                key = ticker + date
                if start >= min_A and start <= max_A:
                    call = black_and_scholes_no_dividend(min_A, max_A, option, volatility, interest )
                    calls_A[key] = call
                else:
                    call = black_and_scholes_no_dividend(min_B, max_B, option, volatility, interest)
                    calls_B[key] = call

            mse_A = mean_squared_error(calls_A, controls)
            mse_B = mean_squared_error(calls_B, controls)
            errors_A[volatility] = mse_A
            errors_B[volatility] = mse_B
            volatility += v_increment
        best_fit_B = get_min(errors_B)[0]
        best_fit_A = get_min(errors_A)[0]
        delta = abs(best_fit_A - best_fit_B)
        deltas[ticker] = delta

    return find_outliers(deltas, standard_deviations= std_dev)



    #get difference, get outliers
    # calc delta for each volatility
    # calc average delta
    # get all outliesr

def parse_csv(csv_file, header_size):
    raw = open(Path(csv_file).resolve(), "r")
    treated = raw.read()
    rows = treated.splitlines(keepends=False)

    data = []
    #dict of control prices to gauge error. key is option ticker
    controls = {}
    #skip the header
    for i in range(header_size):
        del rows[0]

    for r in rows:
        r.strip()
        row = r.split(',')
        strip = []
        #clean data, removing quotation marks and excess whitespace
        for string in row:
            strip.append(string.replace("\"", '').rstrip().lstrip())
        # in given csv, expiry date format was given as dd-mm-yy. years must be transformed from yy to yyyy
        expiry = strip[5][0:-2]+'20'+strip[5][-2:]
        option = {"head_ticker": strip[0], "option_ticker": strip[1] , "date":strip[2] , "heads_close":strip[3] , "options_price":strip[4] , "expiry_date":expiry , "strike_price":strip[6]}
        controls[strip[1]+strip[2]] = (float(strip[4]))
        data.append(option)
    raw.close()
    return (data, controls)


def parse_csv_groupby_options(csv_file, header_size):
    raw = open(csv_file, "r")
    rows = raw.readlines()
    options = {}
    controls = {}

    for i in range(header_size):
        del rows[0]

    for r in rows:
        r.strip()
        row = r.split(',')
        clean = []
        for string in row:
            clean.append(string.replace("\"", '').rstrip().lstrip())
        expiry = clean[5][0:-2]+'20'+clean[5][-2:]
        option_ticker = clean[1]
        option = {"date": clean[2], "heads_close": clean[3], "options_price":clean[4], "expiry_date": expiry, "strike_price":clean[6]}
        controls[option_ticker + clean[2]] = float(clean[4])
        if option_ticker not in options:
            options[option_ticker] = [option]
        else:
            options[option_ticker].append(option)

    raw.close()
    return (options, controls)

def find_outliers(data, standard_deviations):
    outliers = {}
    # Set upper and lower limit to 2 standard deviation
    val = list(data.values())
    std_dev = np.std(val)
    mean = np.mean(val)
    outlier_cut_off = std_dev * standard_deviations
    lower_limit  = mean - outlier_cut_off
    upper_limit = mean + outlier_cut_off

    for option, v in data.items():
        if v > upper_limit or v < lower_limit:
            outliers[option] = v

    return outliers

def mean_squared_error(outputs, controls):
    n = len(outputs)
    sum_sqe = 0
    for k in outputs.keys():
        sqe = (outputs[k] - controls[k])**2
        sum_sqe += sqe
    if sum_sqe == 0.0:
        return 0.0
    return sum_sqe / n

#return volatility value with the lowest mean squared error in regards to its predictive value in the
# black and scholes formula
# volatility_errors: dictionary where key = volatiltiy, value = mean_squared_error
def get_min(volatility_errors):
    min = math.inf
    min_v = math.inf
    for v, mse in volatility_errors.items():
        if mse < min:
            min = mse
            min_v = v
    return (min_v, min)



def days_to_maturity(start, expiry):
    difference = expiry - start
    days = difference.days
    return days

def parse_date_y_m_d(date_string):
    split = date_string.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.date(year, month, day)

def parse_date_d_m_y(date_string):
    split = date_string.split('-')
    day, month, year = int(split[0]), int(split[1]), int(split[2])
    return datetime.date(year, month, day)

def parse_date_m_d_y(date_string):
    split = date_string.split('-')
    month, day, year =int(split[0]), int(split[1]), int(split[2])
    return datetime.date(year, month, day)

def prompt_header():
    while True:
        try:
            header_size = input("Enter size of csv header in lines (0 if none):")
            s = int(header_size)
            return s
        except Exception as e:
            print("Invalid arg: "+header_size)
            print("Header size must be an integer")

help_text = "optimise [csv_file] [min_date] [max_date] [min_v] [max_v] [v_increment] [interest\n" \
            "outlier_volatility [csv_file] [min_date] [max_date] [min_v] [max_v] [v_increment] [interest] [stdev]\n" \
            "outlier_delta [csv_file] [startA] [endA] [startB] [endB] [min_v] [max_v] [v_increment] [interest] [stdev]\n" \
            "---\t---\t---\n" \
            "[csv_file]                 Absolute file path to a csv_file containing the options data for the black and scholes algorithm. \n" \
            "                           CSV format: Heads Ticker,Options Ticker,Date,Heads Close,Options Price,Expiry Date,Strike Price\n"\
            "[min_date], [max_date]     dd-mm-yyyy Valid date range to analyse options. Options out of this range are not considered.\n" \
            "[startA] [endA]            dd-mm-yyyy First date range to analyse options when comparing volatility delta.\n" \
            "[startB] [endB]            dd-mm-yyyy Second date range to analyse options when comparing volatility delta. \n"\
            "min_v, max_v               Highest and smallest possible volatilities to explore when optimising volatility.\n" \
            "v_increment                How much to increment volatility each iteration when optimising volatility\n" \
            "interest                   annualized interest rate for black and scholes formula\n" \
            "stdev                      How many standard deviations a data point must fall out of the average to be\n" \
            "                           considered an outlier"

getcontext().prec = 3
print("Welcome to the Black&Scholes volatility optimiser")
print("Enter 'help' for a list of commands")
while True:
    inp = input("$")
    if inp == 'help':
        print(help_text)
    elif inp == 'quit' or inp == 'q':
        quit()
        break;
    else:
        cmd = inp.split(' ')
        if cmd[0] == 'optimise':
            csv, min_date, max_date, min_v, max_v, v_increment, interest =\
                cmd[1], parse_date_d_m_y(cmd[2]), parse_date_d_m_y(cmd[3]), float(cmd[4]), float(cmd[5]), float(cmd[6]), float(cmd[7])
            header_size = prompt_header()
            data = parse_csv(csv, header_size)
            results = optimise_volatility_for_date_range(data, min_date, max_date, min_v, max_v, v_increment, interest)
            print("Best Fit Volatility: "+str(results[0][0])+"\t MSE: "+str(Decimal.from_float(results[0][1]).quantize(Decimal('0.001'), ROUND_UP)))
            print("Results: ")
            print("Volatility - MSE")
            for k, v in results[1].items():
                print(str(Decimal.from_float(k).quantize(Decimal('0.0001'), ROUND_DOWN))+" - "+str(v))
        elif cmd[0]== 'outlier_volatility':
            csv, min_date, max_date, min_v, max_v, v_increment, interest, stdev =\
            cmd[1], parse_date_d_m_y(cmd[2]), parse_date_d_m_y(cmd[3]), float(cmd[4]), float(cmd[5]), float(cmd[6]), float(cmd[7]), int(cmd[8])
            header_size = prompt_header()
            data = parse_csv_groupby_options(csv, header_size)
            results = get_outlier_volatilies(data, stdev, min_date, max_date, min_v, max_v, v_increment, interest)
            print("Options with outlier best fit volatilities")
            print("Option Ticker - Volatility")
            for k, v in results.items():
                print(k+" - "+str(v))
        elif cmd[0] == 'outlier_delta':
            csv, min_dateA, max_dateA, min_dateB, max_dateB, min_v, max_v, v_increment, interest, stdev = \
                cmd[1], parse_date_d_m_y(cmd[2]), parse_date_d_m_y(cmd[3]), parse_date_d_m_y(cmd[4]), parse_date_d_m_y(cmd[5]),\
                float(cmd[6]), float(cmd[7]), float(cmd[8]), float(cmd[9]), int(cmd[10])
            header_size = prompt_header()
            data = parse_csv_groupby_options(csv, header_size)
            results = get_outlier_volatility_deltas(data, stdev, (min_dateA, max_dateA), (min_dateB, max_dateB), min_v, max_v, v_increment,interest)
            print("Options with outlier shifts in optimum volatity")
            print("Option Ticker - Volatility delta")
            for k, v in results.items():
                print(k+"-"+str(v))
        else:
            print("Command not recognised")
            print("Enter 'help' for a list of commands")
            continue





