# License: BSD 3 clause

"""
The file has two blocks separated by a blank line. Each line in the first
block contains the id and name of a site:
<website id>,<website name>

Each line in the second block contains information about one meme (cascade).
The time is in UNIX time in hours.

<meme id>, <website id>, <timestamp>, <website id>, <timestamp>, <website id>,
<timestamp>...


Example
  ...
  4262,wthr.com
  8588,klkntv.com
  9995,presseportal.de
  10361,wnyc.org
  8709,kswt.com
  7954,woi-tv.com
  ...

  ...
  115642731;2838,366110.853056,5344,366113.987500,5726,366113.987500,...
  32875877;533,362176.518611,24963,362176.519722,1086,362176.519722,...
  32875878;533,362176.518611,24963,362176.519722,1086,362176.519722,...
  93254134;115,365054.000000,1214,365054.000000,1086,365054.004722,...
  48060773;5004,362899.355833,14638,362899.366667,1086,362899.366667,...
  ...

"""

import numpy as np
from numpy.random import default_rng


def read_cascade(filepath, break_ind=None):
    """
    Loading data.

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    break_ind : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    web_dict : TYPE
        DESCRIPTION.
    cascades_list : TYPE
        DESCRIPTION.

    """
    web_dict = {}
    cascades_list = []
    f = open(filepath, encoding="utf8")
    block_1 = True
    if break_ind is None:
        i = 0
        
        for line in f:
            if not bool(line.strip()):
                block_1 = False
            if block_1:
                splitted = line.split(',')
                website_id = splitted[0]
                website_name = splitted[1][:-1]
                web_dict[website_id] = website_name
            if not block_1 and bool(line.strip()):
                cascades_list.append({'meme_id': None, 'cascade times': [],
                                      'cascade web ids': [],
                                      'cascade size': None,
                                      'line': i})
                splitted = line.split(';')
                cascades_list[-1]['meme_id'] = splitted[0]
                splitted_prime = splitted[1][:-1].split(',')
                assert (len(splitted_prime) % 2) == 0
                cascades_list[-1]['cascade size'] = len(splitted_prime)//2
                for ix in range(len(splitted_prime)//2):
                    cascades_list[-1]['cascade web ids'].append(splitted_prime[2*ix])
                    cascades_list[-1]['cascade times'].append(float(splitted_prime[2*ix+1]))
            i += 1
    else:
        i = 0
        for line in f:
            if not bool(line.strip()):
                block_1 = False
            if block_1:
                splitted = line.split(',')
                website_id = splitted[0]
                website_name = splitted[1][:-1]
                web_dict[website_id] = website_name
            if not block_1 and bool(line.strip()):
                cascades_list.append({'meme_id': None, 'cascade times': [],
                                      'cascade web ids': [],
                                      'cascade size': None,
                                      'line': i})
                splitted = line.split(';')
                cascades_list[-1]['meme_id'] = splitted[0]
                splitted_prime = splitted[1][:-1].split(',')
                assert (len(splitted_prime) % 2) == 0
                cascades_list[-1]['cascade size'] = len(splitted_prime)//2
                for ix in range(len(splitted_prime)//2):
                    cascades_list[-1]['cascade web ids'].append(splitted_prime[2*ix])
                    cascades_list[-1]['cascade times'].append(float(splitted_prime[2*ix+1]))
            if i > break_ind:
                break
            i += 1
    f.close()
    return web_dict, cascades_list


# =============================================================================
# Aggregate event cascades
# =============================================================================
def flatten_cascades(cascades_list, mean=0.0, std=10**-5, base_seed=1234,
                     discard_collisions=False):
    time_evs = []
    for i in range(len(cascades_list)):
        for j in range(len(cascades_list[i]['cascade times'])):
            time_evs.append([cascades_list[i]['cascade web ids'][j],
                             cascades_list[i]['cascade times'][j]])

    time_evs = sorted(time_evs, key=lambda L: L[1], reverse=False)
    rng = default_rng(base_seed)
    if discard_collisions:
        mean = 0.
        std = 0.

    times = []
    web_ids = []

    start_index = 0
    while start_index <= len(time_evs)-2:
        end_index = start_index+1
        time_ref = time_evs[start_index][1]

        times.append(time_ref)
        web_ids.append(time_evs[start_index][0])

        local_web_ids = [time_evs[start_index][0]]
        delay = 0.

        while end_index < len(time_evs) and time_evs[end_index][1] == time_ref:
            if time_evs[end_index][0] not in local_web_ids:
                delay += abs(rng.normal(loc=mean, scale=std, size=1)[0])
                t = time_evs[end_index][1]+delay
                if not discard_collisions:
                    times.append(t)
                    web_ids.append(time_evs[end_index][0])
                local_web_ids.append(time_evs[end_index][0])
            end_index += 1
        start_index = end_index
    times = np.array(times)
    web_ids = np.array(web_ids)
    return times, web_ids


def rescale_times(times, rescale_factor, t_min, t_max):
    #   times is in Unix time, in hours
    times = rescale_factor*times
    times = times[(times >= t_min) & (times <= t_max)]
    times -= t_min
    return times

