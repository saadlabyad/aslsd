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

import pandas as pd


def make_topdomain_names_dict(filepath):
    topdomain_names_dict = {}
    i = 0
    break_ind = 245
    f = open(filepath, encoding="utf8")
    for line in f:
        splitted = line.split(';')
        topdomain_names_dict[splitted[0]] = splitted[1]
        if i > break_ind:
            break
        i += 1
    f.close()
    return topdomain_names_dict


def try_url_ext(url, ext):
    n = len(ext)
    res = False
    if len(url) > n:
        if url[-n:] == ext:
            return True
    return res


def webdict2csv(web_dict, keyword, topdomain_filepath, title=None):
    topdomain_names_dict = make_topdomain_names_dict(topdomain_filepath)
    res = {'webid': [], 'weburl': [], 'country': []}
    for key in web_dict.keys():
        res['webid'].append(key)
        res['weburl'].append(web_dict[key])
        #   Country
        url = web_dict[key]
        cntry = ' '
        for topdomain in topdomain_names_dict.keys():
            if try_url_ext(url, topdomain):
                cntry = topdomain_names_dict[topdomain]
        res['country'].append(cntry)
    web_df = pd.DataFrame.from_dict(res)
    if title is None:
        title = 'webdict_'+keyword+'.csv'
    web_df.to_csv(title)


def load_countrywise_webdict(filepath):
    final_webdict = {}
    i = 0
    break_ind = 4999
    f = open(filepath, encoding="utf8")
    for line in f:
        if i > 0:
            splitted = line.split(';')
            final_webdict[splitted[1]] = {'url': splitted[2],
                                          'country': splitted[3][:-1]}
            if i > break_ind:
                break
        i += 1
    f.close()
    return final_webdict
