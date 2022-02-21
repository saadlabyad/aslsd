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


def malaria_rdata_to_csv(filepath):
    