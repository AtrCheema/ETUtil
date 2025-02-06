import os
import site
# add parent directory to path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(wd_dir)
site.addsitedir(wd_dir)

import unittest

from ETUtil.utils import freq_in_mins_from_string

for freq in ['1D', '20D', '5days',  'Daily', 'hourly',
             '5hours', '10H', '3Hour', 'min', 'minute',
             '6min', '6mins', '10minutes',  'Hourly', '3Day']:
    print(freq, freq_in_mins_from_string(freq))

if __name__ == "__main__":
    unittest.main()
