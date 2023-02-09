import pandas as pd
import numpy as np


a = pd.Series([np.nan, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, np.nan])




print(f"-- {a.interpolate(limit=1, limit_direction='both')=} --")
print(f"-- {a.interpolate(limit=1, limit_direction='both', limit_area='inside')=} --")
print(f"-- {a.interpolate(limit=1, limit_direction='both', limit_area='outside')=} --")
