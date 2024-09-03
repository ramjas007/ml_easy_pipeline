import hyperimpute
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from hyperimpute.plugins.imputers import Imputers

print(Imputers().list())

data=pd.concat([data1,data2],axis=0)

plugin = Imputers().get("hyperimpute")
data_fill=plugin.fit_transform(np.array(data))
data_fill=data_fill[:len(data1)]

data_fill=data_fill.rename(columns={"0":"author","1":"geometry","2":"pressure [MPa]","3":"mass_flux [kg/m2-s]",
                          "4":"x_e_out [-]","5":"D_e [mm]","6":"D_h [mm]","7":"length [mm]","8":"chf_exp [MW/m2]"})
