from loc2vec.data_loader import Data_Loader
from loc2vec.config import Params

loader = Data_Loader(x_path=Params.X_PATH.value, x_pos_path=Params.X_POS_PATH.value)

for i in range(loader.batches):
    print(next(loader))