from enum import Enum

class Params(Enum):
    BATCH_SZIE = 60
    LEARNING_RATE = 0.005
    EPOCHS = 100
    X_ANCHOR = '\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec\multipolygons'
    X_POS_ANCHOR = '\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec\multipolygons'
    X_NEG_ANCHOR = '\\SWPUKNAS2201\dst\Data\Projects\RSIB\data\raw\loc2vec\multipolygons'