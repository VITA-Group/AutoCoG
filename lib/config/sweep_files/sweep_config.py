parameters_dict = {
    "MODEL.NUM_LAYERS": {
        'values': [2, 3, 4]
    },
    "MODEL.N_HEADS": {
        'values': [1,2,3,4,5]
    },
    "MODEL.HIDDEN_DIM": {
        'values': [32,64,128,256]
    },
    "MODEL.STABILIZER": {
        'values': [True, False]
    },
    "MODEL.DROPOUT": {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.6
    },
    "TRAIN.LR": {
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.01
    },
    "TRAIN.LR_ARCH": {
        'distribution': 'uniform' ,
        'min': 0.001 ,
        'max': 0.01
    },
}