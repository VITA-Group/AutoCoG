parameters_dict = {
    "GRAPH.NUM_VIEW": {
        'values': [1,2,4,8,16,32]
    },
    "GRAPH.LR": {
        'distribution': 'uniform',
        'min': 1e-4,
        'max': 1e-2
    },
    "GRAPH.SMOOTHNESS_RATIO": {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 1.0
    },
    # "GRAPH.DEGREE_RATIO": {
    #     'distribution': 'uniform' ,
    #     'min': 0.1 ,
    #     'max': 1.0
    # },
    "GRAPH.SPARSITY_RATIO": {
        'distribution': 'uniform' ,
        'min': 0.1 ,
        'max': 1.0
    },
    "GRAPH.LAMBDA": {
        'distribution': 'uniform' ,
        'min': 0.1 ,
        'max': 1.0
    },
    "GRAPH.MU": {
        'distribution': 'uniform' ,
        'min': 0.1 ,
        'max': 1.0
    },
    "GRAPH.EPS": {
        'distribution': 'uniform' ,
        'min': 0.8 ,
        'max': 0.99
    },
    "GRAPH.HIDDEN": {
        'values': [4,8,16,32,64]
    } ,

}

