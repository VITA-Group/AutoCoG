parameters_dict = {
    "MODEL.NAS.AGGR_TYPE": {
        'values': [
            # ['lstm'],
            ["add"],
            ["max"],
            ["concat"],
            # ["add",  "max", "concat"],
            # ['add', 'max', 'lstm']
        ]
    },
    "MODEL.NAS.ATT_TYPE": {
        'values': [
            ['add', 'max'],
            ['gat', 'gat_sym'],
            ['linear', 'generalized_linear'],
            ['gcn', 'cos', 'const'],
            # ["gat", "gcn", "cos", "const", "gat_sym",
            #  'linear', 'generalized_linear', 'add', 'max'],
        ]
    },
    "MODEL.NAS.ACT_TYPE": {
        'values': [
            ['relu', 'leaky_relu', 'relu6', 'elu'],
            ['sigmoid', 'tanh', 'softplus'],
            # ['relu', "sigmoid", "tanh",  "linear",
            #  "softplus", "leaky_relu", "relu6", "elu"]
        ]
    },
    "MODEL.NAS.UPDATE_TYPE": {
        'values':[
            ['identity'],
            ['mlp'],
            # ['identity', 'mlp']
        ]
    },
    "MODEL.NAS.DATA_POLICY": {
        'values': [
            ['none'] ,
            ['edge_perturbation'],
            ['node_drop'] ,
        ]
    } ,
    "MODEL.NAS.SKIP": {
        'values': [
            [True], [False],
        ]
    } ,

}

