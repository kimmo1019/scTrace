from scdec import scDEC

if __name__=="__main__":
    params = {
        'data_type': 'scATAC-seq',
        'dataset': 'Splenocyte',
        'nb_classes': 12,
        'z_dim': 8,
        'x_dim': 50,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':50000,
        'has_label': True,
        'train': True
    }
    scdec_model = scDEC(params)
    scdec_model.train()