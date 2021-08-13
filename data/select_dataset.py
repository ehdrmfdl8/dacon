

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['l', 'low-quality', 'input-only']:
        from data.dataset_l import DatasetL as D

    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D

    elif dataset_type in ['plainpatch']:
        from data.dataset_plainpatch import DatasetPlainPatch as D

    elif dataset_type in ['paired']:
        from data.dataset_paired_image import DatasetPairedImage as D

    elif dataset_type in ['degradation']:
        from data.dataset_degradation import DatasetDegradation as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
