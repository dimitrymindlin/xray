import imgaug.augmenters as iaa

NONE_AUGMENTATION = iaa.Sequential([])

DEFAULT_AUGMENTATION = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                       iaa.Flipud(0.5),  # vertically flip 50% of all images,
                                       # iaa.PadToFixedSize(*run_params['image_resolution'], position='center')
                                       ])

ADVANCED_AUGMENTATION = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                        iaa.Flipud(0.5),  # vertically flip 50% of all images,
                                        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),  # Change brightness
                                        iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
                                        # Zoom in / zoom out
                                        iaa.Sometimes(0.5, iaa.Affine(rotate=(-20, 20), fit_output=True)),  # Rotate
                                        # iaa.PadToFixedSize(*run_params['image_resolution'], position='center')
                                        ])


COMMON_PARAMS = {
    'data_source': 'XR_HAND_PHOTOSHOP',
    'pipeline': {
        'hist_equalisation': True,
        'otsu_filter': True,
        'adaptive_hist_equilization': True,
        'normalisation': (0, 1),
    },
    'random_seed': [42, 4242, 424242, 42424242],
}

MODEL_SPECIFIC_PARAMS = {
    'BaselineAutoencoder': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 1000,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'BottleneckAutoencoder': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'SkipConnection': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'VAE': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'masked_loss_on_val': False,
        'masked_loss_on_train': False,
        'z_dim': 1024,
        'h_dim': 2048,
        'lr': 0.00005,
    },
    'DCGAN': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 80,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': False,
        'spectral_normalisation': True,
        'soft_labels': True,
        'glr': 0.001,
        'dlr': 0.00001,
        'soft_delta': 0.01,
        'z_dim': 2048,
        'checkpoint_frequency': 2000,
    },
    'AlphaGan': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 16,
        'image_resolution': (128, 128),
        'num_epochs': 500,
        'masked_loss_on_val': True,
        'gelr': 0.001,
        'dlr': 0.0001,
        'z_dim': 100,
        'checkpoint_frequency': 2000,
    },
    'SAGAN': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 16,
        'image_resolution': (320, 320),
        'num_epochs': 500,
        'masked_loss_on_val': True,
        'gelr': 0.001,
        'dlr': 0.000005,
        'adv_loss': 'hinge',
        'z_dim': 100,
        'checkpoint_frequency': 2000,
        'lambda_gp': 2,
    }
}
