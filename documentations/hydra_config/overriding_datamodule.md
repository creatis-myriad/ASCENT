# Overriding datamodule configurations in ASCENT

ASCENT now supports more dynamic configurations for datamodules, including the data
augmentations and data loading config groups. The [hierarchical structure](../../ascent/configs/datamodule)
of the datamodule config is as follows:

```
ascent/
├── configs/
│   ├──datamodule/
│   │  ├──augmentation/
│   │  │  ├──aliasing/
│   │  │  ├──flip/
│   │  │  ├──intensity/
│   │  │  ├── ...
│   │  │  ├──default_2d.yaml
│   │  │  └──default_3d.yaml
│   │  │
│   │  ├──loading/
│   │  │  ├──crop/
│   │  │  ├──data_loading/
│   │  │  ├──pad/
│   │  │  ├── ...
│   │  │  ├──default_train.yaml
│   │  │  └──default_test.yaml
│   │  │
│   │  ├──nnunet.yaml
│   │  ├──nnunet_reg.yaml
```

## Overriding data augmentation configurations

The default 2D and 3D data augmentation configurations are stored in
[`default_2d.yaml`](../../ascent/configs/datamodule/augmentation/default_2d.yaml) and
[`default_3d.yaml`](../../ascent/configs/datamodule/augmentation/default_3d.yaml) respectively.
These configurations are automatically adjusted based on the input patch size, whether to do dummy
2D augmentation thanks to the [custom Hydra resolvers](../../hydra_plugins/ascent/resolvers.py).

To override the default data augmentation configurations, you can three options. You are free to choose
any of them based on your needs.

1. Create a new config file in [`datamodule/augmentation/`](../../ascent/configs/datamodule/augmentation).
   An example is given in [`datamodule/augmentation/dealias_2d.yaml`](../../ascent/configs/datamodule/augmentation/dealias_2d.yaml).
   Then, you can specify the augmentation config file in the datamodule config file, e.g.,
   [`datamodule/dealias_2d.yaml`](../../ascent/configs/datamodule/dealias_2d.yaml).
2. Override the augmentation config in the datamodule config file directly. An example is given in
   [`datamodule/nnunet_reg.yaml`](../../ascent/configs/datamodule/nnunet_reg.yaml).
3. Override the augmentation configs in the command line. Examples:
   - Deactivate flip augmentation during the training:
     ```bash
     ascent_train experiment=camus_challenge_2d ~datamodule.augmentation.flip
     ```
   - Add `flip_z` augmentation during the training:
     ```bash
     ascent_train experiment=camus_challenge_2d +datamodule/augmentation/flip=rand_flip_z
     ```
   - Modify the `rand_flip_x` probability to 0.2 during the training:
     ```bash
     ascent_train experiment=camus_challenge_2d datamodule.augmentation.flip.rand_flip_x.prob=0.2
     ```

## Overriding data loading configurations

The default train/val and test data loading configurations are stored in
[`datamodule/loading/default_train.yaml`](../../ascent/configs/datamodule/loading/default_train.yaml) and
[`datamodule/loading/default_test.yaml`](../../ascent/configs/datamodule/loading/default_test.yaml)
respectively. Like the data augmentation configs, these configurations are also automatically adjusted
based on the input patch size by using the [custom Hydra resolvers](../../hydra_plugins/ascent/resolvers.py).
To simplify the management of the data loading configs, `@package` directive is used. More details
can be found [here](https://hydra.cc/docs/advanced/overriding_packages).

Similar to the data augmentation configs, you can override the data loading configs in the datamodule
config file or in the command line.

1. Override loading config in the datamodule config. An example of overriding a group config of
   [`datamodule/loading/`](../../ascent/configs/datamodule/loading) is given in
   [`datamodule/dealiasv.yaml`](../../ascent/configs/datamodule/dealiasv_2d.yaml).
2. Override via the command line. Examples:
   - Use `rand_spatial_crop` instead of `rand_posneg_crop` for the train data loading:
     ```bash
     # the syntax is more complicated to override a group config
     ascent_train experiment=camus_challenge_2d datamodule/loading/crop@datamodule.loading.train.crop=rand_spatial_crop
     ```
   - Change the pad mode in train data loading transform to `reflect`:
     ```bash
     ascent_train experiment=camus_challenge_2d datamodule.loading.train.pad.mode=reflect
     ```
