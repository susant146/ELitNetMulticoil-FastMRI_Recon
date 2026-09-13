import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Data parameters
        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images')

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                          help='Which challenge')
        self.add_argument('--data_path', type=pathlib.Path, required=True,
                          help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--mask-type', choices=['random', 'equispaced'], default='random',
                          help='The type of mask function to use')
        self.add_argument('--accelerations', nargs='+', default=[4, 8], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')
        self.add_argument('--center-fractions', nargs='+', default=[0.08, 0.04], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')
        self.add_argument('--scname_train', type=str, default='singlecoil_train', help='GT folder name')
        self.add_argument('--scname_val', type=str, default='singlecoil_val', help='GT folder name')
        self.add_argument('--scname_test', type=str, default='singlecoil_test', help='GT folder name')
        self.add_argument('--rec_ext', type=str, default='rec_G1_4x', help='rec extension')
        self.add_argument('--rec2_ext', type=str, default=None, help='rec extension')




        # Override defaults with passed overrides
        self.set_defaults(**overrides)
