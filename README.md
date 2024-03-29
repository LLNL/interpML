# Interpolation Models and Error Bounds for Verifiable Scientific Machine Learning (InterpML)
Version 1.0, March 2024.

This repository contains python scripts accompanying the paper:\
**Leveraging Interpolation Models and Error Bounds for Verifiable Scientific Machine Learning**\
Tyler Chang, Andrew Gillette, Romit Maulik\
2024

   

The following subdirectories are included:
 - ``interpml`` contains our interpolation scripts used for all studies
 - ``experiments`` contains scripts demonstrating our experiments with synthetic data
 - ``airfoil`` contains scripts demonstrating our experiments with the UIUC airfoil dataset

Further instructions are provided in each subdirectory.

## Setup

To make interpolation methods importable, use the ``setup.py`` file:
```
python3 -m pip install -e .
```


License
----------------

The code in this respository is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE](https://github.com/LLNL/interpML/blob/main/LICENSE.md) and
[NOTICE](https://github.com/LLNL/interpML/blob/main/NOTICE.md) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE-862386 