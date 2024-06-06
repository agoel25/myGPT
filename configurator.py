"""
Configurator to override specific hyperparameter values in the actual configs
Example: 
$ python train.py config/gpt-2.py --batch_size=8
This will run config/gpt-2.py and override the "batch_size" parameter to the given value 8
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if not arg.startswith('--'):
        # argument is the name of the config
        print(f"Config will be overridden by {arg}")
        with open(arg) as file:
            print(file.read())
        exec(open(arg).read())
    else:
        # split argument in key and value