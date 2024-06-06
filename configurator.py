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
        # argument is the name (key) and value of a hyperparameter
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to evaluate it as boolean, integer, etc.
                eval_val = literal_eval(val)
            except (SyntaxError, ValueError):
                # if evaluation does not work we assume we can use the string
                eval_val = val
            assert type(eval_val) == type(globals()[key])
            print(f"Overriding: {key} = {eval_val}")
            globals()[key] = eval_val
        else:
            raise ValueError(f"Unknown key: {key}")
