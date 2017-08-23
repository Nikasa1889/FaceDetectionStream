import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--welcomemesfile',
    required=True,
    help="Specify welcome messages file"
)

parser.add_argument(
    '--jsfilein',
    required=True,
    help="Specify compiled js file input"
)
parser.add_argument(
    '--jsfileout',
    required=True,
    help="Specify compiled js file output"
)

args = parser.parse_args()
js_in = args.jsfilein
js_out = args.jsfileout

with open(js_in, 'r') as f:
    js_in_str = f.read()


