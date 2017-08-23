import argparse
import json

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
#  parse to get the current list of messages as string
args = parser.parse_args()
mes_file = args.welcomemesfile
js_in = args.jsfilein
js_out = args.jsfileout

with open(js_in, 'r') as f:
    js_in_str = f.read()

str_lookup = "function(e,t){e.exports=[{"
start = js_in_str.find(str_lookup) + len(str_lookup) - 2
assert js_in_str[start] == '[', 'wrong start location, should be [ symbol instead of {}'.format(js_in_str[start])

end = js_in_str.find('}]}', start) + 1
assert js_in_str[end] == ']', 'wrong start location, should be ] symbol instead of {}'.format(js_in_str[end])

print('current string:'.format(js_in_str[start:end+1]))


#  form the new welcome messages
with open(mes_file, 'r') as f:
    mes_dict = json.load(f)

print('mes_dict keys: {}'.format(mes_dict.keys()))
mes_list = []
for name in mes_dict:
    mes_list.append({
        "Navn":name,
        "Firma": "",
        "Velkomsmelding": mes_dict[name],
        "FacebookId": ""
    })

mes_list_str = json.dumps(mes_list).replace("\"Navn\"", "Navn").replace("\"Firma\"", "Firma").replace("\"Velkomsmelding\"", "Velkomsmelding").replace("\"FacebookId\"", "FacebookId")
print('mes_list_str: {}'.format(mes_list_str))

#  write to new js file
js_out_str = js_in_str.replace(js_in_str[start:end+1], mes_list_str)
with open(js_out, 'w') as f:
    f.write(js_out_str)

