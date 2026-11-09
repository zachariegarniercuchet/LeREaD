import json
m = json.load(open('output_coref/test_baseline_coref_strong_from-test_baseline_rules-final/1983CanLII2967L/1983CanLII2967L_resolution_mapping.json', encoding='utf-8'))
print('Total docids mapped:', len(m))
for k, v in list(m.items())[:5]:
    print('  %s: uri=%s | method=%s | matched_key=%s' % (k, v['uri'], v['method'], v['matched_key']))
methods = {}
for v in m.values():
    methods[v['method']] = methods.get(v['method'], 0) + 1
print('Methods:', methods)
