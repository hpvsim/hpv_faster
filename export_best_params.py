"""
Export best calibration parameters from _reduced.obj files to JSON format.

The _reduced.obj files are sciris objdicts with keys: analyzer_results, target_data, df.
Best params come from the df row with minimum mismatch.

Produces individual country JSON files and updates best_params_combined.json.
"""

import sciris as sc
import json

COUNTRIES = {
    'sierra_leone': 'data/sierra_leone_calib_reduced.obj',
}

PARAM_COLS = [
    'beta', 'f_cross_layer', 'm_cross_layer',
    'f_partners_c_par1', 'm_partners_c_par1',
    'hi5_cin_fn_k', 'ohr_cin_fn_k',
]


def flatten_calib(location, obj_path):
    calib = sc.load(obj_path)
    df = calib.df
    best_row = df.loc[df['mismatch'].idxmin()]

    flat = {col: float(best_row[col]) for col in PARAM_COLS if col in best_row.index}
    flat['_country']           = location
    flat['_mismatch']          = float(best_row['mismatch'])
    flat['_calibration_index'] = float(best_row['index'])
    return flat


if __name__ == '__main__':
    combined = {}
    for loc, path in COUNTRIES.items():
        print(f'Loading {path}...')
        pars = flatten_calib(loc, path)
        combined[loc] = pars
        out = f'{loc}_best_params.json'
        with open(out, 'w') as f:
            json.dump(pars, f, indent=2)
        print(f'  Wrote {out}')
        print(f'  Best mismatch: {pars["_mismatch"]:.4f}')
        print(f'  Params: { {k: round(v, 4) for k, v in pars.items() if not k.startswith("_")} }')

    # Merge into combined file (preserves any existing entries for other countries)
    try:
        with open('best_params_combined.json', 'r') as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    existing.update(combined)
    with open('best_params_combined.json', 'w') as f:
        json.dump(existing, f, indent=2)
    print('Updated best_params_combined.json')
