# Script to load in the X-ray and optical catalogue data for pointing our observations
# Created: 2025-03-31, by: Joseph Hall

import numpy as np
from astropy.io import fits
from astropy.table import Table


def load_catalogue(cat_path="/home/ko23871/x-ray_data_analysis/data/emain_wen-han_final_20250328_1052"):
    """
    Loads and cleans the catalogue from TOPCAT cluster matching. Extensions [7] & [10] are the ones to use.
    
    Extensions are:
        0: PrimaryHDU
            (header information)
        1: emain_full_moc 
            (original erosita clusters in the sky region)
        2: wen-han_moc_exps 
            (original wen-han clusters in the sky region)
        3: emainXwh_full 
            (First set of matching from eROSTIA to Wen-Han, reduced to be within match parameters)
        4: whXemain_full 
            (First set of matching from Wen-Han to eROSITA, reduced to be within match parameters)
        5: emainXwh_noWHdupes 
            (removal of duplicate wen-han clusters from [3])
        6: emainXwh_noEMAINnoWHdupes 
            (removal of duplicate emain clusters from [5], and trimmed to the volume + L_min)
        7: emainXwh_final 
            (The final emain selected catalogue, matching [6] and [3] to include emain clusters without WH matches)
        8: whXemain_noEMAINdupes
            (removal of duplicate emain clusters from [4])
        9: whXemain_noWHnoEMAINdupes
            (removal of duplicate WH clusters from [8], and trimmed to volume + lambda_min)
        10: whXemain_final
            (The final wh selected catalogue, matching [9] and [4] to include WH clusters without emain matches)
    """
    # Load the tables
    with fits.open(cat_path) as hdul:
        emain_select = Table(hdul[7].data)
        wh_select = Table(hdul[10].data)

    # Clean the tables up (remove duplicated columns), start by filling columns where the clusters was missed because
    #   the emain version outside the selected volume
    emain_nan_mask = np.isnan(emain_select["RA"])
    emain_col2repl = [
        'DETUID_1','RA','DEC','EXP_1','BEST_Z_1','BEST_ZERR_1','L500_1','L500_L_1',
        'L500_H_1','R500','R500_L_1','R500_H_1','max_dz','max_dTheta_r500',
        'max_dTheta_1750'
        ]
    emain_replcols = [
        'DETUID_2','RA_1','DEC_1','EXP_2','BEST_Z_2','BEST_ZERR_2','L500_2','L500_L_2','L500_H_2',
        'R500_1','R500_L_2', 'R500_H_2','max_dz_1','max_dTheta_r500_1','max_dTheta_1750_1'
        ]
    
    for col2repl, replcol in zip(emain_col2repl, emain_replcols):
        emain_select[col2repl][emain_nan_mask] = emain_select[replcol][emain_nan_mask]

    # Delete the emain_replcols and sort
    emain_select.remove_columns(emain_replcols)
    emain_select.sort("L500_1", reverse=True)

    # Now repeat the process for the WH selected clusters
    wh_nan_mask = np.isnan(wh_select["RA"])
    wh_col2repl = [
        'ID_1','RA','DEC','zCl_1','r500','lambda500_1','M500_1','ERASS_EXP_1','max_dtheta_R500','max_dz',
        'max_dtheta_1750'
    ]
    wh_replcols = [
        'ID_2','RA_1','DEC_1','zCl_2','r500_1','lambda500_2','M500_2','ERASS_EXP_2','max_dtheta_R500_1','max_dz_1','max_dtheta_1750_1'
    ]
    wh_new_cols = [
        'DETUID_1','RA','DEC','BEST_Z_1','R500','lambda500_1','M500_1','ERASS_EXP_1','max_dtheta_R500','max_dz',
        'max_dtheta_1750'
        ]

    for col2repl, replcol, newcol in zip(wh_col2repl, wh_replcols, wh_new_cols):
        wh_select[col2repl][wh_nan_mask] = wh_select[replcol][wh_nan_mask]
        wh_select[newcol] = wh_select[col2repl]

    wh_select["R500"] *= 1000

    wh_select.sort('lambda500_1', reverse=True)
    
    return emain_select, wh_select


if __name__ == "__main__":
    # Get the current directory
    import os
    cur_path = os.getcwd().split('/')
    if cur_path[-1] == "src":
        cur_path = '/'.join(cur_path[:-1])
    else:
        cur_path = '/'.join(cur_path)
    print(cur_path)

    load_catalogue(f"{cur_path}/data/emain_wen-han_final_20250328_1052")
