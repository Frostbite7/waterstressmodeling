import configparser
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
import pandas as pd

from noah_energy.phm.energy_driver import ENERGY_DRIVER_PHM, SOIL_LAYER_CONFIG


# from noah_energy.calibration.phm_cal import config_parameters


def main():
    # site info
    site = 'US-Wkg'
    LAT = 31.7365
    LON = -109.9419
    utc_offset = -7
    parm_path = '/Users/yangyicge/Desktop/watercon/script/noah_energy/parameters/{}/energy_parameter_phm.ini'.format(site)

    # simulation and configuration
    year = 2007
    start = datetime(year, 8, 1)
    end = datetime(year, 10, 1)
    results_path = '/Users/yangyicge/Desktop/watercon/noah_energy_case/applicability/{}/{}_{}/'.format(site, site, start.year)
    results_suffix = '_phm'
    soil_hydro = 0

    # flux data
    flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
    flux_file = flux_path + 'fluxnet/' + 'FLX_US-Wkg_FLUXNET2015_SUBSET_2004-2014_1-4/FLX_US-Wkg_FLUXNET2015_SUBSET_HH_2004-2014_1-4.csv'
    aflux_file = flux_path + 'ameriflux/' + 'AMF_US-Wkg_BASE-BADM_18-5/AMF_US-Wkg_BASE_HH_18-5.csv'

    # read LAI data
    force_start = datetime(year - 1, 1, 1)
    lines_to_skip = 67
    slice_start = (start - force_start).days * 24 + lines_to_skip
    slice_end = (end - force_start).days * 24 + lines_to_skip
    full_model_force = open(
        '/Users/yangyicge/Desktop/watercon/caseout/btran_uso_fluxnet_dry_nrt3/{}/{}_{}-{}_force/force.dat'.format(site[3:], site[3:],
                                                                                                                  year - 1, year), 'r')
    lai_force = []
    i = 0
    for line in full_model_force.readlines():
        if slice_start <= i < slice_end:
            lai_force.append(float(line[157:169]))
        i = i + 1

    # read flux forcing data
    print('reading data...')
    df = pd.read_csv(flux_file)

    time_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TIMESTAMP_START']
    ws_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'WS_F']
    temp_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA_F']
    rh_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH']
    prs_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'PA_F']
    soldn_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SW_IN_F']
    lwdn_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LW_IN_F']
    prec_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'P_F']
    gh_flux = df[(df.TIMESTAMP_START >= int(
        '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'G_F_MDS']

    # inspect gaps
    print('Forcing Gaps! ws, temp, rh, prs, soldn, lwdn, prec, gh:\n', np.sum(ws_flux == -9999), np.sum(temp_flux == -9999),
          np.sum(rh_flux == -9999), np.sum(prs_flux == -9999), np.sum(soldn_flux == -9999), np.sum(lwdn_flux == -9999),
          np.sum(prec_flux == -9999), np.sum(gh_flux == -9999))

    # gap filling
    ws_flux = ws_flux.replace(-9999, np.nan).interpolate()
    temp_flux = temp_flux.replace(-9999, np.nan).interpolate()
    rh_flux = rh_flux.replace(-9999, np.nan).interpolate()
    prs_flux = prs_flux.replace(-9999, np.nan).interpolate()
    soldn_flux = soldn_flux.replace(-9999, np.nan).interpolate()
    lwdn_flux = lwdn_flux.replace(-9999, np.nan).interpolate()
    prec_flux = prec_flux.replace(-9999, np.nan).interpolate()
    gh_flux = gh_flux.replace(-9999, np.nan).interpolate()

    # # read SWC data Ne3
    # dfa = pd.read_csv(aflux_file, skiprows=2)
    # v_number = 3
    # h_number = 1
    # swc_h = []
    # for point in range(h_number):
    #     swc_v_ = []
    #     for depth in range(v_number):
    #         swc_ = dfa[(dfa.TIMESTAMP_START >= int(
    #             '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
    #             '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_{}_{}_{}'.format(
    #             point + 1, depth + 1, '1')]
    #         print('Gaps! SWC_H{}V{}R1:'.format(point + 1, depth + 1), np.sum(swc_ == -9999))
    #         # swc_v_.append(swc_.replace(-9999, np.nan).interpolate().values / 100)
    #         swc_v_.append(swc_.replace(-9999, np.nan) / 100)
    #     swc_h.append(swc_v_)
    # swc_h_ = np.array(swc_h)
    # swc_v = np.nanmean(swc_h_, 0)
    # swc_v = np.maximum(swc_v, 0.02)
    # print('swc_v:', swc_v)

    # read SWC data Wkg
    dfa = pd.read_csv(aflux_file, skiprows=2)
    v_number = 3
    h_number = 1
    swc_h = []
    for point in range(h_number):
        swc_v_ = []
        for depth in range(v_number):
            swc_ = dfa[(dfa.TIMESTAMP_START >= int(
                '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
                '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_{}_{}_{}'.format(
                point + 1, depth + 1, 1)]
            print('Gaps! SWC_H{}V{}R1:'.format(point + 1, depth + 1), np.sum(swc_ == -9999))
            # swc_v_.append(swc_.replace(-9999, np.nan).interpolate().values / 100)
            swc_v_.append(swc_.replace(-9999, np.nan) / 100)
        swc_h.append(swc_v_)
    swc_h_ = np.array(swc_h)
    swc_v = np.nanmean(swc_h_, 0)
    swc_v = np.maximum(swc_v, 0.02)
    print('swc_v:', swc_v)

    # # read full model data
    # slice_start_out = (start - force_start).days * 96
    # slice_end_out = (end - force_start).days * 96
    # out = Dataset('/Users/yangyicge/Desktop/watercon/case_out/btran_uso_flux_old_nrt3/Ne1/Ne1_2006-2007_force/output/OUTPUT.nc')
    # sh2o = np.mean(np.reshape(out['SH2O'][slice_start_out:slice_end_out, :], (-1, 4, 4)), 1)
    # smc = np.mean(np.reshape(out['SMC'][slice_start_out:slice_end_out, :], (-1, 4, 4)), 1)
    # canliq = np.mean(np.reshape(out['CANLIQ'][slice_start_out:slice_end_out], (-1, 4)), 1)
    # fwet = np.mean(np.reshape(out['FWET'][slice_start_out:slice_end_out], (-1, 4)), 1)
    # gh_model = np.mean(np.reshape(out['SSOIL'][slice_start_out:slice_end_out], (-1, 4)), 1)

    # load parameters
    config = configparser.ConfigParser()
    config.read(parm_path)

    # # read calibrated parms (optional)
    # cal_parm_path = '/Users/yangyicge/Desktop/watercon/noah_energy_case/calibration/US-Me2_cal1/' \
    #                 'phm_cal_2013_1000repeat/sceua_opt_phm_parms_2013.npy'
    # cal_parms = np.load(cal_parm_path)
    # config = config_parameters(config, cal_parms)

    # model config
    DT, NSOIL, ZSOIL, DZSNSO = SOIL_LAYER_CONFIG(config)
    if DT == 1800:
        lai_force = np.repeat(lai_force, 2)

    # initialize results
    times = []
    transpiration = []
    soil_evaporation = []
    canopy_evaporation = []
    rssun = []
    rssha = []
    psn = []
    fsun = []
    fsh = []
    sav = []
    sag = []
    fsa = []
    fsr = []
    fira = []
    apar = []
    parsun = []
    parsha = []
    fstomata = []
    swc_related_sim = []
    lai = []
    tr_p = []

    # initialize state variables
    if soil_hydro:
        SH2O_out = np.full(NSOIL, swc_v[0, 0], dtype=float)
        SH2O_out[1] = 0.3
    else:
        SH2O_out = 'holder'

    # for diagnostics
    # print(lai_force)
    # quit()

    # run
    print('\nbegin running...')
    n_records = time_flux.shape[0]
    for i in range(n_records):
        time = str(time_flux.iloc[i])
        # TIME = datetime(int(time[:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12]),
        #                 tzinfo=timezone(timedelta(hours=-6)))
        TIME = datetime(int(time[:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12]),
                        tzinfo=timezone.utc) + timedelta(hours=-utc_offset)
        if i % 240 == 0:
            print('\n', TIME)

        # forcing variables from flux data
        WS = ws_flux.iloc[i]
        SFCTMP = temp_flux.iloc[i] + 273.15
        RH = rh_flux.iloc[i]
        SFCPRS = prs_flux.iloc[i] * 1000
        SOLDN = soldn_flux.iloc[i]
        LWDN = lwdn_flux.iloc[i]
        PRECP = prec_flux.iloc[i] / DT
        GH = gh_flux.iloc[i]

        if soil_hydro:
            # first layer soil moisture from flux data
            SH2O = SH2O_out.copy()
            SH2O[0] = swc_v[0][i]
            SMC = SH2O.copy()
        else:
            # variables from flux data
            # US-Ne3
            # SH2O = [swc_v[0, i], swc_v[1, i], swc_v[2, i], swc_v[3, i]]
            # SMC = SH2O
            # US-Me2
            # SH2O = [swc_v[0, i], np.mean([swc_v[1, i], swc_v[2, i]]), np.mean([swc_v[3, i], swc_v[4, i]]),
            #         np.mean([swc_v[6, i], swc_v[7, i]])]
            # US-Var
            # SH2O = [swc_v[0, i], (swc_v[1, i] + swc_v[2, i]) / 2, swc_v[2, i], 0.3]
            # SMC = SH2O
            # US-Wkg
            SH2O = [swc_v[0, i], (swc_v[0, i] + 2 * swc_v[1, i]) / 3, 0.2, 0.2]
            SMC = SH2O

        # # variables from full model
        LAI = lai_force[i]
        # GH = gh_model[i]
        # FWET = fwet[i]
        # CANLIQ = canliq[i]
        # SH2O = sh2o[i, :]
        # SMC = smc[i, :]

        # pre defined variables
        # LAI = 2.1
        # SH2O = [0.3, 0.3, 0.3, 0.3]
        # SMC = [0.3, 0.3, 0.3, 0.3]
        CANLIQ = 0
        FWET = 0

        SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
        TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
        BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
        QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
        PRECP_out, FSUN, PARSUN, PARSHA, \
        TV_l, FSTOMATA_l, SH2O_out, SMC_out, L12, L23, ROOTU, TR_P = ENERGY_DRIVER_PHM(config, soil_hydro, LAT, LON, TIME, DT, NSOIL,
                                                                                       ZSOIL, DZSNSO, WS, SFCTMP, RH, SFCPRS, SOLDN,
                                                                                       LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ, FWET)

        # output
        # print('WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP:', WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP)
        # print('SAV, SAG, FSA, FSR:', SAV, SAG, FSA, FSR)
        # print('TS, TV, TG, EAH, TAH:', TS, TV, TG, EAH, TAH)

        times.append(time)
        transpiration.append(FCTR)
        soil_evaporation.append(FGEV)
        canopy_evaporation.append(FCEV)
        rssun.append(RSSUN)
        rssha.append(RSSHA)
        psn.append(PSN)
        fsun.append(FSUN)
        fsh.append(FSH)
        sav.append(SAV)
        sag.append(SAG)
        fsa.append(FSA)
        fsr.append(FSR)
        fira.append(FIRA)
        apar.append(APAR)
        parsun.append(PARSUN)
        parsha.append(PARSHA)
        tr_p.append(TR_P)
        fstomata.append(FSTOMATA_l[-1])
        lai.append(LAI)
        swc_related_sim.append(np.concatenate([SH2O_out, [L12, L23], ROOTU]))

    # save results
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results = np.array(
        [times, transpiration, soil_evaporation, canopy_evaporation, rssun, rssha, psn, fsun, fsh, sav, sag, fsa, fsr, fira, apar,
         parsun, parsha, lai, tr_p, fstomata])
    np.save(results_path + 'standalone_flux' + results_suffix, results)
    # np.save(results_path + 'standalone_swc' + results_suffix, swc_related_sim)

    return


if __name__ == '__main__':
    main()
