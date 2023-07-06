from datetime import datetime
from datetime import timedelta
from datetime import timezone
import numpy as np
import pandas as pd
from noah_energy.phm.energy_driver import ENERGY_DRIVER_PHM, SOIL_LAYER_CONFIG


def read_forcing(start, end, flux_path_full, amx_path_full):
    # read flux forcing data
    print('subroutine reading data:')
    df = pd.read_csv(flux_path_full)

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
          np.sum(rh_flux == -9999),
          np.sum(prs_flux == -9999), np.sum(soldn_flux == -9999), np.sum(lwdn_flux == -9999), np.sum(prec_flux == -9999),
          np.sum(gh_flux == -9999))

    # gap filling
    ws_flux = ws_flux.replace(-9999, np.nan).interpolate()
    temp_flux = temp_flux.replace(-9999, np.nan).interpolate()
    rh_flux = rh_flux.replace(-9999, np.nan).interpolate()
    prs_flux = prs_flux.replace(-9999, np.nan).interpolate()
    soldn_flux = soldn_flux.replace(-9999, np.nan).interpolate()
    lwdn_flux = lwdn_flux.replace(-9999, np.nan).interpolate()
    prec_flux = prec_flux.replace(-9999, np.nan).interpolate()
    gh_flux = gh_flux.replace(-9999, np.nan).interpolate()

    # read SWC data
    dfa = pd.read_csv(amx_path_full, skiprows=2)
    swc_v = []
    for depth in range(8):
        swc = dfa[(dfa.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_1_{}_{}'.format(
            depth + 1, 1)]
        print('SWC Gaps! SWC_1_{}_1:'.format(depth), np.sum(swc == -9999))
        swc_v.append(swc.replace(-9999, np.nan).interpolate().values / 100)
    # swc_flux = np.array([swc_v[0], np.mean([swc_v[1], swc_v[2]], 0), np.mean([swc_v[3], swc_v[4]], 0), np.mean([swc_v[6], swc_v[7]], 0)])
    swc_flux = np.array([swc_v[0], np.mean([swc_v[1], swc_v[2]], 0), np.mean([swc_v[3]], 0), np.mean([swc_v[6], swc_v[7]], 0)])

    flux_forcing = [time_flux, ws_flux, temp_flux, rh_flux, prs_flux, soldn_flux, lwdn_flux, prec_flux, gh_flux, swc_flux]
    return flux_forcing


def run_model_time_series(config, flux_forcing, LAT, LON, utc_offset, time_res):
    # print('\nsubroutin run model:')

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

    # read and config forcing
    [time_flux, ws_flux, temp_flux, rh_flux, prs_flux, soldn_flux, lwdn_flux, prec_flux, gh_flux, swc_flux] = flux_forcing
    n_records = time_flux.shape[0]

    # model config
    DT, NSOIL, ZSOIL, DZSNSO = SOIL_LAYER_CONFIG(config)

    # initialize state variables
    SH2O_out = np.full(NSOIL, swc_flux[0][0], dtype=float)
    SH2O_out[1] = 0.3

    for i in range(n_records):
        time = str(time_flux.iloc[i])
        TIME = datetime(int(time[:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12]),
                        tzinfo=timezone.utc) + timedelta(hours=-utc_offset)
        if i % 240 == 0:
            print(TIME)

        # forcing variables from flux data
        WS = ws_flux.iloc[i]
        SFCTMP = temp_flux.iloc[i] + 273.15
        RH = rh_flux.iloc[i]
        SFCPRS = prs_flux.iloc[i] * 1000
        SOLDN = soldn_flux.iloc[i]
        LWDN = lwdn_flux.iloc[i]
        PRECP = prec_flux.iloc[i] / time_res
        GH = gh_flux.iloc[i]

        # variables from flux data
        # SH2O = swc_flux[:, i]
        # SMC = SH2O

        # first layer soil moisture from flux data
        SH2O = SH2O_out.copy()
        SH2O[0] = swc_flux[0][i]
        SMC = SH2O.copy()

        # pre defined variables
        LAI = 2.1
        CANLIQ = 0
        FWET = 0

        # run model for a single step
        SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
        TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
        BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
        QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
        PRECP_out, FSUN, PARSUN, PARSHA, \
        TV_l, FSTOMATA_l, SH2O_out, SMC_out, L12, L23, ROOTU = ENERGY_DRIVER_PHM(config, LAT, LON, TIME, DT, NSOIL, ZSOIL, DZSNSO, WS,
                                                                                 SFCTMP, RH, SFCPRS, SOLDN,
                                                                                 LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ, FWET)

        # record results
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
        fstomata.append(FSTOMATA_l[-1])
        swc_related_sim.append(np.concatenate([SH2O_out, [L12, L23], ROOTU]))

    results = [times, transpiration, soil_evaporation, canopy_evaporation, rssun, rssha, psn, fsun, fsh, sav, sag, fsa, fsr, fira, apar,
               parsun, parsha, fstomata]
    return np.array(results), np.array(swc_related_sim)
