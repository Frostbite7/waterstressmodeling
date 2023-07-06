import numpy as np
from noah_energy.phm.energy import ENERGY
from noah_energy.phm.utilities import CALCULATE_COSZ, ATM
from noah_energy.phm.soil_hydro import SOIL_HYDROLOGY

# global configurations
ISC = 4

# parameters that may change over time but not for now
IGS = 1
FOLN = 1
FVEG = 0.98


def ENERGY_DRIVER_PHM(config, soil_hydro, LAT, LON, TIME, DT, NSOIL, ZSOIL, DZSNSO, WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP, LAI,
                      SH2O, SMC, GH, CANLIQ, FWET):
    # parameters
    HTOP = float(config['vege_rad']['HVT'])
    # NROOT = int(config['soil']['NROOT'])
    ZREF = float(config['vege_flux']['ZREF'])

    # solar position and atmosheric condition, phenology
    COSZ = CALCULATE_COSZ(LAT, LON, TIME)
    Q2, QAIR, EAIR, SOLAD, SOLAI, RHOAIR, SWDOWN = ATM(SFCPRS, SFCTMP, RH, SOLDN, COSZ)
    UU = WS
    VV = 0
    ELAI = LAI
    ESAI = 0

    # set some variables
    CO2AIR = 355e-6 * SFCPRS
    O2AIR = 0.209 * SFCPRS

    # initlalize some flux variables
    TV = SFCTMP
    EAH = EAIR
    TAH = SFCTMP
    PSFC = SFCPRS
    TG = SFCTMP

    # call energy
    SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
    TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
    BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
    QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, FSUN, PARSUN, PARSHA, \
    TV_l, FSTOMATA_l, ROOTU, TR_P = ENERGY(config, ISC, NSOIL, DT, RHOAIR, SFCPRS, QAIR, SFCTMP,
                                     LWDN, UU, VV, ZREF, CO2AIR, O2AIR, SOLAD, SOLAI, COSZ,
                                     IGS, EAIR, HTOP, ZSOIL, ELAI, ESAI, FWET, FOLN, FVEG, DZSNSO,
                                     CANLIQ, TV, TG, EAH, TAH, SH2O, SMC, PSFC, GH)

    # run soil hydrology model
    if soil_hydro:
        SH2O, L12, L23 = SOIL_HYDROLOGY(config, DT, NSOIL, DZSNSO, SH2O, ROOTU, FCTR, LATHEA)
        SMC = SH2O
    else:
        L12 = None
        L23 = None

    return \
        SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
        TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
        BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
        QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
        PRECP, FSUN, PARSUN, PARSHA, \
        TV_l, FSTOMATA_l, SH2O, SMC, L12, L23, ROOTU, TR_P


def SOIL_LAYER_CONFIG(config):
    DT = float(config['configuration']['DT'])
    NSOIL = int(config['configuration']['NSOIL'])
    ZSOIL = np.array([float(idx) for idx in config['configuration']['ZSOIL'].split(',')])
    DZSNSO = np.array([float(idx) for idx in config['configuration']['DZSNSO'].split(',')])

    return DT, NSOIL, ZSOIL, DZSNSO
