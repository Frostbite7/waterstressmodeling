import numpy as np

from noah_energy.phm.energy_constants import CWAT, CPAIR, CICE


def THERMOPROP(config, NSOIL, DZSNSO, DT, SMC, SH2O):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    CSOIL = float(config['soil']['CSOIL'])

    SICE = np.full(4, -9999, 'float64')
    HCPCT = np.full(4, -9999, 'float64')
    DF = np.full(4, -9999.1)
    for IZ in range(NSOIL):
        SICE[IZ] = SMC[IZ] - SH2O[IZ]
        HCPCT[IZ] = SH2O[IZ] * CWAT + (1.0 - SMCMAX) * CSOIL + (SMCMAX - SMC[IZ]) * CPAIR + SICE[IZ] * CICE
        DF[IZ] = TDFCND(config, SMC[IZ], SH2O[IZ])
        # print(DF[IZ])

    # combine a temporary variable used for melting/freezing of snow and frozen soil
    FACT = np.full(4, -9999, 'float64')
    for IZ in range(NSOIL):
        FACT[IZ] = DT / (HCPCT[IZ] * DZSNSO[IZ])

    return DF, HCPCT, FACT


def TDFCND(config, SMC, SH2O):
    # config parameters
    QUARTZ = float(config['soil']['QUARTZ'])
    SMCMAX = float(config['soil']['SMCMAX'])

    # hard coded parameters
    THKO = 2.0
    THKQTZ = 7.7
    THKW = 0.57
    TKICE = 2.2

    # SATURATION RATIO:
    SATRATIO = SMC / SMCMAX

    # solids conductivity
    THKS = (THKQTZ ** QUARTZ) * (THKO ** (1. - QUARTZ))

    # UNFROZEN FRACTION
    XUNFROZ = SH2O / SMC
    # UNFROZEN VOLUME FOR SATURATION
    XU = XUNFROZ * SMCMAX
    # SATURATED THERMAL CONDUCTIVITY
    THKSAT = THKS ** (1. - SMCMAX) * TKICE ** (SMCMAX - XU) * THKW ** XU

    # DRY DENSITY IN KG/M3
    GAMMD = (1. - SMCMAX) * 2700.
    THKDRY = (0.135 * GAMMD + 64.7) / (2700. - 0.947 * GAMMD)

    # Frozen
    if (SH2O + 0.0005) < SMC:
        AKE = SATRATIO
    else:
        if SATRATIO > 0.1:
            AKE = np.log10(SATRATIO) + 1.0
        else:
            AKE = 0.0

    DF = AKE * (THKSAT - THKDRY) + THKDRY
    # print(DF, THKSAT, THKDRY, AKE)

    return DF
