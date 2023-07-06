import numpy as np


def SOIL_HYDROLOGY(config, DT, NSOIL, DZSNSO, SH2O, ROOTU, FCTR, LATHEA):
    # NSOIL must be 2
    if NSOIL != 2:
        raise ValueError('NSOIL must be 2 to use simple soil hydrology model!')

    # config parms
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    PSISAT = float(config['soil']['PSISAT'])
    PSISFC = float(config['soil']['PSISFC'])
    DKSAT = float(config['soil']['DKSAT'])

    # soil hydraulic properties
    WCND = np.full(2, -9999.1)
    WDF = np.full(2, -9999.1)
    SOLPSI = np.full(2, -9999.1)
    for IZ in range(2):
        WCND[IZ], WDF[IZ] = WDFCND1(config, SH2O[IZ])
        SOLPSI[IZ] = max(-PSISAT * (min(SH2O[IZ], SMCMAX) / SMCMAX) ** (-BEXP), -500)

    L12 = 2 * (SOLPSI[0] - SOLPSI[1] + (DZSNSO[0] + DZSNSO[1]) / 2) / (DZSNSO[0] / WCND[0] + DZSNSO[1] / WCND[1])
    if SOLPSI[1] > -PSISFC:
        L23 = WCND[1]
    else:
        L23 = 0
    SH2O[1] = SH2O[1] + (L12 - L23 - ROOTU[1]) * DT / DZSNSO[1]
    # print('L12:', L12, 'L23:', L23, 'ROOTU:', ROOTU, 'ETRAN:', FCTR/LATHEA/1000)

    return SH2O, L12, L23


def WDFCND1(config, SMC):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    DWSAT = float(config['soil']['DWSAT'])
    DKSAT = float(config['soil']['DKSAT'])

    # soil water diffusivity
    FACTR = max(0.01, SMC / SMCMAX)
    EXPON = BEXP + 2.0
    WDF = DWSAT * FACTR ** EXPON

    # hydraulic conductivity
    EXPON = 2.0 * BEXP + 3.0
    WCND = DKSAT * FACTR ** EXPON

    return WCND, WDF
