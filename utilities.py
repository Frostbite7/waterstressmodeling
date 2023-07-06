from pysolar.solar import get_altitude
import numpy as np
from noah_energy.phm.energy_constants import RAIR
from noah_energy.phm.flux_subroutines import ESAT


def CALCULATE_COSZ(lat, lon, utc):
    altitude = get_altitude(lat, lon, utc)
    zenith = 90 - altitude
    return np.cos(zenith / 180 * np.pi)


def ATM(SFCPRS, SFCTMP, RH, SOLDN, COSZ):
    # mixing ratio and specific humidity
    ESW, ESI, DESW, DESI = ESAT(SFCTMP - 273.16)
    if SFCTMP > 273.16:
        SATVP = ESW
    else:
        SATVP = ESI
    EAIR = SATVP * RH / 100
    QAIR = 0.622 * EAIR / (SFCPRS - (1. - 0.622) * EAIR)
    Q2 = QAIR / (1 - QAIR)
    # print('\nATM SATVP, EAIR, RH:', SATVP, EAIR, RH)

    RHOAIR = (SFCPRS - 0.378 * EAIR) / (RAIR * SFCTMP)

    if COSZ <= 0.:
        SWDOWN = 0.
    else:
        SWDOWN = SOLDN

    # partition solar radiation
    SOLAD = np.array([SWDOWN * 0.7 * 0.5, SWDOWN * 0.7 * 0.5])
    SOLAI = np.array([SWDOWN * 0.3 * 0.5, SWDOWN * 0.3 * 0.5])

    return Q2, QAIR, EAIR, SOLAD, SOLAI, RHOAIR, SWDOWN
