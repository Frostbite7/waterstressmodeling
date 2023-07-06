import numpy as np
from noah_energy.phm.energy_constants import SB, CPAIR, VKC
from noah_energy.phm.flux_subroutines import SFCDIF1, ESAT


def BARE_FLUX(SAG, LWDN, UR, UU, VV, SFCTMP, QAIR, EAIR, RHOAIR, DZSNSO, ZLVL, ZPD, Z0M, EMG, DF, RSURF, LATHEA, GAMMA,
              RHSUR, TGB, GHB, PSFC):
    # configurations
    NITERB = 50

    # initialization variables that do not depend on stability iteration
    MPE = 1E-6
    MOZSGN = 0
    H = 0.
    # FV = 0.1

    CIR = EMG * SB
    CGH = 2. * DF[0] / DZSNSO[0]

    # initialize
    MOZ = 0
    FV = 0

    # initialize to shut up
    FM = -9999.1
    FH = -9999.1
    Z0H = -9999.1
    QSFC = CM = EHB = SHB = EVB = IRB = -9999.1

    # begin stability iteration
    TGB_l = []
    DTGB_l = []
    for ITER in range(NITERB):
        Z0H = Z0M

        MOZSGN, MOZ, FM, FH, CM, CH, FV = SFCDIF1(ITER, SFCTMP, RHOAIR, H, QAIR, ZLVL, ZPD, Z0M, Z0H, UR, MPE, MOZ, MOZSGN, FM,
                                                  FH, FV)

        # RAMB = max(1., 1. / (CM * UR))
        RAHB = max(1., 1. / (CH * UR))
        RAWB = RAHB

        # variables for diagnostics
        # EMB = 1. / RAMB
        EHB = 1. / RAHB

        # es and d(es)/dt evaluated at tg
        T = TGB - 273.15
        ESATW, ESATI, DSATW, DSATI = ESAT(T)
        if T > 0.:
            ESTG = ESATW
            DESTG = DSATW
        else:
            ESTG = ESATI
            DESTG = DSATI

        # initialize conductances
        CSH = RHOAIR * CPAIR / RAHB
        CEV = RHOAIR * CPAIR / GAMMA / (RSURF + RAWB)

        # surface fluxes and dtg
        IRB = CIR * TGB ** 4 - EMG * LWDN
        SHB = CSH * (TGB - SFCTMP)
        EVB = CEV * (ESTG * RHSUR - EAIR)
        # GHB = CGH * (TGB - STC[0])

        B = SAG - IRB - SHB - EVB - GHB
        A = 4. * CIR * TGB ** 3 + CSH + CEV * DESTG + CGH
        DTG = B / A

        IRB = IRB + 4. * CIR * TGB ** 3 * DTG
        SHB = SHB + CSH * DTG
        EVB = EVB + CEV * DESTG * DTG
        # GHB = GHB + CGH * DTG

        # update ground surface temperature
        TGB = TGB + DTG

        # for M-O length
        H = CSH * (TGB - SFCTMP)

        T = TGB - 273.15
        ESATW, ESATI, DSATW, DSATI = ESAT(T)
        if T > 0:
            ESTG = ESATW
        else:
            ESTG = ESATI
        QSFC = 0.622 * (ESTG * RHSUR) / (PSFC - 0.378 * (ESTG * RHSUR))

        # QFX = (QSFC - QAIR) * CEV * GAMMA / CPAIR
        # diagnostics
        TGB_l.append(TGB)
        DTGB_l.append(DTG)

        if np.abs(DTG) < 0.1 and ITER >= 4:
            # print('bare_flux break!, iter={}'.format(ITER))
            break
        if ITER == NITERB - 1:
            print('bare_flux iteration not converging, DTGB=', DTG)

    # wind stresses
    TAUXB = -RHOAIR * CM * UR * UU
    TAUYB = -RHOAIR * CM * UR * VV

    # errors in original equation corrected.
    # 2m air temperature
    EHB2 = FV * VKC / np.log((2. + Z0H) / Z0H)
    CQ2B = EHB2

    if EHB2 < 1.E-5:
        T2MB = TGB
        Q2B = QSFC
    else:
        T2MB = TGB - SHB / (RHOAIR * CPAIR) * 1. / EHB2
        Q2B = QSFC - EVB / (LATHEA * RHOAIR) * (1. / CQ2B + RSURF)

    # update CH
    CH = EHB

    return TGB, CM, CH, TAUXB, TAUYB, IRB, SHB, EVB, GHB, T2MB, QSFC, Q2B, EHB2
    # return TGB, CM, CH, TAUXB, TAUYB, IRB, SHB, EVB, GHB, T2MB, QSFC, Q2B, EHB2, TGB_l, DTGB_l
