import numpy as np

from noah_energy.phm.energy_constants import CPAIR, SB, VKC
from noah_energy.phm.flux_subroutines import SFCDIF1, RAGRB, ESAT, STOMATA, STOMATA_A, STOMATA_PHM, PHM_nested


# def VEGE_FLUX(UR, LWDN, SFCTMP, SFCPRS, RH, SH2O, SAV, PARSUN, PARSHA, BTRAN, VAI, LAISUN, LAISHA):
def VEGE_FLUX_PHM_nested(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET, LAISUN, LAISHA, CWP, HTOP, ZLVL, ZPD,
                         Z0M, FVEG, Z0MG, EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN, PARSHA, IGS, FOLN, CO2AIR, O2AIR, SFCPRS, RHSUR, DF,
                         DZSNSO, SAG, GH, UU, VV, EAH, TAH, TV, TG, NSOIL, ZSOIL, SMC):
    # some constants
    MPE = 1e-6
    NITERC = 50
    NITERG = 20

    # initialization variables that do not depend on stability iteration
    MOZSGN = 0
    H = 0
    HG = 0

    # convert grid-cell LAI to the fractional vegetated area (FVEG)
    VAIE = min(6., VAI / FVEG)
    LAISUNE = min(6., LAISUN / FVEG)
    LAISHAE = min(6., LAISHA / FVEG)

    # saturation vapor pressure at ground temperature
    T = TG - 273.16
    ESW, ESI, DESW, DESI = ESAT(T)
    if TG > 273.16:
        ESTG = ESW
    else:
        ESTG = ESI

    # canopy height, wind
    HCAN = HTOP
    UC = UR * np.log(HCAN / Z0M) / np.log(ZLVL / Z0M)
    # print('int UC:', UC)

    # prepare for longwave rad.
    AIR = -EMV * (1. + (1. - EMV) * (1. - EMG)) * LWDN - EMV * EMG * SB * TG ** 4
    CIR = (2. - EMV * (1. - EMG)) * EMV * SB

    # initialize
    MOZ = 0
    FV = 0
    FSTOMATA = 1

    # initialize to shut up
    FM = -9999
    FH = -9999
    FHG = -9999
    Z0H = -9999
    CAH = CM = RAHG = RAWG = RSSUN = RSSHA = PSNSUN = PSNSHA = PSN = TR = ESTV = RSSUN_nc = RSSHA_nc = PSNSUN_nc = \
        PSNSHA_nc = IRC = SHC = EVC = QSFC = SRCD = SOLPSI = VGPSIS = ZMS2G = -9999.1

    # begin stability iteration
    TV_l = []
    DTV_l = []
    FSTOMATA_l = []
    for ITER in range(NITERC):
        # print('ITER:', ITER)
        # print('flux iter EAIR, EAH, TV, RHSUR, EGH, TG:', EAIR, EAH, TV, RHSUR, RHSUR * ESTG, TG)
        Z0H = Z0M
        Z0HG = Z0MG

        # aerodyn resistances between heights zlvl and d+z0v
        MOZSGN, MOZ, FM, FH, CM, CH, FV = SFCDIF1(ITER, SFCTMP, RHOAIR, H, QAIR, ZLVL, ZPD, Z0M, Z0H, UR, MPE, MOZ, MOZSGN, FM,
                                                  FH, FV)

        # RAMC not used
        # RAMC = np.maximum(1., 1. / (CM * UR))
        RAHC = np.maximum(1., 1. / (CH * UR))
        RAWC = RAHC

        # aerodyn resistance between heights z0g and d+z0v, RAG, and leaf boundary layer resistance, RB
        MOZG, FHG, RAMG, RAHG, RAWG, RB = RAGRB(config, ITER, VAIE, RHOAIR, HG, TAH, ZPD, Z0MG, Z0HG, HCAN, UC, Z0H, FV, CWP, MPE, FHG)
        # print('int:', MOZ, FM, FH, FV, MOZG, FHG)

        # es and d(es)/dt evaluated at tv
        T = TV - 273.16
        ESW, ESI, DESW, DESI = ESAT(T)
        if TV > 273.16:
            ESTV = ESW
            DESTV = DESW
        else:
            ESTV = ESI
            DESTV = DESI

        # stomatal resistance, original model only calculates stomatal in the first iteration
        if ITER >= 0:
            # if ITER == 0:
            # print('EI, EAH:', ESTV, EAH)
            RSSUN, PSNSUN = STOMATA_PHM(config, MPE, PARSUN, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, RB, FSTOMATA)
            RSSHA, PSNSHA = STOMATA_PHM(config, MPE, PARSHA, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, RB, FSTOMATA)
            # RSSUN_nc, PSNSUN_nc = STOMATA_PHM(config, MPE, PARSUN, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB,
            #                               FSTOMATA)
            # RSSHA_nc, PSNSHA_nc = STOMATA_PHM(config, MPE, PARSHA, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB,
            #                               FSTOMATA)
            PSN = PSNSUN * LAISUNE + PSNSHA * LAISHAE

        # prepare for sensible heat flux above veg.
        CAH = 1. / RAHC
        CVH = 2. * VAIE / RB
        CGH = 1. / RAHG
        COND = CAH + CVH + CGH
        ATA = (SFCTMP * CAH + TG * CGH) / COND
        BTA = CVH / COND
        CSH = (1. - BTA) * RHOAIR * CPAIR * CVH

        # prepare for latent heat flux above veg.
        CAW = 1. / RAWC
        CEW = FWET * VAIE / RB
        CTW = (1. - FWET) * (LAISUNE / (RB + RSSUN) + LAISHAE / (RB + RSSHA))
        CGW = 1. / (RAWG + RSURF)
        COND = CAW + CEW + CTW + CGW
        AEA = (EAIR * CAW + ESTG * CGW * RHSUR) / COND
        # AEA = (EAIR * CAW + ESTG * CGW) / COND
        BEA = (CEW + CTW) / COND
        CEV = (1. - BEA) * CEW * RHOAIR * CPAIR / GAMMA
        CTR = (1. - BEA) * CTW * RHOAIR * CPAIR / GAMMA
        # print('flux iter RAWC RAWG + RSURF, RB, RSSUN:', RAWC, RAWG + RSURF, RB, RSSUN)

        # evaluate surface fluxes with current temperature and solve for dts
        TAH = ATA + BTA * TV
        EAH = AEA + BEA * ESTV

        IRC = FVEG * (AIR + CIR * TV ** 4)
        SHC = FVEG * RHOAIR * CPAIR * CVH * (TV - TAH)
        EVC = FVEG * RHOAIR * CPAIR * CEW * (ESTV - EAH) / GAMMA
        TR = FVEG * RHOAIR * CPAIR * CTW * (ESTV - EAH) / GAMMA
        EVC = np.minimum(CANLIQ * LATHEA / DT, EVC)

        B = SAV - IRC - SHC - EVC - TR
        A = FVEG * (4. * CIR * TV ** 3 + CSH + (CEV + CTR) * DESTV)
        DTV = B / A

        IRC = IRC + FVEG * 4. * CIR * TV ** 3 * DTV
        SHC = SHC + FVEG * CSH * DTV
        EVC = EVC + FVEG * CEV * DESTV * DTV
        TR = TR + FVEG * CTR * DESTV * DTV

        # plant hydraulic downregulation for next iteration
        FSTOMATA_OLD = FSTOMATA
        FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, ZMS2G = PHM_nested(config, NSOIL, ZSOIL, DZSNSO, LATHEA, TR, SMC, FSTOMATA)
        d_FSTOMATA = FSTOMATA - FSTOMATA_OLD

        # update vegetation surface temperature
        TV = TV + DTV

        # for computing M-O length in the next iteration
        H = RHOAIR * CPAIR * (TAH - SFCTMP) / RAHC
        HG = RHOAIR * CPAIR * (TG - TAH) / RAHG

        # consistent specific humidity from canopy air vapor pressure
        QSFC = (0.622 * EAH) / (SFCPRS - 0.378 * EAH)

        TV_l.append(TV)
        DTV_l.append(DTV)
        FSTOMATA_l.append(FSTOMATA)

        # end loop criteria
        # print('int DTV:', DTV)
        if abs(DTV) < 0.01 and abs(d_FSTOMATA) < 0.01 and ITER >= 5:
            # print('vege_flux veg break!, iter={}'.format(ITER))
            break
        if ITER == NITERC - 1:
            print('PHM nested vege_flux veg iteration not converging, DTV={}, d_FSTOMATA={}'.format(DTV, d_FSTOMATA))
    # print('canopy flux loop End, stability ITER={}\n'.format(ITER))

    # calculate root water extraction
    ROOTU = np.full(NSOIL, -9999.1)
    for IZ in range(NSOIL):
        ROOTU[IZ] = SRCD[IZ] * (SOLPSI[IZ] - VGPSIS - ZMS2G[IZ])

    # under-canopy fluxes and tg
    AIR = - EMG * (1. - EMV) * LWDN - EMG * EMV * SB * TV ** 4
    CIR = EMG * SB
    CSH = RHOAIR * CPAIR / RAHG
    CEV = RHOAIR * CPAIR / (GAMMA * (RAWG + RSURF))
    CGH = 2. * DF[0] / DZSNSO[0]

    # initialize to shut up
    IRG = SHG = EVG = -9999.1

    TG_l = []
    DTG_l = []
    for ITER in range(NITERG):
        T = TG - 273.15
        ESW, ESI, DESW, DESI = ESAT(T)
        if TG > 273.16:
            ESTG = ESW
            DESTG = DESW
        else:
            ESTG = ESI
            DESTG = DESI

        IRG = CIR * TG ** 4 + AIR
        SHG = CSH * (TG - TAH)
        EVG = CEV * (ESTG * RHSUR - EAH)
        # GH  = CGH * (TG         - STC(ISNOW+1))

        B = SAG - IRG - SHG - EVG - GH
        A = 4. * CIR * TG ** 3 + CSH + CEV * DESTG + CGH
        DTG = B / A

        IRG = IRG + 4. * CIR * TG ** 3 * DTG
        SHG = SHG + CSH * DTG
        EVG = EVG + CEV * DESTG * DTG
        # GH = GH + CGH * DTG
        TG = TG + DTG

        TG_l.append(TG)
        DTG_l.append(DTG)

        if np.abs(DTG) < 0.1 and ITER >= 4:
            # print('vege_flux ground break!, iter={}'.format(ITER))
            break
        if ITER == NITERG - 1:
            print('PHM nested vege_flux ground iteration not converging, DTG=', DTG)

    # wind stresses
    TAUXV = -RHOAIR * CM * UR * UU
    TAUYV = -RHOAIR * CM * UR * VV

    # 2 m temperature over vegetation ( corrected for low CQ2V values )
    CAH2 = FV * VKC / np.log((2. + Z0H) / Z0H)
    CQ2V = CAH2
    if CAH2 < 1E-5:
        T2MV = TAH
        Q2V = QSFC
    else:
        T2MV = TAH - (SHG + SHC / FVEG) / (RHOAIR * CPAIR) * 1. / CAH2
        Q2V = QSFC - ((EVC + TR) / FVEG + EVG) / (LATHEA * RHOAIR) * 1. / CQ2V

    # update CH for output
    CH = CAH

    return \
        RSSUN, RSSHA, PSNSUN, PSNSHA, PSN, TR, TV, TAH, TG, EAH, EAIR, ESTV, IRC, SHC, EVC, QSFC, CM, CH, TAUXV, TAUYV, IRG, \
        EVG, SHG, GH, T2MV, Q2V, CAH2, \
        RSSUN_nc, RSSHA_nc, PSNSUN_nc, PSNSHA_nc, \
        TV_l, DTV_l, TG_l, DTG_l, FSTOMATA_l, ROOTU


def VEGE_FLUX(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET, LAISUN, LAISHA, CWP, HTOP, ZLVL, ZPD, Z0M, FVEG,
              Z0MG, EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN, PARSHA, IGS, FOLN, CO2AIR, O2AIR, BTRAN, SFCPRS, RHSUR, DF, DZSNSO,
              SAG, GH, UU, VV, EAH, TAH, TV, TG):
    # some constants
    MPE = 1e-6
    NITERC = 50
    NITERG = 20

    # initialization variables that do not depend on stability iteration
    MOZSGN = 0
    H = 0
    HG = 0

    # convert grid-cell LAI to the fractional vegetated area (FVEG)
    VAIE = min(6., VAI / FVEG)
    LAISUNE = min(6., LAISUN / FVEG)
    LAISHAE = min(6., LAISHA / FVEG)

    # saturation vapor pressure at ground temperature
    T = TG - 273.16
    ESW, ESI, DESW, DESI = ESAT(T)
    if TG > 273.16:
        ESTG = ESW
    else:
        ESTG = ESI

    # canopy height, wind
    HCAN = HTOP
    UC = UR * np.log(HCAN / Z0M) / np.log(ZLVL / Z0M)
    # print('int UC:', UC)

    # prepare for longwave rad.
    AIR = -EMV * (1. + (1. - EMV) * (1. - EMG)) * LWDN - EMV * EMG * SB * TG ** 4
    CIR = (2. - EMV * (1. - EMG)) * EMV * SB

    # initialize
    MOZ = 0
    FV = 0

    # initialize to shut up
    FM = -9999
    FH = -9999
    FHG = -9999
    Z0H = -9999
    CAH = CM = RAHG = RAWG = RSSUN = RSSHA = PSNSUN = PSNSHA = PSN = TR = ESTV = RSSUN_nc = RSSHA_nc = PSNSUN_nc = \
        PSNSHA_nc = IRC = SHC = EVC = QSFC = -9999.1

    # begin stability iteration
    TV_l = []
    DTV_l = []
    for ITER in range(NITERC):
        # print('ITER:', ITER)
        # print('flux iter EAIR, EAH, TV, RHSUR, EGH, TG:', EAIR, EAH, TV, RHSUR, RHSUR * ESTG, TG)
        Z0H = Z0M
        Z0HG = Z0MG

        # aerodyn resistances between heights zlvl and d+z0v
        MOZSGN, MOZ, FM, FH, CM, CH, FV = SFCDIF1(ITER, SFCTMP, RHOAIR, H, QAIR, ZLVL, ZPD, Z0M, Z0H, UR, MPE, MOZ, MOZSGN, FM,
                                                  FH, FV)

        # RAMC not used
        # RAMC = np.maximum(1., 1. / (CM * UR))
        RAHC = np.maximum(1., 1. / (CH * UR))
        RAWC = RAHC

        # aerodyn resistance between heights z0g and d+z0v, RAG, and leaf boundary layer resistance, RB
        MOZG, FHG, RAMG, RAHG, RAWG, RB = RAGRB(config, ITER, VAIE, RHOAIR, HG, TAH, ZPD, Z0MG, Z0HG, HCAN, UC, Z0H, FV, CWP, MPE, FHG)
        # print('int:', MOZ, FM, FH, FV, MOZG, FHG)

        # es and d(es)/dt evaluated at tv
        T = TV - 273.16
        ESW, ESI, DESW, DESI = ESAT(T)
        if TV > 273.16:
            ESTV = ESW
            DESTV = DESW
        else:
            ESTV = ESI
            DESTV = DESI

        # stomatal resistance, original model only calculates stomatal in the first iteration
        if ITER >= 0:
            # if ITER == 0:
            # print('EI, EAH:', ESTV, EAH)
            RSSUN, PSNSUN = STOMATA(config, MPE, PARSUN, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
            RSSHA, PSNSHA = STOMATA(config, MPE, PARSHA, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
            RSSUN_nc, PSNSUN_nc = STOMATA(config, MPE, PARSUN, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
            RSSHA_nc, PSNSHA_nc = STOMATA(config, MPE, PARSHA, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
            PSN = PSNSUN * LAISUNE + PSNSHA * LAISHAE

        # prepare for sensible heat flux above veg.
        CAH = 1. / RAHC
        CVH = 2. * VAIE / RB
        CGH = 1. / RAHG
        COND = CAH + CVH + CGH
        ATA = (SFCTMP * CAH + TG * CGH) / COND
        BTA = CVH / COND
        CSH = (1. - BTA) * RHOAIR * CPAIR * CVH

        # prepare for latent heat flux above veg.
        CAW = 1. / RAWC
        CEW = FWET * VAIE / RB
        CTW = (1. - FWET) * (LAISUNE / (RB + RSSUN) + LAISHAE / (RB + RSSHA))
        CGW = 1. / (RAWG + RSURF)
        COND = CAW + CEW + CTW + CGW
        AEA = (EAIR * CAW + ESTG * CGW * RHSUR) / COND
        # AEA = (EAIR * CAW + ESTG * CGW) / COND
        BEA = (CEW + CTW) / COND
        CEV = (1. - BEA) * CEW * RHOAIR * CPAIR / GAMMA
        CTR = (1. - BEA) * CTW * RHOAIR * CPAIR / GAMMA
        # print('flux iter RAWC RAWG + RSURF, RB, RSSUN:', RAWC, RAWG + RSURF, RB, RSSUN)

        # evaluate surface fluxes with current temperature and solve for dts
        TAH = ATA + BTA * TV
        EAH = AEA + BEA * ESTV

        IRC = FVEG * (AIR + CIR * TV ** 4)
        SHC = FVEG * RHOAIR * CPAIR * CVH * (TV - TAH)
        EVC = FVEG * RHOAIR * CPAIR * CEW * (ESTV - EAH) / GAMMA
        TR = FVEG * RHOAIR * CPAIR * CTW * (ESTV - EAH) / GAMMA
        EVC = np.minimum(CANLIQ * LATHEA / DT, EVC)

        B = SAV - IRC - SHC - EVC - TR
        A = FVEG * (4. * CIR * TV ** 3 + CSH + (CEV + CTR) * DESTV)
        DTV = B / A

        IRC = IRC + FVEG * 4. * CIR * TV ** 3 * DTV
        SHC = SHC + FVEG * CSH * DTV
        EVC = EVC + FVEG * CEV * DESTV * DTV
        TR = TR + FVEG * CTR * DESTV * DTV

        # update vegetation surface temperature
        TV = TV + DTV

        # for computing M-O length in the next iteration
        H = RHOAIR * CPAIR * (TAH - SFCTMP) / RAHC
        HG = RHOAIR * CPAIR * (TG - TAH) / RAHG

        # consistent specific humidity from canopy air vapor pressure
        QSFC = (0.622 * EAH) / (SFCPRS - 0.378 * EAH)

        TV_l.append(TV)
        DTV_l.append(DTV)

        # end loop criteria
        # print('int DTV:', DTV)
        if np.abs(DTV) < 0.01 and ITER >= 5:
            # print('vege_flux veg break!, iter={}'.format(ITER))
            break
        if ITER == NITERC - 1:
            print('PHM vege_flux veg iteration not converging, DTV=', DTV)

    # print('canopy flux loop End, stability ITER={}\n'.format(ITER))

    # under-canopy fluxes and tg
    AIR = - EMG * (1. - EMV) * LWDN - EMG * EMV * SB * TV ** 4
    CIR = EMG * SB
    CSH = RHOAIR * CPAIR / RAHG
    CEV = RHOAIR * CPAIR / (GAMMA * (RAWG + RSURF))
    CGH = 2. * DF[0] / DZSNSO[0]

    # initialize to shut up
    IRG = SHG = EVG = -9999.1

    TG_l = []
    DTG_l = []
    for ITER in range(NITERG):
        T = TG - 273.15
        ESW, ESI, DESW, DESI = ESAT(T)
        if TG > 273.16:
            ESTG = ESW
            DESTG = DESW
        else:
            ESTG = ESI
            DESTG = DESI

        IRG = CIR * TG ** 4 + AIR
        SHG = CSH * (TG - TAH)
        EVG = CEV * (ESTG * RHSUR - EAH)
        # GH  = CGH * (TG         - STC(ISNOW+1))

        B = SAG - IRG - SHG - EVG - GH
        A = 4. * CIR * TG ** 3 + CSH + CEV * DESTG + CGH
        DTG = B / A

        IRG = IRG + 4. * CIR * TG ** 3 * DTG
        SHG = SHG + CSH * DTG
        EVG = EVG + CEV * DESTG * DTG
        # GH = GH + CGH * DTG
        TG = TG + DTG

        TG_l.append(TG)
        DTG_l.append(DTG)

        if np.abs(DTG) < 0.1 and ITER >= 4:
            # print('vege_flux ground break!, iter={}'.format(ITER))
            break
        if ITER == NITERG - 1:
            print('PHM vege_flux ground iteration not converging, DTG=', DTG)

    # wind stresses
    TAUXV = -RHOAIR * CM * UR * UU
    TAUYV = -RHOAIR * CM * UR * VV

    # 2m temperature over vegetation ( corrected for low CQ2V values )
    CAH2 = FV * VKC / np.log((2. + Z0H) / Z0H)
    CQ2V = CAH2
    if CAH2 < 1E-5:
        T2MV = TAH
        Q2V = QSFC
    else:
        T2MV = TAH - (SHG + SHC / FVEG) / (RHOAIR * CPAIR) * 1. / CAH2
        Q2V = QSFC - ((EVC + TR) / FVEG + EVG) / (LATHEA * RHOAIR) * 1. / CQ2V

    # update CH for output
    CH = CAH

    return \
        RSSUN, RSSHA, PSNSUN, PSNSHA, PSN, TR, TV, TAH, TG, EAH, EAIR, ESTV, IRC, SHC, EVC, QSFC, CM, CH, TAUXV, TAUYV, IRG, \
        EVG, SHG, GH, T2MV, Q2V, CAH2, \
        RSSUN_nc, RSSHA_nc, PSNSUN_nc, PSNSHA_nc, \
        TV_l, DTV_l, TG_l, DTG_l


def VEGE_FLUX_A(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET, LAISUN, LAISHA, CWP, HTOP, ZLVL, ZPD, Z0M, FVEG,
                Z0MG, EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN, PARSHA, IGS, FOLN, CO2AIR, O2AIR, BTRAN, SFCPRS, RHSUR, DF, DZSNSO,
                SAG, GH, UU, VV, EAH, TAH, TV, TG,
                RSSUN_P, RSSHA_P, TR_A):
    # some constants
    MPE = 1e-6
    NITERC = 50
    NITERG = 20

    # initialization variables that do not depend on stability iteration
    MOZSGN = 0
    H = 0
    HG = 0

    # convert grid-cell LAI to the fractional vegetated area (FVEG)
    VAIE = min(6., VAI / FVEG)
    LAISUNE = min(6., LAISUN / FVEG)
    LAISHAE = min(6., LAISHA / FVEG)

    # saturation vapor pressure at ground temperature
    T = TG - 273.16
    ESW, ESI, DESW, DESI = ESAT(T)
    if TG > 273.16:
        ESTG = ESW
    else:
        ESTG = ESI

    # canopy height, wind
    HCAN = HTOP
    UC = UR * np.log(HCAN / Z0M) / np.log(ZLVL / Z0M)
    # print('int UC:', UC)

    # prepare for longwave rad.
    AIR = -EMV * (1. + (1. - EMV) * (1. - EMG)) * LWDN - EMV * EMG * SB * TG ** 4
    CIR = (2. - EMV * (1. - EMG)) * EMV * SB

    # initialize
    MOZ = 0
    FV = 0

    # initialize to shut up
    FM = -9999
    FH = -9999
    FHG = -9999
    Z0H = -9999
    CAH = CM = RAHG = RAWG = RSSUN = RSSHA = PSN = TR = ESTV = IRC = SHC = EVC = QSFC = 'holder'
    RB = 'holder'

    # begin stability iteration
    TV_l = []
    DTV_l = []
    for ITER in range(NITERC):
        # print('ITER:', ITER)
        # print('flux iter EAIR, EAH, TV, RHSUR, EGH, TG:', EAIR, EAH, TV, RHSUR, RHSUR * ESTG, TG)
        Z0H = Z0M
        Z0HG = Z0MG

        # aerodyn resistances between heights zlvl and d+z0v
        MOZSGN, MOZ, FM, FH, CM, CH, FV = SFCDIF1(ITER, SFCTMP, RHOAIR, H, QAIR, ZLVL, ZPD, Z0M, Z0H, UR, MPE, MOZ, MOZSGN, FM,
                                                  FH, FV)

        # RAMC not used
        # RAMC = np.maximum(1., 1. / (CM * UR))
        RAHC = np.maximum(1., 1. / (CH * UR))
        RAWC = RAHC

        # aerodyn resistance between heights z0g and d+z0v, RAG, and leaf boundary layer resistance, RB
        MOZG, FHG, RAMG, RAHG, RAWG, RB = RAGRB(config, ITER, VAIE, RHOAIR, HG, TAH, ZPD, Z0MG, Z0HG, HCAN, UC, Z0H, FV, CWP, MPE, FHG)
        # print('int:', MOZ, FM, FH, FV, MOZG, FHG)

        # es and d(es)/dt evaluated at tv
        T = TV - 273.16
        ESW, ESI, DESW, DESI = ESAT(T)
        if TV > 273.16:
            ESTV = ESW
            DESTV = DESW
        else:
            ESTV = ESI
            DESTV = DESI

        # stomatal resistance (relocated to after the TV iteration)
        # # stomatal resistance, original model only calculates stomatal in the first iteration
        # if ITER >= 0:
        #     # if ITER == 0:
        #     # print('EI, EAH:', ESTV, EAH)
        #     RSSUN, PSNSUN = STOMATA(config, MPE, PARSUN, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
        #     RSSHA, PSNSHA = STOMATA(config, MPE, PARSHA, FOLN, TV, ESTV, EAH, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
        #     RSSUN_nc, PSNSUN_nc = STOMATA(config, MPE, PARSUN, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
        #     RSSHA_nc, PSNSHA_nc = STOMATA(config, MPE, PARSHA, FOLN, TV, ESTV, EAIR, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB)
        #     PSN = PSNSUN * LAISUNE + PSNSHA * LAISHAE

        # inversely determine stomatal resistance
        CTW_P = (1. - FWET) * (LAISUNE / (RB + RSSUN_P) + LAISHAE / (RB + RSSHA_P))
        CTW_A = TR_A / max(FVEG * RHOAIR * CPAIR * (ESTV - EAH), MPE) * GAMMA

        if CTW_A == 0:
            f_rs = 1E20
        else:
            f_rs = max(1., CTW_P / CTW_A)
        RSSUN = f_rs * (RB + RSSUN_P) - RB
        RSSHA = f_rs * (RB + RSSHA_P) - RB

        # prepare for sensible heat flux above veg.
        CAH = 1. / RAHC
        CVH = 2. * VAIE / RB
        CGH = 1. / RAHG
        COND = CAH + CVH + CGH
        ATA = (SFCTMP * CAH + TG * CGH) / COND
        BTA = CVH / COND
        CSH = (1. - BTA) * RHOAIR * CPAIR * CVH

        # prepare for latent heat flux above veg.
        CAW = 1. / RAWC
        CEW = FWET * VAIE / RB
        CTW = (1. - FWET) * (LAISUNE / (RB + RSSUN) + LAISHAE / (RB + RSSHA))
        CGW = 1. / (RAWG + RSURF)
        COND = CAW + CEW + CTW + CGW
        AEA = (EAIR * CAW + ESTG * CGW * RHSUR) / COND
        # AEA = (EAIR * CAW + ESTG * CGW) / COND
        BEA = (CEW + CTW) / COND
        CEV = (1. - BEA) * CEW * RHOAIR * CPAIR / GAMMA
        CTR = (1. - BEA) * CTW * RHOAIR * CPAIR / GAMMA
        # print('flux iter RAWC RAWG + RSURF, RB, RSSUN:', RAWC, RAWG + RSURF, RB, RSSUN)

        # evaluate surface fluxes with current temperature and solve for dts
        TAH = ATA + BTA * TV
        EAH = AEA + BEA * ESTV

        IRC = FVEG * (AIR + CIR * TV ** 4)
        SHC = FVEG * RHOAIR * CPAIR * CVH * (TV - TAH)
        EVC = FVEG * RHOAIR * CPAIR * CEW * (ESTV - EAH) / GAMMA
        TR = FVEG * RHOAIR * CPAIR * CTW * (ESTV - EAH) / GAMMA
        EVC = np.minimum(CANLIQ * LATHEA / DT, EVC)

        B = SAV - IRC - SHC - EVC - TR
        A = FVEG * (4. * CIR * TV ** 3 + CSH + (CEV + CTR) * DESTV)
        DTV = B / A

        IRC = IRC + FVEG * 4. * CIR * TV ** 3 * DTV
        SHC = SHC + FVEG * CSH * DTV
        EVC = EVC + FVEG * CEV * DESTV * DTV
        TR = TR + FVEG * CTR * DESTV * DTV

        # update vegetation surface temperature
        TV = TV + DTV

        # for computing M-O length in the next iteration
        H = RHOAIR * CPAIR * (TAH - SFCTMP) / RAHC
        HG = RHOAIR * CPAIR * (TG - TAH) / RAHG

        # consistent specific humidity from canopy air vapor pressure
        QSFC = (0.622 * EAH) / (SFCPRS - 0.378 * EAH)

        TV_l.append(TV)
        DTV_l.append(DTV)

        # end loop criteria
        # print('int DTV:', DTV)
        if np.abs(DTV) < 0.01 and ITER >= 5:
            # print('vege_flux veg break!, iter={}'.format(ITER))
            break
        if ITER == NITERC - 1:
            print('Beta vege_flux veg iteration not converging, DTV=', DTV)

    # photosynthesis with stomatal resistance as input
    PSNSUN = STOMATA_A(config, MPE, PARSUN, FOLN, TV, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB, RSSUN)
    PSNSHA = STOMATA_A(config, MPE, PARSHA, FOLN, TV, SFCTMP, SFCPRS, O2AIR, CO2AIR, IGS, BTRAN, RB, RSSHA)

    # under-canopy fluxes and tg
    AIR = - EMG * (1. - EMV) * LWDN - EMG * EMV * SB * TV ** 4
    CIR = EMG * SB
    CSH = RHOAIR * CPAIR / RAHG
    CEV = RHOAIR * CPAIR / (GAMMA * (RAWG + RSURF))
    CGH = 2. * DF[0] / DZSNSO[0]

    # initialize to shut up
    IRG = SHG = EVG = -9999.1

    TG_l = []
    DTG_l = []
    for ITER in range(NITERG):
        T = TG - 273.15
        ESW, ESI, DESW, DESI = ESAT(T)
        if TG > 273.16:
            ESTG = ESW
            DESTG = DESW
        else:
            ESTG = ESI
            DESTG = DESI

        IRG = CIR * TG ** 4 + AIR
        SHG = CSH * (TG - TAH)
        EVG = CEV * (ESTG * RHSUR - EAH)
        # GH  = CGH * (TG         - STC(ISNOW+1))

        B = SAG - IRG - SHG - EVG - GH
        A = 4. * CIR * TG ** 3 + CSH + CEV * DESTG + CGH
        DTG = B / A

        IRG = IRG + 4. * CIR * TG ** 3 * DTG
        SHG = SHG + CSH * DTG
        EVG = EVG + CEV * DESTG * DTG
        # GH = GH + CGH * DTG
        TG = TG + DTG

        TG_l.append(TG)
        DTG_l.append(DTG)

        if np.abs(DTG) < 0.1 and ITER >= 4:
            # print('vege_flux ground break!, iter={}'.format(ITER))
            break
        if ITER == NITERG - 1:
            print('Beta vege_flux ground iteration not converging, DTG=', DTG)

    # wind stresses
    TAUXV = -RHOAIR * CM * UR * UU
    TAUYV = -RHOAIR * CM * UR * VV

    # 2m temperature over vegetation ( corrected for low CQ2V values )
    CAH2 = FV * VKC / np.log((2. + Z0H) / Z0H)
    CQ2V = CAH2
    if CAH2 < 1E-5:
        T2MV = TAH
        Q2V = QSFC
    else:
        T2MV = TAH - (SHG + SHC / FVEG) / (RHOAIR * CPAIR) * 1. / CAH2
        Q2V = QSFC - ((EVC + TR) / FVEG + EVG) / (LATHEA * RHOAIR) * 1. / CQ2V

    # update CH for output
    CH = CAH

    return \
        RSSUN, RSSHA, PSNSUN, PSNSHA, PSN, TR, TV, TAH, TG, EAH, EAIR, ESTV, IRC, SHC, EVC, QSFC, CM, CH, TAUXV, TAUYV, IRG, \
        EVG, SHG, GH, T2MV, Q2V, CAH2, \
        TV_l, DTV_l, TG_l, DTG_l
