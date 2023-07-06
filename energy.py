import numpy as np

from noah_energy.phm.bare_flux import BARE_FLUX
from noah_energy.phm.energy_constants import GRAV, RW, HVAP, HSUB, CPAIR, SB
from noah_energy.phm.eparm import EPARM
from noah_energy.phm.radiation import RADIATION
from noah_energy.phm.thermoprop import THERMOPROP
from noah_energy.phm.vege_flux import VEGE_FLUX, VEGE_FLUX_A
from noah_energy.phm.flux_subroutines import PHM


def ENERGY(config, ISC, NSOIL, DT, RHOAIR, SFCPRS, QAIR, SFCTMP, LWDN, UU, VV, ZREF, CO2AIR, O2AIR, SOLAD, SOLAI, COSZ,
           IGS, EAIR, HTOP, ZSOIL, ELAI, ESAI, FWET, FOLN, FVEG, DZSNSO, CANLIQ, TV, TG, EAH, TAH, SH2O, SMC, PSFC, GH):
    # config parameters
    Z0MVT = float(config['vege_flux']['Z0MVT'])
    CWPVT = float(config['vege_flux']['CWPVT'])
    EG = float(config['radiation']['EG'])
    SMCWLT = float(config['soil']['SMCWLT'])
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    PSISAT = float(config['soil']['PSISAT'])
    Z0 = float(config['soil']['Z0'])

    # initialize fluxes from veg. fraction
    PSNSUN = 'holder'
    PSNSHA = 'holder'
    RSSUN = 'holder'
    RSSHA = 'holder'

    # wind speed at reference height: ur >= 1
    UR = max(np.sqrt(UU ** 2. + VV ** 2.), 1.)

    # vegetated or non-vegetated
    VAI = ELAI + ESAI
    VEG = 0
    if VAI > 0.:
        VEG = 1

    # ground roughness length
    Z0MG = Z0

    # roughness length and displacement height
    ZPDG = 0
    if VEG:
        Z0M = Z0MVT
        ZPD = 0.65 * HTOP
    else:
        Z0M = Z0MG
        ZPD = ZPDG

    # changed from original Noah-MP. The original one is not the case of flux tower height
    ZLVL = ZREF
    # below the original calculation
    # ZLVL = max(ZPD, HTOP) + ZREF
    if ZPDG >= ZLVL:
        ZLVL = ZPDG + ZREF

    # canopy wind absorption coeffcient
    CWP = CWPVT

    # Thermal properties of soil, snow, lake, and frozen soil
    DF, HCPCT, FACT = THERMOPROP(config, NSOIL, DZSNSO, DT, SMC, SH2O)

    # Solar radiation: absorbed & reflected by the ground and canopy
    FSUN, LAISUN, LAISHA, PARSUN, PARSHA, SAV, SAG, FSA, FSR, FSRV, FSRG, BGAP, WGAP, GAP = RADIATION(config, ISC, COSZ, ELAI,
                                                                                                      ESAI, SMC, SOLAD, SOLAI,
                                                                                                      FVEG)
    # print('LAISUN, LAISHA, PARSUN, PARSHA:', LAISUN, LAISHA, PARSUN, PARSHA)

    # vegetation and ground emissivity
    EMV = 1. - np.exp(-(ELAI + ESAI) / 1.0)
    EMG = EG

    # soil surface resistance for ground evap.
    # RSURF based on Sakaguchi and Zeng, 2009
    # taking the "residual water content" to be the wilting point,
    # and correcting the exponent on the D term (typo in SZ09 ?)
    L_RSURF = (-ZSOIL[0]) * (np.exp((1.0 - min(1.0, SH2O[0] / SMCMAX)) ** 5) - 1.0) / (2.71828 - 1.0)
    D_RSURF = 2.2E-5 * SMCMAX * SMCMAX * (1.0 - SMCWLT / SMCMAX) ** (2.0 + 3.0 / BEXP)
    RSURF = L_RSURF / D_RSURF

    if SH2O[0] < 0.01:
        RSURF = 1.E6
    PSI = -PSISAT * (max(0.01, SH2O[0]) / SMCMAX) ** (-BEXP)
    RHSUR = np.exp(PSI * GRAV / (RW * TG))

    # set psychrometric constant
    if SFCTMP > 273.15:
        LATHEA = HVAP
    else:
        LATHEA = HSUB
    GAMMA = CPAIR * SFCPRS / (0.622 * LATHEA)

    # Surface temperatures of the ground and canopy and energy fluxes

    # initialize to shut up
    ROOTU = TV_l = FSTOMATA_l = Q2V = T2MV = TAUXV = TAUYV = IRG = IRC = SHG = SHC = EVG = GHV = EVC = TR = TGV = CMV = CHV = 'holder'

    if VEG:
        TGV = TG

        # calculate well watered transpiration
        BTRAN_P = 1
        RSSUN_P, RSSHA_P, PSNSUN_P, PSNSHA_P, PSN_P, TR_P, TV_P, TAH_P, TGV_P, EAH_P, EAIR_P, ESTV_P, IRC_P, SHC_P, EVC_P, QSFC_P, CMV_P, CHV_P, TAUXV_P, TAUYV_P, IRG_P, \
        EVG_P, SHG_P, GHV_P, T2MV_P, Q2V_P, CAH2_P, \
        RSSUN_nc_P, RSSHA_nc_P, PSNSUN_nc_P, PSNSHA_nc_P, \
        TV_l_P, DTV_l_P, TG_l_P, DTG_l_P = VEGE_FLUX(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET,
                                                     LAISUN, LAISHA, CWP, HTOP, ZLVL, ZPD, Z0M, FVEG, Z0MG,
                                                     EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN, PARSHA, IGS, FOLN, CO2AIR,
                                                     O2AIR, BTRAN_P, SFCPRS, RHSUR, DF, DZSNSO, SAG,
                                                     GH, UU, VV, EAH, TAH, TV, TGV)
        # print('SFCTMP, TV, ESTV, EAH, EAIR:', SFCTMP, TV, ESTV, EAH, EAIR)

        # calculate hydraulic downregulation factor
        FCTR_d, FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, ROOTU, FCTR_d_l = PHM(config, NSOIL, ZSOIL, DZSNSO, TR_P, SMC)
        TR_A = FCTR_d

        # calculate actual transpiration by inverse calculation
        RSSUN, RSSHA, PSNSUN, PSNSHA, PSN, TR, TV, TAH, TGV, EAH, EAIR, ESTV, IRC, SHC, EVC, QSFC, CMV, CHV, TAUXV, TAUYV, IRG, \
        EVG, SHG, GHV, T2MV, Q2V, CAH2, \
        TV_l, DTV_l, TG_l, DTG_l = VEGE_FLUX_A(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET, LAISUN,
                                               LAISHA, CWP, HTOP, ZLVL, ZPD, Z0M, FVEG, Z0MG, EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN,
                                               PARSHA, IGS, FOLN, CO2AIR, O2AIR, BTRAN_P, SFCPRS, RHSUR, DF, DZSNSO,
                                               SAG, GH, UU, VV, EAH, TAH, TV, TG, RSSUN_P, RSSHA_P, TR_A)

        # old: applying hydraulic downregulation factor to stomata
        # TGV = TG
        # RSSUN, RSSHA, PSNSUN, PSNSHA, PSN, TR, TV, TAH, TGV, EAH, EAIR, ESTV, IRC, SHC, EVC, QSFC, CMV, CHV, TAUXV, TAUYV, IRG, \
        # EVG, SHG, GHV, T2MV, Q2V, CAH2, \
        # RSSUN_nc, RSSHA_nc, PSNSUN_nc, PSNSHA_nc, \
        # TV_l, DTV_l, TG_l, DTG_l, FSTOMATA_l, ROOTU = VEGE_FLUX(config, DT, SAV, LWDN, UR, SFCTMP, QAIR, EAIR, RHOAIR, VAI, GAMMA, FWET,
        #                                                         LAISUN,
        #                                                         LAISHA, CWP, HTOP, ZLVL, ZPD, Z0M, FVEG, Z0MG,
        #                                                         EMV, EMG, CANLIQ, RSURF, LATHEA, PARSUN, PARSHA, IGS, FOLN, CO2AIR,
        #                                                         O2AIR, SFCPRS, RHSUR, DF, DZSNSO, SAG,
        #                                                         GH, UU, VV, EAH, TAH, TV, TGV, NSOIL, ZSOIL, SMC)

        # print('SFCTMP, TV, ESTV, EAH, EAIR:', SFCTMP, TV, ESTV, EAH, EAIR)

    GHB = GH
    TGB = TG
    TGB, CMB, CHB, TAUXB, TAUYB, IRB, SHB, EVB, GHB, T2MB, QSFC, Q2B, EHB2 = BARE_FLUX(SAG, LWDN, UR, UU, VV, SFCTMP, QAIR, EAIR,
                                                                                       RHOAIR, DZSNSO, ZLVL, ZPD, Z0M, EMG, DF,
                                                                                       RSURF, LATHEA, GAMMA,
                                                                                       RHSUR, TGB, GHB, PSFC)

    if VEG:
        TAUX = FVEG * TAUXV + (1.0 - FVEG) * TAUXB
        TAUY = FVEG * TAUYV + (1.0 - FVEG) * TAUYB
        FIRA = FVEG * IRG + (1.0 - FVEG) * IRB + IRC
        FSH = FVEG * SHG + (1.0 - FVEG) * SHB + SHC
        FGEV = FVEG * EVG + (1.0 - FVEG) * EVB
        SSOIL = FVEG * GHV + (1.0 - FVEG) * GHB
        FCEV = EVC
        FCTR = TR
        TG = FVEG * TGV + (1.0 - FVEG) * TGB
        T2M = FVEG * T2MV + (1.0 - FVEG) * T2MB
        TS = FVEG * TV + (1.0 - FVEG) * TGB
        CM = FVEG * CMV + (1.0 - FVEG) * CMB
        CH = FVEG * CHV + (1.0 - FVEG) * CHB
        Q1 = FVEG * (EAH * 0.622 / (SFCPRS - 0.378 * EAH)) + (1.0 - FVEG) * QSFC
        Q2E = FVEG * Q2V + (1.0 - FVEG) * Q2B
    else:
        TAUX = TAUXB
        TAUY = TAUYB
        FIRA = IRB
        FSH = SHB
        FGEV = EVB
        SSOIL = GHB
        TG = TGB
        T2M = T2MB
        FCEV = 0.
        FCTR = 0.
        TS = TG
        CM = CMB
        CH = CHB
        Q1 = QSFC
        Q2E = Q2B
        RSSUN = 0.0
        RSSHA = 0.0
        TGV = TGB
        CHV = CHB

    FIRE = LWDN + FIRA

    # Compute a net emissivity
    EMISSI = FVEG * (EMG * (1 - EMV) + EMV + EMV * (1 - EMV) * (1 - EMG)) + (1 - FVEG) * EMG

    # When we're computing a TRAD, subtract from the emitted IR the
    # reflected portion of the incoming LWDN, so we're just
    # considering the IR originating in the canopy/ground system.
    TRAD = ((FIRE - (1 - EMISSI) * LWDN) / (EMISSI * SB)) ** 0.25

    APAR = PARSUN * LAISUN + PARSHA * LAISHA
    PSN = PSNSUN * LAISUN + PSNSHA * LAISHA

    # effective parameters for PBL and diagnostics
    CHSTAR, TSTAR = EPARM(TAH, TGB, FVEG, CHV, CHB, VEG)

    return \
        SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
        TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
        BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
        QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, FSUN, PARSUN, PARSHA, \
        TV_l, FSTOMATA_l, ROOTU, TR_P
