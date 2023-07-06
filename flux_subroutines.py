import numpy as np

from noah_energy.phm.energy_constants import HVAP
from noah_energy.phm.energy_constants import VKC, GRAV, CPAIR


def SFCDIF1(ITER, SFCTMP, RHOAIR, H, QAIR, ZLVL, ZPD, Z0M, Z0H, UR, MPE, MOZ, MOZSGN, FM, FH, FV):
    # Monin-Obukhov stability parameter moz for next iteration
    TMPCM = np.log((ZLVL - ZPD) / Z0M)
    TMPCH = np.log((ZLVL - ZPD) / Z0H)

    if ITER == 0:
        MOZOLD = MOZ
    else:
        MOZOLD = MOZ
        TVIR = (1. + 0.61 * QAIR) * SFCTMP
        TMP1 = VKC * (GRAV / TVIR) * H / (RHOAIR * CPAIR)
        if np.abs(TMP1) < MPE:
            TMP1 = MPE
        MOL = -1. * FV ** 3 / TMP1
        MOZ = np.minimum((ZLVL - ZPD) / MOL, 1.)

    # accumulate number of times moz changes sign.
    if MOZOLD * MOZ < 0:
        MOZSGN = MOZSGN + 1
    if MOZSGN >= 2:
        MOZ = 0.
        FM = 0.
        FH = 0.

    # evaluate stability-dependent variables using moz from prior iteration
    if MOZ < 0:
        TMP1 = (1. - 16. * MOZ) ** 0.25
        TMP2 = np.log((1. + TMP1 * TMP1) / 2.)
        TMP3 = np.log((1. + TMP1) / 2.)
        FMNEW = 2. * TMP3 + TMP2 - 2. * np.arctan(TMP1) + 1.5707963
        FHNEW = 2 * TMP2
    else:
        FMNEW = -5. * MOZ
        FHNEW = FMNEW

    # except for first iteration, weight stability factors for previous
    # iteration to help avoid flip-flops from one iteration to the next
    if ITER == 0:
        FM = FMNEW
        FH = FHNEW
    else:
        FM = 0.5 * (FM + FMNEW)
        FH = 0.5 * (FH + FHNEW)

    # exchange coefficients
    CMFM = TMPCM - FM
    CHFH = TMPCH - FH
    if np.abs(CMFM) <= MPE:
        CMFM = MPE
    if np.abs(CHFH) <= MPE:
        CHFH = MPE
    CM = VKC * VKC / (CMFM * CMFM)
    CH = VKC * VKC / (CMFM * CHFH)

    # friction velocity
    FV = UR * np.sqrt(CM)

    return MOZSGN, MOZ, FM, FH, CM, CH, FV


def RAGRB(config, ITER, VAI, RHOAIR, HG, TAH, ZPD, Z0MG, Z0HG, HCAN, UC, Z0H, FV, CWP, MPE, FHG):
    # config parameters
    DLEAF = float(config['vege_flux']['DLEAF'])

    # stability correction to below canopy resistance
    MOZG = 0.

    if ITER > 0:
        TMP1 = VKC * (GRAV / TAH) * HG / (RHOAIR * CPAIR)
        if np.abs(TMP1) <= MPE:
            TMP1 = MPE
        MOLG = -1. * FV ** 3 / TMP1
        MOZG = np.minimum((ZPD - Z0MG) / MOLG, 1.)

    if MOZG < 0:
        FHGNEW = (1. - 15. * MOZG) ** (-0.25)
    else:
        FHGNEW = 1. + 4.7 * MOZG

    if ITER == 0:
        FHG = FHGNEW
    else:
        FHG = 0.5 * (FHG + FHGNEW)

    CWPC = (CWP * VAI * HCAN * FHG) ** 0.5

    TMP1 = np.exp(-CWPC * Z0HG / HCAN)
    TMP2 = np.exp(-CWPC * (Z0H + ZPD) / HCAN)
    TMPRAH2 = HCAN * np.exp(CWPC) / CWPC * (TMP1 - TMP2)

    # aerodynamic resistances raw and rah between heights zpd+z0h and z0hg.
    KH = np.maximum(VKC * FV * (HCAN - ZPD), MPE)
    RAMG = 0.
    RAHG = TMPRAH2 / KH
    RAWG = RAHG

    # leaf boundary layer resistance
    TMPRB = CWPC * 50. / (1. - np.exp(-CWPC / 2.))
    RB = TMPRB * np.sqrt(DLEAF / UC)

    return MOZG, FHG, RAMG, RAHG, RAWG, RB


def STOMATA_PHM(config, MPE, APAR, FOLN, TV, EI, EA, SFCTMP, SFCPRS, O2, CO2, IGS, RB, FSTOMATA):
    # config parameters
    BP = float(config['vege_flux']['BP'])
    MP = float(config['vege_flux']['MP'])
    FOLNMX = float(config['vege_flux']['FOLNMX'])
    QE25 = float(config['vege_flux']['QE25'])
    KC25 = float(config['vege_flux']['KC25'])
    AKC = float(config['vege_flux']['AKC'])
    KO25 = float(config['vege_flux']['KO25'])
    AKO = float(config['vege_flux']['AKO'])
    C3PSN = float(config['vege_flux']['C3PSN'])
    AVCMX = float(config['vege_flux']['AVCMX'])
    VCMX25 = float(config['vege_flux']['VCMX25'])

    NITER_S = 100
    # print('stoma ET EA:', EI, EA)

    # initialize RS=RSMAX and PSN=0 because will only do calculations
    # for APAR > 0, in which case RS <= RSMAX and PSN >= 0
    CF = SFCPRS / (8.314 * SFCTMP) * 1.e06
    RS = 1. / BP * CF
    PSN = 0.

    if APAR <= 0:
        return RS, PSN

    FNF = np.minimum(FOLN / np.maximum(MPE, FOLNMX), 1.0)
    TC = TV - 273.15
    PPF = 4.6 * APAR
    J = PPF * QE25
    KC = KC25 * F1(AKC, TC)
    KO = KO25 * F1(AKO, TC)
    # KC = KC25
    # KO = KO25
    AWC = KC * (1. + O2 / KO)
    CP = 0.5 * KC / KO * O2 * 0.21
    VCMX = VCMX25 / (F2(TC) * C3PSN + F3(TC) * (1 - C3PSN)) * FNF * F1(AVCMX, TC)

    # first guess ci
    CI = 0.7 * CO2 * C3PSN + 0.4 * CO2 * (1. - C3PSN)

    # rb: s/m -> s m**2 / umol
    RLB = RB / CF
    # print('stoma RLB:', RLB)

    # constrain ea
    CEA = np.maximum(0.1 * EI * C3PSN + 0.1 * EI * (1. - C3PSN), np.minimum(EA, EI))
    # print('stoma EI EA CEA:', EI, EA, CEA)

    # ci iteration
    RF = 0.2
    for ITER_S in range(NITER_S):
        CI_OLD = CI
        # print('CI, RS:', CI, RS)

        WJ = np.maximum(CI - CP, 0.) * J / (CI + 2. * CP) * C3PSN + J * (1. - C3PSN)
        WC = np.maximum(CI - CP, 0.) * VCMX / (CI + AWC) * C3PSN + VCMX * (1. - C3PSN)
        WE = 0.5 * VCMX * C3PSN + 4000. * VCMX * CI / SFCPRS * (1. - C3PSN)
        PSN = np.min([WJ, WC, WE]) * IGS
        # print('stoma int:', ITER_S, PSN)
        # print('WJ, WC, WE:', WJ, WC, WE)

        CS = np.maximum(CO2 - 1.37 * RLB * SFCPRS * PSN, MPE)
        # print('CS:', CS)

        DL = EI - CEA
        DL = np.maximum(DL, MPE)
        A = 1.
        B = -2. * (BP + 1.6 * PSN * SFCPRS / CS) - (1.6 * PSN * SFCPRS / CS * MP) ** 2 * RLB / DL
        C = BP ** 2 + (2. * BP + 1.6 * PSN * SFCPRS / CS * (1. - MP ** 2 / DL)) * 1.6 * PSN * SFCPRS / CS
        Q1 = (-B + np.sqrt(np.maximum(B * B - 4. * A * C, MPE))) / 2. / A
        Q2 = (-B - np.sqrt(np.maximum(B * B - 4. * A * C, MPE))) / 2. / A
        Q = np.maximum(Q1, Q2)
        RS = 1. / np.maximum(Q, MPE) / FSTOMATA

        DCI = np.maximum(CS - PSN * SFCPRS * 1.65 * RS, 0.) - CI_OLD
        CI = CI_OLD + RF * DCI
        # print('DL', DL)
        # print('A, B, C:', A, B, C)

        DCI_P = np.abs(DCI) / np.maximum(CI_OLD, MPE)
        if DCI_P < 0.01:
            # print('stomata loop converged, ITER=', ITER_S)
            break
        if ITER_S == NITER_S:
            print("stomata loop not converging, ITER={}, DCI/CO2={}".format(ITER_S, DCI / CO2))
            print("stoma DCI_P=", DCI_P)
    # print('stoma RS:', RS)

    # rs, rb:  s m**2 / umol -> s/m
    RS = RS * CF

    return RS, PSN


def STOMATA(config, MPE, APAR, FOLN, TV, EI, EA, SFCTMP, SFCPRS, O2, CO2, IGS, BTRAN, RB):
    # config parameters
    BP = float(config['vege_flux']['BP'])
    MP = float(config['vege_flux']['MP'])
    FOLNMX = float(config['vege_flux']['FOLNMX'])
    QE25 = float(config['vege_flux']['QE25'])
    KC25 = float(config['vege_flux']['KC25'])
    AKC = float(config['vege_flux']['AKC'])
    KO25 = float(config['vege_flux']['KO25'])
    AKO = float(config['vege_flux']['AKO'])
    C3PSN = float(config['vege_flux']['C3PSN'])
    AVCMX = float(config['vege_flux']['AVCMX'])
    VCMX25 = float(config['vege_flux']['VCMX25'])

    NITER_S = 100
    # print('stoma ET EA:', EI, EA)

    # initialize RS=RSMAX and PSN=0 because will only do calculations
    # for APAR > 0, in which case RS <= RSMAX and PSN >= 0
    CF = SFCPRS / (8.314 * SFCTMP) * 1.e06
    RS = 1. / BP * CF
    PSN = 0.

    if APAR <= 0:
        return RS, PSN

    FNF = np.minimum(FOLN / np.maximum(MPE, FOLNMX), 1.0)
    TC = TV - 273.15
    PPF = 4.6 * APAR
    J = PPF * QE25
    KC = KC25 * F1(AKC, TC)
    KO = KO25 * F1(AKO, TC)
    # KC = KC25
    # KO = KO25
    AWC = KC * (1. + O2 / KO)
    CP = 0.5 * KC / KO * O2 * 0.21
    # VCMX = VCMX25 / (F2(TC) * C3PSN + F3(TC) * (1 - C3PSN)) * FNF * BTRAN * F1(AVCMX, TC)
    VCMX = VCMX25 / (F2(TC) * C3PSN + F3(TC) * (1 - C3PSN)) * FNF * F1(AVCMX, TC)

    # first guess ci
    CI = 0.7 * CO2 * C3PSN + 0.4 * CO2 * (1. - C3PSN)

    # rb: s/m -> s m**2 / umol
    RLB = RB / CF
    # print('stoma RLB:', RLB)

    # constrain ea
    CEA = np.maximum(0.1 * EI * C3PSN + 0.1 * EI * (1. - C3PSN), np.minimum(EA, EI))
    # print('stoma EI EA CEA:', EI, EA, CEA)

    # ci iteration
    RF = 0.2
    for ITER_S in range(NITER_S):
        CI_OLD = CI
        # print('CI, RS:', CI, RS)

        WJ = np.maximum(CI - CP, 0.) * J / (CI + 2. * CP) * C3PSN + J * (1. - C3PSN)
        WC = np.maximum(CI - CP, 0.) * VCMX / (CI + AWC) * C3PSN + VCMX * (1. - C3PSN)
        WE = 0.5 * VCMX * C3PSN + 4000. * VCMX * CI / SFCPRS * (1. - C3PSN)
        PSN = np.min([WJ, WC, WE]) * IGS
        # print('stoma int:', ITER_S, PSN)
        # print('WJ, WC, WE:', WJ, WC, WE)

        CS = np.maximum(CO2 - 1.37 * RLB * SFCPRS * PSN, MPE)
        # print('CS:', CS)

        DL = EI - CEA
        DL = np.maximum(DL, MPE)
        A = 1.
        B = -2. * (BP + 1.6 * PSN * SFCPRS / CS) - (1.6 * PSN * SFCPRS / CS * MP) ** 2 * RLB / DL
        C = BP ** 2 + (2. * BP + 1.6 * PSN * SFCPRS / CS * (1. - MP ** 2 / DL)) * 1.6 * PSN * SFCPRS / CS
        Q1 = (-B + np.sqrt(np.maximum(B * B - 4. * A * C, MPE))) / 2. / A
        Q2 = (-B - np.sqrt(np.maximum(B * B - 4. * A * C, MPE))) / 2. / A
        Q = np.maximum(Q1, Q2)
        # RS = 1. / np.maximum(Q, MPE)
        RS = 1. / np.maximum(Q, MPE) / BTRAN

        DCI = np.maximum(CS - PSN * SFCPRS * 1.65 * RS, 0.) - CI_OLD
        CI = CI_OLD + RF * DCI
        # print('DL', DL)
        # print('A, B, C:', A, B, C)

        DCI_P = np.abs(DCI) / np.maximum(CI_OLD, MPE)
        if DCI_P < 0.01:
            # print('stomata loop converged, ITER=', ITER_S)
            break
        if ITER_S == NITER_S:
            print("stomata loop not converging, ITER={}, DCI/CO2={}".format(ITER_S, DCI / CO2))
            print("stoma DCI_P=", DCI_P)
    # print('stoma RS:', RS)

    # rs, rb:  s m**2 / umol -> s/m
    RS = RS * CF

    return RS, PSN


def STOMATA_A(config, MPE, APAR, FOLN, TV, SFCTMP, SFCPRS, O2, CO2, IGS, BTRAN, RB, RS):
    # config parameters
    FOLNMX = float(config['vege_flux']['FOLNMX'])
    QE25 = float(config['vege_flux']['QE25'])
    KC25 = float(config['vege_flux']['KC25'])
    AKC = float(config['vege_flux']['AKC'])
    KO25 = float(config['vege_flux']['KO25'])
    AKO = float(config['vege_flux']['AKO'])
    C3PSN = float(config['vege_flux']['C3PSN'])
    AVCMX = float(config['vege_flux']['AVCMX'])
    VCMX25 = float(config['vege_flux']['VCMX25'])

    NITER_S_A = 100

    # initialize RS=RSMAX and PSN=0 because will only do calculations
    # for APAR > 0, in which case RS <= RSMAX and PSN >= 0
    CF = SFCPRS / (8.314 * SFCTMP) * 1.e06
    # RS = 1. / BP * CF
    RS_MOL = RS / CF
    PSN = 0.

    if APAR <= 0:
        return PSN

    FNF = np.minimum(FOLN / np.maximum(MPE, FOLNMX), 1.0)
    TC = TV - 273.15
    PPF = 4.6 * APAR
    J = PPF * QE25
    KC = KC25 * F1(AKC, TC)
    KO = KO25 * F1(AKO, TC)
    # KC = KC25
    # KO = KO25
    AWC = KC * (1. + O2 / KO)
    CP = 0.5 * KC / KO * O2 * 0.21
    VCMX = VCMX25 / (F2(TC) * C3PSN + F3(TC) * (1 - C3PSN)) * FNF * BTRAN * F1(AVCMX, TC)

    # first guess ci
    CI = 0.2 * CO2 * C3PSN + 0.15 * CO2 * (1. - C3PSN)

    # rb: s/m -> s m**2 / umol
    RLB = RB / CF
    # print('stoma RLB:', RLB)

    # set under relaxation factor
    if RS_MOL > 1e-4:
        RF = 0.02
    elif RS_MOL > 1e-5:
        RF = 0.05
    else:
        RF = 0.15

    # ci iteration
    for ITER_S in range(NITER_S_A):
        WJ = np.maximum(CI - CP, 0.) * J / (CI + 2. * CP) * C3PSN + J * (1. - C3PSN)
        WC = np.maximum(CI - CP, 0.) * VCMX / (CI + AWC) * C3PSN + VCMX * (1. - C3PSN)
        WE = 0.5 * VCMX * C3PSN + 4000. * VCMX * CI / SFCPRS * (1. - C3PSN)
        PSN = np.min([WJ, WC, WE]) * IGS

        CS = np.maximum(CO2 - 1.37 * RLB * SFCPRS * PSN, MPE)

        DCI = np.maximum(CS - PSN * SFCPRS * 1.65 * RS, 0.) - CI
        CI = CI + RF * DCI

        if abs(DCI / CO2) < 0.005:
            # print('stomata loop converged, ITER=', ITER_S)
            break
        if ITER_S == NITER_S_A:
            print("stomata actual loop not converging, ITER={}, DCI/CO2={}".format(ITER_S, DCI / CO2))

    return PSN


def PHM_nested(config, NSOIL, ZSOIL, DZSNSO, LATHEA, FCTR, SMC, FSTOMATA):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    DKSAT = float(config['soil']['DKSAT'])
    PSISAT = float(config['soil']['PSISAT'])
    # WLTSMC = float(config['soil']['SMCWLT'])

    # R2SR = float(config['phm']['R2SR'])
    RAI = float(config['phm']['RAI'])
    VEGH = float(config['phm']['VEGH'])
    VGSP50 = float(config['phm']['VGSP50'])
    VGA2 = float(config['phm']['VGA2'])
    VGKSAT = float(config['phm']['VGKSAT'])
    VGA1 = float(config['phm']['VGA1'])
    SPAI = float(config['phm']['SPAI'])
    VGTLP = float(config['phm']['VGTLP'])
    VGA3 = float(config['phm']['VGA3'])
    fRS = np.array([float(idx) for idx in config['phm']['fRS'].split(',')])

    # set iteration
    relax = 0.15
    FSTOMATA_OLD = FSTOMATA

    # convert transpiration to mm/s
    ETRAN = FCTR / LATHEA

    # set root profile
    RAI = RAI * fRS

    # soil to root conductance
    WCND = np.zeros(NSOIL)
    SRCD = np.zeros(NSOIL)
    SOLPSI = np.zeros(NSOIL)
    ZMS2G = np.zeros(NSOIL)
    for IZ in range(NSOIL):
        FACTR = max(0.01, SMC[IZ] / SMCMAX)
        EXPON = 2 * BEXP + 3
        WCND[IZ] = DKSAT * FACTR ** EXPON
        SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * DZSNSO[IZ])
        # SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ] / (DROOT * DZSNSO[IZ]))
        # if IZ == 2:
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI / DROOT / 1.1)
        # SRCD[IZ] = WCND[IZ] * np.sqrt(RAI) / (np.pi / 1.3)
        SOLPSI[IZ] = max(-PSISAT * (min(SMC[IZ], SMCMAX) / SMCMAX) ** (-BEXP), -500)
        # SOLPSI[IZ] = -PSISAT * (max(WLTSMC / SMCMAX, min(SMC[IZ], SMCMAX) / SMCMAX)) ** (-BEXP)
        ZMS2G[IZ] = max(0.0, -ZSOIL[IZ] - DZSNSO[IZ] * 0.5)

    # soil to root flux, temporary calculation
    VGPSIS = 'holder'
    recalculate = True
    recalculated = False
    while recalculate:
        SRCDt = np.sum(SRCD)
        QS2R_temp = np.zeros(NSOIL)
        for IZ in range(NSOIL):
            QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ] - ZMS2G[IZ])
        QS2R_temp_sum = np.sum(QS2R_temp)
        VGPSIS = (QS2R_temp_sum - ETRAN / 1000) / SRCDt
        # print('VGPSIS:', VGPSIS, 'SOLPSI-ZMS2G', SOLPSI[0] - ZMS2G[0], SOLPSI[1] - ZMS2G[1])

        if recalculated:
            break

        recalculate = False
        for IZ in range(NSOIL):
            if (SOLPSI[IZ] - ZMS2G[IZ] - VGPSIS < 0) and (ETRAN > 0):
                # print(SOLPSI[IZ] - ZMS2G[IZ] - VGPSIS)
                SRCD[IZ] = 0
                recalculate = True
                recalculated = True

    # xylem to leaf flux
    KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)
    VGPSIL = VGPSIS - VEGH - ETRAN / 1000 * VGA1 * VEGH / (KA * SPAI)

    # stomatal downregulation factor
    FSTOMATA_NEW = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
    FSTOMATA = FSTOMATA_OLD + relax * (FSTOMATA_NEW - FSTOMATA_OLD)

    return FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, ZMS2G


def PHM(config, NSOIL, ZSOIL, DZSNSO, FCTR, SMC):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    PSISAT = float(config['soil']['PSISAT'])
    # WLTSMC = float(config['soil']['SMCWLT'])
    DKSAT = float(config['soil']['DKSAT'])

    RAI = float(config['phm']['RAI'])
    VEGH = float(config['phm']['VEGH'])
    VGSP50 = float(config['phm']['VGSP50'])
    VGA2 = float(config['phm']['VGA2'])
    VGKSAT = float(config['phm']['VGKSAT'])
    VGA1 = float(config['phm']['VGA1'])
    SPAI = float(config['phm']['SPAI'])
    VGTLP = float(config['phm']['VGTLP'])
    VGA3 = float(config['phm']['VGA3'])
    fRS = np.array([float(idx) for idx in config['phm']['fRS'].split(',')])

    # set iteration
    NITER = 1000
    relax = 0.07

    # convert transpiration to mm/s
    ETRAN = FCTR / HVAP

    # set root profile
    RAI = RAI * fRS

    # soil to root conductance
    WCND = np.zeros(NSOIL)
    SRCD = np.zeros(NSOIL)
    SOLPSI = np.zeros(NSOIL)
    ZMS2G = np.zeros(NSOIL)
    for IZ in range(NSOIL):
        FACTR = max(0.01, SMC[IZ] / SMCMAX)
        EXPON = 2 * BEXP + 3
        WCND[IZ] = DKSAT * FACTR ** EXPON
        SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * DZSNSO[IZ])
        # if IZ == 2:
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI / DROOT / 1.1)
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI) / (np.pi * 1.1)
        # SOLPSI[IZ] = -PSISAT * max(WLTSMC / SMCMAX, min(SMC[IZ], SMCMAX) / SMCMAX) ** (-BEXP)
        SOLPSI[IZ] = max(-PSISAT * (min(SMC[IZ], SMCMAX) / SMCMAX) ** (-BEXP), -500)
        ZMS2G[IZ] = max(0.0, -ZSOIL[IZ] - DZSNSO[IZ] * 0.5)
    SRCDt = np.sum(SRCD)

    # iteration to solve transpiration
    VGPSIS = VGPSIL = FSTOMATA = KA = 'holder'

    ETRAN_d = ETRAN
    ETRAN_d_l = []
    for i in range(NITER):
        ETRAN_d_old = ETRAN_d
        ETRAN_d_l.append(ETRAN_d)

        # soil to root flux, temporary calculation
        QS2R_temp = np.zeros(NSOIL)
        for IZ in range(NSOIL):
            QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ] - ZMS2G[IZ])
        QS2R_temp_sum = np.sum(QS2R_temp)
        VGPSIS = (QS2R_temp_sum - ETRAN_d / 1000) / SRCDt

        # xylem to leaf flux
        KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)  # Xu et al.
        # KA = VGKSAT * (1 - 1 / (1 + np.exp(A_FENG * (VGPSIS - VGSP50))))  # Feng et al.
        VGPSIL = VGPSIS - VEGH - ETRAN_d / 1000 * VGA1 * VEGH / max(KA * SPAI, 1e-20)

        # stomatal downregulation factor
        FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
        ETRAN_d = ETRAN_d_old + relax * (FSTOMATA * ETRAN - ETRAN_d_old)

        change = (ETRAN_d - ETRAN_d_old) / max(ETRAN_d_old, 1e-6)
        if i > 5 and np.abs(change) < 1e-3:
            # print('itertion converged, iter =', i + 1)
            break

        if iter == NITER:
            print('iteration not converging, change = ', change)

    # convert back to W/m2
    FCTR_d = ETRAN_d * HVAP
    FCTR_d_l = np.array(ETRAN_d_l) * HVAP

    # calculate root water extraction
    ROOTU = np.full(NSOIL, -9999.1)
    for IZ in range(NSOIL):
        ROOTU[IZ] = SRCD[IZ] * (SOLPSI[IZ] - VGPSIS - ZMS2G[IZ])

    return FCTR_d, FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, ROOTU, FCTR_d_l


def _calc_vapor_pressure(rh, T):
    es = 0.611 * np.exp(17.27 * (T - 273.16) / (T - 35.86)) * 1000
    ea = rh * es / 100
    return ea


def F1(AB, BC):
    return AB ** ((BC - 25.) / 10.)


def F2(AB):
    return 1 + np.exp((-2.2E05 + 710. * (AB + 273.16)) / (8.314 * (AB + 273.16)))


def F3(AB):
    return (1. + np.exp(0.3 * (286.15 - AB - 273.16))) * (1. + np.exp(0.3 * (AB + 273.16 - 309.15)))


def ESAT(T):
    A0 = 6.107799961
    A1 = 4.436518521E-01
    A2 = 1.428945805E-02
    A3 = 2.650648471E-04
    A4 = 3.031240396E-06
    A5 = 2.034080948E-08
    A6 = 6.136820929E-11

    B0 = 6.109177956
    B1 = 5.034698970E-01
    B2 = 1.886013408E-02
    B3 = 4.176223716E-04
    B4 = 5.824720280E-06
    B5 = 4.838803174E-08
    B6 = 1.838826904E-10

    C0 = 4.438099984E-01
    C1 = 2.857002636E-02
    C2 = 7.938054040E-04
    C3 = 1.215215065E-05
    C4 = 1.036561403E-07
    C5 = 3.532421810e-10
    C6 = -7.090244804E-13

    D0 = 5.030305237E-01
    D1 = 3.773255020E-02
    D2 = 1.267995369E-03
    D3 = 2.477563108E-05
    D4 = 3.005693132E-07
    D5 = 2.158542548E-09
    D6 = 7.131097725E-12

    ESW = 100. * (A0 + T * (A1 + T * (A2 + T * (A3 + T * (A4 + T * (A5 + T * A6))))))
    ESI = 100. * (B0 + T * (B1 + T * (B2 + T * (B3 + T * (B4 + T * (B5 + T * B6))))))
    DESW = 100. * (C0 + T * (C1 + T * (C2 + T * (C3 + T * (C4 + T * (C5 + T * C6))))))
    DESI = 100. * (D0 + T * (D1 + T * (D2 + T * (D3 + T * (D4 + T * (D5 + T * D6))))))

    return ESW, ESI, DESW, DESI
