import numpy as np


def RADIATION(config, ISC, COSZ, ELAI, ESAI, SMC, SOLAD, SOLAI, FVEG):
    MPE = 1.E-6

    # surface abeldo
    ALBGRD, ALBGRI, ALBD, ALBI, FABD, FABI, FTDD, FTID, FTII, FSUN, FREVD, FREVI, FREGD, FREGI, BGAP, WGAP, GAP = \
        ALBEDO(config, ISC, COSZ, ELAI, ESAI, SMC, FVEG)

    # surface radiation
    FSHA = 1. - FSUN
    LAISUN = ELAI * FSUN
    LAISHA = ELAI * FSHA
    VAI = ELAI + ESAI
    # if VAI > 0:
    #     VEG = 1
    # else:
    #     VEG = 0

    PARSUN, PARSHA, SAV, SAG, FSA, FSR, FSRV, FSRG = SURRAD(MPE, FSUN, FSHA, ELAI, VAI, LAISUN, LAISHA, SOLAD, SOLAI, FABD,
                                                            FABI, FTDD, FTID, FTII, ALBGRD, ALBGRI, ALBD, ALBI, FREVI, FREVD,
                                                            FREGD, FREGI)

    return FSUN, LAISUN, LAISHA, PARSUN, PARSHA, SAV, SAG, FSA, FSR, FSRV, FSRG, BGAP, WGAP, GAP


def ALBEDO(config, ISC, COSZ, ELAI, ESAI, SMC, FVEG):
    # config parameters
    RHOL = [float(idx) for idx in config['radiation']['RHOL'].split(',')]
    RHOS = [float(idx) for idx in config['radiation']['RHOS'].split(',')]
    TAUL = [float(idx) for idx in config['radiation']['TAUL'].split(',')]
    TAUS = [float(idx) for idx in config['radiation']['TAUS'].split(',')]

    # some constants
    NBAND = 2
    MPE = 1.E-06

    # initialize variables
    ALBGRD = np.full(2, -9999.1)
    ALBGRI = np.full(2, -9999.1)
    ALBD = np.full(2, -9999.1)
    ALBI = np.full(2, -9999.1)
    FABD = np.full(2, -9999.1)
    FABI = np.full(2, -9999.1)
    FTDD = np.full(2, -9999.1)
    FTID = np.full(2, -9999.1)
    FTII = np.full(2, -9999.1)
    FTDI = np.full(2, -9999.1)
    FREVD = np.full(2, -9999.1)
    FREVI = np.full(2, -9999.1)
    FREGD = np.full(2, -9999.1)
    FREGI = np.full(2, -9999.1)
    BGAP = 0.
    WGAP = 0.
    GAP = 0.
    FSUN = 0

    # IB 0:vis, 1:nir
    # initialize output because solar radiation only done if COSZ > 0
    for IB in range(NBAND):
        ALBD[IB] = 0.
        ALBI[IB] = 0.
        ALBGRD[IB] = 0.
        ALBGRI[IB] = 0.
        FABD[IB] = 0.
        FABI[IB] = 0.
        FTDD[IB] = 0.
        FTID[IB] = 0.
        FTII[IB] = 0.
        FREVD[IB] = 0.
        FREVI[IB] = 0.
        FREGD[IB] = 0.
        FREGI[IB] = 0.

    # caution: undefined FREVD etc.
    if COSZ <= 0:
        return ALBGRD, ALBGRI, ALBD, ALBI, FABD, FABI, FTDD, FTID, FTII, FSUN, FREVD, FREVI, FREGD, FREGI, BGAP, WGAP, GAP

    VAI = ELAI + ESAI
    WL = ELAI / max(VAI, MPE)
    WS = ESAI / max(VAI, MPE)
    RHO = np.full(2, -9999.1)
    TAU = np.full(2, -9999.1)
    for IB in range(NBAND):
        RHO[IB] = max(RHOL[IB] * WL + RHOS[IB] * WS, MPE)
        TAU[IB] = max(TAUL[IB] * WL + TAUS[IB] * WS, MPE)

    ALBGRD, ALBGRI = GROUNDALB(config, NBAND, ISC, SMC)

    GDIR = -9999.1  # shut warning up
    for IB in range(NBAND):
        IC = 0
        FABD[IB], ALBD[IB], FTDD[IB], FTID[IB], GDIR, FREVD[IB], FREGD[IB], BGAP, WGAP, GAP = TWOSTREAM(config, IB, IC, COSZ, VAI,
                                                                                                        ALBGRD, ALBGRI, RHO, TAU,
                                                                                                        FVEG, BGAP, WGAP)
        IC = 1
        FABI[IB], ALBI[IB], FTDI[IB], FTII[IB], GDIR, FREVI[IB], FREGI[IB], BGAP, WGAP, GAP = TWOSTREAM(config, IB, IC, COSZ, VAI,
                                                                                                        ALBGRD, ALBGRI, RHO, TAU,
                                                                                                        FVEG, BGAP, WGAP)

    EXT = GDIR / COSZ * np.sqrt(1. - RHO[0] - TAU[0])
    FSUN = (1. - np.exp(-EXT * VAI)) / max(EXT * VAI, MPE)
    if FSUN < 0.01:
        FSUN = 0

    return ALBGRD, ALBGRI, ALBD, ALBI, FABD, FABI, FTDD, FTID, FTII, FSUN, FREVD, FREVI, FREGD, FREGI, BGAP, WGAP, GAP


def SURRAD(MPE, FSUN, FSHA, ELAI, VAI, LAISUN, LAISHA, SOLAD, SOLAI, FABD,
           FABI, FTDD, FTID, FTII, ALBGRD, ALBGRI, ALBD, ALBI, FREVI, FREVD, FREGD, FREGI):
    NBAND = 2

    # zero summed solar fluxes
    SAG = 0.
    SAV = 0.
    FSA = 0.

    CAD = np.full(2, -9999.1)
    CAI = np.full(2, -9999.1)
    for IB in range(NBAND):
        # absorbed by canopy
        CAD[IB] = SOLAD[IB] * FABD[IB]
        CAI[IB] = SOLAI[IB] * FABI[IB]
        SAV = SAV + CAD[IB] + CAI[IB]
        FSA = FSA + CAD[IB] + CAI[IB]

        # transmitted solar fluxes incident on ground
        TRD = SOLAD[IB] * FTDD[IB]
        TRI = SOLAD[IB] * FTID[IB] + SOLAI[IB] * FTII[IB]

        # solar radiation absorbed by ground surface
        ABS = TRD * (1. - ALBGRD[IB]) + TRI * (1. - ALBGRI[IB])
        SAG = SAG + ABS
        FSA = FSA + ABS

    # partition visible canopy absorption to sunlit and shaded fractions
    # to get average absorbed par for sunlit and shaded leaves
    LAIFRA = ELAI / max(VAI, MPE)
    if FSUN > 0.:
        PARSUN = (CAD[0] + FSUN * CAI[0]) * LAIFRA / max(LAISUN, MPE)
        PARSHA = (FSHA * CAI[0]) * LAIFRA / max(LAISHA, MPE)
    else:
        PARSUN = 0.
        PARSHA = (CAD[0] + CAI[0]) * LAIFRA / max(LAISHA, MPE)

    # reflected solar radiation
    RVIS = ALBD[0] * SOLAD[0] + ALBI[0] * SOLAI[0]
    RNIR = ALBD[1] * SOLAD[1] + ALBI[1] * SOLAI[1]
    FSR = RVIS + RNIR

    # reflected solar radiation of veg. and ground (combined ground)
    FSRV = FREVD[0] * SOLAD[0] + FREVI[0] * SOLAI[0] + FREVD[1] * SOLAD[1] + FREVI[1] * SOLAI[1]
    FSRG = FREGD[0] * SOLAD[0] + FREGI[0] * SOLAI[0] + FREGD[1] * SOLAD[1] + FREGI[1] * SOLAI[1]

    return PARSUN, PARSHA, SAV, SAG, FSA, FSR, FSRV, FSRG


def GROUNDALB(config, NBAND, ISC, SMC):
    # config parameters
    ALBSAT = [float(idx) for idx in config['radiation']['ALBSAT'].split(',')]
    ALBDRY = [float(idx) for idx in config['radiation']['ALBDRY'].split(',')]

    ALBGRD = np.full(2, -999.1)
    ALBGRI = np.full(2, -999.1)
    for IB in range(NBAND):
        INC = max(0.11 - 0.40 * SMC[0], 0.)
        ALBSOD = min(ALBSAT[IB] + INC, ALBDRY[IB])
        ALBSOI = ALBSOD

        # increase desert and semi-desert albedos
        if ISC == 9:
            ALBSOD = ALBSOD + 0.10
            ALBSOI = ALBSOI + 0.10

        ALBGRD[IB] = ALBSOD
        ALBGRI[IB] = ALBSOI

    return ALBGRD, ALBGRI


def TWOSTREAM(config, IB, IC, COSZ, VAI, ALBGRD, ALBGRI, RHO, TAU, FVEG, BGAP, WGAP):
    # config parameters
    HVT = float(config['vege_rad']['HVT'])
    HVB = float(config['vege_rad']['HVB'])
    RC = float(config['vege_rad']['RC'])
    XL = float(config['vege_rad']['XL'])

    # compute within and between gaps
    if VAI == 0.0:
        GAP = 1.0
        KOPEN = 1.0
    else:
        DENFVEG = -np.log(max(1.0 - FVEG, 0.01)) / (np.pi * RC ** 2)
        HD = HVT - HVB
        BB = 0.5 * HD
        THETAP = np.arctan(BB / RC * np.tan(np.arccos(max(0.01, COSZ))))
        BGAP = np.exp(-DENFVEG * np.pi * RC ** 2 / np.cos(THETAP))
        FA = VAI / (1.33 * np.pi * RC ** 3.0 * (BB / RC) * DENFVEG)
        NEWVAI = HD * FA
        WGAP = (1.0 - BGAP) * np.exp(-0.5 * NEWVAI / COSZ)
        GAP = min(1.0 - FVEG, BGAP + WGAP)
        KOPEN = 0.05

    # calculate two-stream parameters OMEGA, BETAD, BETAI, AVMU, GDIR, EXT.
    # OMEGA, BETAD, BETAI are adjusted for snow. values for OMEGA*BETAD
    # and OMEGA*BETAI are calculated and then divided by the new OMEGA
    # because the product OMEGA*BETAI, OMEGA*BETAD is used in solution.
    # also, the transmittances and reflectances (TAU, RHO) are linear
    # weights of leaf and stem values.
    COSZI = max(0.001, COSZ)
    CHIL = min(max(XL, -0.4), 0.6)
    if abs(CHIL) <= 0.01:
        CHIL = 0.01
    PHI1 = 0.5 - 0.633 * CHIL - 0.330 * CHIL * CHIL
    PHI2 = 0.877 * (1. - 2. * PHI1)
    GDIR = PHI1 + PHI2 * COSZI
    EXT = GDIR / COSZI
    AVMU = (1. - PHI1 / PHI2 * np.log((PHI1 + PHI2) / PHI1)) / PHI2
    OMEGAL = RHO[IB] + TAU[IB]
    TMP0 = GDIR + PHI2 * COSZI
    TMP1 = PHI1 * COSZI
    ASU = 0.5 * OMEGAL * GDIR / TMP0 * (1. - TMP1 / TMP0 * np.log((TMP1 + TMP0) / TMP1))
    BETADL = (1. + AVMU * EXT) / (OMEGAL * AVMU * EXT) * ASU
    BETAIL = 0.5 * (RHO[IB] + TAU[IB] + (RHO[IB] - TAU[IB]) * ((1. + CHIL) / 2.) ** 2) / OMEGAL

    # adjust omega, betad, and betai for intercepted snow
    OMEGA = OMEGAL
    BETAD = BETADL
    BETAI = BETAIL

    # absorbed, reflected, transmitted fluxes per unit incoming radiation
    B = 1. - OMEGA + OMEGA * BETAI
    C = OMEGA * BETAI
    TMP0 = AVMU * EXT
    D = TMP0 * OMEGA * BETAD
    F = TMP0 * OMEGA * (1. - BETAD)
    TMP1 = B * B - C * C
    H = np.sqrt(TMP1) / AVMU
    SIGMA = TMP0 * TMP0 - TMP1
    if abs(SIGMA) < 1.e-6:
        SIGMA = SIGMA / abs(SIGMA) * 1e-6
    P1 = B + AVMU * H
    P2 = B - AVMU * H
    P3 = B + TMP0
    P4 = B - TMP0
    S1 = np.exp(-H * VAI)
    S2 = np.exp(-EXT * VAI)
    if IC == 0:
        U1 = B - C / ALBGRD[IB]
        U2 = B - C * ALBGRD[IB]
        U3 = F + C * ALBGRD[IB]
    else:
        U1 = B - C / ALBGRI[IB]
        U2 = B - C * ALBGRI[IB]
        U3 = F + C * ALBGRI[IB]
    TMP2 = U1 - AVMU * H
    TMP3 = U1 + AVMU * H
    D1 = P1 * TMP2 / S1 - P2 * TMP3 * S1
    TMP4 = U2 + AVMU * H
    TMP5 = U2 - AVMU * H
    D2 = TMP4 / S1 - TMP5 * S1
    H1 = -D * P4 - C * F
    TMP6 = D - H1 * P3 / SIGMA
    TMP7 = (D - C - H1 / SIGMA * (U1 + TMP0)) * S2
    H2 = (TMP6 * TMP2 / S1 - P2 * TMP7) / D1
    H3 = - (TMP6 * TMP3 * S1 - P1 * TMP7) / D1
    H4 = -F * P3 - C * D
    TMP8 = H4 / SIGMA
    TMP9 = (U3 - TMP8 * (U2 - TMP0)) * S2
    H5 = - (TMP8 * TMP4 / S1 + TMP9) / D2
    H6 = (TMP8 * TMP5 * S1 + TMP9) / D2
    H7 = (C * TMP2) / (D1 * S1)
    H8 = (-C * TMP3 * S1) / D1
    H9 = TMP4 / (D2 * S1)
    H10 = (-TMP5 * S1) / D2

    # downward direct and diffuse fluxes below vegetation
    # Niu and Yang (2004), JGR.

    FTD = np.full(2, -9999.1)
    FTI = np.full(2, -9999.1)
    if IC == 0:
        FTDS = S2 * (1.0 - GAP) + GAP
        FTIS = (H4 * S2 / SIGMA + H5 * S1 + H6 / S1) * (1.0 - GAP)
    else:
        FTDS = 0.
        FTIS = (H9 * S1 + H10 / S1) * (1.0 - KOPEN) + KOPEN
    FTD[IB] = FTDS
    FTI[IB] = FTIS

    # flux reflected by the surface (veg. and ground)
    FRE = np.full(2, -9999.1)
    FREV = np.full(2, -9999.1)
    FREG = np.full(2, -9999.1)
    FAB = np.full(2, -9999.1)
    if IC == 0:
        FRES = (H1 / SIGMA + H2 + H3) * (1.0 - GAP) + ALBGRD[IB] * GAP
        # jref - separate veg. and ground reflection
        FREVEG = (H1 / SIGMA + H2 + H3) * (1.0 - GAP)
        FREBAR = ALBGRD[IB] * GAP
    else:
        FRES = (H7 + H8) * (1.0 - KOPEN) + ALBGRI[IB] * KOPEN
        # jref - separate veg. and ground reflection
        FREVEG = (H7 + H8) * (1.0 - KOPEN) + ALBGRI[IB] * KOPEN
        FREBAR = 0
    FRE[IB] = FRES
    FREV[IB] = FREVEG
    FREG[IB] = FREBAR

    # flux absorbed by vegetation
    FAB[IB] = 1. - FRE[IB] - (1. - ALBGRD[IB]) * FTD[IB] - (1. - ALBGRI[IB]) * FTI[IB]

    return FAB[IB], FRE[IB], FTD[IB], FTI[IB], GDIR, FREV[IB], FREG[IB], BGAP, WGAP, GAP
