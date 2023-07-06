def EPARM(TAH, TGB, FVEG, CHV, CHB, VEG):
    # effective exchange coefficient for PBL.
    if VEG:
        CHSTAR = FVEG * CHV + (1. - FVEG) * CHB
        W = FVEG * CHV * TAH + (1. - FVEG) * CHB * TGB
        TSTAR = W / CHSTAR
    else:
        CHSTAR = CHB
        TSTAR = TGB

    return CHSTAR, TSTAR
