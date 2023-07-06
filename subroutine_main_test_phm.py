import datetime
import configparser
import matplotlib.pyplot as plt
from noah_energy.phm.energy_driver import ENERGY_DRIVER_PHM

# load parameters
config = configparser.ConfigParser()
config.read('../parameters/phm/energy_parameter.ini')

# test energy_driver
LAT = 40
LON = -88
TIME = datetime.datetime(2019, 7, 1, 16, 30, tzinfo=datetime.timezone.utc)
WS = 5
SFCTMP = 298
RH = 87
SFCPRS = 1e5
SOLDN = 1500
LWDN = 250
PRECP = 0
LAI = 4
SWC = 0.12
SH2O = [SWC, SWC, SWC, SWC]
SMC = [SWC, SWC, SWC, SWC]
GH = 10
CANLIQ = 0
FWET = 0

SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
PRECP, FSUN, PARSUN, PARSHA, \
TV_l, FSTOMATA_l = ENERGY_DRIVER_PHM(config, LAT, LON, TIME, WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ,
                                     FWET)

print('\nSAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR:', SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV,
      FCTR)
print('TRAD, T2M, PSN, APAR, SSOIL, BTRANI, BTRAN, LATHEA:', TRAD, T2M, PSN, APAR, SSOIL, LATHEA)
print('FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB:', FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB)
print('BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB:', BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB)
print('QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, PRECP:', QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, PRECP)

# plot
fig, ax = plt.subplots(3, 3, figsize=(14, 10))
ax[0, 0].plot(TV_l)

ax[0, 1].plot(FSTOMATA_l)
ax[0, 1].set_ylim(-0.05, 1.05)

fig.tight_layout()

plt.show()
