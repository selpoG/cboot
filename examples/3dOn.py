from __future__ import division, print_function, unicode_literals

import re
import sys
from subprocess import Popen

import numpy as np

import sage.cboot.scalar as cbs

if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    import os
    DEVNULL = open(os.devnull, 'wb')
else:
    from subprocess import DEVNULL

context = cbs.context_for_scalar(epsilon=0.5, Lambda=13)
lmax = 25
nu_max = 12
cbs = dict()
for spin in range(lmax):
    cbs[spin] = context.approx_cb(nu_max, spin)


def make_F(delta, sector, spin, gap_dict, NSO):
    delta = context(delta)
    if (sector, spin) in gap_dict:
        gap = context(gap_dict[(sector, spin)])
    elif spin == 0:
        gap = context.epsilon
    else:
        gap = 2 * context.epsilon + spin
    g_shift = cbs[spin].shift(gap)

    g_num = g_shift.matrix()
    g_pref = g_shift.prefactor()
    F = context.F_minus_matrix(delta).dot(g_num)
    H = context.F_plus_matrix(delta).dot(g_num)

    if sector == "S":
        num = np.concatenate((context.null_ftype, F, H))

    elif sector == "T":
        num = np.concatenate(
            (F, (1 - 2 / context(NSO)) * F, -(1 + 2 / context(NSO)) * H))

    elif sector == "A":
        num = np.concatenate((-F, F, -H))

    return context.prefactor_numerator(g_pref, num)


def make_SDP(delta, gap_dict, NSO=2):
    delta = context(delta)
    pvms = []
    for sector in ("S", "T", "A"):
        if sector is not "A":
            spins = [spin for spin in cbs.keys() if spin % 2 == 0]
        else:
            spins = [spin for spin in cbs.keys() if spin % 2 != 0]
        for spin in spins:
            pvms.append(make_F(delta, sector, spin, gap_dict, NSO))

    norm_F = context.F_minus_matrix(delta).dot(context.gBlock(0, 0, 0, 0))
    norm_H = context.F_plus_matrix(delta).dot(context.gBlock(0, 0, 0, 0))
    norm = np.concatenate((context.null_ftype, norm_F, norm_H))

    obj = norm * 0
    return context.SDP(norm, obj, pvms)


sdpb = "sdpb"
sdpbparams = ["--findPrimalFeasible",
              "--findDualFeasible", "--noFinalCheckpoint"]


def bs(delta, upper=3, lower=1, sector="S", sdp_method=make_SDP, NSO=2):
    upper = context(upper)
    lower = context(lower)
    while upper - lower > 0.001:
        D_try = (upper + lower) / 2
        prob = sdp_method(delta, {(sector, 0): D_try}, NSO=NSO)
        prob.write("3d_Ising_binary.xml")
        sdpbargs = [sdpb, "-s", "3d_Ising_binary.xml"] + sdpbparams
        Popen(sdpbargs, stdout=DEVNULL, stderr=DEVNULL).wait()
        with open("3d_Ising_binary.out", "r") as sr:
            sol = re.compile(
                r'found ([^ ]+) feasible').search(sr.read()).groups()[0]
        if sol == "dual":
            print("(Delta_phi, Delta_{1})={0} is excluded.".format(
                (float(delta), float(D_try)), sector))
            upper = D_try
        elif sol == "primal":
            print("(Delta_phi, Delta_{1})={0} is not excluded.".format(
                (float(delta), float(D_try)), sector))
            lower = D_try
        else:
            raise RuntimeError("Unexpected return from sdpb")
    return upper


if __name__ == "__main__":
    # The default example
    print(bs(0.52))

    # ======================================
    # if you want to derive the bound on Delta_T
    #
    print(bs(0.52, sector="T"))
