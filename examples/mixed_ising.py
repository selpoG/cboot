from __future__ import division, print_function, unicode_literals

import re
import sys
from subprocess import Popen

from sage.misc.cachefunc import cached_function

import sage.cboot.scalar as cbs

if sys.version_info.major == 2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    import os
    DEVNULL = open(os.devnull, 'wb')
else:
    from subprocess import DEVNULL

sdpb = "sdpb"
sdpbparams = ["--findPrimalFeasible",
              "--findDualFeasible", "--noFinalCheckpoint"]

context = cbs.context_for_scalar(epsilon=0.5, Lambda=11)
lmax = 20
nu_max = 8


@cached_function
def prepare_g_0(spin, Delta=None):
    return context.approx_cb(nu_max, spin)


@cached_function
def prepare_g_se(spin, Delta_se, Delta=None):
    g_se = context.approx_cb(
        nu_max, spin, Delta_1_2=Delta_se, Delta_3_4=Delta_se)
    return g_se


@cached_function
def prepare_g_es(spin, Delta_se, Delta=None):
    g_es = context.approx_cb(nu_max, spin, Delta_1_2=-
                             Delta_se, Delta_3_4=Delta_se)
    return g_es


def prepare_g(spin, Delta_se, Delta=None):
    if Delta is None:
        return (prepare_g_0(spin),
                prepare_g_se(spin, Delta_se),
                prepare_g_es(spin, Delta_se))
    else:
        g_0 = context.gBlock(spin, Delta, 0, 0)
        if not (Delta == 0 and spin == 0):
            g_se = context.gBlock(spin, Delta, Delta_se, Delta_se)
            g_es = context.gBlock(spin, Delta, -Delta_se, Delta_se)
        else:
            g_se = None
            g_es = None
        return (g_0, g_se, g_es)


def make_F(deltas, sector, spin, gap_dict, Delta=None):
    delta_s = context(deltas[0])
    delta_e = context(deltas[1])
    delta_mean = (delta_s + delta_e) / 2
    Delta_se = delta_s - delta_e
    if Delta is None:
        if (sector, spin) in gap_dict:
            shift = context(gap_dict[(sector, spin)])
        elif spin == 0:
            shift = context.epsilon
        else:
            shift = 2 * context.epsilon + spin
        gs = [x.shift(shift) for x in prepare_g(spin, Delta_se, Delta=Delta)]
    else:
        gs = prepare_g(spin, Delta_se, Delta=Delta)

    if sector == "even":
        F_s_s = context.dot(context.F_minus_matrix(delta_s), gs[0])
        F_e_e = context.dot(context.F_minus_matrix(delta_e), gs[0])

        F_s_e = context.dot(context.F_minus_matrix(delta_mean), gs[0])
        H_s_e = context.dot(context.F_plus_matrix(delta_mean), gs[0])
        return [[[F_s_s, 0],
                 [0, 0]],
                [[0, 0],
                 [0, F_e_e]],
                [[0, 0],
                 [0, 0]],
                [[0, F_s_e / 2],
                 [F_s_e / 2, 0]],
                [[0, H_s_e / 2],
                 [H_s_e / 2, 0]]]

    elif sector == "odd+":
        F_s_e = context.dot(context.F_minus_matrix(delta_mean), gs[1])
        F_e_s = context.dot(context.F_minus_matrix(delta_s), gs[2])
        H_e_s = context.dot(context.F_plus_matrix(delta_s), gs[2])

        return [0, 0, F_s_e, F_e_s, -H_e_s]

    elif sector == "odd-":
        F_s_e = context.dot(context.F_minus_matrix(delta_mean), gs[1])
        F_e_s = context.dot(context.F_minus_matrix(delta_s), gs[2])
        H_e_s = context.dot(context.F_plus_matrix(delta_s), gs[2])

        return [0, 0, -F_s_e, F_e_s, -H_e_s]
    else:
        raise RuntimeError("unknown sector name")


def make_SDP(deltas):
    pvms = []
    gaps = {("even", 0): 3, ("odd+", 0): 3}
    for spin in range(0, lmax):
        if spin % 2 == 0:
            pvms.append(make_F(deltas, "even", spin, gaps))
            pvms.append(make_F(deltas, "odd+", spin, gaps))
        else:
            pvms.append(make_F(deltas, "odd-", spin, gaps))

    epsilon_contribution = make_F(deltas, "even", 0, dict(), Delta=deltas[1])
    sigma_contribution = make_F(deltas, "odd+", 0, dict(), Delta=deltas[0])
    for m, x in zip(epsilon_contribution, sigma_contribution):
        m[0][0] += x
    pvms.append(epsilon_contribution)
    norm = []
    for v in make_F(deltas, "even", 0, dict(), Delta=0):
        norm.append(v[0][0] + v[0][1] + v[1][0] + v[1][1])
    obj = 0
    return context.sumrule_to_SDP(norm, obj, pvms)


def check(*deltas):
    prob = make_SDP(deltas)
    prob.write("3d_mixed.xml")
    sdpbargs = [sdpb, "-s", "3d_mixed.xml"] + sdpbparams
    Popen(sdpbargs, stdout=DEVNULL, stderr=DEVNULL).wait()
    with open("3d_mixed.out", "r") as sr:
        sol = re.compile(r'found ([^ ]+) feasible').search(sr.read()).groups()[0]
    if sol == "dual":
        print("(Delta_sigma, Delta_epsilon)={0} is excluded.".format(deltas))
    elif sol == "primal":
        print(
            "(Delta_sigma, Delta_epsilon)={0} is not excluded.".format(deltas))
    else:
        raise RuntimeError


if __name__ == "__main__":
    for delta_s in [0.518 + 0.002 * x for x in range(-1, 2)]:
        for delta_e in [1.41 + 0.02 * y for y in range(-1, 2)]:
            check(delta_s, delta_e)
