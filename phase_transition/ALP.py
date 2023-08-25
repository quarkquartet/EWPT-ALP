"""
Title: ALP.py
Author: Isaac R. Wang
This code defines the effective potential of the model in paper.
Python package "cosmoTransitions" is used.
Convention and analytical expressions can be found in the paper.
"""

from collections import namedtuple

import numpy as np
from cosmoTransitions import generic_potential as gp
from cosmoTransitions import pathDeformation as pd
from cosmoTransitions.tunneling1D import SingleFieldInstanton
from finiteT import Jb_spline as Jb
from finiteT import Jf_spline as Jf
from scipy import interpolate, optimize

# Define constants
# Here we use the 1-loop MSbar renormalization scheme, with renormalization scale mZ.
# Computation for renormalization: 1307.3536
# Code implementation: model_setup/Zero_T_full_model.nb
# Scalar contributions to the effective potential is included. Contribution from extra scalar (i.e. the ALP) to the renormalization of the gauge and Yukawa couplings are omitted for simplicity.
mZEW = 91.1876
GF = 1.1663787e-05
v = (2**0.5 * GF) ** (-0.5) / np.sqrt(2)
v2 = v**2
mhpole = 125.13
g1 = 0.3573944734603928
g2 = 0.6510161183564389
yt = 0.9777726923522626


class model_ALP(gp.generic_potential):
    """
    Class of CosmoTransitions.

    Effective potential of the model,
    and some defined functions for computations.

    Input parameters are the observed mS and mixing angle,
    plus the 1-loop MSbar renormalized quantities.

    Parameters include:
    lh: Higgs quartic coupling.
    A: Higgs-ALP coupling.
    muHsq: Higgs mass term squared.
    muSsq: ALP mass term squared.
    f: scale parameter for the ALP, i.e. the "decay constant" of the axion.
    Could be understood as the UV-completion scale.
    delta: phase difference.
    """

    def init(self, mS, sintheta, lh, A, muHsq, muSsq, f, delta):
        self.Ndim = 2
        self.g1 = g1
        self.g2 = g2
        self.yt = yt
        self.mS = mS
        self.sintheta = sintheta
        self.lh = lh
        self.A = A
        self.muHsq = muHsq
        self.muSsq = muSsq
        self.f = f
        self.delta = delta
        self.Tmax = 200
        self.Tmin = 20
        self.renormScaleSq = mZEW**2
        self.Tc = None
        self.action_trace_data = None
        self.Tn = None
        self.action_trace_data_1d = None
        self.Tn1d = None
        self.Tcvev = None
        self.strength_Tc = None

    def V0(self, X):
        """Tree-level potential."""

        # Define field variable quantities
        X = np.asanyarray(X)
        assert X.shape[-1] == 2

        h = X[..., 0]
        S = X[..., 1]

        # tree-level potential
        y_h = (
            -0.5 * (self.muHsq - self.A * self.f * np.cos(self.delta)) * h**2
            + 0.25 * self.lh * h**4
        )

        y_S = -self.f**2 * self.muSsq * (np.cos(S / self.f) - 1)

        y_hS = (
            -0.5
            * self.A
            * self.f
            * (h**2 - 2 * v2)
            * np.cos(-self.delta + S / self.f)
        )

        tot = y_h + y_S + y_hS

        return tot

    def boson_massSq(self, X, T):
        """
        Method of CosmoTransitions.
        Returns bosons mass square, dof and constants.
        The scalar masses are the eigenvalues of the
        full physical scalar matrix,
        plus the Nambu-Goldstone bosons.
        """

        X = np.array(X)
        T = np.asanyarray(T, dtype=float)
        T2 = T * T
        assert X.shape[-1] == 2
        h = X[..., 0]
        S = X[..., 1]

        mgs = (
            self.lh * h**2
            - self.muHsq
            - self.A * self.f * np.cos(S / self.f - self.delta)
            + self.A * self.f * np.cos(self.delta)
        ) + (
            3 * self.g2**2 / 16
            + self.g1**2 / 16
            + 0.5 * self.lh
            + 0.25 * self.yt**2
        ) * T2

        # Scalar mass matrix ((a,c),(c,b))
        aterm = (
            3 * self.lh * h**2
            - self.muHsq
            - self.A * self.f * np.cos(S / self.f - self.delta)
            + self.A * self.f * np.cos(self.delta)
        ) + (
            3 * self.g2**2 / 16
            + self.g1**2 / 16
            + 0.5 * self.lh
            + 0.25 * self.yt**2
        ) * T2

        cterm = -self.A * h * np.sin(S / self.f - self.delta)

        bterm = 0.5 * (
            self.A * (h**2 - 2 * v2 + T2 / 3) * np.cos(-self.delta + S / self.f)
        ) / self.f + self.muSsq * np.cos(S / self.f)

        # Scalar eigenvalues
        mhsq = 0.5 * (aterm + bterm + np.sqrt((aterm - bterm) ** 2 + 4 * cterm**2))
        mSsq = 0.5 * (aterm + bterm - np.sqrt((aterm - bterm) ** 2 + 4 * cterm**2))

        mW = 0.25 * self.g2**2 * h**2
        mWL = mW + 11 * self.g2**2 * T2 / 6
        mZ = 0.25 * (self.g2**2 + self.g1**2) * h**2

        AZsq = np.sqrt(
            (self.g2**2 + self.g1**2) ** 2 * (3 * h**2 + 22 * T2) ** 2
            - 176 * self.g2**2 * self.g1**2 * T2 * (3 * h**2 + 11 * T2)
        )

        mZL = ((self.g2**2 + self.g1**2) * (3 * h**2 + 22 * T2) + AZsq) / 24
        mAL = ((self.g2**2 + self.g1**2) * (3 * h**2 + 22 * T2) - AZsq) / 24

        M = np.array([mSsq, mhsq, mgs, mW, mWL, mZ, mZL, mAL])
        M = np.rollaxis(M, 0, len(M.shape))

        dof = np.array([1, 1, 3, 4, 2, 2, 1, 1])
        c = np.array([1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.5])

        return M.real + 1e-16, dof, c

    def fermion_massSq(self, X):
        """
        Method of CosmoTransitions. Fermion mass square.
        Only top quark is included.
        """

        X = np.array(X)
        h = X[..., 0]

        mt = 0.5 * self.yt**2 * h**2
        Mf = np.array([mt])
        Mf = np.rollaxis(Mf, 0, len(Mf.shape))

        doff = np.array([12.0])

        return Mf, doff

    def V1(self, bosons, fermions, scale=mZEW):
        """
        Method of CosmoTransitions. Overwritten.

        The 1-loop CW correction at the zero-temperature in the
        MS-bar renormalization scheme.
        """

        scale2 = scale**2
        m2, n, c = bosons
        y = np.sum(n * m2 * m2 * (np.log(m2 / scale2 + 1e-100 + 0j) - c), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n * m2 * m2 * (np.log(m2 / scale2 + 1e-100 + 0j) - c), axis=-1)
        return y.real / (64 * np.pi * np.pi)

    def V0T(self, X):
        """
        1-loop corrected effective potential at T=0.
        Not an intrinsic method of CosmoTransitions.
        """
        X = np.asanyarray(X, dtype=float)

        bosons = self.boson_massSq(X, 0)
        fermions = self.fermion_massSq(X)

        y = self.V0(X)
        y += self.V1(bosons, fermions)

        return y

    def V1T(self, bosons, fermions, T, include_radiation=True):
        """
        Method of CosmoTransitions. Should be overwritten.
        The 1-loop finite-temperature correction term.

        `Jf` and `Jb` are modified functions.

        TODO: understand this again, write note, and implement it.
        """

        T2 = (T * T) + 1e-100
        T4 = T2 * T2

        m2, nb, _ = bosons
        y = np.sum(nb * Jb(m2 / T2), axis=-1)
        m2, nf = fermions
        y += np.sum(nf * Jf(m2 / T2), axis=-1)

        return y * T4 / (2 * np.pi * np.pi)

    def Vtot(self, X, T, include_radiation=True):
        """
        Method of CosmoTransitions.
        The total finite temperature effective potential.
        """

        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        assert X.shape[-1] == 2

        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        Vtot = self.V0(X)
        Vtot += self.V1(bosons, fermions)
        Vtot += self.V1T(bosons, fermions, T, include_radiation)

        return Vtot

    """
    So far we have computed the finite temperature effective potential.
    Move on to compute the relevant quantities in during the phase transition.
    """

    def getTc(self):
        """
        Find the critical temperature, using binormial search.
        """
        num_i = 30
        Tmax = self.Tmax
        Tmin = self.Tmin
        T_test = (Tmax + Tmin) * 0.5
        print("Finding Tc...")
        for i in range(num_i + 10):
            # Why use num_i + 10?
            # Binormial search does not guarantee that the result
            # at the number of iteration is higher or lower than the
            # true value.
            # But for us to have a v(T_c), we should perform the search
            # at the temperature lower than the real T_c.
            # So after enough number of iterations, we should
            # continue searching if the current T_test is
            # higher than T_c, until we have a lower one.
            h_range = np.linspace(0, 300, 300)
            V_range = np.array([self.Vmin(i, T_test) for i in h_range])
            V1dinter = interpolate.UnivariateSpline(h_range, V_range, s=0)
            xmin = optimize.fmin(V1dinter, 200, disp=False)[0]
            if V1dinter(xmin) < V1dinter(0) and xmin > 1:
                # This means the current temperature is lower than T_c.
                if i > num_i:
                    # If enough iteration is done, stop computing.
                    self.Tc = T_test
                    self.Tcvev = xmin
                    self.strength_Tc = xmin / T_test
                    break
                else:
                    # Not enough iteration, continue to compute.
                    Tmin = T_test
                    Tnext = (Tmax + T_test) * 0.5
                    T_test = Tnext
            else:
                # Current temperature is higher than T_c.
                Tmax = T_test
                Tnext = (Tmin + T_test) * 0.5
                T_test = Tnext

    def tunneling_at_T(self, T):
        """
        Solve the bounce equation for the transition from the false vacuum
        to the true vacuum, at a given temperature T.
        Called the function `pathDeformation.fullTunneling()`
        in CosmoTransitions.
        Returns the bounce solution, including the bubble profile
        and the action.
        """

        if self.Tc is None:
            self.getTc()

        assert T < self.Tc

        def V_(x, T=T, V=self.Vtot):
            return V(x, T)

        def dV_(x, T=T, dV=self.gradV):
            return dV(x, T)

        false_vev_0 = self.Smin(1e-16, T)
        true_vev_0 = self.Smin(self.Tcvev, T)

        false_vev = optimize.minimize(
            self.Vtot,
            x0=np.array([1e-16, false_vev_0]),
            args=(T,),
            method="Nelder-Mead",
            bounds=[(1e-16, 1e-16), (-np.pi * self.f, np.pi * self.f)],
        ).x

        true_vev = optimize.minimize(
            self.Vtot,
            x0=np.array([self.Tcvev, true_vev_0]),
            args=(T,),
            method="Nelder-Mead",
            bounds=[
                (self.Tcvev * 0.6, self.Tcvev * 1.2),
                (-np.pi * self.f, np.pi * self.f),
            ],
        ).x

        tobj = pd.fullTunneling([true_vev, false_vev], V_, dV_)

        return tobj

    def S_over_T_sol(self, T):
        """
        The S_3/T at a given temperature T.
        Directly call the function to solve the bounce equation to get it.
        """
        Tv = T
        ST = self.tunneling_at_T(T=Tv).action / Tv
        return ST

    def trace_action(self):
        """
        Trace the evolution of the 3d Euclidean action S_3/T as temperature
        cooling down.
        Stores in self.action_trace_data.
        Note: the Tmax and eps can be modified according to what you want.
        Usually this requires some test running.
        There is no general way to get a Tmax and eps automatically.
        """
        if self.Tn1d is None:
            self.find_Tn_1d()
        if self.mS <= 0.05:
            Tmax = self.Tn1d - 0.45
            eps = 0.002
        elif self.mS <= 1:
            Tmax = self.Tn1d - 0.2
            eps = 0.002
        elif self.mS <= 2:
            Tmax = self.Tn1d - 0.15
            eps = 0.003
        else:
            Tmax = self.Tn1d - 0.01
            eps = 0.005

        eps = 0.002

        list = []
        for i in range(0, 1000):
            Ttest = Tmax - i * eps
            print("Tunneling at T=" + str(Ttest))
            trigger = self.S_over_T_sol(Ttest)
            print("S3/T = " + str(trigger))
            list.append([Ttest, trigger])
            if trigger < 140.0:
                break
        Tmin = Ttest
        print("Tnuc should be within " + str(Tmin) + " and " + str(Tmin + eps))
        self.action_trace_data = np.array(list).transpose().tolist()

    def S_over_T_smooth(self, T):
        """
        The 3d Euclidean action at a given temperature T.
        Interpolated from the traced data.
        """
        if self.action_trace_data is None:
            print("No data to be used for interpolation. Tracing data...")
            self.trace_action()
        Tlist = self.action_trace_data[0]
        log_action_list = [np.log10(i) for i in self.action_trace_data[1]]
        y = interpolate.interp1d(Tlist, log_action_list, kind="cubic")
        return 10 ** y(T)

    def find_Tn(self):
        if self.action_trace_data is None:
            print("Tracing action data...")
            self.trace_action()

        def trigger(T):
            return np.log10(self.S_over_T_smooth(T)) - np.log10(140.0)

        self.Tn = optimize.brentq(
            trigger,
            self.action_trace_data[0][-2],
            self.action_trace_data[0][-1],
            disp=False,
            xtol=1e-5,
            rtol=1e-6,
        )
        print("Tnuc = " + str(self.Tn))

    def strength_Tn(self):
        if not self.Tn:
            self.findTn()
        Tnuc = self.Tn
        true_vev_0 = self.Smin(self.Tcvev, Tnuc)
        true_vev = optimize.minimize(
            self.Vtot,
            x0=np.array([self.Tcvev, true_vev_0]),
            args=(Tnuc,),
            method="Nelder-Mead",
            bounds=[
                (self.Tcvev * 0.6, self.Tcvev * 1.2),
                (-np.pi * self.f, np.pi * self.f),
            ],
        ).x[0]
        return true_vev / Tnuc

    def beta_over_H(self):
        """
        Compute the \beta/H quantity at Tn,
        defined as Tn * (d(S_3/T)/dT).
        Use Ridders algorithm.
        """
        if self.Tn is None:
            self.find_Tn()
        Tnuc = self.Tn
        if self.action_trace_data is None:
            self.trace_action()
        eps = 0.5 * (Tnuc - self.action_trace_data[0][-1]) * 0.9
        dev = (
            self.S_over_T_smooth(Tnuc - 2.0 * eps)
            - 8.0 * self.S_over_T_smooth(Tnuc - eps)
            + 8.0 * self.S_over_T_smooth(Tnuc + eps)
            - self.S_over_T_smooth(Tnuc + 2.0 * eps)
        ) / (12.0 * eps)
        return dev * Tnuc

    """
    So far we finished computing everything.
    But for comparison, we need to compute the `1d` phase
    transition quantities.
    """

    def Vmin(self, h, T):
        T2 = T**2
        # Write down the high-T expanded one as the initial guess.
        num = self.A * (3 * (h**2 - 2 * v**2) + T2) * np.sin(self.delta)
        den = 6 * self.f * self.muSsq + self.A * (
            3 * (h**2 - 2 * v**2) + T2
        ) * np.cos(self.delta)
        highT_path = self.f * np.arctan(num / den)
        y = optimize.minimize(
            self.Vtot,
            x0=np.array([h, highT_path]),
            args=(T,),
            method="Nelder-Mead",
            bounds=[(h, h), (-self.f * np.pi, self.f * np.pi)],
        ).fun
        return y

    def Smin(self, h, T):
        T2 = T**2
        # Write down the high-T expanded one as the initial guess.
        num = self.A * (3 * (h**2 - 2 * v**2) + T2) * np.sin(self.delta)
        den = 6 * self.f * self.muSsq + self.A * (
            3 * (h**2 - 2 * v**2) + T2
        ) * np.cos(self.delta)
        highT_path = self.f * np.arctan(num / den)
        y = optimize.minimize(
            self.Vtot,
            x0=np.array([h, highT_path]),
            args=(T,),
            method="Nelder-Mead",
            bounds=[(h, h), (-self.f * np.pi, self.f * np.pi)],
        ).x[1]
        return y

    def tunneling_at_T_1d(self, T):
        # interpolate at first
        if self.Tc is None:
            self.getTc()
        assert T < self.Tc
        fv = 1e-16
        h_range = np.linspace(-2.0 * self.Tcvev, 2.0 * self.Tcvev, 300)
        V_range = np.array([self.Vmin(i, T) for i in h_range])
        V1dinter = interpolate.UnivariateSpline(h_range, V_range, s=0)
        tv = optimize.fmin(V1dinter, self.Tcvev, disp=False)[0]
        gradV1d = V1dinter.derivative()
        tobj = SingleFieldInstanton(tv, fv, V1dinter, gradV1d)
        profile1d = tobj.findProfile()
        action = tobj.findAction(profile1d)
        rtuple = namedtuple("Tunneling_1d_rval", "profile Phi action")
        return rtuple(profile1d, profile1d.Phi, action)

    def S_over_T_sol_1d(self, T):
        Tv = T
        ST = self.tunneling_at_T_1d(T=Tv).action / Tv
        return ST

    def trace_action_1d(self):
        if self.Tc is None:
            self.getTc()
        if self.mS <= 0.05:
            Tmax = self.Tc - 0.05
            eps = 0.002
        elif self.mS <= 1:
            Tmax = self.Tc - 0.02
            eps = 0.002
        else:
            Tmax = self.Tc - 0.01
            eps = 0.005
        list = []
        for i in range(0, 1000):
            Ttest = Tmax - i * eps
            print("Tunneling at T = " + str(Ttest))
            trigger = self.S_over_T_sol_1d(Ttest)
            print("S3/T = " + str(trigger))
            list.append([Ttest, trigger])
            if trigger < 140.0:
                break
        Tmin = Ttest
        print(
            "Tnuc in 1d solution should be within "
            + str(Tmin)
            + " and "
            + str(Tmin + eps)
        )
        self.action_trace_data_1d = np.array(list).transpose().tolist()

    def S_over_T_smooth_1d(self, T):
        if self.action_trace_data_1d is None:
            print("Tracing action data...")
            self.trace_action_1d()
        Tlist = self.action_trace_data_1d[0]
        log_action_list = [np.log10(i) for i in self.action_trace_data_1d[1]]
        y = interpolate.interp1d(Tlist, log_action_list, kind="cubic")
        return 10 ** y(T)

    def find_Tn_1d(self):
        if self.action_trace_data_1d is None:
            print("Tracing action data...")
            self.trace_action_1d()

        def trigger(T):
            return np.log10(self.S_over_T_smooth_1d(T)) - np.log10(140.0)

        self.Tn1d = optimize.brentq(
            trigger,
            self.action_trace_data_1d[0][-2],
            self.action_trace_data_1d[0][-1],
            disp=False,
        )
        print("Tnuc = " + str(self.Tn1d))

    def strength_Tn_1d(self):
        if self.Tn1d is None:
            self.find_Tn_1d()
        Tnuc = self.Tn1d
        true_vev_0 = self.Smin(self.Tcvev, Tnuc)
        true_vev = optimize.minimize(
            self.Vtot,
            x0=np.array([self.Tcvev, true_vev_0]),
            args=(Tnuc,),
            method="Nelder-Mead",
            bounds=[
                (self.Tcvev * 0.6, self.Tcvev * 1.2),
                (-np.pi * self.f, np.pi * self.f),
            ],
        ).x[0]
        return true_vev / Tnuc

    def beta_over_H_1d(self):
        if self.Tn1d is None:
            self.find_Tn_1d()
        Tnuc = self.Tn1d
        eps = 0.5 * (Tnuc - self.action_trace_data_1d[0][-1]) * 0.9
        dev = (
            self.S_over_T_smooth_1d(Tnuc - 2.0 * eps)
            - 8.0 * self.S_over_T_smooth_1d(Tnuc - eps)
            + 8.0 * self.S_over_T_smooth_1d(Tnuc + eps)
            - self.S_over_T_smooth_1d(Tnuc + 2.0 * eps)
        ) / (12.0 * eps)
        return dev * Tnuc
