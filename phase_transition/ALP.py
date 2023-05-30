"""
Title: ALP.py
Author: Isaac R. Wang
This code defines the effective potential of the model in paper.
Python package "cosmoTransitions" is used.
Convention and analytical expressions can be found in the paper.
"""

import numpy as np
import numpy.linalg as la
from cosmoTransitions import generic_potential as gp
from cosmoTransitions import pathDeformation as pd
from finiteT import Jb_spline as Jb
from finiteT import Jf_spline as Jf
from scipy import interpolate

# Define constants
# Here we use the 1-loop MSbar renormalization scheme, with renormalization scale mZ.
# Computation for renormalization: 1307.3536
# Code implementation: model_setup/Zero_T_full_model.nb
# Scalar contributions to the effective potential is included. Contribution from extra scalar (i.e. the ALP) to the renormalization of the gauge and Yukawa couplings are omitted for simplicity.

mZEW = 91.1876  # observed Z-boson mass. Not to be confused with the field-dependent
# Z boson mass term. So use this name here.
GF = 1.1663787e-05
v = (2**0.5 * GF) ** (-0.5)
v2 = v**2
mhpole = 125.13
g1 = 0.3573944734603928
g2 = 0.6510161183564389
yt = 0.9777726923522626


# Define effective potential of this model. Some functions and contributions are manually implemented and overwritten.


class model_ALP(gp.generic_potential):
    """
    Class of CosmoTransitions. Input parameters are the 1-loop renormalized
    parameters (but not the directly-observed quantites such as mixing angle or mass).

    Parameters include:
    lh: Higgs quartic coupling.
    A: Higgs-ALP coupling.
    muHsq: Higgs mass term squared.
    muSsq: ALP mass term squared.
    f: scale parameter for the ALP, i.e. the "decay constant" of the axion. Could
    be understood as the UV-completion scale.
    beta: phase difference between the ALP mass term and the Higgs-ALP interaction term.

    Effective potential of the model, and some defined functions for computation.
    """

    def init(self, mS, sintheta, lh, A, muHsq, muSsq, f, beta):
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
        self.beta = beta
        self.Tmax = 200
        self.renormScaleSq = mZEW**2
        self.Tc = None
        self.action_trace_data = None
        self.ST_func_interpolation = None

    def V0(self, X):
        """Tree-level potential."""

        # Define field variable quantities
        X = np.asanyarray(X)
        assert X.shape[-1] == 2

        h = X[..., 0]
        S = X[..., 1]

        # tree-level potential
        y_h = -0.5 * self.muHsq * h**2 + 0.25 * self.lh * h**4
        y_S = -self.f**2 * self.muSsq * (np.cos(S / self.f) - 1)
        y_hS = (
            -0.5 * self.A * self.f * (h**2 - 2 * v2) * np.sin(self.beta + S / self.f)
        )
        tot = y_h + y_S + y_hS

        return tot

    def boson_massSq(self, X, T):
        """
        Method of CosmoTransitions. Returns bosons mass square, dof and constants. The scalar masses are the eigenvalues of the full physical scalar matrix, plus the Nambu-Goldstone bosons.
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
            - self.A * self.f * np.sin(S / self.f + self.beta)
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
            - self.A * self.f * np.sin(S / self.f + self.beta)
        ) + (
            3 * self.g2**2 / 16
            + self.g1**2 / 16
            + 0.5 * self.lh
            + 0.25 * self.yt**2
        ) * T2

        cterm = -self.A * h * np.cos(S / self.f + self.beta)

        bterm = 0.5 * (
            self.A * (h**2 - 2 * v2) * np.sin(self.beta + S / self.f)
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
        Method of CosmoTransitions. Fermion mass square. Only top quark is included.
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

    def approxZeroTMin(self):
        """
        There are generically two minima at zero temperature in this model,
        and we want to include both of them.
        """
        return [np.array([v, 0]), np.array([-v, 0])]

    def getTc(self):
        """
        Find the critical temperature, using the built-in `getPhases` function
        of cosmoTransitions.
        """
        if self.Tc == None:
            self.calcTcTrans()
        self.Tc = self.TcTrans[0]["Tcrit"]

    def tunneling_at_T(self, T):
        """
        Solve the bounce equation for the transition from the false vacuum to the true vacuum,
        at a given temperature T.
        Called the function `pathDeformation.fullTunneling()` in CosmoTransitions.
        Returns the bounce solution, including the bubble profile and the action.
        """

        if self.Tc == None:
            self.getTc()

        assert T < self.Tc

        def V_(x, T=T, V=self.Vtot):
            return V(x, T)

        def dV_(x, T=T, dV=self.gradV):
            return dV(x, T)

        false_vev = self.TcTrans[0]["high_vev"].tolist()
        true_vev = self.TcTrans[0]["low_vev"].tolist()

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
        Trace the evolution of the 3d Euclidean action S_3/T as temperature cooling down.
        This function did 2 things:
            - store the traced data in self.action_trace_data.
            - Interpolate the action S_3/T as a function of T, store in self.ST_func_interpolation.
        """
        if self.TcTrans == None:
            self.getTc()
        if self.mS <= 1:
            Tmax = self.Tc - 0.02
        elif self.mS <= 0.05:
            Tmax = self.Tc - 0.05
        else:
            Tmax = self.Tc - 0.01
        eps = 0.002
        list = []
        for i in range(0, 1000):
            Ttest = Tmax - i * eps
            print("Tunneling at T=" + str(Ttest))
            trigger = self.S_over_T(Ttest)
            print("S3/T = " + str(trigger))
            list.append([Ttest, trigger])
            if trigger < 140.0:
                break
        Tmin = Ttest
        print("Tnuc should be within " + str(Tmin) + " and " + str(Tmin + eps))
        self.action_trace_data = np.array(list).transpose().tolist()
        Tlist = self.action_trace_data[0]
        log_action_list = [np.log10(i) for i in self.action_trace_data[1]]
        y = interpolate.interp1d(Tlist, log_action_list, kind="cubic")
        self.ST_func_interpolation = y

    def S_over_T_smooth(self, T):
        """
        The 3d Euclidean action at a given temperature T.
        Interpolated from the traced data.
        """
        if self.ST_func_interpolation == None:
            self.trace_action()
        return self.ST_func_interpolation(T)
