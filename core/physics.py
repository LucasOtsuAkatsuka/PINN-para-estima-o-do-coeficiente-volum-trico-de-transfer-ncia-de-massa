import numpy as np
from scipy.integrate import solve_ivp
from .config import Z_T, G_B, CP_B, CP_W_GAS, CP_W_LIQ, DH_VAP, LE, YW_IN, L_IN


def psat(T):


    T = np.clip(T, 260, 380)


    return np.exp(73.649 - 7258.2 / T - 7.3037 * np.log(T) + 4.1653e-6 * T**2)


def yw_sat(T, Patm=101325.0):

    ps = psat(T)


    ps = np.minimum(ps, Patm * 0.99)


    return 0.622 * ps / (Patm - ps)


def odes(z, state, kyam):


    Yw, L, TG, TL = state


    L  = max(L,  0.1)
    TG = max(TG, 260.0)
    TL = max(TL, 260.0)
    Yw = max(Yw, 1e-4)


    Yw_I = yw_sat(TL)


    dYw = Yw_I - Yw


    Cp_h = CP_B + Yw * CP_W_GAS


    hG_aH = LE * kyam * Cp_h


    dYw_dz = kyam * dYw / G_B

    dL_dz = kyam * dYw

    dTG_dz = (hG_aH * (TL - TG) + kyam * dYw * CP_W_GAS * (TL - TG)) / (G_B * Cp_h)

    dTL_dz = (hG_aH * (TL - TG) + kyam * dYw * DH_VAP) / (L * CP_W_LIQ)


    return [dYw_dz, dL_dz, dTG_dz, dTL_dz]


def solve_tower(kyam, TL_in, TG_in, Yw_in=YW_IN, L_in=L_IN,
                n_eval=200, return_profiles=False):

    def _shoot(TL_base):

        y0 = [Yw_in, L_in, TG_in, TL_base]

        try:


            sol = solve_ivp(
                lambda z, s: odes(z, s, kyam),
                [0, Z_T], y0,
                method='RK45',
                rtol=1e-8, atol=1e-10,
                max_step=Z_T / 50
            )

            return sol.y[3, -1] if sol.success else None
        except Exception:

            return None


    TL_lo = TG_in + 0.1
    TL_hi = TL_in - 0.1

    for _ in range(80):


        TL_mid = (TL_lo + TL_hi) / 2.0
        TL_top = _shoot(TL_mid)

        if TL_top is None:

            TL_hi = TL_mid
            continue


        if abs(TL_top - TL_in) < 1e-4:
            break


        if TL_top < TL_in:
            TL_lo = TL_mid
        else:

            TL_hi = TL_mid


    z_eval = np.linspace(0, Z_T, n_eval)

    y0 = [Yw_in, L_in, TG_in, TL_mid]

    sol = solve_ivp(
        lambda z, s: odes(z, s, kyam),
        [0, Z_T], y0,
        method='RK45',
        t_eval=z_eval,
        rtol=1e-8, atol=1e-10,
        max_step=Z_T / 50
    )


    result = {
        'TL_out': sol.y[3, 0],
        'TG_out': sol.y[2, -1],
    }


    if return_profiles:
        result['z']  = sol.t
        result['Yw'] = sol.y[0]
        result['L']  = sol.y[1]
        result['TG'] = sol.y[2]
        result['TL'] = sol.y[3]

    return result
