#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
.. module:: zlel_p3.py
    :synopsis: This module will add to zlel_p2 the ability
     to solve non-linear resistant circuits (diode and transistor).
     It is based on the Newton-Raphson method and the discreet equivalent.


'''


import numpy as np
import sys

if __name__ == "zlel.zlel_p3":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matices.
        If the file has not the proper dimensions it warns and exit.

    Args:
        filename: string with the name of the file

    Returns:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size(b,3)
        cir_ctrl: np array of strings with the element which branch
        controls the controlled sources. size(1,b)
        sim: np array of a list of strings with the firt the type of simulation
        as the first element and its values. size(1,9)

    Rises:
        SystemExit

    """

    try:
        # print(filename)
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    # print("cir:")
    # print(cir)
    # print()
    pos = "a"
    sim = None
    for i in range(len(cir)):
        if cir[i][0][0] == ".":
            pos = i
            break

    if pos != "a":
        cir, sim = cir[:pos, :], cir[pos:, :]

    cir_el = np.array(cir[:, 0], dtype=str)
    cir_nd = np.array(cir[:, 1:5], dtype=int)
    cir_val = np.array(cir[:, 5:8], dtype=float)
    cir_ctr = np.array(cir[:, -1], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctr, sim


def ez_linealak(cir_el_luz):
    '''
    This functions identifies non-linear elements of the circuit. It assigns
    0 to non linear elements, 1 to diodes and 2 to transistors.

    Args:
        cir_el_luz : np array of strings with the elements of the circuit.

    Returns:
        cir_ezl_luz : np array with the elements to identify non linear
        elements in the circuit.
    '''

    cir_ezl_luz = np.zeros(len(cir_el_luz), dtype=float)

    for i in range(len(cir_el_luz)):

        letra = cir_el_luz[i][0].upper()

        # trantsistorea bada 1 zenbakia
        # eta diodoa bada 2 zenbakia
        if letra == "D":
            cir_ezl_luz[i] = 1
        elif letra == "Q":
            cir_ezl_luz[i] = 2

    return cir_ezl_luz


def diode_NR(I0, n, Vdj):
    '''
    This function calculates the g and I of a diode for a
    Newton Raphson discrete equivalent.

    Calculates the g and the I of a diode for a NR discrete equivalent
    Given,

    :math:`Id = I_0(e^{(\\frac{V_d}{nV_T})}-1)`

    The NR discrete equivalent will be,

    :math:`i_{j+1} + g v_{j+1} = I`

    where,

    :math:`g = -\\frac{I_0}{nV_T}e^{(\\frac{V_d}{nV_T})}`

    and

    :math:`I = I_0(e^{(\\frac{V_{dj}}{nV_T})}-1) + gV_{dj}`


    Args:
        I0 : float. reverse-bias saturation current of the diode.
        n : float. ideality factor of the diode.
        Vdj : float. value of Vd

    Returns:
        I : float. The current of the diode.
        g : float. Conductance of the NR discrete equivalent for the diode.
    '''

    Vt = 300*8.6173324*10**(-5)

    Id = I0*(np.exp(Vdj/(n*Vt)) - 1)

    g = -I0/(n * Vt)*np.exp(Vdj/(n*Vt))

    Ij = Id + g*Vdj

    return Ij, g


def NR_bukatu(sol, Vd_lista, eps=10**(-5)):
    '''
    This function evaluates the precision of the NR discrete equivalent for
    each non linear element of the circuit. When all non linear elements meet
    the required precision it returns True.

    Returns:
        bool: while is False it is required another iteration for NR
        method. When its value is True, the precision required is met
        and there is no need for another iteration.
    '''

    for i in range(len(sol)):
        if Vd_lista[i] == 0:
            continue
        else:
            if abs(sol[i] - Vd_lista[i]) > eps:
                return False

    return True


def tentsio_lista_akt(sol_V_lista, Vd_lista):
    '''
    This function updates the values of the list given with the new values
    provided.

    Args:
        sol_V_lista : list with new values.
        Vd_lista : list with values to be updated.

    Returns:
        lista_ber : list. It contains the list with updated values.

    '''

    lista_ber = Vd_lista.copy()

    for i in range(len(Vd_lista)):
        if Vd_lista[i] != 0:
            lista_ber[i] = sol_V_lista[i]

    return lista_ber


def tranNR(Ies, Ics, beta, Vbe, Vbc):
    '''
    This functions calculates the currents and conductances of the transistor.

    Args:
        Ies : float. saturation current of the emitter.
        Ics : float. saturation current of the collector.
        beta : float. Current gain.
        Vbe : float. Tension between the base and the emitter
        Vbc : float. Tension between the base and the collector.

    Returns:
        Ie: float. The current of the emitter.
        Ic: float. The current of the collector.
        g11, g12, g21, g22: float. conductances.

    '''
    alphaf = beta/(1 + beta)
    alphar = Ies * alphaf / Ics

    Vt = 300*8.6173324*10**(-5)

    ie = Ies*(np.exp(Vbe/Vt) - 1) - alphar*Ics*(np.exp(Vbc/Vt) - 1)
    ic = -alphaf*Ies*(np.exp(Vbe/Vt) - 1) + Ics*(np.exp(Vbc/Vt) - 1)

    g11 = -Ies/Vt*np.exp(Vbe/Vt)
    g22 = -Ics/Vt*np.exp(Vbc/Vt)

    g12 = -alphar*g22
    g21 = -alphaf*g11

    Ie = g11*Vbe + g12*Vbc + ie
    Ic = g21*Vbe + g22*Vbc + ic

    # hemen g-ren baliok bueltatu  M eta N matrizetan ordezkatzeko
    return Ie, Ic, g11, g12, g21, g22


# NR trantsistorek ta diodok batea itekotan nola izangozan
def NR(n, b, M, N, Us, A, cir_ezl_luz, cir_val_luz, iter_max=100):
    '''
    This function solves a circuit which contains non linear elements using
    the method of Newton Raphson.

    Args:
        n : int. Number of nodes.
        b : int. Number of branches.
        M : matrix. Contains the voltage related multipliers of each element.
        N : matrix. Contains the current related multipliers of each element.
        Us : matrix. Contains the independent voltage and current sources'
        multipliers.
        A : matrix. Incidence matrix.
        cir_ezl_luz : np array with the elements to identificate non linear
        elements in the circuit.
        cir_val_luz : np array with the values of each element.
        iter_max : int. Maximum number of interactions. The default is 100.

    Returns:
        soluzioa : list. The function returns the w vector, containing the
        value of every variable of the system.

    '''
    Vd_0 = 0.6
    Vbe_0 = 0.6

    it = 0

    M_ber = M.copy()
    N_ber = N.copy()
    Us_ber = Us.copy()

    hurrengoa_Vbc = False

    Vd_lista = cir_ezl_luz.copy()
    for i in range(len(Vd_lista)):
        if Vd_lista[i] == 0:
            Vd_lista[i] = 0.0
        elif Vd_lista[i] == 1:
            Vd_lista[i] = Vd_0

        # biak berdin hasieratzeia ordun igualdigu,
        elif Vd_lista[i] == 2:
            Vd_lista[i] = Vbe_0

    for i in range(len(cir_ezl_luz)):
        # diodok hasieratzen
        if cir_ezl_luz[i] == 1:

            I0 = cir_val_luz[i][0]
            n_d = cir_val_luz[i][1]

            I, g = diode_NR(I0, n_d, Vd_lista[i])

            M_ber[i, i] = g
            N_ber[i, i] = 1

            Us_ber[i] = I

        # trantsistorek hasieratzen
        if cir_ezl_luz[i] == 2:

            if not hurrengoa_Vbc:
                hurrengoa_Vbc = True
                continue

            if hurrengoa_Vbc:
                hurrengoa_Vbc = False

                Ies = cir_val_luz[i][0]
                Ics = cir_val_luz[i][1]
                beta = cir_val_luz[i][2]

                Ie, Ic, g11, g12, g21, g22 = tranNR(Ies, Ics, beta,
                                                    Vd_lista[i-1], Vd_lista[i])

                # ekuaziok hauek diala suposatukot:
                # 1*ie_j+1 + g11*Vbe_j+1 + g12*Vbc_j+1 = Ie
                # 1*ic_j+1 + g21*Vbe_j+1 + g22*Vbc_j+1 = Ic

                # Ie
                # i-1, bc kin aigealako, ordun aurrekoa da be
                N_ber[i-1][i-1] = 1
                M_ber[i-1][i-1] = g11
                M_ber[i-1][i] = g12
                Us_ber[i-1] = Ie

                # Ic
                N_ber[i][i] = 1
                M_ber[i][i-1] = g21
                M_ber[i][i] = g22
                Us_ber[i] = Ic

    while it < iter_max:

        T, U = zl2.TU(n, b, M_ber, N_ber, Us_ber, A)

        soluzioa = zl2.OPsoluzioa(T, U)

        # -1 hoi nodo kopurua 1 etik hastealako, baño listan indizea 0 tik
        sol_V_lista = soluzioa[n - 1: n + b - 1]

        if NR_bukatu(sol_V_lista, Vd_lista):
            return soluzioa
        else:
            Vd_lista = tentsio_lista_akt(sol_V_lista, Vd_lista)
            it += 1

            for i in range(len(cir_ezl_luz)):
                if cir_ezl_luz[i] == 1:

                    I0 = cir_val_luz[i][0]
                    n_d = cir_val_luz[i][1]

                    I, g = diode_NR(I0, n_d, Vd_lista[i])

                    M_ber[i, i] = g
                    N_ber[i, i] = 1

                    Us_ber[i] = I

                elif cir_ezl_luz[i] == 2:

                    if not hurrengoa_Vbc:
                        hurrengoa_Vbc = True
                        continue

                    if hurrengoa_Vbc:
                        hurrengoa_Vbc = False

                        Ies = cir_val_luz[i][0]
                        Ics = cir_val_luz[i][1]
                        beta = cir_val_luz[i][2]

                        Ie, Ic, g11, g12, g21, g22 = tranNR(Ies, Ics, beta,
                                                            Vd_lista[i-1],
                                                            Vd_lista[i])

                        # ekuaziok hauek diala suposatukot:
                        # 1*ie_j+1 + g11*Vbe_j+1 + g12*Vbc_j+1 = Ie
                        # 1*ic_j+1 + g21*Vbe_j+1 + g22*Vbc_j+1 = Ic

                        # Ie
                        # i-1, bc kin aigealako, ordun aurrekoa da be
                        N_ber[i-1][i-1] = 1
                        M_ber[i-1][i-1] = g11
                        M_ber[i-1][i] = g12
                        Us_ber[i-1] = Ie

                        # Ic
                        N_ber[i][i] = 1
                        M_ber[i][i-1] = g21
                        M_ber[i][i] = g22
                        Us_ber[i] = Ic

    return soluzioa


def idatziTR_ezl(hasiera, amaiera, pausua, A, M, N, T, U, n, b,
                 cir_el_luz, cir_val_luz, cir_ezl_luz, filename):
    '''

    This function analizes the time evolution of a given circuit tha contains
    non linear elements. Solving the equation system for every time value for
    a selected range. Writes every value of the time and its related solution
    in a document called the same way as the input circuit + .tr .

    Args:
        hasiera : string. A string containing an integer value which specifies
        the final value of the selected time range.
        amaiera : string. A string containing an integer value which specifies
        the initial value of the selected time range.
        pausua : string. A string containing an integer value which specifies a
        value to be added to the current time in order to complete the TR
        analysis.

    Returns:
        None.

    '''
    hasiera = float(hasiera)
    amaiera = float(amaiera)
    pausua = float(pausua)

    t = hasiera

    with open(filename[: -3] + "tr", 'w') as file:

        header = zl2.build_csv_header("t", b, n)
        print(header, file=file)

        while t < amaiera:
            U_berria = zl2.t_akt(cir_el_luz, cir_val_luz, t, U, n, b)

            soluzioa = NR(n, b, M, N, U_berria[n-1 + b:], A, cir_ezl_luz,
                          cir_val_luz)
            sol_csv = ','.join(['%.9f' % num for num in soluzioa])
            # lerroa = str(t) + "," + sol_csv
            print("%.9f," % (t) + sol_csv, file=file)

            t += pausua


def idatziDC_ezl(hasiera, amaiera, pausua, sorgailua, A, M, N, T,
                 U, n, b, cir_el_luz, cir_val_luz, cir_ezl_luz, filename):
    '''
    This function completes the DC analysis of a circuit that contains non
    linear element with the given voltage source, solving the equation system
    for every voltage value for a selected range. Writes every value of the
    DC voltage source and its related solution in a document called the same
    way as the input circuit + _DCsourceName + .dc .

    Args:
        hasiera : string. A string containing an integer value which specifies
        the final value of the selected DC voltage source.
        amaiera : string. A string containing an integer value which specifies
        the initial value of the selected DC voltage source.
        pausua : string. A string containing an integer value which specifies a
        value to be added to the current value of the selected DC voltage
        source in order to complete the DC analysis.
        sorgailua : string. A string containing the name of the selected DC
        voltage source to be analyzed.
        filename : string. The name of the filename, used to created the name
        of the output document.

    Returns:
        None.
    '''

    # HEMEN LENO sorgailua.upper())[0], azpiko if a gabe zeon ta zlel2
    # ta zlel3 ko zirkuitu guztikin ondo zijoan

    posizioa = np.where(cir_el_luz == sorgailua.upper())
    if len(posizioa) == 0:
        posizioa = np.where(cir_el_luz == sorgailua.lower())
    posizioa = posizioa[0]

    # header = build_csv_header(cir_el_luz[posizioa][0].lower(), b, n)

    elementua = cir_el_luz[posizioa][0]

    header = zl2.build_csv_header(elementua[0].upper(), b, n)

    hasiera = float(hasiera)
    amaiera = float(amaiera)
    pausua = float(pausua)

    balioa = hasiera

    with open(filename[: -4] + "_" + sorgailua + ".dc", 'w') as file:

        header = zl2.build_csv_header("V", b, n)
        print(header, file=file)

        while balioa < amaiera:
            U_berria = U.copy()

            U_berria[n-1 + b + posizioa] = balioa

            soluzioa = NR(n, b, M, N, U_berria[n-1 + b:], A,
                          cir_ezl_luz, cir_val_luz)
            sol_csv = ','.join(['%.9f' % num for num in soluzioa])
            # lerroa = str(balioa) + "," + sol_csv
            print("%.9f," % (balioa) + sol_csv, file=file)

            balioa += pausua

# ================================================================


if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_Q.cir"
        # filename = "../cirs/all/2_zlel_1D.cir"
        # filename = "../cirs/all/2_zlel_2D.cir"
        filename = "../cirs/all/3_zlel_arteztailea.cir"
        filename = "../cirs/all/3_zlel_RC.cir"
        filename = "../cirs/all/2_zlel_Q_ezaugarri.cir"

#    end = time.perf_counter()
#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds

    cir_el, cir_nd, cir_val, cir_ctr, sim = cir_parser(filename)
    # cir_el, cir_nd, cir_val, cir_ctr, sim = cir_parser(filename2)
    # M, N, Us = MNUs(zl1.)
    # print("cir_el:")
    # print(cir_el)
    # print()

    # print("cir_nd:")
    # print(cir_nd)
    # print()

    # print("cir_val:")
    # print(cir_val)
    # print()

    # print("cir_ctr:")
    # print(cir_ctr)
    # print()

    # print("sim:")
    # print(sim)
    # print()

    cir_el_luz, cir_nd_luz, cir_val_luz, cir_ctr_luz = zl1.luzatu(cir_el,
                                                                  cir_nd,
                                                                  cir_val,
                                                                  cir_ctr)

    # print("cir_el_luz:")
    # print(cir_el_luz)
    # print()

    # print("cir_nd_luz:")
    # print(cir_nd_luz)
    # print()

    # print("cir_val_luz:")
    # print(cir_val_luz)
    # print()

    # print("cir_ctr_luz:")
    # print(cir_ctr_luz)
    # print()

    nodo_ezb = zl1.nodo_ezberdinak(cir_nd)

    n, b = zl1.nodo_adar_kop(cir_el_luz, nodo_ezb)
    elementu_kop = zl1.elementu_kop(cir_el)

    # print("nodo ezberdinak: ", nodo_ezb)
    # print("nodo kopurua: ", n)
    # print("adar kopurua:", b)
    # print("elementu kopurua", elementu_kop)
    # print()

    M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz, cir_ctr_luz)

    # print("M:")
    # print(M)
    # print()

    # print("N:")
    # print(N)
    # print()

    # print("Us:")
    # print(Us)
    # print()

    Aa, A = zl1.Aa_eta_A(cir_nd_luz, nodo_ezb, b, n)

    # print("A:")
    # print(A)
    # print()

    cir_ezl_luz = ez_linealak(cir_el_luz)

    # print("lista ez linealak:")
    # print(cir_ezl_luz)
    # print()

    for el in cir_ezl_luz:
        if el == 1 or el == 2:
            print("honea sartzen dao")
            # Us[0] = 5.0
            soluzioa = NR(n, b, M, N, Us, A, cir_ezl_luz, cir_val_luz)
            zl2.print_solution(soluzioa, b, n)

# =============================================================================
#             hasiera = lerroa[5]
#             amaiera = lerroa[6]
#             pausua = lerroa[7]
#             sorgailua = lerroa[8]
#             idatziDC_ezl(hasiera, amaiera, pausua, sorgailua, T, U, n, b,
#                          cir_el_luz, filename)
# =============================================================================

            break
