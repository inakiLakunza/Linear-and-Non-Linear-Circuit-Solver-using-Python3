#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
.. module:: zlel_p4.py
    :synopsis: This module adds to zlel_p3 the ability
     to solve dynamic circuits (capacitor or inductor).
     It is based on the Euler-backwards method.

'''


import numpy as np
import sys
if __name__ == "zlel.zlel_p4":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
    import zlel.zlel_p3 as zl3
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2
    import zlel_p3 as zl3


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
        cir_ctr: np array of strings with the element which branch
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
    sim = []
    for i in range(len(cir)):
        if cir[i][0][0] == ".":
            pos = i
            break

    if pos != "a":
        cir, sim = cir[: pos, :], cir[pos:, :]

    cir_el = np.array(cir[:, 0], dtype=str)

    cir_nd = np.array(cir[:, 1:5], dtype=int)

    cir_val = np.array(cir[:, 5:8], dtype=float)

    cir_ctr = np.array(cir[:, -1], dtype=str)

    return cir, cir_el, cir_nd, cir_val, cir_ctr, sim


def C_edo_L(cir_el_luz):
    '''
    This functions identifies dynamic (time dependent) elements of the circuit.
    It assigns 0 to non dynamic elements, 1 to capacitors and 2 to inductors.

    Args:
        cir_el_luz : np array of strings with the elements of the circuit.

    Returns:
        cir_din_luz : np array with the elements to identify dynamic
        elements in the circuit.

    '''

    cir_din_luz = np.zeros(len(cir_el_luz), dtype=float)

    for i in range(len(cir_el_luz)):
        letra = cir_el_luz[i][0].upper()

        if letra == "C":
            cir_din_luz[i] = 1

        elif letra == "L":
            cir_din_luz[i] = 2

    return cir_din_luz


def EB(n, b, M, N, Us, A, cir_el_luz, cir_din_luz,
       cir_ezl_luz, cir_val_luz, sim, filename):
    '''
    This function solves a circuit that contains dynamic elements using the
    method of Euler Backwards. It calls to NR function defined in zlel_p3
    that solves the circuit for non linear circuits.

    Args:
        n : int. Number of nodes.
        b : int. Number of branches.
        M : matrix. Contains the voltage related multipliers of each element.
        N : matrix. Contains the current related multipliers of each element.
        Us : matrix. Contains the independent voltage and current sources'
        multipliers.
        A : matrix. Incidence matrix.
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_din_luz : np array with the elements to identify dynamic
        elements in the circuit.
        cir_ezl_luz : np array with the elements to identificate non linear
        elements in the circuit.
        cir_val_luz : np array with the values of each element.
        filename : string. The name of the filename, used to created the name
        of the output document.
        sim: np array of a list of strings with the firt the type of simulation
        as the first element and its values. size(1,9).

    Returns:
        soluzioa : list. The function returns the w vector, containing the
        value of every variable of the system.

    '''

    CL_lista = [0 for i in range(len(cir_din_luz))]
    Vc_lista = [0 for i in range(len(cir_din_luz))]
    Ic_lista = [0 for i in range(len(cir_din_luz))]

    M_ber = M.copy()
    N_ber = N.copy()
    Us_ber = Us.copy()

    for i in range(len(cir_din_luz)):
        if cir_din_luz[i] == 1:
            Vc_lista[i] = cir_val_luz[i][1]
            CL_lista[i] = cir_val_luz[i][0]
        if cir_din_luz[i] == 2:
            Ic_lista[i] = cir_val_luz[i][1]
            CL_lista[i] = cir_val_luz[i][0]

    for lerroa in sim:
        if lerroa[0].lower() == ".tr":
            hasiera = float(lerroa[5])
            amaiera = float(lerroa[6])
            h = float(lerroa[7])
            break

    t = hasiera

    if (1 in cir_ezl_luz) or (2 in cir_ezl_luz):
        ezlin = True
    else:
        ezlin = False

    with open(filename[: -3] + "tr", 'w') as file:

        header = zl2.build_csv_header("t", b, n)
        print(header, file=file)

        U = np.zeros(n-1 + 2*b, dtype=float)

        while t < amaiera:

            U[n-1 + b:] = Us_ber
            U_ber = zl2.t_akt(cir_el_luz, cir_val_luz, t, U, n, b)

            Us_ber = U_ber[n-1 + b:]

            for i in range(len(cir_din_luz)):

                if cir_din_luz[i] == 1:

                    M_ber[i][i] = 1
                    N_ber[i][i] = -h/CL_lista[i]

                    # print("Vc_lista[i]",Vc_lista[i])
                    Us_ber[i] = Vc_lista[i]

                elif cir_din_luz[i] == 2:

                    N_ber[i][i] = 1
                    M_ber[i][i] = -h/CL_lista[i]

                    Us_ber[i] = Ic_lista[i]

            T, U_ber = zl2.TU(n, b, M_ber, N_ber, Us_ber, A)

            if ezlin:
                soluzioa = zl3.NR(n, b, M_ber, N_ber, Us_ber, A,
                                  cir_ezl_luz, cir_val_luz)
            else:
                soluzioa = zl2.OPsoluzioa(T, U_ber)

            sol_Vc = soluzioa[n - 1: n + b - 1]
            sol_Ic = soluzioa[n + b - 1:]
            Vc_lista = VIc_akt(Vc_lista, sol_Vc)
            Ic_lista = VIc_akt(Ic_lista, sol_Ic)

            sol_csv = ','.join(['%.9f' % num for num in soluzioa])
            # lerroa = str(t) + "," + sol_csv
            print("%.9f," % (t) + sol_csv, file=file)
            # print()
            # print()
            # print(sol_csv, "eta t:", t)
            # print()
            # print()

            t += h

    return soluzioa


def VIc_akt(VIc_lista, sol):
    '''
    This function updates the values of the list given with the new values
    provided.

    Args:
        VIc_lista : list. Values to be updated
        sol : list. New values.

    Returns:
        lista_ber : list. It contains updated values.

    '''

    lista_ber = VIc_lista.copy()

    for i in range(len(VIc_lista)):
        if VIc_lista[i] > -9000:
            lista_ber[i] = sol[i]

    return lista_ber

# =========================================================


if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # filename = "../cirs/all/2_zlel_Q.cir"
        # filename = "../cirs/all/2_zlel_1D.cir"
        # filename = "../cirs/all/2_zlel_2D.cir"
        # filename = "../cirs/all/3_zlel_arteztailea.cir"
        # filename = "../cirs/all/3_zlel_RC.cir"
        # filename = "../cirs/all/3_zlel_RL.cir"
        # filename = "../cirs/all/3_zlel_RLC.cir"
        # filename = "../cirs/all/3_zlel_arteztailea.cir"
        # filename = "../cirs/all/3_zlel_RC.cir"
        # filename = "../cirs/all/2_zlel_arteztailea.cir"
        filename = "../cirs/all/3_zlel_arteztailea.cir"

#    end = time.perf_counter()
#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds

    cir, cir_el, cir_nd, cir_val, cir_ctr, sim = cir_parser(filename)
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

    cir_ezl_luz = zl3.ez_linealak(cir_el_luz)

    # print("lista ez linealak:")
    # print(cir_ezl_luz)
    # print()

    cir_din_luz = C_edo_L(cir_el_luz)

    # print("lista dinamikoak:")
    # print(cir_din_luz)
    # print()

    for el in cir_din_luz:
        # if el == 1 or el == 2:

        soluzioa = EB(n, b, M, N, Us, A, cir_el_luz, cir_din_luz, cir_ezl_luz,
                      cir_val_luz, sim, filename)

        # print("soluzioa", soluzioa)

        zl2.print_solution(soluzioa, b, n)

        zl2.plot_from_cvs(filename[: -3] + "tr", "t", "v2", "proba")

        # break
