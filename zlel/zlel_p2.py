'''
.. module:: zlel_p2.py
    :synopsis: This module resolves linear resistant
     circuits. New elements have been introduced. It is
     also able to sweep parameters. Solves circuit for .op, .dc and .tr
     simulations.

'''

import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__ == "zlel.zlel_p2":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


def print_solution(sol, b, n):
    """ This function prints the solution with format.

        Args:
            sol: np array with the solution of the Tableau equations
            (e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b)
            b: # of branches
            n: # of nodes

    """

    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=' ')  # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind] = np.array(sol[ind])
        sol = tmp
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        print("e" + str(i) + " = ", sol[i-1])
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        print("v" + str(i) + " = ", sol[i+n-2])
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        print("i" + str(i) + " = ", sol[i+b+n-2])

    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """ This function build the csv header for the output files.
        First column will be v or i if .dc analysis or t if .tr and it will
        be given by argument tvi.
        The header will be this form,
        t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b

    Args:
        tvi: "v" or "i" if .dc analysis or "t" if .tran
        b: # of branches
        n: # of nodes

    Returns:
        header: The header in csv format as string

    """

    header = tvi
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b+1):
        header += ",v" + str(i)
    for i in range(1, b+1):
        header += ",i" + str(i)
    return header


def save_as_csv(b, n, filename):
    """ This function gnerates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file.

    Args:
        b: # of branches
        n: # of nodes
        filename: string with the filename (incluiding the path)

    """
    # Sup .tr
    header = build_csv_header("t", b, n)
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        t = 0
        while t < 10:
            # for t in tr["start"],tr["end"],tr["step"]
            # Recalculate the Us for the sinusoidal sources

            sol = np.full(2*b+(n-1), t+1, dtype=float)
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)
            t = t + 1


def plot_from_cvs(filename, x, y, title):
    """ This function plots the values corresponding to the x string of the
        file filename in the x-axis and the ones corresponding to the y
        string in the y-axis.
        The x and y strings must mach with some value of the header in the
        csv file filename.

    Args:
        filename: string with the name of the file (including the path).
        x: string with some value of the header of the file.
        y: string with some value of the header of the file.

    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=0,
                         skip_footer=1, names=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data[x], data[y], color='r', label=title)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    plt.show()


def idatziDC(hasiera, amaiera, pausua, sorgailua,
             T, U, n, b, cir_el_luz, filename):
    '''

    This function completes the DC analysis of a selected voltage source,
    solving the equation system for every voltage value for a selected range.
    Writes every value of the DC voltage source and its related solution in
    a document called the same way as the input circuit +
    _DCsourceName + .dc .

    Args:
        hasiera : string
            A string containing an integer value which specifies the
            final value of the selected DC voltage source.
        amaiera : string
            A string containing an integer value which specifies the
            initial value of the selected DC voltage source.
        pausua : string
            A string containing an integer value which specifies a value
            to be added to the current value of the selected DC voltage
            source in order to complete the DC analysis.
        sorgailua : string
            A string containing the name of the selected DC voltage
            source to be analyzed.
        filename : string
            The name of the filename, used to created the name
            of the output document.

    Returns:
        None.

    '''

    cir_el_luz_upper = []
    for el in cir_el_luz:
        cir_el_luz_upper.append(el.upper())

    posizioa = np.where(str(cir_el_luz_upper) == str(sorgailua).upper())[0]

    for i in range(len(cir_el_luz)):
        if cir_el_luz_upper[i] == sorgailua.upper():
            posizioa = i

    elementua = cir_el_luz[posizioa][0]

    header = build_csv_header(elementua[0].upper(), b, n)

    hasiera = float(hasiera)
    amaiera = float(amaiera)
    pausua = float(pausua)

    balioa = hasiera

    with open(filename[: -4] + "_" + sorgailua + ".dc", 'w') as file:

        print(header, file=file)

        soluzioa = None

        while balioa < amaiera:
            U_berria = U.copy()

            # print("U:")
            # print(U)
            # print()

            # print("balioa", balioa)
            U_berria[n-1 + b + posizioa] = balioa
            # print("U berria:")
            # print(U_berria)
            # print()
            # print()
            # print("U_berria")
            # print(U_berria)
            soluzioa = OPsoluzioa(T, U_berria)
            sol_csv = ','.join(['%.9f' % num for num in soluzioa])
            # lerroa = str(balioa) + "," + sol_csv
            print("%.9f," % (balioa) + sol_csv, file=file)

            balioa += pausua


def idatziTR(hasiera, amaiera, pausua,
             T, U, n, b, cir_el_luz, cir_val_luz, filename):
    '''
    This function analizes the time evolution of a given circuit.
    Solving the equation system for every time value for a selected range.
    Writes every value of the time and its related solution in
    a document called the same way as the input circuit + .tr .

    Args:
        hasiera : string
            A string containing an integer value which specifies the
            final value of the selected time range.
        amaiera : string
            A string containing an integer value which specifies the
            initial value of the selected time range.
        pausua : string
            A string containing an integer value which specifies a value
            to be added to the current time in order to complete the TR
            analysis.

    Returns:
        None.

    '''

    hasiera = float(hasiera)
    amaiera = float(amaiera)
    pausua = float(pausua)

    t = hasiera

    with open(filename[: -3] + "tr", 'w') as file:

        header = build_csv_header("t", b, n)
        print(header, file=file)

        while t < amaiera:
            U_berria = t_akt(cir_el_luz, cir_val_luz, t, U, n, b)

            soluzioa = OPsoluzioa(T, U_berria)
            sol_csv = ','.join(['%.9f' % num for num in soluzioa])

            print("%.9f," % (t) + sol_csv, file=file)

            t += pausua


def t_akt(cir_el_luz, cir_val_luz, t, U, n, b):
    '''
    This function actualizes the value of each element of the circuit
    given the actual analyzed time t.


    Returns:
        U_berria : matrix
        The returned U matrix is the time actualized U matrix.

    '''

    U_berria = U.copy()
    t_elementuak = {}

    for i in range(len(cir_el_luz)):
        if cir_el_luz[i][0].upper() == "B" or cir_el_luz[i][0].upper() == "Y":
            anp = cir_val_luz[i][0]
            maiz = cir_val_luz[i][1]
            fasea = cir_val_luz[i][2]

            t_elementuak[i] = [anp, maiz, fasea]

    for ind in t_elementuak:
        lista = t_elementuak.get(ind)

        bere_anp = lista[0]
        bere_maiz = lista[1]
        bere_fasea = lista[2]

        bere_bal = bere_anp*np.sin(2*np.pi*bere_maiz*t + np.pi/180*bere_fasea)

        U_berria[n-1 + b + ind] = bere_bal

    return U_berria


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
        cir, sim = cir[: pos, :], cir[pos:, :]

    cir_el = np.array(cir[:, 0], dtype=str)

    cir_nd = np.array(cir[:, 1:5], dtype=int)

    cir_val = np.array(cir[:, 5:8], dtype=float)

    cir_ctr = np.array(cir[:, -1], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctr, sim


def MNUs(b, cir_el_luz, cir_val_luz, cir_ctr_luz):
    '''
    This function calculates the M, N and Us matrixes of the circuit.
    The M matrix will contain the voltage related multipliers of every element.
    The N matrix will contain the current related multipliers of every element.
    The Us matrix will contain the independent voltage and current sources'
    multipliers.
    This matrixes will then be used to calculate the T and U matrixes which
    are necessary to get the soultion of the circuit.

    Returns:
        M : matrix
        The M matrix of the circuit, containing the voltage multipliers,
        its size will be the branch number squared.
        N : matrix
        The N matrix of the circuit, containing the current multipliers,
        its size will be the branch number squared.
        Us : list
        The Us matrix of the circuit, containing the independent source
        multipliers its size will be the branch number.

    '''

    M = np.zeros((b, b), dtype=float)
    N = np.zeros((b, b), dtype=float)
    Us = np.zeros((b), dtype=float)

    # print("M N eta Us")
    # print(M)
    # print(N)
    # print(Us)

    # print()
    # print(type(M[0][0]))
    # print(M[0][0] + 0.2)

    for i in range(b):

        letra = cir_el_luz[i][0].upper()
        balioa = cir_val_luz[i][0]

        if letra == "V" or letra == "B":
            # v = bal

            M[i][i] = 1
            Us[i] = balioa

        elif letra == "H":
            # H_i = rm*ij

            jh = np.where(str(cir_ctr_luz[i][0]).upper() ==
                          str(cir_el_luz[0][0]).upper())[0]

            M[i][i] = 1
            N[i][jh] = -balioa

        elif letra == "I" or letra == "Y":
            # i = bal

            N[i][i] = 1.
            Us[i] = balioa

        elif letra == "F":
            # i = b*ij

            jf = np.where(str(cir_ctr_luz[i][0]).upper() ==
                          str(cir_el_luz[0][0]).upper())[0]

            N[i][i] = 1
            N[i][jf] = -balioa

        elif letra == "R":
            # v -R*i = 0

            M[i][i] = 1.
            N[i][i] = -balioa

        elif letra == "G":
            # G_i = a*V_j

            jg = np.where(str(cir_ctr_luz[i][0]).upper() ==
                          str(cir_el_luz[0][0]).upper())[0]

            N[i][i] = 1
            M[i][jg] = -balioa

        elif letra == "E":
            # dudan nao hola dan edo ez
            # E_i - bal * v_j = 0

            # kontrol elementua zein dan beidtau behar da cir_ctr_luzatun
            # eta gero kontrol elementua ze indizetan daon cir_el_luz en
            je = np.where(str(cir_ctr_luz[i][0]).upper() ==
                          str(cir_el_luz[0][0]).upper())[0]

            M[i][i] = 1
            M[i][je] = -balioa

        elif letra == "A":
            letra2 = cir_el_luz[i][-1].upper()

            if letra2 == "N":  # A_in
                N[i][i] = 1
                continue

            elif letra2 == "T":  # A_out

                M[i][i-1] = 1
                continue

            else:
                print("Zerbait gaizki doa, matrizeak ez daude ondo luzatuta")

# -----------------------------------------------------------------------------
# AZTERKETAKO ATALA:

        elif letra == "N":
            zenbakia = cir_el_luz[i][-1].upper()

            r_pi = cir_val_luz[i][0]
            beta = cir_val_luz[i][1]
            r_0 = cir_val_luz[i][2]

            if zenbakia == "1":  # g1
                N[i][i] = -1
                M[i][i] = 1.0/r_pi
                N[i][i+1] = -(beta + 1.0)/beta
                continue

            elif zenbakia == "2":  # g2

                M[i][i] = -1
                M[i][i-1] = -beta*r_0/r_pi
                N[i][i] = r_0
                continue


# -----------------------------------------------------------------------------

    return M, N, Us


def TU(n, b, M, N, Us, A):
    '''
    This function uses the M, N and Us matrixes to get the T and U matrixes,
    this matrixes are used later to solve the system, due to the fact that
    T*w = U, w being a vector containing the system variables.

    Returns:
        T : matrix
        T matrix of the system, its size is 2b + n-1 squared. It is an
        arrangement of the M and N matrixes.
        U : list
        U matrix of the system, its size is 2b + n-1. It is just the an
        enlargement of the Us matrix, containing just zeros at the first
        indices of the list.

    '''

    tam = n-1 + b + b
    T = np.zeros((tam, tam), dtype=float)
    U = np.zeros(tam, dtype=float)

    # lenengo ilarak eta gero zutabek
    T[: n-1,  n-1 + b:] = A
    T[n-1: n-1 + b, : n-1] = -np.transpose(A)
    T[n-1: n-1 + b, n-1: n-1 + b] = np.eye(b)
    T[n-1 + b:, n-1: n-1 + b] = M
    T[n-1 + b:, n-1 + b:] = N

    U[n-1 + b:] = Us

    return T, U


def OPsoluzioa(T, U):
    '''
    This functions solves the system by solving the T*w = U equation,
    w being a vector containing the system variables.

    Returns:
        soluzioa : list
        The function returns the w vector, containing the value of
        every variable of the system.

    '''

    # T_ald = np.linalg.inv(T) # ezta alderantzizkoakin in behar,
    # zuzenen T kin in behar da

    soluzioa = None

    try:
        soluzioa = np.linalg.solve(T, U)
    except np.linalg.LinAlgError:
        sys.exit("Error solving Tableau equations, check if det(T) != 0.")
    return soluzioa

# ====================================================================


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_V_R_op_dc.cir"
        filename2 = "../cirs/all/1_zlel_B_op_tr.cir"
        filename = filename2
        filename = "../cirs/all/1_zlel_adibide_op.cir"
        filename = "../cirs/all/1_zlel_anpli.cir"
#        filename = "../cirs/all/1_zleL_ekorketa.cir"
        filename = "../cirs/all/1_zlel_OPAMP_E_G_op.cir"

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

    cir_el_luz, cir_nd_luz, cir_val_luz, cir_ctr_luz = zl1.luzatu(
        cir_el, cir_nd, cir_val, cir_ctr)

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

    M, N, Us = MNUs(b, cir_el_luz, cir_val_luz, cir_ctr_luz)

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

    T, U = TU(n, b, M, N, Us, A)

    # print("T:")
    # print(T)
    # print()

    # print("U:")
    # print(U)
    # print()

    # -------------------------------------------

    zl1.print_cir_info(cir_el, cir_nd, b, n, nodo_ezb, elementu_kop)
    # print()
    # print()

    for lerroa in sim:
        mota = lerroa[0][1:].upper()

        if mota == "PR":
            zl1.print_cir_info(cir_el, cir_nd, b, n, nodo_ezb, elementu_kop)

        elif mota == "OP":
            soluzioa = OPsoluzioa(T, U)
            print_solution(soluzioa, b, n)

        elif mota == "DC":

            hasiera = lerroa[5]
            amaiera = lerroa[6]
            pausua = lerroa[7]
            sorgailua = lerroa[8]

            idatziDC(hasiera, amaiera, pausua,
                     sorgailua, T, U, n, b, cir_el_luz, filename)

        elif mota == "TR":

            hasiera = lerroa[5]
            amaiera = lerroa[6]
            pausua = lerroa[7]

            idatziTR(hasiera, amaiera, pausua,
                     T, U, n, b, cir_el_luz, cir_val_luz, filename)
