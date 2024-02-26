'''
.. module:: zlel_p1.py
    :synopsis: Module contains the circuit parser,
      a branch builder, info printing and some integrity checks.

'''

import numpy as np
import sys


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """ Prints the info of the circuit:
            1.- Elements info
            2.- Node info
            3.- Branch info
            4.- Variable info
    Args:
        cir_el: reshaped cir_el
        cir_nd: reshaped cir_nd. Now it will be a (b,2) matrix
        b: # of branches
        n: # number of nodes
        nodes: an array with the circuit nodes sorted
        el_num:  the # of elements.

    """
    # Element info
    print(str(el_num) + ' Elements')
    # Node info
    print(str(n) + ' Different nodes: ' +
          str(nodes))
    # Branch info
    print("\n" + str(b) + " Branches: ")

    for i in range(1, b+1):
        print("\t" + str(i) + ". branch:\t" + cir_el[i-1] +
              ",\ti" + str(i) +
              ",\tv" + str(i) +
              "=e" + str(cir_nd[i-1, 0]) +
              "-e" + str(cir_nd[i-1, 1]))

    # Variable info
    print("\n" + str(2*b + (n-1)) + " variables: ")
    # Print all the nodes but the first (0 because is sorted)
    for i in nodes[1:]:
        print("e"+str(i)+", ", end="", flush=True)
    for i in range(b):
        print("i"+str(i+1)+", ", end="", flush=True)
    # Print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b-1):
        print("v"+str(i+1)+", ", end="", flush=True)
    print("v"+str(b))


# 1) CIR_PARSERRA:
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

    Rises:
        SystemExit

    """

    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    cir_el = np.array(cir[:, 0], dtype=str)

    cir_nd = np.array(cir[:, 1:5], dtype=int)

    cir_val = np.array(cir[:, 5:8], dtype=float)

    cir_ctr = np.array(cir[:, -1], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctr, cir


# 2) Digrafoa irudikatu: LUZATU
def luzatu(cir_el, cir_nd, cir_val, cir_ctr):
    '''
    This function enlarges the cir_el, cir_nd, cir_val and cir_ctr lists in
    order to take into account al branches of the diodes and the amplifiers.


    Returns:
        cir_el_luz, cir_nd_luz, cir_val_luz and cir_ctr_luz.
        All 4 returned elements are lists again.

    '''
    # Q:
    # ezpaiteu kopiak iten elementu berdiñai iteie erreferentzia eta bat
    # aldatzebada bestea ebai aldatzea
    cir_el2 = cir_el.copy()
    cir_nd2 = cir_nd.copy()
    cir_val2 = cir_val.copy()
    cir_ctr2 = cir_ctr.copy()

    # paso aldagaia sortu behar da i!=0 danen matrize berdiñetik
    # hartzen aigealako eta lerro bat gehio gehitzedeunen
    # hasierako range(len(cir_el)) hoi baño matrize
    # haundigok lortzen goazelako
    paso = 0

    for i in range(len(cir_el)):
        if cir_el[i][0].upper() == "Q":

            if i != 0:
                cir_el2 = cir_el2[0: i + paso]
            else:
                cir_el2 = np.array([])
            # matrize hutsa behardeu, bestela aurrekoan gañen iteu append
            cir_el2 = np.append(cir_el2, cir_el[i] + "_be")
            cir_el2 = np.append(cir_el2, cir_el[i] + "_bc")
            cir_el2 = np.append(cir_el2, cir_el[i+1:])

            if i != 0:
                cir_nd2 = cir_nd2[0:i + paso, :]
            else:
                cir_nd2 = np.array([])
            lista1 = [cir_nd[i][1], cir_nd[i][2], 0, 0]
            lista2 = [cir_nd[i][1], cir_nd[i][0], 0, 0]
            cir_nd2 = np.append(cir_nd2, lista1)
            cir_nd2 = np.append(cir_nd2, lista2)
            cir_nd2 = np.append(cir_nd2, cir_nd[i + 1:, :])
            # print(len(cir_nd2))

            if i != 0:
                cir_val2 = cir_val2[0: i + paso, :]
            else:
                cir_val2 = np.array([])
            cir_val2 = np.append(cir_val2, cir_val[i])
            cir_val2 = np.append(cir_val2, cir_val[i])
            cir_val2 = np.append(cir_val2, cir_val[i + 1:, :])
            # print(len(cir_val2))

            if i != 0:
                cir_ctr2 = cir_ctr2[0: i + paso]
            else:
                cir_ctr2 = np.array([])
            cir_ctr2 = np.append(cir_ctr2, cir_ctr[i])
            cir_ctr2 = np.append(cir_ctr2, cir_ctr[i])
            cir_ctr2 = np.append(cir_ctr2, cir_ctr[i+1:])
            # print(len(cir_ctr2))

            # behin baño gehiotan balinbadao Q bat dimentsio bat baño
            # haundigoko matrizek berriro matrize forman jarri beaiteu,
            # ordun hemen reshaipeatu beaiteu eta ez return en
            cir_nd2 = cir_nd2.reshape(len(cir_el2), len(cir_nd[0]))
            cir_val2 = cir_val2.reshape(len(cir_el2), len(cir_val[0]))

            paso += 1

    # A:

    cir_el3 = cir_el2.copy()
    cir_nd3 = cir_nd2.copy()
    cir_val3 = cir_val2.copy()
    cir_ctr3 = cir_ctr2.copy()

    paso = 0

    for i in range(len(cir_el2)):
        if cir_el2[i][0].upper() == "A":

            # if cir_nd2[i][3] != 0:
            #     ref = True
            # else:
            #     ref = False

            if i != 0:
                cir_el3 = cir_el3[0: i + paso]
            else:
                cir_el3 = np.array([])
            cir_el3 = np.append(cir_el3, cir_el2[i] + "_in")
            cir_el3 = np.append(cir_el3, cir_el2[i] + "_out")
            # if ref: cir_el3 = np.append(cir_el3, cir_el[i] + "_ref")
            cir_el3 = np.append(cir_el3, cir_el2[i+1:])

            if i != 0:
                cir_nd3 = cir_nd3[0: i + paso, :]
            else:
                cir_nd3 = np.array([])
            lista1 = [cir_nd2[i][0], cir_nd2[i][1], 0, 0]
            lista2 = [cir_nd2[i][2], cir_nd2[i][3], 0, 0]
            cir_nd3 = np.append(cir_nd3, lista1)
            cir_nd3 = np.append(cir_nd3, lista2)
            # if ref:
            #    lista3 =
            #    cir_nd3 = np.append(cir_nd3, lista3)
            cir_nd3 = np.append(cir_nd3, cir_nd2[i + 1:, :])
            # print(len(cir_nd3))

            if i != 0:
                cir_val3 = cir_val3[0: i + paso, :]
            else:
                cir_val3 = np.array([])
            cir_val3 = np.append(cir_val3, cir_val2[i])
            cir_val3 = np.append(cir_val3, cir_val2[i])
            # if ref: cir_val3 = np.append(cir_val3, cir_val2[i])
            cir_val3 = np.append(cir_val3, cir_val2[i + 1:, :])
            # print(len(cir_val3))

            if i != 0:
                cir_ctr3 = cir_ctr3[0: i + paso]
            else:
                cir_ctr3 = np.array([])
            cir_ctr3 = np.append(cir_ctr3, cir_ctr2[i])
            cir_ctr3 = np.append(cir_ctr3, cir_ctr2[i])
            # if ref: cir_ctr3 = np.append(cir_ctr3, cir_ctr2[i])
            cir_ctr3 = np.append(cir_ctr3, cir_ctr2[i+1:])
            # print(len(cir_ctr3))

            cir_nd3 = cir_nd3.reshape(len(cir_el3), len(cir_nd[0]))
            cir_val3 = cir_val3.reshape(len(cir_el3), len(cir_val[0]))

            paso += 1

# -----------------------------------------------------------------------------
# AZTERKETAKO ATALA:

    # N:

    cir_el4 = cir_el3.copy()
    cir_nd4 = cir_nd3.copy()
    cir_val4 = cir_val3.copy()
    cir_ctr4 = cir_ctr3.copy()

    paso = 0

    for i in range(len(cir_el3)):
        if cir_el3[i][0].upper() == "N":

            if i != 0:
                cir_el4 = cir_el4[0: i + paso]
            else:
                cir_el4 = np.array([])
            cir_el4 = np.append(cir_el4, cir_el3[i] + "_g1")
            cir_el4 = np.append(cir_el4, cir_el3[i] + "_g2")
            # if ref: cir_el3 = np.append(cir_el3, cir_el[i] + "_ref")
            cir_el4 = np.append(cir_el4, cir_el3[i+1:])

            if i != 0:
                cir_nd4 = cir_nd4[0: i + paso, :]
            else:
                cir_nd4 = np.array([])
            lista1 = [cir_nd3[i][0], cir_nd3[i][1], 0, 0]
            lista2 = [cir_nd3[i][2], cir_nd3[i][3], 0, 0]
            cir_nd4 = np.append(cir_nd3, lista1)
            cir_nd4 = np.append(cir_nd3, lista2)

            cir_nd4 = np.append(cir_nd4, cir_nd3[i + 1:, :])

            if i != 0:
                cir_val4 = cir_val4[0: i + paso, :]
            else:
                cir_val4 = np.array([])
            cir_val4 = np.append(cir_val4, cir_val3[i])
            cir_val4 = np.append(cir_val4, cir_val3[i])
            cir_val4 = np.append(cir_val4, cir_val3[i + 1:, :])

            if i != 0:
                cir_ctr4 = cir_ctr4[0: i + paso]
            else:
                cir_ctr4 = np.array([])
            cir_ctr4 = np.append(cir_ctr4, cir_ctr3[i])
            cir_ctr4 = np.append(cir_ctr4, cir_ctr3[i])
            cir_ctr4 = np.append(cir_ctr4, cir_ctr3[i+1:])

            cir_nd4 = cir_nd4.reshape(len(cir_el4), len(cir_nd[0]))
            cir_val4 = cir_val4.reshape(len(cir_el4), len(cir_val[0]))

            paso += 1

# -----------------------------------------------------------------------------

    return cir_el4, cir_nd4, cir_val4, cir_ctr4


# 3) HAUSNARKETA: Nodo zerrenda eta adar kopurua (b)
def nodo_ezberdinak(cir_nd):
    '''
    This function returns the number of nodes of the circuit.

    Args:
        cir_nd : list

    Returns:
        list. Returns a list which contains the nodes of the circuit.

    '''
    return np.unique(cir_nd[:, 0:2])


def nodo_adar_kop(cir_el_luz, nodo_ezb):
    '''
    Returns the number of nodes and branches of the circuit.

    Args:
        cir_el_luz : list
        nodo_ezb : list

    Returns:
        integer, integer
        The first integer specifies the number of nodes of the circuit,
        and the second integer specifies the number of branches.

    '''
    return len(nodo_ezb), len(cir_el_luz)


def elementu_kop(cir_el):
    return len(cir_el)


# 4) Intzidentzia matrizea eta murriztetako intzidentzia matrizea Aa eta A
def Aa_eta_A(cir_nd_luz, nodo_ezb, adar_kop, nodo_ezb_kop):
    '''
    This function calculates the incidence matrix and the asociated incidence
    matrix. A zero-filled matrix is created and then gets the neighboring
    nodes of every branch, taking into account which one is the positive
    and which is the negative. So is the Aa matrix created. And then the
    first line is substracted (the 0 node) to get the A matrix.

    Args:
        cir_nd_luz : list
        nodo_ezb : list
        adar_kop : list
        nodo_ezb_kop : list

    Returns:
        Aa : matrix
        Asociated incidence matrix.
        A : matrix
        Incidence matrix.

    '''
    # Aa = [[0.0 for zutabe_kop in range(adar_kop)]
    #      for ilara_kop in range(nodo_ezb_kop)]
    Aa = [[0 for zutabe_kop in range(adar_kop)]
          for ilara_kop in range(nodo_ezb_kop)]
    Aa = np.array(Aa)
    for i in range(adar_kop):
        plus = cir_nd_luz[i][0].astype(np.int64)
        minus = cir_nd_luz[i][1].astype(np.int64)

        Aa[np.where(nodo_ezb == plus)[0][0]][i] = +1
        Aa[np.where(nodo_ezb == minus)[0][0]][i] = -1

    # suposatzea nodok ordenatuta daudela zenbakiagatik
    # ordun erreferentzia nodoa lenengo ilara izangoa
    # bestela, beste bat bada, zenbatgarren indizen daon jakinda
    # cir_el luzatun lerro guztik hartzeiteu hoi izan ezik
    # [0 : i, :] eta gero [i + 1 : , :] iñez eta bukaeran
    # reshape(nodo_ezb_kop - 1, adar_kop)
    A = np.array(Aa[1:, :])

    return Aa, A


# 5) ERROREEN KONTROLA
def errore_kontrola(Aa, cir, cir_el_luz, cir_nd_luz, cir_val_luz, nodo_ezb, b):
    '''
    This function checks if the given circuit contains any errors:
    A line in the .cir document does not contain 9 lines
    The reference node is not specifies
    The sum of a column of the asociated incidence matrix is not 0
    There are voltage sources of different values in parallel
    There are current sources of different values in series

    Args:
        Aa : matrix
        cir : matrix
        cir_el_luz : list
        cir_nd_luz : list
        cir_val_luz : list
        nodo_ezb : list
        b : integer. Number of branches.

    Returns:
        bool
        True is returned if the .cir document contains no errors.

    '''
    # 5a: (cir fitxategian informazio egokia)

    for lerroa in cir:
        if len(lerroa) != 9:
            sys.exit(".cir FITXATEGIKO LERRO BATEK EZ DITU 9 OSAGAI!!")

    # 5b: (erreferentzia nodoa dago.)
    # if 0 not in nodo_ezb:
    if nodo_ezb[0] != 0:
        sys.exit("Reference node \"0\" is not defined in the circuit.")

    # 5d: (ez dago balio desberdineko tentsio iturririk paraleloan)
    av = {}
    for i in range(0, b):
        if cir_el_luz[i][0].upper() == "V" or cir_el_luz[i][0].upper() == "B":
            av[cir_el_luz[i]] = [cir_nd_luz[i][0],
                                 cir_nd_luz[i][1],
                                 cir_val_luz[i][0]]
    gako_v_lista = list(av.keys())
    balio_v_lista = list(av.values())
    for j in range(len(gako_v_lista)):
        for k in range(j, len(gako_v_lista)):
            if balio_v_lista[j][0] == (balio_v_lista[k][0] and
                                       balio_v_lista[j][1] ==
                                       balio_v_lista[k][1] and
                                       balio_v_lista[j][2] !=
                                       balio_v_lista[k][2]):
                sys.exit("Parallel V sources at branches " + str(j) + " and "
                         + str(k) + ".")
                # sys.exit(str(j) + ". eta " + str(k) + ". adarretan balio \
                #         ezberdineko tentsio iturriak paraleloan")
            elif (balio_v_lista[j][1] == balio_v_lista[k][0] and
                  balio_v_lista[j][0] == balio_v_lista[k][1] and
                  balio_v_lista[j][2] != -balio_v_lista[k][2]):
                sys.exit("Parallel V sources at branches " + str(j) + " and "
                         + str(k) + ".")
                # sys.exit(str(j) + ". eta " + str(k) + ". adarretan balio\
                #         ezberdineko tentsio iturriak paraleloan")

    # 5e: (ez dago balio desberdineko korronte iturririk serian)
    # korronte elementuen nododak eta balioak ai-n gorde
    hiztegi = {}
    aztertzeko_nodoak = []
    for j in nodo_ezb:
        elementu_lista = []
        for i in range(b):
            if cir_nd_luz[i][0] == j or cir_nd_luz[i][1] == j:
                elementu_lista.append(cir_el_luz[i])
        hiztegi[j] = elementu_lista
    nodoak = list(hiztegi.keys())
    elementuak = list(hiztegi.values())
    # print('hiztegi', hiztegi)
    for i in range(len(nodoak)):
        kont = 0
        for j in range(len(elementuak[i])):
            if elementuak[i][j][0].upper() == 'I':
                kont += 1
        if kont == len(elementuak[i]):
            aztertzeko_nodoak.append(nodoak[i])

    batura = 0
    for n in range(len(aztertzeko_nodoak)):
        nn = aztertzeko_nodoak[n]
        for m in range(b):
            batura += int(Aa[nn][m])*int(cir_val_luz[m][0])
        if batura != 0:
            sys.exit("I sources in series at node "
                     + str(nn) + ".")

    # 5c: (ez dago konexio bakarreko nodorik.)
    for j in range(len(nodo_ezb)):
        batura = 0
        for i in range(b):
            batura += abs(Aa[j][i])
        if batura < 2:
            sys.exit("Node "+str(nodo_ezb[j])+" is floating.")

    return True


def print_incidence_matrix_textua(Aa):
    '''
    This function creates a string containing the incidence matrix.
    The text is then printed in the .out files which need the incidence
    matrix to be specified.

    Returns:
        textua : string
        The string contains the sentence 'Indicence Matrix' and then
        the string representation of the matrix.

    '''
    textua = "\nIncidence Matrix: \n"

    # for i in range(len(Aa)):
    #     lerroa = str(Aa[i])
    #     if i == len(Aa) - 1:
    #         textua += " " + lerroa + "]"
    #     elif i == 0:
    #         textua += "[" + lerroa + "\n"
    #     else:
    #         textua += " " + lerroa + "\n"

    textua += str(Aa)

    return textua


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:

        # filename = "../cirs/all/0_zlel_no_elements_def.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_node.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_node_float.cir"  # ONDO

        # filename = "../cirs/all/0_zlel_OPAMP.cir"  # ONDO, baño .out hutsik
        # filename = "../cirs/all/0_zlel_parallel_V_I.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_parallel_V_II.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_parallel_V_III.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_parallel_V_IV.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_parallel_V_V.cir"  # ONDO, .out hutsik

        # filename = "../cirs/all/0_zlel_serial_I_I.cir"  # ONDO
        # filename = "../cirs/all/0_zlel_serial_I_II.cir" # ONDO
        # filename = "../cirs/all/0_zlel_serial_I_III.cir" # ONDO, .out hutsik
        # filename = "../cirs/all/0_zlel_serial_I_IV.cir" # ONDO, .out hutsik
        # filename = "../cirs/all/0_zlel_serial_I_V.cir" # ONDO
        # filename = "../cirs/all/0_zlel_serial_I_VI.cir" # ONDO, .out hutsik
        # filename = "../cirs/all/0_zlel_serial_I_VII.cir" # ONDO, .out hutsik

        # filename = "../cirs/all/0_zlel_tabs.cir" # ONDO, .out hutsik
        # filename = "../cirs/all/0_zlel_V_R.cir" # ONDO, .out hutsik
        # filename = "../cirs/all/0_zlel_V_R_Q.cir" # ONDO, .out hutsik

        # Parse the circuit
        [cir_el, cir_nd, cir_val, cir_ctr, cir] = cir_parser(filename)
        [cir_el_luz, cir_nd_luz, cir_val_luz, cir_ctr_luz] = luzatu(
            cir_el, cir_nd, cir_val, cir_ctr)
        nodo_ezb = nodo_ezberdinak(cir_nd)
        n, b = nodo_adar_kop(cir_el_luz, nodo_ezb)
        el_kop = elementu_kop(cir_el)

        Aa, A = Aa_eta_A(cir_nd_luz, nodo_ezb, b, n)

        print_cir_info(cir_el_luz, cir_nd_luz, b, n, nodo_ezb, el_kop)

    #    print()
    #    print("Incidence Matrix :")
    #    print(Aa)
    #    print(cir_val_luz)
    #    print(cir_nd_luz)
    #    print(cir_nd[4][0])
    #    print(cir_el_luz)

        errore_kontrola(Aa, cir, cir_el_luz, cir_nd_luz, cir_val_luz,
                        nodo_ezb, b)
