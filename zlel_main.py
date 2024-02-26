
import sys
from zlel import zlel_p1 as zl1
from zlel import zlel_p2 as zl2
from zlel import zlel_p3 as zl3
from zlel import zlel_p4 as zl4

# import math
import numpy as np
# import matplotlyb.pyplot as plt


# FUNTZIOAK --------------------------------------------------

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

    # IT IS RECOMMENDED TO USE THIS FUNCTION WITH NO MODIFICATION.


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/examples/0_zlel_V_R_Q.cir"

#        filename = "cirs/all/1_zlel_adibide_op.cir"  # eztoa ondo
#        filename = "cirs/all/1_zlel_anpli.cir"  # ONDO
#        filename = "cirs/all/1_zlel_B_op_tr.cir"  # ONDO
#        filename = "cirs/all/1_zlel_ekorketa.cir"  # ONDO
#        filename = "cirs/all/1_zlel_OPAMP.cir"  # ONDO
#        filename = "cirs/all/1_zlel_OPAMP_E_G_op.cir"  # GAIZKI
#        filename = "cirs/all/1_zlel_parallel_BV_I.cir"  # ONDO
#        filename = "cirs/all/1_zlel_parallel_V_I.cir" # ONDO
#        filename = "cirs/all/1_zlel_serial_YI_VI.cir" # ONDO (konprobatu)
#        filename = "cirs/all/1_zlel_V_R_op_dc.cir" # ONDO

#        filename = "cirs/all/2_zlel_1D.cir" # ONDO
#        filename = "cirs/all/2_zlel_2D.cir" # ONDO, konprobatu
#        filename = "cirs/all/2_zlel_arteztailea.cir" # ONDO
#        filename = "cirs/all/2_zlel_Q.cir" # ONDO
#        filename = "cirs/all/2_zlel_Q_ezaugarri.cir" # ONDO

#        filename = "cirs/all/3_zlel_arteztailea.cir" # EZIN DA KONPROBATU
#        filename = "cirs/all/3_zlel_RC.cir" # ONDO
#        filename = "cirs/all/3_zlel_RC_iragazki.cir" # ONDO
#        filename = "cirs/all/3_zlel_RL.cir" # ONDO
#        filename = "cirs/all/3_zlel_RLC.cir"  # ONDO

# -----------------------------------------------------------------------------

#        filename = "cirs/all/0_zlel_no_elements_def.cir"  # ONDO
#        filename = "cirs/all/0_zlel_node.cir"  # ONDO
#        filename = "cirs/all/0_zlel_node_float.cir"  # ONDO

#        filename = "cirs/all/0_zlel_OPAMP.cir"  # ONDO, baño .out hutsik
#        filename = "cirs/all/0_zlel_parallel_V_I.cir"  # ONDO
#        filename = "cirs/all/0_zlel_parallel_V_II.cir"  # ONDO
#        filename = "cirs/all/0_zlel_parallel_V_III.cir"  # ONDO
#        filename = "cirs/all/0_zlel_parallel_V_IV.cir"  # ONDO
#        filename = "cirs/all/0_zlel_parallel_V_V.cir"  # ONDO, .out hutsik

#        filename = "cirs/all/0_zlel_serial_I_I.cir"  # ONDO
#        filename = "cirs/all/0_zlel_serial_I_II.cir" # ONDO
#        filename = "cirs/all/0_zlel_serial_I_III.cir" # ONDO, .out hutsik
#        filename = "cirs/all/0_zlel_serial_I_IV.cir" # ONDO, .out hutsik
#        filename = "cirs/all/0_zlel_serial_I_V.cir" # ONDO
#        filename = "cirs/all/0_zlel_serial_I_VI.cir" # ONDO, .out hutsik
#        filename = "cirs/all/0_zlel_serial_I_VII.cir" # ONDO, .out hutsik

#        filename = "cirs/all/0_zlel_tabs.cir" # ONDO, .out hutsik
#        filename = "cirs/all/0_zlel_V_R.cir" # ONDO, .out hutsik
#        filename = "cirs/all/0_zlel_V_R_Q.cir" # ONDO, .out hutsik

# -----------------------------------------------------------------------------
# AZTERKETAKO ATALA:

        filename = "cirs/all/4_zlel_N.cir"

# -----------------------------------------------------------------------------

    # Parse the circuit
    # [cir_el,cir_nd,cir_val,cir_ctr]=cir_parser(filename)
    # zl1.cir_parser(filename)

    cir, cir_el, cir_nd, cir_val, cir_ctr, sim = zl4.cir_parser(filename)
#   cir_el, cir_nd, cir_val, cir_ctr, sim = cir_parser(filename2)
#   M, N, Us = MNUs(zl1.)

    # print("cir:")
    # print(cir)
    # print()

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

    # print(cir_el_luz)
    # print()
    # print(cir_nd_luz)
    # print()
    # print(cir_val_luz)
    # print()
    # print(cir_ctr_luz)

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

    if (zl1.errore_kontrola(Aa, cir, cir_el_luz,
                            cir_nd_luz, cir_val_luz, nodo_ezb, b)):

        # print("A:")
        # print(A)
        # print()

        cir_ezl_luz = zl3.ez_linealak(cir_el_luz)

        # print("lista ez linealak:")
        # print(cir_ezl_luz)
        # print()

        cir_din_luz = zl4.C_edo_L(cir_el_luz)

        # print("lista dinamikoak:")
        # print(cir_din_luz)
        # print()

        # print("sim:")
        # print(sim)
        # print()

        if 1.0 in cir_din_luz or 2.0 in cir_din_luz:

            # dinamikoa bada badakigu .tr egin beharko dugula soilik

            soluzioa = zl4.EB(n, b, M, N, Us, A, cir_el_luz, cir_din_luz,
                              cir_ezl_luz, cir_val_luz, sim, filename)

            # print("soluzioa", soluzioa)

            for lerroa in sim:
                eginkizuna = lerroa[0].upper()

                # beti in beharkoa eta goran indeu, ordun honei kasoik ez
                if eginkizuna == ".TR":
                    continue

                if eginkizuna == ".PR":
                    zl1.print_cir_info(cir_el_luz, cir_nd_luz, b, n,
                                       nodo_ezb, elementu_kop)
                    print(zl1.print_incidence_matrix_textua(Aa),
                          flush=True)

                elif eginkizuna == ".OP":

                    if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
                        soluzioa = zl3.NR(n, b, M, N, Us, A, cir_ezl_luz,
                                          cir_val_luz, iter_max=100)
                        zl2.print_solution(soluzioa, b, n)

                    else:

                        M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
                                            cir_ctr_luz)
                        T, U = zl2.TU(n, b, M, N, Us, A)
                        soluzioa = zl2.OPsoluzioa(T, U)
                        zl2.print_solution(soluzioa, b, n)

        else:

            if len(sim) != 0:

                for lerroa in sim:
                    lerroa[0] = lerroa[0].upper()

                eginkizunak = sim[:, 0]

            else:
                eginkizunak = []

            if ".PR" in eginkizunak:
                zl1.print_cir_info(cir_el_luz, cir_nd_luz, b, n,
                                   nodo_ezb, elementu_kop)
                print(zl1.print_incidence_matrix_textua(Aa),
                      flush=True)

            if ".OP" in eginkizunak:
                if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
                    soluzioa = zl3.NR(n, b, M, N, Us, A, cir_ezl_luz,
                                      cir_val_luz, iter_max=100)
                    zl2.print_solution(soluzioa, b, n)

                else:
                    M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
                                        cir_ctr_luz)
                    T, U = zl2.TU(n, b, M, N, Us, A)
                    soluzioa = zl2.OPsoluzioa(T, U)
                    zl2.print_solution(soluzioa, b, n)

            if ".DC" in eginkizunak:
                indizea = np.where(".DC" == eginkizunak)[0][0]
                lerroa = sim[indizea]

                hasiera = lerroa[5]
                amaiera = lerroa[6]
                pausua = lerroa[7]
                sorgailua = lerroa[8]

                M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
                                    cir_ctr_luz)
                T, U = zl2.TU(n, b, M, N, Us, A)

                if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
                    zl3.idatziDC_ezl(hasiera, amaiera, pausua,
                                     sorgailua, A, M, N, T, U, n, b,
                                     cir_el_luz, cir_val_luz,
                                     cir_ezl_luz, filename)
                else:
                    zl2.idatziDC(hasiera, amaiera, pausua,
                                 sorgailua, T, U, n, b,
                                 cir_el_luz, filename)

            if ".TR" in eginkizunak:
                indizea = np.where(".TR" == eginkizunak)[0][0]
                lerroa = sim[indizea]
                hasiera = lerroa[5]
                amaiera = lerroa[6]
                pausua = lerroa[7]

                M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
                                    cir_ctr_luz)
                T, U = zl2.TU(n, b, M, N, Us, A)

                if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
                    zl3.idatziTR_ezl(hasiera, amaiera, pausua, A, M, N,
                                     T, U, n, b, cir_el_luz, cir_val_luz,
                                     cir_ezl_luz, filename)
                else:
                    zl2.idatziTR(hasiera, amaiera, pausua, T, U, n, b,
                                 cir_el_luz, cir_val_luz, filename)

            # ez bada dinamikoa begiratu zer egiteko eskatzen diguten

            # for lerroa in sim:
            #     eginkizuna = lerroa[0].upper()

            #     if eginkizuna == ".PR":
            #         zl1.print_cir_info(cir_el_luz, cir_nd_luz, b, n,
            #                            nodo_ezb, elementu_kop)
            #         print(zl1.print_incidence_matrix_textua(Aa),
            #               flush=True)

            #     elif eginkizuna == ".OP":

            #         if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
            #             soluzioa = zl3.NR(n, b, M, N, Us, A, cir_ezl_luz,
            #                               cir_val_luz, iter_max=100)
            #             zl2.print_solution(soluzioa, b, n)

            #         else:

            #             M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
            #                                 cir_ctr_luz)
            #             T, U = zl2.TU(n, b, M, N, Us, A)
            #             soluzioa = zl2.OPsoluzioa(T, U)
            #             zl2.print_solution(soluzioa, b, n)

            #     elif eginkizuna == ".DC":
            #         hasiera = lerroa[5]
            #         amaiera = lerroa[6]
            #         pausua = lerroa[7]
            #         sorgailua = lerroa[8]

            #         M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
            #                             cir_ctr_luz)
            #         T, U = zl2.TU(n, b, M, N, Us, A)

            #         if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
            #             zl3.idatziDC_ezl(hasiera, amaiera, pausua,
            #                              sorgailua, A, M, N, T, U, n, b,
            #                              cir_el_luz, cir_val_luz,
            #                              cir_ezl_luz, filename)
            #         else:
            #             zl2.idatziDC(hasiera, amaiera, pausua,
            #                          sorgailua, T, U, n, b,
            #                          cir_el_luz, filename)

            #     elif eginkizuna == ".TR":
            #         hasiera = lerroa[5]
            #         amaiera = lerroa[6]
            #         pausua = lerroa[7]

            #         M, N, Us = zl2.MNUs(b, cir_el_luz, cir_val_luz,
            #                             cir_ctr_luz)
            #         T, U = zl2.TU(n, b, M, N, Us, A)

            #         if 1.0 in cir_ezl_luz or 2.0 in cir_ezl_luz:
            #             zl3.idatziTR_ezl(hasiera, amaiera, pausua, A, M, N,
            #                              T, U, n, b, cir_el_luz, cir_val_luz,
            #                              cir_ezl_luz, filename)
            #         else:
            #             zl2.idatziTR(hasiera, amaiera, pausua, T, U, n, b,
            #                          cir_el_luz, cir_val_luz, filename)
