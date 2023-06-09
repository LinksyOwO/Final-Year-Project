from prettytable import PrettyTable
from datetime import datetime
import numpy as np
import os
from moment_functions import *

def main():
    # stdftpara = [feopn tropn Ns Ms al crlx crly OrdPQ]
    #  0: feopn : feature option
    #  1: tropn : translation/rotation/scale option
    #  2: Ns    : transformed image size N
    #  3: Ms    : transformed image size M
    #  4: Omg   : mass of image(for scaling) -> big Omg; no scale if Omg = 0
    #  8: al    : alpha
    #  9: crlx  : skew parameter (horizontally)
    # 10: crly  : skew parameter (vertically)
    # 11: OrdPQ : order
    
    feopn = 0
    tropn = 0
    Ns = 256
    Ms = 256
    Omg = Ns * Ms * 256 / 11.4
    al = -1
    crlx = 0
    crly = 0
    OrdPQ = 20

    stdftpara = np.array([feopn, tropn, Ns, Ms, Omg, al, crlx, crly, OrdPQ])
    print(f"[INFO] Started at {datetime.now()}")

    #### PARAMETERS ####
    # showing the value of parameters that is using
    table = PrettyTable()
    table.field_names = ["Parameter", "Value"]
    params = {
        "Ns": i, "Ms": Ms, "OrdPQ": OrdPQ, "Alpha": al, "crlx": crlx, "crly": crly, "Omg": Omg
    }
    for k, v in params.items():
        table.add_row([k, v])
    print(table)

    #### leaf classes ####
    leaf_classes = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    
    #### constants paths ####
    DATASET_NAME = "Cassava"
    BASE_IMG_PATH = "dataset/cassava/"
    BASE_OUTPUT_PATH = "output/"
    CSV_OUTPUT_PATH = f"{BASE_OUTPUT_PATH}/moment/{DATASET_NAME}/"
    G_OUTPUT_PATH = f"{BASE_OUTPUT_PATH}/g/{DATASET_NAME}"

    for p in [BASE_OUTPUT_PATH, G_OUTPUT_PATH]:
        if os.path.exists(p) is False:
            os.makedirs(p)

    # creating output folders
    for output_type in ["moment", "g"]:
        new_folder = f"{BASE_OUTPUT_PATH}/{output_type}/{DATASET_NAME}/"
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    # creating output subfolders for g
    for split in ["images"]:
        for c in ["CBB", "CBSD", "CGM", "CMD", "Healthy"]:
            new_folder = f"{G_OUTPUT_PATH}/{split}/{c}/"
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

    ######## train images ########
    CBB_PATH = f"{BASE_IMG_PATH}/test/CBB/"
    CBB_test_img_paths = [CBB_PATH + img for img in os.listdir(CBB_PATH)]
    CBSD_PATH = f"{BASE_IMG_PATH}/test/CBSD/"
    CBSD_test_img_paths = [CBSD_PATH + img for img in os.listdir(CBSD_PATH)]
    CGM_PATH = f"{BASE_IMG_PATH}/test/CGM/"
    CGM_test_img_paths = [CGM_PATH + img for img in os.listdir(CGM_PATH)]
    CMD_PATH = f"{BASE_IMG_PATH}/test/CMD/"
    CMD_test_img_paths = [CMD_PATH + img for img in os.listdir(CMD_PATH)]
    Healthy_PATH = f"{BASE_IMG_PATH}/test/Healthy/"
    Healthy_test_img_paths = [Healthy_PATH + img for img in os.listdir(Healthy_PATH)]

    test_img_paths = [CBB_test_img_paths, CBSD_test_img_paths, CGM_test_img_paths, CMD_test_img_paths, Healthy_test_img_paths]

    for path, leaf in zip(test_img_paths, leaf_classes):
        print(f"[INFO] Generating Inm for {leaf} test dataset...")

        # path and filename of output csv file
        csv_path_filename = f"{CSV_OUTPUT_PATH}/{DATASET_NAME}_test_{leaf}_{str(OrdPQ)}.csv"
        g_path = f"{G_OUTPUT_PATH}/images/{leaf}/"
        print(f"[INFO] CSV output filename : {csv_path_filename.replace('//', '/')}")

        params1(paths = path, 
                stdftpara = stdftpara, 
                leaf_class = f"{leaf}",
                csv_output_path_filename = csv_path_filename, 
                g_output_path = g_path)
        print(f"[INFO] Completed at {datetime.now()}")
        print()
main()