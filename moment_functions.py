import math
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

def calc_Tmatrixal_imNormalized_transform_OTRS(Tpq, stdftpara, No, Mo):
	Ns = stdftpara[2]
	Omg = stdftpara[4]
	al = stdftpara[5]
	crlx = stdftpara[6]
	crly = stdftpara[7]
  
	pOrdPQ = 2
	Aoi, Boi, Coi = calc_TABCne(1, No, pOrdPQ)
	Asn, Bsn, Csn = calc_TABCne(1, Ns, pOrdPQ)

	x0 = (Tpq[0, 1] - Boi[1] * Tpq[0, 0]) / (Aoi[1] * Tpq[0, 0])
	y0 = (Tpq[1, 0] - Boi[1] * Tpq[0, 0]) / (Aoi[1] * Tpq[0, 0])
    
	# om 
	if Omg == 0:
		om = 1 # no scale normalization
	else:
		om = math.sqrt(Omg / (No * Tpq[0, 0]))
	
	# tt1 = 0
	mu1 = (2.0 * ((No - 1.0) / 2.0 - x0) * ((No - 1.0) / 2.0 - y0) * Aoi[1] * Tpq[0, 0] + 2.0 * ((No - 1) / 2.0 - x0) * Tpq[1, 0] + 2.0 * ((No - 1.0) / 2.0 - y0) * Tpq[0, 1] + 2.0 * Tpq[1, 1] / Aoi[1])
	mu2 = ((((No - 1.0) / 2.0 - x0) * ((No - 1.0) / 2.0 - x0) - ((No - 1.0) / 2.0 - y0) * ((No - 1) / 2.0 - y0)) * Aoi[1] * Tpq[0, 0] - 2.0 * ((No - 1.0) / 2.0 - y0) * Tpq[1, 0] + 2.0 * ((No - 1.0) / 2.0 - x0) * Tpq[0, 1] - Tpq[2, 0] / Aoi[2] + Tpq[0, 2] / Aoi[2])

	if (mu1 == 0) and (mu2 == 0):
		tt1t = 0.0
	else:
		tt1t = math.atan(mu1 / mu2) / 2.0

	tt1 = tt1t

	if mu2 < 0:
		tt1 = tt1 + math.pi / 2

	# final adj for tt1 so that Tpq(q=0,p=3) >= 0. Use recurrence formula...
	TRSmat = np.array([[om * math.cos(tt1), om * math.sin(tt1), -x0 * om * math.cos(tt1) - y0 * om * math.sin(tt1) + (Ns - 1) / 2],
					   [-om * math.sin(tt1), om * math.cos(tt1), x0 * om * math.sin(tt1) - y0 * om * math.cos(tt1) + (Ns - 1) / 2],
					   [0.0, 0.0, 1.0]])
	tempTRSmat = TRSmat.ravel()
	TRSmatrix = np.append(tempTRSmat,[Ns, Ns, al])
	TRSpq = TcIN_ScTrInv_TRS_Tpq_vrec_v3(Tpq, 3, TRSmatrix, No, Mo)

	if TRSpq[0, 3] < 0:
		# increase tt1 by pi and recalculate TRSpq for later use.
		tt1 = tt1 + math.pi

	TRSmat = np.array([[om * math.cos(tt1), om * math.sin(tt1), -x0 * om * math.cos(tt1) - y0 * om * math.sin(tt1) + (Ns - 1) / 2 + crlx],
						   [-om * math.sin(tt1), om * math.cos(tt1), x0 * om * math.sin(tt1) - y0 * om * math.cos(tt1) + (Ns - 1) / 2 + crly],
						   [0, 0, 1]])
	tempTRSmat = TRSmat.ravel()
	tempMat = np.array([Ns, Ns, al])
	TRSmatrix = np.concatenate((tempTRSmat, tempMat), axis = None)

	return TRSmatrix

def calc_tnx_x_ordn(N, ordn):
    tnx = np.zeros((N, int(ordn + 1)))
    tnx[0, 0] = 1.0 / math.sqrt(N)
    tnx[N - 1, 0] = tnx[0, 0]
    tnx[1, 0 ] = 1.0 / math.sqrt(N)
    tnx[N - 2, 0] = tnx[1, 0]

    for n in range(1,int(ordn)+1):
        tnx[0, n] = -tnx[0, n - 1] * math.sqrt(((N - n) * (2.0 * n + 1.0)) / ((N + n) * (2.0 * n - 1.0)))
        tnx[N - 1, n] = ((-1.0) ** (n)) * tnx[0, n]
        tnx[1, n] = tnx[0, n] * ( 1.0 + (n * (n + 1.0)) / (1.0 - N))
        tnx[N - 2, n]=((-1.0) ** n) * tnx[1, n]

    for x in range(2, int(round(N / 2)) + 1):
        for n in range(0, int(ordn) + 1):
            tnx[x, n] = ((-n * (n + 1) - (2 * x - 1) * (x - N - 1) - x) / (x * (N - x))) * tnx[x-1, n] + ((x - 1) * (x - N - 1)) / (x * (N - x)) * tnx[x - 2, n]
            tnx[N - 1 - x, n] = ((-1.0) ** n) * tnx[x, n]
    return tnx

def calc_Tpq_Ordpq_itr_v3(f):
    M, N = f.shape[:2]
    tpx = calc_tnx_x_ordn(N, N - 1)
    
    if N == M:
        tqy = tpx
    else:
        tqy = calc_tnx_x_ordn(M, M - 1)

    # for q in range(0, OrdPQ + 1):
    #     for p in range(0, OrdPQ - q + 1):
    #         for y in range(0, M):
    #             for x in range(0, N):
    #                 Tpq[q, p] = Tpq[q, p] + f[y, x] * tqy[y, q] * tpx[x, p]
    
    Tpq = np.zeros((M, N))
    Tpq = np.matmul(tqy.transpose(), np.matmul(f, tpx))
    
    return Tpq

def calc_TABCne(e, N, ordn):
    Aen = np.zeros((int(ordn) + 1))
    Bn = np.zeros((int(ordn) + 1))
    Cn = np.zeros((int(ordn) + 1))
    Aen[0] = 1.0 / math.sqrt(N)

    Bn[1] = - (N - 1.0) / 1 * math.sqrt((4.0 * (1.0 ** 2) - 1.0) / (N ** 2 - 1.0 ** 2))
    Aen[1] = - 2.0 * e * Bn[1] / (N - 1.0)
    for n in range(2, int(ordn) + 1):
        Bn[n] = - (N - 1.0) / n * math.sqrt((4.0 * (n ** 2) - 1.0) / (N ** 2 - n ** 2))
        Aen[n] = - 2.0 * e * Bn[n] / (N - 1.0)
        Cn[n] = - (n - 1.0) / n * math.sqrt(((2.0 * n + 1.0) * (N ** 2 - (n - 1.0) ** 2)) / ((2.0 * n - 3.0) * (N ** 2 - n ** 2)))

    return Aen, Bn, Cn

def imTransform_Tmatrix_sub(f, Tmatrix):
    Tr = np.zeros((2, 3))
    Tr[0,:] = Tmatrix[0: 3]
    Tr[1,:] = Tmatrix[3: 6]
    Ns = Tmatrix[9]
    Ms = Tmatrix[10]
    Ns = int(Ns)
    Ms = int(Ms)
    fout = cv2.warpAffine(f, Tr, (Ms,Ns))
    return(fout)

def TcIN_ScTrInv_TRS_Tpq_vrec_v3(Tpq, OrdPQ, Tmatrix, No, Mo):
    a1 = Tmatrix[0]
    a2 = Tmatrix[1]
    a3 = Tmatrix[2]
    b1 = Tmatrix[3]
    b2 = Tmatrix[4]
    b3 = Tmatrix[5]
    Ns = Tmatrix[9]
    alp = Tmatrix[11]
    
    ScTrInvpq = np.zeros((int(OrdPQ) + 1, int(OrdPQ) + 1))
    Aoi, Boi, Coi = calc_TABCne(1, No, OrdPQ + 2)
    Asn, Bsn, Csn = calc_TABCne(1, Ns, OrdPQ + 2)
    ialpqij = np.zeros((int(OrdPQ) + 1 + 2, int(OrdPQ) + 1 + 2, int(OrdPQ) + 1, int(OrdPQ) + 1))
    
    # p=0, q=0, i=0, j=0
    ialpqij[1, 1, 0, 0] = math.sqrt((No * No) / (Ns * Ns))
    
    # p=1, q=0       
    ialpqij[1,1,0,1] = math.sqrt((No * No) / (Ns * Ns)) * Asn[1] * (a3 + (a1 + a2) * (No - 1.0) / 2.0 - (Ns - 1.0) / 2.0)
    ialpqij[1,2,0,1] = math.sqrt((No * No) / (Ns * Ns)) * (a1 * Asn[1] / Aoi[1])
    ialpqij[2,1,0,1] = math.sqrt((No * No) / (Ns * Ns)) * (a2 * Asn[1] / Aoi[1])
    
    # p=0, q=1     
    ialpqij[1,1,1,0] = math.sqrt((No * No) / (Ns * Ns)) * Asn[1] * (b3 + (b1 + b2) * (No - 1.0) / 2.0 - (Ns - 1.0) / 2.0)
    ialpqij[1,2,1,0] = math.sqrt((No * No) / (Ns * Ns)) * (b1 * Asn[1] / Aoi[1])
    ialpqij[2,1,1,0] = math.sqrt((No * No) / (Ns * Ns)) * (b2 * Asn[1] / Aoi[1])
    
    p = 1
    q = 1
    for i in range(0, int(p + q) + 1):
        for j in range(0, int(p + q - i) + 1):
            ialpqij[j + 1, i + 1, q, p] = - ialpqij[j + 1, i + 2, q - 1, p] * b1 * Asn[q] / Aoi[i + 2] * Coi[i + 2] - ialpqij[j + 2, i + 1, q - 1, p] * b2 * Asn[q] / Aoi[j + 2] * Coi[j + 2] + ialpqij[j + 1, i + 1, q - 1, p] * Asn[q] * (b3 + (b1 + b2) * (No - 1) / 2 - (Ns - 1) / 2) + ialpqij[j + 1, i - 1 +1, q - 1 ,p] * b1 * Asn[q] / Aoi[i] + ialpqij[j - 1 + 1, i + 1, q -1 , p] * b2 * Asn[q] / Aoi[j]
            
    for p in range(0, 2):
        for q in range(2, int(OrdPQ - p) + 1):
            for i in range(0, int(p + q) + 1):
                for j in range(0, int(p + q - i) + 1):
                    ialpqij[j + 1, i + 1, q, p] = - ialpqij[j + 1, i + 1 + 1, q - 1, p] * b1 *Asn[q] / Aoi[i + 2] * Coi[i + 2] - ialpqij[j + 1 +1, i + 1, q - 1, p] * b2 * Asn[q] / Aoi[j + 2] * Coi[j + 2] + ialpqij[j + 1, i + 1, q - 1, p] * Asn[q] * (b3 + (No - 1) / 2 * (b1 + b2) - (Ns - 1) / 2) + ialpqij[j + 1, i - 1 + 1, q - 1, p] * b1 * Asn[q] / Aoi[i] + ialpqij[j - 1 + 1, i + 1, q - 1, p] *b2 * Asn[q] / Aoi[j] + ialpqij[j + 1, i + 1, q -2 , p] * Csn[q]
                    
	# by rows using EQA
	# p from 2 to OrdPQ-q
    for q in range(0,int(OrdPQ)+1):
        for p in range(2, int(OrdPQ-q)+1):
            for i in range(0, int(p+q)+1):
                for j in range(0,int(p+q-i)+1):
                    ialpqij[j+1,i+1, q, p] = - ialpqij[j+1,i+1+1,q,p-1]*a1*Asn[p]/Aoi[i+2]*Coi[i+2] - ialpqij[j+1+1,i+1,q, p-1]*a2*Asn[p]/Aoi[j+2]*Coi[j+2] + ialpqij[j+1,i+1,q,p-1]*Asn[p]*(a3+(No-1)/2*(a1+a2)-(Ns-1)/2) + ialpqij[j+1,i-1+1,q,p-1]*a1*Asn[p]/Aoi[i] + ialpqij[j-1+1,i+1,q,p-1]*a2*Asn[p]/Aoi[j] + ialpqij[j+1,i+1,q,p-2]*Csn[p]
	#End loop

    for q in range(0,int(OrdPQ)+1):
        for p in range(0, int(OrdPQ-q)+1):
            for i in range(0, int(p+q)+1):
                for j in range(0,int(p+q-i)+1):
                    ScTrInvpq[q,p] = ScTrInvpq[q,p] + ialpqij[j+1,i+1, q, p]*Tpq[j,i]

    ScTrInvpq = ScTrInvpq * abs(a1 * b2 - a2 * b1)
    if (alp > 0):
        g = (alp / ScTrInvpq[0, 0])
        ScTrInvpq = ScTrInvpq * g
    return ScTrInvpq

def params1(paths, stdftpara, leaf_class, csv_output_path_filename, g_output_path):
    OrdPQ = int(stdftpara[8])
    i = 0
    
    # name of the output CSV file + path
    csv_output_path = csv_output_path_filename
    
    # folder for the generated g image
    img_output_path = g_output_path
    
    for file in tqdm(paths):
        f = cv2.imread(file, 0)
        No, Mo = f.shape[:2]
        old_size = f.shape[:2]
        
        if No > Mo:
            ratio = float(No) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img = cv2.resize(f, (new_size[1], new_size[0]), interpolation = cv2.INTER_AREA)
            delta_w = No - new_size[1]
            delta_h = No - new_size[0]
            top, bottom = int(delta_h / 2), delta_h - int(delta_h / 2)
            left, right = int(delta_w / 2), delta_w - int(delta_w / 2)
            new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
            Mo, No = new_img.shape[:2]
            f = cv2.resize(new_img, (Mo, No), interpolation = cv2.INTER_AREA) 
        elif Mo > No:
            ratio = float(Mo) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img = cv2.resize(f, (new_size[1], new_size[0]), interpolation = cv2.INTER_AREA)
            delta_w = Mo - new_size[1]
            delta_h = Mo - new_size[0]
            top, bottom = int(delta_h / 2), delta_h - int(delta_h / 2)
            left, right = int(delta_w / 2), delta_w - int(delta_w / 2)
            new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
            Mo, No = new_img.shape[:2]
            f = cv2.resize(new_img, (Mo, No), interpolation = cv2.INTER_AREA)
        else:
            f = cv2.resize(f, (Mo, No), interpolation = cv2.INTER_AREA)
        
        # get normalization parameters
        Tpq = calc_Tpq_Ordpq_itr_v3(f)
        Tmatrix = calc_Tmatrixal_imNormalized_transform_OTRS(Tpq, stdftpara, No, Mo)
        
        # save invariance into csv
        path, filename = os.path.split(file)
        label = filename

        # generate g for observation
        Inm1 = TcIN_ScTrInv_TRS_Tpq_vrec_v3(Tpq, OrdPQ, Tmatrix, No, Mo)
        g = imTransform_Tmatrix_sub(f, Tmatrix)
        # saving g
        newfilename = f"g_{filename}"
        output_path = f"{img_output_path}/{newfilename}"
        cv2.imwrite(output_path, g) 

        # saving to csv file
        if i == 0:
            featureName = []
            for counter1 in range(0, len(Inm1)):
                for counter2 in range(0, counter1 + 1):
                    featureName.append("Inm[" + str(counter2) + "," + str(counter1 - counter2) + "]")
            featureName.append('label') # image name
            featureName.append('class') # class name
            with open(csv_output_path, "w", newline='') as output:
                outputfeature = csv.writer(output)
                outputfeature.writerow(map(lambda x: x, featureName))
            i = i + 1
            
        feature = []
        
        for counter1 in range(0, len(Inm1)):
            for counter2 in range(0, counter1 + 1):
                feature.append(Inm1[counter2][counter1 - counter2])
        
        feature.append(label)
        feature.append(leaf_class)
        with open(csv_output_path,"a+", newline='') as output:
            outputfeature = csv.writer(output)
            outputfeature.writerow(map(lambda x: x, feature))