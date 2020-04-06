from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt, rcParams
import re
from os import path, getcwd, listdir, makedirs
import sys
from matplotlib.font_manager import FontProperties

def smooth_pr(prec, rec):
    """
    Smooths precision recall curve according to TREC standards. Evaluates max precision at each 0.1 recall. Makes the curves look nice and not noisy
    """

    n = len(prec)
    m = 11
    p_smooth = np.zeros((m), dtype=np.float)
    r_smooth = np.linspace(0.0, 1.0, m) 
    for i in range(m):
        j = np.argmin( np.absolute(r_smooth[i] - rec) ) + 1
        p_smooth[i] = np.max( prec[:j] )

    return p_smooth, r_smooth

def check_match(im_lab_k, db_lab, num_include):
    """
    Check if im_lab_k and db_lab are a match, i.e. the two images are less than or equal to
    num_include frames apart. The correct num_include to use depends on the speed of the camera, both for frame rate as well as physical moving speed.
    """	
    if num_include == 1:
        if db_lab ==im_lab_k:
            return True
    else:
        # This assumes that db_lab is a string of numerical characters, which it should be	
        #print int(db_lab)-num_include/2, "<=", int(im_lab_k), "<=", int(db_lab)+num_include/2, "?"
        if (int(db_lab)-num_include/2) <= int(im_lab_k) and int(im_lab_k) <= (int(db_lab)+num_include/2):
            return True

    return False

def computeForwardPass(model, im): 
    ###########################################
    #I tried to pass model as a parameter but it failed:
    #                           'module' object is not callable
     #######################################
    """
    Compute the forward pass for the model
    """
    t0 = time.time()
    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC).reshape(1,1,120,160)
    im = torch.from_numpy(im).to(device=device,dtype=torch.float)
    #im = torch.from_numpy(im).float()
    with torch.no_grad():
        
        output = model(im)
        output /= np.linalg.norm(output.cpu())
        descriptor = output
        t_calc = (time.time() - t0)

    return descriptor, t_calc

def get_prec_recall(model,data_path="test_data/CampusLoopDataset", num_include=7, title='Precision-Recall Curve'):
    """
    Input: 
    data_path="test_data/CampusLoopDataset",: Path to data with corresponding images (with corresponding file names) in a <data-dir>/live and <data-dir>/memory
    """
    '''
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    '''
    database = [] # stored pic descriptors
    database_labels = [] # the image labels	

    mem_path = data_path + "\\memory"
    live_path = data_path + "\\live"

    print ("memory path: ", mem_path)
    print ("live path: ", live_path)

    mem_files = [path.join(mem_path, f) for f in listdir(mem_path)]
    live_files = [path.join(live_path, f) for f in listdir(live_path)]

    # same HOG params used to train calc
    hog = cv2.HOGDescriptor((16, 32), (16,16), (16,16), (8,8), 2,1)
    db_hog = []

    t_calc = []

    for fl in mem_files:
        im = cv2.imread(fl)
        print ("loading image ", fl, " to database")
        descriptor, t_r = computeForwardPass(model, im)
        '''
        ################################
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if im.shape[2] > 1:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC)
    
    
        with torch.no_grad():
            t0 = time.time()
            output = model(im)
            output /= np.linalg.norm(output)
            descriptor = output
            tr = (time.time() - t0)
        ################################    
        '''
        t_calc.append(t_r)
        database.append(descriptor)
        database_labels.append(re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1))
        d_hog = hog.compute(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY), (160, 120), interpolation = cv2.INTER_CUBIC))
        db_hog.append(d_hog/np.linalg.norm(d_hog))
        #print(d_hog)
        #db_hog.append(d_hog)

    #print database

    correct = np.zeros((len(live_files),1),dtype=np.uint8) # the array of true labels of loop closure for precision-recall curve for each net
    scores = np.zeros((len(live_files),1))  # Our "probability function" that  simply uses 1-l2_norm

    correct_hog = []
    scores_hog = []
    
    k=0
    t_q = []
    for fl in live_files:
        im_label_k = re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1)
        im = cv2.imread(fl)

        descriptor, t_r = computeForwardPass(model, im)
        '''
        ################################
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if im.shape[2] > 1:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC)
    
    
        with torch.no_grad():
            t0 = time.time()
            output = model(im)
            output /= np.linalg.norm(output)
            descriptor = output
            tr = (time.time() - t0)
        ################################
        '''
        t_calc.append(t_r)

        d_hog = hog.compute(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160, 120), interpolation = cv2.INTER_CUBIC))
        d_hog /= np.linalg.norm(d_hog)

        max_sim = -1.0 
        max_sim_hog = -1.0

        i_max_sim = -1 * np.ones(1, dtype=np.int32)
        i_max_sim_hog = -1
        t_q_single = 0.0
        for i in range(len(database)):
            #print(descriptor.dtype)
            #print(database[i].dtype)
            #print(descriptor.shape)
            #print(database[i].shape)
            #print(descriptor.squeeze())
            #print(database[i].squeeze())
            #curr_sim = torch.dot(descriptor.squeeze(), database[i].squeeze())# Normalizd vectors means that this give cosine similarity
            t0 = time.time()
            curr_sim = np.dot(descriptor.cpu(), database[i].cpu().T)
            print(curr_sim)
            if curr_sim > max_sim: 
                max_sim = curr_sim
                i_max_sim = i
            t1 = time.time()

            curr_sim_hog = np.squeeze(np.dot(d_hog.T,  db_hog[i])) 
            if curr_sim_hog > max_sim_hog: 
                max_sim_hog = curr_sim_hog
                i_max_sim_hog = i
            t_q_single+=(t1-t0)
            #print(t1-t0)
        
        #scores[k] = max_sim.cpu()
        t0 = time.time()
        scores[k] = max_sim
        db_lab = database_labels[i_max_sim]  
        if check_match(im_label_k, db_lab, num_include):
            correct[k] = 1
            # else already 0
        #print(len(database_labels))
        #print(correct.shape)
        #print(i_max_sim)
        #print(k)
        t1 = time.time()
        t_q_single+=(t1-t0)
        t_q.append(t_q_single)
        print(t_q_single)
        print ("Proposed match calc:", im_label_k, ", ", database_labels[i_max_sim], ", score = ", max_sim, ", Correct =", correct[k])

        scores_hog.append( max_sim_hog )
        db_lab_hog = database_labels[i_max_sim_hog]  
        if check_match(im_label_k, db_lab_hog, num_include):
            correct_hog.append(1)
        else:
            correct_hog.append(0)
        print ("Proposed match HOG:", im_label_k, ", ", database_labels[i_max_sim_hog], ", score = ", max_sim_hog,", Correct =", correct_hog[-1])

        print ("\n")
        k += 1

    precisions = []
    recalls = []
    threshold = -1.0

    precision, recall, threshold = precision_recall_curve(correct, scores)
    precisions.append(precision)
    recalls.append(recall)
    # Only get threshold if there's one net. Otherwise we're just coparing them and don't care about a threshold yet
    perf_prec = abs(precision[:-1] - 1.0) <= 1e-6
    #print(precision)
    #print(perf_prec)
    if np.any(perf_prec):
        # We want the highest recall rate with perfect precision as our a-priori threshold
        threshold = np.min(threshold[perf_prec]) # get the largest threshold so that presicion is 1
        #print(threshold.shape)
        print ("\nThreshold for max recall with 1.0 precision = %f" % (threshold) )

    precision_hog, recall_hog, thresholds_hog = precision_recall_curve(correct_hog, scores_hog)
    t_calc = np.asarray(t_calc)
    t_q = np.asarray(t_q)
    #print(t_calc.shape)
    print ("Mean calc compute time = ", np.sum(t_calc)/ np.size(t_calc))
    print ("Mean query compute time = ", np.sum(t_q)/ np.size(t_q))
    

    return precisions, recalls,precision_hog, recall_hog

def plot(model, data_path="test_data/CampusLoopDataset", num_include=7, title='Precision-Recall Curve'):
    """
    Plot the precision recall curve to compare CALC to other methods, or cross validate different iterations of a CALC model.
    """
        
    t0 = time.time()

    precisions, recalls, precision_hog, recall_hog = get_prec_recall(model,data_path, num_include, title)

    rcParams['font.sans-serif'] = 'DejaVu Sans'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'	
    rcParams['axes.labelweight'] = 'bold'	
    rcParams['axes.labelsize'] = 'large'	
    rcParams['figure.figsize'] = [8.0, 4.0]	
    rcParams['figure.subplot.bottom'] = 0.2	
    plots = path.join(getcwd(), "plots")
    rcParams['savefig.directory'] = plots
    if not path.isdir(plots):
        makedirs(plots)

    lines = ['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']

    ax = plt.gca()
    best_auc = -1 
    lab_best_auc = ""
    handles = []
    p_smooth, r_smooth = smooth_pr(precisions, recalls)
    curr_auc = auc(r_smooth, p_smooth)
    label = ' (AUC=%0.2f)' % (curr_auc)

    calc_plt, = ax.plot(r_smooth, p_smooth, '-', label=label, linewidth=2)

    handles.append(calc_plt)
    if curr_auc > best_auc:
        lab_best_auc = label
        best_auc = curr_auc	
    # Only tell the user the most accurate net if they loaded more than one!
    if len(precisions) > 1:
        print("Model with highest AUC:", lab_best_auc)


    print ("\n\n\n\n")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title(title)

    p_smooth, r_smooth = smooth_pr(precision_hog, recall_hog)
    lab = 'HOG (AUC=%0.2f)' % (auc(r_smooth, p_smooth))
    hog_plt, = ax.plot(r_smooth, p_smooth, '-.', label=lab, linewidth=2)
    handles.append(hog_plt)
    fontP = FontProperties()
    fontP.set_size('small')
    leg = ax.legend(handles=handles, fancybox=True, ncol = (1 + int(len(precisions)/30)), loc='best', prop=fontP)
    leg.get_frame().set_alpha(0.5) # transluscent legend 
    leg.set_draggable(True)
    for line in leg.get_lines():
        line.set_linewidth(3)

    print ("Elapsed time = ", time.time()-t0, " sec")
    plt.show()