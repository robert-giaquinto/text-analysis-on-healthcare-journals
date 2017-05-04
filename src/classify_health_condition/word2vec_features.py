from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def load_feats(infile):
    """
    load the features dataset and give it appropriate column labels
    """
    df = pd.read_csv(infile, sep='\t', header=None)
    num_feats = df.shape[1] - 1
    col_names = ['site_id'] + ['vec' + str(t) for t in range(num_feats)]
    df.columns = col_names
    return df


def load_hc(infile):
    col_names = ['site_id', 'health_condition']
    df = pd.read_csv(infile, sep='\t', header=None, names=col_names)
    return df


def load_data(feat_file, hc_file):
    """
    load the two data files and merge them
    """
    feats = load_feats(feat_file)
    hcs = load_hc(hc_file)
    df = pd.merge(feats, hcs, on=["site_id"], how="inner", sort=False)
    return df



def split_data(df, test_size=0.25, random_seed=None):
    """
    split the data, but set aside the site_id in case we want that later
    """
    # split data into x (data features) and y (health condition)
    feats = [c for c in list(df.columns.values) if c != 'health_condition' and c != 'site_id']
    x = df[feats].values
    y = df['health_condition'].values

    # call function to split observations (e.g. 66% of data in train, 33% in test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_seed)    
    return x_train, y_train, x_test, y_test


def run_grid_search(X, y, model, param_grid, cv=5, n_jobs=1, verbose=0):
    """
    Train a model for a range of model parameters (i.e. see how model
    performs depending on how complex you allow the model to become).
    """
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose,  return_train_score=False)
    grid_search.fit(X, y)
    res = grid_search.cv_results_
    print("\nAverage percent of labels correctly classified on hold-out set during cross-validation:")
    for ps, sco, sd, t in zip(res['params'], res['mean_test_score'], res['std_test_score'], res['mean_fit_time']):
        print("Params:", ps, "\tScore:", round(sco, 3), "\t(stdev: " + str(round(sd, 3)) + "), Time to fit:\t", t, "seconds")

    print("Model that generalizes best found to use:", grid_search.best_params_)
    return grid_search.best_estimator_


def split_score(cancer, rest, x, y):
    pred = split_predict(cancer, rest, x)
    correct = np.where(pred == y, 1, 0)
    return correct.mean()


def split_predict(cancer, rest, x, verbose=False):
    # first predict if cancer or not
    cancer_predict = cancer.predict(x)
    if verbose:
        print("Cancer predictions:", pd.Series(cancer_predict).value_counts(normalize=True))

    rest_predict = rest.predict(x)
    if verbose:
        print("Rest predictions:",  pd.Series(rest_predict).value_counts(normalize=True))

    predict = np.where(cancer_predict == "Cancer", cancer_predict, rest_predict)
    if verbose:
        print("Overall Predictions:",  pd.Series(predict).value_counts(normalize=True))
        
    return predict


class similarity_model(object):
    def __init__(self):
        self.cancer = np.array([-0.0606563, 0.0130836, 0.131903, 0.0880681, 0.070882, -0.0359067, 0.0189428, -0.0889493, 0.0704393, -0.0748408, 0.132926, -0.157643, -0.0199808, -0.152181, -0.0461791, 0.0469248, -0.00838161, 0.0908028, -0.0373463, 0.0867815, 0.115667, 0.0739574, 0.0223364, -0.205454, 0.00924462, 0.109079, 0.0606385, -0.0616589, -0.121724, -0.0874695, 0.153075, 0.0690347, -0.0449132, -0.0769086, -0.0587966, -0.103632, 0.0101909, 0.0408025, -0.0218681, 0.0762998, -0.0182491, 0.0387515, -0.125144, -0.110429, -0.133232, 0.0253915, -0.0829147, -0.0312877, 0.10412, 0.0541596, -0.252284, -0.0688221, -0.00235467, 0.117086, 0.087381, -0.0737143, -0.0829349, -0.0740999, 0.0135943, -0.0145437, 0.202859, 0.0232982, -0.104059, 0.167307, 0.0756085, 0.0401989, -0.144599, 0.0300884, 0.121822, 0.0341148, -0.0387681, 0.0677834, 0.0856763, 0.0387525, 0.00679546, 0.185884, 0.10252, -0.042978, -0.0472467, 0.132834, -0.20864, 0.0608536, 0.0488104, 0.000601321, 0.236569, 0.264995, -0.0556295, 0.115524, -0.125346, 0.10047, 0.13247, 0.0154633, 0.00752782, -0.129965, 0.0145959, 0.0475351, -0.102175, 0.0304291, 0.19562, -0.0795821]).reshape((100,1))
        self.surgery = np.array([0.095918, 0.00925408, 0.211078, -0.0309924, -0.0387739, -0.155901, 0.0209568, 0.120832, -0.0414775, 0.00592986, -0.0912102, -0.048366, -0.215252, 0.0157281, -0.00994849, 0.156225, -0.163441, 0.0556315, -0.182209, 0.106327, -0.0408048, -0.0160603, -0.0815995, -0.0752255, -0.155342, 0.0792019, -0.0348346, -0.054769, -0.2251, 0.193243, 0.0783537, -0.0305473, -0.0412046, 0.0851416, -0.0620916, -0.136717, -0.00784571, 0.0181888, 0.112167, -0.0515455, -0.0623433, 0.0707778, -0.0280333, 0.0662785, -0.0643411, -0.156227, -0.0178407, 0.0672092, -0.0828034, -0.0462697, -0.174167, 0.06725, 0.152687, -0.0680572, 0.00917163, -0.028466, 0.141724, 0.0263675, 0.000548022, -0.137579, 0.103347, 0.000901481, -0.0972089, -0.173502, -0.0443603, 0.0380504, 0.0644844, 0.175082, 0.198882, -0.0675338, -0.071102, 0.0748716, -0.181447, -0.054485, -0.128235, 0.0750936, 0.117334, -0.0548028, -0.108929, 0.0895443, -0.0490501, 0.113407, -0.0650783, 0.0433868, -0.105386, 0.0291952, 0.032979, 0.114027, 0.0534178, 0.120577, 0.141225, -0.0290344, 0.0661949, 0.00668953, 0.0882629, -0.0575236, 0.0587363, -0.2147, 0.0826349, 0.0151735]).reshape((100,1))
        self.transplant = np.array([-0.138258, -0.0775432, 0.112159, -0.0501603, 0.0245495, 0.0195862, -0.0209274, 0.0317686, 0.0385108, 0.0254526, -0.0212016, -0.000835881, -0.15993, 0.130778, 0.037726, 0.277603, -0.227029, 0.0367056, -0.136036, 0.0967795, -0.000114658, -0.109563, 0.0135095, -0.0937545, 0.0104258, 0.144789, -0.0603575, -0.184386, -0.200896, -0.102634, 0.13612, -0.231863, -0.133212, 0.142595, 0.0794412, 0.13186, -0.046374, -0.0605256, 0.00131116, 0.00834693, -0.0166573, 0.0658747, -0.0959472, -0.0379167, 0.0429536, -0.120853, -0.10978, -0.0683594, 0.0608955, -0.0381849, -0.123489, 0.181965, 0.164597, 0.0672009, 0.000537814, -0.0047964, -0.0118223, -0.0281023, -0.0723965, -0.065823, 0.0507777, 0.0847029, -0.0116621, -0.126583, -0.0440917, 0.00604468, 0.0281578, 0.134064, 0.0782646, 0.122682, -0.0220208, 0.0227712, -0.0556163, 0.071895, -0.148971, -0.00398414, 0.0402061, -0.0744181, -0.00390267, -0.0461923, 0.0530825, -0.0073085, -0.00707837, 0.134386, -0.0498572, 0.0580078, -0.0230416, 0.0222454, 0.0242848, 0.182017, 0.304188, 0.054849, 0.148514, -0.0780579, 0.154405, 0.103795, -0.0517556, -0.0556574, 0.0483912, -0.0398789]).reshape((100,1))
        self.injury = np.array([0.134233, 0.0871405, -0.083378, 0.0148646, 0.133622, -0.0723327, 0.046792, -0.0810008, 0.157494, -0.111496, 0.0789545, -0.0169083, -0.108089, -0.139935, -0.0513273, 0.117368, -0.13791, 0.0609895, -0.0556167, 0.102466, -0.204409, -0.0951957, 0.0887407, -0.146704, 0.104232, -0.0154215, -0.0175302, -0.0671617, -0.131546, 0.0662128, 0.104957, 0.0573784, -0.117367, 0.0234723, 0.019786, -0.0878561, 0.0258602, -0.0233949, -0.0412459, 0.116362, -0.0863565, -0.126232, -0.0822426, -0.0273357, 0.0698205, -0.121103, -0.0118635, -0.0890743, 0.0764989, 0.0012393, -0.178946, 0.154407, 0.101244, 0.0874561, -0.00795454, 0.100772, 0.177547, -0.0418049, -0.021978, -0.216337, -0.0172867, -0.18545, -0.0615194, -0.0790414, -0.0123385, 0.0266707, -0.0523034, -0.0287564, 0.238271, -0.127418, -0.133864, 0.14912, 0.116666, 0.0224421, -0.0727945, 0.0701011, 0.104365, 0.216235, -0.0260505, 0.139765, -0.0796993, -0.0743173, -0.00613835, 0.0525822, 0.139028, 0.0674354, -0.0315579, 0.223974, 0.0350016, 0.00341661, -0.00152098, -0.0533885, 0.0284541, 0.0195826, 0.0295913, -0.026178, -0.012154, -0.172918, 0.12502, -0.0302473]).reshape((100,1))
        self.stroke = np.array([0.0576513, -0.00299174, -0.0521037, 0.150898, 0.0160791, -0.0918401, -0.0259279, -0.0635552, 0.223365, -0.172265, 0.0852719, -0.0399736, -0.0616868, -0.055948, 0.00574148, 0.0121507, -0.170112, 0.179787, -0.0779576, 0.0834036, -0.151286, -0.0131699, 0.0262185, -0.115511, 0.0427199, 0.0706141, 0.116837, -0.167616, -0.157201, 0.092566, 0.174241, 0.0750146, -0.201486, 0.12528, 0.0352802, -0.0973352, -0.0178036, -0.0962654, -0.128596, 0.0112661, -0.126842, -0.0508162, -0.148942, -0.00258501, 0.0704812, -0.152761, -0.0390439, -0.0252068, 0.0719852, 0.0734438, -0.202995, 0.096017, 0.136728, -0.0750206, 0.0975935, 0.105108, 0.055976, -0.0300726, 0.0848125, -0.215756, 0.0503509, -0.160459, -0.0474053, -0.136035, -0.0742018, 0.00579995, 0.00589069, 0.108491, 0.180162, -0.0191144, -0.145437, 0.179475, 0.0269573, -0.0220476, -0.029751, 0.112538, 0.0455523, 0.030426, -0.0928422, 0.134011, -0.0693618, -0.0758867, -0.029267, -0.00355201, 0.113917, 0.0327757, -0.0149661, 0.167289, 0.0949857, -0.0330509, -0.00355017, 0.0223773, 0.0707538, 0.0207575, 0.0864875, 0.0316356, 0.0317057, -0.124394, 0.110494, -0.0301914]).reshape((100,1))
        self.neurological = np.array([0.06517, -0.011909, 0.105651, 0.0290385, 0.0734544, -0.0624544, 0.0955104, -0.124304, 0.147333, -0.220816, 0.136049, -0.0378655, -0.147513, -0.218765, -0.103222, -0.0628378, -0.171139, 0.0333507, -0.0382891, 0.11935, -0.235031, 0.0899033, 0.230222, -0.117118, -0.100144, 0.142195, 0.000873022, -0.026108, -0.0719552, 0.14154, 0.232286, -0.0105907, -0.200918, 0.125025, 0.147736, 0.0109139, -0.0272979, -0.0979279, -0.0399079, 0.0975663, -0.0643834, -0.026059, -0.0893069, -0.00210596, 0.0847775, -0.201783, -0.0807828, -0.0338313, 0.0278021, 0.132104, -0.00639475, 0.0587198, 0.0590814, 0.123288, 0.0812857, 0.163594, 0.11536, 0.0292109, -0.0797959, -0.158648, 0.0564334, -0.131422, 0.0960385, -0.114073, 0.015755, -0.0182756, -0.0162823, -0.035496, 0.108996, 0.0425518, 0.0340037, 0.0572928, 0.100413, 0.0713783, -0.0674457, 0.0705757, 0.0286546, 0.0581521, -0.142699, 0.0129842, -0.0485261, -0.039925, -0.0676947, -0.0328074, 0.0186602, -0.114134, -0.122649, 0.116195, 0.0736933, 0.103384, -0.00633447, -0.0539618, -0.0732272, 0.0589878, 0.0605681, -0.021523, 0.00114416, -0.0330974, 0.064466, -0.0174716]).reshape((100,1))
        self.infant = np.array([-0.0333418, -0.0133011, -0.223609, -0.0709297, 0.0181273, -0.0561305, -0.0375744, -0.109588, 0.0560335, -0.151539, 0.172222, -0.0417012, -0.0499176, -0.0459481, 0.0659917, 0.0429493, -0.162197, -0.105166, 0.0164973, 0.0172637, -0.164539, 0.0919065, 0.105365, -0.074345, -0.0556186, -0.079134, -0.0423631, -0.118742, -0.132952, -0.0447564, 0.181912, -0.112806, -0.11119, 0.0845963, 0.0997313, -0.047416, -0.0939615, 0.0700484, 0.0976505, 0.158101, 0.00872748, 0.0529278, 0.0573952, -0.104131, 0.089852, -0.253159, -0.095134, -0.140633, -0.0230447, -0.0677504, -0.0201054, -0.0598272, 0.0359609, -0.0403904, -0.00606584, 0.0173566, -0.0266237, -0.0153633, 0.00170652, 0.0941131, 0.0241154, -0.151765, 0.0528567, -0.0672235, 0.108123, 0.135453, -0.227409, -0.00809735, 0.11033, -0.00179257, -0.138743, 0.275917, -0.0561349, -0.00220705, -0.0402322, -0.119752, -0.0993568, -0.107814, -0.0914137, 0.105808, -0.136753, -0.0819711, 0.0133362, -0.098307, 0.0647054, 0.0455944, -0.0708731, -0.111781, 0.154531, 0.0213979, -0.194241, 0.0587535, -0.056716, 0.0350543, -0.0789837, 0.110804, 0.020355, -0.056559, 0.142776, -0.0741045]).reshape((100,1))
        self.congenital = np.array([-0.0686053, 0.0835592, 0.053384, 0.0543002, -0.000192624, 0.00245459, 0.0604487, -0.0256946, 0.0609339, -0.271038, 0.123025, -0.206118, -0.112187, -0.0493175, -0.0273292, 0.0888856, -0.134542, 0.114512, 0.0174946, 0.121449, -0.0380141, -0.0175398, 0.0507215, -0.128337, 0.0117052, 0.0818106, 0.15612, -0.151545, -0.13303, 0.057657, 0.195141, -0.0102184, -0.00740065, 0.0422196, 0.0843466, -0.152507, -0.0929733, -0.0271987, -0.0334531, -0.00123171, -0.0355092, -0.00554343, -0.0887573, -0.047609, 0.106551, -0.19142, 0.064869, -0.0776788, -0.00867241, -0.0336956, -0.0586602, -0.0963187, -0.0815015, -0.0326719, 0.0922363, 0.196201, -0.0171845, 0.000140522, -0.212147, -0.220691, 0.149266, -0.0974942, 0.0864825, -0.0410038, -0.0370009, -0.113408, -0.182274, 0.0541386, 0.00958448, 0.0300531, -0.10713, 0.181795, -0.00251625, -0.019466, -0.104132, -0.087533, 0.094786, -0.117496, -0.0206026, 0.12134, -0.0253703, 0.0377632, 0.150145, 0.124305, 0.0829277, 0.027706, -0.0569112, -0.0201824, 0.108952, -0.0119526, -0.106047, -0.0532346, 0.24163, 0.000637589, 0.00421425, 0.0442666, -0.0601098, 0.0475002, 0.137036, -0.0900885]).reshape((100,1))
        self.immune = np.array([0.0131829, 0.0142167, -0.0438399, 0.0544964, -0.0733399, 0.0635106, -0.0531377, 0.0200575, 0.0492728, -0.107746, 0.184948, 0.133729, -0.156088, -0.146318, 0.0124313, 0.102955, -0.0217859, -0.0710961, 0.0670849, -0.00710042, 0.0792247, -0.084254, 0.0786399, 0.033952, 0.0101971, -0.0380269, -0.238158, -0.0196447, -0.135973, -0.0755233, 0.158763, -0.168098, -0.085657, 0.128992, 0.115479, 0.0518412, -0.03131, -0.0619908, -0.0196632, -0.0454606, -0.0536683, -0.0792315, -0.12818, -0.0528796, 0.101109, -0.080662, -0.149884, -0.128908, 0.0710246, -0.0182192, -0.0294516, -0.0246199, -0.0100877, 0.129784, 0.0858629, -0.097364, -0.0957047, 0.0717626, 0.0218544, -0.10144, -0.00902082, -0.0297839, -0.217687, 0.134505, 0.0677714, -0.114389, -0.0387141, -0.067328, 0.128213, 0.0836677, 0.0177704, -0.182794, 0.0212739, 0.00974299, -0.167506, 0.104588, -0.114338, 0.0270023, 0.0850213, 0.155312, -0.113421, -0.0859593, 0.006657, 0.0671481, 0.32843, 0.170137, 0.228099, -0.0729052, 0.0106879, -0.0119377, -0.0133275, -0.03671, -0.160386, -0.0633617, 0.064809, 0.0170428, -0.0062807, -0.0855288, -0.00236122, -0.0280223]).reshape((100,1))
        self.childbirth = np.array([-0.0836877, -0.0181015, -0.0844071, 0.0989453, -0.0320624, -0.0769346, -0.0612359, -0.0263716, 0.113558, 0.0490366, 0.0778645, 0.0612847, 0.0517153, 0.137359, -0.107482, 0.178784, -0.0876218, 0.0425381, -0.0795737, -0.0257039, -0.198594, 0.0158177, 0.189832, -0.110872, 0.103089, -0.0297189, -0.0110723, 0.0424013, -0.161569, -0.177329, 0.0929172, -0.0858716, -0.0609817, 0.115092, -0.0985789, -0.101564, 0.00919451, 0.130253, 0.0183232, 0.00958328, 0.0029839, 0.114975, 0.0837968, -0.0313, 0.160038, -0.0762705, 0.00235726, 0.177603, -0.0757185, 0.0524889, -0.0490147, -0.078091, 0.05873, -0.0499466, 0.0775218, 0.0816609, 0.193561, 0.0326247, 0.0023296, -0.0387057, 0.13685, -0.180392, 0.0478701, 0.0239932, 0.0621301, -0.105577, -0.0978259, -0.0578027, 0.175086, -0.0366377, -0.0147897, 0.283543, -0.0647449, 0.0603783, -0.0233222, -0.0634247, 0.021232, -0.102014, -0.13804, 0.149244, -0.110501, 0.000684746, -0.0400319, -0.0536896, -0.0377644, 0.0977451, -0.00297511, -0.0302935, -0.0987155, 0.00104959, 0.0285206, -0.17759, 0.19933, 0.0090785, -0.0368666, -0.00579728, -0.0151428, -0.275755, 0.0908632, -0.112596]).reshape((100,1))

    def predict(self, x):
        rval = []
        for site in x:
            site = site.reshape((100,1))
            cancer = cosine_similarity(site, self.cancer).tolist()[0][0]
            surgery = cosine_similarity(site, self.surgery).tolist()[0][0]
            transplant = cosine_similarity(site, self.transplant).tolist()[0][0]
            injury = cosine_similarity(site, self.injury).tolist()[0][0]
            stroke = cosine_similarity(site, self.stroke).tolist()[0][0]
            neurological = cosine_similarity(site, self.neurological).tolist()[0][0]
            infant = cosine_similarity(site, self.infant).tolist()[0][0]
            congenital = cosine_similarity(site, self.congenital).tolist()[0][0]
            immune = cosine_similarity(site, self.immune).tolist()[0][0]
            childbirth = cosine_similarity(site, self.childbirth).tolist()[0][0]
            best = min(cancer, surgery, transplant, injury, stroke, neurological, infant, congenital, immune, childbirth)
            
            if cancer == best:
                rval.append("Cancer")
            elif surgery == best or transplant == best:
                rval.append("Surgery/Transplantation")
            elif injury == best:
                rval.append("Injury")
            elif stroke == best:
                rval.append("Stroke")
            elif neurological == best:
                rval.append("Neurological Condition")
            elif infant == best or childbirth == best:
                rval.append("Infant/Childbirth")
            elif congenital == best or immune == best:
                rval.append("Congenital/Immune Disorder")
            else:
                rval.append("Other")

        return rval

    def score(self, x, y):
        pred = self.predict(x)
        correct = [1 if j == i else 0 for i, j in zip(pred, y)]
        return 1.0 * sum(correct) / len(correct)
    
            
                                                                                    


def main():
    parser = argparse.ArgumentParser(description='Main script for classifying health condition.')
    parser.add_argument('--feats_file', type=str, help='Full path to the features file.',
                        default='/home/srivbane/shared/caringbridge/data/word_embeddings/site_vectors.txt')
    parser.add_argument('--hc_file', type=str, help='Full path to the file with health conditions file.',
                        default='/home/srivbane/shared/caringbridge/data/word_embeddings/health_condition.txt')
    parser.add_argument('--cv', type=int, help='Number of cross-validation folders to use.', default=3)
    parser.add_argument('--n_jobs', type=int, help='Number of cores to use', default=1)
    parser.add_argument('--verbose', type=int, help='Level of verbosity parameter to pass to sklearn', default=1)
    args = parser.parse_args()
    print('main.py: Classify Health Condition with Word Vectors')
    print(args)

    # set a random seed so that results are reproducible
    random_seed = 2017
    
    # load all the data
    print("loading data")
    df = load_data(args.feats_file, args.hc_file)
    print("Size of dataset:", df.shape)

    # split the data into training and test
    print("splitting a portion of the data off for testing later...")
    test_size = 0.2
    x_train, y_train, x_test, y_test = split_data(df, test_size, random_seed)
    y_train_c = np.where(y_train == "Cancer", 'Cancer', 'Not')
    x_train_nc = x_train[np.where(y_train != "Cancer")]
    y_train_nc = y_train[np.where(y_train != "Cancer")]
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape, pd.Series(y_train).value_counts(normalize=True))
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    print("y_train_c.shape:", y_train_c.shape, y_train_c[0:5], pd.Series(y_train_c).value_counts(normalize=True))
    print("x_train_nc.shape:", x_train_nc.shape)
    print("y_train_nc.shape:", y_train_nc.shape, y_train_nc[0:5], pd.Series(y_train_nc).value_counts(normalize=True))

    # train logisitic regression model
    # logistic regression mean we are training a classifier (i.e. a label versus a real valued output)
    print("\nFinding optimal logistic regression")
    logit_param_grid = {'penalty': ['l2'],
                        'solver': ['lbfgs'],
                        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    #logit = run_grid_search(X=x_train, y=y_train, model=LogisticRegression(random_state=random_seed, class_weight='balanced'), param_grid=logit_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #print("Best logistic regression performance on test set:", logit.score(x_test, y_test))
    #y_pred = logit.predict(x_test)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/logit_confusion.txt", X=cnf_matrix, fmt='%1.4f')

    #print("\nLogistic regression splitting cancer up first")
    #logit_cancer = run_grid_search(X=x_train, y=y_train_c, model=LogisticRegression(random_state=random_seed, class_weight='balanced'), param_grid=logit_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #logit_no_cancer = run_grid_search(X=x_train_nc, y=y_train_nc, model=LogisticRegression(random_state=random_seed), param_grid=logit_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #sco = split_score(logit_cancer, logit_no_cancer, x_test, y_test)
    #print("Best split logistic regression performance on test set:", sco)
    #y_pred = split_predict(logit_cancer, logit_no_cancer, x_test, verbose=True)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/split_logit_confusion.txt", X=cnf_matrix, fmt='%1.4f')
    


    
    print("\nFinding optimal kNN")
    knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 28],
                      'algorithm': ['auto'],
                      'weights': ['uniform']}
    #knn = run_grid_search(X=x_train, y=y_train, model=KNeighborsClassifier(), param_grid=knn_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #print("Best kNN performance on test set:", knn.score(x_test, y_test))
    #y_pred = knn.predict(x_test)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/knn_confusion.txt", X=cnf_matrix, fmt='%1.4f')
    
    print("\nkNN splitting cancer up first")
    #knn_cancer = run_grid_search(X=x_train, y=y_train_c, model=KNeighborsClassifier(), param_grid=knn_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #knn_no_cancer = run_grid_search(X=x_train_nc, y=y_train_nc, model=KNeighborsClassifier(), param_grid=knn_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #sco = split_score(knn_cancer, knn_no_cancer, x_test, y_test)
    #print("Best split knn performance on test set:", sco)
    #y_pred = split_predict(knn_cancer, knn_no_cancer, x_test, verbose=True)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/split_knn_confusion.txt", X=cnf_matrix, fmt='%1.4f')

    
    

    # train random forest
    print("\nFinding optimal random forest")
    rf_param_grid = {"max_features": ['sqrt'],
                     'n_estimators': [500],
                     "max_depth": [None]}
    # do cross validation to determine optimal parameters to the model
    #rf = run_grid_search(X=x_train, y=y_train, model=RandomForestClassifier(random_state=random_seed), param_grid=rf_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #print("Best random forest performance on test set:", rf.score(x_test, y_test))
    #y_pred = rf.predict(x_test)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/rf_confusion.txt", X=cnf_matrix, fmt='%1.4f')

    #print("\nRF splitting cancer up first")
    #rf_cancer = run_grid_search(X=x_train, y=y_train_c, model=RandomForestClassifier(random_state=random_seed), param_grid=rf_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #rf_no_cancer = run_grid_search(X=x_train_nc, y=y_train_nc, model=RandomForestClassifier(random_state=random_seed), param_grid=rf_param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose)
    #sco = split_score(rf_cancer, rf_no_cancer, x_test, y_test)
    #print("Best split rf performance on test set:", sco)
    #y_pred = split_predict(rf_cancer, rf_no_cancer, x_test, verbose=True)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    #print(classification_report(y_test, y_pred))
    #np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/split_rf_confusion.txt", X=cnf_matrix, fmt='%1.4f')


    # cosine similarity model
    print("\nUsing cosine similarity model")
    sm = similarity_model()
    y_pred = sm.predict(x_test)
    print("Best cosine similarity performance on test set:", sm.score(x_test, y_test))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred))
    np.savetxt(fname="/home/srivbane/shared/caringbridge/data/word_embeddings/cosine_similarity_confusion.txt", X=cnf_matrix, fmt='%1.4f')
    
    
if __name__ == "__main__":
    main()
