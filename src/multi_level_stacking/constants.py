from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


ART_DATASETS = [
    'circles',  
    'moons',
    'blobs_2',
    'blobs_3',
    'blobs_5',
    'blobs_7',
    'classification_5',
    'classification_7',
    'gaussian_quantiles_2',
    'gaussian_quantiles_3',
    'gaussian_quantiles_5',
    'gaussian_quantiles_7',
    'spirals',
    'two_dnormals',

]

DATASETS = [
    'australian',
    'cylinder_bands',
    'breast_cancer_winsconsin',
    'crx',
    'german',
    'indian_liver_patient',
    'ionosphere',
    'toxicity_2',
    'wine',
    'diabetes',
    'balloons_adult_stretch',
    'balloons_adult_plus_stretch',
    'balloons_yellow_small',
    'analcatdata_fraud',
    'analcatdata_donner',
    'analcatdata_boxing',
    'analcatdata_boxing2',
    'blogger',
    'molecular_promotor_gene',
    'monks1',
    'monks2',
    'qualitative_brankruptcy',
    'shuttle_landing_control',
    'tic_tac_toe',
    'blood_transfusion_service',
    'vertebra_column_2c',
    'qsar_biodegradation',
    'wdbc',
    'analcatdata_creditscore',
    'analcatdata_cyyoung8092',
    'analcatdata_cyyoung9302',
    'analcatdata_lawsuit',
    'biomed'
]

MODEL_DICTIONARY = {
    'decision_tree': lambda X, y: DecisionTreeClassifier().fit(X, y),
    'naive_bayes': lambda X, y: GaussianNB().fit(X, y),
    'svm': lambda X, y: SVC(kernel='rbf', probability=True).fit(X, y),
}