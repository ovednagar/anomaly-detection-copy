import pickle
from beta_calculator import LinearRegBetaCalculator, LinearContext
from feature_calculators import FeatureMeta
from features_picker import PearsonFeaturePicker
from graph_score import KnnScore
from anomaly_picker import ContextAnomalyPicker
from graphs import Graphs
from loggers import PrintLogger
from vertices.attractor_basin import AttractorBasinCalculator
from vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from vertices.betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from vertices.closeness_centrality import ClosenessCentralityCalculator
from vertices.communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from vertices.eccentricity import EccentricityCalculator
from vertices.fiedler_vector import FiedlerVectorCalculator
from vertices.flow import FlowCalculator
from vertices.general import GeneralCalculator
from vertices.k_core import KCoreCalculator
from vertices.load_centrality import LoadCentralityCalculator
from vertices.louvain import LouvainCalculator
from vertices.motifs import nth_nodes_motif
from vertices.multi_dimensional_scaling import MultiDimensionalScalingCalculator
from vertices.page_rank import PageRankCalculator

ANOMALY_DETECTION_FEATURES = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
    #                                                       {"communicability"}),
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
    "flow": FeatureMeta(FlowCalculator, {}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
    # Isn't OK - also in previous version
    # "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "multi_dimensional_scaling": FeatureMeta(MultiDimensionalScalingCalculator, {"mds"}),
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    # "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
    # "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    # "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}
REBUILD_FEATURES = False
RE_PICK_FTR = False


class AnomalyDetection:
    def __init__(self):
        # pearson + linear_regression(simple) + KNN + context
        self._params = {
            'database': 'EnronInc',
            # 'database': 'mc2_vast12',
            # 'database': 'twitter_security',
            'files_path': "../databases/EnronInc/EnronInc_by_day",
            # 'files_path': "../../databases/mc2_vast12/basic_by_minute",
            # 'files_path': "../../databases/twitter_security/data_by_days",
            'date_format': '%d-%b-%Y.txt',  # Enron
            # 'date_format': '%d:%m:%Y_%H:%M.txt',  # vast
            # 'date_format': '%d:%m.txt',  # Twitter
            'directed': False,
            'max_connected': True,
            'logger_name': "default Anomaly logger",
            'ftr_pairs': 25,
            'identical_bar': 0.95,
            # 'single_c': False,
            'dist_mat_file_name': "dist_mat",
            'anomalies_file_name': "anomalies",
            'context_beta': 4,
            'KNN_k': 30,
            'context_split': 4,
            'context_bar': 0.5
        }
        self._ground_truth = ['13-Dec-2000', '11-Sep-2001', '18-Oct-2001', '22-Oct-2001', '19-Nov-2001',
                              '23-Jan-2002', '30-Jan-2002', '04-Feb-2002']     # Enron
        # self._ground_truth = ['1:5', '13:5', '20:5', '24:5', '30:5', '3:6', '5:6', '6:6', '9:6', '10:6', '11:6',
        #                       '15:6', '18:6', '19:6', '20:6', '25:6', '26:6', '3:7', '18:7', '30:7', '8:8',
        #                       '9:8']  # Twitter
        # self._ground_truth = ['4:5:2012_17:51', '4:5:2012_20:25', '4:5:2012_20:26', '4:5:2012_22:16', '4:5:2012_22:21'
        #                       , '4:5:2012_22:40', '4:5:2012_22:41', '4:6:2012_17:41', '4:5:2012_18:11']  # vast

        self._logger = PrintLogger("default Anomaly logger")
        self._graphs = Graphs(self._params['database'], files_path=self._params['files_path'], logger=self._logger,
                              features_meta=ANOMALY_DETECTION_FEATURES, directed=self._params['directed'],
                              date_format=self._params['date_format'], largest_cc=self._params['max_connected'])
        self._graphs.build(force_rebuild_ftr=REBUILD_FEATURES, pick_ftr=RE_PICK_FTR)
        self._graphs._multi_graph._features_matrix_dict = pickle.load(open("old_features.pkl", "rb"))
        # convert to index
        self._ground_truth = [self._graphs.name_to_index(event) for event in self._ground_truth]
        self.print_feature_meta()

    def build(self):
        pearson_picker = PearsonFeaturePicker(self._graphs, size=self._params['ftr_pairs'],
                                              logger=self._logger, identical_bar=self._params['identical_bar'])
        best_pairs = pearson_picker.best_pairs()
        beta = LinearContext(self._graphs, best_pairs, split=self._params['context_beta'])
        beta_matrix = beta.beta_matrix()
        score = KnnScore(beta_matrix, self._params['KNN_k'], self._params['database'],
                         context_split=self._params['context_beta'])
        score.dist_heat_map(self._params['dist_mat_file_name'])
        anomaly_picker = ContextAnomalyPicker(self._graphs, score.score_list(), self._params['database'], logger=None,
                                                split=self._params['context_split'], bar=self._params['context_bar'])

        anomaly_picker.build()
        anomaly_picker.plot_anomalies(self._params['anomalies_file_name'], truth=self._ground_truth,
                                      info_text=self.param_to_str())

    def print_feature_meta(self):
        for ftr in self._graphs.get_feature_meta():
            print(ftr)

    def build_manipulations(self):
        for ftr_num in range(25, 32, 5):
            self._params['ftr_pairs'] = ftr_num
            for identical in range(90, 100, 1):
                self._params['identical_bar'] = round(identical * 0.10, 2)
                pearson_picker = PearsonFeaturePicker(self._graphs, size=self._params['ftr_pairs'],
                                                      logger=self._logger, identical_bar=self._params['identical_bar'])
                best_pairs = pearson_picker.best_pairs()
                # beta = LinearRegBetaCalculator(self._graphs, best_pairs, single_c=self._params['single_c'])
                for context in range(4, 7, 2):
                    self._params['context_beta'] = context
                    beta = LinearContext(self._graphs, best_pairs, split=self._params['context_beta'])
                    beta_matrix = beta.beta_matrix()
                    for k in range(5, min(100, int(self._graphs.number_of_graphs()/context))):
                        self._params['KNN_k'] = k
                        score = KnnScore(beta_matrix, self._params['KNN_k'], self._params['database'],
                                         context_split=self._params['context_beta'])
                        # score = TestScore(beta_matrix, self._params['database'])
                        score.dist_heat_map(self._params['dist_mat_file_name'])
                        anomaly_picker = ContextAnomalyPicker(self._graphs, score.score_list(), self._params['database'], logger=None,
                                                                split=self._params['context_split'], bar=self._params['context_bar'])

                        anomaly_picker.build()
                        anomaly_picker.plot_anomalies(self._params['anomalies_file_name'], truth=self._ground_truth,
                                                      info_text=self.param_to_str())

    def param_to_str(self):
        skip = ['database', 'files_path', 'date_format', 'logger_name', 'context_split', 'context_bar',
                'dist_mat_file_name', 'anomalies_file_name']
        param_str = ""
        for key, val in self._params.items():
            if key in skip:
                continue
            param_str += str(key) + ":" + str(val) + "\n"
        return param_str


if __name__ == "__main__":
    AnomalyDetection().build()
    # A = AnomalyDetection()
    # A.build_manipulations()

