from pattern_discovery.seq_solver.markov_way import MarkovParameters

class HNEParameters(MarkovParameters):
    def __init__(self, path_results, time_str, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 path_data,
                 max_branches, stop_if_twin, no_reverse_seq, error_rate, spike_rate_weight,
                 bin_size=1, cell_assemblies_data_path=None, best_order_data_path=None):
        super().__init__(time_inter_seq=time_inter_seq, min_duration_intra_seq=min_duration_intra_seq,
                         min_len_seq=min_len_seq, min_rep_nb=min_rep_nb, no_reverse_seq=no_reverse_seq,
                         max_branches=max_branches, stop_if_twin=stop_if_twin, error_rate=error_rate,
                         spike_rate_weight=spike_rate_weight,
                         bin_size=bin_size, path_results=path_results, time_str=time_str)
        self.path_data = path_data
        self.cell_assemblies_data_path = cell_assemblies_data_path
        self.best_order_data_path = best_order_data_path
        # for plotting ages
        self.markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
        self.colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
                       "pink", "darkgreen", "gold"]
