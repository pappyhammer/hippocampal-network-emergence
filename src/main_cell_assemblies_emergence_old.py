import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib

# important to avoid a bug when using virtualenv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from matplotlib.ticker import MultipleLocator
import seaborn as sns
# import copy
from datetime import datetime
from collections import Counter
from sklearn import metrics
# import keras
import os
from sortedcontainers import SortedList, SortedDict
from seq_tree_v1 import Tree
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
import pattern_discovery.tools.param as p_disc_param
import pattern_discovery as pattern_discovery
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq
from pattern_discovery.tools.sce_detection import detect_sce_with_sliding_window
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold


# serve to represent the a sequence of spikes based on probability
# from one neuron can have many follower up to n_order.
class Tree_v2(object):
    def __init__(self, neuron=None, father=None, father_id=None, prob=1, acc=0):
        # key are int representing ID of the current node, value are dict, with key neuron, and value Tree_v2
        self.children = dict()
        self.neuron = neuron
        # sortedDict with key being the accumulated metrics from this father, with father ID
        self.fathers = SortedDict()
        # keep in a dictionnary the different father, with the key being the father id and value the tree
        self.fathers_by_id = dict()
        if father is not None:
            self.fathers[round(father.acc[father_id], 6)] = (father, prob)
            self.fathers_by_id[father_id] = father
        # List of ID for each tree among members of the same root tree, list on neuron in the sequence till the given
        # neuron
        if father is None:
            self.ids = [f'{neuron}']
            current_id = f'{neuron}'
        else:
            current_id = self.get_id_using_father_id(father_id)
            self.ids = [current_id]

        self.acc = dict()
        self.acc[current_id] = acc
        self.children[current_id] = dict()
        # ancestors are dict containing dict, first key is the current_id from which the ancestors line is coming,
        # second key is the neuron from the ancestor, and the value is the Tree representing each ancestor
        self.ancestors = dict()
        if father is not None:
            self.ancestors[current_id] = dict(father.ancestors[father_id])
        else:
            self.ancestors[current_id] = dict()
        # this instance is part of its ancestors
        self.ancestors[current_id][self.neuron] = self

    # Used when the neuron is part of a new sequence
    def add_new_instance(self, father=None, father_id=None, prob=1, acc=0):
        if father is not None:
            self.fathers[round(father.acc[father_id], 6)] = (father, prob)
            self.fathers_by_id[father_id] = father
        # List of ID for each tree among members of the same root tree, list on neuron in the sequence till the given
        # neuron
        if father is None:
            self.ids.append(f'{self.neuron}')
            current_id = f'{self.neuron}'
        else:
            current_id = self.get_id_using_father_id(father_id)
            self.ids.append(current_id)
        self.acc[current_id] = acc
        self.children[current_id] = dict()
        if father is not None:
            self.ancestors[current_id] = dict(father.ancestors[father_id])
        else:
            self.ancestors[current_id] = dict()
        # this instance is part of its ancestors
        self.ancestors[current_id][self.neuron] = self

    def disinherit(self, current_id):
        """
        Remove this tree from his father
        :return:
        """
        father_id = Tree_v2.get_id_father(current_id)
        if father_id == "":
            return
        if father_id not in self.fathers_by_id:
            return
        # print("father found")
        father = self.fathers_by_id[father_id]
        father_children = father.children[father_id]
        # if current_id == "0_77_62":
        #     print(f"current_id == 0_77_62, father_id {father_id}")
        del (father_children[current_id])
        # if a tree loose all his children, then it is disinherited as well
        if len(father_children) == 0:
            father.disinherit(father_id)

    def get_father(self, father_id):
        if father_id in self.fathers:
            return self.fathers[father_id]

    def get_id_using_father_id(self, father_id):
        return father_id + '_' + f'{self.neuron}'

    # remove the last neuron, to get the id of the father, return "" if no father
    @staticmethod
    def get_id_father(complete_id):
        reverse_id = complete_id[::-1]
        index = reverse_id.find("_")
        if index == -1:
            return ""
        return reverse_id[(index + 1):][::-1]

    def add_child(self, current_id, child, child_id):
        self.children[current_id][child_id] = child

    def __str__(self):
        result = f"Tree: neuron {self.neuron}\n"
        result += '\n'
        return result

    def is_among_ancestors(self, current_id, neuron):
        """
        Return True if curren_id Node (from self) is among the ancestor
        :param current_id:
        :param neuron:
        :return:
        """
        if self.neuron == neuron:
            return True
        # this means current_id has no ancestors
        if current_id not in self.ancestors:
            return False
        return neuron in self.ancestors[current_id]

    # so far False, to repair
    def get_seq_lists(self):
        # print(f'get_seq_lists neuron {self.neuron}')
        if len(self.children) == 0:
            # print(f'get_seq_lists neuron2 {self.neuron}')
            return [[self.neuron]]
        result = []
        # print(f'get_seq_lists result1: {result}')
        for tree_child in self.children.values():
            child_lists = tree_child.get_seq_lists()
            # print(f'get_seq_lists child_lists {child_lists}')
            for child_l in child_lists:
                # print(f'get_seq_lists child_l {child_l}')
                tmp_list = [self.neuron]
                tmp_list.extend(child_l)
                result.append(tmp_list)
                # print(f'get_seq_lists result2: {result}')
        # print(f'get_seq_lists result: {result}')
        return result


class Parameters:
    def __init__(self, time_inter_seq, seq_can_be_on_same_time, n_order, max_depth,
                 min_len_seq, no_reverse_seq,
                 value_to_add_factor, min_rep_nb, min_len_seq_first_tour,
                 threshold_factor, stop_if_twin, write_on_file,
                 random_mode, threshold_duration, keeping_only_same_diff_seq,
                 split_spike_nums, error_rate, min_duration_intra_seq, mouse,
                 show_heatmap, transition_order, spike_rate_weight):
        self.mouse = mouse
        self.write_on_file = write_on_file
        self.show_heatmap = show_heatmap
        self.random_mode = random_mode
        self.threshold_duration = threshold_duration
        self.time_inter_seq = time_inter_seq
        self.seq_can_be_on_same_time = seq_can_be_on_same_time
        self.n_order = n_order
        self.max_depth = max_depth
        self.min_len_seq = min_len_seq
        self.min_len_seq_first_tour = min_len_seq_first_tour
        self.min_rep_nb = min_rep_nb
        self.value_to_add_factor = value_to_add_factor
        self.threshold_factor = threshold_factor
        self.stop_if_twin = stop_if_twin
        self.split_spike_nums = split_spike_nums
        self.error_rate = error_rate
        self.no_reverse_seq = no_reverse_seq
        self.min_duration_intra_seq = min_duration_intra_seq
        self.keeping_only_same_diff_seq = keeping_only_same_diff_seq
        self.file = None
        self.transition_order = transition_order
        self.spike_rate_weight = spike_rate_weight
        self.max_branches = 5
        self.path_results = None
        self.time_str = None

    def __str__(self):
        s = ""
        s += f"Mouse: {self.mouse}\n"
        s += f"Random mode: {self.random_mode}\n"
        s += f"Time inter seq: {self.time_inter_seq}\n"
        s += f"Seq_can_be_on_same_time: {self.seq_can_be_on_same_time}\n"
        s += f"Nth order: {self.n_order}\n"
        s += f"Max depth: {self.max_depth}\n"
        s += f"Min len seq: {self.min_len_seq}\n"
        s += f"min_len_seq_first_tour: {self.min_len_seq_first_tour}\n"
        s += f"Min rep nb: {self.min_rep_nb}\n"
        s += f"Value_to_add_factor: {self.value_to_add_factor}\n"
        s += f"threshold_factor: {self.threshold_factor}\n"
        s += f"Stop_if_twin: {self.stop_if_twin}\n"
        s += f"Split_spike_nums: {self.split_spike_nums}\n"
        s += f"Error rate: {self.error_rate}\n"
        s += f"No reverse seq: {self.no_reverse_seq}\n"
        s += f"spike_rate_weight {self.spike_rate_weight}\n"
        s += f"min_duration_intra_seq {self.min_duration_intra_seq}\n"
        s += f"threshold_duration {self.threshold_duration}\n"
        s += f"transition_order {self.transition_order}\n"
        return s


class StatSeq:
    def __init__(self, lg_seq):
        self.lg_seq = lg_seq
        # nb of seq of this given length, for a trial if len == 1, otherwise for each trial
        self.nb_seq = []
        # for each seq, nb of repetition (same seq of neurons for a given)
        self.nb_rep = []
        # nb of seq of this given length with the same diff for a trial, the key being the trial number
        self.nb_same_diff = []
        # Std of the sum of diff for this given length seq for a trial
        self.std_sum_diff = []

    def __add__(self, other):
        if self.lg_seq != other.lg_seq:
            return None
        stat_seq = StatSeq(self.lg_seq)
        stat_seq.nb_seq = self.nb_seq + other.nb_seq
        stat_seq.nb_rep = self.nb_rep + other.nb_rep
        stat_seq.nb_same_diff = self.nb_same_diff + other.nb_same_diff
        stat_seq.std_sum_diff = self.std_sum_diff + other.std_sum_diff
        return stat_seq

    def __str__(self):
        s = ""
        s += f"Nb trial: {len(self.nb_seq)}\n"
        s += f"Length seq: {self.lg_seq}\n"
        if len(self.nb_seq) == 1:
            s += f"Nb of seq: {self.nb_seq[0]}\n"
        else:
            s += f"Nb of seq, mean: {np.round(np.mean(self.nb_seq), 3)}, std: {np.round(np.std(self.nb_seq), 3)}, " \
                 f"min: {np.min(self.nb_seq)}, max: {np.max(self.nb_seq)}\n"
        s += f"Nb of rep, mean: {np.round(np.mean(self.nb_rep), 3)}, std: {np.round(np.std(self.nb_rep), 3)}, " \
             f"min: {np.min(self.nb_rep)}, max: {np.max(self.nb_rep)}\n"
        if len(self.nb_same_diff) == 0:
            s += f"No same diff\n"
        else:
            s += f"Nb same diff, mean: {np.round(np.mean(self.nb_same_diff), 3)}, " \
                 f"std: {np.round(np.std(self.nb_same_diff), 3)} " \
                 f"min: {np.min(self.nb_same_diff)}, max: {np.max(self.nb_same_diff)}\n"
        if len(self.std_sum_diff) == 0:
            s += f"No sum diff\n"
        else:
            s += f"Std sum diff, mean: {np.round(np.mean(self.std_sum_diff), 3)}, " \
                 f"std: {np.round(np.std(self.std_sum_diff), 3)} " \
                 f"min: {np.min(self.std_sum_diff)}, max: {np.max(self.std_sum_diff)}\n"
        return s


def get_seq_n_order_2nd_order_dict(trans_dict, neuron_to_start, threshold, n_order, max_depth, depth, param,
                                   father=None, stop_if_twin=True, prob_to_get_there=1):
    """

    :param trans_dict:
    :param neuron_to_start:
    :param threshold:
    :param n_order:
    :param max_depth:
    :param depth:
    :param father:
    :param stop_if_twin: if True, the search will stop if a neuron already in the sequence is found as the next
    one in the sequence, otherwise it will take the 2nd max prob neuron
    :param prob_to_get_there: represent the probability (0 to 1) to arrive to this neuron
    :return:
    """
    neuron = neuron_to_start
    nb_neurons = len(trans_dict[0, 0, :])
    # if father
    tree = Tree(neuron=neuron, n_order=n_order, father=father, prob=prob_to_get_there)
    if (max_depth > 0) and (depth >= max_depth):
        n_order = 1
    # if n_order > 1:
    trans_dict_copy = np.copy(trans_dict)
    # else:
    #     trans_dict_copy = trans_dict
    for i in np.arange(n_order):
        if stop_if_twin:
            next_one = np.argmax(trans_dict_copy[neuron, :])
            n_1 = next_one // nb_neurons
            n_2 = next_one % nb_neurons
            if (trans_dict_copy[neuron, n_1, n_2] > threshold) and \
                    ((father is None) or ((not father.is_in_the_tree(n_1)) and (not father.is_in_the_tree(n_2)))):
                tree.child[i] = Tree(neuron=n_1, n_order=1, father=tree, prob=trans_dict_copy[neuron, n_1, n_2])
                tree.child[i].child[0] = get_seq_n_order_2nd_order_dict(trans_dict, n2, threshold, n_order,
                                                                        max_depth=max_depth,
                                                                        depth=depth + 1, father=tree,
                                                                        stop_if_twin=stop_if_twin,
                                                                        prob_to_get_there=trans_dict_copy[
                                                                            neuron, n_1, n_2], param=param)
            # we used to max prob neuron, now we don't want to find it anymore
            trans_dict_copy[neuron, n_1, n_2] = -1
            # else:
            #     if trans_dict[neuron, next_one] <= threshold:
            #         print('end due to threshold')
            #     else:
            #         if father is None:
            #             print('end due to loop, unique neuron')
            #         else:
            #             print('end due to loop')
        else:
            # Will loop until it get to a neuron which probability is <= threshold
            while True:
                next_one = np.argmax(trans_dict_copy[neuron, :])
                n_1 = next_one // nb_neurons
                n_2 = next_one % nb_neurons
                # if we're deep enough, we stop when we find again the same neuron
                lets_do_that = False
                if lets_do_that:
                    if (max_depth > 0) and (depth >= max_depth):
                        if (father is not None) and (father.is_in_the_tree(n_1) or father.is_in_the_tree(n_2)):
                            break
                if (father is not None) and (father.is_in_the_tree(n_1) or father.is_in_the_tree(n_2)):
                    trans_dict_copy[neuron, n_1, n_2] = -1
                elif trans_dict_copy[neuron, n_1, n_2] > threshold:
                    # print("creating a child")
                    tree.child[i] = Tree(neuron=n_1, n_order=1, father=tree, prob=trans_dict_copy[neuron, n_1, n_2])
                    tree.child[i].child[0] = get_seq_n_order_2nd_order_dict(trans_dict, n_2, threshold, n_order,
                                                                            max_depth=max_depth,
                                                                            depth=depth + 1, father=tree,
                                                                            stop_if_twin=stop_if_twin, param=param)
                    # we used to max prob neuron, now we don't want to find it anymore
                    trans_dict_copy[neuron, n_1, n_2] = -1
                    break
                else:
                    # print(f"get_seq_n_order {np.round(trans_dict_copy[neuron, next_one], 5)} "
                    #       f"<= Threshold at depth {depth}")
                    break

        # we used to max prob neuron, now we don't want to find it anymore
        # if i < (n_order - 1):
        #     trans_dict_copy[neuron, next_one] = -1
    return tree


class Tree_at_depth:
    """
    Special tree made for the SortedList of  nodes_at_depth in bfs_v2 method
    """

    def __init__(self, tree, identifier, acc):
        self.tree = tree
        self.id = identifier
        self.acc = acc

    def __lt__(self, other):
        """
        inferior self < other
        :param other:
        :return:
        """
        return self.acc < other.acc

    def __le__(self, other):
        """
        Lower self <= other
        :param other:
        :return:
        """
        return self.acc <= other.acc

    def __eq__(self, other):
        """
        Equal self == other
        :param other:
        :return:
        """
        return self.acc == other.acc

    def __ne__(self, other):
        """
        non equal self != other
        :param other:
        :return:
        """
        return self.acc != other.acc

    def __gt__(self, other):
        """
        Greater self > other
        :param other:
        :return:
        """
        return self.acc > other.acc

    def __ge__(self, other):
        """
        Greater self >= other
        :param other:
        :return:
        """
        return self.acc >= other.acc


def bfs_v2(trans_dict, param):
    """
    Breadth First Search, build a Tree according to trans_dict, this time there is no first node, each neuron is
    represented by one node (Tree_v2), each node being connected to as many neurons (nodes) as trans_dict implied
    Each node as multiple Id and ancestors line.
    A dict with for each neuron as an int for key, and as value the node corresponding will be return with the Tree
    :param trans_dict:
    :param param:
    :return:
    """
    nb_neurons = len(trans_dict[0, :])
    # determine the maximum nodes of the tree at the deepest level
    max_branches = 10
    mean_trans_dict = np.mean(trans_dict)
    std_trans_dict = np.std(trans_dict)
    neuron_nodes_dict = dict()
    for neuron in np.arange(nb_neurons):
        print(f'bfs_v2 {"|"*(neuron + 1)}{" "*(nb_neurons - neuron - 1)}{"|"} {neuron}')
        current_depth = 0
        # keep list of neurons to explore, for each depth, when no more current_neuron for a depth are yet to explore
        # then we only keep the one with shortest accumulated metric

        if neuron in neuron_nodes_dict:
            tree_root = neuron_nodes_dict[neuron]
            tree_root.add_new_instance()
        else:
            tree_root = Tree_v2(neuron=neuron)
            neuron_nodes_dict[neuron] = tree_root
        tree_at_depth = Tree_at_depth(tree=tree_root, identifier=f'{neuron}', acc=0)
        nodes_to_expend = {0: [tree_at_depth]}
        # keep a sorted list (by acc, the smaller value first) of the nodes (current_neuron) at the current_depth
        nodes_at_depth = dict()
        nodes_at_depth[current_depth] = SortedList()
        nodes_at_depth[current_depth].add(tree_at_depth)
        while True:
            tree_at_depth = nodes_to_expend[current_depth][0]
            current_node = tree_at_depth.tree
            current_id = tree_at_depth.id
            nodes_to_expend[current_depth] = nodes_to_expend[current_depth][1:]
            current_neuron = current_node.neuron
            trans_dict_copy = np.copy(trans_dict)
            i = 0
            while i < max_branches:
                # will take the nth (n = max_branches) current_neuron with the higher probability to spike after current_neuron
                # for i in np.arange(max_branches):
                next_neuron = np.argmax(trans_dict_copy[current_neuron, :])
                prob = trans_dict_copy[current_neuron, next_neuron]
                trans_dict_copy[current_neuron, next_neuron] = -1
                # if the next best current_neuron probability is 0, then we end the for loop
                if prob < mean_trans_dict:  # < (mean_trans_dict + std_trans_dict): #== 0:
                    break
                # if current_neuron already among the elderies, we skip to the next one
                if current_node.is_among_ancestors(current_id, next_neuron):
                    if param.stop_if_twin:
                        break
                    else:
                        continue
                i += 1

                new_tree_acc = current_node.acc[current_id] - np.log(prob)
                if next_neuron in neuron_nodes_dict:
                    tree = neuron_nodes_dict[next_neuron]
                    tree.add_new_instance(father=current_node, father_id=current_id, prob=prob,
                                          acc=new_tree_acc)
                else:
                    tree = Tree_v2(neuron=next_neuron, father=current_node, father_id=current_id, prob=prob,
                                   acc=new_tree_acc)
                    neuron_nodes_dict[next_neuron] = tree
                if (current_depth + 1) not in nodes_at_depth:
                    nodes_at_depth[current_depth + 1] = SortedList()
                new_tree_id = tree.get_id_using_father_id(current_id)
                tree_at_depth = Tree_at_depth(tree=tree, identifier=new_tree_id, acc=new_tree_acc)
                nodes_at_depth[current_depth + 1].add(tree_at_depth)
                current_node.add_child(current_id=current_id, child=tree,
                                       child_id=new_tree_id)

            # limit the number of neurons for each depth and add the best "current_neuron"
            # in term of accumulated metriccs to
            # the current_node

            # we need to check how many leafs are in the tree at this current_depth
            # and remove the one with the lower score
            # adding to nodes_to_expend only the best nodes

            if len(nodes_to_expend[current_depth]) == 0:
                current_depth += 1
                if current_depth in nodes_at_depth:
                    if len(nodes_at_depth[current_depth]) <= max_branches:
                        nodes_to_expend[current_depth] = nodes_at_depth[current_depth]
                    else:
                        # keeping the best ones
                        nodes_to_expend[current_depth] = nodes_at_depth[current_depth][:max_branches]
                        # then removing from the tree the other ones
                        for n in nodes_at_depth[current_depth][max_branches:]:
                            # if n.id == "0_21_0":
                            #     print(f"n.id == 0_21_0")
                            #     for i in nodes_at_depth[current_depth][max_branches:]:
                            #         print(f"id: {i.id}")
                            n.tree.disinherit(n.id)
                else:
                    break

                if current_depth not in nodes_to_expend:
                    break
    return neuron_nodes_dict


def bfs(trans_dict, neuron_to_start, param):
    """
    Breadth First Search, build a Tree according to trans_dict, the first node being neuron_to_start
    :param trans_dict:
    :param neuron_to_start:
    :return:
    """
    current_depth = 0
    # determine the maximum nodes of the tree at the deepest level
    max_branches = 5
    mean_trans_dict = np.mean(trans_dict)
    std_trans_dict = np.std(trans_dict)
    # keep list of neurons to explore, for each depth, when no more current_neuron for a depth are yet to explore
    # then we only keep the one with shortest accumulated metric
    tree_root = Tree(neuron=neuron_to_start, father=None, prob=1, acc=0, n_order=1)
    nodes_to_expend = {0: [tree_root]}
    # keep a sorted list (by acc, the smaller value first) of the nodes (current_neuron) at the current_depth
    nodes_at_depth = dict()
    nodes_at_depth[current_depth] = SortedList()
    nodes_at_depth[current_depth].add(tree_root)
    while True:
        current_node = nodes_to_expend[current_depth][0]
        nodes_to_expend[current_depth] = nodes_to_expend[current_depth][1:]
        current_neuron = current_node.neuron
        trans_dict_copy = np.copy(trans_dict)
        i = 0
        while i < max_branches:
            # will take the nth (n = max_branches) current_neuron with the higher probability to spike after current_neuron
            # for i in np.arange(max_branches):
            next_neuron = np.argmax(trans_dict_copy[current_neuron, :])
            prob = trans_dict_copy[current_neuron, next_neuron]
            trans_dict_copy[current_neuron, next_neuron] = -1
            # if the next best current_neuron probability is 0, then we end the for loop
            if prob < mean_trans_dict:  # < (mean_trans_dict + std_trans_dict): #== 0:
                break
            # if prob < mean_trans_dict:
            # if current_neuron already among the elderies, we skip to the next one
            if (current_node.father is not None) and (next_neuron in current_node.parents):
                if param.stop_if_twin:
                    break
                else:
                    continue
            # print(f"prob in loop for: {prob}")
            i += 1
            tree = Tree(neuron=next_neuron, father=current_node, prob=prob,
                        acc=current_node.acc - np.log(prob), n_order=1)
            if (current_depth + 1) not in nodes_at_depth:
                nodes_at_depth[current_depth + 1] = SortedList()
            nodes_at_depth[current_depth + 1].add(tree)
            current_node.add_child(tree)

        # TODO: See to limit the depth of the tree
        # limit the number of neurons for each depth and add the best "current_neuron" in term of accumulated metriccs to
        # the current_node

        # we need to check how many leafs are in the tree at this current_depth
        # and remove the one with the lower score
        # adding to nodes_to_expend only the best nodes

        if len(nodes_to_expend[current_depth]) == 0:
            current_depth += 1
            if current_depth in nodes_at_depth:
                if len(nodes_at_depth[current_depth]) <= max_branches:
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth]
                else:
                    # keeping the best ones
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth][:max_branches]
                    # then removing from the tree the other ones
                    for n in nodes_at_depth[current_depth][max_branches:]:
                        n.disinherit()
            else:
                break

            if current_depth not in nodes_to_expend:
                break
    return tree_root


# So far, too slow !!
def bfs_2nd_order(trans_dict, neuron_to_start, param):
    """
    Breadth First Search
    :param trans_dict:
    :param neuron_to_start:
    :return:
    """
    print("bfs_2nd_order start")
    nb_neurons = len(trans_dict[0, 0, :])
    mean_trans_dict = np.mean(trans_dict)
    std_trans_dict = np.std(trans_dict)
    median_trans_dict = np.median(trans_dict)
    current_depth = 0
    # determine the maximum nodes of the tree at the deepest level
    max_branches = 2
    # keep list of neurons to explore, for each depth, when no more current_neuron for a depth are yet to explore
    # then we only keep the one with shortest accumulated metric
    tree_root = Tree(neuron=neuron_to_start, father=None, prob=1, acc=0, n_order=1)
    nodes_to_expend = {0: [tree_root]}
    # keep a sorted list (by acc, the smaller value first) of the nodes (current_neuron) at the current_depth
    nodes_at_depth = dict()
    nodes_at_depth[current_depth] = SortedList()
    nodes_at_depth[current_depth].add(tree_root)
    while True:
        current_node = nodes_to_expend[current_depth][0]
        nodes_to_expend[current_depth] = nodes_to_expend[current_depth][1:]
        current_neuron = current_node.neuron
        trans_dict_copy = np.copy(trans_dict)
        print("Before while i < max_branches")
        i = 0
        # will take the nth (n = max_branches) current_neuron with the higher probability to spike after current_neuron
        while i < max_branches:
            # next_neuron = np.argmax(trans_dict_copy[current_neuron, :, :])
            next_one = np.argmax(trans_dict_copy[current_neuron, :])
            n_1 = next_one // nb_neurons
            n_2 = next_one % nb_neurons
            prob = trans_dict_copy[current_neuron, n_1, n_2]
            # if the next best current_neuron probability is 0, then we end the for loop
            if prob < (mean_trans_dict + std_trans_dict):
                break

            trans_dict_copy[current_neuron, n_1, n_2] = -1
            if current_neuron == n_1:
                continue
            if current_neuron == n_2:
                continue
            if n_1 == n_2:
                continue
            # if current_neuron already among the ancestors, we skip (to the next one)
            if (current_node.father is not None) and ((n_1 in current_node.parents) or (n_2 in current_node.parents)):
                break
            i += 1
            # print(f"prob in loop for: {prob}")
            tree_n_1 = Tree(neuron=n_1, father=current_node, prob=prob,
                            acc=current_node.acc - np.log(prob), n_order=1)
            tree_n_2 = Tree(neuron=n_2, father=tree_n_1, prob=prob,
                            acc=tree_n_1.acc - np.log(prob), n_order=1)
            if (current_depth + 1) not in nodes_at_depth:
                nodes_at_depth[current_depth + 1] = SortedList()
            # Not really current depth as we add 2 neuron at each step
            nodes_at_depth[current_depth + 1].add(tree_n_2)
            # checking if this neuron is not already a child, if so, adding tree_n_2 as child to tree_n_1 only
            if tree_n_1.id in current_node.child:
                i = 1
                while True:
                    new_id = tree_n_1.id + f"v{i}"
                    if new_id not in current_node.child:
                        tree_n_1.id = new_id
                        break
            current_node.add_child(tree_n_1)
            tree_n_1.add_child(tree_n_2)

        # TODO: See to limit the depth of the tree
        # limit the number of neurons for each depth and add the best "current_neuron" in term of accumulated metriccs to
        # the current_node

        # we need to check how many leafs are in the tree at this current_depth
        # and remove the one with the lower score
        # adding to nodes_to_expend only the best nodes

        if len(nodes_to_expend[current_depth]) == 0:
            current_depth += 1
            print(f"depth: {current_depth}")
            if current_depth in nodes_at_depth:
                print(f"Nb branches at depth {current_depth}: {len(nodes_at_depth[current_depth])}")
                if len(nodes_at_depth[current_depth]) <= max_branches:
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth]
                else:
                    # keeping the best ones
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth][:max_branches]
                    # then removing from the tree the other ones
                    for n in nodes_at_depth[current_depth][max_branches:]:
                        n.disinherit_2nd_order()
            else:
                break
            if current_depth not in nodes_to_expend:
                break
    return tree_root


def get_seq_n_order(trans_dict, neuron_to_start, threshold, n_order, max_depth, depth, param, father=None,
                    stop_if_twin=True, acc=0, prob=1):
    """
    Depth First Search
    :param trans_dict:
    :param neuron_to_start:
    :param threshold:
    :param n_order:
    :param max_depth:
    :param depth:
    :param father:
    :param stop_if_twin: if True, the search will stop if a neuron already in the sequence is found as the next
    one in the sequence, otherwise it will take the 2nd max prob neuron
    :param prob: probability (float 0 to 1) than this node comes after the previous one
    :param acc: accumulated metric, the first node of the Tree has an acc of 0
    :return:
    """

    neuron = neuron_to_start
    # if father
    tree = Tree(neuron=neuron, n_order=n_order, father=father, prob=prob, acc=acc)
    if (max_depth > 0) and (depth >= max_depth):
        n_order = 1
    # if n_order > 1:
    trans_dict_copy = np.copy(trans_dict)
    # else:
    #     trans_dict_copy = trans_dict
    for i in np.arange(n_order):
        if stop_if_twin:
            next_one = np.argmax(trans_dict_copy[neuron, :])
            if (trans_dict_copy[neuron, next_one] > threshold) and \
                    ((father is None) or (not father.is_in_the_tree(next_one))):
                tree.child[i] = get_seq_n_order(trans_dict, next_one, threshold, n_order, max_depth=max_depth,
                                                depth=depth + 1, father=tree, stop_if_twin=stop_if_twin,
                                                prob=trans_dict_copy[neuron, next_one],
                                                acc=acc - np.log(trans_dict_copy[neuron, next_one]), param=param)
            # else:
            #     if trans_dict[neuron, next_one] <= threshold:
            #         print('end due to threshold')
            #     else:
            #         if father is None:
            #             print('end due to loop, unique neuron')
            #         else:
            #             print('end due to loop')
        else:
            # Will loop until it get to a neuron which probability is <= threshold
            while True:
                next_one = np.argmax(trans_dict_copy[neuron, :])
                # if dict_order == 2:
                #     nb_neurons = len(trans_dict_copy[0,0,:])
                #     n_1 = next_one // nb_neurons
                #     n_2 = next_one % nb_neurons
                # if we're deep enough, we stop when we find again the same neuron
                lets_do_that = False
                if lets_do_that:
                    if (max_depth > 0) and (depth >= max_depth):
                        if (father is not None) and (father.is_in_the_tree(next_one)):
                            break
                if (father is not None) and (father.is_in_the_tree(next_one)):
                    trans_dict_copy[neuron, next_one] = -1
                elif trans_dict_copy[neuron, next_one] > threshold:
                    # print("creating a child")
                    tree.child[i] = get_seq_n_order(trans_dict, next_one, threshold, n_order, max_depth=max_depth,
                                                    depth=depth + 1, father=tree,
                                                    prob=trans_dict_copy[neuron, next_one],
                                                    acc=acc - np.log(trans_dict_copy[neuron, next_one]),
                                                    stop_if_twin=stop_if_twin, param=param)
                    break
                else:
                    # print(f"get_seq_n_order {np.round(trans_dict_copy[neuron, next_one], 5)} "
                    #       f"<= Threshold at depth {depth}")
                    break

        # we used to max prob neuron, now we don't want to find it anymore
        if i < (n_order - 1):
            trans_dict_copy[neuron, next_one] = -1
    return tree


def generate_random_spikes_num(size, model=None, shuffle=True, insert_seq=False, size_seq_to_insert=0):
    if model is not None:
        spike_nums = np.copy(model)
        # spike_nums = np.random.permutation(spike_nums)
        # return spike_nums
        # print(f'len(spike_nums) {len(spike_nums)}')
        for j in np.arange(len(spike_nums[0])):
            spike_nums[:, j] = np.random.permutation(spike_nums[:, j])
        for i, s in enumerate(spike_nums):
            # print(f"pos before permutation {np.where(s)[0]}")
            spike_nums[i] = np.random.permutation(s)
            # print(f"pos after permutation {np.where(s)[0]}")
        return spike_nums
    else:
        spike_nums = np.zeros((size, size))
        return spike_nums


def removing_intersect_seq(set_seq_dict_result, min_len_seq, min_rep_nb, dict_by_len_seq):
    # set_seq_dict_result: each key represent a common seq (neurons tuple)
    # and each values represent a list of list of times
    # dict_by_len_seq: each key is an int representing the length of the seq contains
    # each value is a dict with as they key a tuple of int representing a neuron seq, and as value, a list
    # of list time instances
    keys_to_delete = set()
    for neurons_seq, times_seq_set in set_seq_dict_result.items():
        # in case it would have been already removed
        if neurons_seq not in set_seq_dict_result:
            continue
        if len(times_seq_set) < min_rep_nb:
            keys_to_delete.add(tuple(neurons_seq))
            continue
        if (len(times_seq_set) > 1) and (len(neurons_seq) > min_len_seq):
            # for each seq > at the min length, we take sub sequences starting from the last index
            # and look if it's present in the dictionnary with the same times
            for i in np.arange(len(neurons_seq) - min_len_seq) + 1:
                neurons_tuple_to_test = tuple(neurons_seq[0:-i])
                # print(f'neurons_tuple_to_test: {neurons_tuple_to_test}, ori: {neurons_seq}')
                if neurons_tuple_to_test in set_seq_dict_result:
                    orig_times = set()
                    # need to shorten the times to compare it later
                    for times in times_seq_set:
                        orig_times.add(times[0:-i])
                    times_to_check = set_seq_dict_result[neurons_tuple_to_test]
                    # print(f'times_seq_set: {orig_times}, times_to_check: {times_to_check}, '
                    #       f'eq: {times_to_check == orig_times}')
                    if times_to_check == orig_times:
                        # print(f'removing_intersect_seq neurons_seq {neurons_seq}, '
                        #       f'neurons_tuple_to_test {neurons_tuple_to_test}, times_seq_set: {orig_times}, '
                        #       f'times_to_check: {times_to_check}')
                        keys_to_delete.add(neurons_tuple_to_test)
    for key in keys_to_delete:
        if len(key) in dict_by_len_seq:
            dict_by_len_seq[len(key)].pop(key, None)
        set_seq_dict_result.pop(key, None)


def build_time_diff_dict(times_seq_set, time_diff_threshold=2):
    """
    Take a set of tuple of times and return a dict with the key being a tuple of int representing the diff
    of times and the value a list the times tuples
    :param times_seq_set:
    :param time_diff_threshold:
    :return:
    """
    times_diff_dict = {}
    for times_tuple in times_seq_set:
        times_diff = tuple(np.diff(np.array(times_tuple)))
        if times_diff in times_diff_dict:
            times_diff_dict[times_diff].append(times_tuple)
        else:
            # we look if one time diff is similar with respect of time_diff_threshold
            match_not_found = True
            for k in times_diff_dict.keys():
                if len(k) == len(times_diff):
                    # from tuple to array
                    substr = np.abs(np.array(k) - np.array(times_diff))
                    # print(f'k {k}, times_diff {times_diff}, substr {substr}')
                    similar = np.all(np.less_equal(substr, time_diff_threshold))
                    if similar:
                        times_diff_dict[k].append(times_tuple)
                        match_not_found = False
                        break
            # otherwise new diff
            if match_not_found:
                times_diff_dict[times_diff] = [times_tuple]
    return times_diff_dict


def find_sequences(spike_nums, param,
                   no_print=True):
    """

    :param spike_nums:
    :param param:
    :param no_print:
    :return: return dict_by_len_seq, see comment in the code
    """
    time_inter_seq = param.time_inter_seq
    n_order = param.n_order
    min_len_seq = param.min_len_seq
    stop_if_twin = param.stop_if_twin
    max_depth = param.max_depth
    random_mode = param.random_mode
    keeping_only_same_diff_seq = param.keeping_only_same_diff_seq
    file = param.file
    write_on_file = param.write_on_file

    # Max len in term of repetition of the max len seq in terms of neuron
    max_rep_non_prob = 0
    max_len_non_prob = 0
    # a dict of dict, with each key of the first dict representing the length
    # of the sequences. The 2nd dict will have as key sequences of neurons and as value the time of the neurons spikes
    dict_by_len_seq = dict()
    # used to remove spikes from the same cell, after the cell fires a spike, all spikes fired by the same
    # cell during the following threshold_duration will be removed (1 second = 20)
    if random_mode:
        nb_paste = len(spike_nums[0, :]) // 5000
        for n in np.arange(nb_paste):
            spike_nums[:, (n * 5000):(n * 5000) + 3] = 0
        spike_nums = generate_random_spikes_num(len(spike_nums), model=spike_nums)
        for y, neuron in enumerate(spike_nums):
            positive_indexes = np.where(neuron)[0]
            if len(positive_indexes) < 2:
                continue
            # print(f'positive_indexes {positive_indexes}, threshold_duration {threshold_duration}')
            mask = np.ediff1d(positive_indexes, to_begin=param.threshold_duration + 1) <= param.threshold_duration
            # print(f'len(mask): {len(mask)}, len(positive_indexes): {len(positive_indexes)}')
            spike_nums[y, positive_indexes[mask]] = 0

    print_spikes_count_by_neuron = True
    if print_spikes_count_by_neuron:
        print_save(f'Spikes neurons', param.file, no_print=True, to_write=param.write_on_file)
        for n, neuron in enumerate(spike_nums):
            print_save(f'{n}: {np.sum(neuron)}', param.file, no_print=True, to_write=param.write_on_file)
        print_save(f'total spikes: {np.sum(spike_nums)}', param.file, no_print=True, to_write=param.write_on_file)
        print_save(f'Ratio spikes: {np.sum(spike_nums)/(len(spike_nums)*len(spike_nums[0]))}',
                   param.file, no_print=True,
                   to_write=param.write_on_file)

    # print(f'nb spikes neuron 1: {len(np.where(spike_nums[1,:])[0])}')

    # spike_nums_backup = spike_nums
    spike_nums = np.copy(spike_nums)
    # spike_nums = spike_nums[:, :2000]
    nb_neurons = len(spike_nums)

    if param.transition_order == 2:
        transition_dict = build_2_nd_order_transition_dict_v2(spike_nums=spike_nums, param=param)
    else:
        # transition_dict = build_transition_dict(spike_nums=spike_nums,
        #                                         show_heatmap=param.show_heatmap, param=param)
        transition_dict = build_mle_transition_dict(spike_nums=spike_nums, param=param)
    # print("build_transition_dict done")
    # use to keep strong probability
    threshold_prob = (1 / (nb_neurons - 1)) * param.threshold_factor

    # len of nb_neurons, each element is a dictionary with each key represent a common seq (neurons tuple)
    # and each values represent a list of list of times
    list_dict_result = []

    # key: neuron as integer, value: list of neurons being the longest probable sequence
    max_seq_dict = dict()
    # Start to look for real sequences in spike_nums
    for y, neuron in enumerate(spike_nums):
        if param.transition_order == 2:
            go_for_bfs = True
            # build the tree of sequences with the top node being the neuron y
            if go_for_bfs:
                tree = bfs_2nd_order(trans_dict=transition_dict, neuron_to_start=y, param=param)
                print("end bfs_2nd_order")
            else:
                tree = get_seq_n_order_2nd_order_dict(trans_dict=transition_dict, neuron_to_start=y,
                                                      threshold=threshold_prob,
                                                      n_order=n_order, max_depth=max_depth, depth=0, father=None,
                                                      stop_if_twin=stop_if_twin, param=param)
        else:
            go_for_bfs = True
            # build the tree of sequences with the top node being the neuron y
            if go_for_bfs:
                go_for_bfs_v2 = False
                if go_for_bfs_v2:
                    print("begin bfs v2")
                    res_dict = bfs_v2(trans_dict=transition_dict, param=param)
                    print("end bfs v2")
                else:
                    tree = bfs(trans_dict=transition_dict, neuron_to_start=y, param=param)
            else:
                tree = get_seq_n_order(trans_dict=transition_dict, neuron_to_start=y, threshold=threshold_prob,
                                       n_order=n_order, max_depth=max_depth, depth=0, father=None,
                                       stop_if_twin=stop_if_twin, param=param)
        print_save('@' * 100, param.file, no_print=True, to_write=param.write_on_file)
        print_save(('@' * 50) + f' {y} ' + ('@' * 50), param.file, no_print=True, to_write=param.write_on_file)
        print_save('@' * 100, param.file, no_print=True, to_write=param.write_on_file)
        # print_save(f'max_seq_by_total_prob for neuron {y}: {tree.max_seq_by_total_prob()}', param.file, no_print=False,
        #            to_write=param.write_on_file)
        # print(f'before tree.get_seq_lists()')
        # from tree to list of list
        sequences = tree.get_seq_lists()
        print_save(f'Nb probabilities seq for neuron {y}: {len(sequences)}', param.file, no_print=False,
                   to_write=param.write_on_file)
        # TODO: continue modifications
        # seq = find_seq(trans_dict=transition_dict, neuron_to_start=y, threshold=threshold_prob)
        seq_dict_result = dict()
        # key is a tuple of neurons (int) representing a sequence, value is a set of tuple of times (int) representing
        # the times at which the seq is repeating
        set_seq_dict_result = dict()
        # error_by_seq_dict a dict indicating for each neuron seq instance how many errors was
        # found
        # not correlated to the times of each instance, but allow to do the min and the mean
        error_by_seq_dict = dict()
        # Give the max length among the sequence computed as the most probable
        max_len_prob_seq = 0
        for seq in sequences:
            # print(f"len seq in sequences: {len(seq)}")
            # look for each spike of this neuron, if a seq is found on the following spikes of other neurons
            for t in np.where(neuron)[0]:
                # look if the sequence is right
                index = t + param.min_duration_intra_seq
                if index < 0:
                    index = 0
                elif t >= len(spike_nums[0, :]):
                    index = len(spike_nums[0, :]) - 1
                # if not seq_can_be_on_same_time:
                #     if t >= (len(spike_nums[0, :]) - 1):
                #         break
                #     else:
                #         index = t + 1
                # else:
                #     if t > 1:
                #         index = t-2
                #     else:
                #         index = t
                # keep neurons numbers
                neurons_sequences = []
                times_sequences = []
                # How many times which outpass the fact that one neuron was missing
                nb_error = 0
                # Check if the last neuron added to the sequence is an error
                # the sequence must finish by a predicted neuron
                last_is_error = np.zeros(2)

                # for each neuron of the most probable sequence
                for nb_s, n_seq in enumerate(seq):
                    new_time_inter_seq = time_inter_seq
                    if last_is_error.any():
                        to_add = (time_inter_seq * 2)
                        if last_is_error.all():
                            to_add *= 2
                        if (index + to_add) < len(spike_nums[n_seq, :]):
                            new_time_inter_seq = to_add
                    # If too many neurons actives during the following time_inter_seq
                    # then we stop here
                    # TODO: Keep it ?
                    take_sce_into_account = False
                    if take_sce_into_account:
                        if np.sum(spike_nums[:, index:(index + new_time_inter_seq)]) > \
                                (0.0128 * new_time_inter_seq * len(spike_nums)):
                            print(
                                f'np.sum(spike_nums[:, index:(index + new_time_inter_seq)]) > (0.0128*len(spike_nums))')
                            break
                    nb_neurons_error_in_a_raw = param.error_rate
                    neuron_error_index = 0
                    # look if the neuron n_seq during the next time_inter_seq time is activated
                    if np.any(spike_nums[n_seq, index:(index + new_time_inter_seq)]):
                        time_index_list = np.where(spike_nums[n_seq, index:(index + new_time_inter_seq)])[0]
                        neurons_sequences.append(n_seq)
                        # print(f'time_index {time_index}')
                        next_spike_time = np.min(time_index_list)
                        # print(f'index {index}, next_spike_time {next_spike_time}')
                        times_sequences.append(index + next_spike_time)
                        index += next_spike_time
                        last_is_error = np.zeros(nb_neurons_error_in_a_raw)
                        neuron_error_index = 0
                        if (index + time_inter_seq) >= (len(spike_nums[n_seq, :]) - 1):
                            # if nb_s >= (min_len_seq - 1):
                            #     neurons_sequences = seq[:nb_s + 1]
                            break
                    else:
                        # the last neuron should be a neuron from the sequence, we just skip one, two neurons can't be
                        # skip one after the other one
                        # TODO: see to put more than 2 neurons as error ?
                        if (nb_error < param.error_rate) and (not last_is_error.all()):
                            nb_error += 1
                            last_is_error[neuron_error_index] = 1
                            neuron_error_index += 1
                            # if last_is_error.any():
                            #     last_is_error[1] = 1
                            # else:
                            #     last_is_error[0] = 1
                            # appending n_seq even so it'n_seq not found
                            neurons_sequences.append(n_seq)
                            times_sequences.append(index)
                        else:
                            break
                if last_is_error.any():
                    to_remove = int(-1 * np.sum(last_is_error))
                    nb_error -= np.sum(last_is_error)
                    # print(f'to_remove {to_remove}')
                    if len(neurons_sequences) > 0:
                        neurons_sequences = neurons_sequences[:to_remove]
                        times_sequences = times_sequences[:to_remove]
                # saving the sequence only if its duration has a mimimum length and repeat a minimum of times
                # TODO: verifier que len(times_sequences) ne soit pas identifique à len(neurons_sequences)
                if (len(neurons_sequences) >= param.min_len_seq) and (len(times_sequences) >= param.min_rep_nb):
                    time_neurons_seq = tuple(times_sequences)
                    neurons_sequences = tuple(neurons_sequences)
                    len_seq = len(neurons_sequences)

                    add_seq = True
                    if neurons_sequences not in set_seq_dict_result:
                        set_seq_dict_result[neurons_sequences] = set()
                        set_seq_dict_result[neurons_sequences].add(time_neurons_seq)
                        # we need to check if some time sequences don't intersect too much with this one,
                        # otherwise we don't add the seq to the dictionnary
                    else:
                        for t_seq in set_seq_dict_result[neurons_sequences]:
                            perc_threshold = 0.3
                            inter = np.intersect1d(np.array(time_neurons_seq), np.array(t_seq))
                            if len(inter) > (0.2 * len(t_seq)):
                                add_seq = False
                                break
                        if add_seq:
                            set_seq_dict_result[neurons_sequences].add(time_neurons_seq)

                    if add_seq:
                        # error_by_seq_dict a dict indicating for each neuron seq instance how many errors was
                        # found
                        # not correlated to the times of each instance, but allow to do the min and the mean
                        if neurons_sequences not in error_by_seq_dict:
                            error_by_seq_dict[neurons_sequences] = []
                        error_by_seq_dict[neurons_sequences].append(nb_error)
                        if len(seq) > max_len_prob_seq:
                            # Keep the whole sequences, not just the segment that was actually on the plot
                            # But only if at least one part of sequence was found on the plot, and a minimum
                            # of time (set into param)
                            max_len_prob_seq = len(seq)
                            max_seq_dict[y] = seq
                        # dict_by_len_seq will be return
                        if len_seq not in dict_by_len_seq:
                            dict_by_len_seq[len_seq] = dict()
                        if neurons_sequences not in dict_by_len_seq[len(neurons_sequences)]:
                            dict_by_len_seq[len_seq][neurons_sequences] = set()
                        dict_by_len_seq[len_seq][neurons_sequences].add(time_neurons_seq)

                        if time_neurons_seq not in seq_dict_result:
                            seq_dict_result[time_neurons_seq] = neurons_sequences

                    # neurons_seq.append(neurons_sequences)
                    # time_neurons_seq.append(times_sequences)
        print_save(f'Len max for neuron {y} prob seq: {max_len_prob_seq}', file, no_print=False,
                   to_write=write_on_file)
        # print_save(f'Nb seq for neuron {y}: {len(seq_dict_result)}', file, no_print=True, to_write=write_on_file)
        # if more than one sequence for one given neuron, we draw them
        if len(seq_dict_result) > 1:
            list_dict_result.append(set_seq_dict_result)
            # removing seq for which no instance have less than 2 errors comparing to probabilistic seq
            for k, v in error_by_seq_dict.items():
                if np.min(v) > 1:
                    # print(f'min nb errors: {np.min(v)}')
                    set_seq_dict_result.pop(k, None)
                    dict_by_len_seq[len(k)].pop(k, None)
            # remove intersecting seq and seq repeting less than
            # TODO: change it to work on dict_by_len_seq
            removing_intersect_seq(set_seq_dict_result, min_len_seq, param.min_rep_nb, dict_by_len_seq)
            for neurons_seq, times_seq_set in set_seq_dict_result.items():
                # if (len(times_seq_set) > 2) and (len(neurons_seq) >= param.min_len_seq_first_tour):
                if (len(times_seq_set) > max_rep_non_prob) and \
                        (len(neurons_seq) >= np.max(param.min_len_seq_first_tour)):
                    max_rep_non_prob = len(times_seq_set)
                    max_len_non_prob = len(neurons_seq)

                    # keeping the max len of neurons in a seq, if that seq is repeated more than 2 times.
                    # keeping also the max rep of that given seq
                    last_version = False
                    if last_version:
                        if max_len_non_prob < len(neurons_seq):
                            max_len_non_prob = len(neurons_seq)
                            max_rep_non_prob = len(times_seq_set)
                        if (max_len_non_prob == len(neurons_seq)) and (len(times_seq_set) > max_rep_non_prob):
                            max_rep_non_prob = len(times_seq_set)
            if (not no_print) or write_on_file:
                # set_seq_dict_result: each key represent a common seq (neurons tuple)
                # and each values represent a list of list of times
                print_save(f'Nb unique seq for neuron {y}: {len(set_seq_dict_result)}', file, no_print=True,
                           to_write=write_on_file)
                for neurons_seq, times_seq_set in set_seq_dict_result.items():
                    if (len(times_seq_set) > 1) and (len(neurons_seq) >= min_len_seq):
                        # keeping the max len of neurons in a seq, if that seq is repeated more than 2 times.
                        # keeping also the max rep of that given seq
                        print_save('', file, no_print=True, to_write=write_on_file)
                        print_save('/\\' * 50, file, no_print=True, to_write=write_on_file)
                        if keeping_only_same_diff_seq:
                            # at_least_one = False
                            times_diff_dict = build_time_diff_dict(times_seq_set=times_seq_set, time_diff_threshold=2)
                            for k, v in times_diff_dict.items():
                                if len(v) > 1:
                                    times_diff_list = []
                                    times_diff_sum_list = []
                                    for times_tuple in v:
                                        times_diff_list.append(tuple(np.diff(np.array(times_tuple))))
                                        times_diff_sum_list.append(np.sum(np.diff(np.array(times_tuple))))
                                    print_save(f"{'*' * 5} len {len(neurons_seq)}, nb(seq) {len(v)}, ", file,
                                               no_print=True, to_write=write_on_file)
                                    print_save(f"{'*' * 5} neurons: {neurons_seq}", file, no_print=True,
                                               to_write=write_on_file)
                                    print_save(f"{'*' * 5} same_diff: {times_diff_list}", file, no_print=True,
                                               to_write=write_on_file)
                                    print_save(f"{'*' * 5} sum diff: {times_diff_sum_list}", file, no_print=True,
                                               to_write=write_on_file)
                                    print_save(f"{'*' * 5} times: {v}", file, no_print=True, to_write=write_on_file)
                                    print_save('', file, no_print=True, to_write=write_on_file)
                        times_diff_list = []
                        times_diff_sum_list = []
                        for times_tuple in times_seq_set:
                            times_diff_list.append(tuple(np.diff(np.array(times_tuple))))
                            times_diff_sum_list.append(np.sum(np.diff(np.array(times_tuple))))
                        print_save(f'len {len(neurons_seq)}, nb(seq) {len(times_seq_set)}, ', file, no_print=True,
                                   to_write=write_on_file)
                        print_save(f'neurons: {neurons_seq}', file, no_print=True, to_write=write_on_file)
                        print_save(f'diff: {times_diff_list}', file, no_print=True, to_write=write_on_file)
                        print_save(f'sum diff: {times_diff_sum_list}', file, no_print=True, to_write=write_on_file)
                        print_save(f'times: {times_seq_set}', file, no_print=True, to_write=write_on_file)
                        print_save('', file, no_print=True, to_write=write_on_file)

        print_save('', file, no_print=True, to_write=write_on_file)
    # max_rep_non_prob is the maximum rep of a sequence found in spike_nums
    return list_dict_result, dict_by_len_seq, max_seq_dict, max_rep_non_prob, max_len_non_prob


def plot_sequences(list_seq, list_neurons):
    pass


def seq_candidates(trans_dict, neuron, threshold, nb_of_candidates):
    """

    :param trans_dict:
    :param neuron:
    :param threshold:
    :param nb_of_candidates:
    :return:
    """
    list_candidates = []
    transition_dict = np.copy(trans_dict)
    for i in np.arange(nb_of_candidates):
        # return the index of the max probability neuron
        maxi = np.argmax(transition_dict[neuron, :])
        if trans_dict[neuron, maxi] > threshold:
            list_candidates.append(maxi)
            where_maxi = np.where(transition_dict[neuron, :] == transition_dict[neuron, maxi])[O]
            if len(where_maxi) > 0:
                print(f"More than on max prob: {where_maxi}")
            transition_dict[neuron, maxi] = -1
        else:
            print('seq_candidates break')
            break
    return list_candidates


def get_spike_rates(spike_nums):
    """
    Give the spike rate for each neuron (float between 0 and 1)
    :param spike_nums:
    :return:
    """
    (nb_neurons, nb_time) = spike_nums.shape
    spike_rates = np.zeros(nb_neurons)
    for n, neuron in enumerate(spike_nums):
        spike_rates[n] = np.sum(neuron) / nb_time
    return spike_rates


def weight_value_by_neuron(spike_nums, param):
    nb_neurons = len(spike_nums)
    weights = np.ones(nb_neurons)
    # weights = (weights / (nb_neurons - 1)) * param.value_to_add_factor
    total_spikes = np.sum(spike_nums)
    for n_i, n in enumerate(spike_nums):
        spike_count = np.sum(n)
        if spike_count > 0:
            weights[n_i] = param.value_to_add_factor * (
                    (1 / nb_neurons) / spike_count)  # (spike_count / total_spikes) # * (1/nb_neurons)
        else:
            weights[n_i] = 1
    return weights


def find_following_neurons_spiking(spike_nums, t, param):
    """
    For a giving time, return the neurons firing
    :param spike_nums:
    :param t:
    :param param:
    :return: pos: list of neurons, time_p: list of time corresponding to the neurons
    """
    # t represents the time from which to look
    if (t + param.min_duration_intra_seq) < 0:
        t = 0
    else:
        t = t + param.min_duration_intra_seq
    t_max = t + param.time_inter_seq
    if t_max > (len(spike_nums[0]) - 1):
        t_max = len(spike_nums[0]) - 1
    # print(f'find_following_neurons_spiking t: {t}, t_max: {t_max}, len(spike_nums) {len(spike_nums)}')
    if np.sum(spike_nums[:, t:t_max]) > (0.0128 * (t_max - t) * len(spike_nums)):
        # print(f'spike_nums[:, t:t_max]) > (0.0128 * (t_max-t) * len(spike_nums))')
        return None, None

    pos = np.where(spike_nums[:, t:t_max])[0]
    # looking for the associated time of the neurons found
    time_p = np.zeros(len(pos), dtype="int8")
    for p_i, p in enumerate(pos):
        # print(f'find_following_neurons_spiking p {p}, t {t}')
        new_t = np.where(spike_nums[p, t:t_max])[0]
        time_p[p_i] = new_t[0]

    return pos, time_p


def build_2_nd_order_transition_dict_v2(spike_nums, param):
    nb_neurons = len(spike_nums)
    # keep for each cells the probabilty that the other cell is the following one in a sequence
    # The sum of probability for one cell is 1
    transition_dict = np.zeros((nb_neurons, nb_neurons, nb_neurons))

    # so the neuron with the lower spike rates gets the biggest weight in terms of probability
    if param.spike_rate_weight:
        spike_rates = 1 - get_spike_rates(spike_nums)
    else:
        spike_rates = np.ones(nb_neurons)

    for n, neuron in enumerate(spike_nums):
        # t represents time index when the n neuron is spiking
        for t in np.where(neuron)[0]:
            # dict with key the second neuron of the seq, the value is a list with the 3rd neurons
            neurons_couples = dict()
            pos, time_p = find_following_neurons_spiking(spike_nums, t, param)
            # pos: neuron index / time_p: time index
            if pos is None:
                continue
            # to be sure we don't have 2 times the same neuron, in case time_inter_seq would be > to our threshold
            # to avoid somation
            pos = np.unique(pos)
            mask = np.not_equal(pos, n)
            pos = pos[mask]
            time_p = time_p[mask]
            # then looking for each following neuron, the other following neurons
            for p_i, p in enumerate(pos):
                new_pos, new_time_p = find_following_neurons_spiking(spike_nums, time_p[p_i], param)
                if new_pos is None:
                    continue
                new_pos = np.unique(new_pos)
                # shouldn't be 2 times the same neurons
                new_pos = new_pos[np.not_equal(new_pos, p)]
                new_pos = new_pos[np.not_equal(new_pos, n)]
                for n_p in new_pos:
                    if p not in neurons_couples:
                        neurons_couples[p] = []
                    neurons_couples[p].append(n_p)

            # updating the transition matrix
            for k, v in neurons_couples.items():
                for third_neuron in v:
                    transition_dict[n, k, third_neuron] += (spike_rates[k] + spike_rates[third_neuron])

            if param.no_reverse_seq:
                pass

    for n, neuron in enumerate(spike_nums):
        # the sum of all values should be equal to 1, for a given neuron
        if np.sum(transition_dict[n, :, :]) > 0:
            transition_dict[n, :, :] = transition_dict[n, :, :] / np.sum(transition_dict[n, :, :])
        # print(f"np.sum(transition_dict[n, :, :]) {np.sum(transition_dict[n, :, :])}, "
        #       f"np.min(transition_dict[n, :, :]) {np.min(transition_dict[n, :, :])}, "
        #       f"np.max(transition_dict[n, :, :]) {np.max(transition_dict[n, :, :])}")

    print_transit_dict = False
    if print_transit_dict:
        for i in np.arange(nb_neurons):
            for j in np.arange(nb_neurons):
                print(f'transition dict, n {i}-{j}, sum: {np.sum(transition_dict[i, j, :])}')
                print(f'transition dict, n {i}-{j}, max: {np.max(transition_dict[i, j, :])}')
                print(f'transition dict, n {i}-{j}, nb max: '
                      f'{len(np.where(transition_dict[i, j, :] == np.max(transition_dict[i, j, :]))[0])}')
            # sns.set()
    print(f'mean transition: {np.mean(transition_dict)}')
    print(f'median transition: {np.median(transition_dict)}')
    print(f'std transition: {np.std(transition_dict)}')
    print(f'min transition: {np.min(transition_dict)}')
    print(f'max transition: {np.max(transition_dict)}')

    return transition_dict


# TODO: a revoir
def build_2_nd_order_transition_dict(spike_nums, param):
    nb_neurons = len(spike_nums)
    # keep for each cells the probabilty that the other cell is the following one in a sequence
    # The sum of probability for one cell is 1
    transition_dict = np.ones((nb_neurons, nb_neurons, nb_neurons))
    # The probability that the same neuron fire 2 or 3 times in a row or in the seq of 3 is 0
    # np.fill_diagonal(transition_dict, 0)
    for i in np.arange(nb_neurons):
        for j in np.arange(nb_neurons):
            transition_dict[i, i, j] = 0
            transition_dict[i, j, i] = 0
    transition_dict = transition_dict / (nb_neurons - 1)
    # the value to add to a neuron for which the probability increase to be on a sequence (firing a short
    # time after a given one)
    added_prob_value = (1 / (nb_neurons - 1)) * param.value_to_add_factor

    spikes_weight = weight_value_by_neuron(spike_nums, param)

    for n, neuron in enumerate(spike_nums):
        # t represents time index when the n neuron is spiking
        for t in np.where(neuron)[0]:
            # dict with key the second neuron of the seq, the value is a list with the 3rd neurons
            neurons_couples = dict()
            pos, time_p = find_following_neurons_spiking(spike_nums, t, param)
            # pos: neuron index / time_p: time index
            if pos is None:
                continue
            # to be sure we don't have 2 times the same neuron, in case time_inter_seq would be > to our threshold
            # to avoid somation
            pos = np.unique(pos)
            mask = np.not_equal(pos, n)
            pos = pos[mask]
            time_p = time_p[mask]
            # then looking for each following neuron, the other following neurons
            for p_i, p in enumerate(pos):
                new_pos, new_time_p = find_following_neurons_spiking(spike_nums, time_p[p_i], param)
                if new_pos is None:
                    continue
                new_pos = np.unique(new_pos)
                # shouldn't be 2 times the same neurons
                new_pos = new_pos[np.not_equal(new_pos, p)]
                new_pos = new_pos[np.not_equal(new_pos, n)]
                for n_p in new_pos:
                    if p not in neurons_couples:
                        neurons_couples[p] = []
                    neurons_couples[p].append(n_p)

            # updating the transition matrix
            for k, v in neurons_couples.items():
                n_to_increases = np.array(v)
                # non_zeros_pos represents the seq whose prob is not at 0
                non_zeros_pos = np.greater(transition_dict[n, k, n_to_increases], 0)
                non_zeros = np.greater(transition_dict[n, k, :], 0)
                non_zeros = np.arange(nb_neurons)[non_zeros]
                # neuron not active during a given period after the action of neuron n at time t
                neurons_to_decrease = np.setdiff1d(non_zeros, n_to_increases[non_zeros_pos])
                total_value_added = 0
                # Adding probability to neurons following n, with a weight depending on his spike frequency
                for n_i in n_to_increases:
                    # print(f'{spikes_weight[n_i]}')
                    if transition_dict[n, k, n_i] + spikes_weight[n_i] <= 1:
                        transition_dict[n, k, n_i] += spikes_weight[n_i]
                        total_value_added += spikes_weight[n_i]
                    else:
                        total_value_added += (1 - transition_dict[n, k, n_i])
                        transition_dict[n, k, n_i] = 1
                if len(neurons_to_decrease) > 0:
                    transition_dict[n, k, neurons_to_decrease] -= total_value_added / len(neurons_to_decrease)

            if param.no_reverse_seq:
                pass

    print_transit_dict = False
    if print_transit_dict:
        for i in np.arange(nb_neurons):
            for j in np.arange(nb_neurons):
                print(f'transition dict, n {i}-{j}, sum: {np.sum(transition_dict[i, j, :])}')
                print(f'transition dict, n {i}-{j}, max: {np.max(transition_dict[i, j, :])}')
                print(f'transition dict, n {i}-{j}, nb max: '
                      f'{len(np.where(transition_dict[i, j, :] == np.max(transition_dict[i, j, :]))[0])}')
            # sns.set()
    print(f'mean transition: {np.mean(transition_dict)}')
    print(f'std transition: {np.std(transition_dict)}')
    print(f'min transition: {np.min(transition_dict)}')
    print(f'max transition: {np.max(transition_dict)}')

    return transition_dict


def build_mle_transition_dict(spike_nums, param):
    """
    Maximum Likelihood estimation,
    don't take into account the fact that if a neuron A fire after a neuron B , then it decreases the probability than B fires after A
    :param spike_nums:
    :param param:
    :return:
    """
    nb_neurons = len(spike_nums)
    transition_dict = np.zeros((nb_neurons, nb_neurons))

    # so the neuron with the lower spike rates gets the biggest weight in terms of probability
    if param.spike_rate_weight:
        spike_rates = 1 - get_spike_rates(spike_nums)
    else:
        spike_rates = np.ones(nb_neurons)

    # a first turn to put probabilities up from neurons B that spikes after neuron A
    for n, neuron in enumerate(spike_nums):
        # will count how many spikes of each neuron are following the spike of
        # tmp_count = np.zeros(nb_neurons)
        for t in np.where(neuron)[0]:
            original_t = t
            if (t + param.min_duration_intra_seq) < 0:
                t = 0
            else:
                t = t + param.min_duration_intra_seq
            t_max = t + param.time_inter_seq
            if t_max > (len(spike_nums[0]) - 1):
                t_max = len(spike_nums[0]) - 1

            # TODO: do something similar to Robin, to detect events, with percentile
            # doesn't change anything
            # if np.sum(spike_nums[:, t:t_max]) > (0.0128 * (t_max - t) * len(spike_nums)):
            #     print(f'spike_nums[:, t:t_max]) > (0.0128 * (t_max-t) * len(spike_nums))')
            #     continue

            spike_nums[n, original_t] = 0

            pos = np.where(spike_nums[:, t:t_max])[0]
            # pos = np.unique(pos)
            for p in pos:
                transition_dict[n, p] = transition_dict[n, p] + spike_rates[p]
                if param.no_reverse_seq:
                    # see to put transition to 0 ??
                    # In accordance
                    # with real biological networks we assume that the probability of a connection being
                    # excitatory p+ is five times that of being inhibitory p−,
                    transition_dict[p, n] = transition_dict[p, n] - (spike_rates[p]/5)
            # transition_dict[n, pos] = transition_dict[n, pos] + 1

            spike_nums[n, original_t] = 1
        # just in case, tmp_count[n] should be equal to zero
        if transition_dict[n, n] > 0:
            print(f"transition_dict[{n}, {n}] > 0: {transition_dict[n, n]}")
        transition_dict[n, n] = 0

    # we divide for each neuron by the sum of the probabilities
    for n, neuron in enumerate(spike_nums):
        # print(f'n {n}, len(np.where(transition_dict[n, :] < 0)[0]) {len(np.where(transition_dict[n, :] < 0)[0])}')
        # all negatives values should be put to zero
        transition_dict[n, np.where(transition_dict[n, :] < 0)[0]] = 0
        if np.sum(transition_dict[n, :]) > 0:
            transition_dict[n, :] = transition_dict[n, :] / np.sum(transition_dict[n, :])
        else:
            print(f"np.sum(transition_dict[n, :]) <= 0: {np.sum(transition_dict[n, :])}")

    print_transit_dict = False
    if print_transit_dict:
        for n, neuron in enumerate(spike_nums):
            print(f'transition dict, n {n}, sum: {np.sum(transition_dict[n, :])}')
            print(f'transition dict, n {n}, max: {np.max(transition_dict[n, :])}')
            print(f'transition dict, n {n}, nb max: '
                  f'{np.where(transition_dict[n, :] == np.max(transition_dict[n, :]))[0]}')
    print(f'mean transition: {np.mean(transition_dict)}')
    print(f'std transition: {np.std(transition_dict)}')
    print(f'min transition: {np.min(transition_dict)}')
    print(f'max transition: {np.max(transition_dict)}')
    return transition_dict


# old version
def build_transition_dict(spike_nums, param,
                          show_heatmap=False):
    """
    :param spike_nums:
    :param param:
    :param show_heatmap:
    :return:
    """
    nb_neurons = len(spike_nums)

    # keep for each cells the probabilty that the other cell is the following one in a sequence
    # The sum of probability for one cell is 1
    transition_dict = np.ones((nb_neurons, nb_neurons))
    # The probability that the next neuron to spike is twice the same if null
    np.fill_diagonal(transition_dict, 0)
    transition_dict = transition_dict / (nb_neurons - 1)
    # the value to add to a neuron for which the probability increase to be on a sequence (firing a short
    # time after a given one)
    added_prob_value = (1 / (nb_neurons - 1)) * param.value_to_add_factor
    # added_prob_value = 1/100
    # len 4, nb(seq) 6, neurons: (0, 118, 95, 116

    # transition_dict = np.zeros((len(spike_nums), len(spike_nums)))

    spikes_weight = weight_value_by_neuron(spike_nums, param)

    for n, neuron in enumerate(spike_nums):
        # print(f'In build_transition_dict, neuron {n}')
        # t represents time index when the n neuron is spiking
        for t in np.where(neuron)[0]:
            # putting spike to actual neuron to 0
            # if t == 0:
            #     time_frame[n] = 0
            original_t = t
            if (t + param.min_duration_intra_seq) < 0:
                t = 0
            else:
                t = t + param.min_duration_intra_seq
            t_max = t + param.time_inter_seq
            if t_max > (len(spike_nums[0]) - 1):
                t_max = len(spike_nums[0]) - 1

            # if there is a SCB, we don't take this part in consideration
            if np.sum(spike_nums[:, t:t_max]) > (0.0128 * (t_max - t) * len(spike_nums)):
                print(f'spike_nums[:, t:t_max]) > (0.0128 * (t_max-t) * len(spike_nums))')
                continue
            # np.sum(spike_nums[:, t:t_max]) > (0.3 * len(spike_nums))

            # useful if we don't want to consider neuron active at the same time as a potential follower
            # if not seq_can_be_on_same_time:
            #     if t >= (len(spike_nums[n, :])-1):
            #         break

            spike_nums[n, original_t] = 0
            # if not seq_can_be_on_same_time:
            #     pos_0 = np.where(spike_nums[:, t+1])[0]
            # else:
            pos = np.where(spike_nums[:, t:t_max])[0]
            # pos represent the indexes of neurons activated after n
            # pos = pos_0
            # position of neurons active after (in term of time) the firing of the neuron n,
            # pos_after_0 = np.array([], dtype="int8")
            # print(f"pos {pos}")
            time_to_add = 1
            # if not seq_can_be_on_same_time:
            #     time_to_add = 2
            # for t_i in (np.arange(time_inter_seq) + time_to_add):
            #     if t < (len(spike_nums[n, :]) - t_i):
            #         pos_t_i = np.where(spike_nums[:, t + t_i])[0]
            #         # print(f"pos2 {pos_2}")
            #         pos = np.concatenate((pos, pos_t_i))
            #         pos_after_0 = np.concatenate((pos_after_0, pos_t_i))
            # to be sure we don't have 2 times the same neuron, in case time_inter_seq would be > to our threshold
            # to avoid somation
            pos = np.unique(pos)

            # all neurons that are spiking at the same time as neuron n (represented by pos have an equal
            # chance to be the next one in the sequence. And neuron n become an outsider for being the following event
            # to any of those neurons
            # Keeping those that fires
            non_zeros_pos = np.greater(transition_dict[n, pos], 0)
            non_zeros = np.greater(transition_dict[n, :], 0)
            non_zeros = np.arange(nb_neurons)[non_zeros]
            # neuron not active during a given period after the action of neuron n at time t
            neurons_to_decrease = np.setdiff1d(non_zeros, pos[non_zeros_pos])
            # sum of transition_dict[non_zeros] should be equal to 1
            # TODO: idea: keep in an array the value to add when met, and think about the same but for value to remove
            neurons_to_increase = pos
            # spikes_weight = spikes_weight[neurons_to_increase]
            if len(neurons_to_increase) > 0:  # (len(neurons_to_increase) > 0) and (len(neurons_to_decrease) > 0):
                # done to increase the weight the less neurons fire after n,
                # as it makes them more likely to be the next one
                # added_prob_value_modif = added_prob_value / len(neurons_to_increase)

                it_is_now = True
                if it_is_now:
                    # neurons_to_decrease = np.setdiff1d(np.arange(nb_neurons), pos)

                    total_value_added = 0
                    # to_put_to_one = np.greater_equal(transition_dict[n, neurons_to_increase],
                    #                                  (1 - spikes_weight[neurons_to_increase]))
                    # if np.sum(to_put_to_one) > 0:
                    #     pass
                    # else:
                    #     transition_dict[n, neurons_to_increase] += spikes_weight[neurons_to_increase]
                    # total_value_added = (added_prob_value_modif * len(neurons_to_increase))
                    # transition_dict[n, neurons_to_increase[to_increases]] += added_prob_value_modif
                    # Adding probability to neurons following n, with a weight depending on his spike frequency
                    #
                    for n_i in neurons_to_increase:
                        # print(f'{spikes_weight[n_i]}')
                        if transition_dict[n, n_i] + spikes_weight[n_i] <= 1:
                            transition_dict[n, n_i] += spikes_weight[n_i]
                            total_value_added += spikes_weight[n_i]
                        else:
                            total_value_added += (1 - transition_dict[n, n_i])
                            transition_dict[n, n_i] = 1
                    transition_dict[n, neurons_to_decrease] -= total_value_added / len(neurons_to_decrease)
                # print(f'{transition_dict}')
                it_was_before = False
                if it_was_before:
                    added_prob_value_modif = added_prob_value
                    max_sum_to_substract = np.sum(transition_dict[n, neurons_to_decrease])
                    if (added_prob_value_modif * len(neurons_to_increase)) > max_sum_to_substract:
                        added_prob_value_modif = max_sum_to_substract / len(neurons_to_increase)

                    # section to take into account neuron for which (1- probability) < added_prob_value
                    # to make sure it doesn't end with value > 1
                    to_put_to_one = np.greater_equal(transition_dict[n, neurons_to_increase],
                                                     1 - added_prob_value_modif)
                    to_increases = np.less(transition_dict[n, neurons_to_increase], 1 - added_prob_value_modif)
                    # max_sum_to_add = np.sum(1 - transition_dict[n, neurons_to_increase])

                    # np.sum(1 - transition_dict[n, neurons_to_increase[to_put_to_one]])
                    # print(f"max to_put_to_ine {np.max(transition_dict[n, neurons_to_increase])}")
                    if np.sum(to_put_to_one) > 0:
                        if np.sum(to_put_to_one) < len(neurons_to_increase):
                            print('in if len(to_put_to_one) < len(neurons_to_increase)')
                            sum_above_to_increase = np.sum(1 - transition_dict[n, neurons_to_increase[to_put_to_one]])
                            transition_dict[n, neurons_to_increase[to_put_to_one]] = 1
                            total_value_added = (sum_above_to_increase * np.sum(to_put_to_one)) + \
                                                (added_prob_value_modif * (
                                                        len(neurons_to_increase) - np.sum(to_put_to_one)))
                        else:
                            sum_above_to_increase = np.sum(1 - transition_dict[n, neurons_to_increase[to_put_to_one]])
                            transition_dict[n, neurons_to_increase[to_put_to_one]] = 1
                            total_value_added = sum_above_to_increase
                    else:
                        total_value_added = (added_prob_value_modif * len(neurons_to_increase))
                    transition_dict[n, neurons_to_increase[to_increases]] += added_prob_value_modif

                    to_subtract = total_value_added / len(neurons_to_decrease)
                    # section to take into account neuron which probability is inferior to to_substract
                    # to make sure it doesn't end with negative value
                    to_put_to_zeros = np.less(transition_dict[n, neurons_to_decrease], to_subtract)
                    news_neurons_to_decrease = np.greater_equal(transition_dict[n, neurons_to_decrease], to_subtract)
                    already_updated = False
                    while np.sum(to_put_to_zeros) > 0:
                        if np.sum(to_put_to_zeros) < len(neurons_to_decrease):
                            # print('in if len(to_put_to_zeros) < len(neurons_to_decrease)')
                            sum_under_to_substract = np.sum(transition_dict[n, neurons_to_decrease[to_put_to_zeros]])
                            transition_dict[n, neurons_to_decrease[to_put_to_zeros]] = 0
                            # print(f'to_subtract before: {to_subtract}')
                            to_subtract = ((to_subtract * len(neurons_to_decrease)) - sum_under_to_substract) / \
                                          (len(neurons_to_decrease) - np.sum(to_put_to_zeros))
                            neurons_to_decrease = neurons_to_decrease[news_neurons_to_decrease]
                            to_put_to_zeros = np.less(transition_dict[n, neurons_to_decrease], to_subtract)
                            news_neurons_to_decrease = np.greater_equal(transition_dict[n, neurons_to_decrease],
                                                                        to_subtract)
                            # print(f'to_subtract after: {to_subtract}')
                        else:
                            # print('np.sum(to_put_to_zeros) >= len(neurons_to_decrease):')
                            transition_dict[n, neurons_to_decrease] = 0
                            already_updated = True
                            break
                    if not already_updated:
                        transition_dict[n, neurons_to_decrease[news_neurons_to_decrease]] -= to_subtract

            # for each neuron activating after n, we put the probably for n to be in the next position after p in
            # a sequence to O or diminishing its value (except if p if activate at the same time as n)
            # if False means than neuron that fires in the same time, don't get their transition prob diminish by
            # that fact
            if param.no_reverse_seq:
                # same_time_same_prob = True
                # if same_time_same_prob:
                #     pos = pos_after_0
                for p in pos:
                    # to_dispatch = transition_dict[p, n]
                    # used to put to 0 the neuron n
                    to_dispatch = spikes_weight[p]
                    if transition_dict[p, n] - to_dispatch < 0:
                        to_dispatch = transition_dict[p, n]
                        transition_dict[p, n] = 0
                    else:
                        transition_dict[p, n] -= to_dispatch
                    # transition_dict[p, n] = 0
                    to_dispatch = to_dispatch / (nb_neurons - 2)
                    mask = np.ones(nb_neurons, dtype=bool)
                    mask[[n, p]] = False
                    # index_to_dispatch = np.where((np.where(np.arange(nb_neurons) != n)[0]) != p)[0]
                    # print(f'index_to_dispatch {index_to_dispatch}')
                    # TODO: take into consideration neuron in transition_dict which value is superior at
                    # TODO: (1 - to_dispatch)
                    transition_dict[p, mask] += to_dispatch

            # back to one
            spike_nums[n, original_t] = 1
    print_transit_dict = True
    if print_transit_dict:
        for n, neuron in enumerate(spike_nums):
            print(f'transition dict, n {n}, sum: {np.sum(transition_dict[n, :])}')
            print(f'transition dict, n {n}, max: {np.max(transition_dict[n, :])}')
            print(f'transition dict, n {n}, nb max: '
                  f'{np.where(transition_dict[n, :] == np.max(transition_dict[n, :]))[0]}')
    # sns.set()
    print(f'mean transition: {np.mean(transition_dict)}')
    print(f'std transition: {np.std(transition_dict)}')
    print(f'min transition: {np.min(transition_dict)}')
    print(f'max transition: {np.max(transition_dict)}')
    if show_heatmap:
        ax = sns.heatmap(transition_dict)  # , vmin=0, vmax=1)
        ax.invert_yaxis()
        plt.show()
    return transition_dict


def save_dict(list_seq_dict, parameters, comment, data_used):
    path = "/Users/pappyhammer/Documents/academique/these inmed/michel data/results/"


def print_save(text, file, to_write, no_print=False):
    if not no_print:
        print(text)
    if to_write:
        file.write(text + '\n')


def show_plot_raster(spike_nums, blocs_dict=None, clusters_edges=None, title=None, clusters_to_color=None,
                     cells_seq_to_color=None):
    colors = ["orange", "red", "blue", "green", "pink", "brown"]
    ax = plt.gca()
    # plot
    for y, neuron in enumerate(spike_nums):
        color_neuron = "black"
        if clusters_to_color is not None:
            for c_i, c in enumerate(clusters_to_color):
                if y in c:
                    color_neuron = colors[c_i % len(colors)]
                    break
        plt.vlines(np.where(neuron)[0], y - .5, y + .5, color=color_neuron)
    # coloring sequences
    if cells_seq_to_color is not None:
        color_index = 0
        for k, v in cells_seq_to_color.items():
            color_neuron = colors[color_index % len(colors)]
            for couple in v:
                if not spike_nums[couple[0], couple[1]]:
                    plt.vlines(couple[1], couple[0] - .5, couple[0] + .5, color="red")
                    # print("not real spike")
                else:
                    plt.vlines(couple[1], couple[0] - .5, couple[0] + .5, color=color_neuron)
            color_index += 1
    # if blocs_dict is not None:
    #     for k in blocs_dict.keys():
    #         # print(f'key: {k}')
    #         plt.vlines(k[0] - 1, 0, 10, color='red')
    #         plt.vlines(k[1] + 1, 0, 10, color='blue')
    if clusters_edges is not None:
        for edge in clusters_edges[:-1]:
            plt.hlines(edge, -1000, 0, color='red')

    if False:
        # drawing sequences
        for seq_dict in list_seq_dict:
            for neurons_seq, times_seq_set in seq_dict.items():
                # version dict
                if False:
                    for times in times_seq_set.keys():
                        for n_i, neuron in enumerate(neurons_seq):
                            plt.vlines(times[n_i], neuron - .5, neuron + .5,
                                       color=colors[y % (len(colors) - 1)], linewidth=3)

                # version seq
                try:
                    while True:
                        times = times_seq_set.pop()
                        for n_i, neuron in enumerate(neurons_seq):
                            plt.vlines(times[n_i], neuron - .5, neuron + .5,
                                       color=colors[y % (len(colors) - 1)], linewidth=3)
                except KeyError:
                    pass

    # set the locations and labels of the x_ticks
    # plt.xticks(np.arange(0, len(spike_nums[0, :]) + 1, 2000), np.arange(0, (len(spike_nums[0, :]) // 2) + 1, 1000))
    # minorLocator = MultipleLocator(100)
    # # for the minor ticks, use no labels; default NullFormatter
    # ax.xaxis.set_minor_locator(minorLocator)
    # plt.xlim(-0.5, len(spike_nums[0, :]) + .5)
    plt.ylim(-1, len(spike_nums) + 1)
    if title is None:
        plt.title('Spike raster plot')
    else:
        plt.title(title)
    # Give x axis label for the spike raster plot
    plt.xlabel('Frames')
    # Give y axis label for the spike raster plot
    plt.ylabel('Neurons')
    # Display the spike raster plot
    plt.show()


def finding_blocs(spike_nums, mouse):
    # key is a tuple of int representing times as the beg and end of the bloc
    # the value is a np.array of the neurons in the bloc
    blocs_dict = dict()
    last_bloc_end = -1
    # for p7, bloc_size 18, perc_thr: 0.19
    # for p12, bloc_size 16 or 10, perc_thr: 0.2 or 0.18
    # for P11, bloc_size 20, perc_thr: 0.15
    if mouse == "arnaud":
        bloc_size = 10
        sliding_window = 5
        perc_threshold = 0.2
    elif mouse == "p7":
        bloc_size = 18
        sliding_window = 5
        perc_threshold = 0.19
    elif mouse == "p11":
        bloc_size = 20
        sliding_window = 5
        perc_threshold = 0.15
    elif mouse == "pp":
        bloc_size = 10
        sliding_window = 5
        perc_threshold = 0.7
    else:
        bloc_size = 15
        sliding_window = 5
        perc_threshold = 0.2
    nb_neurons = len(spike_nums)
    max_time = len(spike_nums[0])
    nb_of_slides = max_time // sliding_window
    for i in np.arange(nb_of_slides):
        beg = i * sliding_window
        bloc_end = beg + bloc_size
        if bloc_end < max_time:
            if beg < last_bloc_end:
                continue
            if np.sum(spike_nums[:, beg:bloc_end]) > (nb_neurons * perc_threshold):
                blocs_dict[tuple([beg, bloc_end])] = np.where(spike_nums[:, beg:bloc_end])[0]
                last_bloc_end = bloc_end

    return blocs_dict


def cleaning_spike_nums(spike_nums, threshold_duration, every_5000=True):
    # used to clean spike_nums of the edge effect that happen every 5000 times
    if every_5000:
        nb_paste = len(spike_nums[0, :]) // 5000
        for n in np.arange(nb_paste):
            spike_nums[:, (n * 5000):(n * 5000) + 3] = 0

    for y, neuron in enumerate(spike_nums):
        positive_indexes = np.where(neuron)[0]
        if len(positive_indexes) < 2:
            continue
        mask = np.ediff1d(positive_indexes, to_begin=threshold_duration + 1) <= threshold_duration
        spike_nums[y, positive_indexes[mask]] = 0


def find_cluster_labels_for_neurons(cells_in_peak, cluster_labels):
    cluster_labels_for_neurons = np.zeros(np.shape(cells_in_peak)[0], dtype="int8")
    # sorting neurons spikes, keeping them only in one cluster, the one with the most spikes from this neuron
    # if spikes < 2 in any clusters, then removing spikes
    # going neuron by neuron,
    # removing_multiple_spikes_among_cluster = False

    for n, events in enumerate(cells_in_peak):
        pos_events = np.where(events)[0]
        max_clusters = np.zeros(np.max(cluster_labels) + 1, dtype="int8")
        for p in pos_events:
            max_clusters[cluster_labels[p]] += 1
        if np.max(max_clusters) < 2:
            # if removing_multiple_spikes_among_cluster:
            #     cells_in_peak[n, :] = np.zeros(len(cells_in_peak[n, :]))
            cluster_labels_for_neurons[n] = -1
        else:
            # selecting the cluster with the most spikes from neuron n
            max_cluster = np.argmax(max_clusters)
            cluster_labels_for_neurons[n] = max_cluster
            # clearing spikes from other cluster
            # if removing_multiple_spikes_among_cluster:
            #     cells_in_peak[n, np.not_equal(cluster_labels, max_cluster)] = 0
    return cluster_labels_for_neurons


def show_co_var_first_matrix(cells_in_peak, m_sces, n_clusters, kmeans, cluster_labels_for_neurons):
    # cellsinpeak: np.shape(value): (180, 285)
    plt.subplot(121)
    # list of size nb_neurons, each neuron having a value from 0 to k clusters
    cluster_labels = kmeans.labels_
    # contains the neurons from the SCE, but ordered by cluster
    print(f'// np.shape(m_sces) {np.shape(m_sces)}')
    ordered_m_sces = np.zeros((np.shape(m_sces)[0], np.shape(m_sces)[1]))
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_m_sces[start:start + nb_k, :] = m_sces[e, :]
        ordered_m_sces[:, start:start + nb_k] = m_sces[:, e]
        start += nb_k

    co_var = np.corrcoef(ordered_m_sces)  # cov
    # sns.set()
    ax = sns.heatmap(co_var, cmap="jet")  # , vmin=0, vmax=1) YlGnBu
    # ax.invert_yaxis()

    # plt.show()
    original_cells_in_peak = cells_in_peak

    cells_in_peak = np.copy(original_cells_in_peak)
    plt.subplot(1, 2, 2)

    ordered_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]))
    ordered_n_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]))
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        start += nb_k

    start = 0
    for k in np.arange(-1, np.max(cluster_labels_for_neurons) + 1):
        e = np.equal(cluster_labels_for_neurons, k)
        nb_k = np.sum(e)
        # print(f'nb_k {nb_k}, k: {k}')
        ordered_n_cells_in_peak[start:start + nb_k, :] = ordered_cells_in_peak[e, :]
        start += nb_k

    ax = sns.heatmap(ordered_n_cells_in_peak, cbar=False)
    # ax.invert_yaxis()

    plt.show()


# normalized co-variance
def covnorm(m_sces):
    nb_events = np.shape(m_sces)[1]
    co_var_matrix = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            co_var_matrix[i, j] = np.correlate(m_sces[:, i], m_sces[:, j]) / np.std(m_sces[:, i]) \
                                  / np.std(m_sces[:, j]) / nb_events
    return co_var_matrix


def co_var_first_and_clusters(cellsinpeak, range_n_clusters, shuffling=False, nth_best_clusters=-1,
                              plot_matrix=False):
    """

    :param nth_best_clusters: how many clusters to return, if -1 return them all
    :return:
    """
    # nb_sces = len(list_neurons)
    # if nb_sces == 0:
    #     return
    #
    # # matrix used to do the co-variance matrix and the clustering
    # m_sces = np.zeros((nb_neurons, nb_sces))
    # for i, s in enumerate(list_neurons):
    #     m_sces[s, i] = 1
    # original_m_sces = m_sces
    m_sces = cellsinpeak
    # m_sces = np.transpose(cellsinpeak)
    # m_sces = np.cov(m_sces)
    # print(f'np.shape(m_sces) {np.shape(m_sces)}')
    m_sces = covnorm(m_sces)
    # m_sces = np.corrcoef(m_sces)

    # ax = sns.heatmap(m_sces, cmap="jet")  # , vmin=0, vmax=1) YlGnBu
    # ax.invert_yaxis()
    # plt.show()

    original_m_sces = m_sces
    testing = True

    # key is the nth clusters as int, value is a list of list of SCE
    # (each list representing a cluster, so we have as many list as the number of cluster wanted)
    dict_best_clusters = dict()
    # the key is the nth cluster (int) and the value is a list of cluster number for each cell
    cluster_labels_for_neurons = dict()
    for i in range_n_clusters:
        dict_best_clusters[i] = []

    if testing:
        # if shuffling:
        #     range_n_clusters = [2]
        # else:
        #     range_n_clusters = [n_cluster_var] #np.arange(2, 3)
        # nb of time to apply one given number of cluster
        n_trials = 100
        if shuffling:
            n_shuffle = 100
        else:
            n_shuffle = 1
        silhouettes_shuffling = np.zeros(n_shuffle * range_n_clusters[0])  # *n_trials*range_n_clusters[0])
        for shuffle_index in np.arange(n_shuffle):
            if shuffling:
                m_sces = np.copy(original_m_sces)
                for j in np.arange(len(m_sces[0])):
                    m_sces[:, j] = np.random.permutation(m_sces[:, j])
                for i, s in enumerate(m_sces):
                    # print(f"pos before permutation {np.where(s)[0]}")
                    m_sces[i] = np.random.permutation(s)
            for n_clusters in range_n_clusters:
                best_kmeans = None
                silhouette_avgs = np.zeros(n_trials)
                best_silhouettes_clusters_avg = None
                max_local_clusters_silhouette = 0
                best_median_silhouettes = 0
                silhouettes_clusters_avg = []
                for trial in np.arange(n_trials):
                    # co_var = np.cov(m_sces)
                    kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
                    # print(f'kmeans.labels_ {kmeans.labels_}')
                    cluster_labels = kmeans.labels_
                    silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
                    silhouette_avgs[trial] = silhouette_avg
                    # print(f"Avg silhouette: {silhouette_avg}")
                    sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
                    local_clusters_silhouette = np.zeros(n_clusters)
                    for i in range(n_clusters):
                        # Aggregate the silhouette scores for samples belonging to
                        # cluster i, and sort them
                        ith_cluster_silhouette_values = \
                            sample_silhouette_values[cluster_labels == i]
                        # print(f'ith_cluster_silhouette_values {ith_cluster_silhouette_values}')
                        # print(f'np.mean(ith_cluster_silhouette_values) {np.mean(ith_cluster_silhouette_values)}')
                        avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                        silhouettes_clusters_avg.append(avg_ith_cluster_silhouette_values)
                        # print(ith_cluster_silhouette_values)
                        local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values
                        # ith_cluster_silhouette_values.sort()
                    med = np.median(local_clusters_silhouette)
                    if med > best_median_silhouettes:
                        best_median_silhouettes = med
                        best_silhouettes_clusters_avg = local_clusters_silhouette

                    max_local = med  # np.max(local_clusters_silhouette)  # silhouette_avg
                    # TO display, we keep the group with the cluster with the max silhouette
                    if med > max_local_clusters_silhouette:
                        max_local_clusters_silhouette = max_local
                        best_kmeans = kmeans
                        nth_best_list = []
                        count_clusters = nth_best_clusters
                        if count_clusters == -1:
                            count_clusters = n_clusters
                        for b in np.arange(count_clusters):
                            # print(f'local_clusters_silhouette {local_clusters_silhouette}')
                            arg = np.argmax(local_clusters_silhouette)
                            # print(f'local_clusters_silhouette[arg] {local_clusters_silhouette[arg]}')
                            # print(f'[cluster_labels == arg] {[cluster_labels == arg]}, '
                            #       f'cluster_labels {cluster_labels}')
                            # TODO: put neurons list instead of SCEs
                            nth_best_list.append(np.arange(len(m_sces))[cluster_labels == arg])
                            local_clusters_silhouette[arg] = -1
                        dict_best_clusters[n_clusters] = nth_best_list
                    # silhouettes_clusters_avg.extend(sample_silhouette_values)
                    used = False
                    if used:
                        print(f"Silhouettes: {sample_silhouette_values}")
                if shuffling:
                    print(f'end shuffling {shuffle_index}')
                    index = shuffle_index * n_clusters
                    silhouettes_shuffling[index:index + n_clusters] = best_silhouettes_clusters_avg
                else:
                    print(f'n_clusters {n_clusters}, avg-avg: {np.round(np.mean(silhouette_avgs), 4)}, '
                          f'median-avg {np.round(np.median(silhouette_avgs), 4)}, '
                          f'median-all {np.round(np.median(silhouettes_clusters_avg), 4)}')
                    print(f'n_clusters {n_clusters}, silhouettes_clusters_avg {silhouettes_clusters_avg}')
                    cluster_labels_for_neurons[n_clusters] = find_cluster_labels_for_neurons(cells_in_peak=cellsinpeak,
                                                                                             cluster_labels=best_kmeans.labels_)
                    if plot_matrix:
                        show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_sces,
                                                 n_clusters=n_clusters, kmeans=best_kmeans,
                                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_clusters])
        if shuffling:
            # silhouettes_shuffling contains the mean silhouettes values of all clusters produced (100*k)
            p_95 = np.percentile(silhouettes_shuffling, 95)
            print(f'95th p= {p_95}')
    # m_sces = original_m_sces
    return dict_best_clusters, cluster_labels_for_neurons


def lets_cluster_seq(list_seq_dict, nb_neurons, param):
    list_seq_to_clusters = []
    # list of dict, each dict is produce for one neuron
    for set_seq_dict_result in list_seq_dict:
        # set_seq_dict_result: each key represent a common seq (neurons tuple)
        # and each values represent a list of list of times
        for neurons_seq, times_seq_set in set_seq_dict_result.items():
            if (len(times_seq_set) > 2) and (len(neurons_seq) >= param.min_len_seq):
                list_seq_to_clusters.append(np.array(neurons_seq))

    print(f'Stat co_var_and_clusters len(list_seq_to_clusters): {len(list_seq_to_clusters)}')
    range_n_clusters = [2, 3, 4, 5, 15, 20]
    co_var_and_clusters_by_neurons(list_neurons=list_seq_to_clusters, range_n_clusters=range_n_clusters,
                                   nb_neurons=nb_neurons, plot_matrix=True)


def order_spike_nums_by_seq(spike_nums, param, produce_stat):
    """

    :param spike_nums:
    :param param:
    :param produce_stat:
    :return:
    """
    # ordered_spike_nums = spike_nums.copy()
    nb_neurons = len(spike_nums)
    if nb_neurons == 0:
        return [], [], []
    nb_repet = 1
    if param.random_mode:
        nb_repet = produce_stat
    list_dict_stat = []
    max_len_seq = 0
    for i in np.arange(nb_repet):
        if nb_repet == 1:
            print_save(f'Trial {i+1}', param.file, no_print=False, to_write=True)
        # the key is the length of a seq
        dict_stat = dict()
        # dictionnary to get coordonates of cells from a same sequence, will be used to color the raster.
        #  Key is a number identifying a sequence, then value is a list of list of 2 elements, the first index of the tuple
        # representing the time, and the second represents the cell index
        cells_seq_to_color = dict()
        # list_seq_dict
        list_seq_dict, dict_by_len_seq, max_seq_dict, \
        max_rep_non_prob, max_len_non_prob = find_sequences(spike_nums=spike_nums,
                                                            no_print=True, param=param)

        # TODO: Faire un loss_score sur chaque seq max pour chaque neuron et garder le meilleur !
        # temporary code to test if taking the longest sequence is good option
        max_seq = None
        best_loss_score = 1
        ordered_spike_nums = None
        for k, seq in max_seq_dict.items():
            seq = np.array(seq)
            new_order = np.zeros(nb_neurons, dtype="int8")
            new_order[:len(seq)] = seq
            not_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
                                               seq)
            if len(not_ordered_neurons) > 0:
                new_order[len(seq):] = not_ordered_neurons
            tmp_spike_nums = spike_nums[new_order, :]
            loss_score = loss_function(tmp_spike_nums[::-1, :], param)
            print(f'loss_score neuron {k}, len {len(seq)}: {np.round(loss_score, 4)}')
            if loss_score < best_loss_score:
                best_loss_score = loss_score
                ordered_spike_nums = tmp_spike_nums
                max_seq = np.array(seq)

            # if max_seq is None:
            #     max_seq = np.array(seq)
            # else:
            #     if len(max_seq) < len(seq):
            #         max_seq = np.array(seq)
        print(f'best loss_score neuron {np.round(best_loss_score, 4)}')
        # new_order = np.zeros(nb_neurons, dtype="int8")
        # new_order[:len(max_seq)] = max_seq
        # not_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
        #                                    max_seq)
        # if len(not_ordered_neurons) > 0:
        #     new_order[len(max_seq):] = not_ordered_neurons
        # ordered_spike_nums = spike_nums[new_order, :]
        return ordered_spike_nums[::-1, :], list_seq_dict, None, max_seq[::-1], None

        # ploting raster organized with clusters
        lets_do_it = True
        if lets_do_it:
            neurons_ordered_so_far = []
            seq_edges = []
            nb_neurons_so_far = 0
            # list of np.array of neurons, at the end will serve to organize the new spike_nums
            ordered_list_of_neurons = []
            ordered_spike_nums = np.zeros((len(spike_nums), len(spike_nums[0])))
            not_clustered_neurons = np.arange(nb_neurons)
            # this section takes the max seq defined only by probability
            # without verification on the raster plot
            # max_seq_order = False
            # if max_seq_order:
            #     for first_tour in [True, False]:
            #         seq_max_to_sort = list(max_seq_dict.values())
            #         seq_max_to_sort.sort(key=len)
            #         seq_max_to_sort = seq_max_to_sort[::-1]
            #         for seq_max in seq_max_to_sort:
            #             seq_max = np.array(seq_max)
            #             print(f'neurons_seq {seq_max}')
            #             intersect = np.intersect1d(seq_max, np.array(neurons_ordered_so_far))
            #             if len(intersect) == 0:
            #                 print(f'Ordering raster, intersect empty, one seq added: {seq_max}')
            #                 not_clustered_neurons = np.setdiff1d(not_clustered_neurons, np.array(seq_max))
            #                 # print(f'not_clustered_neurons {not_clustered_neurons}')
            #                 low_index = nb_neurons - 1 - nb_neurons_so_far - len(seq_max)
            #                 high_index = nb_neurons - 1 - nb_neurons_so_far
            #                 ordered_spike_nums[low_index:high_index, :] = \
            #                     spike_nums[seq_max, ::-1]
            #                 seq_edges.append(nb_neurons - 1 - nb_neurons_so_far - len(seq_max))
            #                 neurons_ordered_so_far.extend(seq_max)
            #                 nb_neurons_so_far += len(seq_max)
            #             elif not first_tour:
            #                 new_seq_max = seq_max[np.invert(np.isin(np.array(seq_max),
            #                                                         neurons_ordered_so_far))]
            #                 if len(new_seq_max) > 0:
            #                     print(f'intersect {intersect}')
            #                     print(f'Ordering_ raster, one seq added: {new_seq_max}')
            #                     not_clustered_neurons = np.setdiff1d(not_clustered_neurons,
            #                                                          np.array(new_seq_max))
            #                     low_index = nb_neurons - 1 - nb_neurons_so_far - len(new_seq_max)
            #                     high_index = nb_neurons - 1 - nb_neurons_so_far
            #                     ordered_spike_nums[low_index:high_index, :] = \
            #                         spike_nums[new_seq_max, ::-1]
            #                     seq_edges.append(nb_neurons - 1 - nb_neurons_so_far - len(new_seq_max))
            #                     neurons_ordered_so_far.extend(new_seq_max)
            #                     nb_neurons_so_far += len(new_seq_max)
            #
            # else:
            # list of list
            list_seq_to_insert = []
            print(f'max_rep_non_prob {max_rep_non_prob}')
            print(f'max_len_non_prob {max_len_non_prob}')

            # list become a dict
            # dict_list_seq_dict = dict()
            # for l_i, list_seq in enumerate(list_seq_dict):
            #     dict_list_seq_dict[l_i] = list_seq
            for tour in [1, 3, 2, 4]:
                # for index_set_seq, set_seq_dict_result in dict_list_seq_dict.items():
                # set_seq_dict_result: each key represent a common seq (neurons tuple)
                # and each values represent a list of list of times

                # during tour 1, we select the best sequence of length max_len_non_prob
                # to be a good seq, it needs to repeat as much as possible, then for the same number of repetition
                # the sum of the diff must not be too low (otherwise it would select SCE), then
                # the lower the std of all the diff of the time serie, the more the slope will be the same
                # among sequences
                if tour == 1:
                    best_seq_choice = None
                    times_seq_set_of_best_seq = None
                    lowest_diff_std = 10000
                    mean_value_selected = -1
                    for min_len_seq_first_tour in param.min_len_seq_first_tour[::-1]:
                        if min_len_seq_first_tour not in dict_by_len_seq:
                            continue
                        dict_seq_times = dict_by_len_seq[min_len_seq_first_tour]
                        for neurons_seq, times_seq_set in dict_seq_times.items():
                            if len(times_seq_set) > 5:  # == max_rep_non_prob:
                                times_diff_sum_array = np.zeros(len(times_seq_set))
                                for t_i, times_tuple in enumerate(times_seq_set):
                                    # do the sum of the diff for a given time seq
                                    times_diff_sum_array[t_i] = np.sum(np.diff(np.array(times_tuple)))
                                std_value = np.std(times_diff_sum_array)
                                # a low mean value would mean a sequence almost vertical
                                mean_value = np.mean(times_diff_sum_array)

                                if best_seq_choice is None:
                                    best_seq_choice = neurons_seq
                                    times_seq_set_of_best_seq = times_seq_set
                                    lowest_diff_std = std_value
                                    mean_value_selected = mean_value
                                    print(f'1: best_seq_choice: len {len(best_seq_choice)} {best_seq_choice}, '
                                          f'lowest_diff_std: {lowest_diff_std}, '
                                          f'mean_value_selected: {mean_value_selected}, '
                                          f'len(times_seq_set): {len(times_seq_set)}')
                                else:
                                    # choosing the one with the more repetition
                                    if len(times_seq_set) > len(times_seq_set_of_best_seq):
                                        lowest_diff_std = std_value
                                        best_seq_choice = neurons_seq
                                        times_seq_set_of_best_seq = times_seq_set
                                        mean_value_selected = mean_value
                                        print(f'3: best_seq_choice: {best_seq_choice}, '
                                              f'lowest_diff_std: {lowest_diff_std}, '
                                              f'mean_value_selected: {mean_value_selected}, '
                                              f'len(times_seq_set): {len(times_seq_set)}')
                                    elif (len(times_seq_set) == len(times_seq_set_of_best_seq)) \
                                            and (std_value < lowest_diff_std):
                                        lowest_diff_std = std_value
                                        best_seq_choice = neurons_seq
                                        times_seq_set_of_best_seq = times_seq_set
                                        mean_value_selected = mean_value
                                        print(f'4: best_seq_choice: {best_seq_choice}, '
                                              f'lowest_diff_std: {lowest_diff_std}, '
                                              f'mean_value_selected: {mean_value_selected}')
                                    elif (mean_value_selected < 60) and (mean_value > mean_value_selected):
                                        lowest_diff_std = std_value
                                        best_seq_choice = neurons_seq
                                        times_seq_set_of_best_seq = times_seq_set
                                        mean_value_selected = mean_value
                                        print(
                                            f'2: best_seq_choice: {best_seq_choice}, lowest_diff_std: {lowest_diff_std}, '
                                            f'mean_value_selected: {mean_value_selected}')

                    if best_seq_choice is not None:
                        print(f'Ordering raster, one seq added, tour 1: {best_seq_choice}')
                        ordered_list_of_neurons.append(np.array(best_seq_choice))
                        neurons_ordered_so_far.extend(list(best_seq_choice))
                        len_c = len(cells_seq_to_color)
                        cells_seq_to_color[len_c] = []
                        # times_seq_set_of_best_seq is a set of tuples, each tuples representing the time
                        # of a neuron from the corresponding seq
                        for index_n, n in enumerate(best_seq_choice):
                            for t_s in times_seq_set_of_best_seq:
                                cells_seq_to_color[len_c].append([n, t_s[index_n]])
                        print(f'cells_seq_to_color[len_c] {cells_seq_to_color[len_c]}')
                    else:
                        print('best_seq_choice is None, ANORMAL')
                    continue

                for len_neurons_seq in sorted(dict_by_len_seq.keys()):
                    dict_seq_times = dict_by_len_seq[len_neurons_seq]
                    for neurons_seq, times_seq_set in dict_seq_times.items():
                        neurons_seq = np.array(neurons_seq)
                        # if tour == 1:
                        #     if (len(neurons_seq) == max_len_non_prob) and \
                        #             (len(times_seq_set) == max_rep_non_prob):
                        #         intersect = np.intersect1d(neurons_seq, np.array(neurons_ordered_so_far))
                        #         if len(intersect) == 0:
                        #             print(f'Ordering raster, one seq added, tour {tour}: {neurons_seq}')
                        #             ordered_list_of_neurons.append(np.array(neurons_seq))
                        #             neurons_ordered_so_far.extend(neurons_seq)
                        #             # del dict_list_seq_dict[index_set_seq]
                        #             break
                        #         elif len(intersect) <= 2:
                        #             # print(f"neurons_seq {neurons_seq}, "
                        #             #       f"neurons_ordered_so_far {neurons_ordered_so_far}")
                        #             new_n_seq = neurons_seq[np.isin(np.array(neurons_seq),
                        #                                             np.array(neurons_ordered_so_far),
                        #                                             invert=True)]
                        #             if len(new_n_seq) > 0:
                        #                 print(f'Ordering raster first_tour, one seq added: {new_n_seq}')
                        #                 ordered_list_of_neurons.append(np.array(new_n_seq))
                        #                 neurons_ordered_so_far.extend(new_n_seq)
                        #         else:
                        #             break
                        # 2nd tour: select sequence of a certain length for which none of the neurons has been added
                        # to the ordered seq
                        if tour == 2:
                            if len_neurons_seq > np.min(param.min_len_seq_first_tour):
                                intersect = np.intersect1d(neurons_seq, np.array(neurons_ordered_so_far))
                                if len(intersect) == 0:
                                    print(f'Ordering raster tour 2, one seq added, tour {tour}: {neurons_seq}')
                                    ordered_list_of_neurons.append(np.array(neurons_seq))
                                    neurons_ordered_so_far.extend(neurons_seq)
                                    # del dict_list_seq_dict[index_set_seq]
                        elif tour == 3:
                            to_concatenate = False
                            # look if the prefix/suffix are the same of a sequence already added, if so,
                            # we remove the first 2 or last
                            # neurons of neurons_seq and add the other one before or after the already
                            # selected seq
                            # on how many neurons to look for interesection at one tail
                            len_inter = 3
                            # how many neurons should interesect within len_inter to validate the
                            # concatenation
                            min_inter = 2
                            for l_i, l_neurons in enumerate(ordered_list_of_neurons):
                                if len(l_neurons) > 3:
                                    prefix = neurons_seq[:len_inter]
                                    suffix = l_neurons[len(l_neurons) - len_inter:]
                                    intersect = np.intersect1d(prefix, suffix)
                                    if len(intersect) >= min_inter:
                                        new_n_seq = neurons_seq[min_inter:]
                                        new_n_seq = new_n_seq[np.isin(np.array(new_n_seq),
                                                                      np.array(neurons_ordered_so_far),
                                                                      invert=True)]
                                        if len(new_n_seq) > 3:
                                            ordered_list_of_neurons[l_i] = \
                                                np.concatenate((ordered_list_of_neurons[l_i], new_n_seq))
                                            print(f'Ordering raster 3rd tour, one seq added after: '
                                                  f'{new_n_seq}')
                                            print(f'Original seq: {neurons_seq}')
                                            print(f'order_list: {ordered_list_of_neurons[l_i]}')
                                            neurons_ordered_so_far.extend(new_n_seq)
                                            break

                                    prefix = l_neurons[:len_inter]
                                    suffix = neurons_seq[len(neurons_seq) - len_inter:]
                                    intersect = np.intersect1d(suffix, prefix)
                                    if len(intersect) >= min_inter:
                                        new_n_seq = neurons_seq[:-min_inter]
                                        new_n_seq = new_n_seq[np.isin(np.array(new_n_seq),
                                                                      np.array(neurons_ordered_so_far),
                                                                      invert=True)]
                                        if len(new_n_seq) > 3:
                                            ordered_list_of_neurons[l_i] = \
                                                np.concatenate((new_n_seq, ordered_list_of_neurons[l_i]))
                                            print(f'Ordering raster tour 3, one seq added before: '
                                                  f'{new_n_seq}')
                                            print(f'Original seq: {neurons_seq}')
                                            neurons_ordered_so_far.extend(new_n_seq)
                                            break

                        else:
                            if (len(times_seq_set) > 4) and (len(neurons_seq) >= param.min_len_seq):
                                intersect = np.intersect1d(neurons_seq, np.array(neurons_ordered_so_far))
                                # print(f'intersect {intersect}')
                                if len(intersect) == 0:
                                    print(f'Ordering raster, one seq added, tour {tour}: {neurons_seq}')
                                    ordered_list_of_neurons.append(np.array(neurons_seq))
                                    neurons_ordered_so_far.extend(neurons_seq)
                                    break

                            if len(neurons_ordered_so_far) > 0:
                                new_n_seq = neurons_seq[np.invert(
                                    np.isin(np.array(neurons_seq),
                                            np.array(neurons_ordered_so_far)))]
                            else:
                                new_n_seq = np.array(neurons_seq)
                            # at least sequences of 4 neurons, otherwise go at the end directly
                            if len(new_n_seq) > 4:
                                print(f'Ordering raster__, one seq added: {new_n_seq}')
                                print(f'Original seq: {neurons_seq}')
                                ordered_list_of_neurons.append(np.array(new_n_seq))
                                neurons_ordered_so_far.extend(new_n_seq)

            total_len = 0
            for l in ordered_list_of_neurons:
                total_len += len(l)
            # print(f'total len ordered_list_of_neurons: {total_len}')
            # print(f'ordered_list_of_neurons {ordered_list_of_neurons}')
            # print(f'neurons_ordered_so_far {neurons_ordered_so_far}')
            # print(f'nb_neurons: {nb_neurons}')
            not_clustered_neurons = np.arange(nb_neurons)
            not_clustered_neurons = np.setdiff1d(not_clustered_neurons,
                                                 np.array(neurons_ordered_so_far))
            # print(f'not_clustered_neurons {not_clustered_neurons}, len {len(not_clustered_neurons)}')
            if len(not_clustered_neurons) > 0:
                print(f"len(not_clustered_neurons) > 0: {len(not_clustered_neurons)}")
                ordered_list_of_neurons.append(np.array(not_clustered_neurons))

            # organizing neurons

            # store the new index in ordered_spike_nums for each neurons
            corresponding_neurons_index = np.arange(nb_neurons)
            high_index = nb_neurons
            for array_neurons in ordered_list_of_neurons:
                # print(f"org array_neurons: {array_neurons}, array_neurons[::-1]: {array_neurons[::-1]}")
                print(f"len(array_neurons) {len(array_neurons)}")
                low_index = high_index - len(array_neurons)
                print(f"low_index {low_index}, high_index {high_index}")
                ordered_spike_nums[low_index:high_index, :] = \
                    spike_nums[array_neurons[::-1], :]
                corresponding_neurons_index[array_neurons[::-1]] = np.arange(low_index, high_index)
                seq_edges.append(low_index)
                high_index = low_index

        # to make stat
        for set_seq_dict_result in list_seq_dict:
            # set_seq_dict_result: each key represent a common seq (neurons tuple)
            # and each values represent a list of list of times
            for neurons_seq, times_seq_set in set_seq_dict_result.items():
                if (len(times_seq_set) > 2) and (len(neurons_seq) >= param.min_len_seq):
                    len_seq = len(neurons_seq)
                    if len_seq > max_len_seq:
                        max_len_seq = len_seq
                    if len_seq in dict_stat:
                        stat_seq = dict_stat[len_seq]
                    else:
                        stat_seq = StatSeq(len_seq)
                        dict_stat[len_seq] = stat_seq
                    if len(stat_seq.nb_seq) == 0:
                        stat_seq.nb_seq.append(1)
                    else:
                        stat_seq.nb_seq[0] += 1
                    stat_seq.nb_rep.append(len(times_seq_set))

                    times_diff_dict = build_time_diff_dict(times_seq_set=times_seq_set,
                                                           time_diff_threshold=2)
                    for k, v in times_diff_dict.items():
                        if len(v) > 1:
                            stat_seq.nb_same_diff.append(len(v))

                    times_diff_sum_list = []
                    for times_tuple in times_seq_set:
                        # do the sum of the diff for a given time seq
                        times_diff_sum_list.append(np.sum(np.diff(np.array(times_tuple))))
                    stat_seq.std_sum_diff.append(np.std(np.array(times_diff_sum_list)))

        list_dict_stat.append(dict_stat)
        if nb_repet == 1:
            for l_s in np.arange(param.min_len_seq, max_len_seq + 1):
                if l_s in dict_stat:
                    stat_seq = dict_stat[l_s]
                    print_save(str(stat_seq), param.file, no_print=False, to_write=True)
                    print_save('', param.file, no_print=False, to_write=True)
            print_save('', param.file, no_print=False, to_write=True)
    if nb_repet > 1:
        print_save(f'Over all {nb_repet} trials', param.file, no_print=False, to_write=True)
        if max_len_seq > param.min_len_seq:
            # adding all trials stat
            for i in np.arange(param.min_len_seq, max_len_seq + 1):
                stat_seq = None
                for dict_stat in list_dict_stat:
                    if i in dict_stat:
                        if stat_seq is None:
                            stat_seq = dict_stat[i]
                        else:
                            stat_seq += dict_stat[i]
                if stat_seq is not None:
                    print_save(str(stat_seq), param.file, no_print=False, to_write=True)
                    print_save('', param.file, no_print=False, to_write=True)

    return ordered_spike_nums, list_seq_dict, seq_edges, corresponding_neurons_index, cells_seq_to_color


def sort_it_and_plot_it(spike_nums, param,
                        sliding_window_duration, activity_threshold, title_option="",
                        spike_train_format=False,
                        debug_mode=False):
    if spike_train_format:
        return
    seq_dict_tmp, best_seq, all_best_seq = pattern_discovery.seq_solver.markov_way.order_spike_nums_by_seq(spike_nums,
                                                                                                           param,
                                                                                                           debug_mode=debug_mode)
    # tmp test
    for cell, each_best_seq in enumerate(all_best_seq):
        spike_nums_ordered = np.copy(spike_nums[each_best_seq, :])

        new_labels = np.arange(len(spike_nums))
        new_labels = new_labels[best_seq]
        loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered[::-1, :],
                                                       time_inter_seq=param.time_inter_seq,
                                                       min_duration_intra_seq=param.min_duration_intra_seq,
                                                       spike_train_mode=False,
                                                       debug_mode=True
                                                       )
        print(f'Cell {cell} loss_score ordered: {np.round(loss_score, 4)}')
        # saving the ordered spike_nums
        # micro_wires_ordered = micro_wires[best_seq]
        # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
        #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

        plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                           title=f"cell {cell} raster plot ordered {title_option}",
                           spike_train_format=False,
                           file_name=f"cell_{cell}_spike_nums_ordered_{title_option}",
                           y_ticks_labels=new_labels,
                           y_ticks_labels_size=5,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           save_formats="png")
    return best_seq, seq_dict_tmp


    spike_nums_ordered = np.copy(spike_nums[best_seq, :])

    if debug_mode:
        print(f"best_seq {best_seq}")
    if seq_dict_tmp is not None:
        colors_for_seq_list = ["blue", "red", "orange", "green", "grey", "yellow", "pink"]
        if debug_mode:
            for key, value in seq_dict_tmp.items():
                print(f"seq: {key}, rep: {len(value)}")

        best_seq_mapping_index = dict()
        for i, cell in enumerate(best_seq):
            best_seq_mapping_index[cell] = i
        # we need to replace the index by the corresponding one in best_seq
        seq_dict = dict()
        for key, value in seq_dict_tmp.items():
            new_key = []
            for cell in key:
                new_key.append(best_seq_mapping_index[cell])
            seq_dict[tuple(new_key)] = value

        seq_colors = dict()
        if debug_mode:
            print(f"nb seq to colors: {len(seq_dict)}")
        for index, key in enumerate(seq_dict.keys()):
            seq_colors[key] = colors_for_seq_list[index % (len(colors_for_seq_list))]
            if debug_mode:
                print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    else:
        seq_dict = None
        seq_colors = None
    # ordered_spike_nums = ordered_spike_data
    # spike_struct.ordered_spike_data = \
    #     trains_module.from_spike_nums_to_spike_trains(spike_struct.ordered_spike_data)
    new_labels = np.arange(len(spike_nums))
    new_labels = new_labels[best_seq]
    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered,
                                                   time_inter_seq=param.time_inter_seq,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   spike_train_mode=False,
                                                   debug_mode=True
                                                   )
    print(f'total loss_score ordered: {np.round(loss_score, 4)}')
    # saving the ordered spike_nums
    # micro_wires_ordered = micro_wires[best_seq]
    # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
    #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

    plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                       title=f"raster plot ordered {title_option}",
                       spike_train_format=False,
                       file_name=f"spike_nums_ordered_{title_option}",
                       y_ticks_labels=new_labels,
                       y_ticks_labels_size=5,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       save_formats="png",
                       seq_times_to_color_dict=seq_dict,
                       seq_colors=seq_colors)

    return best_seq, seq_dict_tmp


def use_new_pattern_package(spike_nums, param, activity_threshold, sliding_window_duration, 
                            mouse_id, n_surrogate=2, debug_mode=False):
    # around 250 ms
    # param.time_inter_seq
    # param.min_duration_intra_seq
    # -(10 ** (6 - decrease_factor)) // 40
    # a sequence should be composed of at least one third of the neurons
    # param.min_len_seq = len(spike_nums_struct.spike_data) // 4
    # param.min_len_seq = 5
    # param.error_rate = param.min_len_seq // 4
    param.error_rate = 5
    param.max_branches = 10
    # 500 ms
    # sliding_window_duration = 10

    # print(f"param.min_len_seq {param.min_len_seq},  param.error_rate {param.error_rate}")
    #
    # print(f"spike_nums_struct.activity_threshold {spike_struct.activity_threshold}")
    #
    # print("plot_spikes_raster")

    # if len(spike_nums_struct.spike_nums[0, :]) > 10**8:
    #     spike_nums_struct.spike_nums = spike_nums_struct.spike_nums[:, :10**7]

    labels = np.arange(len(spike_nums))

    if True:
        plot_spikes_raster(spike_nums=spike_nums, param=param,
                           spike_train_format=False,
                           title=f"raster plot {mouse_id}",
                           file_name=f"raw_spike_nums_{mouse_id}",
                           y_ticks_labels=labels,
                           y_ticks_labels_size=4,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           # 500 ms window
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="png")
    # continue

    # 2128885
    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums,
                                                   time_inter_seq=param.time_inter_seq,
                                                   spike_train_mode=False,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   debug_mode=debug_mode)

    print(f'raw loss_score: {np.round(loss_score, 4)}')

    # spike_struct.spike_data = trains_module.from_spike_trains_to_spike_nums(spike_struct.spike_data)

    best_seq, seq_dict = sort_it_and_plot_it(spike_nums=spike_nums, param=param,
                                             sliding_window_duration=sliding_window_duration,
                                             activity_threshold=activity_threshold,
                                             title_option=f"{mouse_id}",
                                             spike_train_format=False,
                                             debug_mode=debug_mode)
    return
    nb_cells = len(spike_nums)

    print("#### REAL DATA ####")
    print(f"best_seq {best_seq}")
    real_data_result_for_stat = SortedDict()
    neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
    if seq_dict is not None:
        for key, value in seq_dict.items():
            print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
            if len(key) not in real_data_result_for_stat:
                real_data_result_for_stat[len(key)] = []
            real_data_result_for_stat[len(key)].append(len(value))
            for cell in key:
                if neurons_sorted_real_data[cell] == 0:
                    neurons_sorted_real_data[cell] = 1

    n_times = len(spike_nums[0, :])

    print("#### SURROGATE DATA ####")
    # n_surrogate = 2
    surrogate_data_result_for_stat = SortedDict()
    neurons_sorted_surrogate_data = np.zeros(nb_cells, dtype="uint16")
    for surrogate_number in np.arange(n_surrogate):
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        tmp_spike_nums = copy_spike_nums

        best_seq, seq_dict = sort_it_and_plot_it(spike_nums=tmp_spike_nums, param=param,
                                             sliding_window_duration=sliding_window_duration,
                                             activity_threshold=activity_threshold,
                                             title_option=f"surrogate {mouse_id}",
                                             spike_train_format=False,
                                             debug_mode=False)

        print(f"best_seq {best_seq}")

        mask = np.zeros(nb_cells, dtype="bool")
        if seq_dict is not None:
            for key, value in seq_dict.items():
                print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                if len(key) not in surrogate_data_result_for_stat:
                    surrogate_data_result_for_stat[len(key)] = []
                surrogate_data_result_for_stat[len(key)].append(len(value))
                for cell in key:
                    mask[cell] = True
            neurons_sorted_surrogate_data[mask] += 1
    # min_time, max_time = trains_module.get_range_train_list(spike_nums)
    # surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
    #                                               min_value=min_time, max_value=max_time)
    print("")
    print("")

    give_me_stat_on_sorting_seq_results(results_dict=real_data_result_for_stat,
                                        neurons_sorted=neurons_sorted_real_data,
                                        title="%%%% DATA SET STAT %%%%%", param=param,
                                        results_dict_surrogate=surrogate_data_result_for_stat,
                                        neurons_sorted_surrogate=neurons_sorted_surrogate_data)


def give_me_stat_on_sorting_seq_results(results_dict, neurons_sorted, title, param,
                                        results_dict_surrogate=None, neurons_sorted_surrogate=None):
    """
    Key will be the length of the sequence and value will be a list of int, representing the nb of rep
    of the different lists
    :param results_dict:
    :return:
    """
    file_name = f'{param.path_results}/sorting_results_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"{title}" + '\n')
        file.write("" + '\n')
        min_len = 1000
        max_len = 0
        for key in results_dict.keys():
            min_len = np.min((key, min_len))
            max_len = np.max((key, max_len))
        if results_dict_surrogate is not None:
            for key in results_dict_surrogate.keys():
                min_len = np.min((key, min_len))
                max_len = np.max((key, max_len))

        # key reprensents the length of a seq
        for key in np.arange(min_len, max_len + 1):
            nb_seq = None
            nb_seq_surrogate = None
            if key in results_dict:
                nb_seq = results_dict[key]
            if key in results_dict_surrogate:
                nb_seq_surrogate = results_dict_surrogate[key]
            str_to_write = ""
            str_to_write += f"### Length {key}: \n"
            real_data_in = False
            if nb_seq is not None:
                real_data_in = True
                str_to_write += f"# Real data: mean {np.round(np.mean(nb_seq), 3)}"
                if np.std(nb_seq) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq), 3)}"
            if nb_seq_surrogate is not None:
                if real_data_in:
                    str_to_write += f"\n"
                str_to_write += f"# Surrogate: mean {np.round(np.mean(nb_seq_surrogate), 3)}"
                if np.std(nb_seq_surrogate) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq_surrogate), 3)}"
            else:
                if not real_data_in:
                    continue
            str_to_write += '\n'
            file.write(f"{str_to_write}")
        file.write("" + '\n')
        file.write("///// Neurons sorted /////" + '\n')
        file.write("" + '\n')

        for index in np.arange(len(neurons_sorted)):
            go_for = False
            if neurons_sorted_surrogate is not None:
                if neurons_sorted_surrogate[index] == 0:
                    pass
                else:
                    go_for = True
            if (not go_for) and neurons_sorted[index] == 0:
                continue
            str_to_write = f"Neuron {index}, x "
            if neurons_sorted_surrogate is not None:
                str_to_write += f"{neurons_sorted_surrogate[index]} / "
            str_to_write += f"{neurons_sorted[index]}"
            if neurons_sorted_surrogate is not None:
                str_to_write += " (surrogate / real data)"
            str_to_write += '\n'
            file.write(f"{str_to_write}")


def main(threshold_duration, min_len_seq_first_tour, plot_raster=False, time_inter_seq=4, seq_can_be_on_same_time=True,
         n_order=1, max_depth=-1,
         threshold_factor=1.0, value_to_add_factor=1.0, min_len_seq=3, stop_if_twin=True, mouse="p12",
         show_heatmap=False, random_mode=True, min_rep_nb=1,
         keeping_only_same_diff_seq=False, split_spike_nums=False, write_on_file=True,
         produce_stat=0, sce_mode=False, spike_rate_weight=True,
         error_rate=2, no_reverse_seq=True, min_duration_intra_seq=0,
         transition_order=1):
    """

    :param plot_raster:
    :param time_inter_seq: represent the time that can be spent between two spikes of the same sequence
    :param seq_can_be_on_same_time:
    :param n_order:
    :param max_depth:
    :param min_len_seq: min len of a sequence to be displayed
    :param value_to_add_factor: used to build the transition dict, is the value added to neuron (B) that
    are found following a given one (A): (p(A|B) += (1/(nb_neurons-1))*value_to_add_factor
    :param threshold_factor: use to stop the algorithm that look for probability seq, if a neuron probability is
    inferior to (1/(nb_neurons-1))*threshold_factor, then the search stop
    :param no_reverse_seq: if True, means that we estimate that there can be reverse sequence, and that if one neuron
    spike after another one, then the probability that the opposite happen is low
    :return:
    """

    if mouse == "pp":
        root_path = "/Users/pappyhammer/Documents/academique/these_inmed/pp_data/"
    elif mouse == "claire":
        root_path = "/Users/pappyhammer/Documents/academique/these_inmed/claire_data/"
    else:
        root_path = "/Users/pappyhammer/Documents/academique/these_inmed/michel_data/"
    path_data = root_path + "data/"
    path_results = root_path + "results/"
    spikenums_file_dict = dict()
    spikenums_file_dict["p12"] = 'P12 171110 000 laura/Spikedigital analysis.mat'
    spikenums_file_dict["p12"] = 'cells event/P12 17_11_10 a000/new data/Onset matrices.mat'
    spikenums_file_dict["p11"] = 'P11 171124 000/spikenums.mat'
    spikenums_file_dict["p7"] = 'P7 171012 000/spikenums.mat'
    spikenums_file_dict["pp"] = 'SpikMtx.mat'
    spikenums_file_dict["arnaud"] = 'spikenumsarnaud.mat'

    cell_events_file_dict = dict()
    cell_events_file_dict["p12"] = 'cells event/P12 17_11_10 a000/new data/Cells to peak.mat'
    events_detection_file_dict = dict()
    events_detection_file_dict["p12"] = 'cells event/P12 17_11_10 a000/Events detection.mat'

    # dot_mat_file = spikenums_file_dict[mouse]
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + f"{time_str}/"
    os.mkdir(path_results)

    param = Parameters(time_inter_seq=time_inter_seq, seq_can_be_on_same_time=seq_can_be_on_same_time, n_order=n_order,
                       max_depth=max_depth, threshold_duration=threshold_duration,
                       min_len_seq=min_len_seq, write_on_file=write_on_file,
                       value_to_add_factor=value_to_add_factor, min_len_seq_first_tour=min_len_seq_first_tour,
                       threshold_factor=threshold_factor, stop_if_twin=stop_if_twin,
                       random_mode=random_mode, split_spike_nums=split_spike_nums,
                       error_rate=error_rate, min_rep_nb=min_rep_nb,
                       keeping_only_same_diff_seq=keeping_only_same_diff_seq,
                       no_reverse_seq=no_reverse_seq, min_duration_intra_seq=min_duration_intra_seq,
                       mouse=mouse, show_heatmap=show_heatmap, transition_order=transition_order,
                       spike_rate_weight=spike_rate_weight)
    param.path_results = path_results
    param.time_str = time_str
    if produce_stat > 0:
        if random_mode:
            file_name = f'{path_results}results_shuffle_stat_{mouse}_{time_str}.txt'
        else:
            file_name = f'{path_results}results_stat_{mouse}_{time_str}.txt'
    else:
        file_name = f'{path_results}results_{mouse}_{time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        param.file = file
        print_save('#' * 100, file, to_write=True)
        print_save(f"Mouse: {mouse}", file, to_write=True)
        print_save(str(param), file, to_write=True)
        print_save('#' * 100, file, to_write=True)
        data = hdf5storage.loadmat(path_data + spikenums_file_dict[mouse])
        data_cell_events = None
        if mouse in cell_events_file_dict:
            data_cell_events = hdf5storage.loadmat(path_data + cell_events_file_dict[mouse])
        data_events_detection = None
        if mouse in events_detection_file_dict:
            data_events_detection = hdf5storage.loadmat(
                path_data + events_detection_file_dict[mouse])
        print_data_values = False
        if print_data_values:
            for all_data in [data, data_cell_events, data_events_detection]:
                if all_data is None:
                    continue
                print_save(str(type(all_data)), file, to_write=write_on_file)  # ==> Out[9]: dict
                print_save('*' * 79, file, to_write=write_on_file)
                for key, value in all_data.items():
                    print_save(f'{key} {type(value)}', file, to_write=write_on_file)
                    print_save(f'np.shape(value): {np.shape(value)}', file, to_write=write_on_file)
                    # if len(np.shape(value)) != 0:
                    #     print(f'len(value): {len(value)}')
                    # print(f'value: {value}')
                print_save('*' * 79, file, to_write=write_on_file)

        if data_cell_events is not None:
            cellsinpeak = data_cell_events['cellsinpeak'].astype(int)
        if 'filt_spikedigital' in data:
            spike_nums = data['filt_spikedigital'].astype(int)
        elif 'spikedigital' in data:
            spike_nums = data['spikedigital'].astype(int)
        elif 'spikenums' in data:
            spike_nums = data['spikenums'].astype(int)
        else:
            spike_nums = data['Mtx']
        nb_neurons = len(spike_nums)

        if mouse == "pp":
            # thresholding
            threshold_pp = 0.4
            mask = np.greater_equal(spike_nums, threshold_pp)
            spike_nums[mask] = 1
            spike_nums[np.invert(mask)] = 0
            spike_nums = spike_nums.astype(int)
            spike_nums = spike_nums[:, :4200]
            print(f'mean(spike_nums) {np.mean(spike_nums)}, median(spike_nums) {np.median(spike_nums)}, '
                  f'std(spike_nums) {np.std(spike_nums)}, np.max(spike_nums) {np.max(spike_nums)}, '
                  f'np.sum(spike_nums) {np.sum(spike_nums)}')
        # ax_sns = sns.heatmap(spike_nums, cmap=plt.cm.jet)
        # plt.show()

        every_5000 = True

        # loss_score = loss_function(spike_nums, param)
        # print(f'raw loss_score: {np.round(loss_score, 4)}')
        # show_plot_raster(spike_nums)
        # return

        if mouse == "arnaud":
            every_5000 = False
            test_random_generator = False
            if not test_random_generator:
                # pass
                # if False:
                # new_order_arnaud = np.random.permutation(np.arange(nb_neurons))
                # spike_nums = spike_nums[new_order_arnaud, :]

                keep_the_best_order = False
                best_spike_nums = None
                best_loss_score = 1
                if keep_the_best_order:
                    nb_repet = 10
                    for i in np.arange(nb_repet):
                        new_spikes_nums = np.copy(spike_nums)
                        new_order_arnaud = np.random.permutation(np.arange(nb_neurons))
                        new_spikes_nums = new_spikes_nums[new_order_arnaud, :]
                        cleaning_spike_nums(spike_nums=new_spikes_nums, threshold_duration=threshold_duration,
                                            every_5000=every_5000)
                        loss_score = loss_function(new_spikes_nums, param)
                        print(f'raw loss_score bis: {np.round(loss_score, 4)}')
                        ordered_spike_nums, list_seq_dict, \
                        seq_edges, corresponding_neurons_index, cells_seq_to_color = order_spike_nums_by_seq(
                            spike_nums=new_spikes_nums,
                            param=param,
                            produce_stat=produce_stat)
                        loss_score = loss_function(ordered_spike_nums, param)
                        print(f'new loss_score: {np.round(loss_score, 4)}')
                        if loss_score < best_loss_score:
                            best_loss_score = loss_score
                            best_spike_nums = ordered_spike_nums
                    print(f'best loss_score: {np.round(best_loss_score, 4)}')
                    title = f'Raster ordered with sequences'
                    show_plot_raster(best_spike_nums, title=title)
                else:
                    pass
                    # new_order_arnaud = np.random.permutation(np.arange(nb_neurons))
                    # spike_nums = spike_nums[new_order_arnaud, :]
            else:
                best_order = None
                best_loss = 1
                cleaning_spike_nums(spike_nums=spike_nums, threshold_duration=threshold_duration,
                                    every_5000=every_5000)
                for i in np.arange(1000):
                    new_order_arnaud = np.random.permutation(np.arange(nb_neurons))
                    spike_nums_r = spike_nums[new_order_arnaud, :]
                    loss_score = loss_function(spike_nums_r, param)
                    print(f"random i: {i}, loss: {np.round(loss_score, 4)}")
                    if loss_score < best_loss:
                        best_loss = loss_score
                        best_order = new_order_arnaud
                print(f'best loss_score : {np.round(best_loss, 4)}')
                show_plot_raster(spike_nums[best_order, :])

        if (not random_mode) and (mouse != 'p12') and (mouse != "pp"):
            if mouse == "pp":
                every_5000 = False
                threshold_duration = 10
            # cleaning is done in find_sequence if random
            cleaning_spike_nums(spike_nums=spike_nums, threshold_duration=threshold_duration, every_5000=every_5000)
            # n_clusters=50
            # kmeans = KMeans(n_clusters=n_clusters).fit(spike_nums)
            # show_co_var_matrix(m_sces=spike_nums, n_clusters=n_clusters, nb_neurons=nb_neurons, kmeans=kmeans)
            # return
        loss_score = loss_function(spike_nums, param)
        print(f'raw loss_score bis: {np.round(loss_score, 4)}')
        # show_plot_raster(spike_nums)

        # show_plot_raster(spike_nums) source ./bin/activate.csh
        # return /Users/pappyhammer/Documents/academique/these\ inmed/python_libraries

        clusters_to_color = dict()
        blocs_dict = None
        if sce_mode and ((mouse == "pp") or (cellsinpeak is not None)):
            # use_seq_ordered_to_cluster = False
            # if use_seq_ordered_to_cluster:
            #     if ordered_spike_nums is not None:
            #         spike_nums = ordered_spike_nums
            nb_repet = 1
            for i in np.arange(nb_repet):
                # spike_nums = generate_random_spikes_num(len(spike_nums), model=spike_nums)

                if mouse == "pp":
                    blocs_dict = finding_blocs(spike_nums, mouse)
                    print(f'Nb assemblies {len(blocs_dict)}')
                    # return
                    assemblies = blocs_dict.values()
                    # making a matrix of the list of list of neurons
                    cellsinpeak = np.zeros((nb_neurons, len(assemblies)))
                    for i, s in enumerate(assemblies):
                        cellsinpeak[s, i] = 1
                # return a dict of list of list of neurons, representing the best clusters
                # (as many as nth_best_clusters).
                # the key is the K from the k-mean
                range_n_clusters = np.arange(5, 6)
                # clusters_sce = co_var_and_clusters(cellsinpeak, shuffling=False,
                #                                               range_n_clusters=range_n_clusters,
                #                                               nth_best_clusters=-1,
                #                                               plot_matrix=True)

                # clusters_sce = co_var_and_clusters_by_neurons(assemblies, len(spike_nums), shuffling=False,
                #                                    range_n_clusters=range_n_clusters, nth_best_clusters=-1,
                #                                    plot_matrix=True)

                clusters_sce, cluster_labels_for_neurons = co_var_first_and_clusters(cellsinpeak, shuffling=False,
                                                                                     range_n_clusters=range_n_clusters,
                                                                                     nth_best_clusters=-1,
                                                                                     plot_matrix=True)
                # changing the format of cluster_labels_for_neurons, still a dict but each value
                # become a list of array, each array corresponding to a set of cell that belong to the same
                # cluster
                for k, v in cluster_labels_for_neurons.items():
                    list_cells = []
                    for cluster in np.arange(-1, np.max(v) + 1):
                        v = np.array(v)
                        list_cells.append(np.arange(nb_neurons)[np.equal(v, cluster)])
                    cluster_labels_for_neurons[k] = list_cells

                clusters_to_color = cluster_labels_for_neurons
                # print(f'clusters_to_color {clusters_to_color}')
                # return
                # ploting raster organized with clusters
                lets_do_it = False
                if lets_do_it:
                    for cluster_i in range_n_clusters:
                        cluster = clusters_sce[cluster_i]
                        print(f'len(cluster) {len(cluster)}')
                        ordered_spike_nums = np.zeros((len(spike_nums), len(spike_nums[0])))
                        not_clustered_neurons = np.arange(nb_neurons)
                        # list of clusters (each cluster is a list of neurons)
                        nb_neurons_so_far = 0
                        clusters_edges = []
                        for k in cluster:
                            # print(f'k {k}, {np.setdiff1d(not_clustered_neurons, k)}')
                            not_clustered_neurons = np.setdiff1d(not_clustered_neurons, k)
                            # print(f'not_clustered_neurons {not_clustered_neurons}')
                            ordered_spike_nums[nb_neurons_so_far:nb_neurons_so_far + len(k), :] = spike_nums[k, :]
                            clusters_edges.append(nb_neurons_so_far + len(k))
                            nb_neurons_so_far += len(k)
                        if len(not_clustered_neurons) > 0:
                            ordered_spike_nums[nb_neurons_so_far:, :] = spike_nums[not_clustered_neurons, :]
                        title = f'Raster ordered with {cluster_i} clusters'
                        plot_raster_for_clusters = True
                        if plot_raster_for_clusters:
                            show_plot_raster(ordered_spike_nums, clusters_edges=clusters_edges, title=title)
                            return
            # if not plot_raster:
            #     return

        if plot_raster:
            show_plot_raster(spike_nums)
            return

        if produce_stat > 0:  # and (not sce_mode):
            # if True, will applu find_sequences on neurons of each cluster, then will concatenante those results
            # to show the raster plot
            order_by_cluster = False
            if sce_mode and order_by_cluster:
                if clusters_to_color is not None:
                    nb_clusters = 5
                    edges_cluster = []
                    new_spikes_nums = np.zeros((len(spike_nums), len(spike_nums[0])))
                    index_high = 0
                    clusters = clusters_to_color[nb_clusters]

                    for c_i, c in enumerate(clusters):
                        if len(c) > 0:
                            # print(f'c {c}')
                            ordered_spike_nums, \
                            list_seq_dict, seq_edges, corresponding_neurons_index, cells_seq_to_color = \
                                order_spike_nums_by_seq(spike_nums=spike_nums[c],
                                                        param=param,
                                                        produce_stat=produce_stat)
                            new_spikes_nums[index_high:index_high + len(ordered_spike_nums), :] = ordered_spike_nums
                            index_high += len(ordered_spike_nums)
                            edges_cluster.append(index_high)
                        else:
                            print(f'len(c) == {len(c)}')
                        # title = f'Raster ordered with sequences for cluster {c_i} of len {len(c)}'
                        # show_plot_raster(ordered_spike_nums, clusters_edges=seq_edges, title=title)
                    title = f'Raster ordered with sequences for {nb_clusters} clusters'
                    show_plot_raster(new_spikes_nums, clusters_edges=edges_cluster, title=title)
                    return

            ###################################################################
            ###################################################################
            # ##############    Sequences detection        ###################
            ###################################################################
            ###################################################################
            # 500 ms
            sliding_window_duration = 10
            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             n_surrogate=20,
                                                             perc_threshold=95,
                                                             debug_mode=False)
            print("Start of use_new_pattern_package")
            use_new_pattern_package(spike_nums=spike_nums, param=param, activity_threshold=activity_threshold,
                                    sliding_window_duration=sliding_window_duration, n_surrogate=1,
                                    mouse_id=mouse, debug_mode=True)
            return

            ordered_spike_nums, list_seq_dict, \
            seq_edges, corresponding_neurons_index, cells_seq_to_color = order_spike_nums_by_seq(spike_nums=spike_nums,
                                                                                                 param=param,
                                                                                                 produce_stat=produce_stat)

            # rearanging cluster_color with the new index
            for k, v in clusters_to_color.items():
                # v is a list of array
                for index, ar in enumerate(v):
                    v[index] = corresponding_neurons_index[ar]

            if cells_seq_to_color is not None:
                # rearanging cells_seq_to_color with the new index
                for k, v in cells_seq_to_color.items():
                    # v is a list of list of 2 elements, first element being the cell index, the second the time
                    for index, couple in enumerate(v):
                        couple[0] = corresponding_neurons_index[couple[0]]

            loss_score = loss_function(ordered_spike_nums, param)
            print(f'new loss_score: {np.round(loss_score, 4)}')
            title = f'Raster ordered with sequences'
            print('Raster ordered with sequences')
            if len(clusters_to_color) > 0:
                for k, cluster_color in clusters_to_color.items():
                    title = f'Raster ordered with sequences'
                    title += f" for {k} clusters"
                    show_plot_raster(ordered_spike_nums, clusters_edges=seq_edges, title=title,
                                     clusters_to_color=cluster_color)
            else:
                show_plot_raster(ordered_spike_nums, clusters_edges=seq_edges, title=title,
                                 cells_seq_to_color=cells_seq_to_color)

            # applying two times order_spike_nums_by_seq
            double_order = False
            if double_order:
                param.random_mode = False
                new_spikes_nums = np.zeros((len(spike_nums), len(spike_nums[0])))
                old_seq_edges = seq_edges
                first_spike_nums = ordered_spike_nums[:old_seq_edges[-2], :]
                first_spike_nums, list_seq_dict, seq_edges, \
                corresponding_neurons_index, cells_seq_to_color = order_spike_nums_by_seq(spike_nums=first_spike_nums,
                                                                                          param=param,
                                                                                          produce_stat=produce_stat)
                new_spikes_nums[:old_seq_edges[-2], :] = first_spike_nums

                second_spike_nums = ordered_spike_nums[old_seq_edges[-2]:, :]
                second_spike_nums, list_seq_dict, seq_edges, \
                corresponding_neurons_index, cells_seq_to_color = order_spike_nums_by_seq(spike_nums=second_spike_nums,
                                                                                          param=param,
                                                                                          produce_stat=produce_stat)
                new_spikes_nums[old_seq_edges[-2]:, :] = second_spike_nums
                # cells_seq_to_color.update(cells_seq_to_color_2)
                # cells_seq_to_color = {**cells_seq_to_color, **cells_seq_to_color_2}
                loss_score = loss_function(new_spikes_nums, param)
                print(f'loss_score after double ordering: {np.round(loss_score, 4)}')
                title = f'Raster ordered with sequences, 2x, '
                show_plot_raster(new_spikes_nums, clusters_edges=[], title=title, cells_seq_to_color=cells_seq_to_color)

        lets_cluster = False
        if lets_cluster:
            lets_cluster_seq(list_seq_dict=list_seq_dict, nb_neurons=len(spike_nums), param=param)

    # frame at 10 Hz, 20 events per second
    # each frame is divided by 2 thus for 12500 frames, spike_nums will have 25000 columns
    if split_spike_nums:
        for i in np.arange(int(len(spike_nums[0]) / 5000)):
            print_save("$" * 100, file, to_write=write_on_file)
            print_save(f'from {(i*5000)} to {((i+1)*5000)}', file, to_write=write_on_file)
            list_seq_dict, max_seq_dict, max_rep_non_prob, max_len_non_prob = \
                find_sequences(spike_nums=spike_nums[:, (i * 5000):((i + 1) * 5000)], no_print=True, param=param)
            print_save("$" * 100, file, to_write=write_on_file)


def loss_function(spike_nums, param):
    """
    Return a float from 0 to 1, representing the loss function.
    If spike_nums is perfectly organized as sequences (meaning that for each spike of a neuron n, the following
    spikes of other neurons (on the next lines) are the same for each given spike of n.
    Sequences are supposed to go from neurons with max index to low index
    :param spike_nums: np.array of 2 dim, first one (lines) representing the neuron numbers and 2nd one the time.
    Binary array. If value at one, represents a spike.
    :return:
    """
    loss = 0.0
    # size of the sliding matrix
    sliding_size = 3
    s_size = sliding_size
    nb_neurons = len(spike_nums)
    max_time = len(spike_nums[0, :])
    rev_spike_nums = spike_nums[::-1, :]

    # max time between 2 adjacent spike of a sequence
    # correspond as well to the highest (worst) loss for a spike
    max_time_inter = param.min_duration_intra_seq + param.time_inter_seq
    nb_spikes_total = np.sum(spike_nums)
    nb_spikes_used = np.sum(spike_nums[s_size:-s_size, :]) * (s_size * 2)
    for i in np.arange(s_size):
        nb_spikes_used += (np.sum(spike_nums[i, :]) + np.sum(spike_nums[-(i + 1), :])) * (i + 1)

    worst_loss = max_time_inter * nb_spikes_used

    # print(f'nb_neurons {nb_neurons}, nb_spikes_total {nb_spikes_total}, worst_loss {worst_loss}')

    for n, neuron in enumerate(rev_spike_nums):
        if n == (nb_neurons - (sliding_size + 1)):
            break
        n_times = np.where(neuron)[0]
        # next_n_times = np.where(rev_spike_nums[n+1, :])[0]
        # if len(n_times) == len(next_n_times):
        #     if np.all(np.diff(n_times) == np.diff(next_n_times)):
        #         continue
        # mask allowing to remove the spikes already taken in consideration to compute the loss
        # mask_next_n = np.ones((sliding_size, max_time_inter*sliding_size), dtype="bool")
        # will contain for each neuron of the sliding window, the diff value of each spike comparing to the first
        # neuron of the seq
        mean_diff = dict()
        for i in np.arange(1, sliding_size + 1):
            mean_diff[i] = []
        # we test for each spike of n the sliding_size following seq spikes
        for n_t in n_times:
            start_t = n_t + param.min_duration_intra_seq
            start_t = max(0, start_t)
            # print(f'start_t {start_t} max_time {max_time}')
            if (start_t + (max_time_inter * sliding_size)) < max_time:
                seq_mat = np.copy(rev_spike_nums[n:(n + sliding_size + 1),
                                  start_t:(start_t + (max_time_inter * sliding_size))])
            else:
                seq_mat = np.copy(rev_spike_nums[n:(n + sliding_size + 1),
                                  start_t:])
            # print(f'len(seq_mat) {len(seq_mat)} {len(seq_mat[0,:])}')
            # Keeping only one spike by neuron
            # indicate from which time we keep the first neuron, the neurons spiking before from_t are removed
            from_t = 0
            first_neuron_t = np.where(seq_mat[0, :])[0][0]
            for i in np.arange(1, sliding_size + 1):
                n_true = np.where(seq_mat[i, from_t:])[0]
                # n_true is an array of int
                if len(n_true) > 1:
                    # removing the spikes after
                    n_true_min = n_true[1:]
                    seq_mat[i, n_true_min] = 0
                # removing the spikes before
                if from_t > 0:
                    t_before = np.where(seq_mat[i, :from_t])[0]
                    if len(t_before) > 0:
                        seq_mat[i, t_before] = 0
                if len(n_true > 0):
                    # keeping the diff between the spike of the neuron in position i from the neuron n first spike
                    mean_diff[i].append((n_true[0] + from_t) - first_neuron_t)
                    from_t = n_true[0] + param.min_duration_intra_seq
                    from_t = max(0, from_t)
            # seq_mat is not used so far, but could be used for another technique.
        # we add to the loss_score, the std of the diff between spike of the first neuron and the other
        for i in np.arange(1, sliding_size + 1):
            if len(mean_diff[i]) > 0:
                # print(f'Add loss mean {np.mean(mean_diff[i])}, std {np.std(mean_diff[i])}')
                loss += min(np.std(mean_diff[i]), max_time_inter)
            # then for each spike of the neurons not used in the diff, we add to the loss the max value = max_time_inter
            # print(f'n+i {n+i} len(np.where(rev_spike_nums[n+i, :])[0]) {len(np.where(rev_spike_nums[n+i, :])[0])}'
            #       f' len(mean_diff[i]) {len(mean_diff[i])}')
            # nb_not_used_spikes could be zero when a neuron spikes a lot, and the following one not so much
            # then spikes from the following one will be involved in more than one seq
            nb_not_used_spikes = max(0, (len(np.where(rev_spike_nums[n + i, :])[0]) - len(mean_diff[i])))
            # if nb_not_used_spikes < 0:
            #     print(f'ERROR: nb_not_used_spikes inferior to 0 {nb_not_used_spikes}')
            # else:
            loss += nb_not_used_spikes * max_time_inter
        # print(f"loss_score n {loss}")

    # loss should be between 0 and 1
    return loss / worst_loss


main(threshold_duration=40, plot_raster=False, time_inter_seq=30,  # used to be 30
     seq_can_be_on_same_time=True, n_order=3, max_depth=3,  # used to be 3 and 3
     min_len_seq=8, min_rep_nb=4, keeping_only_same_diff_seq=True,  # for p12 et arnaud: min_len_seq = 6, min_rep_nb=4
     value_to_add_factor=21, show_heatmap=False,
     threshold_factor=2.1, stop_if_twin=False,  # for p12 and arnaud, threshold_factor to 2.1
     mouse="arnaud",
     min_len_seq_first_tour=[14, 15, 16, 17, 18, 19],  # [6, 7, 8, 9, 10, 11, 12, 13],
     # for p12:[10, 11, 12, 13], # for Arnaud [14, 15, 16, 17, 18, 19], for p7: [7,8]
     random_mode=False,
     split_spike_nums=False, write_on_file=True,
     produce_stat=1, sce_mode=False,
     error_rate=4,  # used to be 5
     spike_rate_weight=True,
     no_reverse_seq=False,  # better results when False
     min_duration_intra_seq=-3,  # used to be -6 then -1
     transition_order=1)

# SCE: synchronous calcium events
# P7: 0.00246
# P11: 0.002205
# P12: 0.00405
# Arnaud: 0.0032
# /Users/pappyhammer/Documents/academique/these_inmed/python_libraries
