
import sys
from SBbadger import buildNetworks
import os
import shutil
import glob
import antimony
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import importlib

pydot_spec = importlib.util.find_spec("pydot")
found_pydot = pydot_spec is not None

if found_pydot:
    import pydot


def model(verbose_exceptions=False, output_dir='models', group_name='test', overwrite=True, n_species=10,
          n_reactions=None, in_dist='random', out_dist='random', joint_dist=None, in_range=None, out_range=None,
          joint_range=None, min_freq=1.0, mass_violating_reactions=True, edge_type='generic', kinetics=None,
          add_enzyme=False, mod_reg=None, rxn_prob=None, rev_prob=0, ic_params=None, dist_plots=False, net_plots=False,
          str_format='ant'):
    """
    Generates a collection of models. This function runs the complete workflow for model generation including
    truncation and re-normalization of the distributions, reaction selection and construction of the network, and the
    imposition of rate-laws. Outputs include distribution data and figures, network data and figures, and the
    final models in Antimony and SBML formats.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_species: Number of species per model.
    :param n_reactions: Specifies the minimum number of reactions per model. Only valid in the completely random case.
    :param out_dist: Describes the out-edge distribution function, the discrete distribution,
        or the frequency distribution.
    :param in_dist: Describes the in-edge distribution function, discrete distribution,
        or frequency distribution.
    :param joint_dist: Describes the joint distribution function, discrete distribution,
        or frequency distribution.
    :param in_range: The degree range for the in-edge distribution.
    :param out_range: The degree range for the out-edge distribution.
    :param joint_range: The degree range for the joint distribution (must be symmetrical, see examples).
    :param min_freq: Sets the minimum number (expected value) of nodes (species) that must be in each degree bin.
    :param mass_violating_reactions: Allow apparent mass violating reactions such as A + B -> A.
    :param edge_type: Determines how the edges are counted against the frequency distributions.
        Current options are 'generic' and 'metabolic'.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param dist_plots: Generate distribution charts.
    :param net_plots: Generate network plots.
    :param str_format: Determines the format of the output string, antimony or sbml. Defaults to ant.
    """

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if ic_params is None:
        ic_params = ['uniform', 0, 10]

    if joint_dist and (in_dist != 'random' or out_dist != 'random'):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("You have provided both a joint distribution "
                        "and one or both of the input and output distributions")

    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated reaction probabilities are {rxn_prob} and they do not add to 1.")

    if mod_reg:
        if round(sum(mod_reg[0]), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated modular regulator probabilities are {mod_reg[0]} and they do not add to 1.")
        if mod_reg[1] < 0 or mod_reg[1] > 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your positive (vs negative) probability is {mod_reg[1]} is not between 0 and 1.")

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    if isinstance(joint_range, list) and joint_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_range, list) and in_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(out_range, list) and out_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_dist, list) and all(isinstance(x[1], int) for x in in_dist) \
            and isinstance(out_dist, list) and all(isinstance(x[1], int) for x in out_dist) \
            and sum(int(x[0]) * int(x[1]) for x in in_dist) != sum(int(x[0]) * int(x[1]) for x in out_dist):

        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("The total in-edges do not match the total out-edges. "
                        "Please revise these frequency distributions.")

    num_existing_models = 0
    path = os.path.join(output_dir, group_name, '')
    if overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))
    else:
        if os.path.exists(os.path.join(path)):
            gd = glob.glob(os.path.join(path, 'antimony', '*'))
            num_existing_models = len(gd)
        else:
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))

    # args_list = [(i, group_name, add_enzyme, n_species, n_reactions, kinetics, in_dist, out_dist, output_dir,
    #               rxn_prob, rev_prob, joint_dist, in_range, out_range, joint_range, min_freq, ic_params,
    #               mod_reg, mass_violating_reactions, dist_plots, net_plots, edge_type, str_format)
    #              for i in range(num_existing_models, num_existing_models + 1)]
    #
    # pool = Pool(1)
    # pool.starmap(generate_model, args_list)
    # pool.close()

    i = num_existing_models
    print(i)

    while i < num_existing_models + 1:

        in_samples = []
        out_samples = []
        joint_samples = []

        rl = [None]
        el = [[]]

        rl_failed_count = -1

        while not rl[0]:

            rl_failed_count += 1
            if rl_failed_count == 100:
                ant_str = "Network construction failed on this attempt, consider revising your settings."
                anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
                break

            in_samples, out_samples, joint_samples = \
                buildNetworks.generate_samples(n_species, in_dist, out_dist, joint_dist, min_freq, in_range, out_range,
                                               joint_range)

            rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                                      rxn_prob, mod_reg, mass_violating_reactions, edge_type)

        if not rl[0]:
            i += 1
            continue

    # if rl[0]:

        net_dir = os.path.join(output_dir, group_name, 'networks', group_name + '_' + str(i) + '.csv')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',[')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(':' + str(every))
                            f.write(']')
                f.write('\n')

        if net_plots and found_pydot:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))

            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'),
                        format='dot')

        if net_plots and not found_pydot:
            print('The pydot package was not found and plots will not be produced')

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        dist_dir = os.path.join(output_dir, group_name, 'distributions', group_name + '_' + str(i) + '.csv')

        with open(dist_dir, 'w') as f:
            f.write('out distribution\n')
            for each in out_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + '\n')
            f.write('\n')
            f.write('in distribution\n')
            for each in in_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + '\n')
            f.write('\n')
            f.write('joint distribution\n')
            for each in joint_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + ',' + str(each[2]) + '\n')
            f.write('\n')

        # todo: write separate script for visualization?
        if dist_plots:

            if in_samples and not out_samples:
                x = [dist_ind[0] for dist_ind in in_samples]
                y = [dist_ind[1] for dist_ind in in_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("Out Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' out edges')
                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_in'
                                         + '.png'))
                plt.close()

            if out_samples and not in_samples:
                x = [dist_ind[0] for dist_ind in out_samples]
                y = [dist_ind[1] for dist_ind in out_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("In Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' in edges')
                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
                                         + '.png'))
                plt.close()

            if in_samples and out_samples:

                y10 = [each[0] for each in out_samples]
                y20 = [each[0] for each in in_samples]
                miny = min(min(y10), min(y20))
                maxy = max(max(y10), max(y20))
                y0 = [m for m in range(miny, maxy + 1)]
                miss1 = list(set(y10) ^ set(y0))
                miss2 = list(set(y20) ^ set(y0))

                for each in miss1:
                    out_samples.append((each, 0))
                for each in miss2:
                    in_samples.append((each, 0))
                out_samples.sort()
                in_samples.sort()

                y1 = [dist_ind[1] for dist_ind in out_samples]
                y2 = [dist_ind[1] for dist_ind in in_samples]
                x1 = [dist_ind[0] for dist_ind in out_samples]
                x2 = [dist_ind[0] for dist_ind in in_samples]
                x0 = list(set(x1).union(set(x2)))
                x0.sort()

                x = np.arange(len(x0))

                width = 0.25
                fig, ax = plt.subplots()
                ax.bar(x-width/2, y1, width, label='outdegree')
                ax.bar(x+width/2, y2, width, label='indegree')
                ax.set_xlabel("Edge Degree")
                ax.set_ylabel("Number of Nodes")
                ax.set_xticks(x)
                ax.set_xticklabels(x0)
                ax.legend()

                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out_in'
                                         + '.png'))
                plt.close()

            if joint_samples:
                x = [dist_ind[0] for dist_ind in joint_samples]
                y = [dist_ind[1] for dist_ind in joint_samples]
                z = [0 for _ in joint_samples]

                dx = np.ones(len(joint_samples))
                dy = np.ones(len(joint_samples))
                dz = [dist_ind[2] for dist_ind in joint_samples]

                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.bar3d(x, y, z, dx, dy, dz)

                ax1.set_xlabel("Out-Edge Degree")
                ax1.set_ylabel("In-Edge Degree")
                ax1.set_zlabel("Number of Nodes")

                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_joint'
                                         + '.png'))
                plt.close()

        sbml_dir = os.path.join(output_dir, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')

        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()

        output_str = None
        if str_format == 'ant':
            output_str = ant_str
        if str_format == 'sbml':
            output_str = sbml

        return output_str


def generate_models(i, group_name, add_enzyme, n_species, n_reactions, kinetics, in_dist, out_dist, output_dir,
                    rxn_prob, rev_prob, joint_dist, in_range, out_range, joint_range, min_freq, ic_params, mod_reg,
                    mass_violating_reactions, dist_plots, net_plots, edge_type):

    in_samples = []
    out_samples = []
    joint_samples = []

    rl = [None]
    el = [[]]

    rl_failed_count = -1

    while not rl[0]:

        rl_failed_count += 1
        if rl_failed_count == 100:
            ant_str = "Network construction failed on this attempt, consider revising your settings."
            anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
            with open(anti_dir, 'w') as f:
                f.write(ant_str)
            break

        in_samples, out_samples, joint_samples = \
            buildNetworks.generate_samples(n_species, in_dist, out_dist, joint_dist, min_freq, in_range, out_range,
                                           joint_range)

        rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                                  rxn_prob, mod_reg, mass_violating_reactions, edge_type)

    if rl[0]:

        net_dir = os.path.join(output_dir, group_name, 'networks', group_name + '_' + str(i) + '.csv')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',[')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(':' + str(every))
                            f.write(']')
                f.write('\n')

        if net_plots and found_pydot:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))

            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'),
                        format='dot')

        if net_plots and not found_pydot:
            print('The pydot package was not found and plots will not be produced')

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        dist_dir = os.path.join(output_dir, group_name, 'distributions', group_name + '_' + str(i) + '.csv')

        with open(dist_dir, 'w') as f:
            f.write('out distribution\n')
            for each in out_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + '\n')
            f.write('\n')
            f.write('in distribution\n')
            for each in in_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + '\n')
            f.write('\n')
            f.write('joint distribution\n')
            for each in joint_samples:
                f.write(str(each[0]) + ',' + str(each[1]) + ',' + str(each[2]) + '\n')
            f.write('\n')

        # todo: write separate script for visualization?
        if dist_plots:

            if in_samples and not out_samples:
                x = [dist_ind[0] for dist_ind in in_samples]
                y = [dist_ind[1] for dist_ind in in_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("Out Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' out edges')
                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_in'
                                         + '.png'))
                plt.close()

            if out_samples and not in_samples:
                x = [dist_ind[0] for dist_ind in out_samples]
                y = [dist_ind[1] for dist_ind in out_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("In Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' in edges')
                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
                                         + '.png'))
                plt.close()

            if in_samples and out_samples:

                y10 = [each[0] for each in out_samples]
                y20 = [each[0] for each in in_samples]
                miny = min(min(y10), min(y20))
                maxy = max(max(y10), max(y20))
                y0 = [m for m in range(miny, maxy + 1)]
                miss1 = list(set(y10) ^ set(y0))
                miss2 = list(set(y20) ^ set(y0))

                for each in miss1:
                    out_samples.append((each, 0))
                for each in miss2:
                    in_samples.append((each, 0))
                out_samples.sort()
                in_samples.sort()

                y1 = [dist_ind[1] for dist_ind in out_samples]
                y2 = [dist_ind[1] for dist_ind in in_samples]
                x1 = [dist_ind[0] for dist_ind in out_samples]
                x2 = [dist_ind[0] for dist_ind in in_samples]
                x0 = list(set(x1).union(set(x2)))
                x0.sort()

                x = np.arange(len(x0))

                width = 0.25
                fig, ax = plt.subplots()
                ax.bar(x-width/2, y1, width, label='outdegree')
                ax.bar(x+width/2, y2, width, label='indegree')
                ax.set_xlabel("Edge Degree")
                ax.set_ylabel("Number of Nodes")
                ax.set_xticks(x)
                ax.set_xticklabels(x0)
                ax.legend()

                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out_in'
                                         + '.png'))
                plt.close()

            if joint_samples:
                x = [dist_ind[0] for dist_ind in joint_samples]
                y = [dist_ind[1] for dist_ind in joint_samples]
                z = [0 for _ in joint_samples]

                dx = np.ones(len(joint_samples))
                dy = np.ones(len(joint_samples))
                dz = [dist_ind[2] for dist_ind in joint_samples]

                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.bar3d(x, y, z, dx, dy, dz)

                ax1.set_xlabel("Out-Edge Degree")
                ax1.set_ylabel("In-Edge Degree")
                ax1.set_zlabel("Number of Nodes")

                plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_joint'
                                         + '.png'))
                plt.close()

        sbml_dir = os.path.join(output_dir, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')

        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()


def models(verbose_exceptions=False, output_dir='models', group_name='test', overwrite=True, n_models=1, n_species=10, 
           n_reactions=None, in_dist='random', out_dist='random', joint_dist=None, in_range=None, out_range=None, 
           joint_range=None, min_freq=1.0, mass_violating_reactions=True, edge_type='generic', kinetics=None, 
           add_enzyme=False, mod_reg=None, rxn_prob=None, rev_prob=0, ic_params=None, dist_plots=False, net_plots=False, 
           n_cpus=1):
    """
    Generates a collection of models. This function runs the complete workflow for model generation including
    truncation and re-normalization of the distributions, reaction selection and construction of the network, and the
    imposition of rate-laws. Outputs include distribution data and figures, network data and figures, and the
    final models in Antimony and SBML formats.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_models: Number of models to produce.
    :param n_species: Number of species per model.
    :param n_reactions: Specifies the minimum number of reactions per model. Only valid in the completely random case.
    :param out_dist: Describes the out-edge distribution function, the discrete distribution,
        or the frequency distribution.
    :param in_dist: Describes the in-edge distribution function, discrete distribution,
        or frequency distribution.
    :param joint_dist: Describes the joint distribution function, discrete distribution,
        or frequency distribution.
    :param in_range: The degree range for the in-edge distribution.
    :param out_range: The degree range for the out-edge distribution.
    :param joint_range: The degree range for the joint distribution (must be symmetrical, see examples).
    :param min_freq: Sets the minimum number (expected value) of nodes (species) that must be in each degree bin.
    :param mass_violating_reactions: Allow apparent mass violating reactions such as A + B -> A.
    :param edge_type: Determines how the edges are counted against the frequency distributions.
        Current options are 'generic' and 'metabolic'.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param dist_plots: Generate distribution charts.
    :param net_plots: Generate network plots.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if ic_params is None:
        ic_params = ['uniform', 0, 10]

    if joint_dist and (in_dist != 'random' or out_dist != 'random'):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("You have provided both a joint distribution "
                        "and one or both of the input and output distributions")

    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated reaction probabilities are {rxn_prob} and they do not add to 1.")

    if mod_reg:
        if round(sum(mod_reg[0]), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated modular regulator probabilities are {mod_reg[0]} and they do not add to 1.")
        if mod_reg[1] < 0 or mod_reg[1] > 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your positive (vs negative) probability is {mod_reg[1]} is not between 0 and 1.")

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    if isinstance(joint_range, list) and joint_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_range, list) and in_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(out_range, list) and out_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_dist, list) and all(isinstance(x[1], int) for x in in_dist) \
            and isinstance(out_dist, list) and all(isinstance(x[1], int) for x in out_dist) \
            and sum(int(x[0]) * int(x[1]) for x in in_dist) != sum(int(x[0]) * int(x[1]) for x in out_dist):

        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("The total in-edges do not match the total out-edges. "
                        "Please revise these frequency distributions.")

    num_existing_models = 0
    path = os.path.join(output_dir, group_name, '')
    if overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))
    else:
        if os.path.exists(os.path.join(path)):
            gd = glob.glob(os.path.join(path, 'antimony', '*'))
            num_existing_models = len(gd)
        else:
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
            os.makedirs(os.path.join(path, 'dot_files'))
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))

    args_list = [(i, group_name, add_enzyme, n_species, n_reactions, kinetics, in_dist, out_dist, output_dir,
                  rxn_prob, rev_prob, joint_dist, in_range, out_range, joint_range, min_freq, ic_params,
                  mod_reg, mass_violating_reactions, dist_plots, net_plots, edge_type)
                 for i in range(num_existing_models, n_models)]

    pool = Pool(n_cpus)
    pool.starmap(generate_models, args_list)
    pool.close()


def generate_distributions(i, group_name, n_species, in_dist, out_dist, output_dir, joint_dist, in_range, out_range,
                           joint_range, min_freq, dist_plots):

    in_samples, out_samples, joint_samples = buildNetworks.generate_samples(
        n_species, in_dist, out_dist, joint_dist, min_freq, in_range, out_range, joint_range)

    dist_dir = os.path.join(output_dir, group_name, 'distributions', group_name + '_' + str(i) + '.csv')

    with open(dist_dir, 'w') as f:
        f.write('out distribution\n')
        for each in out_samples:
            f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        f.write('\n')
        f.write('in distribution\n')
        for each in in_samples:
            f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        f.write('\n')
        f.write('joint distribution\n')
        for each in joint_samples:
            f.write(str(each[0]) + ',' + str(each[1]) + ',' + str(each[2]) + '\n')
        f.write('\n')

    if dist_plots:

        if in_samples and not out_samples:
            x = [dist_ind[0] for dist_ind in in_samples]
            y = [dist_ind[1] for dist_ind in in_samples]
            plt.figure()
            plt.bar(x, y)
            plt.xlabel("Out Degree")
            plt.ylabel("Number of Nodes")
            plt.xticks(x)
            plt.title(group_name + '_' + str(i) + ' out edges')
            plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_in'
                                     + '.png'))
            plt.close()

        if out_samples and not in_samples:
            x = [dist_ind[0] for dist_ind in out_samples]
            y = [dist_ind[1] for dist_ind in out_samples]
            plt.figure()
            plt.bar(x, y)
            plt.xlabel("In Degree")
            plt.ylabel("Number of Nodes")
            plt.xticks(x)
            plt.title(group_name + '_' + str(i) + ' in edges')
            plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
                                     + '.png'))
            plt.close()

        if in_samples and out_samples:

            y10 = [each[0] for each in out_samples]
            y20 = [each[0] for each in in_samples]
            miny = min(min(y10), min(y20))
            maxy = max(max(y10), max(y20))
            y0 = [m for m in range(miny, maxy + 1)]
            miss1 = list(set(y10) ^ set(y0))
            miss2 = list(set(y20) ^ set(y0))

            for each in miss1:
                out_samples.append((each, 0))
            for each in miss2:
                in_samples.append((each, 0))
            out_samples.sort()
            in_samples.sort()

            y1 = [dist_ind[1] for dist_ind in out_samples]
            y2 = [dist_ind[1] for dist_ind in in_samples]
            x1 = [dist_ind[0] for dist_ind in out_samples]
            x2 = [dist_ind[0] for dist_ind in in_samples]
            x0 = list(set(x1).union(set(x2)))
            x0.sort()

            x = np.arange(len(x0))

            width = 0.25
            fig, ax = plt.subplots()
            ax.bar(x-width/2, y1, width, label='outdegree')
            ax.bar(x+width/2, y2, width, label='indegree')
            ax.set_xlabel("Edge Degree")
            ax.set_ylabel("Number of Nodes")
            ax.set_xticks(x)
            ax.set_xticklabels(x0)
            ax.legend()

            plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_out_in'
                                     + '.png'))
            plt.close()

        if joint_samples:
            x = [dist_ind[0] for dist_ind in joint_samples]
            y = [dist_ind[1] for dist_ind in joint_samples]
            z = [0 for _ in joint_samples]

            dx = np.ones(len(joint_samples))
            dy = np.ones(len(joint_samples))
            dz = [dist_ind[2] for dist_ind in joint_samples]

            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(x, y, z, dx, dy, dz)

            ax1.set_xlabel("Out-Edge Degree")
            ax1.set_ylabel("In-Edge Degree")
            ax1.set_zlabel("Number of Nodes")

            plt.savefig(os.path.join(output_dir, group_name, 'dist_figs', group_name + '_' + str(i) + '_joint'
                                     + '.png'))
            plt.close()


def distributions(verbose_exceptions=False, output_dir='models', group_name='test', overwrite=True, n_models=1,
                  n_species=10, out_dist='random', in_dist='random', joint_dist=None, in_range=None, out_range=None,
                  joint_range=None, min_freq=1.0, dist_plots=False, n_cpus=1):
    """
    Generates a collection of frequency distributions from function or bound discrete probabilities.
    Outputs include distribution data and figures.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_models: Number of models to produce.
    :param n_species: Number of species per model.
    :param out_dist: Describes the out-edge distribution function, the discrete distribution,
        or the frequency distribution.
    :param in_dist: Describes the in-edge distribution function, discrete distribution,
        or frequency distribution.
    :param joint_dist: Describes the joint distribution function, discrete distribution,
        or frequency distribution.
    :param in_range: The degree range for the in-edge distribution.
    :param out_range: The degree range for the out-edge distribution.
    :param joint_range: The degree range for the joint distribution (must be symmetrical, see examples).
    :param min_freq: Sets the minimum number (expected value) of nodes (species) that must be in each degree bin.
    :param dist_plots: Generate distribution charts.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if joint_dist and (in_dist != 'random' or out_dist != 'random'):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("You have provided both a joint distribution "
                        "and one or both of the input and output distributions")

    if isinstance(joint_range, list) and joint_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_range, list) and in_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(out_range, list) and out_range[0] < 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(in_dist, list) and all(isinstance(x[1], float) for x in in_dist):
        prob_list = [each[1] for each in in_dist]
        if round(sum(prob_list), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("The provided indegree distribution does not add to 1")

    if isinstance(out_dist, list) and all(isinstance(x[1], float) for x in out_dist):
        prob_list = [each[1] for each in out_dist]
        if round(sum(prob_list), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("The provided outdegree distribution does not add to 1")

    if isinstance(joint_dist, list) and all(isinstance(x[1], float) for x in joint_dist):
        prob_list = [each[2] for each in joint_dist]
        if round(sum(prob_list), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("The provided joint degree distribution does not add to 1")

    if isinstance(in_dist, list) and all(isinstance(x[1], int) for x in in_dist) \
            and isinstance(out_dist, list) and all(isinstance(x[1], int) for x in out_dist) \
            and sum(int(x[0]) * int(x[1]) for x in in_dist) != sum(int(x[0]) * int(x[1]) for x in out_dist):

        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("The total in-edges do not match the total out-edges. "
                        "Please revise these frequency distributions.")

    num_existing_models = 0
    path = os.path.join(output_dir, group_name)

    if overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(os.path.join(path, 'distributions'))
            if dist_plots:
                os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            os.makedirs(os.path.join(path, 'distributions'))
            if dist_plots:
                os.makedirs(os.path.join(path, 'dist_figs'))
    else:
        if os.path.exists(os.path.join(path)):
            gd = glob.glob(os.path.join(path, 'distributions', '*'))
            num_existing_models = len(gd)
        else:
            os.makedirs(os.path.join(path, 'distributions'))
            if dist_plots:
                os.makedirs(os.path.join(path, 'dist_figs'))

    args_list = [(i, group_name, n_species, in_dist, out_dist, output_dir, joint_dist, in_range, out_range,
                  joint_range, min_freq, dist_plots) for i in range(num_existing_models, n_models)]

    pool = Pool(n_cpus)
    pool.starmap(generate_distributions, args_list)
    pool.close()


def generate_networks(i, dists_list, directory, group_name, n_reactions, rxn_prob, mod_reg, mass_violating_reactions,
                      edge_type, net_plots):

    out_dist = False
    in_dist = False
    joint_dist = False
    out_samples = []
    in_samples = []
    joint_samples = []
    with open(os.path.join(directory, group_name, 'distributions', dists_list[i][1])) as dl:
        for line in dl:
            if joint_dist:
                if line.strip():
                    joint_samples.append((int(line.split(',')[0]), int(line.split(',')[1].strip())))
            if line[:-1] == 'joint distribution':
                out_dist = False
                in_dist = False
                joint_dist = True
            if in_dist:
                if line.strip():
                    in_samples.append((int(line.split(',')[0]), int(line.split(',')[1].strip())))
            if line[:-1] == 'in distribution':
                out_dist = False
                in_dist = True
                joint_dist = False
            if out_dist:
                if line.strip():
                    out_samples.append((int(line.split(',')[0]), int(line.split(',')[1].strip())))
            if line[:-1] == 'out distribution':
                out_dist = True
                in_dist = False
                joint_dist = False

    n_species = 0
    out_species = 0
    in_species = 0

    if out_samples and not n_species:
        for each in out_samples:
            out_species += each[1]

    if in_samples and not n_species:
        for each in in_samples:
            in_species += each[1]

    n_species = max(out_species, in_species)

    if joint_samples and not n_species:
        for each in joint_samples:
            n_species += each[1]

    rl = [None]
    el = [[]]

    rl_failed_count = -1

    while not rl[0]:

        rl_failed_count += 1
        if rl_failed_count == 100:
            break

        rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                                  rxn_prob, mod_reg, mass_violating_reactions, edge_type)

    if not rl[0]:

        ant_str = "Network construction failed on this attempt, consider revising your settings."
        anti_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)
    else:
        net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(i) + '.txt')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',(')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(':' + str(every))
                            f.write(')')
                f.write('\n')

    if net_plots and found_pydot:
        edges = []
        for each in el:
            edges.append(('S' + str(each[0]), 'S' + str(each[1])))
        graph = pydot.Dot(graph_type="digraph")
        graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
        for each in edges:
            graph.add_edge(pydot.Edge(each[0], each[1]))
        graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
        graph.write(os.path.join(directory, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'), format='dot')
    if net_plots and not found_pydot:
        print('The pydot package was not found and plots will not be produced')


def networks(verbose_exceptions=False, directory='models', group_name='test', overwrite=True, n_reactions=None, 
             mass_violating_reactions=True, edge_type='generic', mod_reg=None, rxn_prob=None,
             net_plots=False, n_cpus=1):
    """
    Generates a collection of reaction networks. This function requires the existence of previously generated 
    frequency distributions.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param directory: Directory where files are read and stored.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_reactions: Specifies the minimum number of reactions per model. Only valid in the completely random case.
    :param mass_violating_reactions: Allow apparent mass violating reactions such as A + B -> A.
    :param edge_type: Determines how the edges are counted against the frequency distributions.
        Current options are 'generic' and 'metabolic'.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param net_plots: Generate network plots.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated reaction probabilities are {rxn_prob} and they do not add to 1.")

    if mod_reg:
        if round(sum(mod_reg[0]), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated modular regulator probabilities are {mod_reg[0]} and they do not add to 1.")
        if mod_reg[1] < 0 or mod_reg[1] > 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your positive (vs negative) probability is {mod_reg[1]} is not between 0 and 1.")

    net_files = []
    if overwrite:
        if os.path.exists(os.path.join(directory, group_name, 'networks')):
            shutil.rmtree(os.path.join(directory, group_name, 'networks'))
            os.makedirs(os.path.join(directory, group_name, 'networks'))
        else:
            os.makedirs(os.path.join(directory, group_name, 'networks'))

        if os.path.exists(os.path.join(directory, group_name, 'net_figs')):
            shutil.rmtree(os.path.join(directory, group_name, 'net_figs'))
            os.makedirs(os.path.join(directory, group_name, 'net_figs'))
        else:
            os.makedirs(os.path.join(directory, group_name, 'net_figs'))

        if os.path.exists(os.path.join(directory, group_name, 'dot_files')):
            shutil.rmtree(os.path.join(directory, group_name, 'dot_files'))
            os.makedirs(os.path.join(directory, group_name, 'dot_files'))
        else:
            os.makedirs(os.path.join(directory, group_name, 'dot_files'))

    else:
        if os.path.exists(os.path.join(directory, group_name, 'networks')):
            net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'networks'))
                         if os.path.isfile(os.path.join(directory, group_name, 'networks', f))]

        else:
            os.makedirs(os.path.join(directory, group_name, 'networks'))
            os.makedirs(os.path.join(directory, group_name, 'net_figs'))
            os.makedirs(os.path.join(directory, group_name, 'dot_files'))

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]

    path = os.path.join(directory, group_name, 'distributions')
    dist_files = [fi for fi in os.listdir(path) if os.path.isfile(os.path.join(path, fi)) and fi[-3:] == 'csv']
    dists_list = []

    for item in dist_files:
        dists_list.append([int(item.split('_')[-1].split('.')[0]), item])
    dists_list.sort()

    args_list = [(dist[0], dists_list, directory, group_name, n_reactions, rxn_prob, mod_reg,
                  mass_violating_reactions, edge_type, net_plots)
                 for dist in dists_list if dist not in net_inds]

    pool = Pool(n_cpus)
    pool.starmap(generate_networks, args_list)
    pool.close()


def generate_rate_laws(i, nets_list, directory, group_name, add_enzyme, kinetics, rev_prob, ic_params):

    reg_check = False
    with open(os.path.join(directory, group_name, 'networks', nets_list[i][1])) as f:
        if 'a' in f.read():
            reg_check = True
    with open(os.path.join(directory, group_name, 'networks', nets_list[i][1])) as f:
        if 's' in f.read():
            reg_check = True

    if reg_check and 'modular' not in kinetics[0]:
        ant_str = "This model contains regulators that are not accounted for in the selected kinetics."
        anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

    else:
        rl = []
        with open(os.path.join(directory, group_name, 'networks', nets_list[i][1])) as nl:
            for j, line in enumerate(nl):
                if j == 0:
                    rl.append(int(line.strip()))
                else:
                    rl.append([])
                    line_split = line.strip().split(',')
                    for k, each in enumerate(line_split):
                        if k == 0:
                            rl[-1].append(int(each))
                        elif k == 5:
                            rl[-1].append([])
                            each_split = each[1:-1].split(':')
                            for elem in each_split:
                                if elem:
                                    rl[-1][-1].append(elem)
                        else:
                            rl[-1].append([])
                            each_split = each[1:-1].split(':')
                            for elem in each_split:
                                if elem:
                                    rl[-1][-1].append(int(elem))

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)

        anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        sbml_dir = os.path.join(directory, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')
        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()


def rate_laws(verbose_exceptions=False, directory='models', group_name='test', overwrite=True, kinetics=None, 
              add_enzyme=False, mod_reg=None, rxn_prob=None, rev_prob=0, ic_params=None, n_cpus=1):
    """
    Generates a collection of models. This function requires the existence of previously generated networks.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param directory: Directory where files are read and stored.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if kinetics and kinetics[1] != 'uniform' and kinetics[1] != 'loguniform' \
            and kinetics[1] != 'normal' and kinetics[1] != 'lognormal':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please specify the parameter distribution as "uniform", "loguniform", "normal", '
                        'or "lognormal".')

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if ic_params is None:
        ic_params = ['uniform', 0, 10]

    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated reaction probabilities are {rxn_prob} and they do not add to 1.")

    if mod_reg:
        if round(sum(mod_reg[0]), 10) != 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your stated modular regulator probabilities are {mod_reg[0]} and they do not add to 1.")
        if mod_reg[1] < 0 or mod_reg[1] > 1:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception(f"Your positive (vs negative) probability is {mod_reg[1]} is not between 0 and 1.")

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    anti_files = []
    sbml_files = []
    if overwrite:
        if os.path.exists(os.path.join(directory, group_name, 'antimony')):
            shutil.rmtree(os.path.join(directory, group_name, 'antimony'))
            os.makedirs(os.path.join(directory, group_name, 'antimony'))
        else:
            os.makedirs(os.path.join(directory, group_name, 'antimony'))

        if os.path.exists(os.path.join(directory, group_name, 'sbml')):
            shutil.rmtree(os.path.join(directory, group_name, 'sbml'))
            os.makedirs(os.path.join(directory, group_name, 'sbml'))
        else:
            os.makedirs(os.path.join(directory, group_name, 'sbml'))

    else:
        if os.path.exists(os.path.join(directory, group_name, 'antimony')):
            anti_files = [f for f in os.listdir(os.path.join(directory, group_name, 'antimony'))
                          if os.path.isfile(os.path.join(directory, group_name, 'antimony', f))]
        else:
            os.makedirs(os.path.join(directory, group_name, 'antimony'))

        if os.path.exists(os.path.join(directory, group_name, 'sbml')):
            sbml_files = [f for f in os.listdir(os.path.join(directory, group_name, 'sbml'))
                          if os.path.isfile(os.path.join(directory, group_name, 'sbml', f))]
        else:
            os.makedirs(os.path.join(directory, group_name, 'sbml'))

    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(anti_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the antimony and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    path = os.path.join(directory, group_name, 'networks')
    nets_files = [fi for fi in os.listdir(path) if os.path.isfile(os.path.join(path, fi)) and fi[-3:] == 'txt']

    nets_list = []
    for item in nets_files:
        nets_list.append([int(item.split('_')[-1].split('.')[0]), item])
    nets_list.sort()

    args_list = [(net[0], nets_list, directory, group_name, add_enzyme, kinetics, rev_prob, ic_params)
                 for net in nets_list if net not in anti_inds]

    pool = Pool(n_cpus)
    pool.starmap(generate_rate_laws, args_list)
    pool.close()


def generate_linear(i, group_name, add_enzyme, n_species, kinetics, rev_prob, ic_params, output_dir, net_plots):

    rl, el = buildNetworks.generate_simple_linear(n_species)

    if not rl[0]:

        ant_str = "Network construction failed on this attempt, consider revising your settings."
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)
    else:

        net_dir = os.path.join(output_dir, group_name, 'networks', group_name + '_' + str(i) + '.csv')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',(')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(',' + str(every))
                            f.write(')')
                f.write('\n')

        if net_plots and found_pydot:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))
            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'),
                        format='dot')
        if net_plots and not found_pydot:
            print('The pydot package was not found and plots will not be produced')

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)

        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        sbml_dir = os.path.join(output_dir, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')
        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()


def linear(verbose_exceptions=False, output_dir='models', group_name='linear', overwrite=True, n_models=1, n_species=10, 
           kinetics=None, add_enzyme=False, rev_prob=0, ic_params=None, net_plots=False, n_cpus=1):
    """
    Generates a collection of UNI-UNI linear models.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_models: Number of models to produce.
    :param n_species: Number of species per model.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    net_files = []
    anti_files = []
    sbml_files = []
    if overwrite:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'antimony'))
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'sbml'))
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'networks'))
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'net_figs'))
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'dot_files'))
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    else:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            anti_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'antimony'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'antimony', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            sbml_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'sbml'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'sbml', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'networks'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'networks', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'net_figs'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'net_figs', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'dot_files'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'dot_files', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    args_list = [(i, group_name, add_enzyme, n_species, kinetics, rev_prob, ic_params, output_dir, net_plots)
                 for i in range(n_models) if i not in net_inds]

    pool = Pool(n_cpus)
    pool.starmap(generate_linear, args_list)
    pool.close()


def generate_cyclic(i, group_name, add_enzyme, min_species, max_species, n_cycles, kinetics, rev_prob, ic_params,
                    output_dir, net_plots):

    rl, el = buildNetworks.generate_simple_cyclic(min_species, max_species, n_cycles)

    if not rl[0]:

        ant_str = "Network construction failed on this attempt, consider revising your settings."
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)
    else:
        net_dir = os.path.join(output_dir, group_name, 'networks', group_name + '_' + str(i) + '.csv')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',(')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(',' + str(every))
                            f.write(')')
                f.write('\n')

        if net_plots and found_pydot:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph", layout="neato")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))
            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'),
                        format='dot')
        if net_plots and not found_pydot:
            print('The pydot package was not found and plots will not be produced')

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)

        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        sbml_dir = os.path.join(output_dir, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')
        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()


def cyclic(verbose_exceptions=False, output_dir='models', group_name='cyclic', overwrite=True, min_species=10,
           max_species=20, n_cycles=1, n_models=1, kinetics=None, add_enzyme=False, rev_prob=0, ic_params=None,
           net_plots=False, n_cpus=1):
    """
    Generates a collection of UNI-UNI cyclic models.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param min_species: Minimum number of species per cycle.
    :param max_species: Maximum number of species per cycle.
    :param n_cycles: Number of cycles per model.
    :param n_models: Number of models to produce.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    net_files = []
    anti_files = []
    sbml_files = []
    if overwrite:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'antimony'))
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'sbml'))
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'networks'))
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'net_figs'))
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'dot_files'))
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    else:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            anti_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'antimony'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'antimony', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            sbml_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'sbml'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'sbml', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'networks'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'networks', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'net_figs'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'net_figs', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'dot_files'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'dot_files', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    args_list = [(i, group_name, add_enzyme, min_species, max_species, n_cycles, kinetics, rev_prob,
                  ic_params, output_dir, net_plots) for i in range(n_models) if i not in net_inds]

    pool = Pool(n_cpus)
    pool.starmap(generate_cyclic, args_list)
    pool.close()


def generate_branched(i, group_name, add_enzyme, n_species, kinetics, rev_prob, ic_params, output_dir, net_plots, seeds,
                      path_probs, tips):

    rl, el = buildNetworks.generate_simple_branched(n_species, seeds, path_probs, tips)

    if not rl[0]:

        ant_str = "Network construction failed on this attempt, consider revising your settings."
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)
    else:

        net_dir = os.path.join(output_dir, group_name, 'networks', group_name + '_' + str(i) + '.csv')
        with open(net_dir, 'w') as f:
            for j, each in enumerate(rl):
                if j == 0:
                    f.write(str(each))
                else:
                    for k, item in enumerate(each):
                        if k == 0:
                            f.write(str(item))
                        else:
                            f.write(',(')
                            for m, every in enumerate(item):
                                if m == 0:
                                    f.write(str(every))
                                else:
                                    f.write(',' + str(every))
                            f.write(')')
                f.write('\n')

        if net_plots and found_pydot:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))
            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'dot_files', group_name + '_' + str(i) + '.dot'),
                        format='dot')
        if net_plots and not found_pydot:
            print('The pydot package was not found and plots will not be produced')

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)

        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        sbml_dir = os.path.join(output_dir, group_name, 'sbml', group_name + '_' + str(i) + '.sbml')
        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()


def branched(verbose_exceptions=False, output_dir='models', group_name='branched', overwrite=True, n_models=1, 
             n_species=20, seeds=1, path_probs=None, tips=False, kinetics=None, add_enzyme=False, rev_prob=0, 
             ic_params=None, net_plots=False, n_cpus=1):
    """
    Generates a collection of UNI-UNI branched models from a set of seed nodes.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_models: Number of models to produce.
    :param n_species: Number of species per model.
    :param seeds: The number of seed nodes that the network(s) will grow from
    :param path_probs: list of probabilities that govern the rate of branching and converging. Defaults to 
        [branch, grow, combine] = [0.1, 0.8, 0.1].
    :param tips: Confines branching, growth, and converging to the tip of the stems.  
    :param kinetics: Describes the desired rate-laws and parameter ranges. Defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
    :param n_cpus: Provides the number of cores to be used in parallel.
    """

    if kinetics is None:
        kinetics = ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]

    if rev_prob < 0 or rev_prob > 1:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Your reversibility probability is not between 0 and 1')

    if path_probs is None:
        path_probs = [0.1, 0.8, 0.1]

    net_files = []
    anti_files = []
    sbml_files = []
    if overwrite:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'antimony'))
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'sbml'))
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'networks'))
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'net_figs'))
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            shutil.rmtree(os.path.join(output_dir, group_name, 'dot_files'))
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    else:
        if os.path.exists(os.path.join(output_dir, group_name, 'antimony')):
            anti_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'antimony'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'antimony', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'antimony'))

        if os.path.exists(os.path.join(output_dir, group_name, 'sbml')):
            sbml_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'sbml'))
                          if os.path.isfile(os.path.join(output_dir, group_name, 'sbml', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'sbml'))

        if os.path.exists(os.path.join(output_dir, group_name, 'networks')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'networks'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'networks', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'networks'))

        if os.path.exists(os.path.join(output_dir, group_name, 'net_figs')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'net_figs'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'net_figs', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'net_figs'))

        if os.path.exists(os.path.join(output_dir, group_name, 'dot_files')):
            net_files = [f for f in os.listdir(os.path.join(output_dir, group_name, 'dot_files'))
                         if os.path.isfile(os.path.join(output_dir, group_name, 'dot_files', f))]
        else:
            os.makedirs(os.path.join(output_dir, group_name, 'dot_files'))

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    args_list = [(i, group_name, add_enzyme, n_species, kinetics, rev_prob, ic_params, output_dir, net_plots,
                  seeds, path_probs, tips) for i in range(n_models) if i not in net_inds]

    pool = Pool(n_cpus)
    pool.starmap(generate_branched, args_list)
    pool.close()
