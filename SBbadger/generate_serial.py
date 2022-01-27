
import sys
import buildNetworks
import os
import shutil
import glob
import antimony
import matplotlib.pyplot as plt
import numpy as np
import pydot
import io


def models(verbose_exceptions=False, output_dir='models', group_name='test', overwrite=True, n_models=1, n_species=10, 
           n_reactions=None, in_dist='random', out_dist='random', joint_dist=None, in_range=None, out_range=None, 
           joint_range=None, min_freq=1.0, mass_violating_reactions=True, edge_type='generic', kinetics=None, 
           add_enzyme=False, mod_reg=None, rxn_prob=None, rev_prob=0, ic_params=None, dist_plots=True, net_plots=True):
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
    :param kinetics: Describes the desired rate-laws and parameter ranges. Ultimately defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Ultimately defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Ultimately defaults to ['uniform', 0, 10]
    :param dist_plots: Generate distribution charts.
    :param net_plots: Generate network plots.
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

    if joint_dist and (in_dist is not 'random' or out_dist is not 'random'):
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
            and sum(int(x[0])*int(x[1]) for x in in_dist) != sum(int(x[0])*int(x[1]) for x in out_dist):

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
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            os.makedirs(os.path.join(path, 'antimony'))
            os.makedirs(os.path.join(path, 'networks'))
            os.makedirs(os.path.join(path, 'net_figs'))
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
            os.makedirs(os.path.join(path, 'distributions'))
            os.makedirs(os.path.join(path, 'sbml'))
            os.makedirs(os.path.join(path, 'dist_figs'))

    i = num_existing_models
    while i < num_existing_models + n_models:

        in_samples = []
        out_samples = []
        joint_samples = []

        rl = [None]
        el = [[]]

        rl_failed_count = -1

        while not rl[0]:

            rl_failed_count += 1
            if rl_failed_count == 100:
                print(i, 'failed')
                ant_str = "Network construction failed on this attempt, consider revising your settings."
                anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
                break

            in_samples, out_samples, joint_samples = \
                buildNetworks.generate_samples(n_species, in_dist, out_dist, joint_dist, min_freq, in_range,
                                               out_range, joint_range)

            rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                                      rxn_prob, mod_reg, mass_violating_reactions, edge_type)

        if not rl[0]:
            i += 1
            continue

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
                                    f.write(',' + str(every))
                            f.write(']')
                f.write('\n')

        if net_plots:
            edges = []
            for each in el:
                edges.append(('S' + str(each[0]), 'S' + str(each[1])))

            graph = pydot.Dot(graph_type="digraph")
            graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
            node_ids = set()
            for each in edges:
                node_ids.add(each[0])
                node_ids.add(each[1])
            for each in node_ids:
                graph.add_node(pydot.Node(each))
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))
                
            graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
            graph.write(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.dot'), 
                        format='dot')

            # KEEP THIS FOR NOW
            # output graph to Dot object
            # graph_file = graph.create_dot(prog='dot')
            # graph_file = graph_file.decode('ascii')
            # graph_file = pydot.graph_from_dot_data(graph_file)[0]

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)
        anti_dir = os.path.join(output_dir, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        dist_dir = os.path.join(output_dir, group_name, 'distributions', group_name + '_' + str(i) + '.cvs')

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

        i += 1


def distributions(verbose_exceptions=False, output_dir='models', group_name='test', overwrite=True, n_models=1,
                  n_species=10, out_dist='random', in_dist='random', joint_dist=None, in_range=None, out_range=None,
                  joint_range=None, min_freq=1.0, dist_plots=True):
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
    """

    if joint_dist and (in_dist is not 'random' or out_dist is not 'random'):
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

    i = num_existing_models
    while i < num_existing_models + n_models:

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

        i += 1


def networks(verbose_exceptions=False, directory='models', group_name='test', overwrite=True, n_reactions=None, 
             mass_violating_reactions=True, edge_type='generic', mod_reg=None, rxn_prob=None, rev_prob=0, 
             net_plots=True):
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
    :param rxn_prob: Describes the reaction probabilities. Ultimately defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param net_plots: Generate network plots.
    """

    if directory is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a directory.')

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

    else:
        if os.path.exists(os.path.join(directory, group_name, 'networks')):
            net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'networks'))
                         if os.path.isfile(os.path.join(directory, group_name, 'networks', f))]

        else:
            os.makedirs(os.path.join(directory, group_name, 'networks'))
            os.makedirs(os.path.join(directory, group_name, 'net_figs'))

        net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
        path = os.path.join(directory, group_name, 'distributions')
        dist_files = [fi for fi in os.listdir(path) if os.path.isfile(os.path.join(path, fi)) and fi[-3:] == 'csv']
        dists_list = []
        for item in dist_files:
            dists_list.append([int(item.split('_')[-1].split('.')[0]), item])
        dists_list.sort()

        for dist in dists_list:
            if dist[0] not in net_inds:

                ind = dist[0]

                out_dist = False
                in_dist = False
                joint_dist = False
                out_samples = []
                in_samples = []
                joint_samples = []
                with open(os.path.join(directory, group_name, 'distributions', dist[1])) as dl:
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
                if out_samples and not n_species:
                    for each in out_samples:
                        n_species += each[1]
                if in_samples and not n_species:
                    for each in out_samples:
                        n_species += each[1]
                if joint_samples and not n_species:
                    for each in out_samples:
                        n_species += each[1]

                rl = [None]
                el = [[]]

                rl_failed_count = -1

                while not rl[0]:

                    rl_failed_count += 1
                    if rl_failed_count == 100:

                        break

                    rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species,
                                                              n_reactions, rxn_prob, mod_reg, mass_violating_reactions,
                                                              edge_type)

                if not rl[0]:

                    ant_str = "Network construction failed on this attempt, consider revising your settings."
                    net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(ind) + '.txt')
                    with open(net_dir, 'w') as f:
                        f.write(ant_str)
                else:
                    net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(ind) + '.txt')
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

                if net_plots:
                    edges = []
                    for each in el:
                        edges.append(('S' + str(each[0]), 'S' + str(each[1])))
                    graph = pydot.Dot(graph_type="digraph")
                    graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
                    for each in edges:
                        graph.add_edge(pydot.Edge(each[0], each[1]))
                    graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(ind)
                                                 + '.png'))
                    graph.write(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(ind) + '.dot'),
                                format='dot')


def rate_laws(verbose_exceptions=False, directory='models', group_name='test', overwrite=True, kinetics=None, 
              add_enzyme=False, mod_reg=None, rxn_prob=None, rev_prob=0, ic_params=None):
    """
    Generates a collection of models. This function requires the existence of previously generated networks.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param directory: Directory where files are read and stored.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Ultimately defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param mod_reg: Describes the modifiers. Only valid for modular rate-laws.
    :param rxn_prob: Describes the reaction probabilities. Ultimately defaults to
        [UniUni, BiUni, UniBi, BiBI] = [0.35, 0.3, 0.3, 0.05]
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Ultimately defaults to ['uniform', 0, 10]
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

    if directory is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a directory.')

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

    for net in nets_list:
        if net[0] not in anti_inds:
            ind = net[0]

            reg_check = False
            with open(os.path.join(directory, group_name, 'networks', net[1])) as f:
                if 'a' in f.read():
                    reg_check = True
            with open(os.path.join(directory, group_name, 'networks', net[1])) as f:
                if 's' in f.read():
                    reg_check = True

            if reg_check and 'modular' not in kinetics[0]:
                ant_str = "This model contains regulators that are not accounted for in the selected kinetics."
                anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(ind) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)

            else:
                rl = []
                with open(os.path.join(directory, group_name, 'networks', net[1])) as nl:
                    for i, line in enumerate(nl):
                        if i == 0:
                            rl.append(int(line.strip()))
                        else:
                            rl.append([])
                            line_split = line.strip().split(',')
                            for j, each in enumerate(line_split):
                                if j == 0:
                                    rl[-1].append(int(each))
                                elif j == 5:
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

                anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(ind) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)

                sbml_dir = os.path.join(directory, group_name, 'sbml', group_name + '_' + str(ind) + '.sbml')
                antimony.loadAntimonyString(ant_str)
                sbml = antimony.getSBMLString()
                with open(sbml_dir, 'w') as f:
                    f.write(sbml)
                antimony.clearPreviousLoads()


def linear(verbose_exceptions=False, output_dir='models', group_name='linear', overwrite=True, n_models=1, n_species=10, 
           kinetics=None, add_enzyme=False, rev_prob=0, ic_params=None, net_plots=True):
    """
    Generates a collection of linear models.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param n_models: Number of models to produce.
    :param n_species: Number of species per model.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Ultimately defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Ultimately defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
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

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    for i in range(n_models):

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

            if net_plots:
                edges = []
                for each in el:
                    edges.append(('S' + str(each[0]), 'S' + str(each[1])))

                graph = pydot.Dot(graph_type="digraph")
                graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
                for each in edges:
                    graph.add_edge(pydot.Edge(each[0], each[1]))

                graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) 
                                             + '.png'))
                graph.write(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.dot'),
                            format='dot')

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
           net_plots=True):
    """
    Generates a collection of cyclic models.

    :param verbose_exceptions: Traceback for input errors are suppressed.
    :param output_dir: Output directory.
    :param group_name: Name of the group the models belong too and the directory they will be placed in.
    :param overwrite: Overwrite the models in output_dir/models/group_name.
    :param min_species: Minimum number of species per cycle.
    :param max_species: Maximum number of species per cycle.
    :param n_cycles: Number of cycles per model.
    :param n_models: Number of models to produce.
    :param kinetics: Describes the desired rate-laws and parameter ranges. Ultimately defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Ultimately defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
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

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    for i in range(n_models):

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

            if net_plots:
                edges = []
                for each in el:
                    edges.append(('S' + str(each[0]), 'S' + str(each[1])))

                graph = pydot.Dot(graph_type="digraph", layout="neato")
                graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
                for each in edges:
                    graph.add_edge(pydot.Edge(each[0], each[1]))

                graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
                graph.write(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.dot'),
                            format='neato')

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
             ic_params=None, net_plots=True):
    """
    Generates a collection of branching/converging models from a set of seed nodes.

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
    :param kinetics: Describes the desired rate-laws and parameter ranges. Ultimately defaults to
        ['mass_action', 'loguniform', ['kf', 'kr', 'kc'], [[0.01, 100], [0.01, 100], [0.01, 100]]]
    :param add_enzyme: Add a multiplicative parameter to the rate-law that may be used for perturbation
        analysis.
    :param rev_prob: Describes the probability that a reaction is reversible.
    :param ic_params: Describes the initial condition sampling distributions. Ultimately defaults to ['uniform', 0, 10]
    :param net_plots: Generate network plots.
    """

    if output_dir is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a output_dir.')

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

    net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
    anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
    sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

    if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                        "Consider starting over and overwriting them all.")

    for i in range(n_models):

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

            if net_plots:
                edges = []
                for each in el:
                    edges.append(('S' + str(each[0]), 'S' + str(each[1])))

                graph = pydot.Dot(graph_type="digraph")
                graph.set_node_defaults(color='black', style='filled', fillcolor='#4472C4')
                for each in edges:
                    graph.add_edge(pydot.Edge(each[0], each[1]))

                graph.write_png(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))
                graph.write(os.path.join(output_dir, group_name, 'net_figs', group_name + '_' + str(i) + '.dot'),
                            format='dot')

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
