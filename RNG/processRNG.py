
import sys
import buildNetworks
import os
import shutil
import glob
import antimony
import matplotlib.pyplot as plt
import numpy as np
import pydot


def generate_distributions(verbose_exceptions=False, group_name=None, n_models=None, n_species=None, in_dist='random',
                           out_dist='random', output_dir=None, overwrite=False, joint_dist=None, in_range=None,
                           out_range=None, joint_range=None, min_node_deg=1.0, dist_plots=False):

    if group_name is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

    if n_models is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_models (the number of models).')

    if n_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_species (the number of species).')

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

    num_existing_models = 0
    if output_dir:
        path = os.path.join(output_dir, 'models', group_name)
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

    else:
        path = os.path.join('models', group_name)
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
            n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range, out_range, joint_range)

        if output_dir:
            dist_dir = os.path.join(output_dir, 'models', group_name, 'distributions', group_name + '_' + str(i)
                                    + '.csv')
        else:
            dist_dir = os.path.join('models', group_name, 'distributions', group_name + '_' + str(i) + '.csv')

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
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_in'
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
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
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

                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out_in'
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
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_joint'
                                         + '.png'))
                plt.close()

        i += 1


def generate_networks(verbose_exceptions=False, group_name='', n_reactions=None, overwrite=False, rxn_prob=None,
                      rev_prob=False, mod_reg=None, mass_violating_reactions=True, directory='', edge_type='generic',
                      reaction_type=None, net_plots=False):

    if group_name is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

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

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('Your reversibility probability is not between 0 and 1')

    net_files = []
    if directory:
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
                                                              edge_type, reaction_type)

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
                    else:
                        graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(ind)
                                                     + '.png'))


def generate_models(verbose_exceptions=False, group_name='', add_enzyme=False, kinetics=None, overwrite=False,
                    rxn_prob=None, rev_prob=False, ic_params=None, mod_reg=None, directory=''):

    if kinetics is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if group_name is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

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

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('Your reversibility probability is not between 0 and 1')

    anti_files = []
    sbml_files = []
    if directory:
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


def run(verbose_exceptions=False, group_name=None, add_enzyme=False, n_models=None,
        n_species=None, n_reactions=None, kinetics=None, in_dist='random', out_dist='random',
        output_dir=None, overwrite=False, rxn_prob=None, rev_prob=False, joint_dist=None,
        in_range=None, out_range=None, joint_range=None, min_node_deg=1.0, ic_params=None,
        mod_reg=None, mass_violating_reactions=True, dist_plots=False, net_plots=False,
        edge_type='generic', reaction_type=None):

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if group_name is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')
    if n_models is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_models (the number of models).')
    if n_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_species (the number of species).')
    if kinetics is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

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

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
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

    num_existing_models = 0
    if output_dir:
        path = os.path.join(output_dir, 'models', group_name, '')
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

    else:
        path = os.path.join('models', group_name, '')
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
                if output_dir:
                    anti_dir = os.path.join(output_dir, 'models', group_name, 'antimony',
                                            group_name + '_' + str(i) + '.txt')
                else:
                    anti_dir = os.path.join('models', group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
                break

            in_samples, out_samples, joint_samples = \
                buildNetworks.generate_samples(n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range,
                                               out_range, joint_range)

            rl, el = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                                      rxn_prob, mod_reg, mass_violating_reactions, edge_type,
                                                      reaction_type)

        if not rl[0]:
            i += 1
            continue

        if output_dir:
            net_dir = os.path.join(output_dir, 'models', group_name, 'networks', group_name + '_' + str(i)
                                   + '.csv')
        else:
            net_dir = os.path.join('models', group_name, 'networks', group_name + '_' + str(i) + '.csv')
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
            for each in edges:
                graph.add_edge(pydot.Edge(each[0], each[1]))

            if output_dir:
                graph.write_png(os.path.join(output_dir, 'models', group_name, 'net_figs', group_name + '_' + str(i)
                                             + '.png'))
            else:
                graph.write_png(os.path.join('models', group_name, 'net_figs', group_name + '_' + str(i) + '.png'))

        ant_str = buildNetworks.get_antimony_script(rl, ic_params, kinetics, rev_prob, add_enzyme)
        if output_dir:
            anti_dir = os.path.join(output_dir, 'models', group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        else:
            anti_dir = os.path.join('models', group_name, 'antimony', group_name + '_' + str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(ant_str)

        if output_dir:
            dist_dir = os.path.join(output_dir, 'models', group_name, 'distributions', group_name + '_' + str(i)
                                    + '.cvs')
        else:
            dist_dir = os.path.join('models', group_name, 'distributions', group_name + '_' + str(i) + '.cvs')

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

            if in_samples:
                x = [dist_ind[0] for dist_ind in in_samples]
                y = [dist_ind[1] for dist_ind in in_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("Out Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' out edges')
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_in'
                                         + '.png'))
                plt.close()

            if out_samples:
                x = [dist_ind[0] for dist_ind in out_samples]
                y = [dist_ind[1] for dist_ind in out_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("In Degree")
                plt.ylabel("Number of Nodes")
                plt.xticks(x)
                plt.title(group_name + '_' + str(i) + ' in edges')
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
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

                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out_in'
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
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_joint'
                                         + '.png'))
                plt.close()

        if output_dir:
            sbml_dir = os.path.join(output_dir, 'models', group_name, 'sbml', group_name + '_' + str(i) + '.sbml')
        else:
            sbml_dir = os.path.join('models', group_name, 'sbml', group_name + '_' + str(i) + '.sbml')

        antimony.loadAntimonyString(ant_str)
        sbml = antimony.getSBMLString()
        with open(sbml_dir, 'w') as f:
            f.write(sbml)
        antimony.clearPreviousLoads()

        i += 1


def simple_linear(verbose_exceptions=False, group_name='', add_enzyme=False, n_species=None, n_models=None,
                  kinetics=None, overwrite=False, rev_prob=False, ic_params=None, directory='', net_plots=False):

    if group_name is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

    if n_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_species (the number of species).')

    if n_models is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_models (the number of models).')

    if directory is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a directory.')

    if kinetics is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('Your reversibility probability is not between 0 and 1')

    if directory:
        net_files = []
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

            if os.path.exists(os.path.join(directory, group_name, 'networks')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'networks'))
                             if os.path.isfile(os.path.join(directory, group_name, 'networks', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'networks'))

            if os.path.exists(os.path.join(directory, group_name, 'net_figs')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'net_figs'))
                             if os.path.isfile(os.path.join(directory, group_name, 'net_figs', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'net_figs'))

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

            print()
            for each in rl:
                print(each)

            if not rl[0]:

                ant_str = "Network construction failed on this attempt, consider revising your settings."
                anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
            else:

                net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(i) + '.csv')
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

                    graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))

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


def simple_cyclic(verbose_exceptions=False, group_name='', add_enzyme=False, min_species=None, max_species=None,
                  linkage=1, n_cycles=1, n_models=None, kinetics=None, overwrite=False, rev_prob=False,
                  ic_params=None, directory='', net_plots=False):

    if group_name is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

    if min_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide min_species (the minimum number of species per cycle).')

    if max_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide max_species (the maximum number of species per cycle).')

    if n_models is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_models (the number of models).')

    if directory is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a directory.')

    if kinetics is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('Your reversibility probability is not between 0 and 1')

    if directory:
        net_files = []
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

            if os.path.exists(os.path.join(directory, group_name, 'networks')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'networks'))
                             if os.path.isfile(os.path.join(directory, group_name, 'networks', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'networks'))

            if os.path.exists(os.path.join(directory, group_name, 'net_figs')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'net_figs'))
                             if os.path.isfile(os.path.join(directory, group_name, 'net_figs', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'net_figs'))

        net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
        anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
        sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

        if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                            "Consider starting over and overwriting them all.")

        for i in range(n_models):

            rl, el = buildNetworks.generate_simple_cyclic(i, min_species, max_species, linkage, n_cycles)

            print()
            for each in rl:
                print(each)

            if not rl[0]:

                ant_str = "Network construction failed on this attempt, consider revising your settings."
                anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
            else:
                print(i)
                net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(i) + '.csv')
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

                    graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))

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


def linear(verbose_exceptions=False, group_name='', add_enzyme=False, n_species=None, n_reactions=None, n_models=None,
           kinetics=None, overwrite=False, rxn_prob=None, rev_prob=False, ic_params=None, mod_reg=None,
           mass_violating_reactions=True, directory='', edge_type='generic', reaction_type=None,
           mod_species_as_linear=True, strict_linear=False, net_plots=False):

    if 'modular' not in kinetics[0] and mod_reg is not None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Regulators are relevant only to modular kinetics.\n'
                        'Please reset the run with appropriate parameters.')

    if group_name is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a group_name.')

    if n_species is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_species (the number of species).')

    if n_models is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide n_models (the number of models).')

    if directory is '':
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide a directory.')

    if kinetics is None:
        if not verbose_exceptions:
            sys.tracebacklimit = 0
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

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

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception('Your reversibility probability is not between 0 and 1')

    if directory:
        net_files = []
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

            if os.path.exists(os.path.join(directory, group_name, 'networks')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'networks'))
                             if os.path.isfile(os.path.join(directory, group_name, 'networks', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'networks'))

            if os.path.exists(os.path.join(directory, group_name, 'net_figs')):
                net_files = [f for f in os.listdir(os.path.join(directory, group_name, 'net_figs'))
                             if os.path.isfile(os.path.join(directory, group_name, 'net_figs', f))]
            else:
                os.makedirs(os.path.join(directory, group_name, 'net_figs'))

        net_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in net_files]
        anti_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in anti_files]
        sbml_inds = [int(nf.split('_')[-1].split('.')[0]) for nf in sbml_files]

        if set(net_inds) != set(anti_inds) or set(anti_inds) != set(sbml_inds) or set(net_inds) != set(sbml_inds):
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("There exists a discrepancy between the network, antimony, and sbml files.\n"
                            "Consider starting over and overwriting them all.")

        for i in range(n_models):

            rl, el = buildNetworks.generate_linear(n_species, n_reactions, rxn_prob, mod_reg, mass_violating_reactions,
                                                   edge_type, reaction_type, mod_species_as_linear, strict_linear)

            print()
            for each in rl:
                print(each)

            if not rl[0]:

                ant_str = "Network construction failed on this attempt, consider revising your settings."
                anti_dir = os.path.join(directory, group_name, 'antimony', group_name + '_' + str(i) + '.txt')
                with open(anti_dir, 'w') as f:
                    f.write(ant_str)
            else:

                net_dir = os.path.join(directory, group_name, 'networks', group_name + '_' + str(i) + '.csv')
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

                    graph.write_png(os.path.join(directory, group_name, 'net_figs', group_name + '_' + str(i) + '.png'))

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
