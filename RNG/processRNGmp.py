
import sys
import buildNetworks
import os
import shutil
import glob
import antimony
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count


def run_generate_distributions(i, group_name=None, n_species=None, in_dist='random',
                               out_dist='random', output_dir=None, joint_dist=None, in_range=None,
                               out_range=None, joint_range=None, min_node_deg=1.0, plots=False):


    print(i)

    in_samples, out_samples, joint_samples = buildNetworks.generate_samples(
        n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range, out_range, joint_range)

    # print('in samples')
    # for each in in_samples:
    #     print(each)
    # print()
    # print('out samples')
    # for each in out_samples:
    #     print(each)
    # print()
    # print('joint samples')
    # for each in joint_samples:
    #     print(each)

    if output_dir:
        dist_dir = os.path.join(output_dir, 'models', group_name, 'distributions', group_name + '_' + str(i)
                                + '.cvs')
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
    if plots:
        if in_samples:
            x = [dist_ind[0] for dist_ind in in_samples]
            y = [dist_ind[1] for dist_ind in in_samples]
            plt.figure()
            plt.bar(x, y)
            plt.xlabel("Out Degree")
            plt.ylabel("Number of Nodes")
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
            plt.title(group_name + '_' + str(i) + ' in edges')
            plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
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



def generate_distributions(verbose_exceptions=False, group_name=None, n_models=None, n_species=None, in_dist='random',
                           out_dist='random', output_dir=None, overwrite=False, joint_dist=None, in_range=None,
                           out_range=None, joint_range=None, min_node_deg=1.0, plots=False, n_cpus=cpu_count()):

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

    num_existing_models = 0
    if output_dir:
        path = os.path.join(output_dir, 'models', group_name)
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
            else:
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            if os.path.exists(os.path.join(path)):
                gd = glob.glob(os.path.join(path, 'distributions', '*'))
                num_existing_models = len(gd)
            else:
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))

    else:
        path = os.path.join('models', group_name)
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
            else:
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            if os.path.exists(os.path.join(path)):
                gd = glob.glob(os.path.join(path, 'distributions', '*'))
                num_existing_models = len(gd)
            else:
                os.makedirs(os.path.join(path, 'distributions'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))

    args_list = [(i, group_name, n_models, n_species, in_dist, out_dist, output_dir, joint_dist, in_range, out_range,
                  joint_range, min_node_deg, plots) for i in range(num_existing_models, n_models)]

    pool = Pool(n_cpus)
    pool.starmap(run_generate_distributions, args_list)
    pool.close()


def run_generate_networks(i, dists_list, group_name='', add_enzyme=False, n_reactions=None, kinetics=None,
                          rxn_prob=None, rev_prob=False, ic_params=None, mod_reg=None, mass_violating_reactions=True,
                          directory='', edge_type='generic'):

        print(i)
        print(dists_list)
        print(group_name)
        print(add_enzyme)
        print(n_reactions)
        print(kinetics)
        print(rxn_prob)
        print(rev_prob)
        print(ic_params)
        print(mod_reg)
        print(mass_violating_reactions)
        print(directory)
        # quit()


        failed_attempts = 0

        print(i)

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

        # print(out_samples)
        # print(in_samples)
        # print(joint_samples)

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

        rl = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions,
                                              rxn_prob, mod_reg, mass_violating_reactions, edge_type)

        if not rl:
            failed_attempts += 1
            if failed_attempts == 1000:
                sys.tracebacklimit = 0
                raise Exception("There have been 1000 consecutive failed attempts to randomly construct a network.\n"
                                "Consider revising your settings.")

        else:
            failed_attempts = 0

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


def generate_networks(verbose_exceptions=False, group_name='', add_enzyme=False, n_reactions=None,
                      kinetics=None, overwrite=False, rxn_prob=None, rev_prob=False, ic_params=None,
                      mod_reg=None, mass_violating_reactions=True, directory='', edge_type='generic',
                      n_cpus=cpu_count()):

    if kinetics[0] != 'modular' and mod_reg is not None:
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

    n_antimony = 0
    n_sbml = 0

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
                gd = glob.glob(os.path.join(directory, group_name, 'antimony', '*'))
                n_antimony = len(gd)
            else:
                os.makedirs(os.path.join(directory, group_name, 'antimony'))

            if os.path.exists(os.path.join(directory, group_name, 'sbml')):
                gd = glob.glob(os.path.join(directory, group_name, 'sbml', '*'))
                n_sbml = len(gd)
            else:
                os.makedirs(os.path.join(directory, group_name, 'sbml'))

        num_existing_models = 0
        if n_antimony == n_sbml:
            num_existing_models = n_antimony
        else:
            if not verbose_exceptions:
                sys.tracebacklimit = 0
            raise Exception("There exists a discrepancy between the number of antimony and sbml files.\n"
                            "Consider starting over and replacing them all.")

        path = os.path.join(directory, group_name, 'distributions')
        dist_files = [fi for fi in os.listdir(path) if os.path.isfile(os.path.join(path, fi)) and fi[-3:] == 'csv']

        dists_list = []
        for item in dist_files:
            dists_list.append((int(item.split('_')[-1].split('.')[0]), item))
        dists_list.sort()

        args_list = [(i, dists_list, group_name, add_enzyme, n_reactions, kinetics, rxn_prob, rev_prob, ic_params,
                      mod_reg, mass_violating_reactions, directory)
                     for i in range(num_existing_models, len(dists_list))]

        pool = Pool(n_cpus)
        pool.starmap(run_generate_networks, args_list)
        pool.close()


def run_generate_dists_networks(i, group_name=None, add_enzyme=False, n_species=None, n_reactions=None, kinetics=None,
                                in_dist='random', out_dist='random', output_dir=None, rxn_prob=None, rev_prob=False,
                                joint_dist=None, in_range=None, out_range=None, joint_range=None, min_node_deg=1.0,
                                ic_params=None, mod_reg=None, mass_violating_reactions=True, plots=False,
                                edge_type='generic'):

    print(i)

    in_samples, out_samples, joint_samples = \
        buildNetworks.generate_samples(n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range,
                                       out_range, joint_range)

    rl = buildNetworks.generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions, rxn_prob,
                                          mod_reg, mass_violating_reactions, edge_type)

    if not rl:
        pass

    else:

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
        if plots:
            if in_samples:
                x = [dist_ind[0] for dist_ind in in_samples]
                y = [dist_ind[1] for dist_ind in in_samples]
                plt.figure()
                plt.bar(x, y)
                plt.xlabel("Out Degree")
                plt.ylabel("Number of Nodes")
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
                plt.title(group_name + '_' + str(i) + ' in edges')
                plt.savefig(os.path.join('models', group_name, 'dist_figs', group_name + '_' + str(i) + '_out'
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


def generate_dists_networks(verbose_exceptions=False, group_name=None, add_enzyme=False, n_models=None, n_species=None,
                            n_reactions=None, kinetics=None, in_dist='random', out_dist='random', output_dir=None,
                            overwrite=False, rxn_prob=None, rev_prob=False, joint_dist=None, in_range=None,
                            out_range=None, joint_range=None, min_node_deg=1.0, ic_params=None, mod_reg=None,
                            mass_violating_reactions=True, plots=False, edge_type='generic', n_cpus=cpu_count()):

    if kinetics[0] != 'modular' and mod_reg is not None:
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
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            if os.path.exists(os.path.join(path)):
                gd = glob.glob(os.path.join(path, 'antimony', '*'))
                num_existing_models = len(gd)
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))

    else:
        path = os.path.join('models', group_name, '')
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))
        else:
            if os.path.exists(os.path.join(path)):
                gd = glob.glob(os.path.join(path, 'antimony', '*'))
                num_existing_models = len(gd)
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
                if plots:
                    os.makedirs(os.path.join(path, 'dist_figs'))

    args_list = [(i, group_name, add_enzyme, n_species, n_reactions, kinetics, in_dist, out_dist, output_dir,
                 rxn_prob, rev_prob, joint_dist, in_range, out_range, joint_range, min_node_deg, ic_params,
                 mod_reg, mass_violating_reactions, plots, edge_type) for i in range(num_existing_models, n_models)]

    pool = Pool(n_cpus)
    pool.starmap(run_generate_dists_networks, args_list)
    pool.close()
