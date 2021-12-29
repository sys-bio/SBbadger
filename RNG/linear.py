
import processRNG

if __name__ == "__main__":

    processRNG.linear(

        directory='models',
        group_name='test_group',
        overwrite=True,
        n_models=1,
        n_species=20,
        # kinetics=['mass_action', 'trivial', ['kf', 'kr', 'kc']],
        kinetics=['modular_CM', 'trivial', ['kf', 'kr', 'km', 'mol']],
        rxn_prob=[0.0, 0.0, 0.0, 1.0],
        mod_reg=[[0.0, 1.0, 0.0, 0.0], 1.0, 0.0]
    )
