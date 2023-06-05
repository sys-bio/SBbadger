
from SBbadger import generate

if __name__ == "__main__":

    generate.cyclic(
        n_cycles=3,
        min_species=10,
        max_species=20,
        net_plots=True,
        net_layout="neato",
    )
