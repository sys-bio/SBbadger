
import generate

if __name__ == "__main__":

    generate.branched(
        seeds=3,
        path_probs=[.2, .6, .2],
        tips=True,
        net_plots=True,
    )
