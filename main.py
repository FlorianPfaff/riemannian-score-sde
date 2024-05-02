import os
import hydra

@hydra.main(config_path="config", config_name="main", version_base='1.1')
def main(cfg):
    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"

    from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
