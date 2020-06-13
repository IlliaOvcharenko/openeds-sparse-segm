import warnings
warnings.filterwarnings("ignore")

from fire import Fire
from tqdm.cli import tqdm
from src.config import ConfigParser
from src.train import fit
from src.utils import set_global_seed


def run_train(config_filenames, **kwargs):
    set_global_seed(42)

    config_filenames = config_filenames.split(",")
    for config_filename in config_filenames:
        try:
            print("Current config:", config_filename)
            config_parser = ConfigParser(config_filename, False, **kwargs)
            config = config_parser()

            print("Training stage")
            fit(
                config.model,
                config.device,
                config.criterion,
                config.optimizer,
                config.scheduler,
                config.dataloaders,
                metrics=config.metrics,
                metrics_monitor=config.metrics_monitor,
                metrics_lower_is_better=config.metrics_lower_is_better,
                metrics_initial_best_val=config.metrics_initial_best_val,
                writer=config.writer,
                writer_add_visualizations=config.writer_add_visualizations,
                model_folder=config.model_folder,
                model_name=config.model_name,
                remove_previous_ckpt=config.remove_previous_ckpt,
                epochs=config.epochs,
                initial_epoch=config.initial_epoch,
                accumulation_steps=config.accumulation_steps,
            )
        except Exception as exp:
            print(f"Config {config_filename} error: {exp}")


if __name__ == "__main__":
    Fire(run_train)
