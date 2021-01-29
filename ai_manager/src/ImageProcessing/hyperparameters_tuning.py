# Libraries
import shutil
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from functools import partial
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from functools import partial

import tempfile
from pytorch_lightning.utilities.cloud_io import load as pl_load

from CNN import CNN
from MyImageModule import MyImageModule

import pytorch_lightning as pl


def train_model_tune(config, data_dir=None, num_epochs=10, num_gpus=1):
    model = CNN(config)
    image_module = MyImageModule(dataset_size=100, batch_size=32)
    image_module.setup()
    logger = TensorBoardLogger('tb_logs', name='Model_prueba_tuning')
    trainer = pl.Trainer(max_epochs=num_epochs,
                         gpus=num_gpus,
                         logger=logger,
                         deterministic=True,
                         callbacks=[TuneReportCallback({"loss": "ptl/val_loss",
                                                        "mean_accuracy": "ptl/val_accuracy"}, on="validation_end")])
    trainer.fit(model, datamodule=image_module)


def tune_model(num_samples=10, num_epochs=10, gpus_per_trial=1):
    # config the parameters
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # add the scheduler
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=1,
                              reduction_factor=2)

    reporter = CLIReporter(parameter_columns=["lr", "batch_size"],
                           metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_model_tune,
            data_dir='pruebas_tuning',
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    tune.run(
        partial(train_model_tune, epochs=10, gpus=0),
        config=config,
        num_samples=10)

    print("Best hyperparameters found were: ", analysis.best_config)

    shutil.rmtree('pruebas_tuning')


# Code specially prepared to finetune parameters in the desired models

if __name__ == '__main__':
    tune_model(num_samples=10, num_epochs=10, gpus_per_trial=1)

    # TODO : Check por qué no funciona y ver si merece la pena
