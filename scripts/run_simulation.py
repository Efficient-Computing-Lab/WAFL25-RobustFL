import random
import time
import warnings

from flwr.simulation import run_simulation
from loguru import logger

from src.client_app import get_client_fn
from src.server_app import get_server_fn
from src.settings import settings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def simulate() -> None:
    try:
        start_time = time.time()
        random.seed(settings.general.random_seed)
        malicious_ids = random.sample(range(settings.client.num_clients), settings.attack.num_malicious_clients)
        # Start simulation
        run_simulation(
            server_app=get_server_fn(),
            client_app=get_client_fn(malicious_ids),
            num_supernodes=settings.client.num_clients,
            backend_config=settings.backend.__dict__,
        )
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time.__round__(2)} sec")

    except Exception as e:
        logger.error(f"Error in {settings.model.name} Federated Scenario, processing: {str(e)}")
        raise
