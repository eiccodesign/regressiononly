from multiprocessing        import Process, Queue, Manager, set_start_method
from typing                 import Generator, Tuple

import gzip
import numpy                as np
import compress_pickle      as pickle

from config_loader          import ConfigLoader
from exceptions             import DataGeneratorException


class DataGenerator:
    
    def __init__(self, config_loader: ConfigLoader, file_list: list, folder_name: str):
        config = self.config = config_loader
        self.file_list = file_list
        self.folder_name = folder_name
        self.num_files = len(file_list)

        self.processes = []

        data_dir_path = config.OUTPUT_DIR_PATH.resolve() / self.folder_name
        data_dir_path.mkdir(parents=True, exist_ok=True)

        self.processed_file_list = [
            f"{data_dir_path}/data_{file:03d}.p.gz" 
            for file in range(self.num_files)
        ]


    
    """
    The below functions are generator function manager and process to return
    data to the model during training
    """

    def generator(self) -> Generator:
        config = self.config

        batch_queue = Queue(2 * config.NUM_PROCESSES)

        for i in range(config.NUM_PROCESSES):
            process = Process(
                target=self._generator_process, 
                args=(i, batch_queue), 
                daemon=True
            )
            process.start()
            self.processes.append(process)

        while self.processes_are_running() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.001)
            except:
                continue
            yield batch

        for process in self.processes:
            process.join()


    def _generator_process(self, worker_id, batch_queue):
        config = self.config

        batch_graphs = []
        batch_targets = []
        batch_meta = []

        file_num = worker_id

        while file_num < self.num_files:
        
            with gzip.open(self.processed_file_list[file_num], 'rb') as file:
                file_data = pickle.load(file)

            for index in range(len(file_data)):
                batch_graphs.append(file_data[index][0])
                batch_targets.append(file_data[index][1])
                batch_meta.append(file_data[index][2])

                if len(batch_graphs) == config.BATCH_SIZE:
                    batch_queue.put((batch_graphs, batch_targets, batch_meta))

                    batch_graphs = []
                    batch_targets = []
                    batch_meta = []

            file_num += config.NUM_PROCESSES

        if len(batch_graphs) > 0:
            batch_queue.put((batch_graphs, batch_targets, batch_meta))


    def processes_are_running(self) -> bool:
        for process in self.processes:
            if process.is_alive(): 
                return True
        return False


    def kill_processes(self):
        for process in self.processes:
            process.kill()
        self.processes = []

