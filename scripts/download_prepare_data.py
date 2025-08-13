import argparse
import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Dict, Type

import mlflow
import pandas as pd
from datasets import Dataset, load_dataset

from chimera.utils.mlflow_utils import log_reproducibility_passport


@dataclass
class DataPaths:
    """
    Manages directory and file paths related to data processing and reporting.

    This class is used to encapsulate various paths required for processing data
    and generating reports. It provides an organized way of managing file structure
    associated with a project.

    :ivar data_dir: The root directory where raw data is stored.
    :type data_dir: Path
    :ivar processed_dir: The directory where processed data is stored.
    :type processed_dir: Path
    :ivar reports_dir: The directory where reports are stored.
    :type reports_dir: Path
    :ivar full_parquet_path: The file path for the full dataset in Parquet format.
    :type full_parquet_path: Path
    :ivar small_parquet_path: The file path for the smaller subset of the dataset in Parquet format.
    :type small_parquet_path: Path
    :ivar checksums_path: The file path for the checksums JSON file used for verification.
    :type checksums_path: Path
    """

    data_dir: Path
    processed_dir: Path
    reports_dir: Path
    full_parquet_path: Path
    small_parquet_path: Path
    checksums_path: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DataPaths":
        """
        Creates an instance of the class from a given argparse namespace. This method
        extracts required paths from the provided namespace object and uses them to
        construct a new class instance.

        :param args: Argument namespace containing the required data directory path and other
            configurations necessary for constructing the paths.
        :type args: argparse.Namespace
        :return: A new instance of the class initialized with paths derived from the given arguments.
        :rtype: DataPaths
        """
        data_dir = Path(args.data_dir)
        processed_dir = data_dir / "processed"
        reports_dir = Path("reports")

        return cls(
            data_dir=data_dir,
            processed_dir=processed_dir,
            reports_dir=reports_dir,
            full_parquet_path=processed_dir / "flickr8k_full.parquet",
            small_parquet_path=processed_dir / "flickr8k_small.parquet",
            checksums_path=reports_dir / "checksums.json",
        )


class DatasetProcessor(ABC):
    """
    An abstract base class for processing datasets.

    The purpose of this class is to define a common interface for processing datasets.
    Subclasses inheriting from this class must implement the abstract method `process`,
    which is intended to handle specific data processing tasks for a training dataset.
    This design promotes consistency and extensibility when implementing various data
    processing functionalities for datasets.
    """

    @abstractmethod
    def process(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the provided training dataframe to perform specific data processing tasks.

        This method is an abstract method that requires subclasses to implement its functionality,
        which typically transforms or manipulates the training data as per the desired behavior.
        The implementation is expected to output a transformed dataframe.

        :param train_df: A pandas DataFrame containing the input training data to be processed.
        :type train_df: pd.DataFrame
        :return: A pandas DataFrame containing the processed output data based on the implementation.
        :rtype: pd.DataFrame
        """
        pass


class JapaneseFlickr8kProcessor(DatasetProcessor):
    """
    Processes the Japanese Flickr8k dataset into the required format.

    This class is designed to handle and process the Japanese Flickr8k dataset.
    It extracts relevant columns (such as image ID, image, and captions) from
    the provided DataFrame, preparing the dataset for further use in training
    or evaluation tasks.

    It ensures data is extracted in a consistent structure to aid downstream
    operations in machine learning or data analysis workflows.
    """

    def process(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the provided training DataFrame to extract specific columns.

        This method filters the input DataFrame to retain only the columns
        "image_id", "image", and "captions". It is primarily used to restructure
        or preprocess the data for further analysis or machine learning tasks.

        :param train_df: A DataFrame containing various columns including "image_id",
            "image", and "captions".
        :type train_df: pd.DataFrame
        :return: A DataFrame containing only the "image_id", "image", and "captions"
            columns extracted from the input DataFrame.
        :rtype: pd.DataFrame
        """
        return train_df[["image_id", "image", "captions"]]


class StandardFlickr8kProcessor(DatasetProcessor):
    """
    Processes the Flickr8k dataset for image captioning tasks.

    This class provides a method to transform and preprocess the dataset by
    extracting only the relevant columns and renaming them for further use.
    """

    def _find_column_mapping(self, df: pd.DataFrame) -> Dict:
        columns = df.columns.tolist()
        mapping = {}
        image_id_candidates = [
            "image_path",
            "filename",
            "image_filename",
            "file_name",
            "image_id",
        ]
        caption_candidates = ["caption", "text", "description", "sentence", "captions"]
        image_data_candidates = ["image", "img", "raw_image", "picture"]

        for candidate in image_id_candidates:
            if candidate in columns:
                mapping["image_id"] = candidate
                break

        for candidate in caption_candidates:
            if candidate in columns:
                mapping["caption"] = candidate
                break

        for candidate in image_data_candidates:
            if candidate in columns:
                mapping["image"] = candidate
                break

        # Si no encontramos columna de image_id pero tenemos image_filename, usarla
        if "image_id" not in mapping and "image_filename" in columns:
            mapping["image_id"] = "image_filename"

        return mapping

    def process(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the training DataFrame to select and rename specific columns. This function is used to
        extract necessary data for further operations and ensures the correct naming convention is followed.

        :param train_df: The input DataFrame containing training data.
        :type train_df: pd.DataFrame
        :return: A DataFrame containing the processed data with image_id, image, and caption columns.
        :rtype: pd.DataFrame
        """
        print(f"Available columns in dataset: {list(train_df.columns)}")
        column_mapping = self._find_column_mapping(train_df)

        if "image_id" not in column_mapping:
            raise ValueError(
                f"Could not find image_id column. Available columns: {list(train_df.columns)}"
            )

        if "caption" not in column_mapping:
            raise ValueError(
                f"Could not find caption column. Available columns: {list(train_df.columns)}"
            )

        print(f"Using column mapping: {column_mapping}")

        # Prepare result dataframe
        result_data = []

        # Process each row
        for _, row in train_df.iterrows():
            image_id = row[column_mapping["image_id"]]
            caption = row[column_mapping["caption"]]
            image_data = (
                row[column_mapping["image"]] if "image" in column_mapping else None
            )
            if image_data is not None:
                if isinstance(image_data, dict):
                    image_data = image_data["bytes"]
                elif isinstance(image_data, list):
                    image_data = image_data[0]
                else:
                    image_data = image_data

            entry = {
                "image_id": image_id,
                "caption": caption,
                "image": image_data,
            }

            result_data.append(entry)

        return pd.DataFrame(result_data)


class GenericFlickr8kProcessor(DatasetProcessor):
    """
    Processes datasets from the Flickr8k dataset.

    This class is designed to process a dataset specific to the Flickr8k dataset,
    ensuring the required columns are extracted and formatted correctly for further
    use. It provides necessary transformations to maintain compatibility with
    downstream processing.

    """

    def process(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame to retain a subset of specific columns and handle
        the transformation of the 'captions' column if applicable. It ensures that the
        resulting DataFrame contains relevant columns necessary for further operations.

        :param train_df: Input DataFrame containing data to process
        :type train_df: pd.DataFrame
        :return: Processed DataFrame with selected columns and modified content
        :rtype: pd.DataFrame
        """
        columns = ["image_id", "image", "caption"]
        available_columns = [col for col in columns if col in train_df.columns]
        df = train_df[available_columns]

        if "captions" in train_df.columns and "caption" not in df.columns:
            df["caption"] = train_df["captions"]

        return df


class DatasetProcessorFactory:
    """
    Factory class for creating dataset processors.

    This class provides a mechanism to return specific dataset processor instances
    based on the dataset name supplied. It helps in abstracting the selection process
    of different dataset processors for different datasets.

    """

    @staticmethod
    def create_processor(dataset_name: str) -> DatasetProcessor:
        """
        Creates a dataset processor based on the given dataset name.

        This method checks the dataset name and returns the corresponding
        dataset processor object. If the dataset name matches specific
        keywords, it will return a processor tailored for that dataset.
        Otherwise, a generic processor is returned.

        :param dataset_name: Name of the dataset to determine the corresponding
            processor.
        :type dataset_name: str
        :return: A dataset processor instance suitable for the specified
            dataset name.
        :rtype: DatasetProcessor
        """
        if dataset_name == "rinna/japanese-flickr8k":
            return JapaneseFlickr8kProcessor()
        if "tsystems/flickr8k" in dataset_name or "Naveengo/flickr8k" in dataset_name:
            return StandardFlickr8kProcessor()
        return GenericFlickr8kProcessor()


class FileSystemManager:
    """
    Manages file system operations, such as creating directories and checking file
    existence.

    This class provides utility methods to handle directory creation and file
    existence checks. It simplifies interactions with the underlying file system
    and ensures that required directories or files exist as needed.

    :ivar data_dir: Path to the directory where raw data is stored.
    :type data_dir: str
    :ivar processed_dir: Path to the directory where processed data is stored.
    :type processed_dir: str
    :ivar reports_dir: Path to the directory where report files are stored.
    :type reports_dir: str
    """

    @staticmethod
    def create_directories(paths: DataPaths) -> None:
        """
        Creates directories specified within the given paths object.

        This static method ensures the existence of the specified directories
        for storing data, processed data, and reports. If a directory already
        exists, it will not raise an exception, as it uses the `exist_ok` flag.

        :param paths: An instance of `DataPaths` containing directory paths
                      for data, processed data, and reports.
        :type paths: DataPaths
        :return: None
        """
        dirs = [paths.data_dir, paths.processed_dir, paths.reports_dir]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def files_exist(paths: DataPaths, force: bool = False) -> bool:
        """
        Checks if specified files exist in the paths provided. The method confirms the existence
        of the full and small parquet paths. If the `force` parameter is set to True, the method
        will disregard file presence checks and return False.

        :param paths: Object containing paths to check for file existence.
        :type paths: DataPaths
        :param force: Flag to force ignore file exists checks, default is False.
        :type force: bool
        :return: True if files exist and the `force` flag is False, False otherwise.
        :rtype: bool
        """
        return (
            paths.full_parquet_path.exists()
            and paths.small_parquet_path.exists()
            and not force
        )


class ChecksumManager:
    """
    Handles checksum calculations and management for specified files.

    This class provides static methods to calculate checksums for files
    and to generate a checksum file for processed datasets. It is useful
    for verifying file integrity after processing or downloading.

    """

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """
        Calculate and return the SHA256 checksum of a given file.

        This method reads the file in chunks and computes the checksum to ensure
        memory efficiency, especially for large files.

        :param file_path: Path to the file for which the checksum is to be calculated.
        :type file_path: Path
        :return: The SHA256 checksum as a hexadecimal string.
        :rtype: str
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def generate_checksums(paths: DataPaths) -> None:
        """
        Generate checksums for specified parquet files and save them to a JSON file.

        Detailed checksums are calculated for "flickr8k_full.parquet" and
        "flickr8k_small.parquet" using the `calculate_checksum` method of the
        `ChecksumManager`. The resulting checksums are stored in a dictionary and
        written to a file in JSON format.

        :param paths: The DataPaths object containing paths to the full parquet file,
            small parquet file, and the destination path where checksums will be saved.
        :type paths: DataPaths
        :return: None
        """
        print("Generando checksums...")
        checksums = {
            "flickr8k_full.parquet": ChecksumManager.calculate_checksum(
                file_path=paths.full_parquet_path
            ),
            "flickr8k_small.parquet": ChecksumManager.calculate_checksum(
                file_path=paths.small_parquet_path
            ),
        }

        with open(paths.checksums_path, "w") as f:
            json.dump(checksums, f, indent=2)

        print(f"Checksums guardados en {paths.checksums_path}")


class MLflowLogger:
    """
    Responsible for managing MLflow experiments and logging relevant information
    during the data preparation stages.

    This class simplifies interaction with MLflow's experiment and run management
    systems, providing a context manager interface for starting and ending runs
    while also facilitating logging of dataset-specific parameters and tags for
    reproducibility and auditing purposes.

    :ivar experiment_name: Name of the MLflow experiment to set for the current run.
    :type experiment_name: str
    """

    def __init__(self, experiment_name: str = "Data Preparation"):
        """
        Initializes an instance of the class with the provided experiment name or the default value.

        :param experiment_name: The name of the experiment. Defaults to "Data Preparation".
        :type experiment_name: str
        """
        self.experiment_name = experiment_name

    def __enter__(self) -> "MLflowLogger":
        """
        Sets up the experiment and starts a new MLflow run within a context manager.

        Detailed Description:
        This method is part of the context manager functionality, where it initializes
        an MLflow experiment and begins a new run under the specified experiment name.
        It also logs the reproducibility passport for tracking purposes.

        :return: Returns the context-managed instance of the class.
        """
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name="Download_Prepare_Data")
        log_reproducibility_passport()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        """
        Handles the exit logic for a context manager to ensure proper cleanup or termination
        of resources associated with the context. Specifically, it checks if the `_run`
        attribute is present and ends an MLflow run if it is detected.

        :param exc_type: Exception type if an exception occurred during execution or
                         None if no exception occurred.
        :type exc_type: Optional[Type[BaseException]]
        :param exc_val: Exception value providing information about the exception
                        encountered or None if no exception occurred.
        :type exc_val: Optional[BaseException]
        :param exc_tb: Traceback object containing the call stack for the exception
                       or None if no exception occurred.
        :type exc_tb: Optional[TracebackType]
        :return: A boolean indicating whether to suppress the exception or not. Always
                 returns None, as it does not suppress any exceptions.
        :rtype: Optional[bool]
        """
        if hasattr(self, "_run"):
            mlflow.end_run()

    def log_dataset_info(
        self,
        args: argparse.Namespace,
        df: pd.DataFrame,
        small_df: pd.DataFrame,
        paths: DataPaths,
    ) -> None:
        """
        Logs information about the dataset to MLflow, including dataset name, configuration, sizes of the full and
        small datasets, and their checksums. This helps in tracking the data associated with a particular experiment.

        :param args: Command-line or configuration arguments containing `dataset_name` and `dataset_config`.
            Expected to be an object.
        :param df: Full dataset represented as a Pandas DataFrame.
        :param small_df: Subset of the dataset represented as a Pandas DataFrame.
        :param paths: An instance of the `DataPaths` class, which contains paths to the dataset files, including
            attributes `full_parquet_path` and `small_parquet_path`.
        :return: None
        """
        mlflow.log_param("dataset_name", args.dataset_name)
        mlflow.log_param("dataset_config", args.dataset_config)
        mlflow.log_param("full_dataset_size", len(df))
        mlflow.log_param("small_dataset_size", len(small_df))
        mlflow.set_tag(
            "data.full_checksum",
            ChecksumManager.calculate_checksum(paths.full_parquet_path),
        )
        mlflow.set_tag(
            "data.small_checksum",
            ChecksumManager.calculate_checksum(paths.small_parquet_path),
        )


class DatasetDownloader:
    """
    Provides functionality to download datasets from Hugging Face.

    This class includes static methods to simplify the process of
    fetching datasets from the Hugging Face library by specifying
    the dataset name and its configuration.
    """

    @staticmethod
    def download_dataset(dataset_name: str, dataset_config: str) -> Dataset:
        """
        This static method downloads a dataset from Hugging Face based on the
        given name and configuration. It retrieves the dataset and confirms
        successful download.

        :param dataset_name: The name of the dataset to download.
        :type dataset_name: str
        :param dataset_config: The specific configuration or subset of the dataset
                               to download.
        :type dataset_config: str
        :return: The downloaded dataset from Hugging Face.
        :rtype: Dataset
        """
        print(f"Descargando dataset {dataset_name} desde Hugging Face...")
        dataset = load_dataset(dataset_name, dataset_config)
        print("Dataset descargado correctamente.")
        return dataset


class DataProcessor:
    """
    Handles processing of datasets by managing filesystem operations, downloading datasets,
    processing them into desired formats, and storing the results.

    This class acts as a controller for a processing pipeline that includes setting up
    environments, determining if processing is necessary based on file existence and
    checksums, downloading datasets, processing the data, creating subsets, saving the data,
    and logging essential information about the process.

    :ivar fs_manager: Manages filesystem operations such as creating directories and
        checking for file existence.
    :type fs_manager: FileSystemManager
    :ivar checksum_manager: Handles checksum generation and validation for datasets.
    :type checksum_manager: ChecksumManager
    """

    def __init__(self):
        """
        Manages setup and initialization of core managers responsible for file system
        operations and checksum calculations.

        This class initializes and stores the managers required for performing
        operations involving file systems and checksum validation. These managers are
        designed to handle specific tasks and their initialization is a prerequisite
        for other components that depend on their functionality.

        Attributes:
            fs_manager: Manages file system operations.
            checksum_manager: Manages checksum calculations for files.
        """
        self.fs_manager = FileSystemManager()
        self.checksum_manager = ChecksumManager()

    def _setup_environment(self, paths: DataPaths) -> None:
        """
        Sets up the environment by creating the necessary directories defined in the
        provided DataPaths object.

        This method ensures that the required directory structure is initialized
        before proceeding with further operations.

        :param paths: A DataPaths object containing the directory paths to create.
        :type paths: DataPaths
        :return: None
        """
        self.fs_manager.create_directories(paths=paths)

    def _should_skip_processing(self, paths: DataPaths, force: bool) -> bool:
        """
        Determines if the processing of files should be skipped based on their existence and
        integrity status.

        :param paths: The `DataPaths` object containing information about the target
            files and directories.
        :param force: A boolean flag indicating whether to force the processing
            irrespective of existing files.
        :return: A boolean indicating whether the file processing should be skipped.
        """
        if self.fs_manager.files_exist(paths=paths, force=force):
            print("Los archivos ya existen. Usa --force para regenerarlos.")
            if not paths.checksums_path.exists():
                self.checksum_manager.generate_checksums(paths=paths)
            return True
        return False

    def _download_and_process_data(self, args: argparse.Namespace):
        """
        Downloads and processes a dataset based on the provided arguments. This function uses
        a DatasetDownloader to fetch the dataset and a DatasetProcessor to process it into
        the desired format for further use. The function is responsible for transforming the
        raw dataset into a pandas DataFrame and applying specific dataset processing logic as
        defined by the DatasetProcessorFactory.

        :param args: Command-line arguments containing the dataset name and its configuration.
        :type args: argparse.Namespace
        :return: The processed dataset after applying all processing steps appropriate for the
                 specified dataset.
        :rtype: Any
        """
        dataset = DatasetDownloader.download_dataset(
            dataset_name=args.dataset_name, dataset_config=args.dataset_config
        )

        print("Procesando dataset...")
        train_df = dataset["train"].to_pandas()

        processor = DatasetProcessorFactory.create_processor(
            dataset_name=args.dataset_name
        )
        return processor.process(train_df=train_df)

    def _create_small_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a smaller randomized subset of the given DataFrame. The method samples
        a maximum of 1500 rows or the total number of rows in the DataFrame if the total
        is less than 1500. The selection is randomized but reproducible using a fixed
        random seed value.

        :param df: Input DataFrame from which a subset is created
        :type df: pd.DataFrame
        :return: A new DataFrame that is a smaller randomized subset of the input
        :rtype: pd.DataFrame
        """
        small_df = df.sample(min(1500, len(df)), random_state=42)
        return small_df

    def _save_dataset(
        self, df: pd.DataFrame, small_df: pd.DataFrame, paths: DataPaths
    ) -> None:
        """
        Saves the provided datasets to specified locations and generates checksums for them.

        The method writes `df` to the location specified by `paths.full_parquet_path` and
        `small_df` to the location specified by `paths.small_parquet_path` in Parquet format.
        After saving, checksum files are generated for validation and integrity checking.

        :param df: The primary dataset to be saved, containing the full data.
        :type df: pd.DataFrame
        :param small_df: A smaller subset of the dataset to be saved, generally containing sampled or reduced data.
        :type small_df: pd.DataFrame
        :param paths: An object containing file paths for saving both full and small datasets in Parquet format,
            along with paths for checksum files.
        :type paths: DataPaths
        :return: None
        """
        print("")
        df.to_parquet(paths.full_parquet_path, index=False)

        print()
        small_df.to_parquet(paths.small_parquet_path, index=False)

        self.checksum_manager.generate_checksums(paths=paths)

    def _finalize_processing(
        self,
        args: argparse.Namespace,
        df: pd.DataFrame,
        small_df: pd.DataFrame,
        paths: DataPaths,
        logger: MLflowLogger,
    ) -> None:
        """
        Finalizes the dataset processing by logging dataset information and printing success messages
        about saved files and checksums.

        :param args: Parsed command-line arguments containing relevant dataset configurations
        :type args: argparse.Namespace
        :param df: The complete processed dataset
        :type df: pd.DataFrame
        :param small_df: A subset of the processed dataset
        :type small_df: pd.DataFrame
        :param paths: Object encapsulating paths used for storing output data files and checksums
        :type paths: DataPaths
        :param logger: Logger instance for logging dataset-related information
        :type logger: MLflowLogger
        :return: None
        """
        logger.log_dataset_info(args=args, df=df, small_df=small_df, paths=paths)
        print("Dataset procesado correctamente.")
        print(f"Archivos guardados en {paths.data_dir}")
        print(f"Archivos checksums guardados en {paths.checksums_path}")

    def process_dataset(self, args: argparse.Namespace) -> None:
        """
        Processes a dataset by performing various predefined operations including environment setup,
        data downloading, transformation, creating a smaller subset, and saving the results to specified paths.
        The method also utilizes logging to track the processing workflow.

        :param args: Command-line arguments encapsulated in an argparse.Namespace object.
                     These arguments drive the dataset processing configurations.
        """
        paths = DataPaths.from_args(args=args)

        with MLflowLogger() as logger:
            self._setup_environment(paths=paths)

            if self._should_skip_processing(paths=paths, force=args.force):
                return

            df = self._download_and_process_data(args=args)
            small_df = self._create_small_subset(df=df)
            self._save_dataset(df=df, small_df=small_df, paths=paths)
            self._finalize_processing(
                args=args, df=df, small_df=small_df, paths=paths, logger=logger
            )


def main(args: argparse.Namespace) -> None:
    """
    Processes the dataset using the DataProcessor class.

    This function initializes the DataProcessor and processes the dataset
    provided through the arguments. It acts as the main entry point for dataset
    processing logic.

    :param args: Arguments required for dataset processing
    :type args: Any
    :return: None
    """
    processor = DataProcessor()
    processor.process_dataset(args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga y prepara los datos de Flickr8k"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flickr8k",
        help="Directorio donde guardar los datos",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tsystems/flickr8k",
        help="Nombre del dataset en Hugging Face",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help="Configuración del dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar la regeneración de archivos existentes",
        default=True,
    )

    args = parser.parse_args()
    main(args)
