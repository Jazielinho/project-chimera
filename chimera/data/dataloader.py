import ast
import io
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DataSource(Protocol):
    """
    Defines the contract for a data source by specifying required methods to be
    implemented.

    This class is a Protocol that outlines the basic interface any data source
    must comply with. It ensures a standardized way to retrieve the length of
    the data source as well as individual items from it.

    """

    def __len__(self) -> int:
        """
        Determines the number of items in the object.

        This method returns the count of elements contained within the object. It
        provides the length of the object when called using the built-in ``len()``.

        :return: The number of items contained in the object.
        :rtype: int
        """
        ...

    def get_item(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves an item at the specified index.

        This method fetches and returns a dictionary representing the
        item identified by the given index. The structure and contents of
        the returned dictionary can vary depending on its implementation.

        :param idx: Index of the item to retrieve.
        :type idx: int
        :return: Dictionary containing details of the item.
        :rtype: Dict[str, Any]
        """
        ...


class ImageProcessor(ABC):
    """
    Provides an interface for image processing.

    This abstract base class defines a blueprint for implementing image processing
    functionalities. Classes inheriting from this must implement the `process`
    method to define specific image transformation or manipulation logic.
    """

    @abstractmethod
    def process(self, image_data: Any) -> Image.Image:
        """
        Processes the input image data to produce an output image.

        This method should be implemented by all subclasses to provide
        specific image processing functionality.

        :param image_data: The input data representing the image to be processed.
                           It can be of any data type compatible with the subclass
                           implementation.
        :return: The processed image as an instance of `PIL.Image.Image`.
        """
        pass


class CaptionProcessor(ABC):
    """
    Abstract base class for caption processing.

    This class serves as a blueprint for any caption processing implementation.
    The primary purpose is to process caption data into a string that conforms
    to the given maximum length constraint. Subclasses must implement the
    `process` method.

    """

    @abstractmethod
    def process(self, caption_data: Any, max_length: int) -> str:
        """
        Processes and formats the provided caption data based on the given maximum length.

        This method is an abstract method that must be implemented by any subclass
        to ensure consistent handling and transformation of caption data.

        :param caption_data: The input caption data which may be of any type. It
            represents the content to be processed.
        :param max_length: The maximum allowable length for the processed caption.
            The resulting string should not exceed this limit.
        :return: A string representing the processed form of the input caption data
            that adheres to the maximum length constraint.
        """
        pass


class BytesImageProcessor(ImageProcessor):
    """
    Processes image data provided in bytes format and converts it into an RGB
    Image object. This class extends the functionality of the ImageProcessor
    base class, specifically focusing on handling image data in bytes.

    This processor can be used in scenarios where image data is provided in
    binary form and needs to be transformed into a usable image for further
    processing or analysis.
    """

    def process(self, image_data: bytes) -> Image.Image:
        """
        Processes the provided image data and returns a Pillow Image object in RGB format.

        This method takes image data in bytes and converts it to a Pillow Image object. It converts
        the image to RGB mode to ensure compatibility with image-processing workflows.

        :param image_data: The input image data provided in bytes format.
        :return: A Pillow Image object in RGB mode.
        """
        return Image.open(io.BytesIO(image_data)).convert("RGB")


class PathImageProcessor(ImageProcessor):
    """
    Handles image processing when given a path to an image file.

    This class is an extension of the `ImageProcessor` base class and is designed to
    handle image processing specifically for file paths. The `process` method is implemented
    to read an image from the given file path and convert it to RGB format, ready for further
    processing.

    """

    def process(self, image_data: str) -> Image.Image:
        """
        Processes the given image data, loading the image and converting it into
        RGB format.

        :param image_data: Input image data as a string (usually a file path or file-like object)
        :type image_data: str
        :return: An RGB-converted PIL Image object
        :rtype: Image.Image
        """
        return Image.open(image_data).convert("RGB")


class DefaultCaptionProcessor(CaptionProcessor):
    """
    Processes and generates captions from various data types with optional length restriction.

    This class provides functionality to handle caption data of multiple types such as
    lists, numpy arrays, or strings. It processes the input and extracts or generates an
    appropriate caption while ensuring the output adheres to the specified maximum length.

    :ivar caption_data: Source data for the caption. Can be a list, numpy array, or string.
    :type caption_data: Any
    :ivar max_length: Maximum length for the processed caption.
    :type max_length: int
    """

    def process(self, caption_data: Any, max_length: int) -> str:
        """
        Processes caption data by selecting or converting it into a string format and truncating it to a maximum
        length. Handles different types of input, such as lists, numpy arrays, and strings.

        :param caption_data: Caption data that can be either a list, numpy array, or string.
        :param max_length: Maximum length allowed for the processed caption.
        :return: Truncated and processed caption as a string.
        :rtype: str
        """
        if isinstance(caption_data, list):
            caption = random.choice(caption_data)
        elif isinstance(caption_data, np.ndarray):
            # Convertir numpy array a string y luego parsear
            caption_str = (
                str(caption_data.item())
                if caption_data.ndim == 0
                else str(caption_data[0])
            )
            if caption_str.startswith("["):
                try:
                    caption_list = ast.literal_eval(caption_str)
                    if isinstance(caption_list, list) and len(caption_list) > 0:
                        caption = random.choice(caption_list)
                    else:
                        caption = caption_str
                except (ValueError, SyntaxError):
                    caption = caption_str
            else:
                caption = caption_str
        elif isinstance(caption_data, str) and caption_data.startswith("["):
            # Manejar strings que representan listas de Python
            try:
                caption_list = ast.literal_eval(caption_data)
                if isinstance(caption_list, list) and len(caption_list) > 0:
                    caption = random.choice(caption_list)
                else:
                    caption = str(caption_data)
            except (ValueError, SyntaxError):
                caption = str(caption_data)
        else:
            caption = str(caption_data)
        return caption[:max_length]


class ImageProcessorFactory:
    """
    Factory class for selecting appropriate image processors.

    This class provides a method to retrieve an image processor based
    on the type of image data supplied. It supports dynamic selection
    of processors, ensuring flexibility for handling different data
    types related to image processing.

    :ivar _processors: Maps data types to their respective processors.
    :type _processors: dict[type, ImageProcessor]
    """

    _processors = {bytes: BytesImageProcessor(), str: PathImageProcessor()}

    @classmethod
    def get_processor(cls, image_data: Any) -> ImageProcessor:
        """
        Retrieves the appropriate image processor based on the type of the provided image
        data. The method checks the registered processors for the type of the image
        data and returns the matching processor instance. If no processor is found
        that supports the image data type, an error will be raised.

        :param image_data: The image data for which the matching processor is
            required.
        :type image_data: Any
        :return: An instance of the ImageProcessor that can handle the provided image
            data.
        :rtype: ImageProcessor
        :raises ValueError: If the type of the provided image data is not supported.
        """
        processor = cls._processors.get(type(image_data))
        if processor is None:
            raise ValueError(
                f"Tipo de datos de imagen no soportado: {type(image_data)}"
            )
        return processor


class ParquetDataSource:
    """
    Provides an interface to load and access data from a parquet file as a
    pandas DataFrame.

    The class facilitates reading data from a parquet file, accessing the length
    of the data, and retrieving individual rows as dictionaries for further
    processing or analysis.

    :ivar df: The loaded pandas DataFrame containing the data from the
        provided parquet file path.
    :type df: pandas.DataFrame
    """

    def __init__(self, parquet_path: Union[str, Path]):
        """
        Initializes an instance of the class by loading a DataFrame from a given Parquet file.

        This constructor loads data from a specified Parquet file path into a Pandas DataFrame.
        The path can be provided either as a string or a `Path` object.

        :param parquet_path: The file path to the Parquet file to be loaded.
        :type parquet_path: Union[str, Path]
        """
        self.df = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        """
        Computes the length of an object, specifically the number of elements
        contained in the underlying data structure.

        This method provides support for obtaining the count of elements, which
        is typically used in data structures with a concept of size, length, or
        cardinality.

        :return: The number of elements in the object.
        :rtype: int
        """
        return len(self.df)

    def get_item(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve an item from the dataframe at a specified index.

        This method extracts a row from the dataframe based on the provided index,
        converts it to a dictionary, and returns it. The keys in the dictionary
        correspond to the column names of the dataframe, and the values correspond
        to the data in that row.

        :param idx: The index of the row to retrieve.
        :type idx: int
        :return: A dictionary representation of the row at the specified index.
        :rtype: Dict[str, Any]
        """
        return self.df.iloc[idx].to_dict()


class ColumnValidator:
    """
    Provides functionality to validate the existence of required columns
    in a data source. This class ensures that all specified columns are
    present within a given Parquet data source.

    This class contains only static methods for validation and does not
    require instantiation.

    """

    @staticmethod
    def validate(data_source: ParquetDataSource, required_columns: List[str]) -> None:
        """
        Validates that all required columns exist in the DataFrame contained in the provided data source.

        :param data_source: The data source containing the DataFrame to validate.
        :type data_source: ParquetDataSource
        :param required_columns: A list of column names that are required to be present in the DataFrame.
        :type required_columns: List[str]
        :return: None
        """
        for column in required_columns:
            if column not in data_source.df.columns:
                raise ValueError(f"Columna '{column}' no encontrada en el DataFrame")


class DefaultTransformFactory:
    """
    Factory class for creating default image transformations.

    This class provides a static method to create a default sequence of
    transformations commonly used for image preprocessing in machine learning
    and deep learning tasks. The transformation includes resizing, cropping,
    converting to tensor format, and normalization. It ensures that image
    data adheres to the size and scale required by most pretrained neural
    network models.

    Methods:
        create: Constructs and returns a composed transformation sequence.
    """

    @staticmethod
    def create() -> transforms.Compose:
        """
        Creates and returns a composed transformation pipeline.

        This static method generates a composition of several image transformation
        steps, including resizing, center cropping, tensor conversion, and normalization.
        Each operation is applied sequentially to preprocess the input data for model
        compatibility.

        :rtype: torchvision.transforms.Compose
        :return: A composed transformation pipeline containing resizing, center cropping,
            tensor conversion, and normalization transformations.
        """
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class DatasetConfig:
    """
    Configuration for dataset processing.

    This class is used to define and store configuration parameters
    that determine the behavior and structure of dataset processing,
    such as defining column names and limits for data.

    :ivar max_text_length: The maximum allowable length of text in the dataset.
    :type max_text_length: int
    :ivar image_column: The column name in the dataset representing image data.
    :type image_column: str
    :ivar caption_column: The column name in the dataset representing caption text data.
    :type caption_column: str
    :ivar image_id_column: The column name in the dataset representing unique identifiers
        for images.
    :type image_id_column: str
    """

    def __init__(
        self,
        max_text_length: int = 32,
        image_column: str = "image",
        caption_column: str = "caption",
        image_id_column: str = "image_id",
    ):
        """
        Initializes an instance of the class with provided configuration values.

        :param max_text_length: Defines the maximum length of the text/caption. Uses an
            integer value to specify the limit.
        :param image_column: Represents the column name that contains image data. Uses
            a string to define the column name.
        :param caption_column: Represents the column name that contains caption/text
            data. Uses a string to define the column name.
        :param image_id_column: Represents the column name that contains image
            identifiers. Uses a string as the column name.
        """
        self.max_text_length = max_text_length
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_id_column = image_id_column


class Flickr8kDataset(Dataset):
    """
    Represents the Flickr8k dataset, facilitating loading, processing, and transformation
    of image-caption data from a Parquet file. This class integrates image and caption
    processing pipelines, making it suitable for deep learning or other data-intensive
    applications.

    This dataset class allows for transformations on the loaded images and captions
    based on provided or default processors, ensuring compatibility with the intended
    machine learning model architecture. The input dataset must strictly comply with
    the structure and columns defined in the configuration.

    :ivar _data_source: The data source object that provides access to underlying
        Parquet dataset.
    :type _data_source: ParquetDataSource
    :ivar _config: Configuration object defining column names and data processing rules.
    :type _config: DatasetConfig
    :ivar _transform: Image transformation pipeline applied to the processed image data.
    :type _transform: transforms.Compose
    :ivar _image_factory: Factory for creating image processors based on input image data.
    :type _image_factory: ImageProcessorFactory
    :ivar _caption_processor: Processor responsible for processing and optionally
        truncating captions to a specified length.
    :type _caption_processor: CaptionProcessor
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        transform: Optional[transforms.Compose] = None,
        image_processor_factory: Optional[ImageProcessorFactory] = None,
        caption_processor: Optional[CaptionProcessor] = None,
    ):
        """
        Initializes an instance of the dataset class to facilitate data processing
        from a Parquet data source. The configuration, transformations, and processing
        factories can be customized through the provided parameters.

        :param parquet_path: Path to the parquet file containing the dataset.
            Can be a string or a Path object.
        :type parquet_path: Union[str, Path]
        :param config: Optional dataset configuration. If not provided, a default
            `DatasetConfig` instance will be used.
        :type config: Optional[DatasetConfig]
        :param transform: Optional composed transformations to apply to the
            dataset. Defaults to the result produced by `DefaultTransformFactory.create()`.
        :type transform: Optional[transforms.Compose]
        :param image_processor_factory: Optional factory for creating image
            processing instances. Defaults to `ImageProcessorFactory`.
        :type image_processor_factory: Optional[ImageProcessorFactory]
        :param caption_processor: Optional processor for handling captions.
            Defaults to `DefaultCaptionProcessor`.
        :type caption_processor: Optional[CaptionProcessor]

        :raises ValueError: If the provided parquet path is not valid.
        """
        self._config = config or DatasetConfig()
        self._data_source = ParquetDataSource(parquet_path)
        self._transform = transform or DefaultTransformFactory.create()
        self._image_factory = image_processor_factory or ImageProcessorFactory
        self._caption_processor = caption_processor or DefaultCaptionProcessor()

        self._validate_data_source()

    def _validate_data_source(self) -> None:
        """
        Validates the data source against the required columns to ensure its integrity.

        This method checks if the data source contains the necessary columns defined
        in the configuration, such as image_column, caption_column, and image_id_column.
        Validation is performed using the `ColumnValidator.validate` utility.

        Raises an exception if any of the required columns are missing.

        :raises ValueError: If the required columns are not found in the data source.
        """
        required_columns = [
            self._config.image_column,
            self._config.caption_column,
            self._config.image_id_column,
        ]
        ColumnValidator.validate(self._data_source, required_columns)

    def __len__(self) -> int:
        """
        Calculate and return the number of elements in the data source.

        :return: The total number of elements in the data source.
        :rtype: int
        """
        return len(self._data_source)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Retrieves an item at a specific index from the data source and processes
        it for use. This method fetches the data, processes the image, caption,
        and retrieves the image ID.

        :param idx: Index of the item to retrieve
        :type idx: int
        :return: A dictionary containing the processed image, caption, and image ID
        :rtype: Dict[str, Union[torch.Tensor, str]]
        :raises RuntimeError: If an error occurs during the process of retrieving
                              or processing the item
        """
        try:
            item_data = self._data_source.get_item(idx)

            return {
                "image": self._process_image(item_data),
                "caption": self._process_caption(item_data),
                "image_id": self._get_image_id(item_data, idx),
            }
        except Exception as e:
            raise RuntimeError(f"Error procesando item {idx}: {str(e)}") from e

    def _process_image(self, item_data: Dict[str, Any]) -> torch.Tensor:
        """
        Processes an image from the provided item data using the configured processor
        and applies a transformation to it.

        :param item_data: A dictionary containing input data from which the image will be extracted
            and processed based on the designated image column in the configuration.
        :type item_data: Dict[str, Any]
        :return: A tensor representing the processed and transformed image.
        :rtype: torch.Tensor
        """
        image_data = item_data[self._config.image_column]
        processor = self._image_factory.get_processor(image_data)
        image = processor.process(image_data)
        return self._transform(image)

    def _process_caption(self, item_data: Dict[str, Any]) -> str:
        """
        Processes the caption data for a given item.

        The method retrieves caption information from the input `item_data` based on
        a specified column defined in the configuration. Then it processes this data
        to ensure the caption conforms to a maximum length constraint.

        :param item_data: Dictionary containing item data, from which the caption
                          information is retrieved and processed.
        :type item_data: Dict[str, Any]
        :return: The processed caption as a string.
        :rtype: str
        """
        caption_data = item_data.get(self._config.caption_column, "")
        return self._caption_processor.process(
            caption_data, self._config.max_text_length
        )

    def _get_image_id(self, item_data: Dict[str, Any], idx: int) -> str:
        """
        Generates an image ID based on the provided item data and index.

        This method retrieves the image ID using the specified column name from the
        configuration, or defaults to the provided index if the column is not present
        in the input data. It ensures the result is returned as a string.

        :param item_data: Mapping containing the data for an item.
        :type item_data: Dict[str, Any]
        :param idx: Index to be used as the default image ID if the relevant column is
            missing in the item_data.
        :type idx: int
        :return: The generated image ID as a string.
        :rtype: str
        """
        return str(item_data.get(self._config.image_id_column, idx))


class Flickr8kDatasetBuilder:
    """
    Class responsible for constructing Flickr8kDataset instances.

    This class provides a flexible and configurable way to build instances
    of the Flickr8kDataset. It supports chaining methods to set up various
    elements, such as dataset configuration, transforms, image processing
    factories, and caption processors, before finalizing the construction
    of the dataset object.

    :ivar parquet_path: Path to the parquet file containing dataset information.
    :type parquet_path: Union[str, Path]
    """

    def __init__(self, parquet_path: Union[str, Path]):
        """
        Initializes an instance of the class with the specified dataset configuration
        stored in a Parquet file. This constructor prepares the instance for further
        processing by setting default values for image handling and caption processing
        functionalities.

        :param parquet_path: Path to the Parquet file containing the dataset config.
        :type parquet_path: Union[str, Path]
        """
        self._parquet_path = parquet_path
        self._config = DatasetConfig()
        self._transform = None
        self._image_factory = None
        self._caption_processor = None

    def with_config(self, config: DatasetConfig) -> "Flickr8kDatasetBuilder":
        """
        Updates the dataset builder with the given configuration.

        This method allows setting a specific configuration to the dataset builder
        by assigning the provided `DatasetConfig` object to the internal configuration
        attribute. After updating the configuration, it returns the modified dataset
        builder for potential chaining of operations.

        :param config: The dataset configuration to be applied.
        :type config: DatasetConfig
        :return: The updated instance of `Flickr8kDatasetBuilder`.
        :rtype: Flickr8kDatasetBuilder
        """
        self._config = config
        return self

    def with_transform(self, transform: transforms.Compose) -> "Flickr8kDatasetBuilder":
        """
        Sets the transformation operation for the dataset. This is typically used to
        apply preprocessing or augmentation steps to the dataset, encapsulated within
        a transformation pipeline.

        :param transform: Transformation pipeline to apply to the dataset.
        :type transform: transforms.Compose
        :return: Returns the current instance of the dataset builder with the specified
                 transformation set.
        :rtype: Flickr8kDatasetBuilder
        """
        self._transform = transform
        return self

    def with_image_factory(
        self, factory: ImageProcessorFactory
    ) -> "Flickr8kDatasetBuilder":
        """
        Sets the image processing factory used during dataset building.

        This method allows the user to specify a custom image processing factory that
        implements the necessary operations to process images in the dataset.

        :param factory: Custom image processing factory to be used.
        :type factory: ImageProcessorFactory
        :return: An instance of the Flickr8kDatasetBuilder with the specified image factory set.
        :rtype: Flickr8kDatasetBuilder
        """
        self._image_factory = factory
        return self

    def with_caption_processor(
        self, processor: CaptionProcessor
    ) -> "Flickr8kDatasetBuilder":
        """
        Sets the caption processor for the Flickr8kDatasetBuilder and returns the builder instance for method chaining.

        The caption processor is responsible for processing and transforming captions in the dataset.

        :param processor: The caption processor instance used to process dataset captions.
        :type processor: CaptionProcessor
        :return: The updated instance of Flickr8kDatasetBuilder.
        :rtype: Flickr8kDatasetBuilder
        """
        self._caption_processor = processor
        return self

    def build(self) -> Flickr8kDataset:
        """
        Builds and returns an instance of Flickr8kDataset. This method utilizes
        configured settings, processors, and transformations to construct the
        dataset object.

        :raises SomeSpecificError: When certain conditions are not met during
            the dataset building process.

        :return: Constructed Flickr8kDataset object.
        :rtype: Flickr8kDataset
        """
        return Flickr8kDataset(
            parquet_path=self._parquet_path,
            config=self._config,
            transform=self._transform,
            image_processor_factory=self._image_factory,
            caption_processor=self._caption_processor,
        )


class _SpeedMonitor:
    """
    Monitors the speed (frames per second) by keeping track of the number of updates and
    the elapsed time.

    This class is used to calculate frames per second (FPS) by using a counter and
    a timer. The `update` method increments the counters, while the `fps` method
    computes the rate of updates based on elapsed time.

    :ivar t0: Initial reference time (in seconds) based on the performance counter.
    :type t0: float
    :ivar count: The number of updates recorded.
    :type count: int
    """

    def __init__(self):
        """
        A simple class to measure elapsed time and count increments.

        This class initializes a timer using the `time.perf_counter` function and
        provides functionality to keep track of a counter. It is designed for
        tracking performance or time-sensitive operations while maintaining a count.

        Attributes:
            t0 (float): The starting time recorded during initialization.
            count (int): A counter to track the number of increments or operations.

        """
        self.t0 = time.perf_counter()
        self.count = 0

    def update(self, n: int) -> None:
        """
        Updates the internal count by adding the provided value.

        This method increments the count attribute by the given integer value.
        It does not return any value and directly modifies the instance's state.

        :param n: The integer value to add to the current count.
        :type n: int
        :return: None
        """
        self.count += n

    def fps(self) -> float:
        """
        Calculate the frames per second (FPS) based on the elapsed time and frame count.

        This method computes the FPS by dividing the frame count by the time elapsed
        since the initial time marker. It ensures there is no division by zero by
        checking if the elapsed time is positive.

        :return: The calculated frames per second (FPS) as a float. Returns 0 if elapsed time is zero.
        :rtype: float
        """
        dt = time.perf_counter() - self.t0
        return self.count / dt if dt > 0 else 0


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Combines a list of data points into a single batch dictionary. The function processes
    a list of dictionaries, extracts images and captions from each dictionary, and stacks
    them to create a unified format suitable for model input or further preprocessing.

    :param batch: A list of dictionaries where each dictionary contains image tensors
        and corresponding captions.
    :type batch: List[Dict]

    :return: A dictionary containing:
        - "image": A batch of images as a stacked tensor.
        - "caption": A list of captions corresponding to the images in the batch.
    :rtype: Dict[str, torch.Tensor]
    """
    images = torch.stack([item["image"] for item in batch], dim=0).contiguous()
    captions = [item["caption"] for item in batch]
    return {"image": images, "caption": captions}


def _suggest_num_workers(batch_size: int) -> int:
    """
    Determines the suggested number of workers for a given batch size. It calculates
    an appropriate value based on the CPU count of the system to optimize performance.

    :param batch_size: Batch size to process. Used to determine the minimum number of
        workers required. Must be an integer greater than 0.
    :return: Suggested number of workers to use, calculated as the minimum of the
        maximum workers (calculated based on the CPU count) and the maximum between
        2 and the provided batch size. Always returns an integer.
    """
    max_workers = (os.cpu_count() or 8) * 2
    return min(max_workers, max(2, batch_size))


def create_dataloader(
    parquet_path: str | Path,
    batch_size: int = 16,
    num_workers: int | None = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    shuffle: bool = True,
    transform=None,
    max_text_length: int = 32,
    deterministic: bool = False,
    monitor_speed: bool = True,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for efficiently loading and processing data from a
    parquet dataset. The function configures the dataset, applies optional
    transformations, manages worker initialization for deterministic or
    non-deterministic behavior, monitors the speed of data loading if requested,
    and provides an optimized collation function.

    :param parquet_path: Path to the parquet file containing dataset information.
    :type parquet_path: str | Path
    :param batch_size: Number of samples per batch. Defaults to 16.
    :type batch_size: int
    :param num_workers: Number of subprocesses to use for data loading. If None,
        the function will attempt to suggest an optimal number. Defaults to None.
    :type num_workers: int | None
    :param pin_memory: If True, the data loader will copy tensors into CUDA pinned
        memory before returning them. Defaults to True.
    :type pin_memory: bool
    :param prefetch_factor: Number of samples to pre-fetch across all workers.
        Effective only when num_workers > 0. Defaults to 4.
    :type prefetch_factor: int
    :param shuffle: If True, the data will be reshuffled at every epoch. Defaults
        to True.
    :type shuffle: bool
    :param transform: Optional transformation to apply on the dataset. Defaults to
        None.
    :type transform: callable | None
    :param max_text_length: Maximum permissible length for textual data. Defaults
        to 32.
    :type max_text_length: int
    :param deterministic: If True, ensures that data loading is deterministic by
        setting a fixed seed for workers. Defaults to False.
    :type deterministic: bool
    :param monitor_speed: If True, includes a speed monitor to track the data
        loading speed. Defaults to True.
    :type monitor_speed: bool
    :return: A PyTorch DataLoader object configured for the specified dataset and
        loading parameters.
    :rtype: DataLoader
    """
    config = DatasetConfig(max_text_length=max_text_length)
    dataset = (
        Flickr8kDatasetBuilder(parquet_path)
        .with_config(config)
        .with_transform(transform)
        .build()
    )

    if num_workers is None:
        num_workers = _suggest_num_workers(batch_size=batch_size)

    worker_init_fn = None
    if deterministic:

        def _seed(worker_id: int) -> None:
            seed = torch.initial_seed() % 2**32
            torch.manual_seed(seed)

        worker_init_fn = _seed

    speed_monitor = _SpeedMonitor() if monitor_speed else None

    def _wrapper_collate(batch):
        out = _collate_fn(batch)
        if speed_monitor is not None:
            speed_monitor.update(n=len(batch))
        return out

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=_wrapper_collate,
        worker_init_fn=worker_init_fn,
    )

    dataloader.speed = speed_monitor

    return dataloader


if __name__ == "__main__":
    parquet_path = "/media/jahaziel/Datos/publicaciones/project-chimera/data/flickr8k/processed/flickr8k_small.parquet"

    dataloader = create_dataloader(
        parquet_path=parquet_path,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True,
    )

    for batch in dataloader:
        print(batch)
        break
