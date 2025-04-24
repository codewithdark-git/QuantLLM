Dataset API
==========

Dataset Loading
-------------

.. automodule:: quantllm.data.load_dataset
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Preprocessing
------------------

.. automodule:: quantllm.data.dataset_preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Splitting
--------------

.. automodule:: quantllm.data.dataset_splitter
   :members:
   :undoc-members:
   :show-inheritance:

DataLoader
---------

.. automodule:: quantllm.data.dataloader
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-----------

Loading a Dataset
~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import LoadDataset, DatasetConfig

    # Load from HuggingFace
    dataset = LoadDataset().load_hf_dataset("imdb")

    # Load local dataset
    dataset = LoadDataset().load_local_dataset(
        file_path="path/to/data.csv",
        file_type="csv"
    )

Preprocessing
~~~~~~~~~~~

.. code-block:: python

    from quantllm import DatasetPreprocessor

    preprocessor = DatasetPreprocessor(tokenizer)
    train_processed, val_processed, test_processed = preprocessor.tokenize_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        max_length=512,
        text_column="text",
        label_column="label"
    )

Dataset Splitting
~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import DatasetSplitter

    splitter = DatasetSplitter()
    train, val, test = splitter.train_val_test_split(
        dataset,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )

    # Or just train-val split
    train, val = splitter.train_val_split(
        dataset,
        train_size=0.8
    )

Creating DataLoaders
~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import DataLoader

    train_loader, val_loader, test_loader = DataLoader.from_datasets(
        train_dataset=train_processed,
        val_dataset=val_processed,
        test_dataset=test_processed,
        batch_size=8
    )