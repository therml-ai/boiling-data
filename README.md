# Boiling Data

A utility library for converting between boiling data formats

This is primarily intended to have scripts for data conversion,
quality checks, computing normalization values, etc.

## Data Conversion

The `src/boiling_data/boiling_data.py` has 
`to_<format>` and `from_<format>` methods to enable easy conversion
between different file formats. I.e., loading with `from_flashx`
and storing with `to_bubbleml` converts flashx data into the bubbleml
format.

We have a script to convert the Flash-x simulation into the BubbleML format:

```bash
uv run python scripts/flashx_to_bubbleml.py \
    --sim-dir /path/to/flashx_sims \
    --output-dir /path/to/bubbleml_output
```

The output will be an HDF5 file with lots of keys for different fields.
The simulation parameters are a json string stored in the hdf5 file attrs.
To load the simulation parameters,

```python
import h5py
import json

with h5oy.File("", "r") as handle:
    params_str: str = handle.attrs["parameters"]
    params = json.loads(params_str)
    units = json.loads(handle.attrs["units"])
```

## Data loading

`FlashXForecastDataset` reads windows of BubbleML files for forecasting: each
item is an input window and the target window that follows it. `flashx_collater`
batches them into a `FlashXForecastBatch` of two `FlashXBatch`es, `input` and
`target`, and `NormalizerWrapper` normalizes them with the statistics that can
be computed using `scripts/bubbleml_statistics.py`.

```python
import functools

from torch.utils.data import DataLoader

from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.tasks.forecast.flashx_batch import flashx_collater
from boiling_data.tasks.forecast.flashx_dataset import FlashXForecastDataset

field_names = ["dfun", "temperature", "velfacex", "velfacey"]
dataset = FlashXForecastDataset(
    ["/path/to/case-1.hdf5", "/path/to/case-2.hdf5"],
    field_names,
    input_timesteps=8,
    target_timesteps=8,
    start_frame=200,  # skip each run's start-up transient
    stride=4,  # frames between the starts of consecutive windows
)
# the parameters to condition on, by config group
config_keys = {"non_dimensional": ["stefan", "prandtl"], "heaters": ["wall_temp_fraction"]}
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=functools.partial(flashx_collater, config_keys=config_keys),
)
normalizer = NormalizerWrapper.from_json(
    "/path/to/split/statistics.json", vector_fields=[("velfacex", "velfacey")]
)

for batch in loader:  # a FlashXForecastBatch
    inputs = normalizer.normalize(batch.input)  # a FlashXBatch
    targets = normalizer.normalize(batch.target)
    inputs.fields["temperature"]  # [batch, time, y, x]
    inputs.fields["velfacex"]  # [batch, time, y, x + 1], on the x faces
    inputs.config_tensor()  # [batch, 3], the config_keys of every sample
    inputs.stack_field_cells(field_names)  # [batch, time, y, x, 6], channels last
```

`FlashXInMemoryForecastDataset` takes the same arguments and loads every file
into memory up front. With `unroll_steps=2`, each target holds two target
windows. The unrolling can be used for training a model on its own predictions; i.e.,
for the push-forward trick or scheduled sampling.

