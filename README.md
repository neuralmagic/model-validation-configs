# model-validation-configs

This repository contains configurations for model validation.

The configs are organized by model. Each top level folder should be named after the model stub and contain `accuracy` and `performance` sub-folders. Both folders contain configs specific to the activity. Shared configs, e.g. `storage.yml` should be in the common area.

The `accuracy` folder contains YAML configs needed for the model to be validated through the [llm-eval-test](https://github.com/openshift-psap/llm-eval-test). There are 3 config files for each model:
* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the llm-eval-test harness for the model
* accuracy.yml: contains evaluation tasks and accuracy expectations for the model

The `performance` folder contains YAML configs needed for the model to be validated through the [guidellm](https://github.com/neuralmagic/guidellm). There are 2 config files for each model:
* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the guidellm for the model

## Special Cases

### `server.yml` tensor-parallel-size
To take advantage of available GPUs on a test environment, if the `tensor-parallel-size` argument is not specified
in the `server.yml` configuration file, it will be added and set to the value reported by the `torch.cuda.device_count()`
python function.  If it's set to an empty, or `null` value, in `server.yml`, that argument will not get passed to
the `vllm server` command.  If it is set, that value will be provided to the server.
