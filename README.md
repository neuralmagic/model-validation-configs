# model-validation-configs

This repository contains configurations for model validation.

The configs are organized by model. Each top level folder should be named after the model stub and contain `accuracy` and `performance` sub-folders. Both folders contain configs specific to the activity. Shared configs, e.g. `storage.yml` should be in the common area.

The `accuracy` folder contains YAML configs needed for the model to be validated through the [llm-eval-test](https://github.com/openshift-psap/llm-eval-test). There are 3 config files for each model:
* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the llm-eval-test harness for the model
* accuracy.yml: contains evaluation tasks and accuracy expectations for the model

The `benchmark` folder contains YAML configs needed for the model to be validated through the [guidellm](https://github.com/neuralmagic/guidellm). There are 3 config files for each model:
* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the guidellm for the model
