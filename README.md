# model-validation-configs

This repository contains configurations for model validation.

The `accuracy` folder contains YAML files for each model that configures information needed for the model to be validated through the [llm-eval-test](https://github.com/openshift-psap/llm-eval-test). There are 4 config files for each model:

* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the llm-eval-test harness for the model
* accuracy.yml: contains evaluation tasks and accuracy expectations for the model
* storage.yml: specifies where mode and dataset is located

The `benchmark` folder contains YAML files for each model that configures information needed for the model to be validated through the [guidellm](https://github.com/neuralmagic/guidellm). There are  config files for each model:

* server.yml: contains settings to start a vllm server with the model
* client.yml: contains settings for the guidellm for the model
* storage.yml: specifies where mode and dataset is located
