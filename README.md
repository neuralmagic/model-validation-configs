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

### `extra_config.yml` (optional overlays)

Optional per-model overlays live at `<org>/<model>/extra_config.yml`. A shared
default is in `common/extra_config.yml`.

Sections are keyed by purpose so new overlay types can be added later:

```yaml
spec_config:
  speculative-config:
    method: ngram
    num_speculative_tokens: 3
```

Resolution order: model-specific `extra_config.yml`, then `common/extra_config.yml`.
Values under `spec_config.speculative-config` map to vLLM's `--speculative-config`
(JSON object). Use nested YAML mappings (not escaped JSON strings).

Model-specific overrides are sourced from [vLLM Recipes](https://github.com/vllm-project/recipes/tree/main/models)
when available; otherwise the common n-gram default applies.
