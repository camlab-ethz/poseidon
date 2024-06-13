"""
Our version of the Huggingface Trainer class.
It adds learning_rate_time_embedding, learning_rate_embedding_recovery as 
additional learning rates and groups parameters for the optimizer.
It also allows for autoregressive rollouts by using 
trainer.set_ar_steps(AR_STEPS) where AR_STEPS is either a an integer for a 
homogeneous rollout of AR_STEPS steps or a list of integers for a heterogeneous
rollout where each element is the timestep.
If, additionally, output_all_steps is also set, the predict function will
output all intermediate steps as well.

We sublass a Huggingface Trainer to allow for autoregressive rollouts and multiple parameter groups in the optimizer.
It is specifically subclassed for our purpose.

A lot of code is copied over because only slight changes have been made.

The original code of Huggingface Transformers is distributed under the Apache 2.0 license. See below:

Copyright 2018- The Hugging Face team. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_
from transformers import TrainingArguments as TrainingArguments_
from scOT.model import LayerNorm, ConditionalLayerNorm
from dataclasses import dataclass, field


@dataclass
class TrainingArguments(TrainingArguments_):
    learning_rate_embedding_recovery: Optional[float] = field(
        default=None,
        metadata={
            "help": "The initial learning rate for the embedding/recovery. When not provided, falls back to `learning_rate`."
        },
    )

    learning_rate_time_embedding: Optional[float] = field(
        default=None,
        metadata={
            "help": "The initial learning rate for the time embedding. When not provided, falls back to `learning_rate`. Only used when embedding and recovery are also fine-tuned with different lr."
        },
    )

    def set_training(
        self,
        *args,
        learning_rate_embedding_recovery: Optional[float] = None,
        learning_rate_time_embedding: Optional[float] = None,
        **kwargs,
    ):
        self = super().set_training(*args, **kwargs)
        self.learning_rate_embedding_recovery = learning_rate_embedding_recovery
        self.learning_rate_time_embedding = learning_rate_time_embedding
        return self

    def set_optimizer(
        self,
        *args,
        learning_rate_embedding_recovery: Optional[float] = None,
        learning_rate_time_embedding: Optional[float] = None,
        **kwargs,
    ):
        self = super().set_optimizer(*args, **kwargs)
        self.learning_rate_embedding_recovery = learning_rate_embedding_recovery
        self.learning_rate_time_embedding = learning_rate_time_embedding
        return self


class Trainer(Trainer_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_steps = None
        self.output_all_steps = False

    def get_decay_parameter_names(self, model) -> List[str]:
        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, LayerNorm, ConditionalLayerNorm]
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def get_conditional_norm_params(self, model):
        params = []
        for name, module in model.named_modules():
            if isinstance(module, ConditionalLayerNorm):
                for param_name, _ in module.named_parameters():
                    params.append(f"{name}.{param_name}")
        return params

    def create_optimizer(self):
        """This is the same as in the standard trainer, except param groups"""
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            if self.args.learning_rate_embedding_recovery is not None:
                if self.args.learning_rate_time_embedding is not None:
                    time_embedding_params = self.get_conditional_norm_params(self.model)
                    params = {
                        "standard": [],
                        "no_weight_decay": [],
                        "embeddings": [],
                        "time_embedding": [],
                    }
                    for n, p in opt_model.named_parameters():
                        if (
                            "embeddings" in n or "patch_recovery" in n
                        ) and p.requires_grad:
                            params["embeddings"].append(p)
                        elif n in decay_parameters and p.requires_grad:
                            params["standard"].append(p)
                        elif p.requires_grad:
                            if n in time_embedding_params:
                                params["time_embedding"].append(p)
                            else:
                                params["no_weight_decay"].append(p)
                    optimizer_grouped_parameters = [
                        {
                            "params": params["standard"],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["no_weight_decay"],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": params["embeddings"],
                            "lr": self.args.learning_rate_embedding_recovery,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["time_embedding"],
                            "lr": self.args.learning_rate_time_embedding,
                            "weight_decay": 0.0,
                        },
                    ]
                else:
                    params = {"standard": [], "no_weight_decay": [], "embeddings": []}
                    for n, p in opt_model.named_parameters():
                        if (
                            "embeddings" in n or "patch_recovery" in n
                        ) and p.requires_grad:
                            params["embeddings"].append(p)
                        elif n in decay_parameters and p.requires_grad:
                            params["standard"].append(p)
                        elif p.requires_grad:
                            params["no_weight_decay"].append(p)
                    optimizer_grouped_parameters = [
                        {
                            "params": params["standard"],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["no_weight_decay"],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": params["embeddings"],
                            "lr": self.args.learning_rate_embedding_recovery,
                            "weight_decay": self.args.weight_decay,
                        },
                    ]
            elif self.args.learning_rate_time_embedding is not None:
                time_embedding_params = self.get_conditional_norm_params(self.model)
                params = {"standard": [], "no_weight_decay": [], "time_embedding": []}
                for n, p in opt_model.named_parameters():
                    if n in decay_parameters and p.requires_grad:
                        params["standard"].append(p)
                    elif p.requires_grad:
                        if n in time_embedding_params:
                            params["time_embedding"].append(p)
                        else:
                            params["no_weight_decay"].append(p)
                optimizer_grouped_parameters = [
                    {
                        "params": params["standard"],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": params["no_weight_decay"],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": params["time_embedding"],
                        "lr": self.args.learning_rate_time_embedding,
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32"
                        )
                print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def set_ar_steps(self, ar_steps=None, output_all_steps=False):
        self.ar_steps = ar_steps
        if self.ar_steps is not None and output_all_steps:
            self.output_all_steps = True

    def _model_forward(self, model, inputs):
        if self.ar_steps is not None and model.config.use_conditioning:
            channel_difference = (
                model.config.num_channels > model.config.num_out_channels
            )
            # TODO: if outputs is not a dataclass this will break
            if isinstance(self.ar_steps, int):
                inputs = {**inputs, **{"time": inputs["time"] / self.ar_steps}}
                if self.output_all_steps:
                    loss_ = []
                    outputs_ = []
                    hidden_states_ = []
                    attentions_ = []
                    reshaped_hidden_states_ = []
                else:
                    loss = 0
                for i in range(self.ar_steps):
                    outputs = model(**inputs)
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(
                                outputs.reshaped_hidden_states
                            )
                        if outputs.loss is not None:
                            loss_.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss += outputs.loss
                    inputs = {
                        **inputs,
                        **{
                            "pixel_values": (
                                outputs.output.detach()
                                if not channel_difference
                                else torch.cat(
                                    [
                                        outputs.output.detach(),
                                        inputs["pixel_values"][
                                            :,
                                            model.config.num_out_channels :,
                                        ],
                                    ],
                                    dim=1,
                                )
                            )
                        },
                    }
                if self.output_all_steps:
                    outputs.output = torch.stack(outputs_, dim=1)
                    if len(loss_) > 0:
                        outputs.loss = torch.stack(loss_, dim=0)
                    if len(hidden_states_) > 0:
                        outputs.hidden_states = [
                            torch.stack(hs, dim=1) for hs in zip(*hidden_states_)
                        ]
                    if len(attentions_) > 0:
                        outputs.attentions = [
                            torch.stack(att, dim=1) for att in zip(*attentions_)
                        ]
                    if len(reshaped_hidden_states_) > 0:
                        outputs.reshaped_hidden_states = [
                            torch.stack(rhs, dim=1)
                            for rhs in zip(*reshaped_hidden_states_)
                        ]
                else:
                    loss /= self.ar_steps
                    outputs.loss = loss
            elif isinstance(self.ar_steps, list):
                if self.output_all_steps:
                    loss_ = []
                    outputs_ = []
                    hidden_states_ = []
                    attentions_ = []
                    reshaped_hidden_states_ = []
                else:
                    loss = 0
                lead_time = inputs["time"]
                for i in self.ar_steps:
                    inputs = {
                        **inputs,
                        **{"time": lead_time * i},
                    }
                    outputs = model(**inputs)
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(
                                outputs.reshaped_hidden_states
                            )
                        if outputs.loss is not None:
                            loss_.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss += outputs.loss
                    inputs = {
                        **inputs,
                        **{
                            "pixel_values": (
                                outputs.output.detach()
                                if not channel_difference
                                else torch.cat(
                                    [
                                        outputs.output.detach(),
                                        inputs["pixel_values"][
                                            :,
                                            model.config.num_out_channels :,
                                        ],
                                    ],
                                    dim=1,
                                )
                            )
                        },
                    }
                if self.output_all_steps:
                    outputs.output = torch.stack(outputs_, dim=1)
                    if len(loss_) > 0:
                        outputs.loss = torch.stack(loss_, dim=1)
                    if len(hidden_states_) > 0:
                        outputs.hidden_states = [
                            torch.stack(hs, dim=1) for hs in zip(*hidden_states_)
                        ]
                    if len(attentions_) > 0:
                        outputs.attentions = [
                            torch.stack(att, dim=1) for att in zip(*attentions_)
                        ]
                    if len(reshaped_hidden_states_) > 0:
                        outputs.reshaped_hidden_states = [
                            torch.stack(rhs, dim=1)
                            for rhs in zip(*reshaped_hidden_states_)
                        ]
                else:
                    loss /= len(self.ar_steps)
                    outputs.loss = loss
            else:
                raise ValueError(
                    "num_ar_steps must be an integer or a list of integers."
                )
        else:
            outputs = model(**inputs)

        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = self._model_forward(model, inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = self._model_forward(model, inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
