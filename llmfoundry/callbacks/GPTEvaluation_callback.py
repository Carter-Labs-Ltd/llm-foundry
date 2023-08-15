# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
from typing import Any, List, Union, cast

import torch
import wandb
from composer.core import Callback, State, get_precision_context
from composer.loggers import Logger, WandBLogger
from composer.utils import dist, ensure_tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import json
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
from llmfoundry.callbacks.generate_callback import Generate
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key = os.environ['OPENAI_API_KEY']
import concurrent.futures
import logging
logger = logging.getLogger()
import time 
import re

class GPTEvaluation(Generate):
    def __init__(self, test_file: str, template_file:str, batch_log_interval: int,
                 **kwargs: Any):
        '''
        Subclass of generate clalback which loads prompts from file and sends responses to GPT for evaluation
        '''
        
        # load json from test file

        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        prompts = [x['prompt'] for x in test_data]

        with open(template_file, "r") as f:
            template = f.read()
        self.template = template

        super().__init__(prompts, batch_log_interval, **kwargs)
    
    def invoke_openai(self,
        model_name:str, payload:str
    ):
        """
        Invokes OpenAI's API.

        Args:
            model_name: the name of the model to use
            payload: payload to be passed to the endpoint

        Returns:
            result: the generated text
            info: info about the invocation attempt
        """
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": payload['input']}],
            max_tokens=payload['max_tokens'],
            temperature=payload['temperature'],
        )

        content = completion.choices[0].message["content"]
        assert isinstance(content, str)

        return content



    def get_result_with_timeout(self,
        llm_func: Any,
        payload:str,
        timeout:float,
    ):
        """
        Gets the result of an invocation function with a timeout.

        Args:
            llm_func: the function to invoke
            payload: the payload to pass to the function
            timeout: the timeout in seconds

        Returns:
            result: the result of the invocation or None if it timed out
            info: info about the invocation attempt
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            logger.info(f"LLM invocation: {llm_func}, {payload}")
            future = executor.submit(llm_func, payload)
            try:
                return future.result(timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout after {timeout} seconds.")
                
            except Exception as e:
                logger.exception("Failed to get result")
                
            return None
        finally:
            executor.shutdown(wait=False)



    def get_result_with_retry(self, 
        llm_func: Any,
        payload:str,
        timeout:float,
        retry_attempts:int
    ):
        """
        Gets the result of an invocation function with a timeout and retry attempts.

        Args:
            llm_func: the function to invoke
            payload: the payload to pass to the function
            timeout: the timeout in seconds
            retry_attempts: the number of retry attempts

        Returns:
            result: the result of the invocation or None if it timed out
            model_invocation_info: info about the invocation attempt
        """
        logger.info(
            f"Getting LLM result for {payload} with up to {retry_attempts} attempts with {timeout} second timeout each."
        )

    
        for attempt in range(retry_attempts):
            result = self.get_result_with_timeout(llm_func, payload, timeout)
        
            logger.info(f"Attempt {attempt + 1} result: {result}")
            if result is not None:
                return result
            else:
                logger.warning(
                    f"Attempt {attempt + 1} failed."
                    + (" Retrying..." if attempt < retry_attempts - 1 else "")
                )
            time.sleep(1) # hitting gpt4 rate limit with few shot prompt

        logger.error(f"Max retries exceeded.")
        return None

    def evaluate_prompt_response(self, prompt:str, response:str):
        
        template = self.template.replace("{prompt}", prompt)
        template = template.replace("{response}", response)

        payload = {
            "input": template,
            "max_tokens": 500,
            "temperature": 0.0,
        }

        result = self.get_result_with_retry(
            lambda x: self.invoke_openai('gpt-4',x),
            payload,
            timeout=30,
            retry_attempts=5
        )


        objs = self.extract_json_objects_from_string(result)
        
        if len(objs) == 0:
            objs = re.findall(r"[-+]?(?:\d*\.*\d+)", result)
            score = objs[0]
        else:
            score = objs[0]["SCORE"]
        if len(objs) == 0:
            score = 0
        
        return float(score), result

    def generate(self, state: State, logger: Logger):
        model = state.model
        original_mode = model.training
        model.eval()
        tokenizer = cast(Tokenizer, state.model.tokenizer)
        device = state.device

        if not hasattr(model.model, 'generate'):
            raise ValueError(
                f'Cannot generate from model {model.model.__class__.__name__} because it does not have a `generate` method'
            )

        # stash the original original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts,
                                    return_tensors='pt',
                                    padding=True)

        for k, v in tokenized_input.items():
            tokenized_input[k] = device.tensor_to_device(v)

        # dummy forward call needed for FSDP to work consistently
        dummy_input = torch.tensor([[0]], dtype=torch.long)
        dummy_input = device.tensor_to_device(dummy_input)
        with get_precision_context(state.precision):
            with torch.no_grad():
                assert isinstance(model.model, torch.nn.Module)
                _ = model.model(input_ids=dummy_input)

            output_token_ids = model.model.generate(  # type: ignore
                input_ids=tokenized_input['input_ids'],
                attention_mask=tokenized_input['attention_mask'],
                synced_gpus=True,
                **self.generate_kwargs,
            )

        if dist.get_global_rank() == 0:
            if self.wandb_logger is not None:
                assert wandb.run is not None, 'wandb should have started run'

                artifact = wandb.Artifact('generate_samples_' +
                                          str(wandb.run.id),
                                          type='predictions')

                rows = []
                t_score = 0
                n = 0
                for i in range(len(self.prompts)):
                    prompt = self.prompts[i]
                    output_tokens = output_token_ids[i][
                        tokenized_input['input_ids'].shape[1]:]
                    output_text = tokenizer.decode(output_tokens,
                                                   skip_special_tokens=True)
                    
                    try:
                        score, result = self.evaluate_prompt_response(prompt, output_text)
                        n+=1
                        t_score += score
                    except Exception:
                        logger.exception("Failed to evaluate prompt response")
                        result = "Failed to evaluate prompt response"

                    rows.append([prompt, output_text, result, score])



                text_table = wandb.Table(data=rows,
                                         columns=['prompt', 'generation'])
                artifact.add(text_table, 'predictions')
                wandb.log_artifact(artifact)
                wandb.log({'generations': text_table},
                          step=state.timestamp.batch.value)
                wandb.log({'GPT-score:', t_score/n})

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)

    
