import paddle
import paddlenlp.generation as p_generation
from paddlenlp.transformers import LlamaForCausalLM


class Llama2(LlamaForCausalLM):

    def greedy_search(
            self,
            input_ids,
            logits_processors,
            max_length,
            pad_token_id,
            eos_token_id,
            stopping_criteria=None,
            streamer=None,
            fast_ptq_sampling=False,
            **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
        logits_processors = logits_processors if logits_processors is not None else p_generation.LogitsProcessorList()
        eos_token_ids = [eos_token_id, self.generation_config.eos_token_id]
        batch_size, cur_len = input_ids.shape
        attention_mask = model_kwargs["attention_mask"]
        next_token_logits_idx = paddle.sum(attention_mask, axis=-1) - 1
        origin_len = paddle.sum(attention_mask, axis=-1)
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        while True:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)
            logits = outputs[0]
            # [batch_size, vocab_size]
            next_token_logits = logits[range(batch_size), next_token_logits_idx, :]
            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            next_tokens_scores = logits_processors(input_ids, next_token_logits)
            # greedy
            probs = paddle.nn.functional.softmax(next_tokens_scores)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_tokens[~unfinished_flag] = pad_token_id
            next_scores = paddle.index_sample(probs, next_tokens)
            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            next_token_logits_idx += 1
            if paddle.any(next_token_logits_idx >= cur_len):
                input_ids = paddle.concat([input_ids, paddle.full_like(next_tokens, pad_token_id)], axis=1)
                model_kwargs['attention_mask'] = paddle.concat([model_kwargs['attention_mask'],
                                                                paddle.full_like(next_tokens, pad_token_id)], axis=1)
                model_kwargs['position_ids'] = paddle.concat([model_kwargs['position_ids'],
                                                              paddle.full_like(next_tokens, pad_token_id)], axis=1)
            input_ids[range(batch_size), next_token_logits_idx] = next_tokens.squeeze(-1)
            model_kwargs['attention_mask'][range(batch_size), next_token_logits_idx] = 1
            model_kwargs['position_ids'][range(batch_size), next_token_logits_idx] = (
                    model_kwargs['position_ids'][range(batch_size), next_token_logits_idx - 1] + 1)
            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())
            for eos_token_id in eos_token_ids:
                unfinished_flag = unfinished_flag.logical_and(next_tokens != eos_token_id)
            if unfinished_flag.logical_not().all():
                break
            cur_len += 1
            if cur_len >= max_length:
                break
            if fast_ptq_sampling:
                break
        if streamer is not None:
            streamer.end()
        output_list = [input_ids[i, origin_len[i]: next_token_logits_idx[i] + 1] for i in range(batch_size)]
        max_output_len = max([len(output) for output in output_list])
        output_list = [paddle.concat([output, paddle.full([max_output_len - len(output)], pad_token_id,
                                                          dtype=output.dtype)]) for output in output_list]
        return paddle.stack(output_list), scores
